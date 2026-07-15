# -*- coding: utf-8 -*-
"""E-REGIME — banking per-step regime partition (voting-solvable vs verify vs ASK).
설계: BANK_REGIME_PARTITION_PROBE_DESIGN_2026_07_14 (v3). C88 파생.

⋈ 버킷 먼저(⋈-first 게이트). 각 in-situ 실패 ⋈ 스텝(chosen_id != gold_id)을:
  - greedy(T=0) formalize->filter -> id_g -> greedy_ok
  - k회 resample(T=0.7) formalize->filter -> k ids -> maj@k, gold∈support, H_k, k_valid, malformed
  - decidability = 조작적(§7): perfect-formalize(gold파생 criteria) filter가 gold 유일도달? (=verify가능) 아니면 true-dup/underspec(=ASK)
  - 2×2 (greedy_ok × maj_ok) · partition: voting=C(greedy-wrong & maj-ok) / verify=D∩decidable / ASK=D∩non-decidable

무료·로컬(32B·localhost:8140). resample = vLLM n-param(한 호출 k choices).
실행:
  오프라인(서버불요·실패셋+decidability 검증):  python bank_regime_partition.py --dry
  Phase0 스모크(⋈ n~30):                       python bank_regime_partition.py --limit 30 --k 8
  Phase1 ⋈ full:                               python bank_regime_partition.py --k 8
  ⋈ k-curve/반박차단:                          python bank_regime_partition.py --k 32
  T-민감도:                                     python bank_regime_partition.py --k 8 --temperature 1.0
결과 -> sim_results/bank_regime_partition.<tag>.json
"""
import json, gzip, re, argparse, urllib.request, os, math, sys, io
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
SR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
CASES = os.path.join(SR, "bank_xmatch_cases.jsonl.gz")
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
FIELDS = ["date", "merchant", "transaction_type", "amount"]

# ── 결정론 필터 (bank_keystone_formalize.py와 동일·도메인일반) ──
def date_key(v):
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})", str(v or ""))
    return "%02d/%02d" % (int(m.group(1)), int(m.group(2))) if m else None

def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())

def amt_key(v):
    try:
        return round(abs(float(re.sub(r"[^0-9.\-]", "", str(v)))), 2)
    except Exception:
        return None

def match(rec, crit, use):
    if "date" in use and crit.get("date"):
        if date_key(rec.get("date")) != date_key(crit.get("date")):
            return False
    if "merch" in use and crit.get("merchant"):
        cm = _alnum(crit["merchant"]); rd = _alnum(rec.get("description"))
        if cm and cm not in rd:
            toks = [_alnum(t) for t in re.split(r"\s+", str(crit["merchant"])) if len(_alnum(t)) >= 4]
            if not (toks and all(t in rd for t in toks)):
                return False
    if "type_fuzzy" in use and crit.get("transaction_type"):
        rt = re.sub(r"[^a-z]", "", str(rec.get("type") or "").lower())
        toks = [t for t in re.split(r"[^a-z]+", str(crit["transaction_type"]).lower()) if len(t) >= 3]
        if toks and not any(t in rt for t in toks):
            return False
    if "amount" in use and crit.get("amount") is not None:
        ak = amt_key(crit.get("amount"))
        if ak is not None and amt_key(rec.get("amount")) != ak:
            return False
    return True

def filter_id(records, crit, use):
    hits = [r for r in records if match(r, crit, use)]
    return hits[0].get("transaction_id") if len(hits) == 1 else None  # 0/≥2 -> None(abstain)

USE_ALL = {"date", "merch", "type_fuzzy", "amount"}

# ── decidability (조작적·§7): perfect-formalize(gold파생 criteria) filter가 gold 유일도달? ──
def gold_criteria(gold):
    return {"date": gold.get("date"), "merchant": gold.get("description"),
            "transaction_type": gold.get("type"), "amount": gold.get("amount")}

def is_decidable(row):
    """gate_spec(=결정론 filter)이 최선(gold파생) criteria로 gold를 유일 산출 가능? = verify가능. 아니면 ASK."""
    rid = filter_id(row["records"], gold_criteria(row["gold"]), USE_ALL)
    return rid is not None and str(rid) == str(row["gold_id"])

# ── formalize (bank_keystone_formalize와 동형) ──
def formalize_prompt(users):
    return ("The user is disputing/investigating ONE specific transaction. From their messages, extract "
            "identifying criteria for the SINGLE transaction they most recently want to dispute, "
            "as ONE JSON object (NOT a list) with keys %s (use null if not stated). "
            "Dates as MM/DD. 'amount' as a positive number (dollars). "
            "'transaction_type' in plain words (e.g. purchase, atm withdrawal, deposit) if implied.\n"
            "User said:\n- %s\nReply with a single JSON object only." % (FIELDS, "\n- ".join(u[:400] for u in users[-8:])))

def parse_json(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None  # malformed
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except Exception:
        return None

def call_llm(port, prompt, temperature, n, timeout=120):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": 160}
    if n > 1:
        body["n"] = n
    req = urllib.request.Request("http://localhost:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return [c["message"]["content"] for c in d["choices"]]

def sample_to_id(raw, row):
    """LLM raw -> criteria -> filter id. malformed(파싱불가)면 특수값."""
    crit = parse_json(raw)
    if crit is None:
        return "__MALFORMED__"
    rid = filter_id(row["records"], crit, USE_ALL)
    return rid if rid is not None else "__ABSTAIN__"  # None(0/≥2 매칭)=abstain(유효)

def entropy(counter, total):
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h

# ── per-case 측정 ──
def process(row, port, k, temperature):
    gold = str(row["gold_id"])
    out = {"tid": row["tid"], "model": row["model"], "gid": gold, "chosen": str(row.get("chosen_id")),
           "n_disputes": row.get("n_disputes", 1), "decidable": is_decidable(row)}
    try:
        g = call_llm(port, formalize_prompt(row["users"]), 0.0, 1)[0]
        out["greedy_id"] = sample_to_id(g, row)
        raws = call_llm(port, formalize_prompt(row["users"]), temperature, k)
    except Exception as e:
        out["err"] = str(e)[:120]
        return out
    ids = [sample_to_id(r, row) for r in raws]
    malformed = sum(1 for i in ids if i == "__MALFORMED__")
    valid = [i for i in ids if i != "__MALFORMED__"]
    out["k"] = k; out["k_valid"] = len(valid); out["malformed"] = malformed
    out["greedy_ok"] = (out["greedy_id"] == gold)
    if len(valid) < 5:
        out["measurable"] = False
        return out
    out["measurable"] = True
    cnt = Counter(valid)
    maj_id, maj_n = cnt.most_common(1)[0]
    out["maj_id"] = maj_id; out["maj_ok"] = (maj_id == gold)
    out["gold_in_support"] = (gold in cnt)          # k=8에선 하한(설계 §7)
    out["p_gold"] = cnt.get(gold, 0) / len(valid)
    out["H_k"] = round(entropy(cnt, len(valid)), 3)
    out["top_freq"] = round(maj_n / len(valid), 3)
    return out

def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0, (c-h)/d), min(1, (c+h)/d))

def report(rows, results, k, temperature, tag):
    ok = [r for r in results if not r.get("err")]
    errs = [r for r in results if r.get("err")]
    insitu_fail = [r for r in ok if r["chosen"] != r["gid"]]           # ★fix① 실패 universe
    meas = [r for r in insitu_fail if r.get("measurable")]
    print("\n=== E-REGIME ⋈ partition · tag=%s · k=%d · T=%.1f ===" % (tag, k, temperature))
    print("총 %d · err %d · in-situ 실패(chosen≠gold) %d · 측정가능 %d"
          % (len(results), len(errs), len(insitu_fail), len(meas)))
    dec = [r for r in meas if r["decidable"]]; nondec = [r for r in meas if not r["decidable"]]
    print("decidable(verify가능) %d · non-decidable(ASK) %d" % (len(dec), len(nondec)))
    mal = sum(r.get("malformed", 0) for r in meas)
    print("malformed 샘플 총 %d (별도 실패·§5)" % mal)
    if not meas:
        print("측정가능 0 — 서버 실행 필요 or 데이터 확인"); return
    # 2×2 greedy×maj (in-situ 실패 위)
    A = sum(1 for r in meas if r["greedy_ok"] and r["maj_ok"])
    B = sum(1 for r in meas if r["greedy_ok"] and not r["maj_ok"])
    C = sum(1 for r in meas if not r["greedy_ok"] and r["maj_ok"])
    D = sum(1 for r in meas if not r["greedy_ok"] and not r["maj_ok"])
    print("\n2×2 (greedy × maj@k) · n=%d" % len(meas))
    print("            maj_ok  maj_wrong")
    print("greedy_ok   %5d    %5d   (A/B)" % (A, B))
    print("greedy_wr   %5d    %5d   (C=voting-win / D=voting-fail)" % (C, D))
    gw = C + D
    if gw:
        lo, hi = wilson(C/gw, gw)
        print("\nvoting%% = C/(C+D) = %d/%d = %.1f%%  [95%% CI %.1f–%.1f]" % (C, gw, 100*C/gw, 100*lo, 100*hi))
    # partition on greedy-wrong (C+D)
    Dset = [r for r in meas if not r["greedy_ok"] and not r["maj_ok"]]
    verify = sum(1 for r in Dset if r["decidable"]); ask = sum(1 for r in Dset if not r["decidable"])
    print("\npartition (greedy-wrong n=%d):  voting %d · verify %d · ASK %d" % (gw, C, verify, ask))
    print("gold∈support(하한·k=%d): %d/%d" % (k, sum(1 for r in Dset if r.get("gold_in_support")), len(Dset)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="서버불요·실패셋+decidability만")
    a = ap.parse_args()
    with gzip.open(CASES, "rt", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    if a.limit:
        rows = rows[:a.limit]
    insitu = [r for r in rows if str(r.get("chosen_id")) != str(r["gold_id"])]
    dec = sum(1 for r in insitu if is_decidable(r))
    print("cases %d · in-situ 실패 %d · 그중 decidable %d · non-decidable(ASK) %d"
          % (len(rows), len(insitu), dec, len(insitu) - dec), flush=True)
    if a.dry:
        print("[--dry] 오프라인 검증 완료(서버 실행 시 resample partition 산출).")
        # 다중-dispute 구조
        md = sum(1 for r in insitu if r.get("n_disputes", 1) >= 2)
        print("in-situ 실패 중 다중-dispute %d (%.0f%%)" % (md, 100*md/max(len(insitu), 1)))
        return
    tag = "k%d_T%.1f%s" % (a.k, a.temperature, "_n%d" % a.limit if a.limit else "")
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, r, a.port, a.k, a.temperature) for r in rows]
        for i, fu in enumerate(futs):
            results.append(fu.result())
            if (i + 1) % 50 == 0:
                print("  %d/%d" % (i + 1, len(rows)), flush=True)
    out = os.path.join(SR, "bank_regime_partition.%s.json" % tag)
    json.dump({"k": a.k, "T": a.temperature, "results": results}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False)
    report(rows, results, a.k, a.temperature, tag)
    print("결과 -> %s" % out)

if __name__ == "__main__":
    main()
