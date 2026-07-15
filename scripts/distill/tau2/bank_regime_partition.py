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

# ── ★두 decidability 분리 (v4·리뷰 ❶) ──
# oracle_decidable = gold파생 criteria로 filter가 gold 유일도달 (=science 천장·true-dup의 역수·gold 필요·라우터 신호 아님)
# runtime_decidable = 모델 formalize criteria로 filter가 real id 산출 (=gold-free 라우터 신호·maj_id가 abstain/malformed 아님)
# gap = oracle_dec ∧ ¬(runtime이 gold 도달) = formalization error = 라우터 맹점 (§8 핵심수치)
def gold_criteria(gold):
    return {"date": gold.get("date"), "merchant": gold.get("description"),
            "transaction_type": gold.get("type"), "amount": gold.get("amount")}

def oracle_decidable(row):
    """gold파생 criteria filter가 gold 유일도달? = 구조적 resolvability(science). ⚠gold 필요=라우터 신호 아님."""
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
           "n_disputes": row.get("n_disputes", 1), "oracle_decidable": oracle_decidable(row)}
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
    # ★plan/multi-item regime(A·C89): maj가 *다른 유효 dispute*로 앵커링? (C79 mis-pairing=plan, not field-verify)
    out["maj_in_dispute_set"] = str(maj_id) in set(str(x) for x in row.get("dispute_set", []))
    # ★runtime signal (v4·리뷰 ❶❸): gold-free — maj가 real id인가 / abstain / malformed
    out["maj_kind"] = ("abstain" if maj_id == "__ABSTAIN__"
                       else "malformed" if maj_id == "__MALFORMED__" else "real_id")
    out["runtime_decidable"] = (out["maj_kind"] == "real_id")   # 라우터가 유일 id 산출 (gold 불요)
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
    meas = [r for r in ok if r.get("measurable")]     # ★v4(❷): universe=측정가능 전체·voting%는 2×2가 32B-greedy-wrong로 조건화
    ff = sum(1 for r in ok if r["chosen"] != r["gid"])
    print("\n=== E-REGIME ⋈ partition · tag=%s · k=%d · T=%.1f ===" % (tag, k, temperature))
    print("총 %d · err %d · 측정가능 %d" % (len(results), len(errs), len(meas)))
    print("⚠selection bias(§12): 이 케이스셋 %d 중 frontier-failed(chosen≠gold) %d = %.0f%% "
          "→ frontier가 틀린 *더 어려운* ⋈점 편중·voting%%을 아래로 편향(C88 과대평가 위험)·probe모델=32B≠궤적=frontier"
          % (len(ok), ff, 100*ff/max(len(ok), 1)))
    odec = [r for r in meas if r["oracle_decidable"]]; onondec = [r for r in meas if not r["oracle_decidable"]]
    print("oracle-decidable(science 천장) %d · non(true-dup=ASK 천장) %d" % (len(odec), len(onondec)))
    mal = sum(r.get("malformed", 0) for r in meas)
    print("malformed 샘플 총 %d (별도 실패·§5)" % mal)
    if not meas:
        print("측정가능 0 — 서버 실행 필요 or 데이터 확인"); return
    # ── 2×2 greedy×maj (32B) ──
    A = sum(1 for r in meas if r["greedy_ok"] and r["maj_ok"])
    B = sum(1 for r in meas if r["greedy_ok"] and not r["maj_ok"])
    C = sum(1 for r in meas if not r["greedy_ok"] and r["maj_ok"])
    D = sum(1 for r in meas if not r["greedy_ok"] and not r["maj_ok"])
    print("\n2×2 (32B greedy × maj@k) · n=%d" % len(meas))
    print("            maj_ok  maj_wrong")
    print("greedy_ok   %5d    %5d   (A=easy/B=voting-hurts)" % (A, B))
    print("greedy_wr   %5d    %5d   (C=voting-win / D=voting-fail)" % (C, D))
    gw = C + D
    if gw:
        lo, hi = wilson(C/gw, gw)
        print("\nvoting%% = C/(C+D) = %d/%d = %.1f%%  [95%% CI %.1f–%.1f]  (universe=32B-greedy-wrong)"
              % (C, gw, 100*C/gw, 100*lo, 100*hi))
    # ── partition on greedy-wrong (D) — ★4-way(A·C89): voting/plan/verify/ASK ──
    Dset = [r for r in meas if not r["greedy_ok"] and not r["maj_ok"]]
    plan = [r for r in Dset if r.get("maj_in_dispute_set")]                       # 다른 유효 dispute=plan/multi-item→E-PLAN
    rest = [r for r in Dset if not r.get("maj_in_dispute_set")]
    verify = [r for r in rest if r["oracle_decidable"] and r.get("maj_kind") == "real_id"]  # 무관 record·oracle-resolvable=field-verify
    ask = [r for r in rest if r not in verify]                                    # abstain or true-dup=ASK
    print("\npartition (greedy-wrong n=%d):  voting %d · plan %d · verify %d · ASK %d" % (gw, C, len(plan), len(verify), len(ask)))
    print("  plan(maj∈dispute_set=다중-dispute 앵커링→E-PLAN) %d = C79 mis-pairing" % len(plan))
    print("  gold∈support(하한·k=%d): %d/%d" % (k, sum(1 for r in Dset if r.get("gold_in_support")), len(Dset)))
    # ── ★GAP: oracle vs runtime (§8 핵심·gold-free 라우터 실현성) ──
    od_gw = [r for r in meas if r["oracle_decidable"] and not r["greedy_ok"]]  # oracle-resolvable인데 32B가 틀림
    rt_real = sum(1 for r in od_gw if r["runtime_decidable"])   # 라우터가 real id 산출(gold-free)
    rt_correct = sum(1 for r in od_gw if r.get("maj_ok"))       # 그게 gold
    print("\nGAP(§8 gold-free 라우터): oracle-decidable ∧ 32B-greedy-wrong = %d" % len(od_gw))
    print("  런타임 real-id 산출 %d · 그중 gold도달 %d → formalization-gap(맹점) %d = oracle는 풀리는데 라우터 못 닿음"
          % (rt_real, rt_correct, len(od_gw) - rt_correct))
    print("  (gold-free 라우터가 이 gap을 못 좁히면 §8 실현성 미달·H_k marginal은 full-run서 AUROC)")

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
    ff = [r for r in rows if str(r.get("chosen_id")) != str(r["gold_id"])]
    odec = sum(1 for r in rows if oracle_decidable(r))
    print("cases %d · frontier-failed(chosen≠gold) %d · oracle-decidable %d(=science 천장·true-dup 역수) · true-dup(ASK) %d"
          % (len(rows), len(ff), odec, len(rows) - odec), flush=True)
    if a.dry:
        print("[--dry] 오프라인 검증(오라클 decidability만·런타임 신호=서버 필요).")
        # ⚠❹: n_disputes≥2(유저가 ≥2건 dispute) ≠ true-dup(record-collision 18%). 별개 양.
        md = sum(1 for r in rows if r.get("n_disputes", 1) >= 2)
        print("다중-dispute 맥락(n_disputes≥2) %d (%.0f%%)  [주의: true-dup율 %.0f%%와 별개 개념(❹)]"
              % (md, 100*md/max(len(rows), 1), 100*(len(rows)-odec)/max(len(rows), 1)))
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
