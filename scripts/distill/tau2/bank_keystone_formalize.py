# -*- coding: utf-8 -*-
"""reference-filter formalize half 실측 — 리모트 32B e2e (2026-07-14·HANDOFF §0·[[09]] 무료 on-prem).
bank_keystone_extract가 뽑은 ⋈ 케이스 JSONL을 읽어, 각 케이스의 user 발화를 **실제 LLM formalize**
(localhost vLLM Qwen2.5-32B·formalize_reference_criteria와 동형 프롬프트)로 돌려 식별기준을 뽑고,
결정론 filter로 gold 도달 여부를 잰다. = 교정률의 formalize half(perfect-formalize 천장 대비 실측).

이전 replay(bank_keystone_replay)는 criteria를 gold서 파생(perfect-formalize 천장). 이건 user발화서
실제 formalize → 전체 교정률 = formalize정확도 × filter천장. 여러 filter config를 한 formalize 호출로
브래킷(as-specced 3필드 exact + type-fuzzy + amount-numeric).

실행(리모트): python bank_keystone_formalize.py --port 8140 --workers 16
  (localhost vLLM·무료·API비용0). 결과 → sim_results/bank_xmatch_formalize.results.json.
"""
import json, gzip, re, argparse, urllib.request, os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results", "bank_xmatch_cases.jsonl.gz")
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results", "bank_xmatch_formalize.results.json")
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
FIELDS = ["date", "merchant", "transaction_type", "amount"]


def load_cases():
    rows = []
    with gzip.open(CASES, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def formalize_prompt(users):
    # formalize_reference_criteria(t2_resolve.py:406)와 동형 + 단일객체 강제·앵커 명시.
    return ("The user is disputing/investigating ONE specific transaction. From their messages, extract "
            "identifying criteria for the SINGLE transaction they most recently want to dispute, "
            "as ONE JSON object (NOT a list) with keys %s (use null if not stated). "
            "Dates as MM/DD. 'amount' as a positive number (dollars). "
            "'transaction_type' in plain words (e.g. purchase, atm withdrawal, deposit) if implied.\n"
            "User said:\n- %s\nReply with a single JSON object only." % (FIELDS, "\n- ".join(u[:400] for u in users[-8:])))


def call_llm(port, prompt, timeout=90):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 160,
    }).encode()
    req = urllib.request.Request("http://localhost:%d/v1/chat/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def parse_json(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ── 필드-매칭 (도메인일반·현실적: date-prefix·merchant contains·type fuzzy·amount numeric) ──
def date_key(v):
    # MM/DD (연도 무시·데이터 단일연도·user는 연도 거의 안 말함). 시각접미·연도유무 모두 허용.
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})", str(v or ""))
    if not m:
        return None
    return "%02d/%02d" % (int(m.group(1)), int(m.group(2)))


def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def amt_key(v):
    try:
        return round(abs(float(re.sub(r"[^0-9.\-]", "", str(v)))), 2)
    except Exception:
        return None


def match(rec, crit, use):
    """use = set of {date,merch,type_exact,type_fuzzy,amount}. 조건 있는것만 적용(부분기준 허용)."""
    if "date" in use and crit.get("date"):
        if date_key(rec.get("date")) != date_key(crit.get("date")):
            return False
    if "merch" in use and crit.get("merchant"):
        # alnum 정규화 contains (구두점·공백·대소문 무시). 다중토큰이면 어느 토큰이든 등장 시 매칭 완화 회피:
        # 전체 alnum-phrase contains → 실패 시 개별 유의미토큰(≥4) 전부 등장 요구.
        cm = _alnum(crit["merchant"]); rd = _alnum(rec.get("description"))
        if cm and cm not in rd:
            toks = [_alnum(t) for t in re.split(r"\s+", str(crit["merchant"])) if len(_alnum(t)) >= 4]
            if not (toks and all(t in rd for t in toks)):
                return False
    if "type_exact" in use and crit.get("transaction_type"):
        if str(rec.get("type")) != str(crit["transaction_type"]):
            return False
    if "type_fuzzy" in use and crit.get("transaction_type"):
        ct = re.sub(r"[^a-z]", "", str(crit["transaction_type"]).lower())
        rt = re.sub(r"[^a-z]", "", str(rec.get("type") or "").lower())
        # 토큰 겹침: criteria type의 어느 단어가 record type에 등장
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
    if len(hits) == 1:
        return hits[0].get("transaction_id")
    return None                        # 0 or ≥2 (on_ambiguous=none)


CONFIGS = {
    "C1_as_specced(date+merch+type_exact)": {"date", "merch", "type_exact"},
    "C2_date+merch": {"date", "merch"},
    "C3_date+merch+amount": {"date", "merch", "amount"},
    "C4_date+type_fuzzy+amount": {"date", "type_fuzzy", "amount"},
    "C5_all(date+merch+type_fuzzy+amount)": {"date", "merch", "type_fuzzy", "amount"},
}


def process(row, port):
    try:
        raw = call_llm(port, formalize_prompt(row["users"]))
    except Exception as e:
        return {"err": str(e)[:100], "tid": row["tid"]}
    crit = parse_json(raw)
    gid = row["gold_id"]; recs = row["records"]
    res = {"tid": row["tid"], "model": row["model"], "gid": gid, "crit": crit, "raw": raw[:200], "cfg": {}}
    for name, use in CONFIGS.items():
        rid = filter_id(recs, crit, use)
        res["cfg"][name] = "ok" if (rid is not None and str(rid) == str(gid)) else ("wrong" if rid is not None else "none")
    return res


def is_true_dup(row):
    g = row["gold"]; fs = ["date", "amount", "type", "description"]
    return sum(1 for r in row["records"] if all(r.get(k) == g.get(k) for k in fs)) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = load_cases()
    if a.limit:
        rows = rows[:a.limit]
    n = len(rows)
    dup = sum(1 for r in rows if is_true_dup(r))
    print("formalize e2e: %d ⋈ cases (진짜중복 %d) · port %d · workers %d" % (n, dup, a.port, a.workers), flush=True)

    results = []
    done = [0]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, r, a.port) for r in rows]
        for fu in futs:
            results.append(fu.result())
            done[0] += 1
            if done[0] % 100 == 0:
                print("  %d/%d" % (done[0], n), flush=True)

    errs = [r for r in results if r.get("err")]
    ok_rows = [r for r in results if not r.get("err")]
    json.dump({"n": n, "dup": dup, "errs": len(errs), "results": results},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    dec = n - dup
    print("\n=== formalize half 실측 (Qwen2.5-32B·user발화→기준→filter) · n=%d · err=%d ===" % (n, len(errs)))
    print("진짜중복 %d (%.1f%%) · 결정가능부 %d\n" % (dup, 100 * dup / max(n, 1), dec))
    by_tid = {(r["tid"], r["model"]): r for r in rows}
    def ndisp(res):
        row = by_tid.get((res["tid"], res.get("model")))
        return (row or {}).get("n_disputes", 1)
    single = [r for r in ok_rows if ndisp(r) <= 1]
    multi = [r for r in ok_rows if ndisp(r) >= 2]
    print("앵커링 분리: 단일-dispute %d · 다중-dispute %d (다중=formalize가 여러 거래 중 앵커 필요)\n" % (len(single), len(multi)))
    print("%-42s  교정  전체%%  결정가능%%  오답  none  | 단일교정%%  다중교정%%" % "config")
    for name in CONFIGS:
        c = Counter(r["cfg"].get(name) for r in ok_rows)
        okc = c.get("ok", 0)
        sok = sum(1 for r in single if r["cfg"].get(name) == "ok")
        mok = sum(1 for r in multi if r["cfg"].get(name) == "ok")
        print("%-42s  %4d  %5.1f  %7.1f   %4d  %4d  | %6.1f    %6.1f"
              % (name, okc, 100 * okc / max(n, 1), 100 * okc / max(dec, 1), c.get("wrong", 0), c.get("none", 0),
                 100 * sok / max(len(single), 1), 100 * mok / max(len(multi), 1)))
    print("\n(perfect-formalize 천장 대비: gold파생 date+merch+type=75.7%%·+amount=81.9%%·결정가능부 100%%)")
    print("결과 → %s" % OUT)


if __name__ == "__main__":
    main()
