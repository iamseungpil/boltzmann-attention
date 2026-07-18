#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""over-flag 원인 격리 프로브 (무료·2026-07-18 NIGHT+·`RATE_SUBAGENT §2d [?] 확정용`).

iso5서 020/026/028 producer가 12·15·12건 반환(gold 4·4·6) = 격리해도 over-flag. 완주 sim 0이라 궤적
정독 불가([[08]]). ⇒ **격리 서브 프롬프트를 그 태스크 거래에 직접 돌려** 원인을 셋으로 가른다:
  (A) 서브 base_rate 오독      : 서브가 낸 base_rate ≠ gold-implied rate → **모델 부하**([[45]])
  (B) promo 파라미터 오formalize: base는 맞는데 promo_mult/start/end 틀림 → 엔진이 잘못 곱함
  (C) 서브는 맞음·엔진 로직 결함 : 서브 operand 정확한데 engine_rate 합성이 gold와 다름 → 우리 코드

★서브 프롬프트·getter·계약 전부 배포 A2(`isolate`)서 로드(재작성 0). 서브가 스스로 KB 검색(라이브 동형).
gold rate = (dispute 아닌 거래=벤치 옳음) rewards_earned/amount · (dispute 거래) update gold ÷ amount.

Run(리모트): python3 bank_overflag_probe.py --base http://localhost:8140/v1 --rows /home/woori/scratch/task020_rows.json --n 3
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from bank_fab_probes import post  # noqa: E402
import bank_rate_f1_gate_probe as P  # noqa: E402
import t2_scaffold_get as SG  # noqa: E402

A2 = os.path.join(HERE, "a2", "banking_knowledge.gate.json")
DOM = P.DOM_DEFAULT


def isolate_spec():
    spec = json.load(open(A2, encoding="utf-8"))
    found = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("name") == "get_reward_discrepancies" and isinstance(o.get("variants"), dict):
                found.append(o["variants"]["ratefix"]["isolate"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(spec)
    return found[0]


# KB 검색 도구(라이브 동형·env 대신 로컬 문서에서 결정론 검색 — spoon 아님·질의는 서브가 냄)
DOCS = None


def _kb(query, k=5):
    global DOCS
    if DOCS is None:
        dd = os.path.join(DOM, "documents")
        DOCS = [json.load(open(os.path.join(dd, f), encoding="utf-8")) for f in sorted(os.listdir(dd))]
    ql = set(str(query).lower().split())
    scored = sorted(DOCS, key=lambda d: -sum(
        (d.get("title", "") + " " + d.get("content", "")).lower().count(w) for w in ql))
    return "\n\n".join("### %s\n%s" % (d["title"], d["content"]) for d in scored[:int(k or 5)])


KB_TOOL = [{"type": "function", "function": {
    "name": "KB_search_bm25", "description": "Search the bank knowledge base for policy documents.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}}}]


def run_sub(base, model, temp, iso, rows, max_rounds=4):
    raw = [{k: v for k, v in r.items() if k in set(iso["row_fields"])} for r in rows]
    ids = [r["transaction_id"] for r in rows]
    prompt = "%s\n\n=== ITEMS ===\n%s\n\n%s" % (
        iso["instructions"], json.dumps(raw, ensure_ascii=False, indent=1),
        iso["answer_format"].format(schema=json.dumps({i: iso.get("operand_schema", {}) for i in ids},
                                                      ensure_ascii=False)))
    msgs = [{"role": "user", "content": prompt}]
    getter = 0
    for rnd in range(max_rounds):
        payload = {"model": model, "messages": msgs, "tools": KB_TOOL, "temperature": temp,
                   "max_tokens": 4000, "n": 1}
        if rnd == 0:
            payload["tool_choice"] = "required"
        r = post(base, payload, timeout=600)
        m = r["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        if tcs:
            msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": tcs})
            for tc in tcs:
                getter += 1
                try:
                    a = json.loads(tc["function"]["arguments"])
                except Exception:
                    a = {}
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", "c"),
                             "content": _kb(a.get("query", ""), a.get("k", 5))})
            continue
        return SG._merge_json(m.get("content") or "", set(ids)), getter, rnd + 1
    return None, getter, max_rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()

    data = json.load(open(a.rows, encoding="utf-8"))
    rows = data["rows"]
    disputed = set(data["disputed"])
    gold = {r["transaction_id"]: r for r in P.load_gold(DOM)}
    iso = isolate_spec()
    print("★over-flag 원인 프로브 — 거래 %d · dispute %d · 카드 %s" % (len(rows), len(disputed), list(data["accts"])))
    print("  분기: (A)서브 base 오독 / (B)promo 오formalize / (C)엔진결함\n")

    for i in range(a.n):
        try:
            sub, getter, rnds = run_sub(a.base, a.model, a.temp, iso, rows)
        except Exception as e:
            print("  [%d] ERR %r" % (i, str(e)[:80]))
            continue
        if not sub:
            print("  [%d] 서브 파싱실패 (getter %d·%d라운드)" % (i, getter, rnds))
            continue
        base_bad = promo_bad = eng_bad = flagged = 0
        detail = []
        for r in rows:
            tid = r["transaction_id"]
            g = gold.get(tid)
            v = sub.get(tid) or {}
            br = v.get("base_rate")
            amt = r["transaction_amount"]
            if g is None or not isinstance(br, (int, float)):
                continue
            gold_rate = g["gold_pts"] / amt
            # (A) 서브 base_rate가 gold rate의 프로모-제거값과 맞나? gold_rate ∈ {base, base*promo}
            base_ok = abs(br - gold_rate) < 0.01 or abs(br * 2 - gold_rate) < 0.01 or \
                (gold_rate == 0 and br == 0)
            # 엔진 합성
            rate = SG_engine(v, r)
            pts = amt * rate
            eng_ok = abs(pts - g["gold_pts"]) <= 1
            flag = not eng_ok
            flagged += flag
            if not base_ok:
                base_bad += 1
                detail.append(("A base오독", tid[-8:], r["credit_card_type"][:14], r["category"],
                               "sub_base=%s gold_rate=%.1f" % (br, gold_rate)))
            elif not eng_ok:
                # base는 맞는데 최종 틀림 → promo/엔진
                promo_bad += 1
                detail.append(("B/C promo/엔진", tid[-8:], r["credit_card_type"][:14], r["category"],
                               "base=%s pmult=%s pstart=%s → pts=%.0f gold=%.0f"
                               % (br, v.get("promo_mult"), v.get("promo_start"), pts, g["gold_pts"])))
        print("  [%d] getter=%d·%dR · base오독=%d · promo/엔진틀림=%d · 총flagged=%d(gold discrepant=%d)"
              % (i, getter, rnds, base_bad, promo_bad, flagged, len(disputed)))
        for d in detail[:8]:
            print("       %-14s %s %-14s %-11s %s" % d)


def SG_engine(v, r):
    return P.engine_rate(v.get("base_rate"), bool(v.get("promo_start")), v.get("promo_mult", 1),
                         r.get("account_open"), r.get("transaction_date"),
                         v.get("promo_window_months", 6), v.get("promo_start"), v.get("promo_end"))


if __name__ == "__main__":
    main()
