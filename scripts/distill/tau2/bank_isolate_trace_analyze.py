#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""격리 서브 operand trace 분석 (무료·2026-07-18) — over-flag 원인 확정.
trace JSONL(`T2_SG_ISOLATE_TRACE`)의 서브 산출 base_rate를 gold와 대조:
  (A) 서브 base_rate 오독 → over-flag = 서브 문제(검색부실 or 환각)
  (B) 서브 base_rate 정확한데 producer가 over-flag → 엔진/promo 로직 or 다른 원인

Run: python3 bank_isolate_trace_analyze.py <task_id> <trace.jsonl[.gz]>
"""
import json
import gzip
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import bank_rate_f1_gate_probe as P


def main():
    tid = sys.argv[1]
    path = sys.argv[2]
    op = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    recs = [json.loads(x) for x in op if x.strip()]
    gold = {r["transaction_id"]: r for r in P.load_gold(P.DOM_DEFAULT)}
    print("★trace 레코드 %d개 (task=%s)" % (len(recs), tid))
    for i, rec in enumerate(recs):
        if rec.get("error"):
            print("\n[%d] ERROR=%s · queries=%s" % (i, rec["error"], rec.get("queries")))
            continue
        ops = rec.get("operands") or {}
        print("\n[%d] round=%s getter=%s queries=%s · operand %d/%d행"
              % (i, rec.get("round"), rec.get("getter"), rec.get("queries"),
                 rec.get("n_operand"), rec.get("n_ids")))
        bad = 0
        for txn, v in ops.items():
            g = gold.get(txn)
            if not g:
                continue
            br = v.get("base_rate") if isinstance(v, dict) else None
            gr = g["gold_pts"] / g["amount"]
            if not isinstance(br, (int, float)):
                print("   ? %s base_rate=%r (숫자아님)" % (txn[-8:], br))
                bad += 1
                continue
            ok = abs(br - gr) < 0.01 or abs(br * 2 - gr) < 0.01 or (gr == 0 and br == 0)
            if not ok:
                bad += 1
                print("   ✗ %s %-28s %-11s sub_base=%s promo=%s → gold_rate=%.1f"
                      % (txn[-8:], g["card"][:28], g["category"], br, v.get("promo_mult"), gr))
        print("   ★서브 base_rate 오독 %d/%d" % (bad, len(ops)))
    print("\n판정: 오독 多 → (A)서브 문제(검색/환각) · 오독 0인데 producer over → (B)엔진/promo")


if __name__ == "__main__":
    main()
