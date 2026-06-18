#!/usr/bin/env python3
"""facet (3) gate① operand 전수 궤적조사 — new_item_id miss의 *진짜 원인* (aggregate 아님).
저장된 rows(op_ir·old_options·gold_options·gold_set·emitted_set·rid·gold·hit)서 케이스별 분류.

분류 (substitute/create 기준):
  resolved_ok       = hit (정답)
  op_wrong          = emitted op != substitute/create (op 자체 틀림)
  no_set            = set 비었음(emit 안 함)
  missing_key       = gold_set 키 중 일부 누락 (§20-B 과소추출)
  extra_key         = gold_set에 없는 키 변경 (과추출)
  wrong_value       = 키는 맞고 값 틀림 → enum-normalize 가능성 별도 표기
  set_ok_resolve_fail = emitted_set == gold_set 인데 rid != gold (anchor/resolver/카탈로그 문제)
Usage: tau2_native_operand_autopsy.py <result_dir>
"""
import json, sys, glob, os
from collections import Counter


def norm(v):
    return str(v).strip().lower()


def classify(row):
    op = row.get("op")
    if row.get("hit"):
        return "resolved_ok", None
    if op not in ("substitute", "create"):
        return "op_wrong", f"op={op}"
    emit = row.get("emitted_set") or {}
    gold = row.get("gold_set") or {}
    if not emit:
        return "no_set", f"gold_set={gold}"
    ekeys, gkeys = set(emit), set(gold)
    miss = gkeys - ekeys
    extra = ekeys - gkeys
    wrong = {k: (emit[k], gold[k]) for k in (ekeys & gkeys) if norm(emit[k]) != norm(gold[k])}
    # set이 gold와 동일한데 miss면 resolve/anchor 문제
    if emit and not miss and not extra and not wrong:
        return "set_ok_resolve_fail", f"emit==gold={emit} rid={row.get('rid')} gold={row.get('gold')}"
    if miss and not wrong and not extra:
        return "missing_key", f"missing={sorted(miss)} emit={emit} gold={gold}"
    if extra and not miss and not wrong:
        return "extra_key", f"extra={sorted(extra)} emit={emit} gold={gold}"
    if wrong and not miss and not extra:
        return "wrong_value", f"wrong={wrong}"
    return "mixed_error", f"miss={sorted(miss)} extra={sorted(extra)} wrong={wrong} emit={emit} gold={gold}"


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "/home/woori/scratch/depth/c8/facet3/results"
    for tag in ["base", "trained"]:
        for dom in ["retail", "airline"]:
            f = os.path.join(rdir, f"t2native_{tag}__{dom}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f, encoding="utf-8"))
            rows = d.get("rows", [])
            cls = Counter()
            examples = {}
            for r in rows:
                c, detail = classify(r)
                cls[c] += 1
                if c not in ("resolved_ok",) and c not in examples and detail:
                    examples[c] = (r.get("task"), detail)
            n = len(rows)
            ok = cls.get("resolved_ok", 0)
            print(f"\n=== {tag} {dom}  (n={n}, resolved_ok={ok}={ok/max(n,1):.2f}) ===")
            for c, k in cls.most_common():
                if c == "resolved_ok":
                    continue
                print(f"  {c:>20}: {k}")
            for c, (task, det) in examples.items():
                print(f"    ex[{c}] task={task}: {str(det)[:200]}")


if __name__ == "__main__":
    main()
