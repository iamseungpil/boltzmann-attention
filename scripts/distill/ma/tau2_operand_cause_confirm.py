#!/usr/bin/env python3
"""operand 실패의 *진짜* 잔여원인 전수확정 — "decomp+snap만 남았나"를 *시뮬레이션*으로 검증.
케이스별로: ①enum-snap 실제 적용→재해결(snap이 고치나) ②남으면 missing_key(decomp-target)인가
spurious-key/op-wrong/anchor-fail/제3원인인가 전수분류. = 주장 아닌 확인.

τ² 케이스(catalog enum) + gate① rows(emitted op_ir) 조인.
Usage: tau2_operand_cause_confirm.py <cases.jsonl> <rows.json> [tag]
"""
import json, sys
from collections import Counter


def norm(v):
    return str(v).strip().lower().replace("_", " ").replace("-", " ")


def enum_values(catalog, attr):
    return list({it["options"][attr] for it in catalog if attr in it.get("options", {})})


def snap(v, enums):
    """emitted 값 v를 카탈로그 enum 최근접으로 스냅."""
    if not enums:
        return v, "no_enum"
    for e in enums:
        if norm(e) == norm(v):
            return e, "exact"
    # substring 양방향
    cand = [e for e in enums if norm(v) in norm(e) or norm(e) in norm(v)]
    if len(cand) == 1:
        return cand[0], "substr"
    # token overlap
    tv = set(norm(v).split())
    best, bn = None, 0
    for e in enums:
        ov = len(tv & set(norm(e).split()))
        if ov > bn:
            best, bn = e, ov
    if best and bn > 0:
        return best, "token"
    return v, "no_snap"


def resolve(catalog, target):
    """target(attr→val) 완전매치 item_id (유일하면)."""
    m = [it for it in catalog if all(norm(it["options"].get(k)) == norm(v) for k, v in target.items())]
    av = [it for it in m if it.get("available")]
    pool = av or m
    return pool[0]["item_id"] if len(pool) == 1 else None


def main():
    cases = {}
    for l in open(sys.argv[1], encoding="utf-8"):
        c = json.loads(l)
        for ex in c["exchanges"]:
            cases[(c["task_id"], ex["old_item_id"])] = (c, ex)
    rows = json.load(open(sys.argv[2], encoding="utf-8")).get("rows", [])
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    cls = Counter()
    ex_by = {}
    snap_kinds = Counter()
    n = 0
    for r in rows:
        if r.get("hit"):
            cls["already_ok"] += 1
            n += 1
            continue
        n += 1
        ir = r.get("ir") or {}
        op = ir.get("op")
        # 케이스 카탈로그 찾기 (task + old item)
        key = None
        for (tid, oid) in cases:
            if tid == r.get("task") and oid == r.get("gold") or tid == r.get("task"):
                key = (tid, oid)
                if oid == r.get("ir", {}).get("anchor_id"):
                    break
        # fallback: task만으로 첫 케이스
        cobj = None
        for (tid, oid), v in cases.items():
            if tid == r.get("task"):
                cobj = v
                break
        if cobj is None:
            cls["case_not_found"] += 1
            continue
        c, ex = cobj
        cat = ex["variant_catalog"]
        old = ex.get("old_options", {})
        gold_id = ex["gold_new_item_id"]
        catalog_attrs = set().union(*[set(it.get("options", {})) for it in cat]) if cat else set()

        if op not in ("substitute", "create"):
            cls["op_wrong"] += 1
            ex_by.setdefault("op_wrong", f"task{r.get('task')} op={op}")
            continue
        eset = ir.get("set") or {}
        # spurious key (카탈로그 attr 아님)
        spurious = [k for k in eset if k not in catalog_attrs]
        if spurious:
            cls["spurious_key"] += 1
            ex_by.setdefault("spurious_key", f"task{r.get('task')} spurious={spurious} set={eset}")
            continue
        # enum-snap 적용
        snapped = {}
        for k, v in eset.items():
            sv, kind = snap(v, enum_values(cat, k))
            snapped[k] = sv
            snap_kinds[kind] += 1
        base_for = dict(old) if op == "substitute" else {}
        target = dict(base_for); target.update(snapped)
        rid2 = resolve(cat, target)
        if rid2 == gold_id:
            cls["SNAP_FIXES"] += 1
            ex_by.setdefault("SNAP_FIXES", f"task{r.get('task')} set={eset}->snap={snapped}")
            continue
        # snap 후에도 실패 → 잔여
        gold_set = {k: v for k, v in ex["gold_new_options"].items() if norm(old.get(k)) != norm(v)}
        miss = set(gold_set) - set(eset)
        if miss:
            cls["needs_DECOMP(under-extract)"] += 1
            ex_by.setdefault("needs_DECOMP(under-extract)", f"task{r.get('task')} miss={sorted(miss)} emit={eset} gold={gold_set}")
        else:
            cls["RESIDUAL_OTHER"] += 1
            ex_by.setdefault("RESIDUAL_OTHER", f"task{r.get('task')} op={op} emit={eset} snap={snapped} gold={gold_set} rid2={rid2} gold_id={gold_id}")
    print(f"\n===== {tag}  (n={n}) =====")
    for k, v in cls.most_common():
        print(f"  {k:>28}: {v}")
    print("  snap kinds:", dict(snap_kinds))
    for k, e in ex_by.items():
        print(f"    ex[{k}]: {str(e)[:200]}")


if __name__ == "__main__":
    main()
