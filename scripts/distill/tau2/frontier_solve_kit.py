#!/usr/bin/env python
"""frontier_solve_kit — Opus-as-agent로 retail task 풀이/검증 (HANDOFF_2026_06_22_PM §2).

--brief: task 시나리오 + 실데이터(user record·orders·관련 product 변종 카탈로그) 출력. ★gold 비공개(무결성).
--check '<json actions>': 내 행동을 gold(evaluation_criteria.actions)와 대조 + gold 공개.
both-fail 11개(2,3,4,20,21,99,100,103,105,109,110) 전수풀이용.

Run:
  PY frontier_solve_kit.py --task 99 --brief
  PY frontier_solve_kit.py --task 99 --check '[{"name":"...","arguments":{...}}]'
"""
import argparse
import json
import os

RETAIL = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"


def _db():
    return json.load(open(os.path.join(RETAIL, "db.json"), encoding="utf-8"))


def _tasks():
    return {str(t.get("id")): t for t in json.load(open(os.path.join(RETAIL, "tasks.json"), encoding="utf-8"))}


def _find_user(db, name, zip_):
    users = db.get("users", {})
    name_l = (name or "").lower()
    for uid, u in users.items():
        nm = u.get("name", {})
        full = f"{nm.get('first_name','')} {nm.get('last_name','')}".lower()
        z = str((u.get("address") or {}).get("zip", ""))
        if full.strip() and full.strip() in name_l and (not zip_ or z == str(zip_)):
            return uid, u
    # zip-only fallback
    for uid, u in users.items():
        if str((u.get("address") or {}).get("zip", "")) == str(zip_):
            return uid, u
    return None, None


def brief(tid):
    db = _db(); t = _tasks()[tid]
    ins = t["user_scenario"]["instructions"]
    print("=" * 70)
    print(f"TASK {tid}  (actions in gold = {len((t.get('evaluation_criteria') or {}).get('actions') or [])}, HIDDEN)")
    print("=" * 70)
    print("INSTRUCTIONS:\n", json.dumps(ins, ensure_ascii=False, indent=1) if isinstance(ins, dict) else ins)
    # user from known_info (name/zip) — 휴리스틱 추출
    ki = (ins.get("known_info") if isinstance(ins, dict) else "") or ""
    import re
    zipm = re.search(r"\b(\d{5})\b", ki)
    namem = re.search(r"(?:name is|are|You are)\s+([A-Z][a-z]+ [A-Z][a-z]+)", ki)
    uid, u = _find_user(db, namem.group(1) if namem else "", zipm.group(1) if zipm else "")
    if not u:
        print("\n[!] user 자동매칭 실패 — known_info 직접 보고 db.json users 검색 요.")
        return
    print(f"\n=== USER {uid} ===")
    print("payment_methods:", list((u.get("payment_methods") or {}).keys()))
    print("addresses:", json.dumps(u.get("address"), ensure_ascii=False))
    orders = db.get("orders", {})
    pids = set()
    for oid in (u.get("orders") or []):
        o = orders.get(oid) or {}
        print(f"\n--- ORDER {oid} status={o.get('status')} ---")
        for it in (o.get("items") or []):
            print(f"   {it.get('name')} prod={it.get('product_id')} item={it.get('item_id')} ${it.get('price')} {it.get('options')}")
            pids.add(it.get("product_id"))
    # 관련 product 변종 카탈로그
    prods = db.get("products", {})
    for pid in pids:
        p = prods.get(pid) or {}
        print(f"\n=== PRODUCT {p.get('name')} ({pid}) variants ===")
        for vid, v in (p.get("variants") or {}).items():
            print(f"   {vid} | {v.get('options')} avail={v.get('available')} ${v.get('price')}")


def _norm(a):
    """행동 정규화 (name + 핵심 args·item_ids는 정렬해 순서무관)."""
    args = a.get("arguments") or {}
    key = {"name": a.get("name")}
    for k, v in args.items():
        if isinstance(v, list):
            key[k] = sorted(str(x) for x in v)
        else:
            key[k] = str(v)
    return json.dumps(key, sort_keys=True, ensure_ascii=False)


def check(tid, actions_json):
    t = _tasks()[tid]
    gold = (t.get("evaluation_criteria") or {}).get("actions") or []
    mine = json.loads(actions_json)
    gset = {_norm(g) for g in gold}
    mset = {_norm(m) for m in mine}
    print("=== GOLD actions ===")
    for g in gold:
        print("  ", g.get("name"), json.dumps(g.get("arguments"), ensure_ascii=False)[:200])
    print("\n=== MINE ===")
    for m in mine:
        print("  ", m.get("name"), json.dumps(m.get("arguments"), ensure_ascii=False)[:200])
    matched = gset & mset
    print(f"\nMATCH: {len(matched)}/{len(gold)} gold actions matched · extra(mine-gold)={len(mset-gset)}")
    print("VERDICT:", "SOLVED ✓" if gset == mset else ("GOLD⊆MINE(+extra)" if gset <= mset else "MISS"))
    if gset - mset:
        print("MISSING gold:", [json.loads(x).get("name") for x in (gset - mset)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--check", default=None)
    a = ap.parse_args()
    if a.brief:
        brief(a.task)
    elif a.check is not None:
        check(a.task, a.check)


if __name__ == "__main__":
    main()
