#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_turns.py — Claude-user-sim 턴 자동생성(천장 측정용·gpt-4.1 0). task 데이터서 충실 user 턴:
opening(reason 1인칭화) + auth(name/zip) + order(described면 look-up·id-given면 id) + 확정턴들.
출력 {tid:[turns]} → claude_user_batch.py. 사용: python gen_turns.py --tasks 17,20,.. --out turns.json
"""
import json, re, argparse
DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(DOM + "/db.json")); TASKS = {str(t["id"]): t for t in json.load(open(DOM + "/tasks.json"))}
orders, users = db["orders"], db["users"]
DESC = ["look it up", "look up", "don't want to mention", "do not want to mention", "don't want to reveal",
        "do not want to reveal", "don't reveal", "son's", "old address", "default address", "new address",
        "new home", "should be able to look", "in your orders", "in orders profile", "wrong order"]


def naturalize(r):
    r = re.sub(r"\bYou\b", "I", r); r = re.sub(r"\byou\b", "I", r)
    r = re.sub(r"\byour\b", "my", r); r = re.sub(r"\byours\b", "mine", r)
    r = re.sub(r"\bI are\b", "I am", r); r = re.sub(r"\bI want\b", "I want", r)
    return r.strip()


def user_of(t):
    for a in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
        oid = (a.get("arguments") or {}).get("order_id")
        if oid and oid in orders:
            return users.get(orders[oid]["user_id"]), orders[oid]["user_id"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for tid in a.tasks.split(","):
        tid = tid.strip()
        t = TASKS[tid]
        reason = str(t.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))
        u, uid = user_of(t)
        nm = u.get("name", {}) if u else {}
        zip_ = (u.get("address") or {}).get("zip") if u else ""
        # gold order ids
        goids = []
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            oid = (act.get("arguments") or {}).get("order_id")
            if oid and oid not in goids:
                goids.append(oid)
        described = any(k in reason.lower() for k in DESC)
        turns = [f"Hi, I need some help. {naturalize(reason)[:600]}",
                 f"My name is {nm.get('first_name','')} {nm.get('last_name','')} and my zip code is {zip_}."]
        if described:
            turns.append("I don't have the order number handy — please look it up from my orders based on what I described.")
        elif goids:
            turns.append(f"The order{' ids are ' if len(goids)>1 else ' id is '}{', '.join(goids)}.")
        # confirmations (여러 confirm 게이트 대비)
        turns += ["Yes, please go ahead with all of that.",
                  "Yes, I confirm — please proceed.",
                  "Yes, that's correct, proceed.",
                  "Yes, go ahead."]
        out[tid] = turns
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"generated {len(out)} task turn-scripts -> {a.out}")


if __name__ == "__main__":
    main()
