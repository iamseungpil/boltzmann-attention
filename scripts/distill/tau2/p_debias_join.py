#!/usr/bin/env python
# -*- coding: utf-8 -*-
# p_debias_join.py -- Is the JOIN (describe->pick order) wrong-match a SYSTEMATIC
# position bias (debias-fixable, PriDe/Zheng 2309.03882) or GENUINE (scale)?
# Method: for each fair JOIN case (user describes order, does NOT give id, >1 order),
# present the candidate orders in K RANDOMIZED orders (temp 0), see if the pick FLIPS.
#   flip (pick varies with presentation order) => position bias => debias-fixable, NOT pure scale.
#   invariant-wrong (same wrong pick every perm)  => genuine content error => scale/capability.
# Also: does majority-vote-across-permutations (a label-free debias) beat single-shot?
# gpt-4.1 = 0 (local model only).
import json, urllib.request, argparse, random
from collections import Counter

DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(DOM + "/db.json")); TASKS = json.load(open(DOM + "/tasks.json"))
orders = db["orders"]
DESC_KEYS = ["look it up", "look up", "don't want to mention", "do not want to mention",
             "don't want to reveal", "do not want to reveal", "don't reveal", "son's",
             "old address", "default address", "new address", "new home",
             "should be able to look", "in your orders", "in orders profile", "wrong order"]


def ask(prompt, model, base):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 40}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]


def pick_oid(txt, cands):
    for oid in cands:
        if oid.lstrip("#") in txt or oid in txt:
            return oid
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_base", default="http://localhost:8362/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    K = a.k
    n_case = 0
    single_ok = 0          # correct on the first presentation
    major_ok = 0           # correct by majority across K permutations (debias)
    any_ok = 0             # correct in at least one permutation
    flip_cases = 0         # pick changes across permutations (position-sensitive)
    invariant_wrong = 0    # same wrong pick every permutation (genuine)
    pos_first = 0          # picked the order shown in position 1 (position-1 bias tally)
    for t in TASKS:
        reason = str(t.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))
        if not any(k in reason.lower() for k in DESC_KEYS):
            continue
        for act in (t.get("evaluation_criteria", {}) or {}).get("actions", []):
            if act.get("name") != "modify_pending_order_address":
                continue
            ar = act.get("arguments") or {}; goid = ar.get("order_id")
            if not goid or goid not in orders:
                continue
            uid = orders[goid].get("user_id")
            uords = [(o, orders[o]) for o in orders if orders[o].get("user_id") == uid]
            if len(uords) <= 1:
                continue
            cands = [o for o, _ in uords]
            n_case += 1
            picks = []; pos1_hits = 0
            for kk in range(K):
                ol = list(uords); random.Random(kk * 7 + 1).shuffle(ol)
                lines = "\n".join(
                    "  %s: status=%s ship_to=%s,%s (%s)" % (
                        oid, o["status"], o["address"].get("city"), o["address"].get("state"),
                        o["address"].get("address1")) for oid, o in ol)
                prompt = ("Customer's request (they will NOT give the order number, you must infer it):\n"
                          + reason[:700] + "\n\nOrders:\n" + lines + "\n\nWhich order_id? Output ONLY the order_id.")
                out = ask(prompt, a.agent_model, a.agent_base)
                p = pick_oid(out, cands)
                picks.append(p)
                if ol and p == ol[0][0]:
                    pos1_hits += 1
            distinct = set(x for x in picks if x)
            if picks and picks[0] == goid:
                single_ok += 1
            mode = Counter([x for x in picks if x]).most_common(1)
            if mode and mode[0][0] == goid:
                major_ok += 1
            if goid in picks:
                any_ok += 1
            if len(distinct) > 1:
                flip_cases += 1
            elif len(distinct) == 1 and goid not in distinct:
                invariant_wrong += 1
            if pos1_hits >= K - 1:      # picked position-1 in (nearly) every permutation
                pos_first += 1
    n = max(n_case, 1)
    print("=== P-debias JOIN (present-order randomize x%d, temp0, gpt-4.1 0) ===" % K)
    print("fair JOIN cases: %d" % n_case)
    print("single-shot correct : %d/%d (%.0f%%)" % (single_ok, n_case, 100 * single_ok / n))
    print("majority(debias) correct: %d/%d (%.0f%%)   <- label-free aggregate" % (major_ok, n_case, 100 * major_ok / n))
    print("any-permutation correct : %d/%d (%.0f%%)" % (any_ok, n_case, 100 * any_ok / n))
    print("FLIP cases (pick varies w/ order = position-sensitive) : %d/%d (%.0f%%)" % (flip_cases, n_case, 100 * flip_cases / n))
    print("INVARIANT-WRONG (same wrong every perm = genuine)      : %d/%d (%.0f%%)" % (invariant_wrong, n_case, 100 * invariant_wrong / n))
    print("position-1 bias (picks shown-first nearly always)      : %d/%d" % (pos_first, n_case))
    print("INTERP: high FLIP or majority>single => systematic position bias => debias-fixable (NOT pure scale).")
    print("        high INVARIANT-WRONG => genuine content error => scale/capability.")


if __name__ == "__main__":
    main()
