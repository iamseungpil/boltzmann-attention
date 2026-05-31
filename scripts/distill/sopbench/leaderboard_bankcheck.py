#!/usr/bin/env python3
"""Check released SOPBench bank results vs our 'credential-absent unpassable' ceiling claim.

Classifies the 48 should_T bank tasks into buckets (Z impossible8, C cred-absent,
A no-cred, B cred-available) and, for every released model file, reports pass on
should_T overall and on bucket-C specifically. Falsification test of the ceiling.
"""
import json, glob, os

ROOT = "/home/woori/scratch/SOPBench"
ont = json.load(open(f"{ROOT}/induced/ontology_bank.json"))
ops = ont["operators"]
IMP = {"cancel_credit_card", "pay_bill_with_credit_card"}

def needed_creds(goal):
    s = json.dumps(ops.get(goal, {}).get("precondition"))
    nc = []
    if "logged_in_user" in s: nc.append("identification")
    if "authenticated_admin_password" in s: nc.append("admin_password")
    return nc

def bucket(rec):
    goal = rec["task"]["user_goal"]; uk = rec["task"].get("user_known", {}) or {}
    nc = needed_creds(goal)
    if goal in IMP: return "Z_impossible8"
    if not nc: return "A_no_cred"
    if all(c in uk for c in nc): return "B_cred_avail"
    return "C_cred_ABSENT"

# build per-index bucket map from any file (task set identical across runs)
ref = json.load(open(glob.glob(f"{ROOT}/output/bank/*claude-3-7*fc-*tool_full*shuffle_False.json")[0]))
idx_bucket = {}
for i, rec in enumerate(ref):
    if rec["evaluations"][0].get("action_should_succeed"):
        idx_bucket[i] = bucket(rec)
from collections import Counter
print("bucket sizes (should_T):", dict(Counter(idx_bucket.values())))
cabs = [i for i, b in idx_bucket.items() if b == "C_cred_ABSENT"]
print("C_cred_ABSENT indices:", cabs)
print()

rows = []
for f in sorted(glob.glob(f"{ROOT}/output/bank/*.json")):
    name = os.path.basename(f).replace("ast_", "").replace("-dep_full-fmt_structured", "")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if len(d) < 48:
        continue
    st_tot = st_pass = c_tot = c_pass = b_tot = b_pass = 0
    for i, rec in enumerate(d):
        evs = rec.get("evaluations")
        if not evs:
            continue
        ev = evs[0]
        if not ev.get("action_should_succeed"):
            continue
        st_tot += 1; ok = bool(ev.get("success")); st_pass += ok
        bk = idx_bucket.get(i)
        if bk == "C_cred_ABSENT": c_tot += 1; c_pass += ok
        if bk == "B_cred_avail": b_tot += 1; b_pass += ok
    rows.append((st_pass / st_tot if st_tot else 0, name, st_pass, st_tot, c_pass, c_tot, b_pass, b_tot))

rows.sort(reverse=True)
print(f"{'model':<62} {'shouldT':>9} {'bucketC':>8} {'bucketB':>8}")
for r in rows:
    print(f"{r[1]:<62} {r[2]:>3}/{r[3]:<3}={100*r[0]:4.0f}% {r[4]:>2}/{r[5]:<2} {r[6]:>2}/{r[7]:<2}")
