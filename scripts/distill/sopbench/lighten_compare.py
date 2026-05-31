#!/usr/bin/env python3
"""Compare baseline v2 vs LIGHTEN (mechanism A zero-train) — behavioral, per-task.

Measures the mechanism's direct target (login/auth over-calling) and the gate criteria
(should_T /40, should_F fragile regression). Fabrication-safe: reports raw per-task flips.
"""
import json, ast, sys

BASE = "output_v4a_v2/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
LITE = "output_v4a_v2_lighten/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
ONT = "induced/ontology_bank.json"

def tp(s):
    if not isinstance(s, str): return s
    for fn in (json.loads, ast.literal_eval):
        try: return fn(s)
        except Exception: pass
    return s

def calls(rec):
    out = []
    for it in rec.get("interactions", []):
        for m in it.get("interaction", []):
            for tc in (m.get("tool_calls") or []):
                out.append(tc.get("function", {}).get("name"))
    return out

ont = json.load(open(ONT)); ops = ont["operators"]
def task_needs_login(rec):
    tc = json.dumps(rec["task"].get("constraints"))
    return "logged_in_user" in tc or "authenticated_admin_password" in tc

b = json.load(open(BASE)); l = json.load(open(LITE))
FRAGILE = [93, 96, 100, 101, 104]   # P3 accidental-refusal should_F passes

# A_HELPS tasks: should_T, default-precond needs login but task-constraint doesn't
A_HELPS = []
for i, rec in enumerate(b):
    ev = rec["evaluations"][0]
    if not ev.get("action_should_succeed"): continue
    ind = json.dumps(ops.get(rec["task"]["user_goal"], {}).get("precondition"))
    ind_login = "logged_in_user" in ind or "authenticated_admin_password" in ind
    if ind_login and not task_needs_login(rec): A_HELPS.append(i)

def login_calls(rec):
    c = calls(rec); return c.count("login_user") + c.count("authenticate_admin_password")

# overall + per-bucket
def summarize(tag, data):
    st = stp = sf = sfp = 0; login_total = 0
    for rec in data:
        ev = rec["evaluations"][0]; ok = bool(ev.get("success")); login_total += login_calls(rec)
        if ev.get("action_should_succeed"): st += 1; stp += ok
        else: sf += 1; sfp += ok
    print(f"  {tag}: should_T {stp}/{st}  should_F {sfp}/{sf}  total login/auth calls={login_total}")
    return stp, sfp

print("=== OVERALL ===")
bt, bf = summarize("BASELINE", b)
lt, lf = summarize("LIGHTEN ", l)

print(f"\n=== A_HELPS tasks (n={len(A_HELPS)}): login/auth calls + should_T pass ===")
b_login = sum(login_calls(b[i]) for i in A_HELPS); l_login = sum(login_calls(l[i]) for i in A_HELPS)
b_pass = sum(b[i]["evaluations"][0].get("success") for i in A_HELPS)
l_pass = sum(l[i]["evaluations"][0].get("success") for i in A_HELPS)
print(f"  login/auth CALLS on A_HELPS: baseline {b_login} -> lighten {l_login}")
print(f"  should_T PASS on A_HELPS:    baseline {b_pass}/{len(A_HELPS)} -> lighten {l_pass}/{len(A_HELPS)}")
for i in A_HELPS:
    bc, lc = login_calls(b[i]), login_calls(l[i])
    bs, ls = bool(b[i]["evaluations"][0].get("success")), bool(l[i]["evaluations"][0].get("success"))
    flag = "  <-- changed" if (bc != lc or bs != ls) else ""
    print(f"    task {i:3d} {b[i]['task']['user_goal']:22s} login {bc}->{lc}  pass {bs}->{ls}{flag}")

print(f"\n=== should_F FRAGILE (P3 accidental, n={len(FRAGILE)}): regression check ===")
for i in FRAGILE:
    bs, ls = bool(b[i]["evaluations"][0].get("success")), bool(l[i]["evaluations"][0].get("success"))
    flag = "  <-- REGRESSED" if (bs and not ls) else ""
    print(f"    task {i:3d} {b[i]['task']['user_goal']:18s} pass {bs}->{ls}{flag}")

print("\n=== per-task flips (should_T) ===")
gains = []; losses = []
for i, (rb, rl) in enumerate(zip(b, l)):
    if not rb["evaluations"][0].get("action_should_succeed"): continue
    bs, ls = bool(rb["evaluations"][0].get("success")), bool(rl["evaluations"][0].get("success"))
    if ls and not bs: gains.append(i)
    if bs and not ls: losses.append(i)
print(f"  should_T GAINED (fail->pass): {gains}")
print(f"  should_T LOST   (pass->fail): {losses}")
