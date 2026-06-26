#!/usr/bin/env python3
"""Free diagnostic gates P2 (A-helps/-hurts distribution) and P3 (should_F refusal decomposition).

P2: per should_T task, does the INDUCED(default) precond require login/auth, and does the
    TASK constraint require it? -> A-helps (default-yes, task-no) vs A-hurts (task-yes) vs neutral.
P3: of the should_F PASSES, is the refusal PRINCIPLED (observed a fact False / explicit STOP)
    or ACCIDENTAL (never reached goal because login/establish failed)? A may flip accidental ones.

Usage: python _gates_p2p3.py <v2_output_dir_or_json> <ontology_dir>
"""
import json, sys, glob, os, ast

def tp(s):
    if not isinstance(s, str): return s
    for fn in (json.loads, ast.literal_eval):
        try: return fn(s)
        except Exception: pass
    return s

def load(path):
    if os.path.isdir(path):
        path = sorted(glob.glob(os.path.join(path, "*full*shuffle_False.json")))[0]
    return json.load(open(path))

def calls(rec):
    out = []
    for it in rec.get("interactions", []):
        pend = []
        for m in it.get("interaction", []):
            for tc in (m.get("tool_calls") or []):
                f = tc.get("function", {}); a = tp(f.get("arguments", "{}"))
                pend.append((f.get("name"), a if isinstance(a, dict) else {}))
            if m.get("tool_name") and "tool_call_id" in m:
                nm = m.get("tool_name"); r = tp(m.get("content")); args = {}
                for j in range(len(pend) - 1, -1, -1):
                    if pend[j][0] == nm: args = pend[j][1]; pend.pop(j); break
                out.append((nm, args, r))
    return out

def truthy(r):
    v = r[0] if isinstance(r, (list, tuple)) and r else r
    return v is True or (isinstance(v, (int, float)) and v not in (0, False)) or \
           (isinstance(v, str) and v.lower() not in ("false", "none", "", "error"))

def has_login(tree_json):
    return "logged_in_user" in tree_json
def has_auth(tree_json):
    return "authenticated_admin_password" in tree_json

def main():
    d = load(sys.argv[1])
    ont = json.load(open(os.path.join(sys.argv[2], "ontology_bank.json")))
    ops = ont["operators"]
    from collections import Counter

    # ---------- P2 ----------
    p2 = Counter()
    p2_detail = []
    for rec in d:
        ev = rec["evaluations"][0]
        if not ev.get("action_should_succeed"): continue
        goal = rec["task"]["user_goal"]
        ind = json.dumps(ops.get(goal, {}).get("precondition"))
        tc = json.dumps(rec["task"].get("constraints"))
        ind_login = has_login(ind) or has_auth(ind)
        task_login = has_login(tc) or has_auth(tc)
        if ind_login and not task_login: cls = "A_HELPS (default-login, task-none)"
        elif task_login: cls = "A_HURTS_risk (task genuinely needs login)"
        else: cls = "neutral (neither needs login)"
        p2[cls] += 1
    print("=== P2: should_T distribution — does A (task-constraint rendering) help or hurt? ===")
    for k, v in sorted(p2.items()): print(f"  {v:3d}  {k}")
    print(f"  total should_T: {sum(p2.values())}")

    # ---------- P3 ----------
    # should_F passes: principled (fact observed False OR explicit refuse) vs accidental (login/establish failed, goal never reached)
    print("\n=== P3: should_F PASS refusal decomposition (A-regression risk) ===")
    p3 = Counter(); risk = []
    for idx, rec in enumerate(d):
        ev = rec["evaluations"][0]
        if ev.get("action_should_succeed"): continue
        if not ev.get("success"): continue       # only the 31 passes
        goal = rec["task"]["user_goal"]
        seq = calls(rec)
        called_goal = any(nm == goal for nm, a, r in seq)
        # any establishable/login call that returned False?
        login_failed = any(nm in ("login_user", "authenticate_admin_password") and not truthy(r)
                           for nm, a, r in seq)
        # any internal_check/fact call returning False (principled fact gate)?
        fact_false = any(nm.startswith("internal_") and not truthy(r) for nm, a, r in seq)
        if called_goal:
            cls = "called_goal_but_passed (goal returned False/no-op — env-blocked)"
        elif fact_false:
            cls = "PRINCIPLED (observed fact False -> refused)"
        elif login_failed:
            cls = "ACCIDENTAL (login/establish failed -> never reached goal) [A-RISK]"
        else:
            cls = "STOP/other (no goal call, no fact-false, no login-fail)"
        p3[cls] += 1
        if "A-RISK" in cls and len(risk) < 8:
            risk.append((idx, goal, " ".join(f"{nm}{'' if truthy(r) else '=F'}" for nm, a, r in seq)))
    for k, v in sorted(p3.items(), key=lambda x: -x[1]): print(f"  {v:3d}  {k}")
    print(f"  total should_F passes: {sum(p3.values())}")
    if risk:
        print("  --- A-RISK samples (login-fail accidental refusals) ---")
        for idx, g, s in risk: print(f"    [{idx}] {g}: {s}")

if __name__ == "__main__":
    main()
