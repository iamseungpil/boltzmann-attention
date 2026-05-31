#!/usr/bin/env python3
"""§8.1 binding-constraint diagnosis for arm-4a should_T failures.

Fixes the census_shouldT.py default-precond bug (it counted prerequisites from the
DOMAIN-DEFAULT ontology precondition; the real requirement is task["constraints"]).
ALSO fixes the dedup-by-name blind spot that hid transfer_funds' missing DESTINATION
check: required checks are matched ARGS-AWARE (same tool name + different resolved arg
value = two distinct required calls), so "missing dest check" is its own bucket instead
of being swallowed by "dirgraph_violation".

Produces three things the §8.1 review asked for:
  1. should_T failure re-census by TASK constraint (per-task + tally), baseline & lighten
  2. task 111 (transfer_funds) trajectory baseline vs lighten (R8)
  3. should_F GROSS flips baseline->lighten (gain + loss), not just net (R6)

Usage: python binding_diag.py <baseline_json> <lighten_json> <ontology_bank.json>
       (defaults to the /home/woori/scratch/SOPBench arm-4a v2 / lighten outputs)
"""
import json, ast, os, sys
from collections import Counter

ROOT = "/home/woori/scratch/SOPBench"
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "output_v4a_v2/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
LITE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    ROOT, "output_v4a_v2_lighten/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
ONT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "induced/ontology_bank.json")

ont = json.load(open(ONT)); OPS = ont["operators"]; PREDS = ont["predicates"]
GK = ["action_successfully_called", "dirgraph_satisfied", "action_called_correctly",
      "constraint_not_violated", "database_match", "no_tool_call_error"]


def tp(s):
    if not isinstance(s, str): return s
    for fn in (json.loads, ast.literal_eval):
        try: return fn(s)
        except Exception: pass
    return s


def walk(tree, acc):
    if not tree: return
    if isinstance(tree, (list, tuple)) and tree:
        h = tree[0]
        if h == "single":
            nm = tree[1]; pm = tree[2] if len(tree) > 2 else {}; neg = nm.startswith("not ")
            acc.append((nm[4:] if neg else nm, pm, neg))
        elif h in ("and", "or", "chain", "gate"):
            for s in tree[1]: walk(s, acc)


def calls(rec):
    out = []
    for it in rec.get("interactions", []):
        pend = []
        for m in it.get("interaction", []):
            for tc in (m.get("tool_calls") or []):
                f = tc.get("function", {}); a = tp(f.get("arguments", "{}"))
                pend.append((f.get("name"), a if isinstance(a, dict) else {}))
            if m.get("tool_name") is not None and "tool_call_id" in m:
                nm = m.get("tool_name"); r = tp(m.get("content"))
                for j in range(len(pend) - 1, -1, -1):
                    if pend[j][0] == nm: out.append((nm, pend[j][1], r)); pend.pop(j); break
                else: out.append((nm, {}, r))
    return out


def truthy(r):
    v = r[0] if isinstance(r, (list, tuple)) and r else r
    return v is True or (isinstance(v, (int, float)) and v not in (0, False)) or \
        (isinstance(v, str) and v.lower() not in ("false", "none", "", "error"))


def census(data, tag):
    tal = Counter(); detail = []
    for i, rec in enumerate(data):
        ev = rec["evaluations"][0]
        if not ev.get("action_should_succeed") or ev.get("success"): continue
        goal = rec["task"]["user_goal"]
        uk = rec["task"].get("user_known", {})
        leaves = []; walk(rec["task"].get("constraints"), leaves)
        g = {k: ev.get(k) for k in GK}
        seq = calls(rec)
        # ARGS-AWARE: a required check leaf (name, param->slot) is satisfied iff some actual
        # call of `name` had arg values matching uk[slot] for every param.
        checks, ests = [], []
        for nm, pm, neg in leaves:
            info = PREDS.get(nm, {})
            if info.get("kind") == "establishable" and info.get("by"):
                ests.append((nm, info["by"]))
            else:
                checks.append((nm, pm))
        def check_satisfied(nm, pm):
            exp = {p: uk.get(s) for p, s in pm.items()}
            for cn, ca, cr in seq:
                if cn != nm: continue
                if all(ca.get(p) == v for p, v in exp.items() if v is not None):
                    return True
            return False
        miss_checks = []
        for nm, pm in checks:
            if not check_satisfied(nm, pm):
                # label by the distinguishing slot value (e.g. destination_username) when same-name dup
                tag2 = "+".join(f"{p}={s}" for p, s in pm.items()) if pm else nm
                miss_checks.append(f"{nm}({tag2})")
        called = {}
        for nm, a, r in seq: called[nm] = (a, truthy(r))
        goal_called = goal in called
        goal_true = bool(called.get(goal, (None, False))[1])
        miss_est = [(c, by) for c, by in ests if by not in called or not called[by][1]]
        login_failed = any(nm in ("login_user", "authenticate_admin_password") and not truthy(r)
                           for nm, _, r in seq)
        # bucket (priority)
        if not g["no_tool_call_error"]:
            cat = "tool_call_error"
        elif miss_checks:
            cat = f"MISSING_CHECK {sorted(set(miss_checks))}"
        elif miss_est and login_failed:
            cat = "EST_FAILED (login/auth returned False — cred hallucinated/absent)"
        elif miss_est:
            cat = "EST_SKIPPED (login/auth required but not called)"
        elif not goal_called:
            cat = "goal_skipped (all checks+est ok, goal never called)"
        elif goal_called and not goal_true:
            cat = "goal_false (goal returned False/err)"
        elif goal_true and not g["dirgraph_satisfied"]:
            cat = "dirgraph_other (goal ok, checks ok, dirgraph still F — ORDER/extra-call?)"
        elif goal_true and not g["constraint_not_violated"]:
            cat = "constraint_violation"
        elif goal_true and not g["database_match"]:
            cat = "database_mismatch"
        else:
            cat = "other"
        tal[cat] += 1
        detail.append((i, goal, cat, " -> ".join(
            f"{nm}{'' if truthy(r) else '=F'}" for nm, _, r in seq)))
    print(f"\n########## {tag}: should_T FAILURE re-census (TASK-CONSTRAINT, args-aware) ##########")
    for i, gl, cat, s in detail:
        print(f"[{i:3d}] {gl:24s} {cat}")
        print(f"       seq={s[:200]}")
    print(f"  --- {tag} tally (n={sum(tal.values())}) ---")
    for k, v in tal.most_common(): print(f"   {v:3d}  {k}")
    return tal


def main():
    b = json.load(open(BASE)); l = json.load(open(LITE))
    census(b, "BASELINE"); census(l, "LIGHTEN")

    print("\n########## TASK 111 (transfer_funds) trajectory: baseline vs lighten (R8) ##########")
    for tag, rec in [("BASE", b[111]), ("LITE", l[111])]:
        ev = rec["evaluations"][0]
        print(f"-- {tag}: success={ev.get('success')} gates={ {k: ev.get(k) for k in GK} }")
        print(f"   constraints={json.dumps(rec['task'].get('constraints'))}")
        print(f"   user_known={rec['task'].get('user_known', {})}")
        for nm, a, r in calls(rec):
            print(f"     {nm}({a}) -> {str(r)[:80]}")

    print("\n########## should_F GROSS flips baseline->lighten (R6) ##########")
    def short(rec): return " -> ".join(f"{nm}{'' if truthy(r) else '=F'}" for nm, _, r in calls(rec))
    gain, loss = [], []
    for i, (rb, rl) in enumerate(zip(b, l)):
        if rb["evaluations"][0].get("action_should_succeed"): continue
        bs = bool(rb["evaluations"][0].get("success")); ls = bool(rl["evaluations"][0].get("success"))
        if ls and not bs: gain.append(i)
        if bs and not ls: loss.append(i)
    print(f"GAIN (fail->pass) n={len(gain)}: {gain}")
    for i in gain: print(f"   [{i}] {l[i]['task']['user_goal']}: {short(l[i])[:160]}")
    print(f"LOSS (pass->fail) n={len(loss)}: {loss}")
    for i in loss:
        print(f"   [{i}] {b[i]['task']['user_goal']}: base={short(b[i])[:110]} || lite={short(l[i])[:110]}")


if __name__ == "__main__":
    main()
