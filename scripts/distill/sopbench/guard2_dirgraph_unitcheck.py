#!/usr/bin/env python
"""Guard-2 unit-verify (GUARD2_DIRGRAPH_MIRROR_DESIGN): is the Option-A reconstructed dirgraph
leaf-identical to the evaluator's task["directed_action_graph"]? Cause-1 precondition.

A1 reconstruction = the SAME builder the generator used (generation.py:1287):
  dfsgather_invfunccalldirgraph(task["constraints_original"], cl, cp, default_dep, action_params, goal_node)
INPUT AUDIT (Refinement-1): inputs are ONLY {goal, task["constraints_original"], domain dep rules}.
  task["directed_action_graph"] is NEVER read for reconstruction -> not oracle (Option A legitimate).

PASS criteria (Refinement-2/3, asymmetric):
  SAFETY (ship gate)  = current-BOTH (ALL, Refinement-3) have OVER == 0  (no over-deny regression)
  OPTIMALITY          = premature/transfer have UNDER == 0 (full +8 realizable)
  OVER  = node in reconstruction NOT in evaluator graph (reconstruction requires more -> over-deny)
  UNDER = node in evaluator graph NOT in reconstruction (missing -> no gain, but regression-safe)

Run: PYTHONPATH=$CLONE:$REPO/scripts/distill/sopbench seka_python guard2_dirgraph_unitcheck.py
"""
import json, glob, collections, copy
from env.variables import domain_assistant_keys, domain_keys
from env import helpers as H

EVAL_GLOB = "/home/woori/scratch/sft_alias_run/eval_t1c_keeptuple/bank/*full*shuffle_False.json"
DOMAIN = "bank"
DEFAULT_OPT = "required"   # generator/eval default (gather_action_default_dependencies default)

def norm(x):
    """Recursively sort dict keys so comparison ignores key-order (json artifact)."""
    if isinstance(x, dict):
        return {k: norm(x[k]) for k in sorted(x)}
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    return x

def node_key(n):
    """Hashable content key for a graph node: leaf=(name, sorted-args), operator=str."""
    nn = norm(n)
    return json.dumps(nn, sort_keys=True)

def main():
    da = domain_assistant_keys[DOMAIN]
    ds = domain_keys[DOMAIN]()
    cl, cp = da.constraint_links, da.constraint_processes
    add = H.gather_action_default_dependencies(
        da.action_required_dependencies, da.action_customizable_dependencies,
        default_dependency_option=DEFAULT_OPT)
    ap = H.get_action_parameters(ds, da)

    # INPUT AUDIT (Refinement-1): the only task-derived input below is task["constraints_original"].
    # task["directed_action_graph"] is used ONLY as the comparison target, never as a build input.
    AUDIT_INPUTS = {"goal", "task[constraints_original]", "domain:cl/cp/default_dep/action_parameters"}
    print(f"[INPUT AUDIT] reconstruction inputs = {sorted(AUDIT_INPUTS)} ; reads directed_action_graph = NO")

    recs = json.load(open(glob.glob(EVAL_GLOB)[0]))
    def bucket(e):
        if e["dirgraph_satisfied"] and e["action_successfully_called"]: return "BOTH"
        if e["action_successfully_called"]: return "premature"
        return "DENY"

    safety_fail = []; over_tot = under_tot = 0
    bybuck = collections.Counter()
    detail = collections.defaultdict(lambda: [0, 0, 0])  # bucket -> [n, over_tasks, under_tasks]
    for r in recs:
        e = r["evaluations"][0]
        if not e.get("action_should_succeed"): continue
        t = r["task"]; goal = t["user_goal"]; bk = bucket(e); bybuck[bk]+=1
        ugn = (goal, {k: k for k in ap[goal]})
        rb = H.dfsgather_invfunccalldirgraph(t["constraints_original"], cl, cp, add, ap, ugn)
        rb_nodes = [node_key(n) for n in rb["nodes"]]
        tg_nodes = [node_key(n) for n in t["directed_action_graph"]["nodes"]]
        cr, ct = collections.Counter(rb_nodes), collections.Counter(tg_nodes)
        over = list((cr - ct).elements())    # in reconstruction, not in evaluator graph
        under = list((ct - cr).elements())   # in evaluator graph, not in reconstruction
        detail[bk][0]+=1; detail[bk][1]+= (1 if over else 0); detail[bk][2]+= (1 if under else 0)
        over_tot += len(over); under_tot += len(under)
        if bk == "BOTH" and over:
            safety_fail.append((goal, over))

    print(f"\nbuckets(should_T): {dict(bybuck)}")
    print("per-bucket [n, tasks_with_OVER, tasks_with_UNDER]:")
    for bk in ("BOTH","premature","DENY"):
        print(f"  {bk:10s}: {detail[bk]}")
    print(f"\ntotals: OVER nodes={over_tot}  UNDER nodes={under_tot}")
    print(f"\n=== SAFETY GATE (Refinement-2/3): current-BOTH OVER must be 0 over ALL {detail['BOTH'][0]} BOTH ===")
    if not safety_fail:
        print(f"  SAFETY PASS: 0/{detail['BOTH'][0]} BOTH tasks have OVER -> reconstruction never over-requires -> no over-deny regression.")
    else:
        print(f"  SAFETY FAIL: {len(safety_fail)} BOTH tasks OVER (catastrophic, do NOT ship A1):")
        for g, ov in safety_fail[:10]: print(f"    {g}: OVER={ov[:3]}")
    opt = detail["premature"][2] + detail["DENY"][2]
    print(f"\n=== OPTIMALITY: premature/DENY tasks with UNDER (missing establishing) = {opt} (UNDER=0 -> full +8) ===")

if __name__ == "__main__":
    main()
