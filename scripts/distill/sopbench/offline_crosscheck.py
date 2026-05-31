"""Offline evidence-(B): across ALL shipped bank ast_*.json, for each goal+should_succeed=True
task instance, did ANY model ever get evaluation.success == True? Uses the embedded `evaluation`
dict (no re-run). Decides which goals NO released model ever passed."""
import json, glob, os, collections

OUT = "/tmp/ex2/SOPBench-main/output/bank"
files = sorted(glob.glob(os.path.join(OUT, "ast_*.json")))
print(f"bank ast_ files: {len(files)}")

# goal -> {instance_idx -> {"any_pass":bool, "seen":int, "should":bool, "failed":Counter}}
agg = collections.defaultdict(lambda: collections.defaultdict(
    lambda: {"any_pass": False, "seen": 0, "should": None, "failed": collections.Counter()}))
SUB = ["no_tool_call_error", "constraint_not_violated", "database_match",
       "action_called_correctly", "dirgraph_satisfied"]

for f in files:
    d = json.load(open(f))
    for goal, insts in d.items():
        if not isinstance(insts, list):
            insts = [insts]
        for j, rec in enumerate(insts):
            ev = (rec or {}).get("evaluation") or {}
            if not isinstance(ev, dict) or "success" not in ev:
                continue
            cell = agg[goal][j]
            cell["seen"] += 1
            if cell["should"] is None:
                cell["should"] = ev.get("action_should_succeed")
            if ev.get("success"):
                cell["any_pass"] = True
            else:
                for s in SUB:
                    if s in ev and not ev[s]:
                        cell["failed"][s] += 1

# headline: should_succeed=True instances never passed by ANY model (seen>=5 for confidence)
n_should_true = n_impossible = 0
impossible = collections.defaultdict(list)
for goal, insts in agg.items():
    for j, c in insts.items():
        if c["should"] is True:
            n_should_true += 1
            if not c["any_pass"] and c["seen"] >= 5:
                n_impossible += 1
                top = c["failed"].most_common(2)
                impossible[goal].append((j, c["seen"], top))

print(f"should_succeed=True instances: {n_should_true}")
print(f"NEVER passed by any model (seen>=5): {n_impossible} instances / {len(impossible)} goals")
print("--- impossible goals (instance, #models_seen, top failing subchecks) ---")
for goal in sorted(impossible):
    print(f"  {goal}:")
    for (j, seen, top) in impossible[goal]:
        print(f"     inst{j} seen={seen} fails={dict(top)}")
# also: goals where SOME model passed (sanity)
passed_goals = sorted({g for g, insts in agg.items()
                       for c in insts.values() if c["should"] is True and c["any_pass"]})
print(f"--- should=True goals passed by >=1 model: {len(passed_goals)} ---")
print("  ", passed_goals[:25])
