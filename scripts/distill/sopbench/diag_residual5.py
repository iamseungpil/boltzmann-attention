#!/usr/bin/env python3
"""Zero-cost diagnosis of residual ~5 (goal-call cnv) after DGGATE.
Joins eval_t1c_dggate JSON + offload_log_dggate.jsonl by task_sig.
Identifies per-task: BOTH status, premature (acc=True & dirgraph=False & cnv=False),
and the offload-log decision/reason so we can pinpoint the cnv-violating leaf."""
import json, hashlib, collections, sys

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
OFFLOG = "/home/woori/scratch/sft_alias_run/offload_log_dggate.jsonl"

def sig(task):
    return hashlib.md5(json.dumps(
        [task.get("user_goal"), task.get("constraints"), task.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]

d = json.load(open(EVAL))
# offload log: group by task_sig (last-turn decision is what matters for goal-call)
off = collections.defaultdict(list)
with open(OFFLOG) as f:
    for l in f:
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        off[r.get("task_sig")].append(r)

PART_A = {"cancel_credit_card", "pay_bill_with_credit_card"}

rows = []
for e in d:
    t = e["task"]
    ev = e["evaluations"][0]
    if not ev.get("action_should_succeed"):
        continue  # should_T only
    s = sig(t)
    rows.append({
        "sig": s,
        "goal": t.get("user_goal"),
        "cnv": ev.get("constraint_not_violated"),
        "dg": ev.get("dirgraph_satisfied"),
        "acc": ev.get("action_successfully_called"),
        "acalled": ev.get("action_called_correctly"),
        "dbm": ev.get("database_match"),
        "succ": ev.get("success"),
        "ntce": ev.get("no_tool_call_error"),
    })

n_should_t = len(rows)
both = [r for r in rows if r["dg"] and r["acc"]]
print(f"should_T tasks = {n_should_t}")
print(f"BOTH (dirgraph & acc) = {len(both)}")

# residual = should_T NOT both
resid = [r for r in rows if not (r["dg"] and r["acc"])]
print(f"residual (not BOTH) = {len(resid)}\n")

# premature: acc True but dirgraph False
premature = [r for r in resid if r["acc"] and not r["dg"]]
deny = [r for r in resid if not r["acc"]]
print(f"  premature (acc=T, dg=F): {len(premature)}")
print(f"  deny      (acc=F)      : {len(deny)}\n")

def show(label, group):
    print(f"=== {label} (n={len(group)}) ===")
    by_goal = collections.Counter(r["goal"] for r in group)
    for g, c in by_goal.most_common():
        print(f"   {c:2d}  {g}")
    print()

show("PREMATURE should_T", premature)
show("DENY should_T", deny)

print("\n##### PREMATURE detail + offload-log join #####")
for r in premature:
    partA = r["goal"] in PART_A
    logs = off.get(r["sig"], [])
    last = logs[-1] if logs else None
    print(f"\n[{r['goal']}] sig={r['sig']} PartA={partA}")
    print(f"   cnv={r['cnv']} dg={r['dg']} acc={r['acc']} acalled={r['acalled']} dbm={r['dbm']} ntce={r['ntce']}")
    print(f"   offlog turns={len(logs)}")
    if last:
        print(f"   last decision={last.get('decision')} reason={last.get('reason')} "
              f"nfalse={last.get('n_false')} nung={last.get('n_ungathered')} narg={last.get('n_argmismatch')}")
        if last.get("false"):
            print(f"     false leaves: {last['false']}")
        if last.get("ungathered"):
            print(f"     ungathered  : {last['ungathered']}")
