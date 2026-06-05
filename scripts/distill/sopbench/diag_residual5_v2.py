#!/usr/bin/env python3
"""Deep zero-cost diagnosis of premature residual after DGGATE.
For each premature should_T task (acc=True, dirgraph=False, cnv=False):
  - dump goal, constraints_original, user_known (creds), user_instruction
  - full assistant function-call trace (names + args) and tool returns
  - offload-log gate decision (what the gate thought)
Goal: pinpoint the cnv-violating / dirgraph-unsatisfied leaf per task, and
separate PartB-defect (auth leaf, creds genuinely absent) from genuine-fixable.
"""
import json, hashlib, collections

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
OFFLOG = "/home/woori/scratch/sft_alias_run/offload_log_dggate.jsonl"

def sig(t):
    return hashlib.md5(json.dumps(
        [t.get("user_goal"), t.get("constraints"), t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]

d = json.load(open(EVAL))
off = collections.defaultdict(list)
with open(OFFLOG) as f:
    for l in f:
        l = l.strip()
        if l:
            r = json.loads(l)
            off[r.get("task_sig")].append(r)

premature = []
for e in d:
    t = e["task"]; ev = e["evaluations"][0]
    if not ev.get("action_should_succeed"):
        continue
    if ev.get("action_successfully_called") and not ev.get("dirgraph_satisfied"):
        premature.append(e)

print(f"PREMATURE n={len(premature)}\n" + "="*70)
for e in premature:
    t = e["task"]; ev = e["evaluations"][0]; s = sig(t)
    print(f"\n########## [{t['user_goal']}] sig={s} ##########")
    print(f"  cnv={ev.get('constraint_not_violated')} dg={ev.get('dirgraph_satisfied')} "
          f"acc={ev.get('action_successfully_called')} dbm={ev.get('database_match')}")
    print(f"  CONSTRAINTS_ORIG: {json.dumps(t['constraints_original'], ensure_ascii=False)}")
    print(f"  USER_KNOWN: {json.dumps(t['user_known'], ensure_ascii=False)}")
    print(f"  INSTR: {t.get('user_instruction')}")
    # credentials present in initial_database for this user?
    uk = t['user_known']; un = uk.get('username')
    acct = t['initial_database'].get('accounts', {}).get(un, {}) if un else {}
    print(f"  DB[{un}] keys: {sorted(acct.keys())}")
    print(f"  DB[{un}] identification={acct.get('identification')!r} admin_password={acct.get('admin_password')!r}")
    # gate view
    logs = off.get(s, [])
    last = logs[-1] if logs else None
    if last:
        print(f"  GATE last: decision={last['decision']} reason={last['reason']} "
              f"nfalse={last['n_false']} nung={last['n_ungathered']} narg={last['n_argmismatch']}")
        if last.get('false'): print(f"     gate false: {last['false']}")
        if last.get('ungathered'): print(f"     gate ungathered: {last['ungathered']}")
    # call trace
    conv = e["interactions"][0]["interaction"]
    print("  --- CALL TRACE ---")
    for m in conv:
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc["function"]["name"]; ar = tc["function"]["arguments"]
                print("    CALL " + fn + " " + str(ar))
        elif role == "tool":
            print("      -> " + str(m.get("content"))[:140])
