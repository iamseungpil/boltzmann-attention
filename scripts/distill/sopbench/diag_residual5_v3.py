#!/usr/bin/env python3
"""v3: premature root-cause + canonical PartB split.
- Fix call-trace extraction (sender/tool_calls + tool_name responses).
- Match each premature eval-task to canonical bank_tasks.json by augment-INVARIANT
  key (user_goal + constraints_original), recover canonical user_known to decide
  PartB (auth leaf present AND canonical creds absent => defect).
- Report which value-restriction getters were actually CALLED vs the constraint leaves.
"""
import json, hashlib, collections

EVAL = "/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
OFFLOG = "/home/woori/scratch/sft_alias_run/offload_log_dggate.jsonl"
CANON = "/home/woori/scratch/SOPBench/data/bank_tasks.json"

def sig(t):
    return hashlib.md5(json.dumps(
        [t.get("user_goal"), t.get("constraints"), t.get("user_known")],
        sort_keys=True, default=str).encode()).hexdigest()[:12]

# augment-invariant key: goal + constraints_original (user_known excluded)
def inv_key(t):
    return json.dumps([t.get("user_goal"), t.get("constraints_original")],
                      sort_keys=True, default=str)

d = json.load(open(EVAL))
off = collections.defaultdict(list)
with open(OFFLOG) as f:
    for l in f:
        l = l.strip()
        if l:
            r = json.loads(l); off[r.get("task_sig")].append(r)

canon = json.load(open(CANON))
print("CANON type", type(canon).__name__, "n=", len(canon) if hasattr(canon, "__len__") else "?")
# build inv_key -> list of canonical user_known
canon_map = collections.defaultdict(list)
canon_list = canon if isinstance(canon, list) else canon.get("tasks", canon)
if isinstance(canon_list, dict):  # maybe keyed by goal
    flat = []
    for v in canon_list.values():
        flat += v if isinstance(v, list) else [v]
    canon_list = flat
for ct in canon_list:
    if isinstance(ct, dict):
        canon_map[inv_key(ct)].append(ct.get("user_known"))

# leaves that imply auth requirement
AUTH_LEAVES = {"logged_in_user", "authenticated_admin_password"}

def leaves_of(constr):
    """flatten constraint tree -> list of (leaf_fn, params)"""
    out = []
    def rec(node):
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head == "single":
            out.append((node[1], node[2] if len(node) > 2 else {}))
        elif head in ("and", "or", "chain"):
            for sub in node[1]:
                rec(sub)
        else:
            for sub in node:
                rec(sub)
    rec(constr)
    return out

premature = []
for e in d:
    t = e["task"]; ev = e["evaluations"][0]
    if not ev.get("action_should_succeed"):
        continue
    if ev.get("action_successfully_called") and not ev.get("dirgraph_satisfied"):
        premature.append(e)

print(f"\nPREMATURE n={len(premature)}\n" + "="*70)
for e in premature:
    t = e["task"]; ev = e["evaluations"][0]; s = sig(t)
    lvs = leaves_of(t["constraints_original"])
    leaf_names = [fn for fn, _ in lvs]
    auth_needed = [fn for fn in leaf_names if fn in AUTH_LEAVES]
    # canonical user_known for this inv_key
    ck = canon_map.get(inv_key(t), [])
    # PartB heuristic: auth leaf present AND in canonical user_known the matching cred absent
    print(f"\n##### [{t['user_goal']}] sig={s} #####")
    print(f"  leaves: {leaf_names}")
    print(f"  cnv={ev.get('constraint_not_violated')} dg={ev.get('dirgraph_satisfied')} dbm={ev.get('database_match')}")
    print(f"  auth_leaves={auth_needed}")
    print(f"  eval user_known keys: {sorted(t['user_known'].keys())}")
    if ck:
        print(f"  CANON({len(ck)}) user_known keys: {[sorted(u.keys()) if u else None for u in ck]}")
    else:
        print(f"  CANON: NO MATCH on inv_key")
    logs = off.get(s, []); last = logs[-1] if logs else None
    if last:
        print(f"  GATE: {last['decision']}/{last['reason']} nfalse={last['n_false']} nung={last['n_ungathered']} narg={last['n_argmismatch']}"
              + (f" false={last['false']}" if last.get('false') else "")
              + (f" ung={last['ungathered']}" if last.get('ungathered') else ""))
    # call trace
    conv = e["interactions"][0]["interaction"]
    calls = []
    for m in conv:
        if isinstance(m, dict) and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}).get("name") or tc.get("tool_name") or tc.get("name")
                ar = (tc.get("function") or {}).get("arguments") or tc.get("arguments")
                calls.append((fn, ar))
    print(f"  CALLS ({len(calls)}): " + ", ".join(f"{fn}{ar}" for fn, ar in calls))
