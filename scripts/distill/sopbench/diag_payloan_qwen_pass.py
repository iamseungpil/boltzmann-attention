#!/usr/bin/env python3
"""Resolve the pay_loan contradiction: in the released qwen2.5-7b FC pass of pay_loan 92f3/81ba,
what did login_user receive (identification) and RETURN (True/False)? What is the task user_known?
This tells us whether the win path needs creds (then cred-absent = our problem) or login returns
False but pay_loan passes anyway (a no-login-equivalent path)."""
import json, hashlib, glob
def H(x): return hashlib.md5(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:16]
def ident(t): return (t.get("user_goal"),H(t.get("constraints_original")),H(t.get("constraint_parameters")),H(t.get("initial_database")))
EVAL="/home/woori/scratch/sft_alias_run/eval_t1c_dggate/bank/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"
def sig(t): return hashlib.md5(json.dumps([t.get("user_goal"),t.get("constraints"),t.get("user_known")],sort_keys=True,default=str).encode()).hexdigest()[:12]
d=json.load(open(EVAL)); targets={}
for e in d:
    if sig(e["task"]) in ("92f35479191d","81ba61427f77"): targets[ident(e["task"])]=sig(e["task"])

for fp in glob.glob("/home/woori/scratch/SOPBench/output/bank/ast_*qwen2.5-7b*mode_fc*.json"):
    dd=json.load(open(fp))
    for e in dd:
        if "task" not in e: continue
        idt=ident(e["task"])
        if idt not in targets: continue
        evl=e.get("evaluations") or []
        succ=evl[0].get("success") if evl else None
        print(f"\n=== pay_loan {targets[idt]} qwen2.5-7b-fc success={succ} ===")
        print(f"  user_known = {e['task'].get('user_known')}")
        print(f"  initial_db[user] identification = {e['task']['initial_database']['accounts'].get(e['task']['user_known'].get('username'),{}).get('identification')}")
        if evl:
            ev=evl[0]; print(f"  eval: dg={ev.get('dirgraph_satisfied')} cnv={ev.get('constraint_not_violated')} acc={ev.get('action_successfully_called')} dbm={ev.get('database_match')}")
        for il in (e.get("interactions") or []):
            conv=il.get("interaction") if isinstance(il,dict) else None
            if not conv: continue
            pend={}
            for m in conv:
                if not isinstance(m,dict): continue
                for tc in (m.get("tool_calls") or []):
                    fn=(tc.get("function") or {}).get("name") if tc.get("function") else tc.get("tool_name")
                    ar=(tc.get("function") or {}).get("arguments") if tc.get("function") else None
                    print(f"    CALL {fn} {ar}")
                if m.get("tool_name"):
                    print(f"      -> [{m.get('tool_name')}] {str(m.get('content'))[:80]}")
