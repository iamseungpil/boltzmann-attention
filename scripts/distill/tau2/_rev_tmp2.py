import json, glob, re
from collections import Counter, defaultdict
_fam=lambda n: re.sub(r"_\d+$","",str(n or ""))
def _nd(x):
    if isinstance(x,str):
        try:x=json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
ID_KEYS=["transaction_id","card_id","account_id","credit_card_account_id"]
argkeys=defaultdict(Counter)
tools=["pay_credit_card_from_checking","log_credit_card_closure_reason","close_credit_card_account",
       "close_debit_card","close_bank_account","transfer_funds_between_bank_accounts",
       "open_bank_account","order_replacement_credit_card","apply_statement_credit",
       "update_transaction_rewards","approve_credit_limit_increase"]
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d=json.load(open(f,encoding="utf-8"))
    for s in d.get("simulations",[]):
        ri=s.get("reward_info") or {}
        for ac in (ri.get("action_checks") or []):
            a=ac.get("action") or {}
            outer=_nd(a.get("arguments"))
            atn=outer.get("agent_tool_name","")
            if not atn or "arguments" not in outer: continue
            tf=_fam(atn)
            if tf in tools:
                ga=_nd(outer.get("arguments"))
                for k in ga: argkeys[tf][k]+=1
for tf in tools:
    ks=argkeys[tf]
    idlike=[k for k in ks if k.endswith("_id") or k=="user_id"]
    which=[k for k in ID_KEYS if k in ks]
    print(f"{tf}:")
    print(f"   all id-like args: {idlike}")
    print(f"   priority-list hits (in order): {which}  -> picks: {which[0] if which else None}")
