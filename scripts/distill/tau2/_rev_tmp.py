import json, glob, re
from collections import Counter
_fam=lambda n: re.sub(r"_\d+$","",str(n or ""))
_READ=re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_",re.I)
_PROC=re.compile(r"(^log_|_verification$|^kb_search|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_)",re.I)
def _nd(x):
    if isinstance(x,str):
        try:x=json.loads(x)
        except:return {}
    return x if isinstance(x,dict) else {}
gw=Counter(); gp=Counter(); gr=Counter()
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
            if _PROC.search(tf): gp[tf]+=1
            elif _READ.match(tf): gr[tf]+=1
            else: gw[tf]+=1
print("=== GOLD action_checks classified PROCEDURAL (DROPPED) ===")
for k,v in gp.most_common(): print(f"  {v:5d}  {k}")
print("=== GOLD classified READ (FIND) top ===")
for k,v in gr.most_common(15): print(f"  {v:5d}  {k}")
print("=== GOLD classified WRITE top ===")
for k,v in gw.most_common(40): print(f"  {v:5d}  {k}")
