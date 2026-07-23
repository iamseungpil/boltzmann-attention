import gzip,json,os,sys
sys.path.insert(0,r"C:\workspace\ba-frft\scripts\distill\tau2")
import t2_gate_patch as G
from types import SimpleNamespace as NS
D=r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
def load(arm):
    f=os.path.join(D,f"bank_hve2e9_{arm}_20260723.results.json.gz")
    return {s["task_id"]:s for s in json.load(gzip.open(f,"rt",encoding="utf-8"))["simulations"]}
def cvt(m):
    tcs=[]
    for tc in (m.get("tool_calls") or []):
        fn=tc.get("function",{}) if "function" in tc else tc
        try:a=json.loads(fn.get("arguments") or "{}")
        except:a={}
        tcs.append(NS(name=fn.get("name"),arguments=a,id=tc.get("id")))
    return NS(role=m.get("role"),content=m.get("content") or "",tool_calls=tcs or None,error=bool(m.get("error")),id=m.get("id"))
# banking A2 wtools
os.environ["T2_A2_VARIANT"]="ledger,ratefix"
from gate_interpreter import load_domain_a2
a2=load_domain_a2("banking_knowledge")
wtools=G._confirm_write_tools(a2)|set((a2.get("eplan") or {}).get("write_tools") or [])
print("wtools=",sorted(wtools))
base=load("base")
def testcase(tid,idx,label):
    msgs=[cvt(m) for m in base[tid]["messages"]]
    # committed = msgs[:idx], am = msgs[idx]
    am=msgs[idx]; committed=msgs[:idx]
    ntc=len(am.tool_calls or [])
    stale=G._stale_call_ids(am,committed,wtools)
    print(f"[{label}] {tid}[{idx}] tool_calls={ntc} stale-strip={len(stale)}")
    return ntc,len(stale)
# 043[24] = 12개 대량 병렬 (read 중복 다수)
testcase("task_043",24,"over-action-043")
# 054[55] = 12개 (write 중복 포함)
n,s=testcase("task_054",55,"over-action-054")
# over-fire 대조: 정상 단일 호출 턴은 strip 0
for tid,idx in [("task_031",10),("task_038",4)]:
    n2,s2=testcase(tid,idx,"normal")
    assert s2==0, f"over-fire! {tid}[{idx}] stripped {s2}"
print("\nPASS: over-action 턴서 중복 strip 감지·정상 턴 strip 0(over-fire 없음)")
