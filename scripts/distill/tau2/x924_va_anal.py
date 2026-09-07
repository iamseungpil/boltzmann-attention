# -*- coding: utf-8 -*-
import json,io,sys,collections,re
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
va=json.load(open("x923_va.json",encoding="utf-8"))
SIG=['last 4','last four','last-4','four digits','4 digits','correct last 4']
def cur_has(t): 
    tl=(t or "").lower(); return [s for s in SIG if s in tl]
n_cur=sum(1 for x in va if cur_has(x[5]))
print("VA fires",len(va)," 현재 발화에 신호 있음:",n_cur," 없음(=prior 만으로 발화):",len(va)-n_cur)
S={(x[2],x[3],x[1]) for x in va}
print("발화 sim:",len(S))
for sim,r,t in sorted(S,key=lambda z:-z[1]):
    c=sum(1 for x in va if x[2]==sim)
    print("  %s %-9s r=%-4s fires=%d"%(sim[:8],t,r,c))
print("\n--- reward=1.0 sim 의 VA 발화 축자(최대 6건) ---")
ps={s for s,r,_ in S if r==1.0}
k=0
for x in va:
    if x[2] in ps and k<6:
        print("[%s %s r=%s msg%d] sig=%s"%(x[1],x[2][:8],x[3],x[4],cur_has(x[5])))
        print((x[5] or "")[:300]); print(); k+=1
print("\n--- 현재발화에 신호 없이(prior 만) 발화한 예 6건 ---")
k=0
for x in va:
    if not cur_has(x[5]) and k<6:
        print("[%s %s r=%s msg%d]"%(x[1],x[2][:8],x[3],x[4]))
        print((x[5] or "")[:260]); print(); k+=1
