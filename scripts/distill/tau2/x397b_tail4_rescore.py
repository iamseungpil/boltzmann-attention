# -*- coding: utf-8 -*-
"""x397b — 종료마커·인자결손 판정 → B_tail4 재계산 + 표적-대응 부호검정 재실행."""
import io,json,os,re,sys,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import x395_compliance_iso as X, t2_forensic as F

R="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
tab=json.load(io.open(R+"x397_tail4_answerability.json",encoding="utf-8"))
res=json.load(io.open(R+"x395_compliance_iso.json",encoding="utf-8"))

# 마지막 user 메시지 축자 + 종료 마커
docs=X.load_docs()
sims={}
for tag in X.TAGS:
    for sim in F.scored(tag,X.SUF):
        sims[(F.task_id(sim),sim.get("trial"))]=sim
MARK=re.compile(r"###(STOP|TRANSFER|OUT-OF-SCOPE)###")
print("%-9s %-40s %-14s %s"%("task","tool","종료마커","마지막 user 메시지 축자(끝 200자)"))
for r in tab:
    sim=sims[(r["task"],r["trial"])]
    lastu=""
    for m in (sim.get("messages") or [])[-4:]:
        if m.get("role")=="user" and m.get("content"): lastu=" ".join(str(m["content"]).split())
    mk=MARK.search(lastu)
    r["last_user"]=lastu; r["end_marker"]=mk.group(0) if mk else ""
    # 대화 전체 종료 여부: tail4 가 궤적의 끝인가
    r["is_end_of_convo"]=True
    print("%-9s %-40s %-14s %s"%(r["task"],r["tool"][:40],r["end_marker"] or "(없음)",lastu[-200:]))

print("\n## 판정")
hdr=("task","tool","단서(도구명)","인자값 tail4","인자값 A_min","종료마커","오채점")
print("%-9s %-40s %-12s %-14s %-14s %-14s %s"%hdr)
for r in tab:
    c4=r["argcov_t4"]; ca=r["argcov_amin"]
    f4="-" if not c4 else "%d/%d"%(c4[0],c4[1]); fa="-" if not ca else "%d/%d"%(ca[0],ca[1])
    arg_missing = (c4 is not None) and c4[0]<c4[1]
    ended = bool(r["end_marker"])
    r["arg_missing"]=arg_missing; r["ended"]=ended
    r["mis_strict"]=ended or arg_missing        # 기준U: 둘 중 하나라도
    r["mis_arg"]=arg_missing                    # 기준A: 인자값 결손만
    r["mis_end"]=ended                          # 기준E: 종료 신호만
    cue = "절차문장" if r["tool_in_proc"] else ("본문" if r["tool_in_t4"] else "없음")
    print("%-9s %-40s %-12s %-14s %-14s %-14s %s"%(r["task"],r["tool"][:40],cue,f4,fa,
          r["end_marker"] or "-", "예" if r["mis_strict"] else "아니오"))

def hits(arm,task,tool):
    rs=[x for x in res if x["arm"]==arm and x["mode"]=="next" and x["task"]==task and x["tool"]==tool]
    return sum(1 for x in rs if x["hit_exact"]), sum(1 for x in rs if x["said_only"]), len(rs)

print("\n## 표적별 원점수 (next 모드, n=3)")
print("%-9s %-40s %-10s %-10s %-10s %-10s"%("task","tool","A_min","B_tail4","A말만","B4말만"))
pairs=[]
for r in tab:
    ah,asd,an=hits("A_min",r["task"],r["tool"]); bh,bsd,bn=hits("B_tail4",r["task"],r["tool"])
    r["A_hit"],r["A_said"],r["B4_hit"],r["B4_said"],r["n"]=ah,asd,bh,bsd,an
    print("%-9s %-40s %-10s %-10s %-10s %-10s"%(r["task"],r["tool"][:40],"%d/%d"%(ah,an),"%d/%d"%(bh,bn),asd,bsd))
    pairs.append(r)

def signtest(rows,key_a="A_hit",key_b="B4_hit"):
    plus=sum(1 for r in rows if r[key_a]>r[key_b])
    minus=sum(1 for r in rows if r[key_a]<r[key_b])
    n=plus+minus
    if n==0: return plus,minus,None
    p=sum(math.comb(n,k) for k in range(plus,n+1))/float(2**n)
    return plus,minus,p

for label,sel in (("전체 12표적",tab),
                  ("기준A 제외후(인자값 결손 제외)",[r for r in tab if not r["mis_arg"]]),
                  ("기준E 제외후(종료마커 제외)",[r for r in tab if not r["mis_end"]]),
                  ("기준U 제외후(둘 중 하나라도 제외)",[r for r in tab if not r["mis_strict"]])):
    A=sum(r["A_hit"] for r in sel); B=sum(r["B4_hit"] for r in sel); N=sum(r["n"] for r in sel)
    pl,mi,p=signtest(sel)
    print("\n%-34s 표적 %2d개 · A_min %d/%d=%s · B_tail4 %d/%d=%s · 부호 +%d/-%d p=%s"
          %(label,len(sel),A,N,("%.3f"%(A/float(N)) if N else "-"),B,N,("%.3f"%(B/float(N)) if N else "-"),
            pl,mi,("%.4f"%p if p is not None else "정의불가")))

io.open(R+"x397_tail4_answerability.json","w",encoding="utf-8").write(json.dumps(tab,ensure_ascii=False,indent=1))
