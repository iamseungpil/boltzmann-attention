import sys,json,io,re,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import x395_compliance_iso as X, t2_forensic as F
R="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
tab=json.load(io.open(R+"x397_tail4_answerability.json",encoding="utf-8"))
res=json.load(io.open(R+"x395_compliance_iso.json",encoding="utf-8"))
sims={}
for tag in X.TAGS:
    for sim in F.scored(tag,X.SUF): sims[(F.task_id(sim),sim.get("trial"))]=sim
MARK=re.compile(r"###(STOP|TRANSFER|OUT-OF-SCOPE)###")
arms=["A_min","B_tail4","B_tail8","B_tail16","B_tail32","B_full"]
print("%-9s %-38s %-5s %-9s %s"%("task","tool","msgs","Bfull_end"," ".join("%-7s"%a for a in arms)))
tot={a:[0,0] for a in arms}
for r in tab:
    sim=sims[(r["task"],r["trial"])]
    inc=bool(MARK.search(X.convo(sim)))
    cells=[]
    for a in arms:
        rs=[x for x in res if x["arm"]==a and x["mode"]=="next" and x["task"]==r["task"] and x["tool"]==r["tool"]]
        h=sum(1 for x in rs if x["hit_exact"]); cells.append("%d/%d"%(h,len(rs)))
        tot[a][0]+=h; tot[a][1]+=len(rs)
    print("%-9s %-38s %-5d %-9s %s"%(r["task"],r["tool"][:38],r["n_msgs"],"O" if inc else "X"," ".join("%-7s"%c for c in cells)))
print("")
for a in arms:
    print("%-9s %d/%d = %.3f"%(a,tot[a][0],tot[a][1],tot[a][0]/float(tot[a][1])))
