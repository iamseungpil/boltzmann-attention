# x828 — A3 doc_index 가 «절차 문서» 를 라우팅할 수 있는가 (GPU 불요·구조만)
import json,glob,os,re,collections
KB="/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2="/home/woori/scratch/repo_rep1/scripts/distill/tau2/a2/banking_knowledge.gate.json"
idx=(json.load(open(A2)).get("policy_ontology") or {}).get("doc_index") or {}
docs=sorted(glob.glob(KB+"/*.json"))
print("KB 문서 %d · doc_index 군 %d · 계열 총합 %d"%(len(docs),len(idx),sum(len(v) for v in idx.values())))
# 문서 -> (군, 계열)  : 파일명 규약 doc_<group>_<series>_NNN
def split(name):
    b=os.path.basename(name)[4:-5]              # doc_ ... .json
    m=re.match(r"^(.*)_(\d+)$",b)
    return (m.group(1) if m else b)
# 각 군의 계열이 _general_ 뿐인가
onlygen=[g for g,v in idx.items() if set(v)=={"_general_"}]
print("계열이 _general_ 뿐인 군 %d/%d: %s"%(len(onlygen),len(idx),onlygen))
# 절차/내부 문서 = 제목이 Internal: 로 시작하거나 본문에 절차어가 있는 문서 (env 축자·닫힌 술어)
proc=[]
for p in docs:
    try: d=json.load(open(p))
    except Exception: continue
    t=str(d.get("title") or ""); c=json.dumps(d,ensure_ascii=False)
    if t.startswith("Internal:") : proc.append((os.path.basename(p),t))
print("Internal: 로 시작하는 절차 문서 %d건"%len(proc))
# 그 절차 문서들이 어느 군에 떨어지나 · 그 군에 축이 있나
cnt=collections.Counter(); noaxis=0
for fn,t in proc:
    stem=split(fn)
    g=None
    for gg in idx:
        if stem.startswith(gg): 
            if g is None or len(gg)>len(g): g=gg
    cnt[g or "(군없음)"]+=1
    if g is None or set(idx.get(g,{}))=={"_general_"}: noaxis+=1
print("절차 문서 중 **축이 없는 군**(계열=_general_ 뿐이거나 군 미매칭)에 떨어지는 것: %d/%d = %.0f%%"%(
      noaxis,len(proc),100.0*noaxis/max(1,len(proc))))
print("군별 분포 상위:")
for g,n in cnt.most_common(8):
    ser=list(idx.get(g,{})) if g in idx else []
    print("   %-40s %3d건  계열=%s"%(g,n,ser[:4] if ser else "-"))
# 이번 실패의 표적 문서
tgt="doc_bank_accounts_bank_accounts_(general)_042.json"
d=json.load(open(os.path.join(KB,tgt)))
print("\n표적 문서: %s"%d.get("title"))
stem=split(tgt); g=max((gg for gg in idx if stem.startswith(gg)),key=len,default=None)
print("  군=%s  계열=%s  -> 축 %s"%(g,list(idx.get(g,{})),"없음(라우팅 불가)" if set(idx.get(g,{}))=={"_general_"} else "있음"))
