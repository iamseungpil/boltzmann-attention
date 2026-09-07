# -*- coding: utf-8 -*-
r"""x829 — A3 doc_index 가 «절차 문서»로 라우팅할 수 있는가 (격리·[[78]])

구조 결손(x828·GPU 불요): Internal: 절차 문서 47건 중 36건(77%)이
`bank_accounts_bank_accounts` 군에 떨어지고 그 군의 계열은 `_general_` 하나뿐이다
= 인덱스가 스스로 "고를 것이 없는 축"이라 부르는 자리(t2_gate_patch.py:4719).

표적(닫힌 술어·env 출처·gold 불참조): 도구 독스트링이 축자로 요구한다 —
  "The proper transfer reason enum can be found in the knowledge base:
   search it before calling this tool to select the proper applicable reason."
그 enum = tools.py:48 `TransferReasonLiteral` 19개. KB 698문서 중 그 코드를 2개 이상
담은 문서는 **정확히 1건**: doc_bank_accounts_bank_accounts_(general)_042
("Internal: Human Agent Transfer Reason Codes"). 채점은 이 파일명 일치뿐 — 엔진 판단 0.

컷: rep1 task_004 seed 626729 (라이브 FAIL) — verify_identity 가 NOT_VERIFIED 를 돌려주고
손님이 "생일도 주소도 지금 없다"고 답한 직후. 그 sim 은 여기서 이관을 안 하고
「대안 인증·앱 메뉴 경로」를 6라운드 뒤지다 끝났다.
⚠도구 반환은 300자로 자른다 — 이 프로브는 «문서 선택»만 격리한다([[71]] 필요한 것만).

팔은 **선언 오버라이드 한 칸**(주는 재료)만 다르다:
  A_NONE    재료 없음                      = 라이브가 하는 것
  B_GROUPS  A3 doc_index 군 이름 11개      = 우리 온톨로지가 실제로 줄 수 있는 것
  C_GEN36   축 없는 군의 문서 36건 제목    = 축을 만들어 주면?
  D_INT47   Internal: 절차 문서 47건 제목  = 절차 축을 통째로 주면?
"""
import json,glob,os,re,sys,urllib.request,collections
BASE=os.environ.get("X829_BASE","http://localhost:8141/v1")
MODEL="Qwen/Qwen3.8-27B-FP8"
KB="/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2="/home/woori/scratch/repo_rep1/scripts/distill/tau2/a2/banking_knowledge.gate.json"
RES="/home/woori/scratch/repo_rep1/reports/facet_rft_2026/sim_results/rep1_task_004.results.json.gz"
TARGET="doc_bank_accounts_bank_accounts_(general)_042"
N=int(os.environ.get("X829_N","8"))

import gzip
def cut():
    j=json.load(gzip.open(RES,"rt"))
    s=[x for x in j["simulations"] if x.get("seed")==626729][0]
    out=[]
    for m in s["messages"][:26]:
        m=m or {}; r=m.get("role"); c=str(m.get("content") or "")
        tcs=m.get("tool_calls") or []
        if tcs:
            for tc in tcs:
                nm=(tc or {}).get("name") or ((tc or {}).get("function") or {}).get("name")
                ar=(tc or {}).get("arguments") or ((tc or {}).get("function") or {}).get("arguments")
                out.append("AGENT calls %s(%s)"%(nm,json.dumps(ar,ensure_ascii=False)[:160] if not isinstance(ar,str) else ar[:160]))
        elif r=="tool":  out.append("TOOL -> %s"%c[:300].replace("\n"," "))
        elif c.strip():  out.append("%s: %s"%(("CUSTOMER" if r=="user" else "AGENT"),c.strip()))
    return "\n".join(out)
def titles(files):
    o=[]
    for f in files:
        try: t=json.load(open(os.path.join(KB,f))).get("title")
        except Exception: t="?"
        o.append("%s  %s"%(f[:-5],t))
    return o
idx=(json.load(open(A2)).get("policy_ontology") or {}).get("doc_index") or {}
allf=sorted(os.path.basename(p) for p in glob.glob(KB+"/*.json"))
def stem(fn):
    b=fn[4:-5]; m=re.match(r"^(.*)_(\d+)$",b); return m.group(1) if m else b
gen36=[f for f in allf if stem(f).startswith("bank_accounts_bank_accounts")]
int47=[]
for f in allf:
    try:
        if str(json.load(open(os.path.join(KB,f))).get("title") or "").startswith("Internal:"): int47.append(f)
    except Exception: pass
ARMS={
 "A_NONE":  None,
 "B_GROUPS":"Document groups available in our policy ontology:\n"+"\n".join(sorted(idx)),
 "C_GEN36": "Documents in the bank_accounts_bank_accounts group:\n"+"\n".join(titles(gen36)),
 "D_INT47": "Internal procedure documents in the knowledge base:\n"+"\n".join(titles(int47)),
}
ASK=("You are the agent in the conversation above. Identity verification cannot be completed.\n"
     "Name the ONE knowledge-base document you must read next in order to proceed correctly.\n"
     "Reply with ONLY the document id (e.g. doc_xxx_yyy_001), nothing else.")
ASK_B=("You are the agent in the conversation above. Identity verification cannot be completed.\n"
       "Name the ONE document group that contains the document you must read next.\n"
       "Reply with ONLY the group name, nothing else.")
def call(prompt,seed):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],
          "max_tokens":64,"temperature":0.7,"seed":seed}
    req=urllib.request.Request(BASE+"/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()
C=cut()
print("컷 길이 %d자 · 표적 %s · 팔 %d · n=%d"%(len(C),TARGET,len(ARMS),N))
print("  gen36=%d · int47=%d · 군 %d"%(len(gen36),len(int47),len(idx)))
res=collections.defaultdict(list)
for arm,mat in ARMS.items():
    ask=ASK_B if arm=="B_GROUPS" else ASK
    for i in range(N):
        p=C+"\n\n"+(mat+"\n\n" if mat else "")+ask
        try: a=call(p,1000+i)
        except Exception as e: a="ERR:%s"%e
        if arm=="B_GROUPS": ok = "bank_accounts_bank_accounts" in a
        else:               ok = "_042" in a and "bank_accounts" in a
        res[arm].append((ok,a.replace("\n"," ")[:80]))
        print("  %-9s #%d %s %s"%(arm,i,"O" if ok else ".",a.replace("\n"," ")[:70]))
print("\n== 결과 ==")
for arm in ARMS:
    v=res[arm]; k=sum(1 for o,_ in v if o)
    print("  %-9s %d/%d = %5.1f%%"%(arm,k,len(v),100.0*k/len(v)))
    c=collections.Counter(a for _,a in v)
    for a,n in c.most_common(3): print("        %2d회  %s"%(n,a))
