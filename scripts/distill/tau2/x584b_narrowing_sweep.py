# -*- coding: utf-8 -*-
r"""x584b — `_cand9` 를 좁히는 규칙 후보를 **반경 <-> 적중** 으로 맞바꿔 본다 (모델 0 · 무료).

x584 가 잰 무조건 배선의 반경은 검색 턴 432 중 319 (74%) 이고 후보 목록 중앙값이 8 이다.
그건 발화가 아니라 **상시 메뉴**라 [[05]] Q3(발견을 대신해 주지 않는다) 쪽으로 넘어가고
over-action 을 산다([[70]]). 그래서 좁히기 규칙을 후보로 놓고 각각의 반경과 적중을 잰다.

적중 = 표적 도구를 끝내 못 부른 sim 20 개 중, 그 규칙 아래에서 표적이 후보로 제시될 것의 수.
⛔효과가 아니라 반경이다 — 발화가 행동을 바꾸는지는 격리 프로브와 런이 답한다([[78]]).
"""
import gzip, json, os, re, collections
BASE='/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results'
RE=re.compile(r"\b[a-z][a-z0-9_]*_\d{4}\b")
T='apply_checking_account_credit_5829'
TAGS=["bank_t7376_treat_20260828","bank_t7372_control_20260828","bank_t7375_072_20260828",
      "bank_t7369_072_20260828","bank_t7370_radius_20260828","bank_t7368_hard0_20260827"]

def turns(sim):
    """검색 턴마다 (idx, cand, 질의중복?, 그턴까지검색수, 최근tool본문의후보)"""
    deliv=set(); used=set(); seen_q=[]; n=0; out=[]; last_tool=""
    for i,m in enumerate(sim.get('messages') or []):
        if m.get('role')=='tool':
            c=str(m.get('content') or ''); deliv|=set(RE.findall(c)); last_tool=c
        q=None; searched=False
        for tc in (m.get('tool_calls') or []):
            nm=str(tc.get('name') or ''); a=tc.get('arguments')
            a=a if isinstance(a,str) else json.dumps(a or {})
            if nm.startswith('KB_search'):
                searched=True
                try: q=(json.loads(a) if isinstance(a,str) else a).get('query')
                except Exception: q=a
            used|=set(RE.findall(a))|set(RE.findall(nm))
        if searched:
            n+=1
            dup = q in seen_q
            seen_q.append(q)
            cand=sorted(deliv-used)
            recent=[x for x in cand if x in last_tool]
            out.append((i,cand,dup,n,recent))
    return out

rows=[]
for tag in TAGS:
    p=os.path.join(BASE,tag+'.results.json.gz')
    if not os.path.exists(p): continue
    with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f:
        S=json.load(f).get('simulations') or []
    for s in S:
        msgs=s.get('messages') or []
        reach=None; rel=False
        for i,m in enumerate(msgs):
            if m.get('role')=='tool' and T in str(m.get('content') or ''): rel=True
            for tc in (m.get('tool_calls') or []):
                a=tc.get('arguments'); a=a if isinstance(a,str) else json.dumps(a or {})
                if T in a and 'unlock' not in str(tc.get('name') or '') and reach is None: reach=i
        if not rel and reach is None: continue
        rows.append({'tag':tag,'sim':'%s#s%s'%(s.get('task_id'),s.get('seed')),
                     'reach':reach,'turns':turns(s)})

allrows=[]
for tag in TAGS:
    p=os.path.join(BASE,tag+'.results.json.gz')
    if not os.path.exists(p): continue
    with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f:
        S=json.load(f).get('simulations') or []
    for s in S: allrows.append(turns(s))

miss=[r for r in rows if r['reach'] is None]
print('표적 관련 sim %d (못 닿은 것 %d)' % (len(rows), len(miss)))
print()
print('%-34s %-10s %-12s %-16s %s' % ('좁히기 규칙','발화 턴','(반경 대비)','후보 중앙값','못닿은20중 짚음'))
def ev(name, pred):
    fire=[(c) for t in allrows for (i,c,d,n,rc) in t if c and pred(i,c,d,n,rc)]
    tot=sum(1 for t in allrows for _ in t)
    hit=sum(1 for r in miss if any(T in (rc if False else c) and pred(i,c,d,n,rc)
                                   for (i,c,d,n,rc) in r['turns']))
    sz=sorted(len(c) for c in fire) or [0]
    print('%-34s %-10d %-12s %-16d %d/%d'
          % (name, len(fire), '%.0f%%'%(100.0*len(fire)/max(1,tot)), sz[len(sz)//2], hit, len(miss)))
ev('① 무조건(현 제안)',            lambda i,c,d,n,rc: True)
ev('② 2번째 검색부터',             lambda i,c,d,n,rc: n>=2)
ev('③ 3번째 검색부터',             lambda i,c,d,n,rc: n>=3)
ev('④ 같은 질의 반복일 때만',       lambda i,c,d,n,rc: d)
ev('⑤ 후보<=3 일 때만',            lambda i,c,d,n,rc: len(c)<=3)
ev('⑥ 방금 읽은 문서가 이름댄 것만', lambda i,c,d,n,rc: bool(rc))
ev('⑦ ②+⑥',                      lambda i,c,d,n,rc: n>=2 and bool(rc))
