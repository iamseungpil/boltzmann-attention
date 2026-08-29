# -*- coding: utf-8 -*-
import json
A='/home/woori/scratch/tau2-bench/data/simulations/ours_n32int8_floor_bank/results.json'
B='/home/woori/scratch/tau2-bench/data/simulations/bank_x599_q38base_banking_20260829/results.json'
TARGET=['task_002','task_003','task_004','task_005']

def load(p):
    d=json.load(open(p))
    out={}
    for s in d.get('simulations',[]):
        out.setdefault(s.get('task_id'),[]).append(s)
    return out, {t.get('id'):t for t in d.get('tasks',[])}

a,ta=load(A); b,tb=load(B)
print('32B tasks:',len(a),' Q38 tasks:',len(b))

def calls(sim):
    seq=[]
    for m in sim.get('messages',[]):
        for tc in (m.get('tool_calls') or []):
            nm=tc.get('name') or (tc.get('function') or {}).get('name')
            ar=tc.get('arguments') or (tc.get('function') or {}).get('arguments')
            if isinstance(ar,str):
                try: ar=json.loads(ar)
                except Exception: pass
            seq.append((m.get('role'),nm,ar))
    return seq

def summarize(sim,label):
    ri=sim.get('reward_info') or {}
    print('  [%s] reward=%s basis=%s term=%s' % (label, ri.get('reward'), ri.get('reward_basis'), sim.get('termination_reason')))
    dbc=ri.get('db_check') or {}
    if dbc: print('       db_check: match=%s' % dbc.get('db_match'))
    ac=ri.get('action_checks') or []
    for c in ac:
        act=c.get('action') or {}
        print('       action_check: %s met=%s' % (act.get('name') or act.get('action_id'), c.get('action_match')))
    return calls(sim)

for t in TARGET:
    print('\n' + '='*78)
    print('### %s' % t)
    sa=a.get(t); sb=b.get(t)
    if not sa: print('  32B: 없음'); continue
    if not sb: print('  Q38: 아직 미완(x599 진행 중)'); 
    task = ta.get(t) or {}
    ev = task.get('evaluation_criteria') or {}
    ga = ev.get('actions') or []
    print('  gold actions (%d): %s' % (len(ga), [g.get('name') for g in ga]))
    ca = summarize(sa[0],'32B')
    cb = summarize(sb[0],'Q38') if sb else []
    print('  --- 32B 호출열 (%d) ---' % len(ca))
    for i,(r,n,ar) in enumerate(ca): print('    %2d %s %s' % (i,n,json.dumps(ar,ensure_ascii=False)[:150]))
    if cb:
        print('  --- Q38 호출열 (%d) ---' % len(cb))
        for i,(r,n,ar) in enumerate(cb): print('    %2d %s %s' % (i,n,json.dumps(ar,ensure_ascii=False)[:150]))
        # first divergence
        k=0
        while k<min(len(ca),len(cb)) and ca[k][1]==cb[k][1] and ca[k][2]==cb[k][2]: k+=1
        print('  ★첫 분기점 index=%d' % k)
        if k<len(ca): print('     32B: %s %s' % (ca[k][1], json.dumps(ca[k][2],ensure_ascii=False)[:200]))
        else: print('     32B: (호출 종료)')
        if k<len(cb): print('     Q38: %s %s' % (cb[k][1], json.dumps(cb[k][2],ensure_ascii=False)[:200]))
        else: print('     Q38: (호출 종료)')
