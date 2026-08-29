# -*- coding: utf-8 -*-
import json
A='/home/woori/scratch/tau2-bench/data/simulations/ours_n32int8_floor_bank/results.json'
B='/home/woori/scratch/tau2-bench/data/simulations/bank_x599_q38base_banking_20260829/results.json'
def load(p):
    d=json.load(open(p)); o={}
    for s in d.get('simulations',[]): o.setdefault(s.get('task_id'),[]).append(s)
    return o, {t.get('id'):t for t in d.get('tasks',[])}
a,ta=load(A); b,tb=load(B)
def usr(sim):
    return [ (m.get('content') or '').strip() for m in sim.get('messages',[]) if m.get('role')=='user' ]
for t in ['task_002','task_003','task_004','task_005']:
    print('='*70); print('###',t)
    ua=usr(a[t][0]); ub=usr(b[t][0]) if t in b else []
    print(' 32B user turns: %d / Q38 user turns: %d' % (len(ua),len(ub)))
    print(' --- 첫 발화 동일? %s' % (ua[:1]==ub[:1] if ub else 'N/A'))
    if ua: print('  32B[0]:', ua[0][:220].replace('\n',' '))
    if ub: print('  Q38[0]:', ub[0][:220].replace('\n',' '))
    if len(ua)>1: print('  32B[1]:', ua[1][:220].replace('\n',' '))
    if len(ub)>1: print('  Q38[1]:', ub[1][:220].replace('\n',' '))
# task_002 헤더 보충
print(); print('='*70); print('### task_002 상세')
tk = ta.get('task_002') or {}
ev = tk.get('evaluation_criteria') or {}
print(' gold actions:', [g.get('name') for g in (ev.get('actions') or [])])
for lbl,src in [('32B',a),('Q38',b)]:
    if 'task_002' in src:
        ri=src['task_002'][0].get('reward_info') or {}
        print(' [%s] reward=%s basis=%s db_match=%s' % (lbl, ri.get('reward'), ri.get('reward_basis'), (ri.get('db_check') or {}).get('db_match')))
