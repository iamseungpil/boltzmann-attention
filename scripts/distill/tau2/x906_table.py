# -*- coding: utf-8 -*-
import json, os, re
D = os.path.dirname(os.path.abspath(__file__))
g = json.load(open(os.path.join(D,'a2/banking_knowledge.gate.json'), encoding='utf-8'))
r = json.load(open(os.path.join(D,'a2/retail.gate.json'), encoding='utf-8'))
a = json.load(open(os.path.join(D,'a2/airline.gate.json'), encoding='utf-8'))
rows3 = json.load(open(os.path.join(D,'_x903_rows.json'), encoding='utf-8'))['banking_knowledge']
cls4  = {o['key']: o for o in json.load(open(os.path.join(D,'_x904.json'), encoding='utf-8'))}
out=[]
for R in rows3:
    if R['doc']: continue
    k=R['key']; o=cls4[k]
    dom = ('3dom' if (k in r and k in a) else ('2dom' if (k in r or k in a) else 'bank-only'))
    if R['n_cons']==0: v='판정불가'
    elif dom!='bank-only' and (o['cls']!='collection' or o['shp']==1): v='도메인-불변'
    elif dom!='bank-only': v='도메인-불변(증식)'
    elif o['cls']=='collection' and o['n']>1 and o['shp']>1: v='도메인-형상(증식)'
    else: v='도메인-형상'
    cons = R['cons'][:3]
    out.append(dict(key=k, layer=R['layer'], cls=o['cls'], dom=dom, n=R['n'], bytes=R['bytes'],
                    shp=o['shp'], ncons=R['n_cons'], cons=cons, ntools=R['n_tools'], verdict=v))
json.dump(out, open(os.path.join(D,'_x906.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=1)
from collections import Counter
print(Counter(o['verdict'] for o in out))
print(Counter(o['dom'] for o in out))
print()
hdr='%-30s %-6s %-11s %-9s %4s %7s %3s %4s %4s %s'
print(hdr%('KEY','LAYER','CLS','DOM','n','bytes','shp','cons','tool','VERDICT'))
for o in sorted(out,key=lambda x:(x['verdict'],-x['bytes'])):
    print(hdr%(o['key'],o['layer'],o['cls'],o['dom'],o['n'],o['bytes'],o['shp'],o['ncons'],o['ntools'],o['verdict']))
print()
print('bytes by verdict:',{v:sum(o['bytes'] for o in out if o['verdict']==v) for v in set(x['verdict'] for x in out)})
print()
print('=== 대표 소비자 (첫 3개) ===')
for o in sorted(out,key=lambda x:-x['bytes'])[:20]:
    print(' %-28s'%o['key'], o['cons'])
