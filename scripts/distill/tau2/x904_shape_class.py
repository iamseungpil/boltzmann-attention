# -*- coding: utf-8 -*-
"""x904 — 키 유형 분류(collection/record/scalar) + collection 증식 지표(필드 합집합 vs 공통)."""
import json, os, re
D = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(D, '_x903_rows.json'), encoding='utf-8'))['banking_knowledge']
g = json.load(open(os.path.join(D, 'a2/banking_knowledge.gate.json'), encoding='utf-8'))
LIVE = json.load(open(os.path.join(D, '_x901_live.json'), encoding='utf-8'))
src = {m: open(os.path.join(D, m + '.py'), encoding='utf-8').read() for m in LIVE if os.path.exists(os.path.join(D, m + '.py'))}
def used(f):
    return any(('"%s"' % f) in t or ("'%s'" % f) in t for t in src.values())

def sn(v):
    if isinstance(v, dict): return {k: sn(x) for k, x in v.items() if not k.startswith('_')}
    if isinstance(v, list): return [sn(x) for x in v]
    return v

def classify(v):
    if not isinstance(v, (dict, list)): return 'scalar'
    if isinstance(v, list):
        return 'collection' if all(isinstance(x, dict) for x in v) and v else 'list_of_scalar'
    vals = [x for k, x in v.items() if not k.startswith('_')]
    if not vals: return 'empty'
    if all(isinstance(x, (dict, list)) for x in vals) and len(vals) > 1: return 'collection'
    return 'record'

out = []
for r in rows:
    if r['doc']: continue
    v = g[r['key']]
    c = classify(v)
    items = None
    if c == 'collection':
        items = list(v) if isinstance(v, list) else [x for k, x in v.items() if not k.startswith('_')]
        items = [sn(x) for x in items if isinstance(x, dict)]
    fu = fc = None; opt = []
    if items:
        S = [set(x) for x in items]
        U = set().union(*S); C = set.intersection(*S)
        fu, fc = len(U), len(C)
        from collections import Counter
        cnt = Counter(f for s in S for f in s)
        opt = sorted(((f, cnt[f], used(f)) for f in U - C), key=lambda x: -x[1])
    out.append(dict(key=r['key'], cls=c, layer=r['layer'], n=r['n'], bytes=r['bytes'],
                    shp=r['shp_norm'], cons=r['n_cons'], tools=r['n_tools'],
                    fu=fu, fc=fc, n_opt=len(opt),
                    opt_unused=[f for f, k, u in opt if not u]))
json.dump(out, open(os.path.join(D, '_x904.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

from collections import Counter
print('분류:', dict(Counter(o['cls'] for o in out)))
print()
print('%-30s %-13s %-6s %4s %8s %4s %4s %5s %5s %5s' % ('KEY','CLS','LAYER','n','bytes','shp','cons','fUni','fCore','opt'))
for o in sorted(out, key=lambda x: (x['cls'], -x['bytes'])):
    print('%-30s %-13s %-6s %4d %8d %4d %4d %5s %5s %5s' % (o['key'], o['cls'], o['layer'], o['n'], o['bytes'], o['shp'], o['cons'],
        o['fu'] if o['fu'] is not None else '-', o['fc'] if o['fc'] is not None else '-', o['n_opt'] if o['fu'] is not None else '-'))
print()
coll = [o for o in out if o['cls'] == 'collection']
print('collection 키 %d개 · n>1 인 것 %d개' % (len(coll), sum(1 for o in coll if o['n'] > 1)))
print('  그 중 동형(shp==1): %d' % sum(1 for o in coll if o['n'] > 1 and o['shp'] == 1))
print('  그 중 증식(shp>1) : %d' % sum(1 for o in coll if o['n'] > 1 and o['shp'] > 1))
print('  선택필드 총합:', sum(o['n_opt'] for o in coll if o['n_opt']))
print('  선택필드 중 live 엔진에서 리터럴 미발견:', sorted({f for o in coll for f in o['opt_unused']}))
