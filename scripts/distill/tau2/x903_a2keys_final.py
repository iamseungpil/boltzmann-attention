# -*- coding: utf-8 -*-
"""x903 — A2 최상위 키 전수 인벤토리 (banking / retail / airline).
①항목수 ②소비 엔진함수(live 48모듈, 리터럴 grep, 함수명까지) ③형태 동형성(주석키 제거 후) ④도구명 리터럴."""
import json, os, re

D = os.path.dirname(os.path.abspath(__file__))
LIVE = json.load(open(os.path.join(D, '_x901_live.json'), encoding='utf-8'))
src = {}
for m in LIVE:
    p = os.path.join(D, m + '.py')
    if os.path.exists(p):
        src[m] = open(p, encoding='utf-8').read()

# 최상위 def/class 인덱스 (모듈별)
FIDX = {}
for m, txt in src.items():
    idx = []
    for mm in re.finditer(r'^(?:def|class)\s+(\w+)', txt, re.M):
        idx.append((txt.count('\n', 0, mm.start()), mm.group(1)))
    FIDX[m] = idx

def func_at(m, pos):
    line = src[m].count('\n', 0, pos)
    best = '<module>'
    for l, nm in FIDX[m]:
        if l <= line: best = nm
        else: break
    return best

def consumers(key):
    hits = set()
    for p in ('"%s"' % key, "'%s'" % key):
        for m, txt in src.items():
            for mm in re.finditer(re.escape(p), txt):
                hits.add((m, func_at(m, mm.start())))
    return sorted(hits)

NOTEPFX = ('_note', '_quote', '_source', '_ground_note', '_doc', '_why', '_ref')
def strip_notes(v):
    if isinstance(v, dict):
        return {k: strip_notes(x) for k, x in v.items() if not k.startswith('_')}
    if isinstance(v, list):
        return [strip_notes(x) for x in v]
    return v

def shape(v):
    if isinstance(v, dict): return 'D{' + ','.join(sorted(v)) + '}'
    if isinstance(v, list): return 'L[' + '|'.join(sorted({shape(x) for x in v})) + ']'
    return type(v).__name__

env = json.load(open(os.path.join(D, 'a2', 'env_surface.json'), encoding='utf-8'))
env2 = json.load(open(os.path.join(D, 'a2', 'env_surface_airline_retail.json'), encoding='utf-8'))

DOMS = {
 'banking_knowledge': ('a2/banking_knowledge.gate.json', 'a2/banking_knowledge.settings.json', 'a2/banking_knowledge.specific.json'),
 'retail':            ('a2/retail.gate.json',            'a2/retail.settings.json',            'a2/retail.specific.json'),
 'airline':           ('a2/airline.gate.json',           'a2/airline.settings.json',           'a2/airline.specific.json'),
}
base = json.load(open(os.path.join(D, 'a2/base/shared.json'), encoding='utf-8'))
L1KEYS = {k for k in base if not k.startswith('_')}

allrows = {}
for dom, (gp, sp, xp) in DOMS.items():
    gate = json.load(open(os.path.join(D, gp), encoding='utf-8'))
    L2 = set(json.load(open(os.path.join(D, sp), encoding='utf-8')))
    L3 = set(json.load(open(os.path.join(D, xp), encoding='utf-8')))
    src_env = env.get(dom) or env2.get(dom)
    TOOLS = sorted(src_env['tools'], key=len, reverse=True)
    rows = []
    for k, v in gate.items():
        layer = 'L3' if k in L3 else ('L2' if k in L2 else ('L1' if k in L1KEYS else 'loader'))
        its = list(v.values()) if isinstance(v, dict) else (list(v) if isinstance(v, list) else [v])
        kind = 'dict' if isinstance(v, dict) else ('list' if isinstance(v, list) else 'scalar')
        sig_raw = {shape(x) for x in its}
        sig_n   = {shape(strip_notes(x)) for x in its}
        s = json.dumps(v, ensure_ascii=False)
        th = sorted({t for t in TOOLS if re.search(r'\b' + re.escape(t) + r'\b', s)})
        c = consumers(k)
        rows.append(dict(key=k, dom=dom, layer=layer, kind=kind, n=len(its),
                         bytes=len(s.encode('utf-8')),
                         shp_raw=len(sig_raw), shp_norm=len(sig_n),
                         shapes_norm=sorted(sig_n),
                         cons=c, n_cons=len(c), n_tools=len(th), tools=th,
                         doc=k.startswith('_')))
    allrows[dom] = rows
json.dump(allrows, open(os.path.join(D, '_x903_rows.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

for dom, rows in allrows.items():
    data = [r for r in rows if not r['doc']]
    doc  = [r for r in rows if r['doc']]
    print('#### %s : top-level %d = data %d + doc %d | data %dB doc %dB'
          % (dom, len(rows), len(data), len(doc), sum(r['bytes'] for r in data), sum(r['bytes'] for r in doc)))
    from collections import Counter
    print('   layer(data):', dict(Counter(r['layer'] for r in data)))
    print('   dead(cons=0):', [r['key'] for r in data if r['n_cons'] == 0])
    prol = [r for r in data if r['n'] > 1 and r['shp_norm'] > 1]
    print('   multi-shape(norm, n>1): %d ->' % len(prol), [(r['key'], r['n'], r['shp_norm']) for r in prol])
    print('   tool-literal keys: %d/%d' % (sum(1 for r in data if r['n_tools']), len(data)))
    print()
