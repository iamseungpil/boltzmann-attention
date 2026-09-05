# -*- coding: utf-8 -*-
"""Z06 — independent A2 key census: raw layers, loader-merged, gate.json equivalence."""
import json, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D); sys.path.insert(0, D)
import gate_interpreter as GI

A2 = 'a2'
def rd(p):
    p = os.path.join(A2, p)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None

base = rd('base/shared.json') or {}
L1 = {k for k in base if not k.startswith('_')}
L1u = {k for k in base if k.startswith('_')}
print('L1 base/shared.json: %d keys kept, %d underscore-dropped %s' % (len(L1), len(L1u), sorted(L1u)))

rows = {}
for dom in ('banking_knowledge', 'retail', 'airline'):
    s = rd('%s.settings.json' % dom) or {}
    p = rd('%s.specific.json' % dom) or {}
    g = rd('%s.gate.json' % dom) or {}
    L2 = {k for k in s if not k.startswith('_')}; L2n = {k for k in s if k.startswith('_')}
    L3 = {k for k in p if not k.startswith('_')}; L3n = {k for k in p if k.startswith('_')}
    merged = GI.load_domain_a2(dom) or {}
    M = set(merged)
    Mn = {k for k in M if k.startswith('_')}
    G = {k for k in g if not k.startswith('_')}
    rows[dom] = dict(L2=L2, L3=L3, M=M, G=G, L2n=L2n, L3n=L3n, Mn=Mn)
    print('\n=== %s' % dom)
    print('  L2=%d  L3=%d  L2andL3_overlap=%d %s' % (len(L2), len(L3), len(L2 & L3), sorted(L2 & L3)))
    print('  _note-ish keys carried into merged: L2 %d, L3 %d ; merged has %d underscore keys %s'
          % (len(L2n), len(L3n), len(Mn), sorted(Mn)[:8]))
    print('  merged(all)=%d   merged(non-underscore)=%d' % (len(M), len(M - Mn)))
    exp = L1 | L2 | L3
    print('  merged_nonunderscore - (L1|L2|L3)  = %s   <- loader-derived' % sorted((M - Mn) - exp))
    print('  (L1|L2|L3) - merged_nonunderscore  = %s' % sorted(exp - (M - Mn)))
    print('  gate.json non-underscore keys = %d' % len(G))
    print('     gate - merged = %s' % sorted(G - (M - Mn)))
    print('     merged - gate = %s' % sorted((M - Mn) - G))

U = set().union(*[rows[d]['L2'] | rows[d]['L3'] for d in rows]) | L1
print('\nUNION over 3 domains (L1|L2|L3, non-underscore) = %d   (synthesis: 76)' % len(U))
Um = set().union(*[rows[d]['M'] - rows[d]['Mn'] for d in rows])
print('UNION of loader-merged (non-underscore)          = %d' % len(Um))
print('  extra in merged-union: %s' % sorted(Um - U))
