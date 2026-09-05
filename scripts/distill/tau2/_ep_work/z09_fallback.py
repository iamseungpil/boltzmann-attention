# -*- coding: utf-8 -*-
"""Z09 — the synthesis says only 3 of 27 fallback sites are confirmed to fire and the other 23
   need instrumentation that 'does not exist'. But a fallback  A2.get(k) or "<lit>"  fires
   *iff* the loaded A2 lacks k (or its value is falsy). That is decidable STATICALLY from the
   three declarations. No run needed."""
import ast, json, os, sys, collections
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D); sys.path.insert(0, D)
import gate_interpreter as GI
DOMS = ('banking_knowledge', 'retail', 'airline')
A2 = {d: (GI.load_domain_a2(d) or {}) for d in DOMS}
G = json.load(open('_ep_work/x770g.json', encoding='utf-8'))
print('x770g buckets:', {k: len(v) for k, v in G.items()})
fb = [r for r in G['fallback'] if r.get('fn') != '<module>']
print('fallback sites (fn != <module>) = %d' % len(fb))
print(json.dumps(fb[0], ensure_ascii=False))

def find_slot(obj, slot, path=''):
    """does slot appear as a dict key anywhere in the declaration tree? return sample paths+values"""
    hits = []
    def go(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == slot: hits.append((p + '/' + k, v))
                go(v, p + '/' + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o[:400]): go(v, p + '[%d]' % i)
    go(obj, path)
    return hits

print('\n%-18s %-6s %-26s %-24s %s' % ('module', 'line', 'fn', 'slot', 'declared in'))
fires_all, fires_some, never = [], [], []
for r in sorted(fb, key=lambda x: (x['mod'], x['line'])):
    slot = r['slot']
    where = {}
    for d in DOMS:
        h = find_slot(A2[d], slot)
        truthy = [x for x in h if x[1] not in (None, '', [], {}, 0, False)]
        where[d] = len(truthy)
    decl = [d for d in DOMS if where[d]]
    verdict = 'FIRES in all 3' if not decl else ('FIRES in %s' % ','.join(d for d in DOMS if not where[d]) if len(decl) < 3 else 'never fires')
    if not decl: fires_all.append(r)
    elif len(decl) < 3: fires_some.append((r, decl))
    else: never.append(r)
    print('%-18s %-6d %-26s %-24s %-28s %s' % (r['mod'], r['line'], r['fn'], slot, ','.join(decl) or '(none)', verdict))

print('\nSTATIC VERDICT over the 27 fallback sites')
print('  slot declared by NO domain  -> literal fires in every domain : %d' % len(fires_all))
print('  slot declared by SOME       -> literal fires in the rest     : %d' % len(fires_some))
print('  slot declared by ALL 3      -> literal never fires (dead)    : %d' % len(never))
print('\n  --- always-fires (the engine literal IS the behaviour) ---')
for r in fires_all: print('    %s:%d  %s   %s' % (r['mod'], r['line'], r['slot'], r['src'][:90]))
print('\n  --- fires in the domains that do not declare it ---')
for r, decl in fires_some: print('    %s:%d  %s   declared only by %s   %s' % (r['mod'], r['line'], r['slot'], decl, r['src'][:70]))
