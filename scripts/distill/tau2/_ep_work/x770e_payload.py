#!/usr/bin/env python
"""x770e — stronger domain-coupling axis.

Axis: an engine LIVE string literal that also occurs as a *value* (not a
structural key) inside exactly ONE domain's A2 declaration, and does NOT occur
in the shared base layer.  Those are strings the engine names that exist only
because that domain exists.
"""
import ast
import json
import os
import re
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def walk_json(o, out, keys):
    if isinstance(o, dict):
        for k, v in o.items():
            keys.add(k)
            walk_json(v, out, keys)
    elif isinstance(o, list):
        for v in o:
            walk_json(v, out, keys)
    elif isinstance(o, str):
        out.add(o)


def load(p):
    vals, keys = set(), set()
    if os.path.exists(p):
        walk_json(json.load(open(p, encoding='utf-8')), vals, keys)
    return vals, keys


base_v, base_k = load('a2/base/shared.json')
DOM = {}
for d in ('banking_knowledge', 'retail', 'airline'):
    v1, k1 = load('a2/%s.settings.json' % d)
    v2, k2 = load('a2/%s.specific.json' % d)
    DOM[d] = ((v1 | v2), (k1 | k2))

# candidate lexicon: identifier-shaped strings appearing as VALUES (or keys) in
# exactly one domain, absent from base
tok = re.compile(r'^[a-z][a-z0-9_]{4,40}$')
own = collections.defaultdict(set)
allsets = {d: (DOM[d][0] | DOM[d][1]) for d in DOM}
for d in DOM:
    others = set()
    for e in DOM:
        if e != d:
            others |= allsets[e]
    for s in allsets[d]:
        if tok.match(s) and s not in others and s not in base_v and s not in base_k:
            own[d].add(s)
LEX = {s: d for d in own for s in own[d]}
print('single-domain A2 lexicon sizes:',
      {d: len(own[d]) for d in own}, ' total', len(LEX))

LOCAL = {f[:-3] for f in os.listdir('.') if f.endswith('.py')}
IMP = re.compile(r'^\s*(?:from\s+([A-Za-z_]\w*)\s+import|import\s+([A-Za-z_]\w*))', re.M)


def closure(entry):
    seen, stack = set(), [entry]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        if not os.path.exists(m + '.py'):
            continue
        for a, b in IMP.findall(open(m + '.py', encoding='utf-8', errors='replace').read()):
            mod = a or b
            if mod in LOCAL and mod not in seen:
                stack.append(mod)
    return sorted(m for m in seen if os.path.exists(m + '.py'))


MODS = closure('t2_run_gated')
TESTISH = re.compile(r'(selftest|self_test|^test|_test$|smoke|^main$)', re.I)
hits = []
mod_lines = {}
for m in MODS:
    p = m + '.py'
    src = open(p, encoding='utf-8', errors='replace').read()
    mod_lines[m] = src.count('\n') + 1
    tree = ast.parse(src)
    docids = {id(n.value) for n in ast.walk(tree)
              if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
              and isinstance(n.value.value, str)}
    for parent in ast.walk(tree):
        for ch in ast.iter_child_nodes(parent):
            ch._parent = parent
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lits = {n.value for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docids}
        h = sorted(lits & set(LEX))
        if not h:
            continue
        par = getattr(fn, '_parent', None)
        hits.append(dict(mod=m, line=fn.lineno, name=fn.name,
                         cls=par.name if isinstance(par, ast.ClassDef) else '',
                         lex=h, doms=sorted({LEX[s] for s in h}),
                         testish=bool(TESTISH.search(fn.name))))

live = [h for h in hits if not h['testish']]
print('functions naming a single-domain A2 string : %d (%d non-test)' % (len(hits), len(live)))
print()
for h in sorted(live, key=lambda x: (x['mod'], x['line'])):
    print('%s:%d  %s%s  doms=%s' % (h['mod'] + '.py', h['line'],
                                    (h['cls'] + '.' if h['cls'] else ''), h['name'], h['doms']))
    print('      %s' % ', '.join(h['lex'][:12]))
print()
print('--- test/selftest (excluded) ---')
for h in hits:
    if h['testish']:
        print('  %s:%d %s  %s' % (h['mod'] + '.py', h['line'], h['name'], h['lex'][:6]))
json.dump(dict(lexicon={d: sorted(own[d]) for d in own}, hits=hits),
          open('_ep_work/x770e.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
