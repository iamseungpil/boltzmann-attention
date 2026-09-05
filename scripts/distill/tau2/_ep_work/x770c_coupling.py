#!/usr/bin/env python
"""x770c — engine<->domain coupling scan (measurement only).

For every function in the live-wired closure, count LIVE string literals
(docstrings/comment-strings excluded) that are:
  D1  a tool name exposed by exactly ONE domain in env_surface.json
  D2  a tool ARG name that appears in only one domain's tool schemas
  D3  a hook engine_denylist identifier / engine_prose_denylist word
Reports per function and per module, and separates test/selftest functions.
"""
import ast
import json
import os
import re
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

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
ENV = json.load(open('a2/env_surface.json', encoding='utf-8'))
by_tool = collections.defaultdict(set)
for dom, v in ENV.items():
    for t in (v.get('tools') or []):
        by_tool[t].add(dom)
    for t in (v.get('discoverable_user_tools') or []):
        by_tool[t].add(dom)
SINGLE = {t: sorted(d)[0] for t, d in by_tool.items() if len(d) == 1 and len(t) >= 6}
HOOK = json.load(open(r'C:\workspace\.claude\hooks\scaffold_rules.json', encoding='utf-8'))
DENY_ID, DENY_PROSE = HOOK['engine_denylist'], HOOK['engine_prose_denylist']
GUARDED = set(HOOK['guarded_engine'])
TESTISH = re.compile(r'(selftest|self_test|^test|_test$|smoke|demo|main)', re.I)

rows = []
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
        lits = [n.value for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docids]
        # D1: exact-match single-domain tool names (exact literal, or literal prefix use)
        d1 = sorted({t for t in SINGLE if any(t == s or t in s for s in lits)})
        blob = '\n'.join(lits).lower()
        d3a = sorted({x for x in DENY_ID if x.lower() in blob})
        prose = ' '.join(s for s in lits if len(s) >= 8).lower()
        d3b = sorted({w for w in DENY_PROSE if w.lower() in prose})
        if not (d1 or d3a or d3b):
            continue
        par = getattr(fn, '_parent', None)
        rows.append(dict(mod=m, line=fn.lineno, end=fn.end_lineno,
                         cls=par.name if isinstance(par, ast.ClassDef) else '',
                         name=fn.name, d1=d1, d3a=d3a, d3b=d3b,
                         doms=sorted({SINGLE[t] for t in d1}),
                         testish=bool(TESTISH.search(fn.name)),
                         guarded=(p in GUARDED)))

json.dump(rows, open('_ep_work/x770c_rows.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

live = [r for r in rows if not r['testish']]
print('single-domain tool names in env_surface : %d' % len(SINGLE))
print('functions with domain vocab in LIVE literals : %d (%d non-test)'
      % (len(rows), len(live)))
print()
print('--- non-test functions, by module ---')
c = collections.Counter(r['mod'] for r in live)
for m, n in c.most_common():
    print('  %-22s %2d fns   (guarded=%s)' % (m, n, m + '.py' in GUARDED))
print()
print('--- every non-test hit ---')
for r in sorted(live, key=lambda x: (x['mod'], x['line'])):
    print('%s:%d  %s%s  doms=%s' % (r['mod'] + '.py', r['line'],
                                    (r['cls'] + '.' if r['cls'] else ''), r['name'], r['doms']))
    if r['d1']:
        print('      tools : %s' % (', '.join(r['d1'][:6])))
    if r['d3a']:
        print('      hookID: %s' % (', '.join(r['d3a'])))
    if r['d3b']:
        print('      prose : %s' % (', '.join(r['d3b'])))
print()
print('--- test/selftest functions with domain vocab (excluded above) ---')
for r in sorted(rows, key=lambda x: (x['mod'], x['line'])):
    if r['testish']:
        print('  %s:%d %s  %s' % (r['mod'] + '.py', r['line'], r['name'], r['d1'][:4] + r['d3b'][:2]))
