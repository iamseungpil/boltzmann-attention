#!/usr/bin/env python
"""x770f — split engine literals into CONTRACT-DSL vs DOMAIN-PAYLOAD.

Rule (mechanical, reproducible):
  a string S that an engine function names as a live literal is
   * DSL      if S occurs in ANY domain's A2 as a dict KEY (a schema slot the
              engine must be able to address), or as a key in a2/base/shared.json
   * PAYLOAD  if S occurs in A2 only in VALUE position (never a key anywhere),
              or is an env tool name exposed by exactly one domain
   * NEITHER  if S does not occur in A2 at all (engine-internal string)
DOMAIN-SHAPED verdict = the function names >=1 PAYLOAD string.
"""
import ast
import json
import os
import re
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

KEYS, VALS = set(), set()


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            KEYS.add(k)
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, str):
        VALS.add(o)


import glob
A2FILES = ['a2/base/shared.json'] + glob.glob('a2/*.settings.json') + glob.glob('a2/*.specific.json')
for f in A2FILES:
    walk(json.load(open(f, encoding='utf-8')))

ENV = json.load(open('a2/env_surface.json', encoding='utf-8'))
bt = collections.defaultdict(set)
for dom, v in ENV.items():
    for t in list(v.get('tools') or []) + list(v.get('discoverable_user_tools') or []):
        bt[t].add(dom)
SINGLE_TOOL = {t for t, d in bt.items() if len(d) == 1 and len(t) >= 6}

tok = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]{3,44}$')
PAYLOAD = {s for s in VALS if tok.match(s) and s not in KEYS} | SINGLE_TOOL
DSL = {s for s in KEYS if tok.match(s)}
print('A2 dict-KEY strings (DSL slots)      : %d' % len(DSL))
print('A2 VALUE-only identifier strings     : %d' % len(PAYLOAD - SINGLE_TOOL))
print('single-domain env tool names         : %d' % len(SINGLE_TOOL))
print('PAYLOAD lexicon total                : %d' % len(PAYLOAD))
print()

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
rows, mod_lines = [], {}
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
        pl = sorted(lits & PAYLOAD)
        ds = sorted(lits & DSL)
        span = set(range(fn.lineno, fn.end_lineno + 1))
        nested = set()
        for sub in ast.walk(fn):
            if sub is not fn and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested |= set(range(sub.lineno, sub.end_lineno + 1))
        par = getattr(fn, '_parent', None)
        rows.append(dict(mod=m, line=fn.lineno, end=fn.end_lineno, own=len(span - nested),
                         cls=par.name if isinstance(par, ast.ClassDef) else '',
                         name=fn.name, payload=pl, dsl=ds,
                         testish=bool(TESTISH.search(fn.name))))

live = [r for r in rows if not r['testish']]
dirty = [r for r in live if r['payload']]
print('=== VERDICT (%d non-test functions in live-wired closure) ===' % len(live))
print('domain-invariant (names 0 PAYLOAD strings) : %d  (%.1f%%)  own-lines %d'
      % (len(live) - len(dirty), 100.0 * (len(live) - len(dirty)) / len(live),
         sum(r['own'] for r in live if not r['payload'])))
print('domain-shaped    (names >=1 PAYLOAD)       : %d  (%.1f%%)  own-lines %d'
      % (len(dirty), 100.0 * len(dirty) / len(live), sum(r['own'] for r in dirty)))
print()
for r in sorted(dirty, key=lambda x: -x['own']):
    print('%-20s:%-6d %-32s own=%-5d payload=%s'
          % (r['mod'], r['line'], (r['cls'] + '.' if r['cls'] else '') + r['name'],
             r['own'], ', '.join(r['payload'][:10])))
print()
print('--- test/selftest with payload (excluded) ---')
for r in rows:
    if r['testish'] and r['payload']:
        print('  %s:%d %s  %s' % (r['mod'], r['line'], r['name'], r['payload'][:6]))
json.dump(rows, open('_ep_work/x770f.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
