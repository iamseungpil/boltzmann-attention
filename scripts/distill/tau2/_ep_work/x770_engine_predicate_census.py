#!/usr/bin/env python
"""x770 — engine-predicate census (measurement only, no engine edit).

Counts, over the LIVE-WIRED module closure (transitive local imports from
t2_run_gated.py), every function that (a) reads an A2/A3 declaration key by
string literal, and/or (b) takes a declaration-shaped parameter, and reports
whether domain vocabulary (tool names / hook denylists) appears in its body.

Reproducible: all pattern sets are printed in the header of the output.
"""
import ast
import json
import os
import re
import sys
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ---------- 1. live-wired module closure ----------
LOCAL = {f[:-3] for f in os.listdir('.') if f.endswith('.py')}
IMP = re.compile(r'^\s*(?:from\s+([A-Za-z_]\w*)\s+import|import\s+([A-Za-z_]\w*))', re.M)


def closure(entry):
    seen, stack = set(), [entry]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        p = m + '.py'
        if not os.path.exists(p):
            continue
        src = open(p, encoding='utf-8', errors='replace').read()
        for a, b in IMP.findall(src):
            mod = a or b
            if mod in LOCAL and mod not in seen:
                stack.append(mod)
    return sorted(m for m in seen if os.path.exists(m + '.py'))


MODS = closure('t2_run_gated')

# ---------- 2. vocabularies ----------
A2_KEYS = set(json.load(open('_ep_work/a2keys.json', encoding='utf-8')))
TOOLS = set(json.load(open('_ep_work/tools.json', encoding='utf-8')))
HOOK = json.load(open(r'C:\workspace\.claude\hooks\scaffold_rules.json', encoding='utf-8'))
DENY_ID = [x for x in HOOK['engine_denylist']]
DENY_PROSE = [x for x in HOOK['engine_prose_denylist']]
GUARDED = set(HOOK['guarded_engine'])

# declaration-shaped parameter names (A2/A3 payload passed in)
DECL_PARAMS = {'a2', 'a2d', 'spec', 'specs', 'gate', 'gates', 'rules', 'decl',
               'declaration', 'binding', 'bindings', 'node', 'nodes', 'cfg',
               'ops', 'op', 'g', 'sp', 'policy'}


def strs(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return out


def params(fn):
    a = fn.args
    return [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)] + \
           ([a.vararg.arg] if a.vararg else []) + ([a.kwarg.arg] if a.kwarg else [])


rows = []
mod_lines = {}
for m in MODS:
    p = m + '.py'
    src = open(p, encoding='utf-8', errors='replace').read()
    mod_lines[m] = src.count('\n') + 1
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print('PARSE FAIL', p, e, file=sys.stderr)
        continue
    # attach parent class names
    for parent in ast.walk(tree):
        for ch in ast.iter_child_nodes(parent):
            ch._parent = parent
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lits = strs(fn)
        litset = set(lits)
        a2hit = sorted(litset & A2_KEYS)
        ps = params(fn)
        declp = sorted(set(ps) & DECL_PARAMS)
        blob = '\n'.join(lits).lower()
        toolhit = sorted({t for t in TOOLS if t.lower() in blob})
        idhit = sorted({d for d in DENY_ID if d.lower() in blob})
        # prose denylist only over literals >=8 chars (hook's own rule)
        prose_blob = ' '.join(s for s in lits if len(s) >= 8).lower()
        prosehit = sorted({w for w in DENY_PROSE if w.lower() in prose_blob})
        par = getattr(fn, '_parent', None)
        cls = par.name if isinstance(par, ast.ClassDef) else ''
        rows.append(dict(
            mod=m, file=p, cls=cls, name=fn.name, line=fn.lineno,
            end=getattr(fn, 'end_lineno', fn.lineno),
            nlines=getattr(fn, 'end_lineno', fn.lineno) - fn.lineno + 1,
            params=ps, decl_params=declp, a2_keys=a2hit,
            tool_lits=toolhit, deny_id=idhit, deny_prose=prosehit,
            guarded=(p in GUARDED),
        ))

json.dump(rows, open('_ep_work/x770_rows.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---------- 3. summary ----------
tot_lines = sum(mod_lines.values())
tot_fn = len(rows)
tot_fn_lines = sum(r['nlines'] for r in rows)
a2r = [r for r in rows if r['a2_keys']]
declr = [r for r in rows if r['decl_params']]
contract = [r for r in rows if r['a2_keys'] or r['decl_params']]
dirty = [r for r in contract if r['tool_lits'] or r['deny_id'] or r['deny_prose']]

print('=== SCOPE ===')
print('live-wired modules : %d' % len(MODS))
print('live-wired lines   : %d' % tot_lines)
print('all .py in tau2/   : %d files' % len([f for f in os.listdir('.') if f.endswith('.py')]))
print('functions total    : %d  (%d lines, %.1f%% of live lines)'
      % (tot_fn, tot_fn_lines, 100.0 * tot_fn_lines / tot_lines))
print()
print('=== CONTRACT-SHAPED ===')
print('A2-key-reading fns : %d  (%d lines, %.1f%% of live)'
      % (len(a2r), sum(r['nlines'] for r in a2r), 100.0 * sum(r['nlines'] for r in a2r) / tot_lines))
print('decl-param fns     : %d  (%d lines)' % (len(declr), sum(r['nlines'] for r in declr)))
print('union (contract)   : %d  (%d lines, %.1f%% of live)'
      % (len(contract), sum(r['nlines'] for r in contract),
         100.0 * sum(r['nlines'] for r in contract) / tot_lines))
print('  of which carry domain vocab in body literals : %d (%.1f%%)'
      % (len(dirty), 100.0 * len(dirty) / max(1, len(contract))))
print()
print('=== A2-READING FNS BY MODULE ===')
c = collections.Counter(r['mod'] for r in a2r)
for m, n in c.most_common():
    L = sum(r['nlines'] for r in a2r if r['mod'] == m)
    print('  %-24s %3d fns %6d lines / %5d mod lines' % (m, n, L, mod_lines[m]))
print()
print('=== A2 KEYS: which are read by engine code at all ===')
read = collections.Counter()
for r in a2r:
    for k in r['a2_keys']:
        read[k] += 1
unread = sorted(A2_KEYS - set(read))
print('keys read  : %d' % len(read))
print('keys NEVER read as string literal in live engine : %d' % len(unread))
print('  ', unread)
print()
print('=== DOMAIN VOCAB INSIDE CONTRACT FNS (verbatim needed) ===')
for r in sorted(dirty, key=lambda x: -x['nlines'])[:40]:
    print('  %s:%d %s%s  keys=%s tools=%s id=%s prose=%s'
          % (r['file'], r['line'], (r['cls'] + '.' if r['cls'] else ''), r['name'],
             r['a2_keys'][:4], r['tool_lits'][:4], r['deny_id'][:3], r['deny_prose'][:3]))
