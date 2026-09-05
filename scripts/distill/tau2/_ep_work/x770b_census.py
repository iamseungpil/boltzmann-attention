#!/usr/bin/env python
"""x770b — engine-predicate census, v2 (measurement only).

Fixes vs v1:
  - docstrings and bare string-expression "comments" EXCLUDED from literal scan
    (they never reach the model; the hook's own rationale, scaffold_guard.py:315-318)
  - line accounting by SET UNION of line numbers (nested defs no longer double-count)
  - own-body lines = function span minus nested function spans
  - judgment-shape classification added
"""
import ast
import json
import os
import re
import sys
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
A2_KEYS = set(json.load(open('_ep_work/a2keys.json', encoding='utf-8')))
TOOLS = set(json.load(open('_ep_work/tools.json', encoding='utf-8')))
HOOK = json.load(open(r'C:\workspace\.claude\hooks\scaffold_rules.json', encoding='utf-8'))
DENY_ID, DENY_PROSE = HOOK['engine_denylist'], HOOK['engine_prose_denylist']
GUARDED = set(HOOK['guarded_engine'])

DECL_PARAMS = {'a2', 'a2d', 'spec', 'specs', 'gate', 'gates', 'rules', 'decl',
               'declaration', 'binding', 'bindings', 'cfg', 'ops', 'sp', 'policy'}
JUDGE_RE = re.compile(
    r'(deny|violat|unbacked|check|applies|allow|block|verdict|audit|forbid|'
    r'require|eligib|valid|ok$|_is_|^is_|guard|breach|missing|conflict|gate)', re.I)


def doc_nodes(tree):
    """line numbers of docstrings + bare string-expression 'comments'."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) \
           and isinstance(n.value.value, str):
            out.add(id(n.value))
    return out


def live_strs(fn, docids):
    return [n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docids]


def params(fn):
    a = fn.args
    return [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)] + \
           ([a.vararg.arg] if a.vararg else []) + ([a.kwarg.arg] if a.kwarg else [])


def judge_shape(fn):
    """does it return a verdict? (bool / message-or-None / list-of-violations)"""
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    kinds = set()
    for r in rets:
        v = r.value
        if isinstance(v, ast.Constant):
            if v.value is None:
                kinds.add('none')
            elif isinstance(v.value, bool):
                kinds.add('bool')
            elif isinstance(v.value, str):
                kinds.add('str')
        elif isinstance(v, (ast.List, ast.ListComp)):
            kinds.add('list')
        elif isinstance(v, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
            kinds.add('bool')
        elif isinstance(v, ast.Tuple):
            kinds.add('tuple')
    has_bare_none = any(isinstance(n, ast.Return) and n.value is None for n in ast.walk(fn))
    return sorted(kinds), has_bare_none


rows, mod_lines = [], {}
for m in MODS:
    p = m + '.py'
    src = open(p, encoding='utf-8', errors='replace').read()
    mod_lines[m] = src.count('\n') + 1
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print('PARSE FAIL', p, e, file=sys.stderr)
        continue
    docids = doc_nodes(tree)
    for parent in ast.walk(tree):
        for ch in ast.iter_child_nodes(parent):
            ch._parent = parent
    fns = [n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in fns:
        span = set(range(fn.lineno, fn.end_lineno + 1))
        nested = set()
        for sub in ast.walk(fn):
            if sub is fn:
                continue
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested |= set(range(sub.lineno, sub.end_lineno + 1))
        own = span - nested
        lits = live_strs(fn, docids)
        blob = '\n'.join(lits).lower()
        prose_blob = ' '.join(s for s in lits if len(s) >= 8).lower()
        ps = params(fn)
        kinds, bare_none = judge_shape(fn)
        par = getattr(fn, '_parent', None)
        rows.append(dict(
            mod=m, file=p, cls=par.name if isinstance(par, ast.ClassDef) else '',
            name=fn.name, line=fn.lineno, end=fn.end_lineno,
            span=sorted(span), own=len(own),
            nested_fn=sum(1 for s in ast.walk(fn)
                          if s is not fn and isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))),
            params=ps,
            decl_params=sorted(set(ps) & DECL_PARAMS),
            a2_keys=sorted(set(lits) & A2_KEYS),
            tool_lits=sorted({t for t in TOOLS if t.lower() in blob}),
            deny_id=sorted({d for d in DENY_ID if d.lower() in blob}),
            deny_prose=sorted({w for w in DENY_PROSE if w.lower() in prose_blob}),
            ret_kinds=kinds, bare_none=bare_none,
            judge_name=bool(JUDGE_RE.search(fn.name)),
            guarded=(p in GUARDED),
            top=isinstance(par, ast.Module) or isinstance(par, ast.ClassDef),
        ))

for r in rows:
    r['span'] = [r['line'], r['end']]
json.dump(rows, open('_ep_work/x770b_rows.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

TOT = sum(mod_lines.values())


def cover(rs):
    s = set()
    for r in rs:
        s |= set(range(r['line'], r['end'] + 1))
    return len(s)


top = [r for r in rows if r['top']]
a2r = [r for r in rows if r['a2_keys']]
declr = [r for r in rows if r['decl_params']]
contract = [r for r in rows if r['a2_keys'] or r['decl_params']]
judge = [r for r in contract if r['judge_name'] or
         ('bool' in r['ret_kinds']) or (r['bare_none'] and 'str' in r['ret_kinds']) or
         ('list' in r['ret_kinds'])]
dirty = [r for r in contract if r['tool_lits'] or r['deny_id'] or r['deny_prose']]
dirty_j = [r for r in judge if r['tool_lits'] or r['deny_id'] or r['deny_prose']]

print('=== SCOPE (live-wired closure from t2_run_gated.py) ===')
print('modules %d | lines %d | .py files in tau2/ %d'
      % (len(MODS), TOT, len([f for f in os.listdir('.') if f.endswith('.py')])))
print('functions %d (top-level/method %d, nested %d)'
      % (len(rows), len(top), len(rows) - len(top)))
print('lines covered by any def : %d (%.1f%%)' % (cover(rows), 100.0 * cover(rows) / TOT))
print()
print('=== CONTRACT LAYER ===')
print('A2-key-reading fns  %3d | own-lines %6d | span-cover %6d (%.1f%%)'
      % (len(a2r), sum(r['own'] for r in a2r), cover(a2r), 100.0 * cover(a2r) / TOT))
print('decl-param fns      %3d | own-lines %6d | span-cover %6d (%.1f%%)'
      % (len(declr), sum(r['own'] for r in declr), cover(declr), 100.0 * cover(declr) / TOT))
print('UNION contract      %3d | own-lines %6d | span-cover %6d (%.1f%%)'
      % (len(contract), sum(r['own'] for r in contract), cover(contract),
         100.0 * cover(contract) / TOT))
print('  judgment-shaped   %3d | own-lines %6d' % (len(judge), sum(r['own'] for r in judge)))
print()
print('=== DOMAIN VOCAB IN *LIVE* LITERALS (docstrings excluded) ===')
print('contract fns with domain vocab : %d / %d (%.1f%%)'
      % (len(dirty), len(contract), 100.0 * len(dirty) / max(1, len(contract))))
print('judgment fns with domain vocab : %d / %d (%.1f%%)'
      % (len(dirty_j), len(judge), 100.0 * len(dirty_j) / max(1, len(judge))))
print()
for r in sorted(dirty, key=lambda x: (-x['own'])):
    print('  %-22s:%-6d %-34s own=%-5d tools=%s id=%s prose=%s'
          % (r['mod'], r['line'], (r['cls'] + '.' if r['cls'] else '') + r['name'],
             r['own'], r['tool_lits'][:3], r['deny_id'][:2], r['deny_prose'][:2]))
print()
print('=== GUARDED vs UNGUARDED (hook covers %d files) ===' % len(GUARDED))
gm = [m for m in MODS if m + '.py' in GUARDED]
um = [m for m in MODS if m + '.py' not in GUARDED]
print('live-wired & guarded   : %d mods, %d lines' % (len(gm), sum(mod_lines[m] for m in gm)))
print('live-wired & UNGUARDED : %d mods, %d lines' % (len(um), sum(mod_lines[m] for m in um)))
print('  unguarded mods:', ' '.join(um))
