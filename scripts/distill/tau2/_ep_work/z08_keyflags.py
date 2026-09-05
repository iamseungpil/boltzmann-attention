# -*- coding: utf-8 -*-
"""Z08 — K4 mirror test. A key with a textual read-site can still be DEAD in the canonical run
   if every read-site sits under an env flag that go_stack.sh never exports.
   (K4 only asks 'read-site == 0'.)"""
import ast, json, os, re, sys, collections
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D); sys.path.insert(0, D)
import gate_interpreter as GI

live = json.load(open('_ep_work/_z01.json', encoding='utf-8'))['full']
keys = set()
for dom in ('banking_knowledge', 'retail', 'airline'):
    m = GI.load_domain_a2(dom) or {}
    keys |= {k for k in m if not k.startswith('_')}
print('keys under test = %d' % len(keys))

# env flags go_stack.sh exports
GS = open('go_stack.sh', encoding='utf-8', errors='replace').read()
EXPORTED = set(re.findall(r'^\s*export\s+([A-Z0-9_]+)=', GS, re.M))
EXPORTED |= set(re.findall(r'\bexport\s+([A-Z0-9_]+)=', GS))
# multi-var export lines
for line in GS.splitlines():
    if line.strip().startswith('export '):
        for tok in line.strip()[7:].split():
            if '=' in tok: EXPORTED.add(tok.split('=')[0])
print('go_stack.sh exports %d env names' % len(EXPORTED))

trees = {}
def tree(m):
    if m not in trees: trees[m] = ast.parse(open(m + '.py', encoding='utf-8').read())
    return trees[m]

def env_names(node):
    """T2_* env names tested anywhere in this expression."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('get', 'getenv'):
            base = n.func.value
            ok = (isinstance(base, ast.Attribute) and base.attr == 'environ') or \
                 (isinstance(base, ast.Name) and base.id in ('environ', 'os'))
            if ok and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                out.add(n.args[0].value)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == 'environ':
            if isinstance(n.slice, ast.Constant): out.add(n.slice.value)
    return out

sites = collections.defaultdict(list)
for m in live:
    t = tree(m)
    par = {}
    for n in ast.walk(t):
        for c in ast.iter_child_nodes(n): par[c] = n
    for n in ast.walk(t):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in keys:
            # is it a key lookup?  X.get("k") / X["k"] / "k" in X
            p = par.get(n)
            isread = False
            if isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr in ('get', 'setdefault', 'pop'):
                isread = p.args and p.args[0] is n
            elif isinstance(p, ast.Subscript): isread = True
            elif isinstance(p, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in p.ops): isread = True
            if not isread: continue
            # collect env guards on the enclosing If chain
            guards, q = set(), par.get(n)
            while q is not None:
                if isinstance(q, ast.If): guards |= env_names(q.test)
                if isinstance(q, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    pass
                q = par.get(q)
            sites[n.value].append((m, n.lineno, tuple(sorted(guards))))

dead, flagdead, alive = [], [], []
for k in sorted(keys):
    ss = sites.get(k, [])
    if not ss:
        dead.append(k); continue
    # a site is unconditional if it has no env guard, or a guard go_stack exports
    reachable = [s for s in ss if (not s[2]) or any(g in EXPORTED for g in s[2])]
    if reachable: alive.append(k)
    else: flagdead.append((k, ss))

print('\nK4 says DEAD (no read-site at all)      : %d  %s' % (len(dead), dead))
print('\nDEAD IN THE CANONICAL RUN (every read-site is behind a flag go_stack.sh never exports): %d' % len(flagdead))
for k, ss in flagdead:
    gs = sorted({g for s in ss for g in s[2]})
    print('   %-28s sites=%d  guards=%s' % (k, len(ss), gs))
    for m, ln, g in ss[:3]: print('        %s:%d  under %s' % (m, ln, list(g)))
print('\nreachable keys: %d' % len(alive))
