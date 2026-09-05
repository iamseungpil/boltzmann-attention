import ast, json, os, re, itertools, collections
DOMS = ['banking_knowledge','retail','airline']
def leaves(o, acc):
    if isinstance(o, dict):
        for v in o.values(): leaves(v, acc)
    elif isinstance(o, list):
        for v in o: leaves(v, acc)
    elif isinstance(o, str): acc.append(o)
RE_LOW = re.compile(r'^[a-z][a-z0-9_]{2,39}$')
RE_ANY = re.compile(r'^[A-Za-z][A-Za-z0-9_]{2,39}$')
per = {}
for d in DOMS:
    acc = []
    for suf in ['.settings.json','.specific.json']:
        p = 'a2/%s%s' % (d, suf)
        if os.path.exists(p): leaves(json.load(open(p, encoding='utf-8')), acc)
    per[d] = set(acc)
allv = set(itertools.chain(*per.values()))
low = {s for s in allv if RE_LOW.match(s)}
anyc = {s for s in allv if RE_ANY.match(s)}
cap = anyc - low
# base + 3-domain intersection removal (as described)
bacc = []; leaves(json.load(open('a2/base/shared.json', encoding='utf-8')), bacc)
base = set(bacc)
inter = set.intersection(*[{s for s in per[d] if RE_ANY.match(s)} for d in DOMS])
# declared tool names
tools = set()
for f in ['a2/env_surface.json','a2/env_surface_airline_retail.json']:
    if os.path.exists(f):
        j = json.load(open(f, encoding='utf-8'))
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ('tools','tool','name','tool_name'):
                        if isinstance(v, dict): tools.update(v.keys())
                        elif isinstance(v, str): tools.add(v)
                        elif isinstance(v, list):
                            for e in v:
                                if isinstance(e, str): tools.add(e)
                                elif isinstance(e, dict) and isinstance(e.get('name'), str): tools.add(e['name'])
                    walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
        walk(j)
def mk(pool):
    return {s for s in (pool - base - inter) if ('_' in s or s in tools)}
V_strict = mk(low)
V_ext    = mk(anyc)
live = json.load(open('_x901_live.json', encoding='utf-8'))
TESTISH = re.compile(r'(selftest|self_test|^test|_test$|smoke|^main$)', re.I)

def scan(V):
    fns = set(); sites = 0; per_mod = collections.Counter(); ex = []
    for m in live:
        src = open(m + '.py', encoding='utf-8', errors='replace').read()
        try: tree = ast.parse(src)
        except SyntaxError: continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            if TESTISH.search(n.name): continue
            doc = set()
            for st in ast.walk(n):
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                    doc.add(id(st.value))
            hit = []
            for c in ast.walk(n):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and id(c) not in doc:
                    if c.value in V: hit.append((c.lineno, c.value))
            if hit:
                fns.add((m, n.name, n.lineno)); sites += len(hit); per_mod[m] += 1
                ex.append((m, n.name, n.lineno, hit[:4]))
    return fns, sites, per_mod, ex

f1, s1, pm1, ex1 = scan(V_strict)
f2, s2, pm2, ex2 = scan(V_ext)
print('|V_strict| = %d   -> functions %d, sites %d' % (len(V_strict), len(f1), s1))
print('|V_ext|    = %d   -> functions %d, sites %d' % (len(V_ext), len(f2), s2))
print('capitalized terms added: %d ; NEW functions flagged only by them: %d' % (len(cap), len(f2 - f1)))
print('base=%d  3dom_inter=%d  tools=%d' % (len(base), len(inter), len(tools)))
print()
print('--- functions flagged ONLY by capitalized literals (K1 blind spot) ---')
for m, n, l, h in sorted(ex2):
    if (m, n, l) in (f2 - f1):
        print('  %-22s:%-6d %-38s %s' % (m, l, n, h))
