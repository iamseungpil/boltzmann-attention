import ast, json, os, re, itertools, collections
DOMS=['banking_knowledge','retail','airline']
def leaves(o,acc):
    if isinstance(o,dict):
        for v in o.values(): leaves(v,acc)
    elif isinstance(o,list):
        for v in o: leaves(v,acc)
    elif isinstance(o,str): acc.append(o)
RE=re.compile(r'^[a-z][a-z0-9_]{2,39}$')
per={}
for d in DOMS:
    acc=[]
    for suf in ['.settings.json','.specific.json']:
        p='a2/%s%s'%(d,suf)
        if os.path.exists(p): leaves(json.load(open(p,encoding='utf-8')),acc)
    per[d]=set(s for s in acc if RE.match(s))
allv=set(itertools.chain(*per.values()))
bacc=[]; leaves(json.load(open('a2/base/shared.json',encoding='utf-8')),bacc); base=set(bacc)
inter=set.intersection(*per.values())
tools=set()
for f in ['a2/env_surface.json','a2/env_surface_airline_retail.json']:
    if os.path.exists(f):
        j=json.load(open(f,encoding='utf-8'))
        def walk(o):
            if isinstance(o,dict):
                for k,v in o.items():
                    if k in ('tools','tool','name','tool_name'):
                        if isinstance(v,dict): tools.update(v.keys())
                        elif isinstance(v,str): tools.add(v)
                        elif isinstance(v,list):
                            for e in v:
                                if isinstance(e,str): tools.add(e)
                                elif isinstance(e,dict) and isinstance(e.get('name'),str): tools.add(e['name'])
                    walk(v)
            elif isinstance(o,list):
                for v in o: walk(v)
        walk(j)
pool = allv - base - inter
V_strict = {s for s in pool if ('_' in s or s in tools)}
DROPPED  = pool - V_strict           # single-token, not a declared tool name
# how many domains declare each dropped term? 1 = maximally domain-specific
only1 = {s for s in DROPPED if sum(1 for d in DOMS if s in per[d])==1}
live=json.load(open('_x901_live.json',encoding='utf-8'))
TESTISH=re.compile(r'(selftest|self_test|^test|_test$|smoke|^main$)',re.I)
def scan(V):
    fns=set(); sites=[]
    for m in live:
        src=open(m+'.py',encoding='utf-8',errors='replace').read()
        try: tree=ast.parse(src)
        except SyntaxError: continue
        for n in ast.walk(tree):
            if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
            if TESTISH.search(n.name): continue
            doc=set()
            for st in ast.walk(n):
                if isinstance(st,ast.Expr) and isinstance(st.value,ast.Constant) and isinstance(st.value.value,str):
                    doc.add(id(st.value))
            hit=[(c.lineno,c.value) for c in ast.walk(n)
                 if isinstance(c,ast.Constant) and isinstance(c.value,str) and id(c) not in doc and c.value in V]
            if hit:
                fns.add((m,n.name,n.lineno)); sites.append((m,n.name,n.lineno,hit))
    return fns,sites
fs,ss=scan(V_strict)
fd,sd=scan(only1)
print('V_strict           %4d terms -> %3d fns' % (len(V_strict), len(fs)))
print('DROPPED single-tok %4d terms (of which declared by exactly ONE domain: %d)' % (len(DROPPED), len(only1)))
print('  -> functions holding a ONE-DOMAIN single-token literal: %d ; of these NOT flagged by V_strict: %d'
      % (len(fd), len(fd-fs)))
print()
cnt=collections.Counter()
for m,n,l,h in sd:
    for _,v in h: cnt[v]+=1
print('most frequent one-domain single-token literals live in engine code:')
for v,c in cnt.most_common(25):
    ds=[d for d in DOMS if v in per[d]]
    print('   %-16s x%-4d declared only by %s' % (v,c,ds[0]))
print()
print('--- functions that hold a one-domain literal but K1(V_strict) calls CLEAN ---')
for m,n,l,h in sorted(sd):
    if (m,n,l) in (fd-fs):
        print('  %-22s:%-6d %-34s %s' % (m,l,n,[x[1] for x in h][:6]))
