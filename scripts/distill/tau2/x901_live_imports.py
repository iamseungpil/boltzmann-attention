import ast, os, sys, json
D=os.path.dirname(os.path.abspath(__file__))
def mods(p):
    try: t=ast.parse(open(p,encoding='utf-8').read())
    except Exception: return []
    out=[]
    for n in ast.walk(t):
        if isinstance(n,ast.Import):
            for a in n.names: out.append(a.name.split('.')[0])
        elif isinstance(n,ast.ImportFrom):
            if n.module and n.level==0: out.append(n.module.split('.')[0])
            elif n.module: out.append(n.module.split('.')[0])
    return out
seen=set(); stack=['t2_run_gated']
while stack:
    m=stack.pop()
    if m in seen: continue
    p=os.path.join(D,m+'.py')
    if not os.path.exists(p): continue
    seen.add(m)
    for x in mods(p):
        if x not in seen and os.path.exists(os.path.join(D,x+'.py')): stack.append(x)
print(json.dumps(sorted(seen),ensure_ascii=False))
print('N=',len(seen))
