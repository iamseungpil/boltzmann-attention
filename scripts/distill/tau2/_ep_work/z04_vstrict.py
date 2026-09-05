# -*- coding: utf-8 -*-
"""Z04 — rebuild V_strict per the synthesis recipe, verify 591, then K1 the pure-46 and unified()."""
import ast, json, os, re
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)
A2 = 'a2'
ID = re.compile(r'^[a-z][a-z0-9_]{2,39}$')

def leaves(o, out):
    if isinstance(o, dict):
        for k, v in o.items(): leaves(v, out)
    elif isinstance(o, list):
        for v in o: leaves(v, out)
    elif isinstance(o, str):
        out.add(o)

def dom_strs(dom):
    s = set()
    for suf in ('settings.json', 'specific.json'):
        p = os.path.join(A2, '%s.%s' % (dom, suf))
        if os.path.exists(p): leaves(json.load(open(p, encoding='utf-8')), s)
    return {x for x in s if ID.match(x)}

DOMS = ['banking_knowledge', 'retail', 'airline']
per = {d: dom_strs(d) for d in DOMS}
Vdom = set().union(*per.values())
base = set()
leaves(json.load(open(os.path.join(A2, 'base', 'shared.json'), encoding='utf-8')), base)
base = {x for x in base if ID.match(x)}
inter3 = set.intersection(*per.values())
print('V_domain(pre) = %d   base_ids=%d   3dom_intersection=%d' % (len(Vdom), len(base), len(inter3)))

# env-declared tool names
env = set()
for f in ('env_surface.json', 'env_surface_airline_retail.json'):
    p = os.path.join(A2, f)
    if os.path.exists(p):
        j = json.load(open(p, encoding='utf-8'))
        def tools(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ('name', 'tool', 'tool_name') and isinstance(v, str) and ID.match(v): env.add(v)
                    tools(v)
            elif isinstance(o, list):
                for v in o: tools(v)
        tools(j)
        if isinstance(j, dict):
            for k in j:
                if ID.match(k): env.add(k)
print('env-declared tool-ish names = %d' % len(env))

V1 = Vdom - base - inter3
V_strict = {x for x in V1 if ('_' in x) or (x in env)}
print('V_strict = %d  (synthesis claims 591)' % len(V_strict))
json.dump(sorted(V_strict), open('_ep_work/_z04_vstrict.json', 'w'), ensure_ascii=False, indent=0)

# ---- K1 over pure-46 ----
PURE = [l.split() for l in open('_ep_work/_z03_pure.txt', encoding='utf-8').read().split('\n') if l.strip()] \
       if os.path.exists('_ep_work/_z03_pure.txt') else None
