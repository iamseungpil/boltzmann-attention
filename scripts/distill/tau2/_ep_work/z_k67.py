import json, collections, os
DOMS=['banking_knowledge','retail','airline']
CONTRACT9=['declaration','claim_audit','axis_notes','repeat_governor','gates','eplan','action_tools','failure_markers','producers']
def shp(o):
    if isinstance(o,dict): return tuple(sorted(k for k in o if not str(k).startswith('_')))
    return ('<'+type(o).__name__+'>',)
def load(d):
    m={}
    for suf in ['.settings.json','.specific.json']:
        p='a2/%s%s'%(d,suf)
        if os.path.exists(p):
            for k,v in json.load(open(p,encoding='utf-8')).items():
                if not str(k).startswith('_'): m[k]=v
    return m
print('%-22s %-18s %5s %5s  %s'%('key','domain','n','shp','K6/K7 verdict by the report\'s own rule'))
for k in CONTRACT9:
    for d in DOMS:
        m=load(d)
        if k not in m: continue
        v=m[k]
        if isinstance(v,list):
            n=len(v); c=collections.Counter(shp(x) for x in v); s=len(c)
        elif isinstance(v,dict):
            vals=list(v.values())
            if vals and all(isinstance(x,(dict,list)) for x in vals):
                n=len(vals); c=collections.Counter(shp(x) for x in vals); s=len(c)
            else:
                n=1; s=1
        else:
            n=1; s=1
        verd = 'K6 contract' if (n<=1 or s==1) else 'K7 PROLIFERATION'
        print('%-22s %-18s %5d %5d  %s'%(k,d,n,s,verd))
print()
# A3
a3=json.load(open('a2/banking_knowledge.policy_facts.json',encoding='utf-8'))
rows=a3.get('rows') or a3.get('facts') or []
c=collections.Counter(shp(r) for r in rows)
print('A3 policy_facts rows n=%d  distinct shapes=%d -> %s'%(len(rows),len(c),
      'K6 contract' if len(c)==1 else 'K7 PROLIFERATION'))
for s,n in c.most_common(6): print('    x%-5d %s'%(n,list(s)))
