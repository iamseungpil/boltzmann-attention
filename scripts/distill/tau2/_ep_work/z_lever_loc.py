import re, json, collections, os
live=json.load(open('_x901_live.json',encoding='utf-8'))
g=open('go_stack.sh',encoding='utf-8',errors='replace').read()
ex=re.findall(r'^export (T2_[A-Z0-9_]+)=([^\s#]*)',g,re.M)
on=[k for k,v in ex if v.strip('"')!='0']
src={m:open(m+'.py',encoding='utf-8',errors='replace').read().splitlines() for m in live}
UNI=('t2_gate_patch',8685,15522)          # unified() span at HEAD
loc=collections.defaultdict(list)
for lev in on:
    pat=re.compile(re.escape(lev))
    for m,lines in src.items():
        for i,l in enumerate(lines,1):
            if pat.search(l) and 'environ' in l:
                loc[lev].append((m,i))
found={k:v for k,v in loc.items() if v}
inuni=[k for k,v in found.items() if any(m==UNI[0] and UNI[1]<=i<=UNI[2] for m,i in v)]
bymod=collections.Counter()
for k,v in found.items():
    bymod[v[0][0]]+=1
print('go_stack levers ON: %d ; found as os.environ read in live engine: %d ; NOT found: %d'
      %(len(on),len(found),len(on)-len(found)))
print('  ... read at a line INSIDE unified() (8685-15522): %d  (= %.0f%% of located ON levers)'
      %(len(inuni),100.0*len(inuni)/max(1,len(found))))
print('  ... located in t2_scaffold_get.py: %d'%sum(1 for k,v in found.items() if v[0][0]=='t2_scaffold_get'))
print('  ... located in gate_interpreter.py: %d'%sum(1 for k,v in found.items() if v[0][0]=='gate_interpreter'))
print()
print('first-hit module distribution of ON levers:')
for m,n in bymod.most_common(12): print('   %-22s %d'%(m,n))
print()
print('ON levers not located anywhere as an environ read (never consulted?):')
print('   ',[k for k in on if k not in found][:25])
