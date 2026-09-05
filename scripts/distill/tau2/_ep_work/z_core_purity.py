import ast, json, re, collections
rows=json.load(open('_ep_work/x770b_rows.json',encoding='utf-8'))
TESTISH=re.compile(r'(selftest|self_test|^test|_test$|smoke|^main$)',re.I)
RUNTIME={'orch','env','agent','state','ag','self','sim','runner'}
G=json.load(open('_ep_work/x770g.json',encoding='utf-8'))
dirty={(r['mod'],r['fn']) for r in G['fallback']}|{(r['mod'],r['fn']) for r in G['compare']}
dirty={k for k in dirty if k[1]!='<module>' and not TESTISH.search(k[1])}
live=[r for r in rows if not TESTISH.search(r['name'])]
MONO=('t2_gate_patch','unified',8685)
rest=[r for r in live if (r['mod'],r['name'],r['line'])!=MONO]
contract=[r for r in rest if r['a2_keys'] or r['decl_params']]
judge=[r for r in contract if r['judge_name'] or 'bool' in r['ret_kinds'] or (r['bare_none'] and 'str' in r['ret_kinds']) or 'list' in r['ret_kinds']]
pure=[r for r in judge if not (set(r['params']) & RUNTIME)]
pure_clean=[r for r in pure if (r['mod'],r['name']) not in dirty]
key={(r['mod'],r['name'],r['line']) for r in pure_clean}
src={}; tree={}
for m in {k[0] for k in key}:
    src[m]=open(m+'.py',encoding='utf-8',errors='replace').read()
    tree[m]=ast.parse(src[m])
tau=[]; spans=[]
for m,t in tree.items():
    for n in ast.walk(t):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and (m,n.name,n.lineno) in key:
            body=ast.get_source_segment(src[m],n) or ''
            hits=re.findall(r'^\s*(?:from|import)\s+tau2[\w.]*',body,re.M)
            nested=sum((c.end_lineno-c.lineno+1) for c in ast.walk(n)
                       if isinstance(c,(ast.FunctionDef,ast.AsyncFunctionDef)) and c is not n)
            spans.append((m,n.name,n.lineno,n.end_lineno,n.end_lineno-n.lineno+1,nested))
            if hits: tau.append((m,n.name,n.lineno,sorted(set(h.strip() for h in hits))[:3]))
print('pure-contract predicates : %d  own-lines %d'%(len(pure_clean),sum(r['own'] for r in pure_clean)))
print('... whose body imports tau2 (NOT offline-callable): %d'%len(tau))
for m,n,l,h in sorted(tau): print('   %-20s:%-6d %-26s %s'%(m,l,n,h))
print()
print('... whose lexical SPAN (what you must copy to ship it) vs counted own-lines:')
tot_span=0; tot_nested=0
for m,n,l,e,sp,ne in sorted(spans,key=lambda x:-x[4])[:8]:
    print('   %-20s:%-6d %-26s span=%-6d nested_def_lines=%d'%(m,l,n,sp,ne))
for m,n,l,e,sp,ne in spans: tot_span+=sp; tot_nested+=ne
print('   TOTAL span of the 46 = %d lines (counted as %d own-lines); nested-def lines inside = %d'
      %(tot_span,sum(r['own'] for r in pure_clean),tot_nested))
