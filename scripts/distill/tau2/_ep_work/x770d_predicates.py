#!/usr/bin/env python
"""x770d — the predicate table: every judgment-shaped contract function,
with signature, A2 keys read, domain vocab, and offline-callability."""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
rows = json.load(open('_ep_work/x770b_rows.json', encoding='utf-8'))
ENV = json.load(open('a2/env_surface.json', encoding='utf-8'))
import collections
bt = collections.defaultdict(set)
for dom, v in ENV.items():
    for t in list(v.get("tools") or []) + list(v.get("discoverable_user_tools") or []):
        bt[t].add(dom)
SINGLE = {t for t, d in bt.items() if len(d) == 1 and len(t) >= 6}
TESTISH = re.compile(r'(selftest|self_test|^test|_test$|smoke|^main$)', re.I)

# runtime-object parameter names => NOT offline-callable without a live env
RUNTIME = {'orch', 'env', 'agent', 'state', 'ag', 'self', 'sim', 'runner'}

contract = [r for r in rows if r['a2_keys'] or r['decl_params']]
judge = [r for r in contract if r['judge_name'] or ('bool' in r['ret_kinds'])
         or (r['bare_none'] and 'str' in r['ret_kinds']) or ('list' in r['ret_kinds'])]
judge = [r for r in judge if not TESTISH.search(r['name'])]

src_cache = {}
out = []
for r in judge:
    p = r['file']
    if p not in src_cache:
        src_cache[p] = open(p, encoding='utf-8', errors='replace').read().split('\n')
    lines = src_cache[p]
    sig = lines[r['line'] - 1].strip()
    k = r['line']
    while not sig.rstrip().endswith(':') and k < r['end'] and k - r['line'] < 6:
        k += 1
        sig += ' ' + lines[k - 1].strip()
    rt = sorted(set(r['params']) & RUNTIME)
    out.append(dict(mod=r['mod'], line=r['line'], name=r['name'], cls=r['cls'],
                    own=r['own'], sig=sig[:160], a2=r['a2_keys'],
                    rt=rt, offline=not rt,
                    dom=sorted({t for t in r['tool_lits'] if t in SINGLE}) + r['deny_prose'],
                    guarded=r['guarded']))

out.sort(key=lambda x: -x['own'])
tot_own = sum(o['own'] for o in out)
print('judgment-shaped contract predicates (non-test) : %d, own-lines %d' % (len(out), tot_own))
print('offline-callable (no orch/env/agent/state param) : %d / %d (%.0f%%)'
      % (sum(1 for o in out if o['offline']), len(out),
         100.0 * sum(1 for o in out if o['offline']) / len(out)))
print('carrying single-domain vocab                     : %d'
      % sum(1 for o in out if o['dom']))
print()
hdr = '%-20s %-6s %-34s %5s %-7s %-8s %s'
print(hdr % ('module', 'line', 'function', 'own', 'offline', 'domvocab', 'A2 keys read'))
print('-' * 150)
for o in out:
    print(hdr % (o['mod'], o['line'], (o['cls'] + '.' if o['cls'] else '') + o['name'],
                 o['own'], 'yes' if o['offline'] else 'NO:' + ','.join(o['rt'])[:4],
                 ('DOMAIN' if o['dom'] else '-'),
                 ','.join(o['a2'][:6]) + ('...' if len(o['a2']) > 6 else '')))
json.dump(out, open('_ep_work/x770d_predicates.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
