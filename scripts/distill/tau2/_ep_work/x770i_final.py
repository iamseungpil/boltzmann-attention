#!/usr/bin/env python
"""x770i — final consolidated engine-predicate counts."""
import json
import os
import re
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
rows = json.load(open('_ep_work/x770b_rows.json', encoding='utf-8'))
G = json.load(open('_ep_work/x770g.json', encoding='utf-8'))

TESTISH = re.compile(r'(selftest|self_test|^test|_test$|smoke|^main$)', re.I)
RUNTIME = {'orch', 'env', 'agent', 'state', 'ag', 'self', 'sim', 'runner'}
JUDGE_RE = re.compile(r'(deny|violat|unbacked|check|applies|allow|block|verdict|audit|forbid|'
                      r'require|eligib|valid|ok$|_is_|^is_|guard|breach|missing|conflict|gate)', re.I)

live = [r for r in rows if not TESTISH.search(r['name'])]
MONO = ('t2_gate_patch', 'unified', 8685)
mono = [r for r in live if (r['mod'], r['name'], r['line']) == MONO]
rest = [r for r in live if (r['mod'], r['name'], r['line']) != MONO]

mod_lines = {}
for m in {r['mod'] for r in rows}:
    mod_lines[m] = open(m + '.py', encoding='utf-8', errors='replace').read().count('\n') + 1
TOT = sum(mod_lines.values())

# domain-shaped set from the hand-audited fallback/compare census
dirty = {(r['mod'], r['fn']) for r in G['fallback']} | {(r['mod'], r['fn']) for r in G['compare']}
dirty = {k for k in dirty if k[1] != '<module>' and not TESTISH.search(k[1])}

contract = [r for r in rest if r['a2_keys'] or r['decl_params']]
judge = [r for r in contract if r['judge_name'] or 'bool' in r['ret_kinds']
         or (r['bare_none'] and 'str' in r['ret_kinds']) or 'list' in r['ret_kinds']]
pure = [r for r in judge if not (set(r['params']) & RUNTIME)]
pure_clean = [r for r in pure if (r['mod'], r['name']) not in dirty]

def L(rs):
    return sum(r['own'] for r in rs)

print('LIVE-WIRED ENGINE (transitive local imports from t2_run_gated.py)')
print('  modules                          : %d' % len(mod_lines))
print('  lines                            : %d' % TOT)
print('  functions (non-test)             : %d' % len(live))
print('  the wiring monolith  unified()   : %d own-lines (%.1f%% of engine, %.1f%% of t2_gate_patch.py)'
      % (L(mono), 100.0 * L(mono) / TOT, 100.0 * L(mono) / mod_lines['t2_gate_patch']))
print()
print('CONTRACT LAYER (excludes unified())')
print('  reads an A2 key OR takes a declaration param : %d fns, %d own-lines (%.1f%% of engine)'
      % (len(contract), L(contract), 100.0 * L(contract) / TOT))
print('  ... of those, judgment-shaped               : %d fns, %d own-lines' % (len(judge), L(judge)))
print('  ... of those, no runtime object in signature: %d fns, %d own-lines' % (len(pure), L(pure)))
print('  ... and free of domain literals             : %d fns, %d own-lines (%.1f%% of engine)'
      % (len(pure_clean), L(pure_clean), 100.0 * L(pure_clean) / TOT))
print()
print('DOMAIN-SHAPED FUNCTIONS (hand-audited lexicon; selftests excluded)')
print('  fallback sites (A2 slot + domain default)   : %d in %d fns'
      % (len([r for r in G['fallback'] if r['fn'] != '<module>']),
         len({(r['mod'], r['fn']) for r in G['fallback'] if r['fn'] != '<module>'})))
print('  compare sites  (== / in domain literal)     : %d in %d fns'
      % (len([r for r in G['compare'] if r['fn'] != '<module>' and not TESTISH.search(r['fn'])]),
         len({(r['mod'], r['fn']) for r in G['compare']
              if r['fn'] != '<module>' and not TESTISH.search(r['fn'])})))
print('  union of functions                          : %d / %d non-test fns = %.1f%%'
      % (len(dirty), len(live), 100.0 * len(dirty) / len(live)))
print()
print('  per module:')
for m, n in collections.Counter(k[0] for k in dirty).most_common():
    print('    %-20s %d' % (m, n))
print()
print('THE PURE CONTRACT PREDICATES (offline-callable, domain-literal-free, judgment-shaped)')
for r in sorted(pure_clean, key=lambda x: -x['own']):
    print('  %-20s:%-6d %-32s own=%-4d A2=%s'
          % (r['mod'], r['line'], (r['cls'] + '.' if r['cls'] else '') + r['name'], r['own'],
             ','.join(r['a2_keys'][:5]) or '(decl-param only)'))
