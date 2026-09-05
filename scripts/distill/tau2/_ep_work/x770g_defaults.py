#!/usr/bin/env python
"""x770g — the A2-DEFAULT FALLBACK census.

Pattern: the engine reads a declared A2 slot but hardcodes a DOMAIN literal as
the fallback when the declaration is absent:
    spec.get("field") or "merchant_name"
    spec.get("id_field") or "item_id"
    (a2 or {}).get("_domain") or "banking_knowledge"
These read as contract but carry domain knowledge — they are the boundary case.

Also counts the plain form  .get("X", "LIT")  and  == "LIT" / in ("LIT",...)
where LIT is in the DOMAIN lexicon.
"""
import ast
import json
import os
import re
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# hand-audited DOMAIN lexicon (see x770f enumeration of 99 payload strings).
# Included only if the string exists because a specific tau2 domain exists.
DOMAIN = {
    # banking-only env tools + their arg keys
    'give_discoverable_user_tool', 'call_discoverable_agent_tool',
    'call_discoverable_user_tool', 'unlock_discoverable_agent_tool',
    'KB_search', 'KB_search_bm25', 'KB_search_dense', 'verify_identity',
    'agent_tool_name', 'user_tool_name', 'discoverable_tool_name',
    # domain names
    'banking_knowledge', 'retail', 'airline',
    # retail/airline entity + field names
    'order', 'order_id', 'item_id', 'item_ids', 'min_price', 'product',
    'options', 'price',
    # banking record fields
    'merchant', 'merchant_name', 'transaction_type', 'doc_limits', 'doc_minimums',
    # banking credit-card catalog schema (t2_compute catalog_filter)
    'card', 'annual_fee', 'base_cashback', 'cashback', 'category_rates',
    'credit_score', 'min_score', 'invite_only', 'invited', 'virtual_card',
    'needs_virtual_card', 'purchase_protection', 'needs_purchase_protection',
    'min_credit_limit', 'limit_max', 'fx_fee', 'max_fx_fee', 'max_annual_fee',
    'min_cashback', 'min_payment_pct', 'max_min_payment_pct',
    'all_purchases_rate', 'spend_category', 'catalog_filter', 'catalog_compute',
}

LOCAL = {f[:-3] for f in os.listdir('.') if f.endswith('.py')}
IMP = re.compile(r'^\s*(?:from\s+([A-Za-z_]\w*)\s+import|import\s+([A-Za-z_]\w*))', re.M)


def closure(entry):
    seen, stack = set(), [entry]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        if not os.path.exists(m + '.py'):
            continue
        for a, b in IMP.findall(open(m + '.py', encoding='utf-8', errors='replace').read()):
            mod = a or b
            if mod in LOCAL and mod not in seen:
                stack.append(mod)
    return sorted(m for m in seen if os.path.exists(m + '.py'))


MODS = closure('t2_run_gated')
TESTISH = re.compile(r'(selftest|self_test|^test|_test$|smoke|^main$)', re.I)


def enclosing(tree, lineno):
    best = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if fn.lineno <= lineno <= fn.end_lineno:
                if best is None or fn.lineno > best.lineno:
                    best = fn
    return best


FALLBACK, COMPARE, PLAIN = [], [], []
for m in MODS:
    p = m + '.py'
    src = open(p, encoding='utf-8', errors='replace').read()
    lines = src.split('\n')
    tree = ast.parse(src)
    docids = {id(n.value) for n in ast.walk(tree)
              if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
              and isinstance(n.value.value, str)}

    def rec(node, kind, lit, slot=''):
        fn = enclosing(tree, node.lineno)
        nm = fn.name if fn else '<module>'
        if TESTISH.search(nm):
            return
        rec_ = dict(mod=m, line=node.lineno, fn=nm, lit=lit, slot=slot,
                    src=lines[node.lineno - 1].strip()[:140])
        {'fallback': FALLBACK, 'compare': COMPARE, 'plain': PLAIN}[kind].append(rec_)

    for n in ast.walk(tree):
        # A) x.get("slot") or "DOMAINLIT"      /  x.get("slot") or ["a","b"]
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or):
            vals = n.values
            head = vals[0]
            if isinstance(head, ast.Call) and isinstance(head.func, ast.Attribute) \
               and head.func.attr == 'get' and head.args and isinstance(head.args[0], ast.Constant) \
               and isinstance(head.args[0].value, str):
                slot = head.args[0].value
                for tail in vals[1:]:
                    lits = [c.value for c in ast.walk(tail)
                            if isinstance(c, ast.Constant) and isinstance(c.value, str)
                            and id(c) not in docids]
                    hit = sorted(set(lits) & DOMAIN)
                    if hit:
                        rec(n, 'fallback', hit, slot)
        # B) x.get("slot", "DOMAINLIT")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'get' \
           and len(n.args) == 2 and isinstance(n.args[0], ast.Constant) \
           and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str) \
           and n.args[1].value in DOMAIN:
            rec(n, 'fallback', [n.args[1].value], n.args[0].value)
        # C) == "DOMAINLIT"   /   in ("DOMAINLIT", ...)
        if isinstance(n, ast.Compare):
            lits = [c.value for c in ast.walk(n)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)
                    and id(c) not in docids]
            hit = sorted(set(lits) & DOMAIN)
            if hit:
                rec(n, 'compare', hit)

seen = set()
FB = [r for r in FALLBACK if not (r['mod'], r['line'], tuple(r['lit'])) in seen
      and not seen.add((r['mod'], r['line'], tuple(r['lit'])))]
seen = set()
CP = [r for r in COMPARE if not (r['mod'], r['line'], tuple(r['lit'])) in seen
      and not seen.add((r['mod'], r['line'], tuple(r['lit'])))]

print('=== A. A2-slot read with DOMAIN literal as FALLBACK  (n=%d) ===' % len(FB))
for r in sorted(FB, key=lambda x: (x['mod'], x['line'])):
    print('%s:%d  [%s]  slot=%r  domain=%s' % (r['mod'] + '.py', r['line'], r['fn'],
                                               r['slot'], r['lit']))
    print('     %s' % r['src'])
print()
print('=== B. engine COMPARES a value against a DOMAIN literal  (n=%d) ===' % len(CP))
byfn = collections.Counter((r['mod'], r['fn']) for r in CP)
for r in sorted(CP, key=lambda x: (x['mod'], x['line'])):
    print('%s:%d  [%s]  %s' % (r['mod'] + '.py', r['line'], r['fn'], r['lit']))
    print('     %s' % r['src'])
print()
print('functions involved: fallback %d | compare %d | union %d'
      % (len({(r['mod'], r['fn']) for r in FB}), len(byfn),
         len({(r['mod'], r['fn']) for r in FB} | set(byfn))))
json.dump(dict(fallback=FB, compare=CP), open('_ep_work/x770g.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
