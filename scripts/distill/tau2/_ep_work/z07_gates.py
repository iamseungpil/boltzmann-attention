# -*- coding: utf-8 -*-
"""Z07 — normalize cross-domain cost by *policy surface*, not by domain.
   The synthesis compares raw bytes (banking 525,675 vs airline 1,849) and calls airline 'cheap'.
   If banking simply has more gates/tools, the right unit is bytes-per-gate and bytes-per-tool."""
import json, os, sys, collections
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D); sys.path.insert(0, D)
import gate_interpreter as GI

def nbytes(o):
    return len(json.dumps(o, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))

def leaves(o):
    if isinstance(o, dict): return sum(leaves(v) for k, v in o.items() if not str(k).startswith('_'))
    if isinstance(o, list): return sum(leaves(v) for v in o)
    return 1

print('%-20s %7s %7s %7s %7s %9s %9s' % ('domain', 'gates', 'tools', 'eplan', 'prod', 'dataB', 'B/gate'))
tot = {}
for dom in ('banking_knowledge', 'retail', 'airline'):
    m = GI.load_domain_a2(dom) or {}
    data = {k: v for k, v in m.items() if not k.startswith('_')}
    g = data.get('gates') or {}
    at = data.get('action_tools') or {}
    ep = data.get('eplan') or {}
    pr = data.get('producers') or {}
    B = nbytes(data)
    ng = len(g) if isinstance(g, (dict, list)) else 0
    print('%-20s %7d %7d %7d %7d %9d %9s' % (dom, ng,
          len(at) if hasattr(at, '__len__') else 0,
          len(ep) if hasattr(ep, '__len__') else 0,
          len(pr) if hasattr(pr, '__len__') else 0,
          B, ('%d' % (B // ng)) if ng else '-'))
    tot[dom] = (data, g)

print('\n--- gates key only ---')
for dom, (data, g) in tot.items():
    gb = nbytes(g); print('%-20s gates=%d  bytes=%d  leaves=%d  B/gate=%s'
                          % (dom, len(g), gb, leaves(g), ('%d' % (gb // len(g))) if len(g) else '-'))

print('\n--- what fraction of each domain is the two GENERATED keys? ---')
for dom, (data, g) in tot.items():
    B = nbytes(data)
    po = nbytes(data.get('policy_ontology') or {})
    sg = nbytes(data.get('scaffold_get_tools') or {})
    print('%-20s total=%-8d policy_ontology=%-8d(%4.1f%%) scaffold_get_tools=%-8d(%4.1f%%) rest=%d'
          % (dom, B, po, 100.0*po/B, sg, 100.0*sg/B, B - po - sg))

print('\n--- per-key bytes, banking, top 20 ---')
data = tot['banking_knowledge'][0]
for k, v in sorted(data.items(), key=lambda kv: -nbytes(kv[1]))[:20]:
    print('   %-28s %9d  %s' % (k, nbytes(v), type(v).__name__))
