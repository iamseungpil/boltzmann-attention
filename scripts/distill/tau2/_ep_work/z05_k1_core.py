# -*- coding: utf-8 -*-
"""Z05 — K1 (domain-literal in body) applied to (a) the pure-46 'invariant core',
   (b) the unified() monolith that the synthesis parked as K0.
   Lexicon = the published artifact _Vdom.json (715) filtered exactly as the synthesis states:
   keep tokens containing '_' OR present in the env-declared tool list (_ep_work/tools.json, 95)."""
import ast, json, os, re, collections
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)
V = set(json.load(open('_ep_work/_Vdom.json', encoding='utf-8')))
TOOLS = set(json.load(open('_ep_work/tools.json', encoding='utf-8')))
VS = {x for x in V if ('_' in x) or (x in TOOLS)}
print('V_domain artifact=%d  tools=%d  ->  V_strict=%d  (synthesis: 715 / 95 / 591)'
      % (len(V), len(TOOLS), len(VS)))

PURE = """t2_gate_patch 1795 _wev_deny_msgs
t2_gate_patch 5240 _claim_unbacked
t2_ledger 793 eligible_text
t2_gate_patch 2098 _write_arg_ground_deny
t2_dominance 236 requirements_for
t2_gate_patch 1120 _label_mismatch_deny
t2_gate_patch 8425 apply_unified_regen
gate_interpreter 165 _compose_claim_audit
t2_scaffold_get 2393 apply
t2_authority 77 may_suppress
t2_transcribe 107 missing_fields
t2_gate_patch 2890 membership_violation
t2_gate_patch 6973 apply_provenance_regen
t2_resolve 880 action_candidates
t2_resolve 745 _dispatch_since_last_user
t2_dominance 87 dominating_gate
t2_transcribe 159 field_sources
gate_interpreter 125 _resolve_a3_refs
t2_gate_patch 2937 noop_write
t2_prekb_patch 74 _require_before
t2_speak 63 prohibits_target
t2_prekb_patch 266 _argprod_hits
t2_gate_patch 8156 apply_gate_regen
t2_prekb_patch 291 _notice_done
t2_gate_patch 684 _policy_facts
t2_transcribe 84 mismatches
t2_transcribe 139 unknown_ids
t2_signature 36 signature_violation
t2_prekb_patch 155 _effective_fams
t2_gate_patch 13923 _proc_first_deny
t2_handoff_ground 85 check
t2_precedence 72 declarations
t2_ledger 638 ineligible_text
t2_gate_patch 940 _prop_is_string
t2_dominance 77 _exempt
t2_gate_patch 6154 _any_effective_write
t2_gate_patch 7459 _rebuild_gate_state
t2_gate_patch 6145 _is_effective_write
t2_ledger 110 specs_for
t2_authority 69 _declares_any
t2_dominance 116 requirement_text
t2_handoff_ground 129 deny
t2_phase 34 _auth_gates
t2_search 43 _rows
t2_search 61 _ontology
t2_speak 40 _procs"""
want = [tuple(l.split()) for l in PURE.strip().splitlines()]

trees = {}
def tree(m):
    if m not in trees: trees[m] = ast.parse(open(m + '.py', encoding='utf-8').read())
    return trees[m]

def docstring_nodes(fn):
    """Expr-only string statements anywhere in the fn (docstrings) -> exclude per synthesis K1."""
    ds = set()
    for n in ast.walk(fn):
        body = getattr(n, 'body', None)
        for blk in ('body', 'orelse', 'finalbody'):
            b = getattr(n, blk, None)
            if isinstance(b, list):
                for st in b:
                    if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                        ds.add(id(st.value))
    return ds

def hits(fn, direct_only=False):
    """V_strict literals in the function body. direct_only -> skip nested defs (own-lines semantics)."""
    ds = docstring_nodes(fn)
    kids = {id(c) for c in ast.iter_child_nodes(fn) if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))}
    out = []
    def go(n, inside_kid):
        for c in ast.iter_child_nodes(n):
            ik = inside_kid or (id(c) in kids)
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and id(c) not in ds:
                if not (direct_only and ik):
                    v = c.value
                    if v in VS: out.append((c.lineno, v))
                    else:
                        for tok in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', v):
                            if tok in VS: out.append((c.lineno, tok + '  (in-string)')); break
            go(c, ik)
    go(fn, False)
    return out

print('\n=== K1 on the 46 "pure contract predicates" (the claimed 1,237-line invariant core) ===')
bad = 0
for m, ln, name in want:
    ln = int(ln)
    fn = None
    for n in ast.walk(tree(m)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name and n.lineno == ln:
            fn = n; break
    if fn is None: print('  MISS', m, ln, name); continue
    h = hits(fn, direct_only=True)
    if h:
        bad += 1
        seen = collections.OrderedDict()
        for l, v in h: seen.setdefault(v, l)
        print('  HIT %-18s:%-6s %-26s  %d sites  %s'
              % (m, ln, name, len(h), '; '.join('%s@%d' % (v, l) for v, l in list(seen.items())[:6])))
print('  --> %d / %d core functions carry a V_strict literal in their OWN lines' % (bad, len(want)))

print('\n=== K1 inside unified() (parked as K0 "no decomposition unit") ===')
gp = tree('t2_gate_patch')
for n in ast.walk(gp):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'unified' and n.lineno == 8685:
        h_all = hits(n, direct_only=False)
        h_own = hits(n, direct_only=True)
        c = collections.Counter(v.split('  ')[0] for _, v in h_own)
        print('  span L%d-%d  V_strict literal sites: own=%d  incl.nested=%d' % (n.lineno, n.end_lineno, len(h_own), len(h_all)))
        print('  distinct domain tokens = %d' % len(c))
        for v, k in c.most_common(30): print('     %-42s %d' % (v, k))
