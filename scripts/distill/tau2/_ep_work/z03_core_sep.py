# -*- coding: utf-8 -*-
"""Z03 — are the 46 'pure contract predicates' actually separable top-level functions?
   A def nested inside another def is a CLOSURE over the parent's locals -> cannot be lifted
   into a standalone 2,590-line core without a rewrite."""
import ast, os, re, json, collections
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)

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
    if m not in trees:
        trees[m] = ast.parse(open(m + '.py', encoding='utf-8').read())
    return trees[m]

def parents(m):
    t = tree(m); par = {}
    for n in ast.walk(t):
        for c in ast.iter_child_nodes(n):
            par[c] = n
    return par

nested, toplevel = [], []
for m, ln, name in want:
    ln = int(ln); t = tree(m); par = parents(m)
    node = None
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name and n.lineno == ln:
            node = n; break
    if node is None:
        print("MISS", m, ln, name); continue
    chain = []
    p = par.get(node)
    while p is not None:
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chain.append(p.name)
        elif isinstance(p, ast.ClassDef):
            chain.append('class:' + p.name)
        p = par.get(p)
    if chain:
        # free variables: names loaded in body that are not params/locals/globals-of-module
        args = {a.arg for a in node.args.args + node.args.kwonlyargs}
        if node.args.vararg: args.add(node.args.vararg.arg)
        if node.args.kwarg: args.add(node.args.kwarg.arg)
        assigned = set()
        for n2 in ast.walk(node):
            if isinstance(n2, ast.Name) and isinstance(n2.ctx, ast.Store): assigned.add(n2.id)
            if isinstance(n2, ast.arg): assigned.add(n2.arg)
        loaded = {n2.id for n2 in ast.walk(node) if isinstance(n2, ast.Name) and isinstance(n2.ctx, ast.Load)}
        modlevel = set()
        for n2 in ast.iter_child_nodes(t):
            if isinstance(n2, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): modlevel.add(n2.name)
            if isinstance(n2, ast.Assign):
                for tg in n2.targets:
                    for nn in ast.walk(tg):
                        if isinstance(nn, ast.Name): modlevel.add(nn.id)
            if isinstance(n2, (ast.Import, ast.ImportFrom)):
                for al in n2.names: modlevel.add((al.asname or al.name).split('.')[0])
        import builtins
        free = sorted(loaded - args - assigned - modlevel - set(dir(builtins)))
        nested.append((m, ln, name, '>'.join(reversed(chain)), free))
    else:
        toplevel.append((m, ln, name))

print("PURE-46 separability")
print("  module-level (liftable)      : %d" % len(toplevel))
print("  NESTED inside another def    : %d" % len(nested))
for m, ln, name, chain, free in nested:
    print("    %s:%d %s  inside=%s  free_vars=%d %s" % (m, ln, name, chain, len(free), free[:10]))
