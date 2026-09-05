# -*- coding: utf-8 -*-
"""Z02 — is unified() really K0 (no decomposition unit)?"""
import ast, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(D, 't2_gate_patch.py'), encoding='utf-8').read()
t = ast.parse(src)
tot = src.count('\n') + 1
print("t2_gate_patch.py total lines(count nl+1)=%d  wc-l=%d" % (tot, src.count('\n')))
for n in ast.walk(t):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ('unified', 'apply_unified_regen', 'apply_provenance_regen'):
        span = n.end_lineno - n.lineno + 1
        inner = [c for c in ast.walk(n) if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c is not n]
        own = span - sum((c.end_lineno - c.lineno + 1) for c in inner if all(
            not (o.lineno <= c.lineno and c.end_lineno <= o.end_lineno and o is not c) for o in inner))
        print("\n== %s  L%d-%d  span=%d  nested_defs=%d" % (n.name, n.lineno, n.end_lineno, span, len(inner)))
        inner.sort(key=lambda c: c.lineno)
        for c in inner:
            print("    %-34s L%-6d-%-6d %5d" % (c.name, c.lineno, c.end_lineno, c.end_lineno - c.lineno + 1))
