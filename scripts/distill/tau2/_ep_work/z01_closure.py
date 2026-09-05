# -*- coding: utf-8 -*-
"""Z01 — live-closure soundness probe.
Q: the synthesis calls 48 modules "live" via *static* import closure from t2_run_gated.
   But an import inside `if flag:` / `try:` / a function body is NOT live unless the flag is set.
   Measure: unconditional (module-top-level, no enclosing If/Try/FunctionDef) closure vs full closure.
"""
import ast, os, json, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def imports(p):
    """-> list of (module, guarded:bool, lineno, guard_kind)"""
    try:
        src = open(p, encoding='utf-8').read()
        t = ast.parse(src)
    except Exception:
        return []
    out = []
    def walk(node, guards):
        for ch in ast.iter_child_nodes(node):
            g = list(guards)
            if isinstance(ch, ast.If):        g.append('If')
            elif isinstance(ch, ast.Try):     g.append('Try')
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)): g.append('def:'+ch.name)
            elif isinstance(ch, ast.ClassDef): g.append('class')
            elif isinstance(ch, (ast.For, ast.While)): g.append('Loop')
            if isinstance(ch, ast.Import):
                for a in ch.names: out.append((a.name.split('.')[0], bool(g), ch.lineno, '>'.join(g)))
            elif isinstance(ch, ast.ImportFrom):
                if ch.module: out.append((ch.module.split('.')[0], bool(g), ch.lineno, '>'.join(g)))
            walk(ch, g)
    walk(t, [])
    return out

def closure(root, only_unguarded):
    seen, stack, edges = set(), [root], {}
    while stack:
        m = stack.pop()
        if m in seen: continue
        p = os.path.join(D, m + '.py')
        if not os.path.exists(p): continue
        seen.add(m)
        for x, guarded, ln, gk in imports(p):
            if only_unguarded and guarded: continue
            if os.path.exists(os.path.join(D, x + '.py')):
                edges.setdefault(x, []).append((m, ln, gk if guarded else 'TOP'))
                if x not in seen: stack.append(x)
    return seen, edges

full, e_full = closure('t2_run_gated', False)
uncond, e_unc = closure('t2_run_gated', True)

def lines(m):
    p = os.path.join(D, m + '.py')
    return sum(1 for _ in open(p, encoding='utf-8'))

print("FULL closure   N=%d lines=%d" % (len(full), sum(lines(m) for m in full)))
print("UNCOND closure N=%d lines=%d" % (len(uncond), sum(lines(m) for m in uncond)))
only_guarded = sorted(full - uncond)
print("\n--- reachable ONLY through a guarded import (%d mods, %d lines) ---" % (
    len(only_guarded), sum(lines(m) for m in only_guarded)))
for m in only_guarded:
    print("  %-28s %6d lines   via %s" % (m, lines(m), e_full.get(m, [])[:3]))
json.dump({"full": sorted(full), "uncond": sorted(uncond),
           "only_guarded": only_guarded,
           "edges_full": {k: v for k, v in e_full.items()}},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_z01.json'), 'w'),
          ensure_ascii=False, indent=1)
