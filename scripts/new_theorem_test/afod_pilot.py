#!/usr/bin/env python3
"""F8-AFOD pilot — multi-view clustering + NMI orthogonality on τ²-bench tools.

Objective (internal; do NOT cite AFOD/OISA in paper):
  Test whether a multi-perspective ontology discovery (clustering from
  different information sources) produces GENUINELY orthogonal facets on
  τ²-telecom, or whether the candidate facets collapse to redundant views.
  If NMI > 0.5 across most pairs, this EXPLAINS Phase C's "real ≈ permuted"
  result — the facets were never independent, so permuting didn't change
  anything.

Four independent clusterings:
  A. function_action (verb from method name)    — linguistic/behavioral
  B. task co-occurrence community               — usage-level
  C. parameter signature Jaccard                — structural/interface
  D. tool_category (@is_tool READ/WRITE/...)    — semantic-role

Per patent §5.2.4 thresholds:
  NMI < 0.3     → orthogonal (independent facets)
  NMI 0.3-0.5   → soft-orthogonal (same facet family, but salvageable)
  NMI > 0.5     → redundant (merge candidate)

Outputs:
  reports/new_theorem_test/phase_f8_afod/afod_clusterings_telecom.json
  reports/new_theorem_test/phase_f8_afod/nmi_matrix.json
  reports/new_theorem_test/phase_f8_afod/afod_summary.md
"""
from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
TAU2_ROOT = REPO / "external" / "tau2-bench"
DOMAIN = "telecom"
TOOLS_PY = TAU2_ROOT / "src" / "tau2" / "domains" / DOMAIN / "tools.py"
USER_TOOLS_PY = TAU2_ROOT / "src" / "tau2" / "domains" / DOMAIN / "user_tools.py"
TASKS_JSON = TAU2_ROOT / "data" / "tau2" / "domains" / DOMAIN / "tasks.json"
OUT_DIR = REPO / "reports" / "new_theorem_test" / "phase_f8_afod"

ACTION_VERBS = [
    "search", "find", "list", "get", "book", "make", "create",
    "cancel", "exchange", "return", "update", "modify", "set",
    "send", "issue", "calculate", "compute", "transfer",
    "suspend", "resume", "enable", "disable", "refuel", "reseat",
    "reboot", "reset", "toggle", "grant", "disconnect",
    "remove", "add", "check", "verify",
]


# ---------------------------------------------------------------------
# Tool inventory (agent + user tools combined — same pool that appears in
# evaluation_criteria actions)
# ---------------------------------------------------------------------

def parse_tools(path: Path) -> Dict[str, dict]:
    """AST-parse @is_tool methods. Return {name: {args, return_type, tool_type, docstring}}."""
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text())
    out: Dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        tool_type = None
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                fname = dec.func.attr if isinstance(dec.func, ast.Attribute) else getattr(dec.func, "id", None)
                if fname == "is_tool" and dec.args:
                    arg0 = dec.args[0]
                    if isinstance(arg0, ast.Attribute):
                        tool_type = arg0.attr
        if tool_type is None:
            continue
        args = [a.arg for a in node.args.args if a.arg != "self"]
        out[node.name] = {
            "args": args,
            "return_type": _ann(node.returns) if node.returns else "Any",
            "tool_type": tool_type,
            "docstring": (ast.get_docstring(node) or "")[:300],
        }
    return out


def _ann(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ann(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_ann(node.value)}[{_ann(node.slice)}]"
    if isinstance(node, ast.Tuple):
        return ", ".join(_ann(e) for e in node.elts)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return "?"


def collect_tool_inventory() -> Dict[str, dict]:
    """Merge agent + user tools + tools referenced in tasks but missing from code."""
    inv = {}
    inv.update(parse_tools(TOOLS_PY))
    inv.update(parse_tools(USER_TOOLS_PY))
    # Cover any tool referenced in tasks.json that's missing from AST
    tasks = json.load(open(TASKS_JSON))
    referenced = set()
    for t in tasks:
        for a in t.get("evaluation_criteria", {}).get("actions", []):
            referenced.add(a.get("name", ""))
    missing = referenced - set(inv.keys())
    for m in missing:
        if m:
            inv[m] = {"args": [], "return_type": "Any", "tool_type": "UNKNOWN", "docstring": ""}
    return inv


# ---------------------------------------------------------------------
# Clustering A: function_action (verb)
# ---------------------------------------------------------------------

def _extract_verb(name: str) -> str:
    for v in ACTION_VERBS:
        if name.startswith(v + "_") or name == v:
            return v
    return name.split("_", 1)[0]


def clustering_A(inventory: Dict[str, dict]) -> Dict[str, str]:
    return {name: _extract_verb(name) for name in inventory}


# ---------------------------------------------------------------------
# Clustering B: task co-occurrence community (undirected graph, CC)
# ---------------------------------------------------------------------

def clustering_B(inventory: Dict[str, dict]) -> Dict[str, str]:
    tasks = json.load(open(TASKS_JSON))
    tool_names = list(inventory.keys())
    idx = {n: i for i, n in enumerate(tool_names)}
    n = len(tool_names)
    cooc = [[0] * n for _ in range(n)]
    freq = [0] * n
    for t in tasks:
        acts = [a.get("name", "") for a in t.get("evaluation_criteria", {}).get("actions", [])]
        uniq = sorted(set(a for a in acts if a in idx))
        for a in uniq:
            freq[idx[a]] += 1
        for a, b in combinations(uniq, 2):
            i, j = idx[a], idx[b]
            cooc[i][j] += 1
            cooc[j][i] += 1

    # PMI-thresholded edges → connected components
    # edge if cooc[i][j] > max(0.1 * min(freq[i], freq[j]), 2)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if freq[i] == 0 or freq[j] == 0:
                continue
            threshold = max(0.10 * min(freq[i], freq[j]), 2)
            if cooc[i][j] >= threshold:
                union(i, j)
    return {tool_names[i]: f"community_{find(i)}" for i in range(n)}


# ---------------------------------------------------------------------
# Clustering C: parameter signature Jaccard
# ---------------------------------------------------------------------

def clustering_C(inventory: Dict[str, dict]) -> Dict[str, str]:
    tool_names = list(inventory.keys())
    n = len(tool_names)
    sets = [set(inventory[t]["args"]) for t in tool_names]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if not sets[i] and not sets[j]:
                # both empty: only merge if both return same type bucket — handled by D; skip here
                continue
            u = sets[i] | sets[j]
            if not u:
                continue
            jacc = len(sets[i] & sets[j]) / len(u)
            if jacc >= 0.5:
                union(i, j)
    return {tool_names[i]: f"paramgrp_{find(i)}" for i in range(n)}


# ---------------------------------------------------------------------
# Clustering D: tool_category (decorator-based semantic role)
# ---------------------------------------------------------------------

def clustering_D(inventory: Dict[str, dict]) -> Dict[str, str]:
    return {name: f"cat_{info['tool_type']}" for name, info in inventory.items()}


# ---------------------------------------------------------------------
# NMI computation
# ---------------------------------------------------------------------

def nmi(labels_x: List[str], labels_y: List[str]) -> float:
    n = len(labels_x)
    if n == 0 or n != len(labels_y):
        return 0.0
    cx = Counter(labels_x)
    cy = Counter(labels_y)
    cxy = Counter(zip(labels_x, labels_y))

    def H(c: Counter) -> float:
        total = sum(c.values())
        h = 0.0
        for v in c.values():
            if v == 0:
                continue
            p = v / total
            h -= p * math.log(p)
        return h

    h_x = H(cx)
    h_y = H(cy)
    if h_x == 0 or h_y == 0:
        return 0.0  # no information in either side — treat as orthogonal by convention
    mi = 0.0
    for (a, b), v in cxy.items():
        if v == 0:
            continue
        p_ab = v / n
        p_a = cx[a] / n
        p_b = cy[b] / n
        mi += p_ab * math.log(p_ab / (p_a * p_b))
    return mi / math.sqrt(h_x * h_y)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = collect_tool_inventory()
    tool_names = sorted(inventory.keys())
    print(f"[inventory] {len(tool_names)} unique tools (agent + user + tasks-referenced)")

    labels = {
        "A_function_action": clustering_A(inventory),
        "B_task_cooccurrence": clustering_B(inventory),
        "C_param_jaccard": clustering_C(inventory),
        "D_tool_category": clustering_D(inventory),
    }

    # Distribution per clustering
    print("\n[cluster distributions]")
    cluster_stats = {}
    for cname, lbl in labels.items():
        dist = Counter(lbl.values())
        cluster_stats[cname] = {
            "n_clusters": len(dist),
            "distribution": dict(dist),
            "entropy_nats": sum(-(v / len(tool_names)) * math.log(v / len(tool_names))
                                for v in dist.values() if v > 0),
        }
        print(f"  {cname}: {len(dist)} clusters, H={cluster_stats[cname]['entropy_nats']:.3f} nats")
        for c, v in dist.most_common():
            members = [t for t in tool_names if lbl[t] == c]
            print(f"    {c:25s} n={v:3d}  members: {members[:4]}{'...' if v > 4 else ''}")

    # NMI matrix
    keys = list(labels.keys())
    nmi_mat: Dict[str, Dict[str, float]] = {k: {} for k in keys}
    print("\n[NMI matrix]")
    header = " " * 22 + "  ".join(f"{k[:15]:>15s}" for k in keys)
    print(header)
    for i, ki in enumerate(keys):
        row = [f"{ki[:20]:>20s}"]
        for j, kj in enumerate(keys):
            if i == j:
                row.append(f"{'—':>15s}")
                nmi_mat[ki][kj] = 1.0
                continue
            lx = [labels[ki][t] for t in tool_names]
            ly = [labels[kj][t] for t in tool_names]
            v = nmi(lx, ly)
            nmi_mat[ki][kj] = v
            row.append(f"{v:15.3f}")
        print("  ".join(row))

    # Threshold interpretation
    print("\n[interpretation per patent §5.2.4 thresholds]")
    verdicts = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            v = nmi_mat[ki][kj]
            if v < 0.3:
                verdict = "ORTHOGONAL ✓"
            elif v < 0.5:
                verdict = "soft-orthogonal (L_facet_orth)"
            else:
                verdict = "REDUNDANT ✗ (merge)"
            verdicts.append((ki, kj, v, verdict))
            print(f"  {ki} × {kj}:  NMI={v:.3f}  → {verdict}")

    # Save
    (OUT_DIR / "afod_clusterings_telecom.json").write_text(json.dumps({
        "domain": DOMAIN,
        "n_tools": len(tool_names),
        "tool_names": tool_names,
        "labels": {k: v for k, v in labels.items()},
        "cluster_stats": cluster_stats,
    }, indent=2, ensure_ascii=False))
    (OUT_DIR / "nmi_matrix.json").write_text(json.dumps({
        "domain": DOMAIN,
        "keys": keys,
        "nmi": nmi_mat,
        "verdicts": [{"a": a, "b": b, "nmi": v, "verdict": vd} for a, b, v, vd in verdicts],
        "thresholds": {"orthogonal": 0.3, "soft_orth": 0.5},
    }, indent=2))
    print(f"\n[saved] {OUT_DIR}")


if __name__ == "__main__":
    main()
