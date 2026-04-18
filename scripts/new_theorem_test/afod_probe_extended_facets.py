#!/usr/bin/env python3
"""Extended NMI probe — search for orthogonal facets beyond verb × domain.

Builds 10+ candidate clusterings on MetaTool and StableToolBench, computes
all pairwise NMI, reports:
  - ORTHOGONAL pairs (NMI < 0.3)
  - soft-orthogonal (0.3 ≤ NMI < 0.5)
  - REDUNDANT (NMI ≥ 0.5)

Goal: find additional orthogonal axes beyond A_verb × D_domain (already known
from F8b/c/d).  Any new orthogonal pair is a candidate new facet for
multi-facet direction-specificity analysis.

CPU-only, ~2 min total.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "reports" / "new_theorem_test" / "phase_f8_afod"

ACTION_VERBS = [
    "search", "find", "list", "get", "book", "make", "create",
    "cancel", "exchange", "return", "update", "modify", "set",
    "send", "issue", "calculate", "compute", "transfer",
    "suspend", "resume", "enable", "disable", "refuel", "reseat",
    "reboot", "reset", "toggle", "grant", "disconnect",
    "remove", "add", "check", "verify", "translate", "generate",
    "recommend", "provide", "play", "analyze", "convert", "summarize",
    "solve", "explain", "stream", "download", "upload", "lookup",
    "edit", "render", "browse", "read", "write",
]

DOMAIN_KEYWORDS = {
    "travel":         ["hotel", "flight", "trip", "travel", "airbnb", "tourism", "destination"],
    "food":           ["recipe", "food", "restaurant", "menu", "cook", "meal", "nutrition"],
    "finance":        ["stock", "crypto", "finance", "trading", "invest", "bank", "payment"],
    "shopping":       ["shop", "product", "purchase", "retail", "ecommerce", "deal"],
    "entertainment":  ["game", "movie", "music", "song", "video", "show", "trivia"],
    "education":      ["learn", "tutorial", "teach", "course", "homework", "quiz"],
    "news":           ["news", "article", "report", "headline", "press"],
    "coding":         ["code", "program", "develop", "git", "api", "function", "debug"],
    "writing":        ["essay", "text", "content", "blog", "paraphrase", "grammar"],
    "data":           ["data", "database", "chart", "graph", "plot", "statistic", "metric"],
    "image":          ["image", "photo", "picture", "visual", "logo", "icon"],
    "communication":  ["email", "message", "chat", "sms", "call", "notification"],
    "calendar":       ["calendar", "schedul", "appointment", "reminder", "event"],
    "weather":        ["weather", "forecast", "climate", "temperature"],
    "health":         ["health", "medic", "fitness", "diet", "exercise"],
    "social":         ["social media", "tweet", "twitter", "facebook", "instagram"],
    "utility":        ["translate", "summariz", "password", "timer", "qr"],
    "maps":           ["map", "location", "direction", "navigate", "address"],
    "science":        ["science", "research", "paper", "scientific", "physics", "math"],
}


def extract_verb(text: str) -> str:
    text_low = text.lower()
    for v in ACTION_VERBS:
        if re.search(rf"\b{v}\b", text_low):
            return v
    return "other"


def extract_domain(text: str) -> str:
    text_low = text.lower()
    votes = {}
    for dom, patterns in DOMAIN_KEYWORDS.items():
        hits = sum(1 for p in patterns if re.search(p, text_low))
        if hits:
            votes[dom] = hits
    if not votes:
        return "misc"
    return max(votes.items(), key=lambda kv: kv[1])[0]


# ------ Helper ------

def nmi(lx, ly):
    n = len(lx)
    if n == 0 or n != len(ly):
        return 0.0
    cx, cy, cxy = Counter(lx), Counter(ly), Counter(zip(lx, ly))
    def H(c):
        tot = sum(c.values())
        return -sum((v / tot) * math.log(v / tot) for v in c.values() if v > 0)
    h_x, h_y = H(cx), H(cy)
    if h_x == 0 or h_y == 0:
        return 0.0
    mi = 0.0
    for (a, b), v in cxy.items():
        p_ab = v / n
        p_a = cx[a] / n
        p_b = cy[b] / n
        mi += p_ab * math.log(p_ab / (p_a * p_b))
    return mi / math.sqrt(h_x * h_y)


def connected_components(nodes, pairs):
    idx = {n: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        if a in idx and b in idx:
            ra, rb = find(idx[a]), find(idx[b])
            if ra != rb:
                parent[ra] = rb
    return {n: f"cc_{find(idx[n])}" for n in nodes}


def bucket(x, edges, labels):
    """Bucket x by edges; returns label."""
    for e, lbl in zip(edges, labels):
        if x <= e:
            return lbl
    return labels[-1]


# ------ MetaTool extended facets ------

def metatool_extended():
    info = json.load(open("/tmp/MetaTool/dataset/plugin_info.json"))
    st4 = json.load(open("/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json"))

    tools = {}
    for p in info:
        n = p.get("name_for_model") or p.get("name_for_human")
        if not n:
            continue
        tools[n] = {
            "name_human": p.get("name_for_human", n),
            "desc_model": p.get("description_for_model", "") or "",
            "desc_human": p.get("description_for_human", "") or "",
        }
    tool_names = sorted(tools)
    print(f"[MetaTool] {len(tool_names)} plugins")

    # A. verb
    A = {n: extract_verb(tools[n]["desc_model"]) for n in tool_names}
    # B. cooc
    pairs = []
    for entry in st4:
        gt = entry.get("tool", [])
        if isinstance(gt, list) and len(gt) >= 2:
            for a, b in combinations(sorted(set(gt)), 2):
                pairs.append((a, b))
    B = connected_components(tool_names, pairs)
    # C. name jaccard
    def tokens(n):
        return set(re.split(r"[_\W]+", n.lower())) - {""}
    sets = {n: tokens(n) for n in tool_names}
    name_pairs = []
    for i, a in enumerate(tool_names):
        for b in tool_names[i + 1:]:
            u = sets[a] | sets[b]
            if u and len(sets[a] & sets[b]) / len(u) >= 0.5:
                name_pairs.append((a, b))
    C = connected_components(tool_names, name_pairs)
    # D. domain
    D = {n: extract_domain(tools[n]["desc_model"]) for n in tool_names}

    # NEW facets
    # E. name token count: 1 / 2 / 3+
    E = {n: f"ntok_{min(len(tokens(n)), 3):d}" for n in tool_names}
    # F. desc length bucket (word count)
    def wcount(t): return len(t.split())
    F = {}
    for n in tool_names:
        w = wcount(tools[n]["desc_model"])
        F[n] = bucket(w, [30, 60, 100], ["short", "med", "long", "xlong"])
    # G. has underscore in model name
    G = {n: "underscore" if "_" in n else "solid" for n in tool_names}
    # H. first-noun after verb in description (coarse: take first N+ token after verb)
    H = {}
    for n in tool_names:
        low = tools[n]["desc_model"].lower()
        verb = A[n]
        m = re.search(rf"\b{verb}\b\s+(\w+)", low) if verb != "other" else None
        H[n] = m.group(1) if m else "unknown"
    # I. desc mentions generic term ("api" / "tool" / "service")
    I = {}
    for n in tool_names:
        low = tools[n]["desc_model"].lower()
        if re.search(r"\bapi\b", low):
            I[n] = "api_word"
        elif re.search(r"\btool\b", low):
            I[n] = "tool_word"
        elif re.search(r"\bservice\b", low):
            I[n] = "service_word"
        else:
            I[n] = "none"
    # J. name length (char) bucket
    J = {n: bucket(len(n), [8, 15, 25], ["shortname", "medname", "longname", "xlongname"]) for n in tool_names}

    return {
        "domain": "metatool",
        "n_tools": len(tool_names),
        "tool_names": tool_names,
        "A_verb": A, "B_cooc": B, "C_name_jacc": C, "D_domain": D,
        "E_ntokens": E, "F_desc_len": F, "G_has_underscore": G,
        "H_first_noun": H, "I_desc_generic_term": I, "J_name_len": J,
    }


# ------ StableToolBench extended facets ------

def stb_extended():
    path = REPO / "external/StableToolBench/solvable_queries/test_instruction/G1_tool.json"
    data = json.load(open(path))
    apis = {}
    def aid(t, a): return f"{t}::{a}"
    for entry in data:
        for api in entry.get("api_list", []):
            k = aid(api["tool_name"], api["api_name"])
            if k in apis:
                continue
            apis[k] = {
                "category": api.get("category_name", "Misc"),
                "desc": api.get("api_description", "") or "",
                "method": api.get("method", "GET"),
                "req": [p["name"] for p in api.get("required_parameters", [])],
                "opt": [p["name"] for p in api.get("optional_parameters", [])],
                "tmpl": api.get("template_response", {}),
            }
    names = sorted(apis)
    print(f"[StableToolBench G1] {len(names)} APIs")

    # A. verb
    A = {n: extract_verb(apis[n]["desc"] + " " + n.split("::")[-1]) for n in names}
    # B. cooc
    pairs = []
    for entry in data:
        gt = entry.get("relevant APIs", [])
        ids = [aid(x[0], x[1]) for x in gt if len(x) == 2]
        for a, b in combinations(sorted(set(ids)), 2):
            pairs.append((a, b))
    B = connected_components(names, pairs)
    # C. param jaccard
    def pset(n):
        return set(apis[n]["req"]) | set(apis[n]["opt"])
    ps = {n: pset(n) for n in names}
    name_pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            u = ps[a] | ps[b]
            if u and len(ps[a] & ps[b]) / len(u) >= 0.5:
                name_pairs.append((a, b))
    C = connected_components(names, name_pairs)
    # D. category
    D = {n: f"cat_{apis[n]['category']}" for n in names}

    # NEW facets
    # E. method
    E = {n: f"method_{apis[n]['method'].upper()}" for n in names}
    # F. side effect derived from method
    def side(m):
        m = m.upper()
        if m in ("GET", "HEAD", "OPTIONS"):
            return "read"
        if m in ("POST", "PUT", "PATCH", "DELETE"):
            return "write"
        return "other"
    F = {n: side(apis[n]["method"]) for n in names}
    # G. required-param count bucket
    G = {n: bucket(len(apis[n]["req"]), [0, 1, 3], ["req0", "req1", "req2-3", "req4+"]) for n in names}
    # H. optional-param count bucket
    H = {n: bucket(len(apis[n]["opt"]), [0, 2, 5], ["opt0", "opt1-2", "opt3-5", "opt6+"]) for n in names}
    # I. total param count
    I = {n: bucket(len(apis[n]["req"]) + len(apis[n]["opt"]),
                   [1, 3, 7], ["p0-1", "p2-3", "p4-7", "p8+"]) for n in names}
    # J. template response shape
    def tshape(t):
        if not t:
            return "tmpl_empty"
        if isinstance(t, list):
            return "tmpl_list"
        if isinstance(t, dict):
            return "tmpl_dict"
        return "tmpl_scalar"
    J = {n: tshape(apis[n]["tmpl"]) for n in names}
    # K. desc length (word)
    K = {n: bucket(len(apis[n]["desc"].split()),
                   [3, 8, 20], ["desc_short", "desc_med", "desc_long", "desc_xlong"]) for n in names}
    # L. name-token count (api_name after ::)
    def api_tok(n):
        a = n.split("::")[-1]
        return set(re.split(r"[_\W]+", a.lower())) - {""}
    L = {n: f"ntok_{min(len(api_tok(n)), 3):d}" for n in names}

    return {
        "domain": "stabletoolbench_g1",
        "n_tools": len(names),
        "tool_names": names,
        "A_verb": A, "B_cooc": B, "C_param_jacc": C, "D_category": D,
        "E_method": E, "F_side_effect": F,
        "G_req_count": G, "H_opt_count": H, "I_param_total": I,
        "J_tmpl_shape": J, "K_desc_len": K, "L_api_name_ntok": L,
    }


# ------ Probe ------

def probe(name: str, views: dict):
    tool_names = views["tool_names"]
    view_keys = [k for k in views if len(k) >= 2 and k[1] == "_"]

    print(f"\n{'='*80}")
    print(f"{name}  (N={views['n_tools']}, {len(view_keys)} facets)")
    print(f"{'='*80}")

    for k in view_keys:
        lbls = views[k]
        dist = Counter(lbls.values())
        h = sum(-(v / len(tool_names)) * math.log(v / len(tool_names)) for v in dist.values() if v > 0)
        print(f"  {k:22s}  {len(dist):3d} clusters  H={h:.3f}  top3={[f'{a}({b})' for a,b in dist.most_common(3)]}")

    nmi_mat = {k: {} for k in view_keys}
    all_pairs = []
    for i, ki in enumerate(view_keys):
        for j, kj in enumerate(view_keys):
            if i == j:
                nmi_mat[ki][kj] = 1.0
                continue
            lx = [views[ki][t] for t in tool_names]
            ly = [views[kj][t] for t in tool_names]
            v = nmi(lx, ly)
            nmi_mat[ki][kj] = v
            if j > i:
                all_pairs.append((ki, kj, v))

    all_pairs.sort(key=lambda x: x[2])

    orth = [p for p in all_pairs if p[2] < 0.3]
    soft = [p for p in all_pairs if 0.3 <= p[2] < 0.5]
    red = [p for p in all_pairs if p[2] >= 0.5]

    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"  ORTHOGONAL (NMI < 0.3): {len(orth)}")
    print(f"  soft-orth  (0.3 - 0.5): {len(soft)}")
    print(f"  REDUNDANT  (NMI >= 0.5): {len(red)}")

    if orth:
        print(f"\n  ⭐ ORTHOGONAL pairs:")
        for a, b, v in orth:
            print(f"    {a:>22s}  ×  {b:<22s}  NMI={v:.4f}")
    if soft:
        print(f"\n  soft-orthogonal pairs (top 10 closest to 0.3):")
        for a, b, v in soft[:10]:
            print(f"    {a:>22s}  ×  {b:<22s}  NMI={v:.4f}")

    return {
        "dataset": name, "n_tools": views["n_tools"],
        "view_keys": view_keys, "nmi_matrix": nmi_mat,
        "orthogonal": [{"a": a, "b": b, "nmi": v} for a, b, v in orth],
        "soft_orth": [{"a": a, "b": b, "nmi": v} for a, b, v in soft],
        "redundant_count": len(red),
        "total_pairs": len(all_pairs),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- MetaTool ---
    mt_views = metatool_extended()
    mt_res = probe("MetaTool", mt_views)

    # --- StableToolBench ---
    stb_views = stb_extended()
    stb_res = probe("StableToolBench G1", stb_views)

    # Save combined report
    out = {
        "metatool": mt_res,
        "stabletoolbench_g1": stb_res,
        "summary": {
            "metatool_orth_count": len(mt_res["orthogonal"]),
            "metatool_total_pairs": mt_res["total_pairs"],
            "stb_orth_count": len(stb_res["orthogonal"]),
            "stb_total_pairs": stb_res["total_pairs"],
        },
    }
    path = OUT_DIR / "nmi_extended_facets.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {path}")

    # Cross-dataset orthogonal pair intersection
    mt_orth_pairs = {tuple(sorted([p["a"], p["b"]])) for p in mt_res["orthogonal"]}
    stb_orth_pairs = {tuple(sorted([p["a"], p["b"]])) for p in stb_res["orthogonal"]}
    both = mt_orth_pairs & stb_orth_pairs
    print(f"\n==========================")
    print(f"Cross-dataset analysis:")
    print(f"  MetaTool ORTH count: {len(mt_orth_pairs)}")
    print(f"  StableToolBench ORTH count: {len(stb_orth_pairs)}")
    print(f"  Both datasets show ORTH: {len(both)} pairs")
    if both:
        print(f"\n  Universal orthogonal pairs:")
        for a, b in sorted(both):
            print(f"    {a}  ×  {b}")


if __name__ == "__main__":
    main()
