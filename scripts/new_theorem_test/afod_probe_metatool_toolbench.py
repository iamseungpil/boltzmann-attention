#!/usr/bin/env python3
"""F8b/c — Run multi-view NMI probe on MetaTool + StableToolBench.

Compares to F8-AFOD τ²-telecom result (all 6 pairs NMI ≥ 0.38, 5/6 redundant).

Datasets:
  MetaTool: 390 plugins, multi-domain (travel, food, finance, entertainment,
            science, coding, ...).  Co-occurrence from Task2-Subtask4 (497
            two-tool queries).  Description-based verb + domain clustering.

  StableToolBench G1_tool: 158 queries over ~200+ APIs with explicit
            category_name, parameter types, method, template_response.
            Co-occurrence from 'relevant APIs' GT lists.

4-view clustering on each:
  A. function_action (verb from description or api_name)
  B. task co-occurrence (GT co-invocation community, CC)
  C. parameter signature Jaccard (StableToolBench only — MetaTool
     has no formal params; use name-token overlap instead)
  D. domain / category_name

Outputs:
  reports/new_theorem_test/phase_f8_afod/nmi_metatool.json
  reports/new_theorem_test/phase_f8_afod/nmi_stabletoolbench.json
  reports/new_theorem_test/phase_f8_afod/nmi_comparison.md
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
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
    "travel":         [r"hotel", r"flight", r"trip", r"travel", r"airbnb",
                       r"tourism", r"destination", r"vacation"],
    "food":           [r"recipe", r"food", r"restaurant", r"menu",
                       r"cook", r"meal", r"cuisine", r"nutrition"],
    "finance":        [r"stock", r"crypto", r"finance", r"trading",
                       r"invest", r"bank", r"tax", r"currency", r"payment"],
    "shopping":       [r"shop", r"product", r"purchase", r"retail",
                       r"ecommerce", r"deal", r"coupon"],
    "entertainment":  [r"game", r"play", r"movie", r"music", r"song",
                       r"video", r"tv show", r"entertain", r"trivia"],
    "education":      [r"learn", r"study", r"tutorial", r"teach",
                       r"course", r"homework", r"education", r"quiz"],
    "news":           [r"news", r"article", r"report", r"headline",
                       r"journalism", r"press"],
    "coding":         [r"code", r"program", r"develop", r"git", r"api",
                       r"function", r"debug", r"script", r"algorithm"],
    "writing":        [r"write", r"essay", r"text", r"content", r"blog",
                       r"polish", r"rewrit", r"paraphrase", r"grammar"],
    "data":           [r"data", r"database", r"analy", r"chart", r"graph",
                       r"plot", r"statistic", r"metric", r"dashboard"],
    "image":          [r"image", r"photo", r"picture", r"visual", r"logo",
                       r"icon", r"avatar", r"art"],
    "search_web":     [r"web search", r"google", r"search engine",
                       r"browse", r"internet"],
    "communication":  [r"email", r"message", r"chat", r"sms", r"call",
                       r"notification", r"meeting"],
    "calendar":       [r"calendar", r"schedul", r"appointment", r"reminder",
                       r"event", r"agenda"],
    "weather":        [r"weather", r"forecast", r"climate", r"temperature"],
    "health":         [r"health", r"medic", r"fitness", r"diet", r"exercise",
                       r"wellness"],
    "social":         [r"social media", r"tweet", r"twitter", r"facebook",
                       r"instagram", r"linkedin"],
    "utility":        [r"convert", r"translate", r"summariz", r"password",
                       r"timer", r"qr code"],
    "maps":           [r"map", r"location", r"direction", r"navigate",
                       r"geograph", r"address"],
    "science":        [r"science", r"research", r"paper", r"scientific",
                       r"physics", r"chemistry", r"biology", r"math"],
}


def extract_verb(text: str) -> str:
    text_low = text.lower()
    # Try to find first action verb
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


# ---------------------------------------------------------------
# NMI helper (same as afod_pilot.py)
# ---------------------------------------------------------------

def nmi(lx: List[str], ly: List[str]) -> float:
    n = len(lx)
    if n == 0 or n != len(ly):
        return 0.0
    cx, cy, cxy = Counter(lx), Counter(ly), Counter(zip(lx, ly))

    def H(c):
        total = sum(c.values())
        return -sum((v / total) * math.log(v / total) for v in c.values() if v > 0)

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


def connected_components(nodes: List[str], pairs: List[tuple]) -> Dict[str, str]:
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


# ---------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------

def metatool_4views() -> dict:
    info = json.load(open("/tmp/MetaTool/dataset/plugin_info.json"))
    st4 = json.load(open("/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json"))

    # Inventory by name_for_model
    tools: Dict[str, dict] = {}
    for p in info:
        n = p.get("name_for_model") or p.get("name_for_human")
        if not n:
            continue
        tools[n] = {
            "name_human": p.get("name_for_human", n),
            "desc_model": p.get("description_for_model", "") or "",
        }
    tool_names = sorted(tools)
    print(f"[MetaTool] {len(tool_names)} plugins")

    # A. verb from description_for_model
    A = {n: extract_verb(tools[n]["desc_model"]) for n in tool_names}

    # D. domain from description_for_model
    D = {n: extract_domain(tools[n]["desc_model"]) for n in tool_names}

    # B. co-occurrence from ST4 (2-tool queries)
    pairs = []
    for entry in st4:
        gt = entry.get("tool", [])
        if isinstance(gt, list) and len(gt) >= 2:
            for a, b in combinations(sorted(set(gt)), 2):
                pairs.append((a, b))
    # Map GT tool names (name_for_model) to our inventory
    B = connected_components(tool_names, pairs)

    # C. name-token overlap (MetaTool lacks formal params)
    def tokens(n: str) -> set:
        return set(re.split(r"[_\W]+", n.lower())) - {""}
    sets = {n: tokens(n) for n in tool_names}
    name_pairs = []
    for i, a in enumerate(tool_names):
        for b in tool_names[i+1:]:
            u = sets[a] | sets[b]
            if not u:
                continue
            jacc = len(sets[a] & sets[b]) / len(u)
            if jacc >= 0.5:
                name_pairs.append((a, b))
    C = connected_components(tool_names, name_pairs)

    return {
        "domain": "metatool",
        "n_tools": len(tool_names),
        "tool_names": tool_names,
        "A_verb": A, "B_cooccurrence": B, "C_name_jaccard": C, "D_domain": D,
    }


def stabletoolbench_4views() -> dict:
    path = REPO / "external" / "StableToolBench" / "solvable_queries" / "test_instruction" / "G1_tool.json"
    data = json.load(open(path))
    print(f"[StableToolBench G1_tool] {len(data)} queries")

    # Build API inventory
    apis: Dict[str, dict] = {}

    def api_id(tool_name: str, api_name: str) -> str:
        return f"{tool_name}::{api_name}"

    for entry in data:
        for api in entry.get("api_list", []):
            aid = api_id(api["tool_name"], api["api_name"])
            if aid in apis:
                continue
            apis[aid] = {
                "category": api.get("category_name", "Misc"),
                "description": api.get("api_description", "") or "",
                "method": api.get("method", "GET"),
                "required_params": [p["name"] for p in api.get("required_parameters", [])],
                "optional_params": [p["name"] for p in api.get("optional_parameters", [])],
                "template_keys": list(api.get("template_response", {}).keys())
                                 if isinstance(api.get("template_response"), dict) else [],
            }
    api_names = sorted(apis)
    print(f"[StableToolBench] {len(api_names)} unique APIs")

    # A. verb from api_description
    A = {n: extract_verb(apis[n]["description"] + " " + n.split("::")[-1]) for n in api_names}

    # D. category (explicit category_name)
    D = {n: f"cat_{apis[n]['category']}" for n in api_names}

    # B. co-occurrence from "relevant APIs" (≥2 APIs in same GT)
    pairs = []
    for entry in data:
        gt = entry.get("relevant APIs", [])
        ids = [api_id(x[0], x[1]) for x in gt if len(x) == 2]
        for a, b in combinations(sorted(set(ids)), 2):
            pairs.append((a, b))
    B = connected_components(api_names, pairs)

    # C. parameter signature Jaccard (required + optional)
    def param_set(n):
        return set(apis[n]["required_params"]) | set(apis[n]["optional_params"])
    sets = {n: param_set(n) for n in api_names}
    name_pairs = []
    for i, a in enumerate(api_names):
        for b in api_names[i+1:]:
            u = sets[a] | sets[b]
            if not u:
                continue
            jacc = len(sets[a] & sets[b]) / len(u)
            if jacc >= 0.5:
                name_pairs.append((a, b))
    C = connected_components(api_names, name_pairs)

    return {
        "domain": "stabletoolbench_g1",
        "n_tools": len(api_names),
        "tool_names": api_names,
        "A_verb": A, "B_cooccurrence": B, "C_param_jaccard": C, "D_category": D,
    }


# ---------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------

def probe(dataset_name: str, views: dict) -> dict:
    tool_names = views["tool_names"]
    view_keys = [k for k in views if k.startswith(("A_", "B_", "C_", "D_"))]

    stats = {}
    print(f"\n== {dataset_name} ({views['n_tools']} tools) ==")
    for k in view_keys:
        lbls = views[k]
        dist = Counter(lbls.values())
        h = sum(-(v / len(tool_names)) * math.log(v / len(tool_names))
                for v in dist.values() if v > 0)
        stats[k] = {"n_clusters": len(dist), "entropy": h,
                    "top3": dist.most_common(3)}
        print(f"  {k:20s}  {len(dist):3d} clusters  H={h:.3f} nats  "
              f"top3={[f'{name}({cnt})' for name, cnt in dist.most_common(3)]}")

    nmi_mat = {k: {} for k in view_keys}
    verdicts = []
    print(f"\n  NMI matrix:")
    header = " " * 22 + " ".join(f"{k[:12]:>14s}" for k in view_keys)
    print("   " + header)
    for i, ki in enumerate(view_keys):
        row = [f"{ki[:20]:>20s}"]
        for j, kj in enumerate(view_keys):
            if i == j:
                row.append(f"{'—':>14s}")
                nmi_mat[ki][kj] = 1.0
                continue
            lx = [views[ki][t] for t in tool_names]
            ly = [views[kj][t] for t in tool_names]
            v = nmi(lx, ly)
            nmi_mat[ki][kj] = v
            row.append(f"{v:14.3f}")
        print("   " + "  ".join(row))

    for i in range(len(view_keys)):
        for j in range(i + 1, len(view_keys)):
            ki, kj = view_keys[i], view_keys[j]
            v = nmi_mat[ki][kj]
            if v < 0.3:
                verdict = "ORTHOGONAL ✓"
            elif v < 0.5:
                verdict = "soft-orth"
            else:
                verdict = "REDUNDANT ✗"
            verdicts.append({"a": ki, "b": kj, "nmi": v, "verdict": verdict})

    n_orth = sum(1 for v in verdicts if v["verdict"].startswith("ORTHOGONAL"))
    n_soft = sum(1 for v in verdicts if v["verdict"].startswith("soft"))
    n_red  = sum(1 for v in verdicts if v["verdict"].startswith("REDUNDANT"))
    total = len(verdicts)
    print(f"\n  Verdicts: ORTH={n_orth}/{total}, soft={n_soft}/{total}, REDUNDANT={n_red}/{total}")

    return {
        "dataset": dataset_name,
        "n_tools": views["n_tools"],
        "cluster_stats": stats,
        "nmi": nmi_mat,
        "verdicts": verdicts,
        "summary": {"orth": n_orth, "soft": n_soft, "redundant": n_red, "total": total},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mt_views = metatool_4views()
    mt_result = probe("metatool", mt_views)
    (OUT_DIR / "nmi_metatool.json").write_text(json.dumps(mt_result, indent=2))
    (OUT_DIR / "afod_clusterings_metatool.json").write_text(json.dumps({
        "domain": "metatool",
        "n_tools": mt_views["n_tools"],
        "tool_names": mt_views["tool_names"][:50],  # truncate
        "labels_sample": {k: dict(list(mt_views[k].items())[:10])
                          for k in mt_views if k.startswith(("A_", "B_", "C_", "D_"))},
    }, indent=2))

    stb_views = stabletoolbench_4views()
    stb_result = probe("stabletoolbench_g1", stb_views)
    (OUT_DIR / "nmi_stabletoolbench.json").write_text(json.dumps(stb_result, indent=2))

    # Comparison table
    print("\n\n========== CROSS-DATASET COMPARISON ==========")
    print(f"τ²-telecom          (from afod_pilot.py): 0 ORTH / 1 soft / 5 REDUNDANT (of 6)")
    print(f"MetaTool            N={mt_result['n_tools']:>4d}: "
          f"{mt_result['summary']['orth']} ORTH / "
          f"{mt_result['summary']['soft']} soft / "
          f"{mt_result['summary']['redundant']} REDUNDANT (of {mt_result['summary']['total']})")
    print(f"StableToolBench G1  N={stb_result['n_tools']:>4d}: "
          f"{stb_result['summary']['orth']} ORTH / "
          f"{stb_result['summary']['soft']} soft / "
          f"{stb_result['summary']['redundant']} REDUNDANT (of {stb_result['summary']['total']})")


if __name__ == "__main__":
    main()
