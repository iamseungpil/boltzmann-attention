#!/usr/bin/env python3
"""F8d — Extend NMI probe to 4 more benchmarks.

Adds adapters for:
  - TaskBench HF       (23 top-level nodes, 7458 queries)
  - TaskBench Daily    (40 API tools, 4318 queries)
  - AppBench           (800 queries over Schema2 apps; used_app × used_api)
  - C3-Benchmark       (256 tools in OpenAI-function schema, 256 queries)

Uses same 4-view NMI protocol as afod_probe_metatool_toolbench.py:
  A_verb              — action verb from description
  B_cooccurrence      — CC on GT co-invocation in tasks
  C_param_jaccard     — CC on parameter-name Jaccard ≥ 0.5
  D_domain/category   — explicit category or NLP-inferred domain

Output: reports/new_theorem_test/phase_f8_afod/nmi_extended_all.json
        + console comparison vs τ²/MetaTool/StableToolBench
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from afod_probe_metatool_toolbench import (
    extract_verb, extract_domain, nmi, connected_components,
)

OUT_DIR = REPO / "reports" / "new_theorem_test" / "phase_f8_afod"


# ---------------------------------------------------------------
# TaskBench HF (huggingface tasks — 23 top-level ML task categories)
# ---------------------------------------------------------------

def taskbench_hf_4views() -> dict:
    root = REPO / "external" / "JARVIS" / "taskbench" / "data_huggingface"
    nodes = json.load(open(root / "tool_desc.json"))["nodes"]
    with open(root / "data.json") as f:
        queries = [json.loads(l) for l in f]

    tool_ids = [n["id"] for n in nodes]
    tool_desc = {n["id"]: n.get("desc", "") for n in nodes}
    input_type = {n["id"]: (n.get("input-type") or ["text"])[0] for n in nodes}
    output_type = {n["id"]: (n.get("output-type") or ["text"])[0] for n in nodes}
    print(f"[taskbench-hf] {len(tool_ids)} tools, {len(queries)} queries")

    A = {t: extract_verb(tool_desc[t]) for t in tool_ids}
    # Domain = ML-subfield inferred from desc (nlp/vision/audio/generation/...)
    D_KEYS = {
        "text":  ["text", "language", "token", "question", "summar", "translat"],
        "image": ["image", "visual", "segmentation", "picture", "photo"],
        "audio": ["audio", "speech", "voice", "sound"],
        "video": ["video"],
        "multimodal": ["multimodal", "modalit"],
        "tabular": ["tabular", "table"],
    }
    def _dom(desc):
        low = desc.lower()
        votes = {}
        for k, pats in D_KEYS.items():
            c = sum(1 for p in pats if p in low)
            if c:
                votes[k] = c
        return max(votes, key=votes.get) if votes else "other"
    D = {t: _dom(tool_desc[t]) for t in tool_ids}

    # B from sampled_nodes co-occurrence
    pairs = []
    for q in queries:
        nodes_q = q.get("tool_nodes", []) or q.get("sampled_nodes", [])
        ids = sorted(set(n.get("task") if isinstance(n, dict) else n for n in nodes_q))
        ids = [i for i in ids if i in tool_ids]
        for a, b in combinations(ids, 2):
            pairs.append((a, b))
    B = connected_components(tool_ids, pairs)

    # C = Jaccard on input-type + output-type tokens
    sets = {t: {f"in:{input_type[t]}", f"out:{output_type[t]}"} for t in tool_ids}
    cp = []
    for i, a in enumerate(tool_ids):
        for b in tool_ids[i+1:]:
            u = sets[a] | sets[b]
            if u and len(sets[a] & sets[b]) / len(u) >= 0.5:
                cp.append((a, b))
    C = connected_components(tool_ids, cp)

    return {"domain": "taskbench_hf", "n_tools": len(tool_ids),
            "tool_names": tool_ids,
            "A_verb": A, "B_cooccurrence": B, "C_io_jaccard": C, "D_ml_subfield": D}


# ---------------------------------------------------------------
# TaskBench Daily (40 daily-life APIs with parameter-rich signatures)
# ---------------------------------------------------------------

def taskbench_daily_4views() -> dict:
    root = REPO / "external" / "JARVIS" / "taskbench" / "data_dailylifeapis"
    nodes = json.load(open(root / "tool_desc.json"))["nodes"]
    with open(root / "data.json") as f:
        queries = [json.loads(l) for l in f]

    tool_ids = [n["id"] for n in nodes]
    tool_desc = {n["id"]: n.get("desc", "") for n in nodes}
    params = {n["id"]: [p["name"] for p in n.get("parameters", [])] for n in nodes}
    print(f"[taskbench-daily] {len(tool_ids)} tools, {len(queries)} queries")

    A = {t: extract_verb(t.replace("_", " ") + " " + tool_desc[t]) for t in tool_ids}
    D = {t: extract_domain(tool_desc[t] + " " + t.replace("_", " ")) for t in tool_ids}

    pairs = []
    for q in queries:
        nodes_q = q.get("tool_nodes", []) or q.get("sampled_nodes", [])
        ids = sorted(set(n.get("task") if isinstance(n, dict) else n for n in nodes_q))
        ids = [i for i in ids if i in tool_ids]
        for a, b in combinations(ids, 2):
            pairs.append((a, b))
    B = connected_components(tool_ids, pairs)

    sets = {t: set(params[t]) for t in tool_ids}
    cp = []
    for i, a in enumerate(tool_ids):
        for b in tool_ids[i+1:]:
            u = sets[a] | sets[b]
            if u and len(sets[a] & sets[b]) / len(u) >= 0.5:
                cp.append((a, b))
    C = connected_components(tool_ids, cp)

    return {"domain": "taskbench_daily", "n_tools": len(tool_ids),
            "tool_names": tool_ids,
            "A_verb": A, "B_cooccurrence": B, "C_param_jaccard": C, "D_domain": D}


# ---------------------------------------------------------------
# AppBench (multi-app multi-tool)
# ---------------------------------------------------------------

def appbench_4views() -> dict:
    root = REPO / "external" / "AppBench" / "data" / "test"
    files = ["test_ss.json", "test_sm.json", "test_ms.json", "test_mm.json"]
    queries = []
    for fn in files:
        queries.extend(json.load(open(root / fn)))

    # Inventory: API name from any `used_api` dict key
    app_of_api: Dict[str, str] = {}
    api_params: Dict[str, set] = defaultdict(set)
    apis: List[str] = []
    for q in queries:
        out = q.get("output", {})
        apps = out.get("used_app", [])
        apis_list = out.get("used_api", [])
        for app, api_entry in zip(apps, apis_list):
            if not isinstance(api_entry, dict):
                continue
            for api_name, args in api_entry.items():
                if api_name not in app_of_api:
                    apis.append(api_name)
                    app_of_api[api_name] = app
                if isinstance(args, dict):
                    api_params[api_name].update(args.keys())

    apis = sorted(set(apis))
    print(f"[appbench] {len(apis)} unique APIs, {len(queries)} queries")

    A = {n: extract_verb(n.replace("_", " ")) for n in apis}
    D = {n: f"app_{app_of_api.get(n, 'misc')}" for n in apis}

    pairs = []
    for q in queries:
        out = q.get("output", {})
        apis_list = out.get("used_api", [])
        flat = []
        for api_entry in apis_list:
            if isinstance(api_entry, dict):
                flat.extend(api_entry.keys())
        flat = sorted(set(flat) & set(apis))
        for a, b in combinations(flat, 2):
            pairs.append((a, b))
    B = connected_components(apis, pairs)

    sets = {n: set(api_params.get(n, [])) for n in apis}
    cp = []
    for i, a in enumerate(apis):
        for b in apis[i+1:]:
            u = sets[a] | sets[b]
            if u and len(sets[a] & sets[b]) / len(u) >= 0.5:
                cp.append((a, b))
    C = connected_components(apis, cp)

    return {"domain": "appbench", "n_tools": len(apis),
            "tool_names": apis,
            "A_verb": A, "B_cooccurrence": B, "C_param_jaccard": C, "D_app": D}


# ---------------------------------------------------------------
# C3-Benchmark (OpenAI function schema)
# ---------------------------------------------------------------

def c3bench_4views() -> dict:
    root = REPO / "external" / "C3-Benchmark" / "c3_bench"
    with open(root / "multi_agent" / "tools" / "tools_en.jsonl") as f:
        tool_blobs = [json.loads(l) for l in f]
    # Each line is a LIST of tool objects (per-query candidate pool)
    # Flatten to unique tools by function.name
    tool_map: Dict[str, dict] = {}
    for blob in tool_blobs:
        items = blob if isinstance(blob, list) else [blob]
        for item in items:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not name or name in tool_map:
                continue
            tool_map[name] = {
                "desc": fn.get("description", "") or "",
                "params": list(fn.get("parameters", {}).get("properties", {}).keys()),
            }
    names = sorted(tool_map.keys())
    print(f"[c3bench] {len(names)} unique tools")

    # Queries for cooccurrence
    with open(root / "bench_test" / "data" / "C3-Bench.jsonl") as f:
        queries = [json.loads(l) for l in f]
    print(f"[c3bench] {len(queries)} queries")

    A = {n: extract_verb(tool_map[n]["desc"] + " " + n) for n in names}
    D = {n: extract_domain(tool_map[n]["desc"] + " " + n) for n in names}

    pairs = []
    for q in queries:
        task_ids = q.get("task_ids", [])
        # task_ids refer to tool identifiers; many are function names
        ids = [t for t in task_ids if t in tool_map]
        if not ids:
            # Fallback: parse english_answer_list or answer_list for function names
            ans = q.get("english_answer_list", []) or q.get("answer_list", [])
            parsed = []
            for a in ans:
                if isinstance(a, dict):
                    n = a.get("name") or a.get("function", {}).get("name")
                    if n and n in tool_map:
                        parsed.append(n)
                elif isinstance(a, str):
                    m = re.search(r'"name"\s*:\s*"(\w+)"', a)
                    if m and m.group(1) in tool_map:
                        parsed.append(m.group(1))
            ids = parsed
        ids = sorted(set(ids))
        for a, b in combinations(ids, 2):
            pairs.append((a, b))
    B = connected_components(names, pairs)

    sets = {n: set(tool_map[n]["params"]) for n in names}
    cp = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            u = sets[a] | sets[b]
            if u and len(sets[a] & sets[b]) / len(u) >= 0.5:
                cp.append((a, b))
    C = connected_components(names, cp)

    return {"domain": "c3bench", "n_tools": len(names),
            "tool_names": names,
            "A_verb": A, "B_cooccurrence": B, "C_param_jaccard": C, "D_domain": D}


# ---------------------------------------------------------------
# Probe runner (same as in afod_probe_metatool_toolbench.probe)
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
        print(f"  {k:22s}  {len(dist):3d} clusters  H={h:.3f}  "
              f"top3={[f'{name}({cnt})' for name, cnt in dist.most_common(3)]}")

    nmi_mat = {k: {} for k in view_keys}
    verdicts = []
    print(f"\n  NMI matrix:")
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
            verdict = ("ORTH" if v < 0.3 else
                       "soft" if v < 0.5 else "RED")
            verdicts.append({"a": ki, "b": kj, "nmi": v, "verdict": verdict})

    n_orth = sum(1 for v in verdicts if v["verdict"] == "ORTH")
    n_soft = sum(1 for v in verdicts if v["verdict"] == "soft")
    n_red = sum(1 for v in verdicts if v["verdict"] == "RED")
    print(f"\n  ORTH={n_orth} / soft={n_soft} / RED={n_red} (total {len(verdicts)})")

    return {"dataset": dataset_name, "n_tools": views["n_tools"],
            "cluster_stats": stats, "nmi": nmi_mat,
            "verdicts": verdicts,
            "summary": {"orth": n_orth, "soft": n_soft, "redundant": n_red,
                        "total": len(verdicts)}}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, adapter in [
        ("taskbench_hf", taskbench_hf_4views),
        ("taskbench_daily", taskbench_daily_4views),
        ("appbench", appbench_4views),
        ("c3bench", c3bench_4views),
    ]:
        try:
            views = adapter()
            all_results[name] = probe(name, views)
        except Exception as e:
            print(f"[ERR {name}] {type(e).__name__}: {e}")

    (OUT_DIR / "nmi_extended_all.json").write_text(json.dumps(all_results, indent=2))

    # Cross-dataset summary
    print("\n\n========== CROSS-DATASET ORTHOGONALITY SUMMARY ==========")
    # Prior results (from afod_pilot.py / afod_probe_metatool_toolbench.py)
    priors = [
        ("τ²-telecom",       43, 0, 1, 5, "(single-domain)"),
        ("MetaTool",        388, 1, 0, 5, "verb×domain 0.185"),
        ("StableToolBench", 499, 1, 2, 3, "verb×category 0.218"),
    ]
    print(f"{'dataset':<22s} {'N':>5s}  ORTH soft RED  notes")
    for n, N, o, s, r, note in priors:
        print(f"{n:<22s} {N:>5d}  {o:>4d} {s:>4d} {r:>3d}  {note}")
    for name, res in all_results.items():
        s = res["summary"]
        # Find orth pair names
        orth_pairs = [f"{v['a'].split('_',1)[1]}×{v['b'].split('_',1)[1]}={v['nmi']:.3f}"
                      for v in res["verdicts"] if v["verdict"] == "ORTH"]
        note = ", ".join(orth_pairs) if orth_pairs else "(none)"
        print(f"{name:<22s} {res['n_tools']:>5d}  {s['orth']:>4d} {s['soft']:>4d} {s['redundant']:>3d}  {note}")

    print(f"\n[saved] {OUT_DIR/'nmi_extended_all.json'}")


if __name__ == "__main__":
    main()
