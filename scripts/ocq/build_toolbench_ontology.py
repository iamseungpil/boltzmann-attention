#!/usr/bin/env python3
"""Build a ToolBench ontology JSON from StableToolBench solvable_queries.

Parses the 6 test instruction files, collects unique (category, tool, api)
triples along with their natural-language descriptions, groups them by
category (47 categories expected), and writes an ontology JSON with the
same schema consumed by build_qwen_metatool_b_ont.py.

Output: reports/axis2_theoretical_verification/toolbench_ontology.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/woori/workspace_common/boltzmann-attention")
QUERIES_DIR = REPO / "external" / "StableToolBench" / "solvable_queries" / "test_instruction"
DEFAULT_OUT = REPO / "reports" / "axis2_theoretical_verification" / "toolbench_ontology.json"


def describe_api(api: dict) -> str:
    cat = api.get("category_name", "")
    tool = api.get("tool_name", "")
    name = api.get("api_name", "")
    desc = (api.get("api_description") or "").strip()
    if not desc:
        return f"The {name} API of the {tool} tool in the {cat} category."
    return f"The {name} API of the {tool} tool ({cat}): {desc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-dir", default=str(QUERIES_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--min-sentences",
        type=int,
        default=3,
        help="Drop categories with fewer unique api descriptions than this.",
    )
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=40,
        help="Truncate per-category sentence list (longer lists bias extraction).",
    )
    args = ap.parse_args()

    qdir = Path(args.queries_dir)
    per_cat: dict[str, dict[tuple, str]] = defaultdict(dict)
    n_files = 0
    n_queries = 0
    for f in sorted(qdir.glob("*.json")):
        data = json.loads(f.read_text())
        n_files += 1
        n_queries += len(data)
        for q in data:
            for api in q.get("api_list", []):
                key = (api.get("tool_name"), api.get("api_name"))
                cat = api.get("category_name")
                if not cat or not key[0] or not key[1]:
                    continue
                if key in per_cat[cat]:
                    continue
                per_cat[cat][key] = describe_api(api)

    print(f"Parsed {n_files} files, {n_queries} queries.")
    print(f"Found {len(per_cat)} categories.")

    # Build ontology
    ontology: dict[str, list[str]] = {}
    dropped = []
    for cat, kv in per_cat.items():
        sents = list(kv.values())
        if len(sents) < args.min_sentences:
            dropped.append((cat, len(sents)))
            continue
        sents = sorted(sents)[: args.max_sentences]
        ontology[cat] = sents

    print(f"Kept {len(ontology)} categories (dropped {len(dropped)} below min).")
    if dropped:
        print(f"  dropped: {dropped[:10]}")

    # Single-facet structure
    out_payload = {
        "facet_order": ["toolbench_category"],
        "ontology": {"toolbench_category": ontology},
        "facet_counts": {"toolbench_category": len(ontology)},
        "n_categories": len(ontology),
        "source": "StableToolBench solvable_queries test_instruction",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2))
    print(f"Wrote {out}")

    # Summary
    sent_counts = sorted(len(v) for v in ontology.values())
    print(
        f"Per-category sentence counts: min={sent_counts[0]} "
        f"med={sent_counts[len(sent_counts)//2]} "
        f"max={sent_counts[-1]} "
        f"mean={sum(sent_counts)/len(sent_counts):.1f}"
    )


if __name__ == "__main__":
    main()
