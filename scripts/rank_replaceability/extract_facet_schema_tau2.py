#!/usr/bin/env python3
"""extract_facet_schema_tau2.py — E11 prep.

Build facet schemas for tau2-bench retail/telecom/airline domains using
the same deterministic keyword rules as extract_facet_schema_metatool.py.

Tool names are pulled from each domain's tasks.json evaluation_criteria,
descriptions from measure_phi_rank._TAU2_TOOL_DESCRIPTIONS or fallback
(name-based heuristic). Facet assignment: (action_class, sub_domain).
sub_domain is fixed per file (retail/telecom/airline) but the schema also
records action_class for cross-domain transfer (E11).

Usage:
  /home/woori/venvs/seka_env/bin/python3.12 extract_facet_schema_tau2.py \\
    --tau2-root /home/woori/workspace_common/boltzmann-attention/external/tau2-bench \\
    --domains retail,telecom,airline \\
    --out-dir data/facet_schemas
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Reuse same rule tables from metatool extractor.
from extract_facet_schema_metatool import (  # type: ignore
    _ACTION_RULES,
    _DOMAIN_KEYWORDS,
    classify_action,
)

try:
    from measure_phi_rank import _TAU2_TOOL_DESCRIPTIONS, _extract_domain_tools  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"WARN: could not import measure_phi_rank: {exc}", file=sys.stderr)
    _TAU2_TOOL_DESCRIPTIONS = {}
    _extract_domain_tools = None


def name_to_words(name: str) -> str:
    """Convert snake_case tool name to space-separated words for fallback desc."""
    s = name.replace("_", " ")
    return s


def fallback_description(name: str, sub_domain: str) -> str:
    """Generate a minimal description from the tool name when no canonical
    description exists. Keeps the action verb and arguments visible to
    keyword classification."""
    return f"{sub_domain}: {name_to_words(name)}"


def classify_action_robust(name: str, desc: str) -> str:
    """Try description first; if default fallback (no rule matched) and the
    description was empty/synthetic, also probe the tool name itself."""
    a = classify_action(desc)
    if a == "read":
        # Try harder using the snake_case verbs in the tool name.
        words = name_to_words(name).lower()
        for kws, label in _ACTION_RULES:
            for kw in kws:
                if kw in words:
                    return label
    return a


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tau2-root",
        default="/home/woori/workspace_common/boltzmann-attention/external/tau2-bench",
    )
    p.add_argument("--domains", default="retail,telecom,airline")
    p.add_argument("--out-dir", default="data/facet_schemas")
    args = p.parse_args()

    if _extract_domain_tools is None:
        print("ERROR: measure_phi_rank._extract_domain_tools unavailable", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    summary_rows: List[Tuple[str, int, int, int]] = []  # (domain, n_tools, n_actions, n_with_canonical_desc)

    for domain in domains:
        tasks_path = Path(args.tau2_root) / "data" / "tau2" / "domains" / domain / "tasks.json"
        if not tasks_path.exists():
            print(f"WARN: {tasks_path} missing, skipping", file=sys.stderr)
            continue
        tasks = json.load(open(tasks_path))
        tool_names = _extract_domain_tools(tasks)

        schema: Dict[str, Dict[str, str]] = {}
        n_canonical_desc = 0
        for name in sorted(tool_names):
            desc_canon = _TAU2_TOOL_DESCRIPTIONS.get(name, "")
            if desc_canon:
                desc = desc_canon
                n_canonical_desc += 1
            else:
                desc = fallback_description(name, domain)
            action = classify_action_robust(name, desc)
            schema[name] = {
                "action": action,
                "domain": domain,
                "desc_short": desc[:80] + ("…" if len(desc) > 80 else ""),
            }

        n_actions = len({v["action"] for v in schema.values()})
        summary_rows.append((domain, len(schema), n_actions, n_canonical_desc))

        out_path = out_dir / f"tau2_{domain}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# tau2-bench {domain} facet schema (auto-extracted)\n")
            f.write(
                f"# {len(schema)} tools, {n_actions} action classes, "
                f"{n_canonical_desc}/{len(schema)} with canonical description\n"
            )
            f.write("schema_version: 1\n")
            f.write(f"source: tau2-bench/data/tau2/domains/{domain}/tasks.json\n")
            f.write("tools:\n")
            for name in sorted(schema):
                v = schema[name]
                qn = name.replace('"', '\\"')
                if any(c in name for c in ":#&[]{}!|>%@*"):
                    qn = f'"{qn}"'
                f.write(f"  {qn}:\n")
                f.write(f"    action: {v['action']}\n")
                f.write(f"    domain: {v['domain']}\n")
                f.write(f"    desc_short: {json.dumps(v['desc_short'])}\n")

        print(f"# wrote {out_path} ({len(schema)} tools)", file=sys.stderr)

    print("\n# domain summary (domain, n_tools, n_actions, n_with_canonical_desc):", file=sys.stderr)
    for row in summary_rows:
        print(f"#   {row[0]:10s} n={row[1]:3d} actions={row[2]} canonical_desc={row[3]}/{row[1]}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
