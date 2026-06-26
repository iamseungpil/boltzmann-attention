#!/usr/bin/env python3
"""param_dataflow.py — parameter-relationship SEMANTIC LAYER for the ontology.

Adds a data-flow layer over tools: which ENTITY-KEY parameters an action consumes,
and which READ tool produces each (provenance). Auto-derived from the live env tool
schemas (action param names) + shipped read-tool response keys. This is the TBox
relation type (PARAM_PROVENANCE / read->action entity binding) complementing the
existing PARAMETER_FEEDS (action->action) in the 42-relation ontology; instances are
domain-specific (ABox) -> emitted per-domain JSON, swappable for cross-domain.

Purpose: arg-correctness can't be a literal value match (IDs are instance-specific;
tau2 reward is env-assertion). The meaningful check is data-flow BINDING: did the agent
pass each entity param a value that was actually RETRIEVED from a prior read
(provenance), vs hallucinated/empty. procedure_scorecard's arg_bind uses this.

Emits param_dataflow_<domain>.json:
  { entity_keys:[...], provenance:{param:[reads]}, action_params:{tool:[params]},
    read_tools:[...] }

Usage:
  python scripts/distill/param_dataflow.py --domains telecom retail airline \
     --out-dir reports/facet_rft_2026/phase4_distill/induced
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"
READ = ("get_", "list_", "find_", "search_", "lookup")


def is_read(t):
    return t.startswith(READ)


def derive(domain, tau2_root, shipped_dir, n_sims=400):
    sys.path.insert(0, os.path.join(tau2_root, "src"))
    from tau2.registry import registry
    env = registry.get_env_constructor(domain)()
    action_params, read_tools = {}, []
    for t in env.get_tools():
        name = t.openai_schema["function"]["name"]
        props = list(t.openai_schema["function"]["parameters"].get("properties", {}).keys())
        if is_read(name):
            read_tools.append(name)
        else:
            action_params[name] = props
    # scan shipped read responses -> output keys per read tool
    read_keys = Counter()
    read_keys_by_tool = {}
    files = (glob.glob(os.path.join(shipped_dir, f"*_{domain}_default_*4trials.json")) +
             glob.glob(os.path.join(shipped_dir, f"*_{domain}_base_*4trials.json")))
    for f in files[:2]:
        for s in json.load(open(f)).get("simulations", [])[:n_sims]:
            msgs = s.get("messages") or []
            id2name = {}
            for m in msgs:
                if m.get("role") == "assistant":
                    for tc in m.get("tool_calls") or []:
                        id2name[tc.get("id")] = tc.get("name")
            for m in msgs:
                if m.get("role") == "tool":
                    caller = id2name.get(m.get("id"))
                    if not (caller and is_read(caller)):
                        continue
                    c = m.get("content")
                    try:
                        d = json.loads(c) if isinstance(c, str) else c
                    except Exception:
                        d = None
                    rows = d if isinstance(d, list) else [d]
                    for row in rows:
                        if isinstance(row, dict):
                            for k in row.keys():
                                read_keys[k] += 1
                                read_keys_by_tool.setdefault(caller, set()).add(k)
    all_action_params = set(p for ps in action_params.values() for p in ps)
    entity_keys = sorted(p for p in all_action_params if read_keys.get(p, 0) > 0)
    provenance = {p: sorted(r for r, ks in read_keys_by_tool.items() if p in ks)
                  for p in entity_keys}
    return {"domain": domain, "entity_keys": entity_keys, "provenance": provenance,
            "action_params": action_params, "read_tools": read_tools}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for dom in args.domains:
        df = derive(dom, args.tau2_root, args.shipped_dir)
        out = os.path.join(args.out_dir, f"param_dataflow_{dom}.json")
        json.dump(df, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"[{dom}] entity_keys={df['entity_keys']}")
        print(f"        provenance={df['provenance']}")
        print(f"        -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
