#!/usr/bin/env python3
"""extract_facet_schema_metatool.py

One-shot script: enumerate all 78 unique candidate tools in MetaTool ST4,
extract their canonical descriptions from action_prompt, and assign a
deterministic (action_class, domain) facet using keyword rules. Output:
data/facet_schemas/metatool_st4.yaml

Usage:
  /home/woori/venvs/seka_env/bin/python3.12 extract_facet_schema_metatool.py \
    --metatool /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json \
    --out data/facet_schemas/metatool_st4.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Tool entry pattern: "1. tool name: NAME, tool description: DESC\n"
_TOOL_ENTRY_RE = re.compile(
    r'\d+\.\s*tool name:\s*([^,]+?),\s*tool description:\s*(.+?)(?=\n\d+\.\s*tool name:|\n\n|\n\[List|$)',
    re.DOTALL,
)

# Facet keyword rules (priority-ordered first match wins).
# action_class:
#   read    : retrieves/looks up information without side effect
#   search  : query-driven retrieval over external corpus
#   create  : generates new content
#   modify  : transforms or edits existing content
#   compute : calculates / analyzes
#   communicate: sends external messages / calls
#   manage  : tracks / stores / organizes (notes, reminders)
#   convert : format / language conversion
#
# domain: open-vocabulary one-token tag (news, finance, weather, ...).

_ACTION_RULES: List[Tuple[List[str], str]] = [
    (["search", "find", "look up", "lookup", "browse", "discover"], "search"),
    (["generate", "create", "produce", "make", "build", "compose", "design"], "create"),
    (["convert", "translate", "transform", "polish", "rewrite", "compress"], "convert"),
    (["calculate", "compute", "analyze", "estimate", "predict"], "compute"),
    (["send", "email", "message", "post", "share", "communicate"], "communicate"),
    (["modify", "edit", "update", "change", "adjust", "trim"], "modify"),
    (["track", "manage", "remind", "organize", "store", "schedule", "list"], "manage"),
    (["check", "verify", "validate", "inspect", "monitor"], "check"),
]

_DOMAIN_KEYWORDS: List[Tuple[List[str], str]] = [
    (["news", "headline", "press", "media coverage", "current events"], "news"),
    (["finance", "stock", "market", "investment", "trading", "crypto", "currency"], "finance"),
    (["weather", "forecast", "temperature", "climate"], "weather"),
    (["music", "song", "track", "audio", "podcast", "spotify"], "music"),
    (["movie", "film", "tv", "show", "video", "youtube"], "media"),
    (["job", "career", "employment", "resume", "hiring", "salary"], "jobs"),
    (["travel", "flight", "hotel", "vacation", "trip", "destination", "airline"], "travel"),
    (["education", "course", "learn", "study", "tutorial", "lesson", "class"], "education"),
    (["health", "diet", "fitness", "medical", "exercise", "nutrition"], "health"),
    (["law", "legal", "regulation", "compliance"], "law"),
    (["code", "programming", "github", "repo", "developer", "api"], "code"),
    (["map", "location", "navigation", "direction", "geography", "city"], "geo"),
    (["shopping", "product", "discount", "coupon", "deal", "buy", "ecommerce"], "commerce"),
    (["real estate", "house", "apartment", "property", "rent", "buying a home"], "realestate"),
    (["nasa", "space", "astronomy", "earthquake", "scientific", "research", "paper"], "science"),
    (["restaurant", "food", "recipe", "meal", "cuisine"], "food"),
    (["game", "soccer", "football", "sport", "match"], "sports"),
    (["wikipedia", "encyclopedia", "knowledge base", "general information"], "general"),
    (["chart", "graph", "visualization", "plot"], "viz"),
    (["pdf", "document", "file", "url"], "doc"),
    (["meme", "gif", "joke", "fun"], "fun"),
    (["sql", "database", "data retrieval", "query"], "data"),
    (["tts", "speech", "voice"], "speech"),
    (["company info", "business", "corporate"], "business"),
    (["charity", "donation", "non-profit"], "charity"),
    (["note", "reminder", "to-do", "todo"], "memory"),
    (["app builder", "build app", "low-code"], "appdev"),
    (["scratch", "kid", "education for", "extension of mit"], "edu_dev"),
    (["gift", "present"], "gift"),
    (["security", "breach", "credential", "leak", "hack"], "security"),
    (["ad", "marketing", "ppc", "campaign"], "marketing"),
]


def parse_action_prompt(action_prompt: str) -> Dict[str, str]:
    """Extract {tool_name: description} dict from a single action_prompt."""
    text = action_prompt.strip().strip('"')
    out = {}
    for m in _TOOL_ENTRY_RE.finditer(text):
        name = m.group(1).strip().strip('"').strip("'")
        desc = m.group(2).strip().rstrip(".").strip()
        # Description sometimes wraps in [' ... ']: strip those.
        desc = desc.strip("[").strip("]").strip("'").strip('"').strip()
        if name and desc and name not in out:
            out[name] = desc
    return out


def classify_action(desc: str) -> str:
    desc_lc = desc.lower()
    for kws, label in _ACTION_RULES:
        for kw in kws:
            if kw in desc_lc:
                return label
    return "read"  # default if nothing matches


def classify_domain(name: str, desc: str) -> str:
    text = f"{name} {desc}".lower()
    for kws, label in _DOMAIN_KEYWORDS:
        for kw in kws:
            if kw in text:
                return label
    return "general"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metatool", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    raw = json.load(open(args.metatool))
    # Build canonical {tool_name: description} from first occurrence in any record.
    canonical: Dict[str, str] = {}
    candidate_universe = set()
    gt_universe = set()
    for entry in raw:
        ap = entry.get("action_prompt") or ""
        gt = entry.get("tool") or []
        if isinstance(gt, str):
            gt_universe.add(gt)
        elif isinstance(gt, list):
            gt_universe.update(gt)
        descs = parse_action_prompt(ap)
        for name, desc in descs.items():
            candidate_universe.add(name)
            if name not in canonical:
                canonical[name] = desc

    print(f"# unique candidate tools: {len(candidate_universe)}", file=sys.stderr)
    print(f"# unique GT tools:        {len(gt_universe)}", file=sys.stderr)
    print(f"# tools with description: {len(canonical)}", file=sys.stderr)

    # Build facet schema
    schema: Dict[str, Dict[str, str]] = {}
    for name in sorted(candidate_universe):
        desc = canonical.get(name, "")
        action = classify_action(desc)
        domain = classify_domain(name, desc)
        schema[name] = {
            "action": action,
            "domain": domain,
            "desc_short": desc[:80] + ("…" if len(desc) > 80 else ""),
        }

    # Write YAML manually (no PyYAML dep)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# MetaTool ST4 facet schema (auto-extracted, deterministic rules)\n")
        f.write(f"# {len(schema)} tools, {len(set(v['action'] for v in schema.values()))} action classes, "
                f"{len(set(v['domain'] for v in schema.values()))} domains\n")
        f.write("schema_version: 1\n")
        f.write("source: MetaTool/Task2-Subtask4.json\n")
        f.write("tools:\n")
        for name in sorted(schema):
            v = schema[name]
            # Quote tool name if it contains special chars
            qn = name.replace('"', '\\"')
            if any(c in name for c in ":#&[]{}!|>%@*"):
                qn = f'"{qn}"'
            f.write(f"  {qn}:\n")
            f.write(f"    action: {v['action']}\n")
            f.write(f"    domain: {v['domain']}\n")
            f.write(f"    desc_short: {json.dumps(v['desc_short'])}\n")

    print(f"# wrote {out_path}", file=sys.stderr)

    # Summary
    from collections import Counter
    action_counts = Counter(v['action'] for v in schema.values())
    domain_counts = Counter(v['domain'] for v in schema.values())
    print(f"\n# action distribution:", file=sys.stderr)
    for k, c in action_counts.most_common():
        print(f"#   {k:12s} {c}", file=sys.stderr)
    print(f"\n# domain distribution:", file=sys.stderr)
    for k, c in domain_counts.most_common():
        print(f"#   {k:12s} {c}", file=sys.stderr)

    # Show GT-tool subset (these matter most for F1)
    print(f"\n# GT tools (15) facet assignment:", file=sys.stderr)
    for name in sorted(gt_universe):
        if name in schema:
            v = schema[name]
            print(f"#   {name:30s} action={v['action']:12s} domain={v['domain']}", file=sys.stderr)
        else:
            print(f"#   {name:30s} *NOT IN candidate universe*", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
