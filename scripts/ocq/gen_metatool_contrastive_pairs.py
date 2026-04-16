#!/usr/bin/env python3
"""Generate SEKA-format contrastive pairs from MetaTool Subtask1."""
from __future__ import annotations
import argparse, json, random, re, sys
from collections import defaultdict
from pathlib import Path

random.seed(42)

def extract_query(entry):
    if "query" in entry and entry["query"]:
        return entry["query"]
    prompt = entry.get("action_prompt", "")
    lines = prompt.strip().split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^\d+\.\s+tool name:', line):
            return '\n'.join(lines[:i]).strip()
    return lines[0].strip() if lines else ""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subtask1", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json")
    p.add_argument("--max-pairs", type=int, default=500)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = json.load(open(args.subtask1))
    by_tool = defaultdict(list)
    for e in data:
        if e.get("tool", "None") != "None":
            by_tool[e["tool"]].append(e)

    tools = sorted(by_tool.keys())
    print(f"[load] {len(data)} entries, {len(tools)} tools")

    all_entries = [e for t in tools for e in by_tool[t]]
    random.shuffle(all_entries)

    pairs = []
    i = 0
    while i < len(all_entries) - 1 and len(pairs) < args.max_pairs:
        e1 = all_entries[i]
        j = i + 1
        while j < len(all_entries) and all_entries[j]["tool"] == e1["tool"]:
            j += 1
        if j >= len(all_entries):
            i += 1
            continue
        all_entries[i+1], all_entries[j] = all_entries[j], all_entries[i+1]
        e2 = all_entries[i+1]
        q1, q2 = extract_query(e1), extract_query(e2)
        pairs.append({
            "id": len(pairs) + 1,
            "context_1": f"The correct tool for this query is {e1['tool']}.",
            "context_2": f"The correct tool for this query is {e2['tool']}.",
            "question_1": q1,
            "answer_1": e1["tool"],
            "question_2": q2,
            "answer_2": e2["tool"],
        })
        i += 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[done] {len(pairs)} pairs -> {args.out}")

if __name__ == "__main__":
    main()
