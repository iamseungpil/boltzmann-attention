#!/usr/bin/env python3
"""gen_metatool_contrastive_pairs_v2.py — High-quality SEKA pairs from MetaTool Subtask1.

Key improvement over v1: uses FULL action_prompt (with tool candidates) as context,
and the correct tool name as the answer. This gives SEKA's SVD rich signal
from the actual FC prompt structure.

Pair structure:
  context_1: "Query: {query}\nAvailable tools: {tool_list}\nThe best tool is {tool_A}."
  context_2: "Query: {query2}\nAvailable tools: {tool_list2}\nThe best tool is {tool_B}."
  question_1: "Which tool should be used?"
  answer_1: tool_A
  question_2: "Which tool should be used?"
  answer_2: tool_B
"""
from __future__ import annotations
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

random.seed(42)

def parse_candidates(prompt):
    cands = []
    for m in re.finditer(r'\d+\.\s+tool name:\s+([A-Za-z0-9_\-\+\.]+)', prompt):
        cands.append(m.group(1))
    return cands

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
        gt = e.get("tool", "None")
        if gt != "None" and e.get("action_prompt"):
            cands = parse_candidates(e["action_prompt"])
            if cands and gt in cands:
                by_tool[gt].append(e)

    tools = sorted(by_tool.keys())
    print(f"[load] {len(data)} entries, {len(tools)} tools with valid candidates")

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

        q1 = extract_query(e1)
        q2 = extract_query(e2)
        cands1 = parse_candidates(e1["action_prompt"])
        cands2 = parse_candidates(e2["action_prompt"])
        t1, t2 = e1["tool"], e2["tool"]

        ctx1 = f"{q1}\nAvailable tools: {', '.join(cands1)}.\nThe best tool is {t1}."
        ctx2 = f"{q2}\nAvailable tools: {', '.join(cands2)}.\nThe best tool is {t2}."

        pairs.append({
            "id": len(pairs) + 1,
            "context_1": ctx1,
            "context_2": ctx2,
            "question_1": f"Which tool should be used? Answer: {t1}",
            "answer_1": t1,
            "question_2": f"Which tool should be used? Answer: {t2}",
            "answer_2": t2,
        })
        i += 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[done] {len(pairs)} pairs -> {args.out}")
    print(f"[sample]\n{json.dumps(pairs[0], indent=2, ensure_ascii=False)[:600]}")

if __name__ == "__main__":
    main()
