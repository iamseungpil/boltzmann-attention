#!/usr/bin/env python
"""Build TaskBench SFT JSONL (Exp-A distill leg, HANDOFF_2026_06_10 §4).

Each example: user = exact inference.py prompt (tool list + GOAL/REQUIREMENTS +
user request), assistant = gold graph JSON (task_steps/task_nodes[/task_links]).
Stratified sampling overweights chain/dag (edge-F1 is the headroom; node ~saturated).

Usage (remote):
  python tb_build_sft.py --tb_dir /home/woori/scratch/JARVIS_tb/taskbench \
      --domain data_huggingface --out /home/woori/scratch/tb_sft/sft_tb_hf.jsonl \
      --n_single 400 --n_chain 1000 --n_dag 0
(n_X = cap per type, 0 = take all)
"""
import argparse, json, os, random


def pj(v):
    return json.loads(v) if isinstance(v, str) else v


RESOURCE_PROMPT = (
    """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]} """
    + """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""
)

TEMPORAL_PROMPT = (
    """\n# GOAL #:\nBased on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}"""
    + """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;"""
)

SUFFIX = """\n\n# USER REQUEST #: {{user_request}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""


def build_tool_string(tool_list, dep):
    # replicate inference.py: temporal strips parameters to name list
    if dep == "temporal":
        tool_list = [dict(t) for t in tool_list]
        for t in tool_list:
            t["parameters"] = [p["name"] for p in t["parameters"]]
    s = "# TASK LIST #:\n"
    for t in tool_list:
        s += json.dumps(t) + "\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_single", type=int, default=400)
    ap.add_argument("--n_chain", type=int, default=1000)
    ap.add_argument("--n_dag", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = os.path.join(args.tb_dir, args.domain)
    dep = "temporal" if "dailylife" in args.domain else "resource"
    ur = {}
    for l in open(f"{src}/user_requests.json"):
        d = json.loads(l)
        ur[d["id"]] = d["user_request"]
    tool_list = json.load(open(f"{src}/tool_desc.json"))["nodes"]
    tool_string = build_tool_string(tool_list, dep)
    head = RESOURCE_PROMPT if dep == "resource" else TEMPORAL_PROMPT

    by_type = {"single": [], "chain": [], "dag": []}
    for l in open(f"{src}/data.json"):
        d = json.loads(l)
        if d["id"] in ur:
            by_type.setdefault(d.get("type", "single"), []).append(d)

    rng = random.Random(args.seed)
    caps = {"single": args.n_single, "chain": args.n_chain, "dag": args.n_dag}
    picked = []
    for t, recs in by_type.items():
        cap = caps.get(t, 0)
        if cap and len(recs) > cap:
            recs = rng.sample(recs, cap)
        picked += recs
    rng.shuffle(picked)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    skipped = 0
    with open(args.out, "w") as wf:
        for d in picked:
            steps = pj(d["tool_steps"]); nodes = pj(d["tool_nodes"])
            if not isinstance(nodes, list) or any(not isinstance(x, dict) for x in nodes):
                skipped += 1  # rare malformed gold (e.g. dict-shaped tool_nodes)
                continue
            result = {"task_steps": steps,
                      "task_nodes": [{"task": x["task"], "arguments": x.get("arguments", [])} for x in nodes]}
            if dep == "temporal":
                result["task_links"] = pj(d.get("sampled_links", "[]")) or []
            user = tool_string + head + "\n" + SUFFIX.replace("{{user_request}}", ur[d["id"]])
            rec = {"messages": [{"role": "user", "content": user},
                                {"role": "assistant", "content": json.dumps(result)}],
                   "meta": {"id": d["id"], "domain": args.domain, "type": d.get("type", "single")}}
            wf.write(json.dumps(rec) + "\n")
            n += 1
    from collections import Counter
    print(f"[sft] {args.domain} dep={dep} wrote {n} (skipped {skipped} malformed) -> {args.out}")
    print("  picked:", dict(Counter(d.get("type", "single") for d in picked)))
    print("  pool:  ", {t: len(v) for t, v in by_type.items()})


if __name__ == "__main__":
    main()
