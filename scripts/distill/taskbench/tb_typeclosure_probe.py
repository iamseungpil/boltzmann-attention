#!/usr/bin/env python
"""L3 probe (zero-GPU): can a deterministic type-closure gate detect/repair omissions?

For each (pred, gold) pair where pred misses gold nodes (omission), check:
  (a) DETECT: does the pred have a type gap — some node requires an input type that
      no literal arg / no referenced node's output-type provides?
  (b) REPAIR: among tools whose output-type bridges the gap, is the candidate set
      unique — and does it equal the missing gold node? (unique+correct = deterministic
      recovery, multiple = abstain/flag-only, none = undetectable by types)
This is the TaskBench transplant of SOPBench active-H3 (gate drives missing prereqs).

Usage: python tb_typeclosure_probe.py --eval_dir <dir> --llm <tag> \
    --tool_desc data_multimedia/tool_desc.json --out probe.md
"""
import argparse, json
from collections import Counter


def norm(s):
    return str(s).replace("_", " ").strip()


def content_type(s):
    s = str(s).lower().strip()
    for ext, t in ((".jpg", "image"), (".jpeg", "image"), (".png", "image"), (".gif", "image"),
                   (".wav", "audio"), (".mp3", "audio"), (".flac", "audio"),
                   (".mp4", "video"), (".avi", "video"), (".mov", "video")):
        if s.endswith(ext):
            return t
    if s.startswith("http") or "www." in s:
        return "url"
    return "text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tools = json.load(open(a.tool_desc))["nodes"]
    tin = {norm(t["id"]): list(t.get("input-type", [])) for t in tools}
    tout = {norm(t["id"]): list(t.get("output-type", [])) for t in tools}
    producers = {}  # type -> [tool]
    for name, outs in tout.items():
        for t in outs:
            producers.setdefault(t, []).append(name)

    gold = {d["id"]: d for d in map(json.loads, open(f"{a.eval_dir}/data.json"))}
    preds = {}
    for l in open(f"{a.eval_dir}/predictions/{a.llm}.json"):
        x = json.loads(l)
        preds[x["id"]] = x.get("result", x)

    n_omit = 0
    stats = Counter()
    repair_hits = Counter()
    for i, g in gold.items():
        p = preds.get(i)
        if not isinstance(p, dict):
            continue
        pn = p.get("task_nodes")
        if not isinstance(pn, list) or any(not (isinstance(x, dict) and "task" in x) for x in pn):
            continue
        pnames = [norm(x["task"]) for x in pn]
        gnames = [norm(x["task"]) for x in g["task_nodes"]]
        missing = [m for m in gnames if m not in pnames]
        if not missing:
            continue
        n_omit += 1
        # type-gap scan over pred nodes
        gaps = []  # (node, required type)
        for inx, node in enumerate(pn):
            name = pnames[inx]
            req = tin.get(name)
            if req is None:
                continue  # invalid tool name — vocab axis, not omission axis
            have = set()
            for arg in (node.get("arguments") or []):
                if isinstance(arg, dict):
                    arg = list(arg.values())[0] if arg else ""
                if isinstance(arg, list):
                    arg = " ".join(str(x) for x in arg)
                arg = str(arg)
                if "<node-" in arg:
                    try:
                        j = int(arg[arg.index("<node-") + 6:arg.index(">")])
                        if 0 <= j < len(pnames) and j != inx:
                            have.update(tout.get(pnames[j], []))
                    except Exception:
                        pass
                else:
                    have.add(content_type(arg))
            for t in req:
                if t not in have:
                    gaps.append((name, t))
        if not gaps:
            stats["no_gap (type-silent omission)"] += 1
            continue
        stats["gap_detected"] += 1
        # repair: for each gap type, candidate producers; check uniqueness & correctness
        hit = ambiguous = wrong = 0
        for _, t in gaps:
            cands = [c for c in producers.get(t, [])]
            cands_missing = [c for c in cands if c in missing]
            if len(cands) == 1:
                if cands[0] in missing:
                    hit += 1
                else:
                    wrong += 1
            elif len(cands_missing) == 1 and cands_missing[0] in missing:
                # unique once intersected with "useful" set — weaker determinism
                ambiguous += 0  # count separately
                repair_hits["unique_among_missing"] += 1
            elif len(cands) > 1:
                ambiguous += 1
            else:
                wrong += 1
        if hit:
            repair_hits["unique_global_correct"] += 1
        if ambiguous and not hit:
            repair_hits["ambiguous_flag_only"] += 1
        if wrong and not hit and not ambiguous:
            repair_hits["no_or_wrong_candidate"] += 1

    with open(a.out, "w") as wf:
        wf.write(f"# type-closure probe {a.eval_dir} {a.llm}\n")
        wf.write(f"omission cases: {n_omit}\n")
        wf.write(f"detection: {json.dumps(stats)}\n")
        wf.write(f"repair: {json.dumps(repair_hits)}\n")
    print(f"[probe] omission={n_omit} detect={dict(stats)} repair={dict(repair_hits)}")


if __name__ == "__main__":
    main()
