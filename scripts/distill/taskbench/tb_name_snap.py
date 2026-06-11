#!/usr/bin/env python
"""grounded-copy v0 (post-hoc, zero-GPU): snap invalid tool names in predictions to the
nearest valid tool name from the domain's tool list (census 어휘-간섭 축 직접 처방의
가장 싼 변형 — 효과가 양이면 inference-side 제약으로 승격).

Snapping: exact-normalized match passthrough; else difflib closest (cutoff) on
normalized names; no match -> leave as-is (counted). Writes a snapped pred file.

Usage:
  python tb_name_snap.py --pred <pred.json> --tool_desc <tool_desc.json> --out <snapped.json>
"""
import argparse, difflib, json


def norm(s):
    return str(s).replace("_", " ").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=float, default=0.6)
    a = ap.parse_args()

    tools = json.load(open(a.tool_desc))["nodes"]
    canon = {}  # normalized -> canonical id
    for t in tools:
        canon[norm(t["id"])] = t["id"]
    keys = list(canon)

    n_snap = n_miss = n_nodes = 0
    with open(a.out, "w") as wf:
        for l in open(a.pred):
            d = json.loads(l)
            res = d.get("result")
            if isinstance(res, dict) and isinstance(res.get("task_nodes"), list):
                for node in res["task_nodes"]:
                    if not isinstance(node, dict) or "task" not in node:
                        continue
                    n_nodes += 1
                    nm = norm(node["task"])
                    if nm in canon:
                        node["task"] = canon[nm]  # canonicalize case/underscore too
                        continue
                    m = difflib.get_close_matches(nm, keys, n=1, cutoff=a.cutoff)
                    if m:
                        node["task"] = canon[m[0]]
                        n_snap += 1
                    else:
                        n_miss += 1
            wf.write(json.dumps(d) + "\n")
    print(f"[snap] nodes={n_nodes} snapped={n_snap} unmatched={n_miss} -> {a.out}")


if __name__ == "__main__":
    main()
