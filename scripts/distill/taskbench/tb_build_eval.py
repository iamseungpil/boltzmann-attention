#!/usr/bin/env python
"""TaskBench eval builder + runner (HANDOFF_2026_06_10 §5 converter).

Converts original {domain}/data.json (tool_nodes/sampled_links/tool_steps) into
evaluate.py's expected format (task_nodes/task_links/task_steps), id-aligned to
the prediction file, then runs evaluate.py and prints overall metrics.

Usage (remote, tbeval_venv python):
  python tb_build_eval.py --tb_dir /home/woori/scratch/JARVIS_tb/taskbench \
      --domain data_huggingface --llm qwen7b
Dependency type is auto: data_dailylifeapis -> temporal, else resource.
"""
import argparse, json, os, shutil, subprocess, sys


def pj(v):
    return json.loads(v) if isinstance(v, str) else v


def build_eval(src_dir, pred_file, dst, llm):
    os.makedirs(f"{dst}/predictions", exist_ok=True)
    pred_ids = [json.loads(l)["id"] for l in open(pred_file)]
    pset = set(pred_ids)
    gold = {}
    for l in open(f"{src_dir}/data.json"):
        d = json.loads(l)
        if d["id"] in pset:
            tn = pj(d["tool_nodes"]); tl = pj(d.get("sampled_links", "[]")); ts = pj(d["tool_steps"])
            if not isinstance(tn, list) or any(not (isinstance(x, dict) and "task" in x) for x in tn):
                continue  # rare malformed gold (dict-shaped tool_nodes); excluded from eval
            gold[d["id"]] = {
                "id": d["id"], "type": d.get("type", "single"),
                "task_nodes": [{"task": x["task"], "arguments": x.get("arguments", [])} for x in tn],
                "task_links": [{"source": x["source"], "target": x["target"]} for x in tl] if tl else [],
                "task_steps": ts,
            }
    ids = [i for i in pred_ids if i in gold]
    with open(f"{dst}/data.json", "w") as fg:
        for i in ids:
            fg.write(json.dumps(gold[i]) + "\n")
    for aux in ("tool_desc.json", "graph_desc.json"):
        shutil.copy(f"{src_dir}/{aux}", f"{dst}/{aux}")
    shutil.copy(pred_file, f"{dst}/predictions/{llm}.json")
    return len(pred_ids), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--llm", default="qwen7b")
    ap.add_argument("--pred_file", default=None, help="default: {domain}/predictions/{llm}.json")
    ap.add_argument("--dst", default=None, help="default: {domain}_evalfull_{llm}")
    ap.add_argument("--build_only", action="store_true")
    args = ap.parse_args()

    src_dir = os.path.join(args.tb_dir, args.domain)
    pred_file = args.pred_file or f"{src_dir}/predictions/{args.llm}.json"
    dst = args.dst or os.path.join(args.tb_dir, f"{args.domain}_evalfull_{args.llm}")
    dep = "temporal" if "dailylife" in args.domain else "resource"

    n_pred, n_match = build_eval(src_dir, pred_file, dst, args.llm)
    print(f"[build] {args.domain}: preds={n_pred} matched_gold={n_match} -> {dst}")

    if args.build_only:
        return
    cmd = [sys.executable, "evaluate.py", "--data_dir", dst, "--prediction_dir", "predictions",
           "--llm", args.llm, "--splits", "all", "--n_tools", "all", "--mode", "add",
           "--dependency_type", dep, "-m", "f1", "-m", "link", "-m", "argument"]
    r = subprocess.run(cmd, cwd=args.tb_dir)
    if r.returncode != 0:
        sys.exit(r.returncode)
    m = json.load(open(f"{dst}/metrics/{args.llm}.json"))["overall_overall"]
    keys = [k for k in m if any(s in k for s in ("node_micro_f1", "link_binary_f1", "argument"))]
    print(f"[overall] {args.domain} {args.llm}:")
    for k in sorted(keys):
        print(f"  {k} = {m[k]}")


if __name__ == "__main__":
    main()
