#!/usr/bin/env python
"""A-0 edge-miss audit extractor (FIELD_GAP §18.1 / §17.8, zero-GPU BLOCKING before RFT).

Replicates evaluate.py's link reconstruction (resource: <node-j> tags on both gold
and pred, node names '_'->' ') and dumps N sampled cases where gold links are
missed by the prediction, for MANUAL real-error vs valid-alternative judgement.

Usage:
  python tb_a0_audit.py --eval_dir <domain>_evalfull_qwen7b --domain_dir <domain> \
      --llm qwen7b --n 30 --out /home/woori/scratch/tb_a0_audit_mm.md
"""
import argparse, json, random


def tag_links(nodes):
    names = [str(n.get("task", "")).replace("_", " ") for n in nodes]
    links = []
    for inx, node in enumerate(nodes):
        for argument in node.get("arguments", []):
            try:
                if isinstance(argument, dict):
                    argument = list(argument.values())[0]
                if isinstance(argument, list):
                    argument = " ".join(str(a) for a in argument)
                if isinstance(argument, str) and "<node-" in argument:
                    s = argument.index("<node-") + 6
                    e = argument.index(">")
                    j = int(argument[s:e])
                    if j == inx:
                        continue
                    links.append((names[j], names[inx]))
            except Exception:
                pass
    return links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--domain_dir", required=True)
    ap.add_argument("--llm", default="qwen7b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ur = {}
    for l in open(f"{args.domain_dir}/user_requests.json"):
        d = json.loads(l)
        ur[d["id"]] = d["user_request"]
    gold = {d["id"]: d for d in map(json.loads, open(f"{args.eval_dir}/data.json"))}
    preds = {}
    for l in open(f"{args.eval_dir}/predictions/{args.llm}.json"):
        d = json.loads(l)
        preds[d["id"]] = d

    miss_cases = []
    n_eval = 0
    for id_, g in gold.items():
        p = preds.get(id_)
        if not p:
            continue
        res = p.get("result", p)
        pn = res.get("task_nodes") or []
        if not isinstance(pn, list) or any(not isinstance(x, dict) for x in pn):
            continue
        n_eval += 1
        gl = set(tag_links(g["task_nodes"]))
        pl = set(tag_links(pn))
        missed = gl - pl
        if missed:
            miss_cases.append((id_, g, res, sorted(gl), sorted(pl), sorted(missed), sorted(pl - gl)))

    rng = random.Random(args.seed)
    sample = rng.sample(miss_cases, min(args.n, len(miss_cases)))
    with open(args.out, "w") as wf:
        wf.write(f"# A-0 edge-miss audit ({args.eval_dir}, llm={args.llm})\n\n")
        wf.write(f"evaluated={n_eval} with_edge_miss={len(miss_cases)} sampled={len(sample)}\n\n")
        for id_, g, res, gl, pl, missed, extra in sample:
            wf.write(f"## id={id_} type={g.get('type')}\n")
            wf.write(f"REQUEST: {ur.get(id_, '?')}\n")
            wf.write(f"GOLD nodes: {json.dumps(g['task_nodes'], ensure_ascii=False)}\n")
            wf.write(f"PRED nodes: {json.dumps(res.get('task_nodes'), ensure_ascii=False)}\n")
            wf.write(f"GOLD links: {gl}\nPRED links: {pl}\n")
            wf.write(f"MISSED: {missed}\nEXTRA: {extra}\n\n")
    print(f"[a0] evaluated={n_eval} edge-miss cases={len(miss_cases)} "
          f"({100.0 * len(miss_cases) / max(n_eval, 1):.1f}%) sampled={len(sample)} -> {args.out}")


if __name__ == "__main__":
    main()
