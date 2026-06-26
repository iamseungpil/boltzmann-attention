#!/usr/bin/env python
"""Trajectory census for interpreting eval deltas (전수조사 — aggregate F1 만으로
세운 해석을 궤적 단위로 검증). Compares two prediction sets on the same gold.

Per-id signatures: parseable / node-name validity vs tool list / <node-j> tag usage,
self-ref & dangling refs / per-id node & edge F1 / (temporal) task_links presence +
arg-format correctness. Then a delta census A->B bucketed by edge-F1 change, with
signature shifts per bucket + worked examples dumped for manual reading.

Usage:
  python tb_census.py --dir_a <evalA> --llm_a qwen7b --dir_b <evalB> --llm_b tb_x \
      --tool_desc <domain>/tool_desc.json --dep resource --examples 6 --out report.md
(eval dir = output of tb_build_eval.py: data.json + predictions/<llm>.json)
"""
import argparse, json


def norm(s):
    return str(s).replace("_", " ").strip()


def tag_info(nodes):
    names = [norm(n.get("task", "")) for n in nodes]
    links, nself, ndangle, ntag = set(), 0, 0, 0
    for inx, node in enumerate(nodes):
        args_ = node.get("arguments", [])
        if not isinstance(args_, list):
            continue
        for argument in args_:
            try:
                if isinstance(argument, dict):
                    argument = list(argument.values())[0]
                if isinstance(argument, list):
                    argument = " ".join(str(a) for a in argument)
                if isinstance(argument, str) and "<node-" in argument:
                    ntag += 1
                    j = int(argument[argument.index("<node-") + 6:argument.index(">")])
                    if j == inx:
                        nself += 1
                    elif 0 <= j < len(names):
                        links.add((names[j], names[inx]))
                    else:
                        ndangle += 1
            except Exception:
                pass
    return links, ntag, nself, ndangle


def f1(p, g):
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    i = len(p & g)
    pr, rc = i / len(p), i / len(g)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def sig(res, gold, valid_tools, dep):
    out = {"parse": False, "n_nodes": 0, "valid_frac": 0.0, "ntag": 0, "nself": 0,
           "ndangle": 0, "node_f1": 0.0, "edge_f1": 0.0, "links_ok": None, "argdict_frac": None}
    nodes = res.get("task_nodes") if isinstance(res, dict) else None
    if not isinstance(nodes, list) or any(not (isinstance(x, dict) and "task" in x) for x in nodes):
        return out
    out["parse"] = True
    out["n_nodes"] = len(nodes)
    names = [norm(n["task"]) for n in nodes]
    out["valid_frac"] = sum(n in valid_tools for n in names) / max(len(names), 1)
    gnodes = gold["task_nodes"]
    out["node_f1"] = f1(set(names), {norm(x["task"]) for x in gnodes})
    if dep == "temporal":
        gl = {(norm(l["source"]), norm(l["target"])) for l in gold.get("task_links", [])}
        tl = res.get("task_links")
        out["links_ok"] = isinstance(tl, list) and all(
            isinstance(x, dict) and "source" in x and "target" in x for x in tl)
        pl = set()
        if isinstance(tl, list):
            pl = {(norm(x["source"]), norm(x["target"])) for x in tl
                  if isinstance(x, dict) and "source" in x and "target" in x}
        out["edge_f1"] = f1(pl, gl)
        nargs = ndict = 0
        for n in nodes:
            for a in (n.get("arguments") or []):
                nargs += 1
                ndict += isinstance(a, dict) and "name" in a
        out["argdict_frac"] = ndict / nargs if nargs else None
    else:
        gl, _, _, _ = tag_info(gnodes)
        pl, ntag, nself, ndangle = tag_info(nodes)
        out.update(ntag=ntag, nself=nself, ndangle=ndangle, edge_f1=f1(pl, gl))
    return out


def load(d, llm):
    gold = {x["id"]: x for x in map(json.loads, open(f"{d}/data.json"))}
    preds = {}
    for l in open(f"{d}/predictions/{llm}.json"):
        x = json.loads(l)
        preds[x["id"]] = x.get("result", x)
    return gold, preds


def agg(rows, keys):
    n = max(len(rows), 1)
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = (sum(v if not isinstance(v, bool) else int(v) for v in vals) / max(len(vals), 1)
                  ) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True); ap.add_argument("--llm_a", required=True)
    ap.add_argument("--dir_b", required=True); ap.add_argument("--llm_b", required=True)
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--dep", default="resource")
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    valid = {norm(t["id"]) for t in json.load(open(a.tool_desc))["nodes"]}
    gold_a, preds_a = load(a.dir_a, a.llm_a)
    gold_b, preds_b = load(a.dir_b, a.llm_b)
    ids = sorted(set(gold_a) & set(preds_a) & set(gold_b) & set(preds_b))
    # malformed-gold skip (same trap tb_build_eval handles): a few data.json records
    # lack task_nodes and would KeyError inside sig()
    n_raw = len(ids)
    ids = [i for i in ids if "task_nodes" in gold_a[i] and "task_nodes" in gold_b[i]]
    if len(ids) != n_raw:
        print(f"[census] skipped {n_raw - len(ids)} malformed gold (no task_nodes)")

    KEYS = ["parse", "n_nodes", "valid_frac", "ntag", "nself", "ndangle",
            "node_f1", "edge_f1", "links_ok", "argdict_frac"]
    rows = []
    for i in ids:
        sa = sig(preds_a[i], gold_a[i], valid, a.dep)
        sb = sig(preds_b[i], gold_b[i], valid, a.dep)
        rows.append((i, gold_a[i].get("type"), sa, sb))

    buckets = {"improved": [], "worsened": [], "same": []}
    for i, t, sa, sb in rows:
        d = sb["edge_f1"] - sa["edge_f1"]
        buckets["improved" if d > 0.1 else "worsened" if d < -0.1 else "same"].append((i, t, sa, sb))

    with open(a.out, "w") as wf:
        wf.write(f"# census {a.llm_a} -> {a.llm_b} (n={len(ids)}, dep={a.dep})\n\n")
        wf.write(f"## aggregate\nA: {json.dumps(agg([r[2] for r in rows], KEYS))}\n")
        wf.write(f"B: {json.dumps(agg([r[3] for r in rows], KEYS))}\n\n")
        for name, items in buckets.items():
            wf.write(f"## {name}: {len(items)} ({100*len(items)/max(len(ids),1):.1f}%)  ")
            from collections import Counter
            wf.write(f"types={dict(Counter(t for _, t, _, _ in items))}\n")
            wf.write(f"A: {json.dumps(agg([sa for _, _, sa, _ in items], KEYS))}\n")
            wf.write(f"B: {json.dumps(agg([sb for _, _, _, sb in items], KEYS))}\n\n")
        for name in ("improved", "worsened"):
            wf.write(f"## examples: {name}\n")
            for i, t, sa, sb in buckets[name][:a.examples]:
                wf.write(f"### id={i} type={t} edge {sa['edge_f1']:.2f}->{sb['edge_f1']:.2f}\n")
                wf.write(f"GOLD : {json.dumps(gold_a[i]['task_nodes'], ensure_ascii=False)[:600]}\n")
                ra = preds_a[i].get("task_nodes") if isinstance(preds_a[i], dict) else None
                rb = preds_b[i].get("task_nodes") if isinstance(preds_b[i], dict) else None
                wf.write(f"A    : {json.dumps(ra, ensure_ascii=False)[:600]}\n")
                wf.write(f"B    : {json.dumps(rb, ensure_ascii=False)[:600]}\n")
                if a.dep == "temporal":
                    wf.write(f"A links: {json.dumps(preds_a[i].get('task_links'))[:300]}\n")
                    wf.write(f"B links: {json.dumps(preds_b[i].get('task_links'))[:300]}\n")
                wf.write("\n")
    print(f"[census] n={len(ids)} improved={len(buckets['improved'])} "
          f"worsened={len(buckets['worsened'])} same={len(buckets['same'])} -> {a.out}")


if __name__ == "__main__":
    main()
