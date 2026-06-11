#!/usr/bin/env python
"""Node P/R + early-close (short) census — §8 보강의 inline 분석을 박제 (누락축 판정용).

Per pred-set: mean node precision / recall, short rate (n_pred < n_gold),
mean node deficit overall (n_gold - n_pred) and among short ids.

Usage:
  python tb_pr_census.py --set <eval_dir>:<llm> [--set <eval_dir>:<llm> ...]
(eval dir = output of tb_build_eval.py: data.json + predictions/<llm>.json)
"""
import argparse, json


def norm(s):
    return str(s).replace("_", " ").strip()


def stats(d, llm):
    gold = {x["id"]: x for x in map(json.loads, open(f"{d}/data.json"))}
    preds = {}
    for l in open(f"{d}/predictions/{llm}.json"):
        x = json.loads(l)
        preds[x["id"]] = x.get("result", x)
    ids = sorted(set(gold) & set(preds))
    ps, rs, defs_, short = [], [], [], 0
    nparse = 0
    for i in ids:
        res = preds[i]
        nodes = res.get("task_nodes") if isinstance(res, dict) else None
        if not isinstance(nodes, list) or any(
                not (isinstance(x, dict) and "task" in x) for x in nodes):
            continue
        nparse += 1
        pn = {norm(n["task"]) for n in nodes}
        gn = {norm(x["task"]) for x in gold[i]["task_nodes"]}
        inter = len(pn & gn)
        ps.append(inter / len(pn) if pn else 0.0)
        rs.append(inter / len(gn) if gn else 0.0)
        deficit = len(gold[i]["task_nodes"]) - len(nodes)
        defs_.append(deficit)
        short += deficit > 0
    n = max(nparse, 1)
    return {"n_ids": len(ids), "n_parsed": nparse,
            "node_P": sum(ps) / n, "node_R": sum(rs) / n,
            "short": short, "short_rate": short / n,
            "mean_deficit": sum(defs_) / n,
            "mean_deficit_short": (sum(x for x in defs_ if x > 0) / short) if short else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="<eval_dir>:<llm>, repeatable")
    a = ap.parse_args()
    for spec in a.set:
        d, llm = spec.rsplit(":", 1)
        s = stats(d, llm)
        print(f"{llm:16s} n={s['n_ids']} parsed={s['n_parsed']} "
              f"P={s['node_P']:.3f} R={s['node_R']:.3f} "
              f"short={s['short']}/{s['n_parsed']} ({100*s['short_rate']:.1f}%) "
              f"deficit={s['mean_deficit']:+.3f} (short-only {s['mean_deficit_short']:.2f})")


if __name__ == "__main__":
    main()
