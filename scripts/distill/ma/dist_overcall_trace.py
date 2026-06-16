#!/usr/bin/env python3
"""Trajectory trace of the dist adapter's full-catalog PRECISION wall.
For every evaluable sample, diff predicted task_nodes vs gold task_nodes and aggregate
the FALSE-POSITIVE (over-called) tools and FALSE-NEGATIVE (missed) tools. Confirms the
mechanism behind precision 0.44 / recall 0.90 (over-calling from the 40-tool catalog).
"""
import json, argparse
from collections import Counter


def node_names(nodes):
    out = []
    for n in nodes:
        if isinstance(n, dict):
            out.append(n.get("task") or n.get("name") or n.get("id"))
        else:
            out.append(n)
    return out


def trace(data_dir, llm):
    gold = {}
    for line in open(f"{data_dir}/data.json", encoding="utf-8"):
        d = json.loads(line)
        gold[d["id"]] = node_names(d["task_nodes"])
    preds = {}
    for line in open(f"{data_dir}/predictions/{llm}.json", encoding="utf-8"):
        d = json.loads(line)
        preds[d["id"]] = node_names((d.get("result") or {}).get("task_nodes") or [])
    ids = set(gold) & set(preds)
    fp, fn = Counter(), Counter()
    over, exact, under = 0, 0, 0
    tot_pred, tot_gold, tot_tp = 0, 0, 0
    per = []
    for i in ids:
        g, p = set(gold[i]), set(preds[i])
        tot_pred += len(p); tot_gold += len(g); tot_tp += len(g & p)
        fp.update(p - g); fn.update(g - p)
        if len(p) > len(g): over += 1
        elif len(p) == len(g) and p == g: exact += 1
        elif len(p) < len(g): under += 1
        per.append((i, len(g), len(p), len(p - g), len(g - p)))
    prec = tot_tp / max(tot_pred, 1)
    rec = tot_tp / max(tot_gold, 1)
    print(f"== {llm}: {len(ids)} evaluable ==")
    print(f"micro prec={prec:.3f} rec={rec:.3f} | tot_pred={tot_pred} tot_gold={tot_gold} tp={tot_tp}")
    print(f"samples: over-called={over} exact={exact} under-called={under}")
    print(f"avg pred/sample={tot_pred/len(ids):.2f} vs gold/sample={tot_gold/len(ids):.2f}")
    print(f"total FALSE-POSITIVE (over-call) node instances = {sum(fp.values())}")
    print("top-15 over-called (FP) tools:")
    for t, c in fp.most_common(15):
        print(f"   +{c:4d}  {t}")
    print(f"total FALSE-NEGATIVE (missed) = {sum(fn.values())}; top-8:")
    for t, c in fn.most_common(8):
        print(f"   -{c:4d}  {t}")
    return per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/woori/scratch/JARVIS_tb/taskbench/data_dailylifeapis_evalfull_qwen7b")
    ap.add_argument("--llm", default="nfcdist_fc")
    args = ap.parse_args()
    trace(args.data_dir, args.llm)
