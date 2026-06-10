#!/usr/bin/env python
"""Format-interference census (full sweep, zero-GPU).

Verifies the LODO day-1 hypothesis ("held-out drop = output-convention interference,
not tool-semantics") directly from prediction logs instead of score patterns.

For every prediction line (joined with gold by id):
  - parse failure of result
  - argument shape: dict-args ({name,value} = temporal convention) vs str-args
    (flat strings / <node-j> tags = resource convention)
  - task_links emitted? (temporal convention; resource gold never needs them)
  - <node-j> tags present? (resource convention)
  - temporal domains: edge micro-P/R from explicit task_links vs gold sampled_links
  - resource domains: edge micro-P/R from <node-j>-derived links vs gold sampled_links
Stratified by gold type (single/chain/dag) where it matters (links only exist for chain/dag).

Usage (remote):
  python tb_format_census.py --tb_dir /home/woori/scratch/JARVIS_tb/taskbench \
      --domain data_dailylifeapis --preds qwen7b tb_lodo_daily
"""
import argparse, json, os
from collections import Counter


def pj(v):
    return json.loads(v) if isinstance(v, str) else v


def load_gold(tb_dir, domain):
    gold = {}
    for l in open(os.path.join(tb_dir, domain, "data.json")):
        d = json.loads(l)
        nodes = pj(d["tool_nodes"])
        links = pj(d.get("sampled_links", "[]")) or []
        if not isinstance(nodes, list) or any(not isinstance(x, dict) for x in nodes):
            continue
        gold[d["id"]] = {
            "type": d.get("type", "single"),
            "tasks": [x.get("task") for x in nodes],
            "links": {(e.get("source"), e.get("target")) for e in links
                      if isinstance(e, dict) and e.get("source") and e.get("target")},
        }
    return gold


def arg_shape(nodes):
    """Classify a prediction's argument convention. Returns (n_dict, n_str, n_other) node counts."""
    nd = ns = no = 0
    for x in nodes:
        a = x.get("arguments", [])
        if not isinstance(a, list) or not a:
            no += 1
        elif all(isinstance(e, dict) and "name" in e for e in a):
            nd += 1
        elif all(isinstance(e, str) for e in a):
            ns += 1
        else:
            no += 1
    return nd, ns, no


def pred_edges_temporal(nodes, links):
    return {(e.get("source"), e.get("target")) for e in (links or [])
            if isinstance(e, dict) and e.get("source") and e.get("target")}


def pred_edges_resource(nodes):
    """Derive edges from <node-j> tags inside string arguments (resource convention)."""
    edges = set()
    names = [x.get("task") for x in nodes]
    for j, x in enumerate(nodes):
        for a in x.get("arguments", []):
            s = a if isinstance(a, str) else json.dumps(a)
            for i in range(len(names)):
                if f"<node-{i}>" in s and i != j and names[i] and names[j]:
                    edges.add((names[i], names[j]))
    return edges


def census(tb_dir, domain, pred_name, gold):
    dep = "temporal" if "dailylife" in domain else "resource"
    path = os.path.join(tb_dir, domain, "predictions", pred_name + ".json")
    C = Counter()
    edge_tp = edge_pred = edge_gold = 0          # over gold-linked (chain/dag) tasks
    linked_examples = {"links_emitted": 0, "links_empty_or_missing": 0}
    by_type_linkmiss = Counter()
    by_type_n = Counter()
    for l in open(path):
        try:
            d = json.loads(l)
        except Exception:
            C["line_unparseable"] += 1
            continue
        g = gold.get(d.get("id"))
        if g is None:
            C["no_gold"] += 1
            continue
        C["n"] += 1
        r = d.get("result")
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                C["result_parse_fail"] += 1
                continue
        if not isinstance(r, dict) or not isinstance(r.get("task_nodes"), list):
            C["result_parse_fail"] += 1
            continue
        nodes = [x for x in r["task_nodes"] if isinstance(x, dict)]
        nd, ns, no = arg_shape(nodes)
        C["nodes_dictarg"] += nd; C["nodes_strarg"] += ns; C["nodes_otherarg"] += no
        blob = json.dumps(r)
        if "<node-" in blob:
            C["ex_with_nodetag"] += 1
        tl = r.get("task_links")
        if isinstance(tl, list) and len(tl) > 0:
            C["ex_links_emitted"] += 1
        # edge accounting on gold-linked examples only (single tasks have no gold links)
        if g["links"]:
            by_type_n[g["type"]] += 1
            pe = pred_edges_temporal(nodes, tl) if dep == "temporal" else pred_edges_resource(nodes)
            if dep == "temporal":
                if isinstance(tl, list) and len(tl) > 0:
                    linked_examples["links_emitted"] += 1
                else:
                    linked_examples["links_empty_or_missing"] += 1
                    by_type_linkmiss[g["type"]] += 1
            edge_tp += len(pe & g["links"]); edge_pred += len(pe); edge_gold += len(g["links"])
    p = edge_tp / edge_pred if edge_pred else 0.0
    rc = edge_tp / edge_gold if edge_gold else 0.0
    print(f"\n### {domain} :: {pred_name}  (dep={dep})")
    print(f"  examples n={C['n']}  result_parse_fail={C['result_parse_fail']}  line_unparseable={C['line_unparseable']}")
    tot_nodes = C["nodes_dictarg"] + C["nodes_strarg"] + C["nodes_otherarg"]
    if tot_nodes:
        print(f"  arg convention (nodes): dict(temporal)={C['nodes_dictarg']} ({100*C['nodes_dictarg']/tot_nodes:.1f}%)"
              f"  str(resource)={C['nodes_strarg']} ({100*C['nodes_strarg']/tot_nodes:.1f}%)  other={C['nodes_otherarg']}")
    print(f"  ex_with_<node-j>_tag={C['ex_with_nodetag']}  ex_task_links_emitted={C['ex_links_emitted']} / {C['n']}")
    if dep == "temporal":
        le, lm = linked_examples["links_emitted"], linked_examples["links_empty_or_missing"]
        print(f"  gold-linked(chain/dag) ex: links emitted={le}  EMPTY/MISSING={lm}"
              f"  ({100*lm/(le+lm):.1f}% miss)" if (le + lm) else "  gold-linked ex: none")
        if lm:
            print(f"    link-miss by type: {dict(by_type_linkmiss)} of {dict(by_type_n)}")
    print(f"  edge micro vs gold (gold-linked ex only): P={p:.3f} R={rc:.3f}"
          f"  (tp={edge_tp} pred={edge_pred} gold={edge_gold})")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    args = ap.parse_args()
    gold = load_gold(args.tb_dir, args.domain)
    print(f"== {args.domain}: gold={len(gold)} (linked: {sum(1 for g in gold.values() if g['links'])})")
    for p in args.preds:
        census(args.tb_dir, args.domain, p, gold)


if __name__ == "__main__":
    main()
