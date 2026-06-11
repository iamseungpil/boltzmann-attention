#!/usr/bin/env python
"""E6: held-out K-sample gate-selection — inference.py를 K회 돌려 만든 pred 파일들을
id로 조인해 mean/gate(v0/v1)/oracle edge를 측정 (census-식 링크 채점, 상대 갭용).

Usage: tb_kgate_heldout.py --pred k0.json --pred k1.json ... \
           --gold <TB>/data_multimedia/data.json --tool_desc ... --graph_desc ...
"""
import argparse, json
from tb_kgate_select import norm, links_of, f1, load_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--graph_desc", default=None)
    a = ap.parse_args()

    valid = {norm(t["id"]) for t in json.load(open(a.tool_desc))["nodes"]}
    gedges, _ = load_graph([a.graph_desc] if a.graph_desc else None)
    gold = {}
    for l in open(a.gold):
        d = json.loads(l)
        nodes = d.get("task_nodes")
        if isinstance(nodes, list) and all(isinstance(x, dict) and "task" in x for x in nodes):
            gl, _, _, _ = links_of(nodes)
            gold[d["id"]] = gl

    byid = {}
    for p in a.pred:
        for l in open(p):
            d = json.loads(l)
            if d["id"] not in gold:
                continue
            res = d.get("result", {})
            nodes = res.get("task_nodes") if isinstance(res, dict) else None
            byid.setdefault(d["id"], []).append(nodes)

    sums = {"mean": 0.0, "gate_v0": 0.0, "gate_v1": 0.0, "oracle": 0.0}
    n = 0
    for i, cand in byid.items():
        gl = gold[i]
        scored = []
        for nodes in cand:
            if not (isinstance(nodes, list) and all(isinstance(x, dict) and "task" in x for x in nodes)):
                scored.append({"edge": 0.0, "v0": (-1, 0, 0, 0), "v1": (-1, 0, 0, 0, 0)})
                continue
            pl, ntag, nself, ndangle = links_of(nodes)
            names = [norm(x["task"]) for x in nodes]
            vfrac = sum(x in valid for x in names) / max(len(names), 1)
            gmem = (sum(l in gedges for l in pl) / len(pl)) if (gedges and pl) else 0.0
            e = f1(pl, gl)
            scored.append({"edge": e, "v0": (1, vfrac, -(nself + ndangle), ntag),
                           "v1": (1, gmem, vfrac, -(nself + ndangle), ntag)})
        if not scored:
            continue
        n += 1
        sums["mean"] += sum(x["edge"] for x in scored) / len(scored)
        sums["gate_v0"] += max(scored, key=lambda x: x["v0"])["edge"]
        sums["gate_v1"] += max(scored, key=lambda x: x["v1"])["edge"]
        sums["oracle"] += max(scored, key=lambda x: x["edge"])["edge"]

    m = max(n, 1)
    print(f"[kgate-heldout] ids={n} K={len(a.pred)} | mean={sums['mean']/m:.4f} "
          f"gate_v0={sums['gate_v0']/m:.4f} gate_v1={sums['gate_v1']/m:.4f} "
          f"oracle={sums['oracle']/m:.4f}")


if __name__ == "__main__":
    main()
