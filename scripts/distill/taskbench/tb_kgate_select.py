#!/usr/bin/env python
"""L6 lever-② headroom: K-proposal + deterministic gate SELECTION upper bound.

Uses existing RFT rollouts (.all, K=8/prompt, train domains) — zero-GPU.
Per prompt: ①mean single-sample edge-F1 (expected 1-shot) ②gate-selected
(deterministic, gold-free score: parse > valid_frac > no-self/dangling > tag-use)
③oracle best-of-K (true edge-F1 — selection ceiling = Pass@K analog).
Gap(oracle−mean)=selection headroom; gate가 그 갭을 얼마나 회수하나가 판정.

Usage: tb_kgate_select.py --all winners_rft2_mm.jsonl.all --sft_jsonl train_lodo_mm.jsonl \
           --tool_desc <dom>/tool_desc.json [--tool_desc2 ...]
(tool_desc 여러 개 주면 합집합 — train 도메인 2개 커버)
"""
import argparse, json, re


def norm(s):
    return str(s).replace("_", " ").strip()


def parse_result(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        d = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except Exception:
            return None
    n = d.get("task_nodes") if isinstance(d, dict) else None
    if not isinstance(n, list) or any(not (isinstance(x, dict) and "task" in x) for x in n):
        return None
    return n


def links_of(nodes):
    names = [norm(n.get("task", "")) for n in nodes]
    links, nself, ndangle, ntag = set(), 0, 0, 0
    for inx, node in enumerate(nodes):
        for a in (node.get("arguments") or []):
            if isinstance(a, dict):
                a = list(a.values())[0] if a else ""
            if isinstance(a, list):
                a = " ".join(str(x) for x in a)
            if isinstance(a, str) and "<node-" in a:
                ntag += 1
                try:
                    j = int(a[a.index("<node-") + 6:a.index(">")])
                except Exception:
                    ndangle += 1
                    continue
                if j == inx:
                    nself += 1
                elif 0 <= j < len(names):
                    links.add((names[j], names[inx]))
                else:
                    ndangle += 1
    return links, ntag, nself, ndangle


def f1(p, g):
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    i = len(p & g)
    pr, rc = i / len(p), i / len(g)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def load_graph(paths):
    """graph_desc.json들 → 유효 간선 집합 + 도구 타입 맵 (스코어러 v1)."""
    edges, types = set(), {}
    for p in paths or []:
        g = json.load(open(p))
        for l in g.get("links", g.get("edges", [])):
            if isinstance(l, dict) and "source" in l and "target" in l:
                edges.add((norm(l["source"]), norm(l["target"])))
    return edges, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True)
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--tool_desc", action="append", required=True)
    ap.add_argument("--graph_desc", action="append", default=None,
                    help="graph_desc.json (v1 scorer: 링크의 그래프-멤버십 가점)")
    a = ap.parse_args()

    gedges, _ = load_graph(a.graph_desc)
    valid = set()
    for td in a.tool_desc:
        valid |= {norm(t["id"]) for t in json.load(open(td))["nodes"]}
    gold = {}
    for l in open(a.sft_jsonl):
        d = json.loads(l)
        g = json.loads(d["messages"][1]["content"])
        gl, _, _, _ = links_of(g["task_nodes"])
        gold[d["meta"]["id"]] = gl

    sums = {"mean": 0.0, "gate": 0.0, "oracle": 0.0, "reward_pick": 0.0}
    n_prompts = 0
    for l in open(a.all):
        d = json.loads(l)
        if d["id"] not in gold:
            continue
        gl = gold[d["id"]]
        scored = []
        for s in d["samples"]:
            nodes = parse_result(s["text"])
            if nodes is None:
                scored.append({"edge": 0.0, "gate": (-1, 0, 0, 0)})
                continue
            pl, ntag, nself, ndangle = links_of(nodes)
            names = [norm(n["task"]) for n in nodes]
            vfrac = sum(x in valid for x in names) / max(len(names), 1)
            edge = f1(pl, gl)
            # v1: 링크의 도구그래프-멤버십 비율 (gold 아님 — 도메인 그래프 사전)
            gmem = (sum(l in gedges for l in pl) / len(pl)) if (gedges and pl) else 0.0
            # gold-free deterministic gate score (lexicographic; v1=gmem 추가)
            scored.append({"edge": edge, "reward": s.get("reward", 0.0),
                           "gate": (1, gmem, vfrac, -(nself + ndangle), ntag)})
        if not scored:
            continue
        n_prompts += 1
        sums["mean"] += sum(x["edge"] for x in scored) / len(scored)
        sums["gate"] += max(scored, key=lambda x: x["gate"])["edge"]
        sums["oracle"] += max(scored, key=lambda x: x["edge"])["edge"]
        sums["reward_pick"] += max(scored, key=lambda x: x.get("reward", 0.0))["edge"]

    n = max(n_prompts, 1)
    print(f"[kgate] prompts={n_prompts} K~8 | mean(1-shot)={sums['mean']/n:.4f} "
          f"gate-select={sums['gate']/n:.4f} oracle(best-of-K)={sums['oracle']/n:.4f} "
          f"(참고 reward-pick={sums['reward_pick']/n:.4f} — reward는 gold-기반=oracle류)")


if __name__ == "__main__":
    main()
