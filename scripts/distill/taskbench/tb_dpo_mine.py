#!/usr/bin/env python
"""L2: mine early-closure DPO pairs from RFT rollout .all dumps (census 누락축 처방).

Pair per prompt: chosen = best-reward COMPLETE sample (n_nodes >= gold_n, reward >= min_r);
rejected = an EARLY-CLOSED sample (n_nodes < gold_n) with reward <= chosen - margin.
Output = {"prompt", "chosen", "rejected"} jsonl for scripts/distill/sopbench/dpo_train.py.

--balance (v2, §9.6 단방향-overshoot 처방): chosen = best-reward EXACT-length sample
(n_nodes == gold_n, reward >= min_r); rejected = worst short (n < gold_n) AND worst long
(n > gold_n), 각각 별도 쌍으로 — 양방향 길이-이탈을 대칭 페널티.

--mode structure (D1, 2026-06-12): [같은 노드수·둘 다 어휘청정·파싱정상, edge차 ≥ s_margin]
  쌍 — 차이=배선뿐 → 구조축만 학습 (v1 길이-혼입 교훈의 통제 적용).
--mode cost (D2): [edge 동률(±eps)·둘 다 edge≥c_floor·어휘청정, rejected가 노드 +1↑]
  쌍 — 같은 정답성에 군더더기 노드 → parsimony(비용)축만 학습.

Usage:
  python tb_dpo_mine.py --all winners_rft2_mm.jsonl.all --sft_jsonl train_lodo_mm.jsonl \
      --out dpo_earlyclose.jsonl --min_r 0.8 --margin 0.05 [--balance] \
      [--mode structure|cost --tool_desc td1.json --tool_desc td2.json]
"""
import argparse, json, re

from tb_kgate_select import norm, links_of, f1


def parse_nodes(text):
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
    return len(n) if isinstance(n, list) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True)
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_r", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--balance", action="store_true",
                    help="chosen=exact gold length; reject both short AND long")
    ap.add_argument("--mode", default="length", choices=["length", "structure", "cost"])
    ap.add_argument("--tool_desc", action="append", default=None)
    ap.add_argument("--s_margin", type=float, default=0.4, help="structure: edge차 최소")
    ap.add_argument("--s_floor", type=float, default=0.6, help="structure: chosen edge 최소")
    ap.add_argument("--c_eps", type=float, default=0.05, help="cost: edge 동률 허용폭")
    ap.add_argument("--c_floor", type=float, default=0.5, help="cost: 양쪽 edge 최소")
    a = ap.parse_args()

    valid_tools = set()
    for td in (a.tool_desc or []):
        valid_tools |= {norm(t["id"]) for t in json.load(open(td))["nodes"]}

    meta = {}  # id -> (prompt, gold_n, gold_links)
    for l in open(a.sft_jsonl):
        d = json.loads(l)
        g = json.loads(d["messages"][1]["content"])
        gl, _, _, _ = links_of(g["task_nodes"])
        meta[d["meta"]["id"]] = (d["messages"][0]["content"], len(g["task_nodes"]), gl)

    if a.mode in ("structure", "cost"):
        assert valid_tools, "--mode structure/cost는 --tool_desc 필수 (어휘-혼입 차단)"
        n_pairs = 0
        stats = {"no_meta": 0, "no_pair": 0}
        with open(a.out, "w") as wf:
            for l in open(a.all):
                d = json.loads(l)
                if d["id"] not in meta:
                    stats["no_meta"] += 1
                    continue
                prompt, gold_n, gl = meta[d["id"]]
                cands = []
                for s in d["samples"]:
                    t = s["text"].strip()
                    if t.startswith("```"):
                        t = re.sub(r"^```(json)?\s*|\s*```$", "", t, flags=re.S)
                    try:
                        obj = json.loads(t)
                    except Exception:
                        continue
                    nodes = obj.get("task_nodes") if isinstance(obj, dict) else None
                    if not (isinstance(nodes, list) and nodes and
                            all(isinstance(x, dict) and "task" in x for x in nodes)):
                        continue
                    names = [norm(x["task"]) for x in nodes]
                    if any(x not in valid_tools for x in names):
                        continue  # 어휘-혼입 차단: 양쪽 모두 청정만
                    pl, _, _, _ = links_of(nodes)
                    cands.append({"n": len(nodes), "edge": f1(pl, gl), "text": s["text"]})
                best = None
                if a.mode == "structure":
                    # 같은 노드수 그룹 내에서 edge차 최대 쌍 (차이=배선뿐)
                    for i in range(len(cands)):
                        for j in range(len(cands)):
                            ci, cj = cands[i], cands[j]
                            if ci["n"] != cj["n"]:
                                continue
                            diff = ci["edge"] - cj["edge"]
                            if diff >= a.s_margin and ci["edge"] >= a.s_floor:
                                if best is None or diff > best[0]:
                                    best = (diff, ci["text"], cj["text"])
                else:  # cost: edge 동률, rejected가 더 김 (군더더기)
                    for i in range(len(cands)):
                        for j in range(len(cands)):
                            ci, cj = cands[i], cands[j]
                            if (abs(ci["edge"] - cj["edge"]) <= a.c_eps
                                    and min(ci["edge"], cj["edge"]) >= a.c_floor
                                    and cj["n"] >= ci["n"] + 1):
                                gain = cj["n"] - ci["n"]
                                if best is None or gain > best[0]:
                                    best = (gain, ci["text"], cj["text"])
                if best is None:
                    stats["no_pair"] += 1
                    continue
                wf.write(json.dumps({"prompt": prompt, "chosen": best[1],
                                     "rejected": best[2]}) + "\n")
                n_pairs += 1
        print(f"[dpo-mine:{a.mode}] pairs={n_pairs} skip={stats}")
        return

    n_pairs = 0
    stats = {"no_meta": 0, "no_complete": 0, "no_short": 0,
             "pairs_short": 0, "pairs_long": 0}
    with open(a.out, "w") as wf:
        for l in open(a.all):
            d = json.loads(l)
            if d["id"] not in meta:
                stats["no_meta"] += 1
                continue
            prompt, gold_n, _gl = meta[d["id"]]
            best_c = None  # (reward, text)
            worst_s = None
            worst_l = None
            for s in d["samples"]:
                n = parse_nodes(s["text"])
                if n is None:
                    continue
                ok_c = (n == gold_n) if a.balance else (n >= gold_n)
                if ok_c and s["reward"] >= a.min_r:
                    if best_c is None or s["reward"] > best_c[0]:
                        best_c = (s["reward"], s["text"])
                if n < gold_n:
                    if worst_s is None or s["reward"] < worst_s[0]:
                        worst_s = (s["reward"], s["text"])
                elif n > gold_n:
                    if worst_l is None or s["reward"] < worst_l[0]:
                        worst_l = (s["reward"], s["text"])
            if best_c is None:
                stats["no_complete"] += 1
                continue
            rejects = [("pairs_short", worst_s)]
            if a.balance:
                rejects.append(("pairs_long", worst_l))
            wrote = False
            for key, rej in rejects:
                if rej is None or best_c[0] - rej[0] < a.margin:
                    continue
                wf.write(json.dumps({"prompt": prompt, "chosen": best_c[1],
                                     "rejected": rej[1]}) + "\n")
                stats[key] += 1
                n_pairs += 1
                wrote = True
            if not wrote:
                stats["no_short"] += 1
    print(f"[dpo-mine] pairs={n_pairs} skip={stats}")


if __name__ == "__main__":
    main()
