#!/usr/bin/env python
"""D2(비용-표적 DPO) 기각의 전수 궤적 부검 (TB결과 §8.10 엄밀화).

질문: edge −2.0은 어디서 왔나 — 의도(여분-노드 삭제)의 부작용으로 gold-필요 노드까지
삭제했나(과일반화), 아니면 재배선/형식 등 다른 원인인가. 학습 쌍의 제거-어휘와
추론 시 과잉삭제 어휘의 겹침으로 귀속까지.

Usage: tb_d2_autopsy.py --dir_a <rft2 eval dir> --llm_a tb_rft2_mm \
    --dir_b <dpo_cost eval dir> --llm_b tb_dpo_cost \
    --tool_desc .../tool_desc.json --pairs /home/woori/scratch/tb_rft/dpo_cost.jsonl \
    --out /home/woori/scratch/d2_autopsy.md
"""
import argparse, json
from collections import Counter
from tb_census import f1, load, norm, tag_info


def nodes_links(res, dep="resource"):
    nodes = res.get("task_nodes") if isinstance(res, dict) else None
    if not (isinstance(nodes, list) and all(isinstance(x, dict) and "task" in x for x in nodes)):
        return None, None
    names = Counter(norm(x.get("task", "")) for x in nodes)
    links, _, _, _ = tag_info(nodes)
    return names, links


def plan_nodes(text):
    """학습 쌍의 chosen/rejected 문자열 -> 노드 이름 Counter (방어적 파싱)."""
    if isinstance(text, dict):
        d = text
    else:
        s = str(text)
        i = s.find("{")
        if i < 0:
            return None
        try:
            d = json.loads(s[i:])
        except ValueError:
            return None
    tn = d.get("task_nodes")
    if not isinstance(tn, list):
        return None
    return Counter(norm(x.get("task", "")) for x in tn if isinstance(x, dict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True); ap.add_argument("--llm_a", required=True)
    ap.add_argument("--dir_b", required=True); ap.add_argument("--llm_b", required=True)
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gold_a, preds_a = load(a.dir_a, a.llm_a)
    _, preds_b = load(a.dir_b, a.llm_b)
    ids = sorted(i for i in set(gold_a) & set(preds_a) & set(preds_b)
                 if "task_nodes" in gold_a[i])

    # 학습 쌍 제거-어휘 (rejected − chosen = 쌍이 "지워라"고 가르친 노드)
    train_del, n_pairs_ok = Counter(), 0
    if a.pairs:
        for l in open(a.pairs):
            d = json.loads(l)
            cn, rn = plan_nodes(d.get("chosen")), plan_nodes(d.get("rejected"))
            if cn is None or rn is None:
                continue
            n_pairs_ok += 1
            train_del.update(+(rn - cn))

    buckets = {"improved": [], "worsened": [], "same": []}
    short = {"A": 0, "B": 0}
    deficit = {"A": 0, "B": 0}
    pr = {"A": [0, 0, 0], "B": [0, 0, 0]}  # tp, npred, ngold
    parse_fail = {"A": 0, "B": 0}
    rows = []
    for i in ids:
        gn, gl = nodes_links(gold_a[i])
        an, al = nodes_links(preds_a[i])
        bn, bl = nodes_links(preds_b[i])
        if gn is None:
            continue
        for tag, nn in (("A", an), ("B", bn)):
            if nn is None:
                parse_fail[tag] += 1
                continue
            tp = sum((nn & gn).values())
            pr[tag][0] += tp; pr[tag][1] += sum(nn.values()); pr[tag][2] += sum(gn.values())
            if sum(nn.values()) < sum(gn.values()):
                short[tag] += 1
                deficit[tag] += sum(gn.values()) - sum(nn.values())
        if an is None or bn is None:
            continue
        ea, eb = f1(al, gl), f1(bl, gl)
        d = eb - ea
        bucket = "improved" if d > 0.1 else "worsened" if d < -0.1 else "same"

        removed = +(an - bn)            # A에는 있고 B에서 사라진 노드
        added = +(bn - an)
        needed_rm = +(removed & gn)     # gold-필요인데 삭제 = 손상
        spurious_rm = +(removed - gn)   # gold-밖 여분 삭제 = 의도
        lost_correct_edges = len((al & gl) - bl)
        gained_correct_edges = len((bl & gl) - al)
        # worsened 형태 분류
        shape = ("pure-deletion" if removed and not added else
                 "substitution" if removed and added else
                 "pure-rewire" if not removed and not added and al != bl else
                 "no-change")
        rows.append(dict(id=i, bucket=bucket, ea=ea, eb=eb,
                         needed_rm=needed_rm, spurious_rm=spurious_rm, added=added,
                         lost_ce=lost_correct_edges, gained_ce=gained_correct_edges,
                         shape=shape, gold_n=sum(gn.values())))
        buckets[bucket].append(rows[-1])

    def vocab(rows_, key):
        c = Counter()
        for r in rows_:
            c.update(r[key])
        return c

    with open(a.out, "w") as wf:
        wf.write(f"# D2 autopsy {a.llm_a} -> {a.llm_b} (n={len(rows)})\n\n")
        wf.write(f"## prereg-c (P/R/short/deficit)\n")
        for t in ("A", "B"):
            tp, np_, ng = pr[t]
            wf.write(f"{t}: P={tp/max(np_,1):.4f} R={tp/max(ng,1):.4f} "
                     f"short={short[t]} deficit_nodes={deficit[t]} parse_fail={parse_fail[t]}\n")
        wf.write(f"\n## buckets\n")
        for name, items in buckets.items():
            nr, sr = vocab(items, "needed_rm"), vocab(items, "spurious_rm")
            ad = vocab(items, "added")
            lost = sum(r["lost_ce"] for r in items)
            gained = sum(r["gained_ce"] for r in items)
            shapes = Counter(r["shape"] for r in items)
            wf.write(f"### {name}: {len(items)}  shapes={dict(shapes)}\n")
            wf.write(f"  needed-removed nodes={sum(nr.values())} (ids w/>=1: "
                     f"{sum(1 for r in items if r['needed_rm'])}) | spurious-removed={sum(sr.values())} "
                     f"| added={sum(ad.values())} | correct-edges lost={lost} gained={gained}\n")
            wf.write(f"  top needed-removed: {nr.most_common(8)}\n")
            wf.write(f"  top spurious-removed: {sr.most_common(8)}\n")
        # 과일반화 귀속: 추론 시 needed-removed 어휘가 학습 제거-어휘에 있던 비율
        wf.write(f"\n## training-pair attribution (pairs parsed={n_pairs_ok})\n")
        wf.write(f"train removal vocab top: {train_del.most_common(10)}\n")
        for name in ("worsened", "improved"):
            nr = vocab(buckets[name], "needed_rm" if name == "worsened" else "spurious_rm")
            if not nr:
                continue
            inset = sum(c for x, c in nr.items() if x in train_del)
            wf.write(f"{name} {'needed' if name=='worsened' else 'spurious'}-removed in train-vocab: "
                     f"{inset}/{sum(nr.values())} ({100*inset/max(sum(nr.values()),1):.0f}%)\n")
        # gold 크기별 worsened 분포
        gw = Counter(r["gold_n"] for r in buckets["worsened"])
        wf.write(f"\nworsened by gold size: {dict(sorted(gw.items()))}\n")
    print(open(a.out).read())


if __name__ == "__main__":
    main()
