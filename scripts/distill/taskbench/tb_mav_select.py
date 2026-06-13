#!/usr/bin/env python
"""V-1/V-2 (SELECTOR_DESIGN §6): 결정론 축 라이브러리 + MAV-식 validation-선별 집계.

축(후보-단위, 전부 결정론·0원): A1 gmem·A2 valid_frac·A3 -(self+dangle)·
A4 인자-정합(arguments 수 vs tool_desc params — 유사-실행 1보)·A5 DAG-위상(-고립노드).
집계 3형: (i) 이진 승인합산(val-보정 cutpoint=중앙값) (ii) z-선형(val grid 가중)
(iii) lexicographic(음성 통제). val 100 / test 400 (id-층화 seed42 동결).
판정: test gold edge-F1로 SEL-1 베이스라인 대비 (공식 eval은 드라이버 별도).
Usage: tb_mav_select.py --tb_dir <TB>
"""
import argparse, json, random
from collections import Counter
from tb_kgate_select import norm, links_of, f1
from tb_select_official import HM, load_records, sig
from tb_selector_v2 import load_gold


def axes_of(rec, valid, gedges, tooldesc):
    res = rec.get("result", {})
    nodes = res.get("task_nodes") if isinstance(res, dict) else None
    if not nodes:
        return None
    pl, ntag, nself, ndangle = links_of(nodes)
    names = [norm(n.get("task", "")) for n in nodes]
    vfrac = sum(x in valid for x in names) / max(len(names), 1)
    gmem = (sum(l in gedges for l in pl) / len(pl)) if pl else 0.0
    # A4 인자-정합: 노드 인자 수가 tool_desc 파라미터 수 이하/근접 비율
    okarg = 0
    for n in nodes:
        nm = norm(n.get("task", ""))
        spec = tooldesc.get(nm)
        args = n.get("arguments") or []
        if spec is None:
            continue
        np_ = len(spec)
        okarg += 1 if (np_ == 0 or abs(len(args) - np_) <= 1) else 0
    a4 = okarg / max(len(nodes), 1)
    # A5 DAG-위상: 링크에 등장하지 않는 고립 노드 비율 (다중노드 플랜 한정)
    linked = {x for e in pl for x in e}
    iso = sum(1 for x in names if x not in linked)
    a5 = -(iso / max(len(names), 1)) if len(names) > 1 else 0.0
    return {"a1": gmem, "a2": vfrac, "a3": -(nself + ndangle), "a4": a4, "a5": a5,
            "links": pl}


def mbr_util(use_links, w, j):
    num = sum(w[k] * f1(use_links[j], use_links[k]) for k in range(len(use_links)) if k != j)
    den = sum(w[k] for k in range(len(use_links)) if k != j)
    return num / den if den else 0.0


def zs(v):
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5 or 1.0
    return [(x - m) / sd for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--lam", type=float, default=0.15)
    a = ap.parse_args()
    TB = a.tb_dir
    valid = {norm(t["id"]) for t in
             json.load(open(f"{TB}/data_multimedia/tool_desc.json"))["nodes"]}
    g = json.load(open(f"{TB}/data_multimedia/graph_desc.json"))
    gedges = {(norm(l["source"]), norm(l["target"]))
              for l in g.get("links", g.get("edges", []))
              if isinstance(l, dict) and "source" in l and "target" in l}
    td = {norm(t["id"]): (t.get("parameters") or t.get("input-type") or [])
          for t in json.load(open(f"{TB}/data_multimedia/tool_desc.json"))["nodes"]}
    gold = load_gold(TB)

    pools, groups = [], []
    for k in range(8):
        pools.append(load_records(
            f"{TB}/data_multimedia_sub500/predictions/tb_dpo2g_mmk{k}.json"))
        groups.append("dpo2g")
    for m in HM:
        pools.append(load_records(
            f"{TB}/data_multimedia_sub500_eval_{m}/predictions/{m}.json"))
        groups.append(m)
    ids = sorted(set.intersection(*[set(p) for p in pools[:8]]) & set(gold))
    rng = random.Random(42)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    val_ids, test_ids = set(shuffled[:100]), set(shuffled[100:])

    # prior (SEL-1, 전 id — 가중 자체는 gold-free)
    asum, an = {}, {}
    per_id = {}
    for i in ids:
        cands = []
        for p, grp in zip(pools, groups):
            rec = p.get(i)
            if rec is None:
                continue
            ax = axes_of(rec, valid, gedges, td)
            if ax is None:
                continue
            s = sig(rec, valid)
            ok = s[1] if s else False
            cands.append((grp, ax, ok, f1(ax["links"], gold[i])))
        if not cands:
            continue
        per_id[i] = cands
        for grp, ax, _, _ in cands:
            others = [c for c in cands if c[0] != grp]
            if others:
                v = sum(f1(ax["links"], c[1]["links"]) for c in others) / len(others)
                asum[grp] = asum.get(grp, 0.0) + v
                an[grp] = an.get(grp, 0) + 1
    prior = {g_: asum[g_] / an[g_] for g_ in asum}

    AXES = ["a1", "a2", "a3", "a4", "a5"]
    # val-보정 cutpoint = 축별 중앙값 (val 후보 전체)
    valvals = {ax: sorted(c[1][ax] for i in val_ids if i in per_id
                          for c in per_id[i]) for ax in AXES}
    cut = {ax: v[len(v) // 2] for ax, v in valvals.items() if v}

    def select(i, mode, axset):
        cands = per_id[i]
        flt = [c for c in cands if c[2]]
        use = flt if flt else cands
        links = [c[1]["links"] for c in use]
        cnt = Counter(c[0] for c in use)
        w = [(1.0 / cnt[c[0]]) * (prior.get(c[0], 1.0) ** 2) for c in use]
        base_u = [mbr_util(links, w, j) for j in range(len(use))]
        if mode == "sel1":
            sc = base_u
        elif mode == "approve":
            sc = [base_u[j] + a.lam * sum(1 for ax in axset
                                          if use[j][1][ax] >= cut.get(ax, 0))
                  / max(len(axset), 1) for j in range(len(use))]
        elif mode == "zlin":
            zax = {ax: zs([c[1][ax] for c in use]) if len(use) > 1 else [0.0] * len(use)
                   for ax in axset}
            sc = [base_u[j] + a.lam * sum(zax[ax][j] for ax in axset) / max(len(axset), 1)
                  for j in range(len(use))]
        else:  # lex (음성 통제)
            sc = [tuple(use[j][1][ax] for ax in axset) for j in range(len(use))]
        best = max(range(len(use)), key=lambda j: sc[j])
        return use[best][3]

    def mean_f1(idset, mode, axset):
        v = [select(i, mode, axset) for i in idset if i in per_id]
        return sum(v) / len(v)

    base_val = mean_f1(val_ids, "sel1", [])
    print(f"[V-2] val SEL-1 baseline = {base_val:.4f} (val={len(val_ids)} test={len(test_ids)})")
    # greedy 축 선별 (val, zlin 기준)
    chosen = []
    cur = base_val
    for _ in range(len(AXES)):
        gains = [(mean_f1(val_ids, "zlin", chosen + [ax]), ax)
                 for ax in AXES if ax not in chosen]
        gains.sort(reverse=True)
        if gains and gains[0][0] > cur + 1e-4:
            cur = gains[0][0]
            chosen.append(gains[0][1])
        else:
            break
    print(f"[V-2] greedy 채택 축 = {chosen} (val zlin {cur:.4f})")
    for mode in ("sel1", "approve", "zlin", "lex"):
        axset = chosen if chosen else AXES
        tf = mean_f1(test_ids, mode, axset if mode != "sel1" else [])
        print(f"[V-2 test] {mode:8s} axes={axset if mode != 'sel1' else '-'} "
              f"edgeF1={tf:.4f}")


if __name__ == "__main__":
    main()
