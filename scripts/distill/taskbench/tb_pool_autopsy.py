#!/usr/bin/env python
"""음성 2건(ND 풀-확장·NC v3g-풀)의 궤적 전수 부검 (2026-06-13 사용자 발주, zero-GPU).

B(ND): sel1_b2(무확장 67.22) vs sel1_b2_xpool(+Track-B 67.03) — 선택이 *바뀐* id 전수에
  대해 gold edge-F1로 (개선/악화/동률) 분류 + 악화 id의 새-선택 proposer 귀속 +
  바뀐-선택의 합의-유사도(기존 선택과의 f1 = 중복성 측정).
C(NC): dpo2g-AR8 vs v3g-AR8 — id별 ①AR8 내부 다양성(평균 쌍별 1-f1) ②풀(AR8+H6)
  oracle/mean (gold edge-F1) ③AR8과 H6 간 평균 거리 (이종성). 가설: v3g 다양성↓ →
  oracle은 유지돼도 합의-선별 headroom↓.
⚠️gold = data.json tool_links 기반 edge-F1 (공식 link F1과 스케일 상이 — 진단용 내부 척도).
Usage: tb_pool_autopsy.py --tb_dir <TB> --trackb_dir <repo>/reports/facet_rft_2026/trackb_raw/preds/data_multimedia_sub500
"""
import argparse, itertools, json
from tb_kgate_select import norm, f1
from tb_select_official import HM, load_records, sig
from tb_selector_v2 import load_gold

TRACKB = ["qwen25_32b", "qwen25_32b_guided", "qwen25_72b", "qwen25_72b_guided",
          "qwen3_235b_a22b_int4_guided", "qwen3_32b"]


def links_of_rec(rec, valid):
    s = sig(rec, valid)
    return None if s is None else s[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--trackb_dir", required=True)
    a = ap.parse_args()
    TB = a.tb_dir
    valid = {norm(t["id"]) for t in
             json.load(open(f"{TB}/data_multimedia/tool_desc.json"))["nodes"]}
    gold = load_gold(TB)
    P = f"{TB}/data_multimedia_sub500/predictions"

    # ---------- B: ND 선택-diff 전수 ----------
    base_sel = load_records(f"{P}/tb_sel1_b2.json")
    x_sel = load_records(f"{P}/tb_sel1_b2_xpool.json")
    tb_pools = {n: load_records(f"{a.trackb_dir}/{n}.json") for n in TRACKB}
    chg = imp = wor = tie = 0
    wor_by_grp, imp_by_grp = {}, {}
    dup_sims = []
    for i in sorted(set(base_sel) & set(x_sel) & set(gold)):
        lb = links_of_rec(base_sel[i], valid)
        lx = links_of_rec(x_sel[i], valid)
        if lb is None or lx is None or lb == lx:
            continue
        chg += 1
        fb, fx = f1(lb, gold[i]), f1(lx, gold[i])
        # 새 선택의 proposer 귀속 (Track-B 풀에서 동일 링크 후보 탐색)
        grp = "non-trackb"
        for n, pool in tb_pools.items():
            r = pool.get(i)
            if r is not None and links_of_rec(r, valid) == lx:
                grp = n
                break
        dup_sims.append(f1(lx, lb))
        if fx > fb:
            imp += 1
            imp_by_grp[grp] = imp_by_grp.get(grp, 0) + 1
        elif fx < fb:
            wor += 1
            wor_by_grp[grp] = wor_by_grp.get(grp, 0) + 1
        else:
            tie += 1
    print(f"[B ND-diff] changed={chg} improved={imp} worsened={wor} tie={tie}")
    print(f"  worsened by new-pick group: {dict(sorted(wor_by_grp.items(), key=lambda x: -x[1]))}")
    print(f"  improved by new-pick group: {dict(sorted(imp_by_grp.items(), key=lambda x: -x[1]))}")
    if dup_sims:
        print(f"  새-선택↔기존-선택 평균 링크 f1(중복성) = {sum(dup_sims)/len(dup_sims):.3f}")

    # ---------- C: 풀 다양성/oracle 분해 ----------
    for tag, grp in (("tb_dpo2g_mmk", "dpo2g"), ("tb_v3g_mmk", "v3g")):
        ar = [load_records(f"{P}/{tag}{k}.json") for k in range(8)]
        hm = [load_records(f"{TB}/data_multimedia_sub500_eval_{m}/predictions/{m}.json")
              for m in HM]
        ids = sorted(set.intersection(*[set(p) for p in ar]) & set(gold))
        div_ar, dist_ar_hm, oracle, omean, ar_mean, parsed8 = [], [], [], [], [], 0
        for i in ids:
            la = [links_of_rec(p[i], valid) for p in ar if p.get(i) is not None]
            la = [x for x in la if x is not None]
            lh = [links_of_rec(p.get(i), valid) for p in hm if p.get(i) is not None]
            lh = [x for x in lh if x is not None]
            if len(la) >= 2:
                pairs = [1 - f1(x, y) for x, y in itertools.combinations(la, 2)]
                div_ar.append(sum(pairs) / len(pairs))
            if la and lh:
                ds = [1 - f1(x, y) for x in la for y in lh]
                dist_ar_hm.append(sum(ds) / len(ds))
            allc = la + lh
            if allc:
                fs = [f1(x, gold[i]) for x in allc]
                oracle.append(max(fs))
                omean.append(sum(fs) / len(fs))
            if la:
                ar_mean.append(sum(f1(x, gold[i]) for x in la) / len(la))
        n = len(ids)
        print(f"[C {grp}] ids={n} AR8-내부다양성(1-f1)={sum(div_ar)/len(div_ar):.3f} "
              f"AR↔H6거리={sum(dist_ar_hm)/len(dist_ar_hm):.3f} "
              f"AR8-mean={sum(ar_mean)/len(ar_mean):.3f} "
              f"풀oracle={sum(oracle)/len(oracle):.3f} 풀mean={sum(omean)/len(omean):.3f}")


if __name__ == "__main__":
    main()
