#!/usr/bin/env python
"""B1-zero (사용자 발의): 확신도 = "AR 8샘플 중 몇 번 같은 답"(intra-source 빈도) — GPU 없이
기존 전수 궤적에서 직접. B1(logprob)의 zero-GPU 대체이자, "agreement가 정답을 예측하나" 직답.

①진단(핵심): AR그룹(같은 정책 K샘플) 내 plan별 빈도 freq=cluster_size/K. 후보를 freq로 묶어
   gold edge-F1 평균 → freq↑면 정답률↑이면 **agreement=확신=유효 신호**(MBR이 (1/cnt)로 납작히 한 그것).
②선별 테스트: MBR + λ·agree 결합 / AR-mode(최빈 plan) 단독 → 공식 link-F1 vs SEL-1 0.6722/SEL-4 0.6803.

Usage: tb_selfagree.py --tb_dir <TB> --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
  [--out_pred <TBPRED>/tb_selfagree_dpo2g.json --out_mode <TBPRED>/tb_armode_dpo2g.json]
"""
import argparse, json
from collections import Counter, defaultdict
from tb_kgate_select import norm, f1
from tb_select_official import HM, load_records, sig


def load_gold(tb_dir, domain):
    gold = {}
    for l in open(f"{tb_dir}/{domain}/data.json", encoding="utf-8"):
        d = json.loads(l)
        links = d.get("tool_links") or d.get("sampled_links") or []
        if isinstance(links, str):
            links = json.loads(links)
        gold[d["id"]] = {(norm(e["source"]), norm(e["target"])) for e in links
                         if isinstance(e, dict) and "source" in e and "target" in e}
    return gold


def znorm(xs):
    if not xs:
        return xs
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5 or 1.0
    return [(x - m) / sd for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk")
    ap.add_argument("--ar_group", default="dpo2g")
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--hm", default=None)
    ap.add_argument("--prior_beta", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out_pred", default=None, help="MBR+agree 결합 선택본")
    ap.add_argument("--out_mode", default=None, help="AR-mode(최빈) 선택본")
    a = ap.parse_args()
    TB, D = a.tb_dir, a.domain
    hm_list = a.hm.split(",") if a.hm else HM
    valid = {norm(t["id"]) for t in json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]}
    gold = load_gold(TB, D)

    ar = [load_records(f"{TB}/{D}_sub500/predictions/{a.ar_tag}{k}.json") for k in range(8)]
    het = []
    for m in hm_list:
        try:
            het.append((m, load_records(f"{TB}/{D}_sub500_eval_{m}/predictions/{m}.json")))
        except FileNotFoundError:
            pass
    ids = sorted(set.intersection(*[set(p) for p in ar]) & set(gold))

    # ① 진단: AR그룹 내 빈도 -> gold-F1
    freq_f1 = defaultdict(list)      # freq_count(1..8) -> [f1...]
    mode_f1, rand_f1, oracle_in_ar = [], [], []
    K = 8
    for i in ids:
        plans = []  # (frozenset links, f1_vs_gold)
        for p in ar:
            rec = p.get(i)
            s = sig(rec, valid) if rec else None
            if s is None:
                continue
            plans.append((frozenset(s[0]), f1(s[0], gold[i])))
        if not plans:
            continue
        clusters = Counter(pl for pl, _ in plans)
        for pl, fv in plans:
            freq_f1[clusters[pl]].append(fv)
        # AR-mode = 최빈 cluster의 f1 (동률=첫째)
        top = clusters.most_common(1)[0][0]
        mode_f1.append(dict((pl, fv) for pl, fv in plans)[top])
        rand_f1.append(sum(fv for _, fv in plans) / len(plans))  # 평균=무작위 1샘플 기대
        oracle_in_ar.append(max(fv for _, fv in plans))

    print("=== ① 진단: AR 8샘플 내 빈도(k/8) → gold edge-F1 평균 ===")
    print(f"{'freq k/8':>9} {'n_cand':>7} {'mean_F1':>8}")
    for k in range(1, K + 1):
        v = freq_f1.get(k, [])
        if v:
            print(f"{k:>7}/8 {len(v):>7} {sum(v)/len(v):>8.3f}")
    n = len(mode_f1)
    print(f"[요약] AR-mode(최빈) 평균F1={sum(mode_f1)/max(n,1):.3f} vs "
          f"무작위1샘플 기대={sum(rand_f1)/max(n,1):.3f} vs "
          f"AR-oracle(8중 최선)={sum(oracle_in_ar)/max(n,1):.3f}  (n={n})")
    print(f"[해석] mode > 무작위 면 = '많이 나온 답이 더 맞다'=agreement가 확신 신호 (B1 가설 zero-GPU 검증)")

    # ② 선별: MBR + λ·agree (전 풀) → 예측본 (out_pred); AR-mode 단독 (out_mode)
    if a.out_pred or a.out_mode:
        pools = ar + [h[1] for h in het]
        groups = [a.ar_group] * 8 + [h[0] for h in het]
        # SEL-1 prior
        asum, an = {}, {}
        for i in ids:
            cs = [(g, sig(p.get(i), valid)) for p, g in zip(pools, groups)
                  if p.get(i) and sig(p.get(i), valid)]
            for gj, sj in cs:
                others = [s for g2, s in cs if g2 != gj]
                if others:
                    asum[gj] = asum.get(gj, 0.0) + sum(f1(sj[0], s2[0]) for s2 in others)/len(others)
                    an[gj] = an.get(gj, 0) + 1
        prior = {g: asum[g]/an[g] for g in asum}
        wf = open(a.out_pred, "w") if a.out_pred else None
        wm = open(a.out_mode, "w") if a.out_mode else None
        for i in ids:
            cands = []
            for p, g in zip(pools, groups):
                rec = p.get(i)
                s = sig(rec, valid) if rec else None
                if s is None:
                    continue
                cands.append((rec, g, s[0], s[1]))
            use = [c for c in cands if c[3]] or cands
            if not use:
                if wf: wf.write(json.dumps(ar[0][i]) + "\n")
                if wm: wm.write(json.dumps(ar[0][i]) + "\n")
                continue
            # agree = 같은 그룹 내 동일-plan 빈도 / 그룹크기
            gsize = Counter(c[1] for c in use)
            gplan = defaultdict(Counter)
            for c in use:
                gplan[c[1]][frozenset(c[2])] += 1
            agree = [gplan[c[1]][frozenset(c[2])] / gsize[c[1]] for c in use]
            cnt = gsize
            w = [(1.0/cnt[c[1]]) * (prior.get(c[1], 1.0) ** a.prior_beta) for c in use]
            mbr = []
            for j in range(len(use)):
                num = sum(w[k]*f1(use[j][2], use[k][2]) for k in range(len(use)) if k != j)
                den = sum(w[k] for k in range(len(use)) if k != j)
                mbr.append(num/den if den else 0.0)
            if wf:
                zm, za = znorm(mbr), znorm(agree)
                comb = [zm[j] + a.lam*za[j] for j in range(len(use))]
                wf.write(json.dumps(use[max(range(len(use)), key=lambda j: comb[j])][0]) + "\n")
            if wm:
                # AR-mode 단독: AR그룹 최빈 plan (없으면 MBR)
                ar_c = [c for c in use if c[1] == a.ar_group]
                if ar_c:
                    pc = Counter(frozenset(c[2]) for c in ar_c)
                    toppl = pc.most_common(1)[0][0]
                    pick = next(c for c in ar_c if frozenset(c[2]) == toppl)
                else:
                    pick = use[max(range(len(use)), key=lambda j: mbr[j])]
                wm.write(json.dumps(pick[0]) + "\n")
        if wf: wf.close()
        if wm: wm.close()
        print(f"[선별] wrote " + (a.out_pred or "") + " " + (a.out_mode or ""))


if __name__ == "__main__":
    main()
