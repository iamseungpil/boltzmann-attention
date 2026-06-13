#!/usr/bin/env python
"""C: oracle 갭 분해 (zero-GPU) — "0.6803 = 천장"을 선언하기 전 선결 진단.

질문: best-stack(SEL-1+SEL-4) 선택본과 oracle(풀 내 gold edge-F1 최대) 사이 갭의 *얼마가
선별가능*하고 얼마가 *비가역*인가. id별 gap = oracle_f1 - selected_f1 를 3버킷 분해:
  ① gold-limited : oracle_f1 < τ_low  → 최선후보도 나쁨 = gold/과제 한계(선별 무관·비가역)
  ② needle       : oracle를 단 1후보만 달성 → gold-free 선별로 집기 매우 어려움(≈비가역, Stroebl)
  ③ selectable   : oracle를 ≥2후보가 달성(합의 지지 있음)인데 못 고름 → *선별기 개선 여지*
버킷별 (id수, 총 gap 기여 share)로 "선별가능분"을 수치화. selectable 비중이 크면 천장 아님;
needle+gold-limited가 지배하면 천장 주장에 근거.

Usage: tb_gap_decompose.py --tb_dir <TB> --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
         --selected <TBPRED>/tb_sel4_dpo2g.json
"""
import argparse, json
from tb_kgate_select import norm, f1
from tb_select_official import HM, load_records, sig

EPS = 1e-9


def load_gold(tb_dir, domain):
    gold = {}
    for l in open(f"{tb_dir}/{domain}/data.json", encoding="utf-8"):
        d = json.loads(l)
        links = d.get("tool_links") or d.get("sampled_links") or []
        if isinstance(links, str):  # data.json은 필드를 JSON-문자열로 이중 인코딩
            links = json.loads(links)
        gl = {(norm(e["source"]), norm(e["target"])) for e in links
              if isinstance(e, dict) and "source" in e and "target" in e}
        gold[d["id"]] = gl
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk")
    ap.add_argument("--ar_group", default="dpo2g")
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--selected", required=True, help="best-stack 선택본 jsonl (SEL-4 출력)")
    ap.add_argument("--hm", default=None)
    ap.add_argument("--tau_low", type=float, default=0.5, help="gold-limited 판정 oracle_f1 하한")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TB, D = a.tb_dir, a.domain
    hm_list = a.hm.split(",") if a.hm else HM
    valid = {norm(t["id"]) for t in json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]}
    gold = load_gold(TB, D)

    pools = [load_records(f"{TB}/{D}_sub500/predictions/{a.ar_tag}{k}.json") for k in range(8)]
    for m in hm_list:
        try:
            pools.append(load_records(f"{TB}/{D}_sub500_eval_{m}/predictions/{m}.json"))
        except FileNotFoundError:
            pass
    selected = load_records(a.selected)
    ids = sorted(set.intersection(*[set(p) for p in pools[:8]]) & set(gold) & set(selected))

    buckets = {"no_gap": [], "gold_limited": [], "needle": [], "selectable": []}
    sum_oracle = sum_sel = 0.0
    sel_rank_when_gap = []
    for i in ids:
        g = gold[i]
        cand_f1 = []
        for p in pools:
            rec = p.get(i)
            if rec is None:
                continue
            s = sig(rec, valid)
            if s is None:
                continue
            cand_f1.append((f1(s[0], g), s[0]))
        if not cand_f1:
            continue
        oracle = max(c[0] for c in cand_f1)
        n_at = sum(1 for c in cand_f1 if abs(c[0] - oracle) < 1e-6)
        n_distinct_at = len({frozenset(c[1]) for c in cand_f1 if abs(c[0] - oracle) < 1e-6})
        ssig = sig(selected[i], valid)
        sel_f1 = f1(ssig[0], g) if ssig else 0.0
        gap = oracle - sel_f1
        sum_oracle += oracle
        sum_sel += sel_f1
        if gap <= 1e-6:
            buckets["no_gap"].append((i, gap, oracle))
        elif oracle < a.tau_low:
            buckets["gold_limited"].append((i, gap, oracle))
        elif n_distinct_at == 1:
            buckets["needle"].append((i, gap, oracle))
        else:
            buckets["selectable"].append((i, gap, oracle))
        if gap > 1e-6:
            rank = 1 + sum(1 for c in cand_f1 if c[0] > sel_f1 + 1e-6)
            sel_rank_when_gap.append(rank)

    n = len(ids)
    mean_oracle, mean_sel = sum_oracle / max(n, 1), sum_sel / max(n, 1)
    total_gap = mean_oracle - mean_sel
    print(f"[gap] ids={n}  mean_oracle={mean_oracle:.4f}  mean_selected={mean_sel:.4f}  "
          f"mean_gap={total_gap:.4f}")
    print(f"{'bucket':>14} {'n_ids':>6} {'%ids':>6} {'sum_gap':>8} {'%of_total_gap':>13}  설명")
    desc = {"no_gap": "선택=oracle(손실0)", "gold_limited": "oracle<τ=gold한계(비가역)",
            "needle": "oracle 단1후보=gold-free 난(≈비가역)",
            "selectable": "oracle≥2후보=합의지지=선별여지"}
    tot_gap_sum = sum(g for b in buckets.values() for _, g, _ in b)
    for name in ["no_gap", "gold_limited", "needle", "selectable"]:
        bl = buckets[name]
        sg = sum(g for _, g, _ in bl)
        print(f"{name:>14} {len(bl):>6} {len(bl)/max(n,1)*100:>5.1f} {sg/max(n,1):>8.4f} "
              f"{sg/max(tot_gap_sum,1e-9)*100:>12.1f}  {desc[name]}")
    irreducible = sum(g for _, g, _ in buckets["gold_limited"] + buckets["needle"])
    selectable = sum(g for _, g, _ in buckets["selectable"])
    print(f"[분해] 총 gap 중 비가역(gold-limited+needle)={irreducible/max(tot_gap_sum,1e-9)*100:.1f}% "
          f"· 선별가능(selectable)={selectable/max(tot_gap_sum,1e-9)*100:.1f}%")
    if sel_rank_when_gap:
        import statistics
        print(f"[근접도] gap-id에서 선택본의 f1-순위 중앙값={statistics.median(sel_rank_when_gap):.0f} "
              f"(1=oracle·클수록 far-miss; n={len(sel_rank_when_gap)})")
    if a.out:
        json.dump({k: [(i, round(g, 4), round(o, 4)) for i, g, o in v]
                   for k, v in buckets.items()}, open(a.out, "w"), indent=1)
        print(f"[gap] wrote {a.out}")


if __name__ == "__main__":
    main()
