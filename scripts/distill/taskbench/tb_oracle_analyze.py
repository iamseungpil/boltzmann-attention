#!/usr/bin/env python
"""oracle 엣지-단위 분석 (zero-GPU) — plan-atomic oracle의 결함을 넘어 생성기/선택기/검증기 역할 분리.

기존 oracle = max(단일후보 F1) = "계획 통째 고르기" 상한 → 정답이 후보들에 *부품(엣지)별로 흩어진*
경우를 과소평가(needle/selectable/gold-limited 분류가 다 어긋난 근원). 엣지 단위로 재정의:
  - gold 엣지 커버리지 = 풀의 어떤 후보든 그 엣지를 냈나 (생성기 한계: <1이면 그 엣지는 복원 불가)
  - 조립-oracle = (풀에 존재하는 gold 엣지)만 모은 F1 = 완벽 엣지-검증기가 있을 때의 진짜 천장
  - gold 엣지별 독립소스 수 = 선택기 실현성(맞는 엣지가 다중-소스면 합의로 집힘)
  - 오답 엣지 distractor 부하 = 검증기 난이도(맞는/틀린 엣지 구별 부담)

Usage: tb_oracle_analyze.py --tb_dir <TB> --ar_tag tb_dpo2g_mmk --ar_group dpo2g [--selected <f>]
"""
import argparse, json
from collections import defaultdict, Counter
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk")
    ap.add_argument("--ar_group", default="dpo2g")
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--hm", default=None)
    ap.add_argument("--selected", default=None, help="best-stack 선택본(있으면 selected F1도)")
    a = ap.parse_args()
    TB, D = a.tb_dir, a.domain
    hm_list = a.hm.split(",") if a.hm else HM
    valid = {norm(t["id"]) for t in json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]}
    gold = load_gold(TB, D)

    pools, groups = [], []
    for k in range(8):
        pools.append(load_records(f"{TB}/{D}_sub500/predictions/{a.ar_tag}{k}.json"))
        groups.append(a.ar_group)
    for m in hm_list:
        try:
            pools.append(load_records(f"{TB}/{D}_sub500_eval_{m}/predictions/{m}.json"))
            groups.append(m)
        except FileNotFoundError:
            pass
    selected = load_records(a.selected) if a.selected else None
    ids = sorted(set.intersection(*[set(p) for p in pools[:8]]) & set(gold))

    cov_sum = asm_sum = best_sum = sel_sum = 0.0
    gen_limited = 0          # 커버리지<1 = gold 엣지 일부가 풀에 아예 없음
    asm_beats_best = 0       # 조립 천장 > 단일최대 (선택기-조립 여지)
    gold_edge_redund = Counter()   # gold 엣지가 몇 개 독립소스서 나왔나
    wrong_edge_redund = Counter()  # 오답 엣지의 소스 수 (검증기 변별성)
    n_gold_edges = n_wrong_edges = 0
    distractor_per_task = []       # task별 오답 엣지 종류 수
    n = 0
    for i in ids:
        G = gold[i]
        if not G:
            continue
        n += 1
        edge_src = defaultdict(set)   # edge -> {groups}
        best_single = 0.0
        for p, g in zip(pools, groups):
            rec = p.get(i)
            s = sig(rec, valid) if rec else None
            if s is None:
                continue
            best_single = max(best_single, f1(s[0], G))
            for e in s[0]:
                edge_src[e].add(g)
        present_gold = G & set(edge_src)
        cov = len(present_gold) / len(G)
        asm = f1(present_gold, G)   # 존재하는 gold 엣지만 = precision 1
        cov_sum += cov; asm_sum += asm; best_sum += best_single
        if cov < 1.0 - 1e-9:
            gen_limited += 1
        if asm > best_single + 1e-6:
            asm_beats_best += 1
        for e in present_gold:
            gold_edge_redund[len(edge_src[e])] += 1
            n_gold_edges += 1
        wrong = set(edge_src) - G
        for e in wrong:
            wrong_edge_redund[len(edge_src[e])] += 1
            n_wrong_edges += 1
        distractor_per_task.append(len(wrong))
        if selected:
            ss = sig(selected.get(i, {}), valid)
            sel_sum += f1(ss[0], G) if ss else 0.0

    print(f"[oracle 엣지분석] n={n}")
    print(f"  mean gold-엣지 커버리지 = {cov_sum/n:.3f}  (풀이 gold 엣지를 낸 비율 — 생성기 상한)")
    print(f"  mean 조립-oracle F1     = {asm_sum/n:.3f}  (존재 gold엣지 조립 = 완벽 엣지검증기 천장)")
    print(f"  mean 단일최대 F1(구oracle)= {best_sum/n:.3f}  (계획 통째 고르기 상한)")
    if selected:
        print(f"  mean best-stack 선택 F1  = {sel_sum/n:.3f}  (현 선별기 실측)")
    print(f"  ★조립 − 단일최대 = +{(asm_sum-best_sum)/n:.3f}  (선택기-조립이 통째선택 위로 더 딸 여지)")
    print(f"  생성기-한계 id(커버리지<1) = {gen_limited}/{n} = {gen_limited/n*100:.1f}%  "
          f"(이 비율은 *생성기*로만 줄임 — 없는 엣지)")
    print(f"  조립>단일최대 id = {asm_beats_best}/{n} = {asm_beats_best/n*100:.1f}%  (조립 선택기 여지)")
    print(f"=== gold 엣지 독립소스 redundancy (선택기 실현성) ===")
    for k in sorted(gold_edge_redund):
        print(f"  {k}개 소스: {gold_edge_redund[k]:>5} 엣지 ({gold_edge_redund[k]/max(n_gold_edges,1)*100:.1f}%)")
    import statistics
    print(f"  gold엣지 중 단일소스(redund=1) 비율 = "
          f"{gold_edge_redund.get(1,0)/max(n_gold_edges,1)*100:.1f}% (합의로 못 집는 취약 엣지)")
    print(f"=== ★오답 엣지 redundancy (검증기 변별성 — 정답과 분리되나) ===")
    for k in sorted(set(gold_edge_redund) | set(wrong_edge_redund)):
        gp = gold_edge_redund.get(k, 0) / max(n_gold_edges, 1) * 100
        wp = wrong_edge_redund.get(k, 0) / max(n_wrong_edges, 1) * 100
        print(f"  {k}개 소스: gold {gp:>5.1f}%  vs  wrong {wp:>5.1f}%")
    # 엣지-빈도 임계분리: redund>=t 면 정답으로 채택 시 precision/recall
    print(f"  [엣지-빈도 검증기 가능성] redund≥t 채택 시:")
    for t in (2, 3, 4, 5):
        gkeep = sum(v for k, v in gold_edge_redund.items() if k >= t)
        wkeep = sum(v for k, v in wrong_edge_redund.items() if k >= t)
        prec = gkeep / max(gkeep + wkeep, 1)
        rec = gkeep / max(n_gold_edges, 1)
        print(f"    t={t}: 채택 gold={gkeep} wrong={wkeep} → precision={prec:.3f} recall={rec:.3f}")
    print(f"=== distractor(오답 엣지) 부하 — 검증기 난이도 ===")
    print(f"  task당 오답 엣지 종류 중앙값 = {statistics.median(distractor_per_task):.0f} "
          f"평균 = {sum(distractor_per_task)/n:.1f}  (gold 엣지 평균 {n_gold_edges/n:.1f}개 대비)")


if __name__ == "__main__":
    main()
