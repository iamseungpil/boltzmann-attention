#!/usr/bin/env python
"""통합풀 음성(0.626<0.680)의 "다양성×품질" 곱-가정 전수 부검 (2026-06-13, zero-GPU).

질문: P-lora 추가가 왜 best-stack(dpo2g+H6)을 떨어뜨렸나? 3원인 분리:
  (A) mean 희석    — 품질 낮은 후보가 평균만 낮춤 (수동)
  (B) 회수율 붕괴  — 선별기가 P-lora 노이즈에 헷갈려 (sel-mean)/(oracle-mean) ↓ (능동 오염)
  (C) 선택 장악    — P-lora 다수표가 MBR 합의를 자기쪽으로 끌어 dpo2g 정답을 밀어냄
분해: 풀 A(dpo2g+H6=14) vs B(+P-lora8=22)에서 mean·oracle·sel·recovery + 선택 그룹분포
  + 그룹별 선택-edge + A→B 선택전환(P-lora로 간 것·edge 하락) 전수.
gold=tool_links edge-F1(진단 내부척도). prior-MBR(β2) 선택 재현(tb_select_official 동형).
Usage: tb_unified_autopsy.py --tb_dir <TB>
"""
import argparse, json
from collections import Counter
from tb_kgate_select import norm, f1
from tb_select_official import HM, load_records, sig
from tb_selector_v2 import load_gold


def build_pool(TB, valid, include_plora):
    """returns groups: list of (group_name, records_dict)."""
    P = f"{TB}/data_multimedia_sub500/predictions"
    pool = [("dpo2g", load_records(f"{P}/tb_dpo2g_mmk{k}.json")) for k in range(8)]
    # dpo2g 8개는 같은 그룹명 → 1 proposer (K샘플)
    if include_plora:
        for k in range(8):
            pool.append((f"plora{k}", load_records(f"{P}/tb_dl_{k}.json")))
    for m in HM:
        pool.append((m, load_records(f"{TB}/data_multimedia_sub500_eval_{m}/predictions/{m}.json")))
    return pool


def analyze(TB, valid, gold, include_plora, tag):
    pool = build_pool(TB, valid, include_plora)
    # ids = dpo2g 8개 교집합 ∩ gold
    ids = sorted(set.intersection(*[set(r) for _, r in pool[:8]]) & set(gold))
    # per-id 후보수집
    per_id = {}
    for i in ids:
        cands = []
        for gname, recs in pool:
            rec = recs.get(i)
            if rec is None:
                continue
            s = sig(rec, valid)
            if s is None:
                continue
            cands.append((gname, s[0], s[1], f1(s[0], gold[i])))  # group, links, ok, edge
        if cands:
            per_id[i] = cands
    # prior (그룹별 타그룹 합의)
    asum, an = {}, {}
    for cands in per_id.values():
        for g, pl, _, _ in cands:
            others = [c for c in cands if c[0] != g]
            if others:
                v = sum(f1(pl, c[1]) for c in others) / len(others)
                asum[g] = asum.get(g, 0.0) + v
                an[g] = an.get(g, 0) + 1
    prior = {g: asum[g] / an[g] for g in asum}
    # SEL-1 선택 (prior-가중 MBR, validity 필터)
    sel_f1, mean_f1, oracle_f1 = [], [], []
    sel_group, picks = {}, {}
    for i, cands in per_id.items():
        flt = [c for c in cands if c[2]]
        use = flt if flt else cands
        cnt = Counter(c[0] for c in use)
        w = [(1.0 / cnt[c[0]]) * (prior.get(c[0], 0.5) ** 2) for c in use]
        links = [c[1] for c in use]
        best, bu = 0, -1.0
        for j in range(len(use)):
            num = sum(w[k] * f1(links[j], links[k]) for k in range(len(use)) if k != j)
            den = sum(w[k] for k in range(len(use)) if k != j)
            u = num / den if den else 0.0
            if u > bu:
                bu, best = u, j
        gsel = use[best][0]
        gsel_canon = "plora" if gsel.startswith("plora") else (
            "dpo2g" if gsel == "dpo2g" else "H6")
        sel_group[i] = gsel_canon
        sel_f1.append(use[best][3])
        picks[i] = use[best][3]
        mean_f1.append(sum(c[3] for c in cands) / len(cands))
        oracle_f1.append(max(c[3] for c in cands))
    n = len(per_id)
    m, o, s = sum(mean_f1) / n, sum(oracle_f1) / n, sum(sel_f1) / n
    rec = (s - m) / (o - m) if o > m else float("nan")
    print(f"\n=== {tag}: ids={n} groups={len(set(g for c in per_id.values() for g,_,_,_ in c))}")
    print(f"  mean={m:.4f} oracle={o:.4f} sel(SEL-1)={s:.4f} 회수율={rec:.1%}")
    gd = Counter(sel_group.values())
    print(f"  선택 그룹분포: {dict(gd)}")
    # 그룹별 선택-edge (선택됐을 때 정답률)
    for gc in ("dpo2g", "plora", "H6"):
        es = [picks[i] for i in per_id if sel_group[i] == gc]
        if es:
            print(f"    {gc} 선택 {len(es)}건 평균 edge={sum(es)/len(es):.4f}")
    print(f"  prior: dpo2g={prior.get('dpo2g',0):.3f} "
          f"plora평균={sum(prior.get(f'plora{k}',0) for k in range(8))/8:.3f} "
          f"H6평균={sum(prior.get(m_,0) for m_ in HM)/len(HM):.3f}")
    return {"per_id": per_id, "sel_group": sel_group, "picks": picks, "m": m, "o": o, "s": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    a = ap.parse_args()
    TB = a.tb_dir
    valid = {norm(t["id"]) for t in
             json.load(open(f"{TB}/data_multimedia/tool_desc.json"))["nodes"]}
    gold = load_gold(TB)

    A = analyze(TB, valid, gold, False, "풀A = dpo2g+H6 (14, best-stack)")
    B = analyze(TB, valid, gold, True, "풀B = dpo2g+P-lora+H6 (22, 통합)")

    # A→B 선택전환 전수
    common = [i for i in A["picks"] if i in B["picks"]]
    to_plora = same = up = down = down_to_plora = 0
    for i in common:
        ga, gb = A["sel_group"][i], B["sel_group"][i]
        fa, fb = A["picks"][i], B["picks"][i]
        if ga == gb and abs(fa - fb) < 1e-9:
            same += 1
        if gb == "plora" and ga != "plora":
            to_plora += 1
            if fb < fa:
                down_to_plora += 1
        if fb > fa + 1e-9:
            up += 1
        elif fb < fa - 1e-9:
            down += 1
    print(f"\n=== [A→B 선택전환] common={len(common)} 동일={same} "
          f"개선={up} 악화={down}")
    print(f"  ★P-lora로 전환={to_plora} 중 edge-하락={down_to_plora} "
          f"(= P-lora 다수표가 정답 밀어낸 직접 증거)")
    # 원인 귀속
    dm, do, ds = B["m"] - A["m"], B["o"] - A["o"], B["s"] - A["s"]
    print(f"\n=== [곱-가정 분해] Δmean={dm:+.4f} Δoracle={do:+.4f} Δsel={ds:+.4f}")
    recA = (A["s"]-A["m"])/(A["o"]-A["m"]); recB = (B["s"]-B["m"])/(B["o"]-B["m"])
    print(f"  회수율 A={recA:.1%} → B={recB:.1%} (Δ={recB-recA:+.1%})")
    print(f"  해석: Δmean<0=품질희석(수동) / Δ회수율<0=선별기오염(능동) / "
          f"P-lora전환-하락 큼=합의장악")


if __name__ == "__main__":
    main()
