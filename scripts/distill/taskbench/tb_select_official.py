#!/usr/bin/env python
"""N2: 이종-풀 선별(validity-필터 + proposer-가중 MBR)을 공식 척도로 확정 (§8.9b 후속).

AR8+H6 풀에서 per-id 후보를 선별해 inference.py-호환 pred 파일로 조립 →
tb_build_eval --pred_file 로 공식 link/node F1 채점. 통제 = k0 단일(동일 파이프라인).

Usage: tb_select_official.py --tb_dir <TB> --out <selected.json>
"""
import argparse, json
from collections import Counter
from tb_kgate_select import norm, links_of, f1

HM = ["qwen3b", "qwen14b", "qwen3_4b", "qwen3_14b", "tb_lodo_hf", "tb_lodo_daily"]


def load_records(path):
    out = {}
    for l in open(path):
        d = json.loads(l)
        out[d["id"]] = d
    return out


def sig(rec, valid):
    res = rec.get("result", {})
    nodes = res.get("task_nodes") if isinstance(res, dict) else None
    if not (isinstance(nodes, list) and all(isinstance(x, dict) and "task" in x for x in nodes)):
        return None
    pl, _, nself, ndangle = links_of(nodes)
    names = [norm(x.get("task", "")) for x in nodes]
    ok = all(x in valid for x in names) and nself == 0 and ndangle == 0
    return pl, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior_beta", type=float, default=0.0,
                    help="SEL-1 (SELECTOR_DESIGN §2): Smoothie-식 label-free proposer prior"
                         " 가중 w=(1/cnt)*prior^beta. 0=v1(미가중) 동작 보존")
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk",
                    help="AR8 풀 파일 접두 (predictions/{ar_tag}{0..7}.json) — "
                         "v3g 등 다른 K8 풀로 교체용")
    ap.add_argument("--ar_group", default="dpo2g", help="AR8 풀의 proposer 그룹명")
    ap.add_argument("--ar_group_per_slot", action="store_true",
                    help="P-lora: AR8 슬롯 8개가 서로 다른 어댑터 = 각 독립 그룹"
                         "(prior-1표가 8종 이종성 살림; 같은 정책 K샘플이면 OFF)")
    ap.add_argument("--extra", action="append", default=[],
                    help="추가 proposer 'name=path.json' (예: Track-B 32B/72B preds — "
                         "그룹명=name, 풀 확장)")
    ap.add_argument("--domain", default="data_multimedia",
                    help="둘째-기판(⑷) 일반화: tool_desc/예측 경로의 도메인 디렉토리")
    ap.add_argument("--hm", default=None,
                    help="hetero 모델 목록 콤마구분 (기본=MM HM 리스트; hf 등은 명시)")
    ap.add_argument("--no_hm", action="store_true", help="H6 hetero 풀 제외 (순수 AR풀만)")
    a = ap.parse_args()
    TB = a.tb_dir
    D = a.domain
    hm_list = [] if a.no_hm else (a.hm.split(",") if a.hm else HM)
    valid = {norm(t["id"]) for t in json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]}

    pools, groups = [], []
    for k in range(8):
        pools.append(load_records(f"{TB}/{D}_sub500/predictions/{a.ar_tag}{k}.json"))
        groups.append(f"{a.ar_group}{k}" if a.ar_group_per_slot else a.ar_group)
    for m in hm_list:
        pools.append(load_records(f"{TB}/{D}_sub500_eval_{m}/predictions/{m}.json"))
        groups.append(m)
    for ex in a.extra:
        name, path = ex.split("=", 1)
        pools.append(load_records(path))
        groups.append(name)

    ids = sorted(set.intersection(*[set(p) for p in pools[:8]]))

    # SEL-1 prior (gold-free): 전 id 평균의 [그룹 후보 vs 타-그룹 후보 합의 f1]
    prior = {}
    if a.prior_beta:
        asum, an = {}, {}
        for i in ids:
            cands = []
            for p, g in zip(pools, groups):
                rec = p.get(i)
                s = sig(rec, valid) if rec is not None else None
                if s is not None:
                    cands.append((g, s[0]))
            for gj, plj in cands:
                others = [c for c in cands if c[0] != gj]
                if not others:
                    continue
                v = sum(f1(plj, pl2) for _, pl2 in others) / len(others)
                asum[gj] = asum.get(gj, 0.0) + v
                an[gj] = an.get(gj, 0) + 1
        prior = {g: asum[g] / an[g] for g in asum}
        print("[prior]", {k: round(v, 3) for k, v in sorted(prior.items())})

    n_sel_hetero = 0
    with open(a.out, "w") as wf:
        for i in ids:
            cands = []
            for p, g in zip(pools, groups):
                rec = p.get(i)
                if rec is None:
                    continue
                s = sig(rec, valid)
                if s is None:
                    continue
                cands.append((rec, g, s[0], s[1]))
            if not cands:
                wf.write(json.dumps(pools[0][i]) + "\n")
                continue
            flt = [c for c in cands if c[3]]
            use = flt if flt else cands
            cnt = Counter(g for _, g, _, _ in use)
            w = [(1.0 / cnt[g]) * (prior.get(g, 1.0) ** a.prior_beta)
                 for _, g, _, _ in use]
            best, bu = 0, -1.0
            for j in range(len(use)):
                if len(use) == 1:
                    best = 0
                    break
                num = sum(w[k2] * f1(use[j][2], use[k2][2]) for k2 in range(len(use)) if k2 != j)
                den = sum(w[k2] for k2 in range(len(use)) if k2 != j)
                u = num / den
                if u > bu:
                    bu, best = u, j
            rec, g, _, _ = use[best]
            if g != a.ar_group:
                n_sel_hetero += 1
            wf.write(json.dumps(rec) + "\n")
    print(f"[select_official] ids={len(ids)} hetero-selected={n_sel_hetero} -> {a.out}")


if __name__ == "__main__":
    main()
