#!/usr/bin/env python
"""SEL-1~3 (SELECTOR_DESIGN.md §2 — 0원, 기존 AR8+H6 rollout 재분석).

arms (내부 척도 = gold edge-F1; gold는 평가에만, 선별은 전부 gold-free):
  mean        풀 평균 (1-shot 기대값)
  v1          validity-필터 + proposer-1/cnt 가중 MBR  (§8.9b 0.753 재현 sanity)
  sel1        v1 + Smoothie-식 proposer prior 가중 (label-free 전역 합의 품질, w=(1/cnt)*prior^beta)
  sel2        sel1 + soft-approval (graph-membership 연속 점수 lam 합성 — veto 위 보조투표)
  oracle      best-of-pool 천장
판정(사전등록): F5 회수율 (arm-mean)/(oracle-mean) + arm간 paired bootstrap 95% CI (p2).
SEL-3: 최종 arm의 margin(1위-2위 utility 갭) 기반 risk-coverage 곡선 (gold는 곡선 평가에만).

Usage: tb_selector_v2.py --tb_dir /home/woori/scratch/JARVIS_tb/taskbench \
         [--beta 1.0] [--lam 0.1] [--boot 2000]
"""
import argparse, json, random, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tb_kgate_select import norm, links_of, f1  # noqa: E402
from tb_select_official import HM, load_records, sig  # noqa: E402


def load_gold(tb_dir):
    gold = {}
    for l in open(f"{tb_dir}/data_multimedia/data.json", encoding="utf-8"):
        d = json.loads(l)
        links = d.get("tool_links") or d.get("sampled_links") or []
        if isinstance(links, str):  # data.json은 필드를 JSON-문자열로 이중 인코딩
            links = json.loads(links)
        gl = {(norm(e["source"]), norm(e["target"])) for e in links
              if isinstance(e, dict) and "source" in e and "target" in e}
        gold[d["id"]] = gl
    return gold


def mbr_pick(use, w, soft=None, lam=0.0):
    """가중 MBR + (옵션) soft-approval 합성. returns (best_idx, u_sorted_desc)."""
    us = []
    for j in range(len(use)):
        if len(use) == 1:
            us.append(0.0)
            continue
        num = sum(w[k] * f1(use[j][2], use[k][2]) for k in range(len(use)) if k != j)
        den = sum(w[k] for k in range(len(use)) if k != j)
        u = num / den if den else 0.0
        if soft is not None and lam:
            u += lam * soft[j]
        us.append(u)
    best = max(range(len(use)), key=lambda j: us[j])
    return best, sorted(us, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    TB = a.tb_dir
    rng = random.Random(a.seed)

    valid = {norm(t["id"]) for t in
             json.load(open(f"{TB}/data_multimedia/tool_desc.json"))["nodes"]}
    g = json.load(open(f"{TB}/data_multimedia/graph_desc.json"))
    gedges = {(norm(l["source"]), norm(l["target"]))
              for l in g.get("links", g.get("edges", []))
              if isinstance(l, dict) and "source" in l}
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
    print(f"[selector-v2] ids={len(ids)} pool={len(pools)} "
          f"(dpo2g x8 + {len(HM)} hetero)")

    # 진단: 풀별 sig 성공/validity/링크-보유/edge>0 비율 (포맷 드랍 검출)
    for p, grp in zip(pools, groups):
        tot = ok = haslink = pos = 0
        for i in ids[:200]:
            rec = p.get(i)
            if rec is None:
                continue
            tot += 1
            s = sig(rec, valid)
            if s is None:
                continue
            ok += 1
            pl, _ = s
            if pl:
                haslink += 1
                if f1(pl, gold[i]) > 0:
                    pos += 1
        print(f"  [diag] {grp:13s} rec={tot} sig_ok={ok} has_links={haslink} edge>0={pos}")

    # 후보 수집 (use-셋 = validity 하드필터 적용 후, §8.9b v1과 동일)
    per_id = {}
    for i in ids:
        cands = []
        for p, grp in zip(pools, groups):
            rec = p.get(i)
            if rec is None:
                continue
            s = sig(rec, valid)
            if s is None:
                continue
            pl, ok = s
            edge = f1(pl, gold[i])
            gmem = (sum(l in gedges for l in pl) / len(pl)) if pl else 0.0
            cands.append((rec, grp, pl, ok, edge, gmem))
        if not cands:
            continue
        flt = [c for c in cands if c[3]]
        per_id[i] = (cands, flt if flt else cands)

    # SEL-1: Smoothie-식 전역 proposer prior (label-free) —
    # prior_g = 전 id 평균의 [g 후보 vs 타-그룹 후보 평균 합의(f1)]
    agree_sum, agree_n = {}, {}
    for i, (cands, _) in per_id.items():
        for j, (_, gj, plj, _, _, _) in enumerate(cands):
            others = [c for c in cands if c[1] != gj]
            if not others:
                continue
            s = sum(f1(plj, c[2]) for c in others) / len(others)
            agree_sum[gj] = agree_sum.get(gj, 0.0) + s
            agree_n[gj] = agree_n.get(gj, 0) + 1
    prior = {g_: agree_sum[g_] / agree_n[g_] for g_ in agree_sum}
    print("[prior] " + " ".join(f"{k}={v:.3f}" for k, v in sorted(prior.items())))

    # arm별 per-id edge-F1
    arms = {"mean": {}, "v1": {}, "sel1": {}, "sel2": {}, "oracle": {}}
    margins = {}
    from collections import Counter
    for i, (cands, use) in per_id.items():
        arms["mean"][i] = sum(c[4] for c in cands) / len(cands)
        arms["oracle"][i] = max(c[4] for c in cands)
        cnt = Counter(c[1] for c in use)
        w_v1 = [1.0 / cnt[c[1]] for c in use]
        b, _ = mbr_pick(use, w_v1)
        arms["v1"][i] = use[b][4]
        w_s1 = [(1.0 / cnt[c[1]]) * (prior.get(c[1], 0.5) ** a.beta) for c in use]
        b, _ = mbr_pick(use, w_s1)
        arms["sel1"][i] = use[b][4]
        soft = [c[5] for c in use]
        b, us = mbr_pick(use, w_s1, soft=soft, lam=a.lam)
        arms["sel2"][i] = use[b][4]
        # confidence = 승자의 합의 수준 u1 (갭 u1-u2 아님 — 만장일치면 갭 0이 되는 역전 결함;
        # 단일후보 id는 합의 증거 0 = 최저 confidence)
        margins[i] = us[0] if len(us) > 1 else 0.0

    n = len(per_id)
    mean_of = {k: sum(v.values()) / n for k, v in arms.items()}
    rec_of = {k: (mean_of[k] - mean_of["mean"]) / (mean_of["oracle"] - mean_of["mean"])
              for k in ("v1", "sel1", "sel2")}
    for k in ("mean", "v1", "sel1", "sel2", "oracle"):
        r = f" recall={rec_of[k]:.1%}" if k in rec_of else ""
        print(f"  {k:7s} edgeF1={mean_of[k]:.4f}{r}")

    # p2: paired bootstrap 95% CI (sel1-v1, sel2-v1, sel2-sel1)
    idl = sorted(per_id)
    for x, y in (("sel1", "v1"), ("sel2", "v1"), ("sel2", "sel1")):
        diffs = []
        for _ in range(a.boot):
            s = [idl[rng.randrange(len(idl))] for _ in range(len(idl))]
            diffs.append(sum(arms[x][i] - arms[y][i] for i in s) / len(s))
        diffs.sort()
        lo, hi = diffs[int(0.025 * a.boot)], diffs[int(0.975 * a.boot)]
        d = mean_of[x] - mean_of[y]
        print(f"  [boot] {x}-{y}: d={d:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] "
              f"{'SIG' if lo > 0 or hi < 0 else 'ns'}")

    # SEL-3: margin 기반 risk-coverage (최종 arm=sel2)
    order = sorted(idl, key=lambda i: -margins[i])
    print("  [SEL-3 risk-coverage on sel2] coverage: risk(1-edgeF1)")
    for cov in (0.2, 0.4, 0.6, 0.8, 1.0):
        m = order[:max(1, int(cov * len(order)))]
        risk = 1 - sum(arms["sel2"][i] for i in m) / len(m)
        print(f"    {cov:.0%}: {risk:.4f}")


if __name__ == "__main__":
    main()
