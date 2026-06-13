#!/usr/bin/env python
"""다양성-생성 실험 분석 (SELECTOR_DESIGN 큐 ⑸ — matched-K8 정책별 풀 비교 + 회귀).

정책별(K=8 풀): 다양성(평균 쌍별 1-F1)·oracle·mean·SEL-1-lite 선별(동일-정책 풀이라
proposer-prior 무의미 — 균등 MBR)·per-id 선별이득. 사전등록 (i) 회귀: per-id 다양성이
per-id 이득을 예측 (pooled OLS 기울기 + id-bootstrap 95% CI) (ii) 정책 순위.
선별 pred 파일도 기록 → 공식 eval은 드라이버가 tb_build_eval로.
Usage: tb_divgen_analyze.py --tb_dir <TB> --policy P-temp=<glob> --policy P-unguided=<glob> ...
"""
import argparse, glob, itertools, json, random
from tb_kgate_select import norm, f1
from tb_select_official import load_records, sig
from tb_selector_v2 import load_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--policy", action="append", required=True, help="name=fileglob")
    ap.add_argument("--out_prefix", required=True, help="선별 pred 출력 경로 접두")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    TB = a.tb_dir
    valid = {norm(t["id"]) for t in
             json.load(open(f"{TB}/data_multimedia/tool_desc.json"))["nodes"]}
    gold = load_gold(TB)
    rng = random.Random(42)

    points = []  # (policy, id, diversity, gain)
    for pol in a.policy:
        name, pat = pol.split("=", 1)
        files = sorted(glob.glob(pat))[:8]
        if len(files) < 4:
            print(f"[{name}] files={len(files)} <4 — SKIP")
            continue
        pools = [load_records(f) for f in files]
        ids = sorted(set.intersection(*[set(p) for p in pools]) & set(gold))
        div_l, gain_l, orc_l, mean_l, sel_l = [], [], [], [], []
        with open(f"{a.out_prefix}_{name}.json", "w") as wf:
            for i in ids:
                cands = []
                for p in pools:
                    rec = p.get(i)
                    s = sig(rec, valid) if rec is not None else None
                    if s is not None:
                        cands.append((rec, s[0], s[1]))
                if len(cands) < 2:
                    if cands:
                        wf.write(json.dumps(cands[0][0]) + "\n")
                    continue
                flt = [c for c in cands if c[2]]
                use = flt if flt else cands
                if len(use) < 2:  # validity 필터 후 단일 후보 = 다양성 미정의 → 선별 trivial
                    wf.write(json.dumps(use[0][0]) + "\n")
                    continue
                links = [c[1] for c in use]
                pairs = [1 - f1(x, y) for x, y in itertools.combinations(links, 2)]
                d = sum(pairs) / len(pairs)
                fs = [f1(l, gold[i]) for l in links]
                m, o = sum(fs) / len(fs), max(fs)
                # 균등 MBR
                best, bu = 0, -1.0
                for j in range(len(use)):
                    u = sum(f1(links[j], links[k]) for k in range(len(use)) if k != j) \
                        / max(len(use) - 1, 1)
                    if u > bu:
                        bu, best = u, j
                sel = fs[best]
                wf.write(json.dumps(use[best][0]) + "\n")
                div_l.append(d); gain_l.append(sel - m)
                orc_l.append(o); mean_l.append(m); sel_l.append(sel)
                points.append((name, d, sel - m))
        n = len(div_l)
        print(f"[{name}] ids={n} files={len(files)} 다양성={sum(div_l)/n:.4f} "
              f"mean={sum(mean_l)/n:.4f} oracle={sum(orc_l)/n:.4f} "
              f"sel={sum(sel_l)/n:.4f} 이득={sum(gain_l)/n:+.4f}")

    # pooled 회귀: gain ~ diversity (id-bootstrap CI)
    if len(points) > 50:
        def slope(pts):
            xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs) or 1e-9
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        b = slope(points)
        bs = []
        for _ in range(a.boot):
            s = [points[rng.randrange(len(points))] for _ in range(len(points))]
            bs.append(slope(s))
        bs.sort()
        lo, hi = bs[int(0.025 * a.boot)], bs[int(0.975 * a.boot)]
        print(f"[회귀] gain~diversity slope={b:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] "
              f"{'SIG' if lo > 0 else 'ns'} (n={len(points)})")


if __name__ == "__main__":
    main()
