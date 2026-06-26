#!/usr/bin/env python
"""XGrammar validity-floor 측정 실험 (zero-GPU; relwork_arch §3b #1 — 설계 실험).

★질문: grammar-constrained decoding(XGrammar/스키마 = `tb_guided_schema.py`의 enum)이
보장하는 "valid JSON-DAG 하한"이 **선별기 D-oracle 분모를 얼마나 안정화**하는가.
floor 부재 시 후보는 두 층에서 샌다:
  tier-1 sig=None  = task_nodes 구조파손 → 후보 **완전 드롭**(분모 축소)
  tier-2 ok=False  = 파싱되나 **도구명 무효**(enum 위반)/self-loop/dangling → 격하
XGrammar(enum 스키마)는 둘 다 원천 차단. 본 실험은 *기존 풀*에서:
  ① 풀별 tier-1/tier-2/valid 비율 (= floor가 메우는 갭의 위치)
  ② 값싼 post-hoc 대안 = **name-snap repair**(무효 도구명을 최근접 valid id로 스냅)가
     tier-2를 얼마나 회수하는가 (= "unguided 생성 + repair-floor"가 guided-at-gen의 다양성
     자해[P-unguided] 없이 분모를 복구할 수 있는지의 1차 증거)
  ③ id당 usable 후보 수(분모) repair 전/후 — 선별 합의의 안정성 지표

floor 비교 2-arm 설계(GPU 필요분은 차기):
  A guided-at-gen (현 AR8=enum 강제) = 100% valid·다양성↓(P-unguided 부검)
  B unguided-gen + name-snap repair-floor = 다양성 보존 + 본 스크립트의 회수율만큼 valid
  ⇒ A vs B를 D-oracle·선별 공식척도로 비교(동일 풀 재생성 시).

Usage: tb_validity_floor.py --tb_dir <TB> --ar_tag tb_dpo2g_mmk --ar_group dpo2g [--out floor.json]
"""
import argparse, difflib, json
from collections import defaultdict
from tb_kgate_select import norm, links_of, f1
from tb_select_official import HM, load_records, sig


def classify(rec, valid):
    """tier-1(None) / tier-2(invalid) / valid 분류 + 무효 도구명 리스트."""
    res = rec.get("result", {})
    nodes = res.get("task_nodes") if isinstance(res, dict) else None
    if not (isinstance(nodes, list) and all(isinstance(x, dict) and "task" in x for x in nodes)):
        return "none", []
    bad = [norm(x.get("task", "")) for x in nodes if norm(x.get("task", "")) not in valid]
    s = sig(rec, valid)
    return ("valid" if s and s[1] else "invalid"), bad


def snap_repair(rec, valid, valid_list, cutoff=0.6):
    """무효 도구명을 최근접 valid id로 스냅(difflib) — repair된 rec 사본 반환."""
    res = rec.get("result", {})
    nodes = res.get("task_nodes")
    if not isinstance(nodes, list):
        return rec
    newnodes = []
    name_map = {}
    for x in nodes:
        if not isinstance(x, dict):
            newnodes.append(x); continue
        nm = norm(x.get("task", ""))
        if nm not in valid:
            cand = difflib.get_close_matches(nm, valid_list, n=1, cutoff=cutoff)
            if cand:
                name_map[x.get("task", "")] = cand[0]
                x = dict(x); x["task"] = cand[0]
        newnodes.append(x)
    # task_links의 source/target도 동일 스냅
    links = res.get("task_links")
    newlinks = links
    if isinstance(links, list):
        newlinks = []
        for e in links:
            if isinstance(e, dict):
                e = dict(e)
                for side in ("source", "target"):
                    v = norm(e.get(side, ""))
                    if v not in valid:
                        c = difflib.get_close_matches(v, valid_list, n=1, cutoff=cutoff)
                        if c:
                            e[side] = c[0]
            newlinks.append(e)
    new = dict(rec); new["result"] = dict(res); new["result"]["task_nodes"] = newnodes
    if isinstance(links, list):
        new["result"]["task_links"] = newlinks
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk")
    ap.add_argument("--ar_group", default="dpo2g")
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--hm", default=None)
    ap.add_argument("--cutoff", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TB, D = a.tb_dir, a.domain
    hm_list = a.hm.split(",") if a.hm else HM
    valid = {norm(t["id"]) for t in json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]}
    valid_list = sorted(valid)

    pools = []
    for k in range(8):
        pools.append((f"{a.ar_group}{k}",
                      load_records(f"{TB}/{D}_sub500/predictions/{a.ar_tag}{k}.json")))
    for m in hm_list:
        try:
            pools.append((m, load_records(f"{TB}/{D}_sub500_eval_{m}/predictions/{m}.json")))
        except FileNotFoundError:
            print(f"[warn] hetero pool missing: {m}")

    rows = []
    per_id_before = defaultdict(int)
    per_id_after = defaultdict(int)
    for name, pool in pools:
        n = none = inv = val = rec_recovered = 0
        for rid, rec in pool.items():
            cls, bad = classify(rec, valid)
            n += 1
            if cls == "valid":
                val += 1
                per_id_before[rid] += 1
                per_id_after[rid] += 1
            elif cls == "invalid":
                inv += 1
                rep = snap_repair(rec, valid, valid_list, a.cutoff)
                s = sig(rep, valid)
                if s and s[1]:
                    rec_recovered += 1
                    per_id_after[rid] += 1
            else:  # none (구조파손) — post-hoc snap 불가, XGrammar-at-gen만 차단
                none += 1
        rows.append({"pool": name, "n": n, "valid": val, "invalid_names": inv,
                     "struct_broken": none,
                     "valid_rate": round(val / max(n, 1), 3),
                     "snap_recovered": rec_recovered,
                     "valid_rate_after_snap": round((val + rec_recovered) / max(n, 1), 3)})

    ids = set(per_id_before) | set(per_id_after)
    mean_before = sum(per_id_before.values()) / max(len(ids), 1)
    mean_after = sum(per_id_after.values()) / max(len(ids), 1)
    print(f"{'pool':>14} {'n':>5} {'valid%':>7} {'inv-name':>9} {'struct✗':>8} "
          f"{'snap→ok':>8} {'valid%+snap':>11}")
    for r in rows:
        print(f"{r['pool']:>14} {r['n']:>5} {r['valid_rate']*100:>6.1f} {r['invalid_names']:>9} "
              f"{r['struct_broken']:>8} {r['snap_recovered']:>8} {r['valid_rate_after_snap']*100:>10.1f}")
    tot_n = sum(r["n"] for r in rows)
    tot_val = sum(r["valid"] for r in rows)
    tot_inv = sum(r["invalid_names"] for r in rows)
    tot_none = sum(r["struct_broken"] for r in rows)
    tot_rec = sum(r["snap_recovered"] for r in rows)
    print(f"[TOTAL] n={tot_n} valid={tot_val}({tot_val/max(tot_n,1)*100:.1f}%) "
          f"inv-name={tot_inv} struct-broken={tot_none} "
          f"snap-recovered={tot_rec}/{tot_inv} "
          f"valid_after_snap={(tot_val+tot_rec)/max(tot_n,1)*100:.1f}%")
    print(f"[D-oracle 분모] id당 usable 후보 평균: before={mean_before:.2f} "
          f"after_snap={mean_after:.2f} (+{mean_after-mean_before:.2f}) "
          f"= validity-floor가 안정화하는 선별 분모")
    if a.out:
        json.dump({"rows": rows, "mean_cand_before": mean_before,
                   "mean_cand_after": mean_after}, open(a.out, "w"), indent=1)
        print(f"[floor] wrote {a.out}")


if __name__ == "__main__":
    main()
