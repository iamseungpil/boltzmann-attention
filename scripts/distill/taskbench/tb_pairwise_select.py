#!/usr/bin/env python
"""SEL-5 (SELECTOR_DESIGN §2, 큐 ⑶): MBR-shortlist(top-K) + 7B pairwise judge 토너먼트.

SEL-1(prior 가중 prop-MBR)로 shortlist(top-K by MBR)를 만든 뒤, 잔여 oracle 갭을
**직접 쌍대비교**로 공략. judge = 같은 base 7B(chat)에 "어느 plan이 요청을 더 잘 푸나"를
물어 순서-편향 제거(양방향 일치 시만 1승) round-robin → 최다승(동점=MBR 서열). 제약 충족:
gold-free·결정론(temp0)·≤7B/on-prem·구조 출력. LLM-Blender PairRanker(2306.02561)·MAV
계보(shortlist 압축=O(n^2) 비용 정당). MBR이 못 고른 소수-정답을 pairwise가 구제하는가.

Usage (SEL-4 드라이버 서빙 재사용 — base_model @ :8001):
  tb_pairwise_select.py --tb_dir <TB> --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
    --endpoint http://localhost:8001/v1 --served base_model --shortlist 3 --out sel5.json
"""
import argparse, json, urllib.request
from collections import Counter
from tb_kgate_select import norm, f1
from tb_select_official import HM, load_records, sig

JUDGE_SYS = ("You compare two tool-use plans proposed for a user request. Pick the plan that more "
             "correctly and completely fulfills the request: correct tool selection, right ordering, "
             "and appropriate arguments. Answer with ONLY a single character, 'A' or 'B'. No explanation.")


def serialize_plan(rec):
    res = rec.get("result", {})
    nodes = res.get("task_nodes") if isinstance(res, dict) else None
    if not nodes:
        return None
    parts = []
    for n in nodes:
        a = n.get("arguments", [])
        parts.append(f"{n.get('task', '')}({', '.join(map(str, a))})")
    return " -> ".join(parts)


def judge_pair(endpoint, served, instr, plan_a, plan_b):
    """A/B 중 어느 plan이 더 나은가 (단일 호출). 'A'/'B'/None."""
    usr = (f"User request: {instr}\n\nPlan A: {plan_a}\n\nPlan B: {plan_b}\n\n"
           f"Which plan better fulfills the request? Answer A or B.")
    payload = {"model": served, "temperature": 0.0, "max_tokens": 2,
               "messages": [{"role": "system", "content": JUDGE_SYS},
                            {"role": "user", "content": usr}]}
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    c = txt.strip().upper()[:1] if txt.strip() else ""
    return c if c in ("A", "B") else None


def tournament(endpoint, served, instr, plans):
    """순서-편향 제거 round-robin: (A,B)+(B,A) 일치 시만 1승. 반환=각 plan 승수."""
    n = len(plans)
    wins = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            ab = judge_pair(endpoint, served, instr, plans[i], plans[j])  # i=A j=B
            ba = judge_pair(endpoint, served, instr, plans[j], plans[i])  # j=A i=B
            if ab == "A" and ba == "B":
                wins[i] += 1
            elif ab == "B" and ba == "A":
                wins[j] += 1
            # 불일치/None = 무승부(순서편향 = 표 안 줌)
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk")
    ap.add_argument("--ar_group", default="dpo2g")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--served", default="base_model")
    ap.add_argument("--prior_beta", type=float, default=2.0)
    ap.add_argument("--shortlist", type=int, default=3, help="MBR top-K shortlist 크기 (3~5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--ar_group_per_slot", action="store_true")
    ap.add_argument("--hm", default=None)
    ap.add_argument("--no_hm", action="store_true")
    ap.add_argument("--extra", action="append", default=[])
    a = ap.parse_args()
    TB, D = a.tb_dir, a.domain
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
    # SEL-1 prior (tb_reviewer_select와 동일)
    asum, an = {}, {}
    for i in ids:
        cands = [(g, sig(p.get(i), valid)) for p, g in zip(pools, groups)
                 if p.get(i) is not None and sig(p.get(i), valid) is not None]
        for gj, sj in cands:
            others = [s for g2, s in cands if g2 != gj]
            if others:
                v = sum(f1(sj[0], s2[0]) for s2 in others) / len(others)
                asum[gj] = asum.get(gj, 0.0) + v
                an[gj] = an.get(gj, 0) + 1
    prior = {g: asum[g] / an[g] for g in asum}

    n_judged = n_flip = 0  # n_flip = pairwise가 MBR-top을 뒤집은 횟수
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
            if len(use) == 1:
                wf.write(json.dumps(use[0][0]) + "\n")
                continue
            instr = use[0][0].get("user_request", "")
            cnt = Counter(c[1] for c in use)
            w = [(1.0 / cnt[c[1]]) * (prior.get(c[1], 1.0) ** a.prior_beta) for c in use]
            mbr = []
            for j in range(len(use)):
                num = sum(w[k] * f1(use[j][2], use[k][2]) for k in range(len(use)) if k != j)
                den = sum(w[k] for k in range(len(use)) if k != j)
                mbr.append(num / den if den else 0.0)
            order = sorted(range(len(use)), key=lambda j: -mbr[j])
            mbr_top = order[0]
            short = order[:a.shortlist]
            if len(short) == 1:
                wf.write(json.dumps(use[short[0]][0]) + "\n")
                continue
            plans = [serialize_plan(use[j][0]) or "" for j in short]
            wins = tournament(a.endpoint, a.served, instr, plans)
            n_judged += 1
            # 최다승, 동점=shortlist(MBR) 서열 우선
            best_local = max(range(len(short)), key=lambda t: (wins[t], -t))
            best = short[best_local]
            if best != mbr_top:
                n_flip += 1
            wf.write(json.dumps(use[best][0]) + "\n")
    print(f"[sel5] ids={len(ids)} judged={n_judged} pairwise-flipped-MBR-top={n_flip} "
          f"shortlist={a.shortlist} -> {a.out}")


if __name__ == "__main__":
    main()
