#!/usr/bin/env python
"""SEL-4 (SELECTOR_DESIGN §2): 7B reverse-likelihood Reviewer 합성.

Coder-Reviewer (arXiv:2211.16490): p(plan|instr) 단독 재랭킹은 퇴화-해 선호 →
Reviewer 점수 p(instruction|plan)를 곱/합성해 교정. 우리 제약(결정론·gold-free·7B·
후보당 1 pass) 전부 충족. MBR(SEL-1) utility와 직교 신호 = 소수-정답 구제.

구현: 후보 plan(직렬화)을 prompt, instruction을 continuation으로 두고 7B에 통과,
instruction 토큰들의 평균 logprob = reviewer score. echo+prompt_logprobs(vLLM
completions). 최종 = z-norm(MBR_util) + lam * z-norm(reviewer) 로 후보 재선택.

Usage: tb_reviewer_select.py --tb_dir <TB> --ar_tag tb_v3g_mmk --ar_group v3g \
  --endpoint http://localhost:8001/v1 --served base_model --lam 1.0 --out sel4.json
"""
import argparse, json, math, urllib.request
from collections import Counter
from tb_kgate_select import norm, links_of, f1
from tb_select_official import HM, load_records, sig


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


def reviewer_logprob(endpoint, served, plan_str, instruction):
    """p(instruction | plan)의 평균 토큰 logprob (echo+prompt_logprobs)."""
    prompt = (f"Tool-use plan: {plan_str}\n"
              f"User request that this plan fulfills: {instruction}")
    payload = {"model": served, "prompt": prompt, "max_tokens": 0,
               "echo": True, "logprobs": 0, "temperature": 0.0}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    lp = d["choices"][0]["logprobs"]
    toks, offs = lp["tokens"], lp.get("text_offset", [])
    tlp = lp["token_logprobs"]
    # instruction 부분 토큰만 = prompt에서 instruction 시작 char offset 이후
    cut = prompt.rindex(instruction)
    vals = [tlp[i] for i in range(len(toks))
            if tlp[i] is not None and i < len(offs) and offs[i] >= cut]
    return sum(vals) / max(len(vals), 1) if vals else -99.0


def znorm(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5 or 1.0
    return [(x - m) / sd for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ar_tag", default="tb_v3g_mmk")
    ap.add_argument("--ar_group", default="v3g")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--served", default="base_model")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--prior_beta", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--ar_group_per_slot", action="store_true",
                    help="P-lora: AR8 슬롯 8개가 서로 다른 어댑터 = 각 독립 그룹")
    ap.add_argument("--hm", default=None, help="hetero 모델 콤마구분 (기본=MM HM)")
    ap.add_argument("--no_hm", action="store_true", help="H6 hetero 풀 제외 (순수 AR풀만)")
    ap.add_argument("--extra", action="append", default=[], help="'name=path.json' 추가 proposer")
    a = ap.parse_args()
    TB = a.tb_dir
    D = a.domain
    hm_list = [] if a.no_hm else (a.hm.split(",") if a.hm else HM)
    valid = {norm(t["id"]) for t in
             json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]}
    # instruction(user_request)은 pred 레코드에 동봉됨 — 후보에서 직접 읽음

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
    # SEL-1 prior (재사용)
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

    n_rev = 0
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
            rev = []
            for c in use:
                ps = serialize_plan(c[0])
                rev.append(reviewer_logprob(a.endpoint, a.served, ps or "", instr)
                           if ps else -99.0)
            n_rev += 1
            zm, zr = znorm(mbr), znorm(rev)
            comb = [zm[j] + a.lam * zr[j] for j in range(len(use))]
            best = max(range(len(use)), key=lambda j: comb[j])
            wf.write(json.dumps(use[best][0]) + "\n")
    print(f"[sel4] ids={len(ids)} reviewer-scored={n_rev} -> {a.out}")


if __name__ == "__main__":
    main()
