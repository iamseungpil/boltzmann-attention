#!/usr/bin/env python
"""B1 (SELECTOR §7 후속): self-certainty/logprob 신호로 선별 — C 진단의 selectable~50% 검증.

근거: 우리 풀=같은 base의 LoRA 변종 → cross-candidate logprob이 **비교가능한 희귀 레짐**
(relwork_selector §12, Self-Certainty 2502.18581). judge(SEL-5 음성)와 *다른 신호*.
신호 = p(plan | 생성프롬프트)의 평균 토큰 logprob(forward confidence; echo+completions).
SEL-4와 동일 결합틀: best = argmax[ z(MBR) + lam·z(selfcert) ]. 비교용으로 pure-selfcert도 보고.

selectable(합의지지 있는 oracle를 더 지지받는 틀린답에 밀린 케이스)을 confidence가 구제하는가.

Usage: tb_selfcert_select.py --tb_dir <TB> --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
  --endpoint http://localhost:8001/v1 --served base_model --lam 1.0 --out sc.json
"""
import argparse, json, urllib.request
from collections import Counter
from tb_kgate_select import norm, f1
from tb_select_official import HM, load_records, sig
from tb_diffusion_sample import build_prompt


def selfcert_logprob(endpoint, served, gen_prompt, plan_json):
    """p(plan_json | gen_prompt)의 평균 토큰 logprob (echo, completion 부분만)."""
    full = gen_prompt + plan_json
    payload = {"model": served, "prompt": full, "max_tokens": 0,
               "echo": True, "logprobs": 0, "temperature": 0.0}
    req = urllib.request.Request(endpoint.rstrip("/") + "/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    lp = d["choices"][0]["logprobs"]
    toks, offs, tlp = lp["tokens"], lp.get("text_offset", []), lp["token_logprobs"]
    cut = len(gen_prompt)
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
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk")
    ap.add_argument("--ar_group", default="dpo2g")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--served", default="base_model")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--prior_beta", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_pure", default=None, help="pure-selfcert 선택본 별도 출력")
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--ar_group_per_slot", action="store_true")
    ap.add_argument("--hm", default=None)
    ap.add_argument("--no_hm", action="store_true")
    a = ap.parse_args()
    TB, D = a.tb_dir, a.domain
    hm_list = [] if a.no_hm else (a.hm.split(",") if a.hm else HM)
    tool_list = json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]
    valid = {norm(t["id"]) for t in tool_list}
    tool_string = "# TASK LIST #:\n" + "".join(json.dumps(t) + "\n" for t in tool_list)

    pools, groups = [], []
    for k in range(8):
        pools.append(load_records(f"{TB}/{D}_sub500/predictions/{a.ar_tag}{k}.json"))
        groups.append(f"{a.ar_group}{k}" if a.ar_group_per_slot else a.ar_group)
    for m in hm_list:
        pools.append(load_records(f"{TB}/{D}_sub500_eval_{m}/predictions/{m}.json"))
        groups.append(m)
    ids = sorted(set.intersection(*[set(p) for p in pools[:8]]))

    asum, an = {}, {}
    for i in ids:
        cs = [(g, sig(p.get(i), valid)) for p, g in zip(pools, groups)
              if p.get(i) is not None and sig(p.get(i), valid) is not None]
        for gj, sj in cs:
            others = [s for g2, s in cs if g2 != gj]
            if others:
                v = sum(f1(sj[0], s2[0]) for s2 in others) / len(others)
                asum[gj] = asum.get(gj, 0.0) + v
                an[gj] = an.get(gj, 0) + 1
    prior = {g: asum[g] / an[g] for g in asum}

    n_sc = 0
    wf = open(a.out, "w")
    wfp = open(a.out_pure, "w") if a.out_pure else None
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
            if wfp:
                wfp.write(json.dumps(pools[0][i]) + "\n")
            continue
        flt = [c for c in cands if c[3]]
        use = flt if flt else cands
        if len(use) == 1:
            wf.write(json.dumps(use[0][0]) + "\n")
            if wfp:
                wfp.write(json.dumps(use[0][0]) + "\n")
            continue
        instr = use[0][0].get("user_request", "")
        gp = build_prompt(tool_string, instr)
        cnt = Counter(c[1] for c in use)
        w = [(1.0 / cnt[c[1]]) * (prior.get(c[1], 1.0) ** a.prior_beta) for c in use]
        mbr = []
        for j in range(len(use)):
            num = sum(w[k] * f1(use[j][2], use[k][2]) for k in range(len(use)) if k != j)
            den = sum(w[k] for k in range(len(use)) if k != j)
            mbr.append(num / den if den else 0.0)
        sc = []
        for c in use:
            plan_json = json.dumps(c[0].get("result", {}), ensure_ascii=False)
            sc.append(selfcert_logprob(a.endpoint, a.served, gp, plan_json))
        n_sc += 1
        zm, zs = znorm(mbr), znorm(sc)
        comb = [zm[j] + a.lam * zs[j] for j in range(len(use))]
        wf.write(json.dumps(use[max(range(len(use)), key=lambda j: comb[j])][0]) + "\n")
        if wfp:
            wfp.write(json.dumps(use[max(range(len(use)), key=lambda j: sc[j])][0]) + "\n")
    wf.close()
    if wfp:
        wfp.close()
    print(f"[selfcert] ids={len(ids)} scored={n_sc} lam={a.lam} -> {a.out}"
          + (f" + pure {a.out_pure}" if a.out_pure else ""))


if __name__ == "__main__":
    main()
