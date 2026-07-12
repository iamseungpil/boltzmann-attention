#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eref_horizon.py — E-HORIZON: per-step 정확도의 scale-완만 vs verify-급격 실증 (무료·로컬 사다리).

설계 = THINKING_HORIZON_LEVER_SURVIVAL_DESIGN §2·§4(E-HORIZON-1/2). 2509.09677 running-sum 복제.
질문: "짧은 단계로 정확히 verify하면 긴 horizon 생존?" → per-step을 scale이 *느리게* vs verify가 *급격히* 개선.

task (2509.09677 동형): 사전 D(단어→[-99,99]) 문맥제공 + 매 턴 K개 키 지정 → S_t = S_{t-1}+Σ D[k].
  지식·계획 다 줌 = 순수 *실행* 격리. gold = 결정론 누적합(자가채점).

arms (동일 task·동일 모델·차이는 오직 컨텍스트 조립):
  base   : 모델이 매 턴 S_t 산출. 이전 턴 (모델의) S_{t-1}이 문맥에 누적 → self-conditioning.
  verify : 매 턴 후 결정론 재계산으로 gold S_t를 문맥에 주입(모델 오류를 진입 前 교정) → 루프 절단.
  detect : (E-HORIZON-2 기전격리) 오류를 *플래그만* 하고 교정 안 함(오류는 여전히 문맥 진입) → verify와 대조.

지표 (per-run, H턴):
  first_fail  : 처음 틀린 턴 (없으면 H+1) = horizon
  acc_step    : 턴별 정확도(전 턴 = gold 대비 이번 S_t 정확) 평균 = per-step 정확도
  acc_cond    : self-conditioning 곡선 (오류 후 정확도 vs 오류 전)
  survived    : 전 H턴 정확(=완주)

usage:
  py -3 eref_horizon.py --dry                              # task 생성 검증만
  py -3 eref_horizon.py --base http://localhost:8141/v1 --model Qwen/Qwen2.5-7B-Instruct \
        --arms base,verify,detect --H 40 --K 2 --runs 20 --out horiz_7b.jsonl
표준 stdlib만. 커밋=배포용(리모트 git pull).
"""
import argparse, json, random, re, sys
from concurrent.futures import ThreadPoolExecutor

def make_plan(H, K, seed):
    # ★직접-증분(dict-free): 매 턴 K개 정수[1,20]를 직접 제공 → lookup 없음.
    #   단일-스텝 = 러닝합에 작은 정수 더하기(중형 모델 near-perfect) → 실패를 *순수 누적/
    #   self-conditioning*(러닝총합을 턴 넘어 유지)으로 격리. 2509.09677 "near-perfect single-turn" 동형.
    rng = random.Random(seed ^ 0x5bd1e995)
    return [[rng.randint(1, 20) for _ in range(K)] for _ in range(H)]

HDR = ("You are maintaining a running total. It starts at S = 0.\n"
       "Each turn I give you some numbers. Add them to the running total: S_new = S_old + sum(numbers).\n"
       "Answer EACH turn with ONLY the new running total as 'S = <integer>' on its own line. No other text.")

def parse_int(txt):
    if not isinstance(txt, str):
        return None
    m = re.findall(r"S\s*=\s*(-?\d+)", txt)
    if m:
        return int(m[-1])
    m = re.findall(r"-?\d+", txt.replace(",", ""))
    return int(m[-1]) if m else None

def call_server(base, model, messages, timeout=240, max_tokens=600, think=False):
    import urllib.request
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "messages": messages}
    if think is False:
        # Qwen3: thinking 끄기(비-thinking arm). 비지원 모델은 무시됨.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif think is True:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    return out["choices"][0]["message"]["content"]

def run_dialog(base, model, plan, arm, think=False, max_tokens=600):
    """한 run(H턴) 실행. arm ∈ {base,verify,detect}. 반환 = 턴별 기록 리스트."""
    msgs = [{"role": "system", "content": HDR}]
    gold_S = 0
    rec = []
    for t, nums in enumerate(plan):
        step = sum(nums)
        gold_S += step
        user = {"role": "user", "content": "Turn %d: add %s" % (t + 1, ", ".join(str(x) for x in nums))}
        msgs.append(user)
        try:
            ans = call_server(base, model, msgs, max_tokens=max_tokens, think=think)
        except Exception as e:
            rec.append({"t": t + 1, "err": type(e).__name__}); break
        got = parse_int(ans)
        correct = (got == gold_S)
        rec.append({"t": t + 1, "gold": gold_S, "got": got, "correct": bool(correct),
                    "prev_gold": gold_S - step})
        # 문맥 조립 = arm 차이 (핵심)
        if arm == "verify":
            # 결정론 재계산으로 gold를 '확정 상태'로 주입 → 모델 오류 진입 前 교정(self-cond 절단)
            msgs.append({"role": "assistant", "content": "S = %d" % gold_S})
        elif arm == "detect":
            # 오류 탐지는 하되 교정 안 함: 모델값을 그대로 문맥에 (오류가 문맥 진입 = self-cond 유지)
            msgs.append({"role": "assistant", "content": "S = %d" % (got if got is not None else gold_S)})
        else:  # base
            msgs.append({"role": "assistant", "content": ans[:60]})
    return rec

def summarize(rec, H):
    steps = [r for r in rec if "correct" in r]
    if not steps:
        return {"acc_step": 0.0, "first_fail": 0, "survived": False, "n": 0}
    accs = [1 if r["correct"] else 0 for r in steps]
    ff = next((r["t"] for r in steps if not r["correct"]), H + 1)
    # self-conditioning: 첫 오류 전 정확도 vs 후 정확도
    pre = [a for r, a in zip(steps, accs) if r["t"] < ff]
    post = [a for r, a in zip(steps, accs) if r["t"] > ff]
    return {"acc_step": sum(accs) / len(accs), "first_fail": ff, "survived": ff > H,
            "n": len(steps), "acc_pre": (sum(pre) / len(pre)) if pre else None,
            "acc_post": (sum(post) / len(post)) if post else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arms", default="base,verify")
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--nwords", type=int, default=20)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--think", default="off", choices=["off", "on"])
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    think = {"off": False, "on": True}[a.think]
    tasks = []
    for i in range(a.runs):
        plan = make_plan(a.H, a.K, a.seed + i)
        tasks.append((i, plan))
    if a.dry:
        i, plan = tasks[0]
        print("plan[0:3]:", plan[:3])
        g = 0
        for t, nums in enumerate(plan[:5]):
            g += sum(nums)
            print("  turn %d: add %s  gold_S=%d" % (t + 1, nums, g))
        print("arms=%s H=%d K=%d runs=%d think=%s" % (arms, a.H, a.K, a.runs, a.think))
        return
    jobs = [(arm, i, plan) for arm in arms for (i, plan) in tasks]
    def run_one(job):
        arm, i, plan = job
        rec = run_dialog(a.base, a.model, plan, arm, think=think, max_tokens=a.max_tokens)
        s = summarize(rec, a.H)
        return {"arm": arm, "run": i, "model": a.model, "think": a.think, "H": a.H, "K": a.K,
                **s, "rec": rec}
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(run_one, jobs)):
            rows.append(r)
            print("  [%d/%d] arm=%s run=%d acc_step=%.3f first_fail=%d survived=%s"
                  % (k + 1, len(jobs), r["arm"], r["run"], r["acc_step"], r["first_fail"], r["survived"]),
                  flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    # 집계
    print("\n== E-HORIZON 집계 (model=%s think=%s H=%d) ==" % (a.model, a.think, a.H))
    print("arm      runs  acc_step  median_horizon  survived%%  acc_pre  acc_post(self-cond)")
    byarm = {}
    for r in rows:
        byarm.setdefault(r["arm"], []).append(r)
    for arm in arms:
        rs = byarm.get(arm, [])
        if not rs:
            continue
        n = len(rs)
        acc = sum(r["acc_step"] for r in rs) / n
        ffs = sorted(r["first_fail"] for r in rs)
        medff = ffs[n // 2]
        surv = sum(1 for r in rs if r["survived"]) / n
        pres = [r["acc_pre"] for r in rs if r["acc_pre"] is not None]
        posts = [r["acc_post"] for r in rs if r["acc_post"] is not None]
        pre = sum(pres) / len(pres) if pres else float("nan")
        post = sum(posts) / len(posts) if posts else float("nan")
        print("%-8s %4d  %.3f     %6.1f          %.2f      %.3f    %.3f"
              % (arm, n, acc, medff, surv, pre, post))
    print("★per-step: base=scale-완만 / verify=급격(→1.0)·survived=horizon 생존. acc_post<acc_pre=self-conditioning.")

if __name__ == "__main__":
    main()
