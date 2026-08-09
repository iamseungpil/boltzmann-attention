# -*- coding: utf-8 -*-
"""x157 — entrainment 계수 λ 를 **logprob 으로 직접** 재고, 스케일 사다리로 crossover 를 본다.

## 서술 (C87 에너지 토대 위)

    E(y | C) = E_task(y) − Σ_{m∈C} λ · s(y, m),      p(y) ∝ exp(−E/T)

문맥의 각 언급이 그 후보의 에너지를 낮춘다(= logit 을 올린다). 선행이 항의 실재를 잰다
(`2505.09338`: 문맥에 나온 토큰은 **관련성과 무관하게** 확률이 오른다). 우리가 재는 것은
**계수** 다 — 특히 그것이 **모델 크기에 따라 줄어드는가**(= 자기-격리가 스케일이 사는 능력인가).

## 반증 가능한 예측

  P1 누적성 : 앵커 언급을 k 번 늘리면 그 후보의 log-odds 가 대략 **선형**으로 오른다.
              선형이 아니면 이 틀은 틀렸다 — 먼저 적어 둔다([[08]]).
  P3 포화   : 되꽂기 항은 누적 앵커를 압도한다. 어느 k 에서 뒤집히면 그 지점이 λ 비다.
  P4 스케일 : λ 가 모델이 커질수록 **감소**한다 ⇒ 작은 모델 + 외부 격리기 ≥ 큰 모델 단독.

## 방법

측정량 = **첫 토큰 logprob**. 후보 이름의 첫 낱말이 서로 다르므로(`World`·`Navy`·`Hunter`…)
첫 토큰 분포가 후보 분포의 대리로 쓰인다. 샘플링 빈도(n=5)보다 훨씬 촘촘하고 무료다.
앵커 주입은 **구조가 동일한 한 줄을 k 번** 넣는 것뿐이다 — 문구를 바꾸지 않으므로 k 외의
교란이 없다([[57]]: 반복 억제는 횟수가 아니라 인자 변화로 봐야 하고, 여기서는 인자가 k 다).

실행: T2_PROBE_URL=... T2_PROBE_MODEL=... py -3 x157_entrainment_lambda.py [KMAX]
"""
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TASK = "task_099"
GOLD_HEAD = "World"          # gold(`World Blue`)의 첫 낱말 — 채점용일 뿐 프롬프트엔 안 들어간다
ANCHOR = "Navy"              # 앵커로 쓸 오답의 첫 낱말(손님 보유 계좌 = 라이브 실패형)


def first_token_dist(prompt, choices, top=20):
    """★답을 **후보 집합에 강제**하고 첫 토큰 분포를 읽는다.

    자유 생성으로 재려 했더니 무효였다(1차 실행): 첫 토큰이 이름이 아니라 앞말이라
    두 후보 다 top-20 밖으로 나가 logP ≈ −18/−27 바닥이었다. `guided_choice` 로 출력 공간을
    후보로 좁히면 첫 토큰이 곧 후보의 첫 낱말이라 분포가 **후보 분포 그 자체**가 된다.
    """
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 1,
                       "logprobs": True, "top_logprobs": top,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    lp = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {e["token"].strip(): e["logprob"] for e in lp}


def lp_of(dist, head):
    """그 이름의 첫 낱말에 해당하는 토큰들의 확률합 → logprob.

    ⚠1차 실행 버그: 접두 비교가 **1글자 토큰까지** 잡아(`W` → `World`) 없는 확률을 만들었다.
      토큰이 2자 이상이고 이름의 접두일 때만 센다.
    """
    p = 0.0
    for k, v in dist.items():
        kk = k.lower()
        if len(kk) >= 2 and head.lower().startswith(kk):
            p += math.exp(v)
    return math.log(max(p, 1e-12))


def main():
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
    # 후보 집합 = **엔진이 만든 표의 이름들**. 프로브가 목록을 짓지 않는다.
    CHOICES = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]

    # ★앵커는 **실제 궤적**이다 (2026-08-09 자기정정·2차 실행). 합성 한 줄을 6번 넣는 것으로는
    #   꿈쩍도 안 했다(gold 확률 ≈ 1.0 고정) — 자유 생성에서 0/5 를 만드는 실제 문맥은 27 메시지
    #   짜리이고, 내 자극이 그것보다 훨씬 약했다. 자극을 지어내지 말고 **있는 것을 잘라 쓴다**:
    #   궤적 앞에서부터 j 개를 넣고 j 를 늘린다. k 대신 j 가 누적 축이다.
    import x150_choice_ablation as Y
    MSGS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    line = ("Assistant: Looking at what you already have with us, the %s Blue Account "
            "stands out." % ANCHOR)
    hand = ("A separate analysis, working only from the policy constants on record and this "
            "customer's stated situation, selects: %s Blue." % GOLD_HEAD)

    print("model=%s  url=%s" % (MODEL, URL))
    print("%-22s %10s %10s %10s" % ("조건", "logP(gold)", "logP(anchor)", "log-odds"))
    def with_traj(j, tail=""):
        pre = ("Here is a customer-service conversation so far.\n\n"
               + Y.render(MSGS[:j]) + "\n\n") if j else ""
        return pre + base + (("\n\n" + tail) if tail else "")

    xs, ys = [], []
    for k in sorted({0} | {round(len(MSGS) * i / kmax) for i in range(1, kmax + 1)}):
        d = first_token_dist(with_traj(k), CHOICES)
        g, a = lp_of(d, GOLD_HEAD), lp_of(d, ANCHOR)
        xs.append(k)
        ys.append(a - g)
        print("  궤적 %-17s %10.3f %10.3f %10.3f" % ("j=%d/%d" % (k, len(MSGS)), g, a, a - g))
    # 최소제곱 기울기 = λ 의 대리값(언급 1회당 log-odds 이동)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")
    print("\n  P1 누적 기울기 λ̂ = %.3f  (언급 1회당 log-odds, 앵커 쪽이 +)" % slope)

    print("\n  P3 포화 — 같은 j 에 되꽂기를 얹으면")
    for k in (0, len(MSGS) // 2, len(MSGS)):
        d = first_token_dist(with_traj(k, hand), CHOICES)
        g, a = lp_of(d, GOLD_HEAD), lp_of(d, ANCHOR)
        print("    j=%-3d logP(gold)=%7.3f  logP(anchor)=%7.3f  log-odds=%7.3f"
              % (k, g, a, a - g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
