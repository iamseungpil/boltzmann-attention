# -*- coding: utf-8 -*-
"""x160 — 배치(placement) crossover 의 e2e 합성 1판 (전부 무료·로컬).

## 무엇을 합성하나 (§6.1 의 측정된 부품 그대로 — 새 부품 없음)

  [P] 배치 arm  : ⒜ 엔진 필터(`eligible_text`) → ⒝ **14B** 가 깨끗한 문맥에서 선택(자유생성)
                  → ② 그 답을 오염된 원 궤적에 되꽂고 **14B** 가 운반. 32B 무개입.
  [B] 32B 단독  : 같은 base 를 오염 궤적 뒤에 실어 32B 자유생성 — 격리기 없음(라이브 동형).
  [I] 32B+격리기: 같은 구조를 32B 로 — 아키텍처 참조선(x154/x155 의 5/5 재현 기대).

crossover 주장 = P ≥ B (그리고 P ≈ I 이면 선택·운반 자리에서 32B 가 불필요).

## 규율

- 되꽂는 답은 **⒝ 가 실제로 낸 것**(greedy)이다 — gold 를 꽂지 않는다([[03b]]·x157 과 다른 점).
- base 는 세 arm 동일(엔진-필터 표 + 궤적-실재 사실 + 고정 질문) — 표 차이 교락 금지(§8.5 #11).
- 채점은 x149 와 같은 substring. 표본 n=5(greedy 1 + t0.7 ×4)·태스크 2(099·100).

실행: py -3 x160_placement_e2e.py   (8141 에 14B 가 떠 있어야 한다 — /v1/models 로 자가 확인)
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
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL_32B = "http://localhost:8140/v1"
URL_SM = "http://localhost:8141/v1"
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
# 궤적-실재 사실만(x149.FACTS 와 동일 출처·gold 무관): 099=2년차·3만 / 100=65일차·3.1만
TASKS = {
    "task_099": {"days": 730, "case": {"qualifying_deposit_usd": 30000}},
    "task_100": {"days": 65, "case": {"qualifying_deposit_usd": 31000}},
}


def model_of(base_url):
    with urllib.request.urlopen(base_url + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def ask(base_url, model, prompt, temp):
    body = json.dumps({"model": model, "temperature": temp, "max_tokens": 40,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def five(base_url, model, prompt):
    out = []
    for i in range(5):
        try:
            a = ask(base_url, model, prompt, 0.0 if i == 0 else 0.7)
        except Exception as e:
            a = "ERR %r" % (e,)
        out.append(" ".join(a.split()))
    return out


def main():
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    m32 = model_of(URL_32B)
    msm = model_of(URL_SM)
    print("32B=%s  small=%s" % (m32, msm))
    if "14B" not in msm:
        print("⚠ 8141 이 14B 가 아니다 — 배치 arm 의 전제(⒝ 는 14B 부터)가 깨진다. 중단.")
        return 1

    for task, tf in TASKS.items():
        gold = X.GOLD[task]
        table = LG.eligible_text(tf["days"], {}, maps, spec, tf["case"]).strip()
        base = table + "\n\n" + X.FACTS[task] + "\n\n" + X.QUESTION
        msgs = Y.msgs_of(TAG, task)
        # ★T2_DROP_MSG=<idx> — C348 이 지목한 **우리 문장**(#26)을 뺀 baseline(B′) 을 재기 위한
        #   스위치. 기본은 원본. 결함 있는 baseline 으로 crossover 를 주장하면 안 된다(§6.2a).
        drop = os.environ.get("T2_DROP_MSG")
        if drop not in (None, ""):
            d = int(drop)
            if 0 <= d < len(msgs):
                msgs = msgs[:d] + msgs[d + 1:]
                print("  (msg #%d 제거됨 — B′ 모드)" % d)
        traj = ("Here is a customer-service conversation so far.\n\n"
                + Y.render(msgs) + "\n\n")

        # ⒝ 14B 선택 (깨끗) — 되꽂을 답 = greedy 산출물
        sel5 = five(URL_SM, msm, base)
        sel = sel5[0].splitlines()[0][:60].strip().rstrip(".")
        hand = ("A separate analysis, working only from the policy constants on record and "
                "this customer's stated situation, selects: %s." % sel)
        sel32_5 = five(URL_32B, m32, base)
        sel32 = sel32_5[0].splitlines()[0][:60].strip().rstrip(".")
        hand32 = ("A separate analysis, working only from the policy constants on record and "
                  "this customer's stated situation, selects: %s." % sel32)

        arms = [
            ("S  14B선택(깨끗)", sel5),
            ("P  배치: 14B되꽂기+운반", five(URL_SM, msm, traj + base + "\n\n" + hand)),
            ("B  32B단독(오염)", five(URL_32B, m32, traj + base)),
            ("I  32B+격리기(참조)", five(URL_32B, m32, traj + base + "\n\n" + hand32)),
        ]
        print("\n%s  gold=%s  (되꽂은 답: 14B=%r · 32B=%r)" % (task, gold, sel, sel32))
        for label, answers in arms:
            hit = sum(1 for a in answers if X.score(a, gold))
            print("  %-24s %d/5  %s" % (label, hit, [a[:24] for a in answers]))
    print("\n판정: P ≥ B 면 배치 crossover 성립 · P ≈ I 면 선택·운반 자리에 32B 불필요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
