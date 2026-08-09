# -*- coding: utf-8 -*-
r"""x206 — 종류 선택에 **기권을 열면** 어떻게 되나 (격리 A/B · 유료 0 · 엔진 0).

## 왜 (라이브 실측 2026-08-10 · `bank_a3fill_20260810a`)

종류 선택은 098 `checking_accounts` · 100 `business_checking_accounts` 로 맞았는데, **010 은
시행마다 갈렸다**: `credit_cards` → `business_credit_cards`. 뒤엣것은 틀렸다(Wei Chen 은 개인
손님이다).

두 가지가 겹쳐 있다 —

 ⒜ **A3 커버리지 구멍**: 010 의 원장 4행은 Bronze·Gold·Silver·Platinum Rewards Card 인데, A3 에
    개인 `Bronze Rewards Card`·`Gold Rewards Card` 는 **없다**(있는 것은 `Business Bronze …` ·
    `Business Gold …`). 대화에 뜬 이름 절반이 business 쪽에만 존재한다.
 ⒝ **문구가 기권을 막는다**: 우리는 *"어느 군에 속하나? 그 이름만 답하라"* 고 **하나를 강요**한다.
    그런데 010 은 *"왜 보너스가 안 나왔나"* 를 묻는 **진단** 태스크라 고를 상품이 애초에 없다.
    답이 없는 질문에 답을 강요하면 지어낸다([[43]]·[[55]] 우리 문구부터).

⇒ 종류 선택은 사실 **두 겹**이다: *"이 대화가 상품 선택 문제인가"* → *"그렇다면 어느 군인가"*.
   지금 문구는 앞 겹을 빼먹었다.

## 팔

  FORCE    현행 문구 (반드시 하나)          ← 라이브에서 도는 것
  ABSTAIN  같은 문구 + **NONE 을 명시적으로 연다**

## 채점

  098 → `checking_accounts` · 100 → `business_checking_accounts` · **010 → NONE**(기권이 정답).
  ⚠기권을 열어 098·100 이 NONE 으로 새면 그것이 **반대편 비용**이다 — 그 손실까지 같이 센다.
  엔진은 지금도 집합 밖 답이면 안 거르므로, NONE 은 배선 변경 없이 그대로 *"안 거름"* 이 된다.

실행: python x206_kind_abstain.py [N]   (T2_PROBE_URL 로 포트 지정)
"""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8141/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

# 손님의 **실제 첫 발화** (궤적 축자 — 프로브가 저작하지 않는다)
CONV = {
    "task_098": ("Hey! My roommate just got her first real job and she's going to deposit her "
                 "first paycheck - around $600 I think. I want to refer her to open one of the "
                 "accounts you guys offer. Which one should I use to get the best possible "
                 "combined bonus for both of us?"),
    "task_100": ("I want to refer my former business partner who is starting a new company. "
                 "She'll be depositing about $31,000. Which of your business checking accounts "
                 "gives the biggest referral bonus?"),
    "task_010": ("I referred four friends to sign up for your Rho-Bank credit cards, and all four "
                 "of them completed the signup. But I only received referral bonus money for two "
                 "of them. Can you tell me why the other two didn't count?"),
}
GOLD = {"task_098": "checking_accounts", "task_100": "business_checking_accounts",
        "task_010": "NONE"}

ABSTAIN_TAIL = ("\n\nIf the customer is not asking you to pick a product - for example if they are "
                "asking why something happened to an account or a referral they already have - "
                "reply NONE.")


def ask(prompt, choices, temp):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 24,
            "messages": [{"role": "user", "content": prompt}], "guided_choice": list(choices)}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    a2 = load_domain_a2("banking_knowledge")
    sp = next(x for x in a2["ledger_metrics"] if x.get("eligible_text"))
    cfg = sp["eligible"]
    rows = (a2.get("policy_ontology") or {}).get("rows") or []
    kb = LG.subject_kinds(rows, cfg.get("kind_field") or "kind")
    kinds = sorted(set(kb.values()))
    tpl = cfg["kind_prompt"]
    listing = "\n".join("  %s" % k for k in kinds)
    print("후보 종류: %s" % kinds)
    print("⚠A3 에 개인 Bronze/Gold Rewards Card 가 있나: %s"
          % {k: (k in kb) for k in ("Bronze Rewards Card", "Gold Rewards Card",
                                    "Silver Rewards Card", "Platinum Rewards Card")})

    out = {}
    for arm, tail, choices in (("FORCE", "", kinds), ("ABSTAIN", ABSTAIN_TAIL, kinds + ["NONE"])):
        print("\n=== %s (n=%d · %s) ===" % (arm, n, MODEL))
        for task, text in CONV.items():
            c = collections.Counter()
            for i in range(n):
                p = tpl.format(kinds=listing, text=text) + tail
                try:
                    c[ask(p, choices, 0.0 if i == 0 else 0.7)] += 1
                except Exception as e:
                    c["ERR %s" % type(e).__name__] += 1
            hit = sum(v for k, v in c.items() if str(k).strip() == GOLD[task])
            out["%s/%s" % (arm, task)] = [hit, n]
            print("  %-10s 기대 %-26s %d/%d   %s"
                  % (task, GOLD[task], hit, n, c.most_common(3)))
    json.dump(out, open(os.environ.get("T2_X206_OUT", "x206_out.json"), "w"), indent=1)
    print("\n※ 010 이 ABSTAIN 에서 NONE 으로 가고 098·100 이 안 새면 → 문구에 기권을 연다."
          "\n  098·100 이 새면 그것이 반대편 비용이고, 그때는 열지 않는다([[54]] §1.3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
