# -*- coding: utf-8 -*-
"""x168 — 저장소 편집(엔진 필터)은 **분지 강제**를 막는가? 그리고 막히면 모델은 어디로 가나.

## 질문 (사용자 2026-08-09: "엔진 필터가 가족의 정의역을 바꾸는 것으로 보면 되나?")

§6.4 는 개입을 셋으로 나눴다 — 저장소 편집(엔진 필터) / 분지 강제(정박·되꽂기) / 증거 추가(무효).
그런데 **§6.3 은 엔진 필터를 켠 채로 측정됐고, 정박은 그대로 먹혔다.** 즉 두 연산은 축이 다르다:

  - 필터 축 = **자격**(정책 기준) — 행을 *기준으로* 지운다. 가족 구조엔 **눈이 없다**.
  - 분지 축 = **성(姓) 유사도** — 표면형으로 묶인다.

그러므로 저장소 편집이 분지 강제를 막는 것은 **지운 가족을 겨냥한 정박에 한해서**일 것이다.
이것이 §8.4 #1 "계열 범위 필터 = 안전장치" 주장의 기전이고, 여기서 시험한다.

## arm (32B·자유생성이 본 측정 — 표 밖 이름=날조를 봐야 하므로)

  1. full  + Card 정박   : 기지(§6.3) — Business Platinum 이 나온다
  2. **scope + Card 정박**: ★핵심. 카드 가족을 저장소에서 **통째로 지우고** 카드를 지목한다.
       ⒜ gold 로 복귀 → **저장소 편집이 분지 강제를 이긴다**(안전장치 성립)
       ⒝ 표에 없는 카드 이름 산출 → **분지 강제가 저장소 밖으로 민다** = 날조(§3 C43/C45 와 연결)
          ⇒ 범위 필터만으로는 안전하지 않고, 날조 게이트가 함께 필요하다
  3. scope + Green 정박  : 통제 — 남겨 둔 가족을 겨냥하면 여전히 먹혀야 한다(부분 보호 확인)
  4. scope + 정박 없음   : 위생 통제 — gold

⚠**범위 제거는 프로브-측 절제**다. 실제 구현의 범위 필터는 A3 `source.doc`(계열을 기계적으로
가름)를 써야 하며 엔진이 이름 문자열을 뜯어서는 안 된다([[59]]). 여기서는 *가설 시험*이 목적.

실행: py -3 x168_scope_vs_basin.py     (8140=32B)
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

URL = "http://localhost:8140/v1"
TASK = "task_099"
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
TRIGGER = 26
NAMED = "Hunter Green"
CARD = "Business Bronze Rewards Card"


def model_of(u):
    with urllib.request.urlopen(u + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def gen(model, prompt, n=3, mx=24):
    outs = []
    for i in range(n):
        body = json.dumps({"model": model, "temperature": 0.0 if i == 0 else 0.7,
                           "max_tokens": mx,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(URL + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            outs.append(" ".join(json.load(r)["choices"][0]["message"]["content"].split()))
    return outs


def main():
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    full = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()

    head, body = full.split("\n", 1)
    keep, dropped = [], []
    for ln in body.splitlines():
        (dropped if "Card" in ln.split(":")[0] else keep).append(ln)
    scope = head + "\n" + "\n".join(keep)
    names = lambda t: [l.strip().split(":")[0].strip() for l in t.splitlines() if l.startswith("  ")]
    print("full %d행 · scope %d행 (제거 %d = %s)"
          % (len(names(full)), len(names(scope)), len(dropped),
             [d.strip().split(":")[0] for d in dropped]))

    model = model_of(URL)
    MS = Y.msgs_of(TAG, TASK)

    def prompt(table, anchor):
        ms = [dict(m) for m in MS]
        if anchor is None:
            ms = ms[:TRIGGER] + ms[TRIGGER + 1:]
        else:
            ms[TRIGGER]["content"] = str(ms[TRIGGER].get("content") or "").replace(NAMED, anchor)
        base = table + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        return ("Here is a customer-service conversation so far.\n\n"
                + Y.render(ms) + "\n\n" + base)

    # ★통제: 같은 16행이되 **카드를 남기고 최저-보너스 9행을 지운** 표.
    #   arm3 가 gold 로 풀린 것이 "행 수(저장 부하)" 때문인지 "그 카드 행들" 때문인지 가른다.
    #   부하 효과면 여기서도 풀려야 하고, 카드-특정이면 여기선 여전히 Lime Green 이 나와야 한다.
    def bonus(ln):
        for part in ln.split(","):
            if "referrer_bonus_usd" in part:
                try:
                    return int(part.split("=")[1])
                except Exception:
                    return -1
        return -1
    ranked = sorted(body.splitlines(), key=bonus)
    lowdrop = set(ranked[:len(dropped)])
    low16 = head + "\n" + "\n".join(l for l in body.splitlines() if l not in lowdrop)

    # ★★동점 가설(2026-08-09): arm3(카드 제거) 이 정박을 푼 이유가 **최상위 동점 소거**인가?
    #   full·low16 에는 World Blue $300 ↔ Business Platinum $300 동점이 있고, scope 에는 없다.
    #   그래서 **동점 상대 한 행만** 빼고(다른 카드는 전부 유지) 같은 정박을 건다.
    #   풀리면 지배 변수 = 최상위 동점 · 안 풀리면 = 범주/희석 축.
    #   ※제거만 한다 — 정책 상수를 지어내 동점을 *만드는* 설계는 금지([[23]]).
    TIE = "Business Platinum Rewards Card"
    notie = head + "\n" + "\n".join(l for l in body.splitlines()
                                    if l.strip().split(":")[0].strip() != TIE)

    arms = [("1 full  + Card정박", full, CARD),
            ("2 scope + Card정박 ★", scope, CARD),
            ("3 scope + Green정박", scope, NAMED),
            ("4 scope + 정박없음", scope, None),
            ("5 low16 + Green정박 ☆", low16, NAMED),
            ("6 low16 + Card정박", low16, CARD),
            ("7 notie + Green정박 ★★", notie, NAMED),
            ("8 notie + Card정박", notie, CARD),
            ("9 notie + 정박없음", notie, None)]
    print("notie %d행 (동점 상대 %r 만 제거·나머지 카드 유지)" % (len(names(notie)), TIE))
    print("low16 %d행 (제거=%s)"
          % (len(names(low16)), [l.strip().split(":")[0] for l in ranked[:len(dropped)]]))

    print("\n%-22s %-34s %s" % ("arm", "자유생성 ×3", "판정"))
    for label, table, anchor in arms:
        outs = gen(model, prompt(table, anchor))
        tbl = [n.lower() for n in names(table)]
        first = outs[0]
        intbl = any(n in first.lower() for n in tbl)
        gold = X.score(first, X.GOLD[TASK])
        verdict = ("gold ✓" if gold else ("표-안 오답" if intbl else "★표-밖 = 날조"))
        print("%-22s %-34s %s" % (label, str([o[:24] for o in outs])[:34], verdict))
        if not intbl:
            print("      ⚠ 표에 없는 이름: %r" % first[:60])
    print("\n판정: arm2 가 gold 면 **저장소 편집이 분지 강제를 이긴다**(범위 필터=안전장치) · "
          "표-밖 이름이면 **정박이 저장소 밖으로 민다**(범위 필터만으론 불충분·날조 게이트 필요)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
