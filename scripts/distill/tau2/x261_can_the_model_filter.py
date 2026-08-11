# -*- coding: utf-8 -*-
r"""x261 — **모델이 필터를 못 하는가, 아니면 못 본 것인가** (격리 · 유료 0 · 8140 · 새 엔진 0).

## 왜 (사용자 지시 2026-08-11: *"엔진이 해야 된다던 부분을 엄밀히 격리한 적 없지 않나"*)

x260 은 무효였다 — *"이 대화가 이미 회수한 문서만 쓴다"* 로 후보를 짰는데 그 안에 **gold
`Sky Blue` 가 없었다**(회수된 6종: Light Blue·Cobalt Blue·Purple·Lime Green·Blue·True Blue).
**모델은 본 적 없는 것을 고를 수 없다** ⇒ 네 팔 0/8 은 필터 무능의 증거가 아니다.

그래서 다시 묻는다: **후보에 gold 가 들어 있으면 모델이 네 제약의 논리곱을 하는가.**

## 팔 (n · 계기 = 추천한 계좌 이름 하나)

  A_LIVE      라이브 재현(결정 턴 msg 16 까지의 커밋 히스토리)   ← 재현(오답이어야 한다)
  B_MATERIAL  라이브 + **C417 프로덕션 재료**(`material_for`)     ← 회수 결손만 메운다
  C_ISO       **격리** — 손님 제약 + 같은 재료만(대화 없음)       ← C417 이 8/8 낸 형태
  D_NULL      라이브 + 값 없는 같은 길이 한 줄                    ← [[57]] 부정통제

읽는 법 (⛔0 ②):
  `B` 나 `C` 가 높으면 → **모델은 필터를 한다.** 결손은 **회수**이고 엔진은 후보를 **모아 주기만**
    하면 된다 — 제거([[63]])도 argmax 도 짓지 않는다. [[62]] 선을 거의 안 밟는 가장 싼 결말.
  둘 다 낮으면 → 그때 비로소 070 에도 제거가 정당해진다.
  `B` 낮고 `C` 만 높으면 → 재료는 맞는데 **대화가 방해**한다(= 격리 서브의 자리).

★재료는 우리가 짓지 않는다 — 출시 경로(`t2_search.material_for`)를 그대로 부른다([[03b]]).
★제약은 **손님 발화 축자**에서 온다(gold 아님·[[23]]).

실행: T2_PROBE_URL=http://localhost:8140/v1/chat/completions python x261_can_the_model_filter.py [N]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_search as S                                             # noqa: E402
from x248_search_agent_e2e import a2, DOCS, NOW                   # noqa: E402

RES = "/home/woori/scratch/tau2-bench/data/simulations/bank_all6b_20260811/results.json"
GOLD = "Sky Blue"
GROUP = "business_checking_accounts"
ASK = ("The customer asked which ONE business checking account fits ALL of: "
       "ATM fee rebates of at least $15/month · zero overdraft fees · minimum balance "
       "requirement under $10,000 · APY of at least 1%. Name exactly one account.")
NAMES = ["Sky Blue", "Light Blue", "True Blue", "Navy Blue", "Cobalt Blue", "Purple",
         "Hunter Green", "Lime Green", "Beige", "World Blue", "Gold Saver", "Blue"]


def ctx(sim, cut):
    out = []
    for m in sim["messages"][:cut]:
        r = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        tcs = [tc.get("name") for tc in (m.get("tool_calls") or [])]
        if any(tcs):
            out.append("[%s calls] %s" % (r, ", ".join(x for x in tcs if x)))
        if c:
            out.append("[%s] %s" % (r, c[:1500]))
    return "\n".join(out)


def customer_lines(sim, cut):
    return "\n".join("[customer] " + " ".join(str(m.get("content") or "").split())[:900]
                     for m in sim["messages"][:cut] if m.get("role") == "user")


def score(msg):
    txt = str(msg.get("content") or "")
    bold = [n for n in NAMES if re.search(r"\*\*%s[^*]{0,24}\*\*" % re.escape(n), txt)]
    picked = bold or [n for n in NAMES if n in txt]
    if not picked:
        return "(이름 없음)"
    first = picked[0]
    return "HIT" if first == GOLD else "PICK(%s)" % first[:14]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(open(RES, encoding="utf-8"))
    sim = [s for s in d["simulations"] if s["task_id"] == "task_070"][0]
    cut = 16
    live = ctx(sim, cut)

    mat, info = S.material_for(a2, GROUP, DOCS, NOW)
    has = GOLD in (mat or "")
    print("재료 %d자 · info=%s · **gold 포함 %s**"
          % (len(mat or ""), json.dumps(info, ensure_ascii=False)[:160], has))
    if not has:
        print("⚠재료에 gold 가 없다 — 이 프로브도 x260 과 같은 무효가 된다. 중단.")
        return
    print("라이브 문맥 %d자 · 결정 턴 %d · n=%d\n" % (len(live), cut, n))

    matblock = "\n[system] Retrieved account documents:\n" + (mat or "")[:14000]
    nullc = "\n[system] Take a moment before answering; the customer asked for a single account."
    arms = [("A_LIVE", live),
            ("B_MATERIAL", live + matblock),
            ("C_ISO", customer_lines(sim, cut) + matblock),
            ("D_NULL", live + nullc)]
    for label, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body + "\n\n" + ASK, None, 0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[score(r)] += 1
        print("  %-11s 문맥 %6d자 · HIT %d/%d   %s"
              % (label, len(body), c["HIT"], n, c.most_common(4)))
    print("\n※ B 나 C 가 높으면 **모델은 필터를 한다** — 결손은 회수이고 엔진은 모아 주기만 하면 "
          "된다([[62]] 선을 거의 안 밟는다).\n  둘 다 낮을 때에만 제거([[63]])가 070 에 정당해진다."
          "\n  D 가 오르면 프로브 무효.")


if __name__ == "__main__":
    main()
