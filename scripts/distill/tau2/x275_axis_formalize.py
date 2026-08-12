# -*- coding: utf-8 -*-
r"""x275 — 축(계열) 형식화가 **격리에서 되는가** ([[18]] 정보-맞춘 격리 · 유료 0 · 엔진 0).

## 왜 (라이브 실측 `bank_lever_071_20260812f` trial1)

손님 축자 — *"I run a small creative design studio and I'm looking to open a new **business
checking** account … and I also want to open a **business savings** account"* · 뒤에도
*"Let's open **True Blue Business Checking** and the **Gold Saver Account**"*.

그런데 모델의 실제 호출은 `account_type="checking"`(**개인**)·`"savings"`(**개인**)이고 이름도
개인 계열의 실재 이름(`Bluest Account`·`Gold Account`)이었다. 이름은 맞고 **계열이 틀렸다.**

우리 이름 레버(ENUM)는 여기서 **옳게 침묵한다** — 모델이 선언한 축 안에서 이름 실재만 보기
때문이다. 축이 옳은지는 손님 의도 해석 = **열린 술어**라 엔진이 판정하면 안 된다([[22]]·[[62]]).
그래서 먼저 물어야 한다: **격리에서는 맞히는가.**

## 팔 (n=8 · 계기는 `account_type` 4지선다 정답 여부 — env 독스트링 축자 집합)

  A_LIVE     실제 궤적 문맥(write 직전까지)                ← 결손 재현 기대
  B_ISO      **손님 발화만**(격리)                          ← 되면 부하/전달
  C_FIRST    첫 발화 한 줄만                                ← 최소 정보 상한
  D_NOASK    문맥은 주되 요청 문장 제거 (부정 통제·[[57]])   ← 이게 높으면 프로브 무효

닫힌 집합은 **도구 독스트링 축자**에서 온다(gold 무관):
  "Must be one of: 'checking' (personal checking), 'savings' (personal savings),
   'business_checking', 'business_savings'"

⚠엔진이 고르는 형태를 시험하는 것이 아니다 — **LLM 이 형식화하는가**를 잰다.

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x275_axis_formalize.py [N]
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                               # noqa: E402
import x256_dispatcher_write_probe as X256                          # noqa: E402

SIMS = os.environ.get("X275_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
TAG = os.environ.get("X275_TAG", "bank_lever_071_20260812f")
TASK = "task_071"
GOLD = {"business_checking", "business_savings"}
ENUM = ["checking", "savings", "business_checking", "business_savings"]

Q = ("A customer service agent must choose the account_type argument for opening an account.\n"
     "The tool declares it must be one of: 'checking' (personal checking), 'savings' "
     "(personal savings), 'business_checking', 'business_savings'.\n\n"
     "Customer:\n{text}\n\n"
     "Which account_type values does this customer's request call for? "
     "Reply with the value(s) only, one per line.")


def score(raw):
    t = " ".join(str(raw or "").split()).lower()
    picked = set()
    for e in sorted(ENUM, key=len, reverse=True):
        if re.search(r"\b%s\b" % re.escape(e), t):
            picked.add(e)
            t = t.replace(e, " ")
    if picked == GOLD:
        return "EXACT"
    if picked and picked <= GOLD:
        return "PARTIAL(%s)" % ",".join(sorted(picked))
    if picked:
        return "WRONG(%s)" % ",".join(sorted(picked))
    return "NONE"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(io.open(os.path.join(SIMS, TAG, "results.json"), encoding="utf-8"))
    sims = [s for s in d["simulations"]
            if s["task_id"] == TASK and (s.get("reward_info") or None) is not None]
    sim = sims[-1]
    msgs = sim["messages"]
    cut = None
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            if "open_bank_account" in json.dumps(tc.get("arguments") or {}) and cut is None:
                cut = i
    if cut is None:
        cut = max(i for i, m in enumerate(msgs) if m.get("role") == "user")
    users = [str(m.get("content") or "") for m in msgs[:cut] if m.get("role") == "user"]
    live = X256.build(sim, cut, True)
    ask = "\n---\n".join(u for u in users if u.strip())
    first = users[0] if users else ""
    # 부정 통제: 요청 문장(계열을 말하는 발화)을 뺀다
    noask = "\n---\n".join(u for u in users[1:] if u.strip())

    print("cut=%d · 손님 발화 %d개 · live %d자\n" % (cut, len(users), len(live)))
    # ★부정 통제 재설계 (2026-08-12·초판 실패). 초판 `D_NOASK` 는 첫 발화만 뺐는데 남은
    #   발화들이 여전히 "business checking"·"True Blue Business Checking" 을 말해 **신호가
    #   제거되지 않았고** 8/8 이 나왔다 — 내 읽기 규칙대로 그 판은 무효다([[57]]).
    #   제대로 된 부정 통제는 **답이 달라져야 하는 입력**이다: 같은 질문에 *개인* 계좌를
    #   요청하는 손님 텍스트를 넣고 `business_*` 가 **안 나오는지** 본다. 여기서도 gold 가
    #   나오면 그 질문은 텍스트를 읽는 게 아니라 무언가를 되풀이하는 것이다.
    #   ⚠이 텍스트는 프로브 전용 합성이고 A2·엔진에 들어가지 않는다.
    personal = ("Hi, I'd like to open a personal checking account for my own day-to-day "
                "spending, and a personal savings account for my emergency fund. This is "
                "just for me, not for any business.")
    # 그리고 계열 낱말을 지운 판 — 원문에서 'business' 만 제거한다(다른 정보는 그대로).
    stripped = re.sub(r"(?i)\bbusiness\b", "", ask)
    arms = (("A_LIVE", live), ("B_ISO", Q.format(text=ask)),
            ("C_FIRST", Q.format(text=first)),
            ("N1_PERSONAL", Q.format(text=personal)),
            ("N2_STRIPPED", Q.format(text=stripped)))
    for label, body in arms:
        if label == "A_LIVE":
            body = live + "\n\n" + Q.format(text="(see the conversation above)")
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, None, 0.0 if i == 0 else 0.7, 60).get("content")
            except Exception as e:
                r = "ERR %s" % type(e).__name__
            c[score(r)] += 1
        print("  %-9s EXACT %d/%d   %s" % (label, c["EXACT"], n, c.most_common(3)))
    print("\n※ B 높고 A 낮음 ⇒ 축은 **부하** 문제 — 레버는 표면화(두 LLM 출력을 비교해 알림)."
          "\n  A·B 둘 다 낮음 ⇒ 형식화 자체가 경계 — 표면화도 안 통한다."
          "\n  D 가 높으면 요청 문장이 없어도 맞힌 것이니 프로브가 무효다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
