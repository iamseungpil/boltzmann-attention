# -*- coding: utf-8 -*-
r"""x260 — **네 제약의 논리곱을 모델이 하는가** (격리 · 유료 0 · 8140 · 새 엔진 0).

## 왜 (C431 · 사용자 지적 "고객은 먼저 선택하지 않는다")

070 은 손님이 요건만 대고 *"어느 것 **하나**냐"* 를 반복해 묻는 태스크다. 그런데 에이전트는
**다섯 번 추천하고 다섯 번 다른 계좌**를 댄다(Card → Purple → True Blue → Blue → Light Blue).
gold = `Sky Blue`.

**결정적 축자(msg 12)**: True Blue 를 추천하면서 스스로
*"Minimum Balance Requirement: $25,000 (which is **above your $10,000 limit**, so this might not
be suitable)"* 라고 쓰고 **그대로 추천했다** ⇒ 지식 결손이 아니라 **배제 실패**([[63]]).

## ⛔0 — 제거를 짓기 전에 **전달만으로 되는지** 먼저 잰다

  A_LIVE     라이브 재현(결정 턴까지의 커밋 히스토리)     ← 재현 팔(오답이어야 한다)
  B_ROWS     후보 계좌 **행을 축자로** 모아 줌            ← **재료 배달만**(제거 0·판단 0)
  C_MINUS    같은 행에서 **위반 후보를 뺀** 목록          ← 제거([[63]])
  D_NULL     같은 자리·같은 길이·**값 없는** 한 줄        ← [[57]] 부정통제

읽는 법:
  `B` 가 이미 높으면 → 레버는 **재료 배달뿐**. 제거를 짓지 않는다(제일 싸고 안전).
  `B` 낮고 `C` 만 높으면 → [[63]] 의 여섯 번째 사례. 제거가 정당해지되 **"이 칸의 능력을 엔진이
    가져갔다"** 를 원장에 명시해야 한다([[62]] 경계).
  `D` 가 오르면 → 프로브 무효(뭐라도 붙이면 오르는 것).

★재료의 출처: 이 대화에서 **에이전트 자신이 이미 회수한 KB 문서**(role=tool 출력)뿐이다.
  새 문서를 우리가 가져오지 않는다 — 그러면 검색 결손까지 대신 메워 버린다.
★제약의 출처: **손님 발화 축자**(gold 아님). 그래서 `C_MINUS` 도 [[23]] 을 통과한다.

실행: T2_PROBE_URL=http://localhost:8140/v1/chat/completions python x260_conjunction_probe.py [N]
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

RES = "/home/woori/scratch/tau2-bench/data/simulations/bank_all6b_20260811/results.json"
GOLD = "Sky Blue"
# 손님이 **말한** 네 제약 (msg 5·9·13 축자에서 온다 — gold 아님)
CONSTRAINTS = ["ATM fee rebates of at least $15/month",
               "zero overdraft fees",
               "minimum balance requirement under $10,000",
               "APY of at least 1%"]
NAMES = ["Sky Blue", "Light Blue", "True Blue", "Navy Blue", "Cobalt Blue", "Purple",
         "Hunter Green", "Lime Green", "Beige", "World Blue", "Blue"]


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


def rows_from_conversation(sim, cut):
    """이 대화가 **이미 회수한** KB 출력에서 계좌별 조각을 모은다(우리가 새로 안 가져온다).

    자르기만 한다 — 어느 값이 좋은지 판단하지 않는다([[59]]: 뜻은 안 읽는다).
    """
    seen = collections.OrderedDict()
    for m in sim["messages"][:cut]:
        if m.get("role") != "tool":
            continue
        c = str(m.get("content") or "")
        for nm in NAMES:
            if nm not in c:
                continue
            i = c.find(nm)
            seg = c[max(0, i - 120):i + 900]
            seen.setdefault(nm, []).append(" ".join(seg.split()))
    return seen


def score(msg):
    txt = str(msg.get("content") or "")
    picked = [n for n in NAMES if re.search(r"\*\*%s[^*]*\*\*" % re.escape(n), txt)]
    if not picked:
        picked = [n for n in NAMES if n in txt]
    if not picked:
        return "(이름 없음)"
    if picked[0] == GOLD or (GOLD in picked and len(picked) == 1):
        return "HIT"
    return "PICK(%s)" % picked[0][:14]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(open(RES, encoding="utf-8"))
    sim = [s for s in d["simulations"] if s["task_id"] == "task_070"][0]
    # 결정 턴 = 손님이 "딱 하나" 를 요구한 뒤 에이전트가 답을 쓴 자리(msg 16 직전)
    cut = 16
    base = ctx(sim, cut)
    rows = rows_from_conversation(sim, cut)
    print("결정 턴 %d · 문맥 %d자 · 대화가 회수한 계좌 조각 %d종: %s\n"
          % (cut, len(base), len(rows), ", ".join(rows)))

    def block(keep):
        out = ["[system] Account passages already retrieved in this conversation:"]
        for nm in keep:
            out.append("  --- %s ---" % nm)
            out.append("  " + rows[nm][0][:900])
        return "\n".join(out)

    all_rows = block(list(rows))
    # ★제거 팔: **손님이 말한 제약**으로 위반 후보를 뺀다. 어느 계좌가 위반인지는 이 프로브가
    #   문서 축자에서 읽는다(측정용). 라이브 판본은 LLM 형식화 + 엔진 비교여야 한다([[59]]).
    keep = [nm for nm in rows if nm not in ("True Blue", "Purple", "Light Blue", "Blue")]
    minus = block(keep) if keep else all_rows
    nullc = "[system] Take a moment before answering; the customer asked for a single account."

    arms = [("A_LIVE", base),
            ("B_ROWS", base + "\n" + all_rows),
            ("C_MINUS", base + "\n" + minus),
            ("D_NULL", base + "\n" + nullc)]
    print("제거 후 남은 후보 %d종: %s\n" % (len(keep), ", ".join(keep)))
    for label, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body + "\n\nThe customer asked which ONE business checking account "
                                "fits all of: " + " · ".join(CONSTRAINTS), None,
                         0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[score(r)] += 1
        print("  %-9s 문맥 %6d자 · HIT %d/%d   %s"
              % (label, len(body), c["HIT"], n, c.most_common(4)))
    print("\n※ B 가 이미 높으면 레버는 **재료 배달뿐**이다(제거를 짓지 않는다).")
    print("  B 낮고 C 만 높으면 [[63]] 여섯 번째 — 다만 **엔진이 이 칸의 능력을 가져간다**([[62]]).")
    print("  D 가 오르면 이 프로브는 무효다.")


if __name__ == "__main__":
    main()
