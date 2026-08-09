# -*- coding: utf-8 -*-
"""회귀 검정: **2단 재도출이 라이브에서 실제로 발화하는가** (2026-08-09).

무엇을 막는 검정인가 —
 ⒜ **측정된 구성이 조용히 조기 반환되는 것**. x158 n=10 은 목적 구절을 **빼는 쪽**이 이겼다
    (099: 목적 있음 0/10 · 없음 10/10). 그래서 호출부는 `asked=""` 를 넘긴다. 그런데 가드가
    `asked` 를 참으로 요구하고 있었고, 그 결과 유료 런 `bank_rederive_20260809k` 6 sim 이
    `[T2_REDERIVE]` **발화 0** 으로 돌았다 — 재도출을 한 번도 태우지 않고 기준선을 다시 산 것이다.
    ⇒ 빈 목적은 **정상 입력**이고 호출은 일어나야 한다.
 ⒝ **침묵해야 할 때 말하는 것**. 표가 없거나 후보 목록이 없으면 물을 근거가 없다 — 호출 0.
 ⒞ **목록 밖 이름을 옮기는 것**. 답이 후보 원소가 아니면 `None`(침묵)이어야 한다([[22]] 집합
    검사만·C107: 게이트가 날조 통로가 된 실물).
 ⒟ **턴마다 다시 묻는 것**. 같은 재료면 메모가 재호출을 막아야 한다(비용).

오프라인 전용: tau2·서버·LLM 불요(전부 가짜 주입). 실행: py -3 test_rederive_wiring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402

FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


class _Msg(object):
    def __init__(self, content):
        self.content = content


class _UM(object):
    def __init__(self, content=None, role=None):
        self.content = content
        self.role = role


class _Agent(object):
    """재도출이 쓰는 것만 갖춘 최소 대역 — llm·llm_args·메모 슬롯."""

    def __init__(self):
        self.llm = "fake-model"
        self.llm_args = {"temperature": 0.0, "tools": "이건 걸러져야 한다"}


class _LA(object):
    """`generate` 호출을 세는 가짜 llm_agent 모듈."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, model=None, tools=None, messages=None, call_name=None, **kw):
        self.calls.append({"model": model, "call_name": call_name,
                           "prompt": getattr(messages[0], "content", ""), "kw": kw})
        return _Msg(self.answer)


# 도메인 어휘를 짓지 않기 위해 후보 이름은 **무의미 토큰**을 쓴다(검정이 정책을 발명하지 않는다).
SPEC = {"rederive_prompt": "표:\n{table}\n사실:\n{facts}\n목적:\n{asked}\n무엇을 고르겠는가?"}
TABLE = "  alpha_row: x = 1\n  beta_row: x = 2"
ALLOWED = ["alpha_row", "beta_row"]
FACTS = "days since the earliest account was opened = 100"


def run(answer, asked="", table=TABLE, allowed=None, agent=None, la=None):
    agent = agent if agent is not None else _Agent()
    la = la if la is not None else _LA(answer)
    out = LG.rederive_choice(agent, la, _UM, SPEC, table,
                             FACTS, asked, ALLOWED if allowed is None else allowed)
    return out, la, agent


def main():
    # ── ⒜ 빈 목적은 정상 입력이다 (이 검정의 전부) ──────────────────────────
    out, la, _ = run("I would pick beta_row.", asked="")
    chk(len(la.calls) == 1, "목적이 빈 문자열이어도 모델을 부른다  ← 유료 6 sim 을 헛돌린 결손")
    chk(out == "beta_row", "돌아온 답에서 후보 원소를 집어낸다 (%r)" % (out,))

    out, la, _ = run("beta_row", asked=None)
    chk(len(la.calls) == 1 and out == "beta_row", "목적이 None 이어도 죽지 않는다(메모 키 포함)")

    # 목적을 실제로 주는 경로도 살아 있어야 한다(다른 태스크·계열용·초안 §8.2)
    out, la, _ = run("alpha_row", asked="가장 큰 보너스를 받는 것")
    chk(len(la.calls) == 1 and "가장 큰 보너스" in la.calls[0]["prompt"],
        "목적을 주면 그대로 문맥에 실린다")
    chk("tools" not in la.calls[0]["kw"],
        "도구 인자는 걸러진다(깨끗한 문맥 = 표+사실+목적뿐)")
    chk(la.calls[0]["call_name"] == "rederive_choice", "호출이 이름으로 표시된다(회계 가능)")

    # ── ⒝ 근거가 없으면 묻지 않는다 (음성 통제·[[57]]) ──────────────────────
    out, la, _ = run("beta_row", table="")
    chk(out is None and not la.calls, "표가 없으면 호출 0 (물을 근거가 없다)")
    out, la, _ = run("beta_row", allowed=[])
    chk(out is None and not la.calls, "후보 목록이 비면 호출 0")
    la = _LA("beta_row")
    chk(LG.rederive_choice(_Agent(), la, _UM, {}, TABLE, FACTS, "", ALLOWED) is None
        and not la.calls, "A2가 문구를 선언 안 하면 호출 0 (선언 없는 도메인은 거동 0)")
    la = _LA("beta_row")
    chk(LG.rederive_choice(None, la, _UM, SPEC, TABLE, FACTS, "", ALLOWED) is None
        and not la.calls, "에이전트가 없으면 호출 0")

    # ── ⒞ 목록 밖은 침묵 ────────────────────────────────────────────────────
    out, la, _ = run("I recommend gamma_row, which is not on the list.")
    chk(len(la.calls) == 1 and out is None,
        "목록 밖 이름은 옮기지 않는다(묻긴 했다)  ← 게이트가 날조 통로가 되는 자리")

    # 부분 문자열이 다른 이름에 먹히지 않는다(가장 긴 일치)
    la = _LA("alpha_row_extended and alpha_row")
    out = LG.rederive_choice(_Agent(), la, _UM, SPEC, TABLE, FACTS, "",
                             ["alpha_row", "alpha_row_extended"])
    chk(out == "alpha_row_extended", "가장 긴 일치를 고른다(부분 문자열 오식별 방지)")

    # ── ⒟ 같은 재료면 다시 묻지 않는다 ──────────────────────────────────────
    agent, la = _Agent(), _LA("beta_row")
    for _ in range(3):
        LG.rederive_choice(agent, la, _UM, SPEC, TABLE, FACTS, "", ALLOWED)
    chk(len(la.calls) == 1, "같은 재료로 세 번 불러도 모델 호출은 1회(턴마다 오는 자리다)")
    LG.rederive_choice(agent, la, _UM, SPEC, TABLE, FACTS + " (다름)", "", ALLOWED)
    chk(len(la.calls) == 2, "재료가 달라지면 다시 묻는다")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
