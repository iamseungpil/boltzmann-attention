# -*- coding: utf-8 -*-
"""회귀 검정: **결정 블록이 나가는 메시지에서 이름 목록이 빠지는가** (R8b·2026-08-10).

무엇을 막는 검정인가 —
 ⒜ **기본값이 바뀌는 것**. `T2_DECISION_ISOLATE` 미설정이면 종전과 **바이트 동일**이어야 한다.
    (오늘 켠 것을 내일 잊고 비교하면 두 런이 다른 코드가 된다.)
 ⒝ **블록이 없는 턴에서 목록이 사라지는 것**. 그 턴엔 그 목록이 유일한 근거일 수 있다.
 ⒞ **억제가 다른 문장까지 먹는 것**. 빼는 것은 우리가 만든 그 문자열뿐이고, 블록·상태 문장은
    그대로 남아야 한다.

근거(x231 leave-one-in·n=8): 실제 문맥 위에서 `ineligible_text` 두 문장만 얹어도 task_100 이
0/8 로 무너진다. 그 목록은 결정 서브가 이미 소비한 재료다.

오프라인 전용(LLM·서버·tau2 불요 — 전부 가짜 주입). 실행: py -3 test_decision_isolate.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


# ── 가짜 fact DAG: A3 조회 결과만 돌려준다 (엔진은 이 값을 읽기만 한다) ──────────────
_MINS = {"Alpha": (30, "doc_a"), "Bravo": (60, "doc_b"), "Charlie": (90, "doc_c")}
_AX = {"gain": {"Alpha": 35, "Bravo": 175, "Charlie": 300}}


def _install_fake_factdag():
    m = types.ModuleType("t2_factdag")
    m.load = lambda a2: a2

    class _In(object):
        def __init__(self, corpus=None, a3=None):
            self.corpus, self.a3 = corpus, a3
    m.Inputs = _In
    m.evaluate = lambda dag, inp: ({"doc_minimums": _MINS, "doc_limits": {},
                                    "gain_map": _AX["gain"]}, None)
    sys.modules["t2_factdag"] = m


def _install_fake_tau2():
    """엔진이 재도출 경로에서 함수 안에 import 하는 것들 — 이름만 있으면 된다."""
    for name in ("tau2", "tau2.agent", "tau2.data_model"):
        sys.modules.setdefault(name, types.ModuleType(name))
    la = types.ModuleType("tau2.agent.llm_agent")
    la.generate = lambda *a, **k: None
    sys.modules["tau2.agent.llm_agent"] = la
    msg = types.ModuleType("tau2.data_model.message")

    class _UM(object):
        def __init__(self, content=None, role=None):
            self.content, self.role = content, role
    msg.UserMessage = _UM
    sys.modules["tau2.data_model.message"] = msg
    sys.modules["tau2"].agent = sys.modules["tau2.agent"]
    sys.modules["tau2.agent"].llm_agent = la
    sys.modules["tau2"].data_model = sys.modules["tau2.data_model"]
    sys.modules["tau2.data_model"].message = msg


_install_fake_factdag()
_install_fake_tau2()

import t2_gate_patch as G                                      # noqa: E402
import t2_ledger as LG                                           # noqa: E402

SPEC = {
    "ineligible_text": ("Arithmetic on the elapsed time above ({days} days) against the "
                        "minimums you retrieved: not reachable yet - {blocked}. Reachable on "
                        "this criterion - {ok}.\n"),
    "eligible_text": "Eligible rows:\n{rows}\n",
    "eligible": {"axes": ["gain"]},
    "decided_text": "\nIt answers: {choice}.\nfigures: {operands}\nrow: {row}\n",
    "rederive_prompt": "pick one: {rows}",
}
A2 = {"policy_ontology": {"rows": ()},
      "derived": [{"op": "a3_map", "out": "gain_map", "params": {"axis": "gain"}}]}


class _Agent(object):
    def __init__(self, days=61):
        self._t2_ledger_ops = {"acct": {"spec": SPEC, "days": days, "rows": (), "tally": None}}
        self.llm = "fake"
        self.llm_args = {}


def build(days=61, decided=True, isolate=False):
    """엔진 함수를 실제로 태우고, 그 턴에 나갈 문자열을 돌려준다."""
    orig_red, orig_el = LG.rederive_choice, LG.eligible_text

    def fake_red(*a, **k):
        return "Bravo" if decided else None

    def fake_el(days, tally, axm, spec, stated, as_rows=False):
        if as_rows:
            return [("Alpha", {"gain": 35}), ("Bravo", {"gain": 175})]
        return "Eligible rows:\nAlpha; Bravo\n"

    LG.rederive_choice, LG.eligible_text = fake_red, fake_el
    if isolate:
        os.environ["T2_DECISION_ISOLATE"] = "1"
    else:
        os.environ.pop("T2_DECISION_ISOLATE", None)
    try:
        return G._limit_reduce_text(_Agent(days), A2, [])
    finally:
        LG.rederive_choice, LG.eligible_text = orig_red, orig_el
        os.environ.pop("T2_DECISION_ISOLATE", None)


LIST_SIG = "Reachable on this criterion"

print("\n§1 기본값 — 플래그 없으면 종전 그대로")
off = build(decided=True, isolate=False)
chk(LIST_SIG in off, "플래그 OFF·블록 있음 → 이름 목록이 남는다 (거동 불변)")
chk("It answers: Bravo." in off, "플래그 OFF → 결정 블록은 나간다")

print("\n§2 플래그 ON — 블록이 나가는 턴에서만 목록이 빠진다")
on = build(decided=True, isolate=True)
chk(LIST_SIG not in on, "플래그 ON·블록 있음 → 이름 목록이 빠진다")
chk("It answers: Bravo." in on, "플래그 ON → 결정 블록은 그대로 남는다")
chk(len(on) < len(off), "억제는 문자열을 줄인다 (%d < %d)" % (len(on), len(off)))

print("\n§3 블록이 없는 턴은 건드리지 않는다")
nod = build(decided=False, isolate=True)
chk(LIST_SIG in nod, "플래그 ON·블록 없음 → 목록이 남는다 (유일한 근거일 수 있다)")
chk("It answers:" not in nod, "블록 없음이 실제로 재현됐다 (통제)")

print("\n§4 억제가 다른 문장을 먹지 않는다")
chk(off.replace(LIST_SIG, "") != off, "통제: 목록 서명이 OFF 문자열에 실재했다")
rest_off = [s for s in off.split("\n") if s.strip() and LIST_SIG not in s
            and "not reachable yet" not in s]
rest_on = [s for s in on.split("\n") if s.strip()]
chk(all(s in rest_on for s in rest_off),
    "목록 밖 줄은 한 줄도 안 사라졌다 (%d줄 대조)" % len(rest_off))

print("\n%s  (%d/%d)" % ("FAIL" if FAILED else "ALL PASS",
                         8 - len(FAILED), 8))
sys.exit(1 if FAILED else 0)
