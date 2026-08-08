# -*- coding: utf-8 -*-
"""회귀 검정 (C330·C331): 강제할 단계는 **선행 그래프가 정한다**.

배경 — 라이브 2 sim에서 read 강제는 켜져 있었는데 **한 번도 시도되지 않았다**. 옛 규칙이
표적을 *짐작*하느라 조준 보조물(수요 신호 프록시 3종·피의존≥2·1회/sim 캡)을 달고 있었고,
그 프록시가 이 계열에서 원리적으로 못 뜨기 때문이었다(C330 전수 확인).

지금은 짐작이 없다. 큐가 매 턴 *"지금 할 일"*을 satisfier **하나**로 특정해 주고
(GB1→verify_identity · GB3→get_referrals_by_user · reads:→그 read), gate_patch 가 그 머리를
`_t2_demanded_step` 에 심는다. 이 검정은 **정책 순서 ↔ 그 특정 ↔ 핀 판정**이 어긋나지 않는지 본다.

오프라인 전용(서버·LLM 불요). 실행: py -3 test_pin_demand_from_queue.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_pin_read as PR                                      # noqa: E402
import t2_dominance as DOM                                    # noqa: E402
from gate_interpreter import load_domain_a2                    # noqa: E402

FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


class _Reg(object):
    def __init__(self, muts):
        self._m = dict(muts)

    def has_tool(self, n):
        return n in self._m

    def tool_mutates_state(self, n):
        return self._m[n]


class _Env(object):
    def __init__(self, muts):
        self.tools = _Reg(muts)
        self.user_tools = None


class _Tool(object):
    def __init__(self, name):
        self.name = name


class _Orch(object):
    def __init__(self, step=None, own=(), muts=None):
        self.environment = _Env(muts or {})
        self.tools = [_Tool(n) for n in own]
        if step:
            self._t2_demanded_step = step


class _AM(object):
    tool_calls = None


def main():
    os.environ["T2_PIN_READ"] = "1"
    a2 = load_domain_a2("banking_knowledge")
    if not a2:
        print("A2 없음 — skip")
        return 0

    # ── ① 그래프가 각 요건을 도구 **하나**로 특정한다 ─────────────────────
    rs = DOM.requirements_for(a2, [], "submit_referral")
    ids = [str(r.get("id")) for r in rs or []]
    sats = [list(r.get("satisfiers") or []) for r in rs or []]
    print("     요건 순서: %s" % ids)
    print("     satisfiers: %s" % sats)
    chk(bool(rs) and all(len(s) == 1 for s in sats),
        "모든 요건이 satisfier 하나로 특정된다(짐작 0)")

    # ── ② 정책 순서: 게이트가 먼저, 선행 read가 나중 ──────────────────────
    first_reads = next((i for i, x in enumerate(ids)
                        if x.startswith(DOM.READS_PREFIX)), None)
    chk(first_reads is None or first_reads == len(ids) - 1,
        "선행 read 는 게이트 뒤에 온다(머리 규칙이 조기 무장을 막는 근거)")

    # ── ③ 머리가 바뀔 때마다 핀 표적도 그대로 따라간다 ────────────────────
    #     환경 대역: 이 단계들은 전부 상태를 바꾸지 않는다(레지스트리가 그렇게 답한다).
    muts = {s[0]: False for s in sats if s}
    PR._resolve = lambda orch, base: (base + "_3847"
                                      if base == "get_all_user_accounts_by_user_id" else None)
    for req_id, sat in zip(ids, sats):
        step = sat[0]
        # 에이전트 일반 도구인 경우와 discoverable 인 경우를 둘 다 밟는다
        own = (step,) if step != "get_all_user_accounts_by_user_id" else ()
        got = PR.pin_for(_Orch(step, own=own, muts=muts), _AM(), a2, [])
        want = (step, None, None) if own else \
            ("unlock_discoverable_agent_tool", "agent_tool_name", step + "_3847")
        chk(got == want, "머리=%s → 고정 %s" % (req_id, got))

    # ── ④ 기준선: 그래프가 아무것도 특정하지 않으면 고정하지 않는다 ───────
    chk(PR.pin_for(_Orch(muts=muts), _AM(), a2, []) is None,
        "단계 미특정이면 무발화")

    # ── ⑤ 상태를 바꾸는 단계는 이름과 무관하게 배제된다 ───────────────────
    chk(PR.pin_for(_Orch("get_all_user_accounts_by_user_id",
                         muts={"get_all_user_accounts_by_user_id": True}),
                   _AM(), a2, []) is None,
        "레지스트리가 mutating 이라 답하면 read 처럼 생겨도 고정 안 함")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
