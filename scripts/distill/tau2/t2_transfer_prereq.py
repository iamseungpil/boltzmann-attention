"""L5-a — 이관하기 전에 최소한 찾아는 봤는가.

`ACTION_HANDOFF_LEVERS_DESIGN` §3 L5의 세 검정 중 **a절만** 구현한다. 셋을 전수(194 sim)로 재보면:

    a  이관 전 `KB_search` 0회            **9건 / 9 sim**   ← 구현
    c  `summary`가 시도한 도구를 미인용    50건 / 50 sim     ← 보류(§ 아래)
    b′ `summary`의 노드 계수 검증          —                 ← L3 노드 선언이 없어 착수 불가

**c를 보류하는 이유**: 이관 71건 중 50건이 걸린다 = 표적이 아니라 **일괄 요건**이다. 게다가 `summary`는
인계 문서이지 우리 내부 도구명을 적는 자리가 아닐 수 있어, 인용을 요구하면 무관한 이름을 적는
자기정당화 통로가 된다(rev2가 J 술어를 하향한 것과 같은 사유). 정밀도를 먼저 재고 결정한다.

**b′는 선행 부재**: 노드 계수를 검증하려면 A2에 노드 선언(L3)이 있어야 하는데 아직 없다.

★막지 않는다 — 이관 자체는 정당한 행동이고, 여기서 되돌리는 것은 **찾아보지도 않고 넘기는 턴**뿐이다.
그리고 되돌림은 **생성-측 재생성**이라 tool 출력을 만들지 않는다(C210 replay 제약을 건드리지 않는다).
"""

import os

TRANSFER_MARK = "transfer"
SEARCH_MARKS = ("kb_search", "search")


def enabled():
    return os.environ.get("T2_TRANSFER_PREREQ") == "1"


def _names(m):
    for tc in (getattr(m, "tool_calls", None) or []):
        yield str(getattr(tc, "name", "") or "").lower()


def searched_before(messages):
    for m in messages or []:
        if getattr(m, "role", None) != "assistant":
            continue
        for n in _names(m):
            if any(s in n for s in SEARCH_MARKS) and TRANSFER_MARK not in n:
                return True
    return False


def missing_prereq(messages, am):
    """이 턴이 '검색 0회 상태의 이관'인가. 아니면 False = 종전 거동."""
    if not enabled():
        return False
    if not any(TRANSFER_MARK in n for n in _names(am)):
        return False
    return not searched_before(messages)


FEEDBACK = (
    "Error: [TRANSFER-PREREQ] you are handing this conversation to a human without having "
    "searched the knowledge base even once. The customer's question may be answerable with "
    "what is documented here. Search first; if the answer genuinely is not there, transfer "
    "and say in the summary what you looked for and did not find."
)


def selftest():
    class M:
        def __init__(self, role, calls=()):
            self.role, self.tool_calls, self.content = role, [C(c) for c in calls], None

    class C:
        def __init__(self, name):
            self.name = name

    os.environ["T2_TRANSFER_PREREQ"] = "1"
    xfer = M("assistant", ["transfer_to_human_agents"])

    assert missing_prereq([M("user")], xfer)
    print("  ok   검색 0회 이관 = 되돌림")

    searched = [M("assistant", ["KB_search_bm25"])]
    assert not missing_prereq(searched, xfer)
    print("  ok   검색했으면 통과")

    only_reads = [M("assistant", ["get_user_information_by_name"])]
    assert missing_prereq(only_reads, xfer)
    print("  ok   레코드 조회는 검색이 아니다")

    assert not missing_prereq([M("user")], M("assistant", ["get_current_time"]))
    print("  ok   이관 아닌 턴은 대상 아님")

    os.environ["T2_TRANSFER_PREREQ"] = "0"
    assert not missing_prereq([M("user")], xfer)
    os.environ["T2_TRANSFER_PREREQ"] = "1"
    print("  ok   플래그 OFF면 무발화")
    print("PASS (5/5)")


if __name__ == "__main__":
    selftest()
