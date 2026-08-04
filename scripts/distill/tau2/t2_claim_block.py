"""L4 — 아무것도 실행하지 않은 채 "처리됐습니다"라고 말하는 것을 막는다.

`ACTION_HANDOFF_LEVERS_DESIGN` rev2가 구현 전 선결로 남긴 것: *"x59의 느슨한 술어로는 49/80이고
x50의 엄격 술어로는 11이다 — 어느 술어로 걸 것인지 확정하고 표적을 그 술어로 다시 셀 것."*
N97 전수 194 sim으로 네 후보를 재보고 가장 좁은 것을 고른다:

    완료-주장 문장이 있다                                  210건 / 102 sim · pass 10   ← 절반이 걸린다
    ∧ 손님 발화 이후 write 0                               117건 /  82 sim · pass  6
    ∧ sim 전체에 실효 write 0                                     32 sim · pass  3
    **최종 발화가 완료-주장 ∧ sim 전체 write 0**                  16 sim · **pass 0**  ← 과차단 0

마지막 것이 표적이지만 "최종 발화"는 생성 시점에 알 수 없다. 라이브-판정 가능한 동치 근사는
**주장 ∧ 그 메시지에 도구 호출 없음 ∧ 지금까지 실효 write 0**이고, 전수에서 **28건 / 23 sim
(그 sim 중 pass 1)** 이다. 그 1건이 유일한 과차단 후보다.

★막는 것은 **주장**이지 행동이 아니다([[06]] write-강제 금지). 엔진은 "너는 아직 아무것도 실행하지
않았다"는 궤적 사실만 말하고, 무엇을 할지는 모델이 정한다 — 실행하든, 못 한다고 말하든, 이관하든.
"""

import os
import re

# 과거형 완료 주장만. "I will file the dispute"는 계획이고 "the dispute has been filed"는 보고다.
DONE = re.compile(
    r"(has|have) been (successfully )?(filed|submitted|processed|updated|created|closed|"
    r"issued|applied|transferred|completed|credited|reversed|logged|generated|sent)"
    r"|i (have|already) (filed|submitted|processed|updated|created|logged|applied|sent)"
    r"|(is|are) now (filed|submitted|updated|processed|active|complete)",
    re.I)

# 읽기·절차 계열은 실효 write가 아니다(도메인 어휘가 아니라 동사 접두).
READ_PREFIX = ("get_", "list_", "check_", "search_", "view_", "read_", "fetch_",
               "kb_", "think", "transfer", "verify", "log_")


def enabled():
    return os.environ.get("T2_CLAIM_BLOCK") == "1"


def _is_write(name):
    n = str(name or "").lower()
    return bool(n) and not n.startswith(READ_PREFIX)


def executed_writes(messages, eff_name):
    """궤적에 남은 실효 write 수 — 손님이 실행한 것도 센다(그 sim에서 일이 일어났으므로)."""
    n = 0
    for m in messages or []:
        if getattr(m, "role", None) not in ("assistant", "user"):
            continue
        for tc in (getattr(m, "tool_calls", None) or []):
            if _is_write(eff_name(tc)):
                n += 1
    return n


def blocks(messages, content, has_tool_calls, eff_name):
    """이 발화를 되돌려야 하는가. 조건 미충족이면 False = 종전 거동."""
    if not enabled() or has_tool_calls:
        return False
    if not isinstance(content, str) or not DONE.search(content):
        return False
    return executed_writes(messages, eff_name) == 0


FEEDBACK = (
    "Error: [CLAIM-UNBACKED] you are telling the customer that something has been done, but "
    "this conversation contains no executed action - no write tool has been called by you or "
    "by them. Do not report work that did not happen. Either perform the action now, or tell "
    "the customer plainly what you could not do and why."
)


def selftest():
    class M:
        def __init__(self, role, content=None, calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    class C:
        def __init__(self, name):
            self.name = name

    def eff(tc):
        return getattr(tc, "name", "")

    os.environ["T2_CLAIM_BLOCK"] = "1"
    claim = "Your dispute has been submitted successfully."

    hist = [M("user", "please file it")]
    assert blocks(hist, claim, False, eff)
    print("  ok   실행 0 + 완료 주장 = 차단")

    done = hist + [M("assistant", None, [C("submit_cash_back_dispute_0589")])]
    assert not blocks(done, claim, False, eff)
    print("  ok   내가 실행했으면 통과")

    byuser = hist + [M("user", None, [C("apply_for_credit_card")])]
    assert not blocks(byuser, claim, False, eff)
    print("  ok   손님이 실행했어도 통과(그 sim에서 일이 일어났다)")

    reads = hist + [M("assistant", None, [C("get_user_information_by_name"), C("KB_search_bm25")])]
    assert blocks(reads, claim, False, eff)
    print("  ok   조회만 했으면 여전히 차단(읽기는 실효 write 아님)")

    assert not blocks(hist, claim, True, eff)
    print("  ok   같은 턴에 도구를 부르고 있으면 대상 아님")

    assert not blocks(hist, "I will file the dispute now.", False, eff)
    print("  ok   계획 문장은 주장 아님")

    os.environ["T2_CLAIM_BLOCK"] = "0"
    assert not blocks(hist, claim, False, eff)
    os.environ["T2_CLAIM_BLOCK"] = "1"
    print("  ok   플래그 OFF면 무발화")
    print("PASS (7/7)")


if __name__ == "__main__":
    selftest()
