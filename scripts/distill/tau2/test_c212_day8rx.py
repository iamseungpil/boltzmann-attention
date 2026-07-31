#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C212/day8 처방(A1·A2·A4·B1·B2·B3) 오프라인 검증 (2026-07-28·무료·모델 불요).
`DAY8_PRESCRIPTIONS_DESIGN_2026_07_28` §검증 배터리 — day7 중간-포렌식([S]) 표적:
- A1 FOLLOWUP tool_args 이행판정 (022/027 무관-give 침묵 갭)
- A4 TERM_GRANT 유저-직접-방출 폴백 (008 notice-부재 미발동)
- B1 [coverage] 미판정-행 잔존 검출 (019/022/027)
- B3 Unknown-tool 반려 이름 수집 (010/014/015/016)
- A2/B2 = A2 선언 실재 확인
⚠단위통과≠라이브발화([[30]])."""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP        # noqa: E402
import t2_eplan_patch as EP       # noqa: E402

OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


class M:
    def __init__(self, role, content=None, tool_calls=None, mid=None, error=False):
        self.role, self.content, self.tool_calls = role, content, tool_calls
        self.id, self.error = mid, error


class TCall:
    def __init__(self, name, cid="c1", args=None, requestor="assistant"):
        self.name, self.id, self.arguments = name, cid, (args or {})
        self.requestor = requestor


A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
RD = next(t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")


def test_a1_fu_target():
    print("[test_a1] FOLLOWUP tool_args 이행판정")
    ta = (RD.get("follow_up") or {}).get("tool_args")
    chk(ta == {"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        "A2 follow_up.tool_args 선언 실재")
    give_irrel = M("assistant", tool_calls=[TCall(
        "give_discoverable_user_tool", "g1",
        {"discoverable_tool_name": "get_card_last_4_digits"})])
    give_target = M("assistant", tool_calls=[TCall(
        "give_discoverable_user_tool", "g2",
        {"discoverable_tool_name": "submit_cash_back_dispute_0589"})])
    chk(not GP._fu_target_called([give_irrel], "give_discoverable_user_tool", ta),
        "무관-대상 give만 있으면 미이행 (022/027 갭 차단)")
    chk(GP._fu_target_called([give_irrel, give_target], "give_discoverable_user_tool", ta),
        "표적 give 실재 → 이행")
    chk(GP._fu_target_called([give_irrel], "give_discoverable_user_tool", {}),
        "tool_args 미선언 → 도구명 단위(하위호환)")
    user_give = M("user", tool_calls=[TCall(
        "give_discoverable_user_tool", "g3",
        {"discoverable_tool_name": "submit_cash_back_dispute_0589"}, requestor="user")])
    chk(not GP._fu_target_called([user_give], "give_discoverable_user_tool", ta),
        "user-requestor 호출은 계상 안 함(requestor 격리)")


def test_a4_term_grant_userdemand():
    print("[test_a4] TERM_GRANT 유저-직접-방출 폴백 (t2_eplan_patch 소스 검사+거동)")
    import inspect
    src = inspect.getsource(EP._terminal_grant_check)
    chk("T2_TERM_GRANT_USERDEMAND" in src, "폴백 분기 실재")
    chk(src.index("T2_TERM_GRANT_USERDEMAND") > src.index("notice_text"),
        "폴백은 notice 경로 실패 후에만(우선순위 보존)")
    chk("###TRANSFER###" in src, "동의-터미널 토큰 요건(ⓐ′) 유지")
    # 리뷰 반영: 도메인-의존 보강 문구는 엔진 하드코딩 금지 → A2 선언에서만
    esrc = inspect.getsource(EP)
    chk("identifiers" not in esrc, "엔진에 identifiers 문구 하드코딩 없음(리뷰 A4-1)")
    chk("term_grant_reminder_extra" in esrc, "A2 선언 조회 경로 실재")
    ntc = next(g for g in A2["gates"]
               if isinstance(g, dict) and g.get("kind") == "notice")
    chk("identity verification is not required"
        in str(ntc.get("term_grant_reminder_extra") or ""),
        "banking A2에 reminder_extra 선언(verify note 사실 근거)")
    class _FakeEnv:
        domain_name = "no_such_domain_x"
    class _FakeOrch:
        environment = _FakeEnv()
    chk(EP._term_grant_reminder_extra(_FakeOrch()) == "",
        "A2 미선언 도메인 → 빈 문자열(타 도메인 정책위반 지시 없음)")


def test_b1_coverage_pending():
    print("[test_b1] [coverage] 미판정-행 잔존 검출")
    call1 = M("assistant", tool_calls=[TCall("get_reward_discrepancies", "t1")])
    bad = M("tool", "ids...\n[coverage] 22 of 23 rows were checked (1 could not be "
            "verified). The unverified rows are missing input field(s): 'promo_end' "
            "(1 rows).", mid="t1")
    p = GP._coverage_pending([call1, bad])
    chk(p is not None and p[0] == "get_reward_discrepancies",
        "skipped>0 → pending (도구명 귀속)")
    chk(p and "missing input field" in p[1], "coverage 라인 재인용 페이로드")
    call2 = M("assistant", tool_calls=[TCall("get_reward_discrepancies", "t2")])
    good = M("tool", "ids...\n[coverage] 23 of 23 rows were checked (0 could not be "
             "verified).", mid="t2")
    chk(GP._coverage_pending([call1, bad, call2, good]) is None,
        "이후 skipped==0 결과 → 해소")
    chk(GP._coverage_pending([call2, good]) is None, "완전판정만 → pending 없음")


def test_b3_unknown_names():
    print("[test_b3] Unknown-tool 반려 이름 수집")
    e1 = M("tool", "Error: Unknown discoverable tool 'submit_referral'.", mid="x1")
    e2 = M("tool", "Error: Unknown discoverable tool 'file_reward_dispute'.", mid="x2")
    okm = M("tool", "Tool given to user: submit_cash_back_dispute_0589", mid="x3")
    got = GP._unknown_tool_names([e1, e2, okm, M("assistant", "text")])
    chk(got == {"submit_referral", "file_reward_dispute"},
        "양 채널 에러 이름 축자 수집: %s" % sorted(got))
    chk(GP._unknown_tool_names([okm]) == set(), "정상 결과에서 수집 0")


def test_a3_rejected_params():
    print("[test_a3] Unexpected-parameter 반려 인자 수집")
    e1 = M("tool", "Error: Unexpected parameter: correct_rewards", mid="p1", error=True)
    okm = M("tool", "Tool given to user: x", mid="p2")
    chk(GP._rejected_params([e1, okm]) == {"correct_rewards"},
        "반려 인자명 축자 수집(018형)")
    chk(GP._rejected_params([okm]) == set(), "정상 결과에서 수집 0")


def test_b2_reason_guidance():
    print("[test_b2] transfer reason 선택 기준(A2 ask 문구)")
    gate = next(g for g in A2["gates"]
                if isinstance(g, dict) and "transfer_to_human_agents"
                in (g.get("applies_to") or []) and g.get("kind") == "notice") \
        if any(isinstance(g, dict) and g.get("kind") == "notice" for g in A2["gates"]) \
        else None
    asks = " ".join(str(g.get("ask") or "") for g in A2["gates"] if isinstance(g, dict))
    chk("MOST SPECIFIC reason" in asks, "서술형 선택 기준 문구 실재")
    chk("customer_demands_after_unavailable_offer_refusal" not in asks.split("MOST SPECIFIC")[-1],
        "enum 축자 스푼피딩 없음(서술형만)")


def test_go_stack_flags():
    print("[test_flags] go_stack C212 플래그 등록")
    src = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    # ★T2_DISPATCH_ROLE_NOTE는 2026-07-31 폐기(딸린 strip을 V7로 대체) — 부재를 검사한다.
    for f in ("T2_TERM_GRANT_USERDEMAND=1", "T2_COVERAGE_FOLLOWUP=1",
              "T2_UNKNOWN_REPEAT_GUARD=1", "T2_TOOL_SIGNATURE=1"):
        chk(f in src, f)
    chk("export T2_DISPATCH_ROLE_NOTE=1" not in src,
        "폐기 플래그가 스택에 남아 있지 않음(V7 대체)")


if __name__ == "__main__":
    for fn in (test_a1_fu_target, test_a3_rejected_params, test_a4_term_grant_userdemand,
               test_b1_coverage_pending, test_b3_unknown_names,
               test_b2_reason_guidance, test_go_stack_flags):
        fn()
        print()
    print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
