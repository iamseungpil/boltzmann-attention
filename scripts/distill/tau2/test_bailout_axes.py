# -*- coding: utf-8 -*-
"""이탈 감사의 두 축 검정 — 색인 공간과 손님-요구 술어.

이 검정이 존재하는 이유(2026-08-14 야간 실물 결함 2건·둘 다 **집계를 조용히 뒤집었다**):
  ① **색인 공간 혼동**: `transfer_index` 는 **호출 순번**을 주는데 그것을 메시지 색인으로 써서
     궤적을 앞에서 잘랐다(074: 23 vs 실제 60). 결과 — "손님이 이관을 요구함"이 **7건 전부 0**으로
     집계됐고, 하마터면 *"에이전트가 손님 뜻과 무관하게 일을 버린다"* 로 원장에 남을 뻔했다.
  ② **느슨한 술어**: 초판은 'agent' 한 낱말로 세어 상담원을 **언급만** 해도 요구로 읽었다.

두 축이 뒤집히면 결론이 정반대가 되므로(자발 이탈 7 ↔ 0) 여기서 고정한다.
"""
import io
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F                                            # noqa: E402
import bank_bailout_audit as A                                     # noqa: E402

FAIL = []


def chk(c, m):
    if not c:
        FAIL.append(m)
    print("  %s %s" % ("ok  " if c else "FAIL", m))


def M(role, content=None, calls=None):
    d = {"role": role, "content": content}
    if calls:
        d["tool_calls"] = calls
    return d


def TC(name, args=None):
    return {"function": {"name": name, "arguments": args or {}}}


def test_index_space():
    """이관 앞에 손님 발화가 여럿 있어 **호출 순번 ≪ 메시지 색인** 인 궤적."""
    msgs = [M("user", "hello"), M("assistant", None, [TC("KB_search_bm25")]), M("tool", "docs")]
    msgs += [M("user", "please just fix it"), M("assistant", "working on it")] * 6
    msgs += [M("user", "Yes, please transfer me to a human agent."),
             M("assistant", None, [TC("transfer_to_human_agents", {"summary": "x"})])]
    sim = {"messages": msgs, "task_id": "t", "reward_info": {"reward": 0.0}}
    mi = F.transfer_msg_index(sim)
    ci = A.transfer_index(sim)
    chk(mi == len(msgs) - 1, "메시지 색인 = 마지막(%s)" % mi)
    chk(ci == 1, "호출 순번 = 1 (KB 다음) — 두 수는 다르다(%s)" % ci)
    chk(mi != ci, "두 색인 공간이 실제로 갈린다(이 차이가 결함의 자리였다)")
    asked, last = A.user_asked_transfer(sim, mi)
    chk(asked and last, "메시지 색인으로 자르면 손님 요구가 보인다")
    asked2, _ = A.user_asked_transfer(sim, ci)
    chk(not asked2, "호출 순번으로 자르면 못 본다 = 초판이 0으로 셌던 기전")


def test_request_predicate():
    def sim_of(text):
        return {"messages": [M("user", text),
                             M("assistant", None, [TC("transfer_to_human_agents")])],
                "task_id": "t", "reward_info": {"reward": 0.0}}

    pos = ["Yes, please transfer me to a human agent.",
           "I'd like to speak to a real person.",
           "Please connect me with a supervisor.",
           "Yes, please transfer me to a human agent. ###TRANSFER###"]
    neg = ["The agent told me the fee was waived.",
           "Can you check my account balance?",
           "A human agent helped me last time and it was fine."]
    for t in pos:
        a, _ = A.user_asked_transfer(sim_of(t), 1)
        chk(a, "요구로 센다: %s" % t[:44])
    for t in neg:
        a, _ = A.user_asked_transfer(sim_of(t), 1)
        chk(not a, "요구 아님: %s" % t[:44])


def main():
    for t in (test_index_space, test_request_predicate):
        print("[%s]" % t.__name__)
        t()
    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
