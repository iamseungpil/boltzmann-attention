# -*- coding: utf-8 -*-
"""write_evidence_specs 의 `skip_when_tokens` 단위검정 (§S-4 계열·2026-09-01).

무엇을 지키나: 정책이 그 단계를 **건너뛰라**고 말하는 경우에만 요구가 꺼진다.
  ① 선언이 없으면 종전과 **동일하게 반려**(회귀 방지)
  ② 선행 이력 + 같은 id → **통과**(요구 꺼짐)
  ③ 이력이 **다른 id** → 반려 유지(오차단 회피의 반대편)
  ④ 우리 로그가 이미 있으면 애초에 반려가 없다 — 순서 가드는 `found` 와 겹치는 **방어**다.

부호표(게이트 충실·전 코퍼스): **⊖ 0 · ⊕ 37**(049 18 · 048 17 · 046 1 · 043 1).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G

TOK = "record(s) in 'credit_card_closure_reasons'"
AID = "cc_x_green"


class M(object):
    def __init__(self, role, content):
        self.role, self.content = role, content


class TC(object):
    def __init__(self, name, args):
        self.name, self.arguments = name, args


def spec(skip=True):
    s = {"applies_to": "call_discoverable_agent_tool",
         "applies_when": {"arg": "agent_tool_name", "prefix": "close_credit_card_account"},
         "id_key": "credit_card_account_id",
         "require_tokens": ["log_credit_card_closure_reason"],
         "feedback": "Error: [WRITE-EVIDENCE] need log for {id}."}
    if skip:
        s["skip_when_tokens"] = [TOK]
    return [s]


def call():
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": "close_credit_card_account_7834",
               "arguments": '{"credit_card_account_id": "%s"}' % AID})


def hist(aid=AID):
    return M("tool", "Closure reason history for credit card account %s: Found 1 %s: ..." % (aid, TOK))


def ourlog(aid=AID):
    return M("tool", "Executed: log_credit_card_closure_reason_4521 for %s" % aid)


def run(name, specs, msgs, want_deny):
    got = G._wev_deny_msgs(msgs, call(), specs)
    ok = (got is not None) == want_deny
    print(("ok  " if ok else "FAIL") + " " + name)
    return ok


def main():
    r = []
    r.append(run("① 선언 없음 → 반려 유지", spec(False), [], True))
    r.append(run("② 선행 이력 + 같은 id → 통과", spec(True), [hist()], False))
    r.append(run("③ 이력이 다른 id → 반려 유지", spec(True), [hist("cc_other")], True))
    # ④ 우리 로그가 이미 있으면 **애초에 반려하지 않는다**(require_tokens 가 충족).
    #   즉 순서 가드는 `found` 분기와 겹친다 — 방어적일 뿐 부호를 만들지 않는다.
    r.append(run("④ 우리 로그 존재 → 반려 없음(순서 가드는 방어적)", spec(True),
                 [ourlog(), hist()], False))
    print("ALL PASS" if all(r) else "SOME FAILED")
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
