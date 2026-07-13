# -*- coding: utf-8 -*-
"""Lever 4: pre-recommendation 검증(오추천 방지) 유닛 (사용자 2026-07-13).
resolve_recommendation 순수함수 — 에이전트가 give_discoverable_user_tool로 apply 제안 시
card_type을 요구→formalize(올바른 카드) 검증·틀리면 deny."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth")
import t2_resolve as R

BANK = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))


class TC:
    def __init__(self, name, arguments): self.name, self.arguments = name, arguments
class AM:
    def __init__(self, tool_calls=None, content=None):
        self.role, self.tool_calls, self.content = "assistant", tool_calls, content
class M:
    def __init__(self, role, content="", error=False):
        self.role, self.content, self.error = role, content, error


def hist():
    return [M("user", "I need a card with NO foreign transaction fees and purchase protection."),
            M("tool", "Silver Rewards Card: no foreign transaction fees, purchase protection, $120k limit. "
                      "Platinum Rewards Card: has foreign transaction fees, travel perks.")]


# formalize 목킹: 반환할 올바른 카드 지정
class FakeSub:
    def __init__(self, v): self.content = '{"card_type": "%s"}' % v
class FakeLA:
    def __init__(self, v): self._v = v
    def generate(self, **kw): return FakeSub(self._v)
class FakeAgent:
    llm = "m"; llm_args = {}; tools = []
class FakeUM:
    def __init__(self, role=None, content=None): self.role, self.content = role or "user", content


def offer(card):
    return AM(tool_calls=[TC("give_discoverable_user_tool",
              {"discoverable_tool_name": "apply_for_credit_card",
               "arguments": json.dumps({"card_type": card})})])


FAILS = []
def ck(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + ("" if c else " | " + str(d)))
    if not c: FAILS.append(n)


# 1) 오추천(Platinum) + formalize=Silver → deny recommendation-verify
r = R.resolve_recommendation(offer("Platinum Rewards Card"), hist(), BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("wrong_offer_deny", r["status"] == "deny" and r["reason"] == "recommendation-verify", r)
ck("names_correct", "Silver Rewards Card" in r.get("feedback", ""), r)
ck("has_call_anchor", r.get("call") is not None, r)

# 2) 정답 제안(Silver) + formalize=Silver → ok
r = R.resolve_recommendation(offer("Silver Rewards Card"), hist(), BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("right_offer_ok", r["status"] == "ok", r)

# 3) formalize=none(불확실) → ok(미개입·안전)
r = R.resolve_recommendation(offer("Platinum Rewards Card"), hist(), BANK,
                             FakeAgent(), FakeLA("none"), FakeUM)
ck("formalize_none_ok", r["status"] == "ok", r)

# 4) 다른 action 제안(get_referral_link) → 검증 안 함
am = AM(tool_calls=[TC("give_discoverable_user_tool",
        {"discoverable_tool_name": "get_referral_link", "arguments": "{\"card_type\": \"X\"}"})])
r = R.resolve_recommendation(am, hist(), BANK, FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("other_action_noop", r["status"] == "ok", r)

# 5) offer 호출 없음(텍스트만) → ok (이 경로는 미커버·안전)
r = R.resolve_recommendation(AM(content="I recommend the Platinum card."), hist(), BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("text_only_noop", r["status"] == "ok", r)

# 6) recommendation_verify 미설정(retail류) → ok
r = R.resolve_recommendation(offer("Platinum Rewards Card"), hist(), {},
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("no_config_ok", r["status"] == "ok", r)

# 7) agent 없음(오프라인) → ok(우아한 강등)
r = R.resolve_recommendation(offer("Platinum Rewards Card"), hist(), BANK)
ck("no_agent_ok", r["status"] == "ok", r)

# 8) 빈 card_type → ok
am = AM(tool_calls=[TC("give_discoverable_user_tool",
        {"discoverable_tool_name": "apply_for_credit_card", "arguments": "{}"})])
r = R.resolve_recommendation(am, hist(), BANK, FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("empty_operand_ok", r["status"] == "ok", r)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (Lever 4 pre-recommendation verify)")
sys.exit(1 if FAILS else 0)
