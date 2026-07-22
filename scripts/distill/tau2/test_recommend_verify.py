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
    def __init__(self, role, content="", error=False, tool_calls=None):
        self.role, self.content, self.error, self.tool_calls = role, content, error, tool_calls or []


def hist():
    # 에이전트가 KB_search_bm25 함(apply-flow 신호·research_tool 게이트 충족 — 실환경 도구명, e3a8654f)
    return [M("user", "I need a card with NO foreign transaction fees and purchase protection."),
            M("assistant", tool_calls=[TC("KB_search_bm25", {"query": "credit cards no foreign fee"})]),
            M("tool", "Silver Rewards Card: no foreign transaction fees, purchase protection, $120k limit. "
                      "Platinum Rewards Card: has foreign transaction fees, travel perks.")]


# formalize 목킹: 올바른 카드 + apply-intent(applies) 지정
class FakeSub:
    def __init__(self, v, applies=True):
        self.content = '{"applies": %s, "card_type": "%s"}' % ("true" if applies else "false", v)
class FakeLA:
    def __init__(self, v, applies=True): self._v, self._a = v, applies
    def generate(self, **kw): return FakeSub(self._v, self._a)
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

# 5) 텍스트추천(offer 안 씀)+apply-intent 종결 → recommendation-offer(올바른 카드로 제안 유도)
r = R.resolve_recommendation(AM(content="I recommend the Platinum card."), hist(), BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("text_reco_offer_deny", r["status"] == "deny" and r["reason"] == "recommendation-offer", r)
ck("offer_names_correct", "Silver Rewards Card" in r.get("feedback", "")
   and "give_discoverable_user_tool" in r.get("feedback", ""), r)

# 5b) 미추천(카드 언급 없이 포기)+apply-intent → recommendation-offer
r = R.resolve_recommendation(AM(content="I'm not able to help with that, sorry."), hist(), BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("under_reco_offer_deny", r["status"] == "deny" and r["reason"] == "recommendation-offer", r)

# 5c) apply-intent 아님(applies=false) → ok(스퓨리어스 방지)
r = R.resolve_recommendation(AM(content="Have a nice day."), hist(), BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card", applies=False), FakeUM)
ck("not_apply_intent_ok", r["status"] == "ok", r)

# 5d) 이전에 이미 offer함 → 중복 nag 금지
h = hist() + [M("assistant"), ]
h[-1].tool_calls = [TC("give_discoverable_user_tool",
                    {"discoverable_tool_name": "apply_for_credit_card",
                     "arguments": json.dumps({"card_type": "Silver Rewards Card"})})]
r = R.resolve_recommendation(AM(content="Let me know if you need more."), h, BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("already_offered_ok", r["status"] == "ok", r)

# 5e) 아직 작업 중(조회 도구 호출) → ok
r = R.resolve_recommendation(AM(tool_calls=[TC("KB_search", {"query": "x"})]), hist(), BANK,
                             FakeAgent(), FakeLA("Silver Rewards Card"), FakeUM)
ck("still_working_ok", r["status"] == "ok", r)

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
