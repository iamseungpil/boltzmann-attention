# -*- coding: utf-8 -*-
"""Step 2-3: reference-filter 레버 유닛 (keystone·참조축·2026-07-13).
resolve_reference_filter: call_discoverable의 참조 id를 formalize(user→기준)→결정론 filter→검증/교정."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("T2_GATE_KINDS", "auth")
import t2_resolve as R

BANK = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "a2", "banking_knowledge.gate.json"), encoding="utf-8"))

# 수집 record(tool result 텍스트) — ATM 유일·CityFit 2중복
RECTXT = """Transactions for account chk_x:
Found 4 record(s):
1. Record ID: btxn_atm
   transaction_id: btxn_atm
   date: 11/05/2025
   description: RHO-BANK ATM #4827 WITHDRAWAL
   amount: -300.0
   type: atm_withdrawal
2. Record ID: btxn_gym1
   transaction_id: btxn_gym1
   date: 11/06/2025
   description: CITYFIT GYM MONTHLY
   amount: -89.99
   type: debit_card_purchase
3. Record ID: btxn_gym2
   transaction_id: btxn_gym2
   date: 11/06/2025
   description: CITYFIT GYM MONTHLY
   amount: -89.99
   type: debit_card_purchase"""


class TC:
    def __init__(self, name, arguments): self.name, self.arguments = name, arguments
class AM:
    def __init__(self, tool_calls=None): self.role, self.tool_calls = "assistant", tool_calls
class M:
    def __init__(self, role, content="", error=False):
        self.role, self.content, self.error = role, content, error


def hist(user):
    return [M("user", user), M("assistant"), M("tool", RECTXT)]


class FakeSub:
    def __init__(self, j): self.content = j
class FakeLA:
    def __init__(self, j): self._j = j
    def generate(self, **kw): return FakeSub(self._j)
class FakeAgent:
    llm = "m"; llm_args = {}; tools = []
class FakeUM:
    def __init__(self, role=None, content=None): self.role, self.content = role or "user", content


def dispute(tid):
    return AM(tool_calls=[TC("call_discoverable_agent_tool",
              {"agent_tool_name": "file_debit_card_transaction_dispute_6281",
               "arguments": json.dumps({"transaction_id": tid, "dispute_category": "atm_cash_discrepancy"})})])


FAILS = []
def ck(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + ("" if c else " | " + str(d)))
    if not c: FAILS.append(n)


CRIT_ATM = '{"date": "11/05/2025", "merchant": "ATM", "transaction_type": "atm_withdrawal"}'

# 1) 에이전트가 틀린 id(gym1) 지목·기준=ATM/11-05 → filter가 btxn_atm 유일식별 → deny 교정
r = R.resolve_reference_filter(dispute("btxn_gym1"), hist("dispute my ATM withdrawal on Nov 5"),
                               BANK, FakeAgent(), FakeLA(CRIT_ATM), FakeUM)
ck("wrong_ref_deny", r["status"] == "deny" and r["reason"] == "reference-filter", r)
ck("correct_is_atm", r.get("correct") == "btxn_atm", r)

# 2) 정답 id(atm) 지목 → ok
r = R.resolve_reference_filter(dispute("btxn_atm"), hist("dispute my ATM withdrawal on Nov 5"),
                               BANK, FakeAgent(), FakeLA(CRIT_ATM), FakeUM)
ck("right_ref_ok", r["status"] == "ok", r)

# 3) 애매(CityFit 2중복·on_ambiguous=none) → 미개입(filter None)
CRIT_GYM = '{"date": "11/06/2025", "merchant": "CITYFIT", "transaction_type": "debit_card_purchase"}'
r = R.resolve_reference_filter(dispute("btxn_gym1"), hist("dispute the duplicate CityFit charge"),
                               BANK, FakeAgent(), FakeLA(CRIT_GYM), FakeUM)
ck("ambiguous_noop", r["status"] == "ok", r)   # 진짜중복=미개입(18% 수용)

# 4) formalize 실패(빈 기준) → 미개입
r = R.resolve_reference_filter(dispute("btxn_gym1"), hist("help"), BANK, FakeAgent(), FakeLA("{}"), FakeUM)
ck("no_crit_noop", r["status"] == "ok", r)

# 5) agent 없음 → 우아한 강등
r = R.resolve_reference_filter(dispute("btxn_gym1"), hist("x"), BANK)
ck("no_agent_ok", r["status"] == "ok", r)

# 6) reference_filter 미설정(retail류) → ok
r = R.resolve_reference_filter(dispute("btxn_gym1"), hist("x"), {}, FakeAgent(), FakeLA(CRIT_ATM), FakeUM)
ck("no_config_ok", r["status"] == "ok", r)

# 7) 다른 dispatch 도구(prefix 불일치) → 미개입
am = AM(tool_calls=[TC("call_discoverable_agent_tool",
        {"agent_tool_name": "apply_credit_card_1234", "arguments": '{"transaction_id":"btxn_gym1"}'})])
r = R.resolve_reference_filter(am, hist("x"), BANK, FakeAgent(), FakeLA(CRIT_ATM), FakeUM)
ck("prefix_mismatch_noop", r["status"] == "ok", r)

print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (reference-filter 레버)")
sys.exit(1 if FAILS else 0)
