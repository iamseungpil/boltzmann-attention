# -*- coding: utf-8 -*-
"""관문4 WEV형(§2ao·e2e9 031) 오프라인 검증 — dispute write 전 give-flow last4 증거 요구.

라이브 `_wev_deny_msgs` + 실제 A2 스펙을 그대로 잰다([[03b]]). 031 재현: 날조 1654(give-flow 출력 없음)
→ deny·진짜 5320(사용자-실행 출력에 도구명+4자리 공존) → 허용·last4 인자 없는 write → skip(false-block 0).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t2_gate_patch import _wev_deny_msgs  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
SPECS = [s for s in A2["write_evidence_specs"]
         if (s.get("applies_when") or {}).get("prefix") == "file_credit_card_transaction_dispute"]


class _TC:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def main():
    check("A2: last4 WEV 스펙 1건", len(SPECS) == 1)
    disp = _TC("call_discoverable_agent_tool",
               {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                "arguments": json.dumps({"transaction_id": "txn_adea68821a1d",
                                         "card_last_4_digits": "1654",
                                         "card_action": "keep_active"})})
    msgs = [_Msg("tool", "Found 47 record(s) in 'credit_card_transaction_history': ...")]
    # 031 재현: 날조 1654·give-flow 출력 없음 → deny
    deny = _wev_deny_msgs(msgs, disp, SPECS)
    check("031 재현: 날조 last4 → deny", deny is not None and "1654" in deny
          and "give_discoverable_user_tool" in deny)
    # 진짜 give-flow: 사용자-실행 출력에 도구명+5320 공존 → 허용
    give_out = _Msg("tool", "Executed: get_card_last_4_digits_2211\n  - Card: Platinum\n  - Last 4 digits: 5320")
    disp_true = _TC("call_discoverable_agent_tool",
                    {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                     "arguments": json.dumps({"transaction_id": "txn_adea68821a1d",
                                              "card_last_4_digits": "5320"})})
    check("진짜 last4(give-flow 출력 공존) → 허용",
          _wev_deny_msgs(msgs + [give_out], disp_true, SPECS) is None)
    # 날조인데 give-flow 출력은 다른 값 → 여전히 deny (id 공존 불성립)
    check("give-flow 있어도 다른 값 주장 → deny",
          _wev_deny_msgs(msgs + [give_out], disp, SPECS) is not None)
    # last4 인자 없는 dispute 변형 → idv 부재 skip (false-block 0)
    disp_no4 = _TC("call_discoverable_agent_tool",
                   {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                    "arguments": json.dumps({"transaction_id": "txn_x"})})
    check("last4 인자 부재 → skip(false-block 0)",
          _wev_deny_msgs(msgs, disp_no4, SPECS) is None)
    # 무관 도구 → 미적용
    check("무관 write → 미적용",
          _wev_deny_msgs(msgs, _TC("call_discoverable_agent_tool",
                                   {"agent_tool_name": "order_replacement_credit_card_7291",
                                    "arguments": "{}"}), SPECS) is None)

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        sys.exit(1)
    print("ALL PASS — 관문4 WEV형: 날조 차단·진짜 give-flow 허용·인자부재 skip.")


if __name__ == "__main__":
    main()
