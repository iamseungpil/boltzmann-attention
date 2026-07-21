# -*- coding: utf-8 -*-
"""054/050 포렌식 픽스(2026-07-21 §2bc) 오프라인 배선 테스트.
검정: ①eplan 동일-도구 이중역할 → 출력-기반 examined(충족불가 L2 술어 해소)
②WEV 빈-값 deny / 키-부재 skip / 레코드-증거 pass / 날조 deny (dispute 스펙 신판)
③A2 sanity: dispute post-write chain 제거·접미사명 검색 안내 문구 실재.
⚠️단위통과≠라이브발화([[30]]) — 배선만 본다.
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# tau2 스텁 (t2_gate_patch lazy-import 대비·test_unified_regen 패턴)
def mkmod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

mkmod("tau2"); mkmod("tau2.agent"); la = mkmod("tau2.agent.llm_agent")
mkmod("tau2.data_model"); msgmod = mkmod("tau2.data_model.message")
mkmod("tau2.orchestrator"); oo = mkmod("tau2.orchestrator.orchestrator")
msgmod.ToolMessage = type("ToolMessage", (), {})
msgmod.UserMessage = type("UserMessage", (), {})
msgmod.MultiToolMessage = type("MultiToolMessage", (), {})
la.LLMAgent = type("LLMAgent", (), {})
oo.BaseOrchestrator = type("BaseOrchestrator", (), {"__init__": lambda self, **k: None})

import t2_eplan_patch as EP  # noqa: E402
import t2_gate_patch as G    # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))

ok = True
def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ✓ " if cond else "  ✗ ") + msg)


print("① eplan 동일-도구 이중역할 → 출력-기반 examined:")
spec = A2["eplan"]
led = EP.EplanLedger(spec) if hasattr(EP, "EplanLedger") else None
if led is None:  # 클래스명 탐색(이름 다를 수 있음)
    for nm in dir(EP):
        o = getattr(EP, nm)
        if isinstance(o, type) and hasattr(o, "note_read") and hasattr(o, "examined"):
            led = o(spec)
            break
    if led is None:
        for nm in dir(EP):
            o = getattr(EP, nm)
            if isinstance(o, type):
                try:
                    cand = o(spec)
                except Exception:
                    continue
                if hasattr(cand, "note_read"):
                    led = cand
                    break
TXN_OUT = ("Found 2 record(s) in 'credit_card_transaction_history':\n"
           "1. Record ID: txn_584f9c5d00_001\n   transaction_id: txn_584f9c5d00_001\n"
           "   merchant_name: CloudSync Storage\n"
           "2. Record ID: txn_584f9c5d00_002\n   transaction_id: txn_584f9c5d00_002\n"
           "   merchant_name: Home Depot\n")
led.note_read("get_credit_card_transactions_by_user", args={"user_id": "584f9c5d00"},
              output_text=TXN_OUT)
chk(len(led.listed) == 2, "listed=2 (기존 동작 유지)")
chk(led.examined == led.listed and len(led.examined) == 2,
    "출력 전개 entity 전부 examined (구판=공집합·충족불가 해소)")
un = sorted(led.listed - led.examined)
chk(un == [], "unexamined 공집합 → L2 deny 미발화")

print("② WEV 코어: 빈-값 deny / 키-부재 skip / 증거 pass / 날조 deny:")
wev = [w for w in A2["write_evidence_specs"]
       if (w.get("applies_when") or {}).get("prefix") == "file_credit_card_transaction_dispute"]
chk(len(wev) == 1 and wev[0]["require_tokens"] == ["card_last_4_digits"],
    "dispute 스펙 token=card_last_4_digits (§2bc)")

class TC:
    def __init__(self, args):
        self.name = "call_discoverable_agent_tool"
        self.arguments = args

class TM:
    def __init__(self, content):
        self.role, self.content = "tool", content

RECORD = ("Found 1 record(s) in 'credit_card_accounts':\n1. Record ID: cc_x_gold\n"
          "   card_last_4_digits: 7823\n")
GIVE_OUT = ("Card information retrieved successfully.\nExecuted: get_card_last_4_digits\n"
            "Last 4 digits of card: 5320\n")
def dispute_call(last4, omit=False):
    inner = {"transaction_id": "txn_1", "card_action": "cancel_and_reissue"}
    if not omit:
        inner["card_last_4_digits"] = last4
    return TC({"agent_tool_name": "file_credit_card_transaction_dispute_4829",
               "arguments": json.dumps(inner)})

msgs_rec = [TM(RECORD)]
chk(G._wev_deny_msgs(msgs_rec, dispute_call("7823"), wev) is None,
    "레코드-경로 증거(7823·card_last_4_digits 공존) → pass (054 신경로)")
d1 = G._wev_deny_msgs(msgs_rec, dispute_call(""), wev)
chk(bool(d1) and "empty" in d1, "빈-값 write → deny (054 t0 구멍 폐쇄)")
chk(G._wev_deny_msgs(msgs_rec, dispute_call(None, omit=True), wev) is None,
    "키-부재 변형 → skip 유지 (오차단 회피)")
chk(bool(G._wev_deny_msgs(msgs_rec, dispute_call("1654"), wev)),
    "날조(어느 출력에도 없는 1654) → deny (031 무회귀)")
msgs_give = [TM(GIVE_OUT)]
chk(G._wev_deny_msgs(msgs_give, dispute_call("5320"), wev) is None,
    "give-flow 출력(도구명⊃토큰+5320 공존) → pass (031 경로 유지)")

print("③ A2 sanity:")
chk(all(c.get("after") != "file_credit_card_transaction_dispute"
        for c in A2["follow_up_chains"]),
    "dispute post-write chain 제거")
n_kb = sum(1 for w in A2["write_evidence_specs"]
           if "search the knowledge base" in (w.get("feedback") or ""))
chk(n_kb >= 5, "접미사명 KB-검색 안내 문구 5+ 스펙 반영 (실제 %d)" % n_kb)

print("\n%s" % ("PASS — §2bc 픽스 배선 정상 (라이브 발화는 별도 검증·[[30]])" if ok else "FAIL"))
sys.exit(0 if ok else 1)
