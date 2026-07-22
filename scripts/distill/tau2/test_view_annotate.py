# -*- coding: utf-8 -*-
"""T2_VIEW_ANNOTATE(_annotate_view·2026-07-22 §2bs) 오프라인 검증.

rall10 054 실측: 교체주문 후 뷰의 status: CLOSED(舊카드)·account_status: ACTIVE 병존을
"계좌 폐쇄"로 오독→CLI 전면 거부. 라이브 함수(_annotate_view)+실제 A2 선언을 그대로 잰다
([[03b]] 별도구현 금지). 검정: ①공존 → 주석 append(원문 보존) ②부분 일치 → 무간섭
③에러 메시지 → 무간섭 ④이미 주석된 뷰 → 이중 append 없음 ⑤비-tool → 무간섭.
⚠️단위통과≠라이브발화([[30]]) — 라이브 검정=rall11 로그 [T2_VIEW_ANNOTATE] 태그.
"""
import json
import os
import sys
import types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
def mkmod(n):
    m = types.ModuleType(n); sys.modules[n] = m; return m
mkmod("tau2"); mkmod("tau2.agent"); la = mkmod("tau2.agent.llm_agent")
la.generate = lambda **kw: None
class LLMAgent: pass
la.LLMAgent = LLMAgent
mkmod("tau2.data_model"); msgmod = mkmod("tau2.data_model.message")
class _M:
    def __init__(self, **kw): self.__dict__.update(kw)
msgmod.ToolMessage = msgmod.UserMessage = msgmod.MultiToolMessage = _M
msgmod.ToolCall = None
mkmod("tau2.orchestrator"); oo = mkmod("tau2.orchestrator.orchestrator")
class BaseOrchestrator: pass
oo.BaseOrchestrator = BaseOrchestrator
from t2_gate_patch import _annotate_view  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
SPECS = A2["view_field_annotations"]
NOTE = SPECS[0]["note"]

class Msg:
    def __init__(self, role, content, error=False):
        self.role, self.content, self.error = role, content, error

VIEW_054 = ("Found 1 record(s) in 'credit_card_accounts':\n"
            "   account_id: cc_x_gold\n   account_status: ACTIVE\n"
            "   status: CLOSED\n   closed_date: 11/14/2025")

FAILS = []
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: FAILS.append(n)

# ① 공존 → append·원문 보존
out, n = _annotate_view([Msg("tool", VIEW_054)], SPECS)
check("V1_annotated", n == 1 and NOTE in out[0].content and VIEW_054 in out[0].content)
# ② 부분 일치(ACTIVE만) → 무간섭
plain = VIEW_054.replace("status: CLOSED", "reward_points: 5")
out, n = _annotate_view([Msg("tool", plain)], SPECS)
check("V2_partial_untouched", n == 0 and out[0].content == plain)
# ③ 에러 메시지 → 무간섭
out, n = _annotate_view([Msg("tool", VIEW_054, error=True)], SPECS)
check("V3_error_untouched", n == 0)
# ④ 이미 주석된 뷰 → 이중 append 없음
once, _ = _annotate_view([Msg("tool", VIEW_054)], SPECS)
twice, n2 = _annotate_view(once, SPECS)
check("V4_idempotent", n2 == 0 and twice[0].content.count("[FIELD NOTE]") == 1)
# ⑤ 비-tool → 무간섭
out, n = _annotate_view([Msg("assistant", VIEW_054)], SPECS)
check("V5_nontool_untouched", n == 0 and out[0].content == VIEW_054)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILS: %s" % FAILS))
sys.exit(1 if FAILS else 0)
