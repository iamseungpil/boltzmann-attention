# -*- coding: utf-8 -*-
"""x915 — A1 감사 프로브: _have_value_reask_fb / _value_acquire_fb 술어를 실물 문자열로 돌린다.
   [[74]] reports/ 신설 아님 · xNNN_ 프로브."""
import sys, io, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import t2_gate_patch as G

class TC:
    def __init__(self, name, args=None, tid="x"):
        self.name = name; self.arguments = args or {}; self.id = tid; self.requestor="assistant"
class M:
    def __init__(self, role, content="", tool_calls=None, error=False, mid=None):
        self.role=role; self.content=content; self.tool_calls=tool_calls; self.error=error; self.id=mid

SPEC = json.load(open("a2/split/banking_knowledge.discard.json", encoding="utf-8"))
HV = SPEC["have_value_reask"]
VA = SPEC["value_acquisition"]
print("HV signals:", HV[0]["reask_signals"])

PROD_OUT = M("tool", "Executed: get_card_last_4_digits\nLast 4 digits of card: 1652", mid="t1")

# ── 케이스: assistant 발화 축자들 ──────────────────────────────────────────
CASES = [
 ("040 실측형(소유 선언)",
  "I have all the dispute details and the last 4 digits. I need the internal transaction IDs "
  "for the 8 charges before I can file each dispute."),
 ("소유 선언 2",
  "Thanks! I've confirmed the last 4 digits (1652). Now filing the disputes."),
 ("보고형(질문 아님)",
  "The dispute has been filed for the charge ending in the last 4 digits 1652."),
 ("진짜 재요청",
  "Could you please tell me the last 4 digits of your credit card?"),
 ("완전 무관 — 금액 4자리",
  "The statement shows 4 digits after the decimal point in the FX rate."),
 ("완전 무관 — 계좌 마지막 4",
  "I can see your checking account ending in the last 4 digits 8842; is that the one?"),
 ("거절/설명형",
  "I cannot look up the last 4 digits myself - the system does not expose them to me."),
]

def run_hv(cur_text, cur_calls=(), prior_text=None, prod=True):
    msgs=[]
    if prior_text is not None:
        msgs.append(M("assistant", prior_text))
    if prod: msgs.append(PROD_OUT)
    am = M("assistant", cur_text, tool_calls=[TC(n) for n in cur_calls])
    return G._have_value_reask_fb(am, msgs, HV)

print("\n=== HV: prior = 같은 발화(단일 턴 실측 재현) ===")
for name, txt in CASES:
    fb = run_hv(txt, prior_text=txt)
    print("%-28s fired=%-5s" % (name, bool(fb)))

print("\n=== HV: prior = 진짜 재요청 1회, 현재 = 위 발화 ===")
REAL_ASK = "What are the last 4 digits of your card?"
for name, txt in CASES:
    fb = run_hv(txt, prior_text=REAL_ASK)
    print("%-28s fired=%-5s" % (name, bool(fb)))

print("\n--- 발화 문면(040형) ---")
print(run_hv(CASES[0][1], prior_text=CASES[0][1]))

print("\n=== VA (값 미실재) ===")
def run_va(cur_text, prior_text=None):
    msgs=[]
    if prior_text is not None: msgs.append(M("assistant", prior_text))
    am = M("assistant", cur_text, tool_calls=[])
    return G._value_acquire_fb(am, msgs, VA, a2=None, executed=set())
for name, txt in CASES:
    fb = run_va(txt, prior_text=None)
    print("%-28s fired=%-5s" % (name, bool(fb)))
