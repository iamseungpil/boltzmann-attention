# -*- coding: utf-8 -*-
"""x916 — 누락(miss) 측: 진짜 재요청인데 signals 가 못 잡는 문면."""
import sys, io, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import t2_gate_patch as G
class TC:
    def __init__(self,name,args=None,tid="x"): self.name=name; self.arguments=args or {}; self.id=tid
class M:
    def __init__(self,role,content="",tool_calls=None,error=False,mid=None):
        self.role=role; self.content=content; self.tool_calls=tool_calls; self.error=error; self.id=mid
SPEC=json.load(open("a2/split/banking_knowledge.discard.json",encoding="utf-8"))
HV=SPEC["have_value_reask"]; VA=SPEC["value_acquisition"]
PROD=M("tool","Executed: get_card_last_4_digits\nLast 4 digits of card: 1652",mid="t1")
MISS=[
 "Could you confirm the final four numbers printed on your credit card?",
 "What are the ending digits of the card you used for these charges?",
 "Please provide the card number's last portion so I can verify it.",
 "I still need to verify your card before filing - what is the card identifier?",
 "Can you read me the four numbers at the end of the card?",
 "To proceed I need the trailing 4 characters of the card.",
]
def hv(cur, prior):
    msgs=[M("assistant",prior), PROD]
    return G._have_value_reask_fb(M("assistant",cur,tool_calls=[]), msgs, HV)
def va(cur, prior):
    return G._value_acquire_fb(M("assistant",cur,tool_calls=[]), [M("assistant",prior)], VA, a2=None, executed=set())
print("=== 누락: 진짜 재요청인데 미발화? (HV / VA) ===")
for t in MISS:
    print("HV=%-5s VA=%-5s | %s" % (bool(hv(t,t)), bool(va(t,t)), t))
