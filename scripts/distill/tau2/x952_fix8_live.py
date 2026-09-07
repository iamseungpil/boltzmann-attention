# -*- coding: utf-8 -*-
"""x952 - live FIX-8 mispick reproduction (real what-strings pulled from fb_*.jsonl)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t2_gate_patch as G

d = json.load(open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
b = d["banking_knowledge"]
ag = [n for n, v in b["tools"].items() if v.get("side") == "tools"]
REG = sorted(set(ag) - set(b["exposed"]))
USR = sorted(b.get("discoverable_user_tools") or [])
print("registry(agent discoverable) =", len(REG))

live = [
 ("task_004 r=0.0", "Change the user's account email address"),
 ("task_020 r=1.0", "Review rewards and cash back transactions"),
 ("t7xxx",          "inform customer to run apply_for_credit_card"),
 ("t7xxx",          "guide customer to apply for credit card"),
 ("t7xxx",          "check ATM fees for Light Green Account"),
 ("t7xxx",          "recommend checking account class"),
 ("t7xxx",          "retrieve debit card dispute history"),
 ("t7xxx",          "link savings account for overdraft protection"),
 ("t7xxx",          "deposit $8,000 into savings account"),
 ("t7xxx",          "ask for full name or user ID"),
 ("t7xxx",          "enable paper statements for Gold Savings Account"),
]
for tag, w in live:
    own, th, unk = G._split_claims_by_owner([{"what": w, "tool": None}], set(ag), set(USR),
                                            registry=REG)
    m = G._tok_overlap(w, REG, stem=True)
    print("%-14s %-50r -> %s  (cands=%d hits=%s)"
          % (tag, w, (own[0]["tool"] if own else "(unknown)"), len(m),
             G._tok_hits(w, m[0]) if len(m) == 1 else "-"))

# LLM 이 옳은 이름을 선언해도 discoverable 이면 덮어쓴다는 것을 보인다
print()
c = {"what": "inform customer to run apply_for_credit_card", "tool": "order_debit_card_5739"}
own, th, unk = G._split_claims_by_owner([c], set(ag), set(USR), registry=REG)
print("declared tool=order_debit_card_5739 -> emitted tool =",
      own[0]["tool"] if own else "(unknown)")
