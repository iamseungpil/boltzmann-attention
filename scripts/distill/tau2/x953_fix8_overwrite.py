# -*- coding: utf-8 -*-
"""x953 - FIX-8 overwrites the LLM's own declared tool (agent_names = self.tools = exposed 21)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t2_gate_patch as G

d = json.load(open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
b = d["banking_knowledge"]
ag_all = [n for n, v in b["tools"].items() if v.get("side") == "tools"]
EXPOSED = sorted(b["exposed"])                      # == [t.name for t in self.tools]
REG = sorted(set(ag_all) - set(EXPOSED))            # == _agent_discoverable(env)
USR = sorted(b.get("discoverable_user_tools") or [])

cases = [
  ("order replacement credit card",            "order_replacement_credit_card_7291"),
  ("inform customer to run apply_for_credit_card", "give_discoverable_user_tool"),
  ("Change the user's account email address",  "change_user_email"),
  ("file four debit card transaction disputes","file_debit_card_transaction_dispute_6281"),
]
print("declared_tool -> emitted_tool  (agent_names = exposed 21, registry = discoverable 44)")
for what, decl in cases:
    own, th, unk = G._split_claims_by_owner([{"what": what, "tool": decl}],
                                            EXPOSED, USR, registry=REG)
    got = own[0]["tool"] if own else ("(silenced-theirs)" if th else "(unknown)")
    print("  %-46r  decl=%-42s -> %s  %s"
          % (what, decl, got, "OVERWRITTEN" if (own and got != decl) else ""))
