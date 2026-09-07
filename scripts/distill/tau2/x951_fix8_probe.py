# -*- coding: utf-8 -*-
"""x951 - FIX-8 owner-recovery refutation probe (registry = agent DISCOVERABLE only)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t2_gate_patch as G

d = json.load(open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
b = d["banking_knowledge"]
ag = [n for n, v in b["tools"].items() if v.get("side") == "tools"]
exposed = set(b["exposed"])
REG = sorted(set(ag) - exposed)          # live: _agent_discoverable(env)
ALL59 = sorted(ag)                        # probe x902f used this
USR = sorted(b.get("discoverable_user_tools") or [])
print("REG(discoverable agent) =", len(REG))
print("ALL59 =", len(ALL59))

cases = [
 "checking the account status for the customer",
 "I will review your checking account activity",
 "close the loop on your credit card account",
 "look into the referral bonus for your account",
 "I will transfer you to a human agent",
 "I will file a dispute for the Starbucks charge",
]
for name, reg in (("REG38", REG), ("ALL59", ALL59)):
    print("\n=== registry =", name)
    for q in cases:
        own, theirs, unk = G._split_claims_by_owner(
            [{"what": q, "tool": None}], set(ag), set(USR), registry=reg)
        m = G._tok_overlap(q, reg, stem=True)
        print("  %-48r -> own=%s  overlap=%s hits=%s"
              % (q, [c.get("tool") for c in own], m[:4],
                 (G._tok_hits(q, m[0]) if len(m) == 1 else "-")))
