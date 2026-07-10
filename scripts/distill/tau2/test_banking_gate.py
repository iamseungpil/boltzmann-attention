# -*- coding: ascii -*-
"""Unit test: banking_knowledge A2 + applies_when engine extension (local, free)."""
import sys, json, io
sys.path.insert(0, r"C:\workspace\ba-frft\scripts\distill\tau2")
from gate_interpreter import GateInterpreter, load_domain_a2, auth_satisfier_tools, observe_tools

a2 = load_domain_a2("banking_knowledge")
assert a2 is not None, "A2 load failed"
g = GateInterpreter(a2["gates"])

fails = []
def chk(name, got, want):
    ok = (got == want)
    print(("PASS" if ok else "FAIL"), name, "->", got)
    if not ok: fails.append(name)

# 1. unverified: direct user-data tools denied
for t in ["change_user_email", "get_referrals_by_user",
          "get_credit_card_transactions_by_user", "get_credit_card_accounts_by_user"]:
    ok, gid, why = g.check(t, {"user_id": "u1"})
    chk("deny_unverified_" + t, (ok, gid), (False, "GB1_VERIFY_BEFORE_ACCOUNT_ACCESS"))

# 2. unverified: dispatcher with data tool denied
ok, gid, _ = g.check("call_discoverable_agent_tool",
                     {"agent_tool_name": "file_credit_card_transaction_dispute_4829", "arguments": "{}"})
chk("deny_unverified_dispatch_data", (ok, gid), (False, "GB1_VERIFY_BEFORE_ACCOUNT_ACCESS"))

# 3. unverified: transfer/incident inner tools EXEMPT (applies_when.not_in)
for inner in ["initial_transfer_to_human_agent_0218", "initial_transfer_to_human_agent_1822",
              "emergency_credit_bureau_incident_transfer_1114"]:
    ok, gid, _ = g.check("call_discoverable_agent_tool", {"agent_tool_name": inner})
    chk("allow_exempt_" + inner[-4:], (ok, gid), (True, None))

# 4. ungated tools always allowed pre-verification
for t in ["get_user_information_by_id", "get_user_information_by_name", "get_user_information_by_email",
          "log_verification", "get_current_time", "unlock_discoverable_agent_tool",
          "give_discoverable_user_tool", "transfer_to_human_agents", "list_discoverable_agent_tools"]:
    ok, gid, _ = g.check(t, {})
    chk("allow_ungated_" + t, (ok, gid), (True, None))

# 5. observe(log_verification success) -> verified -> everything allowed
g.observe("log_verification", {"user_id": "u1"}, "Verification logged successfully.\n  - User: X (ID: u1)")
ok, gid, _ = g.check("call_discoverable_agent_tool",
                     {"agent_tool_name": "file_credit_card_transaction_dispute_4829"})
chk("allow_after_verify_dispatch", (ok, gid), (True, None))
ok, gid, _ = g.check("get_credit_card_accounts_by_user", {"user_id": "u1"})
chk("allow_after_verify_direct", (ok, gid), (True, None))

# 6. helper exports
chk("auth_satisfiers", auth_satisfier_tools(a2["gates"]), {"log_verification"})
chk("observe_tools", observe_tools(a2["gates"]), {"log_verification"})

# 7. recovery message renders (no crash, mentions satisfier)
g2 = GateInterpreter(a2["gates"])
ok, gid, why = g2.check("change_user_email", {"user_id": "u1", "new_email": "a@b.com"})
assert "log_verification" in why, why
print("PASS recovery_msg:", why[:140])

# 8. retail A2 regression: applies_when absent -> unchanged behavior
r = load_domain_a2("retail")
gr = GateInterpreter(r["gates"])
ok, gid, _ = gr.check("cancel_pending_order", {"order_id": "#W1"})
chk("retail_regression_auth_deny", (ok, gid), (False, "G1_AUTH_FIRST"))

print("\n%d FAIL" % len(fails) if fails else "\nALL PASS")
sys.exit(1 if fails else 0)
