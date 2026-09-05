#!/usr/bin/env python
"""x770h — does a contract predicate actually run with NO orchestrator/env?

Calls each seed predicate with hand-made declarations + a fake message list.
Pass = returns a verdict without touching tau2 runtime objects.
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


class TC:                       # minimal tool-call stand-in (name + arguments)
    def __init__(self, name, args):
        self.name = name
        self.arguments = args
        self.id = "tc1"


class M:                        # minimal message stand-in
    def __init__(self, role, content=None, tool_calls=None, requestor=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.requestor = requestor


ok = []


def t(label, fn):
    try:
        r = fn()
        ok.append((label, 'OK', repr(r)[:110]))
    except Exception as e:
        ok.append((label, 'FAIL', '%s: %s' % (type(e).__name__, e)))


import t2_gate_patch as GP
import gate_interpreter as GI

# ---- 1. _wev_deny_msgs : write-evidence contract, invented domain ------------
SPECS_WEV = [{
    "applies_to": "zz_do_thing",
    "applies_when": {"arg": "ticket_id", "prefix": "TK"},
    "require_tokens_from_arg": "ticket_id",
    "evidence_tools": ["zz_list_tickets"],
    "deny_text": "[WEV] no ledger evidence for {id}",
}]
msgs_no_ev = [M("assistant", tool_calls=[TC("zz_do_thing", {"ticket_id": "TK9"})])]
msgs_ev = [M("tool", content="ticket TK9 open"),
           M("assistant", tool_calls=[TC("zz_do_thing", {"ticket_id": "TK9"})])]
t('_wev_deny_msgs (no evidence)',
  lambda: GP._wev_deny_msgs(msgs_no_ev, TC("zz_do_thing", {"ticket_id": "TK9"}), SPECS_WEV))
t('_wev_deny_msgs (evidence present)',
  lambda: GP._wev_deny_msgs(msgs_ev, TC("zz_do_thing", {"ticket_id": "TK9"}), SPECS_WEV))

# ---- 2. _write_arg_ground_deny : value-transcription contract ---------------
SPECS_WAG = [{
    "applies_to": "zz_do_thing",
    "grounded_args": ["widget_code"],
    "deny_text": "[WAG] {arg}={val} not in any read",
}]
t('_write_arg_ground_deny (ungrounded value)',
  lambda: GP._write_arg_ground_deny(
      [M("tool", content="widgets: AAA111, BBB222")],
      TC("zz_do_thing", {"widget_code": "ZZZ999"}), SPECS_WAG))
t('_write_arg_ground_deny (grounded value)',
  lambda: GP._write_arg_ground_deny(
      [M("tool", content="widgets: AAA111, BBB222")],
      TC("zz_do_thing", {"widget_code": "AAA111"}), SPECS_WAG))

# ---- 3. _claim_unbacked : claim-vs-ledger contract --------------------------
t('_claim_unbacked (claim with no ledger event)',
  lambda: GP._claim_unbacked([{"kind": "zz_update", "what": "did the thing", "tool": "zz_do_thing"}],
                             {}, [], [M("assistant", content="I did the thing")]))

# ---- 4. GateInterpreter : whole gate engine on an INVENTED domain -----------
GATES = [{
    "kind": "auth",
    "applies_to": ["zz_do_thing"],
    "satisfied_by": ["zz_verify_caller"],
    "deny": "[AUTH] verify the caller first",
}, {
    "kind": "confirm",
    "applies_to": ["zz_do_thing"],
    "deny": "[CONFIRM] ask the customer to confirm",
}]
gi = GI.GateInterpreter(GATES, resolvers=None, enable_g2=False)
t('GateInterpreter.check (unauthed, invented domain)',
  lambda: gi.check("zz_do_thing", {"ticket_id": "TK9"}, last_user_msg="please do it"))
gi2 = GI.GateInterpreter(GATES, resolvers=None, enable_g2=False)
gi2.observe("zz_verify_caller", {}, {"ok": True}, ok=True)
t('GateInterpreter.check (after satisfier + confirm)',
  lambda: gi2.check("zz_do_thing", {"ticket_id": "TK9"}, last_user_msg="yes go ahead"))

# ---- 5. compute_facts / _apply_op on an invented catalog --------------------
import t2_compute as CC
t('gate_interpreter.compute_facts (count_where)',
  lambda: GI.compute_facts({"widgets": [{"s": "open"}, {"s": "shut"}, {"s": "open"}]},
                           [{"name": "n_open", "op": "count_where", "over": "widgets",
                             "cond_field": "s", "cond_value": "open"}]))
t('t2_compute count_where (invented schema)',
  lambda: CC._apply_op({"op": "count_where", "over": "rows", "cond_field": "s",
                        "cond_value": "open"},
                       {"rows": [{"s": "open"}, {"s": "shut"}]})
  if hasattr(CC, '_apply_op') else 'n/a')

# ---- 6. signature_violation / dominance / authority ------------------------
import t2_signature as SG
t('t2_signature.signature_violation',
  lambda: SG.signature_violation("zz_do_thing", {"bogus_arg": 1},
                                 {"zz_do_thing": {"required": ["ticket_id"],
                                                  "properties": ["ticket_id"]}}))
import t2_dominance as DM
t('t2_dominance.dominating_gate',
  lambda: DM.dominating_gate("zz_do_thing", {}, {"gates": GATES}, set()))
import t2_authority as AU
t('t2_authority.may_suppress',
  lambda: AU.may_suppress("zz_do_thing", {"suppression_authority": []}, []))

w = max(len(a) for a, _, _ in ok)
for a, b, c in ok:
    print('%-*s  %-4s  %s' % (w, a, b, c))
print()
print('OFFLINE PASS %d / %d' % (sum(1 for _, b, _ in ok if b == 'OK'), len(ok)))
