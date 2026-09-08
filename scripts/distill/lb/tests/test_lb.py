# -*- coding: utf-8 -*-
"""Battery for the LB code base. Script-style: exits 1 on the first failure.

  1. every module's own self-test passes
  2. the migrated banking declaration loads and drives the engines on the 048 / 049 shapes
  3. exactly seven lever flags exist in this code base (T2_LB1..T2_LB7)
"""

import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ["PYTHONIOENCODING"] = "utf-8"

MODULES = ["lb_coordinator", "lb1_requirements", "lb2_decision", "lb3_citation", "lb4_coverage",
           "lb5_resignation", "lb6_load", "lb7_material"]
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print("%s %s %s" % ("ok  " if cond else "FAIL", name, detail))


import glob
import py_compile
for p in sorted(glob.glob(os.path.join(ROOT, "*.py"))):      # the tau2-bound modules have no self-test;
    try:                                                       # a syntax error there killed a probe lane
        py_compile.compile(p, doraise=True)
        check("compiles " + os.path.basename(p), True)
    except py_compile.PyCompileError as e:
        check("compiles " + os.path.basename(p), False, str(e).splitlines()[-1])

for m in MODULES:
    r = subprocess.run([sys.executable, os.path.join(ROOT, m + ".py")], capture_output=True, text=True)
    check("selftest " + m, r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] if r.returncode else "")

import lb_a2
from lb_coordinator import Turn, evaluate, resolve, DENY

lb_a2.migrate("banking_knowledge")
A2 = lb_a2.load("banking_knowledge")
check("a2 sections", all(k in A2 for k in ("dispatch", "LB1", "LB2", "LB3", "LB4", "LB5", "LB6", "LB7")))
procs = {p["id"]: p for p in A2["LB1"]["procedures"]}
check("049: intent chain folded into the closure procedure (signals) and no second engine",
      "signals" in procs["credit_card_closure_retention"]["enter_when"] and '"intent_chains":' not in json.dumps(A2))
check("prescription is a procedure", any(p.startswith("prescription:") for p in procs))


class C(object):
    def __init__(self, name, args=None):
        self.name, self.arguments, self.id = name, args or {}, name


class M(object):
    def __init__(self, role="assistant", content="", calls=()):
        self.role, self.content, self.tool_calls = role, content, list(calls)


# 049: closing a card straight away, entered by the customer's own words -> one LB1 order, computed
am = M(calls=[C("close_credit_card_account_7834", {"account_id": "acct_9001"})])
t = Turn(A2, [M("user", "I want to close my credit card"), M("tool", '{"account_id": "acct_9001", "status": "open"}')], am)
d = resolve(evaluate(t), t)
body = d.denies.get(id(am.tool_calls[0]), "")
check("049: closure blocked by the procedure with the policy checklist", "[PROCEDURE]" in body and "disputes" in body, body[:80])
check("049: no conflicting second voice on the same target", not d.conflicts, d.conflicts)

# 048: the dispute signal appears only inside tool output -> the prescription procedure stays closed
call = C("call_discoverable_agent_tool", {"agent_tool_name": "apply_statement_credit_8472", "arguments": "{}"})
tool_only = [M("tool", "Suspected fraud/unauthorized transactions: see the dispute policy")]
t = Turn(A2, tool_only, M(calls=[call]), executed={"log_verification": 1})
check("048: signal in tool output does not redirect", not any(f.source.startswith("procedure:prescription") for f in evaluate(t)["LB1"]))
t = Turn(A2, [M("user", "this charge is fraudulent, I want to dispute it")], M(calls=[call]), executed={"log_verification": 1})
check("048: signal in customer text does redirect", any(f.source.startswith("procedure:prescription") for f in evaluate(t)["LB1"]))

# sub-call levers: the real declarations drive the deterministic cores offline
import json
import lb2_decision
tools = {t["name"]: t for t in A2["LB2"]["tools"]}
row = {"transaction_amount": 100, "credit_card_type": "Gold Rewards Card", "category": "Dining", "base_rate": 2.5,
       "account_open": "01/01/2020", "promo_start": "01/01/2024", "promo_end": "12/31/2024", "transaction_date": "02/01/2025",
       "promo_window_months": 3, "promo_mult": 2}          # operands the row-mode isolate fills in live runs
tx = [dict(row, transaction_id="t1", rewards_earned=250), dict(row, transaction_id="t2", rewards_earned=100)]
text, err = lb2_decision.run_tool(tools["get_reward_discrepancies"], {"transactions": json.dumps(tx)}, {"kb": [], "ledger": []}, {})
check("verifier get_reward_discrepancies flags the wrong row only", not err and "t2" in text and "t1" not in text.split("):")[-1], text[:120])
fit = tools["check_card_application_fit"]
text, err = lb2_decision.run_tool(fit, {"max_annual_fee": "0", "business": ""}, {"kb": [], "ledger": []}, {})
check("verifier check_card_application_fit filters by a stated constraint", not err and '"excluded"' in text and "Platinum Rewards Card" in text, text[:100])
vi = tools["verify_identity"]
ev = {"__user_text": "my email is a@x.com and my dob is 01/15/1985", "__tool_outputs": {"get_user_information_by_email": "email: a@x.com dob: 01/15/1985"}}
text, err = lb2_decision.run_tool(vi, {"provided": json.dumps({"email": "a@x.com", "date_of_birth": "01/15/1985"}), "record": "{}"}, {"kb": [], "ledger": []}, ev)
check("verifier verify_identity (grounded variant) verifies two matching values", not err and "VERIFIED" in text and "NOT_VERIFIED" not in text, text[:100])
check("derived DAG declared with prompts and texts", any(n.get("text") for n in A2["LB2"]["derived"]) and all(n.get("prompt") for n in A2["LB2"]["derived"] if n["op"] == "formalize"))
check("claims audit and have_value declared", any(s["kind"] == "claims" for s in A2["LB4"]["sets"]) and A2["LB7"]["have_value"])
check("no dead declaration keys", not any(set(p.get("feedback") or {}) - {"unmet"} for p in A2["LB1"]["procedures"]))
check("identifying args declared", "transaction_id" in A2["LB3"]["identifying"]["args"])

# only seven lever flags in this code base
flags = set()
for f in os.listdir(ROOT):
    if f.endswith(".py"):
        src = io.open(os.path.join(ROOT, f), encoding="utf-8").read()
        for token in src.replace('"', " ").replace("'", " ").split():
            if token.startswith("T2_") and token != "T2_" and not token.startswith("T2_LB") and token not in ("T2_FB_SIDECAR", "T2_KB_DOCS_DIR", "T2_AGENT_MAX_TOKENS"):
                flags.add(token)
check("seven lever flags only (T2_LB1..7); harness paths aside", not flags, sorted(flags))
check("no regular expressions in engines", not any("import re" in io.open(os.path.join(ROOT, m + ".py"), encoding="utf-8").read()
                                                    for m in MODULES))
lines = sum(len(io.open(os.path.join(ROOT, f), encoding="utf-8").read().splitlines())
            for f in os.listdir(ROOT) if f.endswith(".py"))
print("code base size: %d lines across %d files" % (lines, len([f for f in os.listdir(ROOT) if f.endswith('.py')])))
print("RESULT: %s (%d/%d)" % ("PASS" if all(OK) else "FAIL", sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
