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
am = M(calls=[C("close_credit_card_account_7834", {"account_id": "x"})])
t = Turn(A2, [M("user", "I want to close my credit card")], am)
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
