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
import tempfile
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
import lb_runtime
import lb2_decision
tools = {t["name"]: t for t in A2["LB2"]["tools"]}
row = {"transaction_amount": 100, "credit_card_type": "Gold Rewards Card", "category": "Dining", "base_rate": 2.5,
       "account_open": "01/01/2020", "promo_start": "01/01/2024", "promo_end": "12/31/2024", "transaction_date": "02/01/2025",
       "promo_window_months": 3, "promo_mult": 2}          # operands the row-mode isolate fills in live runs
tx = [dict(row, transaction_id="t1", rewards_earned=250), dict(row, transaction_id="t2", rewards_earned=100)]
text, err, _ = lb2_decision.run_tool(tools["get_reward_discrepancies"], {"transactions": json.dumps(tx)}, {"kb": [], "ledger": []}, {})
check("verifier get_reward_discrepancies flags the wrong row only", not err and "t2" in text and "t1" not in text.split("):")[-1], text[:120])
fit = tools["check_card_application_fit"]
text, err, _ = lb2_decision.run_tool(fit, {"max_annual_fee": "0", "business": ""}, {"kb": [], "ledger": []}, {})
check("verifier check_card_application_fit filters by a stated constraint", not err and '"excluded"' in text and "Platinum Rewards Card" in text, text[:100])
vi = tools["verify_identity"]
ev = {"__user_text": "my email is a@x.com and my dob is 01/15/1985", "__tool_outputs": {"get_user_information_by_email": "email: a@x.com dob: 01/15/1985"}}
text, err, _ = lb2_decision.run_tool(vi, {"provided": json.dumps({"email": "a@x.com", "date_of_birth": "01/15/1985"}), "record": "{}"}, {"kb": [], "ledger": []}, ev)
check("verifier verify_identity (grounded variant) verifies two matching values", not err and "VERIFIED" in text and "NOT_VERIFIED" not in text, text[:100])
check("derived DAG declared with prompts and texts", any(n.get("text") for n in A2["LB2"]["derived"]) and all(n.get("prompt") for n in A2["LB2"]["derived"] if n["op"] == "formalize"))
check("claims audit and have_value declared", any(s["kind"] == "claims" for s in A2["LB4"]["sets"]) and A2["LB7"]["have_value"])
import lb_a2
FB = set(lb_a2.PROC_FEEDBACK)          # the keys migration keeps; each one must also be read
check("no dead declaration keys",
      not any(set(p.get("feedback") or {}) - FB for p in A2["LB1"]["procedures"])
      and all(k in io.open(os.path.join(ROOT, "lb1_requirements.py"), encoding="utf-8").read() for k in FB))
check("identifying args declared", "transaction_id" in A2["LB3"]["identifying"]["args"])

# what a verifier settles has to reach the rule that consumes it as data. It used to be recovered by
# parsing the sentence the verifier had just rendered: records_in found nothing there, so LB4's
# settled_rows never fired once across 39 simulations that ran the tool, 14 of which ended short of
# the disputes it had found. No text is read back anywhere on this path now.
import lb2_decision as _lb2
_decl = {"op": {"op": "select_discrepant", "id_field": "transaction_id"}, "return_template": "found {ids}"}
_txt, _err, _ids = _lb2.run_tool(_decl, {}, {}, {})
check("run_tool hands the settled ids back as data", _err or _ids == [] or isinstance(_ids, list))
_txt2, _err2, _ids2 = _lb2.run_tool({"op": {"op": "x"}, "return_template": "plain"}, {}, {}, {})
check("a verifier that settles nothing hands back no ids", _ids2 == [])
check("no ids are embedded in the rendered text",
      "[ROWS]" not in io.open(os.path.join(ROOT, "lb2_decision.py"), encoding="utf-8").read())

# base-vs-us divergence inventory: every place our stack leaves tau2's path is raised as
# diverge("<kind>") and listed in lb_runtime's docstring table. If the two drift apart, a comparison
# against base is being made against code nobody enumerated.
RT = io.open(os.path.join(ROOT, "lb_runtime.py"), encoding="utf-8").read()
raised = set()
for piece in RT.split('diverge("')[1:]:
    raised.add(piece.split('"')[0])
doc = RT.split('"""')[1]
table = doc.split("kind             where")[-1].split('"""')[0]
listed = set()
for line in table.splitlines():
    tok = line.strip().split(" ")[0]
    if tok and tok[0].isalpha() and tok == tok.lower() and " " not in tok and len(tok) < 20:
        listed.add(tok)
listed = {k for k in listed if k in raised or "-" in k}
check("every divergence from base is raised and listed", raised and raised == listed,
      "raised-only %s | listed-only %s" % (sorted(raised - listed), sorted(listed - raised)))
# ...and every one of them must survive being called. A keyword that collides with sidecar's own
# parameters raises TypeError at the call site, which killed all 4 sims of bz_task_070 (2026-09-09):
# the inventory check above passed because it only reads source text, never runs it.
_sc = os.path.join(tempfile.gettempdir(), "lb_diverge_probe.jsonl")
if os.path.exists(_sc):
    os.remove(_sc)
os.environ["LB_SIDECAR"] = _sc
try:
    for _k in sorted(raised):
        lb_runtime.diverge(_k, "probe", sim="s", n=1)
    _rows = [json.loads(l) for l in io.open(_sc, encoding="utf-8")]
    check("every divergence kind can actually be raised", len(_rows) == len(raised)
          and {r.get("at") for r in _rows} == raised,
          "wrote %d rows for %d kinds" % (len(_rows), len(raised)))
except TypeError as _e:
    check("every divergence kind can actually be raised", False, str(_e))
finally:
    os.environ.pop("LB_SIDECAR", None)
    if os.path.exists(_sc):
        os.remove(_sc)
# a catalogue that excludes everything is a dead tool. check_card_application_fit returned zero
# eligible cards on every call it ever made: the invite-only constraint read a row that does not
# carry the flag as undocumented rather than unrestricted, so all seven personal cards sat in
# 'unverified'. Asked with nothing constrained, a catalogue must offer something.
for _t in json.load(io.open(os.path.join(ROOT, "a2", "banking_knowledge.specific.json"),
                            encoding="utf-8")).get("scaffold_get_tools") or []:
    if (_t.get("op") or {}).get("op") != "catalog_filter":
        continue
    _r = lb2_decision.evaluate_op(_t["op"], {})
    check("catalogue %s offers something when nothing is constrained" % _t["name"],
          bool(_r and _r.get("eligible")),
          "%d eligible of %d rows" % (len((_r or {}).get("eligible") or []), len(_t["op"].get("table") or [])))
    # a rank that evaluates to None on every row leaves the catalogue in table order while looking
    # like it is sorted: check_referral_options ranked on a plain column and evaluate_op returned
    # None for all eighteen rows. Score the rank against the rows directly, with every number the
    # tool asks for supplied, so a rank that needs an amount is judged on one.
    if _t["op"].get("rank"):
        _ctx = {p: 1 for p, d in (_t.get("params") or {}).items() if str(d).startswith("number")}
        _scored = [lb2_decision.val(dict(_ctx, r=_row), _t["op"]["rank"])
                   for _row in _t["op"].get("table") or []]
        check("catalogue %s actually ranks" % _t["name"],
              any(v is not None for v in _scored),
              "all %d rows scored None" % len(_scored))
        # a catalogue must not present an order it could not compute. task_067 called
        # check_checking_account_fit with every field empty; six accounts came back in catalogue
        # order carrying a score of None, the model opened the third, and the answer was the sixth.
        # With nothing to rank on, the result has to say so, drop the empty score, and keep every
        # row - `top` cuts an unordered list at an arbitrary place.
        _bare = lb2_decision.evaluate_op(_t["op"], {}) or {}
        _rows = _bare.get("eligible") or []
        _k = _t["op"].get("rank_field", "score")
        if _rows and all(_e.get(_k) is None for _e in _rows):
            check("catalogue %s says so when it cannot rank" % _t["name"],
                  "NOT RANKED" in (_bare.get("note") or "")
                  and not any(_k in _e for _e in _rows)
                  and ("cut to the first" in (_bare.get("note") or "")
                       or len(_rows) < (_t["op"].get("top") or 99) + 1),
                  "presented %d rows in table order as though ranked" % len(_rows))
        # scoring the table row is not the test: the caller is handed facts, and keep_fields trims
        # them. check_referral_options ranked on combined_bonus, keep_fields dropped that column, and
        # every eligible row came back with a rank of None in catalogue order while the tool went on
        # telling the caller the rows were sorted.
        _out = lb2_decision.evaluate_op(_t["op"], dict(_ctx))
        _elig = (_out or {}).get("eligible") or []
        _key = _t["op"].get("rank_field", "score")
        check("catalogue %s ranks the rows it hands back" % _t["name"],
              not _elig or any(_e.get(_key) is not None for _e in _elig),
              "all %d eligible rows scored None on %s" % (len(_elig), _key))

# a rule authored as data but never wired to an engine is a rule that does not exist.
# free_text_defaults sat in the declaration from 2026-08-31, naming the very tasks it was measured
# on, and reached no engine: every one of them still failed the same way on 2026-09-10.
_A2 = json.load(io.open(os.path.join(ROOT, "a2", "banking_knowledge.lb.json"), encoding="utf-8"))
_SRC = json.load(io.open(os.path.join(ROOT, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
_wired = {(g.get("when", {}).get("prefix"), g.get("arg"))
          for g in (_A2.get("LB3") or {}).get("grounding") or []}
_want = {(t, a) for t, args in (_SRC.get("free_text_defaults") or {}).items() for a in args}
check("every free-text default reaches an engine", _want and _want <= _wired,
      "declared %s, wired %s" % (sorted(_want), sorted(_want & _wired)))

# the same failure again: the action index was derived from the environment, measured in isolation
# at 10/24 -> 24/24, and appeared zero times in the built A2 while the model grepped for what it
# already had. Anything the ontology carries for the model to read must reach an engine.
_ai = (_A2.get("LB7") or {}).get("action_index") or {}
_po = _SRC.get("policy_ontology") or {}
check("the action index reaches an engine",
      bool(_po.get("action_index")) == bool(_ai.get("rows"))
      and len(_ai.get("rows") or []) == len(_po.get("action_index") or []),
      "declared %d rows, wired %d" % (len(_po.get("action_index") or []), len(_ai.get("rows") or [])))

# Two rules were authored, measured, and silently absent from the live stack - the free-text
# default and the action index, each with the tasks it was measured on written beside it. Nothing
# failed because nothing was checking. A declaration key either reaches an engine or is entered in
# the migration register with a verdict; "pending" is a verdict, and the debt prints on every run.
_decl = {}
for _part in ("settings", "specific"):
    _p = os.path.join(ROOT, "a2", "banking_knowledge.%s.json" % _part)
    if os.path.exists(_p):
        _decl.update(json.load(io.open(_p, encoding="utf-8")))
_code = chr(10).join(io.open(_f, encoding="utf-8").read() for _f in glob.glob(os.path.join(ROOT, "*.py")))
_unwired = sorted(k for k, v in _decl.items() if not k.startswith("_") and v
                  and ('"%s"' % k not in _code and "'%s'" % k not in _code))
_reg = _decl.get("_note_migration") or {}
_unaccounted = [k for k in _unwired if k not in _reg]
check("every declaration key reaches an engine or the migration register", not _unaccounted,
      "unaccounted: %s" % _unaccounted)
_nodekeys = {k for p in _decl.get("procedures") or [] for n in p.get("nodes") or [] for k in n
             if not k.startswith("_")} | {g.get("id") for g in _decl.get("gates") or []}
_stale = sorted(k for k in _reg if k not in _unwired and k not in _nodekeys)
check("the migration register holds nothing already wired", not _stale, "stale: %s" % _stale)
_pending = sorted(k for k, v in _reg.items() if (v or {}).get("verdict") == "pending")
if _pending:
    print("     debt: %d declarations measured and not yet wired - %s"
          % (len(_pending), " ".join(_pending)))


# We must not hand the agent the name of a tool it cannot call. check_card_application_fit's
# description ended "The card_type you then pass to apply_for_credit_card ...", and that tool is the
# customer's - declared in this same file as recommendation_verify.action_tool. task_023 sim1 spent
# 82 shell commands and 61 searches looking for it and ran out of steps at 261 messages; base, which
# never sees the name, hunts zero times in four.
_theirs = {(_decl.get("recommendation_verify") or {}).get("action_tool")} | {
    (_v or {}).get("user_tool") for _v in (_decl.get("arg_producers") or {}).values()}
_theirs.discard(None)
_named = sorted({(_t.get("name"), _u) for _t in _decl.get("scaffold_get_tools") or []
                 for _u in _theirs if _u in (_t.get("description") or "")})
check("our tool descriptions never name a tool the agent cannot call", not _named,
      "; ".join("%s names %s" % (a, b) for a, b in _named))

# A description that commands is the engine deciding. task_007: base holds none of these tools,
# applies for the card on its own and passes 4/4; every simulation of ours called the catalogue,
# and three of four then compared cards for the customer and applied for nothing. When to compare
# is the model's judgement, so a description states the condition and the model chooses.
_cmd = sorted(t["name"] for t in _decl.get("scaffold_get_tools") or []
            if "MANDATORY" in (t.get("description") or ""))
check("our tool descriptions offer a tool, they do not command it", not _cmd,
      "commanded: %s" % _cmd)

# A procedure step its own source makes conditional must not be enforced when the condition holds.
# task_049: the retention protocol says "If records exist for this account within that time frame,
# skip retention offers and proceed directly to processing the closure." The Green card carried such
# a record, the model read it and said it would close directly, and we denied the closure until it
# logged a reason - a row on an account gold never touches, four simulations out of four. The two
# environment texts below are verbatim from that run, and the gate wants the requirement waived on
# the one and standing on the other.
import lb1_requirements
from lb_coordinator import Turn as _Turn


class _C(object):
    def __init__(self, name, args=None, cid=None):
        self.name, self.arguments, self.id = name, args or {}, cid or name


class _M(object):
    # a tool message carries the id of the call it answers; a check that pairs the two needs it
    def __init__(self, role="assistant", content="", calls=(), mid=None):
        self.role, self.content, self.tool_calls, self.id = role, content, list(calls), mid


_HAVE = ("Closure reason history for credit card account cc_x_green:" + chr(10) + "Found 1 record(s) in 'credit_card_closure_reasons':" + chr(10) + "" + chr(10) + "1. Record ID: clsr_x_green_001" + chr(10) + "   credit_card_account_id: cc_x_green" + chr(10) + "   closure_reason: not_using_card" + chr(10) + "   status: LOGGED")
_NONE = ("Closure reason history for credit card account cc_x_crypto:" + chr(10) + "" + chr(10) + "No closure reason records found for this credit card account.")
_A2 = A2
_RUN = {"get_user_dispute_history_7291": 1, "get_pending_replacement_orders_5765": 1,
        "get_closure_reason_history_8293": 1}


def _ran(run, card):
    """The run record with each call's arguments - what a per-subject procedure counts."""
    return [(k, {"credit_card_account_id": card}) for k, v in run.items() for _ in range(v)]


def _closes(card, output, on=None):
    _call = _C("call_discoverable_agent_tool",
               {"agent_tool_name": "close_credit_card_account_7834",
                "arguments": {"credit_card_account_id": card}})
    _t = _Turn(_A2, [_M("tool", output)], _M(calls=[_call]), executed=dict(_RUN),
               ran=_ran(_RUN, on or card))
    return [f for f in lb1_requirements.procedure_findings(_t, _call) if f.primitive == lb1_requirements.DENY]


# migration keeps procedure node keys by whitelist, so a key added to the declaration is dropped in
# silence: skip_when_tokens was declared, the engine read it, and the node it arrived on no longer
# carried it. Every key the declaration puts on a node has to survive the crossing.
_kept = {k for p in (A2.get("LB1") or {}).get("procedures") or [] for n in p.get("nodes") or [] for k in n}
_declared = {k for p in _decl.get("procedures") or [] for n in p.get("nodes") or [] for k in n
             if not k.startswith("_")}
_lost = sorted(_declared - _kept - set(_decl.get("_note_migration") or {}))
check("migration keeps every declared procedure node key or the register carries it", not _lost,
      "dropped: %s" % _lost)

check("a procedure step its source waives is not enforced", not _closes("cc_x_green", _HAVE),
      "denied the closure of a card whose history already holds a record")
check("the same step still stands where the source does not waive it",
      bool(_closes("cc_x_crypto", _NONE)), "no denial where no record exists")

# A gate is a rule, and migration turns it into a tool-prerequisite edge - which it can only do
# when the gate names tools that satisfy it. GB2 is satisfied by sending the customer a sentence,
# so it has no satisfiers, so it was skipped, so the ask-first notice before a transfer does not
# exist here. task_088 transferred with no notice and no work done. A gate either reaches the
# procedures or is entered in the register.
# A gate satisfied by an utterance has to block, not merely mention: GB2 was dropped entirely at
# migration because the loop could only build a tool prerequisite, and task_088 transferred with the
# notice never sent and none of the three writes gold performs done first.
_NOTICE = "TRANSFER NOTICE: Would you like to be transferred to a human agent?"


def _transfer(said):
    _c = _C("transfer_to_human_agents", {"reason": "fraud_or_security_concern"})
    _hist = [_M("user", "help")] + ([_M("assistant", "Sure. " + said)] if said else [])
    _t = _Turn(_A2, _hist, _M(calls=[_c]))
    return [f for f in lb1_requirements.procedure_findings(_t, _c)
            if f.primitive == lb1_requirements.DENY]


_before = _transfer("")
check("a notice gate blocks the action until the notice is on the record",
      bool(_before) and _NOTICE in (_before[0].order or ""),
      "no deny carrying the exact sentence")
check("and stops blocking once it has been said", not _transfer(_NOTICE))

_gate_ids = {g.get("id") for g in _decl.get("gates") or []}
_reached = {str(p.get("id", "")).split(":")[0] for p in (A2.get("LB1") or {}).get("procedures") or []}
_lost = sorted(_gate_ids - _reached - set(_decl.get("_note_migration") or {}))
check("every declared gate reaches the procedures or the register", not _lost, "lost: %s" % _lost)

# An order the procedure states is an edge, not a sentence. The retention protocol says closure
# proceeds only if the customer declines the offer, and outcome held the remedy, the offer and the
# closure in one tool_any node - so the order was nowhere in the graph and 048/049 closed without
# offering. The three cases the document names have to come out of the DAG itself.
_NL = chr(10)
_REC = ("Closure reason history for credit card account cc_x_green:" + _NL
        + "Found 1 record(s) in 'credit_card_closure_reasons':" + _NL
        + "   credit_card_account_id: cc_x_green")
_NONE = ("Closure reason history for credit card account cc_x_crypto:" + _NL
         + "No closure reason records found for this credit card account.")
_RUN = {"get_user_dispute_history_7291": 1, "get_pending_replacement_orders_5765": 1,
        "get_closure_reason_history_8293": 1, "log_credit_card_closure_reason_4521": 1}


def _closure(card, hist, yes=(), on=None, offered=False):
    """yes: which of the step questions the isolated sub-call answers YES to.

    offered: whether the retention offer was actually applied with its tool. Step 5 is a tool in the
    source - task_049's gold applies a $5 statement credit - so saying it is not doing it.
    """
    def _ask(prompt, name="lb_ask"):
        return "YES" if any(k in prompt for k in yes) else "NO"

    _c = _C("call_discoverable_agent_tool",
            {"agent_tool_name": "close_credit_card_account_7834",
             "arguments": {"credit_card_account_id": card}})
    _run = dict(_RUN)
    if offered:
        _run["apply_statement_credit_8472"] = 1
    _t = _Turn(_A2, [_M("tool", hist), _M("assistant", "we have been talking")], _M(calls=[_c]),
               executed=_run, ran=_ran(_run, on or card), extras={"ask": _ask})
    return [f for f in lb1_requirements.procedure_findings(_t, _c)
            if f.primitive == lb1_requirements.DENY]


check("closure waits for the concern and the offer", bool(_closure("cc_x_crypto", _NONE)))
check("the offer alone does not clear it - the concern comes first",
      bool(_closure("cc_x_crypto", _NONE, yes=("Step 5",))))
check("and proceeds once both are on the record",
      not _closure("cc_x_crypto", _NONE, yes=("Step 4",), offered=True))
check("the step question goes to a sub-call, not to a word match",
      "done_when" in io.open(os.path.join(ROOT, "lb1_requirements.py"), encoding="utf-8").read()
      and "said_any" not in io.open(os.path.join(ROOT, "lb1_requirements.py"), encoding="utf-8").read())
check("and proceeds with neither where the source skips retention",
      not _closure("cc_x_green", _REC))


# task_049's customer holds four cards. The smoke closed the Green card legitimately - its history
# already held a record, so the source waived the retention steps - and from that call onward every
# node of this procedure read as done for the Crypto card as well, so the graph let the Crypto
# closure through with its own steps never taken. A procedure that names its subject runs once per
# subject.
check("another card's steps do not satisfy this card's",
      bool(_closure("cc_x_crypto", _NONE, yes=("Step 4",), offered=True, on="cc_x_green")))
check("and the record that does belong to this card still clears it",
      not _closure("cc_x_crypto", _NONE, yes=("Step 4",), offered=True))


# The protocol answers five reasons and the customer's need not be one of them. On the fixed graph
# task_049 ran four simulations: the three that logged simplifying_finances were denied three or four
# times each and all failed, and the one that logged annual_fee drew no denial and passed. A step the
# source has no branch for cannot be a step the graph waits on for ever - and whether the reason is
# one of the five is the sub-call's judgement, not this engine's.
check("a reason the source prescribes no answer for does not deadlock the closure",
      not _closure("cc_x_crypto", _NONE, yes=("prescribes no response",), offered=True))


# task_049's gold applies the retention offer - apply_statement_credit_8472, $5.00, reason
# retention_offer - and then transfers, leaving the Crypto card open. Ours offered the credit in
# words, the customer declined, and the closure went through behind it: eight simulations of
# eight closed a card gold leaves open. A step the source performs with a tool is not done by
# saying it.
check("an offer the source makes with a tool is not satisfied by saying it",
      bool(_closure("cc_x_crypto", _NONE, yes=("Step 4", "Step 5"))))


# A policy sentence carried to a write must survive the exit. Every write in this domain is
# reached through a dispatcher, so the call's own name is call_discoverable_agent_tool; the advice
# window compared that name against the target and was shut for all of them. 085 logged sixteen
# conflict lines for its write rules and emitted none of them, four simulations out of four.
import lb7_material
import lb_coordinator


def _reaches(inner, holds=True):
    """holds: what the isolated sub-call answers when a rule states its own condition."""
    _c = _C("call_discoverable_agent_tool", {"agent_tool_name": inner, "arguments": {}})
    _t = _Turn(_A2, [_M("tool", "the record"), _M("user", "go ahead")], _M(calls=[_c]),
               extras={"ask": lambda prompt, name="": "YES" if holds else "NO"})
    return lb_coordinator.say(_t, lb7_material.write_rules(_t)).advice


_rules = (_A2.get("LB7") or {}).get("write_rules") or []
def _as_registered(name):
    # the environment serves these with a numeric suffix; a rule may be declared with or without one
    return name if name.rsplit("_", 1)[-1].isdigit() else name + "_0000"


_mute = sorted({r["applies_to"] for r in _rules
                if r.get("text") and not _reaches(_as_registered(r["applies_to"]))})
# A rule whose sentence states a condition ("... when multiple duplicates exist") was handed over
# whether the condition held or not. task_036 has two charges from different merchants and a gold
# that files no dispute at all; of twelve simulations with a reward, the two that passed had
# received neither this sentence nor the acquire advice and all ten that failed had received one.
_cond = [r for r in _rules if r.get("when")]
_leak = sorted({r["applies_to"] for r in _cond
                if any(r["text"] in a for a in _reaches(_as_registered(r["applies_to"]), holds=False))})
_held = sorted({r["applies_to"] for r in _cond
                if not any(r["text"] in a for a in _reaches(_as_registered(r["applies_to"])))})
check("a write rule reaches the model on the turn that reaches for its tool", not _mute,
      "silent for: %s" % _mute)

check("a rule that states its own condition stays back when the condition does not hold",
      bool(_cond) and not _leak, "leaked: %s" % _leak)
check("and reaches the model when it does", bool(_cond) and not _held,
      "withheld: %s" % _held)


# The acquire advice names the tool that produces a value. Its condition was first written about the
# customer's intent - "have they asked to dispute a charge" - and task_036 still drew it twice: that
# customer reports two fraudulent charges and asks for a reissue, and a fraud report reads as a
# dispute. A condition about the value carries the argument's own name into the question.
_acq = [v for v in _decl.get("value_acquisition") or [] if v.get("when")]
check("the acquire condition asks about the value, not about the intent",
      bool(_acq) and all("{arg}" in v["when"] for v in _acq))
_built = [h for h in (A2.get("LB7") or {}).get("have_value") or [] if h.get("acquire_when")]
check("and it survives migration onto the spec that carries that argument",
      bool(_built) and all(h.get("arg") for h in _built))


# LB5's contract says to name what the ledger can still show as open at the moment the model leaves,
# and only two kinds were declared - an unread transfer document and an exhausted search. task_016's
# customer asks for a friend's purchase to be put through; the agent verifies the caller with base's
# own arguments, researches the referral terms, and leaves having explained. base writes it 4/4 and
# our arms wrote it 0/4, 1/4, 1/4, 3/4. Neither that tool nor 007's occurs in any of the 698
# documents, so the step cannot be a declared procedure node without transcribing gold.
import lb5_resignation


def _leaves(said, answer, calls=(), once=None, user="please put the purchase through"):
    _t = _Turn(_A2, [_M("user", user)], _M(content=said, calls=list(calls)),
               ran=[("log_verification", {})],
               extras={"ask": lambda prompt, name="": answer,
                       "once": set() if once is None else once})
    return [f for f in lb5_resignation.evaluate(_t)]


_open = _leaves("I hope that helps. Have a good day.", "the friend's $750 purchase was never submitted")
check("what the customer asked for and the record does not show is named as the model leaves",
      bool(_open) and any("never submitted" in f for f in _open[0].facts))
check("and nothing is said when the sub-call finds nothing open",
      not _leaves("I hope that helps. Have a good day.", "NONE"))
_seen = set()
_first = _leaves("Have a good day.", "the purchase was never submitted", once=_seen)
_again = _leaves("Have a good day.", "the purchase was never submitted", once=_seen)
check("it is said once in a simulation, not at every departure", bool(_first) and not _again)
check("and not while the model is still working",
      not _leaves("Let me look that up for you.", "the purchase was never submitted",
                  calls=[_C("KB_search_bm25", {"query": "x"})]))


# task_023's runaways are one shape: the customer asks how to apply for a card that is invitation-only
# and has no application tool, and the agent searches for it - the same query sixteen times, the same
# document eleven - for a hundred assistant turns until the step cap ends the simulation. The
# exhaustion test on file looks for an empty result and every one of those came back full. Over base's
# 388 simulations one identical retrieval result never comes back a fourth time; ours reach four, five
# and six, and both runaways are in that group.
def _loop(times, tool="KB_search_bm25"):
    msgs = []
    for i in range(times):
        c = _C(tool, {"query": "diamond elite apply"}, cid="s%d" % i)
        msgs += [_M(calls=[c]), _M("tool", "the same nine documents", mid="s%d" % i)]
    nxt = _C(tool, {"query": "diamond elite apply"}, cid="next")
    return [f for f in lb5_resignation.evaluate(_Turn(_A2, msgs, _M(calls=[nxt])))
            if f.primitive == lb5_resignation.DENY]


_cap = int((_A2.get("LB5") or {}).get("repeat_cap") or 0)
check("a retrieval that keeps handing back the same result is stopped", bool(_loop(_cap)))
check("and one below the cap is not", not _loop(_cap - 1))
check("the cap is the count base never reaches", _cap >= 4)
check("and a tool that is not a retrieval is left alone",
      not _loop(_cap + 2, tool="get_user_information_by_name"))
check("and stays quiet on a tool it was not written for",
      not _reaches("get_user_information_by_name"))
# The register above sees only the top level. conditional_fields sat four levels down, inside the
# card catalogue's op, declared 2026-07-25 and read by nothing: task_003 held the premium
# subscription that zeroes the Silver card's foreign fee, and the catalogue excluded the one card
# the task wanted, four simulations out of four. Every key an op carries beside its own name is a
# directive the engine is supposed to obey, so every one of them has to appear in the engines.
_directives = {}


def _walk_ops(node, where):
    if isinstance(node, dict):
        if isinstance(node.get("op"), str):
            for _k in node:
                if _k != "op" and not _k.startswith("_"):
                    _directives.setdefault(_k, set()).add(where + "/" + node["op"])
        for _k, _v in node.items():
            _walk_ops(_v, where + "/" + str(_k))
    elif isinstance(node, list):
        for _v in node:
            _walk_ops(_v, where)


for _part in ("settings", "specific"):
    _p = os.path.join(ROOT, "a2", "banking_knowledge.%s.json" % _part)
    if os.path.exists(_p):
        _walk_ops(json.load(io.open(_p, encoding="utf-8")), _part)
_dead_ops = sorted(k for k in _directives
                   if '"%s"' % k not in _code and "'%s'" % k not in _code)
check("every op directive reaches an engine", not _dead_ops,
      "; ".join("%s (%s)" % (k, sorted(_directives[k])[0]) for k in _dead_ops))

# with no lever on, our stack must not touch the model at all
check("levers off delegates to tau2", "if not any_lever():" in RT and RT.count("if not any_lever():") >= 2
      and "_ORIG_TURN(self, message, state)" in RT and "orig_exec(self, tool_calls)" in RT)

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
