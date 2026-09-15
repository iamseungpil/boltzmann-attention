# -*- coding: utf-8 -*-
"""LB3 - citation check (mechanism F2, source axis).

One rule: every value or name the model writes must exist in a source this conversation holds.
The engine never produces a value; it only asks "where is this from?". Declared in A2["LB3"]:

  grounding [{applies_to, when{arg, prefix}, arg, sources[records|customer], field, state, feedback}]
      the value of `arg` must occur in a source text; with `field`, the record's `field` value must be
      one the customer said (reference verification); with `state`, a record naming the value must
      also carry those state words (a dispute RESOLVED before its reward is corrected). A state is an
      environment record's word, never one of our own verifiers' verdicts - demanding our verdict is
      a prescription, and base passes 049 without it.
  verified  [{applies_to, when, arg, verifier, inputs{name: {arg}|{tool, kind}|{near, after, kind}}, value_after, text_until, feedback}]
      the value of `arg` must equal what the named LB2 verifier returns for operands read from the
      call and the records; when it differs the call is declined with the verifier's own line (the
      same text the model would have read had it called the verifier itself). No value is composed.
  names     {feedback_wrong_suffix, feedback_not_discoverable, feedback_rejected}
      a name handed to the unlock / give / call wrappers must be in the registry (agent or user);
      a name the environment already rejected as unknown is not sent again
  schema    {tool: [argument names]}   arguments outside a declared signature are refused
  identifying {args, feedback}
      an argument whose name, or a token of whose name, is in `args` must occur in a record or a
      customer message - the deterministic form of provenance regeneration. Only declared names: a
      value-shape guess ("has digits, no spaces") flagged dates and amounts on the live 048 run.

Presence is judged on normalised text: case folded, punctuation dropped ("#1234" holds "1234"), and a
number is present under any conventional rendering (1500, 1500.0, 1,500.00). Rendering is not evidence.
"""

from lb_coordinator import Finding, DENY, GRADES, fam, fill, records_in, as_dict

LB = "LB3"
LEDGER, ENV, POLICY = GRADES["execution_ledger"], GRADES["env_output"], GRADES["policy_verbatim"]


def applies(spec, turn, call):
    name = str(getattr(call, "name", "") or "")
    if spec.get("applies_to") not in (None, name, fam(name)):
        return False
    w = spec.get("when") or {}
    return not w.get("arg") or str(as_dict(call.arguments).get(w["arg"]) or "").startswith(w.get("prefix", ""))


def norm(text):
    return " ".join("".join(ch if ch.isalnum() else " " for ch in str(text).lower()).split())


def renderings(value):
    """The strings a value may have been written as: itself, and for a number its usual formats."""
    s = str(value).strip()
    out = [norm(s)]
    try:
        x = float(s.replace(",", ""))
    except ValueError:
        return [f for f in out if f]
    for f in ("%g" % x, "%d" % x if x == int(x) else "", "%.1f" % x, "%.2f" % x, "{:,.2f}".format(x)):
        if f and norm(f) not in out:
            out.append(norm(f))
    return [f for f in out if f]


def present(value, text):
    t = norm(text)
    return any(f in t for f in renderings(value))


def grounded(value, turn, sources):
    return (("records" in sources and present(value, turn.tool_text))
            or ("customer" in sources and present(value, turn.user_text))
            or ("kb" in sources and any(present(value, d) for d in (turn.corpus or {}).values())))


def outputs_of(turn, tool):
    """Outputs of one tool, by name family, from the turn's own messages (call id -> tool message)."""
    names = {}
    for m in turn.messages:
        for c in (getattr(m, "tool_calls", None) or []):
            # a tool reached through the dispatcher is named in its arguments, not by the call's own name
            # (nc44 replay: get_user_dispute_history never matched call_discoverable_agent_tool)
            names[getattr(c, "id", None)] = fam(str(turn.named(c) or getattr(c, "name", "") or ""))
    return [str(getattr(m, "content", "") or "") for m in turn.messages
            if getattr(m, "role", None) == "tool" and names.get(getattr(m, "id", None)) == fam(tool)]


def capitalised_words(value):
    """The form 'Word Word-Word': every token starts with a capital letter and holds letters or hyphens
    only - no parenthetical, no lower-case start, no digits. A form check, not a name check."""
    toks = str(value or "").split()
    return bool(toks) and all(t[:1].isalpha() and t[:1].isupper() and all(ch.isalpha() or ch == "-" for ch in t)
                              for t in toks)


def token_after(text, marker, kind):
    """The token that follows `marker` in `text`, read as `kind` (date|number); None when absent."""
    from lb2_decision import date, num
    i = text.find(marker)
    while i >= 0:
        rest = text[i + len(marker):].split()
        if rest:
            tok = rest[0].strip(".,;:()")
            if kind == "date" and date(tok) is not None:
                return tok
            if kind == "number" and num(tok.replace("$", "").replace(",", "")) is not None:
                return tok.replace("$", "").replace(",", "")
            if kind == "bool" and tok.lower() in ("true", "false", "yes", "no"):
                return "1" if tok.lower() in ("true", "yes") else "0"
        i = text.find(marker, i + 1)
    return None


def operand(spec, turn, args, done=None):
    """One verifier operand from the call's own arguments or the conversation's records. Never a guess:
    {arg} copies the call's argument; {tool, kind} takes the first token of that kind in that tool's
    output; {near, after, kind} takes the token after `after` in the record that names the call's
    `near` argument; {near_operand} anchors on an operand already resolved (declaration order);
    kind "text" takes the rest of the line; {count} counts records and this conversation's own writes."""
    from lb2_decision import date, num
    if "arg" in spec:
        return args.get(spec["arg"])
    if "count" in spec:
        return _count_operand(spec["count"], turn, done or {})
    texts = outputs_of(turn, spec["tool"]) if spec.get("tool") else turn.tool_outputs()
    if spec.get("near") or spec.get("near_operand"):
        anchor = str(args.get(spec["near"]) or "") if spec.get("near") else str((done or {}).get(spec["near_operand"]) or "")
        if not anchor:
            return None
        texts = [t[i:i + 600] for t in texts for i in [t.find(anchor)] if i >= 0]
    if spec.get("kind") == "text" and spec.get("after"):
        for t in texts:
            i = t.find(spec["after"])
            if i >= 0:
                line = t[i + len(spec["after"]):].split(chr(10))[0].split(" | ")[0].strip()
                if line:
                    return line
        return None
    for t in texts:
        if spec.get("after"):
            got = token_after(t, spec["after"], spec.get("kind", "number"))
            if got is not None:
                return got
            continue
        for tok in t.split():
            tok = tok.strip(".,;:()")
            if spec.get("kind") == "date" and date(tok) is not None:
                return tok
            if spec.get("kind") == "number" and num(tok) is not None:
                return tok
    return None


def _count_operand(spec, turn, done):
    """Records of one tool's output that carry a marker (optionally with a date inside a window), plus this
    conversation's own successful writes of a family. None when that tool never answered - not evidence.
    039/041: the provisional-credit rule counts disputes filed in the past 12 months including the ones filed
    earlier in this conversation; gold's flags follow that running count exactly (E2 replay 8/8)."""
    from lb2_decision import date
    import datetime, re
    outs = outputs_of(turn, spec["tool"]) if spec.get("tool") else []
    if not outs:
        return None
    marker = str(spec.get("marker") or "")
    n = 0
    ref = date(done.get(spec.get("as_of_operand") or "") or "") if spec.get("date_within_days") else None
    for o in outs:
        parts = o.split(marker)[1:] if marker else []
        for part in parts:
            if ref is not None:
                ds = [date(x) for x in re.findall(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", part)]
                ds = [d for d in ds if d is not None]
                if ds and max(ds) < ref - datetime.timedelta(days=int(spec["date_within_days"])):
                    continue
            n += 1
    pre = str(spec.get("plus_executed_prefix") or "")
    if pre:
        msgs = turn.messages
        for i, m in enumerate(msgs):
            for c in (getattr(m, "tool_calls", None) or []):
                inner = str(turn.named(c) or "")
                if inner.startswith(pre):
                    nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                    res = str(getattr(nxt, "content", "") or "") if nxt is not None and getattr(nxt, "role", None) == "tool" else ""
                    if not res.lower().startswith("error"):
                        n += 1
    return str(n)


def verified_findings(turn, call):
    """A written value the declaration ties to a verifier: the engine runs that verifier on operands read
    from the call and the records, and if the value differs the call is declined with the verifier's own
    line. Nothing is said when they agree, or when an operand is missing (that is not evidence)."""
    out = []
    for spec in (turn.a2.get("LB3") or {}).get("verified") or []:
        if not applies(spec, turn, call):
            continue
        args = turn.args_of(call)
        got = args.get(spec.get("arg"))
        if got in (None, ""):
            continue
        decl = next((t for t in (turn.a2.get("LB2") or {}).get("tools") or []
                     if t.get("name") == spec.get("verifier")), None)
        if decl is None:
            continue
        ops = {}
        for k, v in (spec.get("inputs") or {}).items():
            ops[k] = operand(v, turn, args, ops)
        if any(v in (None, "") for v in ops.values()):
            continue
        import lb2_decision
        res = lb2_decision.run_tool(decl, ops, {"kb": [], "ledger": turn.tool_outputs(),
                                                "ledger_tools": turn.tool_outputs(), "user": [turn.user_text]},
                                    {"__tool_outputs": {}, "__user_text": turn.user_text})
        text, err = res[0], res[1]
        if err:
            continue
        tok = token_after(str(text), spec.get("value_after") or "", spec.get("value_kind") or "number") if spec.get("value_after") else None
        if tok is None:
            continue
        try:
            raw = str(got).strip().lower()
            # a true/false argument is compared as the 1/0 its verifier prints
            have = 1.0 if raw == "true" else 0.0 if raw == "false" else float(raw.replace("$", "").replace(",", ""))
            want = float(tok)
        except (TypeError, ValueError):
            continue
        if abs(want - have) > 0.005:
            shown = str(text).strip()
            cut = spec.get("text_until")
            if cut and cut in shown:
                shown = shown[:shown.find(cut)].strip()      # the fact only, not the verifier's own instruction
            out.append(Finding(LB, DENY, fam(turn.name_of(call)), call,
                               fill(spec.get("feedback"), arg=spec["arg"], got=got, text=shown),
                               grade=LEDGER, source="verified:" + spec["arg"]))
            break
    return out


def grounding_findings(turn, call):
    out = []
    for spec in (turn.a2.get("LB3") or {}).get("grounding") or []:
        if not applies(spec, turn, call):
            continue
        value = turn.args_of(call).get(spec.get("arg"))
        if value in (None, ""):
            continue
        recs = [r for o in turn.tool_outputs() for r in records_in(o) if present(value, str(r.values()))]
        problem = None
        if spec.get("field"):
            said = [str(r[spec["field"]]) for r in recs if r.get(spec["field"])
                    and present(r[spec["field"]], turn.user_text)]
            mine = [str(r[spec["field"]]) for r in recs if r.get(spec["field"])]
            if mine and not said:
                problem = fill(spec.get("feedback"), id=value, arg=spec["arg"], value=mine[-1],
                               mentioned=", ".join(sorted({str(r[spec["field"]]) for o in turn.tool_outputs()
                                                           for r in records_in(o) if r.get(spec["field"])
                                                           and present(r[spec["field"]], turn.user_text)}))
                               or "(none stated)")
        elif spec.get("state"):
            # a state the action depends on (a dispute resolved, an order shipped) must be read from a
            # record that names this value - the customer saying so is not a record. Base 026/027/029:
            # 12 of 12 simulations updated rewards on the customer's word that disputes were approved.
            # the output that names the value must carry the state words: an environment result reads
            # "Arguments: {...transaction_id...}\nStatus: RESOLVED", the state outside the braces
            if not any(present(value, o) and all(t in o for t in spec["state"]) for o in turn.tool_outputs()):
                problem = fill(spec.get("feedback"), id=value, arg=spec["arg"], val=value, value=value)
        elif spec.get("form"):
            # a value's declared form (the tool's own parameter text: "the full official account name").
            # 060-069: 36 of 56 wrong account_class values were forms the tool never accepts ("Green Account
            # (savings)"); the documents themselves use those headings, so presence in the KB is not the
            # test - the form is. Gold holds business names without 'Account' (Cobalt Blue, Sky Blue) and
            # 'Green Fee-Free Account', so the form is capitalised words with hyphens, nothing more.
            if spec["form"] == "capitalised-words" and not capitalised_words(value):
                problem = fill(spec.get("feedback"), val=value, value=value, arg=spec["arg"])
        elif not grounded(value, turn, spec.get("sources") or ["records", "customer"]):
            problem = fill(spec.get("feedback"), val=value, value=value, arg=spec["arg"])
        if problem:
            out.append(Finding(LB, DENY, fam(turn.name_of(call)), call, problem, grade=LEDGER,
                               source="grounding:" + spec["arg"]))
            break
    return out


def rejected_names(turn):
    """Names the environment rejected as unknown: the quoted token after a declared failure marker."""
    marks = [m for m in turn.a2.get("failure_markers") or [] if "unknown" in m.lower()]
    out = set()
    for o in turn.tool_outputs():
        for m in marks:
            i = o.find(m)
            if i >= 0:
                rest = o[i + len(m):]
                q = rest.find("'")
                if q >= 0 and rest.find("'", q + 1) > q:
                    out.add(rest[q + 1:rest.find("'", q + 1)])
    return out


def name_findings(turn, call):
    names = (turn.a2.get("LB3") or {}).get("names") or {}
    value = turn.named(call)
    if not value or not names:
        return []
    registry = set(turn.registry.get("agent", ())) | set(turn.registry.get("user", ()))
    if value in rejected_names(turn) and value not in registry and names.get("feedback_rejected"):
        return [Finding(LB, DENY, fam(value), call, fill(names["feedback_rejected"], name=value), grade=ENV,
                        source="rejected-name", force_call=True)]
    if not registry or value in registry or value in turn.registry.get("user_all", ()):
        return []
    same = any(fam(r) == fam(value) for r in registry)
    tpl = names.get("feedback_wrong_suffix") if same else names.get("feedback_not_discoverable")
    return [Finding(LB, DENY, fam(value), call, fill(tpl, name=value), grade=ENV, source="name-registry",
                    force_call=True)] if tpl else []


def schema_findings(turn, call):
    """The wrapper's own arguments, not the payload it carries - a dispatcher's inner arguments
    belong to the tool being dispatched and are not extra keys on the wrapper."""
    allowed = ((turn.a2.get("LB3") or {}).get("schema") or {}).get(str(getattr(call, "name", "") or ""))
    extra = sorted(k for k in (as_dict(call.arguments) if allowed else {}) if k not in allowed)
    if not extra:
        return []
    return [Finding(LB, DENY, str(call.name), call, grade=POLICY, source="schema",
                    order="Error: [SIGNATURE] '%s' takes only %s; unexpected argument(s): %s."
                    % (call.name, ", ".join(allowed), ", ".join(extra)))]


def identifying_findings(turn, call):
    spec = (turn.a2.get("LB3") or {}).get("identifying") or {}
    if not spec.get("feedback"):
        return []
    names = set(spec.get("args") or [])
    known = set(turn.registry.get("agent", ())) | set(turn.registry.get("user", ()))
    for k, v in turn.args_of(call).items():
        s = str(v).strip()
        declared = k in names or any(tok in names for tok in k.split("_"))
        if declared and s and not present(s, turn.tool_text) and not present(s, turn.user_text) and s not in known:
            return [Finding(LB, DENY, fam(turn.name_of(call)), call, fill(spec["feedback"], arg=k, val=s, value=s),
                            grade=LEDGER, source="identifying:" + k)]
    return []


def identity_findings(turn, call):
    """The check that only speaks when it fails - no tool, no round trip, no turn spent on a yes.

    verify_identity was a tool of ours in the model's list: 312 calls across 372 simulations and not
    one of them came back anything but VERIFIED. It never told the model something it did not already
    believe, and every call cost a round trip - the request, the answer, and on task_016 a further
    instruction to go and fetch the current time. base has no such tool and verifies anyway.

    The same question is answerable without asking: the values the agent is about to record are the
    call's own arguments, and the record is already in the conversation. Count how many of the
    declared fields appear in one retrieved record; under the threshold, say so. Above it, say
    nothing at all.
    """
    spec = (turn.a2.get("LB3") or {}).get("identity") or {}
    if not spec.get("feedback") or fam(turn.name_of(call)) != fam(spec.get("applies_to", "")):
        return []
    args = turn.args_of(call)
    have = [f for f in spec.get("fields") or [] if str(args.get(f) or "").strip()]
    if not have:
        return []
    # counted inside one tool output, not across them: two details that matched two different
    # customers are not a verification. The environment prints its records as a listing, not as JSON,
    # so there is nothing to parse here - the output itself is the record.
    best, who = 0, ""
    for o in turn.tool_outputs():
        hit = [f for f in have if present(args[f], o)]
        if len(hit) > best:
            best, who = len(hit), ", ".join(hit)
    if best >= int(spec.get("threshold") or 2):
        return []
    return [Finding(LB, DENY, fam(turn.name_of(call)), call,
                    fill(spec["feedback"], count=best, threshold=int(spec.get("threshold") or 2),
                         matched=who or "(none)", fields=", ".join(spec.get("fields") or [])),
                    grade=LEDGER, source="identity")]


def evaluate(turn):
    return [f for c in turn.calls for f in grounding_findings(turn, c) + verified_findings(turn, c)
            + name_findings(turn, c) + schema_findings(turn, c) + identifying_findings(turn, c)
            + identity_findings(turn, c)]


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name, args=None):
            self.name, self.arguments, self.id = name, args or {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    A2 = {"dispatch": {"agent_call": "call", "name_args": {"call": "tool", "give": "name"}},
          "failure_markers": ["Unknown discoverable tool"],
          "LB3": {"grounding": [
              {"applies_to": "call", "when": {"arg": "tool", "prefix": "file_"}, "arg": "last4",
               "sources": ["records", "customer"], "feedback": "no {val} for {arg}"},
              {"applies_to": "call", "when": {"arg": "tool", "prefix": "file_"}, "arg": "txn", "field": "merchant",
               "feedback": "{id} is {value}; said {mentioned}"},
              ],
              "names": {"feedback_wrong_suffix": "suffix {name}", "feedback_not_discoverable": "none {name}",
                        "feedback_rejected": "rejected {name}"},
              "schema": {"give": ["name", "arguments"]}}}
    msgs = [M("user", "dispute the Marriott charge"),
            M("tool", '[{"txn": "t1", "merchant": "Marriott"}, {"txn": "t2", "merchant": "Facebook"}] last4 5320')]
    ok = C("call", {"tool": "file_x", "arguments": '{"txn": "t1", "last4": "5320"}'})
    bad = C("call", {"tool": "file_x", "arguments": '{"txn": "t2", "last4": "1234"}'})
    t = Turn(A2, msgs, M(calls=[ok, bad]))
    assert not grounding_findings(t, ok)
    assert grounding_findings(t, bad)[0].order == "no 1234 for last4"
    bad2 = C("call", {"tool": "file_x", "arguments": '{"txn": "t2", "last4": "5320"}'})
    assert grounding_findings(t, bad2)[0].order == "t2 is Facebook; said Marriott"
    A2["LB3"]["grounding"].append({"applies_to": "call", "when": {"arg": "tool", "prefix": "upd_"}, "arg": "txn",
                                   "state": ["RESOLVED"], "feedback": "no record shows {id} resolved"})
    upd = C("call", {"tool": "upd_x", "arguments": '{"txn": "t1"}'})
    assert grounding_findings(t, upd)[0].order == "no record shows t1 resolved"        # the customer's word is not a record
    t_res = Turn(A2, msgs + [M("tool", '[{"txn": "t1", "status": "RESOLVED_CUSTOMER_FAVOR"}]')], M(calls=[upd]))
    assert not grounding_findings(t_res, upd)
    t2 = Turn(A2, [M("tool", "Error: Unknown discoverable tool 'nav_x'")], M(), registry={"agent": {"real_1"}})
    assert name_findings(t2, C("give", {"name": "nav_x"}))[0].order == "rejected nav_x"
    assert name_findings(t2, C("give", {"name": "real_2"}))[0].order == "suffix real_2"
    assert not name_findings(t2, C("give", {"name": "real_1"}))
    assert "extra" in schema_findings(t2, C("give", {"name": "real_1", "arguments": "{}", "extra": 1}))[0].order
    # the payload a dispatcher carries is the inner tool's, not extra keys on the wrapper
    assert not schema_findings(t2, C("give", {"name": "real_1", "arguments": '{"inner_arg": 1}'}))
    A2["LB3"]["identifying"] = {"args": ["user_id", "txn"], "feedback": "no source for {arg}={val}"}
    t3 = Turn(A2, msgs, M())
    assert identifying_findings(t3, C("w", {"txn": "t9x8y7"}))[0].order == "no source for txn=t9x8y7"
    assert identifying_findings(t3, C("w", {"txn_ref": "t9x8y7"}))          # a token of the name is declared
    assert not identifying_findings(t3, C("w", {"txn": "t1"})) and not identifying_findings(t3, C("w", {"user_id": "5320"}))
    assert not identifying_findings(t3, C("w", {"date": "2026-01-01"}))      # undeclared shape: not our business
    # normalisation: "#" and number renderings are the same value
    assert present("1234", "account #1234") and present("1500", "paid $1,500.00") and present(1500.0, "fee 1500")
    assert not present("12", "1 2")
    print("lb3_citation self-test OK")
