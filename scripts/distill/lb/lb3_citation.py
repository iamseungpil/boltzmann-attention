# -*- coding: utf-8 -*-
"""LB3 - citation check (mechanism F2, source axis).

One rule: every value or name the model writes must exist in a source this conversation holds.
The engine never produces a value; it only asks "where is this from?". Declared in A2["LB3"]:

  grounding [{applies_to, when{arg, prefix}, arg, sources[records|customer], tokens, field, feedback}]
      the value of `arg` must occur in a source text; with `tokens`, the record holding it must also
      hold every token (evidence); with `field`, the record's `field` value must be one the customer
      said (reference verification)
  names     {feedback_wrong_suffix, feedback_not_discoverable, feedback_rejected}
      a name handed to the unlock / give / call wrappers must be in the registry (agent or user);
      a name the environment already rejected as unknown is not sent again
  schema    {tool: [argument names]}   arguments outside a declared signature are refused
  identifying {args, min_len, feedback}
      any argument named in `args`, or whose value looks like an identifier (digits, length >= min_len),
      must occur in a record or a customer message - the deterministic form of provenance regeneration
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


def grounded(value, turn, sources):
    s = str(value).strip().lower()
    return bool(s) and (("records" in sources and s in turn.tool_text) or ("customer" in sources and s in turn.user_text))


def grounding_findings(turn, call):
    out = []
    for spec in (turn.a2.get("LB3") or {}).get("grounding") or []:
        if not applies(spec, turn, call):
            continue
        value = turn.args_of(call).get(spec.get("arg"))
        if value in (None, ""):
            continue
        recs = [r for o in turn.tool_outputs() for r in records_in(o) if str(value) in str(r.values())]
        problem = None
        if spec.get("field"):
            said = [str(r[spec["field"]]) for r in recs if r.get(spec["field"])
                    and str(r[spec["field"]]).lower() in turn.user_text]
            mine = [str(r[spec["field"]]) for r in recs if r.get(spec["field"])]
            if mine and not said:
                problem = fill(spec.get("feedback"), id=value, arg=spec["arg"], value=mine[-1],
                               mentioned=", ".join(sorted({str(r[spec["field"]]) for o in turn.tool_outputs()
                                                           for r in records_in(o) if r.get(spec["field"])
                                                           and str(r[spec["field"]]).lower() in turn.user_text}))
                               or "(none stated)")
        elif spec.get("tokens"):
            if not any(all(t in o for t in spec["tokens"]) and str(value) in o for o in turn.tool_outputs()):
                problem = fill(spec.get("feedback"), id=value, arg=spec["arg"], val=value)
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
    allowed = ((turn.a2.get("LB3") or {}).get("schema") or {}).get(str(getattr(call, "name", "") or ""))
    extra = sorted(k for k in (turn.args_of(call) if allowed else {}) if k not in allowed)
    if not extra:
        return []
    return [Finding(LB, DENY, str(call.name), call, grade=POLICY, source="schema",
                    order="Error: [SIGNATURE] '%s' takes only %s; unexpected argument(s): %s."
                    % (call.name, ", ".join(allowed), ", ".join(extra)))]


def identifying_findings(turn, call):
    spec = (turn.a2.get("LB3") or {}).get("identifying") or {}
    if not spec.get("feedback"):
        return []
    names, min_len = set(spec.get("args") or []), int(spec.get("min_len") or 5)
    known = set(turn.registry.get("agent", ())) | set(turn.registry.get("user", ()))
    for k, v in turn.args_of(call).items():
        s = str(v).strip()
        idlike = len(s) >= min_len and any(ch.isdigit() for ch in s) and " " not in s
        if (k in names or idlike) and s and s.lower() not in turn.tool_text and s.lower() not in turn.user_text \
                and s not in known:
            return [Finding(LB, DENY, fam(turn.name_of(call)), call, fill(spec["feedback"], arg=k, val=s, value=s),
                            grade=LEDGER, source="identifying:" + k)]
    return []


def evaluate(turn):
    return [f for c in turn.calls for f in grounding_findings(turn, c) + name_findings(turn, c)
            + schema_findings(turn, c) + identifying_findings(turn, c)]


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
              {"applies_to": "call", "when": {"arg": "tool", "prefix": "upd_"}, "arg": "txn", "tokens": ["RESOLVED"],
               "feedback": "no evidence {id}"}],
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
    assert grounding_findings(t, C("call", {"tool": "upd_x", "arguments": '{"txn": "t1"}'}))[0].order == "no evidence t1"
    t2 = Turn(A2, [M("tool", "Error: Unknown discoverable tool 'nav_x'")], M(), registry={"agent": {"real_1"}})
    assert name_findings(t2, C("give", {"name": "nav_x"}))[0].order == "rejected nav_x"
    assert name_findings(t2, C("give", {"name": "real_2"}))[0].order == "suffix real_2"
    assert not name_findings(t2, C("give", {"name": "real_1"}))
    assert "extra" in schema_findings(t2, C("give", {"name": "real_1", "arguments": "{}", "extra": 1}))[0].order
    A2["LB3"]["identifying"] = {"args": ["user_id"], "min_len": 5, "feedback": "no source for {arg}={val}"}
    t3 = Turn(A2, msgs, M())
    assert identifying_findings(t3, C("w", {"txn": "t9x8y7"}))[0].order == "no source for txn=t9x8y7"
    assert not identifying_findings(t3, C("w", {"txn": "t1"})) and not identifying_findings(t3, C("w", {"user_id": "5320"}))
    print("lb3_citation self-test OK")
