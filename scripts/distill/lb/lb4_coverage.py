# -*- coding: utf-8 -*-
"""LB4 - set-difference completion gate (mechanism F4, coverage).

One rule: requested set minus done set. Every check is a set difference over the execution ledger;
what counts as requested and as done is declared in A2["LB4"]["sets"], one of four kinds:

  ledger        {entity_key, list_tools, write_tools, finalize_writes, feedback}
                entities returned by a list tool minus entities acted on, when the customer asked for
                several ("all", "both", a number) and the model is finishing
  follow_up     {after, requires, decision_tools, feedback, decision_feedback}
                after `after` ran and `requires` are done, a decision tool must run before resigning
  settled_rows  {settle_tool, submit_tool, id_key, feedback}
                ids a settling tool returned minus ids submitted
  once          {applies_to, when{arg, prefix}, keys, feedback}
                a state change whose key already succeeded is not run again
"""

from lb_coordinator import Finding, DENY, SURFACE, GRADES, fam, fill, records_in, as_dict

LB = "LB4"
LEDGER, POLICY = GRADES["execution_ledger"], GRADES["policy_verbatim"]
PLURAL = {"all", "every", "each", "both", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


def _results(turn):
    """[(call, result message)] for every executed call, in order."""
    by_id = {getattr(m, "id", None): m for m in turn.messages if getattr(m, "role", None) == "tool"}
    return [(c, by_id.get(getattr(c, "id", None))) for m in turn.messages
            for c in (getattr(m, "tool_calls", None) or [])]


def _succeeded(turn, res):
    text = str(getattr(res, "content", "") or "").lstrip()
    return res is not None and not getattr(res, "error", False) \
        and not any(text.startswith(m) for m in turn.a2.get("failure_markers") or [])


def plural_request(text):
    words = {w.strip(".,;:!?") for w in text.split()}
    return bool(words & PLURAL) or any(w.isdigit() and int(w) >= 2 for w in words)


def ledger(spec, turn):
    finalizing = not turn.calls or any(fam(turn.name_of(c)) in {fam(x) for x in spec.get("finalize_writes") or []}
                                       for c in turn.calls)
    if not finalizing or not plural_request(turn.user_text):
        return []
    key, listed, acted = spec.get("entity_key"), set(), set()
    for call, res in _results(turn):
        if not _succeeded(turn, res):
            continue
        name = fam(turn.name_of(call))
        if name in {fam(x) for x in spec.get("list_tools") or []}:
            listed |= {str(r[key]) for r in records_in(getattr(res, "content", ""), key)}
        if name in {fam(x) for x in spec.get("write_tools") or []}:
            acted |= {str(v) for v in turn.args_of(call).values() if str(v) in listed or key in turn.args_of(call)}
            acted.add(str(turn.args_of(call).get(key)))
    gap = sorted(listed - acted)
    if not gap:
        return []
    text = fill(spec.get("feedback"), missing=", ".join(gap))
    fin = [c for c in turn.calls if fam(turn.name_of(c)) in {fam(x) for x in spec.get("finalize_writes") or []}]
    if fin:
        return [Finding(LB, DENY, fam(turn.name_of(fin[0])), fin[0], text, grade=LEDGER, source="ledger")]
    return [Finding(LB, SURFACE, "coverage", order=text, grade=LEDGER, source="ledger")]


def follow_up(spec, turn):
    if turn.calls:
        return []
    done = turn.executed_fams()
    if not done & {fam(x) for x in spec.get("after") or []}:
        return []
    decision = {fam(x) for x in spec.get("decision_tools") or []}
    if decision & done:
        return []
    missing = [x for x in spec.get("requires") or [] if fam(x) not in done]
    tpl = spec.get("feedback") if missing else spec.get("decision_feedback")
    return [Finding(LB, SURFACE, sorted(decision)[0], order=fill(tpl, missing=", ".join(missing)), grade=POLICY,
                    source="follow-up")] if tpl else []


def settled_rows(spec, turn):
    if turn.calls:
        return []
    settled, submitted = set(), set()
    for call, res in _results(turn):
        if not _succeeded(turn, res):
            continue
        name = fam(turn.name_of(call))
        if name == fam(spec.get("settle_tool")):
            settled |= {str(r[spec["id_key"]]) for r in records_in(getattr(res, "content", ""), spec.get("id_key"))}
        elif name == fam(spec.get("submit_tool")):
            submitted |= {str(v) for v in _leaves(turn.args_of(call))}
    left = sorted(x for x in settled if not any(x in s for s in submitted))
    return [Finding(LB, SURFACE, spec.get("submit_tool"), order=fill(spec.get("feedback"), ids=", ".join(left)),
                    grade=LEDGER, source="settled-rows")] if left else []


def _leaves(v):
    if isinstance(v, dict):
        return [x for y in v.values() for x in _leaves(y)]
    if isinstance(v, list):
        return [x for y in v for x in _leaves(y)]
    if isinstance(v, str) and v[:1] in ("{", "["):
        parsed = as_dict(v)
        return _leaves(parsed) if parsed else [v]
    return [v]


def once(spec, turn):
    out = []
    for c in turn.calls:
        name = str(getattr(c, "name", "") or "")
        w = spec.get("when") or {}
        if spec.get("applies_to") not in (None, name) or (
                w.get("arg") and not str(as_dict(c.arguments).get(w["arg"]) or "").startswith(w.get("prefix", ""))):
            continue
        key = _key(turn, c, spec.get("keys") or [])
        for call, res in _results(turn):
            if getattr(call, "name", None) == name and _succeeded(turn, res) and _key(turn, call, spec["keys"]) == key:
                out.append(Finding(LB, DENY, fam(turn.name_of(c)), c, grade=LEDGER, source="once",
                                   order=fill(spec.get("feedback") or DEFAULT_ONCE, keys=", ".join(spec["keys"]))))
                break
    return out


DEFAULT_ONCE = ("Error: [DUPLICATE] this state change already succeeded earlier in this conversation for the "
                "same {keys}; it must not run again.")


def _key(turn, call, keys):
    merged = dict(turn.args_of(call))
    merged.update({k: v for k, v in as_dict(call.arguments).items() if k != turn.payload_key})
    return tuple(str(merged.get(k)) for k in keys)


KINDS = {"ledger": ledger, "follow_up": follow_up, "settled_rows": settled_rows, "once": once}


def evaluate(turn):
    out = []
    for spec in (turn.a2.get("LB4") or {}).get("sets") or []:
        check = KINDS.get(spec.get("kind"))
        if check:
            out += check(spec, turn)
    return out


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name, args=None, cid=None):
            self.name, self.arguments, self.id = name, args or {}, cid or name

    class M(object):
        def __init__(self, role="assistant", content="", calls=(), mid=None):
            self.role, self.content, self.tool_calls, self.id = role, content, list(calls), mid

    A2 = {"dispatch": {"agent_call": "call", "name_args": {"call": "tool"}}, "failure_markers": ["Error:"],
          "LB4": {"sets": [
              {"kind": "follow_up", "after": ["submit_x"], "requires": ["submit_x", "read_y"],
               "decision_tools": ["approve_x"], "feedback": "missing {missing}", "decision_feedback": "decide"},
              {"kind": "once", "applies_to": "call", "keys": ["tool", "account_id"]},
              {"kind": "settled_rows", "settle_tool": "settle", "submit_tool": "submit", "id_key": "txn",
               "feedback": "left {ids}"},
              {"kind": "ledger", "entity_key": "oid", "list_tools": ["list_orders"], "write_tools": ["cancel"],
               "finalize_writes": [], "feedback": "not done: {missing}"}]}}
    assert follow_up(A2["LB4"]["sets"][0], Turn(A2, [], M(content="bye"), executed={"submit_x_1": 1}))[0].order == "missing read_y"
    assert follow_up(A2["LB4"]["sets"][0], Turn(A2, [], M(content="bye"), executed={"submit_x": 1, "read_y": 1}))[0].order == "decide"
    prev = C("call", {"tool": "credit_1", "account_id": "A"}, "c1")
    msgs = [M(calls=[prev]), M("tool", "ok", mid="c1")]
    again = C("call", {"tool": "credit_1", "account_id": "A"}, "c2")
    other = C("call", {"tool": "credit_1", "account_id": "B"}, "c3")
    assert once(A2["LB4"]["sets"][1], Turn(A2, msgs, M(calls=[again])))[0].primitive == DENY
    assert not once(A2["LB4"]["sets"][1], Turn(A2, msgs, M(calls=[other])))
    s, sub = C("settle", {}, "s1"), C("submit", {"rows": '[{"txn": "t1"}]'}, "s2")
    msgs = [M(calls=[s]), M("tool", '[{"txn": "t1"}, {"txn": "t2"}]', mid="s1"), M(calls=[sub]), M("tool", "ok", mid="s2")]
    assert settled_rows(A2["LB4"]["sets"][2], Turn(A2, msgs, M(content="done")))[0].order == "left t2"
    lst, cn = C("list_orders", {}, "l1"), C("cancel", {"oid": "o1"}, "c9")
    msgs = [M("user", "cancel both orders"), M(calls=[lst]), M("tool", '[{"oid": "o1"}, {"oid": "o2"}]', mid="l1"),
            M(calls=[cn]), M("tool", "ok", mid="c9")]
    assert ledger(A2["LB4"]["sets"][3], Turn(A2, msgs, M(content="done")))[0].order == "not done: o2"
    assert not ledger(A2["LB4"]["sets"][3], Turn(A2, [M("user", "cancel my order")] + msgs[1:], M(content="done")))
    print("lb4_coverage self-test OK")
