# -*- coding: utf-8 -*-
"""LB2 - decision-point isolation and deterministic execution (mechanism F2, symbolic operand).

One rule: arithmetic and comparisons the policy fixes are done by the engine over parsed records;
the model only supplies the key. Declared in A2["LB2"]["computations"], each one of two kinds:

  ratio_cap  {applies_to, when{arg, prefix}, param, record_key, limit_field, pct_by{field, map}, feedback}
             the value of `param` may not exceed record[limit_field] * map[record[field]]
  distinct   {tool, pairs[[a, b]], feedback}     two arguments the policy defines differently must differ

isolate() is the one door for a formalizing sub-call (the model turns prose into a key); the
engine never guesses a key it was not given.
"""

from lb_coordinator import Finding, DENY, GRADES, fam, fill, records_in, as_dict

LB = "LB2"
LEDGER, POLICY = GRADES["execution_ledger"], GRADES["policy_verbatim"]


def applies(spec, turn, call):
    name = str(getattr(call, "name", "") or "")
    if spec.get("applies_to") not in (None, name, fam(name)):
        return False
    w = spec.get("when") or {}
    return not w.get("arg") or str(as_dict(call.arguments).get(w["arg"]) or "").startswith(w.get("prefix", ""))


def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ratio_cap(spec, turn, call):
    args = turn.args_of(call)
    value, rid = _number(args.get(spec.get("param"))), args.get(spec.get("record_key"))
    if value is None or rid is None:
        return None
    recs = [r for out in turn.tool_outputs() for r in records_in(out, spec.get("record_key"))
            if str(r.get(spec["record_key"])) == str(rid)]
    if not recs:
        return None
    rec, pb = recs[-1], spec.get("pct_by") or {}
    limit, pct = _number(rec.get(spec.get("limit_field"))), (pb.get("map") or {}).get(str(rec.get(pb.get("field"))))
    if limit is None or pct is None or value <= limit * pct:
        return None
    return fill(spec.get("feedback"), value=value, cap=limit * pct, pct=pct, limit=limit)


def distinct(spec, turn, call):
    args = turn.args_of(call)
    for a, b in spec.get("pairs") or []:
        if args.get(a) is not None and args.get(b) is not None and str(args[a]) == str(args[b]):
            return fill(spec.get("feedback"), a=a, b=b)
    return None


KINDS = {"ratio_cap": ratio_cap, "distinct": distinct}


def evaluate(turn):
    out = []
    for c in turn.calls:
        name = turn.name_of(c)
        for spec in (turn.a2.get("LB2") or {}).get("computations") or []:
            check = KINDS.get(spec.get("kind"))
            if check is None or spec.get("tool") not in (None, name, fam(name)) or not applies(spec, turn, c):
                continue
            text = check(spec, turn, c)
            if text:
                out.append(Finding(LB, DENY, fam(name), c, text, grade=LEDGER, source=spec["kind"]))
    return out


def isolate(agent, prompt, call_name="lb2_isolate"):
    """One sub-call over a minimal context; None when no generator is available."""
    gen = getattr(agent, "_lb_generate", None)
    return gen(prompt, call_name) if gen else None


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name, args=None):
            self.name, self.arguments, self.id = name, args or {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    A2 = {"dispatch": {"agent_call": "call", "name_args": {"call": "tool"}},
          "LB2": {"computations": [
              {"kind": "ratio_cap", "applies_to": "call", "when": {"arg": "tool", "prefix": "req_"}, "param": "amount",
               "record_key": "acct", "limit_field": "lim", "pct_by": {"field": "kind", "map": {"A": 0.5}},
               "feedback": "cap {cap} < {value}"},
              {"kind": "distinct", "tool": "w", "pairs": [["x", "y"]], "feedback": "{a} equals {b}"}]}}
    msgs = [M("tool", '{"acct": "1", "lim": 100, "kind": "A"}')]
    over = C("call", {"tool": "req_x", "arguments": '{"acct": "1", "amount": 80}'})
    under = C("call", {"tool": "req_x", "arguments": '{"acct": "1", "amount": 30}'})
    assert evaluate(Turn(A2, msgs, M(calls=[over])))[0].order == "cap 50.0 < 80.0"
    assert not evaluate(Turn(A2, msgs, M(calls=[under])))
    assert evaluate(Turn(A2, msgs, M(calls=[C("w", {"x": 5, "y": 5})])))[0].order == "x equals y"
    print("lb2_decision self-test OK")
