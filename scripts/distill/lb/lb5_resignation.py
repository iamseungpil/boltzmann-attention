# -*- coding: utf-8 -*-
"""LB5 - resignation-timing gate (mechanism F5, persistence / escalation).

One rule, one decision point: when the model is leaving - a text-only turn, or a transfer call -
name what the ledger can still show as open, once, and let the model decide whether to go on. The
engine never decides; whether to persist or hand off is the model's call. Declared in A2["LB5"]:

  transfer_tools   names that hand the conversation off
  doc_feedback     a transfer tool whose defining document was never read      (surface, never deny)
  search_tools     search tools; the same query returning nothing twice is exhaustion
  search_feedback  what the policy says to do when search is exhausted
"""

from lb_coordinator import Finding, SURFACE, GRADES, fam, fill

LB = "LB5"
LEDGER, POLICY = GRADES["execution_ledger"], GRADES["policy_verbatim"]


def spec_of(turn):
    return turn.a2.get("LB5") or {}


def transferring(turn):
    tools = {fam(x) for x in spec_of(turn).get("transfer_tools") or []}
    return [c for c in turn.calls if fam(turn.name_of(c)) in tools]


def exhausted_search(turn):
    tools = {fam(x) for x in spec_of(turn).get("search_tools") or []}
    tpl = spec_of(turn).get("search_feedback")
    if not tools or not tpl:
        return []
    by_id = {getattr(m, "id", None): m for m in turn.messages if getattr(m, "role", None) == "tool"}
    zero = {}
    for m in turn.messages:
        for c in (getattr(m, "tool_calls", None) or []):
            res = by_id.get(getattr(c, "id", None))
            if fam(turn.name_of(c)) in tools and res is not None and _empty(getattr(res, "content", "")):
                q = str(sorted(turn.args_of(c).items()))
                zero[q] = zero.get(q, 0) + 1
    return [Finding(LB, SURFACE, "transfer", order=tpl, grade=POLICY, source="search-exhausted")] \
        if any(n >= 2 for n in zero.values()) else []


def _empty(text):
    t = str(text or "").strip().lower()
    return t in ("", "[]", "{}", "null") or "no results" in t or "no documents" in t or "0 results" in t


def unread_definition(turn):
    tpl = spec_of(turn).get("doc_feedback")
    return [fill(tpl, tool=turn.name_of(c)) for c in transferring(turn)
            if tpl and fam(turn.name_of(c)).lower() not in turn.tool_text]


def leaving(turn):
    """The one decision point: everything still open, in one surfaced note, when the model is leaving."""
    handoff = transferring(turn)
    if not handoff and not turn.resigning():
        return []
    facts = unread_definition(turn) + [f.order for f in exhausted_search(turn)]
    target = fam(turn.name_of(handoff[0])) if handoff else "leaving"
    return [Finding(LB, SURFACE, target, facts=facts, grade=POLICY, source="leaving")] if facts else []


def evaluate(turn):
    return leaving(turn)


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name, args=None, cid=None):
            self.name, self.arguments, self.id = name, args or {}, cid or name

    class M(object):
        def __init__(self, role="assistant", content="", calls=(), mid=None):
            self.role, self.content, self.tool_calls, self.id = role, content, list(calls), mid

    A2 = {"LB5": {"transfer_tools": ["transfer_x"], "doc_feedback": "read the doc for {tool}",
                  "search_tools": ["kb_search"], "search_feedback": "escalate properly"}}
    s1, s2 = C("kb_search", {"q": "a"}, "s1"), C("kb_search", {"q": "a"}, "s2")
    msgs = [M(calls=[s1]), M("tool", "no results", mid="s1"), M(calls=[s2]), M("tool", "[]", mid="s2")]
    f = evaluate(Turn(A2, msgs, M(content="I cannot help further.")))
    assert f[0].source == "leaving" and f[0].facts == ["escalate properly"]
    f = evaluate(Turn(A2, msgs, M(calls=[C("transfer_x")])))
    assert f[0].facts == ["read the doc for transfer_x", "escalate properly"]   # one note, everything open
    f = evaluate(Turn(A2, msgs + [M("tool", "transfer_x: search first")], M(calls=[C("transfer_x")])))
    assert f[0].facts == ["escalate properly"]                                  # the doc is read; the search is still exhausted
    assert not evaluate(Turn(A2, [M("tool", "transfer_x: search first")], M(calls=[C("transfer_x")])))
    print("lb5_resignation self-test OK")
