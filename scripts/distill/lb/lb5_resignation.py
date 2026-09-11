# -*- coding: utf-8 -*-
"""LB5 - resignation-timing gate (mechanism F5, persistence / escalation).

One rule, one decision point: when the model is leaving - a text-only turn, or a transfer call -
name what the ledger can still show as open, once, and let the model decide whether to go on. The
engine never decides; whether to persist or hand off is the model's call. Declared in A2["LB5"]:

  transfer_tools   names that hand the conversation off
  doc_feedback     a transfer tool whose defining document was never read      (surface, never deny)
  search_tools     search tools; the same query returning nothing twice is exhaustion
  search_feedback  what the policy says to do when search is exhausted
  open_request     {question, feedback}  the customer's request the run record does not answer, put to
                   an isolated sub-call at the moment of leaving and named once per simulation
  repeat_tools     retrieval tools whose output repeating is the sign a search cannot go further
  repeat_cap       how many times one identical result may come back before the search is exhausted
  repeat_feedback  what to do instead; this one is denied, because a runaway never reaches a
                   leaving turn and advice does not reach the model on a tool-calling turn
"""

import collections

from lb_coordinator import Finding, DENY, SURFACE, GRADES, fam, fill

LB = "LB5"
LEDGER, POLICY = GRADES["execution_ledger"], GRADES["policy_verbatim"]
KEY = "lb5_open"


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


def read_form(out):
    """Read KIND and REQUEST out of the sub-call's answer. The engine reads; it does not judge.

    A2 declares the kinds and which of them this rule speaks for; whether this conversation is one of
    them is the sub-call's answer, and all that happens here is that the two lines are taken apart.
    """
    kind, named = "", ""
    for line in str(out or "").splitlines():
        s = line.strip()
        if ":" not in s:
            continue
        head, rest = s.split(":", 1)
        head = head.strip().upper()
        if head == "KIND":
            w = rest.strip().upper().split()
            kind = w[0].strip(".,") if w else ""
        elif head == "REQUEST":
            named = " ".join(rest.split())
    return kind, ("" if named in ("", "-", "--") else named)


def open_request(turn):
    """What the customer asked for and the run record does not show, named once as the model leaves.

    This engine's own decision point says to "name what the ledger can still show as open", and the
    only kinds declared were an unread transfer document and an exhausted search. task_016 is the
    missing one: the customer asks for a purchase to be put through on their friend's card, the
    agent verifies the caller correctly, researches the referral terms for sixty messages, decides
    it cannot confirm one clause, and leaves having explained rather than acted. Base does the write
    in four simulations out of four; ours did it in none. Nothing was mis-argued - every simulation
    that wrote at all wrote the same arguments base did.

    Which request is still open is a judgement about this conversation, and no document in the
    corpus names the tool, so the engine cannot hold the answer: it puts the customer's own words
    and the run record to an isolated sub-call and carries back the line it answers with. It speaks
    at most once per simulation. LB1 had a walker that named the next step on every text turn and it
    fired 180 times over 216 simulations; the bound is the difference between naming something once
    on the way out and talking over the whole conversation.
    """
    spec = spec_of(turn).get("open_request") or {}
    ask = (getattr(turn, "extras", None) or {}).get("ask")
    once = (getattr(turn, "extras", None) or {}).get("once")
    speak = {str(k).upper() for k in (spec.get("speak_when") or ())}
    if not spec.get("question") or not spec.get("feedback") or not ask or once is None or not speak:
        return []
    if KEY in once:
        return []
    asked = sum(1 for k in once if isinstance(k, str) and k.startswith(KEY + "#"))
    if asked >= int(spec.get("ask_cap") or 0):
        return []
    ran = [name for name, _ in getattr(turn, "ran", ())]
    if not turn.user_text.strip():
        return []
    kinds = spec.get("kinds") or {}
    nl = chr(10)
    prompt = (spec["question"] + nl + nl
              + nl.join("%s - %s" % (k, v) for k, v in kinds.items()) + nl + nl
              + str(spec.get("acts") or "") + nl + nl
              + "What the customer has said:" + nl + nl + turn.user_text[-8000:] + nl + nl
              + "Every tool the agent actually ran, in order:" + nl + nl
              + (", ".join(ran) or "(none)") + nl + nl
              + str(spec.get("form") or ""))
    kind, named = read_form(ask(prompt, KEY))
    once.add(KEY + "#%d" % asked)
    if kind not in speak or not named:
        return []
    once.add(KEY)
    return [fill(spec["feedback"], open=named)]


def repeated_search(turn):
    """A search that hands back what it has already handed back cannot be exhausted by trying again.

    task_023's three runaways are one shape: the customer asks how to apply for the Diamond Elite
    Card, which is invitation-only and has no application tool, and the agent hunts for it - the same
    query sixteen times, the same document read eleven times - for a hundred assistant turns until
    the step cap or the context window ends the simulation. The exhaustion test that exists looks for
    an empty result, and every one of these came back full: the same documents, again.

    The count is the measurement. Across base's 388 simulations one identical retrieval result never
    comes back a fourth time - the histogram stops at three, in three simulations - while ours reach
    four, five and six, and both of our runaways are in that group. So the cap is declared, not
    guessed, and at four it has never fired on base at all.

    This one is denied rather than surfaced. A runaway never reaches a leaving turn, and advice
    reaches the model only on a text turn or a hand-off, so a note here would never arrive. The denial
    says what cannot work and what to do instead, which is what a denial owes.
    """
    spec = spec_of(turn)
    tools = {fam(x) for x in spec.get("repeat_tools") or []}
    tpl, cap = spec.get("repeat_feedback"), int(spec.get("repeat_cap") or 0)
    if not tools or not tpl or cap < 2:
        return []
    asking = [c for c in turn.calls if fam(turn.name_of(c)) in tools]
    if not asking:
        return []
    by_id = {getattr(m, "id", None): m for m in turn.messages if getattr(m, "role", None) == "tool"}
    seen = collections.Counter()
    for m in turn.messages:
        for c in (getattr(m, "tool_calls", None) or []):
            res = by_id.get(getattr(c, "id", None))
            if fam(turn.name_of(c)) in tools and res is not None:
                seen[str(getattr(res, "content", "") or "")] += 1
    worst = max(seen.values()) if seen else 0
    if worst < cap:
        return []
    return [Finding(LB, DENY, fam(turn.name_of(c)), c, grade=LEDGER, source="search-repeat",
                    order=fill(tpl, n=worst)) for c in asking]


def leaving(turn):
    """The one decision point: everything still open, in one surfaced note, when the model is leaving."""
    handoff = transferring(turn)
    if not handoff and not turn.resigning():
        return []
    facts = unread_definition(turn) + [f.order for f in exhausted_search(turn)] + open_request(turn)
    target = fam(turn.name_of(handoff[0])) if handoff else "leaving"
    return [Finding(LB, SURFACE, target, facts=facts, grade=POLICY, source="leaving")] if facts else []


def evaluate(turn):
    return leaving(turn) + repeated_search(turn)


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
