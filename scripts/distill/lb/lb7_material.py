# -*- coding: utf-8 -*-
"""LB7 - material delivery (the premise of every other engine).

Dropped 2026-09-08: `named_uncalled` (retrieved documents name a tool nobody called). It is the old
`T2_HANDOFF_PREDICATE` / `named-but-not-given`, whose single-variable A/B reads 2/12 <-> 2/12 with
latency 1.90x and 13 context-window terminations against zero, and whose one positive cell (028) the
old ledger disowns: the flip came from environment argument errors disappearing, not from this
predicate. Verdict on record: discard.

One rule: give the model the candidate set it has not seen. Not a seventh mechanism: when the
candidates never reach the model, every mechanism fails for a reason unrelated to itself
(x829: without material 0/8, with 47 document titles 8/8). This engine orders nothing; it
surfaces - the whole set, verbatim from the corpus, no ranking. Declared in A2["LB7"]:

  deliver_for   tools whose defining documents are delivered in full when unread, as far as the
                context has room (model_context minus the reply reserve minus what the view holds);
                when there is no room the documents are named, not delivered
  max_chars     delivery cap
  have_value    [{write, arg, producer_marker, value_after, reask_signals, feedback, acquire_tool, give_tool,
                  acquire_feedback}]  a value a producer already returned is handed back instead of re-asked;
                  when no producer ran and the customer keeps being asked, the acquiring tool is named
"""

from lb_coordinator import Finding, DENY, SURFACE, GRADES, fam, fill

LB = "LB7"
RETRIEVED = GRADES["retrieved_prose"]
HEAD = ("[MATERIAL] Before using '{tool}', the document(s) that define it - which nothing you retrieved so far "
        "contains - are delivered below in full. Read them and choose; nothing here is blocked.")


def docs_naming(tool, corpus):
    f = fam(tool)
    return sorted(d for d, body in corpus.items() if f and f in str(body))


CHARS_PER_TOKEN, RESERVE_TOKENS, MIN_ROOM = 3.5, 9216, 4000


def room(turn, spec):
    cap = int(spec.get("model_context") or turn.a2.get("model_context") or 131072)
    held = sum(len(str(getattr(m, "content", "") or "")) for m in turn.messages)
    return int((cap - RESERVE_TOKENS) * CHARS_PER_TOKEN) - held


def deliver(turn):
    spec = turn.a2.get("LB7") or {}
    tools = {fam(x) for x in spec.get("deliver_for") or []}
    for c in turn.calls:
        name = turn.name_of(c)
        if fam(name) not in tools:
            continue
        unread = [d for d in docs_naming(name, turn.corpus) if d.lower() not in turn.tool_text]
        if not unread:
            continue
        limit = min(int(spec.get("max_chars") or 90000), room(turn, spec))
        if limit < MIN_ROOM:
            return [Finding(LB, SURFACE, fam(name), grade=RETRIEVED, source="deliver",
                            facts=["[MATERIAL] The document(s) defining '%s' are unread and the context has no room "
                                   "to deliver them: %s." % (name, ", ".join(unread))])]
        blob = "\n\n".join("### %s\n%s" % (d, turn.corpus[d]) for d in unread)[:limit]
        # the material has to reach the model before the call runs, so the call is refused once and
        # the material is the refusal; a surfaced note on a call turn is only logged, never read
        return [Finding(LB, DENY, fam(name), c, fill(HEAD, tool=name) + "\n\n" + blob, grade=RETRIEVED,
                        source="deliver")]
    return []


def have_value(turn):
    out, said = [], turn.am_text.lower()
    for sp in (turn.a2.get("LB7") or {}).get("have_value") or []:
        signals = [str(x).lower() for x in sp.get("reask_signals") or []]
        if not signals or not any(x in said for x in signals) or fam(sp.get("write", "")) in turn.executed_fams():
            continue
        outs = [o for o in turn.tool_outputs() if str(sp.get("producer_marker", "")).lower() in o.lower()]
        if outs:
            value = _after(outs[-1], sp.get("value_after"))
            if value and sp.get("feedback"):
                out.append(Finding(LB, SURFACE, fam(sp.get("write")), grade=RETRIEVED, source="have-value",
                                   order=fill(sp["feedback"], value=value, arg=sp.get("arg"), write=sp.get("write"))))
        elif sp.get("acquire_feedback") and sp.get("acquire_tool") and not _given(turn, sp):
            out.append(Finding(LB, SURFACE, fam(sp["acquire_tool"]), grade=RETRIEVED, source="value-acquire",
                               order=fill(sp["acquire_feedback"], arg=sp.get("arg"), acquire_tool=sp["acquire_tool"],
                                          give_tool=sp.get("give_tool"), write=sp.get("write"))))
    return out


def _after(text, marker):
    if not marker or marker not in text:
        return ""
    return (text.split(marker, 1)[1].split() or [""])[0].strip(".,;:")


def _given(turn, sp):
    return any(getattr(c, "name", None) == sp.get("give_tool") and sp["acquire_tool"] in str(c.arguments)
               for m in turn.messages for c in (getattr(m, "tool_calls", None) or []))


def evaluate(turn):
    return deliver(turn) + have_value(turn)


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name):
            self.name, self.arguments, self.id = name, {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    corpus = {"doc_a": "How to use transfer_x: search first", "doc_b": "unrelated"}
    A2 = {"LB7": {"deliver_for": ["transfer_x"]}}
    t = Turn(A2, [M("tool", "grep hit only")], M(calls=[C("transfer_x_1")]), corpus=corpus)
    assert deliver(t)[0].primitive == DENY and "### doc_a" in deliver(t)[0].order and "doc_b" not in deliver(t)[0].order
    assert not deliver(Turn(A2, [M("tool", "### doc_a\n...")], M(calls=[C("transfer_x_1")]), corpus=corpus))
    full = Turn(dict(A2, model_context=20000), [M("tool", "z" * 60000)], M(calls=[C("transfer_x_1")]), corpus=corpus)
    assert deliver(full)[0].facts[0].startswith("[MATERIAL] The document(s) defining") and "doc_a" in deliver(full)[0].facts[0]
    A2["LB7"]["have_value"] = [{"write": "file_x", "arg": "last4", "producer_marker": "Executed: get_last4",
                                "value_after": "Last 4 digits of card:", "reask_signals": ["last 4"],
                                "feedback": "you have {arg}={value}; file {write}", "acquire_tool": "get_last4",
                                "give_tool": "give", "acquire_feedback": "give {acquire_tool} via {give_tool}"}]
    have = Turn(A2, [M("tool", "Executed: get_last4 ... Last 4 digits of card: 5320.")], M(content="What are the last 4 digits?"))
    assert have_value(have)[0].order == "you have last4=5320; file file_x"
    need = Turn(A2, [M("user", "hi")], M(content="please tell me the last 4 digits"))
    assert have_value(need)[0].order == "give get_last4 via give"
    print("lb7_material self-test OK")
