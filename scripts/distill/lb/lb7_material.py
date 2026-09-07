# -*- coding: utf-8 -*-
"""LB7 - material delivery (the premise of every other engine).

One rule: give the model the candidate set it has not seen. Not a seventh mechanism: when the
candidates never reach the model, every mechanism fails for a reason unrelated to itself
(x829: without material 0/8, with 47 document titles 8/8). This engine orders nothing; it
surfaces - the whole set, verbatim from the corpus, no ranking. Declared in A2["LB7"]:

  deliver_for   tools whose defining documents are delivered in full when unread
  max_chars     delivery cap
  names_feedback  documents already retrieved name a tool nobody has called
"""

from lb_coordinator import Finding, SURFACE, GRADES, fam, fill

LB = "LB7"
RETRIEVED = GRADES["retrieved_prose"]
HEAD = ("[MATERIAL] Before using '{tool}', the document(s) that define it - which nothing you retrieved so far "
        "contains - are delivered below in full. Read them and choose; nothing here is blocked.")


def docs_naming(tool, corpus):
    f = fam(tool)
    return sorted(d for d, body in corpus.items() if f and f in str(body))


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
        blob = "\n\n".join("### %s\n%s" % (d, turn.corpus[d]) for d in unread)[:int(spec.get("max_chars") or 90000)]
        return [Finding(LB, SURFACE, fam(name), facts=[fill(HEAD, tool=name) + "\n\n" + blob], grade=RETRIEVED,
                        source="deliver")]
    return []


def named_uncalled(turn):
    tpl = (turn.a2.get("LB7") or {}).get("names_feedback")
    if turn.calls or not tpl:
        return []
    known = turn.executed_fams() | {fam(u) for u in turn.unlocked}
    named = sorted(r for r in turn.registry.get("agent", ()) if r.lower() in turn.tool_text and fam(r) not in known)
    return [Finding(LB, SURFACE, named[0], facts=[fill(tpl, names=", ".join(named))], grade=RETRIEVED,
                    source="named-uncalled")] if named else []


def evaluate(turn):
    return deliver(turn) + named_uncalled(turn)


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name):
            self.name, self.arguments, self.id = name, {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    corpus = {"doc_a": "How to use transfer_x: search first", "doc_b": "unrelated"}
    A2 = {"LB7": {"deliver_for": ["transfer_x"], "names_feedback": "named: {names}"}}
    t = Turn(A2, [M("tool", "grep hit only")], M(calls=[C("transfer_x_1")]), corpus=corpus)
    assert "### doc_a" in deliver(t)[0].facts[0] and "doc_b" not in deliver(t)[0].facts[0]
    assert not deliver(Turn(A2, [M("tool", "### doc_a\n...")], M(calls=[C("transfer_x_1")]), corpus=corpus))
    t2 = Turn(A2, [M("tool", "the doc names other_2")], M(content="done"), registry={"agent": {"other_2"}})
    assert named_uncalled(t2)[0].facts[0] == "named: other_2"
    print("lb7_material self-test OK")
