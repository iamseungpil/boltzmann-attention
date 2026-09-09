# -*- coding: utf-8 -*-
"""LB7 - material at hand (a value already produced is handed back, not re-asked).

One rule: hand back what the conversation already holds instead of asking for it again. This engine
orders nothing and blocks nothing. Declared in A2["LB7"]:

  have_value    [{write, arg, producer_marker, value_after, reask_signals, feedback, acquire_tool, give_tool,
                  acquire_feedback}]  a value a producer already returned is handed back instead of re-asked;
                  when no producer ran and the customer keeps being asked, the acquiring tool is named
  write_rules   [{applies_to, text}]  a policy sentence carried to the decision point. The rule
                  reaches the conversation early, in a retrieved document, and the write comes much
                  later; by then it is far away. Surfaced when the model unlocks that write tool,
                  which is the last moment before it composes the call. Isolation x537 (085, n=4):
                  the decision-point window as it stands 0/12, this sentence placed at the decision
                  point 12/12, an unrelated sentence of the same length 0/12. The engine neither
                  searches nor ranks - it carries the sentence.
  action_index  {text, rows[{title, tools}]}  the documents that describe an action, with the tool each
                  one names. Printed before the first retrieval and nowhere else: once material comes
                  back it is more specific than a list of titles, and adding to it hurts. The engine
                  prints; it does not choose. Measured in isolation at x319 (n=24, blocks of 8): no
                  help 10/24, these titles 24/24, the 91 tool descriptions 23/24, the 91 bare names
                  16/24 - meaning beats naming, and the cheapest of the three is the best.
"""

from lb_coordinator import Finding, SURFACE, GRADES, as_dict, fam, fill

LB = "LB7"
RETRIEVED = GRADES["retrieved_prose"]
def have_value(turn):
    out, said = [], turn.am_text.lower()
    for sp in (turn.a2.get("LB7") or {}).get("have_value") or []:
        signals = [str(x).lower() for x in sp.get("reask_signals") or []]
        if not signals or not any(x in said for x in signals) or fam(sp.get("write", "")) in turn.executed_fams():
            continue
        # only when the retrieved material names the acquiring tool: on task_070 (a business account
        # opening) this pointed at a card-digit tool nothing retrieved had mentioned, twice
        if sp.get("acquire_tool") and fam(sp["acquire_tool"]).lower() not in turn.tool_text:
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


def action_index(turn):
    """The titles of the documents that describe an action, each with the tool it names.

    Only before the first retrieval: this is the fallback for having nothing, and once real material
    is back it is the more specific thing to read. Having no tool output yet happens on exactly one
    turn of a simulation, so this prints once by construction rather than by a flag.
    """
    spec = (turn.a2.get("LB7") or {}).get("action_index") or {}
    rows, head = spec.get("rows") or [], spec.get("text")
    if not rows or not head or turn.tool_outputs():
        return []
    lines = [head]
    for r in rows:
        title = " ".join(str(r.get("title") or "").split())
        tools = ", ".join(r.get("tools") or [])
        lines.append("- %s%s" % (title, (" [%s]" % tools) if tools else ""))
    return [Finding(LB, SURFACE, None, grade=RETRIEVED, source="action-index",
                     order=chr(10).join(lines))]


def write_rules(turn):
    """The policy sentence for a write, carried to the moment the model reaches for that write."""
    out, unlock = [], (turn.a2.get("dispatch") or {}).get("unlock_tool")
    reaching = set()
    for c in turn.calls:
        name = turn.name_of(c)
        reaching.add(fam(name))
        if unlock and name == unlock:
            asked = str(turn.args_of(c).get("agent_tool_name") or as_dict(c.arguments).get("agent_tool_name") or "")
            if asked:
                reaching.add(fam(asked))
    for sp in (turn.a2.get("LB7") or {}).get("write_rules") or []:
        if sp.get("text") and fam(sp.get("applies_to", "")) in reaching:
            out.append(Finding(LB, SURFACE, fam(sp["applies_to"]), grade=RETRIEVED, source="write-rule",
                               order=sp["text"]))
    return out


def evaluate(turn):
    return have_value(turn) + action_index(turn) + write_rules(turn)


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name):
            self.name, self.arguments, self.id = name, {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    A2 = {"LB7": {}}
    A2["LB7"]["have_value"] = [{"write": "file_x", "arg": "last4", "producer_marker": "Executed: get_last4",
                                "value_after": "Last 4 digits of card:", "reask_signals": ["last 4"],
                                "feedback": "you have {arg}={value}; file {write}", "acquire_tool": "get_last4",
                                "give_tool": "give", "acquire_feedback": "give {acquire_tool} via {give_tool}"}]
    have = Turn(A2, [M("tool", "Executed: get_last4 ... Last 4 digits of card: 5320.")], M(content="What are the last 4 digits?"))
    assert have_value(have)[0].order == "you have last4=5320; file file_x"
    assert not have_value(Turn(A2, [M("user", "hi")], M(content="please tell me the last 4 digits")))   # nothing retrieved names the tool
    need = Turn(A2, [M("tool", "doc: use get_last4 to read the digits")], M(content="please tell me the last 4 digits"))
    assert have_value(need)[0].order == "give get_last4 via give"
    A2["LB7"]["action_index"] = {"text": "Consult the relevant one.",
                                 "rows": [{"title": "Closing Personal Checking Accounts",
                                           "tools": ["close_bank_account_7392"]}]}
    first = Turn(A2, [M("user", "close my account")], M(content=""))
    assert action_index(first)[0].order.endswith("- Closing Personal Checking Accounts [close_bank_account_7392]")
    assert not action_index(Turn(A2, [M("tool", "1. some retrieved document")], M(content="")))   # material beats a list
    A2["dispatch"] = {"unlock_tool": "unlock"}
    A2["LB7"]["write_rules"] = [{"applies_to": "file_x", "text": "Dispute the earliest one."}]
    got = write_rules(Turn(A2, [M("user", "hi")], M(calls=[C("file_x_9")])))
    assert got and got[0].order == "Dispute the earliest one.", got
    assert not write_rules(Turn(A2, [M("user", "hi")], M(calls=[C("other")])))
    print("lb7_material self-test OK")
