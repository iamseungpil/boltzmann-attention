# -*- coding: utf-8 -*-
"""LB7 - material at hand (a value already produced is handed back, not re-asked).

One rule: hand back what the conversation already holds instead of asking for it again. This engine
orders nothing and blocks nothing. Declared in A2["LB7"]:

  have_value    [{write, arg, producer_marker, value_after, reask_signals, feedback, acquire_tool, give_tool,
                  acquire_feedback}]  a value a producer already returned is handed back instead of re-asked;
                  when no producer ran and the customer keeps being asked, the acquiring tool is named
"""

from lb_coordinator import Finding, SURFACE, GRADES, fam, fill

LB = "LB7"
RETRIEVED = GRADES["retrieved_prose"]
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
    return have_value(turn)


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
    need = Turn(A2, [M("user", "hi")], M(content="please tell me the last 4 digits"))
    assert have_value(need)[0].order == "give get_last4 via give"
    print("lb7_material self-test OK")
