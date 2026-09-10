# -*- coding: utf-8 -*-
"""LB1 - requirement-graph gate (mechanism F1, compliance / guarantee).

One rule, one walker: a state-changing call may run only after every step the policy names before it
has run, and a running procedure's prohibitions hold. Everything is a procedure DAG in A2["LB1"]:
  procedures     [{id, enforce, enter_when{tool_any, signals} | absent = always active,
                   nodes[{id, tool|tool_any|tool_prefix, requires, min_count}],
                   prohibits{name: {quote}}, feedback{unmet, prohibited}}]
  write_tools    [name]                    the calls that change state; only these can be denied for
                                           being out of order - reading in a different order changes
                                           nothing, and blocking a read only costs turns
A policy prerequisite ("verify before account access") is a two-node procedure that is always active;
a multi-step protocol enters when one of its own tools ran or the customer's wording matched its
signals (customer text only - defect 048). It blocks only when enforce is true and it quotes the
policy sentence that licenses the order; otherwise it surfaces. The engine names no tool and writes no
sentence of its own: every template is the declaration's. Task-specific cases are data here.
"""

from lb_coordinator import Finding, DENY, SURFACE, PIN, GRADES, fam, fill

LB = "LB1"
POLICY = GRADES["policy_verbatim"]


# ---- procedures -----------------------------------------------------------------------------------
def _tools(node):
    return [node["tool"]] if node.get("tool") else list(node.get("tool_any") or [])


def _matches(node, name):
    return name in _tools(node) or (node.get("tool_prefix") and name.startswith(node["tool_prefix"]))


def _done(node, executed, settled=()):
    if node.get("id") in settled:
        return True                        # settled without a call: waived by the source, or already said
    tools = _tools(node)
    if not tools and not node.get("tool_prefix"):
        # A step the source lets you take more than one way is done when any of them is on the
        # record - which one to take is the caller's judgement, not ours. With no way at all
        # declared the step is unobservable and we say so rather than guess.
        return False if (node.get("said_tokens") or node.get("said_any")) else None
    n = sum(v for k, v in executed.items() if _matches(node, k))
    return n >= int(node.get("min_count") or 1)


def active(procs, executed, user_text):
    ran = set(executed)
    out = []
    for p in procs:
        ew = p.get("enter_when")
        if not ew or set(ew.get("tool_any") or []) & ran or any(s.lower() in user_text for s in ew.get("signals") or []):
            out.append(p)
    return out


def settled_nodes(proc, turn, call):
    """Node ids this procedure needs no tool call for.

    Two ways a step is settled without a call. One, the source's own conditional takes it out.

    The retention protocol reads: "If records exist for this account within that time frame, skip
    retention offers and proceed directly to processing the closure." task_049's Green card carried
    such a record; the model read it, said it would close directly, and we denied the closure until
    it logged a reason. That put a row on an account gold never touches - four simulations out of
    four - and the reason word it invented there was the one it then reused on the account gold does
    judge, where the customer had given a different reason out loud. The tokens are the format
    string the environment prints for itself, checked for co-presence with the subject id; nothing
    here parses the record. The policy bounds this by "within the past year" and this test does not
    read the date, so a record older than a year waives a step the policy would still want.
    """
    subject = turn.args_of(call).get(proc["id_key"]) if proc.get("id_key") else None
    out = set()
    for n in proc.get("nodes") or []:
        toks = n.get("skip_when_tokens")
        if toks:
            for o in turn.tool_outputs():
                if all(t in o for t in toks) and (subject is None or str(subject) in o):
                    out.add(n["id"])
                    break
        # Two, the step is a disclosure: it is done when we have already told the customer, on an
        # earlier turn. GB2 - the ask-first notice before a transfer - is that kind, and migration
        # dropped it because it could only express a gate as a tool prerequisite. task_088
        # transferred with no notice sent and no write performed.
        toks = n.get("said_tokens")
        # turn.said is normalised to lower case; the declared sentence is not
        if toks and all(str(t).lower() in turn.said for t in toks):
            out.add(n["id"])
        # said_any is the disjunction: the source names several ways to take one step, and any of
        # the words it uses for them settles it. The engine counts; the caller chooses.
        alt = n.get("said_any")
        if alt and any(str(t).lower() in turn.said for t in alt):
            out.add(n["id"])
    return out


def unmet(proc, node, executed, settled=()):
    """Prerequisite node ids of `node` that have not run (transitive, declaration order)."""
    by_id = {n["id"]: n for n in proc.get("nodes") or []}
    seen, out, stack = set(), [], list(node.get("requires") or [])
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in by_id:
            continue
        seen.add(nid)
        if _done(by_id[nid], executed, settled) is False:
            out.append(nid)
        stack.extend(by_id[nid].get("requires") or [])
    return sorted(out, key=[n["id"] for n in proc.get("nodes") or []].index)


def ready(proc, executed, settled=()):
    """Nodes not done whose own prerequisites are all done."""
    return [n for n in proc.get("nodes") or []
            if _done(n, executed, settled) is False and not unmet(proc, n, executed, settled)]


def mandatory(proc):
    return bool(proc.get("enforce")) and bool(proc.get("_quote_order"))


def changes_state(a2, name):
    """Is this call one the declaration lists as changing state?

    The closure protocol says to check disputes, then replacement cards, then age, then balance.
    On task_048 the model reads those three in exactly that order in every base simulation - and our
    walker still denied two of them ten times, because it was asked before the first read had been
    seen. Denying a read to enforce reading order buys nothing: the read has no effect to undo, and
    the model spends its turns re-asking. The order that matters is the order of the actions, so the
    deny is kept for those and the walker surfaces its checklist for everything else.
    """
    writes = {fam(w) for w in (a2.get("LB1") or {}).get("write_tools") or []}
    return not writes or fam(name) in writes


def state_slots(proc, executed, unlocked, settled=()):
    rows, done = [], 0
    for n in proc.get("nodes") or []:
        ok = _done(n, executed, settled)
        done += ok is True
        rows.append("[%s] %s%s" % ("x" if ok else ("?" if ok is None else " "), n["id"],
                                   (" -> " + "/".join(_tools(n))) if ok is False and _tools(n) else ""))
    cands = ready(proc, executed, settled)
    nxt = cands[0] if len(cands) == 1 and mandatory(proc) else None
    ntool = (_tools(nxt) or [""])[0] if nxt else ""
    return {"procedure": proc.get("id", ""), "done": done, "total": len(proc.get("nodes") or []),
            "checklist": "  ".join(rows), "next": ("%s -> %s" % (nxt["id"], ntool)) if nxt else "",
            "next_tool": ntool, "ready_tools": ", ".join(t for n in cands for t in _tools(n)),
            "unlock_hint": (proc.get("feedback") or {}).get("unlock_hint", "")
            if ntool and ntool not in unlocked else ""}


def procedure_findings(turn, call):
    procs = (turn.a2.get("LB1") or {}).get("procedures") or []
    name = turn.name_of(call)
    names = {name, turn.named(call)} - {""}
    for p in active(procs, turn.executed, turn.user_text):
        fb = p.get("feedback") or {}
        for nm in names:
            spec = (p.get("prohibits") or {}).get(nm)
            if spec and spec.get("quote"):
                return [Finding(LB, DENY, fam(nm), call, grade=POLICY, source="prohibit:" + p["id"],
                                order=fill(fb.get("prohibited", ""), tool=nm, quote=spec["quote"]))]
        node = next((n for n in p.get("nodes") or [] if _matches(n, name)), None)
        if node is None:
            continue
        settled = settled_nodes(p, turn, call)
        missing = unmet(p, node, turn.executed, settled)
        if not missing:
            continue
        slots = state_slots(p, turn.executed, turn.unlocked, settled)
        slots["unlock_hint"] = fill(slots["unlock_hint"], **slots)
        text = fill(fb.get("unmet", ""), tool=name, missing=", ".join(missing),
                    source=", ".join(p.get("_source") or [])[:120], **slots)
        # A gate names the one tool it guards, and that naming is the warrant to block it. The
        # transfer tool is not on the environment's write list, so without this a required
        # disclosure could only be mentioned, never waited for - and task_088 transferred anyway.
        if mandatory(p) and (changes_state(turn.a2, name) or p.get("blocks")):
            pin = _pin(turn, p, missing)
            return [Finding(LB, DENY, fam(name), call, text, grade=POLICY, source="procedure:" + p["id"], pin=pin)]
        # Nothing was blocked here - the call proceeds. Saying "cannot be carried out" reports a
        # refusal that did not happen (016: the read tool get_referrals_by_user was told exactly that).
        surface = fill(fb.get("unmet_surface") or fb.get("unmet", ""), tool=name, missing=", ".join(missing),
                       source=", ".join(p.get("_source") or [])[:120], **slots)
        return [Finding(LB, SURFACE, fam(name), facts=[surface], grade=POLICY, source="procedure:" + p["id"])]
    return []


def _pin(turn, proc, missing):
    """Pin the next generation to the missing step's tool when it is a single, declared tool."""
    d = turn.a2.get("dispatch") or {}
    node = next((n for n in proc.get("nodes") or [] if n["id"] == missing[0]), {})
    tools = _tools(node)
    if len(tools) == 1 and d.get("agent_call") and turn.name_args.get(d["agent_call"]):
        return (d["agent_call"], turn.name_args[d["agent_call"]], tools[0])
    return None


def evaluate(turn):
    return [f for c in turn.calls for f in procedure_findings(turn, c)]
    # A text turn inside a procedure is not judged here. The model asking the customer a question is a
    # text turn too (043: identity details at message 5), and a walker that pinned the next step there
    # fired 180 times over 216 base simulations. Leaving the procedure open at hand-off is LB5's case.


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name, args=None):
            self.name, self.arguments, self.id = name, args or {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    A2 = {"dispatch": {"agent_call": "call", "name_args": {"call": "tool"}},
          "LB1": {"procedures": [{"id": "requires:submit_referral", "enforce": True, "_quote_order": "MUST",
                                  "nodes": [{"id": "log_verification", "tool_prefix": "log_verification"},
                                            {"id": "submit_referral", "tool_prefix": "submit_referral",
                                             "requires": ["log_verification"]}],
                                  "feedback": {"unmet": "[R] '{tool}' needs: {missing}"}},
                                 {"id": "p", "enforce": True, "_quote_order": "MUST",
                                  "enter_when": {"tool_any": ["trigger"], "signals": ["please do the thing"]},
                                  "nodes": [{"id": "a", "tool": "read_a"}, {"id": "b", "tool": "write_b", "requires": ["a"]},
                                            {"id": "c", "tool_prefix": "credit_", "requires": ["b"]}],
                                  "prohibits": {"forbidden_x": {"quote": "Do not"}},
                                  "feedback": {"unmet": "[P] before '{tool}': {missing}",
                                               "prohibited": "[P] '{tool}' forbidden: {quote}"}}]}}
    A2["LB1"]["write_tools"] = ["write_b", "credit_apply", "submit_referral"]
    # an always-active two-node procedure is the old prerequisite: the write waits for its read
    t = Turn(A2, [M("user", "hi")], M(calls=[C("submit_referral_9")]))
    assert evaluate(t)[0].primitive == DENY and evaluate(t)[0].order == "[R] 'submit_referral_9' needs: log_verification"
    assert not evaluate(Turn(A2, [], M(calls=[C("submit_referral_9")]), executed={"log_verification": 1}))
    t = Turn(A2, [M("user", "Please do the thing")], M(calls=[C("write_b")]))
    f = evaluate(t)
    assert f[0].primitive == DENY and f[0].order == "[P] before 'write_b': a"
    # a read out of order is said, not blocked
    A2["LB1"]["procedures"][1]["nodes"].append({"id": "r", "tool": "read_z", "requires": ["a"]})
    t = Turn(A2, [M("user", "Please do the thing")], M(calls=[C("read_z")]))
    assert evaluate(t)[0].primitive == SURFACE
    A2["LB1"]["procedures"][1]["nodes"].pop()
    t = Turn(A2, [M("user", "please do the thing")], M(calls=[C("call", {"tool": "credit_apply_1"})]),
             executed={"read_a": 1})
    assert evaluate(t)[0].order == "[P] before 'credit_apply_1': b"       # tool_prefix node, dispatcher unwrapped
    # 048: the signal only in tool output does not open the procedure
    t = Turn(A2, [M("tool", "policy: please do the thing")], M(calls=[C("write_b")]))
    assert not evaluate(t)
    t = Turn(A2, [M("user", "please do the thing")], M(content="which card?"), executed={"read_a": 1})
    assert not evaluate(t)                                            # a question mid-procedure is not a finding
    t = Turn(A2, [], M(calls=[C("forbidden_x")]), executed={"trigger": 1})
    assert evaluate(t)[0].source == "prohibit:p"
    print("lb1_requirements self-test OK")
