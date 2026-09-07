# -*- coding: utf-8 -*-
"""LB1 - requirement-graph gate (mechanism F1, compliance / guarantee).

One rule: a call may run only after every prerequisite the policy names for it has run, and a
running procedure's prohibitions hold. Both come from A2["LB1"]:

  prerequisites  [{dep, reads}]            dep needs each of reads first (transitive)
  gates          [{id, predicate, satisfiers, applies_to, exempt}]   an action needs a satisfier
  procedures     [{id, enforce, enter_when{tool_any, signals}, nodes[{id, tool|tool_any|tool_prefix,
                   requires, min_count}], prohibits{name: {quote}}, feedback{unmet, absent, prohibited}}]

A procedure is active once one of its own tools ran or the customer's wording matched its signals
(customer text only - defect 048). It blocks only when enforce is true and it quotes the policy
sentence that licenses the order; otherwise it surfaces. The engine names no tool and writes no
sentence of its own: every template is the declaration's. Task-specific cases are data here.
"""

from lb_coordinator import Finding, DENY, SURFACE, PIN, GRADES, fam, fill

LB = "LB1"
POLICY = GRADES["policy_verbatim"]


# ---- prerequisite graph -------------------------------------------------------------------------
def edges_for(a2, target):
    """{tool: [prerequisite tools]} for one target: declared reads plus the gates covering it."""
    spec = a2.get("LB1") or {}
    edges = {}
    for p in spec.get("prerequisites") or []:
        edges.setdefault(fam(p.get("dep")), []).extend(fam(r) for r in p.get("reads") or [])
    for g in spec.get("gates") or []:
        if target in (g.get("applies_to") or []) and target not in (g.get("exempt") or []):
            edges.setdefault(fam(target), []).extend(fam(s) for s in g.get("satisfiers") or [])
    return edges


def first_step(name, done, edges, seen=()):
    """The step that can be taken right now on the way to `name` (the graph walked to its root)."""
    n = fam(name)
    if n in done or n in seen:
        return None if n in done else n
    for p in edges.get(n) or []:
        step = first_step(p, done, edges, set(seen) | {n})
        if step:
            return step
    return n


def requirements_for(a2, target, done):
    """Unmet requirements of `target`, each with the step to take now - all of them, not the first."""
    spec, t, edges, out = a2.get("LB1") or {}, fam(target), edges_for(a2, target), []
    for g in spec.get("gates") or []:
        sat = [fam(s) for s in g.get("satisfiers") or []]
        if target not in (g.get("applies_to") or []) or target in (g.get("exempt") or []):
            continue
        if sat and not set(sat) & done:
            step = [x for x in (first_step(s, done, edges) for s in sat) if x][:1] or sat
            out.append({"id": g.get("id"), "predicate": g.get("predicate") or g.get("id"), "satisfiers": step})
    miss = [first_step(r, done, edges) or r for r in edges.get(t) or [] if r not in done]
    miss = [m for m in dict.fromkeys(miss) if m not in {s for r in out for s in r["satisfiers"]}]
    if miss:
        out.append({"id": "reads:" + ",".join(miss), "satisfiers": miss,
                    "predicate": "the prior read(s) this action requires have been done"})
    return out


def merged_text(a2, reqs, target):
    """One imperative, the rest declarative (a list of four orders lost three of them on task 101)."""
    fb = (a2.get("LB1") or {}).get("feedback") or {}
    r = reqs[0]
    first = "%s (do it with: %s)" % (r["predicate"], ", ".join(r["satisfiers"]))
    if len(reqs) == 1:
        return fill(fb.get("single") or DEFAULT_SINGLE, target=target, requirement=first)
    rest = "; ".join(x["predicate"] for x in reqs[1:])
    return fill(fb.get("merged") or DEFAULT_MERGED, target=target, first=first, rest=rest)


DEFAULT_SINGLE = ("Error: [ORDER] '{target}' cannot be carried out yet. This has to hold first: "
                  "{requirement}. Do that now with the real tool calls.")
DEFAULT_MERGED = ("Error: [ORDER] '{target}' cannot be carried out yet.\nDo this now, with a real tool call: "
                  "{first}\nStill outstanding after that: {rest}")


# ---- procedures -----------------------------------------------------------------------------------
def _tools(node):
    return [node["tool"]] if node.get("tool") else list(node.get("tool_any") or [])


def _matches(node, name):
    return name in _tools(node) or (node.get("tool_prefix") and name.startswith(node["tool_prefix"]))


def _done(node, executed):
    tools = _tools(node)
    if not tools and not node.get("tool_prefix"):
        return None                                   # unobservable step (a bound to check)
    n = sum(v for k, v in executed.items() if _matches(node, k))
    return n >= int(node.get("min_count") or 1)


def active(procs, executed, user_text):
    ran = set(executed)
    out = []
    for p in procs:
        ew = p.get("enter_when") or {}
        if set(ew.get("tool_any") or []) & ran or any(s.lower() in user_text for s in ew.get("signals") or []):
            out.append(p)
    return out


def unmet(proc, node, executed):
    """Prerequisite node ids of `node` that have not run (transitive, declaration order)."""
    by_id = {n["id"]: n for n in proc.get("nodes") or []}
    seen, out, stack = set(), [], list(node.get("requires") or [])
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in by_id:
            continue
        seen.add(nid)
        if _done(by_id[nid], executed) is False:
            out.append(nid)
        stack.extend(by_id[nid].get("requires") or [])
    return sorted(out, key=[n["id"] for n in proc.get("nodes") or []].index)


def ready(proc, executed):
    """Nodes not done whose own prerequisites are all done."""
    return [n for n in proc.get("nodes") or []
            if _done(n, executed) is False and not unmet(proc, n, executed)]


def mandatory(proc):
    return bool(proc.get("enforce")) and bool(proc.get("_quote_order"))


def state_slots(proc, executed, unlocked):
    rows, done = [], 0
    for n in proc.get("nodes") or []:
        ok = _done(n, executed)
        done += ok is True
        rows.append("[%s] %s%s" % ("x" if ok else ("?" if ok is None else " "), n["id"],
                                   (" -> " + "/".join(_tools(n))) if ok is False and _tools(n) else ""))
    cands = ready(proc, executed)
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
        missing = unmet(p, node, turn.executed)
        if not missing:
            continue
        slots = state_slots(p, turn.executed, turn.unlocked)
        slots["unlock_hint"] = fill(slots["unlock_hint"], **slots)
        text = fill(fb.get("unmet", ""), tool=name, missing=", ".join(missing),
                    source=", ".join(p.get("_source") or [])[:120], **slots)
        if mandatory(p):
            pin = _pin(turn, p, missing)
            return [Finding(LB, DENY, fam(name), call, text, grade=POLICY, source="procedure:" + p["id"], pin=pin)]
        return [Finding(LB, SURFACE, fam(name), facts=[text], grade=POLICY, source="procedure:" + p["id"])]
    return []


def _pin(turn, proc, missing):
    """Pin the next generation to the missing step's tool when it is a single, declared tool."""
    d = turn.a2.get("dispatch") or {}
    node = next((n for n in proc.get("nodes") or [] if n["id"] == missing[0]), {})
    tools = _tools(node)
    if len(tools) == 1 and d.get("agent_call") and turn.name_args.get(d["agent_call"]):
        return (d["agent_call"], turn.name_args[d["agent_call"]], tools[0])
    return None


def absent_findings(turn):
    """A procedure was entered and this turn takes no step: surface where it stands."""
    out = []
    for p in active((turn.a2.get("LB1") or {}).get("procedures") or [], turn.executed, turn.user_text):
        fb, slots = p.get("feedback") or {}, state_slots(p, turn.executed, turn.unlocked)
        if not (slots["next"] or slots["ready_tools"]):
            continue
        slots["unlock_hint"] = fill(slots["unlock_hint"], **slots)
        tpl = fb.get("absent") if slots["next"] else fb.get("absent_many")
        if tpl:
            f = Finding(LB, SURFACE, slots["next_tool"] or p["id"], order=fill(tpl, **slots), grade=POLICY,
                        source="absent:" + p["id"])
            if slots["next_tool"] and mandatory(p):
                f.primitive, f.pin = PIN, _pin(turn, p, [ready(p, turn.executed)[0]["id"]])
            out.append(f)
    return out


def evaluate(turn):
    out = []
    for c in turn.calls:
        name = turn.name_of(c)
        reqs = requirements_for(turn.a2, fam(name), turn.executed_fams())
        if reqs:
            out.append(Finding(LB, DENY, fam(name), c, grade=POLICY, source="requirement", reqs=reqs))
        out += procedure_findings(turn, c)
    if not turn.calls:
        out += absent_findings(turn)
    return out


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name, args=None):
            self.name, self.arguments, self.id = name, args or {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    A2 = {"dispatch": {"agent_call": "call", "name_args": {"call": "tool"}},
          "LB1": {"prerequisites": [{"dep": "log_verification", "reads": ["verify_identity"]}],
                  "gates": [{"id": "G", "predicate": "identity verified", "satisfiers": ["log_verification"],
                             "applies_to": ["submit_referral"]}],
                  "procedures": [{"id": "p", "enforce": True, "_quote_order": "MUST",
                                  "enter_when": {"tool_any": ["trigger"], "signals": ["please do the thing"]},
                                  "nodes": [{"id": "a", "tool": "read_a"}, {"id": "b", "tool": "write_b", "requires": ["a"]},
                                            {"id": "c", "tool_prefix": "credit_", "requires": ["b"]}],
                                  "prohibits": {"forbidden_x": {"quote": "Do not"}},
                                  "feedback": {"unmet": "[P] before '{tool}': {missing}", "absent": "[P] NEXT: {next}",
                                               "prohibited": "[P] '{tool}' forbidden: {quote}"}}]}}
    # gate walks to the executable root: log_verification needs verify_identity first
    r = requirements_for(A2, "submit_referral", set())
    assert r[0]["satisfiers"] == ["verify_identity"], r
    assert "verify_identity" in merged_text(A2, r, "submit_referral")
    t = Turn(A2, [M("user", "Please do the thing")], M(calls=[C("write_b")]))
    f = evaluate(t)
    assert f[0].primitive == DENY and f[0].order == "[P] before 'write_b': a"
    t = Turn(A2, [M("user", "please do the thing")], M(calls=[C("call", {"tool": "credit_apply_1"})]),
             executed={"read_a": 1})
    assert evaluate(t)[0].order == "[P] before 'credit_apply_1': b"       # tool_prefix node, dispatcher unwrapped
    # 048: the signal only in tool output does not open the procedure
    t = Turn(A2, [M("tool", "policy: please do the thing")], M(calls=[C("write_b")]))
    assert not evaluate(t)
    t = Turn(A2, [M("user", "please do the thing")], M(content="ok"), executed={"read_a": 1})
    a = absent_findings(t)
    assert a[0].primitive == PIN and a[0].pin == ("call", "tool", "write_b"), a[0].pin
    t = Turn(A2, [], M(calls=[C("forbidden_x")]), executed={"trigger": 1})
    assert evaluate(t)[0].source == "prohibit:p"
    print("lb1_requirements self-test OK")
