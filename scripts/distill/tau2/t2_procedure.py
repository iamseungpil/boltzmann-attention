# -*- coding: utf-8 -*-
"""Walk a procedure the policy states, and say which of its steps is still missing.

Some policies are not a rule to check at one moment but an ordered procedure: submit
the request first, then run these four checks, then decide. `task_051` fails that way —
the decision tool is called while step 1 and one of the checks have never happened, and
an argument that the tier table fixes is filled with a different number. Nothing there
is a judgement; every part of it is written down.

So the procedure lives in A2 as a DAG (`procedures`, L3) and this file is the walker.
It contains no tool name, no field name and no number: those come from the declaration.
What it does is decide three things about a call that enters a procedure:

  unmet          nodes whose prerequisites have not run yet, in declaration order
  arg-from-table an argument whose value the declaration derives from a table
  cap            a bound the declaration attaches to a node

and return the feedback string the declaration itself supplies. It never rewrites an
argument and never blocks a call — the engine states what the policy requires and the
model acts ([[10]]: generator is the model, the check is deterministic).

The operand a table is keyed by (a tier, a class, a category) is supplied by the model,
because the corpus may not map entities to it. When it is missing the table lookup is
skipped rather than guessed — an engine that guesses the key has decided the case.
"""

import re

__all__ = ["find_procedure", "unmet_nodes", "table_expectation", "notes_for_call"]


def _tools_of(node):
    t = node.get("tool")
    if t:
        return [t]
    return list(node.get("tool_any") or [])


def find_procedure(procs, tool_name):
    """The declared procedure this call belongs to, or None.

    Membership is an index, not an inference: a call is inside a procedure only when
    the declaration lists its tool, either as an entry point or on a node. Everything
    else is outside and this file has nothing to say about it.
    """
    for p in procs or []:
        triggers = (p.get("enter_when") or {}).get("tool_any") or []
        if tool_name in triggers:
            return p
        for n in p.get("nodes") or []:
            if tool_name in _tools_of(n):
                return p
    return None


def is_mandatory(proc):
    """Does the policy itself say the order must be followed?

    Enforcement is licensed by the declaration, never by this file: a procedure is
    enforced only when it carries `enforce: true` together with the sentence that
    licenses it (`_quote_order`). A procedure that merely describes a usual sequence
    is surfaced and not blocked, because blocking one the policy did not mandate is
    us inventing a rule.
    """
    return bool((proc or {}).get("enforce")) and bool((proc or {}).get("_quote_order"))


def _node_of(proc, tool_name):
    for n in proc.get("nodes") or []:
        if tool_name in _tools_of(n):
            return n
    return None


def _satisfied(node, executed):
    """A node counts as done when any tool it names has already been executed.

    Nodes that name no tool (a bound the agent must check, not call) cannot be
    observed from the call history; they are reported separately rather than
    silently treated as done.
    """
    tools = _tools_of(node)
    if not tools:
        return None
    return any(t in executed for t in tools)


def unmet_nodes(proc, tool_name, executed):
    """Prerequisites of this call that have not run, in the order declared.

    Walks the requires-graph transitively, so a node that depends on a node that
    depends on the missing step is reported once at the missing step.
    """
    node = _node_of(proc, tool_name)
    if node is None:
        return [], []
    by_id = {n.get("id"): n for n in proc.get("nodes") or []}
    order = [n.get("id") for n in proc.get("nodes") or []]
    seen, missing, unobservable = set(), [], []
    stack = list(node.get("requires") or [])
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in by_id:
            continue
        seen.add(nid)
        n = by_id[nid]
        ok = _satisfied(n, executed)
        if ok is None:
            unobservable.append(nid)
        elif not ok:
            missing.append(nid)
        stack.extend(n.get("requires") or [])
    missing.sort(key=lambda i: order.index(i) if i in order else 0)
    unobservable.sort(key=lambda i: order.index(i) if i in order else 0)
    return missing, unobservable


def table_expectation(proc, tool_name, operands):
    """(arg, expected) when the declaration derives an argument from a table."""
    node = _node_of(proc, tool_name)
    if not node:
        return None
    spec = node.get("arg_from_table")
    if not spec:
        return None
    key = (operands or {}).get(spec.get("operand"))
    if key is None:
        return None                     # the model has not supplied the key — do not guess
    table = (proc.get("tables") or {}).get(spec.get("table")) or {}
    if str(key) not in table:
        return None
    return spec.get("arg"), table[str(key)]


def _fill(tpl, **kw):
    out = str(tpl or "")
    for k, v in kw.items():
        out = out.replace("{%s}" % k, str(v))
    return re.sub(r"\{[a-z_]+\}", "", out).strip()


def notes_for_call(procs, tool_name, args, executed, operands=None):
    """Every line the declaration says about this call. Empty when it says nothing."""
    proc = find_procedure(procs, tool_name)
    if proc is None:
        return []
    fb = proc.get("feedback") or {}
    out = []
    missing, _unobs = unmet_nodes(proc, tool_name, executed)
    if missing and fb.get("unmet"):
        out.append(_fill(fb["unmet"], tool=tool_name, missing=", ".join(missing),
                         source=", ".join(proc.get("_source") or [])[:120]))
    te = table_expectation(proc, tool_name, operands)
    if te and fb.get("arg_from_table"):
        arg, expected = te
        given = (args or {}).get(arg)
        if given is None or str(given) != str(expected):
            out.append(_fill(fb["arg_from_table"], tool=tool_name, arg=arg, value=expected))
    return out


def decide(procs, tool_name, args, executed, operands=None):
    """One verdict for one call: what is missing, what to say, and whether to block.

    `deny` is returned only when the declaration mandates the order (§is_mandatory) and
    a prerequisite that can be observed from the call history has not run. Nodes that
    cannot be observed — a bound the agent is told to check rather than call — never
    produce a block, because their absence is not a fact this engine holds.
    """
    proc = find_procedure(procs, tool_name)
    if proc is None:
        return {"procedure": None, "missing": [], "notes": [], "verdict": "pass"}
    missing, unobservable = unmet_nodes(proc, tool_name, executed)
    notes = notes_for_call(procs, tool_name, args, executed, operands)
    verdict = "deny" if (missing and is_mandatory(proc)) else "pass"
    return {"procedure": proc.get("id"), "missing": missing, "unobservable": unobservable,
            "notes": notes, "verdict": verdict}
