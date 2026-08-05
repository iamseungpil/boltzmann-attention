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

Names are matched exactly against the environment's own tool names. An earlier version
normalised a numeric suffix away, which is the kind of pattern rule this project has
already retired for producing quiet mismatches (C279) — the dispatched call carries the
exact name in its argument, so nothing has to be guessed from spelling.

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


def find_procedure(procs, tool_name, executed=()):
    """The procedure this call is inside, or None — membership needs the procedure to be active.

    Naming a tool on a node is not enough to claim a call belongs to the procedure: the
    same read is used by other flows, and treating every appearance as membership blocked
    28 calls gold wanted when this was measured over the run (x80). So a procedure counts
    as entered only once one of its **activating** tools has been called — the ones that
    exist for this procedure alone — and only then are its nodes checked. A shared read on
    its own leaves the procedure dormant and this file silent.
    """
    done = set(executed or ())
    for p in procs or []:
        triggers = set((p.get("enter_when") or {}).get("tool_any") or [])
        active = bool(triggers & done) or tool_name in triggers
        if not active:
            continue
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
    done = set(executed or ())
    return any(t in done for t in tools)


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
    proc = find_procedure(procs, tool_name, executed)
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


def prohibited(procs, names, executed):
    """(procedure, name, quote) when an active procedure forbids one of these names.

    Some policies do not order steps, they forbid one: the cash-back dispute document
    names the tool to hand over and then says not to collect card details. The engine
    holds neither fact — the prohibition and the sentence licensing it are declared, and
    a prohibition without its quote is ignored, exactly as an order without one is.

    `names` is a set because a handover names its tool inside the call: both the wrapper
    and what it passes are checked, so forbidding a tool also forbids handing it over.
    """
    for p in procs or []:
        block = (p.get("prohibits") or {})
        if not block:
            continue
        triggers = set((p.get("enter_when") or {}).get("tool_any") or [])
        if not (triggers & set(executed or ())) and not (triggers & set(names or ())):
            continue
        for nm in names or ():
            spec = block.get(nm)
            if spec and spec.get("_quote"):
                return p, nm, spec
    return None, None, None


def decide(procs, tool_name, args, executed, operands=None, also_names=()):
    """One verdict for one call: what is missing, what to say, and whether to block.

    `deny` is returned in two cases and no others: an active procedure whose declaration
    mandates the order is missing an observable prerequisite, or an active procedure
    forbids this tool and quotes the sentence that forbids it. Steps that cannot be
    observed — a bound the agent is told to check rather than call — never produce a
    block, because their absence is not a fact this engine holds.
    """
    names = {tool_name} | set(also_names or ())
    pproc, pname, pspec = prohibited(procs, names, executed)
    if pproc is not None:
        return {"procedure": pproc.get("id"), "missing": [], "unobservable": [],
                "prohibited": pname,
                "notes": [(pproc.get("feedback") or {}).get("prohibited", "").replace("{tool}", pname)
                          .replace("{quote}", pspec.get("_quote", "")).strip()],
                "verdict": "deny"}
    proc = find_procedure(procs, tool_name, executed)
    if proc is None:
        return {"procedure": None, "missing": [], "notes": [], "verdict": "pass"}
    missing, unobservable = unmet_nodes(proc, tool_name, executed)
    notes = notes_for_call(procs, tool_name, args, executed, operands)
    verdict = "deny" if (missing and is_mandatory(proc)) else "pass"
    return {"procedure": proc.get("id"), "missing": missing, "unobservable": unobservable,
            "notes": notes, "verdict": verdict}
