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

import collections
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
        if _excluded(p, done):
            continue
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


def _excluded(proc, done):
    """Is this procedure ruled out because another one's own tool has run?

    Two transfer documents share a tool. The incident protocol runs
    1822 -> 0218 -> standard; the purchase-decline protocol uses 0218 three times before
    the standard tool. Entering on 0218 alone would activate both and they disagree about
    the count. Which situation the conversation is in is an open question ([[22]]) that
    this engine must not answer, so the declaration answers a closed one instead: the
    decline protocol is not in play once the incident protocol's own tool has run.
    """
    block = set((proc.get("enter_when") or {}).get("tool_none") or [])
    return bool(block & set(done or ()))


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

    `min_count` is for the policies that count rather than order: the purchase-decline
    document says to use the internal transfer tool "for the first, second, and third
    transfer requests" and the regular one on the fourth. How many times a tool has run
    is as much a fact of the call history as whether it ran, so this stays a closed
    predicate ([[22]]) — but it needs a count, and a set has thrown that away. Callers
    that pass a set still work: without `min_count` nothing reads the multiplicity.
    """
    tools = _tools_of(node)
    if not tools:
        return None
    done = executed if hasattr(executed, "get") else collections.Counter(executed or ())
    need = node.get("min_count")
    if need:
        return sum(done.get(t, 0) for t in tools) >= int(need)
    return any(done.get(t, 0) for t in tools)


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


def active_procedures(procs, executed):
    """Procedures whose entry has already happened — the only entry point that takes no call.

    `find_procedure` answers "which procedure is this call inside", which cannot see the
    failure that dominates the closure cluster: the agent enters and then calls nothing
    at all, so there is no call to ask about. This asks the other question — which
    procedures are running right now — from the execution history alone.
    """
    done = set(executed or ())
    return [p for p in procs or []
            if not _excluded(p, done)
            and set((p.get("enter_when") or {}).get("tool_any") or []) & done]


def checklist(proc, executed):
    """[(node_id, tools, done)] in declaration order — what the policy asked for, and where we are.

    `done` is True, False, or None for a node that names no tool: a bound the agent is
    told to check rather than call cannot be observed from the history, and saying it is
    done would be a claim this engine cannot make.
    """
    return [(n.get("id"), _tools_of(n), _satisfied(n, executed))
            for n in (proc.get("nodes") or [])]


def _blocked_by(proc, node, executed):
    """Prerequisites of this node that have not run, walked transitively.

    The walk has to be transitive for the same reason `unmet_nodes` is: a step can be
    executed out of order, so a node whose direct parent is done may still sit behind an
    unmet grandparent. `task_048` is exactly that — the eligibility check ran first, and
    reading only direct requires then reports the step after it as ready while the two
    checks the policy puts before it never happened. Both functions must read the graph
    the same way or the checklist and the deny disagree about the same state.
    """
    by_id = {n.get("id"): n for n in (proc.get("nodes") or [])}
    seen, out, stack = set(), [], list(node.get("requires") or [])
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in by_id:
            continue
        seen.add(nid)
        if _satisfied(by_id[nid], executed) is False:
            out.append(nid)
        stack.extend(by_id[nid].get("requires") or [])
    return out


def next_step(proc, executed):
    """(candidates, unique) — unmet nodes whose prerequisites are all out of the way.

    The engine does not choose among equals. When several nodes are ready at once — the
    four checks a limit increase requires are all ready the moment the request is
    submitted — every one of them is returned and the caller must show them as a list.
    Naming one would be the engine deciding what the model should do next, which is the
    generator's job ([[10]]), not a deterministic check's.

    A node that cannot be observed never blocks: it is not something the history can
    settle, so treating it as outstanding would freeze the walk at the first such node.
    """
    out = []
    for n in (proc.get("nodes") or []):
        if _satisfied(n, executed) is not False:
            continue
        if _blocked_by(proc, n, executed):
            continue
        out.append(n)
    return out, len(out) == 1


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


def render_state(proc, executed, unlocked=(), pattern=None):
    """Slot values for the declaration's sentence: where the walk is, and what comes next.

    The engine writes no prose. It fills in only what it can observe — which steps ran,
    which tool a step calls, whether that name was ever unlocked — and the declaration
    supplies the sentence around it. That division matters here because the failure this
    is for is not the model refusing a known step: in `task_048` the deny named the node
    id `prior_attempts` ten times while the model called the right tool eight times and
    failed on the one fact nobody told it, that the tool needed unlocking first.

    `next` is filled only when exactly one node is ready. When several are (the four
    checks a limit increase opens at once), they are listed and the slot stays empty —
    picking one is the generator's job ([[10]]), not this file's.
    """
    unlocked = set(unlocked or ())
    rows, done = [], 0
    for nid, tools, ok in checklist(proc, executed):
        if ok is True:
            done += 1
            rows.append("[x] %s" % nid)
        elif ok is None:
            rows.append("[?] %s" % nid)          # 도구를 이름하지 않는 단계 = 이력으로 판정 불가
        else:
            rows.append("[ ] %s%s" % (nid, (" -> " + "/".join(tools)) if tools else ""))
    cands, uniq = next_step(proc, executed)
    ctools = [t for n in cands for t in (_tools_of(n) or [])]
    # ▶ is only ever attached where the policy itself mandates an order. A procedure the
    # declaration merely describes gets its remaining steps listed; naming "the next one"
    # there would be the engine inventing a sequence the policy did not state.
    nxt = cands[0] if (uniq and is_mandatory(proc)) else None
    ntool = (_tools_of(nxt) or [None])[0] if nxt is not None else None
    locked = bool(ntool) and ntool not in unlocked
    return {
        "procedure": proc.get("id") or "",
        "done": done,
        "total": len(proc.get("nodes") or []),
        "checklist": "  ".join(rows),
        "next": ("%s -> %s" % (nxt.get("id"), ntool)) if nxt is not None and ntool else "",
        "next_tool": ntool or "",
        "ready": ", ".join(n.get("id") for n in cands),
        "ready_tools": ", ".join(ctools),
        "locked": locked,
        "name_words": _words(ntool, pattern) if locked else "",
    }


def _words(name, pattern=None):
    """A plain-language query for a tool name — the suffix off, the underscores out.

    `task_048` searched the knowledge base for the tool name itself six times and scored
    nothing every time; the words are what the documents actually contain. The pattern
    comes from the declaration, so the engine holds no spelling rule of its own.
    """
    return re.sub(pattern or r"_\d+$", "", str(name or "")).replace("_", " ").strip()


def _fill(tpl, **kw):
    out = str(tpl or "")
    for k, v in kw.items():
        out = out.replace("{%s}" % k, str(v))
    return re.sub(r"\{[a-z_]+\}", "", out).strip()


def _hint(fb, st):
    """The unlock clause, or nothing — with its own trailing separator.

    `_fill` strips, so a template that ends in a space loses it and the next sentence runs
    into this one. The separator belongs to whoever knows the clause is non-empty.
    """
    if not st.get("locked"):
        return ""
    txt = _fill(fb.get("unlock_hint") or "", **st)
    return (txt + " ") if txt else ""


def absent_note(proc, executed, unlocked=(), pattern=None):
    """What the declaration says when the walk stopped — or nothing, if it says nothing.

    The engine speaks only through the declaration's own sentence; every slot it fills is
    an observation (which steps ran, which tool a step calls, whether that name was ever
    unlocked), never a judgement about what the customer wants.
    """
    fb = proc.get("feedback") or {}
    st = render_state(proc, executed, unlocked, pattern)
    if not (st["next"] or st["ready"]):
        return None
    tpl = fb.get("absent") if st["next"] else fb.get("absent_many")
    if not tpl:
        return None
    return _fill(tpl, unlock_hint=_hint(fb, st), **st)


def notes_for_call(procs, tool_name, args, executed, operands=None, unlocked=(),
                   pattern=None):
    """Every line the declaration says about this call. Empty when it says nothing."""
    proc = find_procedure(procs, tool_name, executed)
    if proc is None:
        return []
    fb = proc.get("feedback") or {}
    out = []
    missing, _unobs = unmet_nodes(proc, tool_name, executed)
    if missing and fb.get("unmet"):
        # ★2026-08-05: 구판은 `{missing}`에 **노드 id**만 채웠다. 048은 `missing=prior_attempts`를
        #   10회 받고 그 이름의 도구를 8회 불렀지만 매번 "unlock 안 됨" 에러였다 — 없던 정보는
        #   단계 이름이 아니라 **호출 가능한 이름과 그 잠금 상태**였다. 같은 체크리스트를 여기서도 준다.
        st = render_state(proc, executed, unlocked, pattern)
        out.append(_fill(fb["unmet"], tool=tool_name, missing=", ".join(missing),
                         source=", ".join(proc.get("_source") or [])[:120],
                         unlock_hint=_hint(fb, st), **st))
    te = table_expectation(proc, tool_name, operands)
    if te and fb.get("arg_from_table"):
        arg, expected = te
        given = (args or {}).get(arg)
        if given is None or str(given) != str(expected):
            out.append(_fill(fb["arg_from_table"], tool=tool_name, arg=arg, value=expected))
    elif fb.get("arg_table_unknown_key"):
        # ★키를 모를 때: 값을 고르지 않고 **표 자체**를 보여 준다. task_053은 정책이 3개월을 정한
        #   자리에 12를 넣었는데, 등급 상수를 모델이 선언한 적이 없어 위 분기는 영원히 침묵한다.
        #   추측하지 않는다는 원칙은 지키되(엔진이 등급을 정하지 않는다) 판단 재료는 준다([[52]]).
        node = _node_of(proc, tool_name)
        spec = (node or {}).get("arg_from_table") or {}
        table = (proc.get("tables") or {}).get(spec.get("table")) or {}
        if spec and table and (operands or {}).get(spec.get("operand")) is None:
            out.append(_fill(fb["arg_table_unknown_key"], tool=tool_name, arg=spec.get("arg"),
                             operand=spec.get("operand"),
                             table=", ".join("%s=%s" % (k, v) for k, v in sorted(table.items()))))
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
        if not block or _excluded(p, executed):
            continue
        triggers = set((p.get("enter_when") or {}).get("tool_any") or [])
        if not (triggers & set(executed or ())) and not (triggers & set(names or ())):
            continue
        for nm in names or ():
            spec = block.get(nm)
            if spec and spec.get("_quote"):
                return p, nm, spec
    return None, None, None


def decide(procs, tool_name, args, executed, operands=None, also_names=(), unlocked=(),
           pattern=None):
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
    notes = notes_for_call(procs, tool_name, args, executed, operands, unlocked, pattern)
    verdict = "deny" if (missing and is_mandatory(proc)) else "pass"
    return {"procedure": proc.get("id"), "missing": missing, "unobservable": unobservable,
            "notes": notes, "verdict": verdict}
