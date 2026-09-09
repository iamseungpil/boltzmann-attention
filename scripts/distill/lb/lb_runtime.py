# -*- coding: utf-8 -*-
"""Runtime: wires the seven engines into the tau2 agent loop. install(domain) patches three hooks.

Generation hook  each turn: LB6 reduces the view, the model generates, the engines evaluate, the
                 coordinator speaks. A deny becomes an error tool result for that call; advice becomes
                 a user message; both go into a non-committed buffer and the model regenerates (at
                 most ROUNDS times). The committed history never sees our text.
Execution hook   LB2 verifier tools (A2["LB2"]["tools"]) are injected into the agent's tool list and
                 executed here deterministically; after a declared read ran, LB2 derived facts are
                 appended to that read's output.
Sub-call door    `ask(prompt, name)` - one generation with no tools over a minimal context - is the
                 only way an engine talks to the model. fetch_formalize lets that sub-call read
                 records through declared getter tools (executed by the environment, off-ledger).

WHERE WE DIFFER FROM BASE - the complete list. Base is `--gate 0`: stock tau2, none of this. Every
row below is a place our stack does something tau2 would not, and each one is (a) reachable only
while some lever is on, and (b) recorded with `diverge(kind)` so a run's sidecar enumerates exactly
what touched it. With T2_LB1..7 all zero, `any_lever()` is false, both hooks delegate to tau2 and
this list is empty - that is what makes the levers-off cell a control. Grep must agree with this
table: tests/test_lb.py fails if a `diverge(` kind is missing here or listed here and never raised.

  kind             where                     gate        what changes for the model
  inject-tools     install/init              LB2         verifier tools appear in the tool list
  our-tool         execute                   LB2         a call is answered by us, not the environment
  facts            append_facts              LB2         a read's output gains "[FACTS] ..."
  merge-order      execute                   any lever   results are reordered to the call order
  empty-batch      execute                   any lever   orig_exec is skipped when we took every call
  fold             turn_hook                 LB6         the view is compacted
  regen            turn_hook                 any lever   the model's message is replaced
  ctx-stop         generate                  any lever   a context-window error ends the run gracefully

Two more differences exist and are recorded, but not through `diverge`: the text a lever speaks is
already one sidecar row per utterance (lb-deny, lb-advice, lb-inject, lb-conflict, lb-release, from
lb_coordinator.say), and `call_name="lb_turn"` only names the debug log file litellm writes - it is
not part of the request, so the model cannot see it.
"""

import collections
import io
import os
import sys

import lb_a2
import lb2_decision
import lb6_load
from lb_coordinator import Turn, evaluate, say, fam, as_dict, sidecar, sim_id, enabled

ROUNDS = 3            # regenerations per turn
REGEN_BUDGET = 12     # regenerations per simulation; after that the model's message stands as generated
GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."
ADVICE_MARK = "[SERVICE LAYER NOTE - not written by the customer; do not reply to it, act on it] "

_ORIG_TURN = None       # tau2's own generation, kept so the all-levers-off cell can run it untouched


def diverge(kind, text="", **meta):
    """One row per place our stack left tau2's path. base raises none of these; a run with levers on
    should be readable as the list of them. Keep every kind in the module docstring's table."""
    sidecar("lb-diverge", text, None, at=kind, **meta)


def any_lever():
    """Is any engine on? With none on, our stack must not reach the model at all - the levers-off
    cell is the control every A/B is read against, and a control that runs our plumbing is not one.
    (2026-09-09: append_facts was ungated and rewrote tool output in 016 and 098 with T2_LB2=0.)
    """
    return any(enabled("LB%d" % i) for i in range(1, 8))


def install(domain):
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    a2 = lb_a2.load(domain)
    if a2 is None:
        raise SystemExit("no a2/%s.lb.json - run: python lb_a2.py migrate %s" % (domain, domain))
    orig_init, orig_exec = BaseOrchestrator.__init__, BaseOrchestrator._execute_tool_calls

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        agent = getattr(self, "agent", None)
        if agent is not None:
            agent._lb_a2, agent._lb_orch = a2, self
            if enabled("LB2"):
                names = inject_tools(agent, a2)
                sidecar("lb-tools", "INJECTED %s" % ", ".join(names), None, sim=sim_id(agent), n=len(names))
                diverge("inject-tools", ", ".join(names), sim=sim_id(agent), n=len(names))

    def exec_hook(self, tool_calls):
        if not any_lever():
            return orig_exec(self, tool_calls)
        return execute(self, a2, tool_calls, orig_exec)

    global _ORIG_TURN
    _ORIG_TURN = LLMAgent._generate_next_message
    BaseOrchestrator.__init__ = init
    BaseOrchestrator._execute_tool_calls = exec_hook
    LLMAgent._generate_next_message = turn_hook
    print("[lb] installed for %s (%d verifier tools)" % (domain, len((a2.get("LB2") or {}).get("tools") or [])), flush=True)


# ---- generation hook ---------------------------------------------------------------------------------
def turn_hook(self, message, state):
    from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
    if not any_lever():
        out = _ORIG_TURN(self, message, state)          # tau2's own path, byte for byte
        am0 = out[0] if isinstance(out, tuple) else out
        trace(self, [am0], turn_len=len(state.messages))  # the sidecar only reads; it writes nothing back
        return out
    state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
    self._system_messages = state.system_messages
    a2 = getattr(self, "_lb_a2", {}) or {}
    trace(self, state.messages[-len(message.tool_messages) if isinstance(message, MultiToolMessage) else -1:])
    view = lb6_load.reduce(a2, state.messages) if enabled("LB6") else list(state.messages)
    fold_mark(self, state.messages, view)
    am = generate(self, view)
    for _ in range(ROUNDS):
        turn = build_turn(self, a2, state.messages, am)
        d = say(turn, evaluate(turn), owner=self)
        # advice reaches the model only through a regeneration: on a text turn, and on a hand-off
        # call (the transfer is deferred once with the open promise named; the model may re-issue it)
        if not d.denies and not (d.advice and (not turn.calls or handing_off(turn))):
            break
        if self.__dict__.get("_lb_regen", 0) >= REGEN_BUDGET:
            print("[lb] regen budget spent - message stands", file=sys.stderr, flush=True)
            sidecar("lb-regen-stop", "budget %d spent; the model message stands" % REGEN_BUDGET,
                    turn, sim=turn.sim, denies=len(d.denies), advice=len(d.advice))
            break
        self._lb_regen = self.__dict__.get("_lb_regen", 0) + 1
        # base never has its message replaced. This is the one event that says ours was, and with
        # what: the trajectory keeps only the replacement (memory 30).
        diverge("regen", "round %d" % self._lb_regen, rnd=self._lb_regen)
        sidecar("lb-regen", "round %d: %d deny, %d advice, force=%s, pin=%s"
                % (self._lb_regen, len(d.denies), len(d.advice), bool(d.force_call),
                   (d.pins[0][0] if d.pins and d.pins[0] else None)),
                turn, sim=turn.sim, rnd=self._lb_regen, denies=len(d.denies), advice=len(d.advice),
                force=bool(d.force_call), pinned=bool(d.pins))
        fb = [am] + [ToolMessage(id=c.id, role="tool", requestor="assistant", error=True,
                                 content=d.denies.get(id(c), GENERIC)) for c in turn.calls]
        # the advice rides in the customer's slot, and on task_070 the model answered it as if the
        # customer had written it; the marker says whose words these are
        fb += [UserMessage(role="user", content=ADVICE_MARK + text) for text in d.advice]
        am = generate(self, view + fb, force=d.force_call, pin=d.pins[0] if d.pins else None)
    trace(self, [am], turn_len=len(state.messages))
    return am


def handing_off(turn):
    transfer = {fam(x) for x in (turn.a2.get("LB5") or {}).get("transfer_tools") or []}
    return any(fam(turn.name_of(c)) in transfer for c in turn.calls)


def trace(agent, msgs, turn_len=None):
    """One sidecar row per message as it happens - the live trajectory. tau2 writes its file only at
    the end, so without this a slow simulation cannot be read while it runs."""
    for m in msgs:
        calls = [(getattr(c, "name", ""), str(getattr(c, "arguments", ""))[:160]) for c in (getattr(m, "tool_calls", None) or [])]
        head = str(getattr(m, "content", "") or "")[:600]
        sidecar("lb-msg", ("%s %s" % (calls, head)) if calls else head, None, sim=sim_id(agent),
                role=str(getattr(m, "role", "")), n=turn_len if turn_len is not None else -1,
                error=bool(getattr(m, "error", False)))


def fold_mark(agent, before, after):
    """The compactor kept no record of whether it ran. 028 hit the context window and the threshold
    had to be recomputed by hand to find out that folding had engaged - and still was not enough."""
    b = sum(len(str(getattr(m, "content", "") or "")) for m in before)
    a = sum(len(str(getattr(m, "content", "") or "")) for m in after)
    if a != b:
        diverge("fold", "%d -> %d chars" % (b, a))
        sidecar("lb-fold", "%d -> %d chars over %d messages" % (b, a, len(before)), None,
                sim=sim_id(agent), before=b, after=a, msgs=len(before))


def generate(self, messages, force=False, pin=None, tools=None, call_name="lb_turn"):
    import tau2.agent.llm_agent as la
    kw = dict(self.llm_args)
    tools = self.tools if tools is None else tools
    if pin:
        tools, choice = pinned(tools, *pin)
        kw["tool_choice"] = choice
    elif force:
        kw["tool_choice"] = "required"
    try:
        # `tools or None` used to sit here and differed from tau2 whenever the list was empty;
        # there is no lever behind that, so it is gone rather than gated.
        return la.generate(model=self.llm, tools=tools, messages=self._system_messages + messages,
                           call_name=call_name, **kw)
    except Exception as e:
        if "ContextWindow" not in type(e).__name__:
            raise
        orch = getattr(self, "_lb_orch", None)
        if orch is not None:
            from tau2.data_model.simulation import TerminationReason
            orch.done, orch.termination_reason = True, TerminationReason.CONTEXT_WINDOW_EXCEEDED
        diverge("ctx-stop", "context window exceeded; ending the run instead of raising",
                sim=sim_id(self))
        print("[lb] context window exceeded -> graceful stop", file=sys.stderr, flush=True)
        from tau2.data_model.message import AssistantMessage
        return AssistantMessage(role="assistant", content="(context limit reached - conversation ending)")


def pinned(tools, tool_name, arg, value):
    import copy
    out = []
    for t in tools or []:
        if getattr(t, "name", None) != tool_name:
            out.append(t)
            continue
        t2 = copy.deepcopy(t)
        try:
            props = t2.openai_schema["function"]["parameters"]["properties"]
            if arg and value is not None and arg in props:
                props[arg] = dict(props[arg], enum=[value])
        except Exception:
            pass
        out.append(t2)
    return out, {"type": "function", "function": {"name": tool_name}}


def ask_fn(agent):
    """The sub-call door: one tool-less generation over the prompt alone. Cached per (name, prompt)."""
    def ask(prompt, name="lb_ask"):
        from tau2.data_model.message import UserMessage
        cache = agent.__dict__.setdefault("_lb_ask_cache", {})
        key = (name, prompt)
        if key not in cache:
            try:
                r = generate(agent, [UserMessage(role="user", content=prompt)], tools=[], call_name=name)
                cache[key] = str(getattr(r, "content", "") or "")
            except Exception as e:
                print("[lb] ask failed: %r" % (e,), file=sys.stderr, flush=True)
                cache[key] = ""
        return cache[key]
    return ask


# ---- turn state ----------------------------------------------------------------------------------------
def build_turn(agent, a2, messages, am):
    env = getattr(getattr(agent, "_lb_orch", None), "environment", None)
    executed, attempted, unlocked, pending = collections.Counter(), collections.Counter(), set(), {}
    dispatch = a2.get("dispatch") or {}
    probe = Turn(a2, [], am)
    for m in messages:
        for c in (getattr(m, "tool_calls", None) or []):
            pending[getattr(c, "id", None)] = probe.name_of(c)
            if getattr(c, "name", None) == dispatch.get("unlock_tool"):
                unlocked.add(probe.named(c))
        if getattr(m, "role", None) == "tool":
            name = pending.pop(getattr(m, "id", None), None)
            text = str(getattr(m, "content", "") or "").lstrip()
            failed = getattr(m, "error", False) or any(text.startswith(k) for k in a2.get("failure_markers") or [])
            if name:
                attempted[name] += 1
                if not failed:
                    executed[name] += 1
    return Turn(a2, messages, am, executed=executed, attempted=attempted, unlocked=unlocked,
                visible_tools={getattr(t, "name", None) for t in (agent.tools or [])},
                registry=registry_of(env), corpus=corpus(),
                extras={"ask": ask_fn(agent), "rows": dict(agent.__dict__.get("_lb_rows") or {})},
                sim=sim_id(agent))


def registry_of(env):
    def names(tk, discoverable):
        try:
            return set(tk.get_discoverable_tools()) if discoverable else set(getattr(tk, "tools", {}) or {})
        except Exception:
            return set()
    return {"agent": names(getattr(env, "tools", None), True), "user": names(getattr(env, "user_tools", None), True),
            "user_all": names(getattr(env, "user_tools", None), False)}


_CORPUS = {}


def corpus():
    d = os.environ.get("LB_DOCS_DIR") or os.environ.get("T2_KB_DOCS_DIR")
    if not d or not os.path.isdir(d):
        return {}
    if d not in _CORPUS:
        docs = {}
        for f in sorted(os.listdir(d)):
            if f.endswith((".md", ".txt", ".json")):
                docs[os.path.splitext(f)[0]] = io.open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
        _CORPUS[d] = docs
    return _CORPUS[d]


# ---- execution hook: verifier tools and derived facts ---------------------------------------------
def inject_tools(agent, a2):
    from tau2.environment.tool import Tool
    have, added = {getattr(t, "name", None) for t in (agent.tools or [])}, []
    for d in (a2.get("LB2") or {}).get("tools") or []:
        if d["name"] in have:
            continue
        params, optional = d.get("params") or {}, set(d.get("optional") or [])
        sig = ", ".join(["%s: str" % p for p in params if p not in optional] + ['%s: str = ""' % p for p in params if p in optional])
        ns = {}
        exec(compile("def %s(%s):\n    pass\n" % (d["name"], sig), "<lb2_tool:%s>" % d["name"], "exec"), ns)
        fn = ns[d["name"]]
        fn.__doc__ = "\n".join([str(d.get("description") or d["name"]).strip(), ""]
                               + [":param %s: %s" % (p, " ".join(str(t).split())) for p, t in params.items()])
        agent.tools.append(Tool(fn, examples=list(d.get("examples") or [])))
        added.append(d["name"])
    return added


def execute(orch, a2, tool_calls, orig_exec):
    from tau2.data_model.message import ToolMessage
    decls = {d["name"]: d for d in (a2.get("LB2") or {}).get("tools") or []}
    ours, rest = {}, []
    for tc in tool_calls:
        d = decls.get(getattr(tc, "name", None)) if getattr(tc, "requestor", "assistant") == "assistant" else None
        if d is None:
            rest.append(tc)
        else:
            ours[id(tc)] = (tc, d)
    if not rest and tool_calls:
        diverge("empty-batch", "every call in this batch was ours; the environment was not asked",
                n=len(tool_calls))
    results = list(orig_exec(orch, rest)) if rest else []
    by_id = {getattr(r, "id", None): r for r in results}
    agent = getattr(orch, "agent", None)
    for tc, d in ours.values():
        args = {k: v for k, v in (as_dict(tc.arguments) or {}).items()}
        iso = d.get("isolate") or {}
        if iso.get("mode") == "fetch_formalize" and not all(args.get(k) for k in iso.get("operand_keys") or []):
            args.update(fetch_formalize(orch, agent, d, iso, args, orig_exec) or {})
        elif iso.get("over") and iso.get("operand_schema"):
            formalize_rows(orch, agent, iso, args, orig_exec)
        text, err, ids = lb2_decision.run_tool(d, args, corpora_of(orch, agent), evidence_of(orch, d))
        if ids and agent is not None:
            # what a verifier settled travels as data. It used to be recovered by parsing the
            # sentence we had just written, which found nothing and left LB4 silent.
            agent.__dict__.setdefault("_lb_rows", {}).setdefault(fam(d["name"]), set()).update(ids)
        by_id[tc.id] = ToolMessage(id=tc.id, role="tool", requestor="assistant", error=err, content=text)
        # result first: the sidecar keeps 4000 chars and a 47-row argument list alone exceeds that
        sidecar("lb-tool", "RESULT %s\nARGS %s" % (text[:2500], json_dumps(args)[:1400]), None, sim=sim_id(agent),
                source=d["name"], error=bool(err))   # the verifier's full input and output, for live forensics
        diverge("our-tool", d["name"], sim=sim_id(agent), error=bool(err))
        print("[lb2] tool %s -> %s" % (d["name"], "error" if err else "ok"), file=sys.stderr, flush=True)
    out = [by_id[getattr(tc, "id", None)] for tc in tool_calls if getattr(tc, "id", None) in by_id]
    if [getattr(r, "id", None) for r in out] != [getattr(r, "id", None) for r in results]:
        diverge("merge-order", "results reordered to the call order", ours=len(ours), rest=len(rest))
    if enabled("LB2"):
        # derived facts are LB2's, and they rewrite a tool result the model reads. Ungated,
        # T2_LB2=0 still put "[FACTS] ..." into 016 and 098 in every simulation of the
        # levers-off cell, so that cell was not the control it was recorded as.
        append_facts(orch, a2, agent, out)
    return out


def corpora_of(orch, agent):
    msgs = orch.get_messages() if hasattr(orch, "get_messages") else []
    tools = [str(getattr(m, "content", "") or "") for m in msgs if getattr(m, "role", None) == "tool"]
    users = [str(getattr(m, "content", "") or "") for m in msgs if getattr(m, "role", None) == "user"]
    return {"kb": list(corpus().values()) + tools, "ledger": tools + users, "ledger_tools": tools, "user": users}


def evidence_of(orch, d):
    msgs = orch.get_messages() if hasattr(orch, "get_messages") else []
    outs, pending = {}, {}
    for m in msgs:
        for c in (getattr(m, "tool_calls", None) or []):
            pending[getattr(c, "id", None)] = getattr(c, "name", None)
        if getattr(m, "role", None) == "tool" and not getattr(m, "error", False):
            n = pending.get(getattr(m, "id", None))
            if n:
                outs[n] = str(getattr(m, "content", "") or "")
    return {"__tool_outputs": outs, "__user_text": " ".join(str(getattr(m, "content", "") or "")
                                                             for m in msgs if getattr(m, "role", None) == "user")}


def fetch_formalize(orch, agent, d, iso, args, orig_exec):
    """A sub-agent with only the declared getter tools reads the records and returns the operands as JSON."""
    from tau2.data_model.message import UserMessage
    import tau2.agent.llm_agent as la
    ref = {k: args.get(k) for k in iso.get("ref_params") or [] if args.get(k) not in (None, "")}
    getters = [t for t in (agent.tools or []) if getattr(t, "name", None) in set(iso.get("getter_tools") or [])]
    if not ref or not getters:
        return None
    prompt = "%s\n\n=== REFERENCE ===\n%s\n\n%s" % (iso.get("instructions", ""),
                                                   "\n".join("%s: %s" % kv for kv in ref.items()), iso.get("answer_format", ""))
    msgs, kw = [UserMessage(role="user", content=prompt)], {k: v for k, v in agent.llm_args.items() if "tool" not in k}
    for rnd in range(int(iso.get("max_rounds", 4))):
        last = rnd == int(iso.get("max_rounds", 4)) - 1
        try:
            resp = la.generate(model=agent.llm, tools=None if last else getters, messages=msgs, call_name="lb2_fetch",
                               **(dict(kw, tool_choice="required") if rnd == 0 else kw))
        except Exception as e:
            print("[lb2] fetch failed: %r" % (e,), file=sys.stderr, flush=True)
            return None
        calls = list(getattr(resp, "tool_calls", None) or [])
        if not calls:
            found = [r for r in lb2_decision.records_in(str(getattr(resp, "content", "") or ""))
                     if set(r) & set(iso.get("operand_keys") or [])]
            return {k: v for r in found for k, v in r.items() if k in set(iso.get("operand_keys") or [])} or None
        msgs.append(resp)
        msgs.extend(orig_exec(orch, calls))
    return None


def formalize_rows(orch, agent, iso, args, orig_exec):
    """Row mode: a sub-agent with the declared getter tools fills each row's operands ({id: {...}}) in place.

    A cited quote must exist in the corpus and a rate must lie in the declared range; otherwise that
    row keeps no operand and the tool reports it as unverified. Nothing else of the old multi-stage
    prompt survives - the declaration's instructions and answer format are the whole prompt.
    """
    from tau2.data_model.message import UserMessage
    import tau2.agent.llm_agent as la
    rows = args.get(iso["over"])
    rows = lb2_decision._list(rows) if isinstance(rows, str) else rows
    if not isinstance(rows, list) or not rows:
        return
    args[iso["over"]] = rows
    idf, fields = iso.get("id_field"), iso.get("row_fields") or []
    items = [{f: r.get(f) for f in fields if r.get(f) is not None} for r in rows if isinstance(r, dict)]
    schema = {str(r.get(idf)): iso["operand_schema"] for r in rows if isinstance(r, dict)}
    prompt = "%s\n\n=== ITEMS ===\n%s\n\n%s" % (iso.get("instructions", ""), json_dumps(items),
                                                 lb2_decision.fill(iso.get("answer_format", ""), schema=json_dumps(schema)))
    getters = [t for t in (agent.tools or []) if getattr(t, "name", None) in set(iso.get("getter_tools") or [])]
    msgs, kw = [UserMessage(role="user", content=prompt)], {k: v for k, v in agent.llm_args.items() if "tool" not in k}
    if iso.get("temperature") is not None:
        kw["temperature"] = iso["temperature"]
    if iso.get("inject_docs"):
        got = inject_operands(agent, iso, rows, schema)
    else:
        got = search_operands(orch, agent, iso, msgs, kw, schema, getters, orig_exec)
    verdict = {}
    if got:
        hay = " ".join(corpora_of(orch, agent)["kb"]).lower()
        lo, hi = (iso.get("rate_range") or [None, None])[:2]
        for r in rows:
            ops = got.get(str(r.get(idf)))
            if not isinstance(ops, dict):
                verdict[str(r.get(idf))] = "absent"
                continue
            quote = str(ops.get(iso.get("quote_field") or "") or "").strip()
            rate = lb2_decision.num(ops.get(iso.get("rate_field") or ""))
            if quote and " ".join(quote.lower().split()) not in " ".join(hay.split()):
                verdict[str(r.get(idf))] = "quote not in corpus"
                continue                                   # unsupported: the row stays unverified
            if rate is not None and lo is not None and not (lo <= rate <= hi):
                verdict[str(r.get(idf))] = "rate out of range"
                continue
            r.update({k: v for k, v in ops.items() if k in iso["operand_schema"] and v not in ("", None)})
            verdict[str(r.get(idf))] = "kept rate=%s" % rate
    sidecar("lb-formalize", "VERDICT %s" % json_dumps(verdict), None, sim=sim_id(agent),
            source=str(iso.get("over")), mode="inject" if iso.get("inject_docs") else "search")


def titled(corpus_docs):
    """{title: body} - a document names the subject it covers in its title ('Silver Rewards Card:
    How to Earn 4% ...'). A document is either a JSON record with a title field or text whose first
    heading line is the title; this corpus is the former and reading it as the latter matched nothing."""
    out = {}
    for body in corpus_docs.values():
        rec = as_dict(body) if body.lstrip()[:1] == "{" else None
        title = str((rec or {}).get("title") or "").strip() or \
            next((l for l in body.splitlines() if l.startswith("#")), "").lstrip("#").strip()
        if title:
            out[title] = str((rec or {}).get("content") or body)
    return out


def inject_operands(agent, iso, rows, schema):
    """Deliver the subject's documents in full instead of making the sub-agent search for them.

    The search mode ended probe 017 with base_rate null on exactly the two rows that held the
    discrepancy: the sub-agent found the bonus-rate document and never the standard-rate one. The
    documents are ours to hand over, so they are handed over - grouped by the declared key, matched
    by title prefix so a neighbouring product's documents ("Business Silver Rewards Card") stay out.
    """
    docs = titled(corpus())
    if not docs:
        return None
    gkeys = iso["group_by"] if isinstance(iso["group_by"], list) else [iso["group_by"]]
    doc_key = iso.get("doc_key") or gkeys[0]
    keep, idf = set(iso.get("row_fields") or []), iso.get("id_field")
    groups = {}
    for r in rows:
        if isinstance(r, dict):
            groups.setdefault(tuple(str(r.get(k)) for k in gkeys), []).append(r)
    out, batch = {}, int(iso.get("max_batch") or 0)
    for grows in groups.values():
        gval = str(grows[0].get(doc_key))
        mine = sorted(t for t in docs if t.startswith(gval + ":"))
        if not mine:
            print("[lb2] inject: no document titled '%s: ...'" % gval, file=sys.stderr, flush=True)
            continue
        blob = "\n\n".join("### %s\n%s" % (t, docs[t]) for t in mine)
        chunks = [grows[i:i + batch] for i in range(0, len(grows), batch)] if batch > 0 else [grows]
        for chunk in chunks:
            items = [{k: v for k, v in r.items() if k in keep} for r in chunk]
            ids = {str(r.get(idf)): iso["operand_schema"] for r in chunk}
            prompt = lb2_decision.fill(iso.get("inject_instructions") or iso.get("instructions", ""),
                                       group=gval, docs=blob, items=json_dumps(items), schema=json_dumps(ids))
            raw = ask_fn(agent)(prompt, "lb2_inject")
            got = next((x for x in lb2_decision.records_in(raw) if set(x) & set(ids)), None) or {}
            sidecar("lb-inject", "GROUP %s docs=%d rows=%d\nREPLY %s" % (gval, len(mine), len(chunk), raw[:1500]),
                    None, sim=sim_id(agent), source=gval, got=len(got))
            out.update({k: v for k, v in got.items() if k in ids})
    return out or None


def search_operands(orch, agent, iso, msgs, kw, schema, getters, orig_exec):
    """The sub-agent searches for what it needs, then answers. Used when no documents are declared."""
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage
    got = None
    for rnd in range(int(iso.get("max_rounds", 4))):
        last = rnd == int(iso.get("max_rounds", 4)) - 1
        if last:
            # searching is over; ask for the answer in so many words. Without this the sub-agent ended
            # probe 017 with an empty message and every row went unjudged.
            msgs.append(UserMessage(role="user", content="Stop searching now and answer. Reply with EXACTLY one "
                                    "JSON object in the format given above and nothing else."))
        try:
            resp = la.generate(model=agent.llm, tools=None if (last or not getters) else getters, messages=msgs,
                               call_name="lb2_rows", **kw)
        except Exception as e:
            print("[lb2] row formalize failed: %r" % (e,), file=sys.stderr, flush=True)
            sidecar("lb-formalize-round", "FAILED %r" % (e,), None, sim=sim_id(agent), rnd=rnd)
            return
        calls = list(getattr(resp, "tool_calls", None) or [])
        sidecar("lb-formalize-round", "CALLS %s\nCONTENT %s" % ([getattr(c, "name", "") for c in calls],
                str(getattr(resp, "content", "") or "")[:1500]), None, sim=sim_id(agent), rnd=rnd, last=last)
        if not calls:
            got = next((r for r in lb2_decision.records_in(str(getattr(resp, "content", "") or "")) if set(r) & set(schema)), None)
            break
        msgs.append(resp)
        msgs.extend(orig_exec(orch, calls))
    return got


def json_dumps(o):
    import json
    return json.dumps(o, ensure_ascii=False)


def append_facts(orch, a2, agent, results):
    """After a declared read ran, LB2 derived facts are appended to its output (reads only)."""
    nodes = (a2.get("LB2") or {}).get("derived") or []
    triggers = {fam(i[5:]) for n in nodes for i in (n.get("inputs") or []) if i.startswith("tool:")}
    if not triggers or agent is None:
        return
    ev = evidence_of(orch, None)
    outs = dict(ev["__tool_outputs"])
    id_to_name = {}
    for m in orch.get_messages() if hasattr(orch, "get_messages") else []:
        for c in (getattr(m, "tool_calls", None) or []):
            id_to_name[getattr(c, "id", None)] = getattr(c, "name", None)
    for r in results:
        name = id_to_name.get(getattr(r, "id", None))
        if not name or fam(name) not in triggers or getattr(r, "error", False):
            continue
        outs[name] = str(getattr(r, "content", "") or "")
        facts = lb2_decision.derived_facts(a2, outs, ask_fn(agent), a3_rows=(a2.get("LB2") or {}).get("a3_rows") or ())
        if facts:
            r.content = outs[name] + "\n\n[FACTS] " + " ".join(t for _o, _v, t in facts)
            diverge("facts", name, n=len(facts))
            print("[lb2] facts appended to %s: %d" % (name, len(facts)), file=sys.stderr, flush=True)
