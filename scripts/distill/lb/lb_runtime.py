# -*- coding: utf-8 -*-
"""Runtime: wires the seven engines into the tau2 agent loop. install(domain) patches two hooks.

Each turn: LB6 reduces the generation view, the model generates, the engines evaluate, the
coordinator speaks. A deny becomes an error tool result for that call; advice becomes a user message;
both go into a non-committed buffer and the model regenerates (at most ROUNDS times). The committed
history never sees our text (replay stays clean). Harness only: context-window overflow ends the
simulation gracefully instead of crashing it.
"""

import collections
import io
import os
import sys

import lb_a2
import lb6_load
from lb_coordinator import Turn, evaluate, say, fam, FAILSAFE_DENY

ROUNDS = 3
GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."


def install(domain):
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    a2 = lb_a2.load(domain)
    if a2 is None:
        raise SystemExit("no a2/%s.lb.json - run: python lb_a2.py migrate %s" % (domain, domain))
    orig_init = BaseOrchestrator.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        agent = getattr(self, "agent", None)
        if agent is not None:
            agent._lb_a2, agent._lb_orch = a2, self

    BaseOrchestrator.__init__ = init
    LLMAgent._generate_next_message = turn_hook
    print("[lb] installed for %s" % domain, flush=True)


def turn_hook(self, message, state):
    from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
    state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
    self._system_messages = state.system_messages
    a2 = getattr(self, "_lb_a2", {}) or {}
    view = lb6_load.reduce(a2, state.messages)
    am = generate(self, view)
    for _ in range(ROUNDS):
        turn = build_turn(self, a2, state.messages, am)
        d = say(turn, evaluate(turn), owner=self)
        if not d.denies and not (d.advice and not turn.calls):
            break
        fb = [am] + [ToolMessage(id=c.id, role="tool", requestor="assistant", error=True,
                                 content=d.denies.get(id(c), GENERIC)) for c in turn.calls]
        fb += [UserMessage(role="user", content=text) for text in d.advice]
        am = generate(self, view + fb, force=d.force_call, pin=d.pins[0] if d.pins else None)
    return am


def generate(self, messages, force=False, pin=None):
    import tau2.agent.llm_agent as la
    kw, tools = dict(self.llm_args), self.tools
    if pin:
        tools, choice = pinned(tools, *pin)
        if choice:
            kw["tool_choice"] = choice
    elif force:
        kw["tool_choice"] = "required"
    try:
        return la.generate(model=self.llm, tools=tools, messages=self._system_messages + messages,
                           call_name="lb_turn", **kw)
    except Exception as e:
        if "ContextWindow" not in type(e).__name__:
            raise
        orch = getattr(self, "_lb_orch", None)
        if orch is not None:
            from tau2.data_model.simulation import TerminationReason
            orch.done, orch.termination_reason = True, TerminationReason.CONTEXT_WINDOW_EXCEEDED
        print("[lb] context window exceeded -> graceful stop", file=sys.stderr, flush=True)
        from tau2.data_model.message import AssistantMessage
        return AssistantMessage(role="assistant", content="(context limit reached - conversation ending)")


def pinned(tools, tool_name, arg, value):
    """Narrow the pinned tool's argument to one value and force that tool (LB1 PIN)."""
    import copy
    out = []
    for t in tools or []:
        if getattr(t, "name", None) != tool_name:
            out.append(t)
            continue
        t2 = copy.deepcopy(t)
        try:
            schema = t2.openai_schema["function"]["parameters"]["properties"]
            if arg and value is not None and arg in schema:
                schema[arg] = dict(schema[arg], enum=[value])
        except Exception:
            pass
        out.append(t2)
    return out, {"type": "function", "function": {"name": tool_name}}


def build_turn(agent, a2, messages, am):
    env = getattr(getattr(agent, "_lb_orch", None), "environment", None)
    executed, unlocked, pending = collections.Counter(), set(), {}
    dispatch, name_args = a2.get("dispatch") or {}, (a2.get("dispatch") or {}).get("name_args") or {}
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
            if name and not failed:
                executed[name] += 1
    return Turn(a2, messages, am, executed=executed, unlocked=unlocked,
                visible_tools={getattr(t, "name", None) for t in (agent.tools or [])},
                registry=registry_of(env), corpus=corpus())


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
            if f.endswith((".md", ".txt")):
                docs[os.path.splitext(f)[0]] = io.open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
        _CORPUS[d] = docs
    return _CORPUS[d]
