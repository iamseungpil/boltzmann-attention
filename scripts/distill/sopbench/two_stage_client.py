"""
two_stage_client.py — arm-3 (L1) TwoStageClient for SOPBench (Zekun Li, 2503.08669).

Drop-in replacement for swarm.llm_handler.OpenAIHandler as Agent.client. Implements the
same `inference(create_params, debug, mode, tool_call_mode) -> {"idx","completion"}`
interface that swarm/core.py:get_chat_completion calls, so NO change to the Swarm loop.

One Swarm turn = planner(LLM) picks the next abstract action -> resolver maps it to a
concrete tool call. This is the L1 (LLM + operators in-context, NO training) rung of the
WORKFLOW_ONTOLOGY_DESIGN §5 planner ladder; the resolver here is rung (b) "ontollm".

ARCHITECTURE (design = WORKFLOW_ONTOLOGY_DESIGN.md §9 / §10):
  PLANNER  : sees goal context + ABSTRACT operator affordances (name + short description
             ONLY, NO concrete param schema) + recent history -> outputs ONE next action
             name. GoalAct-style: re-decides every turn from history. The planner must NOT
             see concrete tool param schemas (transfer-contamination guard, §9.1).
  RESOLVER : sees the chosen action + its FULL concrete tool spec + accumulated slot state.
             rung (b): constrain the model to the single chosen tool (tool_choice) and let
             it fill args in-context. Deterministic shortcut: if all required args already
             live in slot state, emit the tool call WITHOUT an LLM call (counts as
             "deterministic coverage").
  COVERAGE : fraction of tool-call turns resolved deterministically (no resolver LLM call)
             vs. needing the LLM. Diagnostic only (= "how rarely we must invoke an LLM").

DEPLOY: copy to the SOPBench clone `scripts/` (alongside run_two_stage.py). Reused across
tasks; call .reset() per task to clear slot state.
"""
from __future__ import annotations
import json
from openai import OpenAI


class TwoStageClient:
    """OpenAIHandler-compatible client running a 2-stage planner+resolver per turn."""

    def __init__(self, base_url: str, model_name: str,
                 temperature: float = 0.0, max_tokens: int = 512, top_p: float = 0.01):
        self.model_name = model_name
        self.model_name_huggingface = model_name      # swarm/core reads this
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._client = OpenAI(base_url=base_url, api_key="EMPTY")
        # per-interaction state
        self._slot_state: dict = {}    # arg values mined from tool results + user_known
        self._turn: int = 0
        # coverage counters (cumulative across the run)
        self.cov_turns = 0
        self.cov_deterministic = 0

    def reset(self):
        """Call once per task before the interaction."""
        self._slot_state = {}
        self._turn = 0

    # ------------------------------------------------------------------
    # Swarm entry point (called every assistant turn)
    # ------------------------------------------------------------------
    def inference(self, create_params: dict, debug: bool = False,
                  mode: str = "chat", tool_call_mode: str = "fc") -> dict:
        messages = create_params.get("messages", [])
        tools = create_params.get("tools", [])
        temperature = create_params.get("temperature", self.temperature)
        max_tokens = create_params.get("max_tokens", self.max_tokens)
        top_p = create_params.get("top_p", self.top_p)

        if not tools:                          # no tools -> plain text reply (final msg)
            resp = self._client.chat.completions.create(
                model=self.model_name, messages=messages,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            return {"idx": self._turn, "completion": resp}

        self._turn += 1
        self._update_slots(messages)
        chosen_action = self._plan(messages, tools)
        completion = self._resolve(chosen_action, messages, tools,
                                   temperature, max_tokens, top_p)
        return {"idx": self._turn, "completion": completion}

    # ------------------------------------------------------------------
    # STEP 1 — Planner: goal + abstract affordances -> action name
    # ------------------------------------------------------------------
    def _plan(self, messages, tools) -> str:
        op_lines = [f"- {t.get('function',{}).get('name','?')}: "
                    f"{t.get('function',{}).get('description','')[:120]}" for t in tools]
        ops_str = "\n".join(op_lines)

        goal_ctx = next((m.get("content", "")[:400] for m in messages
                         if m.get("role") == "system"), "")

        hist = []
        for m in messages[-6:]:
            role = m.get("role", "")
            if role == "tool":
                hist.append(f"TOOL_RESULT: {str(m.get('content',''))[:80]}")
            elif role == "assistant" and m.get("tool_calls"):
                for tc in (m.get("tool_calls") or []):
                    fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                    name = fn.name if hasattr(fn, "name") else fn.get("name", "?")
                    hist.append(f"CALLED: {name}")
            elif role == "user":
                hist.append(f"USER: {str(m.get('content',''))[:80]}")
        hist_str = "\n".join(hist) if hist else "none"

        planner_prompt = (
            "You are a planning agent. Given the goal context and available operators, "
            "output ONLY the name of the single best next action to call.\n\n"
            f"GOAL CONTEXT:\n{goal_ctx}\n\n"
            f"AVAILABLE OPERATORS (name: description):\n{ops_str}\n\n"
            f"RECENT HISTORY:\n{hist_str}\n\n"
            "Output ONLY the action name, nothing else:")
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": planner_prompt}],
            temperature=0.0, top_p=0.01, max_tokens=32)
        chosen = resp.choices[0].message.content.strip().split()[0].strip(".,:") \
            if resp.choices[0].message.content else ""
        tool_names = {t.get("function", {}).get("name", "") for t in tools}
        if chosen not in tool_names:
            chosen = next(iter(tool_names)) if tool_names else ""
        return chosen

    # ------------------------------------------------------------------
    # STEP 2 — Resolver: action + concrete spec + slot state -> tool call
    # ------------------------------------------------------------------
    def _resolve(self, action_name, messages, tools, temperature, max_tokens, top_p):
        chosen_spec = next((t for t in tools
                            if t.get("function", {}).get("name", "") == action_name),
                           tools[0] if tools else {})
        fn = chosen_spec.get("function", {})
        required = fn.get("parameters", {}).get("required", [])

        self.cov_turns += 1
        if required and all(r in self._slot_state for r in required):
            # deterministic rung: build the call from slot state, NO LLM
            self.cov_deterministic += 1
            args = {r: self._slot_state[r] for r in required}
            return self._make_tool_call_completion(action_name, args)

        # LLM resolver rung (b): force the chosen tool, let the model fill args in-context
        resp = self._client.chat.completions.create(
            model=self.model_name, messages=messages, tools=[chosen_spec],
            tool_choice={"type": "function", "function": {"name": action_name}},
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            parallel_tool_calls=False)
        return resp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_slots(self, messages):
        """Mine arg values from tool results and the user_known dump into slot state."""
        for m in messages:
            if m.get("role") == "tool":
                try:
                    r = json.loads(m.get("content", "{}"))
                    if isinstance(r, dict):
                        self._slot_state.update(r)
                except Exception:
                    pass
            if m.get("role") == "user" and \
               str(m.get("content", "")).startswith("Here is all the information"):
                try:
                    txt = m["content"]
                    self._slot_state.update(
                        json.loads(txt[txt.index("{"):txt.rindex("}") + 1]))
                except Exception:
                    pass

    def _make_tool_call_completion(self, action_name, args):
        """Synthetic ChatCompletion carrying one tool call (deterministic path)."""
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall, Function)
        tc = ChatCompletionMessageToolCall(
            id=f"call_{action_name}_{self._turn}", type="function",
            function=Function(name=action_name, arguments=json.dumps(args)))
        msg = ChatCompletionMessage(role="assistant", content=None, tool_calls=[tc])
        choice = Choice(index=0, message=msg, finish_reason="tool_calls", logprobs=None)
        return ChatCompletion(id=f"det_{self._turn}", object="chat.completion",
                              created=0, model=self.model_name,
                              choices=[choice], usage=None)

    def coverage(self) -> dict:
        return {"turns": self.cov_turns, "deterministic": self.cov_deterministic,
                "llm_resolved": self.cov_turns - self.cov_deterministic,
                "coverage_pct": (self.cov_deterministic / self.cov_turns
                                 if self.cov_turns else 0.0)}

    def kill_process(self):   # OpenAIHandler compat stub (no subprocess to kill)
        pass
