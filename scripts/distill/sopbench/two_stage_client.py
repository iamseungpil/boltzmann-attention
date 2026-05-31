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
import ast
import json
from openai import OpenAI


def _try_parse(s):
    """Parse a tool-result string into a python object (try_eval-style, matches
    run_evaluation.py:try_eval). Returns the object or None if unparseable."""
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


class TwoStageClient:
    """OpenAIHandler-compatible client running a 2-stage planner+resolver per turn.

    use_deterministic_shortcut: rung-(a) opportunism. When True, if every required
    arg of the chosen tool is already in slot state, the call is emitted WITHOUT an
    LLM resolver call. Default False so arm-3 measures CLEAN L1 (planner + LLM
    resolver) without slot-state guesses contaminating pass@1; the would-be coverage
    is still counted as a diagnostic either way (see coverage()).
    """

    def __init__(self, base_url: str, model_name: str,
                 temperature: float = 0.0, max_tokens: int = 512, top_p: float = 0.01,
                 use_deterministic_shortcut: bool = False,
                 planner: str = "naive", abox=None):
        self.model_name = model_name
        self.model_name_huggingface = model_name      # swarm/core reads this
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.use_deterministic_shortcut = use_deterministic_shortcut
        # arm-3v2: structured planner reads the induced ABox (precondition/produces),
        # current slot state, allows STOP=refuse (N1), and is prompted for the cheapest
        # gated path (§11.12). planner="naive" keeps the arm-3 baseline untouched.
        self.planner = planner
        self.abox = abox
        if isinstance(abox, str):
            self.abox = json.load(open(abox)) if abox else None
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
        if self.planner == "v2" and self.abox:
            chosen_action = self._plan_v2(messages, tools)
            if chosen_action in ("STOP", "exit_conversation", ""):
                # refuse / terminate = a no-forbidden-call turn (N1/§7): end via exit_conversation
                self.cov_turns += 1
                return {"idx": self._turn,
                        "completion": self._make_tool_call_completion("exit_conversation", {})}
        else:
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
    # STEP 1' — arm-3v2 structured planner (ABox precondition/produces + state + STOP)
    # ------------------------------------------------------------------
    def _render_precond(self, tree, out, est):
        """Flatten an ABox precondition tree to readable predicate names + establishable hints."""
        if not tree:
            return
        if isinstance(tree, (list, tuple)) and tree:
            head = tree[0]
            if head == "single":
                name = tree[1]
                name = name[4:] if name.startswith("not ") else name
                info = (self.abox.get("predicates", {}) or {}).get(name, {})
                if info.get("kind") == "establishable" and info.get("by"):
                    out.append(name)
                    est[name] = info["by"]
                else:
                    out.append(name)
            elif head in ("and", "or", "chain", "gate"):
                for sub in tree[1]:
                    self._render_precond(sub, out, est)

    def _plan_v2(self, messages, tools) -> str:
        ops = self.abox.get("operators", {})
        tool_names = [t.get("function", {}).get("name", "") for t in tools]
        # operator affordances: name — needs [precondition preds] — gives [produces]
        est_map = {}
        lines = []
        for t in tools:
            nm = t.get("function", {}).get("name", "")
            if nm == "exit_conversation":
                continue
            op = ops.get(nm)
            if op:
                preds, est = [], {}
                self._render_precond(op.get("precondition"), preds, est)
                est_map.update(est)
                needs = ", ".join(dict.fromkeys(preds)) or "nothing"
                gives = ", ".join(op.get("produces", [])) or "the goal/result"
                lines.append(f"- {nm}: needs [{needs}]; gives [{gives}]")
            else:
                desc = t.get("function", {}).get("description", "")[:80]
                lines.append(f"- {nm}: {desc}")
        ops_str = "\n".join(lines)
        est_str = ("\n".join(f"  - to establish '{p}', call {a}" for p, a in est_map.items())
                   or "  (none)")

        user_req = next((str(m.get("content", ""))[:300] for m in messages
                         if m.get("role") == "user" and m.get("content")), "")
        policy = next((str(m.get("content", ""))[:600] for m in messages
                       if m.get("role") == "system"), "")
        hist = []
        for m in messages[-8:]:
            role = m.get("role", "")
            if role == "tool":
                hist.append(f"RESULT[{m.get('tool_name','?')}]: {str(m.get('content',''))[:80]}")
            elif role == "assistant" and m.get("tool_calls"):
                for tc in (m.get("tool_calls") or []):
                    fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                    hist.append(f"CALLED: {fn.name if hasattr(fn,'name') else fn.get('name','?')}")
        hist_str = "\n".join(hist) if hist else "nothing yet"
        slots_str = ", ".join(sorted(self._slot_state.keys())) or "only what the user provided"

        prompt = (
            "You are a planning agent. Pick the SINGLE next tool to call, or STOP.\n\n"
            f"USER REQUEST:\n{user_req}\n\nPOLICY (constraints to honor):\n{policy}\n\n"
            f"TOOLS (name: needs [preconditions]; gives [effects]):\n{ops_str}\n\n"
            f"HOW TO ESTABLISH preconditions:\n{est_str}\n\n"
            f"ALREADY KNOWN/ESTABLISHED: {slots_str}\n"
            f"HISTORY:\n{hist_str}\n\n"
            "RULES:\n"
            "- Call a tool ONLY when its preconditions are already established. If a needed "
            "precondition (e.g. logged_in_user) is not yet established, FIRST call the tool that "
            "establishes it.\n"
            "- Prefer the cheapest path: never repeat a call whose result you already have; avoid "
            "tools you don't need for the goal.\n"
            "- If a required precondition is a fact that is FALSE and no tool can establish it, "
            "output STOP (refusing is the correct answer — do not call the goal tool).\n"
            "- When the goal tool's preconditions are all established, call the goal tool.\n\n"
            "Output ONLY one tool name from the list, or STOP. Nothing else:")
        resp = self._client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, top_p=0.01, max_tokens=24)
        raw = (resp.choices[0].message.content or "").strip()
        low = raw.lower()
        if "stop" in low or "refuse" in low or "exit_conversation" in low:
            return "STOP"
        # copy-grounded: pick the tool name that appears in the output (longest match wins)
        hits = [n for n in tool_names if n and n in raw]
        if hits:
            return max(hits, key=len)
        first = low.split()[0].strip(".,:\"'") if low.split() else ""
        for n in tool_names:
            if n.lower() == first:
                return n
        return "STOP"   # uninterpretable -> refuse rather than blind first-tool (C1/N2 fix)

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
        all_in_slots = bool(required) and all(r in self._slot_state for r in required)
        if all_in_slots:
            # diagnostic: this turn COULD be resolved deterministically
            self.cov_deterministic += 1
            if self.use_deterministic_shortcut:
                # rung (a): build the call from slot state, NO LLM
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
                # SOPBench tool results may be JSON or python-repr (try_eval-style).
                r = _try_parse(m.get("content", ""))
                if isinstance(r, dict):
                    self._slot_state.update(r)
            if m.get("role") == "user" and \
               str(m.get("content", "")).startswith("Here is all the information"):
                try:
                    txt = m["content"]
                    block = _try_parse(txt[txt.index("{"):txt.rindex("}") + 1])
                    if isinstance(block, dict):
                        self._slot_state.update(block)
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
