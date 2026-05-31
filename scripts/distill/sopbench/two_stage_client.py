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
import hashlib
import json
import os
import re
from openai import OpenAI


# ----------------------------------------------------------------------------
# Tool-/predicate-NAME ALIAS masking (TASK_CONSTRAINT_DESIGN §8.5.★ ①, top priority).
# Anti-cheat: present every referential name (tool, predicate, the check/establish tools
# named INSIDE status lines, history, descriptions) as an opaque per-task alias so the
# planner cannot memorize "login->apply" or lexical-shortcut policy<->tool. It is FORCED
# to semantic-match the NL policy/request against the (scrubbed) tool DESCRIPTION. This is
# the validity gate on the LODO transfer headline: with names memorizable, a positive
# transfer number is suspect; with aliases, transfer = the general NL->procedure skill.
#
# NOTE (orthogonality, design review): alias is independent of source1/source3. Aliasing
# only defangs NAME memorization; source3 (drop the constraint-derived STATUS/precond
# "answer key") is what removes structure spoon-feeding. The real anti-cheat = alias ON
# *and* source3. Alias-ON+source1 still hands the anonymized dirgraph (op_7 => VERIFY: op_3).
#
# Train/eval need NOT share the map: within one (prompt,target) pair the map is consistent;
# across train vs eval the salt differs ON PURPOSE so a model that merely memorized an
# alias<->tool binding fails — only genuine description-grounded matching transfers.
# ----------------------------------------------------------------------------
def make_alias_map(terms, salt: str = "") -> dict:
    """Deterministic bijection {real_name -> opaque alias} over the union of tool + predicate
    names. Reproducible across processes (sha256, not the salted builtin hash). `salt` permutes
    the assignment; vary it per task (anti-memorization) and differently in train vs eval."""
    names = sorted({t for t in terms if t and t != "exit_conversation"})
    order = sorted(names, key=lambda n: hashlib.sha256(f"{salt}|{n}".encode()).hexdigest())
    amap = {n: f"op_{i}" for i, n in enumerate(order)}
    amap["exit_conversation"] = "exit_conversation"      # keep STOP/terminate sentinel stable
    return amap


def _alias_text(text: str, amap: dict) -> str:
    """Whole-token replace every real name in `text` with its alias (word boundaries so e.g.
    'get_account_balance' is not partially hit by 'get_account_balance_safety'). Scrubs
    descriptions / policy / history of leaked real names."""
    if not amap or not text:
        return text
    # longest names first is irrelevant with \b boundaries, but keeps behaviour obvious
    for real in sorted(amap, key=len, reverse=True):
        if real == "exit_conversation":
            continue
        text = re.sub(rf"\b{re.escape(real)}\b", amap[real], text)
    return text


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


# ----------------------------------------------------------------------------
# arm-3v2 / arm-4a SHARED planner-prompt builder (train/test consistency: the SFT
# data generator and the inference planner MUST produce byte-identical prompts).
# ----------------------------------------------------------------------------
def _render_precond_mod(tree, predicates, out, est):
    if not tree:
        return
    if isinstance(tree, (list, tuple)) and tree:
        head = tree[0]
        if head == "single":
            name = tree[1]
            name = name[4:] if name.startswith("not ") else name
            info = (predicates or {}).get(name, {})
            out.append(name)
            if info.get("kind") == "establishable" and info.get("by"):
                est[name] = info["by"]
        elif head in ("and", "or", "chain", "gate"):
            for sub in tree[1]:
                _render_precond_mod(sub, predicates, out, est)


def build_v2_prompt(abox, op_names, established, user_req, policy, history_lines, slot_keys,
                    op_descs=None, observed=None, goal_name=None, goal_constraint=None,
                    alias_map=None, source=1):
    """Build the arm-3v2/arm-4a planner prompt. `op_names` = the exact tool-name order shown to
    the model. `established` = establishable preds already satisfied. `observed` = {fact_pred ->
    bool} learned from prior internal-check results in history (fact-visibility, arm-4a v2).

    `goal_name`/`goal_constraint` (TASK_CONSTRAINT_DESIGN.md mechanism A): when given, the GOAL
    operator's precondition is rendered from the TASK-SPECIFIC constraint tree `goal_constraint`
    instead of the domain-default `op["precondition"]`.

    `alias_map` (§8.5.★ ①): {real_name -> opaque alias} applied to EVERY referential name —
    the tool's leading name, its precondition predicate names, the check/establish tools named
    inside status lines, the HOW-TO-ESTABLISH block, history, and (scrubbed) descriptions/policy.
    Aliasing only the leading name leaks the procedure via needs[]/STATUS, so we alias the whole
    graph. `op_names` stays REAL (callers look operators up by real name); aliasing is render-only.
    `source` (§8.5.★ ②): 1 = render the constraint-derived needs/gives/STATUS lines (the
    "answer key"). 3 = render ONLY the (scrubbed) tool description + the NL policy; the planner
    must INFER which checks are required from the policy (no spoon-fed structure). The real
    anti-cheat is alias_map AND source=3 together. Returns the prompt string."""
    ops = abox.get("operators", {})
    predicates = abox.get("predicates", {})
    op_descs = op_descs or {}
    observed = observed or {}
    op_set = set(op_names)
    amap = alias_map or {}
    A = lambda n: amap.get(n, n)                         # noqa: E731 — render-time alias
    S = lambda t: _alias_text(t, amap)                   # noqa: E731 — scrub free text

    est_map, lines = {}, []
    for nm in op_names:
        if nm == "exit_conversation":
            continue
        op = ops.get(nm)
        if source == 3:
            # affordance-only: description (scrubbed) + alias; NO needs/gives/STATUS answer key.
            desc = S(op_descs.get(nm, "") or (op.get("description", "") if op else "") or "(tool)")
            lines.append(f"- {A(nm)}: {desc[:160]}")
            continue
        if op:
            preds, est = [], {}
            precond = (goal_constraint if (goal_name and nm == goal_name and goal_constraint is not None)
                       else op.get("precondition"))
            _render_precond_mod(precond, predicates, preds, est)
            est_map.update(est)
            needs = ", ".join(A(p) for p in dict.fromkeys(preds)) or "nothing"
            gives = ", ".join(A(p) for p in op.get("produces", [])) or "the goal/result"
            # fact preconditions = non-establishable preds; a fact is checkable if it is a callable tool
            facts = [p for p in dict.fromkeys(preds) if p not in est]
            violated = [p for p in facts if observed.get(p) is False]
            uncheckable_unknown = [p for p in facts if p in op_set and p not in observed]
            unmet_est = [p for p in est if p not in established]
            if violated:
                status = "BLOCKED by FACT (a required check returned FALSE) => STOP, do not call this"
            elif unmet_est:
                status = f"BLOCKED — first call: {', '.join(sorted({A(est[p]) for p in unmet_est}))}"
            elif uncheckable_unknown:
                status = f"VERIFY FIRST — call these checks: {', '.join(sorted(A(p) for p in uncheckable_unknown))}"
            else:
                status = "READY (preconditions satisfied/verified)"
            lines.append(f"- {A(nm)}: needs [{needs}]; gives [{gives}]  => {status}")
        else:
            lines.append(f"- {A(nm)}: {S(op_descs.get(nm, '(tool)'))[:80]}")
    ops_str = "\n".join(lines)
    hist_str = S("\n".join(history_lines)) if history_lines else "nothing yet"
    slots_str = ", ".join(sorted(slot_keys)) or "only what the user provided"
    policy_str = S(str(policy))
    req_str = S(str(user_req))
    if source == 3:
        return (
            "You are a planning agent. Pick the SINGLE next tool to call, or STOP.\n\n"
            f"USER REQUEST:\n{req_str}\n\nPOLICY (constraints to honor):\n{policy_str}\n\n"
            f"TOOLS (name: what it does):\n{ops_str}\n\n"
            f"ALREADY KNOWN/ESTABLISHED: {slots_str}\n"
            f"HISTORY:\n{hist_str}\n\n"
            "RULES:\n"
            "- Read the POLICY and decide which conditions must be VERIFIED before the goal.\n"
            "- Call the verification/lookup tool whose description matches each required check "
            "FIRST; only then act.\n"
            "- If a verified condition fails (a required check is false), output STOP — refusing "
            "is correct; do NOT force the goal.\n"
            "- Prefer the cheapest path: never repeat a call whose result you already have.\n"
            "- When every required condition is satisfied and the goal is not yet done, call the "
            "goal tool, then STOP.\n\n"
            "Output ONLY one tool name from the list, or STOP. Nothing else:")
    est_str = ("\n".join(f"  - to establish '{A(p)}', call {A(a)}" for p, a in est_map.items())
               or "  (none)")
    return (
        "You are a planning agent. Pick the SINGLE next tool to call, or STOP.\n\n"
        f"USER REQUEST:\n{req_str}\n\nPOLICY (constraints to honor):\n{policy_str}\n\n"
        f"TOOLS (name: needs [preconditions]; gives [effects]):\n{ops_str}\n\n"
        f"HOW TO ESTABLISH preconditions:\n{est_str}\n\n"
        f"ALREADY KNOWN/ESTABLISHED: {slots_str}\n"
        f"HISTORY:\n{hist_str}\n\n"
        "RULES:\n"
        "- NEVER call a tool marked BLOCKED. Call its 'first call' tool instead.\n"
        "- If a tool is 'VERIFY FIRST', call the listed check tool(s) first to confirm its facts.\n"
        "- If the goal tool is 'BLOCKED by FACT', output STOP (refusing is correct — a required "
        "fact is false; do NOT call the goal).\n"
        "- Prefer the cheapest path: never repeat a call whose result you already have.\n"
        "- When the goal tool is READY and not yet successfully called, call the goal tool, "
        "then STOP.\n\n"
        "Output ONLY one tool name from the list, or STOP. Nothing else:")


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
        # mechanism A (TASK_CONSTRAINT_DESIGN): per-task goal constraint for goal-status rendering.
        # Active only when env SOPBENCH_LIGHTEN is set (zero-train diagnostic toggle).
        self._task_constraints = None
        self._goal_name = None
        self._lighten = bool(os.environ.get("SOPBENCH_LIGHTEN"))
        # §8.5.★ ① tool-name alias masking + ② source (1=render answer-key STATUS, 3=NL-only).
        self._alias = bool(os.environ.get("SOPBENCH_ALIAS"))
        self._source = int(os.environ.get("SOPBENCH_SOURCE", "1"))
        self._alias_map = None        # {real -> alias}, built once per task (reset clears)
        self._alias_inv = None        # {alias -> real}, to de-alias the planner output
        # coverage counters (cumulative across the run)
        self.cov_turns = 0
        self.cov_deterministic = 0

    def reset(self, task_constraints=None, goal=None):
        """Call once per task before the interaction. `task_constraints`/`goal` (optional) feed
        mechanism A's task-specific goal-status rendering (active under env SOPBENCH_LIGHTEN)."""
        self._slot_state = {}
        self._turn = 0
        self._task_constraints = task_constraints
        self._goal_name = goal
        self._alias_map = None        # rebuilt lazily on first plan of this task
        self._alias_inv = None

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
        op_descs = {t.get("function", {}).get("name", ""):
                    t.get("function", {}).get("description", "") for t in tools}
        # established establishable-predicates: an operator that was called and did NOT error/
        # return False establishes its `produces` (deterministic gate-status, from history).
        established = set()
        observed = {}                       # fact_pred -> bool, from internal-check results (v2)
        for m in messages:
            if m.get("role") == "tool":
                nm = m.get("tool_name")
                c = str(m.get("content", ""))
                if nm in ops and "Error" not in c and c.strip() not in ("False", "false", "None", ""):
                    established.update(ops[nm].get("produces", []))
                # fact-visibility: a callable check (tool name == a fact predicate) reveals a bool
                if nm and "Error" not in c:
                    r = _try_parse(c)
                    v = r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else r
                    if isinstance(v, bool):
                        observed[nm] = v
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
        gname = self._goal_name if self._lighten else None
        gconstr = self._task_constraints if self._lighten else None
        # §8.5.★ ①: build the per-task alias map once (union of operator + predicate + tool names).
        # eval salt differs from train salt by design -> a model that memorized an alias<->tool
        # binding fails; only description-grounded semantic matching transfers.
        if self._alias and self._alias_map is None:
            terms = set(ops) | set(self.abox.get("predicates", {})) | set(tool_names)
            salt = f"eval|{self._goal_name}|" + "|".join(sorted(tool_names))
            self._alias_map = make_alias_map(terms, salt)
            self._alias_inv = {v: k for k, v in self._alias_map.items()}
        amap = self._alias_map if self._alias else None
        prompt = build_v2_prompt(self.abox, tool_names, established, user_req, policy,
                                 hist, set(self._slot_state.keys()), op_descs, observed,
                                 goal_name=gname, goal_constraint=gconstr,
                                 alias_map=amap, source=self._source)
        resp = self._client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, top_p=0.01, max_tokens=24)
        raw = (resp.choices[0].message.content or "").strip()
        low = raw.lower()
        if "stop" in low or "refuse" in low or "exit_conversation" in low:
            return "STOP"
        # match against the SHOWN names (aliases when alias is on), then de-alias to the real tool.
        shown = [amap[n] for n in tool_names] if amap else tool_names
        hits = [s for s in shown if s and s in raw]
        if hits:
            best = max(hits, key=len)
            return self._alias_inv.get(best, best) if amap else best
        first = low.split()[0].strip(".,:\"'") if low.split() else ""
        for s in shown:
            if s.lower() == first:
                return self._alias_inv.get(s, s) if amap else s
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
