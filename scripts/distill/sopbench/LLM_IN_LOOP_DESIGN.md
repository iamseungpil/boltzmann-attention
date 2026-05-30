# LLM-in-loop experiment — re-established design (2026-05-31)

Status: **AUTHORITATIVE for the LLM-in-loop track.** Re-centers the experiment after the
user's correction. Supersedes the "deterministic-executor-first (P0)" framing in
`WORKFLOW_ONTOLOGY_DESIGN.md §3/§6` and `EXPERIMENT_DESIGN §16.5`: the deterministic
executor is now explicitly the **oracle / upper bound**, not the contribution.

> Companion docs (still valid for the *ontology schema* and *prior-art positioning*):
> - `WORKFLOW_ONTOLOGY_DESIGN.md` §1 (TBox/ABox split), §2 (8-relation schema), §5 (L0/L1/L2
>   planner + GoalAct/ReWOO/AWM prior art). Those sections are unchanged.
> - `EXPERIMENT_DESIGN §15.13` (neural ABox-conditioned resolver), §15.14 (Routine layers).
> This doc replaces the *experiment center of gravity* and the *primary benchmark*.

---

## 0. The correction (why this doc exists)

The 12-domain deterministic executor (`workflow_executor.py` + `abox/`, mean 97%, 8/12 at
100%) was being read as "the result." **It is not.** SOP-Bench-style benchmarks measure
**how well an LLM, given a goal and tools, completes the task *without an external
deterministic program solving it for it*.** A program that already holds the ground-truth
call graph and walks it will of course score near-ceiling — that is the **oracle / upper
bound**, useful only to (a) validate the env wiring and (b) mark the ceiling.

**The real experiment is LLM-in-loop:** a *learned, domain-general planner* (TBox) + an
*external, swappable operator memory* (ABox) consumed by a resolver, and the headline result
is **transfer** — freeze the planner, swap the ABox to a held-out domain, retrain nothing.
If the system is label-fitting or memorizing, transfer fails; if it learned a general
planning skill, transfer holds. That gap is the contribution.

Honest-axis rule kept from the handoff: **the deterministic 97% is reported as a ceiling
only**, and any domain with no held-out split (label-fit risk) is flagged separately.

---

## 1. Tier-1 dual benchmark (confirmed)

Both run as Tier-1. Name collision is real: **SOPBench** (Zekun Li) ≠ **SOP-Bench** (Amazon).

| | **SOPBench — Zekun Li, arXiv 2503.08669** ★PRIMARY | **SOP-Bench — Amazon, 2506.08119** complementary |
|---|---|---|
| clone | `/home/woori/scratch/SOPBench` | `/home/woori/scratch/SOP-Bench` |
| domains | 7 (bank, dmv, healthcare, hotel, library, online_market, university) | 12 industry |
| operator ABox | ✅✅ **native, formal**: `env/domains/<d>/<d>_assistant.py` (`actions` tool-specs + `action_descriptions/returns/param_descriptions`) + per-task `constraints` / `directed_action_graph` + domain `task_default_dep_full` routines | toolspecs + our compiled `abox/` |
| ground-truth plan | ✅✅ **per task** `directed_action_graph` = nodes `[action, arg-binding]` + `connections` + logical ops (`or`) — the required call DAG | none (we compile/induce) |
| evaluation | ✅✅ **rule oracle** (`env/evaluator.py`, `count_constraint_units`): goal action reached + `constraint_not_violated` + graph conformance. **No LLM judge.** | output-column match (synthetic noise) |
| interaction | multi-turn + user-sim, but **scored by the rule oracle** (so it avoids the tau2 user-sim-as-judge failure) | single-shot |
| extra axes | constraint-violation / jailbreak / **refusal cases** (`action_should_succeed=false`) / `--tool_list full` (distractors = tool-selection@scale = OISA-patent touchpoint) | breadth (12 domains) |
| reported | GPT-5 71–88% / Qwen-7B 5–20% | FC 27% / ReAct 48% (repo ~55–64%) |

**Why SOPBench (Zekun Li) is primary:** it gives us, *natively and formally*, the two things
SOP-Bench (Amazon) makes us reconstruct — (1) a per-task **ground-truth call graph** to
validate any induced/planned workflow against, and (2) a **rule oracle** with zero LLM
judge. That makes the transfer claim auditable. SOP-Bench (Amazon) adds **domain breadth**
(12 vs 7) as the complementary transfer-rotation surface.

**Tier-2 (auxiliary, not headline):** SOP-Maze (2510.08942) = pure decision/QA, no tools →
isolates the planner's SOP *branch-following/reasoning*. Baseline methods to cite/compare:
SOP-Agent (2501.09316), Routine (2507.14447).

---

## 2. The LLM-in-loop architecture (2-stage), grounded in the SOPBench harness

### 2.1 What the harness gives us (verified from the clone)
- `swarm.Swarm(system, max_turns, max_actions, execute_tools=True)` drives the loop: it calls
  `agent.get_chat_completion`, parses tool calls, executes them against the domain DB, appends
  results, repeats until done / `max_turns`.
- `Agent(name, client=OpenAIHandler, instructions, functions, tool_call_mode∈{fc,react,
  act-only,react-v}, temperature, top_p, max_tokens)`. `functions` = the domain's OpenAI
  tool-spec JSON list (`<d>_assistant.py:actions`). The assistant is otherwise a **single
  LLM**: `get_chat_completion` → one model call → tool calls.
- `run_simulation.py` flags: `--domain --assistant_model --user_model --tool_list{oracle,full}
  --tool_call_mode fc --max_num_turns 20 --max_num_actions 10 --num_tasks --env_mode prompt
  --default_constraint_option{full,required}`.
- `tool_list=oracle` → only the task-relevant tools; `full` → all domain tools (distractors).
- Scoring: `env/evaluator.py` rule oracle over the produced action trace vs the task's
  `directed_action_graph` + `constraints`.

### 2.2 The plug-in point
**Replace the single-LLM assistant with our 2-stage policy.** Mirrors the tau2 patch pattern:
the `Agent.client` (or a thin `Agent` subclass) internally runs `planner → resolver` and
returns a `ChatCompletionMessage` carrying the chosen tool call. The Swarm loop, tool
execution, and rule oracle are reused unchanged. So one "turn" of the Swarm = one
(plan-step, resolve-to-tool, execute) cycle.

### 2.3 The two stages
```
goal + user_known + history + slot-state
        │
   ┌────▼─────────────────────────────────────────────┐
   │ PLANNER  (TBox — LEARNED, domain-general)         │
   │  sees: goal, ABSTRACT operator affordances        │
   │        (operator names + precondition/effect      │
   │         TYPES, not the concrete tool schema),     │
   │        global plan G, execution history           │
   │  emits: next ABSTRACT step (subgoal / operator-   │
   │         class), and re-plans G every turn         │
   └────┬──────────────────────────────────────────────┘
        │ abstract step
   ┌────▼─────────────────────────────────────────────┐
   │ RESOLVER (ABox-conditioned)                        │
   │  sees: abstract step + the DOMAIN'S concrete       │
   │        operator ABox (this tool's schema,          │
   │        arg sources) + slot-state                   │
   │  emits: concrete tool call with bound args         │
   │  rungs: (a) deterministic rule  = baseline/diag    │
   │         (b) ontollm (ABox in prompt)               │
   │         (c) neural ABox-conditioned (xattn /        │
   │             per-domain memory)  = ★novelty          │
   └────┬──────────────────────────────────────────────┘
        │ tool call
   ┌────▼─────────────────────────────────────────────┐
   │ ENV  (SOPBench Swarm executes, returns result,     │
   │       updates DB/slot-state)                       │
   └────────────────────────────────────────────────────┘
```

**Why the split carries transfer (the whole point).** The **planner weights** encode the
*general* skill — means-ends planning over operator *types* (which precondition must hold,
which effect fills an unmet goal slot), GoalAct-style global-plan + re-plan each turn (the
structural cure for the tau2 read-loop). The **ABox** (concrete operators) is *external
swappable memory* consumed by the resolver. Freeze the planner, swap the ABox → held-out
domain works with **zero retraining**. Memorize-or-label-fit ⇒ transfer collapses.

### 2.4 Planner rungs (= WORKFLOW_ONTOLOGY_DESIGN §5)
- **L0 — symbolic means-ends** (no learning): forward/backward chaining over operator
  pre/effects (the `directed_action_graph` dependency structure). Measures how far *operators
  alone* determine the workflow. Clean floor.
- **L1 — LLM + operators in context** (no training): goal + operator list + state → next
  operator. The zero-shot "agentic" bar.
- **L2 — learned ABox-conditioned planner** (★contribution): train across N−1 domains'
  success traces to map (goal, operators-as-memory, state) → next operator, operators
  **injected as swappable memory** (cross-attention / per-domain module), not baked into
  weights. Transfer = swap the memory.

---

## 3. Comparison arms

1. **LLM-alone (baseline)** — SOPBench native single-LLM assistant (`tool_call_mode=fc` or
   `react`), full tool/SOP context, plans+executes itself. (Qwen-7B 5–20%, GPT-5 71–88%.)
2. **LLM+structure, no training** — L1 planner (LLM + operators) → resolver (b); plus **L0**
   (symbolic, operators-only) as the floor. Tests whether the 2-stage structure + global-plan
   re-planning alone helps, before any learning.
3. **Ours — L2** — learned ABox-conditioned planner + resolver (c). Trained on N−1 domains.

Ablation within arm 3: greedy next-operator (ReAct-like) vs global-plan+re-plan (GoalAct) —
expect greedy → local sticking (the tau2 read-loop, structurally).

---

## 4. Metrics

- **Rule pass-rate** (SOPBench oracle: goal reached + `constraint_not_violated` + graph
  conformance). **Primary.** On SOP-Bench (Amazon): TSR / ECR / C-TSR.
- **Refusal accuracy** on `action_should_succeed=false` tasks (constraints unsatisfiable →
  the agent must *correctly decline*, not hallucinate success). A distinct, hard axis SOPBench
  gives for free.
- **Tool-selection@scale** — `--tool_list full` vs `oracle`; the degradation = how well the
  policy selects among distractors. (Touchpoint with the OISA patent track, reported jointly
  only on this axis.)
- **Resolver coverage%** (diagnostic) — fraction of steps resolved deterministically (rung a)
  vs needing the LLM. "How rarely we must invoke an LLM to resolve a step."
- **Transfer Δ (★headline)** — held-out-domain rule pass-rate / in-domain rule pass-rate.
  Target ≥ 70% with zero retraining. Plus **ABox-ablation** (empty/wrong operators → L2 must
  collapse → proves it *reads* the ABox).
- **Oracle ceiling** — the deterministic executor's pass-rate, reported as the upper bound
  only (and label-fit domains flagged).

---

## 5. Transfer protocol (Phase 1b — the result)

- **SOPBench (Zekun Li):** 7-domain leave-one-domain-out rotation (bank/dmv/healthcare/hotel/
  library/online_market/university). Train L2 on 6, freeze, swap held-out domain's ABox
  (its `actions` specs + dependency routines), zero retrain → rule pass-rate.
- **SOP-Bench (Amazon):** 12-domain rotation, same protocol → breadth.
- **Success criteria:** (a) held-out rule pass-rate ≥ 70% of in-domain; (b) ABox-ablation
  collapses; (c) L0 < L1 < L2 gap visible; (d) induced-vs-`directed_action_graph` structural
  agreement high (SOPBench gives ground truth → induction is *validated*, not assumed).

Phase 2 (only after 1b holds): AppWorld (457 APIs / 9 apps, goal-only, no SOP) — extend
transfer to fully autonomous, procedure-free planning. Same ABox-ablation control.

---

## 6. Bank pilot — concrete next steps (do these first)

Pilot on **bank** end-to-end before scaling to 7/12 domains. Build order:

1. **Reproduce the baseline.** Install SOPBench requirements into a venv (swarm/litellm etc.;
   `task_default_dep_full` + `env/helpers.py` use `match` → **Python ≥3.10**, so seka_env or a
   3.10+ venv, *not* the system `python` 3.9). Run:
   `python run_simulation.py --domain bank --tool_list oracle --assistant_model <model>
   --num_tasks <small> --max_num_turns 20` then `run_evaluation.py` → get the baseline rule
   pass-rate. Cheapest assistant_model = an OpenRouter model via the `-Or` key (avoids local
   vLLM/GPU). **Open item:** confirm `OpenAIHandler` routes to OpenRouter (base_url/key) vs
   spinning vLLM from `num_gpus`; pick the API path for the pilot.
2. **Resolve the user-sim question (handoff §6).** SOPBench is multi-turn with a `user_model`.
   For pilot reproducibility/cost: start with the **dummy user** (`default_response`/
   `response_repeat`) or a deterministic small `--user_model`; confirm how `user_known` reaches
   the assistant (instructions vs user turns). Record the choice (affects whether the agent
   must *elicit* slots or gets them up front).
3. **Extract the operator ABox** from `bank_assistant.py` (`actions` specs) + the dependency
   routines (`task_default_dep_full('bank', ...)`). Split into: **abstract affordances**
   (operator name + precondition/effect type) for the planner, and **concrete schema**
   (params, arg sources) for the resolver.
4. **Build the 2-stage assistant** as an `Agent` whose `client` runs planner→resolver and
   returns a tool-call `ChatCompletionMessage` per Swarm turn. **Start at L1 + resolver (b)** to
   validate the loop emits valid, oracle-scored tool sequences. Self-test on 1 task: trace the
   plan steps, the resolved calls, and the oracle verdict against that task's
   `directed_action_graph`.
5. **Compare** baseline (arm 1) vs L1-2stage (arm 2) on bank → sanity that structure ≥ parity.
   Then add L0, then wire L2 training data (bank success traces → (goal, operators, state) →
   next-operator labels). Then scale to the 7-domain rotation.

Self-test each step (run on 1 task, inspect the trace vs the ground-truth graph) before
scaling.

---

## 7. Open items / honest risks

- **OpenAIHandler routing** (API vs local vLLM) — resolve in pilot step 1.
- **User-sim determinism/cost** — resolve in pilot step 2; SOPBench multi-turn + user_model is
  the one place tau2-style nondeterminism could leak in (mitigated because scoring is the rule
  oracle, not the user).
- **Planner training data** — L2 needs (goal, operators, state) → next-operator labels mined
  from success traces (analogous to the telret abstract-step SFT). Recipe = GPU train in INFRA.
- **Label-fit / no-holdout domains** — flag separately; the transfer rotation is the guard.
- **Abstract-vs-concrete operator leakage** — the planner must NOT see concrete tool schemas
  (else it can memorize domain specifics and transfer is contaminated). Enforce the
  abstract/concrete split at the prompt/feature boundary.
- Claim retired: "deterministic beat the LLM." Replaced by: "given structure + a learned
  ABox-swappable planner, the LLM-in-loop gains X over baseline **and transfers**."
