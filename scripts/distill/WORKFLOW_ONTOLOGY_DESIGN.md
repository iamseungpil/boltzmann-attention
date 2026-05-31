# Full Workflow Ontology — design (2026-05-30)

> ### ★ EXPERIMENT RE-CENTERED — LLM-in-loop (2026-05-31, integrated)
> The deterministic executor (§3) is the **oracle / upper bound, NOT the contribution.** A
> program that already holds the ground-truth call graph and walks it scores near-ceiling by
> construction (the 12-domain `abox/` mean 97% = that ceiling). **The real experiment is
> LLM-in-loop:** a *learned, domain-general planner* (TBox) + an *external, swappable operator
> memory* (ABox) consumed by a resolver, with the **headline = transfer** (freeze the planner,
> swap the ABox to a held-out domain, retrain nothing). The full LLM-in-loop spec — 2-stage
> loop, comparison arms, metrics, transfer protocol, pilot — is now **§9 of this doc**
> (consolidated here from the former `sopbench/LLM_IN_LOOP_DESIGN.md`).
>
> **★ Tier-1 dual benchmark (naming is confusing — read carefully):**
> - **SOPBench** (Zekun Li, arXiv **2503.08669**, *no hyphen*) = **PRIMARY (主).** 7 domains
>   (bank/dmv/healthcare/hotel/library/online_market/university). **Native formal operators**
>   (`env/domains/<d>/<d>_assistant.py` + per-task `directed_action_graph` + `constraints`) and
>   a **rule oracle** (`env/evaluator.py`, no LLM judge) → the transfer claim is auditable.
>   **Pilot = bank.**
> - **SOP-Bench** (Amazon, arXiv **2506.08119**, *with hyphen*) = **auxiliary (보조).** 12
>   industry domains → breadth / second transfer surface. (`abox/` 12-domain assets live here.)
>
> §1 (TBox/ABox split), §2 (8-relation schema), §5 (L0/L1/L2 planner + prior art), §6 (phased
> plan, 1b transfer = headline) all stay valid and already embody the LLM-in-loop framing;
> §3/§6's "P0-first / deterministic-first" reading is the **oracle/ceiling**.

Status: **DESIGN.** Supersedes the partial "fix-disambiguation only" ontology
(`step_realization_*` + `obs_triggers_*`) and the per-turn reactive resolver
(`ontology_resolver.py` + `two_stage_agent.py:_two_stage_generate`).

> ### ★ BENCHMARK = SOP-bench family (confirmed 2026-05-30 밤; primary corrected 2026-05-31)
> *(Superseded on the primary/auxiliary split by the top banner: **SOPBench / Zekun Li
> 2503.08669 = primary**, SOP-Bench / Amazon = auxiliary. The tau2→SOP-bench-family rationale
> below applies to both.)*
> The target benchmark moved **tau2 → SOP-Bench** (arXiv 2506.08119, amazon-science).
> Rationale + full experiment design = `reports/EXPERIMENT_DESIGN_v1_7_facet_rft.md §16`.
> Why it fits where tau2 didn't: SOP-Bench is **agent-controlled** (the agent calls all
> tools itself), **single-shot** (no user simulator, no dialogue), and the SOP carries
> **explicit If-branches / state-dependent decisions** — exactly our TBox/ABox + Group J
> relations. This **eliminates the tau2 failure modes** diagnosed in §0 below
> (max_steps non-convergence, read-loop, user-side locus): those were artifacts of a
> user-sim-centric benchmark, not of our approach.
>
> Consequences for THIS doc:
> - §0 (tau2 diagnosis) is the **motivation** for moving to an agent-controlled benchmark.
> - §2 schema is benchmark-agnostic → **unchanged** (slots = SOP-Bench CSV columns,
>   tools = toolspecs.json, phases = SOP sections).
> - §3 executor **simplifies**: SOP-Bench has NO user-side tools, so the D2 machinery
>   (NL instruction templates, user-tool-result ingestion) is **dropped** — the executor
>   emits agent tool calls only. See §3 note.
> - Induction gains a 2nd route: **compile sop.txt directly** (the SOP is the ground-truth
>   workflow), and we can **validate induced ontology vs the authored SOP**.
> - §6/§7 updated to SOP-Bench (TSR/Tool-Accuracy, 12-domain ABox-swap transfer).

---

## 0. Why (trajectory-grounded diagnosis)

Re-analysis of the v3 telecom run (N=114, `two_stage_v3/`) showed the previous
"coverage high but wrong timing" story was wrong. The real failure decomposition:

1. **Non-convergence dominates.** 86% of resolver failures are `max_steps` — the sim
   is truncated before completion, so `reward_info` is null (reward 0) regardless of
   tool correctness. DB/action/env actually-wrong cases are a handful (3–7).
2. **Read-loop amplification.** Identical-call repetition ≥4× ⟹ 100% failure; the
   resolver monotonically increases loopers (base 23 → resolver 32 → fallback 44) and
   decreases pass (31→23→13). The planner re-emits `gather_account_context`
   indefinitely; the resolver cannot break it.
3. **Wrong locus.** 86% of agent tool-calls are reads; ~half of tasks have **zero**
   agent write-calls. Telecom resolution runs through **agent NL instructions →
   user-side device diagnostics** (check_sim, check_network_status, reset_apn…), which
   the write-only ontology and the per-turn resolver never touch.

Root causes, structurally:
- The ontology models only **¼ of the workflow** (the `apply_targeted_fix` write
  disambiguation). Reads, diagnostics (14 device tools dumped untriggered into
  `apply_policy_action`), control flow, and closure are blank.
- The "executor" is a **per-turn, stateless tool-rewriter**. The PLANNER (LLM) owns all
  control flow (call vs talk, which step, when to transition). The resolver only swaps
  one concrete tool. It therefore **cannot own a gather→fix transition** — the central
  thing needed to stop the read-loop.

Decision (user, 2026-05-30): **design and reimplement the whole workflow ontology**, so
TBox/ABox express read-decision + diagnosis + write + verify + closure, and the executor
DRIVES the loop deterministically; rebuild all experiments on it.

---

## 1. The practical goal — goal-directed agentic tool use

**Target use case (Agentic AI):** the user gives only a **high-level goal**; the system
**autonomously selects skills/tools and composes a workflow** to complete it. We want a
generalized ontology for *that*, not for executing a hand-written procedure.

This fixes the TBox/ABox split precisely:

| | **ABox** (per-domain, given/inducible) | **TBox** (domain-general, **LEARNED**) |
|---|---|---|
| what it is | the domain's **tool/skill relations only** — each tool's `precondition`, `produces`, `arg`, effect, and which goal-slot it `achieves` | the general **planning / tool-selection skill**: given a goal + tool relations, decide *which tool next, in what order, when done* |
| transfers? | no — **swapped** per domain | yes — the thing that generalizes |
| how obtained | parsed/induced from tool specs + traces | **learned** across many domains' traces |
| analogy | PDDL/HTN domain operators (pre/eff) | the general planner over those operators |

So: **TBox = a learned means-ends planner; ABox = domain operators (tool affordances).** The
agent, given a goal and a domain's operators, plans and executes — no procedure spelled out.

### Two evaluation modes (SOP-Bench supports both)
- **Procedure-given** (easier, validates the executor): the workflow (`next`/`scenario_steps`)
  is compiled from `sop.txt` → the deterministic executor (§3) just runs it. Upper bound.
- **Goal-only (★the real agentic target)**: the agent gets the **goal** (required output
  slots) + the **operator ABox only** (no sequence). The **TBox planner composes the workflow**
  from operators via means-ends (pick a tool whose `precondition` holds and whose effect fills
  an unmet goal/sub-goal slot; repeat until the `output` contract is satisfied). What we
  learn + transfer is exactly this planner.

The earlier "deterministic executor" is now the **procedure-given** special case; the
contribution is the **goal-only** planner that reconstructs the workflow from operators alone.
Computation stays encapsulated in functions (§2.0); the planner only chooses *which function
to call next toward the goal*. This connects to the neural ABox-conditioned resolver
(EXPERIMENT_DESIGN §15.13): TBox = how to consume ABox to choose tools (learned weights),
ABox = operator relations (swappable memory).

---

## 2. SOP Ontology — simple call-graph schema (everything is a function call)

### 2.0 Design principle — push complexity into FUNCTIONS, keep the ontology thin
Surveying all 12 domains showed SOPs carry heavy compute/decide (weighted sums, count-of-TRUE,
formulas, threshold bands, decision tables). **Design decision (simplify): do not model that
logic in the ontology.** Treat **every step — read, compute, decide, write — uniformly as a
FUNCTION CALL**; the complex logic lives *inside* the function. The ontology then needs only a
small, benchmark-general relation set describing the **call graph**: which function, with what
args, under what condition, in what order, producing what.

Function sources (the executor calls; it does not re-implement):
- **Provided tool** — SOP-Bench already exposes most scoring/decisions as tools
  (`CalculateRiskScore`, `make_fulfillment_decision`, `determine_enforcement_action`,
  `calculateChargeback`, `calculateQuantityVariance`, `classifyEmailIntent`, …). Just call it.
- **Wrapped function** — for a pure SOP formula with no tool (e.g. content_flagging BPI), wrap
  as a thin function (small evaluator synthesized from the formula, or LLM-backed). The
  ontology only knows its **name / params / output slot** — never the formula internals.

Why this generalizes: the ontology becomes a typed **dataflow + control call graph** — the
same shape for any tool-using benchmark, not an SOP-specific relation zoo. Complexity is
encapsulated in functions, so the ontology stays inducible, swappable, and small.

### 2.1 Entities
- **Function** (the one unified action): `name`, `params`, `outputs (→slots)`, and a flag
  `kind ∈ {effect (read/write external system), pure (compute/decide — no side effect)}`.
  Provided-tool or wrapped — the call graph treats them identically.
- **Slot** = a state variable = a CSV column (`metadata.input_columns` are given;
  `output_columns` are the goal).
- **Predicate** = a **simple** condition over slots: `slot OP value` (`OP ∈ {=,≠,<,≤,>,≥,
  ∈[a,b], is_missing}`) or a boolean slot. *No composite/threshold/table logic here* — any
  complex condition is computed by a function into a slot, then tested as a simple predicate.
- **Scenario** = a task sub-type (selects which sub-graph runs).

### 2.2 Relations (the WHOLE ontology — 8 simple relations)
ABox instances per domain; TBox = these types + the executor.
- `realizes(step → function)` — which function a step calls. *(= step_realizes_tool)*
- `arg(function, param ← slot)` — bind a parameter from a prior slot. *(dataflow / Routine R1)*
- `produces(function → [slots])` — which slots the call fills.
- `precondition(function → predicate)` — only callable when the (simple) predicate holds.
- `next(step → step | [(predicate → step)])` — control flow: linear next, or a conditional
  fork on simple predicates. *(Routine R3 branch, with an else/default arm)*
- `scenario_select(input_predicate → scenario)` + `scenario_steps(scenario → [steps])` —
  route to a sub-procedure by initial state. *(Routine R4)*
- `terminate(predicate → outcome)` — early-exit / escalate / done with a fixed outcome.
- `output(required:[slots], format)` — final output contract (keys + JSON/XML).

That is the entire schema. Everything else (scoring, banding, tables, imputation, validation,
logging, verification re-checks) is **a function** referenced by `realizes` and consumed via
`produces`/`arg`.

**Operator relations vs plan relations (the §1 split).** Five are **ABox operator
affordances** — pure per-domain tool knowledge, always given/induced: `precondition`,
`produces`, `arg`, plus the function signatures and `achieves(function → goal_slot)`. Three
are **plan** — the control flow: `realizes`/`next`/`scenario_*`. In **procedure-given** mode
the plan is compiled from `sop.txt`; in **goal-only (agentic)** mode the plan is **not given**
— the **TBox planner produces it** at run time from the goal + operator affordances. So the
minimal per-domain ABox is just *operators*; the workflow is either supplied or planned.

### 2.3 Coverage — prior mechanisms map to {simple relation} or {function}
| prior mechanism | where it goes now |
|---|---|
| Group J `repairs_state` (state→tool) | `precondition` + `realizes` |
| Group J `diagnosis_sufficient_for` | `precondition` (decide step requires its input slots filled) |
| Group J `distractor_for` (negative) | not an ontology relation — a *training/eval* signal (kept separate) |
| Group J `escalate_when` | `terminate`/`next` with a predicate |
| Routine R1 variable memory | `arg` + the executor's slot store |
| Routine R2 placeholder slot | `arg` (required) + `produces` |
| Routine R3 branch | `next` (conditional) |
| Routine R4 scenario | `scenario_select` + `scenario_steps` |
| **compute/decide layer** (threshold_band, decision_table, count_score, compute_rule, value_map, impute_rule, range_validate, evidence_gate) | **FUNCTIONS** — provided tool or thin wrapper; ontology only `realizes`+`produces` |
| obligations (log, verify, time, output_contract) | log/verify/SLA = **functions** (a call step); `output` relation covers the final contract |
| robustness_note (fuzzy/typo) | the function for that step is **LLM-backed** (fallback by design) |

So the decision/computation complexity that motivated the survey is **acknowledged but
encapsulated**: it is *why* we make those steps functions, not relations. The ontology proper
stays at 8 relations.

### 2.4 TBox / ABox + induction
- **TBox** (fixed, transfers): the 8 relation types + the entity grammar + the executor (§3).
- **ABox** (per-domain, swaps): the call graph instance — {which functions, `arg` bindings,
  `precondition`/`next` predicates, `scenario` map, `output` contract}. `ontology_<domain>.json`.
- **Induce/compile the call graph only** (orchestration): which function follows which,
  arg provenance (value-match), branch predicates (slot-fork in traces / "If…→" in sop.txt).
  Function *bodies* need no induction — provided tools are given; wrapped formulas are
  compiled once from the sop.txt line or left LLM-backed. (Far simpler than inducing formulas.)
- **Transfer test** = freeze TBox, swap ABox → held-out domain (12-domain rotation).

### 2.5 File layout
```
induced/ontology_<domain>.json  # { functions:[{name,params,outputs,kind}],
                                 #   realizes:{step→function}, arg:{function→{param:slot}},
                                 #   produces:{function→[slots]}, precondition:{function→pred},
                                 #   next:{step→ step | [[pred,step]]},
                                 #   scenario_select:[[pred,scenario]], scenario_steps:{scenario→[steps]},
                                 #   terminate:[[pred,outcome]], output:{required:[slots],format} }
```

---

## 3. Deterministic executor loop (SOP-Bench, single-shot)

The executor just **walks the call graph** (§2.2). State = `slots` (INPUT + every produced
value) and `done`. At each turn: pick the next step whose `precondition` holds, call its
`function` (binding params via `arg`), store `produces` into slots, then follow `next` (a
linear edge or a conditional fork on a simple predicate). Stop on `terminate`; render `output`.

```
def run(inputs, ont, call):                 # call(function, args) = tool OR wrapped/LLM fn
    slots, done = dict(inputs), set()
    scenario = ont.scenario_select(slots)    # R4: pick sub-graph by initial state (or None)
    steps = ont.scenario_steps.get(scenario, ont.all_steps)
    step = steps[0]
    while step is not None:
        t = ont.terminate.match(slots)                       # early-exit / escalate / done
        if t: slots[t.outcome_slot] = t.outcome; break

        fn = ont.realizes[step]
        if ont.precondition.ok(fn, slots) and fn not in done:
            args = { p: slots[s] for p, s in ont.arg[fn].items() }
            out  = call(fn, args)                            # provided tool / wrapped / LLM
            slots.update(zip(ont.produces[fn], out)); done.add(fn)

        step = ont.next(step, slots)         # linear, or conditional fork on simple predicate
    return render(ont.output, slots)         # required keys + JSON/XML format
```

`call(fn, args)` dispatches by function kind: **provided tool** → SOP-Bench tool API;
**wrapped** → a small evaluator compiled from the sop.txt formula; **LLM-backed** → the
fallback path for genuinely generative/fuzzy steps. The executor logic is the *same* for all
three — it only sees a function name, args, and outputs.

**Properties**
- **Simple + general**: 8 relations, one loop. No phase-role machine, no per-type handlers —
  the call graph carries everything. Same executor for any tool-using benchmark.
- **Deterministic where the call graph is**: `precondition`/`next`/`scenario_select` are simple
  predicates over slots; the heavy logic is inside functions (often provided tools) → zero LLM
  on those steps.
- **Clean termination + complete output**: `terminate` + `output` produce exactly the required
  keys/format (SOP-Bench scores the produced state).
- **LLM only inside LLM-backed functions** (free-text fields, typo-tolerant/semantic steps).
  Coverage% = fraction of called functions that are tool/wrapped (deterministic) vs LLM-backed.

---

## 4. Induction plan — induce/compile the CALL GRAPH (8 relations only)

We only extract the **orchestration** (the 8 relations); we do **not** mine formulas/tables —
those are functions. Two extractors per ABox, report agreement + TSR. **Compile** parses the
authored SOP; **Induce** mines teacher SUCCESS traces + the `test_set_with_outputs.csv`
**column dependency graph** (input→intermediate→output). Authored SOP = ground truth → the
induced call graph is *validated*, not just assumed (the research claim; impossible in tau2).

| relation | COMPILE (sop.txt) | INDUCE (traces + columns) | difficulty |
|---|---|---|---|
| `realizes` (step→function) | numbered steps → the tool named in the step | which function call appears at each step in traces | easy |
| `arg` (param←slot) | tool `inputSchema` params ↔ column names; "save … for step N" | value-match provenance (existing miner, all functions) | easy |
| `produces` (function→slots) | step "Output:" / which column it sets | columns newly filled after the call | easy |
| `precondition` (function→pred) | "If <simple cond> …" guarding a step | simple slot-predicate present before the call | easy–med |
| `next` (step→step \| forks) | section order + "If <cond> → <step>" | step-order frequency + slot-fork at branch points | med |
| `scenario_select` / `scenario_steps` | "intent = a\|b\|…" → which sub-steps | cluster GT rows by outcome/issue signature → per-scenario step set | med |
| `terminate` (pred→outcome) | "no further action → <default>", escalation rules | early-terminating GT rows; one-hot outcome columns | easy |
| `output` (required, format) | §6 Output (keys + "json/xml") | `metadata.output_columns` | easy |

**Function bodies are NOT induced.** Provided tools are given by SOP-Bench. A pure-formula step
with no tool is **compiled once** from its sop.txt line into a tiny evaluator (e.g.
`hazard = a+b+c+d`), or left **LLM-backed** if it is genuinely generative/fuzzy. This is the
big simplification: we induce a graph of name/args/conditions, never a formula.

Honest risks:
- Branch/scenario predicates can be noisy when they depend on a value only a function computes
  → ensure that function runs *before* the branch (topological order from `produces`/`arg`).
- Transfer: the 8 relation TYPES transfer by construction; the per-domain call graph swaps. 12
  independent domain schemas = a hard, honest transfer test.
- A wrapped-formula step mis-compiled from prose → falls back to LLM-backed (graceful; measured
  by coverage%).

---

## 5. The TBox planner (the learned, transferable thing)

In **procedure-given** mode there is no planner — the executor (§3) runs the supplied call
graph. The research object is the **goal-only** planner: given the **goal** (the `output`
contract's required slots) + the domain **operator ABox** (`precondition`/`produces`/`arg`/
`achieves` per tool), choose the next tool until the goal slots are filled. This planner is
the **TBox** — domain-invariant, learned, swap the ABox to transfer. Three rungs:

- **L0 — symbolic means-ends planner** (no learning): forward/backward chaining over operator
  pre/effects (PDDL/HTN-style): pick a callable tool whose effect fills an unmet goal/sub-goal
  slot; loop. Measures how far *operators alone* determine the workflow (the clean baseline).
- **L1 — LLM planner, operators in context** (no training): prompt = goal + operator list +
  current slots → next tool. Tests whether a frontier model plans correctly given only the
  affordances (the "agentic" zero-shot bar).
- **L2 — learned ABox-conditioned planner** (★the contribution, EXPERIMENT_DESIGN §15.13):
  train across many domains' traces to map (goal, operators, state) → next tool, with the
  **operators injected as swappable memory** (cross-attention / per-domain module), *not*
  baked into weights. The weights learn the **general planning skill** (TBox); the ABox memory
  is replaced per domain → **transfer with no retraining**. This is exactly "user gives a goal,
  the system auto-selects tools," generalized.

(The killed tau2 TBox-only adapter was a degenerate P1 phase-planner on an obsolete taxonomy.
The real target is L2: a planner conditioned on operator relations, not on a fixed step vocab.)

**Planner operating loop (adopted from GoalAct, §5.1).** L1/L2 do **not** plan greedily one
tool at a time. They maintain a **global plan** `G = [(subgoal_i, skill/operator_i), …, Finish]`
anchored to the goal, and **re-plan after every action** from the execution history
`G_t = π(goal | operators | history_t)` (history = ⟨plan, action, observation⟩ tuples). Two
reasons this matters for us: (a) the persistent global goal **prevents the local-branch /
read-loop sticking** we diagnosed in tau2 (a greedy next-tool policy is exactly what looped);
(b) re-grounding the plan against the **operator preconditions/effects** keeps it
**executable** (no plan steps outside the action space). Plan at the **skill/operator level**,
not micro-steps; each operator then resolves its own concrete call (provided tool / wrapped).

**Recommendation**: L0 first (does the operator ABox alone suffice?), then L1 (zero-shot LLM
bar), then L2 (the learned transferable planner). The gap L0→L1→L2 is the result.

### 5.1 Prior art reflected (searched 2026-05-31)
Our design is positioned in, and borrows from, the LLM-agent planning literature:

- **GoalAct** (arXiv 2504.16563, *Global Planning + Hierarchical Execution*, NCIIP'25 best
  paper): continuously-updated **global plan** of high-level **skills** + execution feedback.
  We adopt its global-plan + re-plan loop and skill-level abstraction (§5.1 above). **Our
  addition**: GoalAct grounds plans only via LLM reasoning over tool descriptions; we add
  **structured operators** (precondition/produces/effect = a checkable action space) so
  executability is *guaranteed*, not hoped — and so the planning skill (TBox) is **learned +
  ABox-swappable**, not prompt-only.
- **Planning survey** (arXiv 2402.02716): taxonomy = Task-Decomposition / Plan-Selection /
  **External-Module** / Reflection / Memory. We are **External-Module + Memory + Reflection**:
  a structured operator module + a slot store + continuous re-plan. (We are *not* pure
  decomposition-by-prompt.)
- **LLM+P / classical planning**: ABox operators = a **PDDL/HTN Domain** (pre/eff); the goal =
  the `output` contract. **L0 = an external symbolic planner** over that domain (LLM+P style);
  **L2 = a neural planner** conditioned on the same operators-as-memory.
- **Plan-and-Execute / Plan-then-Execute** (LangChain; arXiv 2509.08646 security): validates the
  **planner(TBox)/executor split** — a structured, machine-readable plan (our call graph),
  big model for planning, cheap/deterministic executor. We inherit modularity/debuggability and
  the plan-then-execute safety posture (no tool runs outside a vetted plan/operator set).
- **Agent-harness patterns** (harness-engineering, OpenAI/LangChain deepagents): the executor
  is a **stateless orchestration harness** (loop, tool routing, slot memory, tracing, recovery).
  We adopt two patterns explicitly: **Reasoning Sandwich** — strong model for plan+verify,
  deterministic/cheap for intermediate steps (= our determinism split, coverage% measures it);
  **goal re-injection** — keep the global goal in context every step (= GoalAct's global plan),
  the structural cure for premature-exit / non-convergence.

### 5.2 Plan↔execute spectrum — where we sit, and what else we borrow
**Spectrum of plan/execute coupling** (informs our two modes):
`ReAct (re-reason every step, greedy)` ↔ `GoalAct (global plan + re-plan on feedback)` ↔
`ReWOO / LLMCompiler (full plan up front, then execute)`.
- **ReWOO** (arXiv 2305.18323): Planner→Worker→Solver; plan *all* interdependent tool calls up
  front (blueprint with `#E` variable-passing), workers fetch in parallel, 2 LLM calls total
  (vs ReAct's per-step) → 50–70% latency cut. = our **procedure-given executor** (the call graph
  is the blueprint; `arg`/`produces` = `#E` variable passing). 
- **LLMCompiler** (arXiv 2312.04511): plan = a **task DAG with dependencies**, execute
  independent tasks **in parallel**. = our call graph + `parallel_block`; adopt DAG scheduling.
- **Our position**: goal-only mode = GoalAct point (global plan + re-ground on operator
  effects) — between ReAct's loops (which caused our tau2 read-loop) and ReWOO's brittle
  full-upfront plan (GoalAct's "non-executable" critique). Procedure-given mode = the ReWOO/
  LLMCompiler upfront-DAG special case.
- **LATS** (arXiv 2310.04406, ICML'24): MCTS over reason+act+plan with an LM value fn +
  reflection — explores *alternative* operator choices, not one trajectory. Optional **L3**
  rung for ambiguous operator graphs (value-guided search over `next`); heavier, note only.
- **Reflexion / AdaPlanner / Self-Refine**: verbal reflection memory + ± feedback to refine a
  plan. = the re-plan signal in our loop when an operator's `precondition`/effect check fails.

### 5.3 Induce-and-reuse workflows — closest prior, and our delta
- **Agent Workflow Memory** (arXiv 2409.07429, ICML'25): **induces reusable workflows from
  interaction traces** — stored as *workflow skills = trigger + multi-step procedure +
  parameter slots* — and reuses them (offline from train traces or online from test), on
  Mind2Web/WebArena (1000+ tasks, 200+ domains). **This is the closest prior to our
  induction+transfer thesis.** AWM's {trigger, procedure, slots} ≈ our {`scenario_select`,
  `scenario_steps`, `arg`}. **Our delta**: AWM stores **NL workflows reused via prompt**; we
  induce a **structured operator ontology** run by a *deterministic executor* and a *learned,
  ABox-swappable planner* (L2) — verifiable, training-time transfer, not prompt-only. SOP-Bench
  also gives a **ground-truth SOP** to validate the induced workflow (AWM has none).
- **Agentic Plan Caching** (arXiv 2506.14852) + **Learn-When-to-Plan** (arXiv 2509.03581):
  cache structured plan templates; spend LLM planning only when needed. = our **compiled
  per-domain call graph is a plan template** (`scenario_select` = retrieval), and the
  deterministic executor means **LLM is invoked only at uncertain steps** — coverage% is exactly
  "how rarely we must plan with an LLM."
- **Reasoning models + PRM/ORM** (o1/o3/R1; arXiv 2501.09686): L1 may be a reasoning model;
  L2 training can use **process/outcome reward on planning decisions** (reward a correct
  next-operator / penalize off-action-space picks).

---

## 6. Experiment plan — phased (prove transfer cleanly, then scale to autonomy)

> **Scope note (decided 2026-05-31): this is a SEPARATE track from the OISA patent.** The
> patent's claim is *context-dependent tool SELECTION among hundreds–thousands of redundant,
> faceted tools, internalized into weights* → its benchmarks are MetaTool / ToolBench /
> τ²-bench. This document's claim is *learned general PLANNING (TBox) over per-domain operators
> (ABox), with transfer* → benchmarks SOP-Bench then AppWorld. The two tracks share the
> TBox/ABox framing but are evaluated and reported independently.

Strategy: **prove the TBox/ABox transfer mechanism in a clean, ground-truth, single-shot
setting (SOP-Bench) first, then scale to autonomous, procedure-free planning (AppWorld).**
Precise claim: **TBox (the learned general planning/execution skill) transfers; ABox (domain
operators) is *swapped* — freeze TBox, swap ABox → held-out domain works.** (TBox transfers;
ABox is replaced, not "transferred".)

Metrics: **TSR / ECR / C-TSR + Tool Accuracy** + per-phase **deterministic coverage%** (tool/
wrapped vs LLM-backed). Single-shot → no max_steps/read-loop axis.

### Phase 1 — SOP-Bench: quantitatively prove TBox/ABox transfer
- **1a (upper bound + induction validation)**: `compile_sop_ontology(sop.txt)` and
  `induce_ontology(traces)` → call-graph executor (§3). Report TSR ≫ baseline FC/ReAct, and
  **structural agreement of the induced ABox vs the authored `sop.txt`** (ground truth — the
  unique value of SOP-Bench). Establishes the ontology is correct *and* extractable.
- **1b (★the real transfer result)**: the **goal-only L2 learned planner** (§5) — train on
  **N−1 domains**, test the held-out domain with **only its operator ABox swapped in, NO
  retraining**; 12-domain rotation. Plus **ABox-memory ablation** (empty/wrong operators → L2
  must collapse → proves it *reads* the ABox, not memorizes). This is the headline: "the
  general planning skill transfers; only the operators swap."
- **Phase-1 success criteria**: (a) compiled-executor TSR ≫ baseline; (b) induced↔sop.txt
  structural agreement high + TSR ≈ compiled; (c) **L2 held-out TSR ≥ 70% of in-domain with
  zero retraining**; (d) ABox-ablation collapses. (a)/(b) are the *executor/ontology* claim;
  (c)/(d) are the *learned-planner transfer* claim — (c) is the one that matters.
- **Ablations**: L0 (symbolic, operators-only) vs L1 (LLM+operators) vs L2; compile vs induce;
  greedy (ReAct) vs global-plan+re-plan (GoalAct, §5.1) — expect greedy → local sticking.
- **Pilot first**: `customer_service` end-to-end (executor → L0 → L1) before scaling to 12.

### Phase 2 — AppWorld: extend transfer to autonomous, procedure-free planning
Only after Phase-1 (c) holds. AppWorld (457 APIs / 9 apps, goal-only, no SOP, state-based unit
tests): operators = API affordances; **train L2 on N−1 apps, swap the held-out app's API
operators in → autonomous tool selection with no retraining**. Same ABox-ablation control.
This is the full agentic claim (no procedure given anywhere); SOP-Bench Phase-1 having
de-risked the mechanism.

---

## 7. Build order (file changes)

1. `sop_bench_loader.py` (new) — load a SOP-Bench domain dir → (sop.txt, toolspecs,
   tasks rows, ground-truth output columns). Plug our executor in via SOP-Bench's custom
   agent interface (`agents/base.py`, `examples/03_custom_agent.py`).
2. `compile_sop_ontology.py` (new) — parse `sop.txt`: sections→phases, "If <cond>→<action>"
   →branch/precondition_trigger, columns→slots, toolspecs→tool_class/arg_dependency →
   `ontology_<domain>.json` (§2). Plus `induce_ontology.py` (trace-mining route for
   Ours-induced; absorbs `induce_step_realization`+`induce_observation_triggers` logic).
3. `workflow_executor.py` (new) — ExecState + `next_action` (§3), **agent-call only**
   (no user-side path). Replaces `ontology_resolver.py`.
4. eval — wire SOP-Bench CLI (TSR/ToolAcc) + per-phase coverage + domain-swap reporter.
5. (P1 only) phase-planner SFT — regenerate labels to phases from SOP-Bench traces; train.

Self-test each step (compile a domain's SOP → inspect ontology; run executor on 1 task).

---

## 8. Decisions
- **Benchmark** (DECIDED): SOP-Bench (see banner + `EXPERIMENT_DESIGN §16`). tau2 dropped
  as primary; `customer_service` (SOP-Bench) ≈ tau2 telecom offline → controlled contrast.
- **D1** (RE-CENTERED 2026-05-31): the procedure-given deterministic executor (P0) is the
  **oracle/ceiling**, not the deliverable. The contribution is the **goal-only learned planner
  (L2) + ABox-conditioned resolver** and its **transfer** (§9). Build P0 first only to fix the
  ceiling + validate wiring; the headline experiment is §9.
- **D2** (RESOLVED, tau2-only, now moot): tau2 device/probe tools were user-only → drove
  the user-side machinery. SOP-Bench is all-agent-callable, so that machinery is dropped.
- **D3** (open): single `ontology_<domain>.json` vs split files. (Rec: single file.)
- **D5** (new): compile-from-`sop.txt` vs induce-from-traces as the primary ontology
  source. (Rec: build both; compile = clean upper bound, induce = the research claim,
  and their agreement is itself a result.)

---

## 9. The LLM-in-loop experiment (consolidated, AUTHORITATIVE for the experiment)

> Consolidated 2026-05-31 from the former `sopbench/LLM_IN_LOOP_DESIGN.md`. Benchmark =
> **SOP-Bench (Amazon, 2506.08119)**, 12 domains. §1/§2/§5/§6 above supply the schema, the
> TBox/ABox split, and the L0/L1/L2 planner + prior art; this section fixes the *experiment
> center of gravity* and reconciles the "deterministic executor" into its role as the oracle.

### 9.0 The correction (why the executor is not the result)
SOP-Bench-style benchmarks measure **how well an LLM, given a goal and tools, completes the
task *without an external deterministic program solving it for it*.** The 12-domain
deterministic executor (`workflow_executor.py` + `abox/`, mean 97%, 8/12 at 100%) holds the
ground-truth call graph and walks it → it is the **oracle / upper bound**, useful only to
(a) validate the env wiring and (b) mark the ceiling. **The real experiment is LLM-in-loop:**
a *learned, domain-general planner* (TBox) + *external, swappable operator memory* (ABox)
consumed by a resolver, and the headline is **transfer** — freeze the planner, swap the ABox
to a held-out domain, retrain nothing. Label-fit/memorize ⇒ transfer fails; learned general
planning ⇒ transfer holds. Honest-axis: the 97% is reported **as a ceiling only**, and any
no-holdout / label-fit domain (e.g. traffic/content/kyb in the 12) is flagged separately.

### 9.1 The 2-stage loop (planner → resolver → env)
One agent "turn" = (plan abstract step) → (resolve to a concrete tool) → (env executes):
```
goal (output-contract slots) + state (CSV slots) + history
        │
   ┌────▼──────────────────────────────────────────────┐
   │ PLANNER  (TBox — LEARNED, domain-general)           │
   │  sees: goal + ABSTRACT operator affordances         │
   │        (operator name + precondition/effect TYPE,   │
   │         NOT the concrete tool schema), global plan   │
   │        G, execution history; re-plans G every turn   │
   │  emits: next ABSTRACT step (subgoal / operator-class)│
   └────┬──────────────────────────────────────────────┘
        │ abstract step
   ┌────▼──────────────────────────────────────────────┐
   │ RESOLVER (ABox-conditioned)                          │
   │  sees: abstract step + the DOMAIN's concrete operator│
   │        ABox (this tool's schema, arg sources) + state │
   │  emits: concrete tool call with bound args           │
   │  rungs: (a) deterministic rule = baseline/diagnostic  │
   │         (b) ontollm (ABox in prompt)                  │
   │         (c) neural ABox-conditioned (xattn / per-     │
   │             domain memory) = ★novelty (B5*)           │
   └────┬──────────────────────────────────────────────┘
        │ tool call
   ┌────▼──────────────────────────────────────────────┐
   │ ENV (SOP-Bench tool API executes, updates slots)     │
   └──────────────────────────────────────────────────────┘
```
**Why the split carries transfer:** planner weights encode the *general* skill (means-ends over
operator types + GoalAct-style global-plan/re-plan — the structural cure for the tau2
read-loop); the ABox (concrete operators) is *swappable memory* the resolver consumes. Freeze
planner, swap ABox → held-out domain works with zero retraining.
**Critical constraint:** the planner must NOT see concrete tool schemas, or it can memorize
domain specifics and contaminate transfer — enforce the abstract/concrete split at the
prompt/feature boundary.

### 9.2 Comparison arms
1. **LLM-alone (baseline)** — native single-LLM agent (FC / ReAct), full tool context.
   SOPBench (Zekun Li): GPT-5 71–88% / Qwen-7B 5–20%. SOP-Bench (Amazon): FC 27% / ReAct 48%.
2. **LLM+structure, no training** — L1 planner (LLM + operators) → resolver (b); plus **L0**
   (symbolic means-ends, operators only) as the floor. Does the 2-stage structure +
   global-plan re-planning help before any learning?
3. **Ours — L2** — learned ABox-conditioned planner + resolver (c, xattn). Trained on N−1
   domains. Ablation: greedy next-op (ReAct-like) vs global-plan+re-plan (GoalAct).

### 9.3 Metrics
- **Rule pass-rate** (SOPBench/Zekun Li oracle: goal action reached + `constraint_not_violated`
  + `directed_action_graph` conformance, no LLM judge). **Primary.** On the auxiliary SOP-Bench
  (Amazon): TSR / ECR / C-TSR + Tool Accuracy.
- **Refusal accuracy** on `action_should_succeed=false` tasks (constraints unsatisfiable → the
  agent must correctly decline). A hard axis SOPBench (Zekun Li) gives natively.
- **Resolver coverage%** (diagnostic) — fraction of steps resolved deterministically (rung a)
  vs needing the LLM. "How rarely we must invoke an LLM to resolve a step."
- **Transfer Δ (★headline)** — held-out-domain pass-rate / in-domain pass-rate; target ≥ 70%
  with zero retraining. Plus **ABox-ablation** (empty/wrong operators → L2 must collapse →
  proves it *reads* the ABox, not memorizes).
- **Tool-selection@scale** — SOPBench `--tool_list full` (distractors) vs `oracle`; degradation
  = selection among distractors (OISA-patent touchpoint, reported jointly only on this axis).
- **Oracle ceiling** — the deterministic executor's pass-rate, reported as upper bound only;
  label-fit/no-holdout domains flagged.

### 9.4 Transfer protocol (Phase 1b — the result)
**Primary: SOPBench (Zekun Li) 7-domain leave-one-domain-out** (bank/dmv/healthcare/hotel/
library/online_market/university). Train L2 on 6, freeze, swap held-out domain's ABox (its
`actions` specs + dependency routines), zero retrain → rule pass-rate. **Auxiliary: SOP-Bench
(Amazon) 12-domain rotation** → breadth. Success: (a) held-out ≥ 70% of in-domain;
(b) ABox-ablation collapses; (c) L0 < L1 < L2 gap visible; (d) induced ABox ↔ ground truth
(SOPBench `directed_action_graph` / Amazon `sop.txt`) structural agreement high → induction
*validated*. Phase 2 (only after 1b holds): AppWorld (457 APIs / 9 apps, goal-only, no SOP) —
autonomous, procedure-free planning, same ABox-ablation control.

### 9.5 Pilot — bank (SOPBench / Zekun Li), then scale
1. **Fix the ceiling + validate wiring:** run the deterministic executor over bank's
   ground-truth `directed_action_graph` → oracle pass-rate (upper bound).
2. **Baseline (arm 1):** SOPBench native single-LLM assistant on bank (Qwen-7B via local vLLM —
   the weak-model regime where structure has the most headroom; SOPBench reports Qwen-7B
   5–20% / GPT-5 71–88%). `run_simulation.py --domain bank --tool_list oracle ...`.
3. **2-stage assistant (arm 2):** replace the single-LLM `Agent.client` (swarm) with
   planner→resolver; start at **L1 + resolver (b)**, self-test on 1 task (trace plan steps →
   resolved calls → oracle verdict vs that task's `directed_action_graph`).
4. **Compare** baseline vs L1-2stage on bank → structure ≥ parity; add L0; then wire L2
   training data (bank success traces → (goal, operators, state) → next-operator labels).
5. **Scale** to the 7-domain rotation (primary), then the Amazon 12-domain auxiliary surface
   (COWORKER_EXPERIMENT_PLAN matrix).
Harness facts (verified from the clone): `swarm.Swarm` + `Agent(client=OpenAIHandler,
functions=<d>_assistant.py:actions, tool_call_mode=fc)`; ABox = `env/domains/<d>/<d>_assistant.py`;
GT plan = per-task `directed_action_graph`; rule oracle = `env/evaluator.py`; tasks =
`data/<d>_tasks.json` (keyed by goal; `action_should_succeed` = refusal axis); dependency
routines = `task_default_dep_full(domain,{full,required})`. ⚠️ `env/helpers.py` uses `match`
→ Python ≥3.10 (use seka_env 3.12, not system python 3.9). Self-test each step on 1 task.

### 9.6 Open items / honest risks
- **Abstract-vs-concrete operator leakage** (§9.1) — the transfer-validity crux; enforce the split.
- **L2 training data** — mine (goal, operators, state) → next-operator labels from success
  traces (analogous to the telret abstract-step SFT). GPU recipe in INFRA / COWORKER plan B1*.
- **Label-fit / no-holdout domains** — flag separately; the 12-domain rotation is the guard.
- Claim retired: "deterministic beat the LLM." Replaced by: "given structure + a learned
  ABox-swappable planner, the LLM-in-loop gains over baseline **and transfers**."

---

## 10. Step-by-step implementation spec (FOR REVIEW before running)

> Status: **IMPLEMENTED, awaiting design-review → code-review → experiment** (2026-06-01).
> Code lives in `scripts/distill/sopbench/{two_stage_client.py, run_two_stage.py}` (this repo,
> version-controlled + coworker-shared). Deploy = copy both into the SOPBench clone `scripts/`.
> This section documents exactly what was built so the design can be reviewed against §9.

### 10.1 Where it plugs in (verified against the clone)
SOPBench's loop is `swarm.Swarm.run_user_assistant_interaction` → per assistant turn it calls
`get_chat_completion` (swarm/core.py:32), which builds `create_params = {messages, tools,
temperature, top_p, max_tokens, [parallel_tool_calls]}` and calls:
```
agent.client.inference(create_params, debug, mode=mode, tool_call_mode=agent.tool_call_mode)
   -> {"idx": int, "completion": ChatCompletion}
```
So the entire 2-stage policy is injected by giving the assistant `Agent` a **custom client
object that implements `.inference(...)`** with the same return shape. The Swarm loop, tool
execution, DB mutation, and the rule oracle are reused UNCHANGED. (`model_name_huggingface`
attr is also read by core.py for logging → the client exposes it.)

### 10.2 Per-turn algorithm (arm-3 = L1 planner + rung-b resolver)
`TwoStageClient.inference()` does, each turn:
1. **If no tools** in `create_params` → plain chat completion (final natural-language msg).
2. **Slot mining** (`_update_slots`): scan messages; from every `role=="tool"` message parse
   its JSON content into `_slot_state`; from the dummy-user "Here is all the information…"
   message parse the `user_known` JSON block into `_slot_state`. (Accumulates known arg values.)
3. **STEP 1 — Planner** (`_plan`, 1 LLM call, **abstract**):
   - Build operator list = `name: description[:120]` for each tool — **names + descriptions
     ONLY, the concrete param schema is withheld** (the §9.1 transfer-contamination guard).
   - Prompt = goal context (system msg, truncated) + operator list + last-6-turn history
     (CALLED/TOOL_RESULT/USER lines) → "output ONLY the next action name".
   - Parse first token; validate ∈ tool names (else fall back to first tool).
4. **STEP 2 — Resolver** (`_resolve`, rung b):
   - Locate the chosen tool's FULL spec.
   - **Deterministic shortcut**: if every `required` param is already in `_slot_state`, build
     the tool call directly (no LLM) and count it as deterministic coverage.
   - Else **LLM resolver**: `chat.completions.create` constrained to the single chosen tool
     via `tool_choice={"type":"function","function":{"name":action}}`, model fills args
     in-context. (rung b = "ontollm".)
5. Return `{"idx", "completion"}` (real OpenAI object, or a synthetic `ChatCompletion` for the
   deterministic path built via `_make_tool_call_completion`).

`reset()` clears `_slot_state`/turn per task. `coverage()` reports
deterministic / llm_resolved / pct.

### 10.3 Runner (`run_two_stage.py`)
Mirrors run_simulation.py's task loop but self-contained:
- Loads `data/<domain>_tasks.json` (flatten goal→task list, `--num_tasks` head).
- `task_default_dep_full` + `task_initializer` (same calls as run_simulation) → domain_system,
  user_info, assistant_info; `--tool_list oracle` restricts tools to the task's
  `directed_action_graph` nodes, `full` uses all domain tools.
- Builds dummy user (no user_model) with the `user_known` dump as default_response
  (= leaderboard-standard agent-controlled setting).
- Runs `Swarm.run_user_assistant_interaction`, then scores with
  `evaluator_function_directed_graph` (the SAME rule oracle as run_evaluation).
- Saves per-task {task, interaction, evaluation, coverage}; prints pass@1 + coverage%.

### 10.4 Mapping to the §5 / §9 design — what is and isn't done
| design element | this impl (arm-3 L1) | status |
|---|---|---|
| planner = LLM over abstract operators (§5 L1) | `_plan`, names+desc only | ✅ |
| global-plan + re-plan each turn (GoalAct §5.1) | re-decided per turn from history | ⚠️ partial — re-decides per turn but does NOT keep an explicit persistent plan `G`; greedy-ish. Review: is per-turn re-decision enough, or add explicit G? |
| abstract/concrete split (§9.1 guard) | planner sees no param schema | ✅ (review the truncation/leak via descriptions) |
| resolver rung (a) deterministic | slot-state shortcut | ✅ |
| resolver rung (b) ontollm | tool_choice-forced LLM fill | ✅ |
| resolver rung (c) neural xattn (§5 L2) | — | ✗ later (coworker B5*) |
| L0 symbolic planner (§5 L0) | — | ✗ TODO (arm-2) |
| L2 learned planner + transfer (§9.4) | — | ✗ later (coworker B1*/Exp-4) |

### 10.5 KNOWN ISSUES / review points (must resolve before trusting results)
> ★ **Self-review (2026-06-01) found a BLOCKING bug**: `run_two_stage.py`'s inline eval passes
> `func_calls` as tuples but `evaluator_function_directed_graph` expects dicts
> `{"tool_name","arguments","content"}` paired from `interaction[i]↔[i+1]`; saved format also
> differs from run_simulation (key `interaction`, role-stripped, `database` placement).
> **Recommended fix: drop run_two_stage inline eval → add `--two_stage` flag to
> `run_simulation.py` (swap only the assistant client) + reuse standard `run_evaluation.py`.**
> Full self-review = handoff `project_handoff_2026_06_01.md §4.5`.
1. **Import (smoke hit this)**: deploy as same-dir import. `run_two_stage.py` now does
   `sys.path.insert(0, <file dir>)` + `from two_stage_client import TwoStageClient`; run from
   the clone root with both files in `scripts/`. (Original `from scripts.two_stage_client`
   failed: scripts/ not a package.)
2. **Synthetic ChatCompletion** (`_make_tool_call_completion`): must match what
   swarm/core.py:handle_tool_calls expects (`completion.choices[0].message.tool_calls[*]
   .function.name/.arguments`, `tool_call.id`). Verify field-for-field in code review — a
   mismatch silently drops the deterministic-path calls.
3. **Slot mining heuristic**: `_update_slots` assumes tool results are JSON dicts and the
   user_known dump starts with a fixed string. Real SOPBench tool returns may be bare bools/
   strings → review coverage of the miner; a wrong miner inflates/deflates deterministic %.
4. **Planner validation**: on an invalid action name it falls back to "first tool" — crude;
   could bias. Review whether to re-prompt or use a smarter default.
5. **Two LLM calls/turn** (plan + resolve) → ~2× tokens/latency vs arm-1. Acceptable for the
   pilot; note in cost accounting.
6. **No explicit termination policy**: relies on the model calling `exit_conversation`. Weak
   7B may loop to max_turns. Review whether the planner should be allowed to choose exit.
7. **Coverage semantics**: deterministic = "all required args in slot state" — this is rung-a
   opportunism inside arm-3, NOT the §9 oracle. Keep labelled as diagnostic.

### 10.6 Test plan (after code review)
- Smoke: `--domain bank --tool_list full --num_tasks 5 --model Qwen/Qwen2.5-7B-Instruct`
  (vLLM 9100). Inspect 1 trajectory: planner picks → resolver args → oracle verdict vs the
  task's `directed_action_graph`.
- Then bank full N=134: compare arm-3 vs arm-1 baseline (react/full 5.2%, fc/full 3.7%).
  Success gate = **arm-3 full > arm-1 full by ≥5%p on bank** → scale to 7 domains + 14B.
- Report into `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` (Exp-3 row).
