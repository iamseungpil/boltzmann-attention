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
> ✅ **RESOLVED (2026-06-01, Track A)** — the BLOCKING eval-format bug is fixed by the
> recommended route: `apply_two_stage_patch.py` adds `--two_stage`/`--two_stage_det` to the
> author's `run_simulation.py` (swap ONLY the assistant client → identical save schema) and
> the standard `run_evaluation.py` scores it UNCHANGED. Verified end-to-end on bank/full/5
> (eval ran, produced per-task verdicts). `run_two_stage.py` is now DEPRECATED (banner added).
>
> **Independent re-review (2026-06-01, this session) — verified against the live clone source:**
> - **Synthetic ChatCompletion OK** (issue #2 closed): on openai 2.38.0 the synth object's
>   `model_dump_json()` yields `tool_calls[0].function.name/.arguments` + `.id`, matching BOTH
>   core.py:handle_tool_calls AND run_evaluation.py L207 (`interaction[i]["tool_calls"][0]
>   ["function"]["arguments"]`). Tested in isolation before any run.
> - **NEW BUG the smoke surfaced** (never hit before because arm-3 had never run): `swarm/types.py`
>   typed `Agent.client: Union[OpenAIHandler, None]` → pydantic REJECTS a TwoStageClient. Fixed
>   by relaxing to `Optional[object]` (patch #3). Confirms arm-3 was genuinely un-executed.
> - **User protocol now leaderboard-exact**: run_simulation's no-user-model path opens with
>   `task["user_prompt"]` then a dummy user repeats the `user_known` dump — the author standard.
>   (The old run_two_stage opened with the user_known dump itself = non-standard. Another reason
>   the --two_stage route is more faithful.)
> - **Deterministic shortcut made opt-in** (default OFF, `--two_stage_det` to enable): arm-3 now
>   measures CLEAN L1 (planner + LLM resolver); would-be coverage is still counted as a diagnostic.
>   Rationale: the rung-(a) shortcut emits slot-state args with NO type/semantic check, which can
>   produce wrong calls and contaminate pass@1 (issue #3/#7). Keep it as a separate condition.
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

---

## 11. ★ TBox/ABox 완전 분리 학습 설계 — 19-domain, entanglement-free (2026-06-01, 사용자 지시)

> **사용자 지시 (3회 반복, 확정)**: (1) 19개 도메인 모두에서 유효한 **TBox 온톨로지 관계를 전부 도출**하고
> 이를 학습하는 TBox 설계. (2) TBox는 **도메인 특화되지 않은 "의존성 그래프 룰 자체"**만 학습. (3) 실제
> 도메인별 **ABox를 완전히 분리**, 19-도메인 실험에서 **ABox만 교체, TBox는 동결**. (4) **저번처럼 TBox+ABox
> 혼재 SFT 금지.** 결정(2026-06-01): **Phase 1 = Zekun 7 LODO 즉시 + Amazon 12 harness 병행**; 아키텍처 =
> **in-context copy-grounded SFT 먼저 → xattn novelty**.

### 11.0 저번 entanglement 실패 (반복 금지 대상)
`build_abstract_sft.py`가 action turn에 `Plan: <step>` **prefix만 추가하고 구체 tool call을 supervised
target에 그대로 유지** → 가중치가 fault→fix(도메인 ABox)를 암기 = **TBox+ABox entangled** (turn correctness
87%였지만 외부 resolver와 충돌, 전이 시 ABox만 실패). 수정 시도(`build_tbox_sft.py` content→`Plan:` 치환 +
`--mask-toolcalls`)도 **step vocab 자체가 도메인 특화면 여전히 누수.** ⇒ 진짜 분리의 필요조건은 아래 §11.3.

### 11.1 분리 계약 (the separation contract — 이 실험의 불변식)
- **TBox** = 도메인-불변. = {8 관계 타입(§2.2) + 엔티티 문법(Function/Slot/Predicate/Scenario) +
  **means-ends 선택 정책**}. **한 번 학습, 테스트 시 동결.**
- **ABox** = 도메인별 call-graph 인스턴스 = `ontology_<domain>.json` (§2.5). **유일하게 교체되는 것.**
- **하드 제약**: planner 출력 = `f(goal, slot_state, 제공된 ABox)`의 **순수 함수**. target은 **제공된 ABox에
  grounded(copy/pointer)**, 가중치는 **도메인 특화 이름을 절대 생성하지 않음**. ⇒ 가중치엔 룰만, 내용은 ABox에.

### 11.2 19-도메인 공통 TBox 관계 집합 (도출 완료 — 8 관계가 19를 span)
8 관계(§2.2)는 **Amazon 12 도메인 서베이에서 도출**(§2.0)되었고 **Zekun 7도 커버**. 두 벤치 native 포맷 → 8 관계 매핑:

| 8 relation (TBox type) | Zekun-SOPBench(7) 소스 | Amazon-SOP-Bench(12) 소스 |
|---|---|---|
| `realizes`(step→fn) | `directed_action_graph` 노드 | sop.txt 번호 step의 도구 |
| `precondition`(fn→pred) | `constraints`(should-call-before) | "If <cond>" 가드 |
| `produces`(fn→slots) | 도구 반환 필드 | step "Output:" / output_columns |
| `arg`(param←slot) | 도구 inputSchema↔user_known/prior | inputSchema↔column "save for step N" |
| `next`(step→step\|forks) | dirgraph 엣지 | 섹션 순서 + "If→step" |
| `scenario_select`/`_steps` | task `user_goal` 유형 | "intent=a\|b" 서브절차 |
| `terminate`(pred→outcome) | `action_should_succeed`(거부=no-op) | "no further action→default"/escalate |
| `output`(required,format) | GT action + db state | §6 Output keys + json/xml |

**⇒ TBox 관계 집합 = 이 8개. 추가 도출 불요(19 공통).** 도메인 간 차이는 전부 **ABox 인스턴스 값**(어느 fn, 어느
pred, 어느 슬롯)일 뿐 — 관계 *타입*은 불변. 이것이 "TBox 동결, ABox swap"이 성립하는 구조적 근거.

### 11.3 학습되는 TBox = means-ends 선택 정책 (도메인-불변 룰) [v2 — 리뷰 A1–A4 반영]
- **goal_slots := `output.required`** (리뷰 A2: 별도 `achieves` 관계 폐기 → 8관계 닫힘 유지). "operator가
  goal을 achieve" ⟺ **`produces(fn) ∩ goal_slots ≠ ∅`**. (induce도 8관계만.)
- **입력**: goal_slots + **ABox operator affordances**(operator명 + `precondition` pred + `produces` slots —
  **관계적 affordance만; 구체 param schema 제외** =§9.1 전이가드, resolver 몫) + 현재 `slot_state` + history(`G`).
- **출력**: 다음 operator 선택(**제공된 ABox로의 pointer**) 또는 `terminate(done|refuse)`.
- **룰 = 진짜 means-ends + goal-stack 후방 회귀 (리뷰 A1, 구 greedy-forward 수정)**:
  1. goal stack ← 미충족 goal_slots.
  2. top `g` pop: `g`를 `produces`하는 operator X 후보.
     - X의 `precondition`이 현 slot_state로 충족 ∧ X 미실행 → **X 선택**(출력).
     - 아니면 X의 **미충족 precondition 슬롯을 subgoal로 push**(재귀) → 그 subgoal을 produces하는 operator로 하강.
  3. **gate**: precondition 미충족 operator는 절대 출력 안 함.
  4. **종료**: stack 비면 `terminate(done)`. **refuse (리뷰 A3)**: stack 비지 않았는데 적용가능 operator도,
     그 subgoal을 produces하는 operator도 없음 → `terminate(refuse)` (= `action_should_succeed=false` 정답, §9.3 hard축).
     **★검증완(N1, 코드리뷰 §7)**: 평가자 `action_called_correctly=(should_succeed==action_successfully_called)`
     → refuse = **타깃 미호출 후 정지**로 성립(전용 refuse operator 불요). arm-3 강제 tool_choice가 거부 태스크
     66%에서 금지 액션 호출=실패=N1. ⇒ L0/arm-3v2 모두 refuse=no-call 턴.
  5. **tie-break (리뷰 A4 — L0 결정론 필수, gap 주장 보호)**: 후보 복수면 **`next` 토폴로지 순 → 잔여 미충족
     precondition 수 최소 → operator명 사전순** 고정. (L1/L2는 LLM/가중치가 끊고, L0는 이 순서로 재현.)
- **19 도메인 불변** — 도메인은 ABox(operator·pred·slot)로만 들어옴. ★구 greedy-forward는 *subgoal slot만*
  produces하는 중간 operator를 영영 못 골라(precondition 체인서 정지) **arm-3 naive 0%(제약위반 90%)의 유력
  근본원인**(리뷰 A1) → 후방 회귀가 직접 처방. L0에서 이 룰을 결정론 구현, L1/L2는 동일 구조를 컨텍스트/가중치로 근사.

### 11.4 Entanglement-free 학습 (핵심 — 저번 실패의 직접 수정)
1. **매 예제 ABox를 planner 컨텍스트에 직렬화**(해당 도메인 operator affordance). planner는 이걸 **읽어서** 고름.
2. **target = 다음 operator명, 단 그 이름은 컨텍스트 ABox에 verbatim 존재(copy)**. loss는 **operator-선택
   span만** supervise, reasoning·구체 call 내부는 **mask(−100)**. (저번 실패=구체 call을 target에 유지한 것.)
3. **cross-domain 배치**: 각 예제가 자기 도메인 ABox를 운반 → 단일 도메인 operator 집합이 안정적 가중치
   shortcut이 못 됨. (LODO: held-out 제외 6 도메인 trace로 학습.)
4. **anti-memorization (alias는 "(선택)"이 아니라 필수 = 리뷰 B1)**: 예제마다 operator **순서 셔플 +
   operator명 alias 치환을 per-epoch 랜덤화**(같은 trace가 epoch마다 다른 alias). 셔플은 **위치** 암기만 막고,
   alias는 **어휘** 암기(verify_*/check_* 이름 공기만으로 "goal transfer→먼저 verify_identity"를 학습=ABox가
   가중치 누수=§11.0 그 실패)를 막음 → 정책이 **관계 구조(precondition match)로만** 선택. **copy-grounding은 출력
   pointer만 보장, 입력 reading은 보장 못 함(리뷰 B2)** → 충분조건 = **alias 필수 + (ii)(iii) 붕괴를 큰 효과크기로 입증.**
5. **데이터 소스**: GT `directed_action_graph` walk(=oracle SUCCESS 궤적)을 step별로
   `(slot_state_t, ABox, goal) → 다음 operator(라벨)`로 **재라벨**. 궤적이 정답 순서를 주고, 우리는 각 step을
   "제공된 ABox 위에서의 선택"으로 변환. ⇒ tool-call 토큰은 학습 신호에서 빠짐.

### 11.5 ABox 표현 + swap 메커니즘
- `ontology_<domain>.json`(8 관계, §2.5). 추론 시 held-out 도메인 ABox를 (A)컨텍스트 직렬화 또는 (B)메모리
  인코딩. **TBox 가중치 불변.** 19 도메인 = 19 ABox 파일, 정책 코드/가중치는 1벌.

### 11.6 사다리 (모든 rung이 분리 계약 만족) — arm 매핑
| rung | 분리 방식 | arm | 상태 |
|---|---|---|---|
| **L0 symbolic** | 룰 hand-code, 가중치 0, 순수 ABox = **완벽 분리** | arm-2 | TODO |
| **L1 in-context (no train)** | frozen LLM이 프롬프트 ABox 읽고 선택 = 프롬프트 분리 | arm-3v2 | 설계(아래) |
| **L2a in-context copy-grounded SFT** | 선택 정책 cross-domain 학습, copy-target, ablation 증명 | **arm-4a ★이 실험** | 설계완 |
| **L2b xattn ABox-memory** | ABox=swap 메모리뱅크, 가중치 물리적 content-free (§15.13) | arm-4b(novelty) | 후속 |

> **arm-3-naive(현 0%)는 ABox 의존성 그래프를 안 줌** → arm-3v2 = **planner 컨텍스트에 ABox(precondition/
> produces) 주입 + gate + exit 허용**(무학습 L1). arm-4a = 그 선택을 cross-domain copy-grounded SFT로 학습.

### 11.7 분리 증명 ablation (반드시 통과 — "가중치엔 룰만" 입증) [리뷰 B3–B5 반영]
- **(i) ABox-swap LODO** (전이): 6 도메인 학습 → held-out ABox swap, **재학습 0** → pass-rate ≥ in-domain의 70%.
- **(ii) Empty ABox**: operator 제거 → planner **붕괴**(선택 불가). ⇒ ABox를 실제로 읽음.
- **(iii) Wrong-domain ABox**: A task에 B의 ABox → 붕괴/ B operator 선택. ⇒ 도메인 암기 아님.
- **(iv) Operator-shuffle 불변**: 순서 셔플해도 선택 동일. ⇒ 위치 아닌 구조로 선택.
- **(v) Alias 불변 (리뷰 B4)**: operator명을 무작위 alias로 치환해도 선택 동일. ⇒ 어휘 아닌 관계로 선택.
  (alias 없이 통과한 LODO는 어휘 전이 가능성으로 해석 모호 — (v)가 그걸 봉쇄.)
- **(vi) Slot명 alias (P2 스트레치, 리뷰 B5)**: operator명+slot명 둘 다 alias(관계 그래프만 남김) → 가장 깨끗한
  분리 증명. Phase 1엔 과할 수 있음.
- **★붕괴 임계 사전등록 (리뷰 B3, 눈대중 금지)**: "붕괴" ≝ `wrong-ABox pass ≤ 1.2× empty-ABox` **AND**
  `≤ 0.3× correct-ABox`. (ii)(iii)이 음성대조의 핵심 — entangled 모델은 ABox 없이도 동작(=이 임계 위반=실패).
  분리 모델은 ABox 없으면 못 함(=임계 통과=성공조건).

### 11.8 Phasing (사용자 결정: 7 먼저 + Amazon 병행)
- **Phase 1 (즉시, Track A)** — ★실행 순서(리뷰 C-1/C-2): `induce → (induced↔GT dirgraph 대조 = **gate**) →
  **L0/arm-2**(GT ABox로 means-ends 룰 검증, LLM 無 = 가장 싼 (a) 반증/검증) → arm-3v2(무학습 L1) →
  arm-4a(L2 cross-domain copy-grounded SFT, 7 LODO 6→1×7)`. ablation (i)-(vi).
  비교 분해: Δ(naive→3v2)="ABox in-context+gate" / Δ(3v2→4a)="학습된 정책". **★L0서 greedy-vs-regression 둘 다
  돌려** "greedy X% 실패 → regression 해결"을 GPU 없이 데이터로 입증(미결결정1 해소, §11.11).
- **Phase 1b (병행)**: Amazon 12-도메인 harness(loader/executor/eval) 구축 → `ontology_<dom>.json` 12개 induce
  → 19-도메인 ABox 풀 완성. (customer_service 파일럿 자산 `abox/`·`workflow_executor.py` 확장.)
- **Phase 2**: **19-도메인 통합 TBox**(union 학습, 19 LODO) + **L2b xattn** novelty(§15.13).

### 11.9 Build order / 파일 [리뷰 반영 순서]
1. **`induce_ontology_zekun.py`** (신규): `directed_action_graph`+`constraints` → `ontology_<dom>.json`(8 관계) ×7.
2. **induce↔GT 대조 gate (리뷰 C-1, §9.4d)**: induced call-graph vs GT `directed_action_graph` 일치율 검증.
   **통과해야** 다음 진행(arm-3v2 저조를 "룰 오류 vs induce 오류"로 혼동 방지).
3. **`l0_planner.py`** (신규, arm-2): §11.3 means-ends 룰 결정론 구현. **greedy-forward 변형도 같이**(ablation:
   greedy 실패 vs regression 해결 = (a) 데이터 증명). GT ABox 위 LLM 無.
4. **`two_stage_client.py` planner 확장**: ABox-in-context 입력 + **copy-grounded 디코딩**(제공 operator명 제약,
   gate=precondition 미충족 금지, exit/refuse 허용). = arm-3v2(무학습)·arm-4a(학습) 공용 경로.
5. **`build_tbox_planner_sft.py`** (신규, 구 build_abstract_sft 대체): oracle 궤적 → `(goal, ABox-직렬화,
   slot_state, history) → 다음-operator copy-target`, **선택 span만 supervise**, **operator 순서 셔플 + per-epoch
   alias 필수**(리뷰 B1).
6. LODO 러너 + ablation harness (i)-(vi) + 붕괴 임계 자동판정(리뷰 B3).
- 학습 trainer: `lora_train_chat_toolcall.py --mask-toolcalls` 재사용 — target이 copy operator명이고 그 외 전부
  mask인지 데이터 단에서 보장(§11.4-2).

### 11.10 왜 이게 헤드라인인가 (thesis)
"사용자가 목표만 주면 시스템이 도구를 자동 선택"을, **계획 능력(TBox)은 학습으로 일반화하고 도메인 지식(ABox)은
교체 가능한 데이터로 분리**해 달성. ablation (ii)(iii)이 "가중치가 룰만 들고 ABox를 실제로 읽는다"를 증명하고,
(i) LODO가 "재학습 0 전이"를 증명. AWM(NL workflow 프롬프트 재사용, §5.3) 대비 **구조적·검증가능·학습시 전이**가 델타.

### 11.11 설계리뷰 반영 (2026-06-01, `DESIGN_REVIEW_s11_tbox_abox_2026_06_01.md`)
리뷰 판정: 골격(copy-target SFT → LODO → empty/wrong-ABox 음성대조) 건전, **(c) 순서 유지**, **(a)/(b) 정식화 보강**.
전 항목 수용·반영:
- **(a) §11.3 [수정완]**: A1 후방 subgoal 회귀(greedy→means-ends, 0%의 유력 근본원인) · A2 `achieves`→
  `produces∩output.required`(8관계 닫힘) · A3 `refuse` 종단(거부 정확도) · A4 L0 결정론 tie-break.
- **(b) §11.4/11.7 [수정완]**: B1 alias **필수+per-epoch**(어휘 누수 차단) · B2 copy=필요조건일뿐(alias로 충분화)
  · B3 붕괴 임계 사전등록(`wrong ≤1.2×empty AND ≤0.3×correct`) · B4 ablation **(v) alias-불변** · B5 (vi) slot명 alias(P2).
- **(c) §11.8/11.9 [수정완]**: 순서 유지 + C-1 induce↔GT 대조를 **명시 gate** · C-2 **L0를 arm-3v2 앞으로**.

**미결 결정 해소:**
1. **(P0 먼저 vs greedy 먼저)** → **L0-우선으로 해소.** L0(LLM·학습 無)에서 **regression 정식 구현 + greedy 변형**을
   둘 다 GT ABox로 돌려 "greedy 실패율 vs regression 해결"을 **싸게 데이터로 입증** → "고치고 가기"와 "데이터 먼저"를
   동시 달성. greedy-L0가 bank를 의미있게 풀면 회귀 불요 판정도 가능(반증 경로 보존).
2. **(achieves)** → **(ii) `produces ∩ output.required` 채택** (새 관계 불필요, 8관계 닫힘).
