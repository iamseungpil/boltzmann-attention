# Full Workflow Ontology — design (2026-05-30)

Status: **DESIGN.** Supersedes the partial "fix-disambiguation only" ontology
(`step_realization_*` + `obs_triggers_*`) and the per-turn reactive resolver
(`ontology_resolver.py` + `two_stage_agent.py:_two_stage_generate`).

> ### ★ BENCHMARK = SOP-Bench (confirmed 2026-05-30 밤)
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

## 1. Architectural inversion

| | Old (per-turn rewriter) | New (workflow executor) |
|---|---|---|
| Who owns control flow | Planner LLM | **Deterministic executor (ontology state machine)** |
| Resolver scope | swap 1 tool/turn | drive the full phase loop; choose next action |
| LLM role | plan every turn + emit concrete call | **perception (NL→slots) + uncovered-branch fallback only** |
| Transition (gather→fix) | impossible (planner re-plans) | explicit `phase_exit_predicate` |
| Read-loop | unbreakable | impossible (read-dedup + monotone gather progress) |
| Diagnostics | freeform NL → max_steps | executor emits device tool-calls / NL templates deterministically |

The executor becomes a **neuro-symbolic workflow engine**: symbolic where the domain is
a state machine (telecom), LLM only for (a) parsing user free-text into state slots and
(b) genuinely uncovered/ambiguous branches. The TBox is the domain-general control
skeleton (transfers); the ABox is the per-domain bindings (swaps).

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

## 5. Planner & SFT taxonomy under the new design

Because the executor owns control, the planner's role shrinks. Two options to evaluate:

- **(P0) No planner** — pure deterministic executor + LLM perception/fallback. The
  cleanest test of "how far does the workflow ontology go alone." Cheapest.
- **(P1) Phase-planner** — a tiny LLM emits only the PHASE
  (`identify/gather/diagnose/fix/verify/close`) as `Plan: <phase>`; executor does the
  rest. SFT taxonomy collapses from 5 ad-hoc steps → 6 principled phases. Retrain target.

This is why the old TBox-only training was killed: its step taxonomy
(`gather_account_context`/`apply_policy_action`/…) and its assumption that the resolver
only fills fix-writes are both obsolete. If we pursue P1, regenerate SFT with
`build_tbox_sft.py` retargeted to phase labels; if P0, no planner training at all.

**Recommendation**: implement P0 first (fastest signal on the executor), add P1 only if
P0 shows the LLM is needed for phase sequencing.

---

## 6. Experiment reconstruction (SOP-Bench)

Benchmark = **SOP-Bench** (agent-controlled, single-shot, state-based). Metrics (every run):
**TSR / ECR / C-TSR + Tool Accuracy** (SOP-Bench CLI) **+ per-PHASE deterministic coverage**
(fraction of agent tool-calls chosen by the ontology vs LLM-fallback). No tau2 max_steps/
read-loop axis (single-shot removes it).

Runs:
1. **Baseline** — SOP-Bench's own FC / ReAct agent given `sop.txt` as text (~55-64% paper).
2. **Ours-P0** (★primary) — `compile_sop_ontology(sop.txt)` → deterministic
   `workflow_executor`; LLM fallback only on genuinely ambiguous/generative steps. Per
   domain TSR + coverage. Expect TSR ≫ baseline on branch-heavy SOPs.
3. **Ours-induced** — ontology **induced from teacher traces** (not reading sop.txt) →
   (a) TSR, (b) **structural agreement vs the authored SOP** (validates auto-induction,
   §15.14; impossible in tau2).
4. **Transfer (LODO)** — TBox (phase + relation types) fixed, ABox (per-domain slots/
   tools/branches) swapped → held-out domain; 12-domain rotation.
5. **Ablations** — executor w/o `diagnostic_next` (→LLM), w/o `phase_exit`; P0 vs P1;
   compile-from-SOP vs induce-from-traces.
6. **Pilot first** — `customer_service` end-to-end before scaling to all domains.

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
- **D1** (open): P0-first (no planner; SOP→ontology→executor) vs P1 phase-planner. (Rec:
  P0 first — SOP is given, so the planner adds least value initially.)
- **D2** (RESOLVED, tau2-only, now moot): tau2 device/probe tools were user-only → drove
  the user-side machinery. SOP-Bench is all-agent-callable, so that machinery is dropped.
- **D3** (open): single `ontology_<domain>.json` vs split files. (Rec: single file.)
- **D5** (new): compile-from-`sop.txt` vs induce-from-traces as the primary ontology
  source. (Rec: build both; compile = clean upper bound, induce = the research claim,
  and their agreement is itself a result.)
