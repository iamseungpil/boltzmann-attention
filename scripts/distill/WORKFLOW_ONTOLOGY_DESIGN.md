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

## 2. SOP Ontology — rigorous schema (grounded in all 12 SOP-Bench domains)

### 2.0 Empirical basis + design principle
Surveyed **all 12 SOP-Bench domains** (customer_service, content_flagging, dangerous_goods,
know_your_business, patient_intake, aircraft_inspection, email_intent, order_fulfillment,
traffic_spoofing_detection, warehouse_package_inspection, referral_abuse_detection,
video_classification/annotation). SOPs span a spectrum:
**tool-orchestration** (order_fulfillment: 4-step tool chain; video_annotation: 26-tool
pipeline) ↔ **decision/computation** (content_flagging/traffic/referral: scoring →
classification, no external tool) ↔ **hybrid troubleshooting** (customer_service).

**Key finding that forces the redesign**: a large fraction of SOP "actions" are **internal
COMPUTE/DECIDE** (weighted sums, count-of-TRUE indicators, formulas, threshold-banding,
discretization, decision tables) — *not* tool calls. **Group J and Routine R1-R4 model
only tool-orchestration**; they have no decision/computation layer. The old §2.1 schema
(6 telecom-flavored relations: tool_class/arg_dependency/precondition_trigger/diagnostic_
next/phase_exit/verify) covers ≈ the orchestration slice only.

**Principle.** An SOP = ⟨control-flow graph⟩ × ⟨decision logic⟩ × ⟨data-flow + computation⟩
× ⟨obligations⟩. We formalize it as a typed relation set grounded in **BPMN** (control-flow
patterns), **DMN** (decision tables / FEEL predicates), and dataflow provenance. Every
relation is a **TYPE** (TBox, domain-invariant) with per-domain **INSTANCES** (ABox,
swappable), and every relation is both **compilable** from `sop.txt` (a linguistic pattern)
and **inducible** from traces + the column dependency graph.

### 2.1 Entities (typed vocabulary)
- **Phase**, tagged `phase_role ∈ {INGEST_VALIDATE, GATHER, ANALYZE_COMPUTE, DECIDE_CLASSIFY,
  ACT, VERIFY, REPORT_CLOSE, ESCALATE}` (universal skeleton; generalizes the old
  telecom IDENTIFY→GATHER→DIAGNOSE→FIX→VERIFY→CLOSE).
- **Action**, `action_class ∈ {READ (fetch external state), PROBE (reveal state),
  COMPUTE (derive a slot from slots, no tool), DECIDE (predicate→categorical value),
  WRITE_EFFECT (mutate external system), ROUTE (assign outcome/target), LOG (record
  obligation), TERMINATE (end with outcome)}`.
- **Slot** (state variable), `slot_role ∈ {INPUT, FETCHED, COMPUTED, DECISION, OUTPUT,
  CONTROL}`, `dtype ∈ {bool, categorical, numeric, id, text, set}`. (SOP-Bench: slots = CSV
  columns; OUTPUT = `metadata.output_columns`.)
- **Predicate** grammar: *atomic* `slot OP v`, `OP ∈ {=,≠,<,≤,>,≥, ∈[a,b], matches, is_missing}`;
  *composite* `∧ ∨ ¬`; *count* `Σ 1[indicator_i]` (referral violation score).
- **Scenario** (task/sub-procedure type), **Outcome** (terminal output value).

### 2.2 Relation families (TBox relation TYPES) — 5 families, 31 relations
Each: `signature` — definition `[grounding domain]` `{compile-pattern // induce-pattern}`.

**A. CONTROL-FLOW** (procedure skeleton — BPMN patterns)
- `phase_sequence(p_i → p_j)` — default next phase `[all]` `{section order // modal step transitions}`
- `precedes(a → b)` / `requires(action, precond_pred)` — ordering/gate (auth before all) `[customer, aircraft]`
- `branch(point, [(pred_i → target_i)], default)` — XOR exclusive-choice **+else** (extends R3) `[all]` `{"If <cond> → <step>" // step-fork by prior state}`
- `parallel_block(phase → [sub_phases])` — AND-split: independent sub-procedures `[patient_intake 6 outputs, aircraft 7, traffic risk+action]`
- `guard_exit(pred → terminal_outcome)` — early-exit/short-circuit `[dangerous_goods invalid→Unable, warehouse barcode→Returned, customer auth-fail→FAILED, video intake]`
- `iterate(block, until_pred, max_iter)` — bounded loop/retry `[customer troubleshoot→verify, aircraft circuit ×3]`
- `escalate_when(condition → target)` — conditional escalation **routing** (generalizes Group J `escalate_when` to multi-target) `[video ETM>thr, customer Tier2/FieldOps/NetEng]`

**B. DECISION** (state→value; **the extended Group J** + DMN)
- `precondition_trigger(pred → action)` — state gates action (Group J `repairs_state` generalized) `[customer, telecom]`
- `threshold_band(slot, [breakpoints] → category)` — multi-way binning Critical/Warning/Normal `[traffic, aircraft battery, dangerous_goods score→Class]` **(NEW; not in Routine/GroupJ)**
- `range_validate(slot → [lo,hi] | format)` — validation constraint `[dangerous_goods 1-5, aircraft ±2%, content VCT]` **(NEW)**
- `decision_table(inputs:[slots] → outcome, rows:[(pred-tuple→value)])` — N-dim DMN table `[traffic risk×violation→action, email intent×state→action]` **(NEW)**
- `count_score(indicators:[pred] → score_slot)` — additive TRUE-counting `[referral, content BPI gates]` **(NEW)**
- `sufficiency(action, min_evidence:[slots])` — commit boundary (Group J `diagnosis_sufficient_for`) `[all decide phases]`
- `mutual_exclusion({actions} | category)` — at most one branch/class `[all classify]`
- `contrast_negative(goal → wrong_action, why)` — Group J `distractor_for` (negative/contrastive) `[from failure traces]`
- `default_decision(point → fallback_outcome)` — else / "Unable to Decide" / "No Action" `[dangerous_goods, traffic, email]`
- `evidence_gate(decision, requires_evidence:bool)` — outcome conditioned on evidence presence `[traffic low-risk: WITH evidence→Warning vs WITHOUT→NoAction, referral ticket]` **(NEW)**

**C. DATA-FLOW & COMPUTATION** (subsumes Routine R1/R2 + adds compute)
- `arg_binding(action, param → source_slot)` — provenance / **variable memory** (arg_dependency, **R1**) `[order_fulfillment "save for step 4", all]`
- `produces(action → slot)` — which action establishes which slot `[all]`
- `compute_rule(target_slot ← op(input_slots))`, `op ∈ {weighted_sum, sum, formula, count, max, agg}` — derivation `[content BPI/UTC, dangerous_goods hazard=Σ, warehouse variance%, patient lifestyle]` **(NEW; central to scoring SOPs)**
- `value_map(categorical_slot → numeric/label)` — lookup `[patient smoking Never=0/Former=1/Current=2]` **(NEW)**
- `impute_rule(slot, missing_cond → imputation, else default)` — missing-data handling `[dangerous_goods: missing→max(others); >2 missing→Unable]` **(NEW)**
- `accumulate(slot ⊕= flag)` — set-valued / append (multiple findings) `[warehouse append problems, video categories]` **(NEW)**
- `slot_contract(phase → required_in:[slots], produced_out:[slots])` — Routine **R2** placeholder contract `[all]`

**D. CONTRACT & OBLIGATION** (SOP-specific — **NEW family**)
- `output_contract(required:[output_slots], format)` — final-output completeness + format `[customer JSON, dangerous_goods/email/traffic XML, aircraft tagged]` **(NEW)**
- `log_obligation(phase → record:[slots], sink)` — audit/log requirement `[dangerous_goods registry+audit, aircraft chain-of-custody, traffic PVDS]` **(NEW)**
- `verify_obligation(after_action → reprobe, verify_pred)` — act→re-check (old `verify_predicate`) `[customer re-diagnose, aircraft re-inspect]`
- `time_obligation(action → SLA)` — deadline `[video 24h upload]` **(NEW)**
- `robustness_note(step → tolerance)` — fuzzy/typo tolerance → LLM-fallback hint `[video "account for typos", dangerous_goods]` **(NEW)**

**E. SCENARIO / HIERARCHY** (Routine **R4**)
- `scenario_select(input_signature → scenario)` — deterministic routine selection by initial state `[email_intent→per-intent subflow, video role, referral violation-type]`
- `scenario_workflow(scenario → phase_subgraph)` — the DAG per scenario `[all multi-path SOPs]`

### 2.3 Coverage — every prior mechanism maps in (nothing lost; what's NEW)
| prior mechanism | → new relation(s) | status |
|---|---|---|
| §13 base: precondition/effect/achieves_goal, error_fallback, param_dataflow, workflow_role, observation_triggers | precondition_trigger, escalate_when, arg_binding, phase_role, branch | subsumed |
| **Group J** repairs_state / diagnosis_sufficient_for / distractor_for / escalate_when | precondition_trigger / sufficiency / contrast_negative / escalate_when | **extended** (multi-target, +DMN) |
| **Routine R1** variable memory | arg_binding (+ executor runtime var-store) | subsumed |
| **Routine R2** placeholder slot | slot_contract | subsumed |
| **Routine R3** branch/condition | branch (+default) | extended (+else) |
| **Routine R4** scenario | scenario_select + scenario_workflow | subsumed |
| current §2.1 (6) | tool_class→action_class, arg_dependency→arg_binding, precondition_trigger, diagnostic_next→branch/decision_table, phase_exit→slot_contract/sufficiency, verify_predicate→verify_obligation | subsumed |
| **NEW (no prior coverage)** | threshold_band, range_validate, **decision_table**, count_score, evidence_gate, **compute_rule**, value_map, impute_rule, accumulate, parallel_block, guard_exit, iterate, output_contract, log_obligation, time_obligation, robustness_note | **★beyond Group J + Routine** |

The **decision/computation layer** (threshold_band, decision_table, count_score, compute_rule,
value_map, impute_rule) and the **obligation family** are the substantive additions that
Routine's 4 mechanisms and Group J's 4 relations lack — and they are *majority* of several
SOPs (content_flagging, dangerous_goods, traffic, referral, patient are mostly compute+decide).

### 2.4 TBox / ABox split, transfer, dual sourcing
- **TBox** (domain-invariant, fixed): the entity types (§2.1), all 31 relation TYPES (§2.2),
  the phase_role taxonomy, the predicate grammar, the executor (§3). Transfers across all 12.
- **ABox** (per-domain, swappable): the INSTANCES — which concrete slots/columns, thresholds,
  formulas, branch predicates, decision-table rows, tools fill each type. `ontology_<domain>.json`.
- **Transfer test** = freeze TBox, swap ABox → held-out domain (12-domain rotation).
- **Dual sourcing + validation** (the research lever): each ABox is (a) **compiled** from
  `sop.txt` (each relation = a linguistic pattern: "If…→" → branch; "= Σ…" → compute_rule;
  "between a and b" → range_validate; "in format <xml>" → output_contract) AND (b) **induced**
  from teacher traces + column-dependency. Agreement(compiled, induced) and TSR of each are
  results; the authored SOP is the ground truth (impossible in tau2).

### 2.5 File layout
```
induced/ontology_<domain>.json  # { entities:{phases,actions,slots,scenarios},
                                 #   control:{phase_sequence,branch,parallel_block,guard_exit,iterate,escalate_when,precedes},
                                 #   decision:{precondition_trigger,threshold_band,range_validate,decision_table,
                                 #             count_score,sufficiency,mutual_exclusion,contrast_negative,default_decision,evidence_gate},
                                 #   dataflow:{arg_binding,produces,compute_rule,value_map,impute_rule,accumulate,slot_contract},
                                 #   contract:{output_contract,log_obligation,verify_obligation,time_obligation,robustness_note},
                                 #   scenario:{scenario_select,scenario_workflow} }
```

---

## 3. Deterministic executor loop (SOP-Bench, agent-only, single-shot)

The executor walks the §2.2 ontology as a **phase state machine**. State = `ExecState`:
`phase`, `slots` (ObservedState: INPUT + every FETCHED/COMPUTED/DECISION value so far),
`done` (action dedup), `scenario`. **All actions are agent tool calls or internal
COMPUTE/DECIDE** — no user, no NL templates (SOP-Bench has no simulated user). It runs to
TERMINATE, then emits the `output_contract` JSON/XML.

```
def run(task_inputs, ontology, llm):
    st = ExecState(slots=task_inputs, phase=first_phase, scenario=None)
    while st.phase != DONE:
        # G. global guards (any phase): early-exit + escalation
        ex = ontology.guard_exit.match(st.slots)            # e.g. invalid id → Unable
        if ex: st.set_outcome(ex.outcome); break
        esc = ontology.escalate_when.match(st.slots)
        if esc: st.route(esc.target); st.phase = REPORT_CLOSE; continue

        # R. scenario routing (R4), once inputs are read
        if st.scenario is None and ontology.scenario_select.ready(st.slots):
            st.scenario = ontology.scenario_select(st.slots)   # deterministic by signature

        act = next_action_for_phase(st, ontology)            # ↓
        if act is None:                                      # phase complete → advance
            st.phase = ontology.phase_sequence.next(st.phase, st.scenario); continue
        result = execute(act, st, llm)                       # tool call OR compute OR decide
        st.apply(result)                                     # produces/accumulate → slots
        st.done.add(act.key)

    return render(ontology.output_contract, st.slots)        # required keys + format

def next_action_for_phase(st, ontology):
    role = ontology.phase_role(st.phase)
    if role in (INGEST_VALIDATE, GATHER):
        a = next_unsatisfied_read(ontology.arg_binding, st.slots, st.done)  # monotone, dedup
        return a or (None if slot_contract_out_filled(st.phase, st.slots) else FALLBACK)
    if role == ANALYZE_COMPUTE:
        c = next_ready_compute(ontology.compute_rule|value_map|count_score, st.slots, st.done)
        return c or None
    if role == DECIDE_CLASSIFY:
        d = ontology.decision_for(st.slots)   # threshold_band / decision_table / precondition_trigger
        if d and ontology.sufficiency.ok(d.action, st.slots): return d   # commit boundary
        if d is None: return ontology.default_decision(st.phase) or FALLBACK
        return GATHER_MORE                    # sufficiency unmet → read/compute more
    if role == ACT:
        w = ontology.action_for(st.slots)     # WRITE_EFFECT / ROUTE, args via arg_binding
        return w or None
    if role == VERIFY:
        return ontology.verify_obligation.next(st)   # reprobe → verify_pred → loop/done
    if role == REPORT_CLOSE:
        emit_log_obligations(st); return None  # → DONE (render output_contract)
    return FALLBACK
```

**Guarantees** (single-shot, agent-only):
- **No loop/ wasted steps**: reads/computes are *monotone* — each produces an unfilled slot;
  `done` dedups; a phase advances only when its `slot_contract` outputs are filled.
- **Determinism where the SOP is determinate**: threshold_band / decision_table /
  compute_rule / value_map / branch resolve with **zero LLM** — these are the majority of
  scoring/classification SOPs.
- **Clean termination**: `guard_exit` and `output_contract` give an explicit, complete
  final output (right keys + format) — SOP-Bench scores the produced state, so completeness
  matters (`output_completeness`).
- **`sufficiency`** prevents premature DECIDE (commit only when min evidence present).

### 3.1 Where the LLM stays (fallback only)
`FALLBACK` = candidate-restricted LLM, invoked only where the ontology is *intentionally*
non-deterministic:
1. **Generative / fuzzy steps** flagged by `robustness_note` — free-text fields
   (resolution_summary, moderator_notes), typo-tolerant extraction, semantic "significantly
   different from description" (email_intent), classification with no crisp rule.
2. **Uncovered branch** — a phase/state with no matching ontology relation.
Coverage% (deterministic vs FALLBACK) per phase_role is a headline metric; it quantifies
"how much of each SOP the ontology executes without an LLM."

---

## 4. Induction plan — dual sourcing (compile from `sop.txt` ‖ induce from traces)

Each ABox relation has **two independent extractors**; we build both and report agreement +
each one's TSR. **Compile** parses the authored SOP (linguistic patterns); **Induce** mines
teacher SUCCESS traces + the `data.csv`/`test_set_with_outputs.csv` **column dependency
graph** (inputs→intermediate→output). The authored SOP is ground truth → induction is
*validated*, not just assumed (the core research claim, impossible in tau2).

| relation family | COMPILE (sop.txt pattern) | INDUCE (traces + columns) | difficulty |
|---|---|---|---|
| phase_sequence, phase_role | numbered sections 5.x → phases; verbs→role | step-order frequency in traces | easy |
| branch / decision_table / threshold_band | "If <cond> → <step>/<value>", "<Critical:…/Warning:…>" tables | per-decision: prior-slot predicates → outcome (CART/rule-fit on completed rows) | **med** |
| compute_rule / count_score / value_map | "= Σ…", "weight: 0.3", "Never=0/…", "count TRUE" | regress output-col on input-cols on the filled CSV (exact for additive/lookup) | **med** |
| range_validate / impute_rule | "between 1 and 5", "if missing impute max" | column value ranges + missing-handling in GT rows | easy |
| arg_binding / produces / slot_contract | tool `inputSchema` params ↔ column names; "save for step N" | value-match provenance (existing miner, applied to all actions) | easy |
| precondition_trigger / sufficiency | step preconditions; "after sufficient…commit" | last-slots-before-action, min evidence set before first write | easy–med |
| guard_exit / default_decision / mutual_exclusion | "no further action → <default>", "one of …" | early-terminating GT rows; one-hot outcome columns | easy |
| escalate_when / scenario_select / scenario_workflow | escalation rules; "intent = a|b|…" → subflow | cluster GT rows by outcome/issue signature → per-scenario DAG | med |
| output_contract / log_obligation / time_obligation | §6 Output (required keys, "xml/json", "within 24h", "audit trail") | output_columns; presence of LOG actions | easy |
| contrast_negative | — (positive SOP only) | GT-≠ actions in *failure* traces (per goal) | med |

Honest risks:
- `decision_table` / `compute_rule` mining can be noisy when the rule is non-additive or
  uses hidden variables → falls back to LLM for that decision (graceful, measured by
  coverage%). Compile-from-SOP is the cleaner upper bound there.
- Transfer: relation TYPES (TBox) transfer by construction; ABox INSTANCES swap. 12
  independent domain schemas = a hard, honest transfer test.
- Some steps are genuinely generative (`robustness_note`) → never deterministic; the
  ontology marks them so coverage% is interpreted correctly (not counted as failures).

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
