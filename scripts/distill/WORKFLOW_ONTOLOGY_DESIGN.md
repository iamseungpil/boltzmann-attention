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

## 2. Ontology schema

### 2.1 TBox (domain-INVARIANT — transfers across telecom/retail/airline)

**Phases** (ordered control skeleton, with allowed transitions):
```
IDENTIFY  → GATHER → DIAGNOSE → FIX → VERIFY → (DIAGNOSE | CLOSE)
                         ↑__________________|        (re-enter on residual fault)
   any phase → ESCALATE  (guard: escalation trigger)
```
- `IDENTIFY`  : resolve who/which line (needs identity slots).
- `GATHER`    : agent-side account reads (get_*), DAG-ordered, until exit predicate.
- `DIAGNOSE`  : state-revealing probes (check_*/run_speed_test…); pick next by symptom.
- `FIX`       : state-changing ops — split into `device_op` (toggle/reset/reseat) and
                `account_write` (enable_roaming, refuel_data, resume_line, payment).
- `VERIFY`    : re-probe; confirm `verify_predicate`; loop back if residual fault.
- `CLOSE`     : resolution stated + user confirm → STOP (prevents max_steps churn).
- `ESCALATE`  : transfer_to_human (guard predicate).

**Tool classes** (role types, domain-invariant):
`read` (account fetch) · `probe` (state-revealing, no mutation) · `device_op`
(user-device mutation) · `account_write` (carrier-DB mutation, the GT actions) ·
`escalate`.

**Slot types** (state-variable kinds): `identity` (customer_id, line_id, bill_id),
`config_flag` (roaming_enabled, data_saver, airplane_mode, apn_status…),
`status` (line status, cellular_connection, sim_status), `quota` (data_used/limit),
`billing` (total_due, bill status).

**Relation types** (the inducible maps):
- `tool_class(tool) → class`                                  (replaces role heuristic)
- `arg_dependency(tool, param) → producer_tool.field`         read-DAG + write provenance
- `precondition_trigger(tool) → [state predicate]`            state→{probe|device_op|write}
- `diagnostic_next(observed_predicate) → probe`               symptom → next probe
- `phase_exit_predicate(phase) → slot-set / trigger`          when a phase is complete
- `verify_predicate(fault|fix) → state predicate`             what confirms resolution

### 2.2 ABox (per-domain — swaps; this is "transfer = ABox swap")

Concrete instances of every TBox relation for one domain: which concrete tools fill each
class, concrete slot paths, concrete predicates (`get_details_by_id|roaming_enabled=False
→ enable_roaming`), concrete arg bindings, concrete diagnostic rules. Stored per domain:
`ontology_<domain>.json` (one file, replacing the two split files).

### 2.3 File layout (new)
```
induced/ontology_<domain>.json   # {tool_class, arg_dependency, precondition_trigger,
                                  #  diagnostic_next, phase_exit, verify_predicate, slots}
```
Back-compat shim: keep emitting `step_realization_*`/`obs_triggers_*` views for the old
analysis scripts, or migrate them.

---

## 3. Deterministic executor loop

The executor holds **ExecState**: current phase, ObservedState (slots from reads/probes/
user msgs), `done_actions` set (dedup), fault hypotheses. Each agent turn:

```
def next_action(execstate, ontology, llm_perceive):
    obs = execstate.obs
    # 0. ESCALATE guard
    if ontology.escalate_trigger(obs): return Talk-or-Call(escalate)

    if phase == IDENTIFY:
        if not identity_slots_filled(obs): return ask_user_for_identity()  # NL template
        phase = GATHER

    if phase == GATHER:
        nxt = next_unsatisfied_read(ontology.arg_dependency, obs, done_actions)
        if nxt: return Call(nxt)                      # monotone: new read only
        if ontology.phase_exit(GATHER, obs): phase = DIAGNOSE   # exit predicate
        else: phase = DIAGNOSE                        # nothing left to read

    if phase == DIAGNOSE:
        # pick next probe by symptom; skip probes already done
        p = ontology.diagnostic_next(obs, done_actions)
        if p: return Call_or_InstructUser(p)          # device probe
        if any fix trigger fires: phase = FIX
        elif ontology.verify_ok(obs): phase = CLOSE
        else: phase = ESCALATE

    if phase == FIX:
        t = ontology.fix_for(obs)                     # precondition_trigger → write/device_op
        args = ontology.fill_args(t, obs)             # arg_dependency
        if t and args complete: done; phase = VERIFY; return Call_or_InstructUser(t)
        else: phase = DIAGNOSE                         # underdetermined → probe more
                                                       #   (LLM fallback if still stuck)

    if phase == VERIFY:
        if not fresh_probe_done: return Call(verify_probe)
        if ontology.verify_ok(obs):
            if all_known_faults_cleared(obs): phase = CLOSE
            else: phase = DIAGNOSE                     # residual fault → re-loop
        else: phase = DIAGNOSE

    if phase == CLOSE:
        return Talk(resolution_summary) → STOP
```

> **NOTE (SOP-Bench): the entire user-side/D2 block below is tau2-only and is DROPPED.**
> SOP-Bench exposes ALL tools as agent-callable (no user simulator), so DIAGNOSE/FIX are
> ordinary agent tool calls — no NL instruction templates, no user-tool-result ingestion.
> The executor (§3 loop) keeps only the agent-call path. The D2 analysis is retained as
> the evidence that tau2's locus was user-side (motivating the benchmark move).

**D2 RESOLVED (empirical, shipped telecom trajectories — tau2 only):** device/probe tools
(check_*, toggle_*, reset_*, reseat_*, reboot_device, run_speed_test, set_network_*) are
**`requestor=user`** — USER-side, executed on the user's phone (e.g. check_status_bar:
user×130 vs assistant×2). Only account tools (get_*, enable_roaming, refuel_data,
resume_line, send_payment_request, transfer) are **`requestor=assistant`**. So the agent
**cannot call probes/device_ops directly**; it must instruct the user via NL and the user
executes. (The old `apply_policy_action` bucket mis-modeled these as agent-callable — an
artifact of `collect_tools` filtering `requestor==assistant` and catching the rare noise.)

Consequences for the executor:
- `GATHER` (account reads) + `account_write` FIX + escalate + close → **agent tool calls**
  the executor emits directly.
- `DIAGNOSE` probes + `device_op` FIX + `VERIFY` re-probes → the executor emits a
  **deterministic NL instruction template** ("Please run check_apn_settings", "Please
  toggle airplane mode off"); the user runs it; the **user-side tool result** is ingested.
- **ObservedState must ingest user-side tool results.** Today `_observed_from_messages`
  only records reads with `requestor=="assistant"`, so probe outputs never reached the
  resolver — another reason diagnosis was unmodeled. The new executor keys ObservedState
  on **both** assistant reads and user-side probe results.

Key guarantees:
- **No read-loop**: `next_unsatisfied_read` only returns a read whose output slots are
  not yet in obs; `done_actions` dedups. GATHER strictly advances or exits.
- **Convergence (not via collapsing user turns — those are inherent)**: the win is
  removing WASTED turns — (a) no read-loop, (b) deterministic, non-redundant probe
  selection via `diagnostic_next`+dedup (no wandering/repeated probes), (c) optional
  batched instructions ("run check_X and check_Y") to cut round-trips, (d) deterministic
  CLOSE instead of dragging to `max_steps`. Genuine multi-fault tasks may still need a
  higher `max_steps` (3 faults × ~3 user turns each > 50) — measured as a separate axis.
- **Closure**: CLOSE stops the agent, avoiding `max_steps` truncation.
- **LLM fallback** only at the two marked spots (FIX underdetermined, DIAGNOSE no rule):
  candidate-restricted, same as today's `fallback` mode.

### 3.1 Where the LLM stays
1. **Perception**: parse free-text user replies ("I turned it off, now No Signal") into
   slot updates. tau2 also returns the **structured user-tool result** alongside the NL,
   so ingest the structured result first (no LLM); use LLM/regex only when the user
   answers in prose without a tool result.
2. **Uncovered branch**: phase with no matching ontology rule → candidate-restricted LLM.

### 3.2 NL instruction templates are ABox
Each user-side probe/device_op needs an instruction string. Induce per-tool templates
from the agent NL turn immediately preceding each user-probe in success trajectories
(or a simple "Please run <tool>" default). Stored in `ontology_<domain>.json` under
`user_instruction[tool]`.

---

## 4. Induction plan (extend the existing miners)

All mined from **teacher SUCCESS trajectories** (reward≥0.999), reusing
`induce_observation_triggers.py` machinery (`flatten`, `_is_state_field`, value-match
provenance). Difficulty noted.

| Relation | How induced | Difficulty |
|---|---|---|
| `tool_class` | by **requestor** in trajectories (robust, not name heuristic): `requestor=user` → probe (read-like name) or device_op (mutating name); `requestor=assistant` + `is_read` → read; `requestor=assistant` + in `gt_writes` → account_write; escalate = regex. Replace `step_for`. | easy |
| `arg_dependency` (read-DAG) | **apply existing arg-provenance value-match to READ tool args** (currently writes only): for each read with non-identity args, find prior read field with same value. | easy |
| `precondition_trigger` (probe/device/write) | generalize trigger mining from `gt_writes` only → **all mutating + probe tools**: last-observed state fields immediately preceding the call, precision-filtered. | easy–med |
| `diagnostic_next` (symptom→probe) | mine sequence: given ObservedState predicate set before a probe, which probe followed; rank by P(probe \| symptom). Covers the DIAGNOSE decision tree. | **medium** (the procedural core) |
| `phase_exit_predicate(GATHER)` | derive: union of read-fields referenced by any DIAGNOSE/FIX trigger ⇒ GATHER done when those slots observed. Pure post-hoc derivation from triggers. | easy (derived) |
| `verify_predicate` | mine state fields observed AFTER a successful fix that indicate health (e.g., cellular_connection=connected). | medium |

Honest risks:
- `diagnostic_next` is the procedural crux; if mining is noisy, executor falls to LLM in
  DIAGNOSE more often — graceful degradation, not failure.
- Transfer: TBox phase skeleton + relation TYPES should transfer; ABox bindings swap. The
  airline 6.7% failure was an ABox value-binding mismatch (catalogue), which a richer
  ABox with explicit `arg_dependency`/`diagnostic_next` may or may not fix — to be tested.

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
