# Full Workflow Ontology — design (2026-05-30)

Status: **DESIGN, for review.** Supersedes the partial "fix-disambiguation only" ontology
(`step_realization_*` + `obs_triggers_*`) and the per-turn reactive resolver
(`ontology_resolver.py` + `two_stage_agent.py:_two_stage_generate`).

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

**D2 RESOLVED (empirical, shipped telecom trajectories):** device/probe tools
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

## 6. Experiment reconstruction

Metrics (all, every run): **Pass^1 + termination distribution (max_steps rate) +
read-loop signature (≥4× identical) + write/probe sparsity + per-PHASE deterministic
coverage**. (Pass^1 alone hid the real story; `_redx.py`/`analyze_two_stage.py` extended.)

Runs:
1. **telecom in-domain** (N=114, split base): P0 executor vs old base/resolver/fallback
   (`two_stage_v3`) vs entangled abstract. Expect: max_steps↓, read-loop→0, Pass↑.
2. **transfer LODO**: airline (N=50) + retail — swap ABox only, TBox fixed. Tests
   ABox-swap transfer with the richer ontology.
3. **ablations**: executor w/o `diagnostic_next` (DIAGNOSE→LLM), w/o `phase_exit`
   (read-loop returns?), P0 vs P1.
4. **max_steps sensitivity**: rerun old vs new at max_steps ∈ {50, 100} to separate
   "executor convergence" from "raw budget."

---

## 7. Build order (file changes)

1. `induce_ontology.py` (new) — emits `ontology_<domain>.json` with all §2.1 relations;
   internally reuses/absorbs `induce_step_realization.py` + `induce_observation_triggers.py`.
2. `workflow_executor.py` (new) — ExecState + `next_action` (§3); replaces
   `ontology_resolver.py` (kept as a thin compat view if needed).
3. `two_stage_agent.py` — `_two_stage_generate` rewritten: call executor for the next
   action instead of rewriting the planner's call; add a `--mode executor` (P0) and
   `--mode phase_planner` (P1); keep `base` for control.
4. `analyze_two_stage.py` / `_redx.py` — add per-phase coverage, termination, read-loop,
   verify-reach metrics.
5. (P1 only) `build_tbox_sft.py` — retarget labels to 6 phases; regenerate; retrain.

Self-tests at each step (the inducers/resolver already ship `--selftest`/`_selftest`).

---

## 8. Open decisions
- **D1** (open): P0-first (no planner) vs build P1 phase-planner immediately. (Rec: P0 first.)
- **D2** (RESOLVED): telecom device/probe tools are **user-only** (`requestor=user`);
  account tools agent-callable. DIAGNOSE/FIX(device) → deterministic NL templates +
  user-tool-result ingestion. See §3.
- **D3** (open): Single `ontology_<domain>.json` vs keep split files + add new ones. (Rec:
  single file, with a compat exporter for old analysis scripts.)
- **D4** (open): `max_steps` for eval — keep 50 (old default) or raise to 100 for
  multi-fault tasks? (Rec: run both to separate convergence from raw budget.)
