# [CASE STUDY · 2026-07-24] Six Reversals to One Cause: A Worked Example of Isolation Attribution

> Companion section to `PAPER_TRACKC_DRAFT_v0_2026_07_24.md` (Track C). Intended as a self-demonstration
> of Contribution 2 (isolation replay as a per-failure attribution control) and of the §8 over-attribution
> argument, turned on the authors' own reasoning.
> Evidence ledger: RESEARCH_MASTER §3 C135–C142. Grades follow repo convention ([S]=trajectory-verified,
> [M]=measured with stated caveats). Probe scripts: `plan_iso_probe.py`, `position_probe.py`, `exec_iso_probe.py`.
> ⚠Honesty gates: single failure, single domain (τ²-bench `banking_knowledge`), single task (task_043
> "close credit card"); probes are keyword-scored against a gold 7-step chain; gpt-5.2 rows are small-n.

## The failure

In task_043 the customer asks to close a credit-card account. The gold chain has seven steps; the live 32B
agent completes the payment and closing but never executes `apply_credit_card_account_flag` (an annual-fee
retention/waiver step the customer's request does not name). Scored by any taxonomy, this is a wrong/missing
tool call — a **planning capability** failure: the model cannot assemble the full policy-mandated chain. Our
own ledger opened it that way (C135). It took six isolation experiments, each reversing the prior conclusion,
to find the actual mechanism — and three of the reversed conclusions were **ours**.

## The reversal arc

Each row is an information-matched isolation probe that overturned the reading above it. "B" = full-trajectory
prefix; "A" = minimal context (request + account + policy); scores are gold-steps recovered / 7 unless noted.

| # | Isolation pushed | Numbers (from ledger) | Surface reading | Overturned by |
|---|---|---|---|---|
| C135 | plan-only replay, A vs B | A 0.7, B 2.6 | "model can't plan the chain — **capability**" | *retracted at C140* |
| C137 | scale sweep (32B / gpt-4.1 / gpt-5.2) | B/A = 2.6/0.7, 3.0/1.0, 2.7/0.7 | flat curve → "**scale-invariant** → capability crossover" | C138 |
| C138 | inject the procedure doc (full-KB) | C_fullkb = 4.6, 5.2, 5.7 (vs B 2.6/3.0/2.7) | +2–3 jump → "it's **information**, not capability" | refines C137 |
| C139 | BM25 retrieval isolation | plain-word queries retrieve the docs; function-name queries fail; agent *did* retrieve 001/002/003/016 | "not retrieval → **salience/position**" (loosely worded) | C140/C141 |
| C140 | position control (procedure at start/mid/end) | gpt-4.1 P_absent 3.0, P_start 7.0, P_middle 7.0, P_end 6.5 | not lost-in-the-middle; **presence** dominates; low scores were procedure-*absent* all along | retracts C135 |
| C142 | execution isolation (E0 / E_reason / E_act) | both models: plan apply_flag **4/4**, reason eligibility **4/4** | clean context → model does it all → residual is **reactive-execution** | *final* |

(C136 and C141 sit between these as live/consistency checks: C136 showed the soft chain-reminder detects the
gap correctly but the agent still never attempts `apply_flag`; C141 confirmed the retention procedure
(doc_003, Step 4) was actually retrieved and the customer actually eligible, yet the step was skipped — while
softening C140's "position-invariant" to model-dependent: gpt-4.1 flat, 32B primacy 6.7>5.7>4.7, gpt-5.2
noisy and small-n.)

## What each reversal taught

- **C135 → C137**: reading a low B-score (2.6/7) as capability, then "confirming" it across scale, is the
  textbook over-attribution the paper warns about (§8). The scale-flat curve *looked* like a scale-invariant
  residual — the strongest possible capability claim — and was wrong.
- **C138**: the same decision jumps +2–3 steps the instant the procedure is placed in context. Capability
  that reappears when you supply a document was never capability; it was **missing information**. This is the
  attribution flip of Contribution 2, applied to a *planning* decision rather than a reference-selection one.
- **C139 → C140**: our own next guess — "the procedure is retrieved but buried, so it's salience/position" —
  was itself too strong. The position control shows presence, not position, gates planning; the earlier low
  scores (A 0.7, B 2.6, P_absent 3.0) were **simply contexts in which the procedure was absent**. The data
  were consistent from the start (C135's 2.6 was never evidence of anything but absence); only our
  interpretation moved. This retracts C135's capability claim outright.
- **C142**: with a clean context containing the procedure, both models **plan** `apply_flag` 4/4 and **reason**
  the fee-waiver eligibility 4/4. So the residual is none of planning / position / retrieval / information /
  ignorance / reasoning. It is **reactive-execution**: the live agent generates turn-by-turn, never enumerates
  the full plan, passes the retention step on the way to closing, and does not go back. It can plan the step in
  isolation; it does not self-enumerate it during execution.

## Why this is the paper's strongest evidence

The method's value is exactly that the **surface attribution was wrong at every intermediate step**. A single
run of the agent, or a single probe, would have shipped one of six wrong causes — including the seductive
"scale-invariant capability crossover" (C137), which is precisely the headline a capability-centric taxonomy
would report. Only iterated, information-matched isolation reached the truth, and it did so by catching the
**analyst's** premature conclusions (C135, C139, and C140's over-strong wording) alongside the model's. That
the reversed conclusions were largely our own is a feature: the control is an ablation for bias, not just for
the system under test (§8 — "observational labels are hypotheses, not measurements").

The endpoint is a distinct, nameable failure mode — **reactive-execution / plan-non-enumeration** — separable
from the transcription-slip / self-anchor family of the main draft, and it prescribes a concrete lever: an
**isolated planning sub-call in clean context** that emits the full chain, which a deterministic controller
then drives. This vindicates the "plan = LLM, controller = deterministic" division (roles ledger) and
**reverses C135's "plan = LLM fails"** — the LLM can plan; it just won't do so unprompted mid-trajectory.

**Caveats.** One failure, one domain, one task; keyword-scored probes against a gold chain the model may not
internally endorse as mandatory; gpt-5.2 rows are small-n and noisy (C141); the E0/E_reason/E_act split is
4-sample per condition. The arc establishes the *mechanism* and the *method*, not a population rate — that is
the §6 audit's job. ⟦?⟧ Whether reactive-execution generalizes beyond policy-mandated (customer-unnamed) steps
is untested here.

---

## Follow-up (2026-07-25 · ledger C159/C160/C161) — mechanism confirmed, then **eliminated**; one measurement hazard

**The mechanism claim above stands on its own data.** Re-analysis by *execution path* (dispatcher call vs
mere unlock) plus independent DB-diff confirms real `close_credit_card_account` **executions** in the runs this
arc rests on: rall24a @95, rall25a @84 (both with no `apply_flag`), and reg043_base/treat, fix_base. So the
agent genuinely closed on the way past the retention step.

**Two updates the paper must carry:**

1. **The prescribed lever worked (new [S] result).** A deterministic *pre-close* gate — deny the finalizing
   write while prerequisites are outstanding, and resurface the source policy document — removed the behavior:
   matched pair, gate **off** → close executed (fix_base @52); gate **on** → close **0** across fix_treat, dd,
   pr, and all 4 nt4 trials, with `apply_flag` executed in 3/4. This strengthens rather than weakens the
   §"plan = LLM, controller = deterministic" claim, and it contradicts an earlier draft line calling the
   pre-close gate "soft"; that line was based on a miscount (see 2) and must be removed.
2. **⚠ Measurement hazard — do not count tool events by argument-string matching.** In this environment the
   *unlock* call carries the target tool's name **in its arguments**, so a substring match counts unlocks as
   executions. Both a prior session and this one made exactly that error, concluding "the agent still closes"
   when it had only unlocked. **Rule: attribute events by execution path (dispatcher/direct call), and confirm
   with a DB-diff on final state.** Any per-step count in this case study that was produced by name matching
   needs re-derivation before it ships.

**Scope note for the population claim (C160).** A validated 97-task DB-diff re-tally (replay reproduces the
official db_match exactly, 9/97, mismatch 0) finds that behavior failures still dominate: 63% of tasks differ
from gold in **both** state and bookkeeping, 24% in state only, and only **4%** are "behaviorally correct,
blocked solely by discovery-registration bookkeeping." So the residual reported for task_043 after the gate
(bookkeeping) is a *local* endpoint, not the benchmark-wide one.
