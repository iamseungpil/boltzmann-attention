# [DRAFT v0 · 2026-07-24] Not a Semantic Error: Isolation Replay Dissociates Transcription Slips and Self-Anchoring from Capability Limits in LLM Tool Agents

> Working draft (English, markdown → LaTeX later). Placeholders marked ⟦TBD⟧.
> Evidence ledger: RESEARCH_MASTER §3 C121/C124/C125/C126 · sim_results/bank_rall19*/bank_rall20*/bank_rall21* ·
> probe_039_join_iso.py. Grades follow the repo convention ([S]=trajectory-verified, [M]=measured with stated caveats).
> Related-work map: RELWORK_ISOLATION_ATTRIBUTION_2026_07_24.md (105-agent verified survey, 2026-07-24).
> ⚠️Honesty gates carried into the text: single-domain case study until §6 audit lands · B_fullctx replay uses
> flattened-transcript serialization (stated) · rival preprints cited for framing, not numbers (their headline
> mitigation number failed our verification) · self-correction hedge for RL-trained reasoning models.

## Abstract

When an LLM tool agent files a dispute against the wrong `transaction_id`, every existing failure taxonomy
scores it the same way: a semantic reference-selection error — the model could not tell which record the
customer meant. We show that this attribution is often wrong, and give a cheap method to test it. In a live
τ²-bench banking pipeline, a 32B agent asked to dispute a Costco charge of $267.34 filed against a Home Depot
charge of $456.78 — a transaction sharing only the *date* with the intended one, sitting in the *adjacent row*
of a 57-record listing. Forensics show the agent's own intermediate summary had bound the correct row's
attributes (merchant, amount, date) to the neighboring row's id: a **transcription slip**, not a semantic
confusion. Once written into the conversation, the wrong binding is copied forward without re-verification —
**self-anchoring**. We dissociate these from capability limits with **information-matched isolation replay**:
given exactly the customer's messages and the listing (minimal context), the model identifies the correct
record set 9/9 times; given the actual trajectory context, it reproduces the *same wrong id* 9/9 times — a
deterministic attribution flip. The error also recurs across independent live runs (the same wrong-record id,
and — a mechanism we did not anticipate — laundered through the user simulator, which echoes the agent's wrong
id back as apparent customer evidence), and interacts pathologically with evidence gates: hard denial collapses
the episode, pass-through pollutes the database. A mechanism-derived mitigation — an isolated minimal-context re-pick of
reference arguments at write time, accepted only if the answer exists verbatim in the producer listing (so the
repair itself cannot introduce an out-of-listing value) — closes the slip in unit probes (8/8), and in live
runs confirms the design's safety half (0/16 false switches); corrective efficacy remains ⟦TBD: R22⟧. A
retro-audit of failures previously
attributed to semantic reference-matching finds ⟦TBD⟧% flip under isolation replay ⟦E-F3-ISO⟧. We argue that
agent failure taxonomies built on observational labeling systematically overattribute context-induced load to
semantic capability, and that per-failure isolation replay should be a standard attribution control.

## 1. Introduction

Agent evaluation has converged on a comfortable practice: run the agent, collect failures, and sort them into
capability buckets — wrong tool, wrong argument, wrong entity, hallucination. The buckets then drive
conclusions: *reference matching is a capability ceiling*; *scale does not fix entity selection*; *semantic
grounding is the residual*. These conclusions matter — they decide whether teams invest in bigger models,
fine-tuning, or scaffolding.

This paper is about a failure that wears the wrong-entity costume convincingly. In our live banking pipeline
(τ²-bench `banking_knowledge`, Qwen2.5-32B agent, deterministic scaffold), a customer asks to dispute a
warehouse-club charge (Costco, $267.34); the agent retrieves a 57-record transaction listing, correctly
*describes* the target transaction —
right merchant, right amount, right date — and then files the dispute against a different record. Scored by any
existing taxonomy, this is a semantic reference-selection error. Our own evidence ledger initially classified it
exactly that way, into the same bucket as the ~44%-ceiling reference-matching boundary we had measured
elsewhere.

The classification is refuted by a 20-minute experiment. We replay the *decision in isolation*, information-
matched: the model receives verbatim the customer's messages and the full listing — nothing else — and is asked
which records the customer means. It answers perfectly, 9/9 (greedy and sampled). We then replay the decision
with the *actual trajectory context* (the agent's 64 preceding messages, flattened): it reproduces the exact
wrong id, 9/9. Same information, opposite outcomes, deterministic in both directions. The failure is not
in the model's semantic capability; it is induced by the trajectory itself.

Forensics identify the mechanism in two parts:

- **Adjacent-row transcription slip.** The listing's row #41 (the gold target: Costco, $267.34, 11/04) is
  immediately followed by row #42 (Home Depot, $456.78, *same date* 11/04). The agent's intermediate summary
  bound row #41's attributes to row #42's id — attribute-id binding slipped by one row of a long, format-
  homogeneous listing. The semantic selection had already succeeded; the id transcription failed.
- **Self-anchor propagation.** The mis-bound mapping, once emitted into the conversation, is treated as settled:
  every downstream step copies it instead of re-consulting the listing. This is what makes the full-context
  replay deterministic — the wrong binding is *in the prompt*.

The same signature recurs in an independent run on the same task (same wrong id), in a second task (a
date-adjacent wrong pick from a different listing), and in two variants that look like fabrication but are not:
a placeholder-pattern id `txn_a1b2c3d4e502` that is in fact a *real* listing record (Philadelphia Airport
Parking) mis-bound to the wrong item, and a card-digit argument `1234` transplanted from a PIN example in a
knowledge-base document read moments earlier. None of these are semantic confusions; all of them would be
scored as such — and every wrong value is real text already present in the agent's context, never a string
invented from nothing (§4).

Contributions:

1. **A phenomenon**: adjacent-row attribute-id transcription slips inside live agent trajectories, frozen by
   self-anchor propagation — identified end-to-end (trajectory → mechanism → recurrence), with a deterministic
   9/9-vs-9/9 dissociation (§4–5). To our knowledge the transcription mechanism has not been identified inside
   agent pipelines (nearest work: same-key interference probes and single-turn table-QA row/column confusions;
   see §2).
2. **A method**: information-matched isolation replay as a per-failure attribution control — minimal-context vs
   full-trajectory replay of the same decision with the same information (§5). Ancestors exist (isolated
   re-prompting, information-matched single-turn controls); per-failure application to live tool-use failures
   is, to our knowledge, unclaimed.
3. **An audit**: re-attribution of failures previously labeled semantic-reference errors. ⟦TBD E-F3-ISO: N
   failures re-probed, X% flip to load (transcription/self-anchor), Y% ambiguous, Z% remain capability;
   includes the fraction of "wrong pick despite gold record retrieved" cases (43–52% in our earlier census
   [M]) that flip⟧ (§6).
4. **A mitigation**: write-time isolated re-pick of reference arguments — re-run the selection in minimal
   context and substitute only if the answer exists verbatim in the producer listing (so the repair cannot
   itself introduce an out-of-listing value). Unit probes 8/8; live (rall21): 0/16 false switches (safety half
   confirmed), corrective switch ⟦TBD: R22⟧ (§7). We also document why evidence *gates* cannot close this
   failure class: the deny-forever/pass-through dilemma (§7.3).

## 2. Related work

*Self-generated errors propagate.* Hallucination snowballing [Zhang et al., 2023, arXiv:2305.13534] shows LMs
over-commit to early mistakes and produce errors they can recognize as wrong when re-asked in a fresh session
(67%/87% for ChatGPT/GPT-4) — the ancestor of our isolation probe, in single-turn QA. Self-conditioning
[arXiv:2509.09677] establishes causally (by injecting artificial histories with controlled error rates) that
per-step error rates rise when a model's own prior errors are in context, in synthetic long-horizon execution.
"Lost in multi-turn" [Laban et al., 2025, arXiv:2505.06120] provides the strongest information-matched control
precedent: a CONCAT control (same information, one turn) recovers 95.1% of full performance, and an
aptitude/reliability decomposition attributes apparent capability loss mostly to unreliability (−16% n.s. vs
+112%). None of these operate on live tool-use trajectories or entity binding; none replay *specific failed
decisions*. Intrinsic self-correction failure [Huang et al., ICLR 2024, arXiv:2310.01798] supports the premise
that agents do not spontaneously re-verify earlier mappings (hedged for RL-trained reasoning models, which
exhibit trained within-trace re-verification).

*Binding and interference.* PI-LLM [arXiv:2506.08184] shows log-linear retrieval decay under same-key
interference with a diagnostic error signature (earlier values for the same key), dissociated from context
length. Single-turn table-QA work on data-referencing errors [arXiv:2606.32029] separates wrong-cell citation
from answer correctness and observes (qualitatively) that a first referencing error is repeated without
re-consulting the table. These are single-turn probes; the interference unit is a key or a cell, not a record
row inside an agent trajectory. Our companion papers analyze a different unit — same-rule-clause interference in
batched judgments, with an associative-memory account; we deliberately keep this paper at the behavioral/
attribution level and cross-cite (the units differ: structured record rows here, policy clauses there).

*Closest rivals.* Two recent preprints (same author pair) are adjacent: Entity Binding Failures
[arXiv:2606.30531] dissociates wrong-entity actions from tool selection (0% wrong-tool vs 24–26% wrong-entity)
with a taxonomy built on natural-language reference *ambiguity* — which "Alex" — in single-step decisions;
Binding Drift [arXiv:2607.18316] formalizes error propagation of wrong entities in multi-step tool agents and
shows an "entity lock" amplifying *seeded* wrong actions. Neither identifies the adjacent-row transcription
mechanism, performs isolation replay, nor audits attribution; our self-anchor element overlaps theirs partially
— our delta is *spontaneous* (non-injected) anchoring with deterministic reproduction, dissociated from the
transcription slip that seeds it. (We cite these for framing; their headline mitigation number did not survive
our source verification.)

*Failure taxonomies.* ToolCritic [arXiv:2510.17052] and similar taxonomies assign failure categories by
observational labeling; ToolCritic's schema folds typos and semantically wrong values into one bucket, and its
teacher-forcing evaluation (gold history each turn) structurally removes self-anchor propagation from
measurement. We found no taxonomy that validates category assignments with counterfactual replay or ablation
(⟦to assert in final text: direct re-check of MAST/TRAIL/τ-bench error-analysis appendices — pending⟧). The
aptitude/reliability decomposition above supports the overattribution premise at task level; no work
demonstrates per-failure attribution flips in agent taxonomies.

## 3. Setting

τ²-bench `banking_knowledge` (dual-control: agent-side tools, user-side discoverable tools, KB search;
grading = final DB state). Agent: Qwen2.5-32B-Instruct (GPTQ-Int8, vLLM, greedy), user simulator gpt-5.2,
deterministic scaffold stack (generation-level verification gates; the stack is held fixed throughout — this
paper's manipulations are the replay probes and one added lever, §7). All trajectories, probes, and scripts are
committed; every number below traces to a logged artifact.

Tasks referenced: task_039 (8 disputed transactions described in prose, to be matched against a 57-record
listing and filed individually), task_031 (single dispute, card-digit acquisition via user-side tool), plus two
others in the same runs (043, 054) not central here.

## 4. The phenomenon

### 4.1 Case anatomy (task_039, run R19)

The customer enumerates 8 items in prose (merchant, date, amount, reason; no ids: "10/09/2025 – Costco –
$234.56 (Fraud)…"). The agent retrieves the listing (57 records, 17,085 characters, homogeneous format). Its
next summary message enumerates the items *with* resolved ids — and item #3 reads:

> "**11/04/2025 – Costco – $267.34** (Transaction ID: txn_41735bd2d06d)"

The attributes belong to record #41 (`txn_39469d5db822`: Costco, $267.34, 11/04). The id belongs to record
#42 (`txn_41735bd2d06d`: **Home Depot, $456.78**, Business-card, same date 11/04) — the immediately following
row. Semantic selection succeeded; id transcription slipped one row. Every downstream step — user confirmation,
eight `file_dispute` calls — carries the wrong id; the gold record is never filed. [S]

### 4.2 Recurrence (independent runs)

- **R20/task_039** (fresh run, same task): the same wrong id `txn_41735bd2d06d` is filed again — consistent
  with the deterministic full-context replay (§5) — plus another wrong-record id `txn_a1b2c3d4e502` among the
  eight. (Correction to an earlier draft: this id is **not fabricated** — it is a genuine listing record,
  "Philadelphia International Airport Parking $87.00"; its suspicious sequential-nibble form initially led us to
  mislabel it. Verified against the tool output; see §4.4.) [S]
- **R20/task_031** (different task): the agent files against `txn_9a72b84326d1` (Facebook Ads, $203.58,
  11/06, wrong card) where gold is a Marriott charge of $167.34 on **11/07** — a date-adjacent wrong pick from
  a different listing. [S]
- **R19/task_031 (trial 2)**: an argument-level variant — `card_last_4_digits: "1234"`, anchored verbatim to a
  PIN-security example ("Cannot be sequential (e.g., 1234, 4321)") in a KB document the agent had read moments
  earlier; the true digits were never retrieved. Here the value is transplanted from a document rather than a
  neighboring record row — a related but distinct source; we keep it separate from the row-slip family below.
  [S]

The row-slip family signature: the written value is a *real record id from the listing* — a neighboring or
otherwise co-listed row — bound to the wrong item; not a random string and not a semantically plausible
alternative reading of the user's request.

### 4.3 Why gates cannot see it

The pipeline's evidence gates check that written values exist in tool outputs. A neighboring row's id *does*
exist in the listing; a document example *does* exist in a tool output. Substring-existence evidence is
structurally blind to mis-binding — it validates that a value was read, not that it was bound to the right
attributes. (We measured this the hard way: a `1234` transplanted from a document's PIN example passed an
evidence gate because those digits appeared in an unrelated document line; §7.3.)

### 4.4 The error propagates through the user simulator (not just fabrication, not pure self-anchor)

A forensic follow-up (§6 audit) overturned two of our own earlier characterizations and refined the mechanism.
(i) *No fabrication.* Every wrong id we file traces to a real listing record; the "fabricated placeholder"
reading was wrong. (ii) *The channel includes the user.* In R21/task_039, a **user** turn states "Amazon
$129.45 (`txn_a1b2c3d4e502`)" — the user simulator attaches a wrong id (that record is Airport Parking, not the
Amazon charge) to a customer item. Since the customer explicitly has no ids ("I don't have the transaction IDs
handy"), any id in a user turn is the agent's earlier mis-binding reflected back by the simulator, or a
simulator artifact. The self-anchor is therefore not purely intra-agent: **the erroneous binding, once emitted,
is laundered through the conversation partner and returns as apparent user-provided evidence** — which is
exactly why downstream gates and replays treat it as grounded. This strengthens the load account (the error
lives in the trajectory, now demonstrably in the *user-visible* channel) while correcting its mechanism.

## 5. Isolation replay: the attribution flip

**Protocol.** For a failed decision d in trajectory T, construct two probes containing the *same task-relevant
information*:

- **A_minimal**: verbatim customer messages + the producer listing (nothing else) + the decision question.
- **B_fullctx**: the actual trajectory prefix up to d (flattened transcript serialization — stated caveat) +
  the same question.

Run both at the trajectory's decoding settings plus samples (here: greedy + 8 samples at T=0.7, same 32B
model). Attribution rule: failure in both ⇒ capability candidate; success in A_minimal with failure in
B_fullctx ⇒ context-induced load (and B−A localizes the inducing content); success in both ⇒ the live failure
was decoding/orchestration noise, re-examine.

**Result (task_039 decision).** A_minimal: correct 8-record set, **9/9** (every run; the slipped record
included, the neighbor never picked). B_fullctx: **9/9** runs pick `txn_41735bd2d06d` — the exact live error,
reproduced deterministically. [M: single decision; serialization caveat; the "ordered" metric is not used —
set-level only.]

**Localization.** The trajectory prefix contains the agent's own summary with the wrong binding (§4.1);
A_minimal differs from B_fullctx precisely by the absence of self-generated mappings (plus auxiliary dialogue).
The replay therefore identifies self-anchoring as the proximate driver: the model reads its own earlier
resolution instead of re-deriving from the listing. The *original* slip (first binding) is a one-row
transcription error under a 17KB homogeneous listing — an interference-class error consistent with the
single-turn binding literature (§2), here caught inside a live pipeline.

**Cost.** 18 inference calls, no user simulator, no re-run of the episode: minutes per failure on a local GPU.
This is the point — the control is cheap enough to apply per failure, routinely.

## 6. Retro-audit of "semantic" failures ⟦TBD — E-F3-ISO⟧

⟦Protocol locked (repo §1.4 F3-isolation protocol): every failure previously attributed to semantic
reference-matching — including our own ledger's ⋈ bucket and the earlier census finding that 43–52% of wrong
picks occurred despite the gold record having been retrieved [M] — is re-probed with A_minimal/B_fullctx.
Report: N, flip rate to load, split {transcription slip / self-anchor / ambiguity (multi-candidate under
information given) / genuine semantic failure (A_minimal fails)}, per-model. Expected deliverable: the first
measured decomposition of "wrong entity" into mechanism classes; the taxonomy-overattribution claim (§1
contribution 3) stands or falls here.⟧

## 7. Mechanism-derived mitigation

### 7.1 Design

If minimal context solves the selection 9/9, the fix is to *manufacture minimal context at the decision point*:
at write time, for declared reference parameters, re-run the selection as an isolated sub-call whose prompt
contains only (i) verbatim customer messages, (ii) the producer listing, (iii) the pending action's other
arguments — pointedly excluding the agent's own prior mappings, which §5 identified as the corrupting content.
The sub-call's answer is accepted **only if it exists verbatim in the producer listing** (fabrication-proof);
on disagreement the argument is substituted in place; UNSURE or parse failure is a no-op. The conversation is
unchanged; the repair is silent and bounded (per-episode cap).

This differs from re-asking/self-correction (which operates in the polluted context and is known to fail, §2)
and from disambiguation sub-calls that include the full transcript — the essence is *removing* self-generated
mappings, not adding deliberation.

### 7.2 Validation

Unit probes (stubbed sub-call): 8/8 — wrong→gold substitution, correct→keep, UNSURE no-op,
out-of-listing-answer rejection, no-listing no-op, non-target tool no-op, multi-hit conservatism, verdict
memoization. Live (run R21, one episode per task): 16 firings — **keep 8** (the agent's pick was correct this
run and the sub-call confirmed it every time: zero false switches, the safety half of the design validated
[S]), **unsure 8** (in the 8-item task the per-call mapping question under-determined the item; conservative
no-op), **switched 0** — corrective efficacy not yet demonstrated live, because (i) the correctable error did
not recur on the checked calls this run, and (ii) same-value re-checks exhausted the per-episode cap before the
later calls (including two further wrong-record ids) were reached. Three fixes derived and unit-verified (verdict
memoization; multi-hit answers treated as UNSURE; an itemized-mapping instruction); in R22 the lever fires
`switched`/`memo-switch` live ⟦TBD: R22 forensic — does the switch recover the gold id end-to-end, Δspurious⟧.
Notably, the same wrong-record id recurs across independent runs — the wrong *binding*, like the slip that
seeds it, is an attractor, not noise. [S]

### 7.3 Why gating alone fails: the deny/pass-through dilemma

We tried the gate route first, and document its two-sided failure as a finding. With hard denial (an evidence
gate refusing writes whose reference value lacks listing support), the R19/031 episode burned all 8 denials —
the agent, holding a wrong binding, could not produce the right value, mis-explained the denials to the
customer ("your digits are not recorded in our system"), and the episode collapsed to a human transfer. With
pass-through (per-turn re-check limits, needed to avoid deadlock), the R20/031 episode committed the polluted
write. Denial assumes the model can repair a binding it cannot see is wrong; pass-through trusts it. Both lose:
verification without re-derivation cannot fix binding errors — which is why the mitigation above re-derives in
clean context instead of arguing with the model. [S: both live episodes logged]

## 8. Discussion

**For taxonomies.** Observational labels are hypotheses, not measurements. Our case would be scored
"wrong entity / semantic" by every schema we surveyed; isolation replay flips it. Where teacher-forcing
evaluation is used, self-anchoring is structurally excluded from measurement; where labeling is post-hoc,
transcription slips are indistinguishable from semantic confusion. We propose per-failure isolation replay as
the analogue of an ablation control: any capability attribution that has not survived an information-matched
minimal-context replay is provisional. ⟦Strength of final claim pends §6 and direct re-check of MAST/TRAIL/τ²
appendices.⟧

**For scaffolds.** The lever hierarchy shifts: evidence gates (existence checks) close fabrication-from-thin-air
but not binding; binding requires either re-derivation in clean context (§7) or deterministic matching when the
selection criteria are formalizable. The dilemma of §7.3 suggests a general design rule: verification gates
should escalate to *re-derivation*, not to repeated denial.

**Relation to companion work.** Our companion papers analyze same-clause interference in batched judgments with
an associative-memory account; the present paper stays behavioral. The shared thread — correlated context
content corrupts a targeted retrieval/binding — differs in unit (record rows vs policy clauses) and in
deliverable (attribution methodology vs mechanism). PI-LLM is a shared citation; we differentiate explicitly.

**Limitations.** Single domain and model family until §6 lands; B_fullctx uses flattened serialization (the
live path interleaves tool-call structures); the isolated re-pick mitigation assumes a recoverable producer
listing in context; RL-trained reasoning models may re-verify within trace and need separate measurement;
rival preprints (§2) are recent and unreplicated — framing overlap is acknowledged, numbers are not relied on.

## 9. Reproducibility

All trajectories (results.json + logs, gzipped), the isolation-probe script with its committed input artifacts
(the reconstructed listing, customer messages, and trajectory prefix), the mitigation implementation and its
unit tests, and the evidence ledger are in the repository. The §5 result is reproducible by re-running the
probe against a local vLLM endpoint (no benchmark re-runs needed); the full per-run probe transcripts are
Appendix A ⟦TBD — to be regenerated and committed alongside this draft⟧.

---
### Appendix A ⟦TBD⟧: full probe transcripts (A_minimal / B_fullctx, 18 runs)
### Appendix B ⟦TBD⟧: E-F3-ISO per-failure table
### Appendix C: gate-interplay episodes (R19/031 denial collapse; R20/031 pass-through) — trace excerpts
