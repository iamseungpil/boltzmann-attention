# [DRAFT v0 · 2026-07-24] Not a Semantic Error: Isolation Replay Dissociates Transcription Slips and Self-Anchoring from Capability Limits in LLM Tool Agents

> Working draft (English, markdown → LaTeX later). Placeholders marked ⟦TBD⟧.
> Evidence ledger: RESEARCH_MASTER §3 C121/C124/C125/C126 (case 1: 039) · C135–C157 + RESULTS_043_INVESTIGATION_2026_07_25.md
> (case 2: 043) · sim_results/bank_rall19*–rall25*/nt4 · probe_039_join_iso.py. Grades follow the repo convention
> ([S]=trajectory-verified, [M]=measured with stated caveats).
> Related-work map: RELWORK_ISOLATION_ATTRIBUTION_2026_07_24.md (105-agent verified survey) + C145 (106-agent, case-2 lineage).
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
the episode, pass-through pollutes the database. Mitigation follows the mechanism, and lands on determinism: an isolated
minimal-context LLM re-pick of reference arguments is safe in one live run (0/16 false switches) but proves
two-sided in the next — in clean context it can still mis-bind, once switching a *correct* id to a wrong one —
so the robust design removes the LLM from id emission entirely: a deterministic merchant-absence verifier flags
9/9 wrong picks with 26/26 gold passes in engine replay over all recovered trajectories, and a
formalize-then-match repair (LLM emits criteria, code emits the id) closes the residual. A retro-audit of the
recoverable wrong-reference failures (Phase 1, four instances) shows a dose-response — minimal context recovers
the gold set, accumulated context reintroduces the slip, the full trajectory collapses — with 3/4 flipping
cleanly to load; the one residual slip is itself closed by the deterministic matcher (population flip-rate
deferred to Phase 2 ⟦TBD⟧). A second,
independent case study shows the method generalizes beyond binding: a task failure that presents as a
planning/capability deficit (the agent closes a card before running the retention procedure whose policy it had
*already retrieved*) flips to load under the same control — 6/6 correct in clean context vs 0/6 under the
buried-policy context — and yields a graded prescription ladder (pre-action reasoning 3/6, source-document
resurfacing 4/6, clean isolation 6/6), with a different repair than the binding case: resurfacing the source
policy document recovers a policy-conditioned write from 0/6 to 5/6 live-context completions, where resurfacing
a compact plan name yields only wrong-meaning parroting — mirroring, and positively extending, the negative
plan-resurfacing result of prior work. Along the way the method repeatedly corrected *our own* expert
attributions (six refuted surface labels across the two investigations). We argue that agent failure taxonomies
built on observational labeling systematically overattribute context-induced load to semantic capability, and
that per-failure isolation replay should be a standard attribution control.

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
   is, to our knowledge, unclaimed. A second case (§5.5) shows the control generalizes across failure families
   — a binary flip for a binding slip, a graded ladder for a buried-policy execution failure — and that it
   audits the auditor: six of our own surface attributions were refuted by it before the true mechanism held.
3. **An audit**: re-attribution of failures previously labeled semantic-reference errors. Phase 1 (all four
   recoverable wrong-reference instances from three independent runs): a dose-response over three context
   levels — 3/4 flip cleanly to load, one residual slip persists even in minimal context and motivates the
   deterministic matcher (§6). ⟦TBD Phase 2: the full census (incl. the 43–52% "wrong pick despite gold
   retrieved" bucket [M]) and a second domain — population flip-rate reported only then⟧.
4. **A mitigation arc that ends in determinism**: an isolated LLM re-pick of reference arguments is validated
   safe (unit 8/8; live 0/16 false switches) but proves two-sided in a further live run (it can switch gold to
   wrong: the sub-call is still an LLM emitting an id). The robust design removes the LLM from id emission —
   a merchant-absence verifier (engine replay: 9/9 wrong picks flagged, 26/26 gold passes) plus
   formalize-then-match repair (§7). We also document why deny-style evidence gates cannot close this failure
   class: the deny-forever/pass-through dilemma (§7.3), and a repair-matching rule across failure families
   (§7.4).

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
exhibit trained within-trace re-verification). For our second case (§5.5), the decisive near-prior is
Plans-Don't-Persist [arXiv:2606.22953]: plan-following failures are representational (per-action decay
$\sim$4.1$\times$ as the plan recedes in context), and — their negative result — resurfacing the *stale plan*
does not repair them. Our delta is to turn that negative into a positive: resurfacing the *source policy
document* (not the plan) recovers the policy-conditioned action (0/6 → 5/6), with deterministic extraction of
the already-retrieved document rather than LLM re-summarization.

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
listing and filed individually; case 1), task_031 (single dispute, card-digit acquisition via user-side tool),
task_043 (card-closure request whose gold path is a policy-prescribed retention flow; case 2, §5.5), plus
task_054 in the same runs, not central here.

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

### 5.5 A second case: an apparent planning failure is buried-policy load (task_043)

The binding slip of §4–5 could be dismissed as one failure family, so we applied the same control to a failure
with a completely different surface. In task_043 the customer asks to close a Platinum card; the gold behavior
is a *retention* flow (offer the annual-fee waiver the policy prescribes; the customer accepts; the card stays
open). Across four live trials the agent passes 0/4, dominantly by **closing the card before the retention
procedure** — despite having *already retrieved* the policy document that prescribes the waiver step. Scored
observationally this is a planning / instruction-following capability failure, and our own ledger tried, in
sequence, plan-capability, scale-invariant capability, missing-information, salience/position, and
lost-in-the-middle labels — **five surface attributions, each refuted by the next probe** (the first probe's
context had lacked the procedure; retrieval was in fact clean; position was irrelevant, presence dominated) —
before isolation settled it.

**Attribution.** Information-matched isolation at the pre-close decision (policy text 1.3KB, extracted verbatim
from the trajectory's own retrieved document): clean context + policy ⇒ **retain 6/6** (the model infers the
waiver, plans, and acts); the actual polluted context (~27KB), neutral prompt ⇒ **0/6** (reproducing the live
close); same polluted context + a pre-action *reasoning* instruction ⇒ 3/6; + the source policy document
*resurfaced* ⇒ 4/6. A graded ladder — clean 6/6 → resurfaced 4/6 → reasoning 3/6 → neutral 0/6 — rather than
§5's binary flip: the failure is load (policy buried under the trajectory), not capability, and partial context
repair buys partial recovery. [M: single decision, n=6 per condition, single trial per rung]

**Prescription, and another aggregate trap.** At a live pre-close cut, the policy-conditioned write (apply the
waiver flag) goes 0/6 (no resurfacing) → 3/6 (resurfacing a compact *name* of the pending step) → **5/6**
(resurfacing the *source policy document*). String counts understate the gap: read raw, the compact-name arm's
mentions are wrong-meaning parroting (the flag misread as "mark account closed"), while the document arm's are
correct policy inferences (annual fee + tenure ⇒ waiver). Deployed live in a matched pair, the resurfacing
mechanism fires as designed — the waiver action executes with gold-exact arguments in the treated arm (base:
never called), and after fixing two scaffold-induced bugs the treated arm stops closing the card where base
still closes — the causal chain from resurfaced document to corrected action holds end-to-end, though the
episode still misses the benchmark's strict full-DB-hash pass (§8). [S: matched pair, single episode]

**What this adds.** (i) The control is not specific to reference binding: it dissociates load from capability
for an execution-ordering failure too. (ii) Its output is not always binary; graded recovery localizes how much
damage each repair removes. (iii) The repair differs by family — and resurfacing the *source document* is what
works, where resurfacing a compact plan fails (wrong-meaning parroting), consistent with the negative
plan-resurfacing result of Plans-Don't-Persist (§2) and extending it positively. (iv) It corrects the
investigator: the five refuted labels above, plus a sixth during live forensics (a payment step first blamed
for the residual failure was gold-identical; the real residual was an over-action our own scaffold spec had
induced, which we fixed and verified causally), are internal evidence for this paper's thesis — observational
attributions, including ours, do not survive counterfactual probing.

## 6. Retro-audit of "semantic" failures (E-F3-ISO, Phase 1: banking)

We re-probe every wrong-reference failure we can recover from three independent live runs (rall19/20/21, the
`file_dispute` wrong-`transaction_id` cases), as a **dose-response**: three information-matched context levels
of increasing breadth — **S** (the single customer message enumerating the disputed items, id-redacted) +
listing; **A** (all customer turns, id-redacted) + listing; **B** (the full agent trajectory prefix up to the
decision). Greedy + samples, 32B, at the trajectory's settings. Two of our own earlier characterizations were
overturned in the process and are reported as corrections (§4.4): **zero fabrication** (all wrong ids are real
listing records; `true-fab = ∅` in every instance), and **user-simulator laundering** (the wrong id appears in
a *customer* turn in 3 of 4 instances — the agent's error reflected back).

**Results (4 instances).**

| Instance | S (single-msg) | A (multi-turn) | B (full traj.) | Reading |
|---|---|---|---|---|
| r19.039 | 8.0/8, wrong-free | 4.3/8 | 0/8 | textbook dose-response — **LOAD** |
| r20.031 | 1/1 | 1/1 | 0/1 | clean flip — **LOAD** |
| r21.039 | 6.7/8, wrong-free | 6.0/8, wrong-free | 1.0/8 | wrong id never appears in clean context — **LOAD** |
| r20.039 | 7/8, **slip persists** | 6.3/8, slip | 5/8, slip | one item wrong even at S — **residual** |

(gold recovered / gold total, mean over runs.) Three of four flip cleanly to load: the minimal-context probe
recovers the gold set wrong-free and the trajectory collapses; r19.039 shows the dose explicitly (8 → 4.3 → 0).
The load threshold is **low** — accumulated multi-turn context alone (level A, with the agent's own ids
redacted) already reintroduces the slip in the 8-item cases. This is consistent with, and localizes, the
multi-turn degradation reported by Laban et al. (§2): the corrupting content is not only the agent's explicit
mappings but the conversational accumulation itself.

**The residual (r20.039) is the argument for a deterministic matcher, not against load.** Here even the
single-message isolated re-pick reproduces the item-3 slip (Home Depot's id for the Costco item). But the
customer's stated attributes uniquely identify the record: `{merchant=Costco, amount=$267.34}` matches exactly
one listing row (the gold `…5db822`) and never the slip row (`…d2d06d` = Home Depot, $456.78) — verified
directly against the listing, in both r19 and r20. An LLM re-pick, even in clean context, can still transcribe
the neighbor's id; a *formalize-then-match* step (LLM emits the criteria, deterministic code emits the id)
cannot — it closes the residual the isolated re-pick leaves open (§7.1, and the lever in §7 is being upgraded
accordingly).

**Scope and honesty.** Phase 1 is 4 instances in one domain — enough to establish the dose-response and the
mechanism split (load / laundering / residual), not enough for a population flip-rate. The headline
overattribution number (what fraction of the ⋈/"reference-matching ceiling" bucket flips) requires Phase 2:
the full census (the earlier 43–52% "wrong pick despite gold retrieved" finding [M]) run through this protocol,
and the retail domain. We do not report a population percentage until then. Verdict logic note: our first
auto-classifier under-counted load (it keyed on the trajectory reproducing the *specific* wrong id, but B
often fails by emitting other/empty ids); the readings above use the correct signal — S wrong-free and
S-gold ≫ B-gold.

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

**The strongest form removes the LLM from id emission entirely.** The transcription slip is, at bottom, an LLM
copying an id and landing one row off. A *formalize-then-match* step eliminates it: the LLM emits only the
selection *criteria* (the merchant and amount the customer stated — a reading-comprehension task it does well),
and deterministic code emits the id of the uniquely matching row. On our four audited instances this closes the
residual the LLM re-pick leaves open: `{merchant=Costco, amount=$267.34}` matches exactly one listing row (the
gold `…5db822`) and never the slip row (`…d2d06d` = Home Depot $456.78) — including r20.039, where the isolated
LLM re-pick still slipped (§6). Even a purely deterministic *verifier* — no LLM at all — is strikingly
effective as a slip detector: flag any filed dispute whose record's **merchant was never mentioned by the
customer**. Across the four instances this catches **8/8** wrong picks (every slip landed on an unmentioned
merchant: Home Depot, Facebook Ads, Airport Parking, Panera, W.B. Mason, Spotify) with **0/25** false blocks on
gold — because the customer's actual merchants (Costco, Amazon, Marriott) always appear. (Adding an amount
constraint raises false-block risk when the customer gives an approximate figure — "around $160" for a $167.34
charge — so merchant-absence is the robust deny signal; the amount dimension is needed only to disambiguate
*within* a repeated merchant, the harder match that the formalize-then-match repair handles.) This is the
deterministic endpoint of the isolation idea: the load never touches the id at all. We wired this verifier into
the live generation-level gate stack (deny a `file_dispute` whose record's merchant is unmentioned; matching on
significant name tokens, robust to "Marriott Hotels" vs "Marriott hotel"); replayed through the actual engine
path over all recovered trajectories (rall19–22) it flags **9/9** wrong picks — including the exact Facebook-Ads
switch the LLM re-pick caused in R22 — with **26/26** gold passes (zero false blocks). [S: offline + live-engine
replay over committed trajectories; end-to-end live run pending.]

### 7.2 Validation

Unit probes (stubbed sub-call): 8/8 — wrong→gold substitution, correct→keep, UNSURE no-op,
out-of-listing-answer rejection, no-listing no-op, non-target tool no-op, multi-hit conservatism, verdict
memoization. Live (run R21, one episode per task): 16 firings — **keep 8** (the agent's pick was correct this
run and the sub-call confirmed it every time: zero false switches, the safety half of the design validated
[S]), **unsure 8** (in the 8-item task the per-call mapping question under-determined the item; conservative
no-op), **switched 0** — corrective efficacy not yet demonstrated live, because (i) the correctable error did
not recur on the checked calls this run, and (ii) same-value re-checks exhausted the per-episode cap before the
later calls (including two further wrong-record ids) were reached. Three fixes were then derived and unit-verified (verdict
memoization; multi-hit answers treated as UNSURE; an itemized-mapping instruction), and R22 ran the lever with
them enabled — where the isolated LLM re-pick **actively harmed**. In R22/task_031 (a Marriott dispute; gold =
the Marriott $167.34 record), the sub-call switched the *correct* Marriott id to the Facebook Ads $203.58 record
— a merchant the customer never mentioned — and memoized that switch, so the episode filed the wrong id. The
task_039 improvement in the same run (the gold Costco id filed for the first time) was **not** the lever's doing
(it returned UNSURE on all seven 039 calls; the agent simply picked better this run — run-to-run variance).
Net R22 verdict, by forensic (not aggregate): the LLM re-pick's corrective efficacy is **unreliable and
two-sided** — it can move a wrong id to gold, but it can also move gold to wrong (Δspurious > 0), because the
sub-call is still an LLM emitting an id and can itself mis-bind. [S]

**This is the decisive argument for the deterministic endpoint (§7.1).** The exact harm the LLM re-pick caused —
switching to Facebook Ads for a Marriott dispute — is precisely what the merchant-absence verifier prevents:
Facebook Ads is unmentioned (blocked as a target), Marriott is mentioned (the gold is protected). An LLM in the
loop, even in clean context, retains a failure mode that a deterministic merchant check does not. We therefore
do **not** recommend shipping the LLM re-pick as a corrective switch; the deterministic verifier (slip detection,
8/8, 0/25 false blocks, §7.1) plus a formalize-then-match repair for within-merchant disambiguation is the
robust design. The honesty note is itself a finding: R22's aggregate ("5 corrective firings; 039 improved to
6/8") reads as success and would have been mis-reported as such without per-switch forensics — the same
observational-attribution trap this paper is about (§8), turned on our own mitigation.

### 7.3 Why gating alone fails: the deny/pass-through dilemma

We tried the gate route first, and document its two-sided failure as a finding. With hard denial (an evidence
gate refusing writes whose reference value lacks listing support), the R19/031 episode burned all 8 denials —
the agent, holding a wrong binding, could not produce the right value, mis-explained the denials to the
customer ("your digits are not recorded in our system"), and the episode collapsed to a human transfer. With
pass-through (per-turn re-check limits, needed to avoid deadlock), the R20/031 episode committed the polluted
write. Denial assumes the model can repair a binding it cannot see is wrong; pass-through trusts it. Both lose:
verification without re-derivation cannot fix binding errors — which is why the mitigation above re-derives in
clean context instead of arguing with the model. [S: both live episodes logged]

The second case (§5.5) sharpens the soft/hard distinction. There, a pre-close deny-and-resurface gate *fires*
and is simply overridden by the agent across trials; a deny-and-regenerate dispatch gate makes the agent give
up the goal instead of rerouting; explicit prompt rules lose to a knowledge-base "Use X" prior. All of these
are **soft** — they depend on the model complying. The interventions that moved behavior were of a different
kind: content repair (resurfacing the source document into the context the model actually reads) and, for
format-level slips (out-of-schema tool names hallucinated from KB text), constrained decoding — enforcement in
the *decoder*, not the conversation. Gates advise; context and decoding decide.

### 7.4 Matching the repair to the failure family

The two cases converge on one design rule with two instantiations. For **reference bindings** (case 1), the
selection criteria are formalizable, so the endpoint is deterministic: verify by merchant-absence, repair by
formalize-then-match — the LLM never emits the id (§7.1–7.2; the R22 harm case is the proof that even
clean-context LLM re-picks retain the slip mode). For **policy-conditioned actions** (case 2), the decision is
not formalizable without curating domain policy into the scaffold, so the repair must stay content-side:
deterministically *resurface the source document* at the decision point and let the main model decide in full
context. Full isolation is the attribution instrument, not the repair, in this family — an isolated sub-agent
deciding a broad action risks exactly the context-loss harm R22 exhibited for bindings, and resurfacing a
compact plan instead of the source fails by wrong-meaning parroting (§5.5). In both families the common core
is: do not argue with the model in the polluted context; either take the decision away from the LLM
(formalizable case) or repair the context it decides in (non-formalizable case).

## 8. Discussion

**For taxonomies.** Observational labels are hypotheses, not measurements. Our case would be scored
"wrong entity / semantic" by every schema we surveyed; isolation replay flips it. Where teacher-forcing
evaluation is used, self-anchoring is structurally excluded from measurement; where labeling is post-hoc,
transcription slips are indistinguishable from semantic confusion. We propose per-failure isolation replay as
the analogue of an ablation control: any capability attribution that has not survived an information-matched
minimal-context replay is provisional. ⟦Strength of final claim pends §6 and direct re-check of MAST/TRAIL/τ²
appendices.⟧

**For scaffolds.** The lever hierarchy shifts: evidence gates (existence checks) close fabrication-from-thin-air
but not binding; and deny-style gates in general proved *soft* — fired, then overridden or deadlocked (§7.3).
What worked was either removing the LLM from the formalizable decision (deterministic verify/match, case 1) or
repairing the context the LLM decides in (source-document resurfacing, case 2) — §7.4's rule: verification
should escalate to re-derivation, and re-derivation must be scoped to the failure family.

**Relation to companion work.** Our companion papers analyze same-clause interference in batched judgments with
an associative-memory account; the present paper stays behavioral. The shared thread — correlated context
content corrupts a targeted retrieval/binding — differs in unit (record rows vs policy clauses) and in
deliverable (attribution methodology vs mechanism). PI-LLM is a shared citation; we differentiate explicitly.

**Limitations.** Single domain and model family (banking, Qwen2.5-32B) pending Phase-2 audit and a second
domain; B_fullctx uses flattened serialization (the live path interleaves tool-call structures); the
deterministic verifier and formalize-then-match assume a recoverable producer listing and formalizable
criteria; case 2's probe rungs are n=6 single-trial and its reasoning-instruction rung is mildly leading;
case 2's live episode still fails the benchmark's strict full-DB-hash metric even with the corrected action —
that metric requires the full gold read-set through the dispatcher path and zero over-actions (9/97 tasks pass
it under our best stack), so per-step action match may be the more informative signal, a benchmark-metric
question we flag rather than resolve; RL-trained reasoning models may re-verify within trace and need separate
measurement; rival preprints (§2) are recent and unreplicated — framing overlap is acknowledged, numbers are
not relied on.

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
### Appendix D ⟦TBD⟧: case-2 (task_043) probe ladders, resurfacing-arm completions (wrong-meaning parroting vs
policy inference), and the matched-pair (base vs treat) traces
