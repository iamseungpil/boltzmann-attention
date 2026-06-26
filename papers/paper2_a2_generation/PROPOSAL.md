# Paper 2 — Learning a Small, Swappable NL-Policy → GATE_SPEC Compiler with a Fixed Verifier as Runtime Contract

*Research proposal draft (ICLR-style). Claims are marked `[SETTLED]` (measured in our runs / settled prior work) or `[ESTIMATE]` (pre-registered prediction, not yet measured). Estimates are not results.*

---

## 1. Problem & motivation

Tool-using LLM agents that must obey a domain policy (authenticate before write, confirm before a mutating action, deny out-of-scope requests, enforce eligibility/limit conditions) need that policy in a **machine-enforceable** form. In our system this is the **A2 artifact**: a per-domain `GATE_SPEC` — a structured object of gates, each with a `predicate`, `satisfiers` (tool → required inputs), `applies_to` scope, and `terminal`/`ask` disposition — that a **deterministic verifier** can replay against an agent trajectory to guarantee compliance.

Across the portfolio, A2 is the **one component that must be authored per domain**. The bench adapters A1/A4/A5 (tool-catalog → enum schema, domain → LODO split, output → guided grammar) are mechanical, and A3 (evaluator → reward) is wrapping. A2 — NL policy → constraint structure — is the sole research-grade authoring cost `[SETTLED, our framework]`. It is the "common enemy" cost that Papers 1 and 4 treat as *given*. **Paper 2's contribution is to make A2 cheaply and automatically**: an NL-policy → `GATE_SPEC` compiler that a small, on-prem model can run.

**Why a learned small model + verifier rather than a frontier prompt?** The honest answer is *not* raw accuracy. A frontier model in one shot already compiles policies well: in our `P-A2-0` run, a frontier model compiled an unseen 166-line airline policy into a 6-gate spec with **over-deny 0/108** on gold-trajectory replay `[SETTLED]`. The case for a learned small compiler rests on two axes the frontier prompt does not serve:

1. **Sovereignty / on-prem.** Compliance compilation often runs where the policy and customer data cannot leave the premises (air-gapped / regulated). A small model that runs locally is a deployment requirement, not an accuracy play.
2. **Cross-domain transfer (LODO).** We want a generator that *generalizes the act of formalization* to new domains via leave-one-domain-out, not one that memorizes a domain.

This reframes the reviewer's mandatory question — "why not just prompt a frontier model (Prose2Policy-style)?" — onto sovereignty + transfer, and away from accuracy `[SETTLED, positioning]`. Critically, the **verifier is a fixed program** and never learned (it is the same artifact used for live enforcement and for scoring), so the generator is a **swappable part**: parser (L0), frontier one-shot, or trained small model can each plug into the same contract.

---

## 2. Approach

### 2.1 Reverse-rendering data engine (closing the data famine)

Real `(NL, spec)` pairs exist for only ~22 domains. We manufacture training data so the **ground truth is correct by construction**:

1. **Spec sampler (a program).** Sample `GATE_SPEC`s from the grammar: predicate kinds (auth / confirm / scope / limit / time-window …), satisfier tool signatures (with a co-generated fake tool catalog), `applies_to` combinations. Difficulty knobs = gate count, cross-reference depth, exception-clause count.
2. **Frontier NL renderer.** A frontier model renders each spec into policy prose across **K styles** (formal-md / casual-prose / bullet-terse / legalese / Korean / mixed-crossref). Because the **spec comes first**, the GT is exact and no verifier is needed for the synthetic pairs.
3. **Round-trip filter.** Re-compile the rendered NL and require it to match the source spec (gate-recall / applies-F1 / kind-match) as a data-cleanliness gate. Contamination control: rendered NL must not surface spec terms verbatim (alias masking) — style diversity is the actual mechanism for narrowing the synthetic→real gap.

The loop (sampler → render → round-trip filter) is closed and demonstrated: self-sanity 1.0 and KEEP 2/2 on a frontier re-compile seed `[SETTLED]`.

### 2.2 Training ladder (pre-registered)

| Stage | Method | Gate |
|---|---|---|
| **S0** synthetic SFT | LoRA SFT + guided (spec JSON schema) | held-out synthetic **structure-EM** + real-domain replay ≤ frontier one-shot |
| **S1** real-domain verified distill | frontier compiles real policies; **only replay-passing specs continue into SFT** | **LODO** replay-verified gate-F1 |
| **S2** on-policy DPO | self K-samples → verifier-scored → (pass, near-miss) pairs; contrast axis = **structural correctness** (length is de-confounded by the fixed schema) | K=1 accuracy ↑, equal after verifier-selection |
| inference | K-sample + verifier-select; if all fail → **abstain → HITL** | risk-coverage |

### 2.3 Verifier: deterministic replay + cross-stage faithfulness

- **Level-2 (behavioral) replay** — Guard-2-isomorphic: replay gold trajectories and require over-deny = 0 ∧ order-violation = 0. This is the acceptance criterion and is permanently a program.
- **Level-3 (faithfulness / entailment)** — replay has a blind spot: a *fabricated* gate that happens never to fire on the gold trajectory passes behavioral replay. We add a **cross-stage entailment check**: each gate's NL gloss must be entailed by a clause of the source policy → SUPPORTED / FABRICATED (no supporting clause) / UNCERTAIN. Any FABRICATED/UNCERTAIN routes the spec to abstain→HITL. This is **orthogonal and additive** to replay, and directly addresses the published "compile-pass ≠ NL-faithful" risk (FormalAlign; "Do LLMs Game Formalization?"). VeriEquivBench marks the ground-truth-free SOTA ceiling (not implemented).

---

## 3. Results so far

**`[SETTLED]` (measured in our runs):**

- **S0 emergence.** Scaling the synthetic SFT set 135 → 200 pairs (plus sampler-diversity) caused exact structure reproduction to **emerge**: held-in **structure-EM 0/20 → 14/20 (70%)**, canonical-EM 0 → 8/20 (40%), gate-recall 0.248 → 1.000, applies-F1 0.982 → 0.997. (At 135 pairs structure-EM was still 0; the curve is steep but G-A2-1's ≥90% target is not yet met.)
- **S1 rejection (synthetic→real gap).** Held-in synthetic gains do **not** transfer: synthetic-only scaling *worsened* real airline transfer (applies-F1 0.876 → 0.528), and an S1 smoke with 2 real domains × oversample-8 failed its pre-registered bar (airline applies-F1 stayed at 0.528, n_gates collapsed 5 → 1). Interpretation (as pre-registered): real-domain *diversity*, not oversampling, is what's missing → motivates the teacher-pool dose-response.
- **S1-diag (gap decomposition).** Adding 60 concrete-predicate synthetic pairs (program-instantiated predicates/db_checks/prose) lifted airline applies-F1 **0.528 → 0.704 (+18pp)** — the **concreteness gap** (synthetic `"<...>"` placeholders vs. real concrete conditionals) is a confirmed *part* of the loss. But **gate-count shrinkage persists** (n_gates = 1): imitating few-gate real specs teaches the model to emit *fewer* gates — a distinct, first-class bottleneck independent of concreteness.
- **Faithfulness check closes the replay blind spot.** On a real retail spec with an *injected* fabricated gate (G9_LOYALTY_TIER), the entailment judge returned 3 SUPPORTED + **G9 FABRICATED (conf 1.0)** and routed to ABSTAIN, while a clean spec returned all-SUPPORTED with zero false alarms → demonstrated closure of the behavioral-replay blind spot.

**`[ESTIMATE]` (pre-registered, not yet run):**

- S0 reaching G-A2-1 (structure-EM ≥ 90%) via further scale + style expansion.
- S1-v2 dose-response monotonic increase with real-domain count, recovering base transfer at 6+ domains.

---

## 4. Planned experiments

1. **P5 teacher-pool dose-response.** Stratify real-domain exposure to **1 / 3 / 6 / 9** domains (4-arm), training on a large teacher-pool's verified specs; measure the curve "real-domain count → airline transfer (applies-F1)." A curve (not a single pass/fail) answers *how much* diversity is needed even on failure. Pre-registered: monotone increase ∧ base recovery (≈0.815) at 6+ domains `[ESTIMATE]`. Harness ready (`--real` variable); fires on P5 spec arrival.
2. **22-domain verified-distill + LODO.** Frontier compiles all real domains → replay filter → train on N−1, evaluate held-out domain by replay-verified gate-F1. This is the central transfer claim.
3. **Three-compiler table.** Same fixed verifier, same domains (retail + airline → 22 LODO), three swappable generators: **L0 parser / frontier one-shot / small K-sample + verifier-select**. Pre-registered headline: **small (K=8 + verifier-select) ≥ frontier one-shot** on held-out domains (replay pass-rate, gate-F1) `[ESTIMATE]`. Grounding for plausibility: precision reduces to a search problem under K-selection, and StepFun-Formalizer shows a 7B formalizer beating frontier reasoners in its (in-domain) regime.
4. **S2 on-policy DPO** → close the K=1 gap, completing the front-end headline.

---

## 4.5 Related work — NL → symbolic-rule / spec generation (the generator lineage)

Beyond the policy-compiler rivals (§5), an established lineage learns to compile **NL → executable symbolic rules**, mostly in cognitive architectures and neurosymbolic planning. Our generator inherits this shape (LLM proposes a formal spec, a symbolic engine grounds/verifies) and differs in target (policy **GATE_SPEC**, not planning rules), in *verified-distillation + reverse-rendering* data, in *cross-domain LODO transfer*, and in the *fixed verifier as runtime contract*.

- **NL2CA** (arXiv 2512.18189) — the **direct precedent for a *small* NL→symbolic-spec generator**: a fine-tuned **Qwen3-0.6B** compiles NL → LTL → an unsupervised Critic-Tree → `pyactr` (ACT-R) productions, **fully automatic, zero human**. Establishes that a sub-1B model can compile NL to a formal rule artifact (same Generator–Critic shape as NL2GenSym). Our delta: policy-compliance spec (not ACT-R), cross-domain transfer, runtime-contract verifier, abstain-on-fabrication. *(NL2CA is also the direct precedent flagged in our authoritative related-work synthesis for this exact "NL→A2 generator" task.)*
- **NL2GenSym** (arXiv 2510.09355) — LLM Generator → **SOAR production rules**, execution-grounded Generator–Critic with a RAG self-evolving KB; small+framework beats large one-shot. Our delta: A2-swap transfer + learned empty→abstain (NL2GenSym iterates to success/timeout, no abstention).
- **Bootstrapping cognitive agents** (Zhu & Simmons, arXiv 2403.00810) — GPT-4 bootstraps SOAR-syntax productions → symbolic replay/critic/utility verification; runtime = deterministic-if-production-exists / LLM-fallback. The closest precedent for "deterministic-first + verified rule authoring," but large-model, ad-hoc authoring, in-domain growth (no transfer).
- **MERLIN2 / grammar-constrained NL→PDDL** (arXiv 2309.14945) — GBNF grammar-constrained decoding forces structured NL→PDDL output, validated by a symbolic planner — the nearest analogue of our guided-JSON *type*-forcing (we force type, leave content to the model, and verify behavior + faithfulness separately).
- **Solver-aided policy compliance** (arXiv 2603.20449) — NL policy → SMT-LIB → Z3 runtime gate; closest on the *NL→formal-policy* axis, but **human-guided SMT translation** (the cost we automate) and gate-only (no transfer/distillation).
- **Intermediate Languages Matter** (arXiv 2502.17216) — the choice of formal IR is the first-order decision variable in NL→formal; supports our spec-schema design and the round-trip cleanliness filter.

## 5. Expected contribution & positioning

**The unoccupied 4-way intersection.** No single published work occupies: **(swappable small generator) × (fixed verifier as a *runtime contract*) × (verified distillation) × (SOP/policy domain + sovereignty)**. The nearest analogs each miss most of it:

- **Prose2Policy** (2603.15799, Apple — NL → Rego): the closest published task, but **frontier-prompt-only; no distillation, no transfer, no sovereignty.** Honest denominator: its headline 95.3% compile is *post-filter* (371/389 after a ~20% reject of 485→389) ⇒ ~76.5% of raw input; determinism is future work. This is exactly the "why not just prompt" baseline — and exactly what our sovereignty+transfer framing answers.
- **AgentSpec** (2503.18666): 2nd-nearest; o1 rule-recall 70.96%. Cited as a necessary frontier-prompt comparator.
- **StepFun-Formalizer** (2508.04440, AAAI'26): the **only validated small-formalizer precedent** (7B/32B dual-stream distill + RLVR), where the **7B beats o3-pro / Claude-4-thinking / R1-671B and 7B ≈ 32B** — our strongest external evidence that "small generator ≥ frontier" is a *distillation-deficit*, not a *capacity-ceiling*, problem. But it is math-only and in-domain; **cross-domain transfer is our contribution.**
- **FormalAlign / "Do LLMs Game Formalization?" / VeriEquivBench**: motivate and bound the level-3 faithfulness check (compile-pass ≠ faithful; ground-truth-free equivalence as the SOTA ceiling).

**Contributions:** (i) a reverse-rendering data engine with construction-guaranteed GT and a round-trip cleanliness filter; (ii) a three-stage verified-distillation ladder (S0 synthetic → S1 real verified-distill → S2 on-policy DPO) with pre-registered gates; (iii) a fixed deterministic verifier reused three ways (enforce / measure / GT) plus an orthogonal cross-stage faithfulness check that provably closes the replay blind spot; (iv) the three-compiler comparison establishing small-generator-≥-frontier-one-shot *under transfer*, with the value proposition correctly located on sovereignty + LODO rather than raw accuracy.

---

## 6. Relation to Papers 1 and 4

- **Paper 1 (theory: function × scale × lever).** Paper 1 treats A2 as a measurement instrument and asks what scale buys vs. what is scale-invariant (e.g., compliance/guarantee). Paper 2 *produces* the A2 that Paper 1 measures around, and supplies the empirical fact that the **compliance guarantee comes from the fixed deterministic verifier (model-agnostic, violations → 0), not from generator scale** — consistent with Paper 1's scale-invariance claim.
- **Paper 4 (system: cost-optimal {learning, A2, scaffold}).** Paper 4 optimizes the CapEx/OpEx mix treating A2 cost as given. Paper 2 *moves that cost curve*: by learning to generate A2 cheaply on-prem, it changes the per-domain authoring cost that Paper 4's optimization trades against. The dose-response curve (how many real domains buy transfer) is a direct input to Paper 4's cost model.

A2 sits inside the broader frame — **diverse (possibly non-deterministic) generator + fixed deterministic selector/verifier** — so the compliance guarantee is independent of generator determinism; the verifier carries the contract. This is why the small, swappable generator is viable at all.
