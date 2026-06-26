# Paper 4 — The System & Cost Paper: Total-Lifecycle-Optimal Lever Allocation for On-Prem Tool-Use Agents

**Research Proposal (ICLR-style)**

*Portfolio position:* Paper 1 establishes the function × scale × lever map of tool-use (what scale buys, what is genuinely scale-invariant, what scaffold + A2 cover). Paper 2 auto-generates the A2 (NL→GATE_SPEC compiler). Paper 3 learns transferable path-selection heuristics. **Paper 4 (this proposal) is the engineering paper**: given the three levers {model learning, A2 generation, deterministic scaffold} matured in Papers 1–3, it optimizes the *deployed implementation* under a total-lifecycle cost objective (CapEx + OpEx), delivering a cost-optimal lever-allocation guideline and a running on-prem system. The headline quantity is not the smallest model but the **cost knee**: the smallest *total-cost* model size.

---

## 1. Problem & Motivation

The dominant framing of "use a bigger model" hides its true price. Scale costs are paid on **two permanent axes**:

1. **Inference OpEx** — every request consumes GPU-seconds, dollars, and latency, forever. A 2× larger model is roughly 2×+ inference cost and is permanently slower.
2. **Hardware CapEx** — for on-prem deployment the operator *buys* the GPUs: VRAM tier, power, cooling, rack. A 2× model is roughly 2×+ VRAM/GPU/power, paid up front and directly.

A model that "looks free to build" (just download a bigger checkpoint) loses on *both* axes over the deployment lifetime. The central economic claim of this paper is therefore: **a small model + structure (deterministic scaffold + minimal transferable learning) that matches a large model's behavior cuts inference OpEx *and* hardware CapEx simultaneously** — same capability on smaller, cheaper, lower-power hardware.

This matters most exactly where it is hardest: **on-prem / sovereign deployments** (regulated finance, healthcare, government). There, data cannot leave the premises, so a frontier API is off the table; the customer pays the CapEx directly; and the most expensive lever — hand-authored, domain-specific policy/config (the "A2") — must be produced by domain experts *on site*, per field, per change. The payoff of getting the lever allocation right is "same capability on smaller hardware with fewer experts," which is precisely the value an on-prem buyer is willing to pay for.

Paper 1 supplies the load-bearing asymmetry this paper exploits: **capability is largely scale-retired, but guarantee/compliance is scale-invariant.** Measured on a control-flow floor (no gate), pass rises sharply with scale (7B 0.19 → 32B 0.55) as capability-scaffolds become unnecessary, yet the compliant-pass gap stays *flat* across scale (0.058 → 0.064 → 0.056) and **g2 policy-violations do not fall (31 → 38 → 41 across 7B → 32B)**. A deterministic gate zeroes violations at *every* scale. Bigger models do not buy compliance; only structure does. This split is the lever that lets a small model + gate match a large model's *trustworthy* behavior, not just its raw pass rate.

## 1b. Related work — cost-efficiency, routing, deferral, regulatory

The cost-efficiency literature optimizes *cost given black-box components*; we instead **allocate deterministic-vs-learned-vs-scale per capability by measured cost** and deploy a fully-local fleet. Positioning by cluster:

- **Learned cost-routing of frontier calls.** **ToolOrchestra** (arXiv 2511.21689) RL-trains an 8B router to delegate to a frontier model only when worthwhile (80.2%@10.3¢ vs GPT-5 77.7%@31.3¢ on τ²) — the strongest cost-efficiency result, **but it relies on frontier delegation**, disqualifying it for the on-prem / no-data-egress regime; our fleet escalates only within operator-controlled models. **TRUST** (arXiv 2606.06976) gets 4B > 30B by RL but is black-box with introspective Ask/Unable and no deterministic guarantee. Learned routers more broadly — RTR (71.7%↓ tokens, OOD transfer), A2FM (45%↓ cost), xRouter (answer-vs-delegate), ARM2 (format-RL, code offload) — establish that routing-to-minimize-cost is a learned, transferable skill.
- **Cascading / deferral.** **ReDAct** (arXiv 2604.07036) uses calibrated-threshold deferral (small→large) to reach large-model quality at a fraction of cost; **Unified Routing and Cascading** (arXiv 2410.10347) shows a learned quality-estimator is the key factor in cascade routing; this is the FrugalGPT-style cost-cascade family. Our fleet-mix (§4.2) is a *deterministic-router* instance, with the router decidable rather than learned.
- **Determinism as an OpEx lever.** Bootstrapping cognitive agents (arXiv 2403.00810) reports **50–100× fewer reasoning tokens** once work is pushed to deterministic productions — direct evidence that structure cuts inference OpEx. Our own deterministic-vs-learned TCO synthesis (deep-research) characterizes when an engine's amortized cost beats a scaled model. Counter-direction: LLM-ACTR (arXiv 2408.09176) *bakes* the symbolic policy into weights (per-domain retraining; no transfer) — the cost-anti-pattern we avoid.
- **Total-cost framing (scaling stance).** Against a pure-scale reading of Sutton's *Bitter Lesson* (2019), **Brooks' "A Better Lesson" (2019)** argues total cost (engineering + compute + data + lifecycle) must be counted, and **Chollet** (arXiv 1911.01547) that priors/data "buy skill," not generalization. Our knee/TCO is the operationalization of this total-cost view.
- **Regulatory determinism (the value side / moat).** No major framework textually *mandates* determinism, but the buyer value is **verifiability and model-risk-management exemption**: SR 26-2 (2026), EU AI Act (Reg. 2024/1689), and the lone explicit "repeatability" clause in EU MDR Annex I §17.1. This is *why* an on-prem buyer pays for the deterministic-gate compliance (Paper 1's scale-invariant guarantee layer).
- **Industrial baseline.** **Palantir Foundry** (Ontology ≈ deterministic scaffold + per-domain A2; AIP-LLM ≈ what we obtain cheaply with a small model + transferable TBox) — we differentiate on *cost* (HW CapEx/OpEx, expert effort, config-only field-crossing, zero data egress), not on functionality (§5.3).

## 2. Objective Function

We optimize total lifecycle cost as the *simultaneous* minimization of five terms (not a single ordering):

> **minimize { ① cheap GPU tier/count, ② low VRAM (incl. int8/int4 quantization), ③ maintenance (re-configuration per change), ④ human experts, ⑤ generalization cost per new field (ideally = A2-swap only) }** + HW-infra CapEx + inference OpEx.

The terms are coupled, and their gradients point to the same answer:

- **①② (cheap/small HW)** → a **small model** + quantization.
- **③④⑤ (low maintenance / few experts / cheap new-field onboarding)** → **maximize the fixed, domain-general part** (deterministic scaffold + transferable, forgetting-free learning = one-time ML-engineer cost that absorbs change) and **minimize the per-domain variable part (the A2)**, with domain-targeted retraining forbidden.

This yields the paper's organizing insight:

- **Common enemy = the A2.** It is recurring maintenance (③), demands a domain expert per field on site (④), and is the per-new-field marginal cost (⑤). Minimizing the A2 attacks ③④⑤ at once, and is *worst* under on-prem (data cannot leave → no frontier A2-generator → manual authoring).
- **Common friend = fixed scaffold + transferable TBox.** The structure that ①② (small model) *requires* is supplied cheaply and once — built offline, shipped in, no customer data, no re-authoring on change.

**The knee (honest tension).** Shrinking the model (①②↓) demands *more* structure/learning for the same capability. If that structure is **fixed/transferable**, ③④⑤ do not rise (one-time). If it is **A2-/domain-targeted**, ③④⑤ explode. Therefore the optimum is **not the smallest model** but the **smallest-total-cost size** — the knee of the cost curve. Pushing added structure to be fixed/transferable-only is what moves the knee toward smaller models. Quantization (int8/int4) is a direct ②-lever, with capability loss measured as an explicit matrix axis.

## 3. Approach

**(a) The master matrix.** We measure a **capability × lever × scale × quantization × field** matrix. Capabilities are the atomic units C1–C12 from Paper 1 (provenance-check, dependency-sequencing, identity-before-scoped, precondition gate, confirm-before-write, error-recovery, selection-resolution, operand/value-formalize, flow-rule following, NL communication, …). Levers are {prompt, deterministic scaffold, A2-config, minimal transferable learning, scale}. Each cell records pass^1 + assembled cost (HW / OpEx / expert-time / change-scope / generalization) + a no-forgetting check on held-out general ability.

**(b) Decidability-first lever allocation.** For each capability C, evaluate levers in cost-ascending order and short-circuit:

```
1. C decidable from schema+policy+state?      → SCAFFOLD (engine).  [scale-invariant, ~0 per-domain]
2. else C is a domain-specific fact?           → A2 / ABox.          [irreducible domain cost — minimize]
3. else C promptable at deployment size?       → PROMPT.            [cheapest if it works]
4. else scale-emergent behavior, size affordable? → SCALE or cheap-replication method-set
5. else (irreducible NL→formalize)             → minimal transferable LoRA (forgetting-free). [last resort]
```

We further split scaffold into two kinds (Paper 1 result): **capability-scaffold** (present/calc/resolve) *recedes* with scale, while **guarantee-scaffold** (auth/confirm/ownership/precond gates) is *scale-invariant* and remains valuable even at frontier scale — this is why even large models keep a deterministic trust layer.

**(c) Cheap-replication method-set.** For each capability that scale buys, we install it on a small model with *minimal* intervention and no forgetting: e.g. C3 fetch-first via a deterministic autofetch engine (provenance-deny → deterministic producer call → inject real value; learning = 0); C8 error-recovery via a retry-controller vs. minimal LoRA (under measurement); C10 operand via minimal-rank, forgetting-free LoRA only where base is genuinely weak. The claim is "scale is a *decomposable bundle*, and each piece is obtained more cheaply than scale," with an honest boundary where some pieces remain genuinely scale-bound.

**(d) Zero-retrain ABox-swap transfer (generalization cost = ⑤).** The fixed part (model weights + scaffold engine + TBox) must be byte-identical across fields; only the ABox is swapped. We measure this across **structurally different** fields — retail → airline → banking → healthcare — using three instruments: (i) `grep "if field" == 0` and zero retraining (fixed part provably invariant), (ii) the A2-swap authoring effort, and (iii) performance retention. retail↔airline alone is weak evidence (both are customer-service); structural diversity across finance/healthcare/admin is the real test of "generalization cost ≈ A2-swap."

## 4. Results So Far

We mark each result **[SETTLED]** (measured) vs. **[ESTIMATE]** (partial/projected).

- **[SETTLED] ~23× TCO advantage.** On tau2 retail with existing runs: 32B-int8 + deterministic gate, on-prem, **$0.0019/req vs. gpt-4.1 API $0.044/req ≈ 23× cheaper** (16–40× across a GPU price range of $0.2–0.5/hr), and ~3000× cheaper than a human agent (~$6/contact). On-prem, auditable, deterministic gate, zero data egress. Honest trade-off: latency ~6× (178s vs. 30.5s); the on-prem compliance point (0.573) is *below* gpt-4.1 (0.82), so the pure-on-prem headline is "near-compliance, ~23× cheaper," not "equal." [Latency, tool-roundtrips, VRAM = measured; $/req = estimated from token≈chars/4 and an assumed GPU utilization — order-of-magnitude robust, exact multiple pending token-capture fix.]

- **[SETTLED] Capability/guarantee split (from Paper 1).** Floor pass scales (7B 0.19 → 32B 0.55) but the compliant gap is flat and g2 violations do not fall with scale; a deterministic gate zeroes violations at every scale. This is the mechanism that makes the small-model + structure substitution viable for *trustworthy* behavior. [SETTLED, range 7B–32B; 72B confirmation pending.]

- **[SETTLED] Scale monotonicity / decomposition.** pass^1 rises monotonically with scale (7B 0.24 / 14B 0.52 / 32B 0.60, n=342); within it, provenance/recovery-type capability is engine-/scale-replaceable while operand-type residual is the narrow learning target. C3 autofetch alone moves grounding (A 33 → 9) and ~doubles pass (0.14 → 0.264) with zero learning. [SETTLED.]

- **[SETTLED] fleet projection (equal-compliance, cheaper).** A fleet (easy req → 32B on-prem, hard req → frontier escalate) blends to compliance 0.860 (> pure gpt-4.1 0.816) at $0.021/req (~2.1× cheaper than pure gpt-4.1) on clean data. [SETTLED for the blend arithmetic on clean nt1 + gpt-4.1 data; assumes a cheap, decidable router that is not yet implemented.]

- **[ESTIMATE / partial] Multi-field zero-retrain transfer.** retail↔airline ABox-swap shows fixed-part invariance for content-op facets; structurally distant fields (banking, healthcare) and the full ⑤ generalization-cost column are **not yet measured** (next instrument: role-sourcing for A2-swap effort).

## 5. Planned Experiments

1. **Full master matrix.** Fill capability (C1–C12) × lever (5) × scale (1.5/7/14/32/72B + frontier) × quantization (int8/int4) × field (retail/airline/telecom/banking/healthcare). Each cell = pass^1 + assembled cost + no-forgetting check, with GO/NO-GO per cell.
2. **The knee measurement.** Sweep model size against *total* cost (HW CapEx + OpEx + maintenance + experts + generalization) to locate the smallest-total-cost size, generalizing the single-model knee to a **cost-optimal heterogeneous on-prem fleet mix** decided by per-capability L-vs-E crossover data — not a static assumption.
3. **Lifecycle-cost quantification vs. Palantir Foundry.** Position against Palantir's Ontology (≈ our deterministic scaffold + per-domain A2) and AIP-LLM (≈ what we obtain cheaply with small model + transferable TBox). Differentiation is *cost* — HW CapEx/OpEx, expert effort, field-crossing generalization (config-only), zero data egress — **not** a functionality-superiority claim (Palantir governance/integration maturity is strong; we measure the boundary honestly).
4. **$/req precision fix.** Turn on litellm cost/token capture so the estimated $ row becomes measured; report on-prem $ as a range over GPU amortization/utilization assumptions.

## 6. Expected Contribution & Relation to Papers 1/2/3

The deliverables are three, all engineering-grade and honest:

1. **A cost-optimal lever-allocation guideline** — a measured, calibrated decision procedure that, per capability, assigns the minimum-total-lifecycle-cost lever (decidability-first), and locates the cost knee / fleet mix.
2. **A cheap-replication method-set** — concrete, no-forgetting techniques that install each scale-bought capability on a small model with minimal structure/learning, demonstrating "scale is a decomposable bundle, each piece cheaper than scale."
3. **An honest boundary map** — which capabilities are recovered cheaply on small models, and which remain genuinely scale-bound; plus a deployed on-prem system instantiating the guideline.

**Relation to the portfolio.** This paper *consumes* the three levers as developed upstream and answers the engineering question they raise: Paper 1 gives the function × scale × lever map and the capability/guarantee split that this paper turns into a cost objective; Paper 2 supplies the NL→GATE_SPEC compiler that drives down the A2 — the common enemy — making the ④⑤ terms tractable on-prem; Paper 3 supplies the learned transferable path-selection that keeps the fixed part field-invariant under ABox-swap. Paper 4's job is to *allocate and deploy* them at minimum total cost and to report, with marked estimates, exactly where that minimum sits and where the honest scale boundary lies.

*Honesty note.* The thesis defended here is the defensible form: not "the small model does everything," but **"small model + scaffold does most of it, the measured irreducible minority goes to a right-sized on-prem model, at minimum total cost."** The contribution is making the large model *unnecessary* (minimized), not using it.
