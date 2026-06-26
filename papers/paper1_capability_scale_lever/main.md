# What Does Scale Buy in Tool-Use Agents? A Capability×Scale×Lever Map Showing Compliance Is Scale-Invariant

*ICLR submission draft — branch `facet-rft-2026`. Numbers tagged **[SETTLED]** are measured (all with zero paid frontier-API calls; the user-simulator is local); **[EST]** are estimates/placeholders for experiments not yet run.*

---

## Abstract

Tool-use agents (NL request → tool calls under a policy) are usually improved by scaling the base model. We instead ask, function by function: *what does scale actually buy, what can a model genuinely not do at any scale, and what is cheaply covered by deterministic scaffold or by per-domain authored relations (A2)?* Treating four benchmarks (τ²-bench, SOPBench, TaskBench, Synth) purely as measurement instruments, we decompose tool-use failure into **operand execution**, **orchestration load**, and **compliance/guarantee**, and chart each across model scale (7B–32B measured; 72B–235B **[EST]**) and across levers (base / scaffold / A2 / learning). Three findings organize the map. (1) **Operand execution saturates early**: given an explicit specification a 32B base is exactly correct 88/88 (100%), so the residual "operand failures" are not a learnable capability gap. (2) **Capability-type load retires with scale in a definite order**: of five candidate load dimensions, interference and state-multiplicity are predictive of failure at 7B/14B but vanish by 32B, while length and conditional-depth persist. (3) **Compliance is scale-invariant.** As scale rises, task-pass climbs (7B 0.19 → 32B 0.55) but the *compliant*-pass gap stays flat (~0.056) and policy-violations do **not** fall (g2-class 31→41 across 7B→32B); a deterministic gate eliminates them at every scale. We conclude that scale buys *capability* but not *guarantee*, so even a frontier model needs a deterministic guarantee-layer — and a small base + capability-scaffold (which recedes as scale grows) + a scale-invariant guarantee-scaffold + authored A2 is the cost-efficient way to reach frontier-level tool-use. We frame deterministic scaffold as a **load-reduction operator** and give the function×scale×lever map as the paper's contribution.

---

## 1. Introduction

A tool-use agent maps a natural-language request to a sequence of tool calls that must satisfy a domain policy. The default recipe for better agents is a bigger base model. But "bigger" conflates several distinct things, and it is expensive: model size roughly doubles inference OpEx and hardware CapEx. The practical question for on-premise deployment is not "how big a model" but **"which capabilities must scale up, which can be handed to cheap deterministic structure, and which no model — frontier included — can be trusted to do?"**

We answer this with a **function × scale × lever** decomposition. For each functional sub-capability of tool-use we ask, across model scale, whether it (a) grows with scale, (b) is a genuine model limit that scale does not fix, (c) is efficiently covered by deterministic *scaffold*, or (d) is covered by per-domain authored relations/policies (*A2*). The benchmarks are instruments for filling cells of this map, not the object of study.

**Contributions.**
1. A controlled, **frontier-API-free** methodology that isolates operand / orchestration-load / compliance and measures each across scale (§4). Isolated single-shot probes and existing multi-scale runs avoid the cost and noise of full agent rollouts.
2. **Operand execution is scale-saturated, not learnable** (§5.1): a 32B base is 100% correct given a specification; the apparent gap is criterion-interpretation handled deterministically, not a capability deficit. Faithful-formalize SFT is therefore NO-GO at 32B.
3. A **load decomposition** and its **scale-response** (§5.2): the set of load dimensions that bind is *scale-dependent* — scale retires interference and state load by 32B while length and conditional-depth persist. Controlled probes overturn an observational correlation, demonstrating the necessity of controlled (not merely observational) measurement.
4. The central finding: **compliance/guarantee is scale-invariant** (§5.3). Capability rises with scale but the policy-violation rate of the *compliant*-pass metric does not; a deterministic gate closes it at every scale. This is the empirical backbone of a **two-kind scaffold** thesis — capability-scaffold recedes with scale (F3, pass), guarantee-scaffold is scale-invariant (F4, compliant-pass) — and explains *why even frontier systems need a deterministic guarantee layer*.
5. A unifying frame: **scaffold = a load-reduction operator** (§6), giving the prediction `pass ⇔ L_eff < L*(N)` with `L_eff = L − ΔL_scaffold` and an *independent* estimate of `ΔL`.

Our headline is deliberately **not** "small beats large": that result is already held by RL-routing systems (§2). Our contribution is *which* parts of the gap are scale-invariant and must be offloaded, and a map that tells a deployer which lever to pull per capability.

## 2. Related Work

Our contribution sits at the intersection of *verified prior pieces*; the central lineage is conceptual (why an LLM is good at language but must offload deep procedures and cannot self-guarantee compliance), not the recent "small-beats-large" routing systems, which address a narrower problem (§2.5).

**§2.1 Why language is shallow-parallel and deep procedures must be offloaded (the core lineage).** A bounded-depth Transformer forward pass is, asymptotically, a constant-depth parallel circuit (uniform TC⁰; Merrill & Sabharwal, arXiv 2207.00729), and chain-of-thought escapes this only by unrolling depth into serial tokens (Feng et al., arXiv 2305.15408 / 2402.12875; RASP-L length-generalization, Zhou et al.). Natural language, meanwhile, *embeds* procedures of definite computational class: **van Benthem's semantic automata** make "every/some" finite-state but "most/less-than-half" provably pushdown, and this hierarchy is cognitively real (working-memory/PFC recruitment and reaction-time scale with the minimal-automaton class). **Relevance Theory** (Blakemore; Wilson) distinguishes *conceptual* from *procedural* meaning (the latter sub-personal, "machine-language"), and **Goodman's notationality** (disjointness + finite differentiation) gives the formal condition under which a symbol system supports unambiguous mechanical manipulation. The representational-effect correction (Zhang & Norman) clarifies that notation sets *cost*, not computability. Together these say: classifying *which* deep procedure a request embeds is shallow (LLM-learnable, transferable), but *executing* it is serial-deep and must be offloaded — the foundation of our capability/lever routing. (Project synthesis: `NL_PROCEDURE_OFFLOAD_THEORY`; deep-research reports under `deepresearch/`.)

**§2.2 Cognitive architectures, non-introspective decidability, and scale-invariant epistemic limits.** Our system is an instance of the **cognitive-architectures-for-language-agents** umbrella (CoALA, Sumers et al., arXiv 2309.02427, TMLR): an LLM as a probabilistic production within a memory/action/decision loop; we make concrete the substrate CoALA leaves general (a relational-algebra deterministic core, an empty-relation abstention, zero-retrain A2-swap). **SOAR** (Newell, Laird & Rosenbloom 1987; Laird 2012; cognitive-design-patterns, Wray, Kirk & Laird, arXiv 2505.07087) surfaces indecision *architecturally* — an `impasse` observed from the absence of a preference, not from introspection — and subgoals; this is the precedent for our "uncertainty → observable empty/ambiguous relation → ASK" (tie = σ>1, no-change = σ=0). The *deterministic-first / LLM-fallback* execution pattern is established (Bootstrapping cognitive agents, Zhu & Simmons, arXiv 2403.00810, 50–100× fewer reasoning tokens; NL2GenSym, arXiv 2510.09355, LLM→SOAR-rule synthesis). We add what this lineage lacks: a **formalize-faithfulness** axis (SOAR assumes productions are correct), a *learned* empty→ASK, and a small *weight-learned* domain-general skill with A2-swap transfer. On the epistemic side, **hallucination is provably scale-invariant** (Xu, Jain & Kankanhalli, arXiv 2401.11817; Kalai et al., arXiv 2509.04664: scoring rewards guessing over abstention), and verbalized self-confidence is non-diagnostic **on our exact benchmark** (τ²-bench confidence ≈ random, AUROC 0.47–0.69, inflating with trajectory length; arXiv 2602.05073 [pilot-scale]). The mis-formalize failure we target (a wrong-but-non-empty formalization that the deterministic checker cannot catch) is now observed in *frontier* models: "Do LLMs Game Formalization?" (Kim et al., arXiv 2604.19459) finds a detection-evading premise-mistranslation mode (a minority but checker-invisible), the direct analogue of our silent ⓑ residual. Compositional collapse with complexity is documented (Faith and Fate, arXiv 2305.18654; GSM-Symbolic, arXiv 2410.05229; planning collapse, arXiv 2409.13373; correctness-guarantee 0, arXiv 2410.02162). **Our §5.3 result is the policy-enforcement analogue of all of the above**: just as no model self-guarantees truthfulness, none self-guarantees *compliance* — both scale-invariant, both demanding an external deterministic mechanism (abstention/impasse; a gate). *(Honesty, per the source review: faithfulness-gaming is a minority mode — most errors are faithful — and the cited works do not show "more deep ⇒ more fabrication"; we do not claim that.)*

**§2.3 LLM-proposes / deterministic-executes (offload boundary).** The split we inherit is mainstream: LLM-Modulo (Kambhampati et al., arXiv 2402.01817, ICML 2024), PAL (program-aided), and NL→PDDL *formalizers* that beat direct planning (e.g. 100-block 100% vs ~20%), with RLVR systems whose "V" is a deterministic checker (AlphaGeometry, Nature 625, 2024; AlphaProof 2024; Tülu 3, arXiv 2411.15124). Neurosymbolic motivation: Garcez & Lamb (arXiv 2012.05876), Kautz (AI Mag. 43(1), 2022), Marcus (arXiv 2002.06177). **Honest boundary:** offload does *not* always win — LLM-as-formalizer loses to direct solving on simple cases (formalization overhead) and can leak solution reasoning into the spec; the boundary is itself an object of measurement (our `d(e)` / bounded-budget probes).

**§2.4 Guarantees / shielding.** Runtime enforcement from safe-RL shielding (Alshiekh et al., AAAI 2018; Jansen et al., arXiv 1807.06096) and predictive safety filters (Wabersich & Zeilinger, Automatica 2021) to LLM-agent instances: ShieldAgent (arXiv 2503.22738, probabilistic rule circuits, precision<1), AgentSpec (arXiv 2503.18666, hand-written DSL), Formal-LLM (arXiv 2402.00798). These supply a *blocking* safety layer with hand-authored specs or probabilistic checks; our gate is deterministic, *drives* (repairs) preconditions rather than only vetoing, and is paired with a learned proposer and zero-retrain transfer.

**§2.5 Learned routing and "small-model orchestration" (the narrow rivals).** Classification/routing of LLM behaviour is a learned, transferable skill — RTR, A2FM, xRouter, ARM2, ToolkenGPT, and When2Call (tool-necessity linearly decodable from hidden state, AUROC 0.89–0.96; RPO≫SFT); RITE shows math-only training transferring cross-domain — while *verbalized self-confidence is distrusted* (criterion-placement, not metacognition), reinforcing external/learned routing over introspection. **ToolOrchestra (arXiv 2511.21689) belongs here and is narrower than it first appears**: it is an 8B model trained by RL to decide *when to delegate to a frontier model vs. answer locally* — i.e. learned **cost-routing of frontier calls** over black-box components (reported 80.2%@10.3¢ vs GPT-5 77.7%@31.3¢ on τ²). It optimizes *cost given* the components; it does **not** provide deterministic offload, by-construction compliance, decidable-vs-learned accounting, or zero-retrain domain transfer, and it *relies on frontier delegation* — disqualifying for the on-prem / no-frontier-call / no-data-export regime. It establishes that a cost-favorable small-model result *exists*, which is exactly why our contribution is the *analysis and method* (what is scale-invariant, which lever per capability), not that result. ATA (arXiv 2510.16381, FOL+Z3, training-free) is per-domain with no transfer.

**§2.6 Load, long-context, and benchmarks.** Our load framing defers to cognitive-load theory and long-context "lost-in-the-middle" degradation; novelty is only *load-decomposition × scaffold-load-reduction for tool-use*, not a theory of load. Benchmarks as instruments: τ²-bench (arXiv 2406.12045), SOPBench (arXiv 2503.08669), TaskBench, Synth/CFB (arXiv 2501.10132); the floor-measurement precedent separating info-limited vs reasoning-limited failure is "Sufficient Context" (Joren et al., ICLR 2025).

**Whitespace.** Prior work supplies each piece — procedural-semantics/automata (what is deep), Transformer-expressivity (why a forward pass cannot execute it), SOAR/impasse and CoALA (the cognitive-architecture umbrella and non-introspective surfacing of indecision), hallucination-inevitability (scale-invariant epistemic limit), formalize-offload (LLM proposes/engine executes), learned routing (classification transfers), shielding (deterministic enforcement). The unoccupied intersection *this paper* occupies is the **empirical function×scale×lever map**: *which* capabilities scale retires, *which* are genuine scale-invariant limits (compliance, measured), and *which* lever — capability-scaffold (recedes), guarantee-scaffold (invariant), A2, or learning — covers each. The recent routing systems (ToolOrchestra, TRUST) optimize cost over black boxes; we open the box. *(Portfolio note: the "closure-justified finite generator basis + learned rule-abstraction transfer" claim of our prior related-work synthesis — once our intended headline — is, after the operand make-or-break verdict, re-assigned to the companion papers: A2 generation (Paper 2), path-selection heuristics (Paper 3), and the cost-optimal system (Paper 4); the present paper claims the scale-invariance map, not the transfer mechanism, which prior work [schema-guided DST, ToolLLM] already established as a variant.)*

## 3. The Capability×Scale×Lever Framework

We decompose a tool-use trajectory into typed functional steps and, for each function `f`, populate a row of a map indexed by scale `N` and lever `ℓ ∈ {base, capability-scaffold, guarantee-scaffold, A2, learning}`. Each cell records: does capability on `f` grow with `N`; is `f` a genuine model limit; does a lever cover `f` efficiently.

**Three structural claims** organize the map:

- **Operand vs orchestration vs compliance.** Operand = atomic execution given a criterion (pick the variant matching a spec). Orchestration = holding multiple interdependent steps coherent under *load*. Compliance = obeying the domain policy with a *guarantee*. These fail for different reasons and respond to different levers.
- **Decidability-first lever routing.** A function that is exactly specifiable (resolution, eligibility, aggregation, provenance) is handed to deterministic scaffold; domain-specific facts are *provided* via A2/retrieval; only domain-general translation/reasoning is *learned*. This inverts the default "scale it" reflex.
- **Two kinds of scaffold.** *Capability-scaffold* (present/compute/resolve) substitutes for a skill the small model lacks and therefore **recedes as scale grows** (F3, raw pass). *Guarantee-scaffold* (policy gates) provides a property no probabilistic model gives and is therefore **scale-invariant** (F4, compliant-pass). §5.3 measures the split.

**Load as a vector.** Orchestration "load" is not scalar. From failure forensics we operationalize five computable candidate dimensions: **L_len** (context to retain), **L_state** (interdependent state carried), **L_branch** (conditional-resolution depth), **L_interf** (confusable similar entities), **L_contra** (mid-stream contradiction/revision). Crucially, **load ⊥ operand difficulty**: load can be raised while operand difficulty is held fixed.

**Scaffold as a load-reduction operator.** Each scaffold component removes a specific load dimension: present/autofetch externalizes reads (↓L_len, ↓L_interf); compute offloads aggregation (↓L_state); gates enforce eligibility (↓L_branch); a controller holds state (↓L_state, ↓L_len). This gives an *effective* load `L_eff = L − ΔL_scaffold` and the prediction

> `pass(f, N) ⇔ L_eff(f) < L*(N)`,

where `L*(N)` is the model's load-tolerance at scale `N`. To avoid tautology, `ΔL` must be estimated *independently* (from the mechanical feature-reduction the scaffold performs) and then shown to predict the failure-onset shift — not read off that shift. The full predictive test requires the controlled curves of §5.2 at more scales **[EST]**; here we establish the components.

## 4. Method

**Benches as instruments.** We measure on τ²-bench retail (primary), with SOPBench/TaskBench/Synth as the canonical *training* substrate for the lever-allocation that Papers 2–4 build on; τ² is a *transfer* target, never trained on. Scale points: Qwen2.5 7B / 14B / 32B-GPTQ-Int8 **[SETTLED]**; 72B / 235B **[EST]**.

**Cost discipline (frontier-API-free).** The only paid component would be a GPT-4.1 user-simulator; we avoid it. Capability is measured with **isolated single-shot probes** (operand and load probes that pose one decision with a fixed specification) and with **existing multi-scale agent runs** already on disk. A local user-simulator (scripted faithful turns over the local base model) replaces the paid simulator for full-flow checks. All results are persisted under `sim_results/`.

**Conditions.** *floor* (no scaffold) isolates the base model; *g15 / present / nested / assembled* add capability- and guarantee-scaffold; *GIVEN-SPEC vs GOAL* controls the user-simulator input to separate operand execution from criterion interpretation.

**Metrics.** Robust **pass^all** (all trials) over per-trial point estimates; **F3** = raw task pass (bench/strict); **F4** = compliant pass (full, deducting policy violations); per-dimension **partial correlation** of failure with each load feature, controlling operand difficulty; controlled **accuracy-vs-load** onset curves. We do *not* report single-trial point estimates as findings (user-sim noise ≈ 0.11).

## 5. Results

### 5.1 Operand execution is scale-saturated, not a learnable gap [SETTLED]

A controlled experiment varies only the user-simulator input. Given an explicit option specification (**GIVEN-SPEC**), the 32B base selects the correct variant **88/88 (100%)**. Given only the goal (**GOAL**), it is 62/88 (70%); the 30% gap is criterion interpretation (argmax/argmin = deterministic compute, multi-attribute reasoning, conversational fidelity), not execution. Genuine join/disambiguation (user describes, does not name, the order) is 7/13 (54%) and is present-addressable. **Implication:** operand is not a capability gap at 32B, and faithful-formalize SFT is **NO-GO** — there is no residual to learn. (Forensic note: every prior "operand failure" we chased dissolved into a measurement artifact — a calc bug, a premature single case, an ID-given probe artifact — until the controlled probe read 100%.)

The complementary limit is that *selection over a large candidate set does not scale in-head*: on a depth-scale probe, `comparative` over 50 candidates scores **0.02** and `rank@50 ≈ 0.30` even including a 235B model, while the deterministic engine is trivial (1.00) **[SETTLED]**. So "pick the best/most among many" is not done by raw abstraction at any scale; it is offloaded. Operand thus splits cleanly: *atomic execution given a spec* is saturated (base, 100%), and *large-N selection* is a deterministic-compute lever — neither is a learning target.

### 5.2 Capability-type load retires with scale, in order [SETTLED 7B–32B]

Across the five candidate load dimensions, partial correlation of *floor* failure with each feature (controlling operand difficulty) gives:

| dimension | 7B | 14B | 32B |
|---|---:|---:|---:|
| L_len (length) | +0.20 | +0.33 | **+0.37** |
| L_state (state) | **+0.24** | **+0.21** | +0.06 |
| L_branch (conditional) | +0.20 | +0.13 | **+0.26** |
| L_interf (interference) | **+0.30** | **+0.20** | +0.08 |
| L_contra (revision) | +0.06 | +0.13 | +0.04 |

The **binding set is scale-dependent**: at 7B/14B four dimensions predict failure; by 32B only length and conditional-depth survive. Scale *retires* interference and state-multiplicity first; length and conditional-depth persist. L_contra is too sparse in τ² to fit. This is the precise empirical form of "scale buys capability": it removes load dimensions in a definite order.

**Controlled > observational.** The above is observational and confounded with task size. A *controlled* interference probe (fix operand; vary only the number N of confusable variants) does **not** reproduce the strong 7B interference signal:

| N (distractors) | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 32B | 1.00 | 1.00 | 1.00 | 0.97 | 0.87 |
| 7B | 1.00 | 1.00 | 1.00 | 0.90 | 0.80 |

The 7B–32B gap is small (0.80 vs 0.87 at N=16); the observational L_interf correlation was largely a size confound. This is the methodological payoff of controlled generation, and a caution: single-shot probes under-test the inherently *multi-turn* dimensions (L_state, L_contra) **[EST: multi-turn controlled probes pending]**.

**The residual is orchestration, not operand.** Forensically re-running the robust fail-all set under faithful (local, gpt-4.1-free) turns: 2/10 are user-simulator-noise flips, while the dominant residual is *orchestration-under-load* — multi-item batching that violates a non-pending precondition, multi-order tracking that drops an order and fabricates a placeholder id, conditional-branch sequencing, and context-window overflow. An isolation **plan probe** (elicit the abstract action plan with reads pre-supplied, grade structure only) shows the core action structure is planned correctly in 6/10 cases (e.g. the batch the live run split is batched correctly in isolation), i.e. roughly half the orchestration residual is *execution-load* (closable by deterministic plan/execute separation) and half is genuine in-isolation planning miss (conditional / multi-order grounding) — the irreducible piece that motivates a learned path-selection lever (companion Paper 3), not a bigger base. **[SETTLED, forensic; the assembled present+nested+gate stack ceiling is robust pass^all = 0.402.]**

### 5.3 Compliance is scale-invariant — the central finding [SETTLED 7B–32B]

We separate **F3** (task pass) from **F4** (compliant pass; passing trajectories that also commit no policy violation). On *floor* (no gate):

| scale | F3 (pass) | F4 (compliant) | gap | violations (g1 / g2) | total viol% |
|---|---:|---:|---:|---|---:|
| 7B | 0.189 | 0.130 | 0.058 | 57 / 31 | 33.6% |
| 14B | 0.468 | 0.404 | 0.064 | 38 / 38 | 26.3% |
| 32B | 0.547 | 0.491 | 0.056 | 6 / **41** | 14.3% |

Capability (F3) climbs steeply with scale, but **the F3–F4 gap is flat (~0.056) and the policy-violation class g2 does not fall (31→41) — it slightly rises** even as the auth-type class g1 collapses (57→6). With the deterministic gate (g15) the picture is uniform across scale:

| scale | F3 | F4 | gap | viol% |
|---|---:|---:|---:|---:|
| 7B | 0.162 | 0.162 | **0.000** | **0.0%** |
| 14B | 0.469 | 0.469 | 0.000 | 0.0% |
| 32B | 0.547 | 0.547 | 0.000 | 0.0% |

**Interpretation.** Scale solves *capability-type* violations (g1, auth) but **not** *policy-semantic compliance* (g2): the compliant-pass gap is scale-invariant. The gate zeros it at every scale, including the most capable. Therefore a probabilistic model — frontier included — cannot be trusted to *guarantee* policy compliance, and a deterministic guarantee-scaffold is needed at all scales. This is the policy-enforcement analogue of the scale-invariance of hallucination (§2), and it is the empirical backbone of the two-kind scaffold (F3 recedes / F4 invariant). *(Caveat: range is 7B–32B; the gated gap=0 holds by construction, so the informative quantity is the non-closing floor gap; the frontier projection rests on the flat trend plus the hallucination-inevitability priors, not on a >72B measurement.)*

### 5.3b Cheap-replication: a capability-scaffold substitutes for scale [SETTLED]

Raw task-pass rises monotonically with scale (pass^1 = 7B 0.24 / 14B 0.52 / 32B 0.60, n=342) — the headline "scale buys capability" curve, consistent with the floor F3 trend of §5.3. The point of the map is that *each piece scale buys can be obtained more cheaply than scale*. A direct instance: a deterministic **fetch-first** engine (provenance-deny → deterministic producer call → inject the real value) moves grounding errors 33 → 9 and roughly **doubles pass (0.14 → 0.264) with zero learning** — a capability-scaffold standing in for scale. On the lever side, not every scaffold helps: an eligibility-steering gate (G5) has ~zero causal effect (the model does not learn to use the guidance) and a naive retry is counter-productive, while the auth/confirm/ownership/precondition gates (G1–G4) are genuine levers. These results populate the "capability-scaffold recedes / guarantee-scaffold invariant" split with measured cells, and ground the cheap-replication thesis of the companion system paper.

### 5.4 The map, filled

| function | scale-response | genuine limit? | efficient lever |
|---|---|---|---|
| operand execution | saturates by 32B (100% given-spec) | no | base (no learning needed) |
| interference / state load | retire by 32B | no | capability-scaffold (present), or scale |
| length / conditional load | persist at 32B | partial | capability-scaffold (partial); conditional → controller/learn |
| policy compliance (g2) | **scale-invariant** | **yes (guarantee)** | **guarantee-scaffold (gate)** |
| domain facts | n/a | n/a | A2 / retrieval |

## 6. Discussion

The map yields a deployment rule that inverts "scale it": **reduce the load, don't train the model to tolerate it.** Capability-scaffold removes the load dimensions scale would otherwise have to buy (and thus recedes when scale is available); guarantee-scaffold supplies the property scale never buys (and is thus kept at every scale, frontier included). Where a load is irreducible by scaffold — conditional-depth / path-search is our candidate — the lever shifts to a *learned, transferable* heuristic (Paper 3), not to a bigger model. Training the model to tolerate load is, in our framing, training it to do the deterministic controller's job; the two-wing thesis routes that work to determinism by default and to learning only for the residual.

## 7. Limitations

- **Scale range.** Measured 7B–32B; 72B/235B and true frontier are **[EST]**. The scale-invariance claim is a flat-trend projection plus priors, not a >72B measurement.
- **Observational vs controlled.** §5.2's correlations are confounded with task size; only the controlled probe (one dimension done, L_interf) is causal. L_branch/L_len controlled curves and multi-turn (L_state/L_contra) probes are **[EST]**.
- **Single bench for compliance.** F3/F4 measured on τ² retail; cross-domain (airline/bank) and cross-bench (SOPBench/TaskBench/Synth) replication is **[EST]**.
- **Transfer.** Zero-retrain ABox-swap is the design premise; its cross-field validation is Paper 4 and is **[EST/partial]** here (SOPBench cross-domain is partially measured).
- **Construction artifact.** The gated gap=0 is by construction; we rely on the *floor* gap's non-closure for the substantive claim.

## 8. Conclusion

Scaling a tool-use agent buys capability — it retires load dimensions in a definite order and lifts task pass — but it does **not** buy compliance: policy-violation rates are flat across 7B→32B while pass climbs, and a deterministic gate removes them at every scale. The right object is therefore not the model size but the **function×scale×lever map**: a small base, a capability-scaffold that recedes as scale grows, a scale-invariant guarantee-scaffold, and authored A2 — with learning reserved for the domain-general residual. This is why even frontier systems should use deterministic scaffold and A2, and it sets up the system that optimizes the lever mix by cost (Paper 4), the methods that generate A2 (Paper 2), and the learned heuristics for the irreducible path-search residual (Paper 3).

---

*The complete bibliography (229 entries, organized by theme, deduplicated across the project's deep-research and literature-review corpus) follows below. Entries flagged `[unverified]` / future-dated are to be re-checked before submission.*

---
