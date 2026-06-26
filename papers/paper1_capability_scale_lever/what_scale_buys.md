# What Scale Buys in Tool-Use Agents, and How to Buy It Cheaply: A Capability×Scale×Lever×Cost Map Where Compliance Is Scale-Invariant

*Merged ICLR draft (Papers 1+4), branch `facet-rft-2026`. This paper is the realization of the project master plan `EXPERIMENT_DESIGN.md §0★★/★★★` — the capability×scale×lever×**cost** map — combining the measured science (what scale buys; ex-Paper 1) with the cost economics (how to buy it cheaply; ex-Paper 4). Companion papers: A2-generation (Paper 2) and learned path-selection (Paper 3) remain separate.*
*Numbers tagged **[SETTLED]** are measured with **zero paid frontier-API calls** (local user-simulator); **[EST]** are projections/planned. §7 is the funded forward plan ($0.5–1k budget).*

---

## Abstract

Tool-use agents (NL request → tool calls under a policy) are usually improved by scaling the base model. We instead ask, function by function and across scale: **what does scale actually buy, what can no model do at any scale, what is covered cheaply by deterministic scaffold or authored per-domain relations (A2) — and what is the total-cost-optimal way to assemble these?** Treating four benchmarks (τ²-bench, SOPBench, TaskBench, Synth) as measurement instruments, we decompose tool-use failure into **operand execution**, **orchestration load**, and **compliance/guarantee**, chart each across scale (7B–32B measured; 72B–235B **[EST]**) and lever (base / capability-scaffold / guarantee-scaffold / A2 / learning), and fold the map into a **total-lifecycle cost objective** (CapEx + OpEx + maintenance + experts + per-field generalization). Findings: (1) **operand execution saturates early** — a 32B base is 88/88 (100%) exact given a specification, so it is not a learnable gap at 32B. (2) **Capability-type load retires with scale in a definite order** — of five candidate load dimensions, interference and state-multiplicity bind at 7B/14B but vanish by 32B, while length and conditional-depth persist; a controlled probe overturns the observational interference signal. (3) **Compliance is scale-invariant** — task-pass climbs with scale (7B 0.19→32B 0.55) but the *compliant*-pass gap stays flat (~0.056) and policy-semantic violations do not fall (g2 31→41 across 7B→32B) even as auth-type violations collapse (g1 57→6); a deterministic gate eliminates them at every scale. We frame deterministic scaffold as a **load-reduction operator** (`pass ⇔ L_eff < L*(N)`, `L_eff = L − ΔL_scaffold`), and we measure a deployment payoff: on τ²-retail, a 32B-int8 base + deterministic gate runs on-prem at **≈23× lower $/req than a frontier API at near-equal compliance**, and a small-on-prem + escalation **fleet** reaches equal-or-better compliance at ~2.1× lower cost **[SETTLED on existing data; router decidable, projected]**. We conclude that **scale buys capability but not guarantee**, so even a frontier model needs a deterministic guarantee-layer, and the cost-optimal object is not the smallest model but the **total-cost knee**: a small base + a capability-scaffold that recedes with scale + a scale-invariant guarantee-scaffold + authored A2, with learning reserved for a small, scale-dependent residual.

---

## 1. Introduction

A tool-use agent maps a natural-language request to a sequence of tool calls satisfying a domain policy. The default recipe is a bigger base model — but "bigger" conflates distinct capabilities and is expensive on **two permanent axes**: inference OpEx (a 2× model is ~2×+ $/req and slower, forever) and, for on-prem/sovereign deployment, hardware CapEx (the operator *buys* the VRAM/power/cooling). The practical question is therefore not *how big a model* but **which capabilities must scale, which can be handed to cheap deterministic structure, which no model can be trusted to do, and what total-cost mixture is optimal.**

We answer with a **capability × scale × lever × cost** decomposition — the master map of `EXPERIMENT_DESIGN`. For each functional sub-capability we ask, across scale, whether it (a) grows with scale, (b) is a genuine model limit, (c) is covered by deterministic *scaffold*, or (d) is covered by per-domain authored relations (*A2*); we then assemble the per-capability levers under a total-lifecycle cost objective and locate the cost knee. Benchmarks are instruments for filling cells, not the object of study.

**Objective function (the economics, ex-Paper 4).** We minimize, *simultaneously*,
> `{ ① cheap GPU tier/count · ② low VRAM (int8/int4) · ③ maintenance per change · ④ human experts · ⑤ generalization cost per new field (ideally A2-swap only) } + HW CapEx + inference OpEx`.
The terms are coupled and point the same way: ①② → a **small model + quantization**; ③④⑤ → **maximize the fixed domain-general part** (deterministic scaffold + transferable, forgetting-free learning = one-time) and **minimize the per-domain A2**. The **common enemy is the A2** (it drives ③④⑤ and is worst on-prem, where data cannot leave so a frontier A2-generator is unavailable); the **common friend is fixed scaffold + transferable TBox**. The optimum is the **total-cost knee**, not the smallest model.

**Contributions.**
1. A controlled, **frontier-API-free** methodology isolating operand / orchestration-load / compliance across scale (§4): isolated single-shot probes + existing multi-scale runs + a local user-simulator avoid the cost and noise of paid full rollouts.
2. **Operand execution is scale-saturated, not learnable at 32B** (§5.1); the apparent gap is criterion-interpretation (deterministic compute) and join/disambiguation (present-addressable). Faithful-formalize SFT is **NO-GO at 32B** — but possibly **GO at 7B**, where a residual exists (§7).
3. A **load decomposition with a scale-response** (§5.2): the binding load set is *scale-dependent*; scale retires interference/state before length/conditional. Controlled generation overturns an observational correlation — the methodological payoff.
4. **Compliance/guarantee is scale-invariant** (§5.3): capability rises with scale but policy-semantic violation does not; a deterministic gate closes it at every scale. This is the empirical backbone of a **two-kind scaffold** — capability-scaffold recedes with scale (F3), guarantee-scaffold is scale-invariant (F4) — and explains *why even frontier systems need a deterministic guarantee layer*.
5. A **total-cost-optimal lever allocation** with a measured deployment payoff (§5.5–6): decidability-first routing, ≈23× $/req TCO advantage and a 2.1× fleet, and the cost-knee framing — the engineering realization of the map.
6. A unifying frame: **scaffold = a load-reduction operator** (§3), `pass ⇔ L_eff < L*(N)`, with an *independent* estimate of `ΔL` (we establish the components; the full predictive test is §7 [EST]).

Our headline is deliberately **not** "small beats large" (held by RL-routing systems, §2). It is *which* parts of the gap are scale-invariant and must be offloaded, and the cost-optimal map telling a deployer which lever to pull per capability.

## 2. Related Work

The contribution sits at an intersection of verified prior pieces; the central lineage is conceptual (why an LLM is good at language but must offload deep procedures and cannot self-guarantee compliance), not the recent "small-beats-large" routers (§2.5).

**§2.1 Language is shallow-parallel; deep procedures must be offloaded.** A bounded-depth Transformer forward pass is asymptotically uniform TC⁰ (Merrill & Sabharwal, 2207.00729); CoT escapes only by serializing depth (Feng et al., 2305.15408/2402.12875). Natural language embeds procedures of definite class — van Benthem's semantic automata ("most" is pushdown), cognitively real (working-memory/PFC scaling); Relevance Theory's procedural/conceptual split; Goodman notationality (cost not computability, per Zhang & Norman). So *classifying* which deep procedure a request embeds is shallow and learnable; *executing* it is serial-deep and offloaded — the basis of capability/lever routing.

**§2.2 Cognitive architectures, non-introspective decidability, scale-invariant epistemic limits.** Our system instantiates the CoALA umbrella (Sumers et al., 2309.02427, TMLR), making concrete what it leaves general (a relational-algebra deterministic core; empty-relation abstention; zero-retrain A2-swap). SOAR (Newell/Laird/Rosenbloom 1987; Laird 2012; Wray/Kirk/Laird, 2505.07087) surfaces indecision *architecturally* (an `impasse` from absent preference, not introspection) — the precedent for "uncertainty → observable empty/ambiguous relation → ASK." Deterministic-first/LLM-fallback is established (Zhu & Simmons, 2403.00810, 50–100× fewer tokens; NL2GenSym, 2510.09355). We add a **formalize-faithfulness** axis (SOAR assumes productions correct), a *learned* empty→ASK, and a small weight-learned domain-general skill with A2-swap. On epistemics, hallucination is provably scale-invariant (Xu et al. 2401.11817; Kalai et al. 2509.04664: scoring rewards guessing over abstention); **our §5.3 is the policy-enforcement analogue** — no model self-guarantees compliance, just as none self-guarantees truthfulness; both demand an external deterministic mechanism.

**§2.3 LLM-proposes / deterministic-executes.** LLM-Modulo (2402.01817), PAL, NL→PDDL formalizers, RLVR with deterministic checkers (AlphaGeometry/Proof; Tülu 3, 2411.15124). Honest boundary: offload does not always win (formalization overhead on simple cases; spec leakage) — the boundary is itself measured.

**§2.4 Guarantees / shielding.** Safe-RL shielding (Alshiekh et al. 2018; Jansen et al. 1807.06096), predictive safety filters (Wabersich & Zeilinger 2021); LLM-agent instances ShieldAgent (2503.22738), AgentSpec (2503.18666), Formal-LLM (2402.00798). These veto with hand/probabilistic specs; our gate is deterministic, *drives* (repairs) preconditions, and pairs with a learned proposer and zero-retrain transfer.

**§2.5 Learned routing / "small-model orchestration" (the narrow rivals).** Routing is a learned transferable skill (RTR, A2FM, xRouter, ARM2, When2Call — tool-necessity linearly decodable, AUROC 0.89–0.96; RITE cross-domain). **ToolOrchestra (2511.21689)** RL-trains an 8B router to delegate to a frontier model (80.2%@10.3¢ vs GPT-5 77.7%@31.3¢ on τ²) — strongest cost result, **but relies on frontier delegation**, disqualifying for the on-prem/no-egress regime; it optimizes cost *given* black-box components and provides no deterministic offload, by-construction compliance, decidable-vs-learned accounting, or zero-retrain transfer. **TRUST (2606.06976)**: 4B>30B by RL, black-box, no guarantee. **ReDAct (2604.07036)** / Unified Routing-Cascading (2410.10347): calibrated deferral; our fleet is a *deterministic-router* instance. These establish that a cost-favorable small-model result *exists* — which is why our contribution is the *analysis and method* (what is scale-invariant; which lever per capability; the cost knee), not that result.

**§2.6 Cost / regulatory / industrial.** Against a pure-scale reading of the Bitter Lesson, Brooks' "A Better Lesson" (2019) and Chollet (1911.01547) argue total cost / priors must be counted — our knee operationalizes this. Buyer value of determinism = verifiability / model-risk exemption (SR 26-2; EU AI Act 2024/1689; EU MDR Annex I §17.1 repeatability). **Palantir Foundry** (Ontology ≈ scaffold+A2; AIP-LLM ≈ small model + transferable TBox) is the industrial baseline; we differentiate on *cost* (HW, experts, config-only field-crossing, zero egress), **not functionality**.

**§2.7 Load / benchmarks.** Load framing defers to cognitive-load theory and long-context "lost-in-the-middle"; novelty is only *load-decomposition × scaffold-load-reduction for tool-use*. Benches as instruments: τ² (2406.12045), SOPBench (2503.08669), TaskBench, Synth/CFB (2501.10132); floor-measurement precedent: "Sufficient Context" (Joren et al., ICLR 2025).

**Whitespace.** Prior work supplies each piece (procedural-semantics/automata, Transformer-expressivity, SOAR/CoALA, hallucination-inevitability, formalize-offload, learned routing, shielding, cost-routing). The unoccupied intersection is the **empirical capability×scale×lever×cost map**: which capabilities scale retires, which are genuine scale-invariant limits (compliance, measured), which lever covers each, and the **total-cost-optimal assembly**. Recent routers optimize cost over black boxes; we open the box.

## 3. The Capability×Scale×Lever×Cost Framework

We decompose a tool-use trajectory into typed functional steps; for each function `f` we populate a row indexed by scale `N`, lever `ℓ ∈ {base, capability-scaffold, guarantee-scaffold, A2, learning}`, and **assembled cost**.

**Three structural claims.**
- **Operand vs orchestration vs compliance** fail for different reasons and respond to different levers: operand = atomic execution given a criterion; orchestration = holding interdependent steps coherent under *load*; compliance = obeying policy with a *guarantee*.
- **Decidability-first lever routing** (cost-ascending short-circuit): `decidable from schema+policy+state → scaffold`; `else domain fact → A2`; `else promptable at size → prompt`; `else scale-emergent & affordable → scale or cheap-replication`; `else irreducible NL→formalize → minimal transferable LoRA (last resort)`. This inverts "scale it."
- **Two kinds of scaffold.** *Capability-scaffold* (present/compute/resolve) substitutes for a missing skill and **recedes as scale grows** (F3, raw pass). *Guarantee-scaffold* (policy gates) supplies a property no probabilistic model gives and is **scale-invariant** (F4, compliant-pass). §5.3 measures the split — it is *why even frontier keeps a deterministic trust layer*.

**Load as a vector.** Orchestration "load" is not scalar. From failure forensics: **L_len** (context to retain), **L_state** (interdependent state), **L_branch** (conditional depth), **L_interf** (confusable entities), **L_contra** (mid-stream revision). Crucially **load ⊥ operand difficulty** (raise load with operand fixed).

**Scaffold as a load-reduction operator.** Each scaffold removes a dimension: present/autofetch ↓L_len,↓L_interf; compute ↓L_state; gates ↓L_branch; controller ↓L_state,↓L_len. This gives `L_eff = L − ΔL_scaffold` and the prediction `pass(f,N) ⇔ L_eff(f) < L*(N)`, where `L*(N)` is load-tolerance at scale `N`. To avoid tautology, `ΔL` is estimated *independently* (from the mechanical feature-reduction the scaffold performs) and shown to predict the failure-onset shift — not read off it. We establish the components here; the full predictive curve at >32B is §7 **[EST]**.

## 4. Method

**Benches as instruments.** τ²-retail (primary); SOPBench/TaskBench/Synth are the canonical *training* substrate for Papers 2–3 and the lever-allocation; τ² is a *transfer* target, never trained on. Scale: Qwen2.5 7B / 14B / 32B-GPTQ-Int8 **[SETTLED]**; 72B / 235B **[EST]** (§7).

**Cost discipline (frontier-API-free).** The only paid component is a GPT-4.1 user-simulator; we avoid it for discovery. Capability is measured with **isolated single-shot probes** and **existing multi-scale runs on disk**; a **local user-simulator** (faithful scripted turns over a local base model) replaces the paid sim for full-flow checks. Paid full-runs are used only as a **final confirmation, minimal scope, after a free conclusion** (the budget posture of §7). All results persist under `sim_results/`.

**Conditions.** *floor* (no scaffold); *g15 / present / nested / assembled* (capability- and guarantee-scaffold); *GIVEN-SPEC vs GOAL* (separates operand execution from criterion interpretation).

**Metrics.** Robust **pass^all** over per-trial point estimates (user-sim noise ≈ 0.11, so single-trial is not reported as a finding); **F3** = raw task pass; **F4** = compliant pass (deducting policy violations); per-dimension **partial correlation** of failure with each load feature (controlling operand difficulty); controlled **accuracy-vs-load** onset curves; and the **cost row** per cell (latency, tool-roundtrips, VRAM/GPU tier, $/req).

## 5. Results

### 5.1 Operand execution is scale-saturated, not a learnable gap at 32B [SETTLED]
Varying only the user-sim input: given an explicit option spec (**GIVEN-SPEC**) the 32B base selects the correct variant **88/88 (100%)**; given only the goal (**GOAL**) it is 62/88 (70%), the 30% gap being criterion interpretation (argmax/argmin = deterministic compute; multi-attribute reasoning; conversational fidelity), not execution. Genuine join/disambiguation (user describes, not names, the order) is 7/13 (54%) and present-addressable. **Implication:** operand is not a 32B capability gap, and faithful-formalize SFT is **NO-GO at 32B** — no residual to learn. *(Forensic note: every prior "operand failure" dissolved into a measurement artifact — a calc bug, a premature case, an ID-given probe artifact — until the controlled probe read 100%. The corollary that matters for cost: at 7B the base may be genuinely weak even given-spec, so the learning lever can re-acquire a job at small scale — §7.)*

A complementary limit: *selection over a large candidate set does not scale in-head* — `comparative@50 = 0.02`, `rank@50 ≈ 0.30` even including a 235B model, while a deterministic engine is 1.00 **[SETTLED]**. "Pick the best among many" is offloaded at any scale.

### 5.2 Capability-type load retires with scale, in order [SETTLED 7B–32B]
Partial correlation of *floor* failure with each load feature (controlling operand difficulty):

| dimension | 7B | 14B | 32B |
|---|---:|---:|---:|
| L_len (length) | +0.20 | +0.33 | **+0.37** |
| L_state | **+0.24** | **+0.21** | +0.06 |
| L_branch (conditional) | +0.20 | +0.13 | **+0.26** |
| L_interf | **+0.30** | **+0.20** | +0.08 |
| L_contra | +0.06 | +0.13 | +0.04 |

The **binding set is scale-dependent**: at 7B/14B four dimensions predict failure; by 32B only length and conditional-depth survive (interference/state retire first). L_contra is too sparse in τ² to fit. **Controlled > observational:** a controlled interference probe (fix operand; vary only N confusable variants) does *not* reproduce the strong 7B signal —

| N | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 32B | 1.00 | 1.00 | 1.00 | 0.97 | 0.87 |
| 7B | 1.00 | 1.00 | 1.00 | 0.90 | 0.80 |

(0.80 vs 0.87 at N=16): the observational L_interf was largely a size confound. *(Caution: single-shot probes under-test inherently multi-turn dimensions L_state/L_contra; controlled multi-turn probes are §7 [EST].)* **The residual is orchestration, not operand:** re-running the robust fail-all set under faithful local turns, 2/10 are sim-noise flips and the dominant residual is *orchestration-under-load* (batching that violates a precondition; multi-order tracking dropping an order + fabricating an id; conditional sequencing; context overflow). An isolation **plan probe** (elicit the abstract plan, reads pre-supplied, grade structure only) shows correct structure in 6/10 — i.e. ~half the orchestration residual is execution-load (closable by deterministic plan/execute separation) and ~half is in-isolation planning miss (the candidate for a learned path-selection lever, Paper 3). Assembled-stack ceiling = robust pass^all **0.402** **[SETTLED]**.

### 5.3 Compliance is scale-invariant — the central finding [SETTLED 7B–32B]
Separating **F3** (task pass) from **F4** (compliant pass) on *floor* (no gate):

| scale | F3 | F4 | gap | viol (g1 auth / g2 policy) | total viol% |
|---|---:|---:|---:|---|---:|
| 7B | 0.189 | 0.130 | 0.058 | 57 / 31 | 33.6% |
| 14B | 0.468 | 0.404 | 0.064 | 38 / 38 | 26.3% |
| 32B | 0.547 | 0.491 | 0.056 | 6 / **41** | 14.3% |

The F3–F4 gap is flat (~0.056) and **policy-semantic violations (g2) do not fall (31→41)** even as auth-type (g1) collapses (57→6). With the deterministic gate (g15): F3=F4, gap = **0.000**, viol = **0.0%** at every scale (7B=14B=32B).

**Interpretation.** Scale solves *capability-type* (g1) but **not** *policy-semantic* (g2) compliance; the gate zeros it at every scale, frontier included. This is the policy-enforcement analogue of hallucination-inevitability (§2.2) and the backbone of the two-kind scaffold.
**Resolving the completion-rate confound [SETTLED].** Absolute g2 count (31→41) and total viol% (33.6→14.3) are confounded by completion: a higher-pass model completes more trajectories and thus attempts more writes (526→683→680 across 7B/14B/32B), so raw counts conflate ability with exposure. Normalizing to the **per-write-opportunity rate** removes this — g2/write is **0.103 [0.080, 0.132] (7B), 0.070 [0.053, 0.092] (14B), 0.075 [0.058, 0.097] (32B)**: flat across scale, with overlapping 95% Wilson CIs and no significant downward trend (if anything 7B is marginally *higher*). So scale does **not** reduce the confirm-before-write violation rate (~1 in 10–14 writes, at every scale), while the deterministic gate zeros it everywhere — the **strong scale-invariant form holds, not merely the differential** (the differential is also clear: auth-type g1 collapses 57→6 while g2 persists). Range 7B–32B; the 72B point (§7) extends it.

### 5.3b Cheap-replication: a capability-scaffold substitutes for scale [SETTLED]
Raw pass rises monotonically (pass^1 = 7B 0.24 / 14B 0.52 / 32B 0.60, n=342) — the "scale buys capability" curve. The map's point is that *each piece is obtainable more cheaply than scale*: a deterministic **fetch-first** engine (provenance-deny → producer call → inject real value) moves grounding errors 33→9 and roughly **doubles pass (0.14→0.264) with zero learning** — a capability-scaffold standing in for scale. Lever audit: auth/confirm/ownership/precondition gates (G1–G4) are genuine levers; an eligibility-*steering* gate (G5) has ~zero causal effect (the model does not learn to use guidance) and naive retry is counter-productive — populating the "capability-scaffold recedes / guarantee-scaffold invariant" split with measured cells.

### 5.4 The map, filled
| function | scale-response | genuine limit? | efficient lever |
|---|---|---|---|
| operand execution | saturates by 32B (100% given-spec) | no (at 32B) | base; **learning may re-acquire job at 7B** (§7) |
| interference / state load | retire by 32B | no | capability-scaffold (present), or scale |
| length / conditional load | persist at 32B | partial | capability-scaffold (partial); conditional → controller / learn |
| large-N selection | flat at all scales | yes (in-head) | deterministic compute |
| policy compliance (g2) | **scale-invariant (per-write rate flat, CIs overlap)** | **yes (guarantee)** | **guarantee-scaffold (gate)** |
| domain facts | n/a | n/a | A2 / retrieval |

### 5.5 Cost: the deployment payoff [SETTLED on existing data]
- **≈23× TCO advantage.** On τ²-retail with existing runs: 32B-int8 + deterministic gate, on-prem, **$0.0019/req vs frontier-API $0.044/req ≈ 23×** (16–40× over GPU $0.2–0.5/hr), ~3000× cheaper than a human agent (~$6/contact); on-prem, auditable, zero egress. Honest trade-offs: latency ~6× (178s vs 30.5s); pure-on-prem compliance (0.573) is *below* frontier (0.82) → headline is "near-compliance, ~23× cheaper," not "equal." *( $/req estimated from token≈chars/4 + assumed GPU utilization — order-of-magnitude robust; exact multiple pending the litellm token-capture fix, §7.)*
- **Fleet (equal-or-better compliance, cheaper).** Easy→32B on-prem, hard→frontier escalate: blended compliance 0.860 (> pure frontier 0.816) at $0.021/req (~2.1× cheaper) on clean data **[SETTLED arithmetic; assumes a cheap *decidable* router, not yet implemented]**.

## 6. Cost-Optimal Lever Allocation and the Knee

The map yields a deployment rule that inverts "scale it": **reduce the load and supply the guarantee with structure; do not train the model to tolerate either.** Capability-scaffold removes the load dimensions scale would otherwise buy (and recedes when scale is available); guarantee-scaffold supplies the property scale never buys (kept at every scale). Per-capability allocation is decidability-first (§3); the **common enemy is the A2** (minimized; Paper 2 auto-generates it), the **common friend is fixed scaffold + transferable TBox**. The optimum is the **cost knee** — the smallest *total-cost* size — and, more generally, a **heterogeneous on-prem fleet** whose mix is set by per-capability learn-vs-engine crossover, not a static assumption. Where a load is irreducible by scaffold (conditional-depth / path-search is our candidate), the lever shifts to a *learned, transferable* heuristic (Paper 3), not a bigger model. Positioning vs **Palantir Foundry** is on cost (HW/experts/field-crossing/egress), not functionality.

## 7. Funded Forward Plan (advancing the master map; budget ≤ $1k)

This paper fills the *measured* cells of `EXPERIMENT_DESIGN`'s 8-item checklist (items 1–4, 6 partial, 7); the budget below fills the highest-value remaining cells. Discipline: free local verification first; each paid run is a **confirmation of a free conclusion**, minimal scope.

1. **[DONE — FREE] g2 per-opportunity rate + CI** — *settled* the §5.3 backbone: g2/write is flat across scale (0.103/0.070/0.075, overlapping 95% CIs), so the strong scale-invariant form holds (not just the differential). Prerequisite to submission, cleared.
2. **[≈$30–80] 72B compliance point** — extends F3/F4 invariance from ≤32B to ≤72B (eval-item 2/7); the single largest credibility gain for the central claim. One scale, gated arm, local user-sim where possible.
3. **[≈$50–150] 7B/14B scale axis with the *same* scaffold** — eval-item 2/3 (scale plan, §0★★★ AM): does the fixed scaffold+A2 carry small models, and where does the base become too weak so the **learning lever re-acquires a job** (operand/orchestration residual exists at 7B though not 32B)? This is the cost-knee's left edge and tests whether cheap-replication moves the knee toward 7B.
4. **[≈$50–150] cheap-replication confirmations** — fetch-first / present / controller-vs-learn on the orchestration residual (eval-item 3): confirm scaffold ΔL on a clean robust measure; estimate `ΔL` independently and test the load-reduction prediction (§3) at ≥2 scales.
5. **[≈$80–200] multi-field zero-retrain transfer (partial)** — airline / banking A2-swap, fixed-part-invariant (`grep "if field"=0`, zero retrain), A2-swap effort, retention (eval-item 5). Structurally distant fields (healthcare) and the full ⑤ column remain [EST]; this upgrades transfer from premise to *partially measured*.
6. **[≈$40–100] $/req token-capture fix + the knee sweep** — turn on litellm cost/token so §5.5 $ becomes measured; sweep size vs total cost to locate the knee / fleet mix (eval-items 6/8).

Total ≈ $300–800, within budget, prioritized so the backbone (1–2) lands first. Items that would exceed budget or require frontier delegation (full master matrix; 235B; full multi-field) stay [EST].

## 8. Discussion

The result reframes the deployment question from model size to **lever allocation under cost**. Two asymmetries do the work: capability is scale-retired *in a definite order* (so a recede-able capability-scaffold buys the small-model the dimensions scale would), and compliance is *scale-invariant* (so a guarantee-scaffold is kept even at frontier). Training a model to tolerate load is training it to do the deterministic controller's job; the thesis routes that work to determinism by default and to learning only for the residual — a residual that is **scale-dependent** (empty at 32B operand, non-empty at 7B), which is precisely why the cost knee, not a fixed model size, is the right object.

## 9. Limitations

- **Scale range.** Measured 7B–32B; 72B/235B/frontier are [EST] (§7 item 2). The scale-invariance claim is a flat-trend + priors, strengthened by the 72B point once run.
- **Compliance metric.** The strong "scale-invariant" form is now supported by the per-write-opportunity g2 rate (flat across 7B–32B, overlapping CIs; §5.3), not only the differential. It still rests on one violation class (confirm-before-write) on one bench; cross-class and cross-bench replication, and the 72B point, are [EST] (§7).
- **Observational vs controlled.** §5.2 correlations are size-confounded; only the controlled L_interf probe is causal; L_branch/L_len controlled curves and multi-turn (L_state/L_contra) probes are [EST].
- **Cost numbers.** $/req is token-estimated (order-of-magnitude robust); the fleet assumes a decidable router not yet built; the human/Palantir comparisons are external baselines.
- **Transfer.** Zero-retrain A2-swap is the design premise; cross-field validation is [EST/partial] (§7 item 5) — and the "small + schema-swap → unseen domain" mechanism is itself a variant of schema-guided DST / ToolLLM (we claim the scale-invariance map, not the transfer mechanism, which belongs to Papers 2–3).
- **Single bench for compliance.** F3/F4 on τ²-retail; cross-domain/cross-bench is [EST].

## 10. Conclusion

Scaling a tool-use agent buys capability — it retires load dimensions in a definite order and lifts task pass — but it does **not** buy compliance: policy-semantic violations are flat across 7B→32B while pass climbs, and a deterministic gate removes them at every scale. The right object is therefore not the model size but the **capability×scale×lever×cost map**: a small base, a capability-scaffold that recedes as scale grows, a scale-invariant guarantee-scaffold, authored A2, and learning reserved for the scale-dependent residual — assembled at the total-cost knee. This is why even frontier systems should use deterministic scaffold and A2, and it grounds the companion papers that generate the A2 (Paper 2) and learn the path-selection residual (Paper 3).

---
*[SETTLED] cells are measured (frontier-API-free); [EST] cells are the §7 funded plan. Bibliography (≈229 entries, deduplicated across the project corpus) follows; `[unverified]`/future-dated entries to be re-checked before submission.*
