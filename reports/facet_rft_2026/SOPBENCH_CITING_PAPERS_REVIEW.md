# SOPBench (Zekun Li, 2503.08669) citing-paper review — vs our approach (2026-05-31)

Scope: every paper that cites **SOPBench** (Zekun Li et al., arXiv **2503.08669** — the *primary*
benchmark, 7 domains, native formal operators + rule oracle), pulled from the Semantic Scholar
citation graph (8 papers as of 2026-05-31). Goal: careful similarity/difference comparison
against **our approach**. ⚠️ Caveat: S2 lags on the newest arXiv; treat as near-complete, not
provably exhaustive. (Do not confuse with **SOP-Bench**, Amazon 2506.08119 — our *auxiliary*.)

## Our approach (the comparison baseline)
**LLM-in-loop, 2-stage:** a **learned, domain-general planner (TBox)** emits abstract
operator-level steps (means-ends over operator precondition/effect *types*, GoalAct-style
global re-plan), and an **ABox-conditioned resolver** maps each step to a concrete tool call.
The **ABox = domain operators as external, swappable memory** (the resolver's rungs: rule /
ontollm-prompt / **neural cross-attention memory = ★novelty**). **Headline = transfer:** freeze
the planner, **swap the ABox to a held-out domain, retrain nothing** (+ ABox-ablation control).
The deterministic executor over ground-truth call graphs is the **oracle/ceiling, not the
contribution**. Ontology induced from traces and **validated against ground truth**
(`directed_action_graph`). Benchmark: SOPBench (primary) → AppWorld (Phase 2).

---

## A. Method papers closest to us (the real "related work / differentiate hard")

### A1. FM SO.P — Progressive Task Mixture for Cross-Domain SOP Understanding (2602.09336, 2026)
*Evaluated on **SOPBench**, same 7 domains (Bank/DMV/Healthcare/Market/University/Library/Hotel).
32B = 48.3%, 7B = 34.3% ≈ Qwen-2.5-72B (34.4%) at 10× fewer params.*
- **Similar:** same primary benchmark + same 7 domains; **cross-domain SOP generalization** is
  the shared target; staged capability building (concept disambiguation → action-sequence →
  **scenario-aware graph reasoning**) parallels our **L0→L1→L2 ladder** and our scenario/branch
  relations; "scenario-aware graph reasoning for conditional logic" ≈ our `scenario_select`/
  `next`-fork. Strong small-model results (7B≈72B) echo our "procedure is distillable" thesis.
- **Different (the crux):** FM SO.P attains cross-domain ability by a **training-data curriculum
  (SFT mixture)** baked into **one monolithic model's weights** — confirmed: *not* architecture,
  *not* external memory, **no general-planner / domain-operator separation**, and **no
  held-out-domain zero-retrain transfer** (it trains across the domains). We instead **separate
  the general planning skill (weights, TBox) from domain operators (external swappable memory,
  ABox)** and prove transfer by **swapping the memory with zero retraining**. Their eval is an
  automatic multi-agent rubric (LLM-judge-style); ours is the SOPBench **rule oracle** (no judge).
- **Positioning:** our single most important "compare-and-beat." Same benchmark/domains, opposite
  mechanism (curriculum-into-weights vs swappable-operator-memory + transfer). We should run FM
  SO.P–style curriculum SFT as a **baseline arm** and show our memory-swap transfers where their
  baked-in capability would need re-training per new domain.

### A2. CAP-CPT — Analyzing & Internalizing Complex Policy Documents (CC-Gen) (2510.11588, 2025)
*Parses policy → factual/behavioral/conditional categories → Category-Aware Policy Continued
Pretraining; 97.3% prompt-length reduction, +41%/+22% on Qwen-3-32B, also helps τ-Bench.*
- **Similar:** directly adjacent to our **internalization** thread (NONE-vs-FULL, the OISA-patent
  track) and to our ontology categories — their **factual/behavioral/conditional** split ≈ our
  **slot / operator-effect / precondition-&-branch** relations; both target **workflow-complexity
  from conditional policy**; both parse a document into structured specs to drive data synthesis.
- **Different:** CAP-CPT **internalizes each policy into the model's weights** via continued
  pretraining — confirmed: **does *not* separate general reasoning from swappable policy**, and
  describes **no held-out zero-retrain transfer**. Goal = prompt compression (policy→priors). We
  keep operators **external and swappable** so the *same* planner serves new domains without
  re-pretraining. Their "conditional category isolation" stays in-weights, per-policy; ours is a
  runtime operator the planner consumes.
- **Positioning:** closest prior on *policy-into-weights*. Differentiate: internalization-per-
  policy (theirs) vs **general-planner + swappable-operator-memory transfer** (ours). Also a
  bridge to our patent track (context→weights), reported separately.

---

## B. Benchmark / formalization papers (share our framing; eval-only, no transfer method)

### B1. SAGE — Service Agent Graph-guided Evaluation (2604.09285, 2026)
- **Similar:** **formalizes unstructured SOPs into Dynamic Dialogue Graphs** (≈ our call-graph
  ontology) and scores with a **Rule Engine → deterministic ground truth** (≈ our rule oracle /
  `directed_action_graph` conformance); multi-domain (6 industrial). Its headline finding — the
  **"Execution Gap": models classify intent correctly but fail to derive the correct subsequent
  action** — is *direct empirical motivation* for our operator-level planner (the gap is exactly
  what means-ends planning over operators targets).
- **Different:** SAGE is an **evaluation benchmark** (+ data-synthesis + adversarial-intent
  taxonomy), **not a method** — no learned planner, no operator-memory, no transfer. Uses **Judge
  Agents** alongside the rule engine (we avoid LLM judges on the primary). We can cite SAGE's
  Execution-Gap as motivation and its graph formalism as convergent evidence for our ontology.

### B2. SOP-Maze — LLMs on Complicated Business SOPs (2510.08942, 2025) *(already our Tier-2 aux)*
- **Similar:** isolates **SOP branch-following / deep conditional reasoning** (HRS) and
  **wide-option selection** (LRS) — HRS ≈ our planner's branch-following, **LRS ≈ our
  tool-selection@scale (`--tool_list full`)**. Error taxonomy (route-blindness, conversational
  fragility, calculation errors) maps onto failure modes our planner/resolver target.
- **Different:** pure **decision/QA eval, no tool execution, no method/transfer.** We already use
  it as the auxiliary surface to isolate the planner's pure reasoning (tool-free).

### B3. TOD-ProcBench — Complex Instruction-Following in Task-Oriented Dialogues (2511.15976, 2025)
- **Similar:** models procedures as **multi-level condition-action statements** (≈ our
  precondition→action / If-branches); Task-1 = **retrieve relevant statement + predict next
  action** (≈ our planner's next-operator); studies violation detection (≈ our constraint axis).
- **Different:** **multi-turn dialogue / NLU-centric eval** (closer to tau2 than to agent-executes-
  tools), benchmark-only, no transfer method. Useful as a related instruction-following eval, not
  a competing method.

### B4. Complex Logical Instruction Generation — LogicIFGen/LogicIFEval (2508.09125, 2025)
- **Similar:** generates **verifiable instructions from code functions** (conditions, loops,
  **function calls**) — strongly echoes our design principle **"push complexity into functions;
  the ontology is a thin call-graph"**; verifiable/auto-checkable like our rule oracle.
- **Different:** single-turn **instruction-following eval**, not agentic tool use or planning, no
  transfer. Convergent evidence that "logic-rich procedure = code/functions," but not a method peer.

---

## C. Safety / orthogonal (touch our constraint axis only)

### C1. Outcome-Driven Constraint Violations benchmark (2512.20798, 2025)
- **Similar:** **constraint-violation / refusal** focus (≈ our `action_should_succeed=false`
  refusal-accuracy axis + `constraint_not_violated`); multi-step agent tasks; KPI-tied.
- **Different:** an **alignment/safety eval** of *emergent misalignment under KPI pressure*
  (Mandated vs Incentivized), 4-model judge panel — no planning method, no transfer. Orthogonal
  contribution; our refusal metric overlaps but our thesis is planning+transfer, not misalignment.

### C2. LLM Agents Should Employ Security Principles — AgentSandbox (2505.24019, 2025)
- **Similar:** least-privilege / tool-access mediation loosely touches our refusal & tool-scope
  axes.
- **Different:** a **security position paper + conceptual sandbox** (defense-in-depth, complete
  mediation, privacy); cites SOPBench only as an agent-benchmark precedent. **Least relevant** to
  our planning/transfer thesis.

---

## Synthesis — where we stand

1. **Our core mechanism is unclaimed by any citing paper.** None proposes a **learned
   domain-general planner (TBox) + external swappable operator memory (ABox) with zero-retrain
   cross-domain transfer.** The two closest *methods* — FM SO.P (A1) and CAP-CPT (A2) — do the
   **opposite**: they bake SOP/policy capability **into one model's weights** (curriculum SFT /
   continued pretraining). That makes our TBox/ABox separation + memory-swap transfer a sharp,
   defensible novelty *and* gives us two concrete baseline arms to beat.
2. **The problem framing is strongly validated.** SAGE's **"Execution Gap"** (intent-right,
   action-wrong) is textbook motivation for operator-level planning; SAGE/SOP-Maze/TOD-ProcBench/
   LogicIF independently converge on **graph/condition-action/code-function** formalizations of
   SOPs — the same shape as our 8-relation call-graph ontology. Convergent formalisms = the field
   agrees the representation is right; we add the *learned planner + transfer* nobody else has.
3. **Methodological edges to emphasize:** (a) **rule oracle, no LLM judge** (SAGE/constraint-bench
   lean on judge panels); (b) **induced ontology validated against ground-truth** call graphs;
   (c) **refusal/constraint axis** (`action_should_succeed=false`) as a first-class metric, which
   the safety papers (C1) treat as their whole point — we get it for free on the primary.
4. **Risks / must-cite:** FM SO.P (A1) shares our exact benchmark+domains and the "small model
   matches big" result — we must (i) cite it, (ii) run a curriculum-SFT baseline, (iii) show
   transfer-by-memory-swap is the discriminator. CAP-CPT (A2) is the closest *internalization*
   prior — cite and contrast (per-policy-into-weights vs swappable-memory transfer); also ties to
   the separate OISA patent track.
5. **Auxiliary uses already in our plan:** SOP-Maze (B2) as the tool-free planner-reasoning probe;
   SAGE's graph formalism + Execution-Gap as motivation; LogicIF as evidence for "complexity in
   functions."

### One-line table
| Paper | Type | Shares with us | Lacks (vs us) |
|---|---|---|---|
| FM SO.P (A1) | method | SOPBench+7 domains, cross-domain, staged | curriculum-into-weights; no TBox/ABox split; no zero-retrain swap |
| CAP-CPT (A2) | method | policy→structured categories, internalization | per-policy into weights; no swappable memory; no held-out transfer |
| SAGE (B1) | benchmark | SOP→graph, rule-engine GT, Execution-Gap | eval-only; judge agents; no planner/transfer |
| SOP-Maze (B2) | benchmark | branch reasoning (HRS), wide-option (LRS) | tool-free QA; no method/transfer |
| TOD-ProcBench (B3) | benchmark | condition-action, next-action predict | dialogue/NLU eval; no transfer |
| LogicIF (B4) | benchmark | logic from code functions, verifiable | single-turn IF; no agent/transfer |
| Constraint-Violations (C1) | safety eval | constraint/refusal axis | misalignment study; judge panel; no method |
| AgentSandbox (C2) | position | least-privilege/tool-scope | security framing; orthogonal |
