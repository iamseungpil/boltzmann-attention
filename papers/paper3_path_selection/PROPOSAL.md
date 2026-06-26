# Paper 3 (proposal) — Transferable Learned Heuristics for Path-Selection under Tool-Explosion

*Research proposal. ICLR-style. Branch `facet-rft-2026`. Source design doc: `../../scripts/distill/PATH_SELECTION_AXIS_DESIGN_2026_06_19.md`.*
*⚠️ Confidentiality: any proprietary application context (deployment domain, patent specifics) is kept **local-only** and is deliberately omitted here. This proposal uses public benchmarks (TravelPlanner, AppWorld, NATURAL PLAN) and an academic framing only.*

---

## 1. Problem & motivation

Many real tool-use settings present, at a single step, **tens to hundreds of homogeneous tools** (same type, differing parameters) and a **combinatorially exploding space of multi-step paths**. The task is to find the *most appropriate path* under constraints and preferences. This is **search / planning**, and it is a *different failure mode* from the provenance/grounding failures studied in Paper 1 (τ²-retail: id-fabrication, fetch-omission). The two must not be conflated: Paper 1's axis is *value grounding*; this paper's axis is *path search*.

We already have settled evidence that the naive routes fail. In-head selection over large candidate sets does **not** scale: depth-scale probes give `comparative@N=50 = 0.02` (including a 235B model) and `rank@N=50 ≈ 0.30`, while a deterministic engine is trivial (1.00) **[SETTLED]**. So "pick the best among hundreds" cannot be done by raw abstraction even at frontier scale; and a sequential composition of such picks (a path) is at least as hard. This motivates a division of labor.

**Bridge from Paper 1.** Paper 1's load decomposition shows that **conditional-depth / planning load (L_branch) persists at 32B and resists the deterministic capability-scaffold** that closes the other load dimensions. That residual — the part scaffold cannot delete and scale does not retire — is exactly the path-search capability this paper targets. Paper 1 hands it off; Paper 3 addresses it.

## 2. Core research question

> Does a small LLM's **learned, domain-general abstraction** let it *select good paths* in a tool-exploded new domain — or is path-search inherently deterministic offload (search + memory), with the LLM contributing only formalization and heuristics?

## 3. Division of labor (thesis-consistent)

- **Learned (domain-general, ABox-swap transfer):** ① formalize the objective/constraints ("what is a good path"); ② recognize the **applicable operations** at each state (generator routing); ③ estimate a **value/heuristic** ("is this partial path promising").
- **Deterministic / offloaded (decidable):** ④ **search** (MCTS / beam / A*) given the heuristic; ⑤ **memory** (execution success/failure, user preference) as value input.

The **novel, transferable contribution is ②③** — *learned heuristics that make search tractable* and transfer to a new domain with **zero retraining (ABox-swap)**. The search engine and memory (④⑤) are reused from prior work (MCTS, LATS, Tree-of-Thought, success/preference memory), not reinvented.

## 4. Falsifiable hypotheses

- **H_offload:** path-search is pure offload; learned abstraction provides no useful heuristic (depth-scale extrapolation: sequential large-N is *worse* than single). Contribution reduces to "path = offload" (weak, but an honest negative that maps the thesis boundary).
- **H_abstract (headline):** deep learned abstraction yields a **transferable heuristic** that beats **both** (a) heuristic-free search (naive MCTS at the same budget) **and** (b) large-model in-head path selection. Contribution: "small + transferable heuristic > search-alone and large-alone."

**What decides it:** a heuristic **ablation** (LLM-value on/off at fixed search budget) × **transfer** (heuristic learned on training benches → target via ABox-swap, zero retrain) × **scale control** (7B + heuristic + search vs large in-head, extending the depth-scale curve).

## 5. Benchmarks & measurement

- **TravelPlanner** (primary): hundreds of homogeneous options per category (flights/hotels/restaurants) under constraints/preferences; itinerary (path) selection; combinatorial explosion; LLMs notoriously weak (very low final-pass) — bottleneck is path-search/constraint-satisfaction, not provenance. Cleanest isolation of the axis.
- **AppWorld** (stateful, ~457 APIs): path + deep state.
- **NATURAL PLAN**: constraint-bounded combinatorial planning, light tooling.
- **Two-axis comparison:** apply the *same* learned TBox to **τ² (provenance, Paper 1)** vs **TravelPlanner (path-search)** to separate "abstraction helps path-selection (②③ transfer)" from "path is pure offload (LLM irrelevant to ④⑤)."

**Metrics** (deterministic): path quality (constraint-satisfaction, preference, official task-pass); search cost (nodes expanded, tokens, USD — does the heuristic *shrink* search?); transfer (training-bench heuristic → target ABox-swap, zero-retrain retention, reported per-bench, no aggregation); heuristic-contribution ablation (Δ at fixed budget); scale control (7B+heuristic+search vs large in-head).

## 6. Status & risks

- **[SETTLED]** Large-N in-head selection does not scale (depth-scale `comparative@50=0.02`, `rank@50≈0.30`, incl. 235B); deterministic engine trivial.
- **[SETTLED, Paper 1]** Conditional/planning load (L_branch) persists at 32B and resists capability-scaffold — the residual this paper claims.
- **[ESTIMATE]** Everything in §4–§5 (the heuristic transfer result, H_offload vs H_abstract verdict, TravelPlanner numbers) is to-be-run.
- **Risks (honest):** (i) if heuristics don't transfer (H_offload), the headline weakens to "path = offload"; (ii) TravelPlanner/AppWorld harnesses are heavier than a LoRA arm; (iii) **the core risk** — if the learned heuristic is domain-specific, ABox-swap fails and the contribution collapses; (iv) whether ②(recognition)/③(value) genuinely *learn and transfer* or are themselves decidable (= offload) is the open question the ablation must settle.

## 7. Expected contribution & relation to the portfolio

A clean separation of two tool-use failure axes (provenance vs path-search), and either (H_abstract) a transferable learned-heuristic result that makes search tractable across domains without retraining, or (H_offload) an honest boundary showing path-search is pure deterministic offload. **Relation:** Paper 1 provides the load framework and the L_branch hand-off; this paper supplies the *learned* lever for the irreducible path residual; Paper 4 folds the search-cost/heuristic trade-off into the system cost model. Orthogonal to Paper 2 (A2 generation).
