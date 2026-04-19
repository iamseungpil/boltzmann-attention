# Boltzmann Energy Framework v1 — Ontology-Shaped Energy Landscape

**Status**: Brainstorm-locked spec, no experiments yet.
**Date**: 2026-04-19
**Predecessor**: F13 K-rotation track retired (seed-unstable, +13.2pp same-recipe variance, skip-bug retraction).
**Replaces**: NEW_THEOREM_TEST §5 Phase F12-F14 ablation track for ICLR 2027.
**Scope**: Vision paper anchor. F8d ontology infra retained, F13 intervention discarded.

---

## §1. Why this framework

### 1.1 User vision (origin)

Use ontology in a physics + neuroscience way. Prior brainstorm threads (Group 6 reframe, H-Energy hypothesis, Hopfield-attention precedent, project name `boltzmann-attention`) all point to energy-based formulation as the natural anchor.

### 1.2 What F13 failed to deliver

F13 (K-rotation training in B_ont subspace) yielded:
- F13b +3.85pp F1 single seed → reduced to 3-seed mean ≈ baseline once F13k (seed=1) gave 0.671
- 13.2pp same-recipe variance across seeds
- Berry phase mechanism dead (skip_layer_28 was no-op for Qwen2.5-7B with num_layers=28)
- RCO mechanism falsified (G-RCO-3d non-monotone in rank curve)

F13 is retired. F8d ontology measurement infrastructure is preserved.

### 1.3 What survives from F-series

| Asset | Reuse |
|---|---|
| F8d 2-facet NMI (verb × domain, NMI ∈ [0.144, 0.218]) | Ontology cluster definition $c_f$ |
| B_ont basis construction (Phase B) | Direction projection in K-space |
| Phase B2 L27→L28 85× amplification | H-Basin-Hierarchy supporting evidence |
| MetaTool benchmark, Qwen2.5-7B environment | Same testbed |
| MRHD §4.Y idea (head-level localization) | Optional, energy-localization variant |

F13 K-rotation training: discarded.

---

## §2. Locked decisions

### 2.1 D-Facet (LOCKED)

**Use only verb × domain** (2 facets), not 4. Rationale:
- F8d shows cooccur and param_jaccard always redundant (NMI > 0.78 with verb/domain pair)
- Effective informative dimension = 2
- Cleaner H-V-* fitting (less risk of H-V-Sparse fire from spurious facet)
- 2-facet × SO(2) = R=4 pattern from F13 confirms 2 is the natural ontology rank

### 2.2 D-Form (LOCKED)

**Master form + 3 falsifiable H-V-* hypotheses**. Do not commit to a specific $V$ form upfront. Empirical estimator decides.

Master form:
$$
E^{\text{ont}}(t \mid q) = -\langle q, k_t \rangle + V(t, q)
$$

Three hypotheses tested simultaneously by regression on attention-derived energies:
- H-V-Additive (linear in facet mismatch)
- H-V-Hopfield (low-rank centroid product)
- H-V-Sparse (single facet dominant)

### 2.3 D-Cluster centroid (PENDING)

Centroid for H-V-Hopfield: K-space mean activation across tools in cluster $c$. Decision pending on whether to use prompt-end pooling (Phase C convention) or last attention layer pooling.

### 2.4 D-Layer (PENDING)

Layer at which to compute $E_i$. Default candidates:
- L=27 (last attention layer)
- L=18 (where ladapt schedule transitions, Phase B2 boundary)
- Multi-layer sweep (L ∈ {6, 12, 18, 24, 27})

---

## §3. Mathematical framework

### 3.1 Boltzmann form of attention

Standard scaled dot-product attention is a Boltzmann distribution:
$$
P(\text{tool}_i \mid q) = \frac{\exp(-\beta E_i)}{Z}, \quad E_i = -\langle q, k_i \rangle, \quad Z = \sum_j \exp(-\beta E_j)
$$

with $\beta = 1/\sqrt{d_{\text{head}}}$ and free energy $F = -\beta^{-1} \log Z$.

### 3.2 Master form

$$
E^{\text{ont}}(t \mid q) = -\langle q, k_t \rangle + V(t, q)
$$

$V(t, q)$ is the ontology potential. Functional form is uncommitted; inferred empirically.

### 3.3 Three falsifiable form hypotheses

Define facet mismatch indicators (verb × domain only):
$$
\delta_v(t, q) = \mathbb{1}[c_{\text{verb}}(t) \neq c_{\text{verb}}(q)], \quad \delta_d(t, q) = \mathbb{1}[c_{\text{domain}}(t) \neq c_{\text{domain}}(q)]
$$

#### H-V-Additive
$$
V_{\text{add}}(t, q) = a_v \delta_v(t, q) + a_d \delta_d(t, q)
$$
Two parameters $(a_v, a_d)$. Hopfield analog: independent stored axes.

#### H-V-Hopfield
$$
V_{\text{Hop}}(t, q) = -\lambda \sum_{f \in \{v, d\}} \sum_\mu \langle q, \xi^{f, \mu} \rangle \langle \xi^{f, \mu}, k_t \rangle
$$
where $\xi^{f, \mu}$ = K-space centroid of facet $f$, cluster $\mu$. Direct Modern Hopfield (Ramsauer 2020) form. Patterns = ontology cluster prototypes.

#### H-V-Sparse
$$
V_{\text{sp}}(t, q) = a \delta_f(t, q), \quad f \in \{v, d\}
$$
Single facet dominates. Fits one of $a_v$ or $a_d$ alone, the other zero.

### 3.4 Hopfield equivalence (Ramsauer 2020 anchor)

Modern Hopfield retrieval at high $\beta$:
$$
x_{\text{retrieved}} = X^\top \text{softmax}(\beta X q)
$$

Tool selection via attention IS this retrieval, with $X$ = stacked tool keys. Adding ontology potential extends $X$ with cluster-centroid axes:
$$
\tilde{X} = [K; \sqrt{\lambda} \Xi^v; \sqrt{\lambda} \Xi^d]
$$
where $\Xi^f \in \mathbb{R}^{|c_f| \times d_{\text{head}}}$ stacks all facet-$f$ centroids.

Capacity bound (Demircigil 2017): exponential in $d_{\text{head}}$, $C \sim e^{d_{\text{head}}/2}$. Ontology-structured patterns reduce effective capacity demand by basin merging within clusters.

### 3.5 Free energy decomposition

$F = E - TS$ where $T = 1/\beta$ and $S = -\sum_i p_i \log p_i$ is attention entropy.

$$
\frac{\partial F}{\partial T} = -S, \quad \frac{\partial \log Z}{\partial \beta} = -\langle E \rangle
$$

These thermodynamic identities provide H-Free-Energy-Decomposition test without any model intervention.

---

## §4. Five testable hypotheses (all inference-only)

### 4.1 H-Energy-Wells (FIRST EXPERIMENT, ~1 GPU-hr)

For each query $q$ with ground truth tool $t^*$:
- Compute $E_i = -\langle q, k_i \rangle$ for all tools $i$ in catalog at chosen layer $L$
- Compute facet distance $d(t^*, t_i) = \delta_v(t^*, t_i) + \delta_d(t^*, t_i) \in \{0, 1, 2\}$
- Aggregate $\bar{E}(d) = \mathbb{E}_q[E_i \mid d(t^*, t_i) = d]$ over N=100 queries

**Pre-reg G-Wells-1**: $\bar{E}(0) < \bar{E}(1) < \bar{E}(2)$ (strict monotone basin).

**Pre-reg G-Wells-2**: Spearman $\rho(d, E) \geq 0.4$ at single-query level (across 1000+ tool-query pairs).

**Pre-reg G-Wells-3**: Random-cluster control (shuffle facet labels) yields $\rho < 0.1$.

### 4.2 H-V-Form (regression, embedded in 4.1)

Fit three forms (Additive, Hopfield, Sparse) on the same $\Delta E$ data. Compare $R^2$ and AIC.

**Pre-reg outcome matrix**:

| Additive $R^2$ | Hopfield $R^2$ | Sparse $R^2$ | Decision |
|---|---|---|---|
| ≥ 0.5 | ≥ 0.5 + Δ ≥ 0.1 | < 0.7 × full | H-V-Hopfield wins (strongest) |
| ≥ 0.5 | < Additive | < 0.7 × full | H-V-Additive |
| ≥ 0.5 | < Additive | ≥ 0.7 × full Add | H-V-Sparse (single facet, F8d 2-facet claim weakened) |
| < 0.3 | < 0.3 | < 0.3 | $V$ small or non-fit → framework falsified, reconsider |

### 4.3 H-Storage-Capacity (~3 GPU-hr)

Subsample MetaTool catalog at sizes $N \in \{20, 50, 100, 200, 388\}$. Measure F1 at each.

**Pre-reg G-Cap-1**: F1 plateau until $N < N^*$, then drop. $N^*$ predicted by Demircigil bound.

**Pre-reg G-Cap-2**: Onset of capacity collapse correlates with appearance of spurious basins (entropy spike at non-GT tools).

### 4.4 H-Temperature-Modulation (~2 GPU-hr)

Sweep effective $\beta$ at last attention layer (logit scaling). For each $\beta$:
- Measure within-cluster confusion rate $r_{\text{in}}(\beta)$
- Measure across-cluster confusion rate $r_{\text{out}}(\beta)$

**Pre-reg G-Temp-1**: $r_{\text{in}}(\beta) / r_{\text{out}}(\beta) \to 1$ as $\beta \to 0$ (high temp mixes).

**Pre-reg G-Temp-2**: Crossover $\beta^*$ exists where ratio is maximally suppressed (sharpest cluster structure).

### 4.5 H-Basin-Hierarchy (~2 GPU-hr)

For each layer $L$:
- Compute attention basin $B_L^*(q) = \{i : E_i^{(L)} < E_{\min}^{(L)} + \tau\}$
- Compute basin specificity $s_L(q) = |B_L^*(q)|^{-1}$ (smaller basin = more specific)

**Pre-reg G-Basin-1**: $s_L$ monotone in $L$ (deep layers = specific tool, shallow = verb category).

**Pre-reg G-Basin-2**: Spearman $\rho$(layer depth, ontology hierarchy level) ≥ 0.5.

### 4.6 H-Free-Energy-Decomposition (~1 GPU-hr)

Compute $F$ and $S$ at varying $\beta$. Verify thermodynamic identity $\partial F / \partial T = -S$ numerically.

**Pre-reg G-FE-1**: Identity holds within 5% relative error.

**Pre-reg G-FE-2**: Ontology-rich queries (all facets specified) yield lower $S$ than ontology-poor queries (one facet specified).

---

## §5. First experiment protocol (H-Energy-Wells, locked)

### 5.1 Inputs

- Model: Qwen2.5-7B-Instruct
- Dataset: MetaTool Subtask4, N=100 queries (subset of N=147 used in F13)
- Catalog: 388 tools (full MetaTool plugin set)
- Layer: L=27 (last attention layer) — single layer, no sweep yet
- Cluster definition: F8d verb × domain partitions on 388 plugins (already computed)

### 5.2 Procedure

```
For q in queries[:100]:
    h_q = forward(model, prompt(q), capture_layer=27)['attn_out']
    q_vec = h_q[last_token_position]  # (d_head,)
    
    For each tool t in catalog (388 tools):
        h_t = forward(model, tool_description(t), capture_layer=27)['attn_out']
        k_t = h_t[mean_or_last_token]  # (d_head,)
        E[q][t] = -dot(q_vec, k_t)
    
    GT t* = label(q)
    For t in catalog:
        d_v = (cluster_verb(t) != cluster_verb(t*))
        d_d = (cluster_domain(t) != cluster_domain(t*))
        d_total = d_v + d_d
        record (E[q][t], d_v, d_d, d_total, q_id, t_id)
```

### 5.3 Aggregation

- Histogram: $E$ vs $d_{\text{total}}$ across all (q, t) pairs
- Per-query Spearman $\rho(d, E)$, then aggregate
- Random-control: shuffle (verb, domain) labels per tool, repeat measurement
- Three regressions: Additive, Hopfield, Sparse → $R^2$ + AIC comparison

### 5.4 Outputs

- `reports/boltzmann_energy/h_wells_qwen25_metatool_n100.json`:
  - per-query: $\rho$, regression coefficients, $R^2$ for each form
  - aggregated: $\bar{E}(d)$, gate pass/fail
  - random-control: same structure

### 5.5 Cost estimate

- Per-query forward pass: ~0.5 sec
- Catalog forward (cached): ~10 min for 388 tools
- 100 queries × 388 tools energy compute: trivial CPU
- Total: ~1 GPU-hr including overhead

### 5.6 Decision after H-Energy-Wells

| G-Wells-1 | H-V-Form | Next action |
|---|---|---|
| Pass | Hopfield wins | H-Storage-Capacity + paper §4.A draft |
| Pass | Additive wins | H-Storage-Capacity + paper §4.A (Additive variant) |
| Pass | Sparse wins | F8d 2-facet claim weakening, reconsider verb-only form |
| Pass | All weak | Framework partial — proceed to H-Basin-Hierarchy as next test |
| Fail | — | Framework FALSIFIED at first hurdle. Pivot: FEP / Gärdenfors / abandon ontology |

---

## §6. Preemption audit (REQUIRED before paper writing)

### 6.1 Foundation citations (mandatory)

- Hopfield 1982 (associative memory)
- Krotov-Hopfield 2016 (Dense Associative Memory)
- Demircigil 2017 (exponential capacity)
- Ramsauer 2020 (Hopfield = attention, NeurIPS)
- Hoover 2023 (Energy Transformer, NeurIPS)
- Park 2024 (Linear Representation Hypothesis)

### 6.2 Threat audit candidates (3-agent parallel, before first measurement)

| Prior | Search | Threat estimate |
|---|---|---|
| Hopfield-attention recent (2024-2026) | "Hopfield network attention" 2024-2026 | Mid — could subsume framework |
| Energy-based memory in LLMs | "energy-based attention LLM" 2023-2026 | Mid |
| Ontology in attention measurement | "ontology attention measurement" tool LLM | Low-Mid |
| Friston FEP applied to LLM tool use | "free energy principle tool selection" | Low |
| Conceptual spaces in transformers | "Gärdenfors conceptual spaces transformer" | Low |
| Capacity studies of LLM attention as Hopfield | "LLM attention capacity Hopfield" | Mid |
| Boltzmann LLM | "Boltzmann LLM attention" | Mid |

5-axis intersection target: {ontology × energy × attention × training-free × LLM tool selection}. If EMPTY (Wang 2025 audit pattern), framework is novel.

### 6.3 Audit blockers

- Cannot start writing §4.A until 6.2 returns
- Cannot lock contribution claims (C1-C5) until 6.2 returns
- Can run H-Energy-Wells in parallel with audit (independent)

---

## §7. ICLR 2027 paper structure (proposed)

### 7.1 Section map

| § | Content | Source |
|---|---|---|
| §1 Intro | Tool selection as memory retrieval, energy-based view | new |
| §2 Background | Hopfield, Modern Hopfield, attention=energy, conceptual spaces | citation work |
| §3 Framework | Boltzmann form, master $V$, 3 hypotheses, Hopfield equivalence | §3 of this doc |
| §4 Empirical | H-Energy-Wells + H-V-Form + H-Storage + H-Temp + H-Basin + H-FE | §4 of this doc |
| §5 Cross-model + cross-bench | Llama-3-8B replication, BFCL/ToolBench | future |
| §6 Discussion | Connections to FEP, conceptual spaces, neuroscience | new |
| §7 Limitations + Future | Framework boundary, MCTS extension, quantum cognition | new |
| §8 Related | Hopfield revival, energy LLMs, ontology measurement | post-audit |
| Appendix A | F8d ontology construction (from prior work) | salvage |
| Appendix B | Cross-layer basin maps | new |
| Appendix C | F-series retraction notes (F1-F13) | optional |

### 7.2 Five contribution claims (provisional)

- **C1**: Master form $E^{\text{ont}} = -\langle q, k \rangle + V(t,q)$ unifies attention-as-Boltzmann with ontology potential, with three falsifiable form hypotheses
- **C2**: Empirical evidence that LLM attention encodes ontology-shaped energy basins (verb × domain) without explicit ontology training
- **C3**: Capacity bound applied to tool selection — Hopfield Demircigil bound predicts catalog-size scaling
- **C4**: Cross-layer basin hierarchy aligns with ontology hierarchy (basin specificity ∝ layer depth)
- **C5**: Training-free, model-agnostic energy diagnostic that requires only attention scores

### 7.3 Score estimate

- Best case (all 5 H-* partial pass + cross-model + audit clean): 6.5-7.5 borderline accept
- Realistic case (3-4 H-* pass + Qwen-only + audit): 5.5-6.5 borderline
- Pessimistic case (H-Energy-Wells only): 4.5-5.5 reject

---

## §8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hopfield-attention preemption (recent unseen paper) | Mid | §6.2 audit before writing |
| H-Energy-Wells fail (ontology not in energy) | Low-Mid | Pivot path to FEP / abandon ontology layer |
| Last-layer attention not informative for tool selection | Mid | Multi-layer sweep as §5.6 fallback |
| F8d 2-facet too coarse (cooccur/param actually informative) | Low | H-V-Sparse outcome flags this |
| Cluster-centroid Hopfield fit numerically unstable | Mid | Regularization, low-rank approximation |
| Reviewer attack: "energy is just negative attention" | High | C2 sharp prediction defense, multi-layer hierarchy evidence |

---

## §9. Decision queue (next session)

1. D-Centroid: prompt-end vs last-layer pooling for $\xi^{f,\mu}$
2. D-Layer: L=27 single vs multi-layer sweep
3. D-Audit-Trigger: when to spawn 3-agent preemption audit
4. D-F13-Archive: arxiv tech report vs full discard
5. D-Cluster-Source: F8d 2-facet labels (already computed) vs re-derive on 388-tool subset

Recommended next decision: D-Centroid + D-Layer (these are both inputs to §5 protocol).

---

## §10. Hard constraints (inherited from prior memory)

- Brainstorm session: no GPU execution permitted
- Destructive git commands: explicit user approval required
- Reviewer-magnet wording (B1 §5.3): avoid "Berry phase", prefer "energy basin", "Hopfield retrieval"
- ICLR 2027 single track (NeurIPS 2026 withdrawn)
- F13 K-rotation training: archived, not extended
- Pre-registration discipline: gate definitions BEFORE measurement

---

## §11. One-line summary

> Boltzmann Energy Framework v1: ontology-shaped energy landscape for tool-selection attention, formulated as $E^{\text{ont}} = -\langle q, k_t \rangle + V(t,q)$ with master form + 3 H-V-* hypotheses, anchored on F8d 2-facet (verb × domain), tested via 5 inference-only experiments (H-Energy-Wells first, ~1 GPU-hr), grounded in Hopfield-Ramsauer-Demircigil-Hoover lineage, replacing the retired F13 K-rotation track for ICLR 2027.
