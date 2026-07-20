# [DRAFT v0 · 2026-07-19] Degenerate-Cue Interference: An Associative-Memory Account of Same-Rule Failures in Batched LLM Judgments

> Working draft (English, markdown → LaTeX later). Placeholders marked ⟦TBD⟧. Companion to Track A (systems paper);
> behavioral evidence is summarized here and detailed there. Evidence: RATE_SUBAGENT_DESIGN_2026_07_18.md §2l–2m ·
> sim_results/{size_k_sweep, b1_attn_curve, b2_knockout}_20260719 · Track B checklist doc.
> Frame discipline: generation-probe results (k*=2) and logit-readout results (k*=1, target-first primed) are
> reported as separate frames throughout; we never mix thresholds across frames.

## Abstract

When an LLM judges a batch of items against a policy document, items that engage the *same rule clause by
inference* interfere with one another: the model's judgment of a later item collapses to a default once k
similar predecessors precede it, while lexically-anchored items are immune. We give this phenomenon an
associative-memory account. Reading softmax attention as retrieval in a modern Hopfield energy landscape
(Ramsauer et al., 2020), k predecessors that inference-bind clause C constitute quasi-degenerate stored
patterns (multiplicity g ≈ k): the free energy of the clause-retrieval state is eroded by an entropy term,
ΔF = ΔE − T·log g, so retrieval fails at a sharp threshold k* ≈ e^{βΔE−θ} that is large for high-margin
(lexically anchored) items and small for low-margin (inference-bound) ones. This one expression organizes our
behavioral findings — causal asymmetry (only predecessors interfere), similarity gating (dissimilar items leave
g unchanged), item-conditionality (explicit anchors raise ΔE), sharp thresholds with instability at k*, and the
log-linear interference decay reported in prior PI work. We test the account at three levels on an open-weights
model that reproduces the behavior (Qwen2.5-3B): behaviorally (staircase reproduction across 1.5B–32B),
attentionally (clause-attention mass a_C(k) under similar vs dissimilar loading ⟦TBD B1⟧), and causally:
a position-preserving attention-mask knockout of the interfering rows **fully restores** the collapsed judgment
(P(5): 0.10 → 0.99 at k=2, and likewise at k=4, 8), while a size-matched control mask restores nothing and a
readout-only blockade shows the corruption flows through the target item's own representation
(⟦TBD B3: attention-temperature manipulation, which should shift k* exponentially⟧). We
position the result not as a new attention pathology but as the practical manifestation, and unit-level
identification, of a known property of associative memories: correlated patterns corrupt retrieval.

## 1. Introduction

⟦Condensed motivation: the Track A failure — a live agent pipeline judging credit-card transactions against
policy documents assigns a default rate to a specific item exactly when ≥k* same-clause items precede it.
Cross-reference Track A for the full systems story; this paper asks *why*.⟧

The phenomenon to be explained, established behaviorally (all temp-0, [S]-grade trajectories):

- **P1 — causal asymmetry**: only *preceding* similar items interfere; the damage is done in prefill (an
  input/output dissociation holds: emitted-first-but-interior fails, interior-emitted-last passes). ✅
- **P2 — sharp threshold**: k*=2 in the generation frame; 100% failure beyond threshold across 45 varied
  constructions; output *instability* precisely at k* (unit slips: 5 → 500, 5 → 100 in 6/18 threshold cells,
  never past threshold). ✅
- **P5 — similarity gating**: predecessors that engage the same clause *by inference* (no lexical match)
  interfere; same-category items resolving lexically (exclusion-list members, named partners) do not; items of
  other categories do not. Double dissociation kills quota, judgment-budget, and category-cue accounts. ✅
- **Default retreat**: the failure is not garbage — the model returns the *base-rate* answer, i.e., the
  highest-prior alternative, as if the clause simply had not been retrieved.
- **Scale does not immunize**: k=0 baseline holds and collapses under load at every size probed
  (1.5B/3B/14B/32B, logit-readout frame). ✅ (B0; also licenses 3B as the mechanism testbed.)

We propose that all of these are the signature of retrieval failure under *cue overload* in an associative
memory, and test the proposal's untested predictions (P3 attention curves, P6 knockout recovery, P4 temperature)
directly.

## 2. Theory

### 2.1 Attention as associative retrieval

Modern Hopfield networks with exponential energy (Ramsauer et al., 2020) are formally the softmax-attention
update: stored patterns = keys, retrieval cue = query, retrieved content = value mixture. Their capacity theory
distinguishes well-separated patterns (exponential capacity, clean single-pattern retrieval) from **correlated
patterns**, which produce metastable *mixture* states: retrieval returns a blend, or falls into a stronger
nearby attractor.

### 2.2 Degenerate specialization: same-clause rows as correlated patterns

Consider the judgment position for target item t: its query q_t must retrieve clause C's content (the elevated
rate) from the document region. Each predecessor row that *inference-binds* C — a row whose processing required
applying C without a lexical bridge — has, in prefill, (i) absorbed clause content into its residual stream
(causal attention: row tokens attend to C), and (ii) therefore acquired keys k_i = W_K h_i with a component
along C's key direction. The store now contains g ≈ k+1 patterns correlated in the C-direction: the clause
itself and k row-echoes of it.

Softmax normalization splits the retrieval mass among them. Where does the failing retrieval happen? Our causal
data (§5) localize it: not at the final readout (whose access to the interfering rows is causally inert) but at
the **construction of the target row's own representation** — the target row's token queries are the cue q_t,
retrieving "how does a row like me relate to clause C" from a store now crowded with k same-clause row-echoes.
Writing ΔE for the energy (logit) margin between the C-pattern and the row-echoes under cue q_t, the odds of
clean clause retrieval degrade as

  log-odds(retrieve C) ≈ β·ΔE − log g,

an entropy erosion of the free-energy gap: ΔF = ΔE − T·log g. Retrieval fails when log g exceeds βΔE − θ,
i.e., at a sharp threshold **k* ≈ e^{βΔE−θ}** — a staircase in k, not a drift.

### 2.3 What "inference-bound" means: the margin ΔE

Lexically-anchored items retrieve via near-duplicate token matching (induction-like high-margin channels):
ΔE large ⇒ k* astronomically large ⇒ immune. Judgment-dependent items have only a semantic-association margin:
ΔE small ⇒ k* small (measured: 2 in generation frame). The dissociations of P5 are then *forced*: arm-B/C items
(lexical resolution) neither gain C-direction keys strongly (they resolve elsewhere) nor matter if they do
(target's own margin unchanged, their patterns separated); only arm-A items densify the C-cluster.

### 2.4 Derived predictions and their status

| # | prediction | status |
|---|---|---|
| P1 | causal asymmetry (prefill-side; successors harmless) | ✅ behavioral |
| P2 | staircase in k; log-linear decay in aggregate (connects to UF's law) | ✅ behavioral |
| P5 | similarity gating; item-conditionality (anchors ⇒ immunity) | ✅ behavioral |
| — | default retreat = fallback to strongest surviving attractor (the base-rate prior) | ✅ behavioral (interpretive) |
| — | instability at k* = near-tie mixture state (output-level: unit slips; logit-level: mass scatter at 14B k=1) | ✅ behavioral |
| P3 | readout clause-mass a_C(k) decreases under similar loading | ✗ disconfirmed at readout (honest negative, §4) — refined to P3′ at the construction stage ⟦TBD B1b⟧ |
| P6 | blocking judgment-position attention to interfering rows (positions preserved) restores P(5) | ✅ **full recovery** (0.98–0.99 at k=2/4/8; controls fail; §5) |
| P4 | attention-temperature manipulation shifts k* exponentially (k* ~ e^{βΔE−θ}) | ⟦TBD B3⟧ |
| P7 (opt) | interfering-row keys grow a W_K-projected C-direction component with k | ⟦TBD B4 optional⟧ |

## 3. Behavioral evidence (summary; details in companion paper)

⟦Condense from Track A: exclusion chain table, dissociation table, k-staircase, A① robustness (45/45 fail past
threshold, 6/18 slips at threshold), size sweep table with frame caveat (readout k*=1 vs generation k*=2 —
output-ordering freedom is itself protective; consistent with §2: priming the readout removes an alternative
retrieval route, lowering the effective θ).⟧

B0 (testbed licensing): the full-fidelity logit-readout staircase reproduces on Qwen2.5-3B (k=0 P(5)=0.983 →
collapse at k≥1), so mechanism experiments run on open weights at tractable cost; 3B behavior matches the 32B
production model's failure pattern. First attempt honesty note: an earlier P3 attempt on 14B with an abbreviated
prompt failed its k=0 precondition and was voided; all mechanism experiments below use the full-fidelity prompt
verbatim.

## 4. Attention-level test (B1): the readout is the wrong place to look — an honest negative

Design: KV-cache 2-pass — prefill the prompt, then compute the final judgment-position (readout) query's
attention over all positions, per layer/head. Measure a_C(k) = mass on clause-C tokens, echo mass on
interfering-row tokens, and target-row mass, under similar vs dissimilar loading (k ∈ {0,1,2,4,8} / {1,2,4,8}).
Original prediction (P3): a_C falls with k on the similar line only.

Results (layer/head means; behavior columns confirm the frame reproduces the dissociation):

| cond | k | P(5) | a_C | a_iv | a_tgt |
|---|---|---|---|---|---|
| similar | 0 | 0.983 | 0.00009 | — | 0.00301 |
| similar | 1/2/4/8 | 0.22/0.08/0.11/0.04 | 0.00005–0.00010 | 0.0018→0.0032 | 0.0004–0.0007 |
| dissimilar | 1/2/4/8 | 0.33/0.50/0.86/0.90 | 0.00007–0.00021 | 0.0022→0.0028 | 0.0005–0.0010 |

**P3 in its original form is disconfirmed**: every readout-attention aggregate moves the same way under similar
and dissimilar loading (a_C flat and tiny; a_iv rises in both; a_tgt collapses ~5× in both), while behavior
separates cleanly (similar: 0.04–0.11 at k≥2; dissimilar: 0.50–0.90). The readout's attention *distribution*
does not carry the effect — consistent with §5's causal finding that blocking the readout's (and all post-row)
access to the interfering rows restores nothing. What the readout consumes is the target row's already-built
representation; the interference must act where that representation is built. (Per-layer curves are archived;
a layer-specific readout effect remains possible but cannot be the primary channel given §5.) Two frame notes,
recorded honestly: at k=1 both conditions dip (0.22/0.33) — the first predecessor of either type disturbs this
primed frame, and the similarity-specific separation emerges at k≥2; dissimilar P(5) recovers *toward* baseline
with k (0.33→0.90), a release-from-PI-like pattern in-frame.

### 4b. Construction-stage attention (B1b) ⟦TBD — running⟧

The refined prediction P3′: the *target row's own token queries*, during prefill, lose clause mass / gain
interfering-row (echo) mass specifically under similar loading. Design: 2-pass with the split at the target-row
start; pass-2 emits the target-row queries' attention over all prior positions (sdpa prefill, eager measurement
pass). ⟦results⟧

## 5. Causal test (B2): knockout with positions preserved — full recovery

Design: 4D attention-mask knockout — sequence length and position ids unchanged (dissociating content
interference from position effects). Arms: (base) pure-causal mask sanity gate; (ko_full) queries from the
target row onward cannot attend to interfering-row tokens; (ko_last) only the final readout query blocked;
(ctrl) an equal number of unrelated document tokens (clause and target excluded) blocked for the same queries.

Results (P(5), similar loading; k=0 reference 0.983; base arm reproduces the sweep values exactly, validating
the custom-mask path):

| k | base | ko_full | ko_last | ctrl |
|---|---|---|---|---|
| 2 | 0.100 | **0.989** | 0.010 | 0.078 |
| 4 | 0.120 | **0.980** | 0.026 | 0.062 |
| 8 | 0.046 | **0.992** | 0.013 | 0.053 |

Three conclusions. (1) **Causation established**: content-specific, position-preserving blockade of attention
to the interfering rows *fully* restores the judgment — to at or above the k=0 baseline — at every k tested;
the size-matched control restores nothing. The interference is carried by attention to the interfering rows'
tokens, not by their mere presence (positions), length, or mask-size artifacts. (2) **The channel is not the
readout alone**: blocking only the final judgment query does not recover (it even degrades below base, which we
record without over-interpreting) — the corruption travels through the *construction of the target row's own
representation* (and/or subsequent positions), which the readout then consumes. This matches the B1 observation
that the readout's largest attention target is the target row itself, not the clause. (3) The recovery being
*complete* argues the entire behavioral effect is attention-mediated — no residual value-mixture path is needed.
**Fine arms — localization decisive**: blocking *only the target-row queries'* access to the interfering rows
(ko_tgtrow) recovers almost fully (P(5) = 0.959 at k=2, 0.968 at k=4), while blocking only the post-row
queries — schema region and readout included — recovers nothing (ko_post: 0.020, 0.031). The interference
channel is therefore **the construction of the target row's representation**: its tokens' queries read the
same-clause predecessors during prefill, and what they absorb there determines the judgment; everything
downstream, including the readout's direct access to the interfering rows, is causally inert. ⟦optional:
layer-wise knockout⟧

## 6. Temperature test (B3): moving k* ⟦TBD⟧

k* ≈ e^{βΔE−θ} predicts an exponential shift of the threshold under attention-temperature scaling (β = 1/T
applied to attention logits, not output sampling). Implementation: scale attention scores by hooking the
attention modules on 3B; sweep T around 1 and locate k*(T) in the logit-readout frame. ⟦results⟧

## 7. Related work (mechanism axis)

Ramsauer et al. 2020 (modern Hopfield = attention; correlated patterns ⇒ metastable mixtures); Bietti et al.
2023 (transformer as associative memory); induction heads / duplicate-token heads (the high-margin lexical
channel of §2.3); retrieval heads; attention sinks / StreamingLLM (attention-mass bookkeeping); Found-in-the-
Middle (positional attention calibration — a *position*-side correction, complementary to our *content*-side
effect); Unable to Forget (log-linear PI — we read their law as the aggregate signature of the −log g term);
Remember First Forget Last (primacy protection); cognitive-science lineage: fan effect (Anderson), cue-overload
and release-from-PI (Watkins) — we import the paradigm, and the "cue overload" name, explicitly.

## 8. Discussion

- **Not a new pathology.** The claim is identification, not novelty of mechanism: correlated-pattern
  interference is a *theorem-level* property of exponential associative memories; we show it is the operative
  cause of a real agent failure, at clause granularity, and derive its practical geometry (edges protected,
  k* threshold, anchor immunity).
- **Unification.** One free-energy expression with a degeneracy entropy term covers: LiM-style positional
  effects (distance term), PI log-linear decay (−log g), primacy protection (no predecessors ⇒ g=1), and our
  clause gating (only C-correlated patterns count in g). ⟦keep modest: unification *sketch*, formal treatment
  = future work with the HEAT/Boltzmann-attention framework⟧
- **Design consequences.** If interference is degeneracy in the retrieval store, mitigations divide into
  (i) reducing g (batching caps — the Track A fix; deduplication), (ii) raising ΔE (giving items lexical
  anchors, e.g., quoting the clause per item), (iii) raising β (⟦if B3 confirms⟧). Prompt exhortations change
  none of these quantities — explaining their measured impotence.

## 9. Limitations

- Mechanism experiments on one model family (Qwen2.5) and one clause structure; 32B tested behaviorally but
  attention measured on 3B ⟦+14B spot-check if time⟧.
- The logit-readout frame differs from live generation (k* offset by 1); all mechanism claims are within-frame.
- a_C(k) correlation (B1) alone cannot establish causation — that burden is on B2; if B2's restoration is
  partial, the account is partially attention-mediated ⟦quantify⟧.
- Layer/head localization is exploratory (no pre-registered layer hypothesis).

## References ⟦TBD⟧

Ramsauer+ 2020 · Bietti+ 2023 · Olsson+ (induction heads) · Wu+ (retrieval heads) · Xiao+ (StreamingLLM) ·
Found-in-the-Middle · Unable to Forget (2506.08184) · RFFL (2603.00270) · Liu+ 2023 · Anderson 1974 ·
Watkins & Watkins 1975 · Guo & Vosoughi 2025 · Cheng+ 2023 · τ²-bench.

---
### Figure plan
F1 theory overview (degenerate free-energy landscape) / F2 behavioral triptych (exclusion·dissociation·staircase)
/ F3 a_C(k) similar vs dissimilar ⟦B1⟧ / F4 knockout recovery bars ⟦B2⟧ / F5 k*(size) with frame caveat /
F6 k*(temperature) ⟦B3⟧ / F7 (opt) layer localization.
### Provenance
theory §2 = §2m 수학모델 초안 · behavior = Track A draft §4–5 · B0/size = size_k_sweep_20260719 ·
B1 = b1_attn_curve_20260719 ⟦running⟧ · B2 = b2_knockout_20260719 ⟦running⟧ · B3 = ⟦not launched⟧.
