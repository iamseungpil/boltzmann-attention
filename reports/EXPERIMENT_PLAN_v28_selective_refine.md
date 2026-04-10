# Experiment Plan v28: Selective Refinement Wave

## Decision Summary

This wave is narrower than the earlier brainstorms.

- **Proceed now**: cache-aware two-pass selective refinement
- **Proceed later**: query-aware dequantization, but only after a stronger real-trace diagnostic
- **Proceed later**: cross-head agreement, only if the simpler selector saturates
- **Control only**: temperature sharpening
- **Drop**: attention bias
- **Drop for now**: tiny recursive innovation cache

The key correction is simple:

- The synthetic `risk` story survived only in the `budget_pages=1` toy regime.
- On real traces, the current flip-risk metric lost to raw score:
  - Mistral 2-bit: `0.7512 < 0.8017` at Recall@32
  - Mistral 3-bit: `-0.0031` delta
  - Qwen 2-bit: `-0.0116` delta

That means the original QDRP claim is not ready for a real E8 wave.

## Harsh Review

### C. Two-pass reranking / selective refinement

Intent:
- Recover retrieval by refining only a small subset of cache positions selected from cheap proxy scores.

Why it survives:
- It directly targets the failure mode that matters for NIAH: a wrong top-ranked region under low-bit K.
- It has the cleanest ablation story against full 2-bit and full 3-bit.

Fatal objection:
- The first prototype was not cache-correct. It ignored the real `past_key_value` path, so earlier results were not authoritative.

Correction:
- Move selection after cache update and rank over the full KV length, not the local query length.

Verification:
- NIAH on Mistral/Qwen with:
  - `fp16`
  - `uniform_2bit`
  - `uniform_3bit`
  - selective promotion at matched average budget

Kill criterion:
- If cache-aware selective promotion does not beat `uniform_2bit` on NIAH at the same effective budget, stop.
- If it only matches `uniform_3bit`, the method is not worth the complexity.

Decision:
- **Proceed now**

### D. Query-aware dequantization

Intent:
- Use the current query to shift the dequantized representative within each quantization bin.

Why it might matter:
- It changes the decode-time value without extra stored metadata.
- That is more interesting than another selector heuristic.

Fatal objection:
- The current risk proxy is not validated on real traces.
- If the query-conditioned signal cannot beat raw score on real flip calibration, this becomes a story without empirical footing.

Corrected form:
- Do **not** push a full risk-paging claim.
- Restrict this to a local decode-time correction hypothesis:
  - does query-aware bin-centering reduce score error for already selected keys?

Verification:
- Real-trace score MSE / top-k retention before any NIAH claim.

Kill criterion:
- If score correlation or top-k recovery does not improve over plain dequantization, drop.

Decision:
- **Proceed later**

### E. Cross-head agreement

Intent:
- Use agreement across grouped query heads to stabilize refinement decisions.

Why it might matter:
- GQA models already expose a natural grouping structure.

Fatal objection:
- This is a second-order idea stacked on top of an unproven first-order selector.
- If single-head or max-pooled score selection is not working, cross-head agreement is just extra complexity.

Verification:
- Add only after the base selective refinement path is clearly positive.

Kill criterion:
- If it gives only noise-level movement over the base selector, stop immediately.

Decision:
- **Proceed later**

### A. Temperature sharpening

Intent:
- Counteract quantization-induced attention flattening by sharpening logits.

Why it stays alive as a control:
- It is cheap and can explain whether the failure is mostly distributional flattening or actual ranking damage.

Fatal objection:
- As a paper direction, this is weak.
- Reviewers will call it a calibration trick, not a method.

Verification:
- One small ablation alongside the main selective refinement run.

Kill criterion:
- If it does not improve NIAH quickly, remove it from the wave.

Decision:
- **Control only**

### B. Attention bias

Fatal objection:
- No strong theory, no strong novelty, and no prior evidence of gain here.

Decision:
- **Drop**

### TRIC / tiny recursive innovation cache

Fatal objection:
- The tiny recursive predictor still loses to a shared linear predictor in synthetic diagnostics.
- There is no reason to spend E8 time on it right now.

Decision:
- **Drop for now**

## Final Wave

### Wave 1

Method:
- cache-aware two-pass selective refinement

Intent:
- show that a small amount of targeted promotion recovers retrieval more efficiently than blunt uniform precision

Hypothesis:
- a cache-correct score-guided selector beats `uniform_2bit` at the same effective budget on NIAH

Verification:
- Mistral + Qwen, 2k/4k context, small repeat first, then bounded full run

Kill criterion:
- no improvement over `uniform_2bit`, or only trivial movement below run noise

### Wave 1 Control

Method:
- temperature sharpening

Intent:
- test whether flattening alone explains the retrieval loss

Hypothesis:
- a small sharpening factor may recover part of the NIAH drop, but less than selective refinement

Verification:
- one narrow ablation at the same context and repeat grid

Kill criterion:
- zero or negative gain

### Deferred Diagnostic

Method:
- query-aware dequantization

Intent:
- improve decode-time key reconstruction without extra storage

Hypothesis:
- local query-conditioned reconstruction should reduce score error before any end-to-end retrieval claim

Verification:
- real-trace score MSE and top-k retention

Kill criterion:
- if raw score remains better than the query-aware proxy on real traces, stop
