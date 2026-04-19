# H-Wells Rescue Summary (Path A, P1-P4)

**Executed**: 2026-04-19 Qwen2.5-7B × MetaTool Subtask1 N=100, seed=0

**Pre-reg gates** (LOCKED, unchanged across variants):
- G1: strict Ē(0)<Ē(1)<Ē(2) AND Δ_norm ≥ 0.5
- G2: median per-query Spearman ρ ≥ 0.4
- G3: shuffled-label median ρ < 0.1 (null clean)

## Results table

| variant | L | pool | kspace | chat | Δ_norm | G1 | ρ | G2 | ρ_shuf | G3 | Hop R² | joint |
|---|---|---|---|---|---:|:---:|---:|:---:|---:|:---:|---:|---|
| v1 (baseline) | 27 | prompt_end_k_norm_avg_over_n_kv | none | — | -0.192 | ✗ | -0.070 | ✗ | -0.026 | ✓ | -0.000 | FALSIFIED |
| P1 L=6 | 6 | prompt_end | none | — | +0.159 | ✗ | +0.021 | ✗ | -0.008 | ✓ | 0.409 | FALSIFIED |
| P1 L=12 | 12 | prompt_end | none | — | +0.231 | ✗ | +0.025 | ✗ | +0.011 | ✓ | 0.001 | FALSIFIED |
| P1 L=18 | 18 | prompt_end | none | — | +0.139 | ✗ | +0.055 | ✗ | -0.009 | ✓ | 0.304 | FALSIFIED |
| P1 L=24 | 24 | prompt_end | none | — | -0.025 | ✗ | +0.005 | ✗ | +0.018 | ✓ | 0.246 | FALSIFIED |
| P1 L=27 | 27 | prompt_end | none | — | -0.192 | ✗ | -0.070 | ✗ | -0.026 | ✓ | -0.000 | FALSIFIED |
| P2 kmeans-verb L=18 | 18 | prompt_end | verb | — | +0.440 | ✗ | +0.058 | ✗ | -0.008 | ✓ | 0.229 | FALSIFIED |
| P2 kmeans-domain L=18 | 18 | prompt_end | domain | — | +0.579 | ✓ | +0.144 | ✗ | +0.004 | ✓ | 0.274 | WEAK |
| P2 kmeans-both L=18 | 18 | prompt_end | both | — | +0.569 | ✓ | +0.164 | ✗ | +0.009 | ✓ | 0.184 | WEAK |
| P2 kmeans-verb L=12 | 12 | prompt_end | verb | — | +0.294 | ✗ | +0.042 | ✗ | -0.009 | ✓ | 0.000 | FALSIFIED |
| P2 kmeans-domain L=12 | 12 | prompt_end | domain | — | +0.261 | ✗ | +0.102 | ✗ | +0.016 | ✓ | 0.001 | FALSIFIED |
| P3 mean_all afod L=18 | 18 | mean_all | none | — | -0.394 | ✗ | -0.015 | ✗ | +0.009 | ✓ | 0.193 | FALSIFIED |
| P3 first_name afod L=18 | 18 | first_name | none | — | -0.025 | ✗ | -0.030 | ✗ | +0.010 | ✓ | 0.000 | FALSIFIED |
| P3 mean_all kmd L=18 | 18 | mean_all | domain | — | -0.271 | ✗ | -0.088 | ✗ | -0.000 | ✓ | 0.201 | FALSIFIED |
| P3 first_name kmd L=18 | 18 | first_name | domain | — | -0.090 | ✗ | -0.026 | ✗ | +0.010 | ✓ | 0.000 | FALSIFIED |
| P4 chat afod L=18 | 18 | prompt_end | none | ✓ | +0.088 | ✗ | +0.038 | ✗ | -0.006 | ✓ | 0.283 | FALSIFIED |
| P4 chat kmd L=18 | 18 | prompt_end | domain | ✓ | +0.449 | ✗ | +0.121 | ✗ | +0.001 | ✓ | 0.254 | FALSIFIED |
| D.3 BERT-KM domain L=18 | 18 | prompt_end | none | — | +0.143 | ✗ | +0.090 | ✗ | +0.006 | ✓ | 0.299 | FALSIFIED |
| D.3 BERT-KM verb L=18 | 18 | prompt_end | none | — | +0.128 | ✗ | -0.025 | ✗ | -0.015 | ✓ | 0.307 | FALSIFIED |
| D.3 BERT-KM both L=18 | 18 | prompt_end | none | — | +0.048 | ✗ | -0.008 | ✗ | +0.009 | ✓ | 0.302 | FALSIFIED |

## Winner (highest Δ_norm — but see D.3 tautology check below)

**P2 kmeans-domain L=18** — Δ_norm=**+0.579** (G1=PASS), median ρ=+0.144 (G2=FAIL), Hopfield R²=0.274. Joint = **WEAK**.

## D.3 BERT-KM defense — TAUTOLOGY CHECK

Pre-reg tier (locked before D.3 run):
- Δ ≥ 0.30: cross-feature semantic basin (paper tier 5.0-6.0)
- 0.15 ≤ Δ < 0.30: ambiguous (paper tier 4.0-5.0)
- Δ < 0.15: tautology confirmed, pure negative (paper tier 3.5-4.5)

BERT-KM Δ_norm at L=18 prompt_end (best of {verb, domain, both}):
- domain: +0.143 (best, BARELY below 0.15 boundary)
- verb: +0.128 (non-monotone)
- both: +0.048

**TAUTOLOGY CONFIRMED.** afod-domain (+0.139) ≈ BERT-KM-domain (+0.143) << Qwen-K-self-KMeans-domain (+0.579, 4× larger). Two independent semantic spaces (lexical regex, BERT embedding) BOTH produce identical near-zero basin in Qwen K-space; only K-self-derived labels lift the signal. The P2 'rescue' was self-similarity by construction, not semantic structure.

## Per-phase conclusions

### P1 Layer sweep
- Best-L = 12 (Δ_norm = +0.231), but all 5 layers FAIL G1.
- L=6/12/18 weak correct direction; L=24/27 reverse.
- Hop R² spikes at L=6 (0.409) and L=18 (0.304) — cluster structure exists, not monotone.

### P2 K-space KMeans (the rescue)
- **FIRST G1 PASS** across entire rescue: kmeans-domain L=18 (Δ_norm=+0.579, joint=WEAK).
- kmeans-both L=18 also PASS (Δ_norm=+0.569). kmeans-verb L=18 FAIL but +0.440.
- L=12 K-space clustering FAILS — best-K-space layer is L=18, not L=12.
- **afod-heuristic label hypothesis CONFIRMED**: swapping verb/domain labels from regex-extracted afod to KMeans-on-K unlocks the basin.

### P3 Pooling sweep
- mean_all and first_name BOTH fail across both label sets — actually REVERSE Δ_norm direction in most cells.
- **prompt_end pooling is load-bearing.**

### P4 Chat template
- afod L=18 chat: Δ_norm=+0.088 (worse than raw +0.139).
- kmd L=18 chat: Δ_norm=+0.449 (worse than raw +0.579).
- **Chat template HURTS**, not helps. v1's raw-query design was correct.

## Verdict (post-D.3)

**Path A FAILS to rescue H-Energy-Wells framework once D.3 tautology check is applied:**
- P2 kmeans-domain L=18 G1 PASS = self-similarity artifact (Δ=+0.579 with K-self labels collapses to Δ=+0.143 with BERT-independent labels of same K).
- afod (+0.139) ≈ BERT (+0.143) — two independent semantic spaces both fail. Real semantic basin in Qwen attention K-space at L=18 ≈ 0.14σ ≈ noise.
- G2 (per-query Spearman) FAIL throughout — even the artifact-inflated P2 winner only reaches +0.144.
- G3 (shuffled null) PASS throughout — what little signal exists IS structured, just very weak.

**Triple negative**: afod fail + BERT fail + G2 fail. **Tautology confirmed via D.3.**

**Kill criterion technically NOT triggered** (G1 PASS exists, even if artifact), but D.3 reveals it as construction artifact. Effective interpretation: H-Energy-Wells v1 framework is **dead at semantic-basin level**, regardless of methodology variant.

## Next-step options

1. **Paper Option B (pure negative)** — frame as falsification with mechanistic ablations: 'Tool-selection ontology does NOT exist as Hopfield basin in Qwen2.5 attention K-space; reported aggregates under self-derived labels are tautological. afod, BERT, and per-query retrieval all fail.'
2. **Path B brainstorm RE-ACTIVATES**: H-V-NegBasin (anti-basin), FEP (Friston), Gärdenfors conceptual spaces, or abandon ontology axis (attention-only measurements).
3. **Cross-layer + cross-model replication** before paper writing: confirm BERT-KM Δ ≈ afod Δ at L∈{6,12,24} too (if so, framework dead globally) and on Qwen2.5-1.5B / Llama-3-8B.
4. **G2-first redesign**: rebuild framework around per-query metric (e.g., rank of GT in nearest-N) rather than aggregate basin.
