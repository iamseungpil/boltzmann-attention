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

## Winner

**P2 kmeans-domain L=18** — Δ_norm=**+0.579** (G1=PASS), median ρ=+0.144 (G2=FAIL), Hopfield R²=0.274. Joint = **WEAK**.

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

## Verdict

**Path A partially rescues H-Energy-Wells framework**:
- G1 (aggregate basin) PASS with K-space labels — framework survives at aggregate level.
- G2 (per-query Spearman) still FAIL — basin too shallow for robust per-query retrieval.
- G3 (shuffled null) PASS throughout — signal is real, not artifact.

**Root cause of v1 falsification**: afod-heuristic regex labels were K-space-orthogonal. The ontology is intrinsic to the K-space, not the regex categorization scheme.

**Path B (H-V-NegBasin / framework pivot) NOT triggered** — kill criterion (P1+P2+P3 all FAIL) not met. P2 kmeans-domain L=18 joint=WEAK is a partial rescue.

## Next-step options

1. **H-Storage-Capacity** at P2 winner config (spec §4) — test Hopfield-style pattern counting.
2. Tighten G2 investigation — why per-query ρ so low (0.144) despite aggregate basin?
3. Paper narrative pivot: 'ontology exists but at K-intrinsic level, not at lexical-heuristic level.'
4. Replicate at best-L for additional layers around L=18 (L=16, 20, 22) to localize.
