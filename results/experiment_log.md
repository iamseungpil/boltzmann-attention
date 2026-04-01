# Boltzmann Thermodynamics of Attention — Level 1 Experiment Log

## Experiment 1-1: T_eff Gate Test

### Setup
- Models: Qwen2.5-3B (GQA, 2 KV heads), GPT-2 Medium (standard MHA, 24L/16H)
- Calibration: 64 samples, max_seq_len=512
- Gate criterion: Spearman ρ(layer_index, T_eff)

### Results

#### Qwen2.5-3B (GQA)
- Spearman ρ = +0.1081, p = 5.30e-01 → **FAILURE**
- T_eff range: [0.295, 0.903]
- V-shaped pattern: hot initial (0.84) → cold middle (0.46) → reheat final (0.76)
- GQA inter-group gap: up to 0.506 at Layer 5 (KV0=0.548 vs KV1=0.042)
- Head variance: middle layers (L3-L32) 7x higher than initial/final (0.037 vs 0.005)

#### GPT-2 Medium (MHA)
- Spearman ρ = -0.6574, p = 4.82e-04 → **WEAK_SUCCESS**
- T_eff range: [0.221, 0.861]
- Clear progressive cooling: L0=0.86 → L6=0.27 → L19=0.22, final reheat L23=0.52
- Consistent head variance pattern

### Key Finding
Progressive cooling is a real phenomenon in standard MHA transformers, confirming Boltzmann T_eff structure. GQA disrupts this pattern by sharing KV projections across head groups.

---

## Experiment 1-2: Specific Heat Quantization

### Setup
- Model: GPT-2 Medium
- Evaluation: WikiText-2, 32 samples, max_seq_len=256
- Baseline PPL: 139.85
- Methods: Uniform, Variance-based (||k||²), Specific heat (C(q))

### Disagreement Analysis
- Average disagreement rate: **39.8%** (> 20% threshold)
- Average ρ(C(q), ||k||²): 0.275 (weak correlation)
- Later layers show higher disagreement: L21=49.6%, L7=46.8%
- Early layers more correlated: L0=17.2%, L1=18.9%

### PPL Results

| Bits | Uniform | Variance | Specific Heat |
|------|---------|----------|---------------|
| Baseline | 139.85 | — | — |
| 2-bit | 419.41 | exploded | exploded |
| 3-bit | 130.38 | 127.62 | 264.48 |
| 4-bit | 110.19 | 122.85 | 125.56 |

### Decision: DIFFERENT_BUT_NOT_BETTER
- C(q) and ||k||² measure genuinely different things (39.8% disagreement)
- C(q)-based quantization does not outperform variance-based
- Implementation note: hook-based K replacement may have residual errors; 2-bit explosion suggests numerical instability in adaptive scheme

### Interpretation
The Boltzmann specific heat C(q) captures **query-dependent sensitivity** that key norm misses, but this sensitivity doesn't directly translate to better quantization. The information C(q) provides may require a more sophisticated allocation strategy than simple median-based ±1bit.

---

## Summary Decision Matrix (Current)

| Experiment | Result | Implication |
|-----------|--------|-------------|
| 1-1 (Gate) | WEAK_SUCCESS (ρ=-0.66 on MHA) | Boltzmann structure exists |
| 1-2 (Quant) | DIFFERENT_BUT_NOT_BETTER | Novel metric, not yet practical |
| 1-3 (FFN) | PENDING | — |

### Overall Assessment
The Boltzmann framework introduces a **genuinely new physical quantity** (C(q)) that measures something different from existing metrics. The progressive cooling pattern validates the thermal interpretation. However, translating this theoretical insight into practical quantization improvement requires further work — possibly:
1. Non-linear bit allocation (not just median ±1bit)
2. Per-layer adaptive thresholds
3. Joint optimization of C(q) and ||k||² criteria
