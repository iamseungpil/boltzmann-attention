# Experiment 1-1: Boltzmann T_eff Gate Test Results

## Setup
- Models: Qwen2.5-3B (GQA, 2 KV heads) + GPT-2 Medium (standard MHA)
- Calibration: 64 samples, max_seq_len=512
- Metric: Spearman ρ(layer_index, T_eff)

## Results

### Qwen2.5-3B (GQA)
- ρ = +0.1081, p = 5.30e-01
- Gate: **FAILURE**
- Pattern: V-shaped (hot L0-2 → cold L3-5 → fluctuating → hot L34-35)
- GQA KV group gap up to 0.506 within single layer (L5)
- Head variance: Initial/Final 0.005, Middle 0.036

### GPT-2 Medium (MHA)
- ρ = -0.6574, p = 4.82e-04
- Gate: **WEAK_SUCCESS**
- Pattern: Clear progressive cooling (0.86 → 0.22) + final layer reheat (0.52)
- Head variance moderate and consistent

## Key Finding
Progressive cooling EXISTS in standard MHA models but is disrupted by GQA.
GQA's shared KV heads create artificial temperature patterns that mask the underlying Boltzmann structure.

## Next Steps
- Proceed to Exp 1-2 (specific heat quantization) on GPT-2 Medium
- Proceed to Exp 1-3 (FFN frequency selectivity)
- Future: test on non-GQA large models or per-KV-group T_eff analysis
