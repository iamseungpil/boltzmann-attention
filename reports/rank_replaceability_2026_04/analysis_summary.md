# E1 Rank Measurement — Analysis Summary

Auto-generated from: `reports/rank_replaceability_2026_04/*_n*.json`  
Files analyzed: 8

## Headline table (τ=0.95)

| File | Model | Task | N | r*_mean | r*_med | r*_max | r*_p95 | high-rank heads (≥8) | bimodality | prefix_len |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama_metatool_n256.json` | Meta-Llama-3.1-8B-Instruct | metatool_st4 | 256 | 1.38 | 1 | 13 | 3.0 | 6/1024 (0.6%) | 12.00 | 148 |
| `llama_tau2_airline_n256.json` | Meta-Llama-3.1-8B-Instruct | tau2_airline | 50 | 1.10 | 1 | 4 | 2.0 | 0/1024 (0.0%) | 3.00 | 98 |
| `llama_tau2_retail_n256.json` | Meta-Llama-3.1-8B-Instruct | tau2_retail | 114 | 1.06 | 1 | 4 | 1.8 | 0/1024 (0.0%) | 3.00 | 98 |
| `llama_tau2_telecom_n256.json` | Meta-Llama-3.1-8B-Instruct | tau2_telecom | 256 | 1.00 | 1 | 1 | 1.0 | 0/1024 (0.0%) | 0.00 | 98 |
| `qwen_metatool_n256.json` | Qwen2.5-7B-Instruct | metatool_st4 | 256 | 2.25 | 1 | 40 | 8.0 | 43/784 (5.5%) | 39.00 | 144 |
| `qwen_tau2_airline_n256.json` | Qwen2.5-7B-Instruct | tau2_airline | 50 | 1.54 | 1 | 23 | 3.0 | 17/784 (2.2%) | 22.00 | 93 |
| `qwen_tau2_retail_n256.json` | Qwen2.5-7B-Instruct | tau2_retail | 114 | 1.59 | 1 | 28 | 4.0 | 19/784 (2.4%) | 27.00 | 93 |
| `qwen_tau2_telecom_n256.json` | Qwen2.5-7B-Instruct | tau2_telecom | 256 | 1.01 | 1 | 2 | 1.0 | 0/784 (0.0%) | 1.00 | 93 |

## Per-τ summary

### τ = 0.90

| File | r*_mean | r*_med | r*_max | r*_p99 | high-rank heads |
|---|---:|---:|---:|---:|---:|
| `llama_metatool_n256.json` | 1.08 | 1 | 7 | 3.0 | 0/1024 |
| `llama_tau2_airline_n256.json` | 1.02 | 1 | 2 | 2.0 | 0/1024 |
| `llama_tau2_retail_n256.json` | 1.01 | 1 | 2 | 1.8 | 0/1024 |
| `llama_tau2_telecom_n256.json` | 1.00 | 1 | 1 | 1.0 | 0/1024 |
| `qwen_metatool_n256.json` | 1.47 | 1 | 27 | 12.5 | 15/784 |
| `qwen_tau2_airline_n256.json` | 1.27 | 1 | 17 | 10.2 | 12/784 |
| `qwen_tau2_retail_n256.json` | 1.30 | 1 | 21 | 11.2 | 14/784 |
| `qwen_tau2_telecom_n256.json` | 1.00 | 1 | 2 | 1.0 | 0/784 |

### τ = 0.95

| File | r*_mean | r*_med | r*_max | r*_p99 | high-rank heads |
|---|---:|---:|---:|---:|---:|
| `llama_metatool_n256.json` | 1.38 | 1 | 13 | 5.8 | 6/1024 |
| `llama_tau2_airline_n256.json` | 1.10 | 1 | 4 | 2.0 | 0/1024 |
| `llama_tau2_retail_n256.json` | 1.06 | 1 | 4 | 2.0 | 0/1024 |
| `llama_tau2_telecom_n256.json` | 1.00 | 1 | 1 | 1.0 | 0/1024 |
| `qwen_metatool_n256.json` | 2.25 | 1 | 40 | 21.2 | 43/784 |
| `qwen_tau2_airline_n256.json` | 1.54 | 1 | 23 | 14.2 | 17/784 |
| `qwen_tau2_retail_n256.json` | 1.59 | 1 | 28 | 16.5 | 19/784 |
| `qwen_tau2_telecom_n256.json` | 1.01 | 1 | 2 | 2.0 | 0/784 |

### τ = 0.99

| File | r*_mean | r*_med | r*_max | r*_p99 | high-rank heads |
|---|---:|---:|---:|---:|---:|
| `llama_metatool_n256.json` | 4.09 | 3 | 37 | 20.8 | 145/1024 |
| `llama_tau2_airline_n256.json` | 1.98 | 2 | 9 | 6.0 | 4/1024 |
| `llama_tau2_retail_n256.json` | 1.88 | 2 | 9 | 6.0 | 5/1024 |
| `llama_tau2_telecom_n256.json` | 1.04 | 1 | 2 | 2.0 | 0/1024 |
| `qwen_metatool_n256.json` | 7.15 | 3 | 71 | 50.0 | 225/784 |
| `qwen_tau2_airline_n256.json` | 2.98 | 2 | 33 | 24.2 | 54/784 |
| `qwen_tau2_retail_n256.json` | 3.12 | 2 | 44 | 30.3 | 62/784 |
| `qwen_tau2_telecom_n256.json` | 1.08 | 1 | 3 | 2.0 | 0/784 |

## Layer profiles (τ=0.95, mean r* per layer)

### llama_metatool_n256.json

```
L00:  2.59  █████
L01:  1.06  ██
L02:  1.06  ██
L03:  1.12  ██
L04:  1.16  ██
L05:  1.47  ██
L06:  1.12  ██
L07:  1.19  ██
L08:  1.25  ██
L09:  1.19  ██
L10:  1.25  ██
L11:  1.38  ██
L12:  1.56  ███
L13:  1.38  ██
L14:  1.22  ██
L15:  1.19  ██
L16:  1.31  ██
L17:  1.31  ██
L18:  1.22  ██
L19:  1.19  ██
L20:  1.12  ██
L21:  1.19  ██
L22:  1.41  ██
L23:  1.38  ██
L24:  1.69  ███
L25:  1.47  ██
L26:  1.75  ███
L27:  1.47  ██
L28:  1.75  ███
L29:  1.78  ███
L30:  1.69  ███
L31:  1.41  ██
```

### llama_tau2_airline_n256.json

```
L00:  1.22  ██
L01:  1.00  ██
L02:  1.22  ██
L03:  1.09  ██
L04:  1.03  ██
L05:  1.19  ██
L06:  1.09  ██
L07:  1.03  ██
L08:  1.03  ██
L09:  1.06  ██
L10:  1.12  ██
L11:  1.03  ██
L12:  1.00  ██
L13:  1.00  ██
L14:  1.00  ██
L15:  1.06  ██
L16:  1.03  ██
L17:  1.03  ██
L18:  1.06  ██
L19:  1.06  ██
L20:  1.25  ██
L21:  1.09  ██
L22:  1.06  ██
L23:  1.06  ██
L24:  1.09  ██
L25:  1.12  ██
L26:  1.25  ██
L27:  1.25  ██
L28:  1.12  ██
L29:  1.25  ██
L30:  1.31  ██
L31:  1.09  ██
```

### llama_tau2_retail_n256.json

```
L00:  1.22  ██
L01:  1.00  ██
L02:  1.22  ██
L03:  1.09  ██
L04:  1.00  ██
L05:  1.12  ██
L06:  1.03  ██
L07:  1.00  ██
L08:  1.03  ██
L09:  1.03  ██
L10:  1.03  ██
L11:  1.00  ██
L12:  1.00  ██
L13:  1.00  ██
L14:  1.00  ██
L15:  1.03  ██
L16:  1.00  ██
L17:  1.00  ██
L18:  1.03  ██
L19:  1.00  ██
L20:  1.03  ██
L21:  1.03  ██
L22:  1.00  ██
L23:  1.06  ██
L24:  1.03  ██
L25:  1.06  ██
L26:  1.12  ██
L27:  1.22  ██
L28:  1.03  ██
L29:  1.12  ██
L30:  1.19  ██
L31:  1.06  ██
```

### llama_tau2_telecom_n256.json

```
L00:  1.00  ██
L01:  1.00  ██
L02:  1.00  ██
L03:  1.00  ██
L04:  1.00  ██
L05:  1.00  ██
L06:  1.00  ██
L07:  1.00  ██
L08:  1.00  ██
L09:  1.00  ██
L10:  1.00  ██
L11:  1.00  ██
L12:  1.00  ██
L13:  1.00  ██
L14:  1.00  ██
L15:  1.00  ██
L16:  1.00  ██
L17:  1.00  ██
L18:  1.00  ██
L19:  1.00  ██
L20:  1.00  ██
L21:  1.00  ██
L22:  1.00  ██
L23:  1.00  ██
L24:  1.00  ██
L25:  1.00  ██
L26:  1.00  ██
L27:  1.00  ██
L28:  1.00  ██
L29:  1.00  ██
L30:  1.00  ██
L31:  1.00  ██
```

### qwen_metatool_n256.json

```
L00:  4.71  █████████
L01: 11.00  ██████████████████████
L02:  2.36  ████
L03:  5.79  ███████████
L04:  1.32  ██
L05:  1.54  ███
L06:  1.25  ██
L07:  2.54  █████
L08:  1.43  ██
L09:  2.32  ████
L10:  1.29  ██
L11:  1.14  ██
L12:  1.29  ██
L13:  1.39  ██
L14:  1.11  ██
L15:  1.00  ██
L16:  1.04  ██
L17:  1.07  ██
L18:  3.04  ██████
L19:  2.11  ████
L20:  1.71  ███
L21:  1.71  ███
L22:  1.68  ███
L23:  1.07  ██
L24:  2.14  ████
L25:  2.07  ████
L26:  1.18  ██
L27:  2.61  █████
```

### qwen_tau2_airline_n256.json

```
L00:  2.96  █████
L01:  6.89  █████████████
L02:  1.68  ███
L03:  4.21  ████████
L04:  1.25  ██
L05:  1.14  ██
L06:  1.11  ██
L07:  1.50  ███
L08:  1.11  ██
L09:  1.18  ██
L10:  1.07  ██
L11:  1.11  ██
L12:  1.14  ██
L13:  1.07  ██
L14:  1.07  ██
L15:  1.00  ██
L16:  1.00  ██
L17:  1.04  ██
L18:  1.39  ██
L19:  1.04  ██
L20:  1.11  ██
L21:  1.14  ██
L22:  1.14  ██
L23:  1.11  ██
L24:  1.04  ██
L25:  1.18  ██
L26:  1.04  ██
L27:  1.29  ██
```

### qwen_tau2_retail_n256.json

```
L00:  2.89  █████
L01:  8.32  ████████████████
L02:  1.71  ███
L03:  4.39  ████████
L04:  1.29  ██
L05:  1.11  ██
L06:  1.07  ██
L07:  1.36  ██
L08:  1.14  ██
L09:  1.14  ██
L10:  1.11  ██
L11:  1.14  ██
L12:  1.14  ██
L13:  1.07  ██
L14:  1.04  ██
L15:  1.00  ██
L16:  1.00  ██
L17:  1.04  ██
L18:  1.25  ██
L19:  1.04  ██
L20:  1.07  ██
L21:  1.07  ██
L22:  1.14  ██
L23:  1.00  ██
L24:  1.04  ██
L25:  1.11  ██
L26:  1.07  ██
L27:  1.68  ███
```

### qwen_tau2_telecom_n256.json

```
L00:  1.07  ██
L01:  1.25  ██
L02:  1.00  ██
L03:  1.00  ██
L04:  1.00  ██
L05:  1.00  ██
L06:  1.00  ██
L07:  1.00  ██
L08:  1.00  ██
L09:  1.00  ██
L10:  1.00  ██
L11:  1.00  ██
L12:  1.00  ██
L13:  1.04  ██
L14:  1.00  ██
L15:  1.00  ██
L16:  1.00  ██
L17:  1.00  ██
L18:  1.04  ██
L19:  1.00  ██
L20:  1.00  ██
L21:  1.00  ██
L22:  1.00  ██
L23:  1.00  ██
L24:  1.00  ██
L25:  1.00  ██
L26:  1.00  ██
L27:  1.00  ██
```

## Diagnostic notes

- **Theorem 1 prediction**: r*(τ) determines the rank-bound for static prompt replaceability. Mean r*(0.95) ≤ 16 ⇒ corollary 1.1 sufficient condition; > 64 ⇒ corollary 1.2 (need query-conditional).
- **Bimodality**: high `max` with low `median` indicates head specialization. Look for high-rank heads (r* ≥ 8) clustering in particular layers.
- **Caveat (τ² runs)**: this loader uses a *generic* tool-selection system prompt without per-domain tool catalogs. Real τ² evaluation feeds the full RETAIL_TOOLS/TELECOM_TOOLS/etc. catalog. Numbers below should be interpreted as lower bounds; full-catalog measurement is a follow-up.
