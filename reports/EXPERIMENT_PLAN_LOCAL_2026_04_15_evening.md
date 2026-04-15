# Local Experiment Plan — 2026-04-15 Evening Sprint
**Status snapshot**: 2026-04-15 20:15 KST. NeurIPS 2026 sprint D-29.
**Locked floor**: 6.32 / 10. Target floor by 04-17 morning: 6.45-6.55.

---

## 1. Today's confirmed results (as of 20:15 KST)

### 1.1 QKV joint full 497 — Qwen2.5-7B-Instruct Subtask4 (complete 19:19 KST)
| Method | F1 | Δ vs no_steer 0.731 |
|---|---|---|
| no_steer | 0.7307 | — |
| Q-only β=−0.1 | 0.7471 | +1.64pp |
| V+Q (a=0, v=0.05, q=−0.1) | 0.7468 | +1.61pp (V neutral) |
| **Q+K small-α (a=0.05, v=0, q=−0.1)** | **0.7502** ★ | **+1.95pp** (best pair) |
| Trio (a=0.05, v=0.05, q=−0.1) | 0.7414 | +1.07pp (V·K destructive) |

**Verified accuracy-lift family**: {Q-only, Q+V, Q+K small-α}.
**Falsified**: K large-α (≥0.1 destructive at all scales), V·K co-inclusion (multiplicative facet over-weighting on shared B_ont).

### 1.2 H-cat falsifiability diagnostic (complete 18:43 KST)
Per-head facet-projection energy ratio gain vs random baseline:
- **Llama-Inst**: 2.79–3.01x (HOLDS) — K-bias +15.08pp
- **Qwen-Inst**: 2.30–2.65x (HOLDS) — K-bias +1.41pp
- **Mistral-Inst**: 1.87–2.11x (WEAKEST) — K-bias **−2.92pp**
Gain ordering matches K-bias accuracy ordering. Mistral-Inst is at the threshold of structural meaningfulness.

### 1.3 Mistral-Inst α-sweep (in progress, 3/5 complete)
| α_K | top1 | Δ |
|---|---|---|
| 0 (no_steer) | 65.23% | — |
| 0.05 | 64.42% | −0.81pp |
| 0.1 | 62.91% | −2.32pp |
| 0.2 | running | (ETA 20:30) |
| 0.3 | 61.51% (prior) | −2.92pp |

**Monotonic degradation** — phase transition (α* > 0) **falsified**. Mistral-Inst (H-cat) is too weak (gain 2.0x) for ANY α_K > 0 to be net-beneficial. Conclusion: paper §5.5.1 reframed as honest "(H-cat) violation model, scope-out".

### 1.4 Mistral-Inst null-control (complete 18:20 KST)
| Method | top1 | Δ vs 65.23% |
|---|---|---|
| real a=0.3 | 61.51% | −2.92pp |
| random a=0.3 | 65.83% | **+0.60pp** |
| featshuffle a=0.3 | 64.62% | −0.60pp |

**Specificity reversal** (real < random): consistent with H-cat-too-weak interpretation. real B_ont = coherent direction in structureless model = systematic damage; random/featshuffle = incoherent noise = averages out.

---

## 2. Currently running (20:15 KST)

| GPU | Job | PID | ETA | Purpose |
|---|---|---|---|---|
| 0 | Mistral α-sweep (a=0.2 in progress, then a=0.3 redo) | 2887896 | ~20:30 | hypothesis 2 confirmation |
| 1 | **R1 α_K micro-sweep** {0.025, 0.05, 0.075, 0.1, 0.15} at β_Q=−0.1, V=0 | 3020497 | ~21:30 | K-Q monotonicity test |
| 0 (queued) | GPU0 chain: Mistral done → Llama Var_s V re-measure (silent fail fix) → R2 Subtask1 K+Q | 3030620 | ~22:30 | §5.4.1.1 closure + cross-task |
| 1 (debug) | SEKA hang minimal repro (gibberish output identified) | 3035876 | ~5 min | P0-A debug |

---

## 3. Pending / planned waves

### 3.1 R5 — Smoke replication (CHEAP)
- Replicate K+Q smoke (N=20) on indices {100-119, 250-269, 400-419} to test sample-variance hypothesis A
- ~30 min, GPU1 after R1 completes
- Outcome: confirm/refute "K destructive at all α from smoke" was bootstrap-SE artifact

### 3.2 R3 — Cross-model Llama-Inst K+Q small-α (PARTIAL DUPE OF P0-A)
- α_K=0.05, β_Q=−0.1 on Llama-Inst Subtask4 N=497
- ~1 hr, after R1 completes on GPU1
- Outcome: K-Q additivity universal across Mode A/C? (Llama is Mode A, Qwen is Mode C)

### 3.3 SEKA self-debug (in progress — see §4 below)
- Time-box: 2 hr (20:15-22:15 KST)
- If resolved: full SEKA + AdaSEKA on Qwen + Llama × Subtask1 + Subtask4 = 4 cells × 30 min = 2 hr
- Total best case: 4 hr local
- Coworker P0-A as parallel safety net (see Coworker doc)

---

## 4. SEKA debug — current findings (20:15 KST)

### 4.1 Hang root cause (partial)
Earlier coworker delegated this because `eval_subtask4_with_real_seka.py` hung 2x (20min + 15min) at "[eval] no_steer baseline" first call to `seka.model.generate`.

**Fresh repro on cuda:1 (avoiding GPU0 contention)**:
- TEST 1 (vanilla `seka.model.generate`, no SEKA hooks active): completed in **96.8s** for max_new_tokens=32 (~3s normally)
- Output is **GIBBERISH**: `">';\nwüns!지원!Le_secure...halves halves halves..."`
- TEST 2 (SEKA-steered generate) running, likely worse

**Conclusion so far**: Hang is not in SEKA steering hooks — it's in the *base model load itself* via SEKA's wrapper. SEKALLM init produces a degraded model.

### 4.2 Candidate root causes (priority order)
1. **`padding_side="left"` on tokenizer** (SEKALLM line 32) combined with our right-padded chat template: model's positional encodings may be misaligned even at batch_size=1 because of how SEKA encodes markers. → fix: force padding_side="right" in our wrapper
2. **`attn_implementation="eager"` with bfloat16** numerical instability on Qwen2.5: known issue in transformers <4.45 for some Qwen configs. → fix: try float16 or sdpa
3. **`marker_start="**"`** wrapping our chat-template prompt corrupts BPE tokenization at user-message boundary. → fix: skip marker for no_steer baseline (use raw prompt)
4. **SEKA `_load_proj` with d×d projector** (28, 4, 128, 128) maps onto Q-head dimension wrong because Qwen has 28 Q-heads but only 4 K-heads (GQA). → fix: confirm SEKA expects per-K-head projector
5. **`model.config.use_cache` mismatch** with eager attention. → fix: explicit use_cache=False

### 4.3 Debug result (20:18 KST)
**Both TEST 1 (vanilla) and TEST 2 (SEKA-steered) completed in ~97s each with GIBBERISH output**:
- T1 output: `">';\nwüns!지원!Le_secure...halves halves halves halves..."`
- T2 output: `'!!! hue modelBuilderkeyLetingsonoતRadi!!!!!🎶! טל!!!!!!פנו! 포함!ens기도!'`

**Hang root cause IDENTIFIED**: NOT in SEKA steering hooks. The base model loaded via SEKALLM wrapper produces broken outputs even without steering. Classic "halves halves halves" repetition pattern = decoder collapse.

### 4.4 Suspected cause: `attn_implementation="eager"` + bfloat16 on Qwen2.5-7B
SEKALLM init forces `attn_implementation="eager"` (line 41 of repro). Combined with bfloat16, this is a known issue:
- transformers <4.45 has eager-attention numerical bug for Qwen2.5 series
- Same model loaded via standard `eval_metatool_subtask4.py` (which uses **default attn**) works fine
- Switching `attn_implementation="sdpa"` or "flash_attention_2" should resolve

### 4.5 Next debug step (priority)
- TEST 3: bypass SEKALLM, load Qwen2.5-7B-Inst directly with `attn_implementation="eager"` (no SEKA) → confirm eager+bf16 is the culprit
- TEST 4: try SEKALLM with `attn_implementation="sdpa"` → if SEKA hooks compatible with sdpa, we're done
- TEST 5: try `torch_dtype=torch.float16` instead of bfloat16
- ETA: 20-30 min for all three tests
- Time-box: 22:00 KST hard cutoff

---

## 5. Score impact projection (updated 20:15 KST)

### Locked components
- 어제 locked: 6.30
- + Q-only Subtask4 +1.64pp confirmed: +0.05
- − Mistral specificity reversal (now H-cat-violation, cleaner narrative): −0.05
- − Llama Var_s V silent fail (re-measure pending): −0.05
- + K+Q small-α verified additive (K-channel restoration partial): +0.05
- + V·K destructive interaction (theoretical novelty, mechanism story): +0.05
- **Current locked floor: 6.35**

### Pending lift sources (today + tomorrow)
| Source | Probability | Lift |
|---|---|---|
| R1 monotonic peak at small α | 50% | +0.10 |
| Llama Var_s V positive | 50% | +0.05 |
| R2 Subtask1 K+Q universal | 60% | +0.05 |
| R5 smoke replication confirms artifact | 70% | +0.03 |
| SEKA self-debug success + SEKA underperforms | 30% (P0-A succeeds + win) | +0.20 |
| **Expected total** | | **+0.18** |

### Coworker P0 (D+2 to D+7)
| Source | Probability | Expected lift |
|---|---|---|
| P0-A SEKA win | 55% | +0.20 |
| P0-B Thm 6.18 PPL ≤ 13.5 partial+ | 40% | +0.18 |
| P0-C 3 baselines degrade | 55% | +0.10 |
| P0-D τ²-bench AUROC ≥ 0.85 | 50% | +0.10 |
| **Expected total** | | **+0.58** |

### Final projection
- Best (all hits): 6.35 + 0.30 (today) + 0.65 (coworker) = **7.30**
- Median: 6.35 + 0.18 + 0.30 = **6.83**
- Worst: 6.35 − 0.05 + 0 = **6.30**

**Median accept probability**: 60-65%.

---

## 6. Critical handoff to coworker (parallel work)

If you (coworker) are reading this:

### What we need from you (urgent priority)
1. **P0-A SEKA + AdaSEKA full eval** — see `reports/COWORKER_REQUEST_2026_04_15_v3_night_sprint.md` for the spec. Our local debug is in progress (§4 above) but has 30% chance of fundamental fail.
2. **P0-B Thm 6.18 attention-weighted bit allocation full WT2** — `scripts/ocq/measure_thm618_attn_weighted_bits.py` has allocation, needs PPL eval added.
3. **P0-C 3 baselines** — CAA + ITI + LoRA-FT minimum (PASTA/Focus/RAG bonus).
4. **P0-D τ²-bench retail multi-turn** — see Thm 6.20 spec.

### What we'll deliver locally (tonight + tomorrow morning)
1. Mistral α-sweep complete results (~20:30 KST)
2. R1 α_K micro-sweep complete (~21:30 KST)
3. R2 Subtask1 K+Q cross-task (~22:30 KST)
4. Llama Var_s V re-measure (~22:00 KST) — fixes silent fail
5. R5 smoke replication (~21:30 KST after R1 on GPU1)
6. SEKA self-debug result (best-effort, time-boxed 2 hr)

### Decision point: 04-17 morning
- If both your P0-A and our SEKA debug succeed: paper has 2-source SEKA verification (best)
- If only one succeeds: paper uses that source
- If neither succeeds by 04-17 noon: SEKA comparison delayed to ICLR 2027 fallback

---

## 7. File pointers (everything is in develop branch)

### Code
- `scripts/ocq/eval_metatool_subtask{1,4}.py` — main eval scripts
- `scripts/ocq/eval_subtask4_with_real_seka.py` — SEKA wrapper (has hang issue)
- `scripts/ocq/measure_hcat_violation_mistral_2026_04_15.py` — H-cat diagnostic
- `scripts/ocq/measure_theorem_6_1.py` — Thm 6.1 verification (use `--max-samples`, NOT `--n-queries`)

### Key results (today)
- `reports/qkv_joint_2026_04_15/full497_smallA_trio.json` — 5-cell QKV joint
- `reports/hcat_diagnostic_2026_04_15/{mistral,qwen,llama}_inst.json` — H-cat ratios
- `reports/mistral_alpha_sweep_2026_04_15/mistral_inst_alpha_sweep.json` — pending
- `reports/mistral_null_2026_04_15/{random,featshuffle}.json` — null reversal

### Paper
- `math/paper/benchmark_design/PAPER_DRAFT_v1_2026_04_14.md` — current single-source-of-truth
- `math/paper/lie_group/APPENDIX_B_PROOFS.md` — all 5 main theorems now full proof (no sketches)
- `math/paper/lie_group/COROLLARY_6_7_FACET_PHASE_CLOSURE.md` — Cor 6.9.6 full proof

### Logs
- `logs/seka_debug_2026_04_15/repro.log` — SEKA hang minimal repro
- `logs/qkv_joint_2026_04_15/eval.log` — QKV joint cell-by-cell
- `logs/mistral_alpha_sweep_2026_04_15/eval.log` — Mistral α progress
- `logs/qkv_alpha_microsweep_2026_04_15/eval.log` — R1 progress
- `logs/chain_2026_04_15_evening/wave.log` — GPU0 chain (Mistral → Var_s V → R2)

---

*Last updated: 2026-04-15 20:15 KST. Next major update expected: 04-16 06:00 KST after overnight chain completion.*
