# Unified Experiment Plan v4 — Llama Cross-Model + Theorem β* Predictor

**Date**: 2026-04-18 (drafted 2026-04-16, target execution finish Saturday 2026-04-18)
**Supersedes**: `EXPERIMENT_PLAN_UNIFIED_2026_04_17_v3.md`
**Paper target**: `paper/neurips2026_steering_ko` (canonical — English mirror explicitly excluded)

## 0. Scope delta vs v3

v3 explicitly declared Llama out of scope. v4 reverses that single point:

- v4 adds a Llama-3.1-8B-Instruct cross-model strand (L1–L4) that validates C1 survives model-family change.
- v4 adds a Theorem β* empirical-predictor strand (B1–B3) formalizing the sign predictor already in `sections/05_theory.tex` and `sections/09_appendices.tex`. Schema-G variant already failed on telecom (locked on develop); v4 tests whether a logit-lens discriminative redefinition of $\mathcal{G}$ restores sign-agreement, honestly gating the predictor's main-body claim on its result.
- All v3 Qwen-only cells (E1–E9, G1) remain. v4 does not re-run them.

## 1. Top-level claim update

- **C1 (unchanged).** Rotational steering beats stationary steering on multi-tool selection on Qwen2.5-7B.
- **C1b (new).** The regime-split (Q-coverage wins on long-sequence τ², layer-adaptive wins on short MetaTool) survives a model-family change to Llama-3.1-8B-Instruct.
- **C2 (unchanged).** Efficiency — inference-time, no weight update.
- **C3 (new, gated).** Theorem β*'s first-order sign predictor $\operatorname{sign}(\bar r_{\mathcal{G}} - \bar r)$ agrees with the empirical best sign of $\beta$ at rate $\ge 0.7$ on τ²-bench telecom and retail, **only if** $\mathcal{G}$ is redefined via a logit-lens discriminative-token mask (not raw schema tokens).

If C3's logit-lens test fails, Theorem β* is kept as a first-order diagnostic with an honest "schema-G and discriminative-G both fail to recover sign on telecom" paragraph in both `05_theory.tex` and `09_appendices.tex`, and the main-body β* paragraph is softened to "first-order local characterization" without a per-query routing claim.

## 2. Gate check before any run

### G2. Environmental bootstrap gate (blocking)

- **Intent.** v3 silently assumed `external/tau2-bench/` and Qwen B_ont checkpoints exist. On local main branch they do not. Any experiment step fails at step 1 without this gate being green.
- **Hypothesis.** (a) Sierra Research τ²-bench public repo (URL to be confirmed — candidates: `github.com/sierra-research/tau2`, `github.com/sierra-research/tau2-bench`) clones into `external/tau2-bench` and provides `src/tau2/domains/{retail,telecom,airline}/tools.py` and `data/tau2/domains/{retail,telecom,airline}/tasks.json`; (b) Qwen B_ont retail and telecom checkpoints can be rebuilt from `reports/axis2_theoretical_verification/tau2_{retail,telecom}_ontology.json`; (c) Llama-3.1-8B-Instruct weights are cached on HF hub and loadable with `attn_implementation="eager"`.
- **Verification.**
  1. Clone repo (both candidate URLs tried; first success wins); `test -f external/tau2-bench/src/tau2/domains/retail/tools.py`.
  2. Build Qwen retail ontology JSON if missing: `python scripts/ocq/build_tau2_ontology.py --domain retail --out reports/axis2_theoretical_verification/tau2_retail_ontology.json`. Repeat telecom. (Retail + airline already exist locally; telecom needs confirming.)
  3. Build Qwen retail B_ont: `python scripts/ocq/build_qwen_metatool_b_ont.py --model Qwen/Qwen2.5-7B-Instruct --ontology-json reports/axis2_theoretical_verification/tau2_retail_ontology.json --out external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt --target-layers all`. Repeat telecom.
  4. Smoke Llama load: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', torch_dtype='auto', attn_implementation='eager')"`.
  5. Smoke hook: register a dummy forward hook on `model.model.layers[0].self_attn.k_proj`, run one prompt, assert hook fires. This catches SDPA/flash-attn fusion that might bypass the hook path.
- **Decision rule.** All five checks green → proceed. Any red → fix before any Llama or β* run.
- **Cost estimate.** Clone: 2 min. Qwen B_ont retail + telecom: ~30 min combined on A100. Llama load + hook smoke: 3 min.

## 3. Llama cross-model strand (C1b)

### L1. Llama B_ont build for retail and telecom

- **Intent.** Build per-(layer, head) ontology basis $B_{\mathrm{ont}}^{\mathrm{llama}} \in \mathbb{R}^{L \times H_{kv} \times d \times r}$ for Llama-3.1-8B-Instruct on τ² retail and telecom, mirroring the existing Qwen builds.
- **Hypothesis.** The 4-facet Gram-Schmidt basis builder in `build_qwen_metatool_b_ont.py` is architecturally generic (no Qwen-specific config accesses) and emits a valid `.pt` with shape matching Llama's `(L=32, H_kv=8, d=128, r=24)` without code edits. Facet coverage per head is within 10% of the Qwen figure (~85% per head on Qwen 7B per existing logs).
- **Verification.**
  1. `python scripts/ocq/build_qwen_metatool_b_ont.py --model meta-llama/Llama-3.1-8B-Instruct --ontology-json reports/axis2_theoretical_verification/tau2_retail_ontology.json --out external/SEKA/seka_projections/ontology-llama31-8b-tau2-retail/B_ont.pt --target-layers all --device cuda:0`. Repeat telecom.
  2. Assert output tensor `shape == (32, 8, 128, 24)` and per-head facet coverage mean $\ge 0.75$.
- **Decision rule.** Shape correct + mean coverage $\ge 0.75$ → pass. Shape mismatch → fix code; coverage $<0.75$ on more than 20% of heads → log anomaly, proceed but note in paper appendix footnote.
- **Cost estimate.** 15–20 min per domain on A100 (model load + 4-facet × 8-category × 10-sentence forward passes).

### L2. Llama τ² retail N=114 Q-sweep

- **Intent.** Fill the Llama retail row of the new cross-model table (to be inserted into `sections/06_experiments.tex` as Table 2b).
- **Hypothesis.** Q-only negative-β (coverage) wins with $\Delta F_1 \in [+2, +6]$pp peaking near $\beta \in \{-0.03, -0.05\}$, matching the Qwen retail `+5.11pp` regime. Sign is preserved under model-family change — predicted by Theorem β*'s sign statement on the same ontology subspace.
- **Verification.**
  - Baseline: `--methods no_steer`.
  - Sweep: `ocq_qbias_b{-0.10, -0.05, -0.03, 0.01, 0.03, 0.05, 0.10}` (7 β).
  - Also lock layer-adaptive: `ocq_ladapt_k0.05_q-0.03`.
  - `python scripts/ocq/eval_tau2_bench.py --model meta-llama/Llama-3.1-8B-Instruct --device cuda:0 --b-ont external/SEKA/seka_projections/ontology-llama31-8b-tau2-retail/B_ont.pt --methods no_steer ocq_qbias_b-0.10 ocq_qbias_b-0.05 ocq_qbias_b-0.03 ocq_qbias_b0.01 ocq_qbias_b0.03 ocq_qbias_b0.05 ocq_qbias_b0.10 ocq_ladapt_k0.05_q-0.03 --max-samples 114 --domain retail --out reports/tau2_2026_04_18/llama31_retail_full_v4.json`.
- **Decision rule.** Best negative-β cell $\ge$ `no_steer + 2pp` → sign preserved, cross-model claim C1b retail confirmed. Best cell is positive-β or layer-adaptive → regime classification revised for Llama, noted in paper.
- **Cost estimate.** 8 methods × 114 tasks × ~15s/task on A100 (Llama 8B eager-attn with 512 max-new-tokens — more conservative than Qwen given register pressure) ≈ 3.8 GPU-h.
- **Risk.** Llama 3.1 Instruct's tool-call format uses `<|python_tag|>{...}<|eom_id|>` by default. `eval_tau2_bench.py:extract_tool_names` relies on JSON `"name": "..."` pattern or substring fallback — the substring fallback should handle any format, but first-pass regex might miss structured Llama output. If N=10 smoke shows unexpectedly low `no_steer` F1 ($<0.2$), fall back to adding a `<|python_tag|>` extractor branch before running the full sweep.

### L3. Llama τ² telecom N=200 Q-sweep

- **Intent.** Fill the Llama telecom row. Telecom is the strongest Qwen signal (`+24.78pp`), so model-family transfer is the highest-leverage test.
- **Hypothesis.** Q-positive-β (focus) wins, because telecom has short tool namespaces and Qwen's best-β was positive. Expected $\Delta F_1 \in [+5, +20]$pp at $\beta \approx +0.05$. If instead negative-β wins on Llama telecom, the regime classification is model-specific (still interesting but revises the "operator family with regime-dependent ideal β" framing).
- **Verification.** Same as L2 but with `--domain telecom --max-samples 200` and telecom B_ont.
- **Decision rule.** Best cell sign matches Qwen telecom sign → strong C1b confirmation. Sign flipped → scope-reducing edit in paper noting Llama-specific regime.
- **Cost estimate.** 8 methods × 200 tasks × ~18s/task (telecom tasks longer than retail; Llama eager-attn) ≈ 8.0 GPU-h. **This is the tightest item on the critical path; N=200 may be reduced to N=100 if Fri evening is slipping.**

### L4. Llama MetaTool Subtask4 layer-adaptive (gate check)

- **Intent.** Confirm the layer-adaptive K+Q → MetaTool short-sequence regime also survives model change. Currently Qwen MetaTool ST4 shows layer-adaptive `+2.08pp`; check whether Llama ST4 shows same direction.
- **Hypothesis.** Layer-adaptive wins with $\Delta F_1 \in [+0.5, +3]$pp on Llama ST4, mirroring Qwen sign but smaller magnitude (Llama 8B has weaker tool-calling baseline).
- **Verification.** Two-step chain:
  1. **L4a (MetaTool Llama B_ont).** The existing `scripts/ocq/build_qwen_metatool_b_ont.py` uses MetaTool's 4-facet ontology baked into `ontology_facet_basis.ONTOLOGY`. Re-run it with `--model meta-llama/Llama-3.1-8B-Instruct --out external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt`. Expected ~0.7 h on A100 (MetaTool ontology has more categories than τ²).
  2. **L4b (run eval).** `python scripts/ocq/eval_subtask4_dynamic_qk_v2.py --model meta-llama/Llama-3.1-8B-Instruct --b-ont external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt --k-alpha 0.3 --q-beta -0.03 --tau 0.25 --max-samples 497 --out reports/llama_cross_model_2026_04_18/st4_ladapt_llama.json`.
- **Decision rule.** Sign preserved → C1b short-sequence confirmation. Sign flipped → report truthfully; may indicate the τ=1/4 choice is Qwen-specific.
- **Cost estimate.** L4a 0.7 h + L4b 1.0 h (Llama 8B + 497 queries) = 1.7 h. If Sat AM is tight, drop L4 entirely — L2+L3 already support C1b.

**L-strand blocker:** if Saturday deadline is tight, drop L4 first (redundant with L2+L3 for C1b), then L3 (retail preserves main retail story), keeping L1+L2 as minimum viable cross-model evidence.

## 4. Theorem β* predictor strand (C3, gated)

### B1. Reimplement logit-lens discriminative β* measurer

- **Intent.** Replace the known-broken schema-G measurer on develop (`measure_beta_star.py` → telecom schema-G fails at 31.2% sign-agreement) with a logit-lens variant that restricts $\mathcal{G}$ to tokens where the model's next-token logit (projected via the unembedding onto layer-$\ell$ residual) lands above threshold for the GT tool name. Redefining $\mathcal{G}$ from "raw schema tokens" to "discriminatively-attended tokens" is algebraically consistent with the proof (which only requires $\mathcal{G}$ to be any proxy for the actually-used attention mass on GT).
- **Hypothesis.** Sign-agreement rises from 31.2% (schema-G, locked) to $\ge 70\%$ on telecom N=20, and stays $\ge 70\%$ on retail N=30. Bootstrap 95% CI on the rate excludes 50%.
- **Verification.**
  1. Write `scripts/ocq/measure_beta_star_logit.py` that extends develop's `measure_beta_star.py`: adds a `--gt-mode logit_discriminative` option.
     - Capture per-layer residual stream via `output_hidden_states=True` on a single forward pass.
     - Apply the model's final RMSNorm (from `model.model.norm`) then the unembedding `model.lm_head.weight` to each layer's residual at every prompt position: `logits_ℓ,t = model.lm_head(model.model.norm(hidden_ℓ[t]))`.
     - Rank the GT tool's first-BPE token's logit at position $t$ across all positions. Define $\mathcal{G}_\ell$ = top-$k$ positions (`k=max(1, 0.1 * prompt_length)`) or threshold-based (position's GT-token logit $\ge$ prompt-wise mean + $\tau_{\mathrm{disc}} \cdot$ prompt-wise std, default `τ_disc=0.5`).
     - **Tokenization caveat.** For GT tool names containing `_` (e.g., `get_reservation_details`), tokenize with `tokenizer.encode(" " + tool_name, add_special_tokens=False)` — Qwen and Llama tokenize leading-space and no-leading-space differently. Take first token ID.
  2. Run on `reports/beta_star_2026_04_17/telecom_smoke20_*.json` and `retail_smoke30_allpos.json` as the sweep baselines (require develop JSONs cherry-picked first).
  3. Compute `r_t = <P Q, K_t>/√d` per (layer, head), $\bar r_{\mathcal{G}_\ell} - \bar r$, aggregate uniformly across (layer, head), predict sign, compare to empirical best-β sign (available in the Q-sweep JSONs once B2 is done).
  4. Output: `reports/beta_star_2026_04_18/{telecom,retail}_logit_discriminative.json` with per-sample predicted/empirical sign and overall agreement + bootstrap CI.
- **Decision rule.**
  - Agreement $\ge 70\%$ on both domains with bootstrap CI lower bound $> 50\%$ → C3 passes; main-body β* paragraph keeps "sign predictor" framing; appendix gets the logit-lens definition.
  - Agreement $\in [50\%, 70\%)$ → C3 partial; main-body softens to "first-order local characterization with logit-lens consistent direction"; no per-query router claim.
  - Agreement $< 50\%$ or CI contains 50% → C3 fails; main-body drops β* from discussion, moves it to appendix only.
- **Cost estimate.** Coding: ~2 h interactive (including τ_disc sensitivity sweep). Eval runs: ~0.5 GPU-h (single forward pass per prompt, no generation).

### B2. Lock Qwen τ² Q-sweep sign per sample

- **Intent.** For each prompt in τ² retail/telecom, record the empirical best β sign (the `argmax_β F1_per_sample(β) vs F1_per_sample(no_steer)` diff). This gives per-sample ground truth for comparing against β* prediction.
- **Hypothesis.** The per-sample best-β sign is well-defined (i.e., unique max for ≥80% of samples); ties resolved by preferring negative-β if absolute gain is within 0.5pp.
- **Verification.** `scripts/ocq/extract_per_sample_best_beta.py` ingests `reports/tau2_2026_04_17/{retail_full_v2,telecom_N200}.json` (if develop-locked) and emits per-sample `best_beta_sign.json`.
- **Decision rule.** $\ge 80\%$ of samples have unique max → use as ground-truth. $<80\%$ → report only on uniquely-signed subset.
- **Cost estimate.** 0 GPU, 30 min coding.

### B3. Bootstrap CI + honest scope paragraph

- **Intent.** Bootstrap the predictor's sign-agreement rate across 10000 resamples; write the honesty paragraph whose wording is determined by B1's outcome.
- **Verification.** `analyze_q_sign_significance.py` — outputs mean agreement, 95% CI, and writes the final line into `sections/07_discussion.tex` / `sections/09_appendices.tex` depending on B1 decision.

## 5. Execution order and wall-clock schedule

Assume single A100 at 15 GPU-h/day realistic. Deadline: Saturday 2026-04-18 ~18:00 (user takes over if not done by then).

| Slot | Items | GPU-h | Dep | Owner |
|------|-------|-------|-----|-------|
| Thu 16 PM | G2 bootstrap (clone + Qwen B_ont retail/telecom rebuild if needed) | 0.5 | — | me |
| Thu 16 PM | L1 Llama B_ont retail + telecom | 0.6 | G2 | me |
| Thu 16 evening | B2 per-sample best-β extractor (CPU) + develop JSON cherry-pick | 0 | — | me |
| Thu 16 overnight | L3 Llama telecom N=200 Q-sweep (run in background) | 8.0 | L1 | me |
| Fri 17 AM | B1 logit-lens measurer code + Qwen retail/telecom β* eval | 0.5 + 2h coding | B2 | me |
| Fri 17 PM | L2 Llama retail N=114 Q-sweep | 3.8 | L1 (L3 finished overnight) | me |
| Sat 18 AM | L4 Llama MetaTool ST4 ladapt (drop-first) | 1.7 | L1 + MetaTool Llama B_ont | me |
| Sat 18 AM | B3 bootstrap + honesty paragraph + paper edits | 0 | L2 + L3 + B1 | me |
| Sat 18 PM | Cherry-pick all new artifacts to develop + paper commit | 0 | — | me |

**Total GPU**: ~14.6 h spread over 48 h calendar window.

**Minimum viable set (drop-order for time pressure):**
1. G2 + L1 + L3 (overnight) = 9.1 h → Llama telecom cell filled (highest-leverage transfer test)
2. + L2 = 12.9 h → retail also filled
3. + B1 eval = 13.4 h → β* predictor status known
4. + L4 = 15.1 h → ST4 cell
5. + paper edits / commits

If overnight L3 fails (OOM, crash), L2 takes its evening slot and L3 drops to N=100 Sat AM.

**Extended set**: above + L4 + B1+B2+B3 = all boxes ticked.

## 6. Falsification triggers (additions)

- **T4.** L2 best-β on Llama retail is positive (opposite sign from Qwen) → regime-split claim becomes "Qwen-specific"; paper scope retreats to single-model main claim, Llama becomes appendix "model sensitivity" footnote.
- **T5.** L3 best-β on Llama telecom reverses sign vs Qwen → same retreat as T4.
- **T6.** L4 layer-adaptive on Llama ST4 is negative → τ=1/4 choice is flagged as Qwen-calibrated, paper notes "per-model calibration of τ may be needed".
- **T7.** B1 logit-lens agreement $<50\%$ or CI contains 50% → main-body β* paragraph removed; β* stays in theory section as first-order characterization only, no sign-predictor claim.

Two or more of T4–T7 → scope retreat: paper becomes "single-model positive result with first-order local theory, cross-model generality open".

## 7. Open questions the plan does NOT resolve

- Whether the $\tau=1/4$ K+Q split is Pareto-optimal for Llama (only LS-2 tested; Phase 2.5 sweep remains Qwen-only in v3 scope).
- Whether the logit-lens discriminative-G variant reflects true model attention mass or merely a correlated observable. The math (proof via softmax gradient identity) is $\mathcal{G}$-agnostic; the empirical predictor is only as good as its $\mathcal{G}$.
- Llama 70B or other sizes — explicitly out of v4 scope (cost + no 70B available).

## 7.5 Preflight smoke (must pass before any L-strand run)

- **P1. Llama base F1 preflight.** Run `eval_tau2_bench.py --model meta-llama/Llama-3.1-8B-Instruct --domain retail --methods no_steer --max-samples 5 --out reports/tau2_2026_04_18/llama31_retail_preflight.json` with L1's retail B_ont. Expected: `no_steer` F1 $\ge 0.3$. If $<0.2$, Llama's tool-extraction format is incompatible and the L-strand is blocked on a format-adapter fix (estimated 2 h extra).
- **P2. Hook fires.** Part of G2 step 5 — already listed there.
- **P3. tau2-bench git URL.** Part of G2 step 1. If both candidate URLs fail, block and escalate to user.

## 7.6 Background-run resilience

- Every L-strand run launches via `nohup ... &` with per-method JSONL checkpointing already in `eval_tau2_bench.py` (confirmed: the script writes per-sample after each task). A crash mid-run resumes from the last completed method.
- Add a lightweight watchdog: `scripts/ocq/watchdog_llama_runs.sh` that polls the output JSONL every 10 min and appends a line to a status log; if stalled > 30 min, kills + restarts from the last method.

## 8. Rollback/abort conditions

- If L1 build fails due to code incompatibility (e.g., Llama uses MLP-before-attn that breaks the k_proj hook), fall back to running one smaller test with `meta-llama/Llama-3.2-3B-Instruct` as a proof-of-concept + honestly report "8B B_ont blocked by {reason}".
- If G2 tau2-bench clone fails (network/auth), fallback to reusing the two ontology JSONs already under `reports/axis2_theoretical_verification/tau2_{retail,airline}_ontology.json` + manually build telecom JSON from the tau2-bench README if the domain tools.py is retrievable via curl.
- If any τ² run OOMs on A100 with Llama 8B + eager attn, switch to `attn_implementation="sdpa"` and re-verify hooks still fire on `self_attn.k_proj` (SDPA may fuse projections; requires a 10-line check).
