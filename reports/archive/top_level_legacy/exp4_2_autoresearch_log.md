# Exp 4-2 Autoresearch Log

**Date**: 2026-04-01  
**Project**: boltzmann-attention  
**Experiment**: Standard WikiText-2 PPL benchmark (`Exp 4-2`, `Exp 10-1` protocol)  
**Node target**: `tops-caiman`

## Goal

Implement and run the reviewer-grade sliding-window PPL benchmark described in
`FOKVQ_ADDITIONAL_EXPERIMENT_PLAN_v9.docx`.

## Setup

- Goal: compare `FP16`, `Uniform`, and `FOKVQ` under standard WikiText-2 protocol
- Scope: `scripts/fokvq/exp4_2_standard_ppl_benchmark.py`, remote launch commands, result JSONs
- Metric: finite and reproducible `PPL` values from sliding-window evaluation
- Verify: smoke run finishes without crash/OOM and writes result JSON
- Guard: no edits to unrelated user files; keep a local markdown audit trail

## Iterations

### Iteration 1

- Change: stabilized the initial sliding-window benchmark runner so the quantized and fp16 paths score the same suffix tokens and support the installed `transformers` cache API.
- Verification:
  - `tiny-gpt2` CPU smoke completed and wrote `/tmp/fokvq_exp4_2_smoke/tiny-gpt2_standard_ppl.json`
  - token counts aligned across methods (`127` tokens each in the earlier smoke)
- Decision: keep
- Reason: the benchmark became runnable and mechanically comparable.

### Iteration 2

- Change: strengthened internal baselines in `scripts/fokvq/exp4_2_standard_ppl_benchmark.py`.
  - added `variance` mixed-precision key quantization baseline
  - added `kivi`-style per-channel asymmetric key quantization baseline
  - added `identity` and `random` rotated non-uniform baselines sharing the same evaluation path as `fokvq`
- Verification:
  - `tiny-gpt2` CPU smoke:
    - command completed with methods `fp16 uniform variance kivi identity random fokvq`
    - all methods returned finite PPL and identical token counts (`127`)
  - `gpt2` CPU smoke:
    - output: `/tmp/fokvq_exp4_2_gpt2_baseline_smoke/gpt2-baselines-smoke_standard_ppl.json`
    - `fp16=76.64`, `uniform-2=109.30`, `variance-2=117.42`, `kivi-2=76.63`, `identity-2=159.66`, `random-2=121.18`, `fokvq-2=92.06`
    - all methods returned identical token counts (`63`)
  - remote GPU sanity on `metacognition-e8`:
    - output: `/scratch/amlt_code/outputs/exp4_2_remote_sanity/gpt2-remote-sanity_standard_ppl.json`
    - `fp16=57.30`, `uniform-2=81.34`, `variance-2=85.11`, `kivi-2=57.38`, `identity-2=129.31`, `random-2=125.79`, `fokvq-2=104.54`
    - all methods returned identical token counts (`127`)
- Decision: keep
- Reason: the strengthened baselines are executable under the same protocol and already produce differentiated behavior on `gpt2`, which is enough to proceed to longer runs.

### Iteration 3

- Change: staged the updated script on `metacognition-e8` and launched an early longer run on `openai-community/gpt2-medium` with methods `fp16 uniform variance kivi random fokvq` and bits `2 3 4`.
- Verification:
  - node: `metacognition-e8`
  - GPU reservation confirmed; launch and monitoring succeeded
  - output: `/scratch/amlt_code/outputs/exp4_2_gpt2_medium_baselines/gpt2-medium-baselines-early_standard_ppl.json`
  - protocol slice: `context_len=256`, `stride=128`, `calibration_samples=8`, `max_eval_tokens=4096`
  - results:
    - `fp16=26.86`
    - `uniform`: `32.49 / 27.27 / 26.90` at `2 / 3 / 4` bits
    - `variance`: `406.79 / 29.99 / 27.13`
    - `kivi`: `27.75 / 26.97 / 26.88`
    - `random`: `84.48 / 31.21 / 27.04`
    - `fokvq`: `48.28 / 27.24 / 26.99`
  - token counts matched across all runs (`4095`)
  - runtime: `31.72s`, peak memory: `0.85 GiB`
- Decision: keep
- Reason: the remote early run finished cleanly and gives a first ranking under a standard sliding-window setup. On this slice, `kivi` is currently the strongest low-bit baseline, while `fokvq` is not yet competitive at 2-bit and is roughly tied with uniform/kivi at 3-4 bits.

### Iteration 4

- Change: exposed the non-uniform high-bit axis budget as `--fokvq-topk-frac` and tested a wider allocation (`0.50` instead of the old implicit `0.25`).
- Verification:
  - local smoke with `tiny-gpt2` and `--fokvq-topk-frac 0.5` completed successfully
  - remote run:
    - output: `/scratch/amlt_code/outputs/exp4_2_gpt2_medium_topk50/gpt2-medium-topk50_standard_ppl.json`
    - protocol slice matched Iteration 3 except `fokvq_topk_frac=0.5`
  - comparison against the previous `0.25` run on `gpt2-medium`:
    - `random`: `84.48 -> 52.35` at 2-bit, `31.21 -> 28.36` at 3-bit, `27.04 -> 26.96` at 4-bit
    - `fokvq`: `48.28 -> 33.56` at 2-bit, `27.24 -> 27.07` at 3-bit, `26.99 -> 26.98` at 4-bit
    - `kivi` remained stronger at 2-bit (`27.75`), but the `fokvq` gap narrowed substantially
- Decision: keep
- Reason: the fixed 25% split was too narrow for this benchmark slice. Parameterizing the axis budget produced a clear, verifiable improvement for `fokvq`, especially at 2-bit, without destabilizing the runner.

### Iteration 5

- Change: added a conservative `turboquant-style` baseline to the shared `Exp 4-2` harness.
  - implementation scope:
    - random orthogonal rotation per layer/head
    - scalar Lloyd-Max codebook fitted on standard-normal samples
    - rotated-domain scalar quantization with scale restoration
  - added deterministic `--self-test` coverage for:
    - rotation orthogonality
    - monotonic/codebook-size correctness
    - held-out Gaussian Lloyd-Max vs simple scalar uniform reconstruction error
    - shape/dtype/finite preservation
    - higher-bit reconstruction error monotonicity
- Verification:
  - remote self-test on `metacognition-e8` passed:
    - `orthogonality_max_abs_error=4.77e-07`
    - `lloyd_mse=0.1148`
    - `uniform_mse=0.6005`
    - turboquant synthetic reconstruction MSE: `2-bit=0.1196`, `4-bit=0.0097`
  - `tiny-gpt2` harness smoke completed:
    - output: `/scratch/amlt_code/outputs/exp4_2_turboquant_smoke/tiny-gpt2-turbo-smoke_standard_ppl.json`
    - all methods returned identical token counts (`126`)
    - this run only validates path wiring; it is too small to separate methods
  - first `gpt2` sanity with `context_len=128`, `stride=128` was discarded:
    - reason: `context_len == stride` never exercised the quantized prefix-cache path, so every method collapsed to the same result
  - corrected `gpt2` sanity with actual quantized-prefix evaluation:
    - output: `/scratch/amlt_code/outputs/exp4_2_turboquant_gpt2_sanity_v2/gpt2-turbo-sanity-v2_standard_ppl.json`
    - protocol slice: `context_len=256`, `stride=128`, `max_eval_tokens=384`
    - results:
      - `fp16=41.43`
      - `uniform-2=49.70`, `uniform-4=41.51`
      - `kivi-2=43.50`, `kivi-4=41.32`
      - `random-2=82.16`, `random-4=41.21`
      - `turboquant-2=42.52`, `turboquant-4=41.64`
    - token counts matched across all methods (`383`)
  - remote `gpt2-medium` early slice including `turboquant`:
    - output: `/scratch/amlt_code/outputs/exp4_2_gpt2_medium_turboquant/gpt2-medium-turboquant-early_standard_ppl.json`
    - protocol matched the earlier early slice: `context_len=256`, `stride=128`, `calibration_samples=8`, `max_eval_tokens=4096`, `fokvq_topk_frac=0.5`
    - results:
      - `fp16=26.86`
      - `uniform`: `32.49 / 27.27 / 26.90`
      - `kivi`: `27.75 / 26.97 / 26.88`
      - `random`: `52.35 / 28.36 / 26.96`
      - `fokvq`: `33.56 / 27.07 / 26.98`
      - `turboquant`: `27.76 / 27.07 / 27.00`
    - token counts matched across all methods (`4095`)
    - runtime: `35.35s`, peak memory: `0.85 GiB`
- Decision: keep
- Reason: the implementation passed deterministic tests, exercised the actual quantized-cache path, and produced stable same-protocol results on `gpt2` and `gpt2-medium`. At 2-bit on the current `gpt2-medium` slice, `turboquant-style` is effectively tied with `kivi` and materially stronger than `random` or the current `fokvq`.

### Iteration 6

- Change: corrected mixed-precision bit allocation so `variance`, `random`, `identity`, and `fokvq` respect the target average bit budget rather than silently gaining extra bits when the high-precision fraction changes.
  - implemented a fair mixed-precision schedule derived from the actual selected high-precision fraction
  - added self-test checks for the fair schedules:
    - `2-bit @ 25% -> 5/1`
    - `2-bit @ 50% -> 3/1`
    - `3-bit @ 50% -> 4/2`
    - `4-bit @ 50% -> 5/3`
  - allowed a small `0.05 bit` tolerance only to avoid spurious crashes from non-exact token splits in the `variance` method
- Verification:
  - remote self-test on `metacognition-e8` still passed after the fairness correction
  - an attempted full `4096-token` rerun was abandoned as the verification vehicle because duplicate remote launches polluted the node state; those processes were explicitly cleaned up
  - replaced the slow/full rerun with a faster 2-bit proxy slice to verify the direction of the change under the corrected budget:
    - output: `/scratch/amlt_code/outputs/exp4_2_gpt2_medium_fair_proxy/gpt2-medium-fair-proxy-2bit_standard_ppl.json`
    - protocol slice: `context_len=256`, `stride=128`, `calibration_samples=8`, `max_eval_tokens=1024`
    - results:
      - `fp16=29.17`
      - `uniform-2=32.82`
      - `kivi-2=30.07`
      - `random-2=50.49`
      - `fokvq-2=35.35`
      - `turboquant-2=29.94`
    - token counts matched across all methods (`1023`)
    - runtime: `15.24s`, peak memory: `0.85 GiB`
  - interpretation:
    - under a fair average-bit constraint, `fokvq` at `topk_frac=0.5` regresses substantially relative to the earlier non-budget-preserving readout
    - the earlier `33.56` result from Iteration 4 should therefore not be treated as a fair main-table number
- Decision: keep
- Reason: the code change improves methodological correctness even though it hurts the headline `fokvq` metric. The fast proxy check is sufficient to show that the previous gain was at least partly a bit-budget artifact, which is exactly the kind of issue the loop is supposed to catch.

### Iteration 7

- Change: no further code edit; ran a bounded fair-schedule search step on the corrected harness to compare a second exact-fair-ish 2-bit `fokvq` setting against the fair `topk_frac=0.5` proxy.
  - candidate tested: `fokvq_topk_frac=0.3333333333`
- Verification:
  - output: `/scratch/amlt_code/outputs/exp4_2_gpt2_medium_fair_proxy/gpt2-medium-fair-proxy-2bit-topk033_standard_ppl.json`
  - protocol matched the Iteration 6 proxy except `fokvq_topk_frac=0.3333333333`
  - results:
    - `fp16=29.17`
    - `uniform-2=32.82`
    - `kivi-2=30.07`
    - `fokvq-2=39.54`
    - `turboquant-2=29.94`
  - direct comparison against the fair `topk_frac=0.5` proxy from Iteration 6:
    - `fokvq-2`: `35.35 -> 39.54` (worse)
    - `kivi-2` and `turboquant-2` remained much stronger than either `fokvq` setting
- Decision: discard as a candidate configuration
- Reason: moving from the fair `0.5` split toward `1/3` reduced `fokvq` quality further on the same 2-bit proxy slice. It does not help close the remaining gap to `kivi` or `turboquant-style`.

## Current Readout

- The benchmark runner is now usable for fair same-protocol internal comparison.
- `variance` is unstable at 2-bit on `gpt2-medium` and should not be trusted as a strong baseline without further refinement.
- `kivi` is the most credible strengthened baseline so far because it is simple, stable, and strong in the early standard-protocol run.
- the earlier `fokvq topk_frac=0.5` uplift was not fully fair because it increased the average bit budget; after correcting that, `fokvq` again trails `kivi` and `turboquant-style` at 2-bit.
- within the fair-budget proxy experiments tried so far, `fokvq_topk_frac=0.5` is better than `~1/3`, but neither is competitive with `kivi` or `turboquant-style` at 2-bit.
- `turboquant-style` is now a credible paper-aligned baseline inside the same harness, but it must still be described conservatively because this is not an official full TurboQuant reproduction.

## Next Moves

- continue `fokvq` tuning only under fair schedules, not under budget-inflating mixed-precision settings
- test other fair 2-bit schedules only if they are meaningfully different from the already-worse `~1/3` setting; otherwise shift effort toward changing the `fokvq` mechanism itself rather than just retuning the split
- decide whether the next comparison layer should prioritize a cleaner `KIVI-style / TurboQuant-style / FOKVQ` main table or move to external-code baseline integration

### Iteration 8

- Change: ran the planned 4-bit follow-up batch on `metacognition-e8` across all 4 GPUs to test whether the remaining novelty case for `fokvq` appears in:
  - the full `4096-token` standard main table
  - a `PCA vs identity vs random` axis ablation
  - longer-context settings at `context_len=512` and `context_len=1024`
- Verification:
  - all 4 remote runs finished cleanly and wrote JSON outputs:
    - `/scratch/amlt_code/outputs/exp4_2_followup_4bit/main/gpt2-medium-4bit-main_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_4bit/axis_ablation/gpt2-medium-4bit-axis-ablation_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_4bit/longctx512/gpt2-medium-4bit-longctx512_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_4bit/longctx1024/gpt2-medium-4bit-longctx1024_standard_ppl.json`
  - `metacognition-e8` returned to idle after completion (`GPU 0-3: 0 MiB, 0% util`)
  - main 4-bit table (`context_len=256`, `stride=128`, `max_eval_tokens=4096`):
    - `fp16=26.8638`
    - `uniform-4=26.8983`
    - `kivi-4=26.8770`
    - `turboquant-style-4=27.0020`
    - `fokvq-4=26.9954`
  - axis ablation (`context_len=256`, `stride=128`):
    - `identity-4=27.0862`
    - `random-4=26.9608`
    - `fokvq(PCA)-4=26.9954`
  - long-context `512/256`:
    - `fp16=22.5366`
    - `uniform-4=22.5972`
    - `kivi-4=22.5641`
    - `turboquant-style-4=22.6303`
    - `fokvq-4=22.6055`
  - long-context `1024/512`:
    - `fp16=18.3817`
    - `uniform-4=18.4042`
    - `kivi-4=18.3727`
    - `turboquant-style-4=18.4311`
    - `fokvq-4=18.4042`
  - note on execution:
    - per-method measured runtimes inside the JSONs stayed small (`~0.3s` to `~3.0s`)
    - total wall-clock runtime per job was still about `9-10 min`, which points to CPU-thread contention during shared setup/basis-building rather than the evaluation loop itself
- Decision: keep
- Reason: this batch is methodologically valuable even though it is negative for the current headline. At 4-bit, `fokvq` does not beat `kivi`, does not beat a random orthogonal basis, and does not become more compelling at the tested longer contexts. The surviving claim, if any, is not 4-bit superiority on this benchmark slice.

### Iteration 9

- Change: ran a new 2-bit follow-up batch under the original fair `topk_frac=0.25` schedule, plus one low-bit axis ablation, while capping CPU math-library threads at `8` per process to remove the setup-time oversubscription observed in Iteration 8.
  - launch environment:
    - `OMP_NUM_THREADS=8`
    - `MKL_NUM_THREADS=8`
    - `OPENBLAS_NUM_THREADS=8`
    - `NUMEXPR_NUM_THREADS=8`
- Verification:
  - all 4 remote runs finished cleanly and wrote JSON outputs:
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/main/gpt2-medium-2bit-topk025-main_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/axis_ablation/gpt2-medium-2bit-topk025-axis-ablation_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/longctx512/gpt2-medium-2bit-topk025-longctx512_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/longctx1024/gpt2-medium-2bit-topk025-longctx1024_standard_ppl.json`
  - the thread cap fixed the operational bottleneck:
    - Iteration 8 wall-clock runtime per job: about `543-609s`
    - this batch wall-clock runtime per job: about `6.16-12.22s`
    - per-method in-JSON runtimes stayed consistent, so the large wall-clock reduction is attributable to less CPU-thread contention rather than a protocol change
  - main 2-bit table (`context_len=256`, `stride=128`, `max_eval_tokens=4096`):
    - `fp16=26.8638`
    - `uniform-2=32.4926`
    - `kivi-2=27.7507`
    - `turboquant-style-2=27.7524`
    - `fokvq(topk=0.25)-2=48.2813`
  - 2-bit axis ablation (`context_len=256`, `stride=128`):
    - `identity-2=412.4942`
    - `random-2=84.4536`
    - `fokvq(PCA, topk=0.25)-2=48.2813`
  - long-context `512/256`:
    - `fp16=22.5366`
    - `uniform-2=27.3654`
    - `kivi-2=23.9904`
    - `turboquant-style-2=23.3546`
    - `fokvq(topk=0.25)-2=45.9687`
  - long-context `1024/512`:
    - `fp16=18.3817`
    - `uniform-2=22.4624`
    - `kivi-2=20.4618`
    - `turboquant-style-2=19.3868`
    - `fokvq(topk=0.25)-2=42.2597`
- Decision: keep
- Reason: this batch rules out an important rescue hypothesis. Restoring the original `25%` principal-axis split under a fair average-bit budget does not help `fokvq`; it remains far worse than `kivi` and `turboquant-style`, and the gap persists even as context length increases. The operational thread-cap change should also be kept for future remote runs because it reduces wall-clock time by roughly two orders of magnitude without changing the benchmark protocol.

### Iteration 10

- Change: ran the matching full-protocol `2-bit + topk_frac=0.5` batch, including a low-bit axis ablation, to establish the best current fair `fokvq` configuration on the same `4096-token` benchmark slices used above.
- Verification:
  - all 4 remote runs finished cleanly and wrote JSON outputs:
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/main/gpt2-medium-2bit-topk050-main_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/axis_ablation/gpt2-medium-2bit-topk050-axis-ablation_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/longctx512/gpt2-medium-2bit-topk050-longctx512_standard_ppl.json`
    - `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/longctx1024/gpt2-medium-2bit-topk050-longctx1024_standard_ppl.json`
  - main 2-bit table (`context_len=256`, `stride=128`, `max_eval_tokens=4096`):
    - `fp16=26.8638`
    - `uniform-2=32.4926`
    - `kivi-2=27.7507`
    - `turboquant-style-2=27.7524`
    - `fokvq(topk=0.5)-2=34.5466`
  - axis ablation (`context_len=256`, `stride=128`):
    - `identity-2=149.8334`
    - `random-2=52.9955`
    - `fokvq(PCA, topk=0.5)-2=34.5466`
  - long-context `512/256`:
    - `fp16=22.5366`
    - `uniform-2=27.3654`
    - `kivi-2=23.9904`
    - `turboquant-style-2=23.3546`
    - `fokvq(topk=0.5)-2=28.8335`
  - long-context `1024/512`:
    - `fp16=18.3817`
    - `uniform-2=22.4624`
    - `kivi-2=20.4618`
    - `turboquant-style-2=19.3868`
    - `fokvq(topk=0.5)-2=24.0110`
  - direct comparison against Iteration 9 (`topk=0.25`):
    - main: `48.28 -> 34.55`
    - `512`: `45.97 -> 28.83`
    - `1024`: `42.26 -> 24.01`
  - interpretation from the ablation:
    - PCA alignment is materially better than naive basis choices at 2-bit
    - but even the best current fair PCA setting still trails `uniform`, `kivi`, and `turboquant-style` on every tested slice
- Decision: keep
- Reason: this batch establishes the strongest defensible internal readout so far. There is a real axis-selection effect at 2-bit because `fokvq(PCA)` clearly beats `random` and `identity`, but the effect is not strong enough to support a superiority claim against stronger baselines. The most honest paper direction is therefore mechanistic or methodological, not SOTA-style.

### Iteration 11

- Change: fixed a regression in the new variant harness before launching the follow-up wave.
  - bug: `evaluate_quantized_sliding_window()` consumed `cache_stats` but did not initialize the local `key_mse_sum` / `key_mse_count` accumulators
  - auxiliary change: added `PYTHON_BIN` override support to `scripts/fokvq/launch_exp4_2_variant_wave.sh` so the remote node can force the correct `grpo` interpreter instead of the default `ptca` environment
- Verification:
  - local `py_compile` passed after the fix
  - remote `--self-test` on `metacognition-e8` passed again:
    - `orthogonality_max_abs_error=4.77e-07`
    - `lloyd_mse=0.1148`
    - `uniform_mse=0.6005`
    - `turboquant_mse`: `2-bit=0.1196`, `4-bit=0.0097`
  - `tiny-gpt2` smoke was discarded as the final verification vehicle because `uniform-2` became non-finite on that toy model
  - accepted verification vehicle:
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_gpt2_variant_smoke/gpt2-variants-smoke_standard_ppl.json`
    - model: `openai-community/gpt2`
    - protocol slice: `context_len=256`, `stride=128`, `max_eval_tokens=384`
    - methods: `fp16 uniform kivi turboquant fokvq fokvq_lloyd fokvq_adaptive fokvq_clip`
    - result: all methods and both `2/4-bit` settings finished and wrote JSON successfully
- Decision: keep
- Reason: this was a real correctness bug in the newly extended harness. Fixing it was necessary before any full-wave conclusion about the new variants could be trusted.

### Iteration 12

- Change: ran the missing full-protocol `2-bit` variant wave on `metacognition-e8` with the strengthened same-harness comparison set:
  - main / long-context methods:
    - `fp16 uniform kivi turboquant fokvq fokvq_lloyd fokvq_adaptive fokvq_clip`
  - axis ablation methods:
    - `fp16 identity random fokvq fokvq_lloyd fokvq_adaptive fokvq_clip`
  - launch wrapper:
    - `scripts/fokvq/launch_exp4_2_variant_wave.sh`
  - output root:
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_2bit_v2`
- Verification:
  - all 4 runs finished cleanly and wrote JSON outputs:
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_2bit_v2/main/gpt2-medium-2bit_standard_ppl.json`
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_2bit_v2/axis_ablation/gpt2-medium-2bit_standard_ppl.json`
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_2bit_v2/longctx512/gpt2-medium-2bit_standard_ppl.json`
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_2bit_v2/longctx1024/gpt2-medium-2bit_standard_ppl.json`
  - node returned to idle after completion
  - main table (`context_len=256`, `stride=128`):
    - `fp16=26.8638`
    - `uniform-2=32.4926`
    - `kivi-2=27.7507`
    - `turboquant-style-2=27.7524`
    - `fokvq-2=33.6741`
    - `fokvq_lloyd-2=34.3090`
    - `fokvq_adaptive-2=35.2711`
    - `fokvq_clip-2=34.4791`
  - long-context `512/256`:
    - `uniform-2=27.3654`
    - `kivi-2=23.9904`
    - `turboquant-style-2=23.3546`
    - `fokvq-2=28.8335`
    - `fokvq_lloyd-2=27.2620`
    - `fokvq_adaptive-2=33.9548`
    - `fokvq_clip-2=29.0065`
  - long-context `1024/512`:
    - `uniform-2=22.4624`
    - `kivi-2=20.4618`
    - `turboquant-style-2=19.3868`
    - `fokvq-2=24.0110`
    - `fokvq_lloyd-2=28.3483`
    - `fokvq_adaptive-2=37.1294`
    - `fokvq_clip-2=23.6791`
  - axis ablation:
    - `identity-2=149.8334`
    - `random-2=52.9955`
    - `fokvq-2=33.6741`
    - `fokvq_lloyd-2=34.3090`
    - `fokvq_adaptive-2=35.2711`
    - `fokvq_clip-2=34.4791`
  - important proxy-vs-task observation from the same JSONs:
    - `fokvq_lloyd` consistently reduced `avg_key_mse` versus base `fokvq`
    - but that lower reconstruction error did not translate into better `PPL` on the main slice and often regressed it
- Decision:
  - `fokvq_adaptive`: discard as a promising direction under the current design
  - `fokvq_lloyd`: keep only as a mechanistic ablation, not as a candidate improvement
  - `fokvq_clip`: keep as a weak ablation because it gave small wins on some longer-context slices, but not as a headline method
- Reason: the 2-bit wave sharpens the main mechanistic conclusion. Better scalar reconstruction in the rotated space is not sufficient by itself. The one partial positive sign is that `fokvq_lloyd` helped at `longctx512`, and `fokvq_clip` slightly helped at `longctx1024`, but neither closes the gap to `kivi` or `turboquant-style`, and neither is stable enough to claim as the new main method.

### Iteration 13

- Change: ran the matching full-protocol `4-bit` variant wave on the same 4 slices and the same strengthened comparison set.
  - output root:
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_4bit_v2`
- Verification:
  - all 4 runs finished cleanly and wrote JSON outputs:
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_4bit_v2/main/gpt2-medium-4bit_standard_ppl.json`
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_4bit_v2/axis_ablation/gpt2-medium-4bit_standard_ppl.json`
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_4bit_v2/longctx512/gpt2-medium-4bit_standard_ppl.json`
    - `/scratch/boltzmann-attention-run/outputs/exp4_2_variants_4bit_v2/longctx1024/gpt2-medium-4bit_standard_ppl.json`
  - main table (`context_len=256`, `stride=128`):
    - `fp16=26.8638`
    - `kivi-4=26.8770`
    - `uniform-4=26.8983`
    - `fokvq-4=26.9954`
    - `turboquant-style-4=27.0020`
    - `fokvq_adaptive-4=27.0465`
    - `fokvq_clip-4=27.3888`
    - `fokvq_lloyd-4=29.3253`
  - long-context `512/256`:
    - `kivi-4=22.5641`
    - `uniform-4=22.5972`
    - `fokvq_adaptive-4=22.6027`
    - `fokvq-4=22.6055`
    - `turboquant-style-4=22.6359`
    - `fokvq_clip-4=22.9952`
    - `fokvq_lloyd-4=24.0991`
  - long-context `1024/512`:
    - `kivi-4=18.3727`
    - `uniform-4=18.4042`
    - `fokvq-4=18.4042`
    - `fokvq_adaptive-4=18.4086`
    - `turboquant-style-4=18.4266`
    - `fokvq_clip-4=18.4221`
    - `fokvq_lloyd-4=23.4947`
  - axis ablation:
    - `identity-4=27.0862`
    - `random-4=26.9509`
    - `fokvq-4=26.9954`
    - `fokvq_adaptive-4=27.0465`
    - `fokvq_clip-4=27.3888`
    - `fokvq_lloyd-4=29.3253`
- Decision:
  - `fokvq_lloyd`: discard as an improvement candidate at 4-bit
  - `fokvq_adaptive`: discard as an improvement candidate at 4-bit
  - `fokvq_clip`: discard as an improvement candidate at 4-bit
- Reason: the 4-bit wave is even more clearly negative than the 2-bit wave. Once the bit budget is high enough that the main baselines are already close, none of the new `fokvq` variants improves the ranking, and `fokvq_lloyd` becomes dramatically worse despite lower-level reconstruction intuition. This reinforces the claim that the bottleneck is not simply the scalar quantizer in the rotated coordinates.
