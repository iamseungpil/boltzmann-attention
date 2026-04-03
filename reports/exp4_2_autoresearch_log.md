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

### Iteration 14

- Change: validated and redeployed the `v3` full-quant harness on `tops-caiman` with the patched attention wrappers that now call the official eager-attention implementations for both GPT-2 and Qwen2.
  - code path:
    - `/home/v-seungplee/boltzmann-attention-develop/scripts/exp4_2_v3_full_quant_ppl.py`
  - remote sync target:
    - `/scratch/boltzmann-attention-v3-repro`
- Verification:
  - `--self-test` passed after the wrapper fixes and the Q-covariance centering fix.
  - smoke runs on `tops-caiman` produced sane same-harness full-quant readouts:
    - `GPT-2 medium`, `2048` tokens:
      - `fp16=18.1481`
      - `uniform-2=88.7298`
      - `kivi-2=136.0648`
      - `kivi-3=23.4826`
      - `kivi-4=18.5064`
    - `Qwen2.5-7B`, `2048` tokens:
      - `fp16=6.0769`
      - `uniform-2=25825.2416`
      - `uniform-3=19069.2664`
      - `uniform-4=10616.5824`
      - `kivi-2=134.9345`
      - `kivi-3=7.3729`
      - `kivi-4=6.4674`
- Decision: keep
- Reason: this establishes that the repaired `v3` harness is now directionally believable. In particular, `Qwen + kivi` no longer explodes to absurd values at 3/4-bit, while `uniform` remains clearly broken on the harder GQA/RoPE setting. That is exactly the baseline-separation pattern we needed before extending the method set.

### Iteration 15

- Change: added a new Lie-group-inspired same-harness method, `lie_eq`, to `v3`.
  - mechanism:
    - greedy Givens rotations inside `SO(d)` that equalize per-axis variance
    - followed by per-axis Lloyd-Max scalar quantization
  - implementation:
    - `_greedy_variance_equalizing_rotation`
    - `lie_eq_quantize_head`
    - dispatch path `method == "lie_eq"`
- Verification:
  - local self-test on the edited script passed.
  - synthetic anisotropic check:
    - variance spread reduced from `50.20` to `1.05`
    - orthogonality error stayed below `1e-4`
    - 4-bit reconstruction error was lower than 2-bit reconstruction error
- Decision: keep
- Reason: this is a defensible next-step generalization beyond PCA that stays compatible with both MHA and GQA. It does not depend on query-head pooling, and it gives a concrete `SO(d)` comparison point inside the same evaluation harness rather than only in the older offline MSE experiments.

### Iteration 16

- Change: restructured the running experiment queue to reduce iteration cost.
  - killed the slow `lie_eq` smoke jobs that still included `turboquant`
  - relaunched fast `lie_eq` comparison runs without `turboquant`
  - killed the very slow `GPT-2 full` run once it hit the `turboquant` bottleneck and relaunched a `full-fast` run with:
    - `fp16 uniform kivi fokvq fokvq_e2 lie_eq`
  - kept `Qwen full` with `turboquant` alive because it had already yielded a useful `2-bit` reference point
- Verification:
  - partial fast-smoke readouts now available:
    - `GPT-2 medium`, `2048` tokens:
      - `fp16=18.1481`
      - `kivi-2=136.0648`
      - `kivi-3=23.4826`
      - `kivi-4=18.5064`
      - `fokvq-2=21.6894`
      - `fokvq-3=18.3762`

### Iteration 17

- Change: aligned the `v3` harness with the revised paper hypotheses instead of leaving benchmark intent implicit.
  - added `--benchmark-preset` with:
    - `ppl_quality`
    - `rotation_mechanistic`
    - `retrieval_depth`
    - `paper_full`
  - each preset now writes:
    - `intent`
    - `hypothesis`
    - `verification_focus`
    into the JSON summary
  - added alignment guards so that, for example:
    - `rotation_mechanistic` requires `identity/random/fokvq`
    - NIAH depth tests require at least 3 depth points
- Verification:
  - local `--self-test` passed after the preset addition
  - new self-test coverage added for preset metadata and alignment guards
- Decision: keep
- Reason: this was a paper-alignment correction rather than a metric tweak. It reduces the chance of running a benchmark that cannot answer the intended hypothesis.

### Iteration 18

- Change: fixed a real harness mismatch discovered by the new mechanistic smoke.
  - issue:
    - `rotation_mechanistic` preset requested `identity` and `random`
    - but `exp4_2_v3_full_quant_ppl.py` did not dispatch those methods in `quantize_k_tensor`
  - fix:
    - added `identity_basis_quantize_head`
    - added `random_basis_quantize_head`
    - wired both through the `v3` dispatch path
    - expanded self-test coverage so `identity` and `random` are exercised in `2D/3D/4D`
  - auxiliary operational fix:
    - launcher defaults now point to the actual torch-enabled interpreter instead of bare `python3`
- Verification:
  - `--self-test` passed after the dispatch fix
  - local smoke:
    - model: `sshleifer/tiny-gpt2`
    - preset: `rotation_mechanistic`
    - result: all requested methods (`identity`, `random`, `fokvq`, `kivi_residual`, `turboquant_rand`, complex variants) completed without unknown-method errors and wrote JSON successfully
- Decision: keep
- Reason: this was a true correctness bug, not a performance regression. The new paper-aligned benchmark could not even run before the fix, so no further autoresearch wave would have been defensible.
    - `Qwen2.5-7B`, `2048` tokens:
      - `fp16=6.0769`
      - `kivi-2=134.9345`
      - `kivi-3=7.3729`
      - `kivi-4=6.4674`
      - `fokvq-2=76.1748`
      - `fokvq-3=9.7439`
  - partial long run readouts:
    - `GPT-2 medium`, `16384` tokens:
      - `fp16=22.0033`
      - `uniform-2=108.1708`
      - `kivi-2=147.8516`
      - `kivi-3=28.0145`
      - `kivi-4=22.6052`
    - `Qwen2.5-7B`, `16384` tokens:
      - `fp16=6.7544`
      - `uniform-2=13945.5075`
      - `kivi-2=105.6188`
      - `kivi-3=8.0162`
      - `kivi-4=7.0309`
      - `turboquant-2=11.4734`
- Decision: keep
- Reason: this iteration materially improved experiment throughput without changing the scientific question. It also sharpened the emerging interpretation:
  - on `GPT-2/MHA`, `fokvq` remains clearly more promising than `kivi`
  - on `Qwen/GQA`, `kivi` remains stronger at `3/4-bit`, while `fokvq` is competitive only at the harder `2-bit` setting
  - `turboquant` is scientifically relevant on `Qwen`, but its current implementation cost is too high to leave on every branch of the queue

### Iteration 17

- Change:
  - reviewed the current `docx` corpus again before continuing the Lie-algebra branch:
    - `phase4_1_ppl_results.docx`
    - `FOKVQ_ADDITIONAL_EXPERIMENT_PLAN*.docx`
    - `FOKVQ_PAPER_v3.docx`, `FOKVQ_PAPER_v6.docx`
    - `FOKVQ_FINAL_REPORT.docx`
    - `DHS_PAPER_v1.docx`, `DHS_PAPER_v2.docx`, `HEAT_PAPER_v2.docx`
  - added explicit self-tests for:
    - `lie_eq_robust`
    - `lie_qdiag`
    - `lie_qdiag_robust`
    - `quantize_k_tensor` dispatch coverage for the new Lie variants
- Verification:
  - local `--self-test` passed after the additions
  - new coverage now verifies:
    - finiteness of all new Lie branches
    - monotonicity for `lie_eq_robust`
    - non-degenerate query sensitivity for `lie_qdiag`
  - document review outcome:
    - `phase4_1_ppl_results.docx` is now treated as preliminary because it predates the current `v3` full-quant harness
    - `FOKVQ_ADDITIONAL_EXPERIMENT_PLAN_v9.docx` still correctly treats `Exp 4-2` as the decisive reviewer-grade benchmark
    - the DHS/HEAT documents use quantization as a practical implication, not as a settled empirical fact, so they should inherit the new `v3` readout rather than the older optimistic table
- Decision: keep
- Reason: this iteration tightened the verification boundary and aligned the active experiment queue with the most defensible document interpretation.

### Iteration 18

- Change:
  - synchronized the updated `scripts/exp4_2_v3_full_quant_ppl.py` to `tops-caiman`
  - queued the next bounded autoresearch run on the remote node:
    - target: `Qwen2.5-7B`, `2048-token` smoke
    - methods: `fp16 kivi fokvq fokvq_e2 lie_eq lie_eq_robust`
    - bits: `3 4`
    - queue worker: `/tmp/qwen_lie_eq_robust_fast.sh`
- Verification:
  - remote script sync completed successfully
  - remote queue worker is alive on `tops-caiman`:
    - launcher shell PID observed: `547780`
    - log path prepared: `/scratch/boltzmann-attention-v3-repro/logs/qwen_lie_eq_robust_fast.queue.log`
  - current remote status at queue time:
    - all 4 GPUs still occupied by earlier runs
    - `Qwen` fast-smoke partials confirm the current ranking:
      - `kivi-3=7.3729`
      - `fokvq-3=9.7439`
      - `fokvq_e2-3=7.7454`
      - `lie_eq-2=40.9443`
      - `lie_eq-3` already looks clearly weak from partial output
- Decision: keep
- Reason: no GPU was idle immediately, so the most pragmatic next step was to queue the best next hypothesis (`lie_eq_robust`) without interrupting already-running evidence collection.

### Iteration 19

- Change:
  - investigated the new `Qwen` collapse that appeared after the recent `v3` refactor.
  - replaced the brittle custom `Qwen2Attention.forward` rewrite with a wrapper around the official
    `transformers.models.qwen2.modeling_qwen2.eager_attention_forward`.
  - kept quantization confined to the `key` tensor while delegating the actual attention math back to the upstream eager path.
- Verification:
  - decisive remote no-op check on `tops-caiman`:
    - `fp16=6.0769`
    - `kivi-16bit=6.0769`
    - `kivi-4bit=6.4674`
  - interpretation:
    - before this change, even `16-bit` quantized runs could diverge catastrophically from `fp16`, which proved the regression was in the patched forward path rather than in the quantizer.
    - after the wrapper fix, `16-bit == fp16` holds on the actual remote environment and `4-bit` returns the previously sane `kivi` readout.
- Decision: keep
- Reason: this is the core correctness fix. Without forward-equivalence at `16-bit`, none of the Qwen quantization numbers are scientifically interpretable.

### Iteration 20

- Change:
  - redeployed the repaired harness to `tops-caiman`.
  - launched a 4-GPU bounded autoresearch wave to separate:
    - smoke validation of the repaired base methods
    - smoke validation of query-safe Lie variants
    - full `16k` main-table rerun
    - full `16k` Lie-table rerun
- Running queue on `tops-caiman`:
  - `GPU0`: `qwen_regfix_smoke_base`
    - methods: `fp16 kivi fokvq fokvq_e2 lie_eq lie_eq_robust`
    - bits: `3 4`
    - log: `/scratch/boltzmann-attention-v3-repro/logs/qwen_regfix_smoke_base.log`
  - `GPU1`: `qwen_regfix_smoke_qdiag`
    - methods: `fp16 kivi fokvq_e2 lie_qdiag lie_qdiag_robust`
    - bits: `3 4`
    - log: `/scratch/boltzmann-attention-v3-repro/logs/qwen_regfix_smoke_qdiag.log`
  - `GPU2`: `qwen_full_main_regfix_16k`
    - methods: `fp16 kivi fokvq fokvq_e2`
    - bits: `3 4`
    - log: `/scratch/boltzmann-attention-v3-repro/logs/qwen_full_main_regfix_16k.log`
  - `GPU3`: `qwen_full_lie_regfix_16k`
    - methods: `fp16 kivi fokvq_e2 lie_eq lie_eq_robust`
    - bits: `3 4`
    - log: `/scratch/boltzmann-attention-v3-repro/logs/qwen_full_lie_regfix_16k.log`
- Verification:
  - remote process table shows all four jobs alive on the expected GPUs.
  - current smoke partials are sane rather than collapsed:
    - `fp16=6.0769`
    - `kivi-3=7.3729`
    - `kivi-4=6.4674`
    - `fokvq-3=9.7439`
    - `fokvq-4` has entered the expected slow path rather than exploding
  - current long-run partials are also sane:
    - `fp16(16k)=6.7544`
    - `kivi-3(16k)=8.0162`
    - `kivi-4(16k)=7.0309`
    - `fokvq-3(16k)=9.6555`
- Decision: keep running
- Reason: the repaired harness now exhibits consistent, finite Qwen behavior under full-quant evaluation, and the 4-GPU queue is collecting exactly the evidence needed for the next prune-or-keep decision.

### Iteration 21

- Change:
  - audited the public-code gap before running the next Qwen wave.
  - reviewed the official `KIVI` repository directly and confirmed that our old same-harness `kivi` baseline was only a proxy:
    - official KIVI uses grouped quantization,
    - keeps a recent `residual_length` tail in full precision,
    - and jointly treats K/V caches rather than K only.
  - also rechecked the remote `transformers` environment:
    - `transformers==4.52.3`
    - Qwen2 uses the eager-attention signature we wrapped rather than the older brittle path.
- Verification:
  - public KIVI repo inspection showed:
    - `group_size`
    - `residual_length`
    - repeated quant/full cache handling in the official model code
  - remote environment check showed the expected eager signatures for both `qwen2` and `gpt2`.
- Decision: keep
- Reason: this separates two questions that were previously conflated:
  - harness correctness
  - baseline faithfulness
  The Qwen harness was already repaired, but the KIVI baseline still needed a more public-faithful proxy.

### Iteration 22

- Change:
  - implemented two comparison-focused improvements in `v3`:
    - `kivi_residual`: a more public-KIVI-inspired K-only proxy that:
      - quantizes old tokens in sequence groups
      - leaves the most recent `residual_length=32` tokens in full precision
    - `turboquant_rand`: a random-orthogonal-rotation variant closer to the published TurboQuant description than the older deterministic Hadamard proxy
  - extended self-tests to cover:
    - `kivi_residual`
    - `turboquant_rand`
    - updated dispatch paths
  - added an explicit runtime note in the harness output that KIVI/TurboQuant entries are same-harness proxies unless labeled otherwise.
- Verification:
  - local syntax check:
    - `python3 -m py_compile scripts/exp4_2_v3_full_quant_ppl.py`
  - remote self-test passed after sync:
    - `kivi_residual` residual-tail preservation check passed
    - all prior FOKVQ / E2 / Lie tests still passed
- Decision: keep
- Reason: this is the minimum defensible improvement needed before interpreting the next wave of control comparisons.

### Iteration 23

- Change:
  - stopped the mixed long/smoke queue on `tops-caiman`.
  - relaunched a smoke-only 4-GPU wave organized by scientific role:
    - `GPU0`: Qwen core comparison
      - `fp16`, `kivi`, `kivi_residual`, `fokvq`, `fokvq_e2`
    - `GPU1`: Qwen Lie comparison
      - initially `fp16`, `fokvq_e2`, `lie_eq`, `lie_eq_robust`, `lie_qdiag`, `lie_qdiag_robust`
    - `GPU2`: Qwen control comparison
      - initially `fp16`, `uniform`, `quip`, `kvquant`, `turboquant`, `turboquant_rand`
    - `GPU3`: GPT-2 cross-architecture comparison
      - `fp16`, `kivi`, `kivi_residual`, `fokvq`, `fokvq_e2`, `lie_eq`, `lie_qdiag`, `turboquant`, `turboquant_rand`
- Verification:
  - remote self-test was clean before relaunch
  - node reset was confirmed:
    - all 4 GPUs were free before the smoke wave
  - one launch duplication bug briefly created duplicate processes; the older duplicates were explicitly killed so that only one process per GPU remained
- Decision: keep
- Reason: this restores the intended autoresearch structure:
  - smoke first
  - critic on smoke output
  - only then promote survivors to `16k`

### Iteration 24

- Change:
  - ran the first smoke-critic pass and then improved the queue ordering:
    - `GPU1`: restarted `qwen_smoke_lie` with `Lie` methods before `fokvq_e2`
    - `GPU2`: restarted `qwen_smoke_ctrl` with `turboquant` and `turboquant_rand` before slower controls
- Verification:
  - early Qwen core smoke already changed the control interpretation materially:
    - `fp16=6.0769`
    - old `kivi-3=7.3729`, `kivi-4=6.4674`
    - new `kivi_residual-3=6.5734`, `kivi_residual-4=6.1431`
    - `fokvq-3=9.7439`
    - `fokvq-4=6.9743`
  - early Qwen control smoke before reorder:
    - `quip-3=8.3246`
    - `quip-4=6.2592`
    - `kvquant-3=6.78` (partial, already slower than `kivi_residual-3`)
  - early GPT-2 cross-architecture smoke:
    - `fp16=18.1481`
    - old `kivi-2=107.6853`, `kivi-3=22.4750`, `kivi-4=18.9404`
    - new `kivi_residual-2=20.2778`, `kivi_residual-3=18.9892`, `kivi_residual-4=18.6668`
    - `fokvq-2=22.9927`, `fokvq-3=18.8355`, `fokvq-4=18.6213`
- Decision: keep and continue
- Reason: the public-gap fix was not cosmetic; it substantially changed the baseline ranking. The old same-harness `kivi` was too weak. With `kivi_residual`, the Qwen 3/4-bit battleground becomes materially harder, and that is the correct comparison to carry into the next wave.

### Iteration 25

- Change:
  - promoted the validated winners from smoke to `16k` `Qwen` full-quant runs while keeping unresolved smoke questions on the remaining GPUs.
  - active split:
    - `GPU0`: `qwen-full-main-v2`
      - `fp16`, `kivi_residual`, `fokvq`, `fokvq_e2`
    - `GPU3`: `qwen-full-ctrl-v2`
      - `fp16`, `kivi_residual`, `quip`, `turboquant`, `turboquant_rand`
    - `GPU1`: `qwen-smoke-qdiag-fast`
      - `fp16`, `lie_qdiag`, `lie_qdiag_robust`, `fokvq_e2`, `kivi_residual`
    - `GPU2`: `qwen-smoke-turbo-fast`
      - `fp16`, `turboquant`, `turboquant_rand`, `quip`, `kivi_residual`
- Verification:
  - `Qwen 16k` main partial:
    - `fp16=6.7544`
    - `kivi_residual-3=7.2060`
    - `kivi_residual-4=6.7738`
    - `fokvq-3=9.6555`
  - `Qwen 16k` control partial:
    - `fp16=6.7544`
    - `kivi_residual-3=7.2060`
    - `kivi_residual-4=6.7738`
    - `quip-3=9.1795`
  - smoke evidence carried into the promotion:
    - `Qwen 2048`: `kivi_residual-3=6.5734`, `kivi_residual-4=6.1431`
    - `Qwen 2048`: `fokvq-3=9.7439`, `fokvq-4=6.9743`
    - `Qwen 2048`: `turboquant-3=6.96` (older reordered smoke)
    - `Qwen 2048`: `lie_eq-3=23.42`
- Decision: keep
- Reason: one main hypothesis is now effectively settled:
  - `kivi_residual` is the strongest current same-harness practical control on `Qwen`
  - plain `fokvq` and `lie_eq` do not challenge it at `3-bit`
  The remaining useful work is narrowed to `fokvq_e2`, `turboquant_rand`, and `lie_qdiag`.

### Iteration 26

- Change:
  - implemented the first `RoPE-pair-preserving` candidate:
    - `rope_unitary`
    - blockwise `SO(2)` rotations applied independently to each RoPE pair
  - extended self-tests with:
    - pairwise commuting check against a RoPE block
    - finite / monotone reconstruction check
- Verification:
  - remote `--self-test` passed after sync
  - `Qwen 2048` smoke:
    - `fp16=6.0769`
    - `kivi_residual=6.5734 / 6.1431`
    - `rope_unitary=14.3512 / 13.9959`
  - runtime was also poor:
    - `324.1s / 591.5s` for the single-chunk `3 / 4-bit` smoke
- Decision: discard as a practical candidate
- Reason: the RoPE-pair restriction alone does not recover good geometry on
  `Qwen`; it is dramatically worse than both `kivi_residual` and `fokvq_e2`.

### Iteration 27

- Change:
  - closed the outstanding `Qwen 16k` runs and added the first residual-tail
    structured hybrid:
    - `fokvq_e2_residual`
    - recent `32` tokens remain FP16
    - older prefix uses `fokvq_e2`
  - kept the same full-quant harness and same-harness control set
- Verification:
  - completed `Qwen 16k` control results:
    - `fp16=6.7544`
    - `kivi_residual=7.2060 / 6.7738`
    - `quip=9.1795 / 7.1394`
    - `turboquant=7.5671 / 7.0900`
    - `turboquant_rand=7.5587 / 7.0212`
  - completed `Qwen 16k` main results:
    - `fokvq=9.6555 / 7.4720`
    - `fokvq_e2=8.7232 / 8.4917`
  - `Qwen 2048` smoke for the new hybrid:
    - `kivi_residual=6.5734 / 6.1431`
    - `fokvq_e2=7.7454 / 7.3983`
    - `fokvq_e2_residual=7.4612 / 7.4540`
- Decision:
  - keep `turboquant_rand` as the stronger TurboQuant-style control
  - keep `fokvq_e2_residual` only as a mechanistic hybrid
  - discard `fokvq_e2_residual` as a practical winner
- Reason:
  - `fokvq_e2_residual` improves over plain `fokvq_e2` at `3-bit`
  - but it regresses at `4-bit` and still remains far behind `kivi_residual`
  - `kivi_residual` remains the practical leader in the current same-harness queue

### Iteration 28

- Change:
  - started the next `RoPE-phase` hypothesis:
    - `rope_magphase`
    - treat each RoPE pair as `z = r e^{i phi}`
    - quantize `log(1+r)` with Lloyd-Max
    - quantize phase with uniform circular bins and a phase-priority bit split
  - added dispatch and self-test coverage for the new path
- Verification:
  - remote `--self-test` passed after sync
  - launched `Qwen 2048` smoke on `tops-caiman`:
    - `fp16`, `kivi_residual`, `fokvq_e2_residual`, `rope_magphase`
  - initial run state:
    - process active on `GPU0`
    - baseline section already matches the expected `Qwen` reference values
- Decision: in progress
- Reason: this is the next clean test of the document-driven claim that RoPE
  phase structure, not generic Euclidean rotation, is the relevant geometry.

### Iteration 29

- Change:
  - completed the `rope_magphase` smoke
  - compared it directly against `kivi_residual` and `fokvq_e2_residual`
- Verification:
  - `Qwen 2048` smoke:
    - `fp16=6.0769`
    - `kivi_residual=6.5734 / 6.1431`
    - `fokvq_e2_residual=7.4612 / 7.4540`
    - `rope_magphase=750.0469 / 424.5279`
- Decision: discard
- Reason:
  - the method is finite and the harness is correct, but the candidate is
    scientifically noncompetitive and practically unusable
  - naive phase-priority quantization destroys too much structure on `Qwen`

### Iteration 30

- Change:
  - started the next complex-Lie candidate:
    - `complex_unitary_residual`
    - treat each RoPE pair as a complex channel
    - estimate Hermitian covariance on the old prefix
    - rotate by a unitary eigenbasis
    - quantize real/imag parts in that complex basis
    - keep the recent tail in FP16
  - added self-test coverage for:
    - finiteness
    - residual-tail preservation
    - higher-bit monotonicity on the quantized prefix
- Verification:
  - local syntax check passed
  - remote self-test still pending at the time of writing this entry
- Decision: in progress
- Reason:
  - this is the first candidate that matches the intended "complex rotation Lie
    group" interpretation more faithfully than the failed pair-local or direct
    phase-quantization variants

### Iteration 31

- Change:
  - re-ran the current local correctness gate on the revised harness
  - verified both:
    - full `--self-test`
    - a fresh `rotation_mechanistic` smoke on `sshleifer/tiny-gpt2`
- Verification:
  - local `--self-test` passed completely on the current file state
  - smoke JSON now records:
    - `benchmark_meta`
    - `intent`
    - `hypothesis`
    - `verification_focus`
  - all requested mechanistic methods completed without unknown-method errors:
    - `identity`
    - `random`
    - `fokvq`
    - `kivi_residual`
    - `turboquant_rand`
    - `complex_unitary_residual`
    - `banded_complex_unitary_residual`
- Decision: keep
- Reason:
  - this does not add scientific evidence by itself because `tiny-gpt2` is only a wiring smoke
  - but it confirms that the current harness, dispatch path, and benchmark metadata all agree after the recent fixes

### Iteration 32

- Change:
  - fixed launcher drift and result provenance drift
  - launcher fixes:
    - moved outputs to repo-level `results/v3`
    - moved logs to repo-level `reports/`
    - switched Qwen practical launchers to `ppl_quality`
    - upgraded controls from older `kivi` to `kivi_residual`
    - added a dedicated `run_v3_qwen_rotation.sh` for the mechanistic benchmark
  - harness metadata fix:
    - output JSON now also records timestamp, hostname, cwd, and git head
- Verification:
  - launcher paths now align with the report's artifact conventions
  - the current git head can be recovered directly from future result JSONs
- Decision: keep
- Reason:
  - this is a reproducibility correction
  - it reduces the risk that later reports accidentally mix stale failed artifacts with corrected benchmark-aligned runs

### Iteration 33

- Change:
  - added an early alignment guard for bounded PPL smoke runs:
    - fail fast when `max_eval_tokens < context_len`
- Verification:
  - discovered during remote `Qwen` smoke that the old behavior only failed after model load
  - added self-test coverage for this misconfiguration path
- Decision: keep
- Reason:
  - this is a real smoke-critic fix
  - it turns a late expensive failure into an early configuration error

### Iteration 34

- Change:
  - bootstrapped a fresh torch-enabled remote environment on `tops-caiman`
  - installed:
    - `torch 2.6.0+cu124`
    - `transformers 5.3.0`
    - `datasets 4.8.2`
    - `accelerate 1.13.0`
  - re-synced the current harness after the local guard fix
- Verification:
  - remote GPUs were idle before launch
  - remote `torch.cuda.is_available()` returned `True`
  - remote `--self-test` passed completely
- Decision: keep
- Reason:
  - this removes the main operational blocker between local harness validation and real remote smoke runs

### Iteration 35

- Change:
  - launched a bounded remote `Qwen` `rotation_mechanistic` smoke:
    - `context_len=256`
    - `max_eval_tokens=512`
    - methods:
      - `fp16`
      - `identity`
      - `random`
      - `fokvq`
      - `fokvq_e2`
      - `kivi_residual`
      - `turboquant_rand`
      - `complex_unitary_residual`
      - `banded_complex_unitary_residual`
- Verification:
  - early partial results:
    - `fp16=11.8666`
    - `identity-2=19.1351`
    - `identity-3=12.6156`
    - `random-2=26.2928`
    - `random-3=12.6578`
    - `fokvq-2=15.1640`
  - live GPU status while running:
    - `GPU0` active
    - `GPU1-3` idle
- Decision: in progress
- Reason:
  - the first partial result already suggests the mechanistic story survives on bounded `Qwen` smoke:
    - `fokvq-2` is materially better than both agnostic controls seen so far
  - full bounded smoke must finish before deciding the 4-GPU follow-up wave

### Iteration 36

- Change:
  - completed the bounded remote `Qwen` `rotation_mechanistic` smoke and
    recorded the full method table
- Verification:
  - final bounded smoke results:
    - `FP16=11.8666`
    - `identity: 2b=19.1351, 3b=12.6156`
    - `random: 2b=26.2928, 3b=12.6578`
    - `fokvq: 2b=15.1640, 3b=12.2210`
    - `fokvq_e2: 2b=12.4561, 3b=11.5092`
    - `kivi_residual: 2b=13.1795, 3b=11.8421`
    - `turboquant_rand: 2b=14.5626, 3b=11.7115`
    - `complex_unitary_residual: 2b=15.3706, 3b=13.0332`
    - `banded_complex_unitary_residual: 2b=17.2121, 3b=12.0620`
- Decision: keep
- Reason:
  - the structured-basis effect survives on bounded `Qwen`
  - `fokvq_e2` becomes the strongest current structured candidate
  - this changes the search order:
    - promote `fokvq_e2` first
    - keep complex/unitary paths as exploratory challengers
    - attach Hamiltonian-style diagnostics only as descriptive support
