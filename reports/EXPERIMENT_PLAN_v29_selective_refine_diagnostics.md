# Experiment Plan v29: Selective Refinement Diagnostics

## Decision Summary

이번 버전의 계획은 의도적으로 좁다. 지금 당장 밀 수 있는 방향은 `two_pass` 하나뿐이다.

- **Proceed now**: `two_pass` selector diagnostics + harder budget frontier
- **Proceed now**: heuristic control baselines (`recent_k`, `sink_k`, `random_k`)
- **Proceed later**: matched-budget selective stored refinement
- **Drop for now**: `query_dequant`
- **Drop**: `sharp_temp`
- **Diagnostic only**: `QDRP`
- **Drop for now**: `TRIC`

## Why v28 Was Not Enough

v28은 “무엇을 버릴지”를 정리하는 데는 충분했지만, `two_pass`의 남은 공격 지점을 충분히 반영하지 못했다.

남아 있는 주요 objection은 네 가지다.

1. 현재 결과는 novelty보다 strong baseline에 가깝다.
2. 4K NIAH grid는 이미 포화되어 minimum budget frontier를 말해 주지 못한다.
3. attention sink 또는 recency artifact 가능성을 아직 배제하지 못했다.
4. equal-budget accounting이 분리되지 않으면 method claim이 약해진다.

## Experiment 1: Selector Diagnostics

### Intent

`two_pass`가 실제 retrieval region을 복구하는지, 아니면 recency 또는 sink heuristic을 우연히 이용하는지를 구분한다.

### Hypothesis

`two_pass`의 selected set은 simple `recent_k` 또는 `sink_k`보다 needle span과 더 자주 겹치고, same-harness NIAH도 더 높아야 한다.

### Verification

같은 prompt grid에서 다음을 같이 기록한다.

- NIAH accuracy
- prefill-layer needle hit rate
- prefill-layer sink overlap rate
- prefill-layer recent overlap rate
- average selected tokens per call

비교 대상:

- `two_pass`
- `recent_k`
- `sink_k`
- `random_k`

### Kill Criterion

`recent_k` 또는 `sink_k`가 `two_pass`와 비슷한 NIAH를 내거나, `two_pass`가 needle span과 거의 겹치지 않으면 retrieval-specific claim을 중단한다.

## Experiment 2: Harder Budget Frontier

### Intent

포화된 4K grid를 넘어서, 실제로 필요한 refinement budget의 최소값을 찾는다.

### Hypothesis

`two_pass`는 8K 또는 16K에서도 매우 작은 `k`에서 `baseline_2bit`보다 먼저 회복될 것이다.

### Verification

조건:

- context: `8192`, `16384`
- budget: `k = 1, 2, 4, 8, 16`
- models: Mistral, Qwen

기록:

- NIAH accuracy
- minimum `k` for recovery
- runtime
- effective average bits upper bound

### Kill Criterion

harder grid에서 `two_pass`의 이득이 사라지거나, `k`를 줄였을 때 바로 baseline 수준으로 떨어지면 현재 방향의 practical value를 낮게 본다.

## Experiment 3: Fair-Budget Table

### Intent

`two_pass`의 gain이 budget cheating이 아니라는 점을 분리해서 보여준다.

### Hypothesis

같은 effective budget 또는 같은 extra compute 조건에서도 selective refinement는 blunt uniform precision보다 더 효율적일 수 있다.

### Verification

세 regime을 분리한다.

1. equal-storage
2. equal-HBM
3. equal-extra-compute

비교 후보:

- `uniform_2bit`
- `uniform_3bit`
- `two_pass`
- selective stored 3-bit refinement

### Kill Criterion

budget을 공정하게 맞추면 `two_pass` advantage가 사라지면, 이 방향은 paper method가 아니라 engineering trick으로 내리는 것이 맞다.

## Deferred Directions

### QDRP

real flip-risk proxy가 raw score보다 뒤지므로 headline method에서 제외한다. 다만 oracle 또는 revised metric diagnostic은 남겨 둘 수 있다.

### Query-aware dequantization

bounded 4K same-harness에서 baseline을 넘지 못했다. score-MSE diagnostic 없이 다시 NIAH에 올리지 않는다.

### TRIC

shared linear baseline을 못 넘는 동안은 계속 보류한다.
