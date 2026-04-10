# Experiment Plan v30: Validity-First Selective Refinement

## 0. Decision Summary

이번 버전의 원칙은 하나다. **저장 경로가 검증되지 않은 아이디어를 compression method처럼 밀지 않는다.**

- **Proceed now**: attention-path `two_pass`를 diagnostic baseline으로 유지
- **Proceed now**: reproducibility audit on `two_pass`
- **Proceed now**: `two_pass` vs `recent_k` / `sink_k` / `random_k` control
- **Proceed later**: harder 8K/16K frontier, but only after reproduction is stable
- **Proceed next**: true storage-valid selective refinement 구현
- **Diagnostic only**: QDRP
- **Hold**: TRIC
- **Drop for now**: `query_dequant`, `sharp_temp`

## 1. Verified Starting Point

현재까지 확실한 사실은 다섯 가지다.

1. Mistral 4K same-harness에서 `baseline_2bit = 0.333`, `two_pass_k16 = 1.000`, `query_dequant` 최고값은 `0.333`, `sharp_temp`는 `0.000`이었다.
2. Mistral 4K `two_pass_k8/k16/k32/k64`는 모두 `1.000`이었다.
3. Qwen 4K `two_pass_k8/k16/k32/k64`도 모두 `1.000`이었다.
4. QDRP는 synthetic low-budget `budget=1`에서는 `risk_vs_score_recover=+0.3770`이지만, `budget=2`에서는 pure risk가 `-0.1240`으로 무너지고 hybrid도 `+0.0005`에 그쳤다.
5. TRIC는 synthetic에서 shared linear baseline을 못 넘었다. 기록된 `recursive_gain_vs_linear`은 `-3.1270`, `-0.5619`, `-0.2633`이었다.
6. Fresh local bounded rerun at the current checked revision did **not** reproduce the earlier remote `two_pass` win. On Mistral 4K with depths `{0.1, 0.5, 0.9}` and `repeats=1`, `two_pass_k8 = 0.000` and `two_pass_k16 = 0.000`, while heuristic controls were also all `0.000`.

## 2. What Is Invalid Right Now

현재 가장 중요한 결론은 “무엇이 작동했는가”가 아니라 “무엇을 아직 주장할 수 없는가”다.

### 2.1 `exp_query_exploit.py`

- 현재 patcher는 underlying KV cache를 FP16로 유지한다.
- low-bit logic은 attention path에서만 적용된다.
- 따라서 이 스크립트의 `two_pass`, `query_dequant`, `sharp_temp` 결과는 **decode-time intervention diagnostic**이지 stored-cache compression 결과가 아니다.

### 2.2 `exp_cliffkv_niah.py`

- 현재 checked revision도 `key_fp16 = key_states.clone()` 후 attention path에서 promoted key를 다시 주입한다.
- 따라서 이 스크립트도 현재 상태로는 **storage-valid compression benchmark**가 아니다.
- 예전 로그의 “cache-aware cliffkv” 문구는 현재 코드 revision만으로는 방어되지 않는다.

## 3. Why Reviewers Will Attack This

### 3.1 Storage-validity objection

리뷰어는 가장 먼저 “compressed cache를 저장한 것이 아니라 FP16 cache 위에서 읽기만 바꾼 것 아닌가?”라고 묻는다. 현재는 그 objection이 맞다.

### 3.2 Saturated binary NIAH objection

현재 headline result는 사실상 `3 depths x 2 repeats = 6 trials` per setting의 binary success다. 게다가 4K에서는 `k=8`부터 이미 포화라서 minimum useful budget을 말해 주지 못한다.

### 3.3 Artifact objection

현재 `two_pass` selector는 같은 attention wrapper 내부에서 low-bit score를 계산하고, 바로 같은 wrapper 안에서 FP16 key를 다시 넣는다. 따라서 “separate cache read path가 실제로 유효한가”와 “이 wrapper trick이 binary retrieval을 우연히 뒤집는가”가 아직 분리되지 않았다.

### 3.4 Recency / sink objection

needle retrieval이 아니라 recency 또는 sink bias를 잡아도 binary NIAH는 올라갈 수 있다. 이 objection은 control run으로만 막을 수 있다.

## 4. Prior-Work Boundary

현재 방향은 선행연구와 다음처럼 구분해야 한다.

- `two_pass` diagnostic:
  - H2O / SnapKV / QUEST / RocketKV처럼 중요한 token을 고르는 계열과 닿아 있다.
  - novelty는 “query를 쓴다”가 아니라 **low-bit score로 고른 subset만 higher-precision read를 허용했을 때 실제 retrieval failure가 복구되는가**에 있다.
  - 하지만 storage path가 없으면 method novelty보다 **diagnostic baseline**에 가깝다.

- QDRP:
  - score-max selector가 아니라 **quantization uncertainty-aware page risk**를 쓰겠다는 점이 핵심이다.
  - real trace에서 raw score를 못 이기면 이론 framing만 남고 empirical story는 끝난다.

- TRIC:
  - MiniCache / CLA / xKV / AQUA-KV류와 겹친다.
  - 살아남으려면 “recursive”가 아니라 **shared predictor가 conditional innovation entropy를 실제로 줄인다**는 증거가 먼저 필요하다.

## 5. Failure-Mode Inventory

다음 failure mode를 하나씩 죽여야 한다.

1. **Cache-path illusion**: FP16 cache가 그대로 남아 있는 상태에서 생긴 gain
2. **Selector artifact**: 선택과 repair가 같은 wrapper 안에 결합되어 생긴 gain
3. **Recency / sink artifact**: retrieval이 아니라 위치 heuristic이 만든 gain
4. **Saturated task artifact**: 4K binary NIAH가 너무 쉬워서 생긴 gain
5. **Budget accounting artifact**: effective bit proxy가 실제 저장 비용보다 낮게 보이는 문제
6. **Metric mismatch**: binary success만 올라가고 full ranking recovery는 없는 경우
7. **Reproducibility gap**: earlier remote win이 current local revision에서 재현되지 않는 경우

## 6. Workstream 0: Reproducibility Audit

### Intent

이전 remote saturation result와 현재 checked revision의 local bounded rerun 사이의 불일치를 설명한다.

### Hypothesis

차이는 model revision이 아니라 harness revision, generation setting, prompt realization length, 또는 result accounting 차이에서 나왔을 가능성이 크다.

### Verification

- exact script diff between remote-winning revision and current revision
- same prompt grid, same seed, same repeats, same depths
- compare:
  - generated text
  - realized context length
  - selected token diagnostics
  - `two_pass` vs `baseline_2bit`
- rerun both locally and on E8 with identical CLI

### Kill Criterion

remote result를 current code에서 못 재현하면, earlier remote win은 headline evidence에서 내린다.

## 7. Workstream A: Attention-Path Diagnostic Baseline

### Intent

현재 코드가 실제로 주장 가능한 범위 안에서, `two_pass`가 단순 heuristic보다 나은 diagnostic baseline인지 확인한다.

### Hypothesis

같은 top-k budget에서 `two_pass`는 `recent_k`, `sink_k`, `random_k`보다 needle span overlap과 NIAH에서 우세해야 한다.

### Verification

- models: Mistral, Qwen
- context: 4K, 8K, 16K
- budget: `k = 1, 2, 4, 8, 16`
- metrics:
  - NIAH
  - needle hit rate
  - sink overlap rate
  - recent overlap rate
  - selected head slots
  - runtime

### Kill Criterion

`recent_k` 또는 `sink_k`가 `two_pass`를 따라오면 retrieval-specific interpretation을 중단한다.

## 8. Workstream B: True Storage-Valid Selective Refinement

### Intent

attention-path proxy를 끝내고, 실제 저장된 low-bit cache와 promoted side buffer를 가진 구현으로 옮긴다.

### Hypothesis

저장 경로가 실제로 low-bit base + small promoted side buffer로 바뀌어도 selective refinement gain의 일부가 유지될 수 있다.

### Minimal Implementation

1. prefill 시점에 base 2-bit key/value를 실제 cache object에 저장
2. promoted subset은 별도 side buffer에 저장
3. decode 시점에 selector는 base cache만 보고 top-k 결정
4. attention read는 base cache + promoted side buffer만 사용
5. FP16 full cache fallback 금지

### Verification

- equal-storage
- equal-HBM
- equal-extra-compute

비교:

- `uniform_2bit`
- `uniform_3bit`
- stored selective 2bit+3bit
- stored selective 2bit+fp16 side buffer

### Kill Criterion

storage-valid path에서 advantage가 사라지면, 현재 방향은 method가 아니라 diagnostic trick으로 정리한다.

## 9. Workstream C: QDRP Gate

### Intent

QDRP를 되살릴지 완전히 접을지 결정한다.

### Hypothesis

real trace에서 page-level risk proxy가 raw score나 raw margin보다 miss-page recall에서 나아야 한다.

### Verification

Stage 1:
- trace-level calibration only
- metrics:
  - miss-page recall@budget
  - AUROC
  - page-hit
  - recovered top-1 after oracle page refinement

Stage 2:
- only if Stage 1 wins
- small NIAH integration

### Kill Criterion

real trace에서 raw score를 못 넘으면 QDRP는 headline direction에서 완전히 제외한다.

## 10. Workstream D: TRIC Gate

### Intent

TRIC를 다시 볼지 말지 결정한다.

### Hypothesis

real activations에서 shared linear baseline이 설명하지 못하는 conditional innovation reduction이 있어야 한다.

### Verification

Stage 1:
- collect real layerwise KV traces
- compare `copy-last`, shared linear, tiny recursive
- metrics:
  - residual MSE
  - query-logit MSE
  - residual covariance spectrum

Stage 2:
- only if recursive beats shared linear
- bounded end-to-end patch

### Kill Criterion

shared linear를 못 넘으면 TRIC는 중단한다.

## 11. Immediate Execution Order

1. freeze the current checked revision and archive the fresh local rerun
2. audit exact repro gap against the earlier remote-winning run
3. rerun `two_pass` plus `baseline_2bit` under identical CLI on local and E8
4. only if stable, run bounded control sweep for `two_pass` vs heuristic controls
5. only after that, move to harder frontier
6. implement true stored selective refinement before any new compression claim

## 12. Brutal Status

지금 top-venue method로 바로 밀 수 있는 것은 없다.

- `two_pass`: 아직도 좋은 **diagnostic candidate**이지만, 지금은 먼저 reproducibility audit 대상이다
- QDRP: 아직 이론 가설
- TRIC: 아직 synthetic에서 탈락

다음 진짜 분기점은 두 개다. 첫째, earlier remote win이 current code에서 재현되는가. 둘째, storage-valid selective refinement를 구현했는데도 gain이 남는가. 둘 다 아니면 이 wave는 paper method가 아니라 failure analysis chapter로 정리하는 것이 맞다.
