# Experiment Plan v30: Storage Validity First

## Decision Summary

이번 버전의 계획은 이전보다 더 공격적으로 버린다.

- **Primary now**: storage-valid selective refinement
- **Diagnostic only**: attention-path `two_pass` controls
- **Very narrow diagnostic**: `QDRP` only at `budget_pages=1`
- **Stop**: `TRIC`
- **Stop**: `query_dequant`
- **Stop**: `sharp_temp`

핵심 원칙은 하나다. **진짜 저장 경로가 없는 결과는 방법 claim으로 승격하지 않는다.**

## Why v29 Is Not Enough

v29는 objection list는 잘 잡았지만, 여전히 “attention-path diagnostic”과 “storage-valid compression method”를 충분히 분리하지 못했다.

남아 있는 가장 큰 위험은 다섯 가지다.

1. 현재 `two_pass` 결과는 storage-valid method가 아니라 reranking-style diagnostic일 수 있다.
2. 4K NIAH는 너무 쉬워서 `k` frontier를 말해 주지 못한다.
3. gain이 실제 retrieval 복구가 아니라 recency/sink artifact일 수 있다.
4. fair-budget accounting이 불명확하면 reviewer가 바로 budget cheating으로 읽는다.
5. `QDRP`와 `TRIC`는 이미 empirical gate에서 대부분 탈락했다.

## Track A: Storage-Valid Selective Refinement

### Intent

저비트 base cache와 selective promoted side buffer를 **실제 저장 경로**에 넣어, 같은 저장 예산 또는 같은 HBM budget에서 retrieval gain이 남는지 확인한다.

### Hypothesis

needle이 low-bit ranking의 혼잡 구간에 남아 있다면, 2-bit base + sparse promoted side buffer는 uniform 3-bit보다 더 낮은 저장량 또는 더 낮은 HBM budget에서 retrieval을 회복할 수 있다.

### Verification

필수 구현 조건:

1. prefill/write 시점에 base cache를 실제로 2-bit 형태로 저장한다.
2. promoted subset은 별도 side buffer로 저장하거나 재구성한다.
3. 결과 파일에 아래를 모두 기록한다.
   - stored base bits
   - stored promoted bits
   - promoted slot count
   - total stored bytes
   - extra decode FLOPs or wall-clock

비교 표:

- `uniform_2bit`
- `uniform_3bit`
- `2bit + promoted side buffer`

평가:

- 8K, 16K NIAH
- `k = 1, 2, 4, 8, 16`
- same prompt grid
- runtime
- stored-byte budget

### Kill Criterion

아래 중 하나라도 성립하면 이 track은 논문 주축에서 내린다.

1. storage-valid 경로로 바꾸면 이득이 사라진다.
2. uniform 3-bit와 fair budget을 맞추면 advantage가 사라진다.
3. 8K/16K에서 `k`를 줄이면 바로 baseline으로 붕괴한다.

## Track B: Attention-Path Two-Pass Controls

### Intent

현재 강한 `two_pass` gain이 진짜 retrieval-specific signal인지, 아니면 recency/sink/easy-grid artifact인지 구분한다.

### Hypothesis

`two_pass`가 실제로 needle-relevant 후보를 집는다면 `recent_k`, `sink_k`, `random_k`보다 NIAH와 needle-hit diagnostics에서 더 좋아야 한다.

### Verification

같은 prompt grid에서 다음을 모두 저장한다.

- NIAH accuracy
- prefill-layer needle hit rate
- sink overlap rate
- recent overlap rate
- selected unique tokens
- selected head slots

비교:

- `two_pass`
- `recent_k`
- `sink_k`
- `random_k`

조건:

- 4K controls는 smoke로만 사용
- 판단은 8K/16K frontier에서 내린다

### Kill Criterion

아래 중 하나라도 성립하면 retrieval-specific story를 중단한다.

1. `recent_k` 또는 `sink_k`가 `two_pass`와 비슷하게 나온다.
2. selector trace에서 needle hit 없이도 success가 반복된다.
3. 연속형 ranking metric은 회복되지 않는데 binary success만 오른다.

## Track C: QDRP as a Narrow Diagnostic

### Intent

page budget이 극도로 작을 때 (`budget_pages=1`) risk-aware selector가 raw score를 이길 수 있는지 본다.

### Hypothesis

single-page budget에서는 score-max page보다 high-risk page가 true winner page를 더 자주 포함해야 한다.

### Verification

단계:

1. synthetic calibration 유지
2. real trace에서 page-level winner recovery를 다시 측정
3. `score`, `margin`, `risk`, `hybrid`, `oracle`를 같은 trace에서 비교

### Kill Criterion

1. real trace에서 `risk <= score`면 즉시 종료
2. `budget_pages=2` 이상에서 hybrid도 이득이 없으면 headline에서 완전히 제거

## Track D: TRIC Stop Condition

### Current State

synthetic에서도 shared linear predictor를 못 넘는다.

### Rule

새 실험을 하지 않는다. 다시 열려면 다음 두 조건이 먼저 필요하다.

1. 실제 KV trace에서 shared linear baseline이 정의된다.
2. tiny recursive predictor가 그 baseline을 분명히 이긴다.

그 전까지는 이론 메모로만 남긴다.

## Root-Cause Questions To Answer

다음 질문에 답하지 못하면 selective refinement도 paper story가 약하다.

1. needle이 정말 top confusion zone 안에 남아 있는가?
2. two-pass가 needle region을 고르는가, 아니면 recent/sink를 고르는가?
3. binary NIAH가 아닌 연속형 ranking metric도 좋아지는가?
4. extra compute를 storage gain과 분리하면 무엇이 남는가?

## Minimal Next Experiment Set

다음 네 개만 통과하면 다음 wave로 넘어간다.

1. **Code gate**: storage-valid selective refinement write path 구현
2. **Control gate**: `two_pass` vs `recent_k/sink_k/random_k`
3. **Frontier gate**: 8K/16K with `k = 1,2,4,8,16`
4. **Budget gate**: stored-byte matched table vs `uniform_3bit`

이 네 개 중 둘 이상이 실패하면, selective refinement도 논문 방향이 아니라 내부 diagnostic으로 내린다.
