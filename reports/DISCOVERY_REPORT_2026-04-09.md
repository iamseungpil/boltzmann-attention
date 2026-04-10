# Discovery Report (2026-04-09)

## Executive Summary

이번 라운드에서 살아남은 것은 하나뿐이다. `two_pass` selective refinement는 bounded 4K NIAH에서 강한 retrieval recovery diagnostic으로 남았고, `query_dequant`, `sharp_temp`, 실전형 `QDRP`, `TRIC`은 현재 증거로는 연구의 주축이 될 수 없다.

다만 이 결론도 범위를 정확히 잘라서 읽어야 한다. 현재 `scripts/exp_query_exploit.py`와 `scripts/exp_cliffkv_niah.py`는 **저장된 KV cache를 실제로 압축한 실험이 아니라**, FP16 cache를 유지한 채 attention path에서만 저비트 근사와 selective promotion을 적용하는 diagnostic harness다. 따라서 지금 당장 말할 수 있는 것은 “query-time selective refinement가 retrieval failure mode를 강하게 찌른다”는 사실뿐이고, “새로운 KV cache compression method가 storage budget에서 이긴다”는 주장은 아직 쓸 수 없다.

이 보고서의 목적은 처음 보는 사람도 현재 상태를 한 번에 이해하게 만드는 것이다. 무엇이 검증되었고, 무엇이 틀렸고, 다음에 어떤 실험만 남겨야 하는지까지 한 문서에서 정리한다.

## 1. Verified Fact Base

이번 문서의 수치는 아래 파일에서 직접 확인했다.

| ID | Claim | Source |
|----|-------|--------|
| F1 | Mistral 4K same-harness에서 `baseline_2bit = 0.333`, `two_pass_k16 = 1.000`, `query_dequant_s0.25 = 0.333`, `sharp_temp_t{0.5,0.7,0.9} = 0.000`. | `tmp_remote_results/mistral_all_methods_4k.json` |
| F2 | Mistral 4K two-pass sweep에서 `k=8,16,32,64`가 모두 `1.000`. | `tmp_remote_results/mistral_two_pass_4k.json` |
| F3 | Qwen 4K two-pass sweep에서 `k=8,16,32,64`가 모두 `1.000`. | `tmp_remote_results/qwen_two_pass_4k.json` |
| F4 | `QDRP` synthetic low-budget에서는 `risk_vs_score_recover = +0.3770`, `budget=2` pure risk에서는 `-0.1240`, hybrid는 `+0.0005`. | `reports/autoresearch_dynamic_recursive_log.tsv` |
| F5 | `TRIC`은 세 번의 synthetic 반복에서 모두 shared linear baseline을 이기지 못했고, 마지막 기록의 `recursive_gain_vs_linear = -0.2633`. | `reports/autoresearch_dynamic_recursive_log.tsv` |
| F6 | 최신 self-test 기준으로 `exp_query_exploit.py`, `exp_cliffkv_niah.py`, `exp_query_risk_paging.py`, `exp_tric_recursive_predictor.py`는 컴파일 및 smoke self-test를 통과했다. | local run on 2026-04-09 |
| F7 | 현재 `exp_query_exploit.py`와 `exp_cliffkv_niah.py`는 limitation metadata와 docstring에 “attention-path diagnostic, not storage-valid compressed-cache measurement”를 명시한다. | code inspection on 2026-04-09 |

이 문서에서 F1-F7 밖의 숫자는 사용하지 않는다.

## 2. What Changed Since the Earlier Plan

초기 계획은 세 축을 모두 연구 후보로 보고 있었다. 지금은 아니다.

첫째, selective refinement는 살아남았지만 **논문 방법**에서 **diagnostic baseline**으로 격하되었다. 이유는 단순하다. 결과는 강하지만 저장 경로가 아직 진짜 compression이 아니기 때문이다.

둘째, `QDRP`는 “좋은 이론 같아 보이는 방법”에서 “synthetic에서만 잠깐 이기는 selector”로 내려왔다. 실전 trace에서 risk proxy가 raw score보다 못하면, rate-distortion framing이 아무리 예뻐도 방법은 죽는다.

셋째, `TRIC`은 “cross-layer predictive coding”이라는 큰 이야기에서 “linear baseline도 못 넘는 진단 실패”로 정리되었다. 여기서 더 GPU를 쓰는 건 연구가 아니라 미련이다.

## 3. Direction-by-Direction Assessment

### 3.1 Selective Refinement / Two-Pass Promotion

**Intent.** 저비트 key approximation으로 무너진 retrieval을 decode 시점의 selective refinement로 복구할 수 있는지 본다.

**Hypothesis.** needle이 이미 2-bit ranking에서 top confusion zone 안에 있다면, top-k selective promotion이 binary retrieval failure를 뒤집을 수 있다.

**What passed.** F1-F3이 이 가설의 가장 강한 근거다. 같은 harness에서 `two_pass`만 baseline을 명확하게 이겼고, 두 모델 모두 4K에서는 `k=8`부터 포화되었다.

**What failed.** novelty claim과 compression claim은 아직 통과하지 못했다. 지금 경로는 storage-valid method가 아니라 attention-path diagnostic이다. 또 4K에서 `k=8/16/32/64`가 전부 `1.000`이라는 사실은 이 grid가 너무 쉽다는 뜻이기도 하다.

**Verdict.** 계속 가져갈 수는 있다. 단, 이름을 크게 붙일 단계는 아니다. 현재 정체성은 “강한 failure-mode diagnostic”이다.

### 3.2 Query-Dynamic Risk Paging (QDRP)

**Intent.** raw score가 아니라 score margin과 quantization uncertainty를 같이 사용해 “flip risk가 큰 page”만 refinement한다.

**Hypothesis.** 예산이 극도로 작을 때는 raw-score paging보다 risk-aware paging이 true winner page를 더 잘 맞혀야 한다.

**What passed.** synthetic low-budget에서는 맞는다. self-test와 log 기준으로 `risk_vs_score_recover`가 크게 양수다.

**What failed.** real trace에서는 risk proxy가 raw score보다 약하다. `budget=2`에서는 pure risk가 score page를 버리는 바람에 오히려 망가졌고, hybrid도 사실상 tie 수준이다.

**Verdict.** 현재 상태로는 방법이 아니라 진단 도구다. “budget=1 극한 조건”이라는 아주 좁은 niche를 명확히 입증하기 전에는 headline으로 올리면 안 된다.

### 3.3 Tiny Recursive Innovation Cache (TRIC)

**Intent.** 레이어 간 KV redundancy를 shared recursive predictor로 설명하고 innovation residual만 저장한다.

**Hypothesis.** copy-last보다 좋을 뿐 아니라 shared linear predictor도 넘어야 predictive coding story가 선다.

**What passed.** recursive predictor는 copy-last보다 낫다.

**What failed.** shared linear predictor를 계속 못 이긴다. 이 상태에서는 “tiny recursive”가 아니라 그냥 capacity-overhead가 큰 약한 predictor다.

**Verdict.** 지금은 중단이 맞다. 다시 열려면 실제 KV trace에서 linear baseline을 넘는 증거가 먼저 나와야 한다.

## 4. Why Two-Pass May Be Working

현재까지의 증거를 기준으로 보면, `two_pass`가 먹히는 이유는 “query-aware라서”가 아니라 더 좁다.

첫째, 이전 needle-ranking 진단과 이번 NIAH 결과를 함께 읽으면, 2-bit에서도 needle이 아예 사라진 것이 아니라 상위 혼잡 구간에 남아 있을 가능성이 높다. 이 경우 selective promotion은 ranking 전체를 재구성할 필요 없이 소수 후보만 다시 세우면 된다.

둘째, `query_dequant`와 `sharp_temp`가 모두 실패했다는 점은 failure mode가 단순한 bin-centering 문제나 softmax flattening 하나로 설명되지 않음을 시사한다. 즉, 무엇이든 “점수의 아주 작은 전역 보정”보다 “후보 집합 자체를 다시 분리하는 국소적 보정”이 중요하다는 뜻이다.

셋째, 지금 결과는 여전히 artifact 설명을 배제하지 못한다. 아래 네 가지는 실제 원인일 수 있다.

| Alternative cause | Why it is dangerous | What would kill it |
|-------------------|---------------------|--------------------|
| Recency artifact | top-k가 query 직전 토큰을 자주 집으면 retrieval repair가 아니라 recent-token rescue일 수 있다. | `recent_k`가 `two_pass`보다 분명히 낮아야 한다. |
| Attention sink artifact | BOS/초기 sink token을 꾸준히 올려 binary success margin을 우연히 뒤집을 수 있다. | `sink_k`가 `two_pass`보다 낮고, selector trace에서 sink overlap이 과도하지 않아야 한다. |
| Easy 4K cliff task | 4K grid가 너무 쉬우면 어떤 sensible reranker도 1.0이 나온다. | 8K/16K에서 `k=1,2,4,8,16` frontier가 벌어져야 한다. |
| Binary NIAH saturation | top-1 성공만 보면 ranking 전체 복구 없이도 결과가 좋아 보일 수 있다. | answer-margin, needle rank, attention correlation 같은 연속형 metric이 같이 좋아야 한다. |

## 5. Related-Work Map and Real Differentiation

현재 공간은 이미 혼잡하다. 그래서 “우리 방법이 query-aware다” 수준으로는 차별화가 안 된다.

토큰 선택 계열에는 H2O, SnapKV, Quest, RocketKV, ProphetKV처럼 query 또는 proxy attention으로 중요한 토큰을 남기거나 다시 계산하는 흐름이 있다. 이 계열과 비교하면 현재 `two_pass`는 새 방법이라기보다 **bounded reranking diagnostic**에 가깝다. 진짜 차별점은 나중에 **storage-valid selective side buffer**까지 들어갈 때만 생긴다.

양자화 계열에는 KIVI, KVQuant, ZipCache, KVTC, AQUA-KV, MixKVQ가 있다. 이들은 실제 저장 예산 안에서 양자화 단위, saliency, transform coding, cross-layer prediction을 설계한다. 따라서 현재 `exp_query_exploit.py` 결과를 이들과 같은 표에 바로 올리면 비교 자체가 불공정하다.

cross-layer 계열에는 MiniCache와 AQUA-KV가 이미 “깊이 방향 redundancy”를 전면에 세운다. 따라서 `TRIC`이 linear predictor조차 못 넘는 상태에서 recursive framing만으로 승부하는 것은 불가능하다.

## 6. Decision Table

| Direction | Status | Why |
|-----------|--------|-----|
| `two_pass` attention-path diagnostic | Keep | F1-F3 기준으로 retrieval failure mode를 강하게 복구한다. |
| Storage-valid selective refinement | Promote to primary implementation target | 지금 남은 유일한 paper-worthy pivot이다. |
| `QDRP` | Keep only as low-budget diagnostic | synthetic win은 있으나 real proxy가 약하다. |
| `TRIC` | Stop | shared linear baseline 미만이다. |
| `query_dequant` | Stop | bounded same-harness에서 baseline도 못 넘는다. |
| `sharp_temp` | Stop | 전부 실패했다. |

## 7. Concrete Next Steps

다음 계획은 의도적으로 좁다.

1. **Storage-valid selective refinement를 구현한다.**
   2-bit base cache와 promoted side buffer를 실제 저장 경로에 넣고, write-time budget과 read-time budget을 분리해서 측정한다.

2. **현재 two-pass는 diagnostic controls만 더 본다.**
   `recent_k`, `sink_k`, `random_k`, needle hit rate, sink overlap, recent overlap을 같은 harness에서 비교한다.

3. **4K 포화 grid는 버린다.**
   8K/16K에서 `k=1,2,4,8,16` frontier만 측정한다.

4. **QDRP는 budget=1 niche만 남긴다.**
   real trace에서 `risk > score`가 재현되지 않으면 종료한다.

5. **TRIC는 중단한다.**
   다시 열려면 “real KV trace에서 shared linear를 넘었다”는 새 증거가 먼저 필요하다.

## 8. Unverified or Not Yet Admitted

아래 항목은 결과 파일은 있지만 아직 이번 보고서의 사실 기반에는 넣지 않았다.

| Claim | Reason for exclusion |
|-------|----------------------|
| `results/e8_2026-04-09/Mistral-7B-v0_3_fixcheck_cliffkv_mistral_4k.json`의 selective refinement score | 현재 코드 경로와 결과 생성 경로가 정확히 일치하는지 아직 재검증하지 않았다. |
| `cache-aware cliffkv smoke` 로그 한 줄 | 원격 실행 당시 harness와 현재 저장 경로 구현을 동일한 수준으로 재현하지 못했다. |

이 둘은 다시 검증되기 전까지는 paper evidence가 아니라 참고 메모로만 취급한다.
