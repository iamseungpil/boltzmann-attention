# Research Status Report (2026-04-09)

## Executive Summary

현재 가장 중요한 사실은 성능이 아니라 claim scope다. `two_pass`와 `cliffkv` 계열은 bounded NIAH에서 강한 양성 신호를 보였지만, 지금 코드가 입증한 것은 **compressed KV cache 방법**이 아니라 **attention-path proxy diagnostic**이다. 따라서 이 결과를 storage-valid compression 성과처럼 쓰면 바로 무너진다.

동시에 방향성도 정리됐다. `two_pass`만 추가 진단을 진행할 가치가 있고, `QDRP`는 toy low-budget regime을 넘어서면 근거가 약하며, `TRIC`은 shared linear baseline을 넘지 못해 현재 단계에서는 중단이 맞다. 지금 필요한 것은 새 아이디어를 더 늘리는 일이 아니라, 강한 양성 신호 하나를 과장 없이 좁게 검증하는 일이다.

## 1. What Is Actually Verified

이번 wave에서 검증된 수치는 다섯 묶음으로 정리된다. 첫째, Mistral 4K same-harness에서 `baseline_2bit=0.333`이고 `two_pass_k16=1.000`이며, `query_dequant` 최고값은 `0.333`, `sharp_temp`는 전부 `0.000`이다. 둘째, Mistral과 Qwen 모두 4K bounded NIAH에서 `two_pass_k8/k16/k32/k64`가 전부 `1.000`으로 포화된다. 셋째, proxy selective-promotion smoke에서는 Mistral 4K 기준 `uniform_2bit=0.000`, `uniform_3bit=0.667`, `cliffkv_k64_b3=1.000`, `avg_bits_per_dim=2.016`이 나온다. 넷째, QDRP는 synthetic low-budget에서는 `risk_vs_score_recover=+0.3770`이지만, `budget_pages=2`에서는 pure risk가 음수로 돌아서고 hybrid도 `+0.0005`에 그친다. 다섯째, TRIC은 세 번의 synthetic 확대 실험 모두에서 `recursive_gain_vs_linear`가 음수다.

이 수치들이 말해 주는 바는 단순하다. 현재 positive signal은 retrieval repair 쪽에만 있고, 그것도 아직은 storage path가 아니라 attention path에서만 검증되었다.

## 2. What Is Not Verified

지금 검증되지 않은 항목은 더 중요하다. `exp_query_exploit.py`와 `exp_cliffkv_niah.py`는 둘 다 내부적으로 full cached key를 FP16로 유지한 뒤, attention 계산 시점에만 2-bit base와 selective promotion을 적용한다. 즉 저장된 cache 자체가 low-bit representation으로 대체되지 않는다. 이 구조에서는 저장 예산, HBM traffic, cache residency, side-buffer cost를 정직하게 측정할 수 없다.

따라서 다음 문장들은 현재 금지다. 첫째, “storage-valid KV cache compression이 retrieval을 회복했다.” 둘째, “effective average bits가 공정 비교를 보장한다.” 셋째, “cache-correct two-pass method가 완성되었다.” 지금 허용되는 문장은 더 좁다. “FP16 cache 위 attention-path intervention으로서 two-pass selective refinement가 bounded NIAH에서 강한 recovery signal을 보였다.”

## 3. Harsh Verdict By Direction

### 3.1 Selective Refinement / Two-Pass

이 방향은 **살아 있다.** 다만 살아 있는 이유는 novelty가 아니라 signal strength다. Mistral과 Qwen의 4K bounded NIAH에서 같은 하네스 안의 다른 query-time interventions를 모두 누른 것은 무시하기 어렵다. 반대로 이것을 곧바로 논문 방법으로 부르면 곤란하다. 현재 구현은 selector diagnostic으로는 유용하지만, compression method claim으로는 불충분하다.

이 방향의 가장 큰 위험은 세 가지다. 첫째, 4K grid가 너무 쉬워서 `k` frontier가 보이지 않는다. 둘째, recency 또는 attention sink artifact가 아직 배제되지 않았다. 셋째, budget이 upper-bound proxy이기 때문에 공정성 objection에 취약하다. 따라서 다음 단계는 더 많은 heuristic을 붙이는 것이 아니라, `recent_k`, `sink_k`, `random_k` control과 8K/16K tiny-budget sweep으로 양성 신호의 성격을 먼저 확인하는 일이다.

### 3.2 QDRP

이 방향은 **headline method 후보에서 제외**해야 한다. 이론적 framing 자체는 나쁘지 않다. query-conditioned mismatch risk를 직접 겨냥한다는 점은 score-only selector와 구분될 수 있다. 문제는 empirical path가 따라오지 못한다는 데 있다. synthetic low-budget toy에서만 좋고, budget이 조금만 늘어나도 pure risk가 current winner page를 버리는 구조적 문제가 드러난다. 더 결정적인 점은 기존 real-trace calibration에서 raw score보다 낫다는 증거가 없다는 것이다.

따라서 QDRP는 지금 당장 GPU를 더 써서 살릴 방향이 아니다. 살릴 수 있는 최소 조건은 하나다. 실제 trace에서 page-level miss recall이나 winner-page hit에서 raw score를 이기는 revised risk metric이 나와야 한다. 그 전에는 method가 아니라 theory memo다.

### 3.3 TRIC

이 방향은 **중단**이 맞다. shared recursive predictor라는 말은 그럴듯하지만, 현재 synthetic diagnostic에서 shared linear predictor조차 넘지 못한다. copy-last보다 낫다는 것은 아무 의미가 없다. 리뷰어가 보는 기준선은 더 강하다. predictor overhead를 고려하면, shared linear에도 지는 recursive predictor에 end-to-end cache coding을 얹는 순간 복잡성만 늘고 기여는 사라진다.

TRIC을 다시 열 수 있는 조건도 명확하다. 실제 layer traces에서 shared linear 대비 residual energy, query-logit MSE, 또는 conditional R-squared가 유의미하게 더 좋아야 한다. 그 전까지는 연구 분기에서 제거하는 편이 낫다.

## 4. Why Reviewers Will Attack

리뷰어 objection은 이미 정리되어 있다. 첫째, “이건 그냥 top-scoring key를 다시 보는 baseline 아닌가”라는 공격이다. 이 objection은 지금 상태에서 맞는 말에 가깝다. 둘째, “4K NIAH binary success는 너무 쉬워서 과장된 지표다”라는 공격이다. 이것도 맞다. 셋째, “real cache compression이 아니면 budget claim이 성립하지 않는다”는 objection은 치명적이다. 넷째, “gain이 recency/sink artifact일 수 있다”는 objection도 아직 남아 있다.

좋은 소식은 이 objection들이 전부 같은 처방을 요구한다는 점이다. control baseline, harder frontier, storage-valid rewrite, 이 세 가지다. 나쁜 소식은, 이 셋 중 하나라도 빠지면 top venue 기준으로는 방어가 어렵다는 점이다.

## 5. Immediate Code Status

코드 상태는 문서보다 낫다. 두 주요 스크립트는 이미 limitation을 내부 metadata에 기록하고 있었고, 이번 정리에서 top-level payload에도 `attention_path_proxy_diagnostic` claim scope를 명시하도록 보강했다. `py_compile`과 self-test는 통과했다. 즉 지금 코드는 “무엇을 주장할 수 없는지”를 이전보다 더 명확하게 표현한다.

반대로 구현 자체의 연구 상태는 여전히 초기다. selector diagnostic에는 충분하지만, true storage-valid method evaluation에는 아직 충분하지 않다. 후속 실험을 돌리더라도 이 범위를 넘어서는 해석은 금지해야 한다.

## 6. Recommended Plan

지금 가장 좋은 계획은 둘로 나뉜다. 단기 계획은 proxy diagnostic을 끝까지 정리하는 것이다. `two_pass`와 `recent_k/sink_k/random_k`를 같은 prompt grid에서 비교하고, 8K/16K에서 `k=1,2,4,8,16` frontier를 측정한다. 이 단계의 목적은 “signal이 진짜 retrieval repair인지, 아니면 쉬운 task artifact인지”를 가르는 것이다.

중기 계획은 storage-valid selective refinement를 새로 구현하는 것이다. base cache를 실제로 quantized form으로 저장하고, 선택된 token만 side-buffer 또는 stored residual로 올리는 방식이 필요하다. 이 단계가 끝나기 전에는 paper method claim을 하지 않는다. 만약 storage-valid rewrite 뒤에 gain이 사라지면, 이 방향은 논문 축이 아니라 diagnostic appendix로 내려야 한다.

## 7. Final Recommendation

지금 당장은 **Proceed, but only as a diagnostic program**이 맞다. `two_pass`는 계속 본다. `QDRP`는 real-trace revised metric이 나오기 전까지 보류한다. `TRIC`은 닫는다. 새 GPU를 쓰더라도 순서는 바뀌지 않는다. 먼저 control과 harder frontier로 `two_pass`를 압박하고, 그다음 storage-valid rewrite를 해야 한다. 그 전에는 어떤 화려한 이름도 방법의 약점을 가리지 못한다.
