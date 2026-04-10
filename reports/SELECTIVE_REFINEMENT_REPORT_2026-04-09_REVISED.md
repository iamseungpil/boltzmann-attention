# Selective Refinement Status Report (Revised 2026-04-09)

## Executive Summary

이번 라운드에서 살아남은 것은 `two_pass` 하나뿐이다. 하지만 그 문장을 곧바로 “새로운 KV-cache compression method가 검증됐다”로 읽으면 틀린다.

현재 확인된 가장 강한 사실은 다음이다. earlier remote run에서는 Mistral 4K same-harness에서 `baseline_2bit = 0.333`, `two_pass_k16 = 1.000`, `query_dequant` 최고값은 `0.333`, `sharp_temp`는 전부 `0.000`이었다. Qwen 4K에서도 `two_pass_k8/k16/k32/k64`가 모두 `1.000`이었다. 반면 QDRP는 real-trace 근거가 약하고, TRIC는 synthetic에서도 shared linear baseline을 넘지 못했다.

그런데 current checked revision으로 다시 돌린 fresh local bounded rerun은 더 불편한 사실을 추가했다. Mistral 4K, depths `{0.1, 0.5, 0.9}`, `repeats=1`에서 `two_pass_k8 = 0.000`, `two_pass_k16 = 0.000`이었다. heuristic controls도 전부 `0.000`이었지만, 이건 곧바로 좋은 소식이 아니다. 지금은 “two_pass가 강하다”보다 “earlier remote win이 current code에서 안정적으로 재현되는가”가 먼저다.

동시에 가장 중요한 제한도 확인됐다. 현재 `exp_query_exploit.py`와 checked revision의 `exp_cliffkv_niah.py`는 둘 다 FP16 underlying cache를 유지한 채 attention path만 바꾼다. 따라서 지금까지의 positive result는 **storage-valid compression evidence가 아니라 attention-path diagnostic evidence**다.

## 1. What Was Actually Tested

이번 wave의 질문은 “low-bit cache에서 retrieval failure를 query-time intervention으로 복구할 수 있는가”였다. 저장 자체를 다시 설계한 실험이 아니라, 이미 존재하는 key를 decode 시점에 더 잘 읽는 실험이었다.

테스트한 방법은 네 갈래였다.

1. `two_pass`: low-bit score로 일부 위치를 고르고 그 위치만 higher precision으로 다시 읽는 방식
2. `query_dequant`: bin 내부 reconstruction point를 query 방향으로 이동하는 방식
3. `sharp_temp`: low-bit attention의 flattening을 temperature로 보정하는 방식
4. QDRP / TRIC: 각각 risk-aware page selector와 recursive innovation cache 가설

## 2. Verified Facts

이번 문서의 핵심 수치는 다음 파일들에서 직접 확인했다.

| Fact | Value | Source |
|------|-------|--------|
| Mistral 4K `baseline_2bit` | `0.333` | `tmp_remote_results/mistral_all_methods_4k.json` |
| Mistral 4K `two_pass_k16` | `1.000` | `tmp_remote_results/mistral_all_methods_4k.json` |
| Mistral 4K best `query_dequant` | `0.333` | `tmp_remote_results/mistral_all_methods_4k.json` |
| Mistral 4K best `sharp_temp` | `0.000` | `tmp_remote_results/mistral_all_methods_4k.json` |
| Mistral 4K `two_pass_k8/k16/k32/k64` | all `1.000` | `tmp_remote_results/mistral_two_pass_4k.json` |
| Qwen 4K `two_pass_k8/k16/k32/k64` | all `1.000` | `tmp_remote_results/qwen_two_pass_4k.json` |
| QDRP low-budget synthetic | `risk_vs_score_recover=+0.3770` at `budget=1` | `reports/autoresearch_dynamic_recursive_log.tsv` |
| QDRP multi-page synthetic | `risk_vs_score_recover=-0.1240` at `budget=2` | `reports/autoresearch_dynamic_recursive_log.tsv` |
| QDRP hybrid synthetic | `hybrid_vs_score_recover=+0.0005` at `budget=2` | `reports/autoresearch_dynamic_recursive_log.tsv` |
| TRIC diagnostics | `recursive_gain_vs_linear=-3.1270`, `-0.5619`, `-0.2633` | `reports/autoresearch_dynamic_recursive_log.tsv` |
| Fresh local rerun `two_pass_k8` | `0.000` | `results/query_exploit_validity/...validity_two_pass_mistral_4k...json` |
| Fresh local rerun `two_pass_k16` | `0.000` | `results/query_exploit_validity/...validity_two_pass_mistral_4k...json` |
| Fresh local rerun controls | all `0.000` | `results/query_exploit_validity/...validity_controls_mistral_4k...json` |

## 3. What Changed After The Review Loop

처음에는 `two_pass`를 selective compression method처럼 밀 수 있다고 보기 쉬웠다. 지금은 그렇게 쓰면 안 된다는 점이 분명해졌다.

### 3.1 The main positive result stayed

`two_pass`가 bounded 4K NIAH에서 강하게 작동한다는 사실은 남았다. `query_dequant`와 `sharp_temp`는 같은 harness에서 살아남지 못했다. 이 점 자체는 명확하다.

다만 이제 이 문장에는 단서가 붙는다. **earlier remote result 기준으로는** 그렇다. current checked revision의 fresh rerun은 그 positive result를 그대로 재현하지 못했다.

### 3.2 The headline claim got narrower

문제는 구현 경로다. 현재 `exp_query_exploit.py`는 cache update 뒤 `key_fp16 = key_states.clone()`를 잡아 둔 다음, low-bit score로 고른 위치를 same wrapper 안에서 FP16으로 다시 넣는다. checked revision의 `exp_cliffkv_niah.py`도 `key_fp16 = key_states.clone()` 후 promoted token만 attention path에서 higher precision으로 대체한다. 즉, 두 스크립트 모두 **stored compressed cache를 읽는 실험이 아니라, FP16 cache 위에서 attention read를 교정하는 실험**이다.

### 3.3 Some older wording is now stale

`reports/autoresearch_selective_refine_log.tsv`의 “cache-aware cliffkv smoke” 문구는 현재 checked revision의 코드와 맞지 않는다. 로그 한 줄보다 현재 코드가 우선이다. 지금 시점에서 storage-valid claim을 싣는 것은 위험하다.

### 3.4 Fresh rerun exposed a reproducibility gap

새 local bounded rerun은 더 강한 경고다. `two_pass_k8`와 `two_pass_k16`는 모두 `0.000`이었지만, selector diagnostic은 needle overlap을 높게 보였다. `two_pass_k8`의 mean needle hit rate는 `0.708`, `two_pass_k16`은 `0.823`였다. 반면 controls는 모두 `0.000`이었고, `recent_k`의 needle hit는 `0.0`, `sink_k`는 sink overlap만 높았다.

이 패턴은 “selector가 needle 쪽을 자주 포함한다”와 “generation answer가 실제로 회복된다”가 같은 문장이 아니라는 뜻이다. 즉, 지금은 method claim보다 **reproducibility audit**이 먼저다.

## 4. Direction-by-Direction Verdict

### 4.1 `two_pass`

직접 verdict는 이렇다. **좋은 diagnostic candidate, 아직 method는 아니고 지금은 reproducibility audit 대상이다.**

왜 완전히 버리지 않았는지는 단순하다. earlier remote run에서는 실제로 이겼고, fresh rerun에서도 selector는 needle 쪽을 더 잘 잡는다. 왜 밀어붙이지 않는지도 단순하다. generation success가 현재 revision에서 재현되지 않고, 저장-valid path도 아직 없기 때문이다.

### 4.2 `query_dequant`

현재는 중단이 맞다. 4K same-harness에서 baseline을 넘지 못했고, score-MSE나 top-k retention 같은 더 직접적인 diagnostic도 아직 없다.

### 4.3 `sharp_temp`

중단이 맞다. tested temperature 전부 `0.000`이었다. 현재 failure mode는 단순 flattening 보상으로 해결되지 않는다.

### 4.4 QDRP

headline direction으로는 내린다. synthetic `budget=1` toy에서는 꽤 좋아 보이지만, `budget=2`에서 pure risk가 무너지고 hybrid도 사실상 raw score와 동률이다. 기존 real-trace flip calibration에서는 raw score가 더 낫다.

### 4.5 TRIC

지금은 접는 것이 맞다. recursive predictor가 copy baseline은 이겨도 shared linear baseline은 못 이긴다. 이 상태에서 end-to-end GPU를 더 쓰는 것은 연구가 아니라 희망사항이다.

## 5. Failure Modes Still Open

이번 리뷰 루프에서 정리된 남은 가능 원인은 다섯 가지다.

### 5.1 Recency artifact

`two_pass`가 needle을 찾는 것이 아니라 query 바로 앞 토큰을 밀어 올렸을 수 있다.

### 5.2 Attention sink artifact

초기 sink token이 선택되면서 binary success를 만든 것일 수 있다.

### 5.3 Selection-repair coupling

selector와 repair가 같은 wrapper 안에 있어, 실제 시스템 구현보다 유리한 실험 경로일 수 있다.

### 5.4 Saturated 4K binary NIAH

현재 headline result는 사실상 `3 depths x 2 repeats = 6 trials` per setting이다. `k=8`부터 모두 `1.000`인 grid는 frontier를 말해 주지 못한다.

### 5.5 Budget accounting ambiguity

현재 `effective_avg_bits_upper_bound`는 stored budget이 아니라 accessed precision의 상계 proxy다. equal-storage, equal-HBM, equal-extra-compute를 분리하지 않으면 예산 비교가 무너진다.

### 5.6 Reproducibility gap

earlier remote win과 current checked revision의 fresh local rerun이 충돌한다. 이 상태에서 더 큰 frontier나 새 theory를 붙이면 연구가 아니라 로그 수집이 된다.

## 6. Strongest Defensible Claim

지금 바로 써도 되는 가장 강한 문장은 다음 하나다.

“earlier remote attention-path diagnostic harness에서는 `two_pass` selective repair가 bounded 4K NIAH에서 2-bit baseline보다 강했고, `query_dequant`와 `sharp_temp`는 같은 조건에서 실패했다. 그러나 current checked revision의 fresh local rerun은 그 win을 아직 안정적으로 재현하지 못했다.”

이보다 큰 문장, 특히 “compression method”, “stored-cache refinement”, “publication-ready novelty” 같은 표현은 아직 안 된다.

## 7. Plan Going Forward

다음 순서는 세 단계가 맞다.

1. earlier remote run과 current checked revision 사이의 재현성 gap을 먼저 감사한다.
2. 그 다음 `two_pass` vs `recent_k` / `sink_k` / `random_k` control을 같은 CLI로 다시 맞춘다.
3. reproduction이 안정되면 8K/16K frontier로 간다.
4. 그 다음에야 true storage-valid selective refinement를 구현한다.

QDRP와 TRIC는 그 뒤가 아니라 그 아래다. QDRP는 real trace에서 raw score를 이기기 전까지 diagnostic에 묶어 두고, TRIC는 real activation에서 shared linear를 이기기 전까지 보류한다.

## 8. Brutal Bottom Line

지금 상태는 top-venue paper method가 아니다.

- `two_pass`: 살릴 가치가 있는 **diagnostic candidate**, 하지만 지금은 재현성 감사를 먼저 받아야 한다
- QDRP: 아직 **가설**
- TRIC: 현재는 **탈락**

이 wave가 다시 살아나는 조건은 두 개다. **earlier remote win이 current code에서 재현되어야 하고, storage-valid selective refinement를 구현했는데도 gain이 남아야 한다.** 둘 중 하나라도 아니면 이 결과는 method paper보다 failure-analysis appendix나 diagnostic section으로 정리하는 편이 맞다.
