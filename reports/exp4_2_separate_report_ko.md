# Exp 4-2 별도 보고서

**프로젝트**: Boltzmann Attention / FOKVQ  
**작성일**: 2026-04-01  
**범위**: `WikiText-2` sliding-window PPL benchmark, `gpt2-medium`, `max_eval_tokens=4096`  
**주요 비교 대상**: `uniform`, `KIVI-style`, `turboquant-style`, `FOKVQ(PCA)`  

## 1. Executive Summary

이번 추가 실험의 결론은 명확하다. 현재 구현의 `FOKVQ`는 공정한 비트 예산 하에서 `KIVI-style` 및 `turboquant-style`보다 우수하지 않다. `4-bit`에서는 거의 모든 방법이 비슷해져 `FOKVQ`의 이점이 드러나지 않았고, `2-bit`에서는 `PCA` 축 정렬이 `random` 및 `identity`보다는 낫지만 강한 baseline을 넘지는 못했다.

즉, 현재 결과는 "새 방법이 기존 방법보다 좋다"는 종류의 주장보다는, "축 정렬 효과는 존재하지만 strong baseline을 이길 만큼 충분하지는 않다"는 방법론적 해석에 더 가깝다.

## 2. 실험 프로토콜

- 모델: `openai-community/gpt2-medium`
- 데이터: `WikiText-2`
- 평가 방식: sliding-window causal LM perplexity
- 공통 설정:
  - calibration samples: `8`
  - attention implementation: `eager`
  - seed: `42`
- 비교 방법:
  - `uniform`: 대칭 uniform quantization
  - `kivi`: sequence-dimension asymmetric quantization
  - `turboquant-style`: random orthogonal rotation + Lloyd-Max scalar codebook
  - `fokvq`: PCA basis 기반 mixed-precision axis quantization
- 주의:
  - `turboquant-style`은 논문 영감 기반 내부 구현이며, 공식 TurboQuant 완전 재현으로 주장하면 안 된다.
  - `fokvq`는 공정한 평균 비트 예산을 강제하도록 수정된 이후의 결과만 본 보고서에 반영한다.

## 3. 핵심 결과

### 3.1 4-bit main comparison

| Method | PPL |
|---|---:|
| FP16 | 26.8638 |
| KIVI-style | 26.8770 |
| uniform | 26.8983 |
| FOKVQ | 26.9954 |
| turboquant-style | 27.0020 |

출처:
- `/scratch/amlt_code/outputs/exp4_2_followup_4bit/main/gpt2-medium-4bit-main_standard_ppl.json`

해석:
- `4-bit`에서는 방법 간 차이가 매우 작다.
- `FOKVQ`는 `KIVI-style`보다 좋지 않았고, `uniform`보다도 낫지 않았다.
- 따라서 `4-bit`는 현재 구현의 주력 주장 영역이 아니다.

### 3.2 4-bit axis ablation

| Method | PPL |
|---|---:|
| FP16 | 26.8638 |
| uniform | 26.8983 |
| random basis | 26.9608 |
| FOKVQ(PCA) | 26.9954 |
| identity basis | 27.0862 |

출처:
- `/scratch/amlt_code/outputs/exp4_2_followup_4bit/axis_ablation/gpt2-medium-4bit-axis-ablation_standard_ppl.json`

해석:
- `identity`는 가장 나쁘지만, `random`이 `PCA`보다 약간 더 낫다.
- 따라서 `4-bit`에서는 `PCA basis`의 구조적 우위가 보이지 않는다.

### 3.3 2-bit fair comparison, `topk=0.25`

#### Main (`context_len=256`, `stride=128`)

| Method | PPL |
|---|---:|
| FP16 | 26.8638 |
| KIVI-style | 27.7507 |
| turboquant-style | 27.7524 |
| uniform | 32.4926 |
| FOKVQ | 48.2813 |

#### Long context 512

| Method | PPL |
|---|---:|
| FP16 | 22.5366 |
| turboquant-style | 23.3546 |
| KIVI-style | 23.9904 |
| uniform | 27.3654 |
| FOKVQ | 45.9687 |

#### Long context 1024

| Method | PPL |
|---|---:|
| FP16 | 18.3817 |
| turboquant-style | 19.3868 |
| KIVI-style | 20.4618 |
| uniform | 22.4624 |
| FOKVQ | 42.2597 |

출처:
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/main/gpt2-medium-2bit-topk025-main_standard_ppl.json`
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/longctx512/gpt2-medium-2bit-topk025-longctx512_standard_ppl.json`
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/longctx1024/gpt2-medium-2bit-topk025-longctx1024_standard_ppl.json`

해석:
- 원래 설계에 가까운 `topk=0.25`는 현재 구현에서 실패했다.
- 긴 문맥으로 가도 회복되지 않았다.

### 3.4 2-bit fair comparison, `topk=0.5`

#### Main (`context_len=256`, `stride=128`)

| Method | PPL |
|---|---:|
| FP16 | 26.8638 |
| KIVI-style | 27.7507 |
| turboquant-style | 27.7524 |
| uniform | 32.4926 |
| FOKVQ | 34.5466 |

#### Long context 512

| Method | PPL |
|---|---:|
| FP16 | 22.5366 |
| turboquant-style | 23.3546 |
| KIVI-style | 23.9904 |
| uniform | 27.3654 |
| FOKVQ | 28.8335 |

#### Long context 1024

| Method | PPL |
|---|---:|
| FP16 | 18.3817 |
| turboquant-style | 19.3868 |
| KIVI-style | 20.4618 |
| uniform | 22.4624 |
| FOKVQ | 24.0110 |

출처:
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/main/gpt2-medium-2bit-topk050-main_standard_ppl.json`
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/longctx512/gpt2-medium-2bit-topk050-longctx512_standard_ppl.json`
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/longctx1024/gpt2-medium-2bit-topk050-longctx1024_standard_ppl.json`

해석:
- `topk=0.5`는 `0.25`보다 훨씬 낫다.
- 그러나 여전히 `uniform`, `KIVI-style`, `turboquant-style`보다 좋지 않다.
- 따라서 "2-bit에서 FOKVQ가 강하다"는 주장은 현재 성립하지 않는다.

### 3.5 2-bit axis ablation

#### `topk=0.25`

| Method | PPL |
|---|---:|
| FOKVQ(PCA) | 48.2813 |
| random basis | 84.4536 |
| identity basis | 412.4942 |

#### `topk=0.5`

| Method | PPL |
|---|---:|
| FOKVQ(PCA) | 34.5466 |
| random basis | 52.9955 |
| identity basis | 149.8334 |

출처:
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk025/axis_ablation/gpt2-medium-2bit-topk025-axis-ablation_standard_ppl.json`
- `/scratch/amlt_code/outputs/exp4_2_followup_2bit_topk050/axis_ablation/gpt2-medium-2bit-topk050-axis-ablation_standard_ppl.json`

해석:
- 저비트에서는 `PCA basis`가 분명히 `random` 및 `identity`보다 낫다.
- 이 결과는 "축 선택 자체의 효과"를 지지한다.
- 하지만 이 효과만으로 strong baseline을 이기지는 못한다.

## 4. 현재 주장 가능 범위

### 4.1 주장 가능한 것

1. `FOKVQ`에는 저비트에서 의미 있는 `basis-selection effect`가 있다.
2. `PCA basis`는 naive `random` 또는 `identity` basis보다 낫다.
3. 공정한 평균 비트 예산을 강제하면 이전의 일부 개선은 bit-budget artifact였음이 드러난다.

### 4.2 주장하면 안 되는 것

1. `FOKVQ`가 `KIVI-style`보다 우수하다.
2. `FOKVQ`가 `turboquant-style`보다 우수하다.
3. `4-bit`에서 `PCA basis`가 분명한 우위를 보인다.
4. 현재 `turboquant-style` 구현이 공식 TurboQuant 재현이다.

## 5. 연구적으로 가지는 의미

현재 결과는 "성능 우위"보다는 "부정적이지만 유의미한 분석 결과"에 가깝다.

핵심 의미는 다음과 같다.
- 축 정렬은 무의미하지 않다.
- 그러나 단순한 PCA 기반 mixed-precision axis quantization만으로는 low-bit KV-cache quantization의 strong baseline을 넘기 어렵다.
- 따라서 다음 단계의 기여는 새로운 수식이나 서술보다, 왜 PCA 정렬 효과가 strong baseline으로 이어지지 않는지에 대한 구조적 분석이어야 한다.

## 6. 운영상 발견

원격 실험 운영에서도 중요한 발견이 있었다.

- `OMP_NUM_THREADS=8`
- `MKL_NUM_THREADS=8`
- `OPENBLAS_NUM_THREADS=8`
- `NUMEXPR_NUM_THREADS=8`

위 제한을 두지 않은 병렬 실행에서는 wall-clock runtime이 약 `9-10분` 수준까지 불어났다.  
같은 프로토콜에서 thread 제한을 걸면 배치당 `6-12초` 수준으로 줄었다.

즉, 최근 원격 실험에서는 GPU보다 CPU-thread oversubscription이 더 큰 병목이었다. 이후 배치는 이 제한을 기본값처럼 유지하는 것이 맞다.

## 7. 다음 액션 제안

우선순위는 다음 순서가 적절하다.

1. 논문 본문에서는 우월성 주장 대신 "mechanistic / methodological finding"으로 정리한다.
2. `FOKVQ`를 더 밀고 싶다면, 단순 `topk` 조정보다 메커니즘 자체를 바꿔야 한다.
3. 추가 실험을 한다면 다음 세 가지가 우선이다.
   - layer-wise or head-wise adaptive basis selection
   - PCA coefficient clipping / normalization / robust scaling
   - memory or latency까지 포함한 trade-off 분석

## 8. 최종 결론

현재 시점에서의 가장 정직한 결론은 다음과 같다.

> `FOKVQ`는 저비트에서 `PCA` 기반 축 정렬 효과를 보이지만, 공정한 비교 조건에서는 `KIVI-style` 및 `turboquant-style`보다 우수하지 않다.

즉, 이 결과를 그대로 논문화한다면 "새로운 최고 성능 방법"이 아니라, "축 기반 KV quantization의 한계와 조건부 효과를 드러내는 분석 보고"로 정리하는 편이 맞다.
