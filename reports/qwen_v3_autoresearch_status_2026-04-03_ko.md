# Qwen v3 Autoresearch Status

**프로젝트**: Boltzmann Attention / FOKVQ  
**작성일**: 2026-04-03  
**대상 모델**: `Qwen/Qwen2.5-7B-Instruct`  
**평가 프로토콜**: `Exp 4-2 v3`, `post_rope`, full-K quantization, WikiText-2 PPL  
**핵심 질문**: `Qwen/GQA/RoPE` 환경에서 `Lie / complex rotation` 기반 방법이 `kivi_residual`을 이길 수 있는가

## 1. 요약

현재까지의 결론은 명확하다.

- 현재 same-harness 기준 practical leader는 `kivi_residual`이다.
- `fokvq`, `fokvq_e2`, `quip`, `turboquant`, `turboquant_rand` 모두 `Qwen 3/4-bit`에서 이를 넘지 못했다.
- `rope_unitary`, `rope_magphase`처럼 RoPE-aware를 직접 노린 초기 Lie/복소수 계열 후보도 현재 구현에서는 실패했다.
- 다만 `fokvq_e2_residual`은 `3-bit`에서 plain `fokvq_e2`보다 개선되어, "recent residual tail + structured prefix"라는 방향 자체는 계속 볼 가치가 있다.

따라서 현재 논리적으로 맞는 다음 단계는 다음 두 가지다.

1. `phase 직접 양자화`가 아니라, **복소수 Hermitian covariance 기반 unitary basis**에서 양자화하는 진짜 complex-Lie 방향으로 이동한다.
2. practical 측면에서는 `kivi_residual`의 강점을 인정하고, **recent tail 보존 + 오래된 prefix에만 structured transform**을 넣는 하이브리드 계열을 계속 검증한다.

## 2. 프로토콜

- 모델: `Qwen/Qwen2.5-7B-Instruct`
- 데이터: `WikiText-2`
- 평가:
  - non-overlapping chunk
  - `context_len=2048`
  - `post_rope` attention wrapper
  - full K quantization
- dtype: `bfloat16`
- 주요 비교군:
  - `kivi_residual`
  - `quip`
  - `turboquant`
  - `turboquant_rand`
  - `fokvq`
  - `fokvq_e2`
  - `fokvq_e2_residual`
  - `rope_unitary`
  - `rope_magphase`

주의:

- `kivi_residual`, `turboquant`, `turboquant_rand`는 **same-harness proxy**이다.
- 따라서 공식 external reproduction이라고 쓰면 안 되고, 이 보고서에서도 same-harness control로만 해석한다.

## 3. 확정 결과

### 3.1 Qwen 16k full run

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| FP16 | 6.7544 | 6.7544 |
| kivi_residual | 7.2060 | 6.7738 |
| fokvq | 9.6555 | 7.4720 |
| fokvq_e2 | 8.7232 | 8.4917 |
| quip | 9.1795 | 7.1394 |
| turboquant | 7.5671 | 7.0900 |
| turboquant_rand | 7.5587 | 7.0212 |

해석:

- `kivi_residual`이 가장 강한 control이다.
- `turboquant_rand`는 `turboquant`보다 근소하게 낫지만, 여전히 `kivi_residual`보다 뒤처진다.
- `fokvq_e2`는 `fokvq`보다 3-bit에서 좋아졌지만, 4-bit에서는 오히려 매우 약하다.

### 3.2 Qwen 2048 smoke: residual hybrid

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| FP16 | 6.0769 | 6.0769 |
| kivi_residual | 6.5734 | 6.1431 |
| fokvq_e2 | 7.7454 | 7.3983 |
| fokvq_e2_residual | 7.4612 | 7.4540 |

해석:

- `fokvq_e2_residual`은 `3-bit`에서 `fokvq_e2`보다 좋아졌다.
- 그러나 `4-bit`에서는 오히려 소폭 악화되었다.
- 따라서 이 방법은 practical winner는 아니지만, "recent tail + structured prefix" 방향의 mechanistic signal은 제공한다.

### 3.3 Qwen 2048 smoke: Lie / complex 초기 후보

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| kivi_residual | 6.5734 | 6.1431 |
| rope_unitary | 14.3512 | 13.9959 |
| rope_magphase | 750.0469 | 424.5279 |

해석:

- `rope_unitary`는 pair-local `SO(2)` 제약만으로는 충분하지 않았다.
- `rope_magphase`는 실질적으로 붕괴했다.
- 즉, "RoPE-aware"라는 이름만 붙인 단순 변형은 부족하며, 복소수 공간 전체에서의 일관된 unitary basis가 필요하다는 반증으로 볼 수 있다.

## 4. 현재 해석

### 4.1 무엇이 확인됐는가

- `Qwen/GQA/RoPE`에서는 `recent residual tail`이 매우 강한 inductive bias다.
- `MHA/GPT-2`에서 보였던 rotation 계열의 직관은 `Qwen`에 그대로 이식되지 않는다.
- reconstruction MSE를 줄이는 것만으로 PPL이 좋아지지 않는다.

### 4.2 무엇이 아직 남아 있는가

- Lie 군 해석 자체가 무의미하다고 보기는 어렵다.
- 다만 현재까지 실패한 것은:
  - generic real rotation
  - pair-local rotation
  - naive magnitude-phase quantization

즉, 남은 타당한 방향은 **복소수 공간의 Hermitian 구조를 보존하는 unitary transform**이다.

## 5. 다음 실험 방향

### 5.1 우선순위 1

`complex_unitary_residual`

- RoPE pair를 complex channel로 해석
- old prefix에서 Hermitian covariance 추정
- unitary eigenbasis로 회전
- complex basis에서 real/imag를 양자화
- recent tail은 FP16 유지

이 방향은 현재까지 시도한 후보 중 가장 직접적으로 "복소수 회전 Lie 군" 해석에 부합한다.

### 5.2 우선순위 2

`kivi_residual + structured prefix calibration`

- practical leader인 `kivi_residual`을 baseline이 아니라 출발점으로 삼아야 한다.
- structured transform은 전체 cache가 아니라 old prefix에만 제한해야 한다.

## 6. 결론

현재 결과는 negative에 가깝지만, completely dead는 아니다.

- practical claim:
  - 아직 `kivi_residual`을 이길 방법을 찾지 못했다.
- scientific claim:
  - `Qwen/GQA/RoPE`에서는 단순 PCA/real rotation이 통하지 않으며,
    recency-preserving 구조와 complex/unitary geometry를 함께 고려해야 한다.

즉, 지금의 가장 중요한 메시지는 "기존 방법보다 좋다"가 아니라,
"어떤 Lie/rotation 해석은 실패하고, 어떤 geometry만이 살아남는가"를 경계 조건으로 밝혀내는 쪽에 가깝다.
