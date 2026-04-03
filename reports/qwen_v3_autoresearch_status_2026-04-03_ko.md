# Qwen v3 Autoresearch Status

**프로젝트**: Boltzmann Attention / FOKVQ  
**작성일**: 2026-04-03  
**대상 모델**: `Qwen/Qwen2.5-7B-Instruct`  
**평가 프로토콜**: `Exp 4-2 v3`, `post_rope`, full-K quantization, WikiText-2 PPL  
**핵심 질문**: `Qwen/GQA/RoPE` 환경에서 구조적 축 선택 효과가 실제로 남아 있는가, 그리고 그 효과를 Lie/Hamiltonian 언어로 어디까지 설명할 수 있는가

## 1. 요약

현재 상태는 두 층으로 나눠서 읽어야 한다.

- long-budget practical run에서는 `kivi_residual`이 여전히 leader다.
- bounded mechanistic smoke에서는 `fokvq_e2`가 현재 가장 강한 structured candidate다.
- 따라서 지금의 정직한 메시지는 "`이미 SOTA를 이겼다`"가 아니라, "`Qwen/GQA/RoPE`에서도 구조적 축 선택 효과는 남아 있지만 practical win과 이론적 일반화는 별도로 검증해야 한다"이다.

이 해석은 theory framing에도 직접 연결된다.

- `Lie` 해석은 main framing으로 유지할 수 있다.
- `Hamiltonian` 해석은 아직 method claim이 아니라 descriptive diagnostic이어야 한다.
- 즉, quadratic energy drift, phase drift, symplectic-form drift 같은 보조 지표가 bounded mechanistic 결과와 같이 움직일 때에만 설명적 근거로 쓴다.

## 2. 프로토콜

- 모델: `Qwen/Qwen2.5-7B-Instruct`
- 데이터: `WikiText-2`
- 평가:
  - non-overlapping chunk
  - `post_rope` attention wrapper
  - full K quantization
- practical run:
  - `context_len=2048`
  - 16k budget 기준 결과
- bounded mechanistic smoke:
  - `context_len=256`
  - `max_eval_tokens=512`
  - `rotation_mechanistic` preset

주의:

- `kivi_residual`, `turboquant`, `turboquant_rand`는 **same-harness proxy**다.
- 따라서 공식 external reproduction이라고 쓰면 안 되고, 이 문서에서도 same-harness control로만 해석한다.
- `Hamiltonian`은 현재 실험군의 중심 승부 항목이 아니라, 구조 보존을 설명하는 보조 진단 축이다.

## 3. 확정 결과

### 3.1 Qwen 16k full run: practical ranking

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| FP16 | 6.7544 | 6.7544 |
| kivi_residual | 7.2060 | 6.7738 |
| turboquant_rand | 7.5587 | 7.0212 |
| turboquant | 7.5671 | 7.0900 |
| quip | 9.1795 | 7.1394 |
| fokvq_e2 | 8.7232 | 8.4917 |
| fokvq | 9.6555 | 7.4720 |

해석:

- practical leader는 여전히 `kivi_residual`이다.
- `turboquant_rand`와 `turboquant`는 강한 control이지만, 현재 same-harness에서는 `kivi_residual`보다 뒤에 있다.
- `fokvq_e2`는 plain `fokvq`보다 낫지만, long-budget practical winner라고 부를 수준은 아니다.

### 3.2 Qwen 2048 smoke: residual hybrid

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| FP16 | 6.0769 | 6.0769 |
| kivi_residual | 6.5734 | 6.1431 |
| fokvq_e2 | 7.7454 | 7.3983 |
| fokvq_e2_residual | 7.4612 | 7.4540 |

해석:

- `fokvq_e2_residual`은 `3-bit`에서 plain `fokvq_e2`보다 좋아졌다.
- 그러나 `4-bit`에서는 개선이 유지되지 않았다.
- 즉, `recent tail + structured prefix`는 중요한 bias이지만, 그것만으로 practical winner가 되지는 않는다.

### 3.3 Qwen 2048 smoke: 초기 Lie / complex 후보

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| kivi_residual | 6.5734 | 6.1431 |
| rope_unitary | 14.3512 | 13.9959 |
| rope_magphase | 750.0469 | 424.5279 |

해석:

- `rope_unitary`는 pair-local `SO(2)` 제약만으로는 충분하지 않았다.
- `rope_magphase`는 실질적으로 붕괴했다.
- 따라서 "RoPE-aware"라는 이름만 붙인 단순 변형은 설명도 성능도 모두 부족하다.

### 3.4 Qwen bounded mechanistic smoke: structured basis effect는 살아 있다

이 작은 smoke는 practical run이 아니라 `rotation_mechanistic` preset으로 구조적 축 선택 효과를 직접 보는 실험이다.

| Method | 2-bit PPL | 3-bit PPL |
|---|---:|---:|
| FP16 | 11.8666 | 11.8666 |
| identity | 19.1351 | 12.6156 |
| random | 26.2928 | 12.6578 |
| fokvq | 15.1640 | 12.2210 |
| fokvq_e2 | 12.4561 | 11.5092 |
| kivi_residual | 13.1795 | 11.8421 |
| turboquant_rand | 14.5626 | 11.7115 |
| complex_unitary_residual | 15.3706 | 13.0332 |
| banded_complex_unitary_residual | 17.2121 | 12.0620 |

해석:

- bounded mechanistic regime에서는 `fokvq_e2`가 현재 가장 강한 structured candidate다.
- `fokvq_e2`는 `identity`, `random`뿐 아니라 `kivi_residual`, `turboquant_rand`보다도 좋았다.
- complex/unitary 계열은 이번 smoke에서 주력 경로로 승격될 근거를 아직 만들지 못했다.

## 4. 현재 해석

### 4.1 무엇이 확인됐는가

- `Qwen/GQA/RoPE`에서는 `recent residual tail`이 매우 강한 inductive bias다.
- 그와 별개로 bounded mechanistic regime에서는 structured basis effect가 실제로 남아 있다.
- reconstruction MSE를 줄이는 것만으로 PPL이 좋아지지 않는다는 점도 계속 확인되고 있다.

### 4.2 무엇이 아직 확인되지 않았는가

- long-budget practical winner
- complex/unitary 계열의 확실한 우위
- Hamiltonian 언어의 직접적인 실험적 정당화

즉, 지금의 올바른 방향은 세 가지다.

- `fokvq_e2`를 practical track에서 다시 올려 본다.
- Lie 해석은 main framing으로 유지한다.
- Hamiltonian 해석은 descriptive diagnostic으로만 검증한다.

### 4.3 왜 practical winner가 되지 못하는가: 현재 가설

현재까지의 결과를 가장 보수적으로 읽으면, 원인은 하나가 아니라 네 갈래일 가능성이 높다.

첫째, `recency bottleneck`이다. Qwen에서는 최근 tail을 FP16으로 남기는 bias가 매우 강하다. 그래서 구조적 basis가 좋아도, 최신 토큰 근처의 cache를 얼마나 보존하느냐가 practical PPL을 더 크게 좌우할 수 있다.

둘째, `post-rotation quantizer bottleneck`이다. bounded mechanistic run에서 `fokvq_e2`가 강하다는 사실은 basis 자체는 유의미하다는 뜻이다. 그런데 long-budget practical run에서는 그 이득이 충분히 남지 않는다. 이 경우 가장 자연스러운 해석은, 회전 뒤에 붙는 scalar quantizer가 attention 목적함수와 정확히 맞지 않는다는 것이다.

셋째, `GQA bottleneck`이다. Qwen에서는 하나의 KV head가 여러 Q head와 연결된다. 이 구조에서는 K-only basis가 평균적으로는 좋아도, head-group별 query mismatch가 practical 성능을 깎을 수 있다.

넷째, `metric bottleneck`이다. PPL은 평균 next-token likelihood를 본다. 그런데 cache quantization의 장점이 retrieval geometry나 특정 위치의 attention structure에 더 가깝다면, PPL만으로는 그 이득이 희석될 수 있다.

이 네 가설은 서로 배타적이지 않다. 현재로서는 `recency bottleneck + quantizer bottleneck` 조합이 가장 유력하고, `GQA bottleneck`과 `metric bottleneck`이 그 뒤를 잇는다.

## 5. 다음 실험 방향

### 5.1 우선순위 1

`fokvq_e2` practical promotion

- bounded mechanistic win이 medium-budget same-harness PPL에서도 유지되는지 확인
- 최소 비교군:
  - `kivi_residual`
  - `turboquant_rand`
  - `fokvq`
  - `fokvq_e2`
- 여기서도 무너지면 `fokvq_e2`는 practical method가 아니라 mechanistic evidence로 남긴다

### 5.2 우선순위 2

Hamiltonian descriptive probe

- RoPE pair를 `(q, p)` canonical pair처럼 읽었을 때
- quadratic energy drift
- pair phase drift
- symplectic-form drift
를 계산해 bounded mechanistic PPL 순위와 비교한다.

이 단계는 "해밀토니안이 맞다"를 증명하는 단계가 아니다. 어떤 방법이 canonical pair 구조를 덜 망가뜨리는지 설명하는 보조 진단 단계다.

### 5.3 우선순위 3

Practical-gap explanation track

- `fokvq_e2` vs `fokvq_e2_residual`로 recency bottleneck을 본다.
- sampled attention-logit distortion과 top-k attention overlap으로 quantizer bottleneck을 본다.
- grouped Q-vs-KV mismatch를 측정해 GQA bottleneck을 본다.
- PPL과 NIAH가 어긋나는지 비교해 metric bottleneck을 본다.

### 5.4 우선순위 4

`complex_unitary_residual`, `banded_complex_unitary_residual`

- complex/Lie 계열은 계속 보되, 더 이상 default lead candidate로 두지 않는다.
- Hamiltonian/Lie diagnostics에서 의미 있는 보조 신호가 있을 때만 다시 승격한다.

## 6. 결론

현재 결과는 단순 negative result라기보다, claim hierarchy를 다시 세우게 만든 결과다.

- practical claim:
  - 아직 `kivi_residual`을 long-budget same-harness에서 이기지 못했다.
- mechanistic claim:
  - `Qwen/GQA/RoPE`에서도 structured basis effect는 bounded regime에서 살아 있다.
- theory claim:
  - `Lie` 해석은 유지 가능하다.
  - `Hamiltonian` 해석은 아직 descriptive support에 머물러야 한다.

즉, 지금의 가장 중요한 메시지는 "이미 baseline보다 좋다"가 아니라, "어떤 structured effect는 살아남고 어떤 이론 언어는 아직 보조 설명에 머무는가"를 경계 조건으로 분리해 보여 주는 데 있다.
