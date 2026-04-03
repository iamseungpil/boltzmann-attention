# Exp 4-2 Lie-Rotation Update Report

**프로젝트**: Boltzmann Attention / FOKVQ  
**작성일**: 2026-04-03  
**범위**: `GPT-2` same-harness 결과, `Qwen2.5-7B` full-K 결과, Lie/rotation 해석 기준의 재정리

## Executive Summary

이번 정리의 핵심은 단순하다. 현재 결과는 `FOKVQ가 baseline을 이긴다`는 논문보다, `회전 기반 KV-cache 양자화를 어떤 수학적 문제로 봐야 하는가`를 정리하는 논문에 더 잘 맞는다.

확정적으로 말할 수 있는 것은 네 가지다. 첫째, K-space의 비등방성과 구조적 축 선택 효과는 실재한다. 둘째, GPT-2에서는 `PCA basis`가 `random`과 `identity`보다 확실히 낫다. 셋째, Qwen/GQA의 long-budget practical run에서는 `kivi_residual`이 여전히 leader다. 넷째, bounded Qwen mechanistic smoke에서는 `fokvq_e2`가 `kivi_residual`과 `turboquant_rand`를 앞서는 새로운 positive signal이 확인됐다.

따라서 가장 정직한 현재 결론은 다음과 같다. `Lie 군 기반 회전 선택 해석`은 의미가 있지만, 그 해석이 자동으로 `완성된 low-bit SOTA quantizer`를 만들어 주지는 않는다. 지금의 병목은 회전 존재 여부보다 `어떤 generator가 살아남는가`와 `회전 후 quantizer가 충분히 강한가`에 있다.

## Fact Base와 출처 정책

이번 보고서는 아래 자료만 근거로 삼는다.

| 구분 | 현재 사용 여부 | 근거 |
|---|---|---|
| `results/v3/gpt2-medium_full_quant_ppl_v3.json` | 사용 | 현재 레포에 남아 있는 GPT-2 v3 raw JSON |
| `reports/exp4_2_separate_report_ko.md` | 사용 | GPT-2 same-harness 표와 원격 raw artifact 경로를 함께 정리한 보고서 |
| `reports/exp4_2_autoresearch_log.md` Iteration 24-29 | 사용 | 수정된 Qwen queue의 최종 수치와 해석이 남아 있는 작업 로그 |
| `results/v3/qwen2.5-7b_full_quant_ppl_v3.json` | 사용 안 함 | 과거 실패 런으로, hook/runtime error가 포함된 stale JSON |
| `/tmp/exp4_2_v3_rotation_smoke_20260403/...json` | 제한적 사용 | 현재 harness wiring과 `benchmark_meta` 기록 여부를 확인하는 smoke 전용 자료 |

즉, 이 보고서에서 Qwen practical 표는 오래된 `results/v3/qwen2.5-7b_full_quant_ppl_v3.json`이 아니라, 수정 후 작업 로그와 별도 상태 보고서를 source of truth로 사용한다.

## 질문별 요약

| 질문 | 의도 | 관찰 결과 | 현재 판단 |
|---|---|---|---|
| `Q1. 회전 선택은 실제로 중요한가?` | basis effect 자체를 검증 | GPT-2 2-bit에서 `identity > random > PCA` 순서의 큰 차이 확인 | 지지됨 |
| `Q2. 현재 FOKVQ가 practical winner인가?` | low-bit main table에서 baseline 우위 확인 | GPT-2 2-bit, Qwen 3/4-bit 모두 strong control을 넘지 못함 | 지지되지 않음 |
| `Q3. GQA/RoPE에서 어떤 bias가 중요한가?` | 현대 모델에서 살아남는 구조 확인 | long-budget practical에서는 `kivi_residual`, bounded mechanistic에서는 `fokvq_e2`가 가장 강함 | 부분 지지 |
| `Q4. 단순 RoPE-aware Lie 후보가 충분한가?` | 복소수/회전 해석의 직접 구현 검증 | `rope_unitary`, `rope_magphase`는 크게 실패했고, 현재 bounded smoke에서도 complex/unitary 계열은 lead가 아님 | 반증됨 |
| `Q5. Hamiltonian 언어를 본문에 둘 수 있는가?` | Lie보다 더 강한 이론 언어의 타당성 점검 | 아직 직접 검증 없음, descriptive diagnostic으로만 사용 가능 | 제한적 |
| `Q6. 현재 harness는 새 가설을 시험할 준비가 되었는가?` | smoke-critic gate 확인 | self-test 통과, mechanistic smoke 통과, preset metadata 기록 확인 | 예 |

## 1. 무엇이 바뀌었는가

초기 문서군은 두 종류의 주장을 섞고 있었다.

- 방법론 주장: `PCA 기반 FOKVQ가 기존 방법보다 좋다`
- 해석 주장: `회전 기반 양자화는 Lie 군 작용으로 볼 수 있고, 좋은 회전 선택이 핵심이다`

현재까지의 실험 결과는 두 번째 주장을 더 강하게 지지한다. 그래서 이번 업데이트는 논문과 계획을 모두 그 방향으로 다시 정리한다.

## 2. 현재까지 검증된 사실

### 2.1 GPT-2: basis-selection effect는 분명하다

공정한 same-harness 2-bit 비교에서 `topk=0.5` 설정의 축 ablation 결과는 다음과 같다.

| Basis | PPL |
|---|---:|
| identity | 149.8334 |
| random | 52.9955 |
| FOKVQ(PCA) | 34.5466 |

이 결과는 두 가지를 보여 준다.

1. `PCA basis`는 단순한 장식이 아니다.
2. 그러나 이 효과만으로 `KIVI-style` 또는 `turboquant-style`을 이기지는 못한다.

같은 main table에서 `FOKVQ`는 `34.5466`, `KIVI-style`은 `27.7507`, `turboquant-style`은 `27.7524`였다. 즉 `basis는 맞아도 quantizer는 아직 약하다`는 해석이 가장 자연스럽다.

### 2.2 GPT-2: 4-bit에서는 basis effect가 거의 사라진다

4-bit main table은 거의 포화 영역이다.

| Method | PPL |
|---|---:|
| FP16 | 26.8638 |
| KIVI-style | 26.8770 |
| uniform | 26.8983 |
| FOKVQ | 26.9954 |
| turboquant-style | 27.0020 |

4-bit axis ablation에서도 `random=26.9509`, `FOKVQ(PCA)=26.9954`, `identity=27.0862`로 차이가 매우 작다. 따라서 현재 논문의 핵심 mechanistic evidence는 4-bit가 아니라 2-bit stress regime에 있다.

### 2.3 Qwen: practical leader는 `kivi_residual`

Qwen2.5-7B full-K quantization 16k run은 현재 practical ranking을 더 분명하게 보여 준다.

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| FP16 | 6.7544 | 6.7544 |
| kivi_residual | 7.2060 | 6.7738 |
| turboquant_rand | 7.5587 | 7.0212 |
| turboquant | 7.5671 | 7.0900 |
| quip | 9.1795 | 7.1394 |
| fokvq_e2 | 8.7232 | 8.4917 |
| fokvq | 9.6555 | 7.4720 |

이 표가 말하는 것은 단순하다. Qwen/GQA/RoPE에서는 `recent residual tail`을 살리는 practical bias가 매우 강하다. 같은 이유로 `fokvq_e2_residual`은 plain `fokvq_e2`보다 3-bit에서 좋아졌지만, 여전히 `kivi_residual`과는 거리가 있다.

### 2.4 초기 Lie 후보의 생존 여부

초기 RoPE-aware 변형은 성공적이지 않았다.

| Method | 3-bit PPL | 4-bit PPL |
|---|---:|---:|
| kivi_residual | 6.5734 | 6.1431 |
| rope_unitary | 14.3512 | 13.9959 |
| rope_magphase | 750.0469 | 424.5279 |

이 결과는 `RoPE-aware`라는 이름만으로는 충분하지 않다는 반증이다. pair-local 회전이나 단순 magnitude-phase 분해는 현재 구현에서 실제로 도움을 주지 못했다.

### 2.5 Qwen bounded mechanistic smoke: `fokvq_e2`가 구조적 후보군 선두로 올라왔다

`2026-04-03` bounded Qwen smoke는 기존 practical 표와는 다른 질문을 던진다. 이 실험은 "어느 방법이 바로 제일 좋은가"가 아니라, "약한 control과 strong same-harness control을 포함해 구조적 축 선택 효과가 실제로 살아 있는가"를 묻는다.

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

이 결과는 세 가지를 추가로 보여 준다.

1. `Qwen/GQA/RoPE`에서도 구조적 basis effect는 사라지지 않았다.
2. 현재 가장 강한 structured candidate는 `fokvq_e2`다.
3. complex/unitary 계열은 아직 설명 후보이지, 주력 constructive candidate는 아니다.

## 3. 지금까지의 해석

이 결과들을 함께 읽으면 다음 구조가 보인다.

### 3.1 이미 지지되는 해석

- 회전 선택은 진짜 문제다.
- 정적 basis choice는 quality에 영향을 준다.
- PCA는 적어도 정적 회전의 강한 baseline이다.
- GQA/RoPE에서는 `recent tail + old prefix structured transform`이 중요한 inductive bias다.
- bounded Qwen mechanistic regime에서는 `fokvq_e2`가 구조적 후보군 중 가장 강하다.

### 3.2 아직 지지되지 않는 해석

- 현재 FOKVQ가 완성된 practical winner다.
- 임의의 Lie-inspired transform이 baseline보다 낫다.
- reconstruction MSE를 줄이면 PPL도 자동으로 좋아진다.
- Hamiltonian 언어가 이미 직접 검증되었다.

특히 `fokvq_lloyd`는 rotated-space reconstruction MSE는 줄였지만 PPL은 개선하지 못했다. 이 사실은 지금의 병목이 단순 scalar quantizer 교체가 아니라는 점을 보여 준다.

## 4. 벤치마크를 어떻게 다시 잡아야 하는가

이 논문이 실제로 검증하고 싶은 것은 `좋아 보이는 숫자`가 아니라 `Lie 군 기반 회전 선택 가설`이다. 따라서 벤치마크를 네 층으로 나눠야 한다.

### 4.1 Quality benchmark

- 목적: pure LM quality retention 측정
- 현재 선택: `WikiText-2` full-K PPL
- 이유: 기존 KV-cache 논문과 연결되고, iterative loop에 적합하다

### 4.2 Rotation mechanistic benchmark

- 목적: `identity vs random vs PCA/structured basis` 비교
- 현재 선택: same-harness axis ablation
- 이유: main table negative를 `rotation은 의미 없다`로 오해하지 않기 위해 필요하다

### 4.3 Retrieval-depth benchmark

- 목적: position-sensitive geometry가 실제로 유지되는지 확인
- 현재 선택: NIAH depth sweep
- 이유: 평균 PPL만으로는 cache geometry 보존 여부를 직접 보기 어렵다

### 4.4 Optional external benchmark

- 목적: broad long-context generalization
- 상태: B1-B3가 안정화된 뒤로 미룬다

## 5. 코드/프로토콜에서 바로 수정한 점

이번 정리 과정에서 코드 정합성도 함께 점검했다.

1. `exp4_2_v3_full_quant_ppl.py`에 `benchmark preset` 개념을 추가했다.
2. preset마다 `intent / hypothesis / verification focus`가 JSON에 남도록 수정했다.
3. `rotation_mechanistic` preset에 필요한 `identity`, `random` basis가 v3 dispatch에 빠져 있던 버그를 수정했다.
4. local launcher가 기본 `python3`를 써서 `torch`가 없는 환경에서 실패하던 문제를 수정하고, 기본 interpreter를 명시적으로 고정했다.
5. launcher 출력 경로를 repo-level `results/`와 `reports/`로 정리해, 이후 보고서가 `scripts/results/...` 같은 드리프트 경로를 참조하지 않게 했다.
6. 결과 JSON에 `generated_at`, `hostname`, `cwd`, `git_head`를 남기도록 수정해 provenance를 추적 가능하게 했다.

수정 후 self-test는 통과했고, 작은 `rotation_mechanistic` preset smoke도 정상 종료됐다. 단, 이 smoke는 wiring 검증일 뿐이며 scientific evidence로 해석하면 안 된다.

## 6. 다음 단계

이제 autoresearch는 무작정 많은 표를 돌리는 방식이 아니라 아래 순서로 진행해야 한다.

1. `self-test` 통과
2. preset smoke 통과
3. `rotation_mechanistic`에서 basis-level gain 확인
4. 그 다음에만 `ppl_quality` 또는 `retrieval_depth` full run

현재 가장 타당한 active queue는 네 가지다.

- `fokvq_e2` practical promotion
- Hamiltonian descriptive diagnostics
- `complex_unitary_residual`
- `banded_complex_unitary_residual`

여기서 우선순위는 명확하다. `fokvq_e2`가 bounded win을 medium-budget에서도 유지하는지 먼저 확인해야 하고, Hamiltonian은 그 결과를 설명할 수 있는 descriptive axis인지 검증해야 한다. complex/unitary 계열은 이 두 축을 보조하는 challenger로 남는 편이 맞다.

## 7. 결론

현재까지의 결과는 부정적이지만 무의미하지는 않다. 지금 밝혀진 가장 중요한 사실은 `어떤 회전이든 넣으면 좋아지는 것이 아니라, 회전 선택 문제와 quantizer 문제를 분리해서 봐야 한다`는 점이다. GPT-2는 이 분리를 강하게 지지하고, Qwen은 practical winner가 어디에 있는지 냉정하게 보여 준다.

따라서 지금의 가장 좋은 논문 방향은 `Lie 군 관점에서 회전 기반 KV-cache 양자화를 재해석하고, 그 해석이 어디까지 empirically 살아남는지를 정리하는 것`이다. 해밀토니안 언어는 여기서 더 강한 claim이 아니라, pairwise energy와 symplectic structure를 덜 망가뜨리는 방법이 실제 bounded mechanistic 결과와 같이 움직이는지 보는 보조 설명 축으로 써야 한다. 그 위에서 추가 실험이 성공하면 method paper로 확장할 수 있고, 실패하더라도 분석 논문으로서의 가치는 유지된다.
