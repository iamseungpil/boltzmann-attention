# 실험 계획 v15: PPL 검증 강화 및 Theory-Practice Gap 해소

**날짜**: 2026-04-05
**기반**: Report v2 발견 (3모델 PPL, Lloyd-Max 실패, Mistral 예외, WF 불일정)
**목표**: 4가지 theory-practice gap의 원인을 진단하고 해결책을 검증

---

## Exp V15-1: KVTC 직접 비교 (공유 PCA vs 헤드별 PCA)

```
의도 (Intent):
  KVTC (ICLR 2026)는 전체 KV head를 합쳐서 하나의 공유 PCA basis를 계산한다.
  우리 방법은 각 KV head에서 독립적으로 PCA를 계산한다 (정리 6.16.3).
  Fischer inequality에 의해 헤드별 PCA가 MSE에서 항상 우위이지만,
  PPL 차이가 실질적인지 확인한다.

가설 (Hypothesis):
  H1: 헤드별 PCA MSE < 공유 PCA MSE (이론 보장, 확인용)
  H2: 헤드별 PCA PPL ≤ 공유 PCA PPL (PPL 전환 확인)
  H3: 이득은 헤드 간 공분산 이질성이 높은 모델에서 더 큼

검증 방법 (Verification):
  모델: Llama-3.1-8B (KVTC 논문 주력 모델)
  방법:
    a. 공유 PCA + Uniform (모든 KV head 합쳐서 PCA 1개)
    b. 헤드별 PCA + Uniform (KV head별 PCA)
  비트: 2, 3, 4
  메트릭: PPL (WikiText-2, 50K tokens)
  성공 기준: PPL 차이 > 0.1% (실질적)
  실패 대응: 차이 무시 → "공유 PCA는 near-optimal approximation" 결론
```

## Exp V15-2: Calibration 크기가 PCA 안정성에 미치는 영향

```
의도 (Intent):
  Mistral 2-bit에서 PCA(6.46)가 TurboQuant(6.37)에 진 원인을 진단한다.
  가설: R_aniso=131.62인 극단적 이방성 헤드의 PCA 방향이 2048 토큰
  calibration으로 불안정하여, 더 많은 calibration이 이를 안정화시킬 수 있다.

가설 (Hypothesis):
  H1: calibration 토큰 수 증가(2048→4096→8192)에 따라 Mistral PCA PPL 개선
  H2: R_aniso > 100인 헤드에서 calibration 효과가 가장 큼
  H3: Qwen/Llama(R_aniso<10)에서는 calibration 크기 효과 미미

검증 방법 (Verification):
  모델: Mistral-7B (예외 모델)
  Calibration 토큰: {1024, 2048, 4096, 8192}
  방법: PreRoPE PCA + Uniform, 2-bit
  메트릭: PPL + 극단적 헤드(R_aniso>100)의 MSE 변동
  성공 기준: 8192 토큰에서 PCA PPL < TurboQuant PPL
  실패 대응: calibration 불안정이 아닌 구조적 문제 → R_aniso 상한 보고
```

## Exp V15-3: Adaptive Lloyd-Max (min-max 초기화)

```
의도 (Intent):
  Lloyd-Max의 PPL 재앙이 고정 Gaussian codebook의 tail sensitivity 때문인지
  검증한다. 데이터 적응적 initialization(per-channel min-max로 codebook 초기화
  후 Lloyd iteration)이 PPL을 구제하는지 확인한다.

가설 (Hypothesis):
  H1: min-max 초기화 Lloyd-Max가 고정 Gaussian Lloyd-Max보다 PPL 개선
  H2: 적응적 Lloyd-Max PPL ≤ Uniform PPL (축 2 이론 복원)
  H3: MSE 이득은 Gaussian Lloyd-Max보다 작지만, PPL은 안정적

검증 방법 (Verification):
  구현: per-column에서
    1. min/max로 초기 centroid를 균등 배치
    2. Lloyd iteration 10회 (expectation step + centroid update)
    3. 수렴된 codebook으로 양자화
  모델: Qwen2.5-7B (기존 결과와 직접 비교)
  비트: 2, 3, 4
  메트릭: MSE + PPL
  성공 기준: adaptive Lloyd PPL < Gaussian Lloyd PPL AND ≤ Uniform PPL
  실패 대응: scalar 양자화기 자체의 한계 → "Class C 경계" 결론 확정
```

## Exp V15-4: Water-Filling 최소 비트 하한

```
의도 (Intent):
  WF가 Qwen 2-bit에서 PPL을 2배 악화(11.37 vs 7.98)시킨 원인을 진단한다.
  가설: 저분산 PCA 차원에 1bit만 할당하면 해당 차원의 정보가 완전 소실되어
  PPL catastrophe 발생. 최소 비트 하한(floor=2)을 걸면 안정화될 것이다.

가설 (Hypothesis):
  H1: floor=2인 WF가 floor=1(현재)인 WF보다 PPL 개선 (Qwen 2-bit)
  H2: floor=2인 WF가 uniform보다 PPL 개선 (이론적 이득 복원)
  H3: Llama에서는 floor 효과 미미 (이미 WF가 좋으므로)

검증 방법 (Verification):
  구현: compute_wf_alloc()에 min_bits=2 파라미터 추가
    b_alloc = np.maximum(b_alloc, min_bits)  # floor 적용
    총 비트 예산 재분배 (floor로 올린 만큼 다른 채널에서 감소)
  모델: Qwen2.5-7B, Llama-3.1-8B
  비트: 2, 3
  메트릭: PPL
  성공 기준: floor=2 WF PPL < uniform PPL
  실패 대응: WF 자체가 PPL에 부적합 → "MSE 최적 ≠ PPL 최적" 결론
```

---

## 실행 순서

```
V15-1 (KVTC 비교)      ~15분  ← 리뷰어 질문 대응
V15-2 (Calibration)     ~30분  ← Mistral 예외 해명
V15-3 (Adaptive Lloyd)  ~30분  ← 축 2 복원 시도
V15-4 (WF floor)        ~20분  ← 축 3 안정화 시도
```

## 전체 성공 판정

| 실험 | 최선 결과 | 논문 의미 |
|------|----------|----------|
| V15-1 PASS | 헤드별 PCA > 공유 PCA (PPL) | KVTC 대비 이론적+실질적 우위 |
| V15-2 PASS | 8K cal에서 Mistral PCA 역전 | 예외 해소, 보편적 우위 |
| V15-3 PASS | adaptive Lloyd < uniform (PPL) | 축 2 이론-실무 gap 해소 |
| V15-4 PASS | floor WF < uniform (PPL) | WF 안정화, 축 3 부분 복원 |

ALL FAIL 시: "Pre-RoPE PCA + Uniform이 실무적 최적이며, 이론적 상한(Lloyd, WF)과의 gap은 scalar quantizer의 구조적 한계. 이것이 프레임워크가 밝히는 Class C의 경계."
