# NeurIPS 2026 실험 계획서 v13

**날짜**: 2026-04-05
**기반**: NEURIPS_REVIEW_v1.md + 기존 검증 결과 분석
**목표**: 3축 최적성 이론의 다중 모델 검증
**핵심 변경**: Lloyd-Max 센터링 버그 수정, Llama-3-8B 추가, NIAH 장문맥 확장

---

## 0. 현황 요약

### 완료된 검증
| 항목 | 모델 | 결과 | 상태 |
|------|------|------|------|
| Level 1 예측 (Σ → MSE 순서) | Qwen-7B, Qwen-1.5B, Mistral-7B | R_aniso 계산 완료 | ✅ |
| 축 1: Pre-RoPE PCA MSE | Qwen-7B | 모든 비트에서 BEST | ✅ |
| 축 1: Pre-RoPE PCA PPL | Qwen-7B | 모든 비트에서 BEST | ✅ |
| Post-RoPE 2bit 패배 확인 | Qwen-7B | Cor 6.16.4(d) 일치 | ✅ |

### 미완료 / 실패
| 항목 | 문제 | 원인 |
|------|------|------|
| 축 2: Lloyd-Max | MSE가 uniform의 100-4000x 나쁨 | **센터링 버그** (mean 미차감) |
| 축 3: NIAH | 대부분 100% 포화 | context 2048 너무 짧음 |
| Llama-3-8B | 미실행 | KVTC 직접 비교 모델 |
| 3모델 PPL 비교 | 부분적 | 통합 테이블 미완성 |

---

## Exp V13-1: Lloyd-Max 센터링 수정 검증

```
의도 (Intent):
  축 2의 핵심 주장 "Gaussian Lloyd-Max가 uniform quantizer를 이김"을 검증한다.
  이전 구현은 per-channel centering을 하지 않아 N(μ,σ) 데이터에 N(0,σ) 코드북을
  적용하여 모든 값이 한 bin에 매핑되는 치명적 버그가 있었다.

가설 (Hypothesis):
  H1: 센터링 수정 후, per-channel Gaussian Lloyd-Max MSE < per-channel uniform MSE
      for all bits ∈ {2, 3, 4} and all rotations (identity, random, Pre-RoPE PCA)
  H2: Lloyd-Max gain (uniform_MSE / lloyd_MSE)은 이론적 예측값
      G_LM(b) = (AM/GM ratio of N(0,1) at b bits)에 근접
  H3: PrePCA + Lloyd-Max가 모든 조합 중 최소 MSE를 달성

검증 방법 (Verification):
  1. Self-test: N(μ, σ²) 합성 데이터 (μ=0,5,-3,100, σ=0.1,1,10)에서
     Lloyd-Max MSE ≤ uniform MSE × 1.1 확인
  2. Qwen2.5-7B 112 헤드에서 6개 조합 비교:
     {NoRot, TurboQuant, PrePCA} × {Uniform, Lloyd-Max} × {2,3,4}bit
  3. 성공 기준: Lloyd-Max gain > 1.0 for ALL rotations at ALL bits
  4. 실패 시: Gaussian 가정 위반 정도를 kurtosis로 정량화

실행:
  python verify_3axis_unified.py --model Qwen/Qwen2.5-7B --axis 2 --device cuda:0
```

## Exp V13-2: Llama-3-8B Level 1+2 검증

```
의도 (Intent):
  KVTC 논문의 주력 모델인 Llama-3-8B에서 Level 1 (MSE 순서 예측)과
  Level 2 (PPL 예측)를 검증한다. 이것은 NeurIPS 리뷰어가 가장 먼저
  비교를 요구할 모델이다.

가설 (Hypothesis):
  H1: Llama-3-8B의 모든 KV 헤드에서 MSE 순서가 이론 예측과 일치:
      PreRoPE PCA+WF < PreRoPE PCA+uni < No rotation ≈ TurboQuant
  H2: Pre-RoPE PCA+WF가 TurboQuant 대비 R_aniso배 더 낮은 MSE 달성
  H3: PPL 순서가 MSE 순서와 일치

검증 방법 (Verification):
  1. predict_benchmarks.py로 Llama-3-8B의 R_aniso, η(b), G값 계산
  2. verify_3axis_unified.py로 축 1 + 축 2 검증 (MSE)
  3. verify_prerope_pca.py를 Llama용으로 실행 (PPL)
  4. 성공 기준: MSE 순서 100% 성립, R² > 0.8 (PPL 예측)

실행:
  python verify_3axis_unified.py --model meta-llama/Llama-3.1-8B --axis 1 2 --device cuda:0
```

## Exp V13-3: Mistral-7B 극단 이방성 검증

```
의도 (Intent):
  R_aniso=80.89 (극단적)인 Mistral-7B에서 이론 예측의 정합도를 확인한다.
  R_aniso가 극단적일 때 이론-실측 오차가 커지는지, 아니면 비례적으로
  더 큰 이득이 확인되는지를 본다.

가설 (Hypothesis):
  H1: Pre-RoPE PCA+WF가 TurboQuant 대비 현저히 큰 이득 (>10x)
  H2: 이론 예측(80.89x)과 실측의 비율이 Qwen(4.07→1.98~3.39x)과
      유사한 패턴 (이론의 50-80% 수준)
  H3: 일부 극단적 이방성 헤드(R_aniso>1000)에서 수치 불안정성 확인

검증 방법 (Verification):
  1. 축 1 MSE 검증 (verify_3axis_unified.py --axis 1)
  2. R_aniso > 100인 헤드와 < 10인 헤드를 분리해 이론-실측 비교
  3. 이상치 헤드 제외한 robust 통계량도 보고

실행:
  python verify_3axis_unified.py --model mistralai/Mistral-7B-v0.3 --axis 1 2 --device cuda:0
```

## Exp V13-4: NIAH 장문맥 검증

```
의도 (Intent):
  4K/8K/16K 컨텍스트에서 양자화가 needle 검색 정확도에 미치는 영향을
  확인한다. 기존 2048 context에서는 모든 방법이 96-100%로 포화되어
  차별화가 불가능했다.

가설 (Hypothesis):
  H1: 컨텍스트가 길어질수록 (4K→8K→16K) 양자화 품질 차이가 증폭됨
  H2: 2-bit에서 Pre-RoPE PCA > TurboQuant > Identity (NIAH 정확도 순서)
  H3: NIAH 정확도 순서가 MSE 순서와 일치

검증 방법 (Verification):
  1. Qwen2.5-7B에서 {4K, 8K} × {2, 3}bit × {identity, turbo, pre_pca} NIAH
  2. 각 조합 10 trials × 5 depths
  3. 성공 기준: 8K 이상에서 방법 간 유의미한 차이 (>5% gap)
  4. 실패 시: Qwen의 max context가 짧아서일 수 있음 → Llama(128K context)로 대체

실행:
  python verify_3axis_unified.py --model Qwen/Qwen2.5-7B --axis 3 \
    --niah-contexts 4096 8192 --device cuda:0
```

---

## 실행 순서 및 의존성

```
V13-1 (Lloyd-Max fix)  ←  가장 급함: 축 2 주장의 생사 결정
  ↓
V13-2 (Llama Level 1+2)  ←  KVTC 비교의 핵심
V13-3 (Mistral extreme)  ←  이론 검증력 보강
  ↓
V13-4 (NIAH long)  ←  축 3 차별화
```

## 예상 소요 시간

| 실험 | GPU | 예상 시간 |
|------|-----|----------|
| V13-1 smoke (self-test + Qwen axis 2) | cuda:0 | 5분 |
| V13-2 Llama axis 1+2 | cuda:0 | 15분 |
| V13-3 Mistral axis 1+2 | cuda:0 | 15분 |
| V13-4 NIAH 4K+8K | cuda:0 | 60분 |

총 예상: ~2시간 (직렬 실행 기준)

## 성공 판정 기준

| 항목 | 기준 | 논문 영향 |
|------|------|----------|
| 축 1 (3모델) | PreRoPE PCA+WF = BEST at all bits | 정리 6.16.3 확인 |
| 축 2 (Lloyd-Max) | Lloyd gain > 1.0 all cases | 축 2 주장 가능 |
| 축 3 (NIAH 8K) | 방법 간 >5% 차이 | Table 2 소재 |
| Level 1 (3모델) | MSE 순서 100% 예측 | 핵심 기여 |

## 실패 시 대응

- 축 2 실패 (Lloyd-Max가 여전히 나쁨): Gaussian 가정 위반 분석,
  "이론적 예측은 성립하나 현재 스칼라 양자화기로는 이득이 미미"로 논문 프레이밍 변경
- NIAH 포화: 더 긴 컨텍스트 (32K) 시도, 또는 PPL 우위에 집중
- Llama에서 MSE 순서 불일치: 해당 헤드의 non-Gaussianity (kurtosis) 분석
