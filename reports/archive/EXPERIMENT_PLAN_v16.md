# 실험 계획 v16: NeurIPS 테이블 완성 및 마무리 검증

**날짜**: 2026-04-05
**기반**: V15 결과 (WF floor=2 돌파, KVTC 비교 완료, Lloyd-Max 한계 확정)
**목표**: NeurIPS 제출에 필요한 마지막 데이터 확보

---

## 현황: 확보된 것 vs 미확보

| 항목 | 상태 | 비고 |
|------|:----:|------|
| 3모델 PPL Table (2-3bit) | ✅ | PCA+WF(f=2) BEST |
| 3모델 PPL Table (4-bit) | ❌ | WF floor=2 미측정 |
| KVTC 직접 비교 | ✅ | +46.3% at 2-bit |
| Level 2 R² | ❌ | WF floor=2 데이터로 업데이트 필요 |
| NIAH with WF floor=2 | ❌ | 최적 방법으로 NIAH 재확인 |
| 3축 Ablation table | ❌ | 논문 Table 3용 |

---

## Exp V16-1: 4-bit WF floor=2 PPL (3모델)

```
의도: NeurIPS Table 1의 4-bit 열을 완성한다. 3-4bit에서는
      차이가 미미할 것으로 예상되지만, completeness를 위해 필요.

가설: PCA+WF(floor=2) PPL ≤ PCA+Uniform PPL at 4-bit (3모델)
      이득은 0.1-0.3% 수준으로 미미할 것

검증: verify_v15_experiments.py --exp 4 --bits 4 on each model
      (기존 실행에서 2-3bit만 했으므로 4-bit 추가)
```

## Exp V16-2: 3축 Ablation Table (Qwen2.5-7B)

```
의도: 논문 Table 3 (Ablation)을 위해 각 축의 독립적 기여를 정량화.
      축 1 (rotation), 축 2 (quantizer type), 축 3 (WF bit alloc)의
      조합별 PPL을 측정.

가설: 축 간 이득이 대략 곱셈적 (명제 6.18.2 독립성)

검증:
  6개 조합 × 2-bit PPL (Qwen2.5-7B):
    a. No rot + Uni            (baseline)
    b. PCA + Uni               (축 1만)
    c. No rot + WF(f=2)        (축 3만)
    d. PCA + WF(f=2)           (축 1+3)
    e. TurboQuant + Uni        (reference)
    f. TurboQuant + WF(f=2)    (turbo + 축 3)
  
  독립성 체크:
    gain(축1) = PPL(a) / PPL(b)
    gain(축3) = PPL(a) / PPL(c)
    predicted PPL(d) ≈ PPL(a) / (gain(1) × gain(3))
    |actual - predicted| / actual < 10%
```

## Exp V16-3: Level 2 R² 정량화

```
의도: "Σ만으로 최적 방법 추천" 주장을 R²로 뒷받침.
      MSE와 ln(PPL/fp16)의 선형 상관을 정량화.

가설: R² > 0.85 (3모델 통합, main methods only)

검증:
  1. MSE 데이터: 3모델 axis 1 결과에서 추출
  2. PPL 데이터: 최신 PPL table (WF floor=2 포함)
  3. 회귀: ln(PPL/fp16) = c × MSE + intercept
  4. 모델별 + 통합 R² 보고
  5. Figure: scatter plot (MSE vs ln(PPL/fp16))
```

---

## 실행 순서

```
V16-1: 4-bit WF floor=2      ~15분 (3모델 순차)
V16-2: 3축 Ablation           ~20분 (Qwen만)
V16-3: R² 계산                ~5분 (기존 데이터 분석)
```

총 ~40분. 이후 Report v3 최종 업데이트 → 논문 작성 시작.
