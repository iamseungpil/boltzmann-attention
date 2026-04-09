# 실험 계획 v23.1: 회전-양자화기 상호작용 분석 (2026-04-09 updated)

## 핵심 발견 (확정)

| # | 사실 | 증거 |
|:-:|------|------|
| 1 | Pre-RoPE PCA는 Class C 내 MSE 최적 | 증명 + 624/624 |
| 2 | Per-head > Shared PCA | Fischer 증명 + 46.3% |
| 3 | 3-bit에서 PCA PPL 4/4 최선 | 49K test |
| 4 | MMLU: PCA +9.2%p | Qwen 2-bit |
| 5 | Lloyd L∞ > Uniform L∞ | 624/624 측정 |
| 6 | PCA + Lloyd PPL > PCA + Uniform PPL | **9/9 설정** (3모델 × 3비트) |
| 7 | Mistral 전체 완료 | 2b: 2.46×, 3b: 1.25×, 4b: 1.06× |

## 핵심 미해결: "Lloyd 자체가 문제인가, PCA+Lloyd 조합이 문제인가?"

현재 모든 Lloyd 실험은 **PCA 회전 위에서만** 수행됨.
PCA가 분산 이방성을 극대화 → 고분산 차원에서 Lloyd 꼬리 gap 극대화.
TurboQuant는 random rotation으로 분산 균등화 → Lloyd 안전하게 작동.

**이것을 분리하는 실험이 필수.**

## Phase 2: Root Cause 분리 실험 (3×2 ablation)

### 실험 설계

```
          | Uniform    | Lloyd      | Clipped Lloyd |
----------+------------+------------+---------------+
No Rot    | A (KIVI식) | B          | G             |
PCA       | C ✓(6.40)  | D ✓(15.75) | H (새 방법)   |
Random    | E (QuaRot) | F (Turbo)  | I             |
```

### 각 실험의 의도-가설-검증

#### A: NoRot + Uniform (baseline)
```
의도: 회전 없이 Uniform의 PPL 기준점
가설: 이미 알고 있음 (Mistral ~7.28)
검증: PPL 측정
```

#### B: NoRot + Lloyd ★★★ (가장 중요)
```
의도: Lloyd 자체가 PPL에서 나쁜지 확인
가설 H1: NoRot + Lloyd ≈ NoRot + Uniform → "Lloyd 자체는 OK, PCA가 문제"
가설 H2: NoRot + Lloyd >> NoRot + Uniform → "Lloyd가 본질적으로 나쁨"
검증: Mistral 2-bit PPL
해석:
  H1이면: PCA가 고분산 차원을 만들어 Lloyd를 망침 → Clipped Lloyd로 해결 가능
  H2이면: Lloyd의 꼬리 gap이 회전과 무관하게 PPL에 해로움
```

#### E: Random Rotation + Uniform
```
의도: TurboQuant의 rotation 효과 (분산 균등화) 확인
가설: Random + Uniform ≈ PCA + Uniform (rotation 유형보다 양자화가 중요)
검증: Mistral 2-bit PPL
```

#### F: Random Rotation + Lloyd ★★ (TurboQuant 재현)
```
의도: TurboQuant 방식에서 Lloyd가 잘 작동하는지 확인
가설: Random + Lloyd < Random + Uniform (분산 균등화로 Lloyd 안전)
검증: Mistral 2-bit PPL
해석: 
  맞으면: "분산 균등화가 Lloyd를 살린다" → 회전-양자화기 상호작용 확정
  틀리면: Lloyd가 본질적으로 나쁨 (TurboQuant도 Uniform이 나았을 것)
```

#### H: PCA + Clipped Lloyd ★★★ (새 방법)
```
의도: PCA의 MSE 이점 + 꼬리 보호 결합
방법: 바깥 2개 centroid를 calibration min/max에 고정, 안쪽만 Lloyd
가설: PCA + Clipped Lloyd < PCA + Uniform (MSE 이점이 PPL로 전이)
검증: 3모델 × 3비트 PPL
해석:
  맞으면: SOTA 후보. "이론 → 진단 → 방법 → 성능" 완전한 논문
  틀리면: 안쪽 centroid 위치가 무관 → "양자화에서 중요한 것은 꼬리 coverage뿐"
```

### 부가 측정 (실험 중 동시 수집)

- **Kurtosis**: 각 PCA 차원의 초과 첨도 (>3이면 heavy-tail)
- **Eval-time L∞**: 각 config에서 최대 양자화 오차
- **MSE**: 각 config에서 평균 양자화 오차

## Phase 3: 결과별 논문 방향

| B 결과 | F 결과 | H 결과 | 논문 |
|--------|--------|--------|------|
| B ≈ A (Lloyd OK without PCA) | F < E (Lloyd OK with random) | H < C (Clipped beats Uniform) | **BEST: "회전-양자화기 상호작용 + Clipped Lloyd = SOTA"** |
| B ≈ A | F < E | H ≈ C | "상호작용 발견, but Uniform으로 충분" |
| B >> A (Lloyd always bad) | F >> E | H ≈ C | "Lloyd는 KV cache에 부적합 (근본적)" |
| B ≈ A | F ≈ E | — | "양자화기 무관, 회전이 전부" |

## 실행 계획 (E8, 4 GPUs idle)

| GPU | 실험 | 시간 |
|:---:|------|:----:|
| 0 | B (NoRot+Lloyd) + A (NoRot+Uniform) — Mistral 2/3/4-bit | 30분 |
| 1 | F (Random+Lloyd) + E (Random+Uniform) — Mistral 2/3/4-bit | 30분 |
| 2 | H (PCA+Clipped Lloyd) — Mistral 2/3/4-bit | 30분 |
| 3 | H (PCA+Clipped Lloyd) — Qwen + Llama 2-bit | 30분 |

**총 ~30분으로 핵심 ablation 완료 가능.**
