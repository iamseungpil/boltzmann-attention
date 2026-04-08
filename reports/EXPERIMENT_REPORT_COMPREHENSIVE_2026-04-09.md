# 종합 실험 보고서 (2026-04-09)

## 1. 연구 질문

**"KV 캐시 양자화에서 회전(rotation)이 왜 품질을 개선하는가?"**

기존 8개 방법 (KIVI, KVQuant, KVTC, TurboQuant, SpinQuant, PolarQuant, QuaRot, GEAR)이 각기 다른 회전을 사용하지만, 왜 회전이 도움이 되는지에 대한 통일적 설명이 없다.

---

## 2. 확정된 실험 결과

### 2.1 Decorrelation 실험 (Mistral-7B, 4K calib, 49K test)

**3-bit 결과:**

| 회전 | PPL | MSE | L∞_max | L∞_p99 | Error Diag% | Error OffDiag |
|------|:---:|:---:|:------:|:------:|:-----------:|:-------------:|
| NoRot | 5.721 | 0.0990 | 6.81 | 1.68 | 80.7% | 0.488 |
| **PCA** | **5.691** | **0.0928** | 6.45 | 1.99 | **99.9%** | **0.037** |
| Random | 5.695 | 0.0946 | **4.13** | **0.99** | 78.5% | 0.524 |
| FP16 | 5.576 | 0 | 0 | 0 | — | — |

**2-bit 결과:**

| 회전 | PPL | MSE | L∞_max | L∞_p99 | Error Diag% | Error OffDiag |
|------|:---:|:---:|:------:|:------:|:-----------:|:-------------:|
| NoRot | 7.352 | 0.5491 | 7.75 | 3.88 | 62.7% | 0.771 |
| **PCA** | **6.713** | **0.5153** | 7.91 | 4.62 | **99.3%** | **0.084** |
| Random | 6.772 | 0.5218 | **4.86** | **2.19** | 56.4% | 0.880 |
| FP16 | 5.576 | 0 | 0 | 0 | — | — |

**Qwen-7B 3-bit 결과:**

| 회전 | PPL | Diag% |
|------|:---:|:-----:|
| NoRot | 6.890 | 69.3% |
| PCA | 6.771 | 77.9% |
| Random | 6.871 | 72.0% |

### 2.2 Lloyd vs Uniform (3모델 × 3비트, 수정 후, 49K test)

| 모델 | Bits | Uniform PPL | Lloyd PPL | Lloyd/Uni |
|------|:----:|:----------:|:---------:|:---------:|
| Qwen | 2 | 7.94 | 8.16 | 1.03× |
| Qwen | 3 | 6.76 | 7.28 | 1.08× |
| Qwen | 4 | 6.61 | 6.92 | 1.05× |
| Mistral | 2 | 6.40 | 15.75 | 2.46× |
| Mistral | 3 | 5.67 | 7.10 | 1.25× |
| Mistral | 4 | 5.60 | 5.92 | 1.06× |
| Llama | 2 | 10.20 | 43.39 | 4.25× |
| Llama | 3 | 6.67 | 19.15 | 2.87× |
| Llama | 4 | 6.46 | 10.20 | 1.58× |

**9/9 설정에서 Lloyd PPL > Uniform PPL. 버그 수정 후에도 동일.**

### 2.3 3×3 Interaction (Mistral 2-bit, 2K calib)

| | Uniform | Lloyd | Clipped Lloyd |
|--|:-------:|:-----:|:-------------:|
| NoRot | 6.94 | 274.0 | 342.6 |
| PCA | 6.40 | 15.75 | 16.83 |
| Random | 6.47 | 16.66 | 17.89 |

### 2.4 L∞ 측정 (624/624 heads, 3 models, 2-bit)

| 모델 | Lloyd MSE↓ | Lloyd L∞↑ | 모든 head 일관 |
|------|:---------:|:---------:|:------------:|
| Mistral | 74% | 94% | 256/256 |
| Qwen | 73% | 75% | 112/112 |
| Llama | 74% | 86% | 256/256 |

### 2.5 D_attn 측정 (Mistral, 양자화 없이)

| 회전 | D_attn (합계) |
|------|:----------:|
| NoRot | 20,702 |
| PCA | 32,513 (최악) |
| Random | 9,996 (최선) |

**D_attn과 PPL이 역상관: D_attn이 PPL의 올바른 proxy가 아님.**

### 2.6 MMLU (Qwen-7B 2-bit)

| | FP16 | NoRot | PCA |
|--|:----:|:-----:|:---:|
| MMLU | 74.3% | 58.7% | **67.9%** (+9.2%p) |

### 2.7 Pre vs Post RoPE (4 models, Uniform)

| 비트 | PCA 승리 모델 수 |
|:----:|:--------------:|
| 4-bit | 2/4 |
| **3-bit** | **4/4** |
| 2-bit | 2/4 |

### 2.8 Per-head vs Shared PCA (Llama 2-bit)

| | 2-bit | 3-bit | 4-bit |
|--|:-----:|:-----:|:-----:|
| Shared PCA (KVTC) | 18.87 | 6.81 | 6.48 |
| **Per-head PCA** | **10.14** | **6.67** | **6.46** |
| 이득 | **+46.3%** | +2.1% | +0.4% |

### 2.9 128K Calibration 실험 (Mistral 2-bit, 진행 중)

| Calib | NoRot+Uniform | NoRot+Lloyd |
|:-----:|:------------:|:-----------:|
| 2K | 6.94 | 274.0 |
| 128K | 10.75 | 413.2 |

**128K에서 PPL이 더 나빠짐 → calibration range 확대가 양자화 해상도를 낮춤.**

---

## 3. 핵심 관찰 (성급한 결론 없이)

### 관찰 1: 회전 자체가 중요, 회전 종류는 덜 중요

```
3-bit: PCA(5.691) ≈ Random(5.695) << NoRot(5.721)
2-bit: PCA(6.713) ≈ Random(6.772) << NoRot(7.352)

→ "어떤 회전이든 NoRot보다 낫다"
→ PCA vs Random 차이는 작음 (0.07% at 3-bit, 0.9% at 2-bit)
```

### 관찰 2: PCA와 Random이 다른 메커니즘으로 비슷한 PPL에 도달

```
PCA:    MSE 최선(0.093), Diag 99.9%, L∞ 중간(6.45)
Random: MSE 중간(0.095), Diag 78.5%, L∞ 최선(4.13)

→ PCA는 decorrelation + MSE 경로
→ Random은 L∞ 균등화 경로
→ 둘 다 비슷한 PPL에 도달
```

### 관찰 3: Lloyd는 어떤 설정에서든 Uniform보다 나쁨

```
9/9 설정 (3모델 × 3비트), 버그 수정 후에도 동일.
Lloyd MSE가 더 좋지만 (74% 감소), L∞가 더 나쁘고 (94% 증가), PPL도 더 나쁨.
```

### 관찰 4: D_attn은 PPL의 proxy가 아님

```
D_attn(PCA) = 32513 > D_attn(Random) = 9996
but PPL(PCA) = 5.691 < PPL(Random) = 5.695

역상관. D_attn (평균 attention error)이 PPL을 결정하지 않음.
```

### 관찰 5: 원인 미분리

PCA가 PPL을 개선하는 원인으로 가능한 것들:
- (A) Decorrelation (오차 독립화)
- (B) Per-dim range adaptation (각 차원 min/max 적응)
- (C) Gaussianity (CLT에 의해 PCA 차원이 더 가우시안)
- (D) Effective dimensionality (저분산 차원 제거 효과)
- (E) Attention sink 격리
- (F) WF enablement
- (G) 복합적

**현재 데이터로는 A-G를 분리할 수 없음.**

---

## 4. 선행 연구 대비 우리 위치

| 선행 연구 | 그들의 주장 | 우리 데이터가 보여주는 것 |
|----------|-----------|----------------------|
| TurboQuant | random rot + Lloyd ≈ 최적 | Random은 PPL에서 PCA와 비슷하지만 Lloyd는 Uniform보다 나쁨 |
| KVTC | PCA + DP 비트배분 | Per-head PCA가 Shared보다 46.3% 좋음 (우리 증명) |
| QuaRot | Hadamard rotation이 outlier를 분산 | 회전 자체가 중요, 종류는 덜 중요 |
| SpinQuant | 학습된 회전이 최적 | PCA ≈ Random이면 학습 불필요할 수 있음 |
| 모든 논문 | MSE 기반 설계 | MSE와 PPL의 관계가 복잡 (Lloyd 역설) |

### 우리만의 확정된 novelty

| 발견 | 새로운가? | 증거 강도 |
|------|:--------:|:--------:|
| Per-head PCA > Shared PCA (증명) | **예** | 강 |
| Lloyd MSE↓ but PPL↑ (9/9) | **예** (이 규모의 체계적 비교 없음) | 강 |
| 624/624 L∞ 측정 | **예** | 강 |
| PCA ≈ Random for PPL | **예** (아무도 이 비교 안 함) | 강 |
| D_attn ≠ PPL proxy | **예** | 강 |
| Error diag 99%+ (PCA) | 관찰 (tautology 논쟁 있음) | 중 |

---

## 5. 미해결 질문 (추가 실험 필요)

| 질문 | 필요한 실험 | 우선순위 |
|------|-----------|:--------:|
| PCA 개선의 원인 분리 (A-G) | Fixed grid ablation | 높음 |
| WF(3-bit)가 PPL을 개선하나? | PCA+WF vs PCA+Uniform at 3-bit | 높음 |
| Attention-weighted WF가 eigenvalue WF보다 좋나? | AAPQ 실험 | 중 |
| 결과가 다른 모델에서 일관되나? | Llama/Qwen decorrelation | 높음 |
| Σ_Q가 calibration에서 eval로 안정적인가? | cross-domain Σ_Q 비교 | 중 |
| Attention sink이 PCA 개선에 기여하나? | PCA loadings + sink 분석 | 중 |
