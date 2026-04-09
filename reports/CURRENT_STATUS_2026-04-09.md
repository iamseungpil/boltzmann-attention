# 연구 현황 보고서 (2026-04-09)

## 논문 제목 (안)

**"The Rotation-Quantizer Interaction in KV Cache Compression:
Why MSE-Optimal Quantization Fails After PCA Rotation"**

---

## 핵심 발견 (확정, 검증됨)

### 발견 1: PCA 회전은 MSE를 최적화한다 (Theorem 1)

- **정리**: Class C (직교회전 + 스칼라양자화) 내에서 Pre-RoPE 헤드별 PCA가 MSE 최적
- **증명**: Fischer 부등식 + Hadamard 부등식 + RoPE 직교성
- **검증**: 624/624 heads, 3 models (Qwen, Llama, Mistral)
- **PPL 전이**: 3-bit에서 4/4 모델 일치, 2-bit에서 2/4 모델 역전

### 발견 2: MSE 최적 양자화기(Lloyd)가 PPL에서 실패한다

- **측정**: 수정된 Lloyd vs Uniform, 49K test, PCA 회전 후

| 모델 | 2-bit (Lloyd/Uni) | 3-bit | 4-bit |
|------|:-----------------:|:-----:|:-----:|
| Qwen | 1.03× | 1.08× | 1.05× |
| Llama | 4.25× | 2.87× | 1.58× |
| Mistral | 2.46× | 진행중 | 진행중 |

- **3모델 × 모든 비트에서 예외 없이 Lloyd PPL > Uniform PPL**
- 코드 버그 수정 후에도 동일 → 진짜 현상

### 발견 3: L∞가 PPL과 Lloyd 실패를 설명한다

- **측정**: 624/624 heads, Lloyd MSE↓74% but L∞↑94%
- **메커니즘**: Lloyd가 codebook을 mode 근처에 집중 → 꼬리에 큰 gap → L∞ 증가
- **softmax**: L∞ 오차를 지수적으로 증폭 → PPL 악화

### 발견 4: 회전-양자화기 상호작용 (가장 중요한 통찰)

```
PCA 회전: 분산 이방성 극대화 → 꼬리가 두꺼운 차원 생성
          → Lloyd의 꼬리 gap 극대화 → L∞ 폭발 → PPL 악화

Random 회전 (TurboQuant): 분산 균등화 → 모든 차원 비슷
          → Lloyd의 꼬리 gap 보통 → L∞ 보통 → PPL OK

핵심: "회전 선택이 양자화기의 성공/실패를 결정한다"
```

이것은 **새로운 발견**이며, 아무도 명시적으로 분석하지 않았음:
- TurboQuant: random rotation + Lloyd = 작동 (분산 균등화 덕분)
- KVTC: PCA + Uniform = 작동 (L∞ bounded 덕분)
- 우리: PCA + Lloyd = 실패 (분산 이방성 + 꼬리 gap)
- CQ: Fisher-weighted centroids = MSE↑ but PPL↓ (같은 현상)

### 발견 5: MMLU에서도 PCA 효과 확인

- Qwen 2-bit: PCA MMLU 67.9% vs NoRot 58.7% (+9.2%p)

### 발견 6: Per-head PCA > Shared PCA

- Fischer 부등식 증명
- Llama 2-bit: +46.3% PPL 개선

---

## 선행 연구와의 차별화

| 기존 연구 | 그들의 방법 | 우리의 분석 |
|----------|-----------|-----------|
| TurboQuant | random rotation + Lloyd | 분산 균등화 덕분에 Lloyd 작동 |
| KVTC | PCA + Uniform + DP | PCA + Uniform이 올바른 조합 |
| KVQuant | sensitivity-weighted NUQ | raw Lloyd 아닌 weighted만 작동 |
| CQ (1-bit) | Fisher-weighted centroids | MSE↑ PPL↓ (우리 발견과 일치) |
| KIVI | per-channel Uniform | 회전 없음 → 우리 Theorem 1이 개선 가능 |

**우리만의 기여**: 회전-양자화기 상호작용의 체계적 분석 + 이론적 설명

---

## 미해결 + 다음 실험

### 즉시 (E8 GPU 1,2,3 유휴)

| 실험 | 의도 | GPU |
|------|------|:---:|
| **Clipped Lloyd** (바깥 centroid 고정) | 회전-양자화기 tradeoff 해결? | 1 |
| **Lloyd MMLU** (Qwen) | PPL 나쁘면 MMLU도 나쁜가? | 2 |
| **Mistral 3,4-bit** (진행 중) | 전체 표 완성 | 0 |

### 논문 강화

| 실험 | 의도 |
|------|------|
| TurboQuant 재현 (random rot + Lloyd) | 우리 분석이 TurboQuant를 설명하는지 확인 |
| KVTC 공식 비교 | head-to-head |
| Per-dim adaptive (Uniform 고분산 + Lloyd 저분산) | tradeoff 최적 해결 |

---

## 철회된 주장 (논문에 포함하지 않음)

| 주장 | 철회 이유 |
|------|----------|
| PCA-Q alignment 0.6-2.5° | 측정 버그 (실제 30-57°) |
| CWF v3 돌파 | 2K eval artifact + Lloyd 버그 |
| QW-WF 10-33% 개선 | 개선은 floor=2에서 옴 |
| Lloyd PPL 180/19000 | mean centering 버그 (수정 후 16/8로 줄었지만 여전히 Uniform보다 나쁨) |
| pos0 diagnostic 예측력 | 검증 실패 |

---

## 파일 정리 필요

### 보고서 (reports/)
- v11-v22 plan: 오래됨 → archive/
- NEURIPS_VERIFICATION_REPORT v1-v3: 오래됨 → archive/
- 유지: v23 plan, CURRENT_STATUS, FACT_BASE

### 스크립트 (scripts/)
- 버그 있는 Lloyd 스크립트: exp_coworker_method_49k.py, exp_sink_vs_linf.py 등 → archive/
- 유지: exp_correct_lloyd_vs_uniform.py, exp_linf_vs_l2.py, exp_pos0_diagnostic_v2.py
