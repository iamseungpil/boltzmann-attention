# Axis 2 / Axis 3 Failure Diagnosis - Run Guide

## Overview

V3 검증에서 발견된 두 이상 현상의 근본 원인을 특정하는 실험:
- **축 2 실패**: Lloyd-Max PPL 재앙 (Llama 2-bit 10.14 → 65.46)
- **축 3 부분 불일치**: WF(floor=1) 붕괴 (Qwen 2-bit 11.26 vs uniform 7.98)

## 실험 3개 (Critical Path: E1 → E4 → E2, 총 ~8시간)

| 실험 | 가설 | 내용 | 예상 시간/모델 |
|------|------|------|----------------|
| **E1 (MK)** | A1: 쿼리 메트릭 미스매치 | Σ_q 기반 Mahalanobis Lloyd-Max | ~1.5h |
| **E4 (Integer WF)** | B2: 1-bit 금지 | b_j ∈ {0, ≥2} 제약 water-filling | ~0.7h |
| **E2 (Sink)** | A2: 싱크 토큰 파괴 | n_sink 토큰 FP16 보존 | ~0.7h |

## Quick Start

### 환경 요구사항
- GPU: A100 또는 A6000 (48GB+)
- Python 패키지: transformers, torch, datasets, numpy

### 3모델 병렬 실행 (2 GPU 활용)

```bash
cd scripts/

# GPU 0: Qwen, GPU 1: Llama (동시)
CUDA_VISIBLE_DEVICES=0 bash run_diagnosis_qwen.sh &
CUDA_VISIBLE_DEVICES=1 bash run_diagnosis_llama.sh &
wait

# 그 후 Mistral
CUDA_VISIBLE_DEVICES=0 bash run_diagnosis_mistral.sh
```

### 개별 실험만 실행

```bash
# E1만 (Qwen)
python exp_axis2_axis3_diagnosis.py --experiment E1 \
    --model-name Qwen/Qwen2.5-7B --model-key qwen2.5-7b \
    --device cuda:0 --dtype bfloat16 --context-len 2048 \
    --bits 2 3 --attn-implementation eager \
    --output-dir ../results/diagnosis

# E4만 (Llama)
python exp_axis2_axis3_diagnosis.py --experiment E4 \
    --model-name meta-llama/Llama-3.1-8B --model-key llama-3.1-8b \
    --device cuda:0 --dtype bfloat16 --context-len 2048 \
    --bits 2 3 --attn-implementation eager \
    --output-dir ../results/diagnosis
```

### Self-test (GPU 불필요)

```bash
python exp_axis2_axis3_diagnosis.py --self-test
```

## 결과 파일

`results/diagnosis/` 에 JSON 저장:
```
results/diagnosis/
├── qwen2.5-7b_diagnosis.json
├── llama-3.1-8b_diagnosis.json
├── mistral-7b-v0.3_diagnosis.json
└── run_diagnosis_*.log
```

### JSON 스키마
```json
{
  "model_name": "Qwen/Qwen2.5-7B",
  "fp16_ppl": 8.12,
  "results": [
    {
      "experiment": "E1",
      "method": "mk",
      "bits": 2,
      "ppl": 9.45,
      "mse": 0.0012,
      "config": {"mk_calib_tokens": 2048}
    }
  ]
}
```

## 판정 기준 요약

### E1 (MK): A1 쿼리 메트릭 미스매치
- MK PPL < Uniform PPL (3모델 2-bit) → **A1 확정**, 축 2 클레임 완전 복구
- MK ≈ Gaussian Lloyd → **A1 기각**, A2/A3로

### E4 (Integer WF): B2 정수 제약
- WF(no-one) ≈ WF(floor=2) → **B2 확정**, floor=2는 이론적 최적
- WF(no-one) > WF(floor=2) → **B2 기각**

### E2 (Sink): A2 싱크 토큰
- n_sink=4만으로 PPL 재앙 소멸 → **A2 확정**
- 효과 없음 → **A2 기각**

## Coworker 작업 분배 제안

| 머신 | GPU | 모델 | 담당 |
|------|-----|------|------|
| 로컬 (A6000×2) | 0,1 | Qwen + Llama 동시 | mais |
| Azure (A100) | 0 | Mistral | coworker |

또는:

| 머신 | 모델 |
|------|------|
| 로컬 | Qwen |
| Azure | Llama + Mistral |
