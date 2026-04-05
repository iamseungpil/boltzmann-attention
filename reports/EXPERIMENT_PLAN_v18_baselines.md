# 실험 계획 v18: 선행연구 Baseline 비교 (경로 A 강화)

**날짜**: 2026-04-05
**목표**: 공개 코드가 있는 선행연구와 동일 프로토콜로 공정 비교
**인프라**: tops caiman GPU 1, 2 (각 80GB A100)

---

## 재현 가능한 Baseline 우선순위

| 우선순위 | 방법 | KV-only? | 코드 | 시간 | 이유 |
|:--------:|------|:--------:|------|:----:|------|
| 1 | **KIVI** | ✅ | github.com/jy-yuan/KIVI | 1일 | 가장 직접적 KV-only baseline, ICML'24 |
| 2 | **GEAR** | ✅ | github.com/HaoKang-Timmy/GEAR | 1일 | residual KV 양자화, NeurIPS'24 |
| 3 | **KVQuant** | ✅ | github.com/SqueezeAILab/KVQuant | 1일 | Pre-RoPE 채널별, NeurIPS'24 |
| 4 | SpinQuant | ❌ (W+KV) | github.com/facebookresearch/SpinQuant | 2-3일 | 학습 필요, ICLR'25 |
| 5 | QuaRot | ❌ (W+KV) | github.com/spcl/quarot | 1-2일 | Hadamard, weight 포함 |

## Exp V18-1: KIVI 비교

```
의도: 가장 널리 사용되는 KV-only baseline (KIVI)과 공정 비교.
      KIVI는 회전 없이 채널별/토큰별 비대칭 양자화를 사용하므로,
      우리 프레임워크에서는 A=0 (항등 회전)에 해당한다.
      Pre-RoPE PCA 회전이 무회전 + 채널별 양자화를 이기는지 확인.

가설:
  H1: Pre-RoPE PCA + uniform > KIVI at 2-bit PPL
  H2: 이득은 KIVI가 사용하지 않는 "회전에 의한 decorrelation" 때문
  H3: KIVI의 비대칭 양자화가 우리의 symmetric보다 tail 처리에서 유리할 수 있음

검증:
  1. KIVI 코드 설치 (pip install 또는 git clone)
  2. 동일 모델(Llama-3.1-8B)에서 KIVI 2/3/4-bit PPL 측정
  3. 우리 Pre-RoPE PCA 결과와 직접 비교
  4. 동일 WikiText-2 test, 동일 context length

  GPU: tops caiman GPU 1
  시간: ~4시간 (모델 로드 + 3 비트 × PPL 측정)
```

## Exp V18-2: GEAR 비교

```
의도: Residual 기반 KV 양자화 (GEAR)와 비교.
      GEAR는 저비트 양자화 후 잔차를 SVD로 보정한다.
      이것은 우리의 "회전 선택" 접근과 직교적이며,
      결합 가능성도 있다.

가설:
  H1: GEAR의 residual 보정이 저비트에서 효과적일 수 있음
  H2: Pre-RoPE PCA + GEAR의 결합이 가능하다면 추가 이득

검증:
  1. GEAR 코드 설치
  2. Llama-3.1-8B에서 GEAR 2/3/4-bit PPL 측정
  3. 우리 결과와 직접 비교

  GPU: tops caiman GPU 2
  시간: ~4시간
```

## Exp V18-3: Downstream Task (MMLU)

```
의도: PPL 이득이 downstream 정확도로 전이되는지 확인.
      선행연구들은 모두 MMLU를 보고하므로, 이것 하나만이라도 추가하면
      논문의 실험 범위가 크게 확장된다.

가설:
  H1: Pre-RoPE PCA의 MMLU accuracy > No rotation MMLU accuracy (2-bit)
  H2: 정확도 순서가 PPL 순서와 일치

검증:
  1. lm-evaluation-harness + k_proj hook 방식
  2. Llama-3.1-8B × {FP16, No rot, Random rot, Pre-RoPE PCA} × 2-bit
  3. MMLU 5-shot

  GPU: tops caiman GPU 1 (KIVI 끝난 후)
  시간: ~3시간/방법 × 4방법 = ~12시간
```

---

## 실행 순서 (tops caiman GPU 1, 2 병렬)

```
GPU 1:
  [1] KIVI 설치 + Llama PPL (4시간)
  [2] MMLU 평가 (12시간)

GPU 2:
  [1] GEAR 설치 + Llama PPL (4시간)
  [2] KVQuant 설치 + Llama PPL (4시간, 가능하면)

총 예상: ~16시간 (하루 + 밤새)
```

## 성공 기준

NeurIPS Table 1에 추가되어야 할 행:
- FP16 (baseline)
- No rotation + per-channel uniform (우리 구현)
- KIVI (공식 코드)
- GEAR (공식 코드)
- Haar random rotation (우리 구현, TurboQuant 회전 성분)
- Pre-RoPE PCA + uniform (본 논문)
- Pre-RoPE PCA + WF(floor=2) (본 논문)
