# FOKVQ 실험 계획서 v12

**문서 버전**: v12.0
**작성일**: 2026-04-02
**기반**: v11.2 + Needle-in-a-Haystack(NIAH) 벤치마크 추가 + 전체 실험 재구성
**핵심 변경**:
- **NIAH 벤치마크 추가**: TurboQuant(ICLR 2026)이 사용한 long-context 검색 평가 도입
- PPL + NIAH 이중 평가 체계로 NeurIPS 리뷰어 요구 사항 대응
- 논문 Table 구성 확정: Table 1(PPL), Table 2(NIAH), Table 3(Ablation)
**코드베이스**: https://github.com/iamseungpil/boltzmann-attention

---

## 0. v11.2 → v12 변경사항

### 0.1 왜 NIAH가 필요한가

| 벤치마크 | 측정 | KV 양자화 민감도 | 비고 |
|---------|------|----------------|------|
| WikiText-2 PPL | 평균적 다음 토큰 예측 | 낮음 — 오차 평균화 | 차이가 작게 보임 |
| **NIAH** | **긴 컨텍스트 특정 정보 검색** | **높음 — 단일 key 정확 attend 필수** | **양자화 차별점 극대화** |

PPL은 전체 토큰의 평균 NLL → 양자화 오차가 평균에 묻힘.
NIAH는 "건초더미에서 바늘 찾기" → 특정 위치의 K를 정확히 attend해야 성공.
양자화로 그 K가 왜곡되면 즉시 실패 → **FOKVQ의 비균일 bit 할당이 진가를 발휘하는 벤치마크**.

### 0.2 TurboQuant의 NIAH 결과 (우리의 비교 기준)

TurboQuant(ICLR 2026) 보고 수치 (4x 압축):

| Method | NIAH Score |
|--------|-----------|
| Full-precision | 1.000 |
| **TurboQuant** | **0.997** |
| KIVI | 0.981 |
| PyramidKV | 0.895 |
| SnapKV | 0.858 |

**우리 목표**: `fokvq_full` NIAH score > KIVI(0.981), 가능하면 TurboQuant(0.997)에 근접.

### 0.3 전체 실험 구조 (v12)

```
평가 체계 1: WikiText-2 PPL (v3 프로토콜, 100% K 양자화)
  → 13 methods x 3 bits x 2 models = 78 runs
  → 논문 Table 1

평가 체계 2: Needle-in-a-Haystack (신규)
  → 13 methods x {4K, 8K, 16K, 32K} context x needle depth {10%, 30%, 50%, 70%, 90%}
  → 논문 Table 2 + Figure (heatmap)

평가 체계 3: E1/E2/E3 Ablation
  → fokvq, fokvq_qw, fokvq_e2, fokvq_e3, fokvq_full
  → 논문 Table 3
```

---

## 1. 평가 체계 1: WikiText-2 PPL (v11.2에서 승계)

### 1.1 프로토콜

- v3: 100% K 양자화 (attention 내부 monkey-patch)
- Non-overlapping chunks
- KVQuant/KIVI 논문과 동일 기준

### 1.2 비교 대상: 13 methods

| # | Method | 논문 | 구현 상태 |
|---|--------|------|----------|
| 1 | FP16 | baseline | 완료 |
| 2 | Uniform | baseline | 완료 |
| 3 | KIVI | ICML 2024 | 완료 |
| 4 | QuIP# | 2024 | 완료 |
| 5 | KVQuant | NeurIPS 2024 | 완료 |
| 6 | GEAR | NeurIPS 2024 | 완료 |
| 7 | ZipCache | NeurIPS 2024 | 완료 |
| 8 | TurboQuant | ICLR 2026 | 완료 |
| 9 | FOKVQ | Ours | 완료 |
| 10 | FOKVQ-QW (E1) | Ours | 완료 |
| 11 | FOKVQ-E2 | Ours | 완료 |
| 12 | FOKVQ-E3 | Ours | 완료 |
| 13 | FOKVQ-Full | Ours (E1+E2+E3) | 완료 |

### 1.3 모델

| 모델 | 아키텍처 | RoPE | GPU | 상태 |
|------|---------|------|-----|------|
| GPT-2 Medium | MHA, d=64 | X | GPU 1 | E1/E2/E3 실행 중, ALL 대기 |
| Qwen2.5-7B | GQA, d=128 | O | GPU 0 | ALL 실행 중 |
| Mistral-7B | GQA, d=128 | O | - | Week 1 |
| Llama-3-8B | GQA, d=128 | O | - | Week 1 |

### 1.4 GPT-2 v3 중간 결과 (100% 양자화, 핵심 발견)

| Method | 2bit | 3bit | 4bit | FP16 |
|--------|------|------|------|------|
| FP16 | | | | 21.36 |
| Uniform | **99.24** | 24.97 | 21.77 | |
| KIVI | **138.56** | 27.15 | 21.96 | |
| FOKVQ | **25.65** | 실행 중 | 대기 | |

**2bit**: FOKVQ(25.65) = Uniform(99.24)의 1/4, KIVI(138.56)의 1/5. v2(50%)에서 안 보이던 차이.

**KIVI가 GPT-2에서 Uniform보다 나쁜 이유 (신규 발견)**:
- Per-channel(seq 축) 양자화: seq=1024 범위에서 2bit(4 levels) → step 매우 큼
- GPT-2는 RoPE 없음 → KIVI의 per-channel 이점(RoPE 차원 쌍 보존) 없음
- KIVI 논문은 GPT-2(non-RoPE) 미평가

---

## 2. 평가 체계 2: Needle-in-a-Haystack (NIAH) — 신규

### 2.1 벤치마크 설명

긴 컨텍스트 중간에 특정 "바늘(needle)" 문장을 삽입하고, 모델이 이를 정확히 검색하는지 평가.

```
Haystack = [filler text ... needle "The secret code is 7392" ... filler text]
Query = "What is the secret code?"
Success = 모델이 "7392"를 정확히 생성
```

### 2.2 실험 설계

**모델**: Qwen2.5-7B (128K context), Llama-3-8B (8K context)
- GPT-2(1024 max)는 NIAH에 부적합 → 제외

**컨텍스트 길이**: 4K, 8K, 16K, 32K tokens

**Needle 위치 (depth)**: 10%, 30%, 50%, 70%, 90% of context

**Methods**: 13개 전부 (PPL과 동일)

**Bits**: 2, 3, 4

**측정**: needle 검색 성공률 (0~1), 5회 반복 평균

**총 runs**: 13 methods x 4 lengths x 5 depths x 3 bits = 780 evaluations per model
(각 evaluation은 짧은 generation이므로 개당 ~10초, 총 ~2시간)

### 2.3 NIAH가 FOKVQ에 유리한 이유

FOKVQ의 비균일 bit 할당: 상위 PCA 주성분에 더 많은 bit → **K의 핵심 정보 보존**.
Needle token의 K 에너지가 상위 주성분에 집중되어 있다면, FOKVQ가 그 정보를 더 잘 보존.
반면 Uniform/KIVI는 모든 차원 동일 bit → needle의 핵심 정보가 희석.

**E1(Q-가중 PCA)의 추가 이점**: NIAH에서 query는 needle 위치를 정확히 타겟. Q-가중 축이 이 query 방향의 K 정보를 보존하면 → needle 검색 성공률 극대화.

### 2.4 구현 방법

```python
# v3의 attention monkey-patch를 그대로 사용
# 차이: PPL 대신 generation + needle 검색 판정

def evaluate_niah(model, tokenizer, method, bits, context_len, needle_depth):
    """
    1. haystack 텍스트 생성 (context_len tokens)
    2. needle_depth 위치에 needle 문장 삽입
    3. v3 attention patch로 K 양자화 활성화
    4. model.generate()로 답변 생성
    5. needle 내용이 답변에 포함되는지 판정
    """
```

v3의 `AttentionKQuantPatcher`를 그대로 재사용 → 구현 비용 최소.

### 2.5 성공 기준

| 기준 | 조건 |
|------|------|
| **HOME RUN** | fokvq_full NIAH > TurboQuant(0.997) |
| **STRONG PASS** | fokvq_full NIAH > KIVI(0.981) |
| **PASS** | fokvq_full NIAH > fokvq |
| **FAIL** | fokvq_full ≤ fokvq → E1/E2/E3가 NIAH에서도 효과 없음 |

### 2.6 논문 표현

**Table 2**: NIAH Score by Method and Context Length

```
| Method       | 4K   | 8K   | 16K  | 32K  | Avg  |
|-------------|------|------|------|------|------|
| FP16        | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TurboQuant  | ?.?? | ?.?? | ?.?? | ?.?? | ?.?? |
| KIVI        | ?.?? | ?.?? | ?.?? | ?.?? | ?.?? |
| FOKVQ-Full  | ?.?? | ?.?? | ?.?? | ?.?? | ?.?? |
| ...         |      |      |      |      |      |
```

**Figure**: NIAH Heatmap (depth x context_len, per method)
- TurboQuant 논문과 동일 형식 → 직접 비교 가능

---

## 3. E1/E2/E3 실험 (v11.2에서 승계)

### 3.1 E1: Q-가중 PCA (`fokvq_qw`) — 구현 완료

축 개선: `Σ_{K|Q} = Σ_Q^{1/2} · Σ_K · Σ_Q^{1/2}` 의 고유벡터
- Q 방향에서 K의 비등방성을 포착
- Q·K 내적 오차 최소화

### 3.2 E2: 축별 Lloyd-Max (`fokvq_e2`) — 구현 완료

양자화기 개선: 각 PCA 축의 실분포에서 직접 codebook fitting
- Synthetic test에서 FOKVQ 대비 54% MSE 감소

### 3.3 E3: MK 왜곡 (`fokvq_e3`) — 구현 완료

왜곡 측도 개선: 유클리드 MSE → 마할라노비스 `(K-K̂)^T Σ_K^{-1} (K-K̂)`
- 분산 작은 축의 오차에 더 큰 페널티 → Fisher-Rao 최적

### 3.4 E1+E2+E3 조합 (`fokvq_full`) — 구현 완료

```
fokvq_full = Q-가중 PCA축 + MK-가중 Lloyd-Max codebook
```
세 개선이 **독립적 축**이므로 간섭 없이 조합.

---

## 4. 현재 실행 상태

| 실험 | GPU | 모델 | Methods | 상태 |
|------|-----|------|---------|------|
| GPT-2 E1/E2/E3 | GPU 1 | gpt2-medium | 8개 | 실행 중 |
| Qwen ALL | GPU 0 | Qwen2.5-7B | 13개 (SOTA 포함) | 실행 중 |
| GPT-2 ALL | GPU 1 | gpt2-medium | 13개 | 대기 (GPT-2 E1 후) |
| NIAH | - | Qwen2.5-7B | 13개 | **구현 예정** |

---

## 5. 실행 순서 (NeurIPS 마감 역산: 05-06)

```
[완료] E1/E2/E3/full + SOTA 5종 구현 (04-02)
[실행 중] GPT-2 E1/E2/E3 + Qwen ALL (04-02 ~ 04-03)

Week 1 (04-03 ~ 04-07):
  04-03 AM: GPT-2 ALL 13 methods 실행 (~6h)
  04-03 PM: NIAH 벤치마크 구현 (~3h)
  04-04: Qwen NIAH 실행 (13 methods, ~2h)
  04-04: GPT-2 + Qwen PPL 결과 분석 → Table 1 초안
  04-05: Mistral-7B PPL + NIAH (~8h)
  04-06: Llama-3-8B PPL + NIAH (~8h)
  04-07: 전체 결과 정리 + 논문 방향 확정

Week 2 (04-08 ~ 04-14):
  NIAH heatmap 생성 + 분석
  LongBench downstream (선택적)
  Latency 벤치마크
  논문 초안 작성 시작

Week 3-5 (04-15 ~ 05-06):
  논문 완성 + 부록 + 코드 정리 + 제출
```

---

## 6. 논문 구성 (v12 확정)

### Table 1: WikiText-2 PPL (주력)

```
| Model      | Method       | FP16  | 4bit  | 3bit  | 2bit  |
|------------|-------------|-------|-------|-------|-------|
| GPT-2 Med  | Uniform     | 21.36 | ?.??  | ?.??  | 99.24 |
|            | KIVI        |       | ?.??  | ?.??  |138.56 |
|            | TurboQuant  |       | ?.??  | ?.??  | ?.??  |
|            | FOKVQ       |       | ?.??  | ?.??  | 25.65 |
|            | FOKVQ-Full  |       | ?.??  | ?.??  | ?.??  |
| Qwen2.5-7B | (동일)      | ?.??  | ...   | ...   | ...   |
| Llama-3-8B | (동일)      | ?.??  | ...   | ...   | ...   |
```

### Table 2: NIAH Score (신규, TurboQuant 직접 비교)

```
| Method       | 4K   | 8K   | 16K  | 32K  | Avg  |
|-------------|------|------|------|------|------|
| FP16        | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TurboQuant  | ?.?? | ?.?? | ?.?? | ?.?? | ?.?? |
| KIVI        | ?.?? | ?.?? | ?.?? | ?.?? | ?.?? |
| FOKVQ-Full  | ?.?? | ?.?? | ?.?? | ?.?? | ?.?? |
```

### Table 3: E1/E2/E3 Ablation

```
| Variant     | Axis    | Quantizer    | Distortion | 2bit PPL | 3bit PPL |
|-------------|---------|-------------|------------|----------|----------|
| FOKVQ       | K-PCA   | Uniform     | Euclidean  | 25.65    | ?.??     |
| +E1 (QW)    | Q-PCA   | Uniform     | Euclidean  | ?.??     | ?.??     |
| +E2 (Lloyd) | K-PCA   | Lloyd-Max   | Euclidean  | ?.??     | ?.??     |
| +E3 (MK)    | K-PCA   | Lloyd-Max   | Mahalanobis| ?.??     | ?.??     |
| Full        | Q-PCA   | Lloyd-Max   | Mahalanobis| ?.??     | ?.??     |
```

### Figure 1: NIAH Heatmap

- X축: context length (4K, 8K, 16K, 32K)
- Y축: needle depth (10%, 30%, 50%, 70%, 90%)
- 색상: 검색 성공률 (0~1)
- 패널: FP16, KIVI, TurboQuant, FOKVQ-Full (4개 비교)

---

## 7. 판정 매트릭스

### 7.1 PPL + NIAH 종합 판정

| PPL 결과 | NIAH 결과 | 논문 포지셔닝 |
|---------|----------|-------------|
| fokvq_full < KIVI | NIAH > KIVI | **최강**: PPL + long-context 모두 SOTA |
| fokvq_full ≈ KIVI | NIAH > KIVI | **strong**: PPL 동등 + long-context 우위 |
| fokvq_full > KIVI | **NIAH > KIVI** | **viable**: PPL 열위지만 실용적 long-context 우위 |
| fokvq_full > KIVI | NIAH ≤ KIVI | **재설계 필요** |

### 7.2 E1/E2/E3 Ablation 판정

(v11.2 §8.2와 동일)

| E1 | E2 | E3 | 해석 |
|----|----|----|------|
| PASS | PASS | PASS | 축 + 양자화기 + 왜곡 모두 → fokvq_full 최강 |
| PASS | FAIL | - | 축이 핵심 → Q-가중 PCA 논문 |
| FAIL | PASS | - | 양자화기 핵심 → 적응 양자화기 논문 |
| FAIL | FAIL | FAIL | PCA 기반 한계 → honest negative |

---

## 8. 이전 계획서와의 차이

| 항목 | v11.0 | v11.2 | v12 (본 문서) |
|------|-------|-------|--------------|
| PPL 프로토콜 | v2 (50%) | v3 (100%) | v3 (100%) |
| **NIAH 벤치마크** | **없음** | **없음** | **추가** |
| SOTA PPL 비교 | KL만 | 5종 구현 | 5종 구현 |
| 비교 method 수 | 4개 | 13개 | 13개 |
| 평가 차원 | PPL만 | PPL만 | **PPL + NIAH** |
| 논문 Table | 1개 | 1개 | **3개 + Figure** |
| TurboQuant 직접 비교 | 없음 | PPL만 | **PPL + NIAH** |
| NeurIPS 경쟁력 | 25% | 35% | **50%+** (NIAH 결과에 따라) |

---

## 9. 파일 목록

| 파일 | 위치 | 용도 |
|------|------|------|
| `exp4_2_v3_full_quant_ppl.py` | `4-2_ex/` | v3 메인 코드 (13 methods, E1/E2/E3/SOTA) |
| `exp4_2_standard_ppl_benchmark_v2.py` | `4-2_ex/` | v2 코드 (참조용) |
| `FOKVQ_CODE_REVIEW.md` | `4-2_ex/` | cowork vs 우리 코드 비교 |
| `V3_PROTOCOL_DESIGN.md` | `4-2_ex/` | v3 프로토콜 설계 |
| `EXPERIMENT_REPORT_20260401.md` | `4-2_ex/` | 04-01 실험 결과 보고서 |
| `EXPERIMENT_PLAN_v11.1.md` | `4-2_ex/` | v11.1 (superseded) |
| `EXPERIMENT_PLAN_v11.2.md` | `4-2_ex/` | v11.2 (superseded) |
| **`EXPERIMENT_PLAN_v12.md`** | **`4-2_ex/`** | **본 문서 (최신)** |
| `run_v3_gpt2_all.sh` | `4-2_ex/` | GPT-2 ALL 13 methods |
| `run_v3_qwen_all.sh` | `4-2_ex/` | Qwen ALL 13 methods |
| `exp_niah_benchmark.py` | `4-2_ex/` | **NIAH 벤치마크 (구현 예정)** |

---

**Date**: 2026-04-02
**Author**: CDP PoC Team
**Status**: PPL 13 methods 구현 완료 + 실행 중, NIAH 구현 예정
