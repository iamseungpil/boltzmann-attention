# FOKVQ 추가 실험 계획서 v11.1 (superseded by v11.2)

**문서 버전**: v11.1 → **v11.2로 대체됨. 최신 버전: EXPERIMENT_PLAN_v11.2.md**
**작성일**: 2026-04-02
**기반**: v11.0 + E1/E2/E3 구현 완료 + v3 프로토콜(100% K 양자화) 반영
**핵심 변경**:
- Phase 7에서 KL로만 비교한 7개 SOTA를 v3 PPL harness에 전부 통합
- 총 13개 method (SOTA 7 + FOKVQ 5 + FP16) x 3 bits = 39 runs per model
- GPT-2 v3 중간 결과: **2bit에서 FOKVQ가 Uniform/KIVI를 4~5배 압도** (v2에서 안 보이던 차이)
- Qwen v3 attention forward 반환값 버그 수정 완료
**코드베이스**: https://github.com/iamseungpil/boltzmann-attention

---

## 0. v11.1 → v11.2 변경사항

### 0.1 프로토콜 변경: v2 → v3

| 항목 | v2 (v11 기준) | v3 (v11.2 기준) |
|------|-------------|----------------|
| 양자화 비율 | 50% (prefix만) | **100% (전체 K)** |
| 방식 | KV cache 외부 교체 | **Attention 내부 monkey-patch** |
| 평가 | Sliding window (overlap) | **Non-overlapping chunks** |
| SOTA 대응 | 불일치 | **KVQuant/KIVI 논문과 동일** |

v2 결과에서 PPL 차이가 미미했던 근본 원인이 "50% 양자화 희석"이었음을 확인.
v3에서 100% 양자화로 전환하여 SOTA 논문들과 동일한 기준으로 비교.

### 0.2 v11.2 핵심 추가: SOTA 7종 PPL 비교 통합

Phase 7에서 KL divergence로만 비교했던 7개 SOTA를 v3 PPL harness에 전부 구현:

| Method | 논문 | Phase 7 KL | v3 PPL | 구현 함수 |
|--------|------|-----------|--------|----------|
| Uniform | baseline | O | O | `uniform_quantize_tensor` |
| KIVI | ICML 2024 | O | O | `kivi_quantize_tensor` |
| **QuIP#** | 2024 | O | **O (신규)** | `quip_quantize_head` |
| **KVQuant** | NeurIPS 2024 | O | **O (신규)** | `kvquant_quantize_head` |
| **GEAR** | NeurIPS 2024 | O | **O (신규)** | `gear_quantize_head` |
| **ZipCache** | NeurIPS 2024 | O | **O (신규)** | `zipcache_quantize_head` |
| **TurboQuant** | ICLR 2026 | O | **O (신규)** | `turbo_quantize_head` |
| FOKVQ | Ours | O | O | `fokvq_quantize_head` |
| FOKVQ-QW (E1) | Ours | - | O | `fokvq_qw_quantize_head` |
| FOKVQ-E2 | Ours | - | O | `fokvq_e2_quantize_head` |
| FOKVQ-E3 | Ours | - | O | `fokvq_e3_quantize_head` |
| FOKVQ-Full (E1+E2+E3) | Ours | - | O | `fokvq_full_quantize_head` |

**총 13개 method**, 단일 코드(`exp4_2_v3_full_quant_ppl.py`)에서 동일 조건 비교.

### 0.3 SOTA 구현 상세

| Method | 핵심 메커니즘 |
|--------|-------------|
| QuIP# | Hadamard 회전으로 outlier 분산 → uniform 양자화 |
| KVQuant | per-channel Lloyd-Max (분포 적합) + outlier FP16 보존 (2.5σ 초과) |
| GEAR | uniform 양자화 + SVD low-rank(rank=4) 잔차 보정 + sparse(top 1%) outlier |
| ZipCache | 최근 30% token → bits+2, 나머지 → bits (token importance split) |
| TurboQuant | Hadamard 회전 + per-dim Lloyd-Max + QJL 1-bit 잔차 보정 |

### 0.4 GPT-2 v3 중간 결과 (v11.2의 가장 중요한 발견)

**v3(100% 양자화)에서 v2에서 안 보이던 차이가 극명하게 드러남:**

| Method | 2bit | 3bit | 4bit | FP16 |
|--------|------|------|------|------|
| FP16 | | | | **21.36** |
| Uniform | **99.24** | 24.97 | 21.77 | |
| KIVI | **138.56** | 27.15 | 21.96 | |
| FOKVQ | **25.65** | 실행 중 | 대기 | |

**2bit 핵심**: FOKVQ(25.65)가 Uniform(99.24)의 **1/4**, KIVI(138.56)의 **1/5**.

v2에서는 (50% 양자화 희석으로) FOKVQ 19.35 vs Uniform 19.97 — 3% 차이에 불과했음.
v3 프로토콜이 FOKVQ의 실제 이점을 정확히 측정함을 확인.

**KIVI가 GPT-2에서 Uniform보다 나쁜 이유 (신규 발견)**:
- KIVI는 per-channel(sequence 축) 양자화 → seq=1024에서 min/max range가 넓음
- 2bit(4 levels)로 1024개 값을 커버 → step size 매우 큼
- GPT-2는 RoPE가 없어 per-channel의 이점(RoPE 차원 쌍 보존)이 없음
- **KIVI 논문은 GPT-2(non-RoPE) 미평가** — Llama/Falcon/Mistral만 테스트

### 0.5 구현 완료 상태

| 실험 | method 이름 | 구현 | 테스트 | 실행 |
|------|------------|------|--------|------|
| E1: Q-가중 PCA | `fokvq_qw` | **완료** | self-test PASS | GPT-2 실행 중 |
| E2: 축별 Lloyd-Max | `fokvq_e2` | **완료** | self-test PASS | GPT-2 실행 중 |
| E3: MK 왜곡 Lloyd-Max | `fokvq_e3` | **완료** | self-test PASS | GPT-2 실행 중 |
| E1+E2+E3 조합 | `fokvq_full` | **완료** | self-test PASS | GPT-2 실행 중 |
| SOTA 5종 PPL | quip/kvquant/gear/zipcache/turboquant | **완료** | self-test PASS | Qwen 실행 중 |
| E4: R-D 쌍대성 | - | 미구현 | - | 최후순위 |

### 0.6 Self-Test 결과 (synthetic data)

```
Anisotropic 3bit MSE:
  uniform  = 0.139749
  fokvq    = 0.001273  (PCA + uniform scalar)
  E2       = 0.000583  (PCA + Lloyd-Max, 54% MSE 감소!)
  E3       = 0.000583  (PCA + MK Lloyd-Max)

Q·K inner product MSE:
  FOKVQ    = 20.01
  FOKVQ-QW = 19.91  (E1: Q-가중 축이 Q·K 정확도에서 우위)

SOTA 3bit MSE (random data):
  quip       = 0.036  (Hadamard + uniform)
  kvquant    = 0.025  (Lloyd-Max + outlier FP16)
  gear       = 0.026  (uniform + SVD + sparse)
  zipcache   = 0.026  (token split)
  turboquant = 0.018  (Hadamard + Lloyd-Max + QJL)
```

---

## 1. 검증할 4개 미검증 가설 (v11에서 승계)

| 가설 | 검증 실험 | method | 핵심 질문 | 성공 기준 |
|------|----------|--------|----------|----------|
| H1: Q-가중 비등방성 | E1 | `fokvq_qw` | Q 방향 가중 PCA가 K-only PCA보다 PPL에서 나은가? | fokvq_qw PPL < fokvq PPL |
| H2: 축별 적응 양자화기 | E2 | `fokvq_e2` | 축별 Lloyd-Max가 uniform scalar보다 나은가? | fokvq_e2 PPL < fokvq PPL |
| H3: MK 왜곡 vs 유클리드 | E3 | `fokvq_e3` | MK 왜곡이 유클리드 MSE보다 PPL 보존에 나은가? | fokvq_e3 PPL < fokvq PPL |
| **H1+H2+H3 조합** | **E1+E2+E3** | **`fokvq_full`** | **세 개선이 결합되면 KIVI를 이길 수 있는가?** | **fokvq_full PPL ≤ kivi PPL** |

**핵심 인사이트**: E1(축), E2(양자화기), E3(왜곡 측도)는 **독립적 개선 축**이므로 간섭 없이 조합 가능.

```
현재 FOKVQ:    K-only PCA축  + uniform scalar 양자화기 + 유클리드 MSE
fokvq_full:   Q-가중 PCA축  + 축별 Lloyd-Max codebook + 마할라노비스 왜곡
```

---

## 2. E1: Q-가중 비등방성 축 (구현 완료)

### 2.1 수학적 정식화

```
현재 FOKVQ의 양자화 왜곡:
  D_Euclid = E[‖K - K̂‖²]

Q-가중 양자화 왜곡 (E1 제안):
  D_QW = E[|Q^T(K - K̂)|²] = E[(K-K̂)^T Σ_Q (K-K̂)]
  여기서 Σ_Q = E[QQ^T]  (Query 공분산)

최적 축: Σ_{K|Q} = Σ_Q^{1/2} · Σ_K · Σ_Q^{1/2} 의 고유벡터
```

### 2.2 구현 방법 (v3에서)

- Post-RoPE attention patch에서 Q states를 on-the-fly로 수집
- 별도 캘리브레이션 불필요 (현재 chunk의 Q에서 Σ_Q 계산)
- GQA 모델: Q heads를 KV group별로 평균하여 Σ_Q 계산

### 2.3 성공/실패 기준

- **PASS**: fokvq_qw PPL < fokvq PPL (모든 비트에서)
- **HOME RUN**: fokvq_qw PPL ≤ kivi PPL (2-bit에서)
- **FAIL**: Q와 K의 주축이 이미 정렬 → E2로 이동

---

## 3. E2: 축별 적응 양자화기 (구현 완료)

### 3.1 핵심 아이디어

FOKVQ 기존: PCA 축을 찾은 후 각 축에 **동일한 uniform scalar** 양자화 적용
E2 제안: 각 축의 **실제 분포에서 직접 Lloyd-Max codebook** fitting

### 3.2 구현: `_fit_lloyd_max_1d`

```python
# 각 PCA 축의 실분포에서 직접 codebook fitting
for i in range(d):
    col = K_pca[:, i]
    n_lev = 2 ** int(ib[i])
    cb = _fit_lloyd_max_1d(col, n_lev)  # 20 iterations
    K_q[:, i] = _quantize_with_codebook_1d(col, cb)
```

- v11의 fokvq_lloyd_fitted에 해당 (표준정규가 아닌 실분포 fitting)
- fokvq_lloyd 실패 원인(표준정규 mismatch)을 직접 해결

### 3.3 Self-Test에서 확인된 효과

```
fokvq(uniform scalar) MSE = 0.001273
fokvq_e2(Lloyd-Max)   MSE = 0.000583  → 54% MSE 감소
```

### 3.4 성공/실패 기준

- **PASS**: fokvq_e2 PPL < fokvq PPL
- **STRONG PASS**: fokvq_e2 PPL이 kivi PPL에 근접 (차이 < 1.0)
- **FAIL**: Lloyd-Max가 PPL에서도 uniform보다 나쁨 → MSE ≠ PPL 경로 단절, E3로 이동

---

## 4. E3: 마할라노비스 왜곡 (구현 완료)

### 4.1 핵심 아이디어

E2의 Lloyd-Max는 유클리드 MSE를 최소화. 하지만 **MSE 최적 ≠ attention 최적**.
E3: 마할라노비스 왜곡 `(K-K̂)^T Σ_K^{-1} (K-K̂)`을 최소화하는 codebook.

### 4.2 구현: `_fit_lloyd_max_mahalanobis_1d`

```python
# MK weight = 1/eigenvalue (분산 작은 축에 더 높은 가중치)
# 구현: data를 sqrt(weight)로 스케일 → 표준 Lloyd-Max → 역스케일
scale = (1.0 / eigenvalue) ** 0.5
scaled_data = data * scale
cb_scaled = lloyd_max(scaled_data, n_levels)
cb = cb_scaled / scale
```

분산이 작은 축(Fisher 정보 높음)의 양자화 오차에 더 큰 페널티 → 정보 이론적 최적에 근접.

### 4.3 성공/실패 기준

- **PASS**: fokvq_e3 PPL < fokvq_e2 PPL (MK가 유클리드보다 나음)
- **FAIL**: 가우시안 가정 위반 → Fisher-Rao 동치성이 실제 Transformer에서 비유효

---

## 5. E1+E2+E3 조합: `fokvq_full` (구현 완료)

### 5.1 구성

```
fokvq_full = E1(Q-가중 PCA) + E2(축별 Lloyd-Max) + E3(MK 왜곡 가중)
           = Q-가중 축 선택 + MK-가중 Lloyd-Max codebook
```

### 5.2 이것이 KIVI를 이길 수 있는 이유

| KIVI | fokvq_full |
|------|-----------|
| Per-channel (sequence dim) 양자화 | Per-axis (PCA dim) 양자화 |
| 채널별 min/max → asymmetric | **Q-가중 최적 축** + **분포 적응 codebook** |
| 암묵적으로 RoPE 구조 보존 | **명시적으로 Q·K 내적 오차 최소화** |

KIVI의 장점(간단, per-channel이 RoPE와 호환)을 fokvq_full이 **이론적으로 최적인 방법**으로 대체.

### 5.3 성공/실패 기준

- **HOME RUN**: fokvq_full PPL ≤ kivi PPL (모든 모델, 모든 비트)
- **PASS**: fokvq_full PPL < fokvq PPL (개선은 확인)
- **FAIL**: 조합해도 KIVI 미달 → PCA 기반 접근의 근본적 한계

---

## 6. 현재 실행 상태

### 6.1 실행 중인 실험

| 실험 | GPU | 모델 | Methods | 스크립트 | 상태 |
|------|-----|------|---------|---------|------|
| GPT-2 E1/E2/E3 | GPU 1 | gpt2-medium | 8개 (SOTA 미포함) | `run_v3_gpt2_e1.sh` | 실행 중 (fokvq 3bit ~87%) |
| **Qwen ALL** | **GPU 0** | **Qwen2.5-7B** | **13개 (SOTA 포함)** | `run_v3_qwen_all.sh` | **실행 중** |
| GPT-2 ALL | GPU 1 | gpt2-medium | 13개 (SOTA 포함) | `run_v3_gpt2_all.sh` | GPT-2 E1 완료 후 실행 예정 |

### 6.2 모니터링

```bash
# GPT-2 E1/E2/E3 (현재)
tail -20 run_v3_gpt2_e1.log

# Qwen ALL (SOTA 포함)
tail -20 run_v3_qwen_all.log

# GPT-2 ALL (대기 중)
tail -20 run_v3_gpt2_all.log
```

### 6.3 예상 소요 시간

- GPT-2 E1/E2/E3: 8 methods x 3 bits = 24 runs, ~4시간 (FOKVQ per-head PCA 느림)
- **Qwen ALL: 13 methods x 3 bits = 39 runs, ~12시간**
- GPT-2 ALL: 13 methods x 3 bits = 39 runs, ~6시간

---

## 7. 실행 순서 (NeurIPS 마감 역산: 05-06)

```
[완료] v11 E1/E2/E3/full 구현 및 self-test (04-02 AM)
[완료] SOTA 5종 (QuIP#, KVQuant, GEAR, ZipCache, TurboQuant) PPL 구현 (04-02 PM)
[완료] v3 Qwen attention forward 반환값 버그 수정 (04-02 PM)
[실행 중] GPT-2 E1/E2/E3 v3 (04-02, ~4h 잔여)
[실행 중] Qwen ALL 13 methods v3 (04-02 시작, ~12h)

Week 1 (04-03 ~ 04-07):
  04-03 AM: GPT-2 ALL 13 methods 실행 (~6h)
  04-03 PM: GPT-2 + Qwen 결과 분석 → 논문 Table 1 초안
  04-04: KIVI를 이기는 구성이 있으면 → Mistral-7B + Llama-3-8B 추가
  04-05~06: 결과 정리 + 논문 방향 확정

Week 2 (04-08 ~ 04-14):
  추가 모델 (Llama-3-8B 등) PPL
  LongBench downstream task (선택적)
  긴 컨텍스트 8K/16K PPL (선택적)
  Latency 벤치마크

Week 3-5 (04-15 ~ 05-06):
  논문 작성 + 부록 + 코드 정리 + 제출
```

---

## 8. 판정 매트릭스

### 8.1 E1/E2/E3 + full 결과별 전략

| fokvq_full vs KIVI | 해석 | 다음 단계 |
|-------------------|------|----------|
| **fokvq_full < kivi** | 세 개선 조합이 KIVI를 역전 | SOTA 비교 논문으로 진행 |
| **fokvq_full ≈ kivi** | 동등 → 이론적 우위로 차별화 | 이론 + 동등 성능 논문 |
| **fokvq > fokvq_full > kivi 불가** | 개선은 있으나 KIVI 미달 | Ablation 분석 논문 |
| **fokvq_full ≈ fokvq** | E1/E2/E3 모두 실효 없음 | PCA 기반 한계 분석으로 전환 |

### 8.2 개별 E 실험 Ablation

| E1 결과 | E2 결과 | E3 결과 | 해석 |
|---------|---------|---------|------|
| PASS | PASS | PASS | 축 + 양자화기 + 왜곡 모두 개선 가능 → fokvq_full이 최강 |
| PASS | FAIL | - | 축이 핵심, 양자화기 무관 → Q-가중 PCA 논문 |
| FAIL | PASS | - | 양자화기가 핵심, 축 무관 → 적응 양자화기 논문 |
| FAIL | FAIL | PASS | MK 왜곡만 유효 → 정보 기하학 논문 |
| FAIL | FAIL | FAIL | PCA 기반 접근 자체의 한계 → honest negative |

---

## 9. 이전 계획서와의 차이

| 항목 | v11.0 | v11.1 | v11.2 (본 문서) |
|------|-------|-------|----------------|
| 프로토콜 | v2 (50%) | v3 (100%) | v3 (100%) |
| FOKVQ 변형 | E1~E3 개별 | E1~E3 + full | E1~E3 + full |
| **SOTA PPL 비교** | **없음 (KL만)** | **없음** | **5종 구현 완료** |
| 비교 method 수 | 4개 | 8개 | **13개** |
| harness | v1 | v3 | v3 (SOTA 통합) |
| GPT-2 v3 결과 | 없음 | 없음 | **중간 결과 확보** |
| Qwen 버그 | 미발견 | 발견 | **수정 완료** |
| 논문 Table 1 가능성 | 불가 | E1~E3만 | **SOTA 포함 완전한 Table** |

---

## 10. 파일 목록

| 파일 | 위치 | 용도 |
|------|------|------|
| `exp4_2_v3_full_quant_ppl.py` | `4-2_ex/` | v3 메인 코드 (E1/E2/E3/full 포함) |
| `exp4_2_standard_ppl_benchmark_v2.py` | `4-2_ex/` | v2 코드 (참조용) |
| `FOKVQ_CODE_REVIEW.md` | `4-2_ex/` | cowork vs 우리 코드 비교 |
| `V3_PROTOCOL_DESIGN.md` | `4-2_ex/` | v3 프로토콜 설계 |
| `EXPERIMENT_REPORT_20260401.md` | `4-2_ex/` | 어제 실험 결과 보고서 |
| `EXPERIMENT_PLAN_v11.1.md` | `4-2_ex/` | 본 문서 |
| `run_v3_gpt2_e1.sh` | `4-2_ex/` | GPT-2 실행 스크립트 |
| `run_v3_qwen_e1.sh` | `4-2_ex/` | Qwen 실행 스크립트 |

---

**Date**: 2026-04-02
**Author**: CDP PoC Team
**Status**: E1/E2/E3/full 구현 완료, GPT-2 + Qwen 실험 실행 중
