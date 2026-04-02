# FOKVQ 실험 결과 보고서 (2026-04-01)

**Date**: 2026-04-01 ~ 04-02
**GPU**: NVIDIA RTX A6000 48GB x 2
**총 실행 시간**: ~4.3시간 (성공한 실험 기준)

---

## 실험 목록 및 상태

| # | 실험 | GPU | 모델 | 상태 | 소요 시간 |
|---|------|-----|------|------|----------|
| 1 | Exp 10-1: WikiText-2 PPL | GPU 1 | GPT-2 Med + Qwen2.5-7B | **완료** | 124분 |
| 2 | Exp 10-3: WikiText-2 PPL | GPU 1 | Mistral-7B | **완료** | 89분 |
| 3 | Exp 6-5: 2D Hybrid Quant | GPU 0/1 | GPT-2 Med + Qwen2.5-7B | **완료** | 0.5분 |
| 4 | Exp 6-2: QW-PCA | GPU 0 | Qwen2.5-7B | **실패** | 10분 |
| 5 | Phase 4-1: PPL (선행) | GPU 1 | GPT-2 Med + Qwen2.5-7B | **완료** | 68분 |

---

## 1. 핵심 결과: WikiText-2 PPL 종합

### 1.1 전체 모델 비교 (3bit 기준)

| 모델 | 아키텍처 | FP16 | Uniform 3bit | FOKVQ 3bit | FOKVQ vs FP16 | FOKVQ vs Uniform |
|------|---------|------|-------------|------------|---------------|-----------------|
| GPT-2 Medium | MHA, d=64 | 18.23 | 18.33 (+0.6%) | **18.30** (+0.4%) | +0.4% | -0.1% |
| Qwen2.5-7B | GQA+RoPE, d=128 | 7.36 | 6,380 (붕괴) | **9.12** (+24%) | +24% | -99.9% |
| Mistral-7B | GQA+RoPE, d=128 | 5.05 | 5.08 (+0.6%) | **5.08** (+0.5%) | +0.5% | -0.1% |

### 1.2 전체 모델 비교 (4bit 기준)

| 모델 | FP16 | Uniform 4bit | FOKVQ 4bit | FOKVQ vs FP16 |
|------|------|-------------|------------|---------------|
| GPT-2 Medium | 18.23 | 18.27 (+0.2%) | **18.25** (+0.1%) | +0.1% |
| Qwen2.5-7B | 7.36 | 5,721 (붕괴) | **7.47** (+1.5%) | **+1.5%** |
| Mistral-7B | 5.05 | 5.06 (+0.1%) | **5.06** (+0.1%) | +0.1% |

### 1.3 전체 모델 비교 (2bit 기준)

| 모델 | FP16 | Uniform 2bit | FOKVQ 2bit | FOKVQ vs FP16 |
|------|------|-------------|------------|---------------|
| GPT-2 Medium | 18.23 | 19.98 (+9.6%) | **19.35** (+6.1%) | +6.1% |
| Qwen2.5-7B | 7.36 | 6,723 (붕괴) | **16.84** (+129%) | +129% |
| Mistral-7B | 5.05 | 5.83 (+15.4%) | **5.27** (+4.4%) | +4.4% |

---

## 2. Exp 10-1: GPT-2 Medium + Qwen2.5-7B 상세

### GPT-2 Medium (MHA, 16 heads, d_head=64, float16, context=1024, stride=512)

| Method | 2bit | 3bit | 4bit | FP16 |
|--------|------|------|------|------|
| Uniform | 19.98 | 18.33 | 18.27 | 18.23 |
| **FOKVQ** | **19.35** | **18.30** | **18.25** | — |

**GATE: PASS** — FOKVQ 3bit(18.30) < Uniform 3bit(18.33), FP16 대비 +0.4%

### Qwen2.5-7B (GQA, 4 KV heads, d_head=128, bfloat16, context=2048, stride=1024)

| Method | 2bit | 3bit | 4bit | FP16 |
|--------|------|------|------|------|
| Uniform | 6,723 | 6,380 | 5,721 | 7.36 |
| **FOKVQ** | **16.84** | **9.12** | **7.47** | — |

**GATE: PARTIAL** — FOKVQ 4bit PASS (+1.5%), FOKVQ 3bit은 +24%로 10% 기준 초과

---

## 3. Exp 10-3: Mistral-7B 상세

**설정**: Mistral-7B-Instruct-v0.3 (GQA, 8 KV heads, d_head=128, bfloat16, context=2048, stride=1024)
**데이터**: WikiText-2 test 328,835 tokens

| Method | Bits | PPL | vs FP16 | vs Uniform |
|--------|------|-----|---------|------------|
| FP16 | 16 | 5.05 | — | — |
| Uniform | 2 | 5.83 | +15.4% | — |
| **FOKVQ** | **2** | **5.27** | **+4.4%** | **-9.5%** |
| Uniform | 3 | 5.08 | +0.6% | — |
| **FOKVQ** | **3** | **5.08** | **+0.5%** | **-0.1%** |
| Uniform | 4 | 5.06 | +0.1% | — |
| **FOKVQ** | **4** | **5.06** | **+0.1%** | **+0.0%** |

**GATE: PASS** — FOKVQ 3bit = Uniform 3bit (5.08), FP16 대비 +0.5%

**주목할 점**: Mistral-7B는 Qwen과 동일한 GQA+RoPE 아키텍처이나, Uniform이 붕괴하지 않는다. Qwen의 Uniform 붕괴는 GQA+RoPE의 일반적 특성이 아닌 **Qwen 고유 현상**일 가능성이 있다.

---

## 4. Exp 6-5: 2D Hybrid Quantization

3가지 전략 비교 (target B_avg=3 bits):
1. **Dim-only**: FOKVQ B=3 (PCA 기반 차원 비균일 할당)
2. **Token-only**: 최근 32 tokens 4bit, 나머지 ~2.9bit
3. **2D Combined**: 최근 32 tokens FOKVQ(4bit), 나머지 FOKVQ(2bit)

### 결과 (KL Divergence, 낮을수록 좋음)

| 모델 | Dim-only | Token-only | 2D Combined | 판정 |
|------|---------|-----------|-------------|------|
| GPT-2 Med | **0.0160** | 0.0929 | 0.1122 | FAIL |
| Qwen2.5-7B | **0.0529** | 0.8813 | 0.1683 | PARTIAL |

**결론**: 2D 결합은 Dim-only(FOKVQ) 단독보다 나쁘다. PCA 차원 할당만으로 충분하며, 토큰 축 비균일 할당은 불필요하다.

---

## 5. Exp 6-2: QW-PCA (실패)

- Qwen2.5-7B에서 Q-weighted PCA 실험 시도
- 데이터 수집(32/32 배치) 완료 후, Q 추출 단계에서 에러 발생
- **원인**: `Qwen2Attention.forward()` 인터페이스 변경 (transformers 5.x에서 `position_embeddings`, `attention_mask` 필수 인자 추가)
- **수정 필요**: `self_attn(hidden_norm)` → `self_attn(hidden_norm, position_embeddings=..., attention_mask=...)` 로 변경

---

## 6. 핵심 발견 요약

### 6.1 MHA vs GQA+RoPE 양자화 특성

| 특성 | MHA (GPT-2) | GQA+RoPE (Mistral) | GQA+RoPE (Qwen) |
|------|------------|-------------------|-----------------|
| Uniform 3bit | +0.6% (양호) | +0.6% (양호) | 붕괴 (x867) |
| FOKVQ 3bit | +0.4% | +0.5% | +24% |
| FOKVQ 4bit | +0.1% | +0.1% | **+1.5%** |
| FOKVQ 필수성 | 선택적 | 선택적 | **필수** |

### 6.2 Qwen vs Mistral: 같은 GQA+RoPE인데 왜 다른가?

Mistral-7B에서 Uniform이 정상 작동하는 반면 Qwen에서 붕괴하는 이유 후보:
- **KV heads 수**: Qwen 4개 vs Mistral 8개 → Qwen의 KV head당 부하가 더 큼
- **RoPE theta 차이**: base frequency가 다를 수 있음
- **KV 값 분포**: Qwen의 K 벡터 값 범위가 더 넓어 양자화 오류 증폭
- **→ 추가 실험 필요**: Llama-3-8B(8 KV heads)로 교차 검증

### 6.3 FOKVQ 운영 bit 권장

| 시나리오 | 권장 bit | 근거 |
|---------|---------|------|
| 논문 headline 결과 | **4bit** | 모든 모델에서 FP16 대비 +1.5% 이내 |
| MHA 모델 운영 | 3bit | +0.4%, 5.3x 압축 |
| GQA+RoPE 운영 (안전) | 4bit | +1.5%, 4x 압축 |
| GQA+RoPE 운영 (공격적) | 3bit | +24% (Qwen), 용도 한정 |
| 극한 압축 | 2bit | MHA +6%, GQA 모델별 차이 큼 |

### 6.4 2D Hybrid 결론

PCA 차원 할당(Dim-only)이 토큰 축 할당보다 일관되게 우수. 2D 결합은 오히려 성능 저하. **FOKVQ는 차원 축 비균일 할당으로 충분**하다.

---

## 7. 논문 Table 1 구성 제안

```
| Model           | Method  | FP16  | 4bit          | 3bit          | 2bit          |
|-----------------|---------|-------|---------------|---------------|---------------|
| GPT-2 Med       | Uniform | 18.23 | 18.27 (+0.2%) | 18.33 (+0.6%) | 19.98 (+9.6%) |
| (MHA, d=64)     | FOKVQ   |   —   | 18.25 (+0.1%) | 18.30 (+0.4%) | 19.35 (+6.1%) |
| Qwen2.5-7B      | Uniform |  7.36 | 5,721 (x777)  | 6,380 (x867)  | 6,723 (x913)  |
| (GQA+RoPE,d=128)| FOKVQ   |   —   |  7.47 (+1.5%) |  9.12 (+24%)  | 16.84 (x2.3)  |
| Mistral-7B      | Uniform |  5.05 |  5.06 (+0.1%) |  5.08 (+0.6%) |  5.83 (+15.4%)|
| (GQA+RoPE,d=128)| FOKVQ   |   —   |  5.06 (+0.1%) |  5.08 (+0.5%) |  5.27 (+4.4%) |
```

---

## 8. 다음 단계 우선순위

| 우선순위 | 실험 | 목적 |
|---------|------|------|
| **1** | Exp 4-2 v2 (GPT-2) | 수정된 코드로 PPL 재검증 (현재 실행 중) |
| **2** | Exp 10-3: Llama-3-8B | GQA+RoPE 3번째 모델로 범용성 확인 |
| **3** | Exp 6-2 수정 | Qwen2 Attention 인터페이스 수정 후 QW-PCA 재실행 |
| **4** | Qwen Uniform 붕괴 원인 분석 | Mistral과의 차이 규명 (KV head 수, RoPE theta) |
| **5** | Exp 10-5: 긴 컨텍스트 | Qwen 3bit +24%가 긴 시퀀스 문제인지 검증 |

---

## 9. GATE 판정 종합

| 모델 | 3bit GATE | 4bit GATE | 비고 |
|------|-----------|-----------|------|
| GPT-2 Medium | **PASS** (+0.4%) | **PASS** (+0.1%) | |
| Qwen2.5-7B | PARTIAL (+24%) | **PASS** (+1.5%) | Uniform 대비 99.9% 개선 |
| Mistral-7B | **PASS** (+0.5%) | **PASS** (+0.1%) | |

**종합 판정**: 4bit 기준 **전 모델 PASS**. 논문 주력 주장을 4bit 중심으로 구성 권장.

---

**작성**: CDP PoC Team
**결과 원본 위치**:
- `docs/architecture/patent/exp10_1_wikitext2_results.md`
- `docs/architecture/patent/exp10_3_llama3_results.md`
- `docs/architecture/patent/exp6_5_2d_hybrid_results.md`
- `docs/architecture/patent/phase4_1_ppl_results.md`
- `/tmp/exp10_1.log`, `/tmp/exp10_3.log`
