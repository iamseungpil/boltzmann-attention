# Exp 4-2 v3: Full KV Cache Quantization PPL Benchmark -- Protocol Design

**Date**: 2026-04-02
**Author**: FOKVQ Team
**Status**: Implementation complete, pending execution

---

## 1. Problem Statement

### v2 Protocol (current)
```
context=1024, stride=512
window = [prefix 512 | eval 512]
  -> prefix K: quantized
  -> eval K: FP16 (NOT quantized)
  -> attention key sequence: [quant_K 512 | fp16_K 512]
  -> quantization ratio: 50%
```

### Result
- GPT-2 3bit: FP16 vs FOKVQ = 18.23 vs 18.30 (+0.4%)
- 양자화 효과가 50% 희석되어 실제 영향을 측정할 수 없음

### Real Deployment
- 전체 KV cache가 양자화됨 (KIVI, KVQuant, vLLM 등)
- 양자화 비율: **100%**
- 논문의 PPL 벤치마크가 이를 반영해야 함

---

## 2. SOTA Papers Protocol Analysis

### KVQuant (NeurIPS 2024)
- **GitHub**: https://github.com/SqueezeAILab/KVQuant
- **코드 분석**: `quant/llama_simquant.py`, `kvquant/simquant_module_quantizer.py`
- **프로토콜**:
  1. k_proj, v_proj Linear layer를 `QuantLinearSim`으로 교체
  2. `QuantLinearSim.forward()`에서 `y = x @ weight` 후 출력을 즉시 양자화
  3. **모든 forward에서 100% 양자화 자동 적용**
  4. PPL 평가: non-overlapping 청크 (sliding window 아님)
  5. `use_cache = False` -- layer-by-layer activation 전파
- **양자화 위치**: pre-RoPE (k_proj 출력 직후, RoPE 적용 전)

### KIVI (ICML 2024)
- **GitHub**: https://github.com/jy-yuan/KIVI
- **코드 분석**: `models/llama_kivi.py`
- **프로토콜**:
  1. `LlamaAttention`을 커스텀 attention으로 완전 교체
  2. Attention 내부에서 `triton_quantize_and_pack_along_last_dim()`으로 K,V 양자화
  3. `residual_length` 만큼의 최근 토큰만 FP16, 나머지 전부 양자화
  4. PPL 직접 측정하지 않음 -- LongBench, CoQA 등 generation task로 평가
- **양자화 위치**: post-RoPE (attention 내부에서 K 생성 후)
- **특이점**: sliding window 개념으로 최근 32토큰은 FP16 유지

### SKVQ (2024)
- **프로토콜**: sequence length 4096에서 PPL 측정
- **양자화**: channel reordering + clipped dynamic quantization
- **최근 window**: FP16 유지

### 공통 패턴
| 논문 | 양자화 통합 방식 | PPL 평가 방식 | 양자화 비율 |
|------|----------------|-------------|-----------|
| KVQuant | Module 교체 (k_proj/v_proj) | Non-overlapping chunks | 100% |
| KIVI | Attention 완전 교체 | Generation tasks | ~95% (residual FP16) |
| SKVQ | Channel reorder + quant | Chunk evaluation | ~95% (window FP16) |
| **FOKVQ v2** | **외부 wrapper (prefix만)** | **Sliding window** | **50%** |

---

## 3. v3 Protocol Design

### 핵심 변경
1. **양자화 비율**: 50% -> 100%
2. **평가 방식**: Sliding window -> Non-overlapping chunks
3. **양자화 삽입 위치**: 외부 KV cache 교체 -> 모델 내부 hook/patch

### Protocol A: Pre-RoPE Hook (KVQuant 방식)
```python
# k_proj Linear에 register_forward_hook
# k_proj 출력 (pre-RoPE K)을 즉시 양자화
hook = k_proj.register_forward_hook(quantize_output)
```
- 장점: 간단, 모델 아키텍처 불문
- 단점: RoPE 적용 전 양자화 = 실제 배포와 다를 수 있음

### Protocol B: Post-RoPE Attention Patch (KIVI 방식)
```python
# Attention module의 forward를 monkey-patch
# RoPE 적용 후 K를 양자화
attn.forward = patched_forward_with_k_quantization()
```
- 장점: 실제 배포와 동일 (post-RoPE K 양자화)
- 단점: 모델별 attention 구현에 의존

### 선택: **Protocol B (post_rope)를 기본으로 사용**

이유:
1. FOKVQ의 PCA는 K의 이방성을 활용 -- RoPE가 이방성의 주요 원인
2. Post-RoPE K에서 PCA가 RoPE 구조를 캡처하여 효과적 비트 할당
3. 실제 배포에서 KV cache에 저장되는 것은 post-RoPE K
4. v2 결과와의 비교 일관성 (v2도 post-RoPE K를 양자화)

Protocol A (pre_rope)는 ablation study로 사용:
- Pre-RoPE vs Post-RoPE 양자화의 차이를 보여주는 것도 논문에 유용

---

## 4. Evaluation Protocol Details

### Non-overlapping Chunk Evaluation
```
전체 시퀀스: [chunk_0 | chunk_1 | chunk_2 | ... | chunk_N]
각 chunk: 독립 forward (use_cache=False)
NLL: shift_logits[:, :-1] vs labels[:, 1:]
PPL = exp(sum(NLL) / total_tokens)
```

### v2 (sliding window) vs v3 (non-overlapping) 차이
- v2: 각 window에서 eval 토큰은 prefix context의 도움을 받음
- v3: 각 chunk은 독립 -- chunk 시작 부분은 context 없이 예측
- **결과**: v3의 FP16 baseline PPL이 v2보다 높을 수 있음 (context 없는 토큰이 있으므로)
- **하지만**: v2와 v3의 FP16 baseline 차이는 양자화 방법 간 비교에 영향 없음
  (모든 방법이 동일한 baseline을 사용하므로)

### 이 차이가 문제가 되는가?
아닐 수 있음. GPT-2는 `context_len=1024`가 max position이므로 sliding window나
non-overlapping이나 1024 토큰 안에서만 attention. 차이는 chunk 시작 부분의
context 부재뿐.

실제로 HuggingFace의 표준 PPL 평가 (`transformers/perplexity.py`)도
non-overlapping을 기본으로 사용하며, 논문 비교 시 통일된 프로토콜이면 공정함.

---

## 5. Expected Results

### 가설
1. **GPT-2 3bit**: v2(+0.4%) -> v3에서 더 높은 열화 예상 (+1~3%)
   - v2에서 50%만 양자화되어 희석되었으므로, 100%면 2배 정도 차이 가능
2. **Qwen 3bit**: v2(+24%) -> v3에서 유사하거나 약간 증가
   - v2에서도 prefix의 RoPE 구조가 강하게 영향했으므로
3. **Qwen 4bit**: v2(+1.5%) -> v3에서 +2~5% 예상
4. **FOKVQ vs Uniform 격차**: v3에서 더 명확해질 것
   - 100% 양자화에서 FOKVQ의 PCA 기반 비트 할당이 더 큰 이점

### 논문에 유리한 시나리오
v3에서 Uniform의 열화가 크게 증가하고 FOKVQ는 상대적으로 적게 증가하면:
- "Full quantization 환경에서 FOKVQ의 이점이 극대화된다"
- 특히 Qwen(GQA+RoPE)에서 Uniform 완전 붕괴 vs FOKVQ 안정

---

## 6. Files

| File | Purpose |
|------|---------|
| `exp4_2_v3_full_quant_ppl.py` | Main experiment code |
| `run_v3_gpt2.sh` | GPT-2 Medium execution script |
| `run_v3_qwen.sh` | Qwen2.5-7B execution script |
| `V3_PROTOCOL_DESIGN.md` | This document |

### Usage
```bash
# Self-test
python3 exp4_2_v3_full_quant_ppl.py --self-test

# GPT-2 Medium
bash run_v3_gpt2.sh

# Qwen2.5-7B
bash run_v3_qwen.sh

# Pre-RoPE ablation (Protocol A)
python3 exp4_2_v3_full_quant_ppl.py \
    --model-name gpt2-medium \
    --model-key gpt2-medium \
    --context-len 1024 \
    --protocol pre_rope \
    --methods fp16 uniform fokvq \
    --bits 2 3 4 \
    --output-dir results/v3_pre_rope
```

---

## 7. v2 vs v3 Comparison Plan

실험 완료 후 작성할 비교 테이블:

```
| Model | Method | Bits | v2 PPL (50% quant) | v3 PPL (100% quant) | Delta |
|-------|--------|------|--------------------|---------------------|-------|
| GPT-2 | FP16   | 16   | 18.23              | ???                 |       |
| GPT-2 | Uniform| 3    | 18.33 (+0.6%)      | ??? (+?%)           |       |
| GPT-2 | FOKVQ  | 3    | 18.30 (+0.4%)      | ??? (+?%)           |       |
| Qwen  | FP16   | 16   | 7.36               | ???                 |       |
| Qwen  | Uniform| 3    | 6,380 (exploded)   | ??? (same?)         |       |
| Qwen  | FOKVQ  | 3    | 9.12 (+24%)        | ??? (+?%)           |       |
| Qwen  | FOKVQ  | 4    | 7.47 (+1.5%)       | ??? (+?%)           |       |
```
