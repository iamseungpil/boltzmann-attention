# 실험 계획 v25: Query-Conditional PCA Subspace Selection (2026-04-09)

## 방향 검증

### 의도에 부합하는가?
- 목표: NIAH(긴 문맥 검색)에서 쿼리에 맞게 KV 캐시를 압축
- 핵심 아이디어: 양자화된 PCA 차원 중 현재 쿼리와 관련된 상위 m개만 사용
- 기존 발견과의 연결: PCA가 error decorrelation(99.9%)을 달성하므로 차원별 독립 선택이 가능

### 선행 연구에 없는 방향인가?
| 선행 연구 | 방법 | 차이점 |
|----------|------|--------|
| H2O (ICML'24) | 토큰-레벨 eviction (Heavy Hitter Oracle) | 토큰 단위, 우리는 차원 단위 |
| SnapKV (ICML'24) | 토큰-레벨 selection (attention-based) | 토큰 단위, 우리는 PCA 차원 단위 |
| MixKVQ (arXiv'25) | 채널-레벨 static bit allocation | 정적 할당, 우리는 쿼리-동적 선택 |
| KVTC (ICLR'26) | PCA + DP bit allocation (shared PCA) | 정적 할당, shared basis |
| TurboQuant (ICLR'26) | random rotation + Lloyd | 회전만, 차원 선택 없음 |
| A2ATS (ACL'25) | query-aware VQ | VQ 기반, 우리는 scalar quant + dim selection |

**결론: per-query dynamic PCA dimension selection은 아직 없음**

## 핵심 경고 (Codex 리뷰)

1. **RoPE 비정합성**: PCA는 pre-RoPE K에서 학습, attention은 post-RoPE에서 발생. PCA 좌표에서의 차원 선택이 post-RoPE 공간에서 의미가 있는지 불확실.
2. **Lloyd 역설 반복 위험**: attention MSE↓가 PPL↓을 보장하지 않음. 체계적 bias 발생 가능.
3. **저장량 불변**: 모든 128차원을 2-bit로 저장하므로 압축률은 변하지 않음. 이것은 "더 나은 양자화기"가 아니라 "양자화된 캐시 위의 query-adaptive sparse attention".
4. **스코링 함수**: `q_j² × λ_j`가 아닌 `q_j² × max(λ_j - σ²_ε,j, 0)` 사용 (SNR 보정)

## 실험 A: Attention Score MSE Microbenchmark

### 의도
양자화된 PCA 공간에서 쿼리 조건부 차원 선택이 attention score 정확도를 개선하는지 측정

### 가설 (반증 가능)
m ∈ {32, 48, 64}에서 PCA-top-m (2-bit)이 PCA-all-128 (2-bit)보다 낮은 attention MSE를 달성.
예측: 10-30% MSE 감소 (20-40%에서 하향 조정, Codex 지적)

### 검증 방법
- 모델: Mistral-7B (1차), Qwen-7B, Llama-8B
- 데이터: WikiText-2 test 10K 토큰
- 측정: attention weight MSE vs FP16, per-layer 분해
- 추가 측정: entropy shift, signed logit bias

### 통제 실험 (Codex 요구)
- **fp16_topk_query**: FP16 키 + query-conditional 선택 → 양자화 상호작용 분리
- **pca_topk_fixed**: λ_j만으로 정적 선택 → 쿼리 의존성 분리
- **pca_topk_random**: 랜덤 m개 → 선택 자체의 효과 분리
- **pca_topk_oracle**: FP16 키 기반 oracle → 이론적 상한

### 해석 매트릭스
| query-cond < pca_all? | 최적 m | fp16_topk도 개선? | 해석 |
|:---:|:---:|:---:|---|
| Yes, >10% | 32-64 | No | 양자화 노이즈 제거 효과 | → Exp B 진행 |
| Yes, >10% | 32-64 | Yes | 차원 선택 자체가 유익 (양자화 무관) | → 방향 재고 |
| Yes, <10% | 64-96 | No | 약한 효과 | → Exp B 조건부 진행 |
| No | 128 | - | 모든 차원이 유용 | → 방향 폐기 |

## 실험 B: PPL with Query-Conditional Selection

### 의도
attention MSE 개선이 PPL 개선으로 전이되는지 검증

### 가설
PCA-top-m (2-bit)이 PCA-all-128 (2-bit, PPL=6.713) 대비 PPL 0.1+ 개선

### 검증 방법
- WikiText-2 49K test, 2048 토큰 청크
- 비교: FP16, pca_all, pca_topk_query (best m), pca_topk_fixed (best m)
- 2-bit (1차), 3-bit (2차)

### 성공 기준
- 2-bit: PPL < 6.613 (0.1 개선)
- 강한 결과: PCA-top-m < Random (6.772) → PCA ≈ Random 깨기

## 실험 C: NIAH Retrieval (Exp B 성공 시)

### 의도
긴 문맥 검색에서 query-conditional 선택의 효과 검증

### 가설
PCA-top-m (2-bit)이 PCA-all-128 (2-bit) 대비 NIAH 정확도 10%p+ 개선 (16K 문맥)

### 전제 조건
- FP16 NIAH 정확도가 80%+ 유지되는 문맥 길이에서만 의미 있음
- Exp B에서 PPL 개선이 확인된 경우에만 진행

## GPU 배분 (tops-caiman A100×4)

| GPU | 실험 | 시간 |
|:---:|------|:----:|
| 0 | Exp A: Mistral-7B microbench | ~2h |
| 1 | Exp A: Qwen-7B microbench | ~2h |
| 2 | Exp A: Llama-8B microbench | ~2h |
| 3 | (대기: Exp B 후 NIAH) | - |

## 스크립트
- `scripts/exp_query_conditional_pca.py` — 모든 실험 통합
