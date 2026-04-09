# 실험 계획서 — Phase B 메인 방법 (X+Y+Z 3-axis Contribution)

**버전**: v1.0
**작성일**: 2026-04-10
**작성자**: mais (develop 브랜치)
**목표 venue**: NeurIPS 2026 / ICLR 2027 main track
**담당**: mais (develop, 공유 GPU) + iamseungpil (origin/main, **A100 80GB × 4 전용**)
**상태**: Kill-switch PASS 확정 (2026-04-09 full 995 MetaTool +11.15pp) → 실행 가능 상태

---

## 0. 한 줄 요약

도구 catalog만으로 LLM의 호출 가능 도구 vocabulary를 학습 없이 확장하는 training-free 방법을 제안한다. 세 가지 기여:

1. **[Z] 문제**: Training-free tool vocabulary expansion — 재학습 없이, demonstration 없이, calibration data 없이, 도구 catalog만으로 새 도구 추가
2. **[Y] 메커니즘**: Asymmetric Q/K ontology projection — Q는 intent subspace (`B_intent`)에, K는 content subspace (`B_content`)에 투영. 새 도구 추가 = `B_content` 확장만, `B_intent`는 catalog-invariant
3. **[X] 정리**: Phase-closure 보장 — `‖Q·B_intent‖²/‖Q‖² < ε`이면 attention score 변화량이 `f(ε)` 이하. Vocabulary expansion이 non-tool query (MMLU 등)를 **구조적으로** 저하시킬 수 없음

가장 가까운 선행 **AdaSEKA (ICLR 2026, arXiv:2603.01281)** 에 대해 세 차원 모두에서 차별화. Counter-theorem으로 AdaSEKA의 max-normalized routing이 phase-closure를 원리적으로 제공 불가능함을 증명.

**Kill-switch 확정 결과 (2026-04-09 저녁 run, 995 샘플)**:
- `no_steer`: 75.58%
- `ocq_bias α=0.3`: 86.73% (**+11.15pp**, kill-switch 3pp 기준 3배 이상 PASS)
- α curve에서 0.25에 −10pp 딥 발견 — 논문에 honest 보고 필요
- `ocq_quant` 단독 −20.7pp, `ocq_quant + bias α=0.3` −18.6pp → **dual-claim 불가**, Claim A (tool selection) 단독 논문으로 재작성

**일정**: ~10일에 paper draft 완성. 약 45 A100·시간이 coworker 쪽 4× A100 80GB로 분리 실행, wall-clock 3-3.5일.

---

## 1. 문제 정의 (Z) — 상세

### 1.1 배경 및 동기

Tool-calling agent로 배포된 LLM은 고정된 도구 집합 `T = {t_1, ..., t_N}` 을 시스템 프롬프트에 등록한다. 새 도구 `t_{N+1}` 추가 시 현재 옵션들:

| 방법 | 정확도 | 비용 | 유연성 | 한계 |
|---|---|---|---|---|
| **Retrieval-augmented generation (BGE/DPR)** | 중 | 쿼리당 retrieval latency | 즉시 추가 가능 | Retriever 품질이 ceiling, out-of-distribution 도구에서 취약 |
| **Prompt engineering** | 중-고 | Context window 선형 증가 | 즉시 추가 | Scale 증가 시 context window 폭발, 정확도 저하 |
| **LoRA fine-tuning** | 고 | 학습 시간 + labeled data | Catalog 변경 시 재학습 | Catalog 업데이트 불가, 학습 데이터 필수 |
| **Full fine-tuning** | 최고 | 매우 높음 | 재학습 필수 | 잦은 catalog 변경에 부적합 |

### 1.2 연구 문제

**"도구 catalog만으로, 학습 데이터 없이, 모델 가중치 수정 없이, 추론 시 retrieval overhead 없이, LLM의 유효 tool vocabulary를 확장할 수 있는가?"**

이는 "attention steering for tool selection" (Phase B plan v1 framing)과 본질적으로 다른 문제다:
- Steering은 **고정 catalog** 에서 성능 최적화
- Expansion은 **catalog 자체**를 확장

제안 방법은 도구 catalog과 frozen LLM 사이에 위치하는 **training-free, incremental, zero-data** 모듈.

### 1.3 왜 이 문제가 중요한가

1. **현실적 배포 시나리오**: Enterprise 도구 카탈로그는 주간/월간 단위로 변화. 매번 LoRA 재학습은 비현실적.
2. **Zero-shot catalog adaptation**: 모델 소유자와 도구 공급자가 분리된 현실 (e.g., OpenAI API + 고객사 사내 도구 카탈로그)에서 LoRA 불가능.
3. **Deployment friction**: Context 창 크기로 카탈로그 확장성 제한 (Claude 200K, Qwen 32K 등) → 199개 도구 catalog이 이미 시스템 프롬프트 상당 부분 차지.
4. **Novelty**: AdaSEKA / SEKA / ASA / PASTA / FGA / InstABoost / Spotlight 어느 선행도 이 문제를 다루지 않음 (문헌 검토 결과 2026-04-10 기준).

### 1.4 평가 프로토콜 (상세)

#### 1.4.1 Expansion scenario

MetaTool catalog에서 `N_0 = 20` 도구로 시작 (random seed=42로 199개 중 sampling). 점진적 추가 단계: `N ∈ {20, 50, 100, 150, 199}`.

각 단계 `N_k` 에서:
1. `B_content^{(N_k)}` 재구축 (incremental update)
2. `M^{(N_k)}` 재계산 (ridge regression)
3. `B_intent` 는 고정 (catalog-invariant 주장 empirical 검증)

#### 1.4.2 측정 metric

각 단계에서 다음을 측정:

| Metric | 측정 방법 | 목적 |
|---|---|---|
| **Tool selection top-1** | MetaTool Subtask1 held-out (`N_k` 도구만 candidate) | 주 tool-selection 성능 |
| **MMLU 5-shot accuracy** | cais/mmlu 1000 샘플 subset | 비-tool query degradation |
| **Phase-closure bound 실측** | MMLU 샘플별 `ε(Q)` vs `|ΔS|` 분포 | 정리(X)의 empirical 검증 |
| **Catalog update latency** | 새 도구 `k`개 incorporate 시간 (초) | Deployment 현실성 |
| **σ_max(M^{(N_k)})** 추이 | singular value decomposition | Corollary bound의 `N` 의존성 검증 |

#### 1.4.3 Success criteria

**핵심 criterion (Z의 main claim)**:
- **MMLU degradation slope가 catalog 크기 `N` 에 대해 통계적으로 0과 구별 불가** (one-sided t-test, α=0.05)
- 즉, `MMLU(N=199) − MMLU(N=20) ≈ 0 ± noise`
- 이것이 정리(X)가 뒷받침하는 main claim

**Tool-selection criterion**:
- Expanded catalog에서 top-1 accuracy가 full-catalog LoRA finetuned baseline 대비 2pp 이내
- 즉, "zero training data + 즉시 catalog update + ~2pp accuracy 희생" trade-off

**부가 criterion**:
- Catalog update latency < 30초 (199 도구 전체)
- `σ_max(M^{(N_k)})` 가 `N` 에 대해 로그 증가 또는 상수 수렴

#### 1.4.4 비교 baseline

| Baseline | 학습 필요 | Catalog update 시 비용 | 기대 성능 |
|---|---|---|---|
| **no_steer** | — | — | Lower bound |
| **Retrieval-BGE top-5** | — | 재인덱싱 (~1분) | 중-고 |
| **Prompt engineering (all tools)** | — | 프롬프트 수정 | 중 |
| **LoRA r=16 finetune** | 필요 (8h/7B) | 재학습 필요 | 상한 reference |
| **Proposed (ours)** | — | 20초 (incremental) | LoRA 대비 −1~3pp 허용 |

### 1.5 Benchmark 후보

Primary:
- **MetaTool Subtask1** (ICLR 2024, arXiv:2310.03128): 995 queries, 199 tools, similar-choices disambiguation, MIT license

Secondary (expansion scenario에도 확장):
- **BFCL V4 Agentic** (gorilla-llm, 2024): 60+ tools, function calling, irrelevance/relevance subsets
- **MTU-Bench** (arXiv:2602.04935 관련): 3000 multi-turn tool calls, ASA 직접 비교

Non-tool (phase-closure 검증):
- **MMLU 5-shot** (cais/mmlu)
- **TriviaQA dev** (7993 questions)
- **HellaSwag** (10,042 questions)
- **MT-Bench subset** (20 prompts, GPT-4o-mini judge)

추가 옵션 (여유 시):
- **Seal-Tools**, **UltraTool** — overlap-heavy
- **AppSelectBench** (2511.19957) — enterprise tool selection
- **ToolE** (2310.03128) — explicit overlap
- **NESTFUL**, **ComplexFuncBench** (arXiv:2501.10132), **StableToolBench**, **FunctionChat-Bench** (arXiv:2411.14054)

---

## 2. Method (Y) — Asymmetric Q/K Ontology Projection

### 2.1 Q와 K의 역할 차이 (why asymmetric)

**핵심 통찰**: Attention 메커니즘에서 Q와 K의 역할은 본질적으로 다르다.

- **Q (query)**: "이 토큰이 무엇을 알고자 하는가 / 무엇을 원하는가" — **의도(intent)** 표현
- **K (key)**: "이 토큰이 무엇을 담고 있는가" — **내용(content)** 표현

Attention은 `softmax(Q·K^T / √d)` 로 routing. 선택은 Q가 한다 (K에서 자기와 맞는 것을 찾아감).

기존 선행연구 (SEKA, AdaSEKA, Focus Directions, PASTA, InstABoost, Spotlight)는 모두 **symmetric projector** 를 Q와 K에 동일하게 적용하거나, K에만 (또는 Q에만) 적용한다. 이는 Q와 K의 역할 차이를 **구조적으로 무시**한다.

본 논문은 Q와 K에 **서로 다른 subspace** 를 적용한다:
- Q → `B_intent` (사용자 의도 관련 방향)
- K → `B_content` (도구/도메인/내용 관련 방향)

이 asymmetric design의 이점:
1. **Linguistic alignment**: Intent vs content의 언어학적 직관과 일치
2. **Vocabulary expansion friendly**: `B_intent` (의도 유형 시스템)는 도구 추가와 무관하게 고정. `B_content` (도구별 content)만 확장 → 새 도구 추가가 intent 표현을 해치지 않음
3. **Factorized control**: Gate 함수가 Q의 intent subspace 에너지만 보고 판단, K의 content와 독립

### 2.2 Notation

- `d = head_dim` (Qwen2.5-7B: 128)
- `H = n_kv` (Qwen2.5-7B: 4)
- `L` (Qwen2.5-7B: 28)
- `B_intent ∈ ℝ^{d × r_I}`: intent subspace basis (orthonormal columns), `r_I << d`, default `r_I = 8`
- `B_content ∈ ℝ^{d × r_C}`: content subspace basis (orthonormal columns), `r_C << d`, default `r_C = 24`
- `M ∈ ℝ^{r_I × r_C}`: intent → content linear mapping
- `Q, K ∈ ℝ^{T × d}`: per-head attention tensors (pre-RoPE)

모든 tensor는 per-(layer, head)이지만 `(ℓ, h)` 인덱스 생략.

### 2.3 Basis 구축 (training-free, catalog만 사용)

#### 2.3.1 `B_content` 구축

각 도구 `t_n` 에 대해:
1. `t_n` 을 4 facet anchor sentence 집합으로 렌더링 (function_action, io_type, domain, tool_category)
2. 각 anchor를 frozen LLM에 통과
3. `k_proj` forward hook에서 pre-RoPE K 추출: `K_anchor,n ∈ ℝ^{T_n × d}` per (layer, head)
4. Per-tool K-covariance 누적: `Σ_content = Σ_n Σ_t K_{anchor,n,t} · K_{anchor,n,t}^T`
5. `B_content = top-r_C eigenvectors of Σ_content`

기존 인프라 재활용: `scripts/ocq/build_qwen_metatool_b_ont.py`.

#### 2.3.2 `B_intent` 구축

**핵심 속성**: `B_intent` 는 facet *type system* 에만 의존, 구체적 도구에는 의존하지 않음.

1. 각 `function_action` facet value (verb: `translate`, `summarize`, `recommend`, `book`, `search`, ...) 마다 intent sentence 렌더링 ("I want to {verb} this", "Please {verb} the {object}")
2. `q_proj` forward hook에서 pre-RoPE Q 추출
3. `Σ_intent = Σ Q_anchor · Q_anchor^T`
4. `B_intent = top-r_I eigenvectors of Σ_intent`

기존 facet value를 쓰는 새 도구가 추가되어도 `B_intent` 는 변경되지 않음 (Z의 catalog-invariant 주장의 근거).

#### 2.3.3 `M` 구축 (intent → content mapping)

Per-tool (intent vector, content vector) 쌍에서 ridge regression:

1. 각 도구 `t_n`:
   - `i_n = B_intent^T · Q_anchor(t_n) ∈ ℝ^{r_I}` (intent embedding)
   - `c_n = B_content^T · K_anchor(t_n) ∈ ℝ^{r_C}` (content embedding)
2. `M = argmin_M Σ_n ‖M·i_n - c_n‖² + λ‖M‖_F²`, solved as `M = (I^T I + λ I)^{-1} I^T C` with `I = [i_1; ...; i_N]`, `C = [c_1; ...; c_N]`
3. `λ = 0.01` default

### 2.4 Attention score 수정

모든 (layer, head)에서:

```
S'(Q_i, K_j) = (Q_i · K_j^T) / √d  +  α(Q_i) · (Q_i · B_intent) · M · (B_content^T · K_j) / √d
```

여기서 `α(Q_i)` 는 per-query self-gate:

```
α(Q_i) = α_max · σ(β · (ε(Q_i) - τ))
ε(Q_i) = ‖Q_i · B_intent‖² / ‖Q_i‖²
```

Hyperparameter defaults:
- `α_max = 1.0` (maximum intervention strength)
- `β = 10` (gate steepness)
- `τ = 0.1` (gate threshold: Q가 intent subspace에 ≥10% 에너지일 때 gate 열림)
- `σ`: sigmoid

**핵심 속성**:
1. `ε(Q) ≪ τ` (non-tool query): `α(Q) ≈ 0` → score 변화 ≈ 0 (**phase-closure**)
2. `ε(Q) ≫ τ` (tool query): `α(Q) ≈ α_max` → 전체 개입 적용
3. 개입은 **asymmetric**: Q는 `B_intent`, K는 `B_content`, `M`이 bridge
4. K cache 수정 없음 — vocabulary expansion이 cache rebuild 불필요

### 2.5 Catalog expansion 절차

새 도구 `t_{new}` 추가:

1. `t_{new}` anchor K 계산 (facet별 4 sentence × forward pass)
2. `Σ_content += K_anchor_new · K_anchor_new^T` 업데이트 (incremental)
3. `Σ_content` 재-eigendecompose → 새 `B_content` (rank `r_C + δ_r`)
4. Incremental ridge regression으로 `M` 에 새 row 추가
5. `B_intent` 는 **변경 없음**

총 비용 per 도구: **O((d² + r_C²) · L · H)**
- Qwen2.5-7B (28×4×128²×199 도구): CPU에서 도구당 ~100ms, 199 전체 ~20초 (예측치)

### 2.6 구현 전략

**Primary**: HF transformers의 `attention_mask` dtype-bias convention으로 pre-softmax additive bias 주입
- `q_proj` hook에서 `Q · B_intent` 저장
- `k_proj` hook에서 `K · B_content` 저장
- Attention 계산 전 `α(Q) · (Q·B_intent) · M · (B_content^T · K)^T` 를 `attention_mask` 에 additive bias로 주입
- `LlamaAttention.forward` / `Qwen2Attention.forward` monkey-patch 회피

**Fallback**: mask bias 주입이 내부에서 막히면 `LlamaAttention.forward` monkey-patch
- transformers 4.56 lock
- `attn_implementation="eager"` 강제 (SDPA/Flash attention 비호환)

기존 인프라 재활용:
- `scripts/ocq/eval_metatool_subtask1.py` (hook 기반 K-bias 이미 구현)
- `scripts/ocq/build_qwen_metatool_b_ont.py` (B_content 빌더)

---

## 3. Phase-Closure Theorem (X) — 상세

### 3.1 정리 statement

**Theorem 1 (Phase-Closure for Asymmetric Ontology Projection)**

Orthonormal column을 가진 `B_intent ∈ ℝ^{d × r_I}`, `B_content ∈ ℝ^{d × r_C}`, mapping matrix `M ∈ ℝ^{r_I × r_C}` 가 주어짐. Gate 함수 `α: ℝ^d → [0, α_max]` 는 `f`-Lipschitz이며 `f(0) = 0`. `ε(Q) := ‖Q·B_intent‖² / ‖Q‖²` 정의.

모든 `Q, K ∈ ℝ^d` 에 대해:

```
|S'(Q, K) - (Q·K^T)/√d|  ≤  (α(Q) / √d) · ‖Q·B_intent‖ · σ_max(M) · ‖K·B_content‖
                          ≤  (α(Q) / √d) · √(ε(Q)) · ‖Q‖ · σ_max(M) · ‖K‖
                          ≤  (α_max · f(ε(Q)) / √d) · √(ε(Q)) · ‖Q‖ · ‖K‖ · σ_max(M)
```

**따름**:
- `ε(Q) → 0` 시 score perturbation은 `O(f(ε(Q)) · √ε(Q))` 속도로 소멸
- Default gate `α(Q) = α_max · σ(β(ε - τ))` 에서 `ε → 0` 근방에 `f(ε) ≈ α_max β e^{-βτ} · ε` (sigmoid Taylor expansion)
- 따라서 phase-closed regime에서 **cubic decay** `|ΔS| = O(ε^{3/2})`

### 3.2 증명 sketch

**Step 1**: 기본 bound.
```
|S' - Q·K^T/√d|  =  |α(Q) · (Q·B_intent) · M · (B_content^T · K) / √d|
                 ≤  (α(Q)/√d) · ‖Q·B_intent‖ · ‖M (B_content^T K)‖
                 ≤  (α(Q)/√d) · ‖Q·B_intent‖ · σ_max(M) · ‖B_content^T K‖
```

**Step 2**: `B_intent`, `B_content` 가 orthonormal이므로:
- `‖Q·B_intent‖² = ‖B_intent^T Q‖² ≤ ‖Q‖²` (Parseval)
- `‖B_content^T K‖² ≤ ‖K‖²`

**Step 3**: `ε(Q) = ‖Q·B_intent‖²/‖Q‖²` 정의에서:
- `‖Q·B_intent‖ = √(ε(Q)) · ‖Q‖`

**Step 4**: Gate Lipschitz 조건 `α(Q) ≤ f(ε(Q))` 대입 (`f(0)=0` → `α(0)=0`).

**Step 5**: 종합:
```
|ΔS|  ≤  (α_max f(ε(Q)) / √d) · √ε(Q) · ‖Q‖ · ‖K‖ · σ_max(M)
```

**Equality case**: `Q ∈ span(B_intent)` and `K` 가 `M^T B_intent^T Q` 방향의 `B_content` column에 정렬될 때 bound이 tight. (Cauchy-Schwarz equality.)

### 3.3 Corollary 1 (Vocabulary Expansion Safety)

`N` 개 도구를 base catalog에 추가한 후의 content basis를 `B_content^{(N)}`, 수정된 score를 `S'^{(N)}`.

`ε(Q) < ε_0` 인 모든 `Q` 에 대해:

```
sup_K  |S'^{(N)}(Q, K) - (Q·K^T)/√d|  ≤  (α_max · f(ε_0) / √d) · √(ε_0) · ‖Q‖ · ‖K‖ · σ_max(M^{(N)})
```

Bound은 `σ_max(M^{(N)})` 을 통해서만 `N` 에 의존.

**핵심 관찰**: 
- Well-conditioned M^{(N)} (ridge regression with λ > 0)에서 `σ_max(M^{(N)})` 은 `N` 에 대해 **로그 순 증가** (경험적 bound; worst case O(√N) → ridge로 bounded)
- 따라서 **도구 추가가 out-of-domain query를 `ε_0` 고정 floor 이상으로 저하시키지 못함**

### 3.4 Corollary 2 (Bit-exact 무개입 for perfectly orthogonal Q)

`Q ⊥ span(B_intent)` 이면 `ε(Q) = 0` → `α(Q) = α_max · σ(-βτ) ≈ 0` (for large `βτ`) → `|ΔS| ≤ 0` (exactly).

즉 intent subspace와 완전 직교인 query에 대해서는 **bit-exact vanilla score** 가 보장됨. Vocabulary expansion scenario에서 "이 query는 도구와 무관"이라고 판단되는 순간 intervention이 완전히 사라짐.

### 3.5 Counter-theorem (AdaSEKA-style routing의 불가능성)

**주장**: 다음 형태의 attention score 수정은 phase-closure를 만족할 수 없다:

```
S'(Q, K) = (Q·K^T)/√d  +  (1/√d) · Σ_{m=1}^M α_m(Q) · Q · P_m · K^T
```

여기서 `α_m(Q)` 가 max-normalized routing:
```
α_m(Q) = score_m(Q) / max_{m'} |score_{m'}(Q)|
```

**주장**: `ε(Q) → 0` 인 `Q` 가 존재하지만 어떤 `K` 에 대해 `|S'(Q,K) - (Q·K^T)/√d| > C > 0` (non-vanishing perturbation).

**증명 sketch**:

Consider `Q ⊥ ∪_m span(B_m)` (모든 expert subspace와 직교). 그러면:
- `score_m(Q) = Q · P_m · Q_anchor_m = 0` for all `m`
- Max-normalized formula: `α_m(Q) = 0 / max(0,...,0) = 0/0` — undefined

실제 구현에서는 0/0 resolution으로 다음 중 하나 선택:
1. **Uniform fallback**: `α_m = 1/M` → non-zero intervention
2. **Epsilon floor**: `α_m = score_m / max(score, ε)` → numerator 0이면 `α = 0`, OK
3. **Softmax approximation**: `α_m = exp(score_m/T) / Σ exp(score_{m'}/T)` → `α_m = 1/M` when all scores equal

AdaSEKA 논문의 실제 구현 (`Σ(q·u_k)·σ_k / max_{m'} |·|`)은 option (1)에 해당 → max가 0일 때 분모도 0이고, 실제 구현은 `1/M` fallback 또는 epsilon regularization 사용.

**결정적 case**: `ε(Q) → 0+` (완전 orthogonal이 아니지만 점근적)
- Numerator는 `O(ε)`
- Denominator (max)는 **가장 큰 expert** 의 기여로 결정 — `Q` 가 완전 orthogonal이 아닌 이상 어떤 expert가 약간이라도 larger
- 따라서 `α_m*(Q) = O(ε) / O(ε) = O(1)` (non-vanishing)
- 이로 인해 `|ΔS| ≈ α_max · (Q·B_m*) · ... · K = O(√ε · ‖Q‖ · ‖K‖)`

Phase-closure를 정의하기에는 `O(√ε)` 이 너무 빠르게 vanish하지 않음. 우리 Option 4 (self-gate)는 `O(ε^{3/2})`, AdaSEKA-style max-routing은 `O(√ε)` → **차수 차이**.

더 엄밀히: phase-closure를 "bit-exact vanilla when `ε(Q) = 0`" 로 정의하면, AdaSEKA의 max-routing은 `0/0` 특이점에서 regularization에 의해 어떻게 정의되든 `ε → 0+` 극한에서 discontinuity를 가짐. 우리 방법은 `ε = 0` 에서 continuous.

**이 counter-theorem이 formal하게 증명되면 AdaSEKA 대비 가장 강한 novelty argument**: 단순히 "우리 방법이 다르다"가 아니라 "그들의 방법은 우리가 제공하는 property를 **원리적으로** 제공할 수 없다".

### 3.6 Empirical validation 프로토콜

정리의 empirical 검증 (mais 담당, 그림은 paper의 main figure):

1. MMLU에서 1000 query sampling
2. 각 query의 last token에서 `ε(Q) = ‖Q·B_intent‖²/‖Q‖²` 계산
3. 모든 (Q, K) pair에서 `|S'(Q,K) - (Q·K^T)/√d|_∞` 측정
4. Scatter plot: x축 `ε(Q)`, y축 `|ΔS|`, envelope로 theoretical bound 그리기
5. 기대 결과:
   - 모든 empirical point가 theoretical bound 아래 (sanity)
   - `ε(Q) < 0.1` 영역에서 empirical `|ΔS|` 가 `O(ε^{3/2})` 속도로 수렴
   - AdaSEKA baseline의 동일 plot은 `ε(Q) → 0` 에서도 bounded-below → counter-theorem의 empirical evidence

### 3.7 정리가 reviewer 공격을 방어하는 방식

| Reviewer 공격 | 정리의 방어 |
|---|---|
| "이건 AdaSEKA + ontology" | Counter-theorem: AdaSEKA는 phase-closure 불가능, 우리만 가능 |
| "MMLU 결과는 cherry-picked" | 정리가 **structural guarantee** — empirical noise가 아니라 mathematical property |
| "Vocabulary expansion이 실제론 기존 도구만 건드림" | Corollary 1: bound이 `N` 에 대해 `σ_max(M^{(N)})` 에만 의존, well-conditioned case에서 로그 순 |
| "Training-free라고 주장하지만 calibration 필요" | Corollary 2: `Q ⊥ B_intent` 면 bit-exact, calibration이 필요한 구간이 formally characterized |
| "`τ`, `β` hyperparameter 민감" | Theorem은 `τ`, `β` 선택과 무관하게 성립 (단 `f(0)=0` 만 요구); bound의 prefactor만 변경 |

---

## 4. 선행연구 차별화 — Full Prior Art Matrix

### 4.1 핵심 prior art 상세 (memory/prior_art_attention_steering.md + KBIAS_STEERING_EXPERIMENT_PLAN.md + PHASE_B_PAPER_PLAN_v1.md 통합)

#### 4.1.1 Attention steering 계열

**SEKA** (Li et al., ICLR 2026, arXiv:2603.01281)
- Operator: `k' = k + (1/2)(g+ · P+ · k + g- · P- · k)` (K tensor 직접 수정)
- Direction source: SVD of contrastive cross-covariance (GPT-4o synthetic prompt)
- Gate: per-(task, model) scalar `g+, g-`
- Benchmark: CounterFact, BiasBios, Pronoun, Lost-in-Middle. **MMLU/TriviaQA/NQ/KL/perplexity 전부 0 hits**
- Model: Qwen3 + Gemma3 only (no Llama, no Mistral)
- Overhead: +0.03s, +0MB. FlashAttention 호환 (주 selling point)
- 우리와의 차이: K only, per-task gain, SVD 방향, 평가 범위 없음

**AdaSEKA** (같은 논문 §3.3)
- Operator: `P_dynamic(q) = Σ_m α_m(q) · U^+_m · (U^+_m)^T`
- Routing: `α_m(q) = Σ_k (q·u^+_{m,k}) · σ^+_{m,k} / max_{m'} |·|` (max-normalized mixture)
- M = 4 task-specific experts: synthetic, CounterFact, BiasBios, HotpotQA
- K=5 top singular vectors per expert
- Benchmark: CounterFact, BiasBios, Pronoun, Lost-in-Middle (AdaSEKA 자체의 평가)
- **우리와 가장 가까운 선행**. §4.2 에서 point-by-point 차별화.

**Focus Directions** (Zhu et al., 2025, arXiv:2503.23306)
- Operator: `W = softmax((Q + α·d_Q)(K + α·d_K)^T / √F)` (Q와 K에 additive linear shift)
- Direction source: gradient-trained (AdamW, lr=1e-3, 10 epoch on Multi-Doc QA 2654 samples)
- Head selection: Llama-3.2-3B 기준 middle-late layer 8–18, 672 head 중 contextual score >0.2인 head 단 2개 (0.3%), top-20 optimal
- α sweep: {−0.2, 0.2, 0.3, 0.5}, optimal α=0.3
- Benchmark: HELMET only (NQ, TriviaQA, HotpotQA, PopQA, MS MARCO)
- **측정 안 한 것**: MMLU delta, perplexity, fact preservation, hallucination, side-effect, capability degradation
- **Ablate 안 한 것**: K-only vs Q-only vs K+Q (항상 joint)
- 우리와의 차이: Linear shift (우리 bilinear), shared α (우리 Q-conditional), gradient-trained (우리 training-free)

**PASTA** (Zhang et al., NeurIPS 2023, arXiv:2311.02262)
- Operator: post-softmax row reweighting `[T(A)]_{ij} = α·A_{ij}/C_i for j ∈ G`, renormalize
- Default α = 0.01, sweep {0.05, 0.01, 0.002, 1e-3}
- Head selection: multi-task profiling, |H| = 50–150 optimal
- Benchmark: JSON Format, Pronouns Change, BiasBios, CounterFact
- **측정 안 한 것**: MMLU, TruthfulQA, hallucination metric, CAA/ITI/RepE/ActAdd 비교
- 우리와의 차이: Post-softmax (우리 pre-softmax), position-based (우리 semantic), no subspace

**ASA** (Wang et al., Feb 2026, arXiv:2602.04935)
- Operator: `h'_L(x) = h_L(x) + Gate(h_L(x)) · α · MoV(h_L(x))` (residual stream)
- Layer: Qwen2.5-1.5B L=18, LLaMA-8B L=21
- MoV: 4 domain vectors (Code, Math, Search, Translation) + 1 global
- Vector source: class-conditional mean difference, 320 calibration samples, no backprop
- Router: learned linear softmax on standardized hidden state
- Gate: ternary {+1, 0, −1}, threshold τ ∈ [0.5, 0.7]. **Gate 없으면 FPR 0.05 → 0.50 폭발** (Table 6)
- Benchmark: **MTU-Bench only**
- Headline: Qwen2.5-1.5B L=18 α=4.0: F1 0.18 → 0.50, FPR 0.15 → 0.05
- Storage: ~20KB (vs LoRA r16 ~19MB)
- **0.5B model에서 완전 실패** (Recall=0)
- **비교 baseline**: LoRA/Q-LoRA/Prefix/BitFit/prompt. **CAA/ITI/PASTA/RepE 중 어느 것과도 비교하지 않음**
- 우리와의 차이: Residual stream (우리 attention score), learned router (우리 training-free), disjoint domain 가정 (우리 homonym 허용)

**InstABoost** (arXiv:2506.13734, 2025)
- Operator: `β_ij = α_ij · M if 0 ≤ j < K_inst, else α_ij`, then renormalize (post-softmax multiplicative boost)
- M ∈ [2, 20] hyperparameter
- Instruction token: position-based (처음 K_inst 토큰)
- Uniform per (query position i, head, layer)
- "5 lines of code" 단순성
- 우리와의 차이: Post-softmax scalar boost, position-only, no subspace, no Q-gating

**SpotLight** (arXiv:2505.12025, 2025)
- Operator: `B_j = log(ψ_target/ψ_current) if j ∈ S, else 0`; `L'_{ij} = L_{ij} + B_j` (pre-softmax additive)
- `ψ_current = Σ_{j∈S} A_{ij} / Σ_k A_{ik}` (attention mass in user-marked span)
- Dynamic bias (attention mass gap) but position-independent per query i
- User-marked spans
- 우리와의 차이: Scalar bias, attention-mass-based (not subspace), user annotation 필요

**Fact Grounded Attention (FGA)** (Gupta 2025, arXiv:2509.25252)
- Operator: `S_FGA = S + α ⊙ G`, `G = B_qf · A`, `B_qf = Q · K_fact^T / √d_k`, `α = sigmoid(W_α[Q; C] + b_α)`
- **2.1M trainable parameters** (W_K, W_α, b_α); base LLM frozen
- `K_fact` 는 학습된 entity 임베딩, `A` 는 binary entity-to-token mask
- Layer 20-27 of Llama 3.2 3B (deep layers optimal)
- Flat KB: 137 entities × 12 attributes, no hierarchy
- **§6.2.1 admits**: "Future work should explore hierarchical and compositional fact representations" — **공식적으로 우리 ontology 방향을 초대**
- Benchmark: 1107 spec QA (Vanilla 6.3% → FGA-Zero 87.1% → FGA-FT 99.7%), NQ 23.4→41.2, TriviaQA 31.8→48.3, PopQA 12.3→38.7
- **측정 안 한 것**: MMLU, perplexity, TruthfulQA, HaluEval, activation steering 비교 (CAA/ITI/PASTA/RepE/ActAdd)
- Code: github.com/ayushgupta4897/FGA
- 우리와의 차이: Learned fact keys (2.1M params; 우리는 0), binary token mask (우리는 low-rank subspace), token-local (우리는 subspace-global), factual QA target (우리는 tool selection)

**Gated Attention** (NeurIPS 2025, Qwen3-Next architecture)
- Query-dependent head-specific sigmoid gate after SDPA output
- Element-wise sparsity, non-linearity before output projection
- Architectural modification, requires training
- Attention sink 현상 제거
- 우리와의 차이: Architectural (우리 inference-time), post-SDPA (우리 pre-softmax score bonus), trained (우리 training-free)

**Differential Gated Self-Attention (M-DGSA)** (arXiv:2505.24054)
- Per-head input-dependent gating, 각 head를 excitatory/inhibitory branch로 분할, dual softmax fusion
- Architectural, trained
- 우리와의 차이: Architectural, different mechanism

#### 4.1.2 Activation steering family (residual stream / output)

**CAA** (Rimsky et al., ACL 2024, arXiv:2312.06681)
- Residual stream addition, single layer (L=13 for Llama-2-7B-Chat)
- Vector: mean difference at answer-letter token position, A/B contrastive pairs (290–1000 pairs per behavior)
- Multiplier: ±1 (paper never runs curve)
- MMLU at ±1: baseline 0.63, worst drop −0.06 (Hallucination at −1)
- **§9.1 future work explicitly invites "steering outside the residual stream"** — K-bias는 이 future work의 실행
- 우리와의 차이: Residual stream, not attention; no subspace; no self-gating

**ITI** (Li et al., NeurIPS 2023, arXiv:2306.03341)
- Intervention site: per-head attention OUTPUT (input to `o_proj`), pre-softmax-weighted-V
- 수학적으로 `W_O.bias` 상의 constant bias와 동등 — input-independent
- Attention pattern UNTOUCHED
- Probe: per-(l,h) logistic regression on TruthfulQA (5918 QA pairs)
- Direction: mass mean shift (Table 3에서 probe weight 보다 우월)
- Llama-7B optimal: K=48/1024 heads, α=15
- Headline: Llama-7B TruthfulQA True×Info 30.5 → 43.5
- MMLU improves slightly (35.71 → 40.16)
- 우리와의 차이: Attention output (우리 score), input-independent (우리 Q-dependent), probe-trained

**Activation Addition / ActAdd** (Turner et al.)
- Residual stream, contrastive prompt pair 에서 direction 추출
- RepE / CAA 의 직접 조상

**Representation Engineering (RepE)** (Zou et al. 2023, arXiv:2310.01405)
- LAT scan 으로 concept direction 추출, residual stream 에 project
- 우리와의 차이: Residual stream, mechanism level

**SAE-TS** (Chalnev et al., Nov 2024, arXiv:2411.02193)
- SAE feature space steering
- Gemma-2-2B layer 12, 9 tasks, 256 completions × 32 tokens per cell
- Metric: GPT-4o-mini Behavioral rubric + Coherence rubric, scalar summary = Behavioral × Coherence
- **Strict matched-effect 는 하지 않음** (A3 amendment 정정): α sweep + Pareto curve, no pinning
- 우리와의 차이: SAE feature space (우리 attention score), Pareto curve (우리 strict matched-effect 는 차용 candidate)

**Stickland KTS** (Stickland et al., Jun 2024, arXiv:2406.15518)
- Fine-tune + steering combined
- **KL 은 training loss** (Eq. 2), 평가 metric 이 아님
- Actual side-effect metric: MT-Bench
- Matched Prefill-ASR=74%: base MT-Bench 4.67, KTS 5.17
- 우리와의 차이: Fine-tuning 결합 (우리 training-free)

**SteeringControl / SteeringSafety** (arXiv:2509.13450)
- Side-effect 통합 benchmark
- 우리가 metric 으로 차용 고려

#### 4.1.3 외부 지식 → attention 계열

**Fact Grounded Attention**: 위 §4.1.1 참조

**GUIDE** (arXiv:2409.19001), **InstABoost** (2506.13734), **SpotLight** (2505.12025)
- 모두 attention-score family, user 마킹 또는 position-based

#### 4.1.4 Quantization (별개 claim, 우리 dual-claim 불가 확정 후 제외)

- **KIVI** (Liu et al., ICML 2024, arXiv:2402.02750): per-channel K, per-token V, R=128 fp16 residual
- **KVQuant** (Hooper et al., NeurIPS 2024, arXiv:2401.18079)
- **GEAR** (Kang et al. 2024, arXiv:2403.05527)
- **AQUA-KV** (Pinaev et al., ICML 2025, arXiv:2501.19392)
- **KVSink** (Su & Yuan, COLM 2025, arXiv:2508.04257): **no public code**
- **More for Keys, Less for Values** (Feb 2025, arXiv:2502.15075)
- **KITTY** (Nov 2025, arXiv:2511.18643)
- **StreamingLLM** (Xiao et al., ICLR 2024, arXiv:2309.17453): attention sinks
- **TurboQuant**

Kill-switch 결과 `ocq_quant` 단독 −20.7pp, `ocq_quant + bias` −18.6pp → dual-claim 불가 → 본 논문에서 quantization 쪽 claim 제외, Appendix 또는 별도 논문으로.

#### 4.1.5 Tool benchmark

- **MetaTool** (Huang et al., ICLR 2024, arXiv:2310.03128): 47 category, 199 plugins, Subtask1 "similar choices" 995 queries. MIT license. **Primary**.
- **BFCL v3/v4** (Berkeley, 2024/2025): gorilla.cs.berkeley.edu/leaderboard.html
- **ToolBench / ToolLLM** (Qin et al., ICLR 2024, arXiv:2307.16789)
- **MTU-Bench** (ASA 직접 비교)
- **NESTFUL**, **ComplexFuncBench** (THUDM 2025, arXiv:2501.10132), **StableToolBench**
- **τ-bench** (Yao et al. 2024, arXiv:2406.12045)
- **FunctionChat-Bench** (Kakao 2024, arXiv:2411.14054)
- **Seal-Tools**, **UltraTool**, **AppSelectBench** (arXiv:2511.19957)

#### 4.1.6 Long context benchmark (non-tool baseline)

- **LongBench v1** (15 tasks, 4k–32k)
- **RULER** (Hsieh et al., NVIDIA 2024, arXiv:2404.06654)
- **HELMET** (Zhu's Focus Directions benchmark)
- **TriviaQA**, **NQ**, **HotpotQA**, **PopQA**, **MS MARCO**

#### 4.1.7 Fact preservation metric

- **ROME** (Meng 2022): CounterFact efficacy/generalization/specificity triple
- **Yu/Merullo/Pavlick** (2310.15910, 2511.05919): context-memory override rate
- **ROME CounterFact specificity**: neighbourhood fact accuracy
- **WikiText-2 PPL** (legacy quant benchmark)

### 4.2 AdaSEKA Point-by-Point 차별화

| 차원 | AdaSEKA | 제안 방법 | 차이 강도 |
|---|---|---|---|
| **문제** | Factual editing, bias correction, long context | Training-free tool vocabulary expansion | **강** — 완전히 다른 application |
| **Operator 구조** | `S = Q·K^T + Σ_m α_m(Q)·Q·P_m·K^T`, symmetric projectors | `S = Q·K^T + α(Q)·Q·B_intent·M·B_content^T·K^T`, **asymmetric Q/K** | **강** |
| **Gate 함수** | `Σ(q·u_k)·σ_k / max_{m'}` (max-normalized mixture, 항상 어떤 expert 선택) | `α_max·σ(β(ε - τ))` with `ε = ‖Q·B_intent‖²/‖Q‖²` (absolute energy self-gate, 완전 close 가능) | **강** — phase-closure 가능성의 원천 |
| **Expert 구조** | Flat 4 task-level (CounterFact, BiasBios, HotpotQA, synthetic) | 단일 asymmetric pair, ontology facet hierarchy | **중** |
| **Direction source** | 각 task별 benchmark 데이터 → SVD contrastive cross-covariance | Tool catalog anchor sentences, **zero benchmark data** | **중-강** |
| **형식적 보장** | 없음 | Phase-closure theorem + counter-theorem | **강** — 새 theoretical contribution |
| **Evaluation** | CounterFact, BiasBios, Pronoun, Lost-in-Middle | MetaTool, BFCL, MMLU, vocab expansion | **강** — non-overlapping |
| **Storage** | M=4 experts × L × H × d² | (d·r_I + d·r_C) + r_I·r_C per (L,H), ~100배 작음 | **중** |
| **Tool selection** | 없음 | Primary application | **강** |
| **Hierarchical structure** | 없음 (flat mixture) | 4 facet hierarchy | **중** |
| **Phase-closure** | **불가능** (counter-theorem) | **가능** (theorem) | **결정적** |

### 4.3 Baseline 실험 목록

1. **no_steer** (vanilla)
2. **Retrieval-BGE top-5**
3. **LoRA r=16** (MetaTool train split finetune)
4. **SEKA** (static projector, single task expert)
5. **AdaSEKA** (4 experts, max-normalized routing)
6. **Focus Directions** (K+Q linear shift, gradient-trained on small MetaTool subset)
7. **PASTA** (post-softmax row reweight with catalog-marked spans)
8. **InstABoost** (post-softmax boost on instruction tokens)
9. **Spotlight** (pre-softmax attention-mass bias on user-marked spans)
10. **FGA-Zero** (rule-based entity matching, no fine-tuning)
11. **Flat K-bias** (`ocq_bias_a0.3`, kill-switch에서 측정됨: 86.73%)
12. **Proposed: asymmetric Q/K self-gated** (main method)
13. **Ablation: symmetric variant** (`B_intent = B_content`)
14. **Ablation: no self-gate** (`α ≡ α_max`)
15. **Ablation: no M mapping** (`M = I` identity if `r_I = r_C`)
16. **Ablation: single-facet hold-out** (4 configurations)

---

## 5. 실험 계획

### 5.1 자원 분담

- **mais** (develop, 공유 GPU 2장): Qwen2.5-7B 중심, theorem drafting, MMLU 검증, paper 작성
- **iamseungpil** (origin/main, **A100 80GB × 4 전용**): Cross-model replication, LoRA baseline 학습, full benchmark suite, 대규모 vocabulary expansion

**A100 80GB × 4 활용**:
- Qwen2.5-14B 단일 GPU 로드 (28GB, 여유)
- Qwen2.5-32B 단일 GPU 로드 (64GB, 여유) — 옵션 scaling
- Llama-3.1-70B 2 GPU 분산 (140GB)
- 큰 batch size (MMLU 5-shot 가속)

### 5.2 모델 목록

| 모델 | 크기 | 위치 | 목적 |
|---|---|---|---|
| **Qwen2.5-7B** | 7B | mais | Primary, B_ont 존재, kill-switch PASS |
| **Qwen2.5-0.5B-instruct** | 0.5B | mais | Small-model failure analysis |
| **Llama-3.1-8B** | 8B | coworker | Cross-model replication |
| **Mistral-7B-Instruct-v0.3** | 7B | coworker | Cross-family (Phase 1.3에서 negative transfer 경험) |
| **Qwen2.5-14B** | 14B | coworker | Scaling validation |
| **Qwen2.5-32B** (옵션) | 32B | coworker | 대규모 scaling (A100 80GB라 단일 GPU 가능) |
| **Llama-3.1-70B** (옵션) | 70B | coworker | 초대규모 (2 GPU 분산) |

### 5.3 Hyperparameter

MetaTool Subtask1 dev split (첫 100 queries)에서만 tuning 후 고정:

- `r_I = 8`, `r_C = 24`
- `τ = 0.1`, `β = 10`
- `α_max ∈ {0.3, 0.5, 1.0}` sweep

### 5.4 Work Breakdown Structure

**범례**: `[L]` = mais (local), `[C]` = coworker (A100 × 4)

#### Week 1 (Day 0–3): Implementation + Qwen2.5-7B grid + theorem 초안

| ID | 작업 | 담당 | Compute | 시간 |
|---|---|---|---|---|
| W1.1 | Kill-switch 결과 확정 (완료됨: +11.15pp) | L | — | 완료 |
| W1.2 | Asymmetric `B_intent`/`B_content` + `M` builder 구현 | L | CPU | 4h |
| W1.3 | Self-gated attention mask bias 구현 on Qwen2.5-7B | L | CPU + 1 GPU | 6h |
| W1.4 | MetaTool grid: flat, symmetric, asymmetric, facet-level, no-gate, AdaSEKA, SEKA | L | 2 GPU × 8h | 8h |
| W1.5 | MMLU eval driver 구현 (5-shot, 5 subject) | L | CPU | 4h |
| W1.6 | MMLU grid on Qwen2.5-7B × 7 methods | L | 2 GPU × 5h | 5h |
| W1.7 | Theorem 3.1 formal statement + proof sketch | L | — | 4h |
| W1.8 | Counter-theorem 3.3 formal sketch | L | — | 3h |

**Day 3 Milestone**: Qwen2.5-7B grid 완료, theorem 초안, MMLU 결과.

#### Week 2 (Day 4–7): Cross-model + LoRA baseline + theorem 완성

| ID | 작업 | 담당 | Compute | 시간 |
|---|---|---|---|---|
| W2.1 | Llama-3.1-8B `B_intent`+`B_content` 구축 | C | A100 × 1 × 2h | 2h |
| W2.2 | Mistral-7B 구축 | C | A100 × 1 × 2h | 2h |
| W2.3 | Qwen2.5-14B 구축 | C | A100 × 1 × 3h | 3h |
| W2.3b | (옵션) Qwen2.5-32B 구축 | C | A100 × 1 × 5h | 5h |
| W2.4 | Cross-model MetaTool grid: 3-4 models × 7 methods | C | A100 × 4 × 6h | 6h |
| W2.5 | Cross-model MMLU grid: 3-4 models × 3 critical methods | C | A100 × 4 × 5h | 5h |
| W2.6 | LoRA baseline: Qwen2.5-7B + Llama-3.1-8B on MetaTool | C | A100 × 2 × 6h | 6h |
| W2.7 | BFCL V4 eval on Qwen2.5-7B × 7 methods | C | A100 × 2 × 3h | 3h |
| W2.8 | MTU-Bench eval on Qwen2.5-7B × 7 methods (ASA 직접 비교) | C | A100 × 2 × 3h | 3h |
| W2.9 | Theorem 3.1 full proof, typeset | L | — | 4h |
| W2.10 | Counter-theorem 3.3 full proof | L | — | 5h |
| W2.11 | Empirical phase-closure plot on Qwen2.5-7B | L | 1 GPU × 2h | 2h |

**Day 7 Milestone**: Cross-model 전체 결과, LoRA baseline, theorem proof, BFCL/MTU-Bench.

#### Week 3 (Day 8–10): Vocabulary expansion + ablation + paper draft

| ID | 작업 | 담당 | Compute | 시간 |
|---|---|---|---|---|
| W3.1 | Vocabulary expansion 절차 구현 (incremental B_content + M) | L | CPU | 3h |
| W3.2 | Vocabulary expansion on Qwen2.5-7B: 20→50→100→150→199 | L | 1 GPU × 6h | 6h |
| W3.3 | Vocabulary expansion on Llama-3.1-8B | C | A100 × 1 × 5h | 5h |
| W3.4 | Ablation: symmetric vs asymmetric | L | 1 GPU × 2h | 2h |
| W3.5 | Ablation: no self-gate | L | 1 GPU × 2h | 2h |
| W3.6 | Ablation: facet hold-out (4 configs × 2 models) | C | A100 × 1 × 4h | 4h |
| W3.7 | Retrieval-BGE baseline on MetaTool + BFCL | C | A100 × 1 × 3h | 3h |
| W3.8 | Paper draft: Intro + Related Work | L | — | 6h |
| W3.9 | Paper draft: Method (Y) + Theorem (X) | L | — | 8h |
| W3.10 | Paper draft: Experiments + main table + figures | L | — | 10h |
| W3.11 | Paper draft: Problem (Z) + Discussion | L | — | 4h |
| W3.12 | Internal review + revision | L | — | 6h |

**Day 10 Milestone**: Complete paper draft.

### 5.5 Critical path

Serial: W1.1 → W1.4 → W2.4/W2.5 → W3.2/W3.3 → W3.10 → W3.12, 약 7-8일 parallelism 활용시.

병렬 독립 track:
- W1.7-W1.8 (theorem): GPU 불필요
- W2.6 (LoRA): 독립, Day 4부터 시작
- W2.7-W2.8 (BFCL/MTU): 독립
- W3.7 (retrieval): 독립
- Paper 작성 (W3.8+): Day 7부터 병렬

---

## 6. Coworker 위임 패키지

**iamseungpil (A100 80GB × 4 전용)** 에게 위임되는 작업만 나열. 각 task는 self-contained.

### 6.1 사전 조건

- `develop` 브랜치 clone, `scripts/ocq/*`, `external/SEKA/*` 접근
- HF transformers 4.56 (eager attention)
- MetaTool dataset: `/data/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json`
- BFCL V4 (gorilla-llm/berkeley-function-calling-leaderboard)
- MTU-Bench (MuTooL/MTU-Bench)
- MMLU (cais/mmlu)
- TriviaQA, HellaSwag (Day 2+)

### 6.2 위임 작업 (우선순위 순)

---

#### [C1] Cross-model B_ont / B_content / B_intent 구축 (Week 2, Day 4–5)

**입력**:
- 스크립트: `scripts/ocq/build_qwen_metatool_b_ont.py` (develop), `scripts/ocq/build_b_intent.py` (Day 3 commit 예정)
- 모델: Llama-3.1-8B, Mistral-7B-Instruct-v0.3, Qwen2.5-14B, (옵션) Qwen2.5-32B
- Ontology: `reports/axis2_theoretical_verification/metatool_ontology.json`

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env

for MODEL in meta-llama/Llama-3.1-8B mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-14B; do
    SHORT=$(basename $MODEL)
    # B_content
    python scripts/ocq/build_qwen_metatool_b_ont.py \
        --model $MODEL \
        --ontology reports/axis2_theoretical_verification/metatool_ontology.json \
        --out external/SEKA/seka_projections/ontology-$SHORT-metatool/B_content.pt
    # B_intent
    python scripts/ocq/build_b_intent.py \
        --model $MODEL \
        --ontology reports/axis2_theoretical_verification/metatool_ontology.json \
        --out external/SEKA/seka_projections/ontology-$SHORT-metatool/B_intent.pt
    # M matrix
    python scripts/ocq/compute_intent_content_map.py \
        --model $MODEL \
        --b-intent external/SEKA/seka_projections/ontology-$SHORT-metatool/B_intent.pt \
        --b-content external/SEKA/seka_projections/ontology-$SHORT-metatool/B_content.pt \
        --out external/SEKA/seka_projections/ontology-$SHORT-metatool/M.pt
done
```

**출력**: 모델당 3 파일 (`B_content.pt`, `B_intent.pt`, `M.pt`)

**Acceptance**: 각 모델 `r_content` ≥ 20, `r_intent` ≥ 6. 모델당 3h 이내.

**Compute**: 2-3h per model × 3 = 6-9h A100 serial, ~3h with 4× parallel.

---

#### [C2] LoRA baseline 학습 (Week 2, Day 4–6)

**입력**:
- MetaTool train/eval split: `Task2-Subtask1.json` 에서 random 80/20 (seed=42)
- 모델: Qwen2.5-7B, Llama-3.1-8B
- HF peft: `r=16, alpha=32, dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"]`
- 3 epochs, batch_size 16 (A100 80GB), lr 5e-5, AdamW
- Hyperparameters: AlpacaEval reference or HF PEFT tool-calling example

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env
python scripts/baselines/train_lora_metatool.py \
    --model Qwen/Qwen2.5-7B \
    --train_data metatool_train.json --eval_data metatool_eval.json \
    --output_dir lora_qwen25_7b_metatool \
    --epochs 3 --batch_size 16 --lr 5e-5 --lora_r 16

python scripts/baselines/train_lora_metatool.py \
    --model meta-llama/Llama-3.1-8B \
    --train_data metatool_train.json --eval_data metatool_eval.json \
    --output_dir lora_llama31_8b_metatool \
    --epochs 3 --batch_size 16 --lr 5e-5 --lora_r 16
```

(`train_lora_metatool.py` mais Day 3 commit)

**출력**: 2개 LoRA adapter + dev eval accuracy

**Acceptance**: Loss monotonic, dev acc >75% (no_steer 75.58% 기준 넘어야 함)

**Compute**: Qwen2.5-7B ~6h, Llama-3.1-8B ~7h (A100 80GB, batch 16)

---

#### [C3] Cross-model MetaTool grid (Week 2, Day 5–6)

**입력**:
- [C1] 출력 (`B_content.pt`, `B_intent.pt`, `M.pt` per model)
- `scripts/ocq/eval_metatool_asymmetric.py` (mais Day 3 commit)
- 7 methods per model: `no_steer`, `retrieval_bge`, `lora`, `seka`, `adaseka`, `flat_kbias_a0.3`, `ours`

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env

for MODEL in Llama-3.1-8B Mistral-7B-v0.3 Qwen2.5-14B; do
    for METHOD in no_steer retrieval_bge lora seka adaseka flat_kbias_a0.3 ours; do
        python scripts/ocq/eval_metatool_asymmetric.py \
            --model $MODEL --method $METHOD \
            --b-content external/SEKA/seka_projections/ontology-$MODEL-metatool/B_content.pt \
            --b-intent external/SEKA/seka_projections/ontology-$MODEL-metatool/B_intent.pt \
            --m-matrix external/SEKA/seka_projections/ontology-$MODEL-metatool/M.pt \
            --out results/metatool_${MODEL}_${METHOD}.json
    done
done
```

**출력**: 21개 JSON (3 models × 7 methods), top-1 accuracy + pred_counts + runtime

**Acceptance**: NaN/inf 없음, `no_steer`가 Qwen2.5-7B 75.58% 기준 ±3pp 이내 (sanity)

**Compute**: ~15분 per (model, method) × 21 = ~5.3h, 4× A100 parallel ~1.5h

---

#### [C4] Cross-model MMLU eval (Week 2, Day 6)

**입력**: [C1] 출력, `scripts/eval/eval_mmlu.py` (mais Day 4 commit), MMLU cais/mmlu

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env

for MODEL in Llama-3.1-8B Mistral-7B-v0.3 Qwen2.5-14B; do
    for METHOD in no_steer flat_kbias_a0.3 adaseka ours; do
        python scripts/eval/eval_mmlu.py \
            --model $MODEL --method $METHOD --n_samples 1000 \
            --b-content external/SEKA/seka_projections/ontology-$MODEL-metatool/B_content.pt \
            --b-intent external/SEKA/seka_projections/ontology-$MODEL-metatool/B_intent.pt \
            --m-matrix external/SEKA/seka_projections/ontology-$MODEL-metatool/M.pt \
            --out results/mmlu_${MODEL}_${METHOD}.json
    done
done
```

**출력**: 12개 JSON (3 models × 4 critical methods), MMLU 5-shot + per-subject

**Acceptance**: 
- `no_steer` HF leaderboard ±2pp
- `ours`가 `no_steer` ±1pp 이내 (**phase-closure empirical 증거**)
- `flat_kbias_a0.3` 은 non-trivial degradation 예상 (대조군)
- `adaseka` 도 degradation 예상 (counter-theorem empirical 증거)

**Compute**: ~30분 per (model, method) × 12 = ~6h, 4× A100 parallel ~1.5h

---

#### [C5] BFCL V4 + MTU-Bench (Week 2, Day 7)

**입력**: BFCL V4 repo + data, MTU-Bench, `scripts/eval/eval_bfcl.py`, `scripts/eval/eval_mtu.py` (mais Day 5 commit)

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env

# BFCL V4
for METHOD in no_steer retrieval_bge lora seka adaseka flat_kbias_a0.3 ours; do
    python scripts/eval/eval_bfcl.py \
        --model Qwen/Qwen2.5-7B --method $METHOD \
        --out results/bfcl_qwen25_7b_${METHOD}.json
done

# MTU-Bench (ASA 직접 비교)
for METHOD in no_steer adaseka flat_kbias_a0.3 ours; do
    python scripts/eval/eval_mtu.py \
        --model Qwen/Qwen2.5-7B --method $METHOD \
        --out results/mtu_qwen25_7b_${METHOD}.json
done
```

**출력**: BFCL 7 JSON + MTU 4 JSON, per-category accuracy + F1

**Acceptance**: MTU-Bench ASA reference F1 0.50 (Qwen2.5-1.5B L18 α=4.0)이 ceiling reference. 우리가 Qwen2.5-7B에서 이보다 높아야 함.

**Compute**: ~3-4h total on 4× A100

---

#### [C6] Retrieval-BGE baseline (Week 3, Day 8)

**입력**: `BAAI/bge-small-en-v1.5`, MetaTool 도구 정의

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env

for K in 1 3 5 10; do
    python scripts/baselines/eval_retrieval_bge.py \
        --model Qwen/Qwen2.5-7B \
        --tool-catalog metatool_tools.json \
        --retriever BAAI/bge-small-en-v1.5 \
        --top-k $K \
        --eval-set metatool_subtask1.json \
        --out results/retrieval_bge_k${K}_qwen25_7b.json
done
```

(`eval_retrieval_bge.py` mais Day 6 commit)

**출력**: k ∈ {1,3,5,10} 별 top-1 accuracy

**Acceptance**: BGE top-5 >75% (강한 retrieval baseline)

**Compute**: ~2h on 1× A100

---

#### [C7] Vocabulary expansion on Llama-3.1-8B (Week 3, Day 9)

**입력**: `scripts/ocq/eval_vocab_expansion.py` (mais Day 8 commit), [C1] Llama-3.1-8B B_content/B_intent, MetaTool catalog, random ordering seed=42

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env
python scripts/ocq/eval_vocab_expansion.py \
    --model meta-llama/Llama-3.1-8B \
    --b-intent-out B_intent_llama_frozen.pt \
    --b-content-init B_content_llama_init_20tools.pt \
    --tool-order metatool_random_order_seed42.json \
    --steps 20 50 100 150 199 \
    --eval-metatool --eval-mmlu \
    --out results/vocab_expansion_llama_3.1_8b.json
```

**출력**: (step, metatool_acc, mmlu_acc, catalog_size, sigma_max_M, update_latency_s) tuples

**Acceptance**:
- 단계별 MMLU drift ±1pp 이내 (phase-closure empirical evidence)
- `σ_max(M)` 이 N에 대해 로그 순 증가 또는 bounded
- Update latency per 도구 <500ms

**Compute**: ~5h on 1× A100

---

#### [C8] Single-facet ablation (Week 3, Day 9)

**입력**: [C1] B_content files, `scripts/ocq/eval_facet_ablation.py` (mais Day 7 commit)

**절차**:
```bash
source /home/woori/workspace_common/CDP/poc/set.env

for MODEL in Qwen2.5-7B Llama-3.1-8B; do
    for HOLDOUT in function_action io_type domain tool_category; do
        python scripts/ocq/eval_facet_ablation.py \
            --model $MODEL \
            --holdout $HOLDOUT \
            --b-content external/SEKA/seka_projections/ontology-$MODEL-metatool/B_content.pt \
            --out results/facet_ablation_${MODEL}_${HOLDOUT}.json
    done
done
```

**출력**: 8개 JSON (2 models × 4 facets), MetaTool accuracy per hold-out

**Acceptance**: 최소 하나 facet hold-out이 >3pp degradation → facet decomposition 의미 검증

**Compute**: ~4h on 1× A100

---

### 6.3 위임 compute 합계

| Task | Week | Compute (A100·h) | Parallel |
|---|---|---|---|
| C1 Cross-model B_ont | 2 | 6-9 | 3 models × 1 GPU |
| C2 LoRA baselines | 2 | 13 | 2 models × 2 GPU |
| C3 Cross-model MetaTool | 2 | 5 | 21 runs × 4 GPU |
| C4 Cross-model MMLU | 2 | 6 | 12 runs × 4 GPU |
| C5 BFCL + MTU | 2 | 4 | parallel |
| C6 Retrieval-BGE | 3 | 2 | 1 GPU |
| C7 Vocab expansion Llama | 3 | 5 | 1 GPU |
| C8 Facet ablation | 3 | 4 | parallel |
| **합계** | | **~45 h** | Wall-clock ~12-15h on 4× A100 |

**Wall-clock**: 집중 A100 사용 ~2일 (12-15h / 4 GPU) + setup/data/upload ~1일 = **3-3.5 calendar days**

### 6.4 mais 가 위임 전에 commit 해야 할 파일

**Day 3 deadline**:
- `scripts/ocq/build_b_intent.py`
- `scripts/ocq/compute_intent_content_map.py`
- `scripts/ocq/eval_metatool_asymmetric.py` (main method 구현)
- `scripts/baselines/train_lora_metatool.py`

**Day 4 deadline**:
- `scripts/eval/eval_mmlu.py`

**Day 5 deadline**:
- `scripts/eval/eval_bfcl.py`
- `scripts/eval/eval_mtu.py`
- `scripts/baselines/eval_retrieval_bge.py`
- `scripts/baselines/eval_adaseka.py` (AdaSEKA baseline)
- `scripts/baselines/eval_seka.py`
- `scripts/baselines/eval_focus_directions.py`
- `scripts/baselines/eval_pasta.py`
- `scripts/baselines/eval_instaboost.py`
- `scripts/baselines/eval_spotlight.py`
- `scripts/baselines/eval_fga_zero.py`

**Day 7 deadline**:
- `scripts/ocq/eval_facet_ablation.py`
- `scripts/ocq/eval_vocab_expansion.py`

각 스크립트는 mais가 Qwen2.5-7B 단일 샘플로 functional test 완료 후 넘김.

### 6.5 Coordination protocol

- **Daily sync**: 일일 `reports/STATUS_DAILY_YYYY-MM-DD.md` develop 에 commit
- **Blockers**: 즉시 Slack/email, daily sync 대기 금지
- **결과 delivery**: JSON files → `results/` 디렉토리, develop 에 commit
- **Merge protocol**: coworker는 `results/` 만 commit, `scripts/` 수정은 mais review 필수
- **Bug report**: `reports/BUG_REPORT_eval_arch_two_bugs_2026-04-09.md` 포맷 따름

---

## 7. Risk 관리

### 7.1 Technical risks

| Risk | 가능성 | 영향 | 완화 |
|---|---|---|---|
| Phase-closure theorem trivial ("당연") | 중 | 고 | Counter-theorem §3.5 추가; tight-bound equality case 분석 |
| Asymmetric이 symmetric 대비 empirical gain 없음 | 중 | 고 | Day 5 ablation 조기 (W3.4); 실패 시 facet-only 쪽으로 pivot |
| AdaSEKA가 MetaTool에서 우리와 비슷 | 중 | 고 | MMLU phase-closure로 차별화 (theoretical + empirical); main argument 는 non-degradation |
| Self-gate `τ` per-model tuning 필요 | 중 | 중 | τ sweep 포함; validation MMLU 기반 자동 calibration |
| `B_intent` catalog-invariance가 실제로 깨짐 | 중 | 중 | Incremental rebuild 허용; empirical property 로 보고 |
| Monkey-patch가 transformers >4.56 에서 깨짐 | 저 | 저 | 버전 lock; mask-bias fallback |
| Mistral 실패 (Phase 1.3 선례) | 중 | 중 | 모델별 보고; "failure mode" 섹션 |
| α=0.25 dip 재현 안됨 or 다른 모델에서 다른 양상 | 중 | 중 | Per-model α curve 필수 측정; honest 보고 |

### 7.2 Timeline risks

| Risk | 가능성 | 영향 | 완화 |
|---|---|---|---|
| Theorem proof rework | 중 | 중 | Day 4 완료 목표 (Day 7 아님) |
| LoRA baseline 수렴 실패 | 저 | 중 | HF PEFT reference hyperparameter |
| Paper draft 3일 초과 | 고 | 저 | Day 4부터 병렬 작성 |
| Coworker A100 가용성 변동 | 저 | 고 | 위임 패키지 self-contained; mais fallback 2 GPU |
| Script commit 지연으로 coworker idle | 중 | 중 | Day 3 hard deadline; mais가 tested one-sample 증거 함께 |

### 7.3 Venue/scope risks

| Risk | 가능성 | 영향 | 완화 |
|---|---|---|---|
| "이건 AdaSEKA + ontology" | 중-고 | Critical | §4.2 point-by-point + counter-theorem 강조 |
| "세 contribution 모두 incremental" | 중 | 고 | "Only bundle enables vocab expansion with safety" framing |
| "Trivial theorem" | 중 | 고 | Counter-theorem 3.3 가 main novelty; abstract 강조 |
| "Tool selection 덜 흥미로움" | 저 | 중 | Vocab expansion framing; enterprise case 인용 |
| 유사 논문 arxiv 등장 | 저 | Critical | Week 중 weekly sweep; scoop 시 specific 차별화 pivot |

---

## 8. Decision Log & 미해결 질문

### 8.1 결정된 사항 (2026-04-09 ~ 2026-04-10 세션)

1. **Target venue**: Main venue (NeurIPS 2026 / ICLR 2027)
2. **Paper framing**: X+Y+Z triple bundle
3. **Method 이름**: TBD. 후보: FAPC (Facet-Asymmetric Phase-Closed), ICAE (Intent-Content Attention Expansion)
4. **Primary benchmark**: MetaTool Subtask1; Secondary: BFCL, MTU-Bench, MMLU
5. **Primary model**: Qwen2.5-7B (kill-switch +11.15pp 확정); Secondary: Llama-3.1-8B, Mistral-7B, Qwen2.5-14B, (옵션) Qwen2.5-32B / Llama-3.1-70B
6. **Dual-claim 포기**: `ocq_quant` −20.7pp, `ocq_quant + bias` −18.6pp 로 Claim B drop, Claim A 단독 contribution
7. **origin/main 수정 금지**: 모든 개발은 develop, coworker 결과도 develop `results/`
8. **Asymmetric Q/K가 main mechanism**: Q intent / K content 구별이 symmetric SEKA/AdaSEKA 대비 main differentiation
9. **Phase-closure가 main theoretical contribution**: AdaSEKA 불가 counter-theorem 으로 강화

### 8.2 미해결 질문 (사용자 결정 필요)

1. Method 이름 확정 시점?
2. Coworker 쪽 shared storage (HF Hub / S3 / 공유 디스크)?
3. LoRA hyperparameter 정확한 reference?
4. NeurIPS 2026 vs ICLR 2027 (마감일 결정)?
5. Theorem 작성 공동 저자 영입?
6. Qwen2.5-32B / Llama-3.1-70B 옵션 scaling 포함?
7. BFCL/MTU-Bench data prep 주체?

---

## 9. Appendix

### 9.1 Develop 파일 inventory

이미 commit (재활용):
- `scripts/ocq/build_qwen_metatool_b_ont.py`
- `scripts/ocq/build_metatool_ontology.py`
- `scripts/ocq/eval_metatool_subtask1.py`
- `scripts/ocq/eval_hook_mode.py`
- `scripts/ocq/quantizer.py`
- `external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt` (Qwen2.5-7B B_content)
- `reports/axis2_theoretical_verification/metatool_ontology.json`
- `reports/BUG_REPORT_eval_arch_two_bugs_2026-04-09.md` (pushed as 43cf9dc)
- Kill-switch results: `/tmp/metatool_FULL995_alpha_sweep_cuda0.json`, `/tmp/metatool_FULL995_ablations_cuda1.json`
- `memory/phase_b_tool_selection_plan.md`, `memory/prior_art_attention_steering.md`, `memory/baseline_recipes_attention_steering.md`, `memory/eval_arch_two_bugs_2026_04_09.md`, `memory/metatool_subtask1_first_signal_2026_04_09.md` (업데이트 예정)

Week 1 중 mais commit 필요: §6.4 목록 참조

### 9.2 References (전체, KBIAS + PHASE_B_PAPER_PLAN 에서 통합)

**Attention steering family**:
- **SEKA / AdaSEKA**: Li et al., ICLR 2026, arXiv:2603.01281 (github.com/waylonli/SEKA)
- **Focus Directions**: Zhu et al. 2025, arXiv:2503.23306
- **PASTA**: Zhang et al. NeurIPS 2023, arXiv:2311.02262 (github.com/QingruZhang/PASTA)
- **InstABoost**: arXiv:2506.13734 (github.com/BrachioLab/InstABoost)
- **SpotLight**: arXiv:2505.12025
- **Fact Grounded Attention (FGA)**: Gupta 2025, arXiv:2509.25252 (github.com/ayushgupta4897/FGA)
- **Gated Attention** (Qwen3-Next): NeurIPS 2025
- **Differential Gated Self-Attention**: arXiv:2505.24054
- **GUIDE**: arXiv:2409.19001
- **SEA** (SEKA's inspiration): Qiu et al. 2024

**Activation steering family**:
- **CAA**: Rimsky et al. ACL 2024, arXiv:2312.06681 (github.com/nrimsky/CAA)
- **ITI**: Li et al. NeurIPS 2023, arXiv:2306.03341 (github.com/likenneth/honest_llama)
- **RepE**: Zou et al. 2023, arXiv:2310.01405
- **Activation Addition**: Turner et al.
- **ASA**: Wang et al. Feb 2026, arXiv:2602.04935
- **SAE-TS**: Chalnev et al. Nov 2024, arXiv:2411.02193
- **Stickland KTS**: arXiv:2406.15518

**External knowledge → attention**:
- **FGA** (above)

**Fact preservation / side-effect**:
- **ROME**: Meng 2022
- **Yu/Merullo/Pavlick**: arXiv:2310.15910, arXiv:2511.05919 (context-memory override)
- **SteeringControl / SteeringSafety**: arXiv:2509.13450

**Tool benchmarks**:
- **MetaTool**: Huang et al. ICLR 2024, arXiv:2310.03128 (github.com/HowieHwong/MetaTool)
- **BFCL**: Berkeley, gorilla.cs.berkeley.edu/leaderboard.html
- **ToolBench / ToolLLM**: Qin et al. ICLR 2024, arXiv:2307.16789 (github.com/OpenBMB/ToolBench)
- **MTU-Bench**: arXiv:2602.04935 related
- **NESTFUL**, **ComplexFuncBench**: arXiv:2501.10132, THUDM 2025
- **τ-bench**: Yao et al. 2024, arXiv:2406.12045
- **StableToolBench**
- **FunctionChat-Bench**: Kakao 2024, arXiv:2411.14054
- **Seal-Tools**, **UltraTool**
- **AppSelectBench**: arXiv:2511.19957

**Long context / non-tool**:
- **LongBench v1**
- **RULER**: Hsieh et al. NVIDIA 2024, arXiv:2404.06654
- **HELMET**
- **TriviaQA**, **NQ**, **HotpotQA**, **PopQA**, **MS MARCO**, **MMLU**, **TruthfulQA**, **MT-Bench**, **AlpacaEval**, **HellaSwag**

**KV quantization (reference only, dual-claim 제외)**:
- **KIVI**: Liu et al. ICML 2024, arXiv:2402.02750 (github.com/jy-yuan/KIVI)
- **KVQuant**: Hooper et al. NeurIPS 2024, arXiv:2401.18079 (github.com/SqueezeAILab/KVQuant)
- **GEAR**: Kang et al. 2024, arXiv:2403.05527
- **AQUA-KV**: Pinaev et al. ICML 2025, arXiv:2501.19392
- **KVSink**: Su & Yuan COLM 2025, arXiv:2508.04257 (no public code)
- **More for Keys, Less for Values**: Feb 2025, arXiv:2502.15075
- **KITTY**: Nov 2025, arXiv:2511.18643
- **StreamingLLM**: Xiao et al. ICLR 2024, arXiv:2309.17453
- **TurboQuant**

### 9.3 Change log

- **2026-04-10 v1.0**: Initial plan. Kill-switch PASS (+11.15pp) 확정 후 작성. X+Y+Z triple bundle, coworker 4× A100 80GB 활용. Dual-claim drop, Claim A 단독 재작성. Full prior art matrix 통합.
