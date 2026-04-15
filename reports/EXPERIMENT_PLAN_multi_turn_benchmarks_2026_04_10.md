# 실험계획서 — Multi-Turn Tool Selection Benchmarks for Facet-Gated K-Bias

**버전**: v1.0
**작성일**: 2026-04-10
**작성자**: `mais` (Claude session)
**목표 venue**: NeurIPS 2026 / ICLR 2027 main (empirical + theoretical bundle)
**보완 문서**:
- `reports/PHASE_B_PAPER_PLAN_v1.md` (상위 paper plan, 2026-04-10 reframing note 포함)
- `reports/ADASEKA_COMPARISON_2026_04_10.md` (AdaSEKA 실증 비교)
- `reports/COWORKER_REQUEST_cross_model_2026_04_10.md` (coworker A100×4 요청)
- `math/paper/lie_group/COROLLARY_6_7_FACET_PHASE_CLOSURE.md` (신규 이론 corollary 6.7-6.10)
- `memory/adaseka_vs_ours_differentiation_2026_04_10.md`
- `memory/existing_theorem_package_2026_04_10.md`

---

## 0. Motivation 및 이 문서의 위치

Phase B 의 core paper concept (patent-safe rephrasing):

> *현실 enterprise agent scenario — 수십~수백 개의 workflows / plans / tools — 에서 catalog 구조로부터 **tens of orthogonal semantic facets** 을 자동 구성하고, 이를 **K** 에 반영해 per-token multiple focus dimensions 가 simultaneously 활성화되도록 한다. Context-integrated intent detection 은 multi-turn 대화에서 특히 강력하다.*

이 core concept 의 empirical 증명은 **multi-turn tool selection benchmark** 에서만 가능하다. 이유:

1. **Single-turn benchmark (MetaTool 등)** 에서는 Q 가 이미 명확하므로 "context-integrated Q-gate" 의 이점이 드러나지 않는다. Flat K-bias 와 facet-gated K-bias 의 차이가 작게 보일 가능성이 높다.
2. **Multi-turn scenario 에서는** 단일 turn 의 query 가 ambiguous 하고, 이전 turn 의 context 가 누적되어야 intent 가 명확해진다. 이 규제가 바로 우리 method 의 설계 motivation.
3. **"LLMs Get Lost in Multi-Turn Conversation"** (Laban et al. 2025, arXiv:2505.06120) 이 single-turn → multi-turn 에서 30-40% performance drop 을 보고했고, "intent alignment gap" 을 root cause 로 지적했다. 우리 method 는 이 gap 을 context-integrated facet-gating 으로 closing 하는 것이 목표.

즉 multi-turn benchmark 는 **우리 paper 의 main empirical 주장이 성립할 유일한 장** 이다. Single-turn 에서 lift 가 작아도 multi-turn 에서 크게 이기면 paper story 가 성립한다. 반대로 multi-turn 에서 이기지 못하면 paper 의 main contribution 이 무너진다.

---

## 1. 선정된 benchmark 및 우선순위

### Primary benchmarks (paper main table 필수)

#### 1.1 τ²-bench (Sierra Research)

| 속성 | 값 |
|---|---|
| Repo | `https://github.com/sierra-research/tau2-bench` |
| Paper | arXiv:2506.07982 (τ²-bench), arXiv:2406.12045 (τ-bench 원본) |
| 도메인 | 4개: Airline / Retail / Telecom / Mock |
| 형식 | Tool-agent-user 3자 simulation, realistic multi-turn |
| 특징 | Dual-control (agent + user 모두 tool 호출), turn-based half-duplex |
| License | Public GitHub |
| Leaderboard | `https://tau-bench.com/` |

**Paper 에서의 역할**: **Main empirical evidence 의 첫 번째 기둥**. Multi-turn tool use 의 현재 가장 rigorous benchmark. 우리의 context-integrated Q-gate 주장을 직접 검증하는 유일한 일반적으로 인정받는 benchmark.

**우선 실행 도메인**: Retail (가장 일반적, 가장 많이 reported), Airline (보조). Telecom 과 Mock 은 시간 여유 있으면 추가.

#### 1.2 BFCL v3 Multi-Turn Subset (Berkeley)

| 속성 | 값 |
|---|---|
| Repo | `https://github.com/ShishirPatil/gorilla` (bfcl 하위) |
| Paper | ICML 2025 poster, OpenReview id=2GmDdhBdDk |
| Leaderboard | `https://gorilla.cs.berkeley.edu/leaderboard.html` |
| 형식 | v3 에서 multi-turn interactions 도입, v4 에서 agentic |
| 특징 | AST 기반 evaluation, single-turn → multi-turn → agentic 점진적 |
| License | Apache-2.0 |

**Paper 에서의 역할**: **Community standard reproducibility**. 모든 tool-use 논문이 cite. Reviewer expectation 충족. τ²-bench 와 달리 **state-tracking tool sequencing** 까지 포함하므로 우리 method 의 generalization test.

**우선 실행**: v3 multi-turn subset 전체 (약 ~200 cases 예상). v4 agentic 은 optional.

### Secondary benchmark (F scaling 주장 empirical 증명)

#### 1.3 ToolDial (Lim et al., ICLR 2025)

| 속성 | 값 |
|---|---|
| Repo | `https://github.com/holi-lab/ToolDial` |
| Paper | arXiv:2503.00564 (ICLR 2025) |
| 규모 | 11,111 multi-turn dialogues, 평균 **8.95 turns/dialogue** |
| APIs | **473 real APIs × 23 domains** (RapidAPI 기반) |
| Actions | 16 user/system actions (request, clarify, fail-inform 등) |
| License | Public GitHub |

**Paper 에서의 역할**: **"F = 10-100 orthogonal facets" scaling claim 의 결정적 empirical 증거**. 473 API × 23 domain → 자동 구성 facet 수가 10-30+ 달성 예상. 우리 patent core purpose 의 "수십 개 facet" 주장을 empirically 증명 가능.

**우선 실행 전략**: 전체는 너무 큼. **Domain subset (5-10 도메인)** 으로 축소해 F=10+ 달성 여부 확인.

### Motivation citation (benchmark 는 아님)

#### 1.4 "LLMs Get Lost in Multi-Turn Conversation" (Laban et al. 2025)

| 속성 | 값 |
|---|---|
| Paper | arXiv:2505.06120 (Microsoft Research) |
| 후속 | arXiv:2602.07338 (Intent Mismatch follow-up) |
| 발견 | 30-40% performance drop single-turn → multi-turn for ALL top LLMs |
| Root cause | "intent alignment gap" (intent 이 턴마다 underspecified) |

**Paper 에서의 역할**: **Introduction + related work motivation**. Paper 의 gap statement:
- Laban et al. 이 30-40% drop 을 보고했고 intent alignment gap 을 원인으로 지목
- 우리는 context-integrated facet gating 으로 이 gap 을 closing
- 실측: τ²-bench 에서 우리 method 가 flat baseline 대비 multi-turn 에서 X pp 이김

Benchmark 자체를 돌리지는 않음. Citation 만.

### 제외된 후보 (이유 명시)

| 후보 | 제외 이유 |
|---|---|
| **MetaTool** | **Single-turn only** (확인 완료: Task2-Subtask1~4 모두 단일 query). Multi-turn 주장 불가. 단, single-turn baseline 으로 유지. |
| **ToolTalk** | 28 APIs 만, 규모 부족. 현대 tool-use paper 에 잘 cite 되지 않음. |
| **MTU-Bench** | Multi-turn 전용 아님 (multi-granularity). Paper 의 "multi-turn intent" 주장과 fit 이 덜함. |
| **MultiIF** | Tool-use 전용 아님, 일반 instruction following. |
| **AgentBench** | Agentic 전체 evaluation 이라 scope 과다. Tool selection 에 focus 가 아님. |
| **τ-bench v1** | v2 (τ²-bench) 로 superseded. v1 은 single-control 이라 v2 보다 약함. |
| **BFCL v4 agentic** | v3 multi-turn 먼저. v4 agentic 은 v3 통과 후 optional extension. |

---

## 2. 실험 구조 및 main 비교 grid

### Method 리스트 (모든 benchmark 에서 동일)

| 코드 | Method | Intervention | Gate source | Direction source |
|---|---|---|---|---|
| **A** | `no_steer` | 없음 | — | — |
| **B** | `ocq_bias_a0.3` (flat K-bias, 2026-04-09 확정) | k_proj forward hook | 없음 (uniform α) | Catalog ontology (4-facet, r_ont=24) |
| **C** | `ocq_facet_gated_a1.0` (new 2026-04-10) | k_proj forward hook | `g_f(k_t) = ‖B_f^⊤ k_t‖² / ‖k_t‖²` — **token-local** per-facet | Same catalog ontology |
| **D** | `ocq_q_gated` (new, to-implement) | Attention score bonus | `g_f(q_intent) = ‖B_f^⊤ q_last-user‖² / ‖q_last-user‖²` — **context-integrated** | Same catalog ontology |
| **E** | `ocq_qk_gated` (new, to-implement) | Attention score bonus | `g_f(q, k_t) = max(g_f(q_intent), g_f(k_t))` — **composition** | Same catalog ontology |
| **F** | `adaseka_3expert` | K-side via `AdaptiveSEKALLM` | `α_m(q)` max-normalized routing | Per-expert SVD (benchmark-specific contrastive) |
| **G** | `seka_vanilla_single` | K-side via `SEKALLM` | Fixed g+/g- | Single expert SVD |
| **H** | `cot_prompt` (prompt engineering baseline) | None (prompt-only) | — | — |
| **I** | `retrieval_bge` (RAG baseline) | None (prompt injection) | BGE top-k retrieval | — |

**총 9개 method**. A-E 는 우리 제안 method 및 그 variants, F-G 는 prior art 직접 비교, H-I 는 naive baselines.

### Benchmark × Method grid

| Benchmark | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| **τ²-bench retail** (primary) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **τ²-bench airline** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **BFCL v3 multi-turn** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **ToolDial domain subset (5 domains)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| **MetaTool single-turn (legacy)** | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| **MMLU subset (phase-gating check)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

**총 셀 수**: 52. 모든 셀 × 3 models (Qwen2.5-7B / Llama-3.1-8B / Mistral-7B-v0.3) = **156 실험**.

### Metric 정의

**Primary metric (τ²-bench, BFCL v3, ToolDial)**:
- **Turn-level tool selection accuracy**: 각 agent turn 에서 올바른 tool 이 호출되었는가
- **Trajectory success rate**: 전체 dialogue 가 올바르게 완료되었는가 (state-tracking 기준)
- **Tool-call sequence correctness**: 도구 호출 순서 맞음 / 잘못 (BFCL AST 기준)

**Secondary metric**:
- **Turn-wise intent alignment score**: 각 turn 에서 모델의 latent intent ($q_{\text{intent}}$) 가 ground-truth intent 와 얼마나 일치하는가 (cosine similarity on intent subspace)
- **Facet activation profile**: turn 별로 어떤 facet 이 활성화됐는지 (g_f 값의 histogram). Qualitative analysis.

**Phase-gating metric (MMLU)**:
- 각 method 의 MMLU 정확도와 no-steer baseline 의 차이. 우리 주장: **0 차이 또는 양수 차이** (non-tool query 에서는 intervention 이 effective 0).

---

## 3. 구현 계획 (순차적 task)

### Week 1 — Benchmark 환경 구축

**T1.1 τ²-bench clone + 환경** (0.5일)
```bash
cd /home/woori/workspace_common/boltzmann-attention/external
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
source /home/woori/workspace_common/CDP/poc/set.env && pip install -e .
tau2 --help
```
- Task 형식, API schema 파악
- 기본 `gpt-4.1` agent run 으로 sanity check

**T1.2 τ²-bench catalog → facet ontology 포팅** (1일)
- `external/tau2-bench/src/tau2/domains/retail/tools.py` 에서 tool schema 추출
- `scripts/ocq/build_metatool_ontology.py` 를 τ²-bench catalog format 에 맞게 포팅 → `scripts/ocq/build_tau2_ontology.py`
- 예상 facet: function_action (retail: search/purchase/return/refund/...), io_type (product/order/user/transaction), domain (retail-specific), tool_category (CRUD vs query vs side-effect)
- Output: `reports/axis2_theoretical_verification/tau2_retail_ontology.json`

**T1.3 BFCL v3 multi-turn clone + harness** (0.5일)
```bash
cd /home/woori/workspace_common/boltzmann-attention/external
git clone https://github.com/ShishirPatil/gorilla
cd gorilla/berkeley-function-call-leaderboard
# bfcl 폴더의 multi-turn eval 형식 파악
```
- Multi-turn categories: `multi_turn_base` / `multi_turn_miss_func` / `multi_turn_miss_param` / `multi_turn_long_context`
- Evaluation format (AST 기반) 확인

**T1.4 ToolDial clone + domain subset 선정** (0.5일)
```bash
git clone https://github.com/holi-lab/ToolDial
# 23 domains 중 orthogonal 하게 선택된 5-10 개
```
- 5 domains 선택 (tentative): e.g., Travel, Finance, Shopping, Healthcare, Education
- 각 domain 의 API 수 확인 → F (facet 수) 예상

### Week 2 — Multi-turn eval driver

**T2.1 Multi-turn prompt 구조 통합** (1일)
- τ²-bench, BFCL, ToolDial 각각 prompt format 이 다름
- 통일된 `MultiTurnDialogue` 데이터 구조 정의
- Turn boundary marker (`<|user|>`, `<|assistant|>`) detection 로직

**T2.2 Q-gated score bonus hook 구현 (Method D, E)** (2일)
- `scripts/ocq/eval_multi_turn.py` (new driver)
- `install_q_gated_score_hooks(...)` — attention forward patch
  - HF `LlamaAttention.forward` monkey-patch (eager attention 강제)
  - `q_intent = hidden_state[last_user_turn_boundary_position]`
  - Per-facet gate 계산 + score bonus 적용
- Method tag: `ocq_q_gated_a{α}`, `ocq_qk_gated_a{α}`
- Sanity check: Qwen2.5-7B, 10 τ²-bench cases, errors 없는지
- **주의**: Transformers 버전 lock 필수 (이미 vllm_env 에 고정되어 있음)

**T2.3 AdaSEKA port to multi-turn** (1일)
- 현재 `AdaptiveSEKALLM` 은 CounterFact 전용 (single-query eval)
- Multi-turn dialogue 에 적용하려면: per-turn forward 에 expert routing 적용
- `eval_multi_turn.py` 에 method `adaseka_{expert_set}` 추가
- Expert set: `{synthetic, biasbios, hotpot}` (catalog 무관) 또는 `{synthetic, tau2_retail, tau2_airline}` (domain-specific)

**T2.4 SEKA vanilla port to multi-turn** (0.5일)
- `SEKALLM` 을 multi-turn driver 에 통합

**T2.5 Baseline (no_steer, cot_prompt, retrieval_bge) 통합** (0.5일)
- BGE retrieval: `BAAI/bge-large-en-v1.5`, top-k 도구 injection

### Week 3 — Main experiments (Qwen2.5-7B primary)

**T3.1 τ²-bench retail sanity (50 cases)** (1일)
- 9 methods × 50 cases = 450 runs
- Latency 측정, error 잡기
- Sanity baseline: no_steer ≈ 40-60% (tau2 typical)

**T3.2 τ²-bench retail full** (1-2일)
- 전체 retail case (약 250개 추정)
- 9 methods × 250 cases = 2250 runs
- Report: trajectory success rate, turn-level accuracy

**T3.3 τ²-bench airline full** (1일)
- Retail 과 동일 protocol
- Cross-domain robustness check

**T3.4 BFCL v3 multi-turn** (1일)
- Multi-turn subset 전체 (약 200 cases)
- AST 기반 scoring

**T3.5 MetaTool legacy** (0.5일, 기존 결과 재활용)
- 현재 flat α=0.3 결과 있음 (+11.15pp on Qwen)
- Method C/D/E 추가만 필요

### Week 4 — Cross-model + ToolDial

**T4.1 Cross-model τ²-bench retail** (coworker A100×4, 2일)
- Llama-3.1-8B, Mistral-7B-v0.3 + Qwen2.5-7B baseline
- Core benchmark 만 (retail) 이라도 3 모델 전부
- **Mistral 이 결정적**: Phase 1.x negative 이력 있음

**T4.2 ToolDial domain subset** (1일)
- 5 domains, 500-1000 cases
- Facet build: F ≥ 10 확인
- Method C/D/E 위주 (AdaSEKA 생략)

**T4.3 MMLU phase-gating check** (1일)
- 1000 MMLU 샘플, 9 methods × 3 models
- 목표: 우리 method (C/D/E) 는 no_steer 대비 ±0.5pp 이내

### Week 5 — Analysis + paper draft

**T5.1 Theorem 6.1/6.2 factor decomposition on multi-turn data** (2일)
- `verify_prerope_pca.py` 를 adapt 해서 qaMSE / Var_s[V] / Λ 직접 측정
- 우리 method 의 qaMSE(q; E) vs AdaSEKA 의 qaMSE(q; E) 비율
- Cor 6.10 의 empirical verification: ratio 가 end-to-end accuracy ordering 과 일치하는지

**T5.2 Multi-turn vs single-turn lift gap 정량화** (1일)
- 같은 model, 같은 method 로 single-turn MetaTool 결과와 multi-turn τ²-bench 결과 비교
- Laban et al. 의 30-40% drop 이 우리 method 로 얼마나 closing 되는지

**T5.3 Paper draft 작성** (2-3일)
- Paper structure (tentative):
  - §1 Introduction: Laban et al. gap citation + core concept statement
  - §2 Related Work: AdaSEKA, SEKA, Focus Directions, τ²-bench, BFCL, ToolDial
  - §3 Method: facet construction + Q-gated K-bias operator + Cor 6.7-6.10 statement
  - §4 Theory: Theorem 6.1 (inherited) + Cor 6.7-6.10 (new) + Λ-cancellation comparison
  - §5 Experiments: main grid (Week 3-4)
  - §6 Analysis: turn-wise facet activation, MMLU phase-gating
  - §7 Discussion: limitations, future work

---

## 4. Resource 배정

### GPU 배분

| GPU | 기간 | 담당 실험 | 책임자 |
|---|---|---|---|
| A6000×1 (local cuda:0) | Week 1-2 | Method D/E 구현, sanity, τ²-bench retail 50 sample smoke | mais |
| A6000×1 (local cuda:1) | Week 1-2 | AdaSEKA multi-turn port + τ²-bench airline sanity | mais |
| A100 80GB × 4 (coworker) | Week 2-4 | Cross-model (Llama/Mistral), τ²-bench retail full, ToolDial domain subset | iamseungpil (coworker) |
| A6000×2 (local) | Week 3-4 | BFCL v3 multi-turn, MetaTool legacy re-run, MMLU phase-gating | mais |
| A100 (coworker) | Week 5 | Theorem factor verification, additional cross-model ablations | iamseungpil |

### Coworker 추가 요청 (기존 `COWORKER_REQUEST_cross_model_2026_04_10.md` 확장)

기존 요청: Llama-3.1-8B / Mistral-7B-v0.3 on MetaTool full 995 α sweep. 추가로:

- **R1 (Week 2 초)**: τ²-bench retail + airline domain 에서 Qwen / Llama / Mistral 3모델 × Method A-E baseline eval. 각 모델 약 250 cases × 5 methods = 1250 runs × 3 모델 = **3750 runs**. Multi-turn dialogue 은 turn 수가 평균 8-10 이므로 single-turn 대비 8-10× 느림. A100 4장으로 예상 **2-3일**.
- **R2 (Week 3)**: BFCL v3 multi-turn 전체 benchmark 를 3 모델에 run. 약 200 cases × 9 methods × 3 models = **5400 runs**. **1-2일**.
- **R3 (Week 4)**: ToolDial domain subset (5 domains, 500 cases) × 3 models × Methods A-E = **7500 runs**. **2일**.

별도 요청 문서: `reports/COWORKER_REQUEST_multi_turn_2026_04_10.md` (작성 예정).

---

## 5. Gate 정의 (각 단계의 PASS/FAIL 판정)

**Gate M1 — τ²-bench retail sanity (Week 3, day 1)**
- Qwen2.5-7B on 50 τ²-bench retail cases
- PASS 조건: method C 또는 D 가 method B (flat) 대비 ≥ 2pp lift in trajectory success rate
- FAIL 대응: Method D/E 구현에 버그 가능성, 재검증. 다중 시행 시 FAIL 유지되면 "multi-turn benefit 없음" 판정 → paper scope 축소 (single-turn only, flat α=0.3 주장 유지)

**Gate M2 — τ²-bench retail full + cross-model (Week 3-4)**
- 3 models × retail full
- PASS 조건: method D 또는 E 가 ≥ 2/3 모델에서 method B 대비 ≥ 3pp lift
- FAIL 대응: paper 의 main claim 을 "context-integrated Q-gate is effective in specific domains" 로 축소. Retail 단독 결과로만 paper. Venue 하향 (NeurIPS → EMNLP findings).

**Gate M3 — BFCL v3 multi-turn (Week 3)**
- BFCL 는 state-tracking 이 포함되므로 stricter
- PASS 조건: method D 또는 E 가 flat 대비 ≥ 1pp lift (BFCL 는 전반적으로 baseline 이 높아 lift 작음 예상)
- FAIL 대응: BFCL 는 cite only, τ²-bench 중심으로 paper

**Gate M4 — MMLU phase-gating (Week 4)**
- Method C/D/E 가 MMLU 에서 degradation 없음 증명
- PASS 조건: |Δ MMLU| ≤ 0.5pp for methods C, D, E vs no_steer
- FAIL 대응: phase-gating 주장 retract, paper 의 Cor 6.7/6.8 subsection 을 "empirical observation only" 로 격하

**Gate M5 — ToolDial F scaling (Week 4)**
- ToolDial domain subset 에서 auto-constructed facet 수 ≥ 10
- PASS 조건: F ≥ 10 AND 각 facet pair 의 NMI < 0.3 (직교성)
- FAIL 대응: "F scaling" 주장 축소, 4 facet 고정으로 paper 작성

### 전체 GO/NO-GO

- **GO (main venue)**: M1, M2, M4 모두 PASS + M3 또는 M5 중 최소 하나 PASS
- **GO (EMNLP/ACL main)**: M1, M2 PASS + M3 또는 M4 또는 M5 중 하나 PASS
- **Workshop only**: M1 PASS + 나머지 중 하나라도 partial PASS
- **Abort / reframe**: M1 FAIL (multi-turn 에서 이점 없음 = paper 의 main claim 무효)

---

## 6. Known risks 및 완화

### R1: τ²-bench 의 LLM-as-judge evaluation

τ²-bench 는 agent 와 user 모두 LLM simulation 이라, trajectory success 가 user simulator 의 판단에 의존. 우리 method 가 user-side 판정에 bias 줄 수 있음 (동일 LLM 을 양쪽에 쓰면).

**완화**: user simulator 는 다른 family (e.g., GPT-4o via API) 로 고정. 우리 method 는 agent-side 만 적용. 이 분리가 τ²-bench standard protocol 에 부합.

### R2: BFCL v3 multi-turn state-tracking 복잡도

BFCL v3 multi-turn 은 previous turn 의 tool 실행 결과가 다음 turn 에 state 로 들어감. 우리 method 는 attention steering 이므로 state 추론까지 책임지지 않음.

**완화**: BFCL 결과가 lift 작게 나오는 것은 expected. "우리 method 는 tool selection 에 집중, state tracking 은 직교적 문제" 라는 caveat 을 paper 에 명시.

### R3: ToolDial 473 API 중 직교 facet 수 부족

473 API 가 많은 중복을 가질 수 있음. 실제 orthogonal facet 수는 10 미만일 가능성.

**완화**: Week 1 단계에서 facet build 결과 확인. F < 10 이면 ToolDial 을 scaling proof 로 쓰지 않고 일반 multi-turn benchmark 로만 사용. "F scaling" 주장은 reserved for future work.

### R4: Multi-turn eval 의 latency

τ²-bench 의 dialogue 는 평균 10+ turns. 각 turn 마다 prefill + decode. 전체 benchmark 돌리는 데 single-turn MetaTool 대비 **8-15× 느림**.

**완화**: sample 수 축소 (retail 250 → 100 if 시간 부족). Coworker A100 4장 병렬 활용. Methods 축소 (H, I baseline 생략 가능).

### R5: Method D (Q-gated score bonus) 의 구현 복잡도

HF attention forward patch 는 transformers 버전에 민감. 버전 upgrade 시 깨질 수 있음.

**완화**: transformers 버전 lock (이미 `vllm_env` 에 고정). Patch 를 monkey-patch 형태로 유지하되 LlamaAttention, Qwen2Attention, MistralAttention 세 가지 모두 지원 필요. ~300줄 예상.

### R6: "LLMs Get Lost" simulation benchmark 재사용 위험

Laban et al. 2025 의 simulation 을 돌리는 것은 그들의 methodology 재현이지 우리 contribution 이 아님.

**완화**: Benchmark 로 쓰지 않음. Introduction 의 motivation citation 으로만. 30-40% drop 을 "gap we aim to close" 로 기술.

---

## 7. 성공 기준 (어떤 숫자가 나와야 "이겼다"고 할 수 있는가)

**Minimum viable result (paper 성립 조건)**:
- τ²-bench retail + BFCL v3 multi-turn 에서 method D 또는 E 가 flat K-bias (B) 대비 **≥ 3pp lift in trajectory success rate**
- Cross-model (최소 2/3 models) 에서 재현
- MMLU phase-gating 에서 **|Δ| ≤ 0.5pp**
- AdaSEKA 대비 **≥ 5pp lift** (CounterFact 에서 이미 확보, multi-turn 에서 재확인)

**Strong result (main venue competitive)**:
- τ²-bench retail 에서 method D 가 flat 대비 **≥ 5pp lift**
- 3/3 모델 재현
- ToolDial F ≥ 10 달성 + facet ablation 이 "F 가 클수록 유리" 를 보임
- Cor 6.8 의 empirical verification: `qaMSE(q; E)` 의 `ε_q` 선형 감소가 측정됨

**Breakthrough result (ICLR/NeurIPS spotlight 가능)**:
- τ²-bench 에서 context-integrated Q-gate 가 flat 대비 **≥ 10pp lift**
- 모든 모델 + benchmark 에서 monotone improvement
- Laban et al. 의 30-40% drop 을 > 50% closing (30pp drop → 15pp drop)
- F scaling 에서 F=4 → F=20 으로 갈 때 성능 monotone 증가

---

## 8. 타임라인 요약

| Week | 주요 활동 | Gate | Deliverable |
|---|---|---|---|
| 1 | Benchmark clone + facet build | — | τ²-bench + BFCL v3 + ToolDial 로컬 환경 |
| 2 | Multi-turn eval driver + Method D/E 구현 | — | `scripts/ocq/eval_multi_turn.py` |
| 3 | τ²-bench + BFCL sanity + full (Qwen) | **M1, M3** | Main table 의 Qwen column |
| 4 | Cross-model + ToolDial + MMLU | **M2, M4, M5** | Full main table |
| 5 | Theorem verification + paper draft | GO/NO-GO | Paper draft v1 |

**총 5주** (parallel 실행 포함). Coworker 협력 필수.

---

## 9. 기존 2026-04-10 작업과의 관계

이 계획서는 오늘 진행된 작업 위에 builds on:

| 오늘 완료 | 이 계획서에서의 역할 |
|---|---|
| AdaSEKA 2-expert CounterFact eval (ES 48.2) | 이미 확보, Phase 1.x appendix table |
| AdaSEKA 3-expert CounterFact eval (ES 86.8) | 이미 확보, CounterFact comparison 완료 |
| `install_facet_gated_hooks` (Method C) 구현 | Method C 로 본 grid 에서 재사용 |
| `COROLLARY_6_7_FACET_PHASE_CLOSURE.md` 작성 | Paper §4 Theory section 의 core |
| `adaseka_vs_ours_differentiation_2026_04_10.md` 메모리 | Paper §2 Related Work 의 core |
| `existing_theorem_package_2026_04_10.md` 메모리 | Theorem 6.1/6.2/Cor 6.3 재활용 근거 |
| MetaTool 1a/1c 실험 (모두 catastrophic) | Paper 의 "quant path 는 본 method 의 main 방향이 아님" 을 정당화. Negative appendix. |
| Coworker cross-model request (기존) | Week 2-4 의 R1/R2/R3 로 확장됨 |

이 계획서는 **새 문서** 가 아니라 기존 작업의 **multi-turn 축 연장** 이다. Core concept, theorem package, AdaSEKA comparison 은 모두 유지. Multi-turn benchmark 가 추가됨으로써 우리 paper 의 **main empirical 주장이 성립할 장** 이 확보된다.

---

## 10. 즉시 다음 행동

1. **(지금)** 이 계획서를 `reports/` 에 저장 ✓ (이 파일)
2. **(다음)** `reports/COWORKER_REQUEST_multi_turn_2026_04_10.md` 작성 — coworker 에게 R1/R2/R3 전달
3. **(Week 1 day 1)** τ²-bench clone + `uv sync` + 환경 sanity
4. **(Week 1 day 2-3)** τ²-bench retail catalog → facet ontology 빌드
5. **(Week 1 day 4-5)** BFCL v3 multi-turn 환경 + 통합 prompt format
6. **(Week 2)** Method D (Q-gated score bonus) 구현 시작

현재 backgrounds:
- cuda:0: `ocq_quant_bias_a0.3` 1a mode 진행 중 (결과는 이미 catastrophic 예상 — bias 가 구할 가능성 낮음)
- cuda:1: `ocq_quant_bias_a0.3` 1c mode 진행 중 (동일)

이 background 2개 완료 후 GPU 재배분.

---

**Document signed**: mais (Claude), 2026-04-10
**Review pending**: user approval
**Next update trigger**: Gate M1 결과 (τ²-bench retail 50 sample sanity)
