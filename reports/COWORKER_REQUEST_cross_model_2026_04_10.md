# Coworker 실험 요청 v2 — Cross-model + Multi-Turn Benchmark 검증

**작성일**: 2026-04-10 (v1) → **2026-04-10 저녁 업데이트 v2**
**요청자**: mais (develop 브랜치)
**대상**: iamseungpil (origin/main, **A100 80GB × 4**)
**긴급도**: 높음 — Phase B paper 의 main empirical 주장이 이 결과에 의존
**예상 GPU 시간**: R1 (~3시간) + R2 (2-3일) + R3 (1-2일) + R4 (2일) — 총 약 5-7일 (병렬 최대 활용 시)

---

## 0. v2 업데이트 배경 (2026-04-10 오후-저녁 진행된 작업)

v1 작성 이후 오늘 다음이 진행됨:

1. **AdaSEKA 2-expert/3-expert 실험 완료 (CounterFact 500, Qwen3-4B)**:
   - 2-expert held-out: ES **48.2** (near-baseline 40.2, only +8pp)
   - 3-expert in-domain: ES **86.8** (여전히 single-expert SEKA vanilla 95.2 보다 낮음)
   - Ontology rank-8 α=3.0: ES **96.8** (best, 우리 방법)
   - **결론**: AdaSEKA 의 Q-adaptive routing 은 mixture dilution 때문에 structurally 약함. See `reports/ADASEKA_COMPARISON_2026_04_10.md`.

2. **기존 theorem package 재발견**: `math/paper/lie_group/APPENDIX_B_PROOFS.md` 에 Theorem 6.1, 6.2, 6.16.3 + Cor 6.3-6.6 모두 이미 증명 완료. qaMSE framework (query-projected attention-weighted MSE) 가 우리 facet-gated K-bias 에 직접 적용됨. See `memory/existing_theorem_package_2026_04_10.md`.

3. **새 corollaries 작성 (6.7-6.12)**: `math/paper/lie_group/COROLLARY_6_7_FACET_PHASE_CLOSURE.md`. 4개 + 추가 2개 = 총 6개 corollary. 기존 Theorem 6.1/6.2/6.3 에만 의존, 총 ~2페이지 proof.

4. **1a / 1c per-token hard selection 실패 (MetaTool 995)**:
   - `ocq_quant 1a` (sign-only): **0.90%** top-1 (catastrophic, -74.68pp)
   - `ocq_quant 1c` (per-token argmax): **1.41%** top-1 (-74.17pp)
   - **Cor 6.11 (hard selection penalty) 의 empirical verification** — 실패 실험이 이론 support 로 전환됨

5. **Multi-turn benchmark plan 확정**: `reports/EXPERIMENT_PLAN_multi_turn_benchmarks_2026_04_10.md`. τ²-bench / BFCL v3 / ToolDial 3개 benchmark 선정.

6. **Paper core concept 확정**:
   > *현실 enterprise scenario (수십~수백 workflows/plans/tools) 에서 catalog 로부터 **수십 개 직교 facet** 을 자동 구성, **K 에 반영** 하여 **multiple focus 가 simultaneously** 활성화. Multi-turn 대화에서 context-integrated intent detection 이 특히 강력.*

v2 요청은 **이 확정된 방향에 맞춰 cross-model (R1 기존) + multi-turn (R2/R3 추가) + phase-gating (R4 추가)** 네 가지 track.

---

## 1. R1 — Cross-model MetaTool α sweep (기존 v1 요청, 변경 없음)

### 배경 (v1 그대로)

2026-04-09 Qwen2.5-7B / MetaTool Subtask1 (995 queries) 전체 α sweep:

| Method | top-1 | Δ vs no_steer |
|---|---|---|
| no_steer | 75.58% | baseline |
| ocq_bias α=0.20 | 84.02% | +8.44 |
| ocq_bias α=0.25 | 65.23% | **−10.35 (dip)** |
| ocq_bias α=0.30 | **86.73%** | **+11.15 ✅** |
| ocq_bias α=0.35 | 83.12% | +7.54 |
| ocq_bias α=0.40 | 73.37% | −2.21 |

### 요청 내용

**Task R1.A — Llama-3.1-8B B_ont build + α sweep**

```bash
# Build (A100 1장, ~10분)
source /home/woori/workspace_common/CDP/poc/set.env && \
cd /home/woori/workspace_common/boltzmann-attention && \
python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --ontology-json reports/axis2_theoretical_verification/metatool_ontology.json \
    --out external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt \
    --target-layers all \
    --device cuda:0

# Eval (A100 1장, ~30분 — 6 methods)
python scripts/ocq/eval_metatool_subtask1.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --device cuda:0 \
    --methods no_steer ocq_bias_a0.2 ocq_bias_a0.25 ocq_bias_a0.3 ocq_bias_a0.35 ocq_bias_a0.4 \
    --b-ont external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt \
    --out /tmp/metatool_FULL995_llama31_8b_alpha_sweep.json
```

**Task R1.B — Mistral-7B-v0.3 B_ont build + α sweep**

위 R1.A 와 동일, `--model mistralai/Mistral-7B-v0.3`, output path `ontology-mistral-7b-v03-metatool/`, `/tmp/metatool_FULL995_mistral_7b_v03_alpha_sweep.json`.

**주의 (Mistral 관련)**: Phase 1.x 에서 Mistral 은 toy ontology 로 negative 였음 (`memory/phase1_3_ontology_beats_seka.md`). **Catalog-derived MetaTool ontology 로 전환되면서 결과가 바뀌는지** 가 핵심 관심사. 이 결과가 paper direction 을 결정.

### R1 병렬화 (A100 × 4 활용)

- cuda:0 → Llama build + eval
- cuda:1 → Mistral build + eval
- cuda:2, cuda:3 → optional: α=0.15, 0.45, 0.50 추가값 또는 다른 seed 로 variance 측정

**R1 총 소요**: 4장 병렬 시 **~2-3시간**.

### R1 Deliverables

1. `/tmp/metatool_FULL995_llama31_8b_alpha_sweep.json`
2. `/tmp/metatool_FULL995_mistral_7b_v03_alpha_sweep.json`
3. `external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt`
4. `external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool/B_ont.pt`
5. 각 build 로그에서 `r_per_pair`, `r_min/r_median/r_max`, `n_skipped` 숫자

### R1 PASS/FAIL 판정

- **PASS**: no_steer 대비 α=0.2, 0.3, 0.35 중 최소 1개에서 ≥ +5pp lift
- **FAIL**: 모든 α 에서 lift < 3pp
- Llama PASS + Mistral PASS → Phase B 유지 + multi-turn 확장
- Llama PASS + Mistral FAIL → "Qwen+Llama-family" scope, Mistral 별도 section
- 둘 다 FAIL → Qwen-specific, paper scope 축소

---

## 2. R2 — τ²-bench Multi-Turn Tool Selection (NEW)

### 배경

**Multi-turn tool use 의 현재 most rigorous benchmark**. Sierra Research (NeurIPS 2024 원본 + 2025 τ²). Retail, Airline, Telecom, Mock 4 도메인. Realistic multi-turn dialogue simulation.

**왜 필요한가**: 우리 paper 의 main 주장 — "context-integrated intent detection via multi-facet K-bias" — 은 **single-turn benchmark (MetaTool) 에서는 드러나지 않음**. Multi-turn 에서 이전 turn 의 context 가 누적되어 intent 가 명확해지는 scenario 가 필요. τ²-bench 는 이 scenario 의 **community standard**.

**Motivation citation**: Laban et al. 2025 (`arXiv:2505.06120`, "LLMs Get Lost in Multi-Turn Conversation") 는 single-turn → multi-turn 에서 모든 top LLMs 에 **30-40% performance drop** 을 보고. 우리 method 는 이 gap 을 closing 하는 것이 목표.

### 요청 내용

**Task R2.A — τ²-bench clone + 환경 구축** (30분)

```bash
cd /home/woori/workspace_common/boltzmann-attention/external
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
source /home/woori/workspace_common/CDP/poc/set.env && pip install -e .

# Sanity: GPT-4 API 없이 로컬 모델로 retail 도메인 10 cases run 가능한지 확인
tau2 --help
```

**Task R2.B — τ²-bench catalog → facet ontology 포팅** (이건 mais 쪽에서 진행, coworker 는 대기)

mais 가 `scripts/ocq/build_tau2_ontology.py` 를 작성 후 coworker 에게 전달. 그 후:

```bash
# Build facet basis for τ²-bench retail + airline catalogs
python scripts/ocq/build_tau2_ontology.py \
    --domain retail \
    --out reports/axis2_theoretical_verification/tau2_retail_ontology.json

python scripts/ocq/build_tau2_b_ont.py \
    --model Qwen/Qwen2.5-7B \
    --ontology-json reports/axis2_theoretical_verification/tau2_retail_ontology.json \
    --out external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt
```

같은 방식으로 `airline` 도메인.

**Task R2.C — τ²-bench retail + airline 3-model full eval** (A100 × 4, **2-3일**)

3 모델 × 2 도메인 × methods 조합:

Methods (from `EXPERIMENT_PLAN_multi_turn_benchmarks_2026_04_10.md` §2):
- **A**: `no_steer`
- **B**: `ocq_bias_a0.3` (flat K-bias)
- **C**: `ocq_facet_gated_a1.0` (per-facet K-side gate)
- **D**: `ocq_q_gated_a1.0` (Q-gated score bonus, multi-turn focus) — **mais 가 구현 후 전달**
- **E**: `ocq_qk_gated_a1.0` (Q+K composition)
- **F**: `adaseka_3expert` — AdaSEKA baseline with {synthetic, counterfact, hotpot} experts or τ²-adapted experts
- **G**: `seka_vanilla`

**τ²-bench eval 실행 방법 (mais 가 `eval_tau2.py` 작성 후 전달)**:

```bash
python scripts/ocq/eval_tau2.py \
    --model Qwen/Qwen2.5-7B \
    --domain retail \
    --device cuda:0 \
    --methods no_steer ocq_bias_a0.3 ocq_facet_gated_a1.0 ocq_q_gated_a1.0 ocq_qk_gated_a1.0 \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt \
    --num-trials 1 \
    --out /tmp/tau2_retail_qwen25_7b.json
```

- 3 모델 × 2 도메인 × 5-7 methods = 30-42 configurations
- Per-config: τ²-bench retail 은 약 250 cases, dialogue 평균 8-10 turns → 단일 model 한 domain 기준 ~2-3시간
- A100 4장 병렬: **2-3일 전체 소요**

### R2 Deliverables

1. `/tmp/tau2_retail_{qwen25_7b,llama31_8b,mistral_7b_v03}.json` (3개)
2. `/tmp/tau2_airline_{qwen25_7b,llama31_8b,mistral_7b_v03}.json` (3개)
3. τ²-bench 도메인별 facet ontology + B_ont 파일
4. τ²-bench 실행 중 만난 이슈 (OOM, prompt format, API key 필요 여부, LLM-as-judge 사용 등) 기록

### R2 PASS/FAIL 판정

**Gate M1 (sanity, sample 50)**:
- PASS: method C 또는 D 가 method B (flat) 대비 ≥ 2pp lift in trajectory success rate
- FAIL: single-turn flat 이 multi-turn 에서도 최선 → main claim 무효

**Gate M2 (full τ²-bench, 3 models)**:
- PASS: method D 또는 E 가 ≥ 2/3 models 에서 flat 대비 ≥ 3pp lift
- FAIL: paper scope 축소 (single-turn only)

---

## 3. R3 — BFCL v3 Multi-Turn (NEW)

### 배경

**Community standard** for LLM tool calling. Berkeley Gorilla team, ICML 2025. v3 에서 multi-turn categories 도입 (multi_turn_base / miss_func / miss_param / long_context). AST 기반 evaluation.

**왜 필요한가**: Reviewer expectation — 모든 tool-use paper 가 cite. τ²-bench 가 "multi-turn quality" 를 보여준다면 BFCL 은 "community reproducibility" 를 보장.

### 요청 내용

**Task R3.A — BFCL clone + harness** (30분)

```bash
cd /home/woori/workspace_common/boltzmann-attention/external
git clone https://github.com/ShishirPatil/gorilla
cd gorilla/berkeley-function-call-leaderboard
pip install -e .

# Sanity: single-turn subset 10 cases local Qwen run
bfcl generate --test-category simple --model Qwen/Qwen2.5-7B
```

**Task R3.B — BFCL multi-turn eval (3 models × 5 methods)**

Multi-turn categories:
- `multi_turn_base` (~200 cases)
- `multi_turn_miss_func` (~100 cases)
- `multi_turn_miss_param` (~100 cases)
- `multi_turn_long_context` (~50 cases)

Methods: A (no_steer), B (flat α=0.3), C (facet-gated), D (Q-gated), E (Q+K composition)

**구현 notes**: BFCL v3 는 stateful (previous turn's tool output affects next turn). 우리 K-bias hook 은 tool selection 에 focus, state tracking 은 inherent 하게 모델이 처리. Eval 은 BFCL 의 AST correctness metric 그대로 사용.

**소요**: A100 4장 병렬, **1-2일**.

### R3 Deliverables

1. `/tmp/bfcl_multiturn_{base,miss_func,miss_param,long_context}_{qwen25_7b,llama31_8b,mistral_7b_v03}.json` (총 12개)
2. BFCL 의 AST-based score 요약 (method × category × model matrix)

### R3 PASS/FAIL 판정

- PASS: method D 또는 E 가 multi-turn 4 categories 중 최소 2개에서 flat 대비 ≥ 1pp lift (BFCL 는 전반적으로 lift 작음 예상)
- FAIL: BFCL 은 cite only, τ²-bench 결과로만 paper

---

## 4. R4 — MMLU Phase-Gating Check (NEW, Cor 6.7 empirical verification)

### 배경

**Corollary 6.7/6.8 의 empirical verification**. 우리 facet-gated operator 는 non-tool query (MMLU 문제) 에 대해 **intervention 이 사실상 zero** 여야 함 (Cor 6.7: `q ⊥ Range(B) ⇒ qaMSE = 0`). MMLU 는 factual QA 로 tool ontology subspace 와 직교에 가까우므로 우리 method 는 baseline 유지해야 함.

**반면 flat K-bias (method B) 는 phase-closure property 가 약함** — 모든 query 에 uniform 증폭 → MMLU 에서도 degradation 발생 가능. 이 비교가 우리 method 의 "safe for non-tool queries" 주장의 근거.

### 요청 내용

**Task R4.A — MMLU 1000 샘플 eval**

MMLU 5-shot 1000 샘플 (random 또는 high-confidence subset), 5 methods × 3 models = 15 configurations.

```bash
python scripts/ocq/eval_mmlu_subset.py \
    --model Qwen/Qwen2.5-7B \
    --device cuda:0 \
    --n-samples 1000 \
    --seed 42 \
    --methods no_steer ocq_bias_a0.3 ocq_facet_gated_a1.0 ocq_q_gated_a1.0 adaseka_3expert \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --out /tmp/mmlu_phase_gating_qwen25_7b.json
```

**주의**: `eval_mmlu_subset.py` 는 mais 쪽에서 작성 후 전달.

**소요**: A100 4장 병렬 (3 models 동시), **~2시간**. MMLU 는 단순 multiple-choice 라 multi-turn eval 대비 빠름.

### R4 Deliverables

1. `/tmp/mmlu_phase_gating_{qwen25_7b,llama31_8b,mistral_7b_v03}.json` (3개)

### R4 PASS/FAIL 판정

- **PASS**: method C (facet-gated) 와 D (Q-gated) 가 no_steer 대비 **|Δ MMLU| ≤ 0.5pp**
- **FAIL**: degradation ≥ 1pp → Cor 6.7 retraction or scope 축소

우리 paper 의 "phase-closure empirically validated" 주장의 유일한 근거.

---

## 5. 전체 우선순위 및 timeline

### 권장 순서 (gating 고려)

| 단계 | Task | 기간 | Blocker |
|---|---|---|---|
| **Phase 1** (Week 1) | R1 (Cross-model MetaTool) | ~3시간 A100 × 4 | 없음, 바로 시작 가능 |
| **Phase 2** (Week 1-2) | R4 (MMLU phase-gating) | ~2시간 × method 수 | mais 가 `eval_mmlu_subset.py` 전달 |
| **Phase 3** (Week 2-3) | R2 (τ²-bench) | 2-3일 | mais 가 `build_tau2_ontology.py` + `eval_tau2.py` 전달 |
| **Phase 4** (Week 3) | R3 (BFCL v3 multi-turn) | 1-2일 | Phase 3 완료 후 (경험 재활용) |

**총 소요**: Week 1-3, A100 × 4 활용 시 **5-7일**.

### 병렬화 권장

- R1 은 **즉시 시작** (기존 v1 요청, 모든 준비 완료)
- R4 는 R1 완료 후 **같은 model 위에서 추가 eval** 가능 (모델 load 재사용)
- R2 는 mais 의 τ²-bench 포팅 완료 후 시작 (Week 1 말)
- R3 는 R2 경험 재활용해 빠르게

### R1 우선 시작 (2026-04-10 밤)

R1 은 이미 v1 에서 준비 완료. **지금 바로 시작 가능**. 다른 task 는 mais 쪽 준비가 필요하므로 대기.

**따라서 R1 먼저 launch 하시면 됩니다.** R2/R3/R4 는 mais 가 준비되는 대로 추가 전달 예정.

---

## 6. 참고 자료 (coworker 용)

### 핵심 읽기 자료 (추천 순서)

1. **`reports/ADASEKA_COMPARISON_2026_04_10.md`** — 오늘 CounterFact 에서 AdaSEKA 실험 결과. 왜 AdaSEKA 가 약한지 empirical + structural.
2. **`reports/EXPERIMENT_PLAN_multi_turn_benchmarks_2026_04_10.md`** — 오늘 확정된 multi-turn benchmark plan. R2/R3 의 상세 설명.
3. **`memory/adaseka_vs_ours_differentiation_2026_04_10.md`** — 다음 세션을 위한 anchor. AdaSEKA vs 우리 차별점 4-axis table.
4. **`memory/existing_theorem_package_2026_04_10.md`** — 기존 Theorem 6.1/6.2/6.3 재활용 + Cor 6.7-6.12 새로 추가.
5. **`math/paper/lie_group/COROLLARY_6_7_FACET_PHASE_CLOSURE.md`** — 새 corollary 6개 (6.7-6.12) 전체 증명.
6. **`memory/metatool_subtask1_first_signal_2026_04_09.md`** — Qwen MetaTool 결과 + 진행 중 실험 status (중복 방지용).

### 코드 (develop branch)

- `scripts/ocq/build_qwen_metatool_b_ont.py` — B_ont builder (model-agnostic)
- `scripts/ocq/build_metatool_ontology.py` — MetaTool catalog → 4-facet ontology
- `scripts/ocq/eval_metatool_subtask1.py` — 기존 MetaTool eval driver
  - 새로 추가된 method: `ocq_facet_gated_a{α}` (per-facet K-side energy gate)
  - `install_facet_gated_hooks` + `build_facet_masks` 함수
- `scripts/ocq/quantizer.py` — OCQ quant modes (1a sign / 1b mean-split / 1c argmax) — **1a/1c 는 MetaTool 에서 catastrophic 확인됨**, paper 의 negative appendix
- `scripts/ocq/eval_hook_mode.py` — WT2 PPL eval (R1~R4 와 무관, FOKVQ paper 용)
- `external/SEKA/` — SEKA + AdaSEKA 코드, CounterFact + BiasBios 실험 완료

### 향후 추가 예정 (mais 가 작성)

- `scripts/ocq/build_tau2_ontology.py` (Week 1 말)
- `scripts/ocq/build_tau2_b_ont.py` (Week 1 말)
- `scripts/ocq/eval_tau2.py` (Week 2)
- `scripts/ocq/eval_multi_turn.py` (Week 2, BFCL + τ² 통합 driver)
- `scripts/ocq/eval_mmlu_subset.py` (Week 1)
- `install_q_gated_score_hooks` (Week 2, attention forward patch) — **가장 구현 복잡, Qwen/Llama/Mistral 3개 attention class 모두 monkey-patch 필요**

### 관련 외부 benchmark

- **τ²-bench**: https://github.com/sierra-research/tau2-bench (Sierra Research)
- **BFCL v3**: https://github.com/ShishirPatil/gorilla (Berkeley Gorilla)
- **ToolDial**: https://github.com/holi-lab/ToolDial (ICLR 2025, optional secondary)
- **Laban et al. "LLMs Get Lost"**: arXiv:2505.06120 (motivation, benchmark 아님)

### 원본 dataset

- MetaTool: `/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json`
- 기타 benchmark 는 clone 후 repo 안에

---

## 7. Paper core concept (한 문단)

Coworker 가 실험 결과를 해석할 때 참조할 우리 paper 의 핵심 방향:

> *현실 enterprise agent deployment — 수십~수백 개의 워크플로우, 계획, 도구 — 에서 catalog 구조로부터 **tens of orthogonal semantic facets** 을 자동 구성하고, 이를 **K 텐서에 반영** 하여 per-token **multiple focus dimensions 가 simultaneously** 활성화되도록 한다. Intervention 은 K-side, composition 은 additive (winner-take-most 가 아님), facet 수는 F = 10-100 이며 catalog-derived (benchmark contrastive data 불필요). Multi-turn 대화에서 context-integrated intent detection 이 특히 강력하다.*

**Against AdaSEKA**: Q-adaptive 1-of-M expert winner-take-most vs 우리 K-side F-simultaneous catalog-driven. Cor 6.9 (rank ceiling) + Cor 6.12 (unified failure mode) 가 이 차별점을 formally 증명.

**Against flat K-bias**: Cor 6.7/6.8 (phase-closure) 가 non-domain query 에서의 safety 를 증명. 플랫은 phase-closure 없음, 우리는 `O(ε_q)` linear 감소.

**Against 1a/1c hard selection (negative result)**: Cor 6.11 (hard selection penalty) 가 per-token hard selection 의 `((R-k)/R)²` 패널티를 설명. 1a (0.90%), 1c (1.41%) 는 theory 의 empirical verification.

이 6개 corollary (6.7-6.12) 는 모두 기존 Theorem 6.1/6.2/6.3 (이미 proven) 에만 의존. 총 ~2페이지 proof.

---

## 8. 질문 / blocker / 예상 이슈

### 예상 이슈 1 — τ²-bench 의 LLM-as-judge

τ²-bench 는 agent 와 user 모두 LLM simulation. 같은 model 을 양쪽에 쓰면 bias 가능.

**완화**: user simulator 는 **GPT-4o (API)** 또는 **별도 family (e.g., Claude)** 로 고정. Agent 쪽만 우리 method 적용. 이 분리가 τ²-bench standard protocol 에 부합.

**Coworker 에게 질문**: A100 rack 에서 **OpenAI / Anthropic API key 접근 가능한가**? 아니면 user simulator 로 Llama-70B 를 내부 host 해야 하나?

### 예상 이슈 2 — BFCL v3 stateful state tracking

BFCL v3 multi-turn 은 previous tool output 이 context 에 들어가고 다음 turn 의 input 이 됨. 우리 method 는 stateful tracking 책임지지 않음.

**완화**: BFCL 결과가 작게 나오는 것은 expected. Paper 에 "tool selection 에 focus, state tracking 은 orthogonal" 이라는 caveat.

### 예상 이슈 3 — Mistral 7B negative recurrence

Phase 1.x Mistral negative 이력 때문에 R1.B 결과가 negative 일 가능성.

**완화**: Mistral 결과가 negative 라도 paper abort 하지 않음. "Catalog-ontology 는 Mistral 의 attention structure 에 less aligned, facet basis 재조정 필요" 라는 honest observation 으로 보고.

### 예상 이슈 4 — 7B-8B 모델은 single A100 에 fit

A100 80GB × 1 로 충분. Memory 이슈 없음. 복수 GPU 는 **병렬 실행 용** (서로 다른 model 동시 run).

### 예상 이슈 5 — `eval_metatool_subtask1.py` 의 새 method `ocq_facet_gated_a*`

오늘 mais 가 추가. `install_facet_gated_hooks` 함수 + `build_facet_masks` 함수 + parser. Payload 에 `r_per_pair` dict 있어야 facet mask 빌드 가능 (B_ont 저장 시 자동 포함됨).

**Coworker 에게 확인**: `scripts/ocq/eval_metatool_subtask1.py` 최신 버전 pull 후 `ocq_facet_gated_a0.3` method 가 동작하는지 sanity check. 50 샘플 smoke 권장.

---

## 9. 즉시 행동 요약 (coworker view)

1. **지금 (2026-04-10 밤)**: R1 (Cross-model MetaTool) 시작 — 기존 v1 그대로, 변경 없음. 3시간 내 완료. **최우선**.
2. **R1 완료 직후**: mais 에게 결과 report (JSON 파일 경로 + 간단 요약). mais 가 판정.
3. **Week 1 말 (mais 준비 후)**: R2 (τ²-bench) 시작. mais 가 `build_tau2_*.py`, `eval_tau2.py` 전달.
4. **R2 완료 후**: R4 (MMLU phase-gating) 병렬 start.
5. **Week 2-3**: R3 (BFCL v3 multi-turn).

### Blocker 면 즉시 연락

- `build_qwen_metatool_b_ont.py` 가 Llama/Mistral 에서 shape error 나면 → mais 가 model-specific config 수정
- τ²-bench 환경 구축 중 Sierra repo 의존성 문제 → mais 가 확인
- BFCL v3 가 newer version 이라 format 변경됐으면 → mais 가 adapter 작성
- GPU memory 이슈 (multi-turn 대화는 long context) → batch size 1, gradient checkpointing

---

**질문이나 blocker 있으면 바로 연락 주세요.** R1 결과가 도착하는 대로 mais 가 next phase 준비 (τ²-bench 코드) 를 완료할 예정입니다.

**Document history**:
- v1 (2026-04-10 오후): Cross-model MetaTool R1 만.
- **v2 (2026-04-10 저녁)**: AdaSEKA 결과 추가, theorem 재발견 + Cor 6.7-6.12 추가, 1a/1c 실패 결과 추가, multi-turn plan 추가 (R2/R3/R4). v1 R1 은 변경 없음.

**Signed**: mais (Claude session), 2026-04-10 저녁
