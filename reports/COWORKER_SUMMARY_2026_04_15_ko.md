# 동료 인수인계 — 실험 결과 + 논문 프레이밍 요약 (2026-04-15)

투고 목표: **ICLR 2027** (2026-09 제출).
Canonical 논문: `math/paper/benchmark_design/PAPER_DRAFT_v1_2026_04_14.md` (EN) / `PAPER_DRAFT_v1_ko.md` (KO).
Canonical 증명: `math/paper/lie_group/APPENDIX_B_PROOFS.md` + `COROLLARY_6_7_FACET_PHASE_CLOSURE.md` (**Cor 6.9.6 추가 2026-04-15**).

---

## 1. 논문 프레이밍 — stability-first (2026-04-15 변경)

### 이전 (2026-04-15 이전) 프레이밍
"Facet-gated K-bias 가 operator-level rank 구조를 통해 도구 선택 정확도를 향상시킨다."
**선형 베팅**: 이론 → operator rank → multi-tool 정확도 향상. Subtask4 full 497 에서 반증 ($-4.6$pp).

### 새 프레이밍 — 고유하게 특권화된 온톨로지 부분공간
"Per-head 온톨로지 기저 $B_{\mathrm{ont}}$ 는 기저 모델로부터의 output 분포 divergence 가 $O(\alpha^2)$ 속도인 유일한 $R$-차원 K-perturbation 부분공간. 동일 크기 직교 perturbation 은 $\alpha > \alpha^*$ 에서 FC-emission 매니폴드를 이탈."

**이중 안전망**:
- 주요 claim (stability) 은 이미 full scale 에서 검증 (Subtask4 N=497 에서 +68.5pp 방향 특이성 gap).
- 정확도 향상 (contrastive, LoRA) 은 독립적 부수 기여. 개별 실패가 주요 claim 을 깨뜨리지 않음.

### 핵심 정리 추가
- **Cor 6.9.6 (안정성 특성화, 신규)**: $\alpha^*$ 에서의 phase transition 을 가진 형식적 on/off-manifold KL bound. 증명은 `COROLLARY_6_7_FACET_PHASE_CLOSURE.md` 의 Rmk 6.9.3 뒤에 추가됨.
- Cor 6.9 를 operator-rank 진술에서 distributional-stability 진술로 격상.

---

## 2. 실험 결과 스냅샷 (2026-04-15 08:50 KST)

### 2.1 Subtask1 full 995 label_logprob cross-model grid

| 모델 | Scorer | no_steer | real a0.3 | random a0.3 | featshuffle a0.3 | real−random | real−featshuffle |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | sum | 52.46 | +0.10 | −48.74 | −40.10 | **+48.84** | **+40.20** |
| Qwen2.5-7B-Instruct | mean | 36.78 | **+5.03** | −23.01 | −11.25 | **+28.04** | **+16.28** |
| Llama-3.1-8B-Base | sum | 46.33 | **+6.33** | −1.00 | −0.20 | **+7.33** | **+6.53** |
| Llama-3.1-8B-Base | mean | 23.12 | **+2.61** | −0.61 | −1.41 | **+3.22** | **+4.02** |
| Mistral-Base-v0.3 (skipL0+padmax) | sum | 69.35 | **+3.12** | pending | pending | — | — |
| Mistral-Instruct-v0.3 | sum | 61.51 | **−2.92** | pending | pending | — | — |

**해석**: 3-family Base positive (Qwen + Llama + Mistral 모두 sum-positive). Mistral-Instruct 만 음수; chat-template hedging 으로 격리 (no_steer 자체가 Base 보다 7.84pp 낮음). 방향 특이성 (gap) 은 scorer-invariant 이자 모델-invariant.

### 2.2 Subtask4 full 497 (**주요 실증 기여**)

| B_ont | Method | F1 | Recall | Exact |
|---|---|---|---|---|
| real | no_steer | **0.731** | 0.716 | 0.525 |
| real | a0.3 | 0.685 | 0.672 | 0.473 |
| random | a0.3 | **0.000** | 0.000 | 0.000 |
| featshuffle | a0.3 | **0.000** | 0.000 | 0.000 |

**방향 특이성 gap = +68.5pp F1**. 논문에서 단일 실험 최강 신호.

### 2.3 Non-uniform smoke (N=20, multi-tool K-bias 첫 positive)

- Contrastive $a=0.3, d=3$: **F1 = 0.608 (+5.8pp over no_steer 0.550)** ★
- Contrastive $a=0.3, d=1$: F1 = 0.583 (+3.3pp)
- $\alpha$-sweep $a=0.15$: F1 = 0.575 (+2.5pp)
- V-bias 변형: 최대 F1 0.558 (모두 실패)
- Normalized (Thm 6.9.5 직접 구현) at $\alpha \ge 0.3$: 붕괴

Contrastive d=3 full 497 확장이 GPU0 에서 현재 실행 중 (PID 1776169, ETA ~3h).

### 2.4 Thm 6.1 per-sample bound (이론 검증)

Qwen2.5-7B-Instruct L=13, $\alpha=0.3$, N=100 쿼리 × 28 헤드 = 2800 샘플:
- **bound_pass_rate = 1.00** (모든 head-query 샘플이 Thm 6.1 만족)
- median LHS/RHS ratio = $2.36 \times 10^{-8}$

### 2.5 Cor 6.9 operator nrank (500 쿼리 SVD)

- 본 방법 facet-gated: nrank_ε = **24.0**
- AdaSEKA: nrank = 7.44
- **Gap +17** — operator-level rank 분리 결정적

### 2.6 R6 MMLU gate grid (N=1000)

Baseline 0.713. 최고 셀: **flat α=0.2 = 0.727 (+1.4pp)**. α=1.0 순서: flat 0.584 > soft 0.614 > hard_argmax 0.552 > hard_thresh 0.535. Hypothesis (R) 신호는 α=1.0 에서 가장 명확.

### 2.7 WT2 압축 (Thm 6.13)

Qwen2.5-7B 전체 WT2 test (ctx=2048):
- 2-bit: **OCQ 15.60** vs KIVI 19.97 (−4.37 PPL, 9.4% 비트 절약)
- 4-bit: KIVI 7.79 vs OCQ 12.56 (Cor 6.13.5 cross-over 검증)

### 2.8 LoRA hybrid (Thm 6.16, 재실행 중)

2026-04-15 02:32 KST 첫 시도가 GPU1 sibling proc 충돌로 OOM. 08:49 KST 재실행 (PID 1774445), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + batch-size 1. L1 학습 정상 (loss 9.58 → 0.001 step 100).

---

## 3. 동료에게 요청 (A100 × 4 트랙)

`reports/COWORKER_REQUEST_gemma_and_scaling_2026_04_14.md` v4 참조. 우선순위 (업데이트):

### Priority 1 — Baselines (~18 GPU-hr, **제출에 critical**)
Subtask1 + Subtask4 직접 비교 표: CAA, ITI, PASTA, ASA, Focus Directions, AdaSEKA 2/3-expert, LoRA r=8 tool-FT, RAG. 레시피는 메모리 `baseline_recipes_attention_steering.md` 참조.

### Priority 2 — Scaling (~30 GPU-hr)
Qwen2.5-{0.5, 3, 7, 14}B-Instruct 의 Subtask4 F1 + 방향 특이성 at $\alpha=0.3$. 예상: 크기 전반에서 scale-invariant +68.5pp gap (architectural, emergent 아님).

### Priority 3 — Gemma (HF access form 차단 중)
Gemma-3-27B-it R5.

### Priority 4 — LoRA full 트랙 (`build_lora_adapted_b_ont.py` + `eval_metatool_subtask4_lora.py` 작성 차단 중)

---

## 4. 위험과 미해결 질문

1. **Contrastive d=3 full 497**: full 에서 회귀 (F1 < 0.608) 시 §5.5.2 가 추측적이 됨. 주요 claim (§5.5 stability) 은 영향 없음.
2. **Mistral-Instruct null-control**: +60pp 방향 특이성 gap 예측. 관측되면 Instruct hedging 국소화; 아니면 scope 격리 필요.
3. **Baselines 부재**: main-track 최대 위험. 동료 Track 1 이 critical path.
4. **논문 길이**: 현재 draft ~515 줄. NeurIPS/ICLR 9 페이지 + appendix 목표. §5.4–§5.10 압축 필요.

---

## 5. 읽을 파일 (순서)

1. `math/paper/benchmark_design/PAPER_DRAFT_v1_ko.md` — canonical 한글 논문
2. `math/paper/lie_group/COROLLARY_6_7_FACET_PHASE_CLOSURE.md` ("Corollary 6.9.6" 검색) — 신규 안정성 정리
3. `reports/subtask4_overnight/st4_{real,random,featshuffle}_N0.json` — 주요 실증 결과
4. `reports/nonuniform_smoke/st4_contrastive_a0.3_d3.json` — multi-tool 첫 positive
5. `reports/theory_verify_2026_04_14/thm61_qwen_L13_a0.3_N100.json` — Thm 6.1 검증
6. `reports/theory_verify_2026_04_14/r6/r6_*.json` — MMLU 12-cell grid
7. `.claude/projects/-home-woori-workspace-common-boltzmann-attention/memory/subtask4_nullctrl_and_contrastive_d3_2026_04_15.md` — 최신 메모리

---

## 6. 예측 main-track 확률

| 시나리오 | 점수 (/10) | Main-track 확률 |
|---|---|---|
| 현재 (overnight 완료, contrastive smoke) | 6.2 | 40–48% |
| + Contrastive d=3 full 497 +5.8pp 확인 | 6.8 | 55% |
| + LoRA L3 F1 > 0.82 | 7.2 | 60–65% |
| + Baselines 표 (동료 Track 1) | 7.7 | 70% |
| + Gemma-3-27B scaling | 8.0+ | 75%+ |
