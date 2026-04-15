# Coworker 실험 요청 — 2026-04-15 (revised v2)

**보낸 사람**: develop side (mais)
**날짜**: 2026-04-15 (v2 update with critical implementation rule)
**대상**: A100×4 보유 coworker
**우선순위 변경**: Baselines = **P0 (critical)**. Gemma/Scaling 은 P1 로 격하.

## ⚠️ 0. CRITICAL implementation rule (반드시 읽기)

**External baseline 알고리즘을 비교할 때는 반드시 기존 구현 (external/ 또는 official repo) 을 그대로 사용하시고, paper 만 보고 proxy 를 작성하지 마세요.**

### 이번 develop side 사고 (2026-04-15) — 같은 실수 반복 방지

- 한 일 (잘못): Cor 6.9 의 max-of-M routing 설명만 보고 `install_adaseka_proxy_hooks` 작성 — Q-side per-step softmax routing on B_ont
- 결과: F1=0.768 (+3.7pp 우리 Q-coverage 대비) 측정 → paper §5.5.3 에 "AdaSEKA proxy beats us" 로 commit
- 발견 (사용자 지적 후): `external/SEKA/src/model/{seka_llm,adaptive_seka_llm}.py` 의 진짜 AdaSEKA 는 **K-side** + per-query routing (last prompt token) + steer_mask 사용. 5 가지 차이:
  | 항목 | 진짜 AdaSEKA | proxy (틀림) |
  |---|---|---|
  | Hook side | K-side | Q-side |
  | Routing schedule | per-query (last token only) | per-step (모든 token) |
  | Token selection | steer_mask | 전체 |
  | Routing input | SVD U-matrix coefficients | B_ont 분할 energy |
  | Forward 횟수 | 2회 (routing + generation) | 1회 |

- 비용: ~4 시간 paper writing 무효, retraction commit (`c5e4f2f`) 필요. Reviewer 에게 발견되었으면 paper credibility 손상.

### 적용 — coworker 가 baselines 작성 시

**P0 8 baselines 별 구현 sourcing**:

| Baseline | 권장 sourcing |
|---|---|
| **SEKA** | `external/SEKA/src/model/seka_llm.py` (이미 있음) — `SEKALLM` class 직접 사용 |
| **AdaSEKA 2-expert / 3-expert** | `external/SEKA/src/model/adaptive_seka_llm.py` (이미 있음) — `AdaptiveSEKALLM` class 직접 사용 |
| **CAA** | https://github.com/nrimsky/CAA — clone 하여 `external/CAA/` |
| **ITI** | https://github.com/likenneth/honest_llama — clone 하여 `external/ITI/` |
| **PASTA** | https://github.com/QingruZhang/PASTA — clone 하여 `external/PASTA/` |
| **Focus Directions** | (Zhu et al. 2025) repo 검색 후 clone |
| **LoRA-tool-FT** | peft 표준 — 우리 `scripts/ocq/lora_train_metatool_v3.py` 참조 |
| **RAG-prompt** | LangChain 표준 retrieval — 직접 작성 가능 |

**proxy 작성 금지 원칙**:
- **Source 검색 먼저**: `find external/ -iname "*<method>*"` 와 GitHub 검색
- **Source 발견 시**: wrapper 만 작성 (input 형식 조정 + Subtask1/Subtask4 driver 와 통합)
- **Source 부재 시만 proxy**: 그러나 반드시 (a) original repo 가 진짜 없는지 확인, (b) paper 의 algorithm 명세 충분한지 검증, (c) `install_<method>_PAPER_PROXY_hooks` 로 라벨 + docstring 에 모든 추정 명시
- **Paper 텍스트 절대 금지**: "<method> proxy beats X" — *real implementation 이거나 우리 새 방법으로 라벨*

### Develop side 가 이미 작성 + 검증한 wrapper

- `scripts/ocq/eval_subtask4_with_real_seka.py` (NEW): real SEKA 사용 wrapper
  - B_ont (L,H,d,r=24) → P_pos (L,H,d,d) via P = B B^T 변환
  - User query 를 `**...**` markers 로 wrap (steer_mask 자동 생성)
  - SEKALLM.generate(steer=True) 호출
  - amplify_pos sweep
- coworker 는 SEKA / AdaSEKA 부분에서 이 script 패턴 재사용 가능

---

## 0. 핵심 한 줄 요청

> **CAA / ITI / PASTA / ASA / Focus Directions / AdaSEKA 2-3 expert / RAG-prompt / LoRA-tool-FT 8 개 baseline 을 MetaTool Subtask1 (N=995) + Subtask4 (N=497) 위에서 직접 비교 표로 만들어 주세요. 이것이 paper main-track 당락의 #1 결정 요인입니다.**

예상 GPU 비용: A100×1 에서 18 시간, A100×4 병렬화 시 5–6 시간.

---

## 1. 2026-04-15 develop side 진행 상황 요약

### 1.1 검증된 결과 (paper main claims)

| Item | 결과 | Status |
|---|---|---|
| **Cor 6.9.6 stability** (Subtask4 N=497) | real F1=0.685 vs random/featshuffle F1=0.000 → **+68.5pp gap** | ✅ |
| Thm 6.1 per-sample bound (Qwen L13, 2800 samples) | pass_rate **1.00**, median LHS/RHS 2.36×10⁻⁸ | ✅ |
| Cor 6.9 operator nrank | ours 24.0 vs AdaSEKA 7.44, **gap +17** | ✅ |
| Cross-model Subtask1 sum (full 995) | Qwen +0.10 / Llama-Base +6.33 / Mistral-Base +3.12 | ✅ |
| Direction specificity (Subtask1 sum) | Qwen gap +48.84 / Llama-Base +7.33 | ✅ |
| Thm 6.13 WT2 2-bit | OCQ 15.60 < KIVI 19.97 (−4.37 PPL, 9.4% fewer bits) | ✅ |
| **Q-coverage Subtask4 +1.6pp lift** (full 497, NEW today) | F1 0.731 → 0.747 at β=−0.1 | ✅ NEW |
| R6 MMLU flat α=0.2 | +1.4pp over baseline 0.713 | ✅ |

### 1.2 Falsified / weak (honestly reported)

- **Cor 6.9 원래 multi-tool accuracy-lift 예측**: full 497 에서 −4.6pp → reframed as "stability" 방향 (해결됨)
- **Contrastive K-bias (smoke +5.8pp)**: full 497 에서 −3.6pp → small-N artifact 로 판명, §5.5.2 retracted
- **LoRA v1 (Cor 6.16.1)**: F1 0.533 — 학습/평가 token 분포 mismatch (plain text vs chat-template). v2 진행됨.
- **LoRA v2** (chat-template fix): F1 0.219 — single-tool training bias 로 multi-tool 능력 손상. v3 (synthetic multi-tool) 진행 중.
- **Mistral-Instruct sum**: −2.92pp (chat-template hedging artifact, scope 한계로 명시)

### 1.3 새 정리 (Option C 통합 채택, 2026-04-15)

논문이 "steering paper" 에서 "**steering + KV-compression Pareto 통합**" 으로 확장:

- **Cor 6.9.6** (신규): on/off-manifold KL bound, $\mathrm{span}(B_{\mathrm{ont}})$ 가 $\alpha^2$ 안정성을 가진 유일 부분공간
- **Thm 6.17** (신규): QKV-joint coverage-aware steering first-order optimality
  - Refined Thm 6.17′ (실측 검증): small-α regime 만 first-order separable, $\alpha_{\mathrm{coupling}} \approx 0.1$
- **Thm 6.18** (신규): attention-weighted bit allocation $b^*(t,f) = \tfrac12 \log_2(\lambda^* \pi(t,f) \sigma_f^2)_+$
- **Thm 6.19** (신규): joint Pareto optimality — 동일 $B_{\mathrm{ont}}$ 가 steering + compression 양쪽 Pareto-optimal

새 paper 제목: "**A Uniquely Privileged Subspace: Joint Pareto-Optimality of Ontology-Based Steering and KV-Cache Compression in Instruction-Tuned Transformers**"

### 1.4 현재 진행 중 (develop side autonomous)

| 작업 | 위치 | 예상 종료 |
|---|---|---|
| GPU0 PM wave 1: Llama Subtask4 full 497 + Llama Subtask1 Q-coverage | PID 1907195 | ~13:00 KST |
| GPU1 PM wave 1: Q-bias extended sweep + null-control + Mistral/Llama smoke + R6 | PID 1907194 | ~14:00 KST |
| PM wave 2 (대기 중): LoRA v3 학습 + L3 평가, K×Q small-α joint 검증, V-only full | PID 1978440 | ~18:00 KST |

---

## 2. Coworker 요청 — 갱신된 우선순위

### 🔴 P0 (CRITICAL — 즉시) — Baselines 직접 비교표

이전 v4 의 Track D 를 P0 로 격상. 이유: paper 의 모든 주요 claim (Cor 6.9.6 stability, Thm 6.17 accuracy lift) 이 비교 baseline 없이는 reviewer 가 "ASA / Focus Directions 보다 좋은가?" 를 물었을 때 답 못함.

**8 개 method × 2 dataset = 16 cells. A100×4 병렬화 시 5–6h 총합.**

#### Track A — 8 baselines on Subtask1 (N=995)
모든 method 동일 protocol: Qwen2.5-7B-Instruct, label_logprob sum scorer, max_new_tokens=32.

⚠️ **Source-first 정책**: 각 method 의 *source* 컬럼 보고, 가능하면 original code wrapper 사용. proxy 는 source 부재 시만, label 명시.

| Method | Source | 핵심 hyperparams (correction!) | 비고 |
|---|---|---|---|
| **SEKA** (Feng et al. 2025) | ✅ `external/SEKA/src/model/seka_llm.py` | layers=last10, amplify_pos=1-5 sweep, P_pos = B_ont @ B_ont.T | **K-side** + steer_mask. wrapper: `eval_subtask4_with_real_seka.py` 패턴 |
| **AdaSEKA** M=2/3 (Kim et al. 2026) | ✅ `external/SEKA/src/model/adaptive_seka_llm.py` | combination_method='weighted_top_k', top_k=24, T sweep | **K-side** + per-query routing. **NOT Q-side max-norm** (source 확인) |
| **CAA** (Rimsky 2024) | 🔴 clone https://github.com/nrimsky/CAA → `external/CAA/` | α=2.0, layer 16 (residual stream rank-1 bias) | 우리 `install_caa_hooks` 는 *B_ont 1st col* — original 은 contrastive paired-data 학습. **다른 알고리즘**, 별도 비교 필요 |
| **ITI** (Li et al. 2023) | 🔴 clone https://github.com/likenneth/honest_llama → `external/ITI/` | α=15, top-48 heads, mean diff | per-head Q-side intervention |
| **PASTA** (Zhang et al. 2023) | 🔴 clone https://github.com/QingruZhang/PASTA → `external/PASTA/` | α=0.01, all heads, attention bias | attention map 직접 수정 |
| **Focus Directions** (Zhu et al. 2025) | 🔴 GitHub 검색 후 clone | α=3.0, top-3 directions | Q-side rank-3 |
| **LoRA-tool-FT** | ✅ peft 표준 + 우리 `scripts/ocq/lora_train_metatool_v3.py` 패턴 | r=16, q/k/v/o/up/down_proj, mixed single+synthetic 2-tool | 우리 v3 결과 (F1 0.333) 와 비교용 |
| **RAG-prompt** | LangChain retrieval (직접 작성 OK) | top-k 후보 retrieval + prompt injection | text-only baseline |
| (참고: Q-coverage β=−0.1) | 우리 코드 | B_ont rank-24 Q-side 사영 빼기 | (자체 실행, 비교용) |
| (참고: K-bias α=0.3) | 우리 코드 | B_ont rank-24 K 사영 더하기 | (자체 실행, 비교용) |

**🔴 표시된 baselines**: source repo clone 작업이 baseline 평가 자체보다 시간 더 걸릴 수 있음. 추정:
- SEKA + AdaSEKA: source 이미 있음 → 1 셀당 ~1h
- CAA / ITI / PASTA / Focus: clone + integration ~2h × 4 + eval ~1h × 4 = ~12h
- LoRA + RAG: 직접 작성 + eval ~2h × 2 = ~4h

→ A100×4 병렬화 시 ~6h wall-clock 가능 (각 트랙 별도 설치).

#### Track B — 같은 8 baselines on Subtask4 (N=497, multi-tool)
동일 method 들. **multi-tool 가능한 방법** (AdaSEKA 2-3 expert) 만 multi-tool emission 가능; 나머지는 single-tool only — 이게 paper 의 핵심 주장 ("Q-side rank-1 routing 은 multi-tool emission 불가") 의 결정적 evidence.

#### 결과 보고 형식
```json
{
  "method": "CAA",
  "model": "Qwen2.5-7B-Instruct",
  "dataset": "Subtask1",
  "n_queries": 995,
  "scorer": "label_logprob_sum",
  "no_steer_top1": 0.5246,
  "method_top1": 0.XX,
  "delta_pp": +X.XX,
  "hyperparams": {...}
}
```

`reports/baselines/{method}_{dataset}.json` 으로 저장 + push.

#### 코드 시작점
- `scripts/ocq/eval_metatool_subtask1.py` 의 `install_kbias_hooks` 패턴 참조
- 각 baseline 에 대한 hook function 신규 작성: `install_caa_hooks`, `install_iti_hooks`, ... (memory `baseline_recipes_attention_steering` 에 정확한 layer/α 명시됨)
- `scripts/ocq/eval_metatool_subtask4.py` 의 `run_method` 분기 추가

---

### 🟡 P1 — Scaling curve (이전 P3, 동일 priority 유지)

Track C — Qwen2.5-{0.5, 3, 14, 32}B-Instruct on Subtask4 N=497 + Subtask1 full 995, **K-bias α=0.3 + Q-coverage β=−0.1** 만 (하루 총합 ~15 GPU-h).

목적: 본 method 의 scale-invariance 검증. Subtask4 stability gap (+68.5pp) 이 모델 크기 무관한지, Q-coverage lift 가 모든 사이즈에서 보이는지.

#### B_ont 빌드 필요
- Qwen2.5-0.5B/3B/14B-Instruct 위에서 `scripts/ocq/build_qwen_metatool_b_ont.py` 실행 (메모리 ~1h each on A100)
- 32B 는 별도 (Track A 의 Gemma 와 비슷한 메모리 부담)

#### 결과 보고
- `reports/scaling/qwen{0.5,3,7,14,32}b_st4_full497.json`
- `reports/scaling/qwen{0.5,3,7,14,32}b_st1_full995.json`

---

### 🔵 P2 — Gemma-3-27b-it (이전 P1, deprioritized)

이전 v4 Track A 그대로. **HuggingFace gated 승인 받은 후에만**.

이유: develop side 는 8B/7B 위주 결과로도 main-track 가능; Gemma 27B 는 "deployment-scale" 보조 evidence 일 뿐 main claim 의존성 없음.

#### Gemma 우선순위 낮춘 이유
- 오늘 Q-coverage +1.6pp 가 Cor 6.9.6 stability 를 보강 → main contribution 이 deployment scale 의존성 약해짐
- 27B 는 baseline / scaling 후 여유 시간에

---

### ⚪ P3 — develop side 가 자체 처리할 항목 (요청 X)

다음 항목은 develop side 가 처리하므로 coworker 부하 X:
- LoRA v3 (synthetic multi-tool), L3 평가
- Q-coverage cross-model on Llama (PM wave 진행 중)
- Step-adaptive Q-coverage 구현 (Thm 6.17 full version) — 향후
- Thm 6.18 attention-weighted quantizer 구현 — 향후
- 한글 paper 동기, memory 갱신

---

## 3. 실행 가이드

### 3.1 시작 전 준비 (develop 에서 push 필요)

| 파일 | 상태 | 비고 |
|---|---|---|
| `scripts/ocq/eval_metatool_subtask1.py` | ✅ pushed | install_q_bias_hooks, install_qkv_joint_hooks 추가됨 |
| `scripts/ocq/eval_metatool_subtask4.py` | ✅ pushed | qbias / qkv_joint 분기 추가됨 |
| `scripts/ocq/lora_train_metatool_v2.py`, `v3.py` | ✅ pushed | 참조용 |
| memory `baseline_recipes_attention_steering.md` | ✅ already in `.claude/.../memory/` | 정확한 hyperparams |
| `scripts/ocq/install_caa_hooks` (CAA hook) | ❌ TO WRITE | coworker 작성 |
| `scripts/ocq/install_iti_hooks` (ITI hook) | ❌ TO WRITE | coworker 작성 |
| `scripts/ocq/install_pasta_hooks` (PASTA hook) | ❌ TO WRITE | coworker 작성 |
| `scripts/ocq/install_asa_hooks` (ASA hook) | ❌ TO WRITE | coworker 작성 |
| `scripts/ocq/install_focusdir_hooks` (Focus Directions hook) | ❌ TO WRITE | coworker 작성 |

추정: hook 작성에 method 당 ~1h, total 5h. 그 후 평가 8 method × 2 dataset × ~30min/cell = 8h. **A100×4 병렬 → 총 wall-clock ~5h**.

### 3.2 Develop side 와의 동기

- 결과 push: `reports/baselines/`, `reports/scaling/`
- Develop 에서 매 6시간마다 git pull 하여 결과 통합
- 긴급한 결정 (e.g., baselines 가 모두 우리보다 좋다면 paper framing 수정 필요) 은 Slack 으로 즉시 보고

### 3.3 결과의 paper 위치

| Coworker 결과 | Paper 섹션 |
|---|---|
| Track A baselines on Subtask1 | §5.4 새 표 (현재 grid 옆 baselines 컬럼 추가) |
| Track B baselines on Subtask4 | §5.5 새 표 (cross-method 비교) |
| Track C scaling | §5.10 E7 (현재 placeholder) |
| Track D Gemma | §5.10 E10 (조건부 추가) |

---

## 4. 점수 영향 (paper main-track 확률)

현재 develop side 추정:
- 현 시점 NeurIPS 2026 main-track: **45–55%**
- + Track A baselines 완료: **+10–15% → 60–70%**
- + Track A + B + C 완료: **+15–20% → 65–75%**
- + Track A + B + C + Gemma: **+5% → 70–80%**

**즉 Track A 단독으로 가장 큰 score boost.** 이것이 P0 인 이유.

---

## 5. 질문 / 차단 요인

차단 / 모호한 부분 발견 시 즉시 회신:
- Hyperparameter ambiguity (예: ASA 의 T 가 0.05 인지 0.1 인지)
- B_ont path 부재 (scaling 시 0.5B/3B/14B 위 B_ont 빌드 우선)
- Compute 부족 (Gemma 가 27B 라 메모리 빠듯)

---

## 6. 시간선

| 시점 | 마일스톤 |
|---|---|
| **D-day = 2026-04-15** | (오늘) coworker 시작 |
| D+1 = 04-16 | Hooks 작성 + 첫 셀 검증 |
| D+2 = 04-17 | Track A Subtask1 8 cells 완료 |
| D+3 = 04-18 | Track B Subtask4 8 cells 완료 → **paper §5.4-5.5 baselines 표 완성** |
| D+5 = 04-20 | Track C scaling 완료 |
| D+7 = 04-22 | Track D Gemma (승인 시) |
| 2026-05-15 | NeurIPS 2026 deadline |

---

## 7. 한 줄 요약

**Track A baselines (CAA/ITI/PASTA/ASA/Focus/AdaSEKA/RAG/LoRA on Subtask1+4)** 가 paper main-track 당락의 #1 변수입니다. A100×4 로 5–6 시간이면 가능. 이걸 우선 실행해 주시면 develop side 는 LoRA v3 + step-adaptive Q-coverage + Thm 6.18 implementation 에 집중하겠습니다.

질문 있으시면 언제든.
