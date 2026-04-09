# K-Bias Attention Steering 실험 계획서

**가제**: *Ontology-Guided K-Side Attention Bias: Focus Shifting without Fabrication*
**작성일**: 2026-04-09
**상태**: 계획 단계 (코드 작성 전)
**OISA 와의 관계**: 독립된 standalone 논문. 부록에 "OISA 와의 융합 가능성" 절을 포함하지만 주 framing 은 아님.

---

## 0. 코드 작성 전 반드시 충족해야 할 전제

0.1. **세 논문을 abstract 만 보지 말고 전체 읽을 것**
  - Zhu et al. 2025, "Focus Directions Make Your Language Models Pay More Attention to Relevant Contexts" — arXiv:2503.23306
  - Wang et al. 2026, "ASA: Activation Steering for Tool-Calling" — arXiv:2602.04935
  - Zhang et al. 2023, PASTA — arXiv:2311.02262

  이 셋은 직접적인 prior art 이다. 본 계획의 모든 설계 결정은 "이미 X 가 한 것 아닌가" 라는 reviewer 질문 (X 는 위 셋 중 하나) 에 답할 수 있어야 한다.

0.2. **GGB_experiment 의 η=0 padding artifact 를 먼저 고치고** 12/128 effective facet span 을 재측정할 것. "특정 의미 축에 대한 효과적 제어" 라는 재해석은 이 재측정 수치에 의존한다. Fix 이전 수치는 논문에 인용할 수 없다.

0.3. **비교 baseline 목록을 미리 확정할 것** (§5). 하드웨어 예산으로 인해 어떤 baseline 을 빼야 한다면 시작 전에 문서화하라 — 조용히 건너뛰지 말 것.

---

## 1. Prior art 대비 positioning

### 1.1 확실히 새로운 것 (방어 가능한 novelty)

2026-04-09 문헌 조사 후, 다음 세 가지 claim 은 방어 가능하다:

- **Ontology 에서 유도한 intervention direction.** 지금까지 발견된 모든 attention-level steering 방법은 direction 을 (a) gold response 에 대한 gradient training (Zhu 2025), (b) contrastive activation pair (CAA, ITI, SADI), (c) classifier gradient 학습 (GAME), (d) SAE feature (Golden Gate), 또는 (e) 외부 fact KB (Fact Grounded Attention) 로부터 유도한다. **사전 정의된 의미 ontology/taxonomy 를 attention-level steering basis 로 사용한 논문은 존재하지 않는다.**
- **K-only bias 와 K+Q, Q-only ablation.** Zhu 2025 는 K 와 Q 에 대칭적으로 bias 를 준다. K vs Q vs K+Q 를 ablation 하고 그 결과를 fact-preservation metric 과 연결한 논문은 아직 없다.
- **Focus-shift 방법 (PASTA, Focus Directions, K-bias) 과 content-injection 방법 (CAA, ITI, SAE clamping) 사이의 공식적인 matched-effect Pareto 곡선, fact-preservation metric 기반.** 부분 조각들은 있다 (PASTA 가 attention reweight 을 함, SteeringControl 이 side-effect 측정, SAE-TS 가 matched-effect 비교) 그러나 본 논문이 그려야 할 곡선을 직접 그린 논문은 없다.

### 1.2 새롭지 않은 것 (인용 필수, claim 금지)

- "Residual stream 대신 attention activation 을 bias" — PASTA, GUIDE, InstABoost, Spotlight, Focus Directions 가 먼저 함
- "K activation 에 bias vector 추가" — Focus Directions 가 K+Q 로 이미 함
- "Attention steering 의 focus-shift 해석" — PASTA (2023) 에 implicit 함
- "Tool selection 에 activation steering 적용" — ASA (2026)
- "외부 구조적 지식 → attention-level bias" — Fact Grounded Attention (2509.25252) 이 factual KB 로 함
- "Fine-tuning 없이 steering" — 위 survey 에 인용된 모든 논문

### 1.3 한 줄 thesis

> **Intervention direction 이 외부 의미 ontology 에서 유도되고 bias 가 K-side (K+Q 가 아니라, residual 이 아니라) 에 적용될 때, attention steering 은 동등한 steering effect 강도에서 content-injection 방법 (CAA, ITI, SAE clamping) 이 달성할 수 없는 수준으로 context fact 와 non-target knowledge 를 보존한다.**

이 thesis 는 Phase 3 에서 반증 가능하다. Pareto 곡선이 분리되지 않으면 논문은 negative-result tech report 가 된다.

---

## 2. Thesis 를 testable hypothesis 로 분해

| H# | 가설 | 측정 방법 | 실패 시 논문 사망? |
|---|---|---|---|
| H1 | K-only bias 가 non-destructive 강도에서 target behavior 에 non-trivial effect 를 낸다 | Target concept probe accuracy ↑ at β, where MMLU Δ ≤ 1% | 예 (gate) |
| H2 | K-only bias 가 matched-effect CAA 보다 held-out neighbourhood fact 를 더 잘 보존한다 | CounterFact specificity (ROME protocol), matched effect magnitude 기준 | 예 (핵심 claim) |
| H3 | K-only bias 가 bias direction 에 반하는 context fact 를 더 잘 보존한다 | Context-memory override rate (Yu et al. 2023 protocol) | 예 (핵심 claim) |
| H4 | K-only 가 matched effect 에서 K+Q bias 와 Pareto-competitive 하다 | Ablation cell: K-only vs Q-only vs K+Q 에서 H1–H3 | 아니오 (약화하지만 사망 아님) |
| H5 | Ontology direction 이 gradient-trained direction 과 동등하거나 더 나은 target accuracy 를 낸다 | Matched β 에서 effect-vs-budget 비교 | 아니오 (negative 도 흥미: "ontology 가 더 싸다") |
| H6 | Multi-layer schedule 이 total budget 고정 시 single-layer 보다 낫다 | Layer schedule sweep | 아니오 |
| H7 | 두 facet 의 동시 K-bias 가 interference 없이 additive 효과를 낸다 | Dual-facet probe accuracy vs single-facet | 아니오 |
| H8 | Homonymous tool 이 있는 실제 tool-selection benchmark 에서 K-bias 가 tool-metadata fabrication 없이 selection accuracy 를 개선한다 | BFCL V4 hallucination metric + MetaTool overlap subtask | 논문 생존 여부는 아니오, 응용 framing 은 예 |

**H1, H2, H3 가 모두 통과하면 논문 진행.** 나머지는 supporting 구조.

---

## 3. 실험 단계 (gated)

### Phase 0 — 인프라 (3–5 일)

- P0.1. GGB_experiment pipeline 의 padding 수정. Effective facet span 재측정.
- P0.2. K-bias hook 구현: 선택한 layer 들의 `k_proj` 출력에 forward pre-hook 을 걸어 `Σ_f β_f · v_f` 를 더한다. `{v_f}` 는 ontology basis.
- P0.3. Ablation 용 Q-bias, K+Q bias hook 구현.
- P0.4. Baseline 최소 재현: CAA, ITI, PASTA, Focus Directions (K+Q). Full-paper 재현 아니라 headline number 재현.
- P0.5. Metric harness: KL-on-benign (Stickland), CounterFact specificity (ROME), context-memory override, target-concept probe, MMLU, MT-Bench subset.
- P0.6. 모델: Llama-3-8B-Instruct (primary), Qwen2.5-1.5B-Instruct (ASA 비교용), Mistral-7B-Instruct (tertiary, 기존 GGB_experiment 데이터 재활용).

**Phase 0 exit criterion**: 모든 baseline 이 원 논문의 headline number 를 ±2pp 이내로 재현. 재현 실패 시 진행 금지.

### Phase 1 — 단일 facet 에 대한 K vs Q vs K+Q ablation (3–4 일)

Ontology 의 단일 facet (예: "landmark" 또는 "product category"). 단일 layer → 전체 layer 순서.

- P1.1. β ∈ {0.5, 1, 2, 4, 8, 16} sweep, 세 조건 (K-only, Q-only, K+Q).
- P1.2. 각 (조건, β, layer) 에 대해 target probe accuracy, MMLU delta, KL-on-benign 측정.
- P1.3. K, Q, K+Q 에 대한 Pareto 곡선 (effect vs side-effect) 작성.

**Exit criterion (H1, H4)**: K-only 가 MMLU delta ≤ 1% 인 β 에서 target-concept probe 를 baseline 대비 ≥ 20pp 개선해야 한다. 불가능하면 K-only viable 여부 재고 — 첫 번째 go/no-go gate.

### Phase 2 — Ontology vs gradient-trained direction (3–4 일)

- P2.1. 동일 facet 에 대해 direction 을 세 방식으로 유도:
  - (a) **Ontology projection**: 기존 taxonomy label embedding 을 해당 layer 의 K-space 에 투영
  - (b) **Gradient-trained** (Focus Directions 레시피): target example 의 cached activation 에 AdamW 로 `d_K` 학습
  - (c) **Contrastive mean-diff** (CAA 레시피를 K-space 에 적용): positive example 의 K 평균 − negative example 의 K 평균
- P2.2. Target probe 상에서 matched effect magnitude 기준으로 KL-on-benign, CounterFact specificity, context-memory override 측정.

**Exit criterion (H5)**: Ontology direction 이 target 에 대해 gradient-trained 의 90% 이내 성능 유지, side-effect metric 중 하나 이상에서 동등하거나 우위. 모든 metric 에서 열위면 ontology framing 은 사망 — "기존 taxonomy 로부터 training-free direction 추출" 이라는 efficiency contribution 으로 pivot.

### Phase 3 — Content-injection 대비 matched-effect Pareto (5–7 일)

**이것이 핵심 실험이다. 앞의 모든 phase 는 이 실험을 가능하게 하기 위해 존재한다.**

- P3.1. 비교 방법 (matched target-effect magnitude 기준):
  - K-only bias (본 연구)
  - CAA (residual stream)
  - ITI (attention head output)
  - PASTA (attention score reweight)
  - Focus Directions K+Q (가장 가까운 mechanistic prior)
  - SAE clamping on matching feature (Gemma Scope / Llama SAE 에 해당 feature 존재 시)
  - Prompt-only baseline (instruction following)

- P3.2. Effect matching protocol (SAE-TS, Chalnev 2024 에서 차용):
  - Effect = target-concept probe accuracy delta over baseline
  - 각 방법에 대해 held-out prompt set 상에서 동일한 effect magnitude E* 를 내는 hyperparameter (β, α, scale) 를 찾음
  - Matched E* 에서 모든 preservation metric 측정

- P3.3. Preservation metric 세트:
  - **Fluency**: held-out WikiText-103 에서 CE
  - **General capability**: MMLU 5-shot delta
  - **Near-fact specificity**: CounterFact neighbourhood accuracy
  - **Context-memory override rate**: contrary-fact stress test (Yu/Merullo/Pavlick 2310.15910 + 2511.05919 benchmark 가용 시)
  - **KL-on-benign**: NQ-open dev prompt 에서 KL(steered || base)
  - **Gold-token rank delta**: factual completion prompt 에 대한 logit lens

- P3.4. 주 figure 생성: **matched-effect Pareto 상에서 K-bias 가 preservation 축 최소 두 개에서 어느 방법에도 dominate 되지 않아야 함.**

**Exit criterion (H2, H3)**: K-only 가 matched-effect Pareto 에서 최소 CounterFact specificity 또는 context-memory override 에 대해 CAA, ITI, SAE-clamp 모두 위에 있어야 함. 실패 시 논문 등급 하락 — workshop tech report.

### Phase 4 — Multi-layer schedule (3 일)

- P4.1. Layer-wise β 스케줄: flat, ramp-up, mid-layer peak, Zhu 의 contextual-head 서브셋
- P4.2. `Σ_ℓ β_ℓ` 예산 고정 상태에서 effect 를 최대화하는 schedule 탐색

**Secondary contribution** — gate 아님. 결과는 논문의 "recommended default" 절에 반영.

### Phase 5 — 두 facet 의 compositional focus (3 일)

- P5.1. Orthogonal 한 두 facet (OISA AFOD 스타일 NMI < 0.3 검증)
- P5.2. 동시 bias `β_1 v_1 + β_2 v_2`
- P5.3. 두 probe 가 additive 하게 활성화되는지, 한쪽이 다른 쪽을 crowd out 하는지 측정

**Secondary contribution** — Phase 1–3 통과 시에만 진행. 핵심 claim 아님.

### Phase 6 — Tool selection benchmark (5–7 일)

- P6.1. Benchmark: BFCL V4 (hallucination subcategory) + MetaTool overlap subtask + MTU-Bench (ASA 직접 비교)
- P6.2. Benchmark 의 tool-category label 로 ontology facet basis 구성 — hand design 금지, benchmark 가 제공하는 label 만 사용
- P6.3. K-bias vs ASA (residual mid-layer) vs retrieval-only vs prompt-only 비교
- P6.4. 주 metric: homonymous/overlapping tool case 에서의 tool-selection accuracy
- P6.5. Headline fact-preservation metric: **tool-signature fabrication rate** — steered 모델이 카탈로그에 없는 parameter 명이나 tool field 를 지어내는가?

**Exit criterion (H8)**: K-bias ≥ ASA on tool-selection accuracy AND < ASA on tool-signature fabrication rate. 이것이 applied-value claim. 둘 중 하나만 성립하면 응용 framing 은 약화되지만 Phase 1–5 의 mechanistic story 는 유지.

---

## 4. Metric 정의 (정확히, baseline 이 빠져나갈 여지 없도록)

**Target effect magnitude `E`**: held-out target-concept probe 의 top-1 accuracy delta (unsteered baseline 대비). Probe 는 별도의 concept-labeled set 에 대한 hidden state 로 학습한 linear classifier.

**CounterFact specificity**: N 개의 ROME neighbourhood fact (steering 이 건드리면 안 되는 fact) 각각에 대해 steered vs baseline top-1 accuracy 측정. **Accuracy retention rate** = (#steering 후에도 맞는 neighbourhood fact) / (#baseline 에서 맞는 fact) 로 보고.

**Context-memory override rate**: 각 stress prompt `"Context: [fact X]. Question: [related Q]"` 에서 X 가 steering direction 과 모순될 때, steered 모델이 X 를 entail 하는 응답 (good) 대 steering direction 을 entail 하는 응답 (bad) 을 내는 비율 측정. NLI 판정: DeBERTa-v3-MNLI.

**KL-on-benign**: 500 개 NQ-open dev prompt 에 대해 KL(π_steer(·|x) || π_base(·|x)) 평균. Median 과 90th percentile 모두 보고.

**Tool-signature fabrication rate** (Phase 6 only): steered 모델이 emit 한 각 tool call 을 (tool_name, parameters) 튜플로 파싱하고 catalog schema 에 대해 exact match 검사. Fabrication = tool_name 이 카탈로그에 없음 OR parameter 명이 schema 에 없음 OR parameter type 위반. Emit 된 call 중 비율로 보고.

**Effect matching tolerance**: `|E_method - E*| < 0.02` 를 만족하도록 hyperparameter 선택. Baseline 이 이 조건을 만족하는 값을 찾지 못하면 가장 가까운 값을 쓰고 mismatch 를 결과에 명시.

---

## 5. Baseline 목록 (완전판 — 말 없이 건너뛰지 말 것)

| Baseline | Mechanism | 필수 여부 | Direction 출처 |
|---|---|---|---|
| No steering | Identity | 예 | — |
| Prompt-only (instruction) | Prompt prefix | 예 | — |
| CAA | Residual stream addition | 예 | Contrastive mean-diff |
| ITI | Per-head output shift | 예 | 학습된 probe |
| PASTA | Marked span 에 대한 attention score reweight | 예 | 사용자가 mark 한 span (ontology label 로 치환) |
| Focus Directions | K + Q additive bias | **예 (critical)** | Gold response gradient-trained |
| SAE feature clamping | Feature-space clamping | 해당 feature 가 Gemma Scope 에 있을 때 | SAE feature |
| ASA | Residual mid-layer mixture-of-vectors | Phase 6 만, critical | Probe-contrastive |
| K-bias ontology (본 연구) | K-side additive bias | 예 | Ontology embedding projection |
| K-bias gradient (본 연구, Phase 2) | K-side additive bias | 예 | Gradient-trained (Focus Directions 레시피, K-only) |
| K-bias contrastive (본 연구, Phase 2) | K-side additive bias | 예 | CAA-style mean-diff in K-space |

---

## 6. 모델과 컴퓨트

- **Primary**: Llama-3-8B-Instruct — Zhu 2025, CAA, ITI 와의 비교 가능성
- **Secondary**: Qwen2.5-1.5B-Instruct — MTU-Bench 상 ASA 와 직접 비교
- **Tertiary**: Mistral-7B-Instruct-v0.3 — GGB_experiment 데이터 재활용
- **Optional**: Gemma-2-9B — target facet 에 해당하는 public Gemma Scope SAE feature 존재 시 (SAE-clamp baseline 에 필요)

컴퓨트 추정: A100-80G 기준 전체 ~300 GPU-hour. Phase 1–3 만이면 ~150 GPU-hour.

---

## 7. Timeline 과 go/no-go gate

| Phase | 기간 | Gate | 실패 시 |
|---|---|---|---|
| 0 | 3–5 일 | 모든 baseline 이 ±2pp 이내 재현 | 통과 후 진행 |
| 1 | 3–4 일 | H1 (MMLU Δ ≤ 1% 에서 K-only 가 ≥ 20pp effect) | 중단: 논문 infeasible, negative note 작성 |
| 2 | 3–4 일 | H5 (ontology 가 target 상 gradient-trained 의 90% 이상) | "Training-free direction" framing 으로 pivot |
| 3 | 5–7 일 | **H2 + H3** (specificity 또는 override rate 상 K-bias 가 Pareto frontier) | Workshop tech report 로 강등, Phase 4–6 생략 |
| 4 | 3 일 | — (informational) | — |
| 5 | 3 일 | — (informational) | — |
| 6 | 5–7 일 | H8 (K-bias ≥ ASA on accuracy, < ASA on fabrication) | 응용 절 제거, mechanistic story 유지 |

**모든 것이 통과 시 총**: 집중 작업 약 4 주. Phase 3 실패 시 tech report 까지 약 2 주.

Phase 3 가 논문의 생사. 부분 결과가 baseline retuning 에 반영될 수 있을 만큼 시간을 할당할 것.

---

## 8. 리스크와 대응

**R1. Focus Directions (Zhu 2025) 가 모든 metric 에서 K-only 를 dominate.** K+Q 가 모든 preservation 축에서 K-only 보다 엄격히 우월하면 K-only ablation 은 story 가 없다.
- *대응*: Ablation 은 여전히 과학적 발견이다 ("symmetry helps"). 이 경우 논문을 새 method 가 아니라 Focus Directions 의 mechanistic analysis 로 재포지셔닝. 기여는 작아지지만 workshop 수준에서는 여전히 viable.

**R2. Ontology direction 이 gradient-trained 에 비해 엄격히 열위.** Ontology projection 이 K-space 의 useful direction 에 착륙하지 못하면 Phase 2 가 붕괴한다.
- *대응*: Ontology framing 이 quality claim 에서 efficiency claim 으로 이동 ("zero-training direction 으로 gradient-trained 의 X%"). X > ~80% 이고 wall-clock 비용이 설득력 있으면 여전히 publishable.

**R3. Matched-effect Pareto 가 K-bias 와 CAA 를 분리하지 못함.** 핵심 thesis 사망.
- *대응*: Phase 1 ablation 을 독립 contribution 으로 보고. "Attention-site steering direction 에 대한 체계적 연구, K-only 가 site 선택지 중 capability 를 가장 잘 보존함을 경험적으로 입증" 으로 재framing — 약해지지만 정직.

**R4. Focus Directions 가 appendix 에 K-only ablation 을 숨겨놨을 가능성.** 검색으로는 발견하지 못했으나, Zhu 2025 논문 전문을 읽어야 확인.
- *대응*: Phase 1 시작 전에 이 full read 를 반드시 할 것. Zhu 가 이미 K-only 를 돌렸다면 즉시 novelty 재정의, user 에게 통지, center of gravity 를 ontology + matched-effect Pareto + tool-selection 쪽으로 이동.

**R5. BFCL V4 / MetaTool 이 homonymous-tool subset 을 노출하지 않음.** Phase 6 applied story 가 약화.
- *대응*: MetaTool overlap subtask 나 Seal-Tools 에서 synthetic overlap benchmark 를 구성. Cherry-pick 이 아님을 reviewer 가 이해할 수 있도록 명시.

**R6. ASA (arXiv:2602.04935) 의 follow-up 에 ontology 또는 attention-side intervention 이 이미 있을 가능성.** "최초" 우려 재등장.
- *대응*: 실험 기간 내내 2602.04935 를 citing 하는 arXiv 신규 listing 을 monthly 점검. 새 경쟁 논문 발견 즉시 flag.

---

## 9. Deliverable

- **Primary**: 논문 초고 (~8 쪽). Phase 1–6 전부 통과 시 NeurIPS / ICLR main target. Phase 1–3 만 통과 시 COLM / BlackboxNLP workshop. Phase 1 만 통과 시 arXiv tech report.
- **Secondary**: 오픈소스 구현 (PyTorch forward hook, 최소), ontology → K-direction projection 유틸리티, 모든 baseline 에 대한 재현 스크립트
- **Artifact**: Matched-effect Pareto plot (주 figure), ablation table, BFCL V4 상 Phase 6 leaderboard row

---

## 10. Appendix A — OISA 와의 융합 가능성 (secondary interest)

본 실험은 standalone 논문으로 설계되었으나, 다음의 자연스러운 OISA 연결 고리가 존재한다. Phase 1–3 통과 후 본 실험이 끝나면 탐색 가능.

- **AFOD 를 direction 출처로 사용**: OISA 의 AFOD 는 이미 F 개의 orthogonal facet 과 basis vector 를 생성한다. Phase 2 의 ontology projection 은 기업 catalog 에 대한 AFOD 출력을 재사용할 수 있다. Hand-designed label 외에 경험적으로 grounded 된 두 번째 ontology source 가 확보된다.
- **FC-LoRA 와의 상보성**: K-bias 는 OISA FC-LoRA 의 학습 manifold 에 포함되지 않은 facet 조합에 대한 training-free fallback 을 제공한다. "FC-LoRA 위에 K-bias" 조합 실험으로 두 방법이 additive 인지 interfere 하는지 테스트 가능.
- **FOKVQ 와의 KV cache 호환성**: K-bias 는 attention score 계산 *전* 의 K 에 작용. FOKVQ 는 저장된 KV cache 의 precision 에 작용. 수학적으로 composable. 두 방법을 함께 적용했을 때 FOKVQ quantization noise 가 K-bias direction 품질과 상호작용하는지 측정 가능.
- **공유 benchmark**: OISA Exp 2 (homonym resolution) 와 Exp 5 (unseen facet combination) 는 본 계획 Phase 6 와 overlap. 결과는 cross-populate 가능.

위 넷 모두 standalone 논문에는 load-bearing 아님. 팀이 나중에 재발견하지 않도록 미리 기록.

---

## 11. Appendix B — Prior art 치트 시트

| 논문 | Mechanism | Direction 출처 | 인용 이유 |
|---|---|---|---|
| Zhu et al. 2025 (2503.23306) | Contextual head 에 K+Q bias | Gradient-trained | 직접적 mechanistic 경쟁 |
| Zhang et al. 2023 PASTA (2311.02262) | Attention score reweight | 사용자 mark span | Canonical focus-shift 선행 |
| Wang et al. 2026 ASA (2602.04935) | Residual mid-layer mixture | Probe-contrastive | Tool 에 대한 유일한 선행 activation steering |
| Rimsky et al. 2023 CAA (2312.06681) | Residual stream addition | Contrastive mean-diff | Content-injection baseline |
| Li et al. 2023 ITI (2306.03341) | Attention head output shift | 학습된 probe | Attention-family baseline |
| Chalnev et al. 2024 SAE-TS (2411.02193) | SAE feature space | SAE | Matched-effect protocol 출처 |
| Stickland et al. 2024 KTS (2406.15518) | Fine-tune + steering | — | KL-on-benign metric 출처 |
| Meng et al. ROME (2022) | Weight editing | — | CounterFact specificity protocol 출처 |
| Yu et al. 2023 (2310.15910) | — | — | Context-memory override protocol 출처 |
| SteeringControl (2509.13450) | — | — | Side-effect 통합 benchmark |
| Fact Grounded Attention (2509.25252) | Pre-softmax score bias | 외부 fact KB | 외부 지식 → attention-level bias 의 가장 가까운 선행 |

---

*계획 끝. 실행 중 본 계획에서 벗어나는 결정은 본 파일 하단에 일자별 수정 사항으로 기록할 것.*

---

## Amendment 2026-04-09 (A1): Zhu 2025 Focus Directions 전문 정독

### 확인된 사실 (arXiv:2503.23306 full read)

- **개입 수식 (Eq. 5)**: `W = softmax((Q + α·d_Q)(K + α·d_K)^T / √F)` — K 와 Q 에 **동일한 단일 스칼라 α** 공유. 두 component 를 분리할 수 없음.
- **Direction 학습**: `d_K`, `d_Q` joint optimization. AdamW, lr=10⁻³, 10 epoch. 손실 `L = -S_C^d` (relevant context 에 대한 attention 양 자체를 maximize).
- **학습 데이터**: Multi-Document QA from "Lost in the Middle", 2654 샘플, 50/50 train/test.
- **Layer/head 선택**: Llama-3.2-3B 기준 middle-late layer 8–18, 672 개 head 중 contextual score ≫ 0.2 인 head 는 단 2 개 (0.3%). Top-20 head 가 optimal.
- **Hyperparameter**: α ∈ {−0.2, 0.2, 0.3, 0.5}, optimal α=0.3.
- **벤치마크**: HELMET 전체 (Recall / RAG / Re-ranking / ICL / Long QA). NQ, TriviaQA, HotpotQA, PopQA, MS MARCO 등.
- **Baseline**: no-intervention, gold context, split-softmax (Li et al. 2024a).
- **Zhu 가 측정하지 않은 것**: MMLU delta, perplexity, fact preservation, hallucination, side-effect, capability degradation. **하나도 없음.**
- **Zhu 가 ablate 하지 않은 것**: K-only vs Q-only vs K+Q. K 와 Q 는 언제나 joint.
- **명시된 limitation**: (a) "focus directions may be task dependent", (b) "applying overly strong focus directions can inadvertently heighten attention to irrelevant contexts" — 정량화 없음.

### R4 판정: 해소

Zhu 는 K-only ablation 을 돌리지 않았다. Appendix 에도 없음. 우리 Phase 1 의 K/Q/K+Q ablation 은 공개 지점이다.

### 문제 설정 차이의 중요성

Zhu 의 문제: "long-context distraction" — 모델이 긴 context 에서 relevant token 을 무시하는 현상. d_K/d_Q 는 특정 relevant token 을 더 보게 만드는 pointer 역할.

우리 문제: 여러 valid 한 선택지 중 의미 축을 따라 focus 편향. d_v_f 는 ontology facet 방향이지 특정 token pointer 아님.

**두 문제는 다른 family 에 속한다.** Zhu 의 direction 은 gold answer 있는 task 에서만 학습 가능 (task-dependent — 본인들 인정). 우리 direction 은 ontology 만 있으면 학습 불필요 (task-independent by construction). 이 차이가 두 번째 독립 novelty 축이 된다.

### Thesis 수정

기존:
> Intervention direction 이 외부 의미 ontology 에서 유도되고 bias 가 K-side (K+Q 가 아니라, residual 이 아니라) 에 적용될 때, attention steering 은 동등한 steering effect 강도에서 content-injection 방법이 달성할 수 없는 수준으로 context fact 와 non-target knowledge 를 보존한다.

수정 (task-independence 축 추가):
> Intervention direction 이 외부 의미 ontology 에서 유도되고 (**training-free, task-independent**), bias 가 K-side 에만 적용될 때, attention steering 은 (a) Zhu 2025 의 gradient-trained direction 이 요구하는 per-task training 없이 동등한 focus 효과를 내고, (b) 동등한 steering effect 강도에서 content-injection 방법이 달성할 수 없는 수준의 fact preservation 을 달성하며, (c) 고강도 steering 에서 gradient-trained direction 보다 graceful 하게 열화된다.

### 새 가설 H9, H10

| H# | 가설 | 측정 방법 | 실패 시 |
|---|---|---|---|
| H9 | Ontology direction 의 "effective α range" (target gain ≥ baseline + 20pp 이면서 irrelevant leakage ≤ 10% 인 구간) 가 gradient-trained direction 보다 넓다 | α sweep {0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0} 에서 target probe + irrelevant attention mass 동시 측정 | 논문 사망 아님, "low-α 에서만 경쟁력" 으로 claim 약화 |
| H10 | Zhu 의 HELMET NQ 벤치마크에서 ontology 유래 K-only direction 이 gradient-trained K+Q direction EM 의 90% 이상 유지 | HELMET NQ + TriviaQA subset 재현 실험 | 논문 사망 아님, "Zhu 의 task 에서는 약함, 우리 task (tool selection) 에서는 강함" 으로 재frame |

### Phase 1 확장: HELMET cross-check

Phase 1 마지막에 추가 실험:

- **P1.4**. Zhu 2025 의 HELMET NQ + TriviaQA subset 재현.
  - (a) Zhu 의 레시피대로 K+Q gradient-trained direction 학습 (Multi-Document QA 2654 샘플, AdamW, 10 epoch)
  - (b) 동일 layer/head set (Zhu 의 top-20 contextual head) 에 우리 ontology-derived direction 을 K-only 로 적용
  - (c) α sweep {0.2, 0.3, 0.5} 에서 EM 비교
  - (d) **Exit criterion H10**: K-only ontology 의 EM 이 K+Q gradient-trained 의 90% 이상

### Phase 2 확장: High-α degradation curve

Phase 2 에 서브 실험 추가:

- **P2.3**. Zhu 의 acknowledged limitation ("overly strong directions bleed") 정량화.
  - α ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0} 넓은 sweep (Zhu 는 {-0.2, 0.2, 0.3, 0.5} 만 보고)
  - 두 direction 출처 (ontology projection, Focus Directions gradient-trained) 각각에 대해 세 곡선 측정:
    - Target probe accuracy
    - **Irrelevant-context attention mass**: steered 모델의 attention weight 중 ontology 기준으로 irrelevant 한 token 에 할당된 비율 (Zhu 가 측정 안 한 것)
    - CounterFact specificity retention
  - **Exit criterion H9**: Ontology 의 "effective α range" 가 gradient-trained 보다 넓음

이 실험 단독으로도 workshop paper 가치가 있다 — Zhu 의 명시된 failure mode 를 처음으로 정량화한 결과가 된다.

### Risk 수정

**R4 제거** (Zhu 의 숨은 K-only ablation 가능성): Zhu 2025 전문 정독 결과 존재하지 않음. 확인 완료.

**R7 추가**: HELMET NQ 에서 ontology K-only 가 Zhu 의 gradient-trained K+Q 에 크게 뒤처질 위험.
- *원인*: Zhu 의 task 는 context 내 relevant token pointer 문제이므로 gradient 신호가 강함. Ontology embedding 은 semantic axis 이지 token pointer 가 아님. HELMET 에서의 간극은 어느 정도 예견됨.
- *대응*: (a) HELMET 결과를 정직하게 보고 — "Zhu 의 task 에서는 Zhu 가 강함", (b) 우리 tool-selection task 는 pre-existing ontology 가 있는 reality 이므로 gradient-training 이 불필요한 점이 advantage 라고 재frame, (c) 간극이 크면 제목에서 "focus directions" 과의 직접 경쟁 뉘앙스를 빼고 "training-free, task-independent attention steering" 쪽으로 무게 이동.

### Phase 0 precondition 상태 (A1 시점)

- [x] 0.1.a Zhu 2025 전문 정독 — **완료 (2026-04-09)**
- [ ] 0.1.b ASA 2026 전문 정독 — 진행 중
- [ ] 0.1.c PASTA 2023 전문 정독 — 진행 중
- [ ] 0.2 GGB_experiment padding fix
- [ ] 0.3 baseline 목록 확정

---

## Amendment 2026-04-09 (A2): ASA / PASTA / CAA / ITI 전문 정독

A1 이후 네 개 주요 선행 논문을 병렬로 정독했다. 이 amendment 는 각 논문에서 추출한 사실과 그로부터 파생된 계획 수정 사항을 기록한다.

### A2.1 ASA (arXiv:2602.04935, Feb 2026) 전문 정독 결과

**개입 수식 (Eq. 14)**: `h'_L(x) = h_L(x) + Gate(h_L(x)) · α · MoV(h_L(x))`

- 개입 위치: Pre-LN residual stream, **단일 layer L**, **마지막 non-padding prompt token 1 개**, pre-fill 1 회. 디코딩 중 매 step 이 아님.
- MoV = `v̂_{d̂} + β·v̂_global`. 벡터 수 **총 5 개** (1 global + 4 domain: Code/Math/Search/Translation).
- 벡터 구축: class-conditional mean difference (CAA-style), 320 calibration 샘플. Backprop 없음.
- Router: linear softmax `W^r h̃_L + b^r` 로 domain 예측.
- Gate: ternary `{+1, 0, −1}` signed. `Gate=+1` if `p(x)>τ`, `-1` if `p(x)<1-τ`, 그 외 `0`. **Gate 없으면 FPR 이 0.05 → 0.50 폭발 (Table 6)** — 이건 ASA 에서 본질적 element.
- 벤치마크: **MTU-Bench 만**. 4 개 disjoint domain, 각 domain cross-cosine <0.4 (Table 2) — ASA 의 핵심 가정.
- Headline (Qwen2.5-1.5B, L=18, α=4.0): F1 0.1818 → 0.5037, FPR 0.1458 → 0.0521.
- Layer: Qwen2.5-1.5B L=18, LLaMA-8B L=21. Cross-model 전이 시 layer 재조정 필요.
- 0.5B 모델에서 Recall=0 (완전 실패).
- Baseline 비교: LoRA/Q-LoRA/Prefix/BitFit/prompt. **CAA/ITI/PASTA/RepE 중 어느 것과도 비교하지 않음**.
- Fact preservation 측정 없음: Format Acc, Tool Name Acc (0.7436), Argument Acc, FPR 만.
- Limitations 섹션 없음. 산재된 언급: "ASA cannot create tool-use behavior from scratch" (§4.2), "routing accuracy is the bottleneck".

**결정적 통찰**: ASA 는 **domain 단위 routing** 이지 **tool 단위 disambiguation** 이 아니다. 4 개 domain 이 disjoint 하다는 가정 (cosine <0.4) 은 homonymous tool 이 존재하는 순간 무너진다. **Homonym disambiguation 은 ASA 의 범위 밖 problem 이다.** 우리가 선점 당한 것이 아니라 다른 problem 을 풀고 있다.

### A2.2 PASTA (arXiv:2311.02262, ICLR 2024) 전문 정독 결과

**개입 수식 (Eq. 2)**: `[T(A)]_ij = α·A_ij/C_i` for `j ∈ G^-`, `A_ij/C_i` otherwise. Row renormalization with `C_i = Σ_{j∈G} A_ij + Σ_{j∈G^-} α·A_ij`.

- 개입 위치: **post-softmax attention weight matrix** `A`. Pre-softmax logit 이 아니고, value/output 도 아님.
- 기본 α = 0.01 (0 은 피해야 함 — context 삭제 효과).
- G 는 **사용자가 marking 한 token index 집합**. `*...*` 이나 `""..."` 로 표시. **Semantic axis 없음, ontology 없음, concept 개념 없음.** 순전히 positional.
- Head 선택: **Multi-task profiling** — 각 (l, h) 에 대해 single-head ablation 을 ~1000 샘플에 돌려 task accuracy 측정. Task 별 top-k 의 intersection 이 default.
- 7B 에서 |H| = 50–150 optimal (k ∈ {300, 400, 500} of 1024 heads).
- Llama-7B Multi-task PASTA headline (Table 1): JSON F/P 96.6/85.1 (vs zero-shot 60.0/54.9), Pronouns 96.4/95.8 (vs 71.8/66.3), BiasBios 95.3 (vs 87.4), CounterFact ES/PS 99.6/99.6 (vs 58.5/52.0). 평균 95.5 vs 67.3.
- **"focus-shift" 용어는 논문에 등장하지 않는다.** "directing", "steering", "emphasizing", "bold/italics analogy" 만.
- Baseline 비교: zero-shot, `*`-marked, `""`-marked, 3-shot few-shot. **ITI/CAA/RepE/ActAdd/ROME/MEMIT 중 어느 것과도 비교하지 않음** (Related Work 에 언급만).
- Capability retention 측정 없음: MMLU 없음, TruthfulQA 없음, perplexity-on-corpus 없음. 엔트로피 >3.0 fluency filter 만.
- Failure mode: 50–150 head 초과 시 JSON Pred Acc 와 fluency 하락 (Figure 3a/b). α=0 은 context 파괴. 용어로 표현은 없지만 **U-shape trade-off** 가 존재함.
- Dedicated limitations 섹션 없음. Task-dependence of head selection, absence of representation-engineering baseline, factual knowledge 보존 여부 — 모두 open 으로 남김.

**결정적 통찰**: PASTA 는 focus-shift 의 **존재 증명** 이지 focus-shift 라는 **category 의 claim** 이 아니다. 우리가 "focus-shift vs content-injection" 을 **공식화** 하는 것은 여전히 novel 기여다. 단 "first focus-shift attention steering method" 는 claim 불가 — PASTA 가 먼저 한 attention-level method 이기 때문.

PASTA 는 또한 **ontology / semantic axis interface** 를 전혀 건드리지 않았다. PASTA 의 G 는 순수 positional. 이건 우리 ontology 차별화의 근거를 강화한다.

### A2.3 CAA (arXiv:2312.06681, ACL 2024) 전문 정독 결과

**Recipe (§3, Eq. 1)**: Multiple-choice contrastive pair, 답 letter 위치의 residual stream mean difference.
`v_MD = (1/|D|) Σ [a_L(p, c_p) − a_L(p, c_n)]`

- 학습: Llama-2-7B-Chat L=13, Llama-2-13B-Chat L=14 또는 15 (Figure 3 layer sweep 결과).
- Pair 수: behavior 당 290 (Corrigibility) ~ 1000 (Sycophancy).
- 적용: **prompt 뒤 모든 token position** 의 residual stream 에 `c · v_MD` 더함. ActAdd 가 prompt 의 첫 token 에 더하는 것과 다름.
- Normalization: **across behavior 정규화** (behavior 간 multiplier 비교 가능), **across layer 정규화 안 함** (residual stream norm 이 layer 에 따라 지수 증가하므로).
- Multiplier: **모든 headline 결과가 오직 ±1**. §4.2: *"steering with larger multipliers results in a degradation in the quality of the open-ended text"* — **정량화 없음, curve 없음**.
- 7 개 behavior: AI Coordination, Corrigibility, **Hallucination** (GPT-4 로 생성된 synthetic MC dataset), Myopic Reward, Survival Instinct, Sycophancy, Refusal.
- MMLU (Table 5, 13B L=14, ±1): baseline 0.63. 최대 drop 은 Hallucination at −1 → 0.57 (absolute −0.06). 나머지 <0.04. 저자 주장: "our intervention does not significantly affect MMLU performance".
- TruthfulQA (Appendix H): Sycophancy vector 빼기만 테스트. Llama-2-13B-Chat: +0.02 개선, 빼면 −0.03 악화. 7B: +0.01/−0.05. "more investigation needed".
- **Fact-contradicting context 실험 없음**. Closed-book QA 없음. 외부 hallucination benchmark 사용 안 함.
- Random seed 없음, error bar 없음, significance test 없음 (§10 인정).
- **§9.1 Future Work 에 직접 명시**: *"steering outside the residual stream ... e.g., after the MLP but before merging"* — attention-site 개입은 CAA 저자가 **직접 future work 로 초대**한 방향.

**결정적 통찰**: K-side attention bias 는 CAA §9.1 에 적혀 있는 future work 를 실행한 것이다. 이건 정당화의 황금이다 — 논문 intro 에 *"CAA (Rimsky et al. 2023) explicitly suggested steering outside the residual stream as future work; we do this."* 라고 쓸 수 있다.

CAA 는 또한 multiplier-resolved capability curve 를 그리지 않았다. 오직 ±1 에서만 측정. Curve 를 그리는 것 자체로 부분 기여가 된다.

### A2.4 ITI (arXiv:2306.03341, NeurIPS 2023) 전문 정독 결과

**개입 수식 (Eq. 2, §3.1)**: `x_{l+1} = x_l + Σ_h Q_l^h(Att_l^h(P_l^h x_l) + α·σ_l^h·θ_l^h)`

- 개입 위치: **per-head attention output**, softmax-weighted V aggregation **이후**, W_O projection **이전**. Head-space 의 D 차원.
- **Eq. 3 으로 접힘**: 실제로는 `Bias_l = α Σ_h Q_l^h(σ_l^h θ_l^h)` — **W_O bias 에 대한 상수 offset**. **Input-independent**. Offline 으로 W_O 에 bake 가능.
- **결정적 사실**: ITI 는 attention pattern (softmax 가중치, K/Q dot product) 을 **건드리지 않는다**. 어떤 token 이 보이는지 바꾸지 않고, 보고 난 후 residual stream 에 쓰는 것만 바꾼다.
- Head 선택: per-(l,h) logistic regression probe on TruthfulQA. 5918 개 QA pair 재구성. Train/val 4:1 분할. Llama-7B 기준 top-K=48 / 1024 (4.7% sparse).
- Direction: **Mass mean shift 가 probe weight direction 과 CCS 를 이김** (Table 3). Per-head 로 각자 다른 방향.
- α: Llama-7B 에서 α=15 가 optimal (K=48).
- TruthfulQA headline: Llama-7B baseline 30.5 → ITI 43.5 True×Info. Alpaca 32.5 → 65.1. Vicuna 51.5 → 74.0.
- **MMLU/NQ/TriviaQA (Table 4)**: MMLU 35.71 → 40.16 (소폭 **상승**). NQ 46.6 → 51.3. TriviaQA 89.6 → 91.1. 능력 감소 없음.
- CE(OWT) 2.16 → 2.48, KL=0.40. Vicuna+ITI 에서 KL=1.41 로 큼. 생성 분포에 측정 가능한 shift 있음.
- α trade-off: "upside-down U curve". Peak 지나면 refusal ("I have no comment") 증가. Pointwise selection α=15 에서 CE=4.01, KL=1.95 — 모델 심하게 손상.
- Framing: "shifting along truthful direction" 만. Focus vs content 논의 없음.

**결정적 통찰**: ITI 와 K-bias 는 **인과적으로 완전히 다른 site** 다. 한 줄로: *"ITI changes what each head writes; K-bias changes what each head reads."* 이건 가장 깨끗한 ablation cell 이다 — 두 방법은 mechanistically 같은 경계선의 반대편에 있다.

ITI 는 또한 input-independent (Eq. 3). K-bias 는 input-dependent (QK^T 에 영향). 이것도 causal distinction 이다.

### A2.5 종합: Novelty surface 가 **넓어졌다**

초기 survey 는 우리 novelty 를 좁게 잡았는데, 전문 정독 결과 다음 다섯 지점이 전부 열려 있다:

1. **Focus-shift vs content-injection 의 공식화.** PASTA 가 용어를 쓰지 않았다. Dichotomy 를 formal 하게 정의하는 것 자체가 novel 기여 가능. 단 "first focus-shift method" 는 claim 불가.
2. **Matched-effect Pareto across {PASTA, Focus Directions, K-bias, ITI, CAA, SAE clamping}.** 네 논문 어느 것도 matched-effect 비교를 하지 않았다. 단일 figure 로 독립 기여.
3. **Multiplier-resolved capability curve for CAA, ITI, PASTA.** 셋 다 고정 α 에서만 측정. CAA 는 본인들이 "quality degrades at higher multipliers" 라고 qualitative 하게만 인정. Curve 를 그리는 것 자체로 부분 기여.
4. **Homonymous tool disambiguation.** ASA 의 disjoint-domain 가정 (cosine <0.4) 은 homonym 이 존재하는 순간 붕괴. 우리 problem 은 ASA 의 범위 밖.
5. **K-bias 는 CAA §9.1 이 초대한 future work.** "steering outside the residual stream" 을 직접 실행. 정당화의 근거로 인용 가능.

### A2.6 Thesis 재수정 (A1 이후 2차 수정)

A1 의 thesis:
> Intervention direction 이 외부 의미 ontology 에서 유도되고 (training-free, task-independent), bias 가 K-side 에만 적용될 때, attention steering 은 (a) Zhu 2025 의 gradient-trained direction 이 요구하는 per-task training 없이 동등한 focus 효과를 내고, (b) 동등한 steering effect 강도에서 content-injection 방법이 달성할 수 없는 수준의 fact preservation 을 달성하며, (c) 고강도 steering 에서 gradient-trained direction 보다 graceful 하게 열화된다.

A2 수정 (causal distinction 을 명시):
> **Attention steering 은 "head 가 무엇을 쓰는가" 를 바꾸는 개입 (content injection: CAA, ITI, SAE clamping) 과 "head 가 무엇을 읽는가" 를 바꾸는 개입 (focus shift: PASTA, Focus Directions, 본 연구의 K-bias) 으로 인과적으로 구분된다.** Intervention direction 이 외부 의미 ontology 에서 유도되고 (training-free, task-independent), bias 가 K-side 에만 적용될 때, focus-shift 개입은 (a) 동등한 target effect 강도에서 content-injection 방법이 달성할 수 없는 수준의 **parametric fact preservation** 과 **in-context fact faithfulness** 를 달성하고, (b) CAA (Rimsky et al. 2023) §9.1 에서 명시적으로 요청된 "attention-site steering" 에 대한 첫 정량적 응답이며, (c) ASA (Wang et al. 2026) 가 disjoint-domain 가정으로 풀지 못하는 **homonymous tool disambiguation** 에 적용 가능하다.

### A2.7 새 가설 (H11–H13)

| H# | 가설 | 측정 방법 | 실패 시 |
|---|---|---|---|
| H11 | Focus-shift 방법 (PASTA, Focus Directions, K-bias) 이 content-injection 방법 (CAA, ITI, SAE clamping) 보다 matched effect 에서 CounterFact neighbourhood specificity 를 체계적으로 더 잘 보존한다 | 6 방법 × 5 가지 multiplier × 3 가지 specificity metric 의 매치드 비교, Llama-2-7B-Chat L=13 (CAA) 와 Zhu-recommended L=8–18 (attention family) | 주 thesis 사망, workshop tech report |
| H12 | ITI 와 K-bias 는 **동일한 contrastive pair 로 학습된 동일한 semantic direction** 에 대해 causally distinct 한 behavioral signature 를 만든다. ITI 는 "identity shift" 를, K-bias 는 "topical redirection" 을 생성한다. | ITI mass-mean direction 을 K-space 로 projection, 같은 direction 으로 ITI / K-bias 두 방식 적용, open-ended generation 을 LLM-as-judge 로 category 분류 (self-reference vs topic-redirection) | 주 thesis 약화, "mechanism 차이 뚜렷하지 않음" 보고 |
| H13 | Homonymous tool pair 가 있는 벤치마크에서 ontology-aware K-bias 가 ASA 보다 disambiguation accuracy 는 동등 이상, tool-signature fabrication rate 는 낮다 | MTU-Bench 를 cross-domain homonym 추가로 augment, ASA 가 정의한 4 domain 에 "search" 와 "code search" 같은 homonym 추가 | 응용 framing 약화, mechanism story 만 유지 |

### A2.8 Baseline 구현 recipe (정독 기반 확정)

각 baseline 에 대해 정독에서 추출한 정확한 recipe:

**CAA** (github.com/nrimsky/CAA):
- Layer 13 for Llama-2-7B-Chat, Layer 14 or 15 for 13B
- Pair: multiple-choice, single-token A/B difference
- Vector: mean difference of residual stream at answer-letter token position
- Apply: add `c · v` to residual stream at **all post-prompt token positions** of generation
- Normalize across behaviors, do NOT normalize across layers
- Multiplier sweep {0, ±0.5, ±1, ±1.5, ±2, ±2.5, ±3} (CAA 원 논문은 ±1 만, 우리가 curve 를 그려야 함)
- 7 behaviors 중 Hallucination 과 Sycophancy 를 우리 실험의 behavior 로 사용

**ITI** (github.com/likenneth/honest_llama):
- Intervention site: per-head attention output, post-softmax-weighted-V, pre-W_O
- Equivalent: offline bake into W_O bias
- Probe: per-(l,h) logistic regression on labeled activation, last-token position
- Direction: **mass mean shift** (not probe weight), per-head
- K=48 for Llama-2-7B, α=15
- Eval metric set: TruthfulQA True×Info + CE(OWT) + KL(OWT) + MMLU + NQ + TriviaQA

**PASTA** (github.com/QingruZhang/PASTA):
- Intervention site: post-softmax attention weights, row-renormalized
- α = 0.01 default, sweep {0.05, 0.01, 0.002, 1e−3}
- |H| ∈ {25, 50, 100, 150}, intersection-of-top-k multi-task profiling
- G = user-marked token index set (ontology 적용 시: ontology 용어로 substring matching → token index)
- 7B profiling: k ∈ {300, 400, 500} out of 1024 heads
- Benchmark: JSON Format, Pronouns Change, BiasBios, CounterFact (원 논문), plus MMLU + closed-book QA (우리가 추가)

**ASA** (arxiv.org/html/2602.04935, 공식 코드 유무 확인 필요):
- Layer L=18 for Qwen2.5-1.5B, L=21 for LLaMA-8B
- MoV = 1 global + 4 domain (but generalize to our tool domains)
- Linear router + per-domain sigmoid probe + ternary gate (τ ∈ [0.5, 0.7])
- 320 calibration samples
- Apply: residual stream, last non-padding prompt token, pre-fill only
- α ∈ {0.5, 1, 2, 4}
- **Gate ablation 은 필수** — Without-gate 는 내부 ablation 에서도 FPR 폭발
- Benchmark: MTU-Bench (우리가 공유해야 할 benchmark)

### A2.9 Phase 1 확장 (A2)

A1 에서 Phase 1 마지막에 HELMET cross-check (P1.4) 를 추가했다. A2 에서 여기에 더:

- **P1.5**. **Direction 을 공유한 ITI vs K-bias 비교**. 동일한 contrastive pair (CAA/ITI 방식) 로 mass-mean direction 을 얻되, 이 direction 을 (a) ITI 방식으로 W_O bias 에 bake, (b) K-space 로 projection 하여 K-bias 로 적용. 같은 strength 에서 MMLU, TruthfulQA, open-ended generation 을 측정.
  - **Exit criterion (H12 약버전)**: 두 방법이 matched effect 에서 open-ended generation signature 가 구별 가능한 차이를 보임 (LLM-as-judge 로 판정).
  - 이 실험이 성립하면 **causal distinction** 을 처음으로 empirical 하게 보인 figure 가 된다.

### A2.10 Phase 3 확장 (A2)

Phase 3 의 matched-effect Pareto 에 baseline 추가:

- **Focus-shift family**: PASTA, Focus Directions (K+Q), K-bias ontology (ours)
- **Content-injection family**: CAA, ITI, SAE clamping (Gemma Scope 가용 시), ASA (tool setting 에서)

매치드 기준: target concept probe accuracy 의 delta (이전 계획 그대로).

새 측정: **method 를 두 family 로 grouping 하고 family 차이가 family 내 차이보다 큰지 통계 검정**. 이게 H11 (focus-shift > content-injection on specificity) 의 검정이다.

### A2.11 Phase 6 확장 (A2)

A1 의 Phase 6 (tool selection) 에 homonym-specific 실험 추가:

- **P6.6. Homonym benchmark 구성**: ASA 의 MTU-Bench 4 domain 을 기반으로 homonymous tool pair 추가. 예: Search domain 에 "web_search" 와 "database_search" 가 둘 다 존재, 문맥에 따라 하나가 정답. Cross-domain cosine 을 0.5 이상으로 강제하여 ASA 의 disjoint 가정이 깨지는 condition 을 실험적으로 만듦.
- **P6.7. ASA vs ontology-K-bias on homonym benchmark**: 동일 benchmark 에서 ASA (원 recipe), ASA + homonym-aware domain 확장, K-bias ontology 세 조건 비교.
- **Exit criterion (H13)**: K-bias 가 disambiguation accuracy 에서 ASA 를 동등 이상으로 유지하면서 tool-signature fabrication rate 에서 개선.

### A2.12 Risk 추가 (R8, R9)

**R8**. **네 baseline 재현이 계획보다 오래 걸림**. CAA, ITI, PASTA, Focus Directions, ASA 다섯 개를 재현하는 건 Phase 0 의 3–5 일을 초과할 수 있다. 각각 github 공식 코드가 있어도 우리 모델 (Llama-3-8B, Qwen2.5, Mistral) 에 맞추는 porting 작업이 있음.
- *대응*: Phase 0 를 **5–8 일로 확장**. 재현 실패 baseline 은 원 논문 모델로 제한하고 그 사실을 명시.

**R9**. **Causal distinction (H12) 측정이 어려움**. ITI 와 K-bias 가 같은 direction 으로 적용되었을 때 behavioral signature 가 구별되지 않을 가능성. 이 경우 "mechanism 은 다르지만 effect 는 같음" 이 결론.
- *대응*: H12 실패 시 thesis 에서 "causal distinction" 축을 제거하고 "empirical preservation advantage" 축만 유지. 여전히 H11 (matched-effect Pareto) 로 논문 성립 가능.

### A2.13 Phase 0 precondition 상태 (A2 시점)

- [x] 0.1.a Zhu 2025 전문 정독 — 완료 2026-04-09
- [x] 0.1.b ASA 2026 전문 정독 — 완료 2026-04-09
- [x] 0.1.c PASTA 2023 전문 정독 — 완료 2026-04-09
- [x] 0.1.d CAA 2023 전문 정독 — 완료 2026-04-09
- [x] 0.1.e ITI 2023 전문 정독 — 완료 2026-04-09
- [ ] 0.1.f Fact Grounded Attention (2509.25252) — 남음 (외부 KB → attention-score bias, 가장 가까운 external-knowledge 선행)
- [ ] 0.1.g SAE-TS (2411.02193) — 남음 (matched-effect protocol 출처)
- [ ] 0.1.h Stickland KTS (2406.15518) — 남음 (KL-on-benign metric 출처)
- [ ] 0.2 GGB_experiment padding fix
- [ ] 0.3 baseline 목록 확정 — **A2.8 에서 확정 완료**

---

## Amendment 2026-04-09 (A3): FGA / SAE-TS / Stickland KTS 전문 정독

A2 이후 남은 세 개 방법론 선행 논문을 병렬로 정독했다. **두 개의 중요한 정정 사항이 발생했다** — 이전 amendment 에서 metric attribution 이 부정확했다.

### A3.1 FGA (arXiv:2509.25252, Sep 2025) 전문 정독 결과

Aayush Gupta (독립 저자), Apple M4 Max 로 실행. 제목: "Fact Grounded Attention: Eliminating Hallucination in LLMs Through Attention Level Knowledge Integration".

**개입 수식 (Eq. 7, 8)**: `S_FGA = S + α ⊙ G`, `Attention_FGA(Q,K,V) = softmax(S_FGA) V`

- 개입 위치: **pre-softmax attention score 행렬 S** (L×L). K projection 도 Q projection 도 아님.
- `G = B_qf · A`, where `B_qf = Q K_fact^T / √d_k ∈ ℝ^{L×M}` (Eq. 3, 4), `A ∈ {0,1}^{M×L}` binary entity-to-token assignment (Eq. 5). K_fact = `W_K V_fact` 는 **별도로 학습된** projection matrix (약 2.1M params).
- G 는 rank-≤M L×L matrix. Per-pair additive bias. Row-stochastic 아님.
- α = sigmoid(W_α [Q; C] + b_α) ∈ [0,1] (Eq. 6). Q 와 context feature C 에 의존하는 **learned gate**.
- Hard constraint: α ≥ 0.8 일 때 vocabulary-level hard mask (θ_hard=0.8).
- Layer: Llama 3.2 3B 기준 **layer 20–27** (top 8 of 28). Ablation: shallow (1–4) 67.2%, deep (24–28) 88.9%, main config top-8 99.7%.
- Head: per-head selection 없음. Shared gate across heads (ablation: per-head 96.3% vs shared 99.7%).
- **KB 형태**: **Flat (entity, attribute, value) tuple**. 137 entity (47 smartphones + 52 laptops + 38 EVs) × 12 attribute. **Ontology 아님**, taxonomy 아님, hierarchy 아님. Entity 인식은 rule-based chunked recognition (stride s=16).

**Headline (Table 3)**: Vanilla Llama 3.2 3B 6.3% → FGA-Zero 87.1% → FGA-FT 99.7% on 1107 spec QA.

**Public benchmark (Table 1)**: NQ 23.4 → 41.2, TriviaQA 31.8 → 48.3, PopQA 12.3 → 38.7, FEVER 67.2 → 78.9.

**측정하지 않은 것**: MMLU, perplexity, TruthfulQA, HaluEval, 그리고 **어떤 activation-steering method 와도 비교하지 않음** (CAA, ITI, PASTA, RepE, ActAdd 모두 benchmark 에 없음).

**Ablation (Table 4)**: 가장 중요한 component 는 **entity assignment matrix A** — 제거 시 42.3% 로 급락. Gate 제거 71.4%, hard constraint 제거 79.2%.

**명시된 future work (§6.2.1)**: *"Future work should explore hierarchical and compositional fact representations"*. Homonym, 개념 계층, facet 은 현재 지원 안 함.

**결정적 통찰**: FGA 는 **flat-KB method**, 저자 스스로 hierarchy/compositionality 를 future work 로 남겼다. 우리 ontology approach 는 FGA 가 **공식적으로 초대한 확장**이다. 또한 FGA 는 score matrix S 를 건드리고 우리는 K projection 을 건드린다 — 같은 attention 계산의 서로 다른 mathematical site. FGA 는 우리 작업에 대한 **두 번째 공식 초대장**이다 (첫 번째는 CAA §9.1).

추가: FGA 의 deep layer 발견 (20–27) 은 Zhu 의 middle layer 발견 (8–18) 과 다르다. 이 차이는 task-dependent — Zhu 는 long-context distraction, FGA 는 factual grounding. 우리 multi-layer phase 는 **두 range 모두** 테스트해야 한다.

### A3.2 SAE-TS (arXiv:2411.02193, Nov 2024) 전문 정독 결과 — 중요한 정정

Chalnev, Siu, Conmy. "Improving Steering Vectors by Targeting Sparse Autoencoder Features".

**중요한 정정: SAE-TS 는 strict matched-effect pinning 을 하지 않는다.** A2 에서 "matched-effect protocol 을 SAE-TS 에서 차용" 이라고 한 서술은 부정확하다. 실제로 SAE-TS 는:

- **α 를 sweep 하고 full Pareto curve 를 그리는 방식** (Figure 3 주 comparison, Appendix G Figure 8 trade-off curve).
- Scalar summary 는 **peak Behavioral × Coherence** across sweep (Table 2).
- "Matched" 가 등장하는 유일한 곳은 **training data 수집 시점** — 50,000 개 vector 에 대해 ΔCE = 0.5 nats 가 되도록 α 를 per-vector search. 이건 evaluation 이 아니라 **training data curation**.

**Evaluation metric**:
- **Effect (Behavioral score)**: GPT-4o-mini rubric 1–10, normalized to [0,1]. Concept-specific rubric.
- **Preservation (Coherence score)**: GPT-4o-mini 1–10 normalized. "Semantic 일관성과 fluency".
- **Primary scalar**: `Behavioral × Coherence` (product, not weighted sum).
- **Only one preservation axis** — KL 없음, perplexity 없음, MMLU 없음.
- Unsteered baseline: Coherence 0.64, Behavioral 0.

**Methods compared**: CAA (their variant, mean-diff of positive/negative prompt activations), SAE clamping (decoder row d_j as steering vector), SAE-TS (theirs). **ITI 없음, ActAdd 없음, PASTA 없음, Focus Directions 없음.**

**Model and setup**: Gemma-2-2B base (not instruction-tuned). Layer 12 residual stream. 9 tasks (Anger/Christian/Conspiracy/French/London/Love/Praise/Want-to-die/Wedding). 256 completion × 32 token per (method, task, α).

**Headline (Table 2, peak Behavioral × Coherence)**:
| Task | CAA | SAE | SAE-TS |
|---|---|---|---|
| Average (9 tasks) | 0.2165 | 0.1290 | **0.3600** |
| London | 0.0476 | 0.0061 | **0.5380** |
| Wedding | 0.1768 | 0.2626 | **0.5432** |
| Christian | **0.3486** | 0.0902 | 0.3302 |

SAE-TS 가 9 개 중 7 개에서 승, Christian 에서 CAA 에 패배, Conspiracy 는 동점.

**SAE clamping failure mechanism (Table 1)**: **Golden Gate 현상의 수학적 설명**. Decoder row `d_j` 를 steering vector 로 쓰면, 실제로 가장 크게 활성화되는 feature 는 **j 자신이 아니라** activation density 45% 의 "no clear pattern" feature (6810, effect magnitude 4.58). **Decoder direction ≠ causal steering direction**. SAE-TS 의 Eq. 3 `s = M_j/||M_j|| − λ(Mb)/||Mb||` 에서 두 번째 항은 이 generic high-density feature 활성화를 **명시적으로 빼주는** correction term.

그럼에도 고 α 에서는 SAE-TS 도 collapse — Appendix I 의 "London at α>160" 은 반복적 fashion-show text 로 degenerate. **Failure 는 higher α 로 밀려났을 뿐 제거되지 않는다.**

**결정적 통찰 (두 가지)**:

1. **우리의 matched-effect protocol 재정의 필요.** SAE-TS style sweep + Pareto curve 는 "peak-pick in hindsight" 이다. 더 엄격한 버전 — 각 method 를 matched preservation level (예: ΔCE = 0.5) 에서 compare — 은 SAE-TS 의 strengthening 이지 borrowing 이 아니다. 우리 논문에서는 **두 protocol 을 모두 실행**: (a) SAE-TS style sweep + peak Behavioral × Coherence, (b) strict matched-strength binary search. 후자는 SAE-TS 의 training-time criterion 을 evaluation 으로 lift 한 것으로 명시.

2. **Golden Gate 의 수학적 기제가 이제 인용 가능**. SAE-TS Table 1 + Eq. 3 이 "decoder direction 은 causal direction 과 다르다" 를 처음으로 empirically measure 했다. 우리 논문의 narrative 는: *"Golden Gate-style fabrication 은 SAE clamp 에서 고질적. SAE-TS 가 그 원인을 처음으로 정량화 (Table 1). 우리 K-bias 는 동일 문제를 mechanism 레벨에서 회피 — V/MLP 에 content 를 주입하지 않기 때문에 decoder-vs-causal mismatch 가 발생할 수 없음."*

### A3.3 Stickland KTS (arXiv:2406.15518, Jun 2024) 전문 정독 결과 — 중요한 정정

Stickland, Lyzhov, Pfau, Mahdi, Bowman (NYU + Anthropic). "Steering Without Side Effects".

**중요한 정정: Stickland 는 KL 을 evaluation metric 으로 사용하지 않는다.** KL 은 **KTS LoRA fine-tuning 의 training loss**. 그들이 보고하는 side-effect metric 은 **MT-Bench score degradation** 이다.

A2 에서 "KL-on-benign from Stickland" 이라고 한 것은 부정확한 attribution 이다. 내가 쓰려는 것은 그들의 **training objective 를 evaluation 에 재purpose** 한 것이지 그들이 metric 으로 사용한 것을 차용한 게 아니다.

**Stickland 의 실제 KL 정의 (Eq. 2)**:
```
E_{v~V}[D_KL[LLM_v(x) || LLM(x)]]
```
- **Direction**: Forward KL, `KL(steered || base)`. Steered 를 P 로, base 를 Q 로 — steered 의 novel behavior 를 penalize.
- **Prompt x 분포**: **UltraChat** (Ding et al. 2023), general benign QA dataset.
- **Training**: minibatch 당 steering vector 1 개 sample, strength `k ~ U[-c, c]`, 12.5% steer dropout, LoRA rank-128, 4 epoch, 192 step, 768 example (384 harmless + 384 UltraChat), ~50분 on A100/V100.
- **Token position**: 본문에 명시 없음. 표준 distillation 읽기는 response token 에 대해 per-token softmax KL 을 mean.

**KTS pipeline (Algorithm 1)**:
1. Box 1: Steering vector 를 mean-diff 로 구성 (Zou et al. 2023a / RepE).
2. Box 2: LoRA fine-tune base model 을 minimize Eq. 2. Base model 이 *steering 에 robust* 하게 만드는 **self-distillation under random steering perturbation**.
3. Box 3: Fine-tuned model 위에서 inference time steering 적용.

**Evaluation 축 (실제)**:
- **Side-effect (benign)**: **MT-Bench** (80 multi-turn questions, GPT-4 judge). Llama-2-chat-7B baseline 6.53.
- **Adversarial**: Jailbreak ASR, Prefill ASR.
- **Pareto axes**: ASR (y) vs **MT-Bench** (x), parameterized by multiplier k. **KL 은 어떤 축에도 나오지 않는다.**

**Headline 숫자 (정확)**:
- KTS alone (no steering): MT-Bench 6.53 → **6.63** (실제로 소폭 상승).
- KTS + steer (k=−0.5): ASR 17.7%, MT-Bench 6.43. Base + steer: ASR 19.3%, MT-Bench 6.05. 즉 KTS 가 MT-Bench drop 을 크게 줄임.
- **가장 깨끗한 matched-effect 비교 (§5.1)**: matched Prefill-ASR=74% 에서 base 는 MT-Bench 4.67, KTS 는 **5.17**. 같은 adversarial 효과에서 capability 가 0.5 포인트 더 높음.
- **"44% jailbreak blocked"** (abstract): KTS + LoRA-DPO **merged** 에서 달성, MT-Bench "almost on par with original".

**Limitation (§5.4)**: Steering 은 LoRA FT 의 대체가 아니라 보완. Single model (Llama-2-chat-7B) 만. **KL 의 evaluation 사용에 대한 caveat 없음** — 왜냐하면 KL 을 evaluation 에 사용하지 않았기 때문.

**결정적 통찰 (세 가지 정정)**:

1. **Attribution 정정**: 우리 "KL-on-benign" 은 Stickland 의 Eq. 2 training objective 를 evaluation 으로 adapt 한 것. 논문 methods 섹션에 명시: *"Following the training objective of Stickland et al. (2024, Eq. 2), we compute `KL(steered || base)` on UltraChat prompts as an evaluation metric. Note that Stickland et al. use this quantity as a distillation loss during KTS fine-tuning and do not report it as a side-effect scalar; our use is an adaptation."*

2. **Direction 명시**: Forward KL `KL(steered || base)` 을 사용 (Stickland 과 일치). 이유: steering-induced novel behavior 를 penalize 하려는 목적에 forward KL 이 적합. Reverse KL 은 mode-dropping 을 penalize 하는데 이건 우리가 측정하려는 것과 다름.

3. **Heavy-tail 주의**: KL 은 natural text 에서 heavy-tailed. Mean 과 median 모두 보고. Trimmed mean 도 고려. Stickland 는 이 문제를 논하지 않았는데 그들의 KL 은 training 중 expectation 안에 있기 때문.

4. **KL 은 capability metric 을 대체하지 않는다**: Stickland 본인들도 MT-Bench 를 user-facing side-effect 로 사용. 우리도 KL 을 보조 metric 으로 쓰고, **MT-Bench 와 MMLU 를 primary capability metric** 으로 유지해야 함.

### A3.4 계획 수정 (A3)

#### 수정 1: Metric 세트 재정의 (Phase 3)

Phase 3 의 preservation metric 세트를 다음으로 확정:

**Primary capability metrics (Stickland 의 실제 metric 따름)**:
- MT-Bench (first-turn, 80 prompts) — **Stickland 의 실제 side-effect metric**, primary
- MMLU 5-shot — cross-validation with CAA / ITI evaluation

**Auxiliary preservation metrics (우리 adaptation)**:
- KL-on-benign: `KL(steered || base)` on UltraChat 500 prompts, forward direction, per-token mean, report mean + median + 90th percentile
- CounterFact neighbourhood specificity (ROME) — primary fact preservation
- Context-memory override rate (Yu et al. 2310.15910) — contrary-fact stress
- CE on WikiText-103 held-out — fluency proxy

**Behavioral (effect) metric**:
- Target concept probe accuracy (linear probe, held-out)
- GPT-4o-mini Behavioral rubric (SAE-TS style) — cross-validation with SAE-TS comparability
- **Scalar summary**: Behavioral × Coherence (SAE-TS convention) **AND** probe accuracy — report both

#### 수정 2: Matched-effect protocol 이중화

Phase 3 에서 두 protocol 을 모두 실행:

**Protocol P3a (SAE-TS style sweep + curve)**:
- 각 method 에 대해 α sweep (log-spaced, 10 points over 3 decades)
- (Behavioral, Coherence) 를 각 point 에 대해 계산
- Pareto curve 를 그리고, peak Behavioral × Coherence 를 scalar summary 로 보고
- SAE-TS Figure 3, Figure 8 과 직접 비교 가능

**Protocol P3b (strict matched-strength, our strengthening)**:
- 각 method 에 대해 binary search 로 α 를 찾아 target coherence = 0.90 (10% drop from baseline 1.00) 에 pin
- 그 α 에서 모든 preservation metric 측정
- Stickland §5.1 matched-ASR 비교의 strengthening 으로 positioning
- 이게 "first strict matched-effect Pareto for attention-level steering" claim 의 근거

두 protocol 모두 논문에 포함. P3b 가 main figure 의 테이블 버전, P3a 가 curve figure.

#### 수정 3: FGA 를 baseline 목록에 추가 (Phase 3 과 Phase 6)

A2.8 의 baseline 목록에 추가:

**FGA** (github.com/ayushgupta4897/FGA):
- 개입 위치: pre-softmax attention score S, layer 20–27 of Llama 3.2 3B (다른 모델은 top 8 of ~30)
- Bias 구축: `G = B_qf · A`, W_K 는 별도 학습
- α = learned sigmoid gate (FGA-FT) 또는 heuristic 0.8/0.2 (FGA-Zero)
- 우리 실험에서 쓸 때는 **FGA-Zero 만** 구현 (FGA-FT 는 2 시간 training 이 필요하고 우리 ontology 와의 integration 이 복잡). Ontology node 를 "entity" 로 취급, ontology embedding 을 V_fact 로 사용, rule-based entity matching 으로 A 구성.
- **Phase 3 에서**: attention-score level baseline (PASTA 와 나란히 보고)
- **Phase 6 에서**: factual grounding baseline (tool schema 를 "facts" 로 취급하여 동일 framework 내에서 비교)

#### 수정 4: Multi-layer range 에 FGA 의 deep-layer 발견 반영

Phase 4 (multi-layer schedule) 의 layer range 를 다음으로 확장:

- **Middle range (Zhu 의 contextual head 위치)**: layer 8–18
- **Deep range (FGA 의 factual grounding 위치)**: layer 20–27
- **Full range sweep**: 각 layer 개별 테스트 후 Pareto plot

**새 가설 H14**: K-bias 는 task 에 따라 optimal layer range 가 다르다 — concept/identity steering 은 middle layer, factual tool disambiguation 은 deep layer 에서 더 효과적. 이게 성립하면 "intervention site 는 task 의 대표가 residual stream 의 어느 layer 에 응결되는지에 따라 결정되어야 한다" 는 mechanistic 결론.

#### 수정 5: Golden Gate narrative 를 SAE-TS Table 1 에 anchor

논문의 motivation section 을 다음으로 재작성:

> SAE clamping based steering (Templeton et al. 2024, "Golden Gate Claude") generates fabrications at high clamp values. SAE-TS (Chalnev et al. 2024, Table 1) provides the first quantitative mechanistic explanation: the decoder direction `d_j` of a target feature `j` does not selectively activate `j` in generation; instead it activates high-density generic features (e.g. feature 6810 with 45% activation density dominates with effect magnitude 4.58). This is the "decoder ≠ causal direction" problem. SAE-TS partially mitigates it via an effect-approximator bias-subtraction term, but high-α collapse persists (Appendix I). We propose that **K-side attention bias avoids this failure mode at the mechanism level**: because K-bias does not inject content into V/MLP, there is no "decoder-vs-causal mismatch" to compensate — the intervention only changes which existing tokens are attended to, not what content is written.

이 narrative 는 SAE-TS 의 발견을 직접 인용하여 우리 thesis 의 motivation 을 강화한다.

### A3.5 Risk 추가 (R10, R11)

**R10**. **FGA 재현이 ontology integration 에서 어려울 수 있음.** FGA 는 entity 가 flat 하고 rule-based matching 을 가정. Ontology 를 "entity" 로 취급하려면 hierarchical matching 이 필요하고, 이건 FGA 의 원래 설계 밖.
- *대응*: FGA-Zero 만 구현, ontology leaf node 만 "entity" 로 취급 (flat 화). Hierarchical matching 은 포기하고 FGA-Zero 를 기준 comparison 으로만 사용. 만약 FGA-Zero 를 구현하기조차 어렵다면 FGA 는 citation 만 하고 baseline 에서 제외.

**R11**. **MT-Bench GPT-4 judge cost**. 80 prompt × 6 methods × 10 α values × 2 runs (variance) = 9600 judge call. API cost 와 시간이 클 수 있음.
- *대응*: 첫 sweep 은 GPT-4o-mini 로 (SAE-TS 가 이걸 사용), 최종 main figure 만 GPT-4 로 재평가. 또는 MT-Bench subset (20 prompt) 으로 축소.

### A3.6 계획 상태 update

#### Phase 0 precondition 상태 (A3 시점)

- [x] 0.1.a Zhu 2025 전문 정독 — 완료 2026-04-09
- [x] 0.1.b ASA 2026 전문 정독 — 완료 2026-04-09
- [x] 0.1.c PASTA 2023 전문 정독 — 완료 2026-04-09
- [x] 0.1.d CAA 2023 전문 정독 — 완료 2026-04-09
- [x] 0.1.e ITI 2023 전문 정독 — 완료 2026-04-09
- [x] 0.1.f Fact Grounded Attention (2509.25252) — **완료 2026-04-09**
- [x] 0.1.g SAE-TS (2411.02193) — **완료 2026-04-09**
- [x] 0.1.h Stickland KTS (2406.15518) — **완료 2026-04-09**
- [ ] 0.2 GGB_experiment padding fix — **다음 할 일**
- [x] 0.3 baseline 목록 확정 — A2.8 + A3.3 에서 확정

### A3.7 현재 Novelty surface (A3 최종)

정독 완료 후 방어 가능한 novelty 축 (정리):

1. **Focus-shift vs content-injection 의 formal dichotomy 와 empirical demonstration** — PASTA 가 용어 없이 instance, 우리가 formalization.
2. **Matched-effect strict-strength Pareto across 6 methods** — SAE-TS 의 curve comparison 을 strengthening. 최초의 strict comparison for attention-level steering.
3. **Ontology-derived direction** — 네 prior (Zhu gradient, CAA/ITI/ASA mean-diff, PASTA user-mark, FGA flat-KB) 어느 것도 ontology 사용 안 함. FGA §6.2.1 에 "hierarchical/compositional" 이 future work 로 명시.
4. **K-only ablation with independent α** — Zhu 는 K+Q joint, 확인.
5. **Training-free task-independent** — Zhu 와 ASA 는 task-specific training 필요, PASTA 는 head profiling 필요, 우리는 ontology 만 있으면 됨.
6. **Multi-layer schedule across middle + deep range** — Zhu middle, FGA deep, 우리가 task-conditional optimal 을 찾음.
7. **K-bias as mechanism-level fix for SAE clamp failure** — SAE-TS Table 1 기반 narrative, K-bias 는 V/MLP 를 건드리지 않아 decoder-vs-causal mismatch 구조적으로 회피.
8. **Homonymous tool disambiguation** — ASA 의 disjoint-domain 가정 밖, 기존 어느 방법도 접근하지 않음.
9. **ITI vs K-bias 의 causal distinction demonstration** — "input-independent bias on W_O" vs "input-dependent bias on K". 첫 empirical 구분 실험.
10. **CAA §9.1 + FGA §6.2.1 가 각각 우리 방향을 future work 로 초대** — 두 개의 official invitation 을 동시에 인용 가능.

초기 survey 의 세 축 (ontology, K-only, fact preservation) 에서 **열 축**으로 확장. Narrow novelty 에 대한 걱정은 더 이상 근거가 없다. 오히려 계획된 실험을 전부 수행하면 **한 논문에 너무 많은 contribution** 이 들어가는 역의 문제가 발생할 수 있다 — Phase 3 가 실패할 경우 축소할 우선순위를 미리 정해두어야 한다.

### A3.8 우선순위 축소 순서 (비상 계획)

Phase 3 (matched-effect Pareto) 가 실패하거나 부분적으로만 성공할 경우, 논문을 축소하는 우선순위:

1. 가장 먼저 버릴 것: **Phase 5 (compositional focus)** — 흥미롭지만 main thesis 와 독립적
2. 다음: **Phase 6 (tool selection)** — 응용 framing 만 잃음, mechanism story 유지
3. 다음: **Phase 4 multi-layer schedule 의 deep range** (FGA 영향) — middle range 만 유지
4. 절대 버리지 말 것: **Phase 1 K-only ablation, Phase 3 matched-effect Pareto, Phase 2 ontology vs gradient 비교, causal distinction (H12)**

최악의 경우 Phase 1 + Phase 3 subset 만으로도 workshop paper 성립 가능.

---

*Amendment A3 끝. 다음 단계: Phase 0.2 (GGB_experiment padding fix) 로 실험 실행 단계에 진입.*


