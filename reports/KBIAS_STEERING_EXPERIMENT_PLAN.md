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

### Phase 0 precondition 상태

- [x] 0.1.a Zhu 2025 전문 정독 — **완료 (2026-04-09)**
- [ ] 0.1.b ASA 2026 전문 정독 — 진행 중
- [ ] 0.1.c PASTA 2023 전문 정독 — 진행 중
- [ ] 0.2 GGB_experiment padding fix
- [ ] 0.3 baseline 목록 확정


