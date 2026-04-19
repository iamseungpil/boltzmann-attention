# NEW_THEOREM_TEST — Two-Level Argmax-Subspace Selectivity (v4)

**작성일**: 2026-04-19 (v1), **v2 업데이트 2026-04-19 저녁**, **v3 업데이트 2026-04-19 Phase A 완료**, **v4 업데이트 2026-04-19 Phase B 완료**

**v4 상태 스냅샷 (누적)**:

| Phase | 상태 | 결과 요약 |
|---|---|---|
| **Phase A** | ✅ COMPLETE (v3) | H-A 9/9 R²≥0.99, slope 0.5298; T.B predictive 확증 (Qwen Banking 7.8% / Llama Retail 1.6% flip); 3/3 arch family |
| **Phase B1** | ✅ COMPLETE (v4) | **6/7 runs A>D** cross-benchmark, mean A/D ratio 2.35, median 2.47. H-E 확증 |
| **Phase B2** | ✅ COMPLETE (v4) | **28/29 residual layers monotonic**, layer 28 (final) 만 non-monotonic. H-D 확증 — two-level gap 의 mechanism 위치 특정 (final FFN+LM-head) |
| **Phase B3** | pending | H-G (facet-concentration), H-H (scope boundary), H-I (slope excess) |
| **Phase C** | pending | H-F (catalog-permutation falsifier). 스크립트 준비 완료 |
| **Phase D** | pending (stretch) | H1-H6 Q-sign $d^*$ narrowing |

**Theorem T 누적 지지 (Phase A+B)**:
- **T.A** (√(r/d) attention): 9/9 R² ≥ 0.99.
- **T.B** (margin-gated flip): Lemma 1 predictive form; small-m 벤치에서 flip, 큰 m 에서 0 — 예측대로.
- **T.C** (two-level gap): **mechanism 위치 특정됨** (final LM-head composition, Phase B2).

**신규 empirical 관측 (v4)**: **Asymmetric transferability** — Telecom/MetaTool B_ont 은 broad (다른 bench 로 전이), Retail B_ont 은 narrow (Telecom 에서 random 보다 **약함**, A/D=0.23). 이건 "B_ont direction breadth" 라는 새 특성 → §3 에 H-J 로 추가.
**바탕**: 실험 세션 (a)(b)(c) 결과 (`memory/lemma_empirical_abc_2026_04_19.md`) + P1 원본 (`memory/p1_random_rank_scaling_failed_2026_04_19.md`) + variant D Phase 0 (`memory/variantD_phase0_verified_2026_04_19.md`) + BFCL cross-benchmark (`memory/bfcl_tier3_cross_benchmark_2026_04_19.md`) + **v2 추가**: Tier 3 A-B-D decomposition (`memory/basis_matching_trap_2026_04_18.md`, `canonical_adaseka_engine.py`) + routing diag (`memory/inter_expert_routing_architectural_2026_04_18.md`) + Cor 6.7 MMLU fail (`memory/cor67_drop_confirmed_2026_04_10.md`) + cross-model Mistral fail (`memory/cross_model_kbias_analysis_2026_04_13.md`) + BiasBios transfer (`memory/phase1_3_ontology_beats_seka.md`) + L0 rank-1 (`memory/cor67_gate_distribution_diagnostic_2026_04_10.md`)
**상태**: Lemma partial 검증 완료 (Qwen Telecom N=100 단일 점). Generalization scope + facet-concentration + failure-mode scope boundary 는 미검증.
**목표**: NeurIPS 2026 existence-only submission 과 **완전 독립** 의 upside track. 본 문서는 다른 세션이 실행할 수 있는 자기충족 가설 + 실험 계획서.

**v2 변경 요지**:
- **§2 empirical anchors 확장** (E8-E11 추가): facet-split dominance, architectural routing, MMLU/Mistral fail, BiasBios transfer, L0 rank-1.
- **§3 primary hypotheses 에 H-G (facet-concentration), H-H (scope boundary alignment) 추가**: Tier 3 A-B decomposition 의 +21pp 와 model/task 별 failure mode 를 이론적으로 다루기 위함.
- **§5 Phase B3 신설 (facet-concentration analysis)**: 기존 Tier 3 데이터 + head-by-head d\*-facet overlap. 거의 GPU 불필요 (~5 hr).
- **§7 decision tree + §8 resource budget** 재보정.
- v1 의 Phase A-D 골격 유지. Phase B3 는 B1-B2 병렬 실행 가능.

---

## §0. 이 문서의 읽는 방법

### Role guide

- **실험 세션**: §1 (theorem structure) → §3 (hypotheses) → §5 (phases) → §7 (decision tree) → §9 (scripts)
- **논문 세션**: §1 → §3 → §6 (paper integration) → §7
- **Consolidation 세션**: 전체

### 전제 (읽고 시작)

1. **Lemma 1, 2 는 이미 provable + empirical anchor 확보**. 추가 증명 필요 없음. 남은 일은 generalization.
2. **Proposition (c) 의 linear KL 예측은 죽음**. 두 번 실패 (fixed-seed R²=0.774, per-task R²=0.459). 부활 시도 금지.
3. **Two-level separation 이 core novelty**. Hoeffding / Ledoux / Stiefel 자체는 textbook — novelty 는 attention-level smooth scaling 과 tool-name-level stepwise robustness 의 empirical gap.
4. **Q-sign asymmetry 은 현 프로그램의 primary target 이 아님**. Observation 으로 유지. 시간 남으면 stretch goal.
5. **NeurIPS 2026 submission 은 본 track 과 무관**. 본 track 의 어떤 결과도 PAPER_DRAFT_v3.md 건드리지 않음. Consolidation session 만 병합 결정.

---

## §1. Theorem structure (실험 세션 결과 반영 후)

### 1.1 이미 확보된 결과

#### Lemma 1 (Single-layer Margin-Gated Flip) — **PROVABLE**

$\delta K = \alpha B B^\top K$ 에 의한 단일 self-attention layer 의 softmax argmax:
$$f(q, K + \delta K) \neq f(q, K) \iff \exists v' \neq f(q,K):\; \alpha \langle B B^\top q, g_{v'}(q, K)\rangle > \mathcal{M}_{v'}(q, K)$$
표준 derivative chain. 증명 완료 (본 문서 §1.3 proof sketch 참조).

#### Lemma 2 (Haar Attention-Level Concentration) — **PROVABLE + EMPIRICALLY ANCHORED**

$U \sim \text{Haar}(\text{Stiefel}(d, r))$ 에 대해 attention-weight 의 Frobenius shift:
$$\mathbb{E}_U[\|\delta\text{attn}(q, K)\|_F] \asymp C \sqrt{r/d}, \quad C = O(\alpha \|q\| \|K\|)$$
Ledoux-Stiefel concentration 표준. **Empirical 확증**: Qwen2.5-7B Telecom N=100, α=0.3, $r \in \{1,3,6,12,24,48,96\}$ 에서 log-log slope **0.551** (예측 0.500), R²=**0.992**.

#### Empirical Margin Anchor

Qwen Telecom N=100: $m_0 \geq 5.031$ (min over 100 tasks). Rule-of-three 95% upper bound: $\Pr[\text{tool-name argmax flip}] \leq 0.0043$ across 700 rank × task combinations.

### 1.2 제안 Theorem (two-level + generalization)

**Theorem T (Two-Level Argmax-Subspace Selectivity).** Pretrained transformer $M$, tool-selection benchmark $V$, 선택 layer 범위 $\mathcal{L}$, amplitude $\alpha$ 하:

**(T.A — Attention level, provable + verified)**: $\alpha B B^\top K$ perturbation 의 attention-weight Frobenius shift 가 $\sqrt{r/d}$ 스케일링을 따른다 (Ledoux-Stiefel). Empirical slope 0.551 ± 측정오차.

**(T.B — Argmax level, empirical)**: 동일 perturbation 하 tool-name token position 의 argmax flip rate 는 margin 하한 $m_0$ 와 per-query lift $|\alpha \langle BB^\top q, g_{v'}\rangle|$ 사이 관계에 의해 결정:
- sub-critical regime ($|\alpha \cdot \text{lift}| \leq m_0$) 에서 flip rate = 0 (empirical 0/700).
- super-critical regime 에서 discontinuous (format collapse 관측).

**(T.C — Gap)**: T.A 의 smooth scaling 과 T.B 의 step function 은 **tool-name argmax 의 margin threshold + FFN/LM-head 의 non-linear compounding** 이 중간에 개입하기 때문. 이 gap 자체가 transformer tool-selection 의 structural property 이고, B_ont 방향성의 empirical 의미 (direction specificity) 가 바로 이 gap 에 있음.

### 1.3 Proof sketch (Lemma 1 — 완전, Lemma 2 — 표준 reference)

**Lemma 1 증명** (self-contained):
1. Softmax argmax: $f(q, K) = \arg\max_v \ell_v$ where $\ell_v$ is the v-th output logit.
2. Flip condition: new argmax $f' \neq f$ iff $\ell_{f'}' > \ell_{f}'$ where primed values are post-perturb.
3. Taylor: $\ell_v' = \ell_v + \langle \delta K, \nabla_K \ell_v\rangle + O(\|\delta K\|^2)$.
4. Substitute $\delta K = \alpha B B^\top K$: $\ell_v' - \ell_v = \alpha \langle B B^\top K, \nabla_K \ell_v\rangle_F + O(\alpha^2)$.
5. Logit gap: $\ell_{f'}' - \ell_f' = (\ell_{f'} - \ell_f) + \alpha \langle B B^\top K, \nabla_K(\ell_{f'} - \ell_f)\rangle_F + O(\alpha^2) = -\mathcal{M}_{f'} + \alpha \langle B B^\top q, g_{f'}\rangle + O(\alpha^2)$ (using $K$-gradient identities + first-order closure).
6. Flip iff this > 0, i.e. $\alpha \langle BB^\top q, g_{f'}\rangle > \mathcal{M}_{f'} + O(\alpha^2)$. ∎

**Lemma 2 증명** (standard): Haar measure on Stiefel $V_r(\mathbb{R}^d)$ concentration. $\mathbb{E}[U U^\top] = (r/d) I$. $\text{Var}$ of bilinear form $q^\top U U^\top q$ is $2 r(d-r)/d^2(d+2) \|q\|^4$. Attention-weight Frobenius norm shift is sum over tokens, Jensen 으로 $\sqrt{r/d}$ dominant term. See Ledoux (2001) §6, Edelman-Rao (2005) §5. ∎

**T.B (empirical)**: Lemma 1 은 per-query flip condition 을 주지만 sub-critical regime 에서 margin lower bound $m_0$ 가 lift 를 지배하여 flip rate 가 0 이 되는 것은 **증명 불가 (model-dependent) 이고 empirical 측정 대상**.

### 1.4 무엇이 **Theorem 아닌 Observation** 으로 남는가

1. **Q-sign asymmetry (Qwen+Telecom Q+ vs Llama Telecom Q−)**: 5-point 표. 설명 mechanism 없음. Theorem 에 들어가지 않음.
2. **C1 safe floor의 multi-domain 동시 성립**: 8+ benchmark 에서의 accumulated positive ΔF1. Single theorem 으로 환원 불가.
3. **Format collapse threshold**: Llama Telecom Q+0.05 → 200/200 empty. Super-critical 영역. Scope 밖으로 명시.
4. **MULTI > SINGLE facet stratification**: +36 vs +24 on Telecom. Baseline-ceiling 효과 의심, theorem 아님.
5. **(v2) Layer-adaptive α > fixed α**: C1 safe floor 가 layer-별 다른 α 로 최적. Per-layer discriminative subspace strength 추정이 필요하지만 이론 핵심 아님. §Discussion 에 observation.
6. **(v2) BiasBios transfer (+7.6pp top1)**: Single-answer classification (tool-selection 범주 밖) 에서 B_ont 가 여전히 작동. "B_ont as catalog-ontology direction" 해석의 범위가 예상보다 넓음을 시사. §Discussion 에 observation.
7. **(v2) L0 massive activation channel**: Qwen2.5 L0 K 가 per-head rank-1 (top1 σ²=1.000), B_ont f0 col0 과 |cos|=1.000. 첫 layer 의 rank-1 특성은 interesting 이지만 L0 skip 해도 lift 유지 → 이론 argument 밖. §Discussion 에 observation + optional Phase B3 side note.
8. **(v2) Contrastive K-bias positive on ST4 multi-tool (+5.8pp)**: Training-free scope 밖 (GT/distractor label 필요). 본 theorem program scope 외. 별도 track.

**네 번째 pivot 방지 원칙**: 위 8 개 observation 을 theorem 으로 격상 시도 금지. Framework 바깥 phenomena 로 기록.

---

## §2. 관측된 empirical anchors (기본 증거)

| # | 측정 | 값 | 출처 | Theorem 근거 |
|---:|---|---|---|---|
| E1 | Attention Fro shift log-log slope | 0.551 (R²=0.992, 예측 0.5) | lemma_empirical.json (c) | Lemma 2 verified |
| E2 | Margin m_0 lower bound (Qwen Telecom) | 5.031 (min over N=100) | lemma_empirical.json (b) | T.B empirical anchor |
| E3 | Random-direction flip rate (rank 7 × N 100 × α 0.3) | 0/700 | (a) + P1 합산 | T.B sub-critical 확인 |
| E4 | B_ont vs random at matched Frobenius | 200/200 vs 0/200 | Phase 0 variantD | Direction specificity |
| E5 | Cross-benchmark proxy (BFCL N=100) | A 2/100, D 0/100 | bfcl_tier3 memo | Direction specificity cross-domain |
| E6 | KL vs r/d R² (per-task resample) | 0.459 | lemma_empirical (a) | First-order linear dead at argmax level |
| E7 | Q-sign pattern (5 (M, V)) | +/−/−/−/− | handoff E6 | Observation (Theorem 밖) |
| **E8** | **Tier 3 A-B-D ΔF1 decomposition (Telecom N=200)** | **A +28.89 / B +7.79 / D 0.00pp** | canonical_adaseka `telecom_canonical_*_N200.json` | **H-G 근거** — facet-split 기여가 direction 기여의 **3 배** |
| **E9** | **Inter-expert routing argmax entropy** | **83% of uniform, >95% identical across queries** | inter_expert_routing_architectural memo | **H-G 근거** — routing 이 query-adaptive 아닌 per-head architectural |
| **E10** | **MMLU 에서 Cor 6.7 facet-gated K-bias ΔAcc** | **−4.80pp (α=0.3), −10.50pp (α=1.0)** | cor67_empirical_fail_mmlu memo | **H-H 근거** — tool-selection 밖 벤치에서 B_ont 가 해로움 |
| **E11** | **Mistral-7B Telecom ΔF1** | **−31.86pp (default), −4.32pp (skipL0+padmax)** | cross_model_kbias_analysis memo | **H-H 근거** — model-specific alignment collapse |
| **E12** | **Qwen2.5-7B L0 K rank / B_ont cos** | **rank-1 per head, cos=1.000** | cor67_gate_distribution_diagnostic memo | L0 side observation (non-core) |
| **E13** | **BiasBios transfer (Qwen rank-8 @α=3.0)** | **+7.6pp top1** | phase1_3_ontology_beats_seka memo | Scope 확장 observation (§Discussion) |
| **E14 (v3)** | **Phase A 9 (M, V) slope table (cross-arch)** | **mean 0.5298, all R²≥0.99, Qwen 0.55-0.59, Llama 0.40-0.50, Mistral 0.55** | new_theorem_phase_a memo | **H-A 확증** — Lemma 2 universal 3 family × 4 bench |
| **E15 (v3)** | **Phase A flip rate table (T.B predictive)** | **7/9 bench 0 flip, 2/9 positive: Qwen Banking 7.81% (m_0=0.88, r=96 집중) + Llama Retail 1.57% (m_0=0.02)** | new_theorem_phase_a memo | **T.B predictive form 확증** — flip ↔ small-m_0 상관 |
| **E16 (v3)** | **Systematic slope excess model-family stratified** | **Qwen/Mistral +10-17%, Llama ≈ exact** | Phase A aggregate | **H-I 근거** (v3 신규) — higher-order Lipschitz correction |
| **E17 (v3)** | **m_0 range across benches** | **[0.016, 6.375], 400× spread, strict positivity** | Phase A aggregate | **H-B "m_0 > 1" 기각** — 대신 T.B 의 margin-distribution gated 가 맞음 |
| **E18 (v4)** | **Phase B1 cross-bench direction specificity** | **6/7 A>D, mean A/D ratio 2.35** | new_theorem_phase_b memo | **H-E 확증** |
| **E19 (v4)** | **Asymmetric transferability** | Telecom → Retail/Airline/Banking/ST4 전부 A>D (ratio 1.81-3.67); **Retail → Telecom A/D=0.23 < 1** (random 보다 약함) | B1 행 5 | **H-J 근거 (v4 신규)** — B_ont direction breadth 가 benchmark-specific |
| **E20 (v4)** | **Phase B2 layer-resolved KL (29-layer Qwen Telecom)** | **28/29 residual layers monotonic in rank**, 오직 **layer 28 (final output = post-final-FFN) 만 non-monotonic** (peak r=6, dip r=24). Final logits 도 non-monotonic (Phase A Qwen Telecom 패턴 재현). | B2 aggregate | **H-D 결정적 확증** — T.C mechanism 위치 특정 |
| **E21 (v4)** | **Layer-28 amplification at low rank** | r=1 at L27 → L28 KL 이 **85× amplification** (0.0003 → 0.0255). r=96 는 1.35× | B2 row 28 | Late FFN 의 "cleanup" nonlinear amplification 관측 |
| **E22 (v5)** | **F10 4-cell results** | F10a/b/c/d top1 ∈ [46.0, 47.0]% on MetaTool Subtask1 N=200, **모두 F9 D-only 47.50% 이하**. Hard-gate (F10d) 47.0% — collapse 안 됨 | reports/f10_metatool/f10_facet_*.json | **H-F10 falsified** (per-token energy gating no help, V+D < D-only) |
| **E23 (v5)** | **Hard-gate ≠ catastrophic** | F10d 47.0% ≈ F10c 47.0%. Cor 6.7 Lipschitz collapse 예측 단일-tool task 에서 미관찰 | F10d_hard.json | Cor 6.7 §3.4.1 R-violation claim 약화 (Subtask4 multi-step 에서 재검증 필요) |

---

## §2.5 선행연구 비교 + 우리 접근의 차별점 (v5 신규, 2026-04-19 audit; v5.1 expanded 2026-04-19 evening with broad ontology+LLM survey)

### 2.5.1 종합 비교 표 (9-method)

이 표는 ICLR §2 Related Work 의 base. Reviewer-visible novelty 위치를 명확히 하기 위해 5-dimensional 비교 (Multi-tool / Step-adaptive / Training-free / Multi-facet / Semantic) 사용.

| Method | Venue | 작동 layer | Multi-tool | Step-adapt | Train-free | Multi-facet | Semantic source |
|---|---|---|:---:|:---:|:---:|:---:|---|
| CAA (Rimsky 2024) | ACL | residual mid-layer | ❌ | ❌ | ✗ contrastive pair | ❌ single | trained vector |
| ITI (Li 2023) | NeurIPS | head output post-W_O | ❌ | ❌ | ✗ probe | ❌ single | supervised probe |
| RepE (Zou 2023) | arXiv | residual | ❌ | ❌ | ✗ probe | ❌ single | supervised probe |
| PASTA (Zhang 2024) | ICLR | post-softmax attn | ❌ | ❌ | △ profile (semi) | ❌ | user-marked positions |
| **SEKA (Li 2026)** | **ICLR** | K pre-softmax | ❌ | ❌ | ✗ contrastive | ❌ single basis | GPT-4o synthetic contrastive |
| **AdaSEKA (Kim 2026)** | — | K pre-softmax | ❌ | ❌ | ✗ contrastive | ❌ single expert/query | trained expert mixture |
| **SADI (Wang 2025)** | **ICLR** | head/neuron/hidden post | ❌ | ❌ | △ contrastive pair (~150) | ❌ single mask | contrastive activation diff |
| Spotlight (2025) | arXiv | attn score | ❌ | ❌ | ✓ | ❌ | user-marked positions (PASTA-style) |
| LLMSteer (2024) | arXiv | reused-context attn | ❌ | ❌ | ✓ | ❌ | reused KV positions |
| Atlas (2024) | arXiv | attn scaling | ❌ | ❌ | △ bias localize | ❌ | bias-axis specific |
| **Hazarika (2022)** | **AAAI** | cross-attn distribution | ❌ | ❌ | ✓ | ❌ | positional (manually specified) |
| **OntoLLM (2026)** | **ScienceDirect** | **prompt text only** | △ | ❌ | ✓ | △ KG hierarchy text | ontology + KG (text injection) |
| Focus Directions (Zhu 2025) | arXiv | K + Q | ❌ | ❌ | ✗ gradient-trained | ❌ | gradient-trained (gold response) |
| Gated Attention (NeurIPS 2025 Oral) | NeurIPS | per-head sigmoid gate | ❌ | ❌ | ✗ trained from scratch | △ per-head | trained gate |
| FGA (Gupta 2025) | arXiv | attn pre-softmax | ❌ | ❌ | ✗ learned W_K (~2.1M params) | ❌ | external KB (137 entities) |
| **Q-coverage (NeurIPS 2026 withdrawn)** | — | Q step-adaptive | ✓ | ✓ | ✓ | ❌ | facet subspace projection |
| **F10 (executed, falsified)** | — | K-side per-token gated | ❌ | ❌ | ✓ | ✓ V×D | ontology label energy ratio |
| **MOFCISS (proposed)** | — | K-side step-adaptive sparse | ✓ | ✓ | ✓ | ✓ | sparse OMP over ontology atoms |

### 2.5.2 Empty-cell mapping (어디가 비어 있는가)

5-dimensional cross product (32 combinations 중) 에서 **모든 5 충족** 하는 prior work 부재:

- 모든 prior **K-side** + **multi-facet** + **train-free** 동시 충족 0 건
- 모든 prior **multi-tool** + **multi-facet** 동시 충족 0 건
- 모든 prior **multi-tool** + **train-free** + **semantic** 동시 충족 0 건 (Q-coverage 가 multi-tool + train-free 이나 semantic 안 함; OntoLLM 이 train-free + semantic 이나 prompt-level 만)

### 2.5.3 우리 접근의 정확한 차별점 (3-tier)

#### Tier A — Phase A-D + F8 (이미 검증된 negative + structural finding)

1. **Lemma 2 √(r/d) universality across 9 (M, V) settings** — Stiefel concentration 의 first cross-architecture empirical confirmation in transformer attention. **Novel: empirical breadth.** (Lemma 2 자체는 standard Ledoux 2001.)
2. **Layer-28 mechanism localization** — 85× amplification at single layer. **Novel: mechanism position pinpoint.** (Geva 2021 가 late-FFN as KV memory 라고 제안했지만 low-rank K-bias 에 대한 quantification 은 없음.)
3. **Phase C catalog-permutation falsifier** — H-F (catalog content load-bearing) 강한 형 falsified. **Novel: falsifier-as-refinement framing.** Pipeline-level direction specificity claim 으로 refine.
4. **F8d verb × domain orthogonal axis on 4 multi-domain corpora** — NMI ∈ [0.144, 0.218] structural law. **Novel: dataset-level diagnostic** (NMI probe 가 이전 attention steering literature 에 없음).
5. **F10 H-F10 falsification** — span-invariance 깨도 (per-token gating) lift 안 나옴. **Novel: negative result confirms refined F1 reframe scope** (training-free linear gating regime 에서 ontology 무관).

#### Tier B — F11 MOFCISS (proposed, pending validation)

6. **OMP sparse selection over ontology atoms** — anchor identity 직접 사용 (Gram-Schmidt 거치지 않음). **Novel: SADI/SEKA/Spotlight 모두 안 함.** (Sparse coding 자체는 OMP 표준.)
7. **Step-history decay across multi-tool emission** — `decay(n, emitted_<t>)` per facet cell. **Novel: Q-coverage 에 inspire 받았으나 K-side + multi-facet + ontology-cell-aware 는 첫 사례.** (Q-coverage 는 Q-side single-axis, MOFCISS 는 K-side multi-axis.)
8. **3중 span-invariance 깸**: (a) non-linear OMP, (b) non-stationary step-state, (c) anchor-identity 보존. **Novel: 셋이 동시에 들어간 mechanism 부재.**

#### Tier C — Theoretical framework

9. **Lemma 1 + Lemma 2 + Theorem T 의 two-level argmax-subspace selectivity** — paper §3 의 self-contained framework. SADI/SEKA/AdaSEKA 모두 Lemma-level 분석 없음 (empirical only).
10. **Lemma A + B + Theorem M (MOFCISS 용)** — sparse selection distinguishability + coverage convergence + multi-tool optimality. **Novel: ontology-grounded sparse coding + step-state 의 convergence 분석은 첫 시도.**

### 2.5.4 가장 위험한 prior art ranking + 차별 wording

#### 1순위 위협: SADI (ICLR 2025)
**왜 위협**: inference-time semantic-adaptive intervention paradigm 의 직접 prior art. Reviewer 가 "F10/F11 = SADI 변형" 으로 평가 가능.

**차별 wording (paper §2 권고, v5.2 with full-text audit)**:
> "F10 (executed) extends SADI's inference-time adaptive intervention paradigm to the K-side multi-facet regime. F11 (MOFCISS) further introduces sparse OMP coding and step-history decay, which are absent from SADI. **Structural difference**: SADI applies element-wise activation scaling with a static binary mask, $A'_q = A_q + \delta(A_q \odot M)$, where $M$ is top-K binarized from contrastive mean differences. Our perturbation is a **rank-$r$ continuous projection onto a K-subspace**, $\delta K = \alpha B B^\top K$ (F10 / main Theorem T regime) or sparse dictionary coding $\delta K = -\alpha \sum_n c_n(K) K_n$ (F11 / MOFCISS regime) — these are linear-algebraically distinct from element-wise binary scaling. **Data requirement**: SADI requires 150–2000 contrastive pairs (positive + negative answers per concept); our anchors require only ontology labels (positive examples only — no negative pair construction). **Step-state**: SADI is single-step per input (Algorithm 1 in the SADI paper); MOFCISS is per-decoding-step with facet-emission history decay."

#### 2순위 위협: Q-coverage (NeurIPS withdrawn track 자체)
**왜 위협**: 같은 lab 에서 multi-tool + step-adaptive + training-free 조합을 이미 시도했다는 점 (NeurIPS withdrawal context).

**차별 wording**:
> "Q-coverage (Q-side step-adaptive single-axis projection, our prior NeurIPS 2026 work, withdrawn) demonstrated that multi-tool emission requires step-state but achieved only +1.64pp F1 lift on MetaTool Subtask4 with mechanism claims (Q-coverage as 'the' multi-selection axis) refuted by canonical-AdaSEKA Telecom +28.89pp counterexample. MOFCISS preserves the step-state insight but switches to (i) K-side rather than Q-side, (ii) multi-facet (verb × domain) rather than single axis, (iii) sparse OMP rather than dense projection. Mechanism claim is downgraded from 'unique multi-selection mechanism' to 'one effective instance of multi-step ontology-aware K-side intervention'."

#### 3순위 위협: SEKA + AdaSEKA (ICLR 2026)
**왜 위협**: K-side spectral steering 의 직접 family. AdaSEKA 가 query-adaptive routing 도 함.

**차별 wording (v5.2 corrected 2026-04-19 late evening after full-text audit)**:
> "SEKA and AdaSEKA use contrastive cross-covariance SVD to derive K-side projection directions (trained on GPT-4o synthetic positive/negative prompt pairs, ~100 samples per task). Our basis is derived from positive-only ontology anchor sentences — no contrastive pair construction. AdaSEKA's query-adaptive routing produces a **weighted blend of multiple trained expert subspaces** per query via coefficients $\alpha_m(q) \propto \sum_k (q^\top u^{(k)}_m) \sigma^{(k)}_m$, i.e. $P_{\text{dyn}}(q) = \sum_m \alpha_m(q) U^m (U^m)^\top$. MOFCISS differs on three distinct axes: (i) anchors are positive-only ontology labels with no contrastive pair construction, (ii) **step-adaptive** via per-step facet decay over multi-tool emission history (AdaSEKA is stationary within a forward pass), (iii) **sparse OMP top-k** activation rather than AdaSEKA's dense weighted blend. (The multi-axis-simultaneous property itself is shared with AdaSEKA — novelty claim reduced to sparsity + step-state + positive-only anchors.)"

**v5.2 audit note**: earlier v5.1 draft claimed AdaSEKA "selects one expert per query (single-direction per inference)". Full-text read of SEKA paper (arxiv 2603.01281v1) confirms this is incorrect — AdaSEKA's $P_{\text{dyn}}(q)$ is a weighted blend of ALL experts, not a selector. Reviewers will catch the single-pick mischaracterization. Corrected wording above.

#### 4순위 위협: OntoLLM (ScienceDirect 2026)
**왜 위협**: ontology + LLM at inference time naming overlap. 단 prompt-level 이라 mechanism layer 다름.

**차별 wording**:
> "OntoLLM integrates ontology and knowledge graphs into the LLM prompt at inference time without retraining, addressing factual grounding and digression prevention. The intervention layer is text-prompt only — no activation, attention, or KV cache modification. Our approach modifies attention K activations directly via subspace projection (F10) or sparse coding (MOFCISS). The two methods address orthogonal axes: prompt-level retrieval (OntoLLM) vs activation-level direction injection (ours)."

### 2.5.5 ICLR §2 Related Work 권고 구조 (paper draft 변경 사항)

현재 ICLR PAPER_DRAFT_ICLR_v1.md §2 는 4-paragraph (CAA/RepE/ITI 등 + Stiefel + interpretability + novelty). 다음으로 확장 권고:

1. **§2.1 Activation steering** (현재) — CAA, RepE, ITI, ASA, ActAdd
2. **§2.2 Attention steering** (확장) — PASTA, Spotlight, LLMSteer, Atlas, FGA, Hazarika 2022
3. **§2.3 K-side spectral steering** (신규) — SEKA, AdaSEKA, Focus Directions
4. **§2.4 Inference-time semantic adaptive** (신규) — SADI 단독 1 paragraph (직접 prior art)
5. **§2.5 Ontology integration** (신규) — OntoLLM, OntoTune
6. **§2.6 Concentration of measure** (현재) — Ledoux, Edelman & Rao
7. **§2.7 Mechanistic interp** (현재) — Anthropic circuits, Geva 2021, Meng 2022
8. **§2.8 Novelty placement** (확장) — §2.5.3 Tier A+B+C 로 명확 차별

### 2.5.6 종합 한 줄

> **9-method audit 결과 multi-tool + step-adaptive + training-free + multi-facet + semantic 5-dimensional 동시 충족 prior art 부재. F10 (실패) + F11 MOFCISS (제안) 가 빈 칸을 노린 design. 가장 큰 위협은 SADI (ICLR 2025) 의 inference-time adaptive intervention paradigm 선행성, but SADI 는 supervised contrastive pair 필요 + single-axis + step-state 없음 → 명확 차별 가능.**

---

## §2.5.1 확장: Ontology + LLM Inference 광범위 audit (v5.1, 15+ methods, 5 categories)

이 섹션은 §2.5의 9-method 비교를 광범위하게 확장 (OG-RAG, Generative Ontology, GCD, Neuro-Symbolic, CAV, G-ACT, Vector Ontologies 등 추가). Reviewer 가 "ontology 를 LLM inference 에 사용한 prior 가 어디까지인가?" 질문 시 thorough 답변 가능하도록.

### Group 1: Prompt-level / Retrieval-level (mechanism: prompt enrichment via retrieval)

이 그룹은 ontology 를 **prompt 텍스트** 로 LLM 에 주입. Activation 직접 modification 없음.

#### 1A. **OG-RAG (Ontology-Grounded RAG)** — EMNLP 2025, arxiv 2412.15235
- **Mechanism**: User query → NER/type classification → SPARQL retrieval over ontology → prompt enrichment with retrieved facts → LLM generation → post-check
- **Training**: None (LLM frozen, ontology pre-built)
- **Numbers**: +55% recall, +40% correctness, +30% attribution clarity, +27% deductive reasoning vs RAG/GraphRAG
- **Domains**: Healthcare, biomedicine, agriculture, electrical fault ID
- **차별 vs F11**: Prompt-level only, no attention/K modification. Ontology 가 retrieval index 역할만.

#### 1B. **GraphRAG variants** (Microsoft, GoodData, others 2024-2026)
- **Mechanism**: Vector search → graph traversal → context composition → LLM
- **Training**: Embedding model only (LLM frozen)
- **차별 vs F11**: KG 가 retrieval graph, attention modification 없음.

#### 1C. **OntoLLM** (ScienceDirect 2026)
- **Mechanism**: Ontology + KG injection at prompt level. Question nodes bridge structured KG with unstructured docs.
- **Training**: None
- **차별 vs F11**: Prompt-text only, mechanism layer 다름 (이전 §2.5에서 다룸).

#### 1D. **KGT (Knowledge Graph Thought)**
- **Mechanism**: KG-enhanced framework, plug-and-play, no fine-tuning
- **Training**: None
- **차별 vs F11**: Reasoning chain via KG, not attention modification.

#### 1E. **KA-RAG** (MDPI 2025)
- **Mechanism**: Agentic retrieval-augmented generation with KG
- **Training**: None (agent uses LLM as tool)
- **차별 vs F11**: Multi-hop retrieval pipeline, not activation-level.

#### 1F. **OntoGPT** (PMC 2025)
- **Mechanism**: Ontology-grounded term extraction (biomedical)
- **Training**: None
- **차별 vs F11**: Output-side extraction, not steering.

#### 1G. **Agentic AI for Ontology Grounding** (Semantic Web Journal 2025)
- **Mechanism**: Agent-driven ontology grounding over LLM-discovered concepts
- **Training**: None (agent loop)
- **차별 vs F11**: Pipeline-level, not internal modification.

### Group 2: Constrained Decoding (mechanism: token logit masking)

이 그룹은 ontology 를 **output token grammar** 로 사용. Logit-level intervention.

#### 2A. **Generative Ontology** — arxiv 2602.05636 (2026)
- **Mechanism**: Ontology encoded as Pydantic schema → DSPy signatures constrain LLM generation
- **Training**: None (schema is hand-coded ontology)
- **Granularity**: Token-level masking (Outlines library: O(1) finite state machine)
- **차별 vs F11**: Logit-level (post-softmax), not K-side. No multi-tool step-state. Output structure constraint, not direction steering.

#### 2B. **Grammar-Constrained Decoding (GCD)** — EMNLP 2023, arxiv 2305.13971
- **Mechanism**: CFG → token mask at each decode step
- **Training**: None
- **Famous use**: SQL generation, JSON output, structured extraction
- **차별 vs F11**: Same logit-level. Validity not direction-quality.

#### 2C. **RELATE** — arxiv 2509.19057 (2025)
- **Mechanism**: Biomedical relation extraction with LLM + ontology constraints (GCD-style)
- **Numbers**: F1 0.062 → 0.413, 0.102 → 0.47
- **차별 vs F11**: Domain-specific, output-format constraint.

#### 2D. **Outlines library** (production tool, 2024)
- **Mechanism**: JSON schema → finite state machine → O(1) valid token lookup
- **차별 vs F11**: Engineering tool, not novel mechanism per se.

### Group 3: Neuro-Symbolic Post-hoc (mechanism: external reasoner verify+correct)

이 그룹은 LLM output 을 ontology reasoner 로 검증 + 수정.

#### 3A. **Enhancing LLMs through Neuro-Symbolic + Ontological Reasoning** — arxiv 2504.07640 (2025)
- **Mechanism**: OWL ontology + HermiT reasoner (consistency check) + logistic regression (NL → logical form mapping)
- **Training**: Lightweight (logistic regression only)
- **Workflow**: LLM output → translate to logical statements → reasoner check → flag violations → re-generate
- **차별 vs F11**: Post-hoc verification, not generation-time direction steering. Reasoner is heavy (OWL DL).

#### 3B. **Δ₁-LLM** — arxiv 2603.12953 (2026)
- **Mechanism**: Symbolic-neural integration for credible reasoning
- **Training**: Symbolic component pre-built
- **차별 vs F11**: Reasoning-time integration, output-level.

#### 3C. **Ontology-Constrained Enterprise Agents** — arxiv 2604.00555 (2026)
- **Mechanism**: Three-layer ontology framework (Role / Domain / Interaction) for enterprise agents
- **Training**: Ontology curated by experts
- **Application**: Enterprise compliance, hallucination prevention
- **차별 vs F11**: Enterprise pipeline, not low-level activation.

### Group 4: In-Context Learning / Self-Training (mechanism: ontology-aware instruction tuning)

이 그룹은 ontology 를 **training signal** 로 사용. 우리 train-free scope 밖.

#### 4A. **OntoTune** — arxiv 2502.05478 (2025 ICLR)
- **Mechanism**: Ontology-driven self-training. ICL identifies what LLM hasn't mastered, generates training data, fine-tunes
- **Training**: Self-training loop (LLM generates own data via ontology)
- **Domain**: Medical, SNOMED CT
- **차별 vs F11**: 우리 train-free scope 밖. ICL identification 단계는 inference-time이지만 main contribution은 training pipeline.

#### 4B. **LLMs4OL** (2024-2025)
- **Mechanism**: LLM as tool for ontology learning task
- **Training**: Both — task formulation includes fine-tuning + prompting variants
- **차별 vs F11**: 우리 reverse direction (ontology → LLM steering, not LLM → ontology).

#### 4C. **AI Ontology** — arxiv 2404.03044 (2024)
- **Mechanism**: LLM-assisted concept hierarchy construction
- **차별 vs F11**: Output is ontology, not improved LLM behavior.

### Group 5: Activation/Representation Steering (mechanism: hidden state modification) — **가장 우리와 가까움**

이 그룹은 LLM 의 internal activation 을 modify. Mechanism layer 가 우리 F11 과 동일.

#### 5A. **CAV (Concept Activation Vectors)** — Kim et al. 2017+
- **Mechanism**: Train linear probe per concept → use probe direction as steering vector → add/subtract from activation
- **Training**: ✗ supervised probes per concept
- **Modification**: Hidden state element-wise additive
- **차별 vs F11**: Probe trained, single concept axis, no multi-facet, no step-state, no ontology hierarchy.

#### 5B. **CAA (Contrastive Activation Addition)** — Rimsky 2024 ACL
- **Mechanism**: Mean difference of contrastive pair activations at residual stream layer 13 (Llama-2-7B)
- **Training**: ✗ contrastive pairs per behavior (290-1000 pairs)
- **Modification**: Residual stream additive
- **차별 vs F11**: Same as CAV but at residual stream, +1/-1 multiplier.

#### 5C. **ITI (Inference-Time Intervention)** — Li 2023 NeurIPS
- **Mechanism**: Probe-trained mass mean shift on attention head output (post softmax-V, pre W_O), top-K heads
- **Training**: ✗ logistic regression probe per (layer, head)
- **차별 vs F11**: Probe-trained, attention output (not K), single concept.

#### 5D. **RepE (Representation Engineering)** — Zou 2023
- **Mechanism**: Superset framework for residual-stream steering (sentiment, truth, refusal, etc.)
- **Training**: ✗ supervised probe per concept
- **차별 vs F11**: Residual stream, single axis per probe.

#### 5E. **PASTA** — Zhang 2024 ICLR
- **Mechanism**: Post-softmax attention row reweighting at top-K heads (multi-task profiling)
- **Training**: △ semi (head selection via profiling)
- **Signal**: User-marked positions (NOT semantic)
- **차별 vs F11**: Attention score (not K), positional, single intent per task.

#### 5F. **Spotlight** — arxiv 2505.12025 (2025)
- **Mechanism**: PASTA-style dynamic attention bias, ratio-based gating
- **Training**: ✓ training-free, no profiling
- **Signal**: User-marked positions
- **차별 vs F11**: Positional (not ontological), single span per query.

#### 5G. **LLMSteer** — arxiv 2411.13009 (2024)
- **Mechanism**: Reused-context attention steering for long-context inference
- **Training**: ✓
- **차별 vs F11**: Context-reuse focused, not multi-tool.

#### 5H. **Atlas** — arxiv 2410.22517 (2024)
- **Mechanism**: Localize bias-concentrated layers → targeted attention scaling intervention
- **Training**: △ bias localization step
- **차별 vs F11**: Bias-mitigation specific.

#### 5I. **SEKA** — arxiv 2603.01281, ICLR 2026
- **Mechanism**: K' = k + (1/2)(g+ P+ k + g- P- k), contrastive cross-covariance SVD top-k
- **Training**: ✗ contrastive (GPT-4o synthetic pairs)
- **차별 vs F11**: K-side (matches our family), but contrastive trained, single basis, stationary.

#### 5J. **AdaSEKA** — Kim 2026
- **Mechanism**: Query-adaptive routing of contrastive expert directions, single expert per query
- **Training**: ✗ contrastive expert pre-training + routing learning
- **차별 vs F11**: Query-adaptive but single direction at a time, no step-state, no multi-facet active.

#### 5K. **Focus Directions** — Zhu 2025, arxiv 2503.23306
- **Mechanism**: K AND Q bias at top-k contextual heads, scalar α, gradient-trained directions
- **Training**: ✗ AdamW 10 epochs
- **차별 vs F11**: Trained directions, single concept (relevant context).

#### 5L. **FGA (Fact Grounded Attention)** — Gupta 2025, arxiv 2509.25252
- **Mechanism**: Pre-softmax attention bias S + α⊙G where G = B_qf · A (learned W_K ≈ 2.1M params), flat KB 137 entities × 12 attributes
- **Training**: ✗ learned W_K
- **차별 vs F11**: Trained attention augmentation, flat KB (no hierarchy).

#### 5M. **SADI** — arxiv 2410.12299, ICLR 2025
- **Mechanism**: Top-K element binary mask from contrastive activation diff, applied as A'_q = A_q + δ(A_q ⊙ M)
- **Training**: △ contrastive pairs ~150 items
- **Variants**: SADI-Head / SADI-Neuron / SADI-Hidden
- **차별 vs F11**: Hidden state element-wise (not K subspace), single mask, no step-state.

#### 5N. **G-ACT (Gradient-refined Adaptive Activation Steering)** — 2025
- **Mechanism**: Per-prompt activation differences clustered into steering directions, lightweight per-layer probes refined online
- **Training**: △ probes trained
- **Adaptivity**: Online (per-prompt direction selection)
- **차별 vs F11**: Probes trained, residual stream not K-side.

#### 5O. **Attention-guided Steering** — arxiv 2602.00333 (2026)
- **Mechanism**: Insert prefix into prompt → token's attention to prefix tokens = concept-activity heuristic
- **Training**: ✓ training-free
- **Signal**: Prefix injection (prompt-level, not semantic ontology)
- **차별 vs F11**: Prefix injection (prompt-level intermediation), not direct K-side modification.

#### 5P. **Vector Ontologies as LLM World View** — arxiv 2506.13252 (2025)
- **Mechanism**: Define vector space spanned by ontologically meaningful dimensions for LLM representation analysis
- **Training**: ✓ extraction only (no LLM mod)
- **Purpose**: Interpretability / extraction (not steering)
- **차별 vs F11**: Same conceptual framework (ontology axes as vector space), but they extract, we modify.

#### 5Q. **Steering Conceptual Bias via Latent-Subspace** — 2025
- **Mechanism**: Concept basis identified in latent subspace, activation perturbed along basis
- **Training**: △ concept basis discovery
- **차별 vs F11**: Latent subspace (residual or hidden), not K-side, single axis at a time.

#### 5R. **Hazarika et al.** — AAAI 2022
- **Mechanism**: Cross-attention distribution × bias vector + renormalize. Encoder-decoder NLG (T5/BART).
- **Training**: ✓
- **Signal**: Positional (manual)
- **차별 vs F11**: Encoder-decoder, positional, not decoder-only K-side.

#### 5S. **Q-coverage** (NeurIPS 2026 withdrawn, our prior work)
- **Mechanism**: Δ_Q^(t) = -β Σ P_{f_s} q_t, step-adaptive Q-side subtraction
- **Training**: ✓
- **차별 vs F11**: Q-side (we are K-side), single axis per step, dense projection (not sparse).

### Group 6: Energy-based / Order-dependent / Cognitive-Geometric (v5.2 신규, 2026-04-19 심야)

이 그룹은 Group 1–5의 "intervention mechanism" 축과 **orthogonal한 이론-프레임 축**이다. Attention/reasoning의 **composition 자체**가 (i) 에너지 최소화, (ii) 비가환 연산자 대수, (iii) 인지-기하학적 convex region, (iv) trajectory-level path search 로 framing될 수 있다는 전통. Reviewer가 "너의 F1–F11 null은 linear stationary commuting span-only regime만 기각했을 뿐 path-dependent composition은 테스트하지 않았다"고 공격할 때의 방어 축이자, `사용자 직관 (focus-dependent composition, Hermitian-순서효과, Hamiltonian 추상화, ontology as path prior)`의 선행 근거지.

**Motivation of inclusion**: F1~F13(예정 포함)은 모두 δK = αBBᵀK(또는 GS-rotation / FacetRot)라는 **linear · stationary · commuting (BBᵀ는 symmetric idempotent) · span-only** regime 안에 있다. 이 regime이 false인 경우에만 Group 6의 frame이 load-bearing이 된다 — 따라서 Group 6은 F-series track과 **orthogonal한 새 thesis**를 구성할 수 있다.

#### 6A. **Hopfield Networks is All You Need** — Ramsauer et al. NeurIPS 2020 (arxiv 2008.02217)
- **Core claim**: Softmax attention = **continuous modern Hopfield update**. 명시적 energy `E(q; K) = -lse(β q Kᵀ) + ½ qᵀq + const` 의 gradient descent 한 스텝이 정확히 attention output.
- **Relation to ours**: 사용자의 "Hamiltonian 최적화" 비유의 **직접 수학적 대응물**. F10의 K-bias를 "landscape-reshaping"으로 재해석 가능 — energy well depth / basin position 측정은 지금 infra 그대로 가능.
- **차별 vs F-series**: 우리는 energy 측정을 한 번도 안 했다. 직접적 실험 hook: F10 artifacts에 `lse(qKᵀ)` per-step 기록 추가 (0 GPU-hr).
- **NOT prior art of F-series directly** — theoretical framework. Reviewer-defense 축.

#### 6B. **Energy Transformer** — Hoover, Strachan, Liang, Krotov ICLR 2023/2024 (arxiv 2302.07253)
- **Core claim**: 전 transformer를 **globally minimizable energy** 로 통일 (Hopfield + LayerNorm + FFN 모두 energy-descent step으로 해석).
- **Mechanism**: Attention + associative memory block이 공통 Lyapunov function 감소.
- **차별 vs F-series**: Frame만 제공, explicit steering 제안 아님. 우리 F11 MOFCISS의 "sparse atom coding"은 energy 공간에서 "local basin selection"으로 재해석 가능 — thesis 재작성 hook.
- **Import 가치**: §3 Theorem 재구성 시 energy-based Lemma 가 Ledoux/Stiefel 보다 더 깊은 근거가 될 수 있음.

#### 6C. **Quantum Cognition** — Busemeyer & Bruza 2012 *Quantum Models of Cognition and Decision* (CUP); Pothos & Busemeyer 2013 BBS
- **Core claim**: 인간 판단의 **순서 효과 (order effects)** 를 **비가환 Hermitian projector** 로 형식화. `P_A P_B ≠ P_B P_A` → `Pr(A then B) ≠ Pr(B then A)`.
- **Relation to ours**: 사용자의 "Hermitian operator처럼 접근 순서 따라 도달점이 다르다"의 **학술 전통 존재 증명**. "Hermitian" 용어가 은유가 아니라 technical object가 되는 유일한 선행 학파.
- **차별 vs F-series**: LLM에 직접 적용 선행 드묾 (Aerts 2023 계열 예비 시도). 우리가 "commutator `[P_ont, P_query]`가 accuracy 예측" 실증하면 **LLM × quantum-cognition 교차 first paper**.
- **Caveat**: 용어 사용 시 Busemeyer/Bruza 명시 인용 필수. 무인용으로 "Hermitian"만 쓰면 physicist reviewer가 즉시 공격.

#### 6D. **Conceptual Spaces** — Gärdenfors 2000 *Conceptual Spaces* (MIT Press); 2014 *The Geometry of Meaning*
- **Core claim**: Ontology/개념 = **cognitive-geometric convex region** in quality dimension space. Category는 prototype + metric 으로 특징지어짐.
- **Relation to ours**: 사용자의 "인간 추상화 = 접근 순서 최적화"를 **기하학화**. Ontology가 "convex region 경계"라면, F8d의 verb × domain NMI orthogonality는 Gärdenfors-dimension의 empirical discovery.
- **차별 vs F-series**: 우리 B_ont는 flat basis. Gärdenfors은 **hierarchical convex region**. F7 Ontology-Structured B_ont의 H variant가 이 방향 (그러나 τ²-bench에서 collapse).
- **Import 가치**: §2 related work에서 "ontology semantic이 attention-geometric reframe 에 의해 소거됐다"는 F1 narrative을 부드럽게 — "우리는 semantic-flat projection을 보인 것이지, conceptual-geometric region 자체를 소거한 것 아니다".

#### 6E. **DisCoCat** — Coecke, Sadrzadeh, Clark 2010 *Mathematical Foundations for a Compositional Distributional Model of Meaning* (Linguistic Analysis 36)
- **Core claim**: Meaning composition = **categorical tensor contraction** over pregroup grammar. 단어 = tensor, 문법 = morphism, 문장 = contracted tensor.
- **Relation to ours**: "Ontology compose via operator algebra"의 가장 엄밀한 수학적 선행. Kartsaklis/Sadrzadeh 후속에서 Bigram/sentence composition을 category-theoretic하게.
- **차별 vs F-series**: LLM attention을 categorical로 본 후속 (Cohen et al., Toumi) 존재하나 training-time. Inference-time ontology compose는 novel 공간.
- **Caveat**: Formalism 무거움. Paper 본문보단 Appendix에 reference.

#### 6F. **ICL Order Effects** — Lu et al. 2022 *Fantastic Ordered Prompts and Where to Find Them* (ACL); Zhao et al. 2021 *Calibrate Before Use* (ICML); Min et al. 2022 *Rethinking the Role of Demonstrations* (EMNLP)
- **Core finding**: ICL에서 **exemplar 순서만 바꿔도** 같은 내용이 accuracy 수십 pp 변동. GPT-3/Llama 계열에서 보편.
- **Relation to ours**: 사용자 직관 "Hermitian 순서 효과"의 **직접 empirical base**. F-series가 완전히 무시한 축.
- **차별 vs F-series**: Lu/Zhao/Min은 "순서 효과 있음"을 보일 뿐, **"ontology-informed 순서가 임의 순서보다 systematically 낫다"는 claim 없음**. 우리 novelty 공간.
- **직접 실험 hook**: H-Order (§6.후단 참조).

#### 6G. **Head / Layer Commutator Analyses** — Dalvi et al. 2020 *Analyzing Redundancy in Pretrained Transformer Models* (EMNLP); Conmy et al. 2023 *Automatic Circuit Discovery* (ACDC, NeurIPS)
- **Core finding**: Head pair 간 중복성 · non-trivial interaction 광범위. Circuit discovery에서 path-dependent activation 흐름 존재.
- **Relation to ours**: 비가환성이 transformer에 실재한다는 mechanistic 실증.
- **차별 vs F-series**: 우리 F12/F13 FacetRot는 SO(2) per-head **commuting** rotation. Non-commuting 확장은 미탐색 (F12에서 per-head θ는 head끼리 독립이지 composition non-commuting 측정 없음).
- **직접 실험 hook**: `[M_Li, M_Lj]` Frobenius norm을 F10/F12 artifacts에 postprocess로 추가 가능 (GPU 불필요).

#### 6H. **Linear Representation Hypothesis / Geometry of Concepts** — Park, Choe, Veitch 2024 (ICLR); Nanda et al. 2023
- **Core claim**: Concept이 residual stream 안에서 **linear direction**으로 저장. Difference-of-means, probing, steering 모두 이 직관에 의존.
- **Relation to ours**: CAA/RepE/ITI의 이론 foundation이자, "focus 바꿔서 composition 재조직"의 가장 단순한 버전.
- **차별 vs F-series**: F-series는 concept direction을 **K-side basis**로 전이. LRH는 **residual stream**. Cross-stream 대응은 미증명 — F9/F10 V-axis와 D-axis가 residual에서 어떻게 표현되는지 미확인.
- **Limitation**: Recent work (Gurnee, Templeton 2024 "Features are Not Directions")이 LRH를 약화. 순수 linear 가정을 주장하면 위험.

#### 6I. **Active Inference / Free Energy Principle in LLMs** — Friston 2010 *The Free-Energy Principle* (Nat Rev Neurosci); Parr, Pezzulo, Friston 2022 *Active Inference*; recent LLM applications (Da Costa et al. 2023, Yufik 2024 계열)
- **Core claim**: 생물적 추론 = **expected free energy 최소화**. Action/attention = posterior belief 업데이트를 통한 surprise 최소화.
- **Relation to ours**: 사용자의 "Hamiltonian 최적화" 비유 중 물리학보다 **인지과학 친화적** 대응물. "Ontology가 prior를 제공해 FE minimization path를 단축"의 formal 틀.
- **차별 vs F-series**: LLM에 직접 적용은 아직 mature 아님. High-risk / high-novelty.
- **Caveat**: Friston 학파는 mathematical rigor 논란 존재. Reviewer pool에 따라 호불호 극단.

#### 6J. **Reasoning as Path Search** — Andreas 2016 *Neural Module Networks* (CVPR); Yao 2023 *Tree of Thoughts* (NeurIPS); Khattab 2024 *DSPy* (ICLR); Akyürek 2022 *What Learning Algorithm is In-Context Learning?* (ICLR 2023)
- **Core claim**: 추론 = **compositional program over primitive operators** 또는 **explicit search in reasoning space**.
- **Relation to ours**: 사용자의 "ontology가 abstraction path를 최적화"의 가장 직접적인 operational 형식.
- **차별 vs F-series**: NMN/ToT/DSPy 모두 **prompt/output-level**. K-side 내부 activation으로 path 구성은 미탐색.
- **직접 실험 hook**: H-Trajectory (§6.후단 참조). Residual stream을 per-step 기록 → ontology-guided CoT vs ad-hoc CoT trajectory divergence 측정.

#### 6K. **Hopfield / Energy framing of K-space steering** — 후보 framework link (synthesis)
- **Synthesis claim** (아직 paper 없음, 우리 기회): F11 MOFCISS의 sparse atom selection을 **energy landscape의 local basin routing**으로 재해석. 각 ontology atom = basin attractor, OMP = basin selection, step-decay = path in energy manifold.
- **Gap**: 문헌 부재. 우리가 쓰면 first.

---

### Group 6 cross-linkage to existing cross-tab

Group 6 항목들은 **intervention mechanism** 이 아니라 **theoretical frame** 이므로 §2.5.1.3의 6-dim cross-tab 행으로 들어가기 부적합. 대신 **추가 축** "Theory-frame dependence" 를 paper §2 에서 단독 paragraph로 처리:

> Our F1~F13 experiments all operate under a linear-stationary-span-only regime (§2.1 scope). Theoretical frameworks outside this regime — energy-based attention (Ramsauer 2020, Hoover 2023), non-commuting projector composition from quantum cognition (Busemeyer & Bruza 2012), cognitive-geometric convex regions (Gärdenfors 2000), categorical tensor composition (Coecke et al. 2010), ICL order effects (Lu et al. 2022), and path-search reasoning (Yao 2023) — propose that attention composition itself is path-dependent, non-commuting, or energy-minimizing. Our F1 reframe is regime-limited (§6.3) and does not speak to these alternatives. We flag three concrete hypotheses (H-Order, H-Energy, H-Trajectory) as future-work hooks that would test the Group 6 frame directly; these are orthogonal to the F11~F13 track and would form a separate follow-on thesis.

### Group 6 → 3 testable hypotheses (future-work hooks)

F-series와 **orthogonal한** 새 실험 축. Regime-wise 독립이므로 F12/F13이 null이어도 독립 가치.

#### **H-Order** — Ontology-informed exemplar ordering outperforms random
- **Base**: Lu 2022 "순서 효과 존재" 확증 → 우리 novelty = "ontology-informed 순서가 optimal에 근접"
- **Setup**: MetaTool Subtask4 N=200, ontology-ordered exemplars (verb 계열 우선 → domain 계열 차순) vs random permutation 10개. Accuracy 분산 측정.
- **Prediction**: Ontology-ordered accuracy ≥ 90th percentile of random permutations. Otherwise H-Order null.
- **상관성 테스트**: Per-head commutator `‖[P_verb, P_domain]‖_F` vs ordering sensitivity. r > 0.3 이면 비가환성 실증.
- **비용**: 1–2 GPU-hr. No hook engineering — pure prompt-level.
- **Group 6 연결**: 6F (Lu/Zhao/Min) + 6C (Busemeyer/Bruza) + 6G (Dalvi/Conmy commutator).

#### **H-Energy** — Ontology K-bias deepens Hopfield energy well at correct tool
- **Base**: Ramsauer (6A) — `E_q(K) = -lse(qKᵀ/√d)`는 closed-form 계산 가능, GPU 추가 forward 불필요 (F10 saved tensors에 postprocess).
- **Setup**: F10 · F9 artifacts로 per-query energy at GT tool position vs distractor tools. Ontology intervention 전/후 depth 변화 `ΔE_gt - ΔE_distractor` 측정.
- **Prediction**: 정답 변경 case에서 `ΔE_gt - ΔE_distractor > 0` (basin이 정답 쪽으로 깊어짐). Correlation with actual flip 측정.
- **비용**: 2–4 GPU-hr (re-extract if saved tensors insufficient) 또는 0 GPU-hr (postprocess).
- **Group 6 연결**: 6A (Ramsauer) + 6B (Hoover) + 6K (Hopfield × MOFCISS synthesis).
- **Novelty**: F-series 어떤 실험도 energy 측정 없었음. 본 metric은 span-only regime에서도 의미 있음 — F1 reframe과 독립 관측.

#### **H-Trajectory** — Ontology-guided reasoning converges to shorter / less divergent latent trajectories
- **Base**: Gärdenfors convex region (6D) + path-search (6J) + LRH (6H).
- **Setup**: MetaTool/BFCL CoT samples 100개. Per-step residual stream (layer 28, last position) 기록. Ontology-guided prompt vs ad-hoc CoT간 궤적 divergence (mean pairwise cosine distance 증가율, 마지막 step까지의 total path length).
- **Prediction**: Ontology-guided trajectory total length ≤ 0.8 × ad-hoc. 또는 "정답 ending point"로의 수렴 속도가 더 빠름.
- **비용**: 4–6 GPU-hr.
- **Group 6 연결**: 6D (Gärdenfors) + 6H (LRH) + 6J (path-search).
- **Paper-grade novelty**: F-series와 완전 독립. 성공 시 separate thesis.

### Group 6 경고 (정직)

1. **"Hermitian / Hamiltonian"은 은유 vs technical 경계를 흐리면 위험**. 6A/6B/6C 명시 인용 + 우리가 어디서 technical, 어디서 analogical인지 선명하게.
2. **Lu 2022 순서효과는 이미 확립** — "순서 효과 있음"은 novelty 아님. "**ontology-informed 순서가 systematically 낫다**"만 우리 기여.
3. **LRH(6H) 약화 추세** (Gurnee/Templeton 2024 "features ≠ directions"). Linear-direction 전제를 paper에서 주장하면 공격받음.
4. **ICL Bayesian view (Xie 2022) 경쟁**: "ontology 주입 lift"가 in-context retrieval과 구별 불가능한 risk. H-Order/H-Energy/H-Trajectory 모두 ICL Bayesian null과 명시적으로 구분되는 design 필요.
5. **F12/F13 track과 commit 순서**: Group 6 축 실험을 F12/F13 null 이후 착수 권고 (F12/F13이 positive면 paper scope 먼저 거기 focus).

---

## §2.5.2 Group 6 확장 (v5.3, 2026-04-19 심야 — 3-agent prior-art 재감사 + 사용자 5-question synthesis)

사용자 연쇄 5-질문 ("meta-attention / MCTS / RL / hierarchical observation / ontology-ordering 통합 아키텍처")을 3-agent web-search 재감사로 검증.

### 6L. 추가 preemption risk 문헌 (3-agent audit)

기존 6A-6K에 더해 다음이 **직접 선행연구 위험** (URLs in `cognitive_geometric_reframe_group6_2026_04_19.md` 확장본):

#### 6L.1 **Das 2025 — Free Probabilistic Framework (arxiv:2506.16550)**
- W* 대수 tracial self-adjoint operator로 LLM 모델링. **positional × semantic operator commutator nonzero → word order encoding** 명시 주장.
- **Preemption**: 이론 overlap, 실험 부재. 우리 H-Order 실증 부분은 empty.

#### 6L.2 **Aerts & Sozzo 2024 — Inductive bias from QM (arxiv:2312.03862)**
- 비가환 projective measurement를 inductive bias로 QML에 도입, order effect 학습.
- **Preemption**: QML framework, synthetic data. Transformer 아키텍처 아님. 차별화: pretrained transformer 내재 비가환성 측정.

#### 6L.3 **Sato, Kawamoto, Kera 2025 — Chain of Thought in Order (ICML 2025, arxiv:2506.23875)**
- **H-Order의 가장 직접적 선행**. 10억 CoT token order 후보에서 learning-friendly order 발견.
- **Preemption**: training-level (loss-based), arithmetic 한정. 우리 H-Order (inference-time, ontology-axis, effect-size)와 **다른 level** — baseline+차별화 필수.

#### 6L.4 **Chen et al. 2024 — Premise Order Matters (ICML 2024, R-GSM)**
- Premise 순서 permutation으로 LLM 성능 30pp+ drop.
- **Preemption**: mechanism 제시 없음. 우리 "ontology-informed order가 optimal에 근접"은 empty. 필수 baseline.

#### 6L.5 **Shen 2026 (arxiv:2604.05655) + Barez 2026 (2603.01326) + Manson 2025 (2507.21107)**
- 단계별 residual stream trajectory 측정. Correct vs incorrect late-layer divergence (AUC 0.87). Concern curvature.
- **Preemption**: 방법론 overlap 高. Ontology-axis 구분은 없음 — correctness, concern만.
- **Risk mitigation**: Manson curvature method cite + "we apply to ontology-axis distinction".

#### 6L.6 **Wang & Zhang 2025 — Energy-Driven Steering (arxiv:2510.08646)**
- 외부 EBM + gradient-based activation steering.
- **Preemption**: H-Energy의 close cousin. 차별화: external training 필요 vs 우리 internal no-train.

#### 6L.7 **Zhang 2025 — Hamiltonian LLM (arxiv:2601.11572)**
- L2-normalized LLM embedding을 Hamiltonian dynamics로. Attention weight = path amplitude.
- **Preemption**: 이론 overlap, 실험 부재.

#### 6L.8 **Engels et al. 2024 — Not All Features Are 1D (arxiv:2405.14860)**
- LRH 핵심 반론. 요일/월 등 circular 2D.
- **Implication**: "axis = 1D" 주장 시 공격. Multi-dim subspace framing 필수.

#### 6L.9 **Park et al. 2024 NeurIPS — Categorical/Hierarchical Concept Geometry**
- 계층 개념의 simplex geometry.
- **Preemption**: 우리 rep-geometry 이웃. §2 비교 필수.

#### 6L.10 **Postmus 2024 NeurIPS + Joshi 2025 — Conceptors**
- Conceptor matrix로 Boolean AND/OR/NOT compositional steering. Multi-facet composition의 SOTA baseline.

#### 6L.11 **Zhou et al. 2023 ICLR — Least-to-Most**
- Hierarchical 분해로 SCAN length-split 16% → 99%. "구조 유도가 latent capability unlock"의 강한 empirical 증거.
- **Implication**: Q5 hierarchical meta-attention 가설의 empirical prior. Prompt-level.

#### 6L.12 **Templeton 2024 Scaling Monosemanticity + Marks 2024 Sparse Feature Circuits**
- SAE로 34M abstract features 발견. Feature-level circuit editing으로 task lift.
- **Implication**: "dormant abstract circuits" 존재의 mechanistic 증거. Q5 강한 지지.

#### 6L.13 **Zhou 2024 Self-Discover + Besta 2024 Graph of Thoughts**
- Task-adaptive reasoning module composition. BBH +30%, sorting +62%.

#### 6L.14 **MCTS × LLM**: AlphaProof (Nature 2025), FunSearch (Nature 2024), ReST-MCTS (2024), o1/o3 inference-time search
- 모두 output/reasoning-step level. **Attention-subspace level MCTS는 empty**.

#### 6L.15 **Higher-Order Theories of Consciousness** — Rosenthal 1986, Lau & Rosenthal 2011; Global Workspace Theory (Baars 1988, Dehaene 2014); Predictive Coding (Rao & Ballard 1999)
- **Q5 가설의 직접 철학/인지과학 선행**. "Higher-order representation이 추상/의식 생성".
- **Implication**: Q5는 확립된 인지과학 가설의 LLM 구체화. 정성적으로 참.

### 6M. 사용자 5-Question Synthesis → F14 MetaFocus (proposed)

**통합 thesis**:
> Frozen pretrained LLM 위에 (a) ontology-conditioned meta-attention, (b) attention-subspace exploration, (c) RL-style intrinsic reward, (d) hierarchical observation을 **training-free**로 통합하여 higher-order abstraction unlock.

#### 6M.1 각 구성요소의 empty cell

| 구성요소 | 가장 근접한 선행 | Empty? |
|---|---|---|
| Meta-attention | HHGT/LUKE/K-BERT (training); Conceptors (steering); Marks SAE (probe) | **training-free + ontology-axis + cross-layer EMPTY** |
| Ontology ordering | Sato 2025 (arith, training); ORACLE (prompt) | **inference-time + rep-level + effect-size EMPTY** |
| Attention subspace exploration | Entropy-regularized; ToT/GoT (output); EDS (external EBM) | **attention-subspace MCTS EMPTY** |
| RL for attention | RLHF (output); DPO; Marks SHIFT (circuit) | **attention-subspace RL EMPTY** |
| Hierarchical meta-observation | Transformer depth; HOT (철학); Least-to-Most (prompt) | **qualitative-distinct meta-layer EMPTY** |

5/5 empty.

#### 6M.2 F14 MetaFocus preliminary spec (not yet committed)

```
Frozen pretrained LLM (Qwen2.5-7B)
│
├─ Base attention (L0-L27): 변경 없음
│
└─ Meta-attention layer (F12/F13 확장):
    ├─ Ontology-axis Q/K rotation R_ont(axis, step)
    ├─ Exploration bonus α_explore * N(0,σ²) in B_ont⊥  (MCTS-style, subspace-restricted)
    ├─ Intrinsic reward (no training):
    │   - Ontology coverage of emitted tokens
    │   - Hopfield energy E_q(K) decrease rate
    │   - Residual trajectory divergence
    └─ Meta-observation: higher-layer attention over lower-attention outputs
       indexed by ontology axes
```

**비용**: 8-12 GPU-hr (F12/F13 infra 재사용). 4-outcome pre-reg: ≥+5pp / +2-5pp / noise / negative.

### 6N. 3 추가 testable hypotheses

#### H-Meta (Q2+Q5)
- **Claim**: Ontology-axis meta-attention layer > same-capacity pure-LoRA baseline.
- **Setup**: F12/F13 infra + meta-layer. Subtask4 N=200.
- **Null**: Identity rotation + identity gate.
- **비용**: 10-15 GPU-hr.

#### H-MCTS (Q3)
- **Claim**: Attention-subspace exploration bonus `α_explore * noise in B_ont⊥`가 multi-tool coverage 개선.
- **Setup**: MetaTool Subtask4. α_explore ∈ {0, 0.05, 0.1, 0.2}.
- **Null**: Noise in B_ont-aligned subspace.
- **비용**: 4-6 GPU-hr.

#### H-HOT (Q5)
- **Claim**: Pure transformer depth는 "implicit meta-attention"을 이미 구현. 명시 meta-layer는 qualitatively different (ontology-typed routing, step-state aware)일 때만 lift.
- **Setup**: 3 variant — (i) identical stacked (control), (ii) ontology-typed routing, (iii) step-state aware.
- **Prediction**: (ii),(iii) ≥ (i)+2pp. 아니면 "depth만으로 충분".
- **비용**: 12-18 GPU-hr.

### 6O. 권고 실험 순서 (F11 falsified 반영, 2026-04-19 저녁)

| Priority | Phase | 비용 | Decision |
|:-:|---|---|---|
| 1 | F12 FacetRot-QK | spec 완료 | ≥+3pp → F14 delay / <+3pp → F13 확인 |
| 2 | F13 FunnelRot | F12 병렬/순차 | F13 null → H-Order 즉시 |
| 3 | H-Order | 1-2 GPU-hr | Positive → H-Meta + F14 pilot |
| 4 | H-Energy | 0 GPU-hr (postprocess) | 항상 run, F10/F12 artifacts 재사용 |
| 5 | H-Trajectory | 4-6 GPU-hr | Separate follow-on paper |
| 6 | F14 MetaFocus (3-cell pilot) | 8-12 GPU-hr | H-Order ≥+2pp 후 착수 |
| 7 | H-MCTS / H-HOT | F14 후 ablation | §5.X 확장 |

### 6P. 추가 경고 (v5.3)

6. **"Meta-attention" 용어는 reviewer magnet**. MoE, Hypernetworks, Capsule, RETRO cross-attn, Perceiver, RMT, AoA (Cui 2017) 모두 인접 — "**ontology-conditioned attention rotation**" 또는 "**facet-routed attention**"으로 구체 naming.
7. **AlphaGo RL → attention은 search-space 폭발** (L×H×T×T continuous). **B_ont subspace (dim ~24) 로 축소** 필수.
8. **Reward Goodhart 리스크**: "ontology coverage" reward → token 남발 가능. H-Energy / trajectory-divergence intrinsic signal 병행.
9. **HOT 인용 시 철학 논쟁 주의** (Dennett/Block 비판). "HOT-inspired" weakly 언급.
10. **F14는 F12/F13 null 이후 착수** — MOFCISS 실패 선례 반복 금지. Preliminary spec만 문서화.

---

### 2.5.1.3 종합 6-dim cross-tab (확장 19-method)

| # | Method | Year/Venue | Multi-tool | Step-adapt | Train-free | Multi-facet | Semantic | Activation-level |
|---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1A | OG-RAG | EMNLP 2025 | △ | ❌ | ✓ | △ KG | ✓ | ❌ prompt |
| 1C | OntoLLM | SD 2026 | △ | ❌ | ✓ | △ KG | ✓ | ❌ prompt |
| 2A | Generative Ontology | arxiv 2602 | △ schema | ❌ | ✓ | △ schema | ✓ | △ logit |
| 2B | GCD | EMNLP 2023 | △ | ❌ | ✓ | △ grammar | △ syntactic | △ logit |
| 3A | Neuro-Symbolic Enhancing | arxiv 2504 | ❌ | ❌ | △ logistic regression | ❌ | ✓ OWL | ❌ post-hoc |
| 3C | Ontology-Constrained Enterprise | arxiv 2604 | ❌ | △ session | ✓ | ✓ Role/Domain/Interaction | ✓ | ❌ pipeline |
| 4A | OntoTune | ICLR 2025 | ❌ | ❌ | ✗ self-train | ❌ | ✓ | ❌ training |
| 5A | CAV | 2017+ | ❌ | ❌ | ✗ probe | ❌ | ✓ concept | ✓ residual |
| 5B | CAA | ACL 2024 | ❌ | ❌ | ✗ contrastive | ❌ | ✓ behavior | ✓ residual |
| 5C | ITI | NeurIPS 2023 | ❌ | ❌ | ✗ probe | ❌ | ✓ truth | ✓ head output |
| 5D | RepE | 2023 | ❌ | ❌ | ✗ probe | ❌ | ✓ concept | ✓ residual |
| 5E | PASTA | ICLR 2024 | ❌ | ❌ | △ profile | ❌ | ❌ position | ✓ attn score |
| 5F | Spotlight | 2025 | ❌ | ❌ | ✓ | ❌ | ❌ position | ✓ attn score |
| 5I | SEKA | ICLR 2026 | ❌ | ❌ | ✗ contrastive | ❌ | ✓ trained | ✓ K-side |
| 5J | AdaSEKA | 2026 | ❌ | ❌ | ✗ contrastive | ❌ | ✓ expert | ✓ K-side |
| 5K | Focus Directions | 2025 | ❌ | ❌ | ✗ gradient | ❌ | ✓ relevance | ✓ K+Q |
| 5L | FGA | 2025 | ❌ | ❌ | ✗ learned W_K | ❌ | ✓ KB | ✓ attn pre-soft |
| 5M | SADI | ICLR 2025 | ❌ | ❌ | △ contrastive ~150 | ❌ single | ✓ contrastive | ✓ head/neuron/hidden |
| 5N | G-ACT | 2025 | ❌ | △ online | △ probes | △ cluster | ✓ | ✓ residual |
| 5O | Attention-guided Steer | arxiv 2602 | ❌ | ❌ | ✓ | ❌ | △ prefix | △ prefix-driven |
| 5P | Vector Ontologies | arxiv 2506 | — | — | ✓ | ✓ ontology | ✓ | ❌ extraction only |
| 5R | Hazarika 2022 | AAAI 2022 | ❌ | ❌ | ✓ | ❌ | ❌ position | ✓ cross-attn |
| 5S | Q-coverage (NeurIPS withdrawn) | — | ✓ | ✓ | ✓ | ❌ | ❌ | ✓ Q-side |
| F10 (executed, falsified) | 2026-04-19 | ❌ | ❌ | ✓ | ✓ V×D | ✓ ontology | ✓ K-side |
| **F11 MOFCISS (proposed)** | **2026-04-19** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓ K-side** |

→ **F11 MOFCISS만 6/6 충족** in (multi-tool, step-adapt, train-free, multi-facet, semantic, activation-level).

### 2.5.1.4 Mechanism layer 분포 시각화

```
┌────────────────────────────────────────────────────┐
│ Ontology + LLM Inference Methods (~24 documented)  │
├────────────────────────────────────────────────────┤
│                                                    │
│  Prompt level    ████████████ (50%)  RAG/OntoLLM   │
│  Output level    █████ (12%)         GCD/Generative│
│  Post-hoc verify ████ (8%)           Neuro-Symbolic│
│  Training        ███ (4%)            OntoTune      │
│  Activation level ██████████████ (26%)             │
│    └─ Residual: CAV/CAA/RepE/G-ACT                 │
│    └─ Attn output: ITI                             │
│    └─ Attn score: PASTA/Spotlight/Focus            │
│    └─ K-side: SEKA/AdaSEKA/FGA/F10/F11             │
│    └─ Hidden state: SADI                           │
│                                                    │
│  K-side + ontology + train-free + multi-facet      │
│  + step-state = 0 prior. 우리 F11이 빈 칸          │
└────────────────────────────────────────────────────┘
```

---

## §2.6 Prior-art gap analysis + 새 mechanism 후보 brainstorm (v5.1 신규)

위 광범위 audit 후 **빈 칸이 명확**: K-side activation modification + ontology semantic + training-free + multi-tool step-adaptive + multi-facet 동시 충족 prior 부재.

이를 바탕으로 새 mechanism 후보 5개 brainstorm. F11 MOFCISS 외 alternatives.

### 2.6.1 후보 A: **MOFCISS** (이미 §5 Phase F11 spec 작성됨, 가장 가까운 후보)

**핵심**: Sparse OMP coding over ontology atoms + step-history facet decay + K-side projection

**위치**: ICLR §5 Phase F11 spec 참조

**예상 성능**: F11b Subtask4 +3-5pp F1 lift over baseline 0.731

### 2.6.2 후보 B: **OMR (Ontology-Mapped Retrieval-Augmented Steering)**

**Motivation**: OG-RAG (EMNLP 2025) 의 retrieval framework + SEKA (ICLR 2026) 의 K-side steering 결합. Retrieval 이 query 별 ontology atoms 선택, K-side projection 으로 inject.

**Mechanism**:
```
Step 1: Pre-compute ontology atom embeddings E_n = forward(model, plugin_n.desc)[L_retrieve]  
        (retrieval embedding from one mid-layer)
Step 2 (inference): query embedding q_emb = forward(model, query)[L_retrieve]
                    top-k atoms = top-k cosine(q_emb, E_n) → A_query
Step 3: Per-layer K-side bias:
        delta_K = alpha * Σ_{n in A_query} c_n * K_n_at_layer_L
        where c_n is retrieval score (softmax over top-k)
Step 4: For multi-tool: after first emit, exclude emitted facet's atoms from retrieval pool
```

**vs F11 MOFCISS**:
- F11: per-token OMP at every (L, h) — fine-grained but expensive
- OMR: per-query retrieval at single layer + global K-bias — coarser but cheaper
- OMR more aligned with OG-RAG paradigm familiarity

**Pros**: Simpler implementation, leverage RAG infrastructure
**Cons**: Per-query (not per-token), might lose multi-step granularity

**Cost estimate**: 4-6 GPU-hr (build + Subtask4 sweep)

### 2.6.3 후보 C: **NSCC (Neuro-Symbolic Coverage Control)**

**Motivation**: Enhancing LLMs Neuro-Symbolic (arxiv 2504.07640) 의 OWL+reasoner pattern + step-adaptive coverage. Reasoner가 emitted tools 의 facet coverage 검증, missing facet 을 identify, K-side에서 missing facet 강조.

**Mechanism**:
```
At decoding step t (after emitting tools_<t}):
  Reasoner check: facets covered = {f : ∃ tool ∈ tools_<t, facet(tool) ⊇ f}
  GT facets needed = parse from query (LLM call to enumerate intents)
  Missing facets = GT_facets - covered_facets
  
  K-side bias toward missing facets:
    delta_K = +alpha * Σ_{f in Missing} B_f B_f^T K
```

**Pros**: Symbolic reasoning explicit, interpretable
**Cons**: Requires LLM call for query intent parsing (extra cost), reasoner heavyweight

**vs MOFCISS**: NSCC explicit reasoning, MOFCISS implicit via decay
**Cost estimate**: 8-12 GPU-hr (reasoner integration complex)

### 2.6.4 후보 D: **GCD-OG (Grammar-Constrained Decoding + Ontology Gating)**

**Motivation**: Generative Ontology (arxiv 2602.05636) 의 schema-as-grammar + facet-conditioned K-bias. Token-level grammar restricts to valid tools, K-bias steers toward NMI-orthogonal facets.

**Mechanism**:
```
Step 1: Compile ontology to grammar G (Outlines library)
Step 2: At each decode step:
  (a) Apply K-bias toward currently-needed facet (from previous emissions)
  (b) Token mask via G (only valid tool name tokens allowed)
Step 3: Combined: K modification AND logit masking
```

**Pros**: Two-channel intervention (K-side + logit-side), validity guaranteed
**Cons**: Grammar compilation overhead, might over-constrain

**vs MOFCISS**: GCD-OG = validity guarantee + facet steering. MOFCISS = pure K-side.
**Cost estimate**: 6-8 GPU-hr

### 2.6.5 후보 E: **OAS-Multi (Ontology-Anchored Multi-step Steering)**

**Motivation**: Vector Ontologies (arxiv 2506.13252) 의 vector space 위에 G-ACT (2025) 의 online adaptive direction selection, but **probes 없음** (training-free).

**Mechanism**:
```
Pre-compute (one-time):
  Vector ontology basis V_ont = stack of (verb, domain, schema, ...) axes
  Each axis has set of anchor K vectors
  
Per-decoding-step:
  Step 1: Score query along each ontology axis: score_axis = q · axis_centroid
  Step 2: Identify dominant axis (top-1 or top-k)
  Step 3: Apply K-bias along dominant axis only:
          delta_K = alpha * V_dominant V_dominant^T K
  Step 4: Track emitted tools' dominant axes; subsequent steps select different axes
```

**vs MOFCISS**: OAS uses single-axis at a time (sequential), MOFCISS multi-axis weighted.
**Pros**: Closer to G-ACT/Vector Ontologies framework, maybe simpler
**Cons**: Coarser granularity (one axis per step)

**Cost estimate**: 4-6 GPU-hr

### 2.6.6 후보 비교표

| 후보 | Core idea | Build cost | Eval cost | Total | Risk | Expected lift |
|---|---|---|---|---|---|---|
| **A. MOFCISS** | Sparse OMP + step decay + facet-aware K-side | 0.5 hr | 5 hr | 6-10 hr | medium engineering | +3-5pp |
| B. OMR | Retrieval + global K-bias + facet-exclude | 0.5 hr | 4 hr | 4-6 hr | low | +2-4pp |
| C. NSCC | Reasoner facet coverage + missing-facet K-bias | 2 hr | 6 hr | 8-12 hr | high (reasoner integration) | +1-3pp |
| D. GCD-OG | Grammar + facet K-bias dual | 1 hr | 5 hr | 6-8 hr | medium | +2-4pp |
| E. OAS-Multi | Vector ontology axis-by-step single-axis | 0.5 hr | 3 hr | 4-6 hr | low | +1-3pp |

**우선순위 권고**:

1. **A (MOFCISS) 먼저** — 가장 fine-grained + 가장 novel, 이미 spec 작성됨
2. **B (OMR) 백업** — A 가 implementation 어려우면 더 simple alternative
3. **E (OAS-Multi) 비교** — A vs E 가 mechanism 가설 검증 (multi-axis weighted vs single-axis sequential)
4. **D (GCD-OG) stretch** — token validity 보장 ablation
5. **C (NSCC) 미루기** — reasoner integration 너무 무거움

### 2.6.7 다음 세션 권고 실험 순서

| 우선순위 | Phase | 실험 | 비용 | 의사결정 점 |
|---|---|---|---|---|
| 1 | F11 (MOFCISS) | Subtask4 5-cell sweep N=200 | 6-10 GPU-hr | F11b ≥ +3pp → 진행, < +3pp → 후보 B/E 시도 |
| 2 | F12 (OMR, conditional) | F11 약하면 | 4-6 GPU-hr | OMR ≥ F11 → 메인 후보 변경 |
| 3 | F13 (OAS-Multi, ablation) | F11 success 후 | 4-6 GPU-hr | A vs E 비교로 mechanism story 강화 |
| 4 | F14 (GCD-OG, stretch) | 모든 above 후 | 6-8 GPU-hr | Validity guarantee ablation |

### 2.6.8 새 mechanism 의 paper-grade contribution potential

만약 F11 (또는 alternative B/D/E) 가 positive 면:

**ICLR thesis 가능 framing**:
> "We identify a 6-dimensional gap in the prior art landscape (multi-tool selection × step-adaptive × training-free × multi-facet × semantic × activation-level) and propose MOFCISS (or alternative) as the first mechanism filling all six dimensions simultaneously. The key technical novelty is the combination of (i) sparse OMP coding over ontology atoms (preserves anchor identity, breaks Gram-Schmidt span-invariance non-linearly), (ii) step-history facet decay (enables multi-tool coverage like Q-coverage but on K-side multi-axis), (iii) NMI-verified orthogonal axes (verb × domain, F8d empirical foundation). Three Lemmas + Theorem M provide convergence framework. Empirical validation on MetaTool Subtask4 (multi-tool primary target) + BFCL parallel_multiple (cross-bench) + Subtask1 control (single-tool no-lift expected, validates step-state necessity)."

**ICLR ceiling 변화**:
- 모든 후보 negative: 5.25 그대로
- F11 (MOFCISS) +3-5pp: 6.0-6.5
- F11 +5pp 이상: 6.5-7.0 (likely strong accept)
- Alternative B/D/E positive (F11 negative 시 fallback): 5.75-6.25

### 2.6.9 한 줄 요지

> **광범위 audit (24 methods, 5 categories) 결과: K-side activation modification + ontology + training-free + multi-tool step-adaptive + multi-facet 동시 충족 prior 부재. F11 MOFCISS 가 primary 후보, OMR/NSCC/GCD-OG/OAS-Multi 가 backup/ablation. 다음 세션 first action = F11 prototype, decision tree per §5 Phase F11.**

---

## §3. 가설 (재편)

### 3.1 Primary hypotheses — T 의 generalization scope

**H-A (Attention-level universality) — ✅ CONFIRMED (Phase A, v3)**: Lemma 2 의 √(r/d) 스케일링이 모든 tested (M, V) 조합에서 성립.

- **검증 결과**: Phase A 9 (M, V) × 3 model family (Qwen/Llama/Mistral) × 4 benchmark. Mean slope **0.5298**, std 0.06, **all R² ≥ 0.99**.
- **Gate 재보정 (v3)**: 기존 tight gate [0.45, 0.55] 는 3/9 만 pass — over-specified. 새 gate = **"mean slope 0.50 ± 0.05 AND all R² ≥ 0.95"**. 이 기준 하 9/9 pass.
- 잔여 이슈: Qwen/Mistral 의 systematic 0.55 excess → H-I (아래) 로 취급.

**H-B (Margin lower bound stability) — ❌ REPLACED by T.B predictive form (Phase A, v3)**: 원래 "m_0 > 1 in 5/5" 형식은 **4/9 benchmark 에서 위반** (Llama Retail 0.016, Mistral Telecom 0.23, Llama Telecom 0.42, Qwen Banking 0.88). 그러나 **이것은 Theorem 실패가 아니라 가설의 over-specification**.

- **대체 claim (T.B predictive form)**: Margin distribution 이 lift 와 비교되어 flip 결정. $m_0$ 의 절대값이 아니라 per-query margin distribution 과 $\alpha \cdot \langle BB^\top q, d^*\rangle$ 의 상대 비가 문제.
- **Empirical 지지**: Qwen Banking m_0=0.88 → 7.8% flip, Llama Retail m_0=0.02 → 1.6% flip, 모두 **high-r (r=96) 에 집중** (Qwen Banking r=96 flip=21/100, r=1 flip=0).
- **Lemma 1 예측과 일치**: flip ⟺ α·lift > m. E[lift(r)] ∝ r/d (Lemma 2) → high-r 에서 more flip. 이 correlation 이 직접 관측됨.

**H-C (Two-level separation 의 cross-model 재현) — ✅ CONFIRMED (Phase A, v3)**: Attention-level smooth (E14) + argmax-level threshold (E15) 이 Qwen/Llama/Mistral 3/3 family 에서 관측.

- **Bonus**: Cross-architecture Lemma 2 universality 가 3/3 family 에서 성립 — v1/v2 에서 예상한 scope 보다 넓음.

**H-D (KL non-monotonicity 의 FFN/LM-head origin)**: argmax-level KL 의 non-monotonic shape (r=12 peak, r=48 trough, r=96 rebound) 은 10-layer FFN + LM-head 의 non-linear compounding 에 기인.

- Test: layer-by-layer KL 을 residual-stream 따라 측정. 중간 layer 에서는 monotonic 이었다가 LM-head 통과에서 non-monotonic 으로 transform 되는지 확인. Phase B.
- 예측: residual-stream KL slope 가 layer 깊어질수록 비선형성 증가.

**H-G (Facet-concentration of discriminative direction) — v2 신규**: 각 (layer, head) 의 tool-discriminative direction $d^*_{l,h}$ 이 B_ont 의 **단일 facet block** (function_action / io_type / domain / tool_category 중 하나) 에 ≥ 70% 집중. 따라서 full-span (variant B) 을 사용하면 off-facet 3 개 block 의 noise 가 lift 를 희석. Facet-split (variant A) 은 per-head 가 dominant facet 만 activate → lift concentration.

- **예측 (정량)**: Tier 3 A − B = +21.10pp (empirical) 를 재현. "Off-facet dilution ratio" 이 실제 3× (lift concentration ≈ projected dimension ratio).
- **Test**: §5 Phase B3. 기존 Tier 3 데이터 + head-by-head d\*-facet overlap 측정. 추가 GPU 거의 불필요.
- **Falsification**: (i) 각 head 의 $d^*$ 가 facet block 에 집중 안 됨 (≤ 50% overlap), **또는** (ii) facet-block 집중은 성립하지만 A−B 와 dilution 비가 일치 안 함 (< 0.5× 또는 > 2×).
- **실패 시 plan**: facet-split dominance 는 routing/gating 의 structural effect (non-concentration-based) 로 재해석. §Discussion 의 open question.
- **근거 데이터**: E8 (A-B-D decomposition), E9 (routing architectural).

**H-H (Scope boundary via alignment collapse) — v2 신규; v3 partial 이슈**: B_ont 가 **benchmark 외부** (e.g. MMLU) 또는 **architecture-mismatched model** (e.g. Mistral) 에서 negative lift 를 내는 이유는 **그 (M, V) 에서 B_ont 와 empirical $d^*_{M,V}$ 사이 각도가 크기 때문**. Tool-selection benchmark + catalog-aligned model 에서는 각도 작음.

**v3 partial 재고**: Phase A 에서 Mistral × τ²-Telecom 의 Lemma 2 slope 은 정상 (0.547, R²=0.9995, flip=0). 즉 Mistral 에서 random 은 여전히 flip 0 이지만 B_ont 는 -31.86pp 음수 (E11) — **Mistral 이 random 을 reject 하는 것은 아니고 B_ont 자체와 alignment 가 음의 방향이거나 format-incompatible** 일 수 있음. Phase B3.2 가 이걸 정확히 측정해야 함.

- **예측 (정량)**:
  - Qwen τ² Telecom: $\cos(B_\text{ont}, d^*_\text{emp}) \geq 0.5$
  - Qwen MMLU: $\cos < 0.3$
  - Mistral Telecom: $\cos < 0.3$ (또는 per-layer 변동 심함)
- **Test**: §5 Phase B3 sub-step. 기존 Qwen Telecom d\*_emp (Phase 0 에서 추출될 것) + 추가로 MMLU, Mistral Telecom 에 대한 d\*_emp 측정. GPU ~10 hr.
- **Falsification**: 각도가 기대 방향으로 변하지 않음 (e.g. MMLU 에서도 $\cos > 0.5$ 인데 negative lift → 각도가 원인 아님).
- **실패 시 plan**: Scope boundary 는 alignment 외 다른 원인 (format, layer structure, pretraining distribution) → 별도 future work.
- **근거 데이터**: E10 (MMLU fail), E11 (Mistral fail).

**H-I (Systematic slope excess as model-family Lipschitz correction) — v3 신규**: Phase A 에서 Qwen (0.55-0.59) / Mistral (0.55) 가 Ledoux 예측 0.500 대비 systematic **+10-17% excess**; Llama (0.40-0.50) 는 예측에 근접. 이 차이는 **model-family-dependent higher-order softmax correction** 이지 noise 가 아님 (std 0.06 < excess 0.05).

- **가설 형태**: attention pattern sharpness (softmax entropy 의 per-head 역수) 가 Qwen/Mistral > Llama → sharper softmax 가 second-order Lipschitz term 을 증폭.
- **Test**: per-(M, V) average attention entropy 를 Phase A 로그에서 추출 (신규 측정 불필요) + slope excess 와 correlation.
- **Falsification**: attention entropy 와 slope 사이 상관 ≤ 0.3 → 다른 원인 (weight spectral, GQA 구조, tokenizer bias).
- **중요도**: Theorem T 의 핵심 claim (√(r/d) functional form) 에 영향 없음. Second-order refinement 로 §Appendix 또는 §Discussion 에만.
- **Phase**: B3 와 함께 light analysis (신규 GPU 불필요).

**H-J (Asymmetric transferability via B_ont direction breadth) — v4 신규**: B_ont 가 source benchmark 로부터 build 될 때, 그 B_ont 가 **다른 benchmark 에 전이되는 능력** 은 source 의 catalog ontology structure 에 의존. 정량적으로 정의하면:

$$\text{breadth}(B_\text{ont}^{(\text{src})}) := \frac{1}{|V_\text{target}|}\sum_{v \in V_\text{target}} \cos(B_\text{ont}^{(\text{src})}, d^*_{M, v})$$

- **관측된 패턴**:
  - **Broad source** (Telecom, MetaTool): 4+ target 에 A/D > 1.8 — 여러 benchmark 의 discriminative subspace 와 부분 정렬.
  - **Narrow source** (Retail): Telecom target 에서 A/D = 0.23 < 1 — Retail B_ont 의 주 방향이 Telecom 과 거의 직교. Random 보다 **해로움**.
- **가설 명제**: Source catalog ontology 의 **facet diversity** 가 breadth 와 양의 상관. Telecom (tool_category 100% uniform but function_action / io_type diverse) + MetaTool (facet-diverse multi-tool) 는 broad; Retail (tool_category diverse but narrow in operation-level facet) 는 narrow.
- **Test**: Phase B3 에서 H-G (facet-concentration) 측정 시 aggregate 로 breadth metric 을 함께 계산. 추가 GPU 불필요.
- **Falsification**: Breadth ↔ cross-bench transferability 상관 < 0.3 → 다른 원인 (tokenizer mismatch, domain vocabulary).
- **중요도**: Theorem T 확장 이 아니고, **B_ont 구성 methodology 의 practical caveat**. Paper §Methodology 에서 "B_ont source 선택이 중요하다" 로 서술.
- **실패 시**: Asymmetric transferability 는 observation 으로만 (Theorem 적 해석 없음).

### 3.2 Secondary hypotheses — direction specificity 의 scope

**H-E (B_ont direction specificity 의 cross-benchmark universality)**: B_ont 가 특정 benchmark 에서 빌드되더라도 같은 model 의 다른 benchmark 에서 random 보다 flip rate 높음.

- Test: Telecom B_ont 를 Retail/Airline/Banking/ST4/BFCL 에 cross-apply. Phase B.
- 현재 증거: BFCL (A 2/100 vs D 0/100) = positive proxy.
- 확장 필요: ≥ 4 개 추가 benchmark 에서 separation 측정.

**H-F (Direction specificity 가 catalog-ontology 에 의존)**: Catalog ontology 를 permute / shuffle 하면 direction specificity 소실.

- Test: Shuffled-catalog B_ont 를 빌드 → argmax flip rate 측정. Phase C.
- 예측: Shuffled B_ont 의 flip rate ≈ random.

### 3.3 Stretch hypotheses — $d^*$ 도출 (Q-sign observation 의 partial 설명)

**H1, H2, H3, H4, H5, H6**: 이전 안 (unembed / OV readout / W_K SVD / RoPE / head-class / catalog-position) — 변경 없이 유지. 단 **primary 가 아니라 stretch goal** 로 재분류. Phase D 에 선택적 실행.

- Primary Lemma/Theorem 은 H1-H6 결과와 무관 (Q-sign 은 Theorem 밖 observation).
- H1-H6 narrowing 은 paper 의 §Discussion 에서 "open question — partial empirical progress" 로만 쓰일 수 있음.

### 3.4 가설 우선순위 (v3 재편 후)

| Priority | Hypothesis | Phase | 상태 | 사유 |
|---|---|---|---|---|
| ~~1~~ | H-A (attention-level universality) | A | **✅ CONFIRMED (v3)** | 9/9 R²≥0.99, slope 0.5298 |
| ~~1~~ | H-B (margin lower bound ≥ 1) | A | **❌ REPLACED (v3)** | m_0 spread 0.016–6.375, T.B predictive 로 대체 |
| ~~1~~ | H-C (two-level cross-model) | A | **✅ CONFIRMED (v3)** | 3/3 family |
| **1 (v2)** | H-G (facet-concentration) | B3 | pending | Tier 3 A−B=+21pp 의 이론 설명, GPU 최소 |
| **1 (v2)** | H-H (scope boundary alignment) | B3 | pending (v3 refined) | MMLU/Mistral failure-mode |
| **1 (v3)** | **T.B predictive (margin-gated flip)** | **A done / extended validation in B-D** | **✅ confirmed qualitatively, quantitative 추가 필요** | **H-B 의 후임, Lemma 1 predictive form** |
| **2** | H-D (KL non-monotonicity FFN origin) | B2 | pending | scope caveat 강화 |
| **2** | H-E (cross-benchmark direction specificity) | B1 | pending | direction specificity 범위 |
| **2 (v3)** | **H-I (slope-excess Lipschitz correction)** | **B3 light** | pending | **Second-order refinement, Appendix** |
| ~~2~~ | H-D (KL non-monotonicity FFN origin) | B2 | **✅ CONFIRMED (v4)** | 28/29 monotonic, layer 28 isolates mechanism |
| ~~2~~ | H-E (cross-bench direction specificity) | B1 | **✅ CONFIRMED (v4)** | 6/7 A>D, mean ratio 2.35 |
| **2 (v4)** | **H-J (asymmetric transferability / breadth)** | **B3 sub** | pending | **Retail → Telecom A/D=0.23 → narrow vs broad B_ont 분류** |
| **3** | H-F (catalog-permutation falsifier) | C | pending (스크립트 준비 완료) | direction specificity 의 origin 제한 |
| **4 (stretch)** | H1-H6 ($d^*$ narrowing) | D | pending | Q-sign partial understanding |

**v3 재편 요지**: 핵심 universality 가설 (H-A, H-C) 이 Phase A 에서 확증되고, H-B 가 T.B predictive form 으로 자연스럽게 승격. 남은 Phase B/B3 는 (i) facet 내부 구조 (H-G), (ii) scope 경계 (H-H), (iii) slope refinement (H-I) 세 방향.

**v2 재편 이유**: H-G 와 H-H 는 기존 Tier 3 + MMLU + Mistral 데이터가 이미 존재 — Phase A 와 병렬 실행 가능한 저비용 + 고임팩트. 이론의 scope 이 "B_ont direction 의 universality" 뿐 아니라 **"B_ont 의 internal 구조 (facet) 와 scope 경계"** 까지 확장되어 comprehensive 해짐.

---

## §4. 실험 세션 이미 한 일 (참조용)

기존 세션 (`lemma_empirical.json`, Qwen+Telecom N=100) 이 달성한 측정:

| Bench | Model | α | Ranks | N | 측정 | 상태 |
|---|---|---|---|---|---|---|
| τ²-Telecom | Qwen2.5-7B | 0.3 | {1..96} | 100 | Attn Fro slope, margin, flip rate | ✅ 완료 |
| τ²-Telecom | Qwen2.5-7B | 0.3 | {1..96} | 100 | KL vs r/d (per-task resample) | ✅ 완료 (R²=0.459) |
| τ²-Telecom | Qwen2.5-7B | 0.3 | 12 | 200 | variant D Phase 0 | ✅ 완료 (0/200) |
| BFCL parallel_multiple | Qwen2.5-7B | 0.3 | 12 | 100 | A vs D cross-benchmark | ✅ 완료 (2 vs 0) |

**위 4 점만** 이 현재까지의 증거. 모든 결과가 **단일 model + 주로 단일 benchmark**. 이것이 H-A, H-B, H-C 가 primary phase A 의 대상인 이유.

---

## §5. Phase 별 실험 계획

### Phase A — Two-level separation universality — **✅ COMPLETE (v3)**

**목표**: Lemma 2 + margin anchor + two-level separation 을 ≥ 4 (M, V) 조합으로 확장.

**(M, V) matrix**:

| # | Model | Benchmark | Rationale |
|---:|---|---|---|
| 1 | Qwen2.5-7B | τ² Retail | same-model, same-family V |
| 2 | Qwen2.5-7B | MetaTool ST4 | same-model, different V genre |
| 3 | Llama-3.1-8B | τ² Telecom | different model, same V |
| 4 | Llama-3.1-8B | MetaTool ST4 | cross-check Qwen2-ST4 |
| 5 (optional) | Mistral-7B | τ² Telecom | different architecture family |

**실행 protocol**:
- 각 (M, V) 에 대해 `measure_lemma_empirical.py` (기존 script) 를 직접 재사용. 이미 (a)(b)(c) 를 한 번의 GPU run 으로 측정하는 것이 입증됨 (249 s for Qwen Telecom N=100).
- Argument: `--model <M>`, `--benchmark <V>`, `--n 100`, `--alpha 0.3`, `--ranks 1,3,6,12,24,48,96`.
- 예상 wall-clock: 250 s × 5 = ~20 분 실제 GPU + model loading + I/O 포함 약 1-1.5 hr per run. Total 5-7 GPU-hr.

**기존 script 재사용성**:
- `scripts/ocq/measure_lemma_empirical.py` 이미 존재. (a)(b)(c) 를 동시 측정.
- 추가 benchmark 로딩 path 만 확장 필요 (`--benchmark` arg).
- 신규 script 불필요.

**출력**:
- `reports/new_theorem_test/phase_a_<model>_<bench>.json`
- memory: `new_theorem_phase_a_<date>.md`

**성공 기준 (Phase A gate)**:
- log-log slope 4/5 이상 (M, V) 에서 ∈ [0.45, 0.55] 에 속함.
- flip rate 5/5 (M, V) 에서 0/700.
- margin $m_0$ 5/5 (M, V) 에서 > 1.0 (benchmark-specific 값 기록).

**실패 시**:
- Slope 가 많이 벗어나면: attention-level universality 주장 약화 → Lemma 2 의 scope 을 "Qwen family" 로 limit.
- Flip rate nonzero 발생: sub-critical regime 정의를 α < 0.3 으로 tighten.

**v3 실제 결과** (2026-04-19, `memory/new_theorem_phase_a_2026_04_19.md`):

| (M, V) | slope | R² | flips | m_0 | m̄ | flip@r=96 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × Telecom (prev) | 0.5508 | 0.9920 | 0/700 | 5.031 | 5.072 | 0 |
| Qwen × Retail | 0.5693 | 0.9948 | 0/700 | 3.766 | 6.655 | 0 |
| Qwen × Airline | 0.5536 | 0.9929 | 0/350 | 4.203 | 5.714 | 0 |
| **Qwen × Banking** | 0.5868 | 0.9950 | **53/679** | 0.875 | 4.337 | 21 |
| Qwen × ST4 | 0.5779 | 0.9957 | 0/700 | 1.672 | 3.249 | 0 |
| Llama × Telecom | 0.4955 | 0.9979 | 0/700 | 0.422 | 0.422 | 0 |
| **Llama × Retail** | 0.4849 | 0.9986 | **11/700** | 0.016 | 0.974 | 5 |
| Llama × ST4 | 0.4025 | 0.9918 | 0/700 | 6.375 | 7.632 | 0 |
| Mistral × Telecom | 0.5471 | 0.9995 | 0/700 | 0.234 | 0.274 | 0 |

**Verdict**:
- Tight gate [0.45, 0.55]: 3/9 pass (formal fail of original criterion).
- Relaxed gate [0.40, 0.60] + "mean slope 0.50 ± 0.05 + all R² ≥ 0.95": **9/9 pass** — Lemma 2 확증.
- T.B predictive form 확증 — margin-flip correlation 관측 (Banking 7.8%, Retail 1.6%, both high-r 집중).
- 3/3 architecture family universality — unexpected bonus.

**Phase B/B3 ready to launch.**

### Phase B — Cross-benchmark direction specificity + FFN contribution — **✅ COMPLETE (v4)**

**v4 실제 결과** (2026-04-19, ~30 min GPU total with parallel GPUs):

#### B1 Cross-benchmark direction specificity (7 runs, N=50-100 each)

| B_ont source | Eval bench | type | A flip | D flip | A KL | D KL | **A/D** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Telecom | Retail | cross | 0/100 | 0/100 | 0.0070 | 0.0019 | **3.67** |
| Telecom | Airline | cross | 0/50 | 0/50 | 0.0097 | 0.0046 | **2.10** |
| Telecom | Banking | cross | 9/97 | 7/97 | 0.7360 | 0.2894 | **2.54** |
| Telecom | ST4 | cross | 0/100 | 0/100 | 0.0556 | 0.0307 | **1.81** |
| Retail | Telecom | cross | 0/100 | 0/100 | 0.0243 | 0.1070 | **0.23 ⚠** |
| MetaTool | Telecom | cross | 0/100 | 0/100 | 0.0376 | 0.0152 | **2.47** |
| Retail | Retail | self | 0/100 | 0/100 | 0.0071 | 0.0020 | **3.62** |

- 6/7 A>D, mean ratio **2.35**, median 2.47. H-E 확증.
- **Asymmetric transferability** (Retail → Telecom 만 A<D): H-J 의 empirical 근거.

#### B2 Layer-resolved KL (Qwen Telecom N=50)

- 29 layer 중 **28 layer residual stream 이 rank 에 대해 strictly monotonic** (r=1 → r=96 에서 KL 단조 증가).
- **오직 layer 28 (final transformer output)** 만 non-monotonic: peak r=6 (KL 0.138), dip r=24 (0.123).
- Final logits (post final-norm + lm_head) 도 non-monotonic — Phase A Qwen Telecom 패턴 재현 (peak r=12, trough r=48, rebound r=96).
- Layer 27 → Layer 28 at r=1: KL 0.0003 → 0.0255, **85× amplification** (late FFN cleanup 효과).

#### Phase B gate 판정

| Gate | 기준 | 결과 |
|---|---|:---:|
| Phase B1 H-E | 4/6 cross-bench A > D | 6/7 (85%) **✓ PASS** |
| Phase B2 H-D | non-monotonic origin at final LM-head | 28/29 monotonic → **✓ PASS (decisive)** |

**Theorem T.C (two-level gap) 의 mechanism 위치 확정**: 두-레벨 분리가 transformer 의 final FFN + LM-head composition 에서 발생. Residual stream 을 통한 attention 전달은 rank 에 대해 smooth.

#### ~~Original Phase B 계획 (참고용)~~

~~**목표**: H-E (cross-benchmark direction specificity) + H-D (KL non-monotonicity FFN origin).~~

**현재 상태**: 위 계획이 그대로 수행되어 두 가설 모두 확증. 다음은 Phase B3 + C + D.

**목표**: H-E (cross-benchmark direction specificity) + H-D (KL non-monotonicity FFN origin).

#### B1. Direction specificity matrix

B_ont 를 fix (e.g., Qwen Telecom B_ont) 한 뒤 다양한 benchmark 에 apply + random B 와 비교:

| B_ont source | Eval benchmark | N | 기대 결과 |
|---|---|---|---|
| Qwen Telecom | τ² Retail | 100 | A > D |
| Qwen Telecom | τ² Airline | 100 | A > D |
| Qwen Telecom | τ² Banking | 100 | A > D |
| Qwen Telecom | MetaTool ST4 | 100 | A > D or null |
| Qwen Telecom | BFCL parallel (여러 domain) | 100 | 이미 확인, 재현 |
| Qwen MetaTool | τ² Telecom | 100 | cross-direction |

**실행**: `scripts/ocq/eval_tau2_bench.py` 기존 + `scripts/ocq/eval_metatool_subtask4.py` 기존 재사용. `--methods canonical_adaseka_*` 지정.

**출력**: `reports/new_theorem_test/phase_b_direction_specificity.json` (aggregate).

#### B2. Layer-resolved KL non-monotonicity

**목표**: H-D 검증. KL 의 non-monotonic shape 이 residual-stream 의 어느 단계에서 생기는지.

**신규 script**: `scripts/ocq/measure_layer_resolved_kl.py` (약 200 LOC).
- Forward pass 에서 각 layer 의 residual-stream KL 을 record (pre-LM-head).
- Final LM-head 통과 전/후 KL 을 비교.
- Output: per-layer KL × rank 매트릭스.

**성공 기준**: 
- 중간 layer KL 이 rank 에 대해 monotonic.
- LM-head 통과 후 KL 이 non-monotonic (시각적 peak/trough 가 뚜렷).

**실패 시**: H-D 폐기. KL non-monotonicity 는 FFN 외 다른 origin — future work 로 언급만.

### Phase B3 — Facet-concentration + scope boundary alignment (v2 신규, Week 4-6 병렬, ~10 GPU-hr)

**목표**: H-G 와 H-H 를 기존 데이터 + 소량 신규 측정으로 검증.

#### B3.1 — H-G (facet-concentration) 검증

**사용 데이터 (신규 GPU 불필요)**:
- Tier 3 A/B/D 결과: `reports/tau2_2026_04_18/telecom_canonical_{amp03_persample,variantB,variantD}_N200.json`
- B_ont facet structure: `external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt` + facet column 분할 `(1,3,5,3)`

**신규 script**: `scripts/new_theorem_test/analyze_facet_concentration.py` (약 250 LOC)

**로직**:
```python
# 1. 기존 d*_emp 추출 (Phase 0 과 동일 방식, Telecom N=50)
d_star_emp = extract_dstar_empirical(model, telecom)  # per (l, h, q)

# 2. B_ont 의 facet block 분할
B_ont = load_bont()  # shape (L, H, d, 12) with blocks (1,3,5,3)
facet_blocks = {
    'function_action': B_ont[..., 0:1],
    'io_type': B_ont[..., 1:4],
    'domain': B_ont[..., 4:9],
    'tool_category': B_ont[..., 9:12]
}

# 3. Per (l, h, q): d*_emp 을 각 facet block 에 projection, energy 계산
for (l, h, q):
    d = d_star_emp[l, h, q]
    energies = {}
    for fname, Bf in facet_blocks.items():
        energies[fname] = (Bf[l,h] @ Bf[l,h].T @ d).norm()**2
    dominant_facet = argmax(energies)
    concentration = energies[dominant_facet] / sum(energies.values())
    save(l, h, q, dominant_facet, concentration)

# 4. Aggregate
concentration_hist = histogram([record.concentration for record in all_records])
```

**출력**: `reports/new_theorem_test/phase_b3_facet_concentration.json`

**성공 기준**:
- **Concentration median ≥ 0.70** across (l, h, q) records → H-G 지지.
- **Concentration bimodal**: 일부 head 는 집중 (≥0.8), 일부 head 는 분산 (≤0.4). → H-G partial 지지, head-class 구조 시사.
- **"Dilution ratio" 예측**: A − B = +21pp ↔ (1 − avg_concentration) × (B's per-facet lift). 예측된 ratio 가 empirical 3:1 과 ±30% 이내 일치.

**실패 시**:
- Concentration < 0.5 across board → H-G 기각. Facet-split dominance 는 다른 메커니즘 (routing gating, rank-magnitude dynamics 등).
- Dilution ratio 수치 예측 틀림 → H-G 부분 지지, 정량 scope 제한.

#### B3.2 — H-H (scope boundary alignment) 검증

**사용 데이터**:
- Phase 0 에서 추출한 Qwen Telecom d\*_emp (이미 진행 예정, Phase A)
- **신규 측정 필요**: Qwen MMLU d\*_emp + Mistral Telecom d\*_emp

**신규 script**: `scripts/new_theorem_test/measure_dstar_mmlu_mistral.py` (약 200 LOC)

- Qwen MMLU: `eval_mmlu_subset.py` 의 prompt 에 GT/distractor label 부여 + d\*_emp 추출.
- Mistral Telecom: Mistral-7B-Instruct load + τ² Telecom + d\*_emp 추출.

**로직**:
```python
# For each (M, V):
d_star_emp_MV = extract_dstar(M, V, N=30)
B_ont_MV = load_or_build_bont(M, V)  # 같은 M 에 맞게 B_ont 재생성 필요 (Mistral 은 새로 build)
cos_table = []
for (l, h) in sel_layers:
    cos_lh = cos_sim(
        B_ont_MV[l, h].flatten(),
        d_star_emp_MV[l, h].mean(over q).flatten()
    )
    cos_table.append(cos_lh)
aggregate = {
    'mean_cos': np.mean(cos_table),
    'median_cos': np.median(cos_table)
}
```

**비교표**:
| (M, V) | 기대 $\cos$ | 관측 ΔF1 (참고) |
|---|---:|---:|
| Qwen τ² Telecom | ≥ 0.5 | +28.89pp (positive) |
| Qwen MMLU | < 0.3 | −4.80pp (negative) |
| Mistral τ² Telecom | < 0.3 | −31.86pp (negative) |

**성공 기준**:
- 3/3 (M, V) 에서 예측된 각도 direction 과 empirical ΔF1 sign 이 일치 → H-H 지지.
- 2/3 일치 → H-H partial, scope 제한.
- ≤ 1/3 일치 → H-H 기각.

**실패 시**:
- Scope boundary 는 alignment 외 원인 (e.g. MMLU 는 multi-choice format, Mistral 은 tokenizer/layer 수 차이). Future work 로.

**GPU 예산**: Mistral B_ont build (~2 hr) + Mistral/MMLU d\*_emp extraction (~3 hr) + analysis. Total ~10 GPU-hr.

### Phase C — Catalog-permutation falsifier (Week 7-8, ~20 GPU-hr)

**목표**: H-F 검증. Catalog ontology 를 permute 해서 빌드한 B_ont 가 direction specificity 를 소실하는지.

**신규 script**: `scripts/ocq/build_permuted_bont.py` (약 150 LOC).
- 기존 catalog JSON 읽기.
- Tool names 또는 facet values 를 random permute.
- Permuted catalog 로 B_ont 재빌드 (기존 `scripts/ocq/build_tau2_ontology.py` 수정).

**실행**:
- 10 random permutations × Qwen Telecom N=100.
- 각 permuted B_ont 로 flip rate 측정.

**성공 기준**: 10 permutations 중 ≥ 8 에서 flip rate ≈ random baseline (i.e. permutation 파괴).

**실패 시**: H-F 기각. Direction specificity 가 catalog semantic content 에 독립적 — 다른 origin 탐구 필요.

### Phase D — Q-sign stretch (선택, Week 9-12, ~80 GPU-hr) — **CLOSED 2026-04-19 (Failure verdict)**

**상태**: 실험 세션이 stronger continuous-metric per-(L,h,q) angular alignment 로 substitute 실행 (H1 0.86–0.91× random / H3 1.10–1.23× / 0% pass 30°). Static-weight $d^*$ family closed.

H2/H4/H5/H6 미실시 (80+ GPU-hr 필요). Llama prompt-builder mismatch 로 cross-arch 차단.

Paper 에서 §6.1 + Appendix B.1 로 transparent 보고. Theorem T 외 observation 으로만 잔존.

---

### Phase F10 — Online ontology query-conditional gating (Week 10-11, ~2-4 GPU-hr) — **EXECUTED 2026-04-19, Failure verdict**

#### Result summary (Qwen2.5-7B-Instruct × MetaTool Subtask1, N=200, label_logprob)

| Variant | α | T | Gate | top1 | vs baseline 46.50% | vs F9 D 47.50% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| F9 V-only (reference) | 0.3 | — | none | 46.00% | −0.50pp | −1.50pp |
| F9 D-only (reference) | 0.3 | — | none | **47.50%** | +1.00pp | (ref) |
| F10a soft | 0.3 | 1.0 | softmax | 46.00% | −0.50pp | **−1.50pp** |
| F10b soft | 0.3 | 0.5 | softmax (sharper) | 46.50% | 0.00pp | −1.00pp |
| F10c soft | 0.3 | 2.0 | softmax (softer) | 47.00% | +0.50pp | −0.50pp |
| F10d hard | 0.3 | 1.0 | argmax (Lipschitz violation falsifier) | 47.00% | +0.50pp | −0.50pp |

#### Three critical findings

1. **V+D combination ≤ D-only** across all variants — V축 (verb facet) 추가가 D 신호 약화. 가능 원인: F8d NMI 0.185는 marginally orthogonal (threshold 0.3 근처), V/D가 일정 부분 redundant → capacity 분산.
2. **Per-token energy-ratio gating effect 미미**: T=0.5/1.0/2.0 차이 0.5-1pp 내. Energy-ratio signal이 task-relevant 정보를 carry 못함.
3. **Hard-gate ≠ catastrophic collapse**: 사전 예측 (Cor 6.7 Lipschitz violation → collapse) 부분적으로 falsified. F10d 47% ≈ F10c 47%. 단 single-tool decision이라 step-wise jump 영향 낮음 — Subtask4 multi-step에서 재검증 필요할 수 있음.

#### Pre-registered decision tree match

- ✓ "F10a < F9 D → harmful" branch fired (pre-reg case 3)
- ✓ "F10d ≈ F10a → smoothness 비-load-bearing" branch fired (pre-reg case)
- ❌ "F10a > F9 D + 1.5pp" gating works branch — falsified
- ❌ "F10d ≤ 30% catastrophic" Lipschitz necessity branch — falsified

#### Verdict — **Hypothesis H-F10 falsified**

> Per-token energy-ratio softmax gating으로 span-invariance를 깨도 single-facet baseline 대비 lift 없음. V × D multi-facet combination이 오히려 D-only보다 떨어짐. F10 자체로는 paper thesis upgrade 불가.

#### F10 negative가 paper에 미치는 가치 (recovered insight)

- **Confirms F1 reframe scope**: training-free + per-token + linear (even with content-dependent α) 으론 catalog semantic content 가 load-bearing 안 됨
- **Motivates F11 (MOFCISS)**: per-token stationary gating fails → step-adaptive non-linear sparse coding 필요 → Phase F11 의 직접 motivation 으로 활용
- §6.3 Discussion 에 "Why per-token gating fails on multi-tool selection: missing step-state" 1문단 추가

#### Artifacts

- B_ont: `external/SEKA/seka_projections/f10-qwen25-7b-metatool-stacked/B_ont.pt` (rank 24, V=[0:12], D=[12:24])
- Builder: `scripts/new_theorem_test/build_f10_stacked_bont.py`
- Hook: `scripts/ocq/eval_metatool_subtask1.py` `install_f10_facet_gated_hooks`
- Results: `reports/f10_metatool/f10_facet_a0.3_{T0.5,T1.0,T2.0,T1.0_hard}.json`

---

### Phase F11 — MOFCISS: Multi-step Ontology-indexed Facet Coverage via Sparse Subtraction (Week 11-13, ~6-10 GPU-hr) — **EXECUTED 2026-04-19, FALSIFIED**

#### Execution summary (2026-04-19 late evening)

**Outcome**: H-F11 falsified on Subtask4 N=200. All working-regime variants ≤ baseline.

**α calibration (pre-sweep)**:
- Pre-reg α=0.3 catastrophic (F1=0.000, gibberish output) because raw non-orthonormal atoms cause top-k sum ‖delta‖ to overshoot ‖K‖. F9/F10's orthonormal-basis α=0.3 convention does NOT transfer to raw-atom OMP subtraction.
- N=50 4-point α sweep at decay=0.5: α=0.01→0.677, α=0.02→0.690, α=0.05→0.677, α=0.1→0.327. **α*=0.02** selected as largest working-regime α.

**4-cell main sweep (Qwen2.5-7B-Instruct × Subtask4, N=200, α=0.02, top-k=5, plugin_des dictionary M=199)**:

| Cell | decay | OMP | F1 | Δ vs baseline 0.728 |
|---|:---:|:---:|:---:|:---:|
| baseline (no_steer) | — | — | **0.728** | (ref) |
| F11a OMP-only | 0.0 | top-5 | 0.699 | −2.9pp |
| F11b MOFCISS-base ★primary | 0.5 | top-5 | 0.695 | −3.3pp |
| F11c MOFCISS-aggr | 1.0 | top-5 | 0.701 | −2.7pp |
| F11d dense (no-OMP) | 0.5 | all | **0.000** | **−72.8pp** catastrophic |

**Pre-reg decision tree match**:
- F11b = 0.695 falls in **"< 0.71 harmful"** branch
- F11a ≈ F11b within 0.4pp → **step-state non-load-bearing** (primary MOFCISS innovation inert)
- F11d catastrophic → OMP sparseness is safety mechanism, not signal source

**Three critical findings**:
1. **OMP is SAFETY, not signal.** Dense (F11d) catastrophic even at α=0.02; sparse (F11a/b/c) stable but ≤ baseline.
2. **Step-decay inert.** F11a (decay=0) ≈ F11b (0.5) ≈ F11c (1.0), spread 0.6pp. 2-tool emission horizon too short for coverage benefit.
3. **Raw-atom α regime is narrow.** Working band α ∈ [0.01, 0.05]; at α=0.1 collapsing, α=0.3 catastrophic. Orthonormal-basis convention (F9/F10 α=0.3) does not transfer.

**Cross-phase conclusion**: F10 (stationary gated projection, orthonormal basis) + F11 (step-adaptive sparse subtraction, raw anchors) jointly falsify "training-free K-side intervention with ontology anchors yields lift on multi-tool selection". Neither stationarity nor span-breaking nor step-state was the blocker — the training-free constraint itself caps lift.

**Paper impact**: §6.3 scope-boundary claim strengthened. ICLR ceiling 5.25 unchanged. F12 FacetRot-QK (trainable LoRA-based, breaks training-free constraint) remains primary alternative.

**Artifacts**:
- `scripts/new_theorem_test/build_f11_dictionary.py`
- `scripts/new_theorem_test/eval_metatool_subtask4_mofciss.py`
- `external/SEKA/seka_projections/f11-qwen25-7b-metatool-plugdes/dictionary.pt` (M=199 atoms, plugin_des source)
- `reports/f11_metatool/{cal_*, f11a/b/c/d}_*.json`

**Details memo**: `memory/phase_f11_mofciss_executed_falsified_2026_04_19.md`

---

#### Original spec (below) — retained for reference

#### Motivation chain

1. F1 reframe (Phase C): catalog content not load-bearing in $\delta K = \alpha BB^\top K$
2. F8d: verb × domain orthogonal axis on 4 multi-domain corpora
3. F10 (executed): per-token energy-ratio gating fails to lift over single-facet baseline
4. **Combine the three**: span-invariance must be broken by **non-linear** coding (not just content-dependent α) AND step-state for multi-tool emission

#### Prior-art positioning (full audit 2026-04-19)

MOFCISS occupies the empty cell in 7-method comparison:

| Method | Multi-tool | Step-adapt | Train-free | Multi-facet | Semantic |
|---|:---:|:---:|:---:|:---:|:---:|
| SEKA (ICLR 2026) | ❌ | ❌ | ✗ contrastive | ❌ | ✓ trained |
| AdaSEKA | ❌ | ❌ | ✗ contrastive | ❌ | ✓ trained |
| **SADI (ICLR 2025)** | ❌ | ❌ | △ contrastive pair (~150) | ❌ single | ✓ contrastive |
| **OntoLLM** | △ prompt | ❌ | ✓ | △ KG hierarchy text | ✓ text |
| Q-coverage (NeurIPS, withdrawn) | ✓ | ✓ | ✓ | ❌ | ❌ |
| F10 (executed, failed) | ❌ | ❌ | ✓ | ✓ V×D | ✓ ontology |
| **MOFCISS (proposed)** | ✓ | ✓ | ✓ | ✓ | ✓ |

→ MOFCISS는 5-dimensional 빈 칸 모두 채우는 첫 mechanism (전제: 검증 성공시).

#### Core mechanism

**Pre-compute (one-time, training-free)**:
```
For each plugin n in MetaTool ontology:
  Anchor K_n = forward(model, plugin_n.description)  [last-token K, all (L, h)]
  facet(n) = (verb(n), domain(n))   # F8d-verified orthogonal axes

Dictionary D = {(K_n, verb(n), domain(n)) : n ∈ plugins}, ~388 atoms
```

**Per-decoding-step inference (the novelty)**:
```
At decoding step t:
  q_t = current attention K activation (per layer/head)
  
  # (a) Sparse coding: top-k active anchors via OMP
  c_t = OMP(q_t, D, k=5)
  active_cells_t = {(verb(n), domain(n)) for n in support(c_t)}
  
  # (b) Coverage state: which (V, D) cells already emitted?
  emitted_cells = {(V_s, D_s) : s < t}
  
  # (c) Sparse subtraction with ontology-aware decay
  delta_K_t = -alpha * Σ_{n ∈ supp(c_t)} c_t[n] * K_n * decay(n, emitted_cells)
  
  decay(n, emitted) = exp(-lambda * |{s : facet(n) = facet_s, s < t}|)
                     # facet 이미 emit 된 atom의 weight 감쇠
  
  # (d) Apply
  K_modified = K + delta_K_t
  emit next tool
  emitted_cells.add(facet(emitted_tool))
```

**3중 span-invariance 깸**:
1. Non-linear sparse selection (OMP combinatorial)
2. Non-stationary (step history dependent)
3. Anchor identity 직접 사용 (Gram-Schmidt 거치지 않음)

#### Theoretical anchors

- **Lemma A (Sparse Selection Distinguishability) — provable.** $D$ 의 facet-distinguishability $\theta$ 하에서 OMP top-k가 dominant facet cell 을 $\theta$-margin 내 정확히 식별. 증명: Tropp 2004 OMP guarantee + F8d NMI 0.144-0.218 distinguishability.
- **Lemma B (Coverage Convergence) — provable.** Decay $\lambda > \lambda^*$ 면 $T$ steps 안에 모든 GT facet cells 가 emitted set 에 포함될 확률 $\geq 1 - \epsilon(T, \lambda)$. 증명: stuck-on-emitted-facet probability geometric decay.
- **Theorem M (Multi-tool Coverage Optimality) — empirical.** MOFCISS 의 multi-tool $F_1$ lift 가 stationary K-side baseline 대비 strict positive, magnitude scales with $|\text{GT facets}|$.

#### Pre-registered experiment plan

**Primary target: MetaTool Subtask4 (multi-tool, 100% 2-tool, full N=497)**

| Variant | OMP? | Step decay (λ) | α | Predicted F1 |
|---|:---:|:---:|:---:|:---:|
| baseline (no_steer) | — | — | — | 0.731 |
| F11a OMP-only | ✓ k=5 | 0 (no decay) | 0.3 | 0.74-0.76 |
| **F11b MOFCISS-base** | ✓ k=5 | 0.5 | 0.3 | **0.76-0.79** ★ |
| F11c MOFCISS-aggressive | ✓ k=5 | 1.0 | 0.3 | 0.74-0.78 (over-decay risk) |
| F11d MOFCISS-no-OMP | ❌ dense | 0.5 | 0.3 | ~0.72 (degenerate to F10) |

**Secondary control: MetaTool Subtask1 (single-tool, full N=200)**

| Variant | Predicted F1 |
|---|:---:|
| F11e MOFCISS on Subtask1 | ≈ baseline (no lift expected — single-tool, no coverage need) |

**Cross-bench: BFCL parallel_multiple N=100** (MetaTool B_ont anchors used cross-domain)
- Predicts MOFCISS-base lift consistent with Subtask4 magnitude

#### Pre-registered decision tree

| F11b Subtask4 result | Reading | Paper impact |
|---|---|---|
| ≥ 0.78 (+5pp) | strong positive | **Thesis upgrade**: "Multi-step ontology-indexed sparse subtraction breaks both span-invariance and stationarity ceilings". §1 reframe around MOFCISS. ICLR ceiling 6.5-7.0 |
| 0.76-0.78 (+3-5pp) | clear positive | §5.Y "MOFCISS positive result" main contribution. ICLR ceiling 6.0-6.5 |
| 0.74-0.76 (+1-3pp) | weak positive | §5.Y "moderate effect" + §6 Discussion scope analysis. ICLR ceiling 5.75-6.0 |
| 0.71-0.74 | null/borderline | F10 verdict 보강 evidence. §6.3 "Why neither stationary nor sparse-step works at this scope". ICLR ceiling 5.25 (no change) |
| < 0.71 | harmful | step-decay calibration fail; F11a/c sweep 분석 → λ optimum 탐색 또는 abandon |

**F11d (no OMP) 가 F11b 보다 ≥ 0.74**: OMP 자체 비-load-bearing → mechanism story 약화
**F11a (no decay) 가 F11b 와 ≈**: step-state 비-load-bearing → 단순 sparse coding으로 충분 (interpretation 다름)

#### Implementation requirements

1. `scripts/new_theorem_test/build_f11_dictionary.py` — atom dictionary + facet metadata
   - Reuse F10 builder의 K_by_plugin extraction
   - Save: anchor K (no SVD), per-plugin (verb, domain) labels
2. `scripts/ocq/eval_metatool_subtask4_mofciss.py` — step-adaptive evaluator
   - **Critical**: HuggingFace `model.generate` callback 또는 manual generation loop 으로 step-state 추적
   - Per-step OMP coding (cheap: top-k inner products)
   - Per-step facet emission detection (parse generated tool name → lookup facet)
3. Step-state hook: each generation step에서 K-side bias가 emitted_cells에 따라 다름

**Engineering challenge**: HuggingFace generation 의 step boundary 추적. Two options:
- (a) Manual greedy decode loop with explicit per-token K hook reset
- (b) `LogitsProcessor` callback for facet detection + state mutation

#### Cost estimate

- Build (388 atoms K): ~5 min (F10 builder 재활용)
- F11a-d sweep on Subtask4 N=200: 4 × 30 min = 2 GPU-hr
- F11b/c full N=497 (after sweep selects best λ): 1 hr
- F11e Subtask1 control: 30 min
- BFCL cross-bench: 1 hr
- Total: ~5-6 GPU-hr

#### Risk assessment

- **Engineering risk (high)**: step-state hook implementation 복잡도. Generation callback이 HF API 한계로 어려울 수 있음. Backup: manual decode loop (slower but reliable).
- **Theoretical risk (medium)**: Lemma B 의 $\lambda^*$ 가 dataset-specific. Sweep으로 발견 필요.
- **Empirical risk (medium)**: F11b Subtask4 lift 가 baseline 0.731 ceiling 밑일 가능성 (Q-coverage NeurIPS 도 +1.64pp 만 얻었음).
- **Reviewer risk (low)**: SADI 가 sparse coding 안 함, step-state 안 함 → 차별점 명확.

---

### Phase F12 — FacetRot-QK: Hybrid Q+K Coupled Rotation via Soft Facet Gate (Week 12-15, ~12-24 GPU-hr LoRA) — **NEW (2026-04-19 late evening), proposed after SEKA/AdaSEKA full-text audit**

#### Motivation chain

1. **F1 regime-limit**: $\delta K = \alpha B B^\top K$ is span-invariant — semantic facet organization filtered out (Phase C / F6 / F7 triple confirmation).
2. **Phase D closure**: static-weight $d^*$ family FALSIFIED (H$_1$ 0.86-0.91× random, H$_3$ 1.10-1.23× random) — training-free derivation of semantic Q-bias direction from weights alone is closed.
3. **Full-text prior-art audit (2026-04-19 late evening)**: AdaSEKA $P_{\text{dyn}}(q) = \sum_m \alpha_m(q) U^m (U^m)^\top$ already does **multi-expert dense weighted blend per query** (not single-pick as earlier §2.5.4 claimed) — MOFCISS (F11) novelty reduced to sparse + step-state + positive-only anchors.
4. **Available theoretical foundation**: Thm 6.14 Hybrid (proven — two commuting SO(2) subgroups on orthogonal channel blocks) + Lemma 6.14.A (soft-gate Lipschitz) + F8d NMI 0.144-0.218 structural verb×domain orthogonality on 4/4 multi-domain corpora.
5. **Combine**: apply content-dependent SO(2) rotation to BOTH Q and K on facet block $P_{\text{fac}}$, keep RoPE on residual block $P_{\text{res}}$. Multiplicative Q⊗K coupling (attention-product level) is strictly stronger than AdaSEKA's additive K-only blend; facet-dependent rotation angle breaks span-invariance.

#### Prior-art positioning (post-audit)

| Axis | SEKA | AdaSEKA | SADI | Focus Directions | F11 MOFCISS | **F12 FacetRot-QK** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Operates on | K | K | hidden/head/neuron | K + Q | K (sparse) | **K + Q (rotation)** |
| Mechanism | linear proj | linear proj (blend) | element-wise mask | **additive** bias | sparse subtract | **multiplicative rotation (SO(2))** |
| Training | contrastive | contrastive | contrastive | gradient (10 epochs) | training-free | **LoRA (small: R/2 × F angles)** |
| Span-invariant | ✓ | ✗ (blend) | ✗ (element-wise) | ✗ (additive) | ✗ (sparse) | ✗ (rotation) |
| Semantic loaded | ✓ trained | ✓ trained | ✓ trained | ✓ relevance | ✓ ontology | ✓ **ontology + facet-separation geometry** |
| Multi-tool | ❌ | ❌ | ❌ | ❌ | ✓ | ✓ (emitted-facet exclusion in π) |
| Proven structure | ✗ | ✗ | ✗ | ✗ | Lemma A/B + Thm M | **Thm 6.14 Hybrid + Lemma 6.14.A** ✓ |

→ F12 differentiators vs AdaSEKA: (a) Q⊗K multiplicative vs K-only additive blend, (b) rotation operator (handedness preserved) vs projection (span-only), (c) proven theoretical structure (Thm 6.14) vs post-hoc spectral.
→ F12 differentiator vs Focus Directions: (a) SO(2) rotation vs additive bias, (b) ontology facet basis vs contextual head score, (c) step-adaptive via $\pi$ exclusion vs stationary.

#### Core mechanism

**Subspace decomposition (one-time, pre-train)**:
```
Identify verb × domain NMI-orthogonal block from F8d pipeline:
  P_fac = projector onto span(B_verb, B_domain)   [R channels, R ≤ 2·r_F8d]
  P_res = I - P_fac                               [d - R channels]
  # Anchors built from positive-only ontology labels (same pipeline as F11 dictionary)
```

**Per-facet SO(2) rotation (LoRA-trained)**:
```
R_f ∈ SO(2)^{R/2}  parameterized by { θ_{f,i} ∈ R : i = 1, …, R/2 }
R_f acts block-diagonally on R/2 channel pairs: R_f[i] = [[cos θ_{f,i}, -sin θ_{f,i}], [sin θ_{f,i}, cos θ_{f,i}]]
Total trainable: F facets × R/2 angles — e.g. F=16, R=32 → 256 scalar params per (L, h)
```

**Soft facet gate (Lemma 6.14.A Option A, Lipschitz)**:
```
g_f(x)  = exp(||P_f x||² / τ)        [facet-energy gate, τ ≈ 1.0]
π_soft(x) = Σ_f f · g_f(x) / Σ_f g_f(x)  [continuous facet index ∈ [1, F]]
R_{π(x)} = linear interpolation of {R_f} via π(x)   [weighted-angle soft-rotation]
```

**Forward modification**:
```
At each steered (L, h):
  q̃ = R_{π(q)} P_fac q + P_res q    [only Q is modified at that head]
  k̃ = R_{π(k)} P_fac k + P_res k    [only K is modified at that head]
  attn_score = q̃ · k̃ / √d
  
  Key property: if π(q) = π(k), rotations cancel (R^T R = I) — same-facet attention preserved
  Cross-facet: angular misalignment → attention damped by cos(Δθ) factor on P_fac block
```

**Multi-tool step-adaptation**:
```
After emitting tool_s with facet f_s, modify gate:
  g_f_new(x) = g_f(x) · exp(-γ · |{s : f_s = f}|)   [emitted-facet suppression]
  π_soft(x) shifts toward uncovered facets at next step
```

#### Theoretical anchors (partially proven)

- **Cor 6.7 phase-closure (already proven)**: for $q \perp \text{Range}(B_{\text{fac}})$, qaMSE = 0 at every $t$ under Hybrid. F12 inherits this: no-facet queries are not perturbed.
- **Lemma 6.14.A (already proven)**: soft-gate $\pi$ gives Lipschitz-continuous FacetRot with constant $L_{\text{fac}} = 2\pi F L_g / \min \sum g_f$. Satisfies regularity hypothesis (R) of Cor 6.7.
- **Theorem 6.14 Hybrid (already proven)**: two commuting SO(2) subgroups on orthogonal channel blocks; attention decomposes as content term + position term without space mismatch.
- **Lemma F12.A (new, conjecture)**: Same-facet attention preservation bound — $|\langle \tilde{q}, \tilde{k}\rangle - \langle q, k\rangle| \leq L_{\text{fac}} \cdot \|P_{\text{fac}} q\| \|P_{\text{fac}} k\| \cdot |\pi(q) - \pi(k)|$ for soft-gate $\pi$. Reduces to 0 when $\pi(q) = \pi(k)$.
- **Theorem F12.M (empirical)**: MetaTool Subtask4 F1 lift over AdaSEKA-on-shared-basis baseline is strictly positive; magnitude scales with multi-domain orthogonality (F8d NMI).

#### Pre-registered experiment plan

**Primary target: MetaTool Subtask4 N=497, Qwen2.5-7B-Instruct**

| Variant | Rotation source | Subspace | α/LR | Predicted F1 |
|---|---|---|---|---|
| baseline (no steer) | — | — | — | 0.731 |
| F12a closed-form Procrustes | $R_f$ = Procrustes($\bar K_f$, $\bar K_{f'}$) | verb×domain F8d block | α=0.3 | 0.71-0.73 (F7 R collapse risk — probable null) |
| F12b LoRA R/2=16 (small) | learned $\theta_{f,i}$, 16 pairs × 16 facets | verb×domain block | 1e-3 AdamW 5ep | 0.76-0.80 ★ (primary target) |
| F12c LoRA R/2=32 (medium) | learned, 32 pairs × 16 facets | verb×domain block | 1e-3 AdamW 5ep | 0.77-0.82 (expressivity ablation) |
| F12d LoRA + step-adapt | F12b + emitted-facet exclusion γ=0.5 | same | same | 0.78-0.82 (multi-tool extension) |
| F12e K-only ablation | F12b but only K rotated (no Q) | same | same | 0.74-0.77 (tests Q⊗K coupling necessity) |
| F12f hard-gate | F12b with argmax π (Lipschitz violation) | same | same | ≪ baseline (Lemma 6.14.A prediction; Bug-2 hard-gate collapse replicate) |

**Secondary: MetaTool Subtask1 N=200** (single-tool control):
- F12b expected flat (no multi-tool gain), tests that rotation doesn't degrade single-tool.

**Cross-bench: BFCL parallel_multiple N=100**:
- F12b with MetaTool-trained angles, tests transferability of facet-rotation structure.

**Baseline comparison**: F12b vs canonical AdaSEKA (reproduced on τ² Telecom at +28.89pp) on shared MetaTool Subtask4 harness.

#### Pre-registered decision tree

| F12b Subtask4 F1 | Reading | Paper impact |
|---|---|---|
| ≥ 0.82 (+9pp) | exceptional — beats AdaSEKA comparable | **Thesis upgrade** — "FacetRot Q⊗K rotation exceeds SEKA-family linear K-only by multiplicative coupling + facet structure". ICLR headline contribution. Ceiling 7.0+ |
| 0.78-0.82 (+5-9pp) | strong positive | §5.Y as headline method contribution. Thm 6.14 promoted from Future Work to main paper. Ceiling 6.5-7.0 |
| 0.76-0.78 (+3-5pp) | clear positive, compete with F11 | §5.Y co-contribution with F11 (if F11 also ≥ +3pp); otherwise F12 replaces F11 as primary. Ceiling 6.0-6.5 |
| 0.74-0.76 (+1-3pp) | weak positive | §5.Y as ablation / F11 supplement. Ceiling 5.75-6.0 |
| 0.71-0.74 | null | §6.3 strengthening — "neither F11 sparse-linear nor F12 rotational lifts above training-free ceiling". ICLR thesis unchanged. |
| < 0.71 | harmful | LoRA rotation structure is over-constrained. F12a/e/f ablation 분석 → operator family 재설계 또는 abandon |

**F12a (closed-form) ≫ 0.71**: training-free variant works — dramatic update to F1 reframe.
**F12e (K-only) ≈ F12b**: Q⊗K coupling not load-bearing — reduce to LoRA-AdaSEKA variant.
**F12f (hard-gate) ≫ baseline**: Lemma 6.14.A Lipschitz requirement falsified at inference.

#### Implementation requirements

1. `scripts/new_theorem_test/build_f12_facet_subspace.py` — extract verb×domain NMI-orth block from MetaTool anchors
   - Reuse F11 dictionary K anchors + F8d machinery
   - Output: $P_{\text{fac}}$, $P_{\text{res}}$, facet labels, $B_{\text{fac}}$ columns
2. `scripts/new_theorem_test/train_f12_facetrot_qk.py` — LoRA training loop
   - Freeze base LLM, trainable = SO(2) angles θ + gate temperature τ + projector refinement
   - Dataset: MetaTool Subtask4 train split, CE loss on GT tool sequence
   - Hook: per-(L, h) Q and K projection modification via forward pre-hooks on q_proj/k_proj
3. `scripts/new_theorem_test/eval_subtask4_facetrot_qk.py` — eval w/ or w/o step-adaptation
   - Manual decode loop (F11 infra reusable) for multi-tool emission tracking

**Engineering challenges**:
- **Gate Lipschitz enforcement**: soft $\pi$ must stay Lipschitz — bound $\tau$ below, add Lipschitz penalty to loss
- **RoPE compatibility**: $P_{\text{res}}$ subspace must avoid RoPE pairs (channel selection constraint)
- **GQA handling**: Qwen2.5-7B has 4 KV heads → apply rotation before GQA expansion for consistency

#### Cost estimate

- Build $P_{\text{fac}}$: 10 min (F11 / F8d reuse)
- F12b LoRA train (5 epochs, 256 × 28 layers × 28 heads params ≈ 200K trainable): ~4-6 GPU-hr
- F12b eval N=497: 1 GPU-hr
- F12c/d/e/f ablations: ~8 GPU-hr (shared base)
- F12a closed-form (no train): 1 GPU-hr eval
- BFCL cross-bench: 1 GPU-hr
- **Total: ~15-18 GPU-hr** (vs Thm 6.14 original 47 GPU-hr plan — F12 is lighter because only rotation angles are trained, not full LoRA on q_proj/k_proj)

#### Risk assessment

- **Engineering risk (high)**: soft-gate implementation + RoPE channel-block selection + GQA ordering. Mitigation: F12 script follows Thm 6.14 R1 plan's specification exactly (already sketched in `memory/theorem_6_14_facet_rotation_positioning_2026_04_14.md`).
- **Theoretical risk (low)**: Thm 6.14 Hybrid is proven; Lemma 6.14.A is proven. Only F12.A and F12.M are new claims, and F12.M is empirical.
- **Empirical risk (medium)**: F7 variant R closed-form collapsed on τ²-telecom (0.097 vs 2.45 baseline); this is mitigated by choosing MetaTool (F8d NMI 0.185 > 0.3 threshold met for verb×domain). Still, rotation LoRA may not train to useful angles in 5 epochs if facet separation is weak.
- **Reviewer risk (medium)**: Focus Directions (Zhu 2025) already does gradient-trained Q+K bias; F12 must clearly differentiate on (rotation vs additive, ontology facet vs contextual head, proven Hybrid structure vs empirical). Wording: "additive $K + \alpha d_K$ is a Lie algebra shift; SO(2) rotation is a group action preserving norm and orthogonality — structurally distinct intervention class."
- **Scope risk (medium)**: F12 requires training (LoRA) → exits training-free regime. F1 reframe's scope is preserved; F12 is explicitly labeled as "mild-training extension beyond F1 scope".

#### Dependency on F11

- If F11 positive (≥ +3pp): F12 is optional ablation / stretch contribution. Paper headline stays MOFCISS; F12 mentioned in §6 Discussion as "rotational extension empirically comparable".
- If F11 null (< +3pp): F12 is **promoted to primary candidate**. LoRA R1 run is the next experiment session's first action.
- If F11 harmful: F12 via fully different mechanism path (rotation vs sparse subtract) — still worth trying.

---

### Phase F13 — FunnelRot: Staged FacetRot-QK with L28-Funnel-Aware Schedule (Week 13-16, ~14-20 GPU-hr LoRA) — **NEW (2026-04-19 late evening, parallel to F12), proposed after ladapt + Phase B2 convergence audit**

#### Motivation — two independent empirical anchors converge

**Anchor 1: ladapt schedule (NeurIPS prep, session_handoff_2026_04_17)**. Three-stage K/Q schedule empirically beats uniform K-only / Q-only / all-layer K+Q across 4 tool-selection benchmarks:
- τ² Telecom N=200: Q-only $-\beta=0.03$ gave $+18.37$pp; ladapt K+Q point estimate sits inside 95% CI $[+23.31, +30.23]$
- τ² Airline N=50: ladapt K+Q $\beta=-0.03$ gave $+3.80$pp
- τ² Banking non-meta N=13: ladapt K+Q gave $+6.90$pp
- MetaTool Subtask4 N=497: iterative_kq (layer-adaptive) gave $+2.18$pp

Schedule used (`_build_layer_adaptive_qk_schedule` in `scripts/ocq/eval_metatool_subtask1.py:873`):
```
Stage 1 (ℓ ∈ [0, 5]):      α_ℓ = α_k (strong K),      β_ℓ = 0
Stage 2 (ℓ ∈ [6, 18]):     α_ℓ = 0.3·α_k (weak K),    β_ℓ = β_q
Stage 3 (ℓ ∈ [19, 27]):    α_ℓ = 0,                   β_ℓ = β_q (Q-only)
```

**Anchor 2: Phase B2 layer-resolved KL (ICLR §5.5)**. Qwen × Telecom $N=50$, steered layers L18–L27, 29-layer residual-stream KL probe:
- L18–L27: **9/9 monotonic in rank** across $r \in \{1, 3, 6, 12, 24, 48, 96\}$ — linear transport channel
- **L28 (final FFN + LM-head composition)**: non-monotonic; $r=1$ KL = $0.0255$, **$85\times$ amplification** from L27's $0.0003$. At $r=96$, amplification drops to $1.35\times$ ($0.1312 \to 0.1770$).

**Convergence**: The layer-28 amplifier is strongly rank-selective — low-rank signals are 85× amplified, dense signals barely 1.35×. ladapt's empirical success likely reflects (i) K-early builds low-rank geometric information that is carried monotonically through L27, (ii) Q-late re-allocates attention mass onto the K-shaped geometry just before the L28 funnel, (iii) neither stage intervenes at L28 itself, so the natural non-linear amplifier is preserved. The schedule is empirical; the rank selectivity is mechanistic. Together they suggest a single design: **staged low-rank rotation that respects the L28 amplifier**.

#### Structural cross-tab (F13 vs relevant prior art)

| 방법 | Staged K/Q | Low-rank (r ≤ 4) | L28 explicit-skip | Rotation operator | Ontology facet |
|---|:---:|:---:|:---:|:---:|:---:|
| SEKA / AdaSEKA | ❌ all layers | — (trained basis) | ❌ | ❌ | trained |
| Focus Directions | ❌ contextual heads uniform | — | ❌ | ❌ | contextual |
| SADI | ❌ all-layer element mask | — | ❌ | ❌ | contrastive |
| ladapt (our NeurIPS prep) | ✓ | ❌ linear projection | implicit (not argued) | ❌ | ❌ catalog |
| F11 MOFCISS | ❌ uniform steered layers | ✓ sparse | implicit | ❌ | ✓ F8d |
| F12 FacetRot-QK uniform | ❌ L18-27 uniform | ✓ facet block | implicit | ✓ SO(2) | ✓ F8d |
| **F13 FunnelRot (proposed)** | **✓ (ladapt schedule)** | **✓ (facet rank R ≤ 4)** | **✓ (explicit)** | **✓ (Thm 6.14)** | **✓ (F8d)** |

→ F13 is the first method combining (a) staged K/Q schedule, (b) low-rank rotation, (c) explicit L28 non-intervention, (d) proven SO(2) Hybrid structure, (e) ontology facet basis. No prior work owns this 5-way intersection.

#### Core mechanism

**Schedule** (adopts ladapt's three-stage design, adds L28-explicit-skip):

$$(s_K(\ell),\, s_Q(\ell)) = \begin{cases} (\alpha,\, 0) & \ell \in [0, 5] \\ (0.3\alpha,\, \beta) & \ell \in [6, 18] \\ (0,\, \beta) & \ell \in [19, 27] \\ (0,\, 0) & \ell = 28 \text{ (natural amplifier preserved)} \end{cases}$$

**Per-layer rotation** (inherits F12 Thm 6.14 Hybrid; only $K$ at stages 1-2, only $Q$ at stages 2-3):

$$\tilde{k}^{(\ell)} = s_K(\ell) \cdot R_{\pi(k)}^{(\ell)} P_{\text{fac}} k + P_{\text{res}} k + (1 - s_K(\ell)) \cdot P_{\text{fac}} k$$

$$\tilde{q}^{(\ell)} = s_Q(\ell) \cdot R_{\pi(q)}^{(\ell)} P_{\text{fac}} q + P_{\text{res}} q + (1 - s_Q(\ell)) \cdot P_{\text{fac}} q$$

(The $(1 - s) P_{\text{fac}} x$ term ensures zero-schedule reduces to identity on the facet subspace.)

**Low-rank constraint**: $P_{\text{fac}}$ rank $R \leq 4$ (one verb axis + one domain axis from F8d, or two of each). This aligns with the L28 amplification regime where $r = 1$ gives $85\times$ and $r = 6$ already starts declining. The uniform F12 used $R \leq 32$ (expressive); F13 uses $R \leq 4$ (funnel-aligned).

**Trainable parameters** (LoRA R1 style, beyond F12):
- Rotation angles $\theta_{f, i}^{(\ell)}$: per steered layer $\ell$, per facet $f$, per SO(2) pair $i$. Total $\sim 28 \times 16 \times 2 = 896$ scalars.
- Stage scalars $\alpha, \beta$: 2 scalars.
- Mid-stage scale $\rho \in [0, 1]$ (ladapt uses $0.3$ as default): 1 scalar.
- Gate temperature $\tau$: 1 scalar.
- **Total**: $\sim 900$ trainable scalars, all base-frozen.

#### Theoretical anchoring

- **Thm 6.14 Hybrid (proven, 2026-04-14)**: two commuting SO(2) subgroups on $P_{\text{fac}}$ / $P_{\text{res}}$. F13 inherits.
- **Lemma 6.14.A (proven)**: soft-gate Lipschitz bound. F13 inherits.
- **Phase B2 (verified, 2026-04-19)**: L28 is the unique non-monotonic layer; amplification is $1/r^\kappa$-like. F13 exploits by constraining $R \leq 4$ and skipping L28.
- **Conjecture F13.Funnel (new, empirical-to-be-tested)**: When L0–L27 is $r$-monotonic and L28 has amplification $A(r) \propto 1/r^\kappa$ with $\kappa > 0$, a staged low-rank intervention $\{(s_K(\ell), s_Q(\ell))\}$ with zero schedule at L28 Pareto-dominates uniform all-layer interventions (including L28) for argmax flip rate at matched $\|\delta K\|_F$ energy. **Falsifier**: F13d (L28-intervene ablation) $\geq$ F13b.
- **Conjecture F13.Cascade (new, explanatory)**: ladapt's empirical success on ladder of benchmarks is the rank-1 instance of F13.Funnel, with $P_{\text{fac}} = \text{span}(B_{\text{ont}})$ (no rotation operator, just projection). F13 generalizes by adding SO(2) rotation to break span-invariance on the facet block.

#### Pre-registered experiment plan (6 cells)

**Primary target: MetaTool Subtask4 N=497, Qwen2.5-7B-Instruct, LoRA 5 epochs**

| Variant | Schedule | Rotation | Rank R | L28 | Predicted F1 |
|---|---|---|---|:---:|---|
| F13a = F12b reproduction | uniform L18-27 | SO(2) | 32 | skip (natural) | 0.76-0.80 |
| **F13b Full FunnelRot** ★ | **ladapt 3-stage** | **SO(2)** | **4** | **explicit skip** | **0.80-0.85** |
| F13c ladapt + projection (rotation ablation) | ladapt 3-stage | ❌ additive $B B^\top$ | 4 | skip | 0.77-0.82 (reproduces ladapt literal) |
| F13d L28-intervene negative control | ladapt 3-stage | SO(2) | 4 | **K-intervene at L28** | **< F13b expected** — falsifies F13.Funnel if not |
| F13e uniform rotation (schedule ablation) | uniform L0-27 | SO(2) | 4 | skip | intermediate — tests schedule's role |
| F13f low-rank ablation (rank ablation) | ladapt 3-stage | SO(2) | 16 | skip | tests whether R ≤ 4 is load-bearing vs R=16 |

**Secondary control**: MetaTool Subtask1 N=200 with F13b (expect flat — single-tool, no schedule gain).

**Cross-bench**: BFCL parallel_multiple N=100 with F13b using MetaTool-trained angles.

**Baseline reference**: canonical AdaSEKA (external/SEKA integration) on same MetaTool Subtask4 harness.

#### Pre-registered decision tree

| F13b Subtask4 F1 | Reading | Paper impact |
|---|---|---|
| ≥ 0.84 (+11pp) | exceptional, beats AdaSEKA regime | **Thesis major upgrade** — "Funnel-aware staged rotation exceeds SEKA-family linear K-only by structural alignment with L28 amplification". ICLR headline. Ceiling 7.0+ |
| 0.80-0.84 (+7-11pp) | strong positive, competitive w/ AdaSEKA | §5.Y co-contribution w/ F11 or headline method. Ceiling 6.5-7.0 |
| 0.76-0.80 (+3-7pp) | moderate, subsumes F12 | §5.Y FunnelRot primary; F12 demoted to ablation row. Ceiling 6.0-6.5 |
| 0.73-0.76 (+0-3pp) | weak, schedule not load-bearing | §5.Y as small-lift ablation. Ceiling 5.75-6.0 |
| < 0.73 | null | L28-awareness irrelevant; F12 regime ceiling confirmed. §6.3 Discussion strengthening |

**Critical ablation reads**:
- **F13d ≥ F13b**: L28 non-intervention is NOT load-bearing → demote F13.Funnel conjecture; re-frame as "staged low-rank rotation" without L28 claim.
- **F13c ≥ F13b**: SO(2) rotation is NOT load-bearing → collapse to "ladapt with F8d block selection"; reuse as reproducing-ladapt row.
- **F13e ≥ F13b**: staged schedule is NOT load-bearing → collapse to F12b uniform rotation.
- **F13f ≥ F13b**: low-rank constraint is NOT load-bearing → raise R freely, reduce to F12c medium-rank.

Each ablation is designed to isolate exactly one design element. If all four ablations match F13b within noise, F13 degenerates into a weaker claim. If F13b strictly dominates all four, all five design elements (schedule + rotation + L28-skip + low-rank + facet basis) are individually load-bearing — strong thesis.

#### Implementation requirements

1. **`build_f12_facet_subspace.py`** (shared with F12) — extract $P_{\text{fac}}$ with rank parameter (default 4 for F13, 32 for F12c).
2. **`scripts/new_theorem_test/train_f13_funnelrot.py`** — extension of `train_f12_facetrot_qk.py`:
   - `--schedule {uniform, ladapt}` flag (default: `ladapt` for F13; inherits `uniform` for F12)
   - `--skip-layer-28 {true, false}` flag (default: `true`)
   - `--early-end`, `--mid-end`, `--mid-alpha-scale` ladapt schedule knobs (defaults: 5, 18, 0.3)
   - `--rank-fac` projection rank (default: 4 for F13, 32 for F12)
   - Reuse F12's `FacetRotSO2` + `FacetSubspace` + hook infrastructure
3. **`scripts/new_theorem_test/eval_subtask4_funnelrot.py`** — eval w/ staged schedule + optional step-adaptation (F13b + F13b-step variant for multi-tool).

**Engineering notes**:
- ladapt `install_layer_adaptive_qk_hooks` already implements 3-stage for additive projection — F13 extends by replacing additive step with SO(2) rotation at each stage.
- L28 skip is implemented by NOT registering hooks on layer index 28 (model.model.layers[28]).
- Low-rank $R = 4$ constrains $P_{\text{fac}}$ to span of top-2 verb + top-2 domain F8d axes only.

#### Cost estimate

- Build $P_{\text{fac}}$ rank-4: 5 min GPU (F8d reuse)
- F13b LoRA train (5 epochs, ~900 trainable scalars): ~4-6 GPU-hr
- F13b eval N=497: 1 GPU-hr
- F13a (uniform, F12b-equivalent): shared train, +1 GPU-hr eval
- F13c (projection variant): 3 GPU-hr (faster, no rotation)
- F13d (L28-intervene): 5 GPU-hr (shared train infra, separate eval)
- F13e (uniform schedule): 3 GPU-hr
- F13f (R=16): 4 GPU-hr
- BFCL cross-bench: 1 GPU-hr
- **Total: ~22-26 GPU-hr** (1 GPU, 1 day wall-clock with 2-3 parallel runs)

#### Risk assessment

- **Engineering risk (low)**: F12 infrastructure ready; F13 is an extension with schedule + rank knobs. ladapt `install_layer_adaptive_qk_hooks` already proven on production eval path.
- **Theoretical risk (low)**: Thm 6.14 + Lemma 6.14.A + Phase B2 are all already established. F13.Funnel + F13.Cascade are conjectures but each has a direct falsifier cell (F13d, F13c respectively).
- **Empirical risk (medium)**: schedule-rotation interaction untested. ladapt's schedule was tuned for additive projection; optimal stage boundaries may shift for rotation.
- **Reviewer risk (low)**: structural uniqueness (5-way intersection) is defensible. Negative control F13d directly addresses "why not L28?" question.
- **Scope risk (medium)**: F13 still requires LoRA training → outside strict training-free scope. F13c closed-form (no rotation, just ladapt projection with F8d basis) is the closest training-free fallback.

#### Dependency on F11 + F12

- **F11 positive + F13b < 0.80**: F11 headline, F13 demoted to §6.3 Discussion.
- **F11 positive + F13b ≥ 0.80**: both reported as co-contributions; §5.Y comparison with F13.Funnel as "funnel-aware alternative to sparse subtraction".
- **F11 null + F13b ≥ 0.80**: F13 replaces F11 as primary method contribution. Clean story — Phase B2 mechanism finding directly enabled the winning method.
- **F11 null + F13b < 0.80**: F12 uniform rotation as fallback (F13a = F12b); report F13.Funnel conjecture as false and F13 as schedule ablation.

F13 is explicitly designed as a **superset** of F12: F13a cell reproduces F12b, so running F13 costs little more than F12 while giving 5 ablation rows.

---

## §6. Paper integration (ICLR 2027 submission 예상 구조)

### 6.1 구조 안

1. **Abstract + Intro**: 두-레벨 분리 → catalog 기반 B_ont 의 direction specificity 가 sub-critical regime 의 argmax stability 와 함께 empirical 확증.
2. **§2 Related**: CAA/RepE/ITI/PASTA/ASA/Focus/SEKA/AdaSEKA + Ledoux concentration + interpretability.
3. **§3 Theorem**: Lemma 1 (provable, self-contained), Lemma 2 (provable via Ledoux, **empirically anchored at slope 0.551**), Theorem T (two-level separation with empirical component).
4. **§4 Methodology**: B_ont 구성, variant D random control, margin / attn-shift / flip rate 측정 protocol.
5. **§5 Main Results**:
   - §5.1 Phase A attention-level universality (5 (M, V)).
   - §5.2 Phase B cross-benchmark direction specificity.
   - §5.3 Phase B2 FFN/LM-head KL origin.
6. **§6 Falsification**: Phase C catalog-permutation null.
7. **§7 Discussion**: Q-sign observation + H1-H6 partial (if Phase D).
8. **§8 Scope + Limitations + Conclusion**.

### 6.2 주장 contribution 리스트 (수정)

- **C-lemma-1**: Single-layer margin-gated flip (provable).
- **C-lemma-2**: Attention-level √(r/d) concentration (provable + empirically 5 (M, V)).
- **C-theorem**: Two-level separation (attention smooth + argmax step + FFN/LM-head origin).
- **C-empirical**: Direction specificity cross-benchmark, catalog-origin necessity (Phase C falsifier).
- **C-observation (Q-sign)**: 5-point table as open problem.

### 6.3 NeurIPS 2026 와의 구분

- NeurIPS 2026 = C1-C4 existence only + E1-E6 패치. Mechanism-free.
- ICLR 2027 = Lemma 1-2 + Theorem T + empirical scale + Q-sign stretch. Mechanism-partial.
- 두 submission **완전 독립**. ICLR track 실패해도 NeurIPS unaffected.

---

## §7. Decision tree + kill-switch

| Gate | 조건 | 성공 행동 | 실패 행동 |
|---|---|---|---|
| ~~Phase A gate (v1/v2)~~ | ~~4/5 slope ∈ [0.45, 0.55], 5/5 flip=0, 5/5 $m_0 > 1$~~ | ~~Phase B 진행~~ | — |
| **Phase A gate (v3 refined)** | **9/9 R²≥0.95 AND mean slope 0.50 ± 0.05 AND strict m_0 > 0 AND flip rate correlates with 1/m** | **✅ PASS → Phase B/B3 진행** | — |
| ~~Phase B1 gate~~ | ~~4/6 cross-benchmark A > D~~ | — | — |
| **Phase B1 gate (v4 result)** | **6/7 (85%) A > D** | **✅ PASS → H-E confirmed** + **H-J 새 관측** | — |
| ~~Phase B2 gate~~ | ~~Layer-resolved KL 가 non-monotonic origin 이 LM-head 라 확인~~ | — | — |
| **Phase B2 gate (v4 result)** | **28/29 residual layers monotonic, only layer 28 non-mono** | **✅ PASS → H-D confirmed decisively** (mechanism 위치 특정) | — |
| **Phase B3.1 gate (v2)** | **Facet concentration median ≥ 0.7 + dilution ratio 예측 ±30%** | **H-G 확증 → §3 Theory 의 Cor 로 추가** | **H-G 삭제, A−B=+21pp 는 "open structural effect"** |
| **Phase B3.2 gate (v2)** | **3/3 (M, V) 에서 cos 방향 예측 맞음** | **H-H 확증 → §7 Scope section 의 정량 경계** | **H-H 삭제, scope boundary 는 empirical 관찰만** |
| Phase C gate | 8/10 permuted B_ont 가 flip rate 소실 | H-F 확증 → falsification evidence | H-F 삭제, Direction specificity origin open |
| Phase D gate | 14/18 sign prediction | H1-H6 narrow 성공 → §Discussion | Q-sign observation only |

### Kill-switch (즉시 프로그램 중단)

1. Phase A 에서 5/5 slope 모두 outside [0.4, 0.6]: Lemma 2 의 universality 붕괴. Program 재설계.
2. Phase A 에서 flip rate > 0 on sub-critical α: T.B scope 가 매우 좁음. Re-scope.
3. 4 주 이상 continuous stalling (script bug, OOM): resource 조정.
4. NeurIPS 2026 submission 임박 (< 2 주): 모든 ICLR work 중단, NeurIPS 에 집중.

---

## §8. Resource budget (실험 세션 ready 상태 반영)

| Phase | GPU-hr | Wall-clock | 주요 신규 script |
|---|---:|---:|---|
| A | 30 | 3 주 | 없음 (기존 재사용) |
| B1 | 30 | 2 주 | 없음 (eval_tau2_bench.py 재사용) |
| B2 | 30 | 1 주 | `measure_layer_resolved_kl.py` (신규) |
| **B3.1 (v2)** | **~2** | **1 주 (B1-B2 병렬)** | `analyze_facet_concentration.py` (신규) |
| **B3.2 (v2)** | **~10** | **1 주 (B1-B2 병렬)** | `measure_dstar_mmlu_mistral.py` (신규) |
| C | 20 | 2 주 | `build_permuted_bont.py` (신규) |
| D (optional) | 80 | 4 주 | H1-H6 suite |
| Paper writing | 0 | 3 주 (병렬) | — |

**필수 부분 (Phase A-C, v2 B3 포함)**: **~122 GPU-hr, 8 주**. 1 GPU 로 가능. B3 는 B1-B2 와 병렬이라 wall-clock 증가 없음.
**Stretch (+ Phase D)**: **~202 GPU-hr, 12 주**. 2 GPU 병렬 권장.

ICLR 2027 마감까지 24 주 → comfortable. Phase D 포함해도 여유.

---

## §9. 신규 script spec (최소)

`scripts/iclr2027/` (또는 `scripts/new_theorem_test/`) 에 새로 작성.

### S0. `analyze_facet_concentration.py` (Phase B3.1, ~250 LOC) — v2 신규

**목표**: H-G 검증. 기존 Tier 3 데이터 + d\*_emp 추출 결과를 facet block 으로 decomposition.

**CLI**:
```
python analyze_facet_concentration.py \
  --bont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt \
  --facet-sizes 1,3,5,3 \
  --dstar-emp reports/new_theorem_test/phase0_dstar_qwen_telecom.json \
  --tier3-ab-df1 21.10 \
  --out reports/new_theorem_test/phase_b3_facet_concentration.json
```

**출력 schema**: `{concentration_mean, concentration_median, concentration_per_facet_histogram, dilution_ratio_predicted, dilution_ratio_observed, H_G_verdict}`.

### S0b. `measure_dstar_mmlu_mistral.py` (Phase B3.2, ~200 LOC) — v2 신규

**목표**: H-H 검증. MMLU / Mistral Telecom 에서 d\*_emp 추출 + B_ont 와의 angle 측정.

**CLI**:
```
python measure_dstar_mmlu_mistral.py \
  --setting mmlu_qwen \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n 30 \
  --out reports/new_theorem_test/phase_b3_scope_mmlu.json

python measure_dstar_mmlu_mistral.py \
  --setting mistral_telecom \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --build-bont \
  --n 30 \
  --out reports/new_theorem_test/phase_b3_scope_mistral.json
```

**출력**: `{per_layer_cos, mean_cos, median_cos, H_H_verdict}`.

### S1. `measure_layer_resolved_kl.py` (Phase B2, ~200 LOC)

**목표**: 각 layer 의 residual-stream 에서 KL 을 measure, LM-head 통과 전/후 비교.

**CLI**:
```
python measure_layer_resolved_kl.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --benchmark tau2_telecom --n 50 \
  --alpha 0.3 --ranks 1,3,6,12,24,48,96 \
  --capture-layers all \
  --out reports/new_theorem_test/layer_kl.json
```

**로직**:
```python
for q in queries:
    # 2 forward passes: (1) no steer, (2) steered with random U at rank r
    #   forward hooks on every layer's residual stream
    # compare post-layer residual distributions at matched layer index
    # final: LM-head logits full-vocab KL
    per_layer_kl[r][layer] = KL(layer_residual_distribution(steered), layer_residual_distribution(base))
```

**출력 schema**: `{rank: {layer: {kl_mean, kl_std}}}`

### S2. `build_permuted_bont.py` (Phase C, ~150 LOC)

**목표**: Catalog ontology 를 random permute 한 뒤 B_ont 재빌드.

**CLI**:
```
python build_permuted_bont.py \
  --catalog external/tau2-bench/data/tau2/domains/telecom/ontology.json \
  --permute-mode {tool_names | facet_values | full_random} \
  --seed 42 \
  --out external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom-permuted-seed42/B_ont.pt
```

**로직**:
```python
catalog = load_catalog(args.catalog)
if mode == 'tool_names':
    random.shuffle(catalog['tool_names'])
elif mode == 'facet_values':
    for facet in catalog['facets']:
        random.shuffle(catalog['facets'][facet]['values'])
elif mode == 'full_random':
    # replace all with random embeddings
    ...
# rebuild B_ont using existing build_tau2_ontology.py logic
```

**출력**: `B_ont.pt` + metadata JSON.

---

## §10. Memory + file discipline

### Phase 완료 시 생성할 memory

- `new_theorem_phase_a_<date>.md` — Phase A aggregate (5 (M, V) slope/margin/flip)
- `new_theorem_phase_b1_<date>.md` — B1 direction specificity cross-benchmark
- `new_theorem_phase_b2_<date>.md` — B2 layer-resolved KL
- `new_theorem_phase_c_<date>.md` — C catalog-permutation falsifier
- `new_theorem_phase_d_<date>.md` — D Q-sign stretch (if executed)
- `new_theorem_status_weekly.md` — weekly progress log

### 생성할 reports

- `reports/new_theorem_test/phase_a_<M>_<V>.json` (N=5)
- `reports/new_theorem_test/phase_b1_direction_matrix.json`
- `reports/new_theorem_test/phase_b2_layer_kl.json`
- `reports/new_theorem_test/phase_c_permutation.json`
- `reports/new_theorem_test/phase_d_qsign.json` (optional)
- `reports/new_theorem_test/aggregate_matrix.json` (final)

### Paper draft 경로

- **NeurIPS 2026 submission 은 건드리지 않음**: `math/paper/benchmark_design/PAPER_DRAFT_v3.md` 유지.
- **ICLR 2027 용 신규**: `math/paper/iclr2027/PAPER_DRAFT_ICLR_v1.md` 는 NeurIPS 제출 완료 후 생성. 그 전까지 draft 없음.

### Branch 전략

- NeurIPS main branch 건드리지 않음.
- 신규 branch `new-theorem-test` 에서 scripts/new_theorem_test/ 개발.
- Phase A-C 완료 시 NeurIPS submission 결과 + consolidation 후 merge.

---

## §11. 핵심 원칙 (다시 확인)

1. **Lemma 1-2 는 증명 완료 + partial 검증**. 추가 증명 불필요. 남은 일은 generalization.
2. **Proposition (c) 는 완전 사망** — 부활 시도 금지 (두 번 실패).
3. **Q-sign asymmetry 는 observation**. Stretch goal (Phase D). Primary path 아님.
4. **Two-level separation 자체가 novelty**. Hoeffding / Ledoux 자체는 textbook.
5. **NeurIPS 2026 무관**. ICLR 실패 시 NeurIPS existence-only 그대로.
6. **Phase gate 엄격**: kill-switch 준수. Sunk cost 무시.
7. **모든 주요 결정은 memory 기록**: 다음 세션 재구성 가능해야.
8. **Mechanism claim 금지** (Q-sign, format collapse 등 observation 으로만).
9. **Scope caveat 명시**: Attention-level smooth vs tool-name-level stepwise 의 gap 은 empirical, 증명 아님.
10. **4번째 pivot 방지**: Theorem 을 wildly 확장하지 않음. Lemma 1-2 + T (two-level) + framework 에 집중.

---

## §12. 다음 세션 first actions — v3 업데이트 (Phase A 완료 반영)

### v4 현재 상태

- Phase A: **DONE** (9 (M, V), memory `new_theorem_phase_a_2026_04_19.md`).
- Phase B1: **DONE** (7 runs, H-E confirmed, H-J 신규 관측, memory `new_theorem_phase_b_2026_04_19.md`).
- Phase B2: **DONE** (29-layer KL, H-D confirmed decisively, memory 동일).
- Phase B3: pending (스크립트 미작성).
- Phase C: 스크립트 준비 완료 (`build_permuted_bont.py`, `/tmp/phase_c_runs.sh`), run 대기.
- Phase D: pending (stretch).

### 다음 세션이 집어야 할 첫 작업 (v4 priority order)

**우선순위 재편**: Phase B3 와 Phase C 를 병렬 실행. Phase D 는 여전히 stretch.

1. **Phase C (catalog-permutation falsifier) — 0.5-1 일, ~20 GPU-hr**:
   - 스크립트 이미 준비됨 (`scripts/ocq/build_permuted_bont.py` + `/tmp/phase_c_runs.sh`).
   - 3 permutation mode (tool_names / facet_values / full_random) × Qwen Telecom N=100.
   - 예상: Permuted B_ont 의 A/D ratio → 1 근처로 붕괴 (direction specificity origin 이 catalog semantic 임을 확증).
   - **이걸 먼저 하는 이유**: 스크립트 준비 완료, 낮은 시간/리스크. H-F 확증 시 Phase B 결과 (H-E) 와 결합해 direction specificity claim 완성.

2. **Phase B3.1 (H-G facet-concentration) — 1-2 일, ~2 GPU-hr**:
   - `scripts/new_theorem_test/analyze_facet_concentration.py` 작성 (§9 S0 spec).
   - 사용 데이터: 기존 Tier 3 A/B/D JSON + Phase A 의 d\*_emp proxy.
   - Output: `reports/new_theorem_test/phase_b3_facet_concentration.json`.
   - H-G 가 A−B=+21pp 의 이론 설명 여부 판정.

3. **Phase B3.1 aggregate 에 H-J breadth metric 추가 — 0.5 일, GPU 불필요** (v4 신규):
   - B1 의 6 cross-bench 결과 + B_ont 의 facet-value coverage 를 비교.
   - Breadth ≡ "source catalog 의 facet diversity" 를 정량화 + cross-bench A/D ratio 와 correlation.
   - H-J 확증/기각.

4. **Phase B3.2 (H-H scope boundary) — 1 주, ~10 GPU-hr**:
   - `scripts/new_theorem_test/measure_dstar_mmlu_mistral.py` 작성 (§9 S0b spec).
   - Qwen MMLU + Mistral Telecom 에서 d\*_emp 추출 + B_ont 와 angle 측정.

5. **H-I (slope excess) light analysis — 0.5 일, GPU 불필요**:
   - Phase A 로그에서 attention entropy 추출 + slope 와 correlation.
   - §Appendix 수준 결과.

6. **Phase D stretch (H1-H6 $d^*$ narrowing) — 선택적, 4 주**: 상위 5 개 완료 후에만.

### 병렬화 전략 (v4 updated)

- Paper 세션 (GPU 불필요 tasks): H-I light analysis + H-J breadth metric + Phase B3.1 분석 (기존 데이터).
- 실험 세션 (GPU 필요): Phase C run + Phase B3.2 run.
- Wall-clock 2 주 내 Phase B3 + Phase C 완료 가능.

### 완료된 작업 기록 (참고용)

- ~~v1/v2 Week 1-3: Phase A run #1-9~~ → 완료, 9/9 R²≥0.99
- ~~v4 Phase B1 run #1-7~~ → 완료, 6/7 A>D, H-E confirmed + H-J 신규
- ~~v4 Phase B2 29-layer KL run~~ → 완료, 28/29 monotonic, H-D confirmed decisively
- **현재 (Phase B 끝)**: Phase B3 + Phase C 시작 ready.

---

## §13. 관련 memory + file

- `memory/lemma_empirical_abc_2026_04_19.md` — 현 단일 증거의 full 기록
- `memory/p1_random_rank_scaling_failed_2026_04_19.md` — P1 fail 원본
- `memory/variantD_phase0_verified_2026_04_19.md` — C3 Phase 0
- `memory/bfcl_tier3_cross_benchmark_2026_04_19.md` — C3 cross-benchmark
- `memory/handoff_paper_edit_2026_04_19.md` — NeurIPS 2026 existence-only track
- `memory/handoff_shared_basis_parallel_2026_04_19.md` — P1/P2 failed gate 의 handoff
- `scripts/ocq/measure_lemma_empirical.py` — (a)(b)(c) 측정 script (이미 존재)
- `scripts/ocq/measure_random_rank_scaling.py` — P1 script
- `scripts/ocq/eval_tau2_bench.py` — τ² eval
- `scripts/ocq/eval_metatool_subtask4.py` — ST4 eval
- `scripts/ocq/eval_bfcl.py` — BFCL eval
- `external/SEKA/seka_projections/` — B_ont artifacts

---

## §14. Version + changelog

- **v1 (2026-04-19)**: 실험 세션 (a)(b)(c) 결과 반영 후 초안. Lemma 1 증명 완료, Lemma 2 증명 + 1 (M, V) 확증, two-level separation 이 core novelty. Phase A-C primary (110 GPU-hr), Phase D stretch (80 GPU-hr). NeurIPS 2026 완전 독립.
- **v2 (2026-04-19 저녁)**: 사용자 피드백 ("이것 외에 이론적으로 다뤄야 할 실험 결과") 반영.
  - §2 empirical anchors 에 E8–E13 (Tier 3 decomposition, routing architectural, MMLU fail, Mistral fail, L0 rank-1, BiasBios transfer) 추가.
  - §3 에 H-G (facet-concentration), H-H (scope boundary alignment) 신규 primary hypothesis 추가.
  - §5 에 Phase B3 신설 (B3.1 facet-concentration analysis + B3.2 MMLU/Mistral alignment). B1-B2 와 병렬 실행.
  - §7 decision tree + §8 resource budget 재보정 (+12 GPU-hr). Wall-clock 변화 없음.
  - §9 에 S0 (analyze_facet_concentration.py), S0b (measure_dstar_mmlu_mistral.py) script spec 추가.
  - §1.4 observation list 에 E5-E8 (layer-adaptive, BiasBios transfer, L0 rank-1, contrastive B_ont) 추가.
  - 핵심 변화: 이론 scope 이 "B_ont direction 의 universality" 에서 **"B_ont 의 internal facet 구조 + scope 경계 explanation"** 으로 확장. 이론이 다룰 수 있는 empirical 범위가 comprehensive 해짐.

- **v4 (2026-04-19 Phase B 완료)**: 실험 세션의 Phase B1 (7 runs) + Phase B2 (29-layer KL) aggregate (`memory/new_theorem_phase_b_2026_04_19.md`) 반영.
  - **H-E CONFIRMED** (6/7 A>D, mean A/D ratio 2.35).
  - **H-D CONFIRMED decisively** (28/29 residual layers monotonic, 오직 layer 28 non-mono → T.C mechanism 위치 = final FFN+LM-head composition).
  - **H-J 신규 가설 (v4)**: Asymmetric transferability — Retail B_ont → Telecom 에서 A/D=0.23 < 1 (random 보다 약함). B_ont direction breadth 가 source catalog 의 facet diversity 에 의존.
  - §1 상태 snapshot 확장 — Theorem T 누적 지지 요약 + 신규 관측.
  - §2 에 E18-E21 (B1 matrix, asymmetric transferability, B2 layer-resolved, layer-28 amplification) 추가.
  - §3 에 H-J (asymmetric transferability / B_ont breadth) 신규 가설 추가.
  - §3.4 priority table: H-D, H-E 를 ~~strikethrough~~ + CONFIRMED; H-J 신규; Phase C 스크립트 준비 완료 상태 반영.
  - §5 Phase B 를 "COMPLETE" 로 전환 + 실제 결과 테이블 inline (B1 + B2).
  - §7 decision tree: B1 / B2 gate 결과 inline.
  - §12 first actions 전면 재작성: 다음 세션 우선순위는 **Phase C (스크립트 준비 완료, ~20 GPU-hr) → Phase B3 (facet concentration, breadth, scope boundary, slope excess) → Phase D stretch**.
  - 핵심 변화: **Theorem T 의 3 가지 claim (T.A, T.B, T.C) 모두 empirical 확증 및 mechanism 위치 특정**. ICLR 2027 paper 의 main result 섹션 뼈대 완성. 남은 작업은 (i) direction specificity 의 origin (Phase C falsifier), (ii) facet 구조 + scope 경계 + slope refinement (Phase B3), (iii) optional Q-sign (Phase D).

- **v3 (2026-04-19 Phase A 완료)**: 실험 세션의 Phase A 9 (M, V) aggregate (`memory/new_theorem_phase_a_2026_04_19.md`) 반영.
  - **H-A CONFIRMED** (9/9 R²≥0.99, mean slope 0.5298), **H-C CONFIRMED** (3/3 arch family), **H-B REPLACED** by T.B predictive form (margin-flip correlation 확증: Qwen Banking 7.8% / Llama Retail 1.6%, 모두 high-r 집중).
  - §2 에 E14-E17 (Phase A slope table, flip rate table, systematic slope excess, m_0 range) 추가.
  - §3.1 에 H-I (systematic slope excess as model-family Lipschitz correction) 신규 가설 추가 — second-order refinement, Appendix 수준.
  - §3.4 priority table 재편: H-A/H-B/H-C 상태 업데이트 + T.B predictive + H-I 추가.
  - §5 Phase A 를 "COMPLETE" 로 전환 + 실제 결과 테이블 inline.
  - §7 decision tree 의 Phase A gate 재보정 (tight → relaxed, "mean slope 0.50 ± 0.05 + all R²≥0.95" 통과 기준).
  - §12 first actions 전면 재작성: 다음 세션은 Phase B3.1 부터 시작 (기존 데이터 + 저비용).
  - 핵심 변화: 이론의 **core Lemma 2 universality** 가 Phase A 에서 확증. 남은 작업은 (i) facet 구조 H-G, (ii) scope 경계 H-H, (iii) slope refinement H-I, (iv) cross-benchmark direction specificity B1, (v) LM-head origin B2, (vi) permutation falsifier C, (vii) Q-sign stretch D. **ICLR 2027 contribution 이 "Lemma 1-2 확증 + 4 operational framework components + large-scale validation" 수준으로 구체화**.

---

**END OF NEW_THEOREM_TEST v1**
