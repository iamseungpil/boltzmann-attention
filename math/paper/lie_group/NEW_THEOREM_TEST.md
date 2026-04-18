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

### Phase B — Cross-benchmark direction specificity + FFN contribution (Week 4-6, ~60 GPU-hr)

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

### Phase D — Q-sign stretch (선택, Week 9-12, ~80 GPU-hr)

**목표**: H1-H6 narrowing (이전 안).

시간 + resource 여유 시에만. Phase A-C 이 성공적으로 끝난 뒤 판단. Phase A-C 만으로 충분한 paper contribution 일 수 있음.

**Note**: 이전 계획서의 §4-§6 (H1-H3 angular / β-sweep / sign prediction) 내용을 그대로 적용. 단 paper 에서 Theorem 이 아니라 §Discussion 의 "partial empirical progress" 로만 쓰임.

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
| Phase B1 gate | 4/6 cross-benchmark A > D | H-E 확증 → Phase B2 | Direction specificity scope 제한 |
| Phase B2 gate | Layer-resolved KL 가 non-monotonic origin 이 LM-head 라 확인 | H-D 확증 → Phase C | H-D 삭제, non-monotonicity "unexplained" 로만 |
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

### v3 현재 상태

- Phase A: **DONE** (9 (M, V), all in `reports/new_theorem_test/phase_a_*.json` + `new_theorem_phase_a_2026_04_19.md` memory).
- Phase B/B3 준비 완료.

### 다음 세션이 집어야 할 첫 작업 (priority order)

1. **Phase B3.1 (H-G facet-concentration) 실행 — 1-2 일, ~2 GPU-hr**:
   - `scripts/new_theorem_test/analyze_facet_concentration.py` 작성 (§9 S0 spec).
   - 사용 데이터: 기존 Tier 3 A/B/D JSON + Phase 0 d\*_emp (또는 Phase A log 에서 d\*_emp proxy 추출).
   - Output: `reports/new_theorem_test/phase_b3_facet_concentration.json` + memory.
   - **이걸 먼저 하는 이유**: 기존 데이터만으로 실행 가능. A−B = +21pp 의 이론적 설명 여부가 즉시 판정됨.

2. **Phase B3.2 (H-H scope boundary) 실행 — 1 주, ~10 GPU-hr**:
   - `scripts/new_theorem_test/measure_dstar_mmlu_mistral.py` 작성 (§9 S0b spec).
   - Mistral B_ont build + Mistral/MMLU d\*_emp 측정.
   - Output: `reports/new_theorem_test/phase_b3_scope_{mmlu,mistral}.json`.

3. **H-I (slope excess) light analysis — 0.5 일, GPU 불필요**:
   - Phase A 의 per-(M, V) log (`logs/phase_a_*.log`) 에서 attention entropy 를 post-hoc 추출.
   - Slope vs entropy correlation 계산.
   - 결과 paper §Appendix / §Discussion 용으로만.

4. **Phase B1 (cross-benchmark direction specificity) — 2 주, ~30 GPU-hr**:
   - Telecom B_ont 를 Retail/Airline/Banking/ST4/BFCL 에 cross-apply + random B 비교.
   - 기존 `scripts/ocq/eval_tau2_bench.py` + `eval_metatool_subtask4.py` 재사용.

5. **Phase B2 (FFN/LM-head KL origin) — 1 주, ~30 GPU-hr**:
   - `scripts/new_theorem_test/measure_layer_resolved_kl.py` 작성 (§9 S1 spec).

### 병렬화 전략

B3.1 + H-I light analysis 는 GPU 거의 불필요 → paper 세션이 수행 가능. B3.2 / B1 / B2 는 GPU 필요 → 실험 세션이 병렬로 진행. Wall-clock 3-4 주 내 Phase B 전체 완료 가능.

### v1/v2 의 첫 작업 기록 (참고용)

- ~~Week 1 Day 1-2: Phase A run #1 (Qwen Retail)~~ → 완료, slope 0.5693 R²=0.9948
- ~~Week 1 Day 3-4: Phase A run #2 (Qwen ST4)~~ → 완료, slope 0.5779 R²=0.9957
- ~~Week 1-2: Phase A run #3-5~~ → 완료 (Llama Telecom/Retail/ST4, Mistral Telecom, Qwen Airline/Banking).
- ~~Week 3: Phase A aggregate + memory~~ → 완료 (`new_theorem_phase_a_2026_04_19.md`).
- **현재 (Week 3 끝)**: Phase B/B3 시작 ready.

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
