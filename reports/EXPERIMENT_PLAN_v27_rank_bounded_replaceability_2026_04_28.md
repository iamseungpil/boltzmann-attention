# 실험 계획 v27: Rank-Bounded Prompt Replaceability (2026-04-28)

> **Working title**: "Prompt Internalization for Agentic LLMs via KV-Side Intervention — Rank-Bounded Replaceability and Query-Conditional Correction"
>
> **상위 문서**: `math/paper/iclr2027_prompt_internalization/PAPER_DRAFT_v0.md` (이론 정리 + 증명)
>
> **무관 문서 (untouched)**: `math/paper/iclr2027/PAPER_DRAFT_ICLR_v1.md` (two-level argmax-subspace selectivity, 별 thesis)

---

## 0. 한 페이지 요약

**문제.** Agentic LLM (Graphify / Anthropic Opus 4.7 agentic harness 등)은 도구·계획 정의를 위해 *수천 토큰 길이의 시스템 프롬프트*를 매 호출마다 부담한다. TTFT, KV cache 메모리, 갱신 비용이 모두 이 prefix에 비례.

**목표.** 이 prefix prompt $P$를 **frozen LLM 위의 KV-side intervention**으로 대체하면서 도구·계획 선택 정확도를 동등 이상 유지.

**중심 가설.**  Prefix prompt $P$가 attention output에 기여하는 함수
$$\Phi_P(q) := \lambda_P(q)\cdot \mathrm{attn}(q;K_P,V_P)\quad (\text{He et al. 2022, Eq. 6})$$
의 task-distribution $\mathcal Q$ 위에서의 **effective rank** $r^*$가 작을 때만, 정적 rank-$k$ intervention(현 $B_{\mathrm{ont}}$ 계열)으로 prompt 대체가 *이론적으로* 가능. 큰 경우 query-conditional intervention $f:q\mapsto V_{\mathrm{steer}}(q)$가 필요.

**판정 가능한 단일 측정.** $r^*(\tau{=}0.95)$ on $\mathcal Q \in \{$MetaTool ST4, $\tau^2$-bench retail/telecom/airline$\}$, layer/head 별.

**결과 시나리오 (모두 paper가 됨).**
- $r^* < 16$ → 정적 $B_{\mathrm{ont}}$로 prompt 대체 정당화. Phase 1 자산 그대로 main figure.
- $r^* > 64$ → Query-conditional 보정 필수. 현 Q-bias가 *왜* 작동하는지(1차 Taylor 보정으로) 이론적 설명. F13b seed bimodality 자연 해석.
- 중간 → 둘이 정확히 trade-off하는 영역. Q-bias = 1차 보정 정리가 핵심 기여.

---

## 1. 방향 검증

### 1.1 의도에 부합하는가?

| 항목 | 판정 |
|---|---|
| 카톡(4/20) "포커스 랭크 이동으로 프롬프트 대체" | ○ — 본 thesis와 직결 |
| 카톡 "프롬프트 없이 온톨로지만으로" | △ — $r^*$가 작은 영역에서만. 이번 측정으로 결판 |
| 카톡 "QK rank ≡ Graph RAG (He 확장)" | ✗ — He에서 안 나옴. 본 plan에서 다루지 않음 |
| 기존 자산($B_{\mathrm{ont}}$, Q-bias, MetaTool/τ²-bench harness) | ○ — 전부 재사용 |

### 1.2 선행 연구와의 차별

| 선행 | 대체 대상 | 측정 | 우리와의 차이 |
|---|---|---|---|
| Prefix-tuning (Li & Liang 2021) | task prompt | task acc | **학습 필요**, query-conditional 아님 |
| He et al. 2022 unified view | — (이론) | — | Form 정리만, content/rank bound 없음 |
| Akyürek 2023, von Oswald 2023 (ICL≡GD) | ICL prompt | linear regression | 우리 task(agentic tool selection)와 setup 다름 |
| Petrov, Torr, Bibi 2024 (prompting limits) | prefix tuning | expressivity | Convex hull bound, *rank* bound 아님 |
| Hyperdecoders / HyperLoRA | task prompt | NLP task | **학습 기반 query-conditional**, frozen 아님 |
| LLMLingua / 500xCompressor | prompt 토큰 압축 | task acc | **토큰 단위** 압축, KV-space intervention 아님 |
| GIST tokens (Mu et al. 2023) | instruction prompt | task acc | 압축된 KV로 학습, frozen LLM 아님 |
| SEKA / AdaSEKA / our $B_{\mathrm{ont}}$ | facts (CounterFact) | edit success | Tool selection 아님, *prompt 대체* angle 아님 |

**결론**: "Frozen LLM, *rank-bounded* prompt replacement, *query-conditional* 보정의 필요성을 함수공간 SVD로 정량화" 자리는 비어있음.

---

## 2. 이론적 가설 (paper 본문에 정리)

증명은 paper draft에 있음. 여기서는 측정 가능한 statement만 정리.

**Setup.** Frozen decoder LLM. Layer $\ell$, head $h$. Task query 분포 $\mathcal Q$. Prefix prompt $P$. He 2022 Eq. 6에 의해 attention output은
$$o^{(\ell,h)}(q) = (1-\lambda_P^{(\ell,h)}(q))\cdot \mathrm{attn}(q; K_x, V_x) + \underbrace{\lambda_P^{(\ell,h)}(q)\cdot \mathrm{attn}(q; K_P, V_P)}_{=:\Phi_P^{(\ell,h)}(q)}.$$

**Definition (effective rank).** $\Phi_P^{(\ell,h)}: \mathcal Q \to \mathbb R^{d_h}$. 행렬 $M^{(\ell,h)} := [\Phi_P^{(\ell,h)}(q_1), \dots, \Phi_P^{(\ell,h)}(q_N)]^\top \in \mathbb R^{N\times d_h}$ 의 SVD $M = U\Sigma V^\top$ 에 대해
$$r^*(\tau) := \min\Bigl\{ k : \tfrac{\sum_{i\le k}\sigma_i^2}{\sum_i \sigma_i^2} \ge \tau \Bigr\}.$$

**Hypothesis H1 (rank-bounded replaceability).** rank-$k^*$ static intervention $V_{\mathrm{steer}}\in\mathbb R^{d_h\times k^*}$ ($k^*\ge r^*(\tau)$) 가 존재하여, Eckart–Young에 의해 prefix-attn output을 layer/head별 $L_2$ 손실 $\le \sqrt{1-\tau}\cdot\|\Phi_P\|$ 안에서 근사.

**Hypothesis H2 (query-conditional necessity).** $r^*$이 큰 layer/head에서, *어떤* 정적 rank-$k$ intervention도 task acc upper bound가 query-conditional intervention 대비 정확히 $1-\tau$만큼 낮음.

**Hypothesis H3 (Q-bias as 1st-order correction).** 현 Q-bias $\beta\cdot(B_{\mathrm{ont}} B_{\mathrm{ont}}^\top)Q$ 는 $\Phi_P$의 $q$-방향 1차 Taylor 보정과 일치하는 형태이며, 그 sign이 $\partial_q \lambda_P(q)$의 sign과 정합할 때만 정확도 개선. **이게 retail β−, telecom β+ regime flip의 메커니즘 설명 후보.**

---

## 3. 실험 (E1–E6)

### E1. Effective Rank Measurement (가장 중요)

**의도.** $r^*$를 직접 측정. Hypothesis H1 vs H2의 결판.

**가설 (반증 가능).**
- 강한 정적 가설(H1): 모든 layer × head에서 $r^*(0.95) \le 16$.
- 강한 query-cond 가설(H2): 평균 $r^*(0.95) \ge 64$.
- 사후 가설: head 분포에서 **bimodal** — 일부는 작고(static-OK) 일부는 큼(needs query-cond).

**검증 방법.**
- 모델: Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (cross-family).
- Prefix prompt: 각 task 표준 system prompt (MetaTool ST4 보유 / τ²-bench retail-telecom-airline 보유).
- Query 분포 $\mathcal Q$: 해당 task의 user query 256–512개.
- 측정:
  1. 각 query $q_i$ 에 대해 standard forward pass. Layer $\ell$, head $h$의 attention output 중 prefix-position 기여분만 추출:
     $$\Phi_P^{(\ell,h)}(q_i) = \sum_{p\in P} a_p^{(\ell,h)}(q_i)\cdot V_p^{(\ell,h)}, \quad a_p = \mathrm{softmax}(qK^\top)_p.$$
  2. $M^{(\ell,h)}$ 구성, SVD.
  3. $r^*(\tau)$ at $\tau \in \{0.90, 0.95, 0.99\}$.
- 보고: layer × head heatmap, mean ± std, 분포 histogram.

**컨트롤.**
- **Random query control**: $\mathcal Q$를 task 외 random query로 교체. 같은 prefix지만 task와 무관 → $r^*$ 변화로 task-conditioning 효과 측정.
- **Shuffled prefix control**: prefix 토큰 순서 shuffle (기존 Phase C 재사용 가능). $r^*$가 유지되어야 정상 (공식상 invariant).
- **Random prefix control**: prefix를 random 토큰으로 교체. $r^*$가 *작아져야* 정상 (정보 없는 prefix는 단순 직류 bias).

**해석 매트릭스.**
| 평균 $r^*(0.95)$ | 사전 예측 결과 | 다음 단계 |
|:---:|---|---|
| < 16 | 정적 $B_{\mathrm{ont}}$ 충분 | E3 정적 recovery 곧장 |
| 16–64 | bimodal head 의심 | E3 + E4 둘 다 |
| > 64 | query-cond 필수 | E4 + E5에 자원 집중 |

**리소스.** 모델 inference만, 학습 없음. A6000 1대 × 6h × 2 모델 × 4 task = 48 GPU·h.

**스크립트 (신규).** `scripts/rank_replaceability/measure_phi_rank.py`

---

### E2. Layer/Head Specialization

**의도.** $r^*$가 layer · head에 어떻게 분포하는지. 정적 intervention이 *어느 layer*에 들어가야 하는지 결정.

**가설.**
- F13b "K early-layers (L/4) imprint, Q all-layers cover" 처방이 우연이 아니라면, $r^*$ 분포가 *early layer에서 작고 late layer에서 큼* 패턴을 보일 것 (내가 새 정보를 *imprint*하는 곳은 압축 가능, *통합*하는 곳은 압축 불가).
- 반증: random 분포 → F13b 처방은 lucky guess.

**검증.** E1의 layer × head heatmap의 layer-wise marginal 분석. K-S test로 layer 간 차이 유의성.

**리소스.** E1 결과 재사용, GPU 추가 없음.

---

### E3. Static rank-$k$ Recovery (H1 직접 검증)

**의도.** 측정된 $r^*$ 에 맞춰 static intervention을 잘라 적용하고 task acc를 측정. Eckart–Young upper bound가 실제로 달성되는지.

**가설.**
- $k = r^*(0.95)$ 정도의 static intervention으로 기존 prompt 대비 task acc loss $\le 5$pp.
- $k = 4$ (= F13b 현 설정) 로는 $r^* > 16$ layer에서 정확도 손실 측정 가능.

**검증 방법.**
1. E1에서 얻은 SVD에서 top-$k$ components 추출 → static $V_{\mathrm{steer}}^{(\ell,h)}$ 구성.
2. **Prompt를 빼고** $V_{\mathrm{steer}}$만 inject (layer $\ell$, head $h$의 attention output에 더하기).
3. MetaTool ST4 / τ²-bench eval.
4. $k \in \{1, 2, 4, 8, 16, 32, 64\}$ sweep.
5. 비교: full prompt baseline / no-prompt baseline / 현 $B_{\mathrm{ont}}$ static / E3 ranked-$k$.

**Pass 기준.**
- $k = 16$ 에서 task acc $\ge$ full prompt − 5pp on MetaTool ST4.
- $k = r^*(0.95)$ 가 $k$-acc 곡선의 elbow와 일치 (이론-실측 정합).

**리소스.** A6000 × 12h × 2 모델 × 4 task × 7 $k$ ≒ 좀 큼. **A100 1개 sweep으로 12h 안에 가능.**

**스크립트.** `scripts/rank_replaceability/static_recovery_eval.py`

---

### E4. Query-Conditional Oracle (H2 검증, upper bound)

**의도.** Per-query oracle steering vector → query-cond intervention의 *upper bound*. 정적 vs query-cond gap의 실측치.

**가설.** $r^*$이 큰 task/layer에서 oracle - static gap $\ge 10$pp. 작은 곳에서는 $\le 2$pp.

**검증 방법.**
1. 각 query $q$에 대해 prefix-attn output $\Phi_P(q)$를 *그 query에 한해* 정답으로 사용. 즉 prompt를 빼고 그 자리에 $\Phi_P(q)$ 자체를 inject.
2. Task acc 측정. **이게 query-cond intervention의 oracle upper bound.**
3. E3의 best static $k$ 와 비교.

**Pass 기준.** Oracle - static gap > 10pp on at least one task → query-cond direction 정당화. 작으면 정적으로 충분.

**리소스.** Forward-pass overhead 2배. A100 × 12h.

---

### E5. Q-bias as 1st-Order Correction (H3 검증)

**의도.** 현재 보유한 **regime-dependent sign flip** 데이터를 H3 예측과 정합 검증. **paper의 mechanism 단락을 살릴 핵심 실험.**

**가설.** Q-bias가 $\Phi_P$의 1차 보정이면:
- $\partial_q \lambda_P(q)$ 가 task에서 양의 평균 → β+ 가 acc 개선
- 음의 평균 → β− 가 acc 개선
- τ²-bench retail (β−가 best, +5.11pp), telecom (β+가 best, +24.78pp) — 두 task에서 $\partial_q \lambda_P$의 평균 부호가 *예측대로 반대*여야 H3 통과.

**검증 방법.**
1. 각 task 256 query에 대해 $\lambda_P(q)$ 계산.
2. Query-direction에 대한 finite-difference $\partial_q \lambda_P$ 추정 (또는 jvp).
3. 평균 부호 측정.
4. 기존 β-sweep 결과(retail β−, telecom β+)와 부호 일치 검정.

**Pass 기준.** 4 task 중 ≥ 3 에서 부호 예측 일치 → mechanism 정리 paper에 박을 수 있음.

**리소스.** E1 forward-pass 결과 재사용 + jvp 추가. A6000 × 4h.

**스크립트.** `scripts/rank_replaceability/qbias_taylor_check.py`

---

### E6. Sanity Gate — Random Basis Ablation (필수)

**의도.** B_ont가 정말 load-bearing인지 cross-check. "Random direction으로도 sign-flip이 살아남으면 ontology framing은 전부 빼야 함" — 이전 의견에서 권고한 sanity gate.

**가설.** Random direction으로는 sign-flip 약화 또는 소실.

**검증.** Q-bias의 $B_{\mathrm{ont}}$ 자리에 random orthonormal basis 대입, retail/telecom β-sweep 재현.

**Pass 기준.** Random에서 sign-flip 소실 → ontology load-bearing 확인. 보이면 framing 전면 재검토.

**리소스.** A6000 × 4h.

**경고.** 이게 negative로 나오면 본 thesis도 영향 받음. 그래도 정직하게 해야 함.

---

## 4. 통합 결정 트리

```
E1 (rank measurement)
 ├─ r* < 16 (모든 layer/head)
 │   └─ E3 static recovery 직진
 │       ├─ 통과 → Paper Story A: "Static rank-k intervention suffices"
 │       └─ 실패 → 측정 오류 의심, E4로 디버깅
 │
 ├─ r* in [16, 64], bimodal
 │   └─ E3 + E4 양쪽
 │       └─ Paper Story B: "Hybrid — static for low-r* layers, query-cond for high-r* layers"
 │
 └─ r* > 64 (대부분 layer/head)
     └─ E4 + E5 집중
         └─ Paper Story C: "Static is fundamentally limited; Q-bias is 1st-order correction"

E5 (Q-bias as 1st-order)
 ├─ retail/telecom 부호 예측 일치 ≥ 3/4 → mechanism 단락 확보
 └─ 불일치 → mechanism은 phenomenological로 남음 (paper 약점)

E6 (random basis)
 ├─ sign-flip 소실 → ontology load-bearing 확인 (B_ont framing 살림)
 └─ sign-flip 유지 → ontology 빼고 "any low-rank direction works"로 framing 변경
```

---

## 5. GPU 할당 및 일정

### 5.1 자원 요약

| GPU | 실험 | 시간 | 비고 |
|:---:|---|:---:|---|
| A6000 #0 | E1 Qwen × 4 task | 24h | rank measurement |
| A6000 #1 | E1 Llama × 4 task | 24h | rank measurement |
| A100 #0 | E3 static-recovery sweep | 12h | 7 k × 4 task |
| A100 #0 | E4 oracle eval | 12h | 4 task |
| A6000 #2 | E5 Taylor check | 4h | jvp |
| A6000 #2 | E6 random basis | 4h | sanity |

**총 약 80 GPU·h.**

### 5.2 일정 (Day 0 = 2026-04-29)

- **Day 0–1**: E1 launch on Qwen + Llama. 동시에 E5/E6 스크립트 준비.
- **Day 2**: E1 결과 분석. $r^*$ 분포 확인 → 결정 트리 분기.
- **Day 3–4**: E3 / E4 (분기에 따라).
- **Day 5**: E5 + E6.
- **Day 6**: 결과 통합, paper draft v0 → v1 업데이트.

**중간 게이트 (Day 2 저녁)**: $r^*$ 분포가 strong 정적 / strong query-cond / 중간 중 무엇인지 1쪽 메모. 이거 없이 E3/E4 진행 안 함.

---

## 6. 리스크 & 감산 (preregistered)

| 리스크 | 확률 | 감산 |
|---|:---:|---|
| $\Phi_P$ 정확 추출이 RoPE/GQA로 어려움 | 중 | layer-wise GQA-aware 구현. Qwen2.5는 GQA, Llama-3.1도 GQA. KV-head 단위 SVD로 처리 |
| $r^*$가 layer 간 너무 다양해 단일 결정 불가 | 중 | per-layer story로 split. paper figure도 layer-resolved로 |
| E3 static recovery가 task acc 회복 못 함 | 중 | E1 결과로 사전 예측. acc 손실이 Eckart-Young 예측과 정합하면 (이론-실측 일치) 자체로 paper |
| E5 부호 일치 실패 → mechanism 약화 | 중 | 그래도 phenomenological 보고는 가능. mechanism은 future work로 |
| E6에서 random basis도 동작 → ontology framing 무너짐 | **낮음 (Phase B 선행 결과로 비등방 효과 있음)** | framing을 "structured low-rank direction" 으로 수정. Theorem은 그대로 |
| F13b seed bimodality가 E3에서 재현 | 높음 | $k$별로 seed sensitivity 측정, paper에 명시 |
| GPU 부족 (A100 코워커 가용성) | 중 | E3 우선, E4 oracle은 줄여서 single task만 |

**Preregistration 원칙**: E1 결과 보기 전에 모든 결정 트리 가지에서의 paper story를 적어둔다. H-Energy-Wells 폐기 패턴(데이터 본 후 framing 변경) 재발 방지.

---

## 7. 산출물 (Deliverables)

| 산출물 | 위치 | 마감 |
|---|---|---|
| E1 rank heatmap | `reports/rank_replaceability_2026_04/r_star_heatmap.{pdf,json}` | Day 2 |
| 중간 게이트 메모 | `reports/rank_replaceability_2026_04/gate_memo_day2.md` | Day 2 |
| E3 acc curve | `reports/rank_replaceability_2026_04/static_recovery_curve.{pdf,json}` | Day 4 |
| E4 oracle gap | `reports/rank_replaceability_2026_04/oracle_gap.{pdf,json}` | Day 4 |
| E5 Taylor sign 표 | `reports/rank_replaceability_2026_04/qbias_taylor_signs.json` | Day 5 |
| E6 random ablation | `reports/rank_replaceability_2026_04/random_basis_ablation.json` | Day 5 |
| Paper v1 (이론 + Phase A 실험) | `math/paper/iclr2027_prompt_internalization/PAPER_DRAFT_v1.md` | Day 6 |

---

## 8. Out-of-Scope (명시)

- **He 2022 → Graph RAG 동치 주장**: 본 plan에서 다루지 않음. 카톡 framing 중 살리지 않은 한 줄.
- **카탈로그 semantic encoding**: Phase C (untouched paper) 결과로 이미 falsified. 본 plan은 K-subspace pipeline 가정만 사용.
- **학습 기반 query-conditional intervention** (HyperLoRA 등): 본 plan은 frozen LLM 한정. 학습 도입은 후속 paper.
- **PAPER_DRAFT_ICLR_v1.md (two-level argmax-subspace selectivity)**: 완전 untouched. 무관.

---

## 9. 메모 — 카톡 framing과의 정합성

| 카톡 (4/19~25) | 본 plan에서의 처리 |
|---|---|
| "NeurIPS 폐기 → ICLR 새 thesis" | ○ — 본 plan이 새 thesis. F13b 자산 그대로 사용 |
| "Goal/Act 내재화로 LLM local minimum 해결" | △ — "prompt 내재화"로 좁혀서 측정 가능하게 |
| "포커스 랭크 이동" | ○ — 정확히 rank-$k$ intervention의 He-확장 |
| "프롬프트 없이 온톨로지만으로" | △ — $r^*$ 측정으로 결판. 작으면 ○, 크면 △ |
| "QK rank ≡ Graph RAG (He 확장)" | ✗ — 다루지 않음 |
| "온톨로지가 distillation 역할" | △ — Phase C 결과(catalog 내용 not load-bearing)와 충돌, B_ont 는 *K-subspace pipeline*으로 한정 |
| "이미 He에서 증명" | ✗ — He는 form만. Content/rank bound는 본 paper에서 *우리가* 증명 |

**한 줄**: 카톡 직관 중 "포커스 랭크 이동으로 prompt 대체"만 살리고, 나머지는 측정으로 결판하거나 범위 밖으로 둔다.
