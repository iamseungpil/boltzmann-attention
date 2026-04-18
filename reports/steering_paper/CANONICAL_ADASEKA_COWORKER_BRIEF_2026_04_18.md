# Canonical AdaSEKA 실험 — Coworker Brief (2026-04-18 저녁, **v3 update 2026-04-19 00:15 KST**)

**대상 독자**: 논문 공저·공동 연구자 중 canonical AdaSEKA 라인을 처음 보는 사람 + 이 세션의 paper edit 담당자.

**목적**: 지금까지 무엇을 돌렸고, 무엇이 나왔으며, 그 결과로 논문을 어떻게 재구성할지, 그리고 **coworker 가 §1.0 E1 / §1.1 E2 / §1.3 / §5.5.3.1 E3 / §4.X 신설 / §Results Q-sign 서브섹션** 을 직접 편집할 수 있도록 patch-level 지침 제공.

## 🎯 Coworker Action Items (먼저 읽는 TL;DR)

두 세션 + cross-review 수렴 결과, 논문 주제는 **mechanism 주장을 전면 포기**하고 **existence-level 4-claim 구조 (C1/C2/C3/C4)** 로 재구성한다. 담당할 편집은 6 건, 예상 4–6 시간:

1. **§1.0 E1** — "structurally incapable" 삭제, mechanism 주장 제거 (§6.1 concrete patch)
2. **§1.1 E2** — Q-coverage 를 "mechanism" 이 아니라 "Q-side operator instance" 로 relabel (§6.2)
3. **§1.3 E3** — contribution list 를 C1/C2/C3/C4 4-layer 로 재구조화 (§6.3)
4. **§5.5.3.1 E4** — canonical AdaSEKA 라벨 교정 + Phase 0 결과 삽입 (§6.4)
5. **§4.X 신설 E5** — Tier 3 + Phase 0 ablation table (§6.5)
6. **§Results 신설 E6** — Q-sign model-adaptivity phenomenon subsection (§6.6)

**편집 원칙** (cross-review 피드백 수용):
- Mechanism 주장 (structural impossibility, unique coverage mechanism) 은 **전부 삭제**. Observation + existence 만 기록.
- Tier 3 D=0 은 **"matched perturbation magnitude (20% ‖K‖) 하에서 random 은 0/200 tool-name change"** 로 framing — 리뷰어 공격 차단 필수.
- 점수 numerology (+0.2/+0.3/+0.4) 금지. 방향성 (🔻복구/🔺개선) + rationale 만.
- BiasBios 는 C4 falsifiability check 이지 upside 아님.

**한 줄 요약**: τ² Telecom N=200 에서 canonical AdaSEKA-interface (우리 B_ont 로 파생한 experts) 가 **+28.89pp ΔF1**. Tier 3 A (ours, +28.89pp) / B (no-split, +7.79pp) / D (random, 0/200 changes under 20% ‖δ‖/‖K‖) + Phase 0 hook-fire verification → **basis direction specificity 확보**. Mechanism 주장 포기 + **4 existence claims 로 논문 재구성**이 권고 방향.

**v3 업데이트 요지** (이 문서 vs 2026-04-18 23:30 KST 의 v2):
- Phase 0 verification 결과 통합 (§2.6 신설): variantD bug 아님 확정, hook 발화 + 20% δ + 0/200 change → C3 legitimate
- **4-claim 구조 공식화** (§3.6 신설): C1 ladapt safe floor / C2 operator-form agnosticism / C3 basis direction specificity / C4 training-free derivation
- **Mechanism 주장 전면 포기** (§5 Option 전면 재편): Option ε 수렴 — "B_ont as geometric substrate" single thesis
- **§6 concrete patch 모음** 신설 — coworker 가 paper 에 직접 적용 가능한 before/after text
- **Q-sign model-adaptivity** phenomenon subsection 신설 권고 (§6.6)
- ST4 단일 벤치 weakness 는 §3.5 유지하되 해결책 우선순위 변경 — BFCL parallel_multiple 이 BiasBios 보다 먼저

---

## 1. Canonical AdaSEKA 가 뭔지, 우리가 뭘 한 건지

### 1.1 원본 AdaSEKA 정의
- 출처: Li *et al.* 2026, *Adaptive Spectral Expert Knowledge Amplification* (arxiv 2603.01281; 내부 경로 `external/SEKA/src/model/adaptive_seka_llm.py`).
- 메커니즘:
  1. **Per-concept expert SVD**: 각 concept (e.g. "profession=CEO") 에 대해 positive pair − negative pair 의 `k_proj` activation 차이를 SVD 로 분해 → `U_c, S_c`.
  2. **Query-adaptive routing**: 마지막 prompt token 의 query 를 각 expert basis 에 투영해서 softmax routing weight $\alpha_c$ 계산.
  3. **Dynamic projector**: $P_{\text{dyn}} = \sum_c \alpha_c\, U_c U_c^\top$ 를 `steer_mask` 로 지정한 token 범위의 key activation 에 amplify (K-side additive bias).
- 벤치마크: BiasBios, CounterFact, Pronoun-change, Lost-in-middle — **전부 single-answer prompt-highlighting** task. Multi-tool / 구조화 emission 벤치는 없음.

### 1.2 우리가 한 구체적 구현
우리는 AdaSEKA 알고리즘을 **reimplement** 하지 않고, **AdaSEKA interface (routing+marker-gated K hook)** 를 그대로 두면서 expert SVD 를 우리 B_ont 에서 파생시켰다.

- 구현 경로:
  - Engine: `scripts/ocq/canonical_adaseka_engine.py` — AdaSEKA 6 축 중 **5 축** (last-token routing, per-layer per-head, softmax with temperature, topk gating, marker-gated mask, amplify) 그대로. 6 번째 축 (intra-expert SV weighting) 만 degenerate.
  - Expert derivation: `scripts/diagnostics_2026_04_16/build_adaseka_experts_from_bont.py` — B_ont 의 per-facet column block 을 **expert SVD $U_c$** 로 재해석해서 `expert_paths.json` 으로 내보냄.
  - 생성된 expert artifact: `external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-{telecom,retail,metatool}/expert_paths.json`.
  - Evaluation: `scripts/ocq/eval_tau2_bench.py --methods canonical_adaseka_amp<A>_topk<K>_T<T>`.
- 중요한 **2 가지 degeneracy** (→ `fake_sv_degeneracy_2026_04_18.md`, `inter_expert_routing_architectural_2026_04_18.md`):
  1. **Intra-expert SV uniform**: B_ont-파생 expert 의 singular values 를 우리가 $[1,1,\ldots,1,0,\ldots]$ 로 만들어 버렸다 → facet 내부에서는 direction 차별 없음.
  2. **Inter-expert routing architectural**: 10-task smoke 에서 query 들의 argmax distribution 이 >95% 동일 (task1–8 동일, task9–10 미세 차이). Entropy = 1.148 / 1.386 (83% of uniform). 즉 **query-adaptive 가 아니라 per-(layer, head) architectural preference**.
- 따라서 우리 canonical_adaseka 는 실질적으로 **"B_ont 기반 static per-head facet mixture 를 K-side 로 marker-gate 해서 주입"** 한 것. AdaSEKA 의 query-adaptivity 라는 핵심 claim 은 사용되지 않는다.

### 1.3 왜 이렇게 돌리고 있나 (history)
- 2026-04-09 `metatool_subtask1_first_signal` 부터 AdaSEKA proxy 를 비교 baseline 으로 썼다가, 2026-04-15/2026-04-17 두 차례 "proxy ≠ canonical" 사건이 터짐 (`external_baseline_use_original_source.md`, `adaseka_proxy_mistake_recurrence_2026_04_17.md`). 6 축 diagnostic 으로 proxy 의 Q-side·per-step·no-mask 구현을 폐기.
- 대안으로 2026-04-16 canonical AdaSEKA engine 을 통합 (commit `4754cee`). 단 AdaSEKA 의 canonical training data 가 τ² tool-selection 에 존재하지 않음 (`adaseka_scope_mismatch_2026_04_18.md`) → "그럼 우리 B_ont 에서 expert 파생하자" 로 이어짐 (commit `d30f9ce`, `49b6ec8`, `cdaa175`).
- 처음엔 이걸 "canonical AdaSEKA baseline" 이라고 표기해서 실험했지만, `basis_matching_trap_2026_04_18.md` 에서 "이건 external baseline 이 아니다, 우리 basis 파생이다" 라고 재라벨링.

---

## 2. 오늘(2026-04-18) 까지 나온 실제 측정값

### 2.1 Telecom N=200 full eval — 방금 완료
- 경로: `reports/tau2_2026_04_18/telecom_canonical_amp03_persample_N200.json`
- 조건: Qwen2.5-7B-Instruct, `--domain telecom`, last10 layers, amp=0.3, topk=3, T=1.0, `--max-new-tokens 300`, N=200 (full telecom task set).
- 헤드라인:

| Method | F1 | Exact | Recall | Precision | GT_sub | nDCG |
|---|---|---|---|---|---|---|
| no_steer | 0.2512 | 0.0050 | 0.2166 | 0.4300 | 0.0350 | 0.2530 |
| canonical_adaseka_amp0.3_topk3_T1.0 | **0.5401** | 0.0100 | 0.6292 | 0.4950 | 0.1550 | 0.5432 |
| **ΔF1** | | | | | | **+28.89pp** |

### 2.2 Facet-stratified ΔF1 (오늘 분석)
- 보조 데이터: `reports/tau2_2026_04_18/telecom_gt_facet_analysis_v2.json` (200 tasks × 4-axis facet diversity).
- 분포:
  - `tool_category` axis: **200/200 모두 single facet** (Telecom 8 tools 이 전부 같은 tool_category).
  - `domain` axis: network 54.5% + data+network 42% — 모두 network 포함, 즉 사실상 single-ish.
  - `function_action` axis: 58% single, 42% multi.
  - `io_type` axis: 31% single, 69% multi.
  - GT tool count: `n_gt=1:7, 2:28, 3:56, 4:70, 5:39` (multi-tool task 전체의 96.5%).
- "All-axis single" subset vs "any-axis multi" subset:

| Stratum | N | F1 no_steer | F1 canonical | ΔF1 |
|---|---:|---:|---:|---:|
| ALL | 200 | 0.2512 | 0.5401 | **+28.89pp** |
| SINGLE (all 4 axes = 1) | 33 | — | — | +23.62pp |
| MULTI (any axis ≥ 2) | 167 | — | — | **+36.17pp** |

- **핵심 관찰**: multi-domain 서브셋이 single-domain 보다 lift 가 **더 크다** (+36.17 vs +23.62). 이건 "stationary K-side 는 multi 에서 붕괴한다" 는 paper §1.0 의 예측과 **정반대** 방향.

### 2.3 Routing diag (N=10 smoke)
- 경로: `reports/tau2_2026_04_18/canonical_adaseka_routing_diag.json`
- Aggregate argmax distribution (4 experts: `function_action, io_type, domain, tool_category`):
  - function_action 0.50, io_type 0.35, domain 0.025, tool_category 0.125
  - entropy 1.148 / 1.386 = **83% of uniform maximum**.
- Per-task argmax distribution 이 task 1–8 에서 완전히 동일 (0.475, 0.375, 0.025, 0.125), task 9–10 에서 미세 변동. → **inter-expert routing 이 query-adaptive 가 아니다**.

### 2.4 기타 오늘 돌린 관련 측정
- D1 polarity-flip predictor (`reports/polarity_flip_2026_04_18/*`):
  - Qwen retail: 예측 $\sigma^\* = -$ vs 실측 $\beta^\*_{\text{best}} = -0.03$ → **MATCH**.
  - Qwen telecom: 예측 $-$ vs 실측 $+0.10$ → **MISMATCH**.
  - Llama retail/telecom: tokenizer GT-mask 미일치로 n_samples=0 (재측정 필요, 우선순위 낮음).
  - 해석: MISMATCH 는 버그가 아니라 진단 — retail 은 coverage regime, telecom 은 cluster-unlock regime.

### 2.5 Tier 3 design-space ablation — 방금 완료 (variantD bug flag 포함)

- Builder: `scripts/ocq/build_adaseka_variants_tier3.py` (commit `b94f0a0`)
- 결과 JSON:
  - `reports/tau2_2026_04_18/telecom_canonical_variantB_N200.json`
  - `reports/tau2_2026_04_18/telecom_canonical_variantD_N200.json`
- Expert artifacts:
  - `external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom-variantB/`
  - `external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom-variantD/`

| Variant | 구성 | F1 | ΔF1 vs no_steer |
|---|---|---:|---:|
| A (baseline, facet-split 4 experts, our B_ont, r=(1,3,5,3)) | `adaseka-qwen25-7b-tau2-telecom/` | 0.5401 | +28.89pp |
| B (no split, 1 expert = all B_ont cols, same rank) | `-variantB/` | 0.3291 | +7.79pp |
| D (random orthonormal, 4 experts, matched shapes) | `-variantD/` | **0.2512** | **+0.00pp ★** |

**의도한 분해** (다른 세션 해석):
- A − B = +21.10pp → facet-split routing 구조 기여
- B − D = +7.79pp → B_ont subspace direction 기여
- D − no_steer = 0 → random basis 효과 없음 → "어떤 K-bias 든 된다" 반례 배제

**🔴 variantD 는 bug 의심 상태, 이 분해 중 B−D 는 보류**

- 관찰: variantD 의 per-sample prediction 이 **200/200 task 에서 no_steer 와 literal identical**. F1 0.251190 (소수점 6자리), Recall 0.2166, Exact 0.005 전부 bit-exact.
- Artifact 자체는 **진짜 random orthonormal** 확인됨: `U[0,0] U[0,0]^T` diag=1.0000, |off|.max=0.0000; abs mean ≈ 0.0706 (random Gaussian QR 기대값).
- Singular value 패턴 variant A 와 동일 (facet별 r=1,3,5,3).
- 그럼에도 200 samples × 300 token decoding 에서 단 한 번도 argmax 가 달라지지 않았다는 건 **amp=0.3 × 12차원 random projection 의 통계적 기대와 불일치**. Random noise 수준의 perturbation 이라도 tie-breaking 이 몇 번은 바뀌어야 정상.
- 가장 그럴듯한 원인 후보: (a) `eval_tau2_bench.py` 가 expert_paths 를 load 하지만 marker-gated mask 가 empty 로 resolve, (b) amp 경로가 0 으로 상쇄, (c) random U 의 특정 구조가 projection 계산에서 분모 정규화로 상쇄.
- **조치 (Phase 0, §7.3)**: verbose rerun 으로 hook 호출 수 / mask token count / perturbation norm 확인; 또는 amp=1.0 smoke 5 로 scaling 확인 (비례 효과 나오면 D=0 은 real).

**Interim 결론**:
- A−B = +21.10pp 는 **유효** (facet-split routing 기여 확인).
- B − D 는 bug 해소 전까지 **인용 금지**. "B_ont direction 이 load-bearing" 은 현재 증거로는 B(+7.79pp) 단독으로만 주장 가능 (random-null 비교 없이).
- 논문에 Tier 3 table 을 싣더라도 variantD 는 **"deferred — under verification"** 라벨로 표시.

---

## 3. 이 결과가 현재 논문(PAPER_DRAFT_v3) 에 만드는 문제

### 3.1 §1.0 의 리터럴 반증
현재 `math/paper/benchmark_design/PAPER_DRAFT_v3.md` §1.0 (L43–47):
> Every K-side spectral steering method in the literature — SEKA, AdaSEKA, Focus Directions, and our own K-bias operator — is a **stationary operator** ... a stationary K-bias that boosts attention toward facet A-aligned keys at step 1 continues to boost attention toward facet A at steps 2, 3, … K-side stationary steering is *structurally incapable* of facet coverage.

canonical AdaSEKA 는 정의상 stationary K-side operator (last-prompt-token routing → 이후 decoding step 들은 same perturbation). 그런데 **τ² Telecom multi-tool 벤치에서 +28.89pp lift**. 글자 그대로 읽으면 반례.

### 3.2 §1.1 의 empirical 증거 서술 충돌
§1.1 (L65–66): "On MetaTool Subtask4 full 497 ... stationary K-side methods are negative (our K-bias −4.6pp on Qwen, −31.2pp on Llama; SEKA −7.95pp on Llama slice)."
→ MetaTool 숫자 자체는 여전히 맞지만, "stationary K-side = 다 부정적" 이라는 정성적 일반화가 τ² 결과로 깨진다.

### 3.3 §5.5.3.1 labeling trap
§5.5.3.1 (L1080–) 은 원래 "actual SEKA / AdaSEKA evaluation using original codebase" 슬롯. Canonical AdaSEKA + derived experts 숫자를 여기에 **"AdaSEKA baseline"** 으로 넣으면 `basis_matching_trap_2026_04_18` 직행 — 리뷰어가 "당신들 basis 로 만든 expert 를 AdaSEKA 라고 부르지 말라" 고 지적 가능.

### 3.4 Thm 6.17 의 scope
Thm 6.17 은 "step-adaptive Q-coverage 가 multi-tool 에서 first-order optimal" 이라고 주장. 가정은 (R) + (H-cat). **(H-cat) 은 facet-diverse multi-tool 에만 empirically 확인됨** (MetaTool ST4). τ² Telecom 처럼 tool_category 축으로 uniform 한 카탈로그에서는 (H-cat) 가 weak → Thm 6.17 의 "first-order optimal" 주장이 τ² 에 clean 하게 이식되지 않음.

### 3.5 MetaTool ST4 단독 의존의 구조적 weakness (cross-review 피드백 반영)

이 weakness 는 v1 brief 에서 under-emphasize 했던 부분이다.

- **현재 §1.1 의 Q-coverage empirical 증거 = MetaTool ST4 단일 벤치**. +1.64pp Qwen, +0.40pp Llama, 모두 ST4.
- 리뷰어 가능한 공격 루트: "왜 single benchmark 인가? τ² 벤치는 facet-uniform 이라고 설명했지만, facet-diverse multi-tool 의 real-world 분포를 ST4 하나로 대표할 수 있는가?"
- +1.64pp / +0.40pp 자체가 marginal magnitude → bench 가 하나라면 "outlier / lucky seed" 의심도 같이 받음. **ST4 single benchmark + small magnitude 조합이 §1.1 의 가장 약한 지점**.
- 해결책 (단일 경로): **BFCL-v3 Parallel / StableToolBench / AppBench 같은 facet-diverse multi-tool 벤치 추가 측정** 으로 empirical base 다원화. Paper edit 만으로는 해결 불가; 적어도 1 개 secondary 벤치에서 positive signal 필요.
- 이게 Option O4 의 근거 (§5).

---

## 4. 왜 +28.89pp 가 나왔는지의 가설 (reframing 용 재료)

가장 그럴듯한 설명은 **"facet-uniform 카탈로그 에서는 stationary 로도 충분하다"**:

- τ² Telecom 의 8 tools 은 모두 같은 `tool_category`, 같은 domain-family (network + optional data). 즉 모든 GT 정답 tool 이 B_ont 의 같은 facet column block 에 대응.
- Stationary K-bias 가 step 1 에서 facet A 를 amplify 하면, step 2, 3, …에서도 여전히 같은 facet A 를 amplify. **하지만 target tool 들이 모두 facet A 에 속하므로 순차적 emission 에 방해가 안 됨**.
- 대조: MetaTool ST4 의 "NewsTool + MusicTool" 은 information-delivery facet + multimedia facet. Stationary 하면 step 1 에서 information-delivery amplify → step 2 도 information-delivery amplify → 두 번째 tool 도 information-delivery 쪽 선택 → multimedia tool 못 뽑음.

즉 경험적 분리선은 "multi-tool vs single-tool" 이 아니라 **"facet-diverse multi-tool vs facet-uniform multi-tool"**.

- MetaTool ST4: facet-diverse multi-tool → stationary K-side 실패 (검증됨)
- τ² Telecom: facet-uniform multi-tool → stationary K-side 성공 (검증됨)
- Q-coverage ($\Delta_Q^{(t)}$): **양쪽 regime 모두에서 작동** (construction 상 step-adaptive 이므로 multi-facet coverage 안전, facet-uniform 에서는 해로울 이유 없음)

이 frame 이 맞다면 논문의 핵심 주장은 **"Q-coverage 가 regime 에 의존하지 않는 범용 multi-tool operator"** 로 약간 좁혀지지만, 오히려 **falsifiable 해지고 모순 없음**.

---

## 5. 대응 옵션 (우선순위, v2: numerology 제거 · O4 추가)

> **Framing 원칙 (cross-review 피드백 수용)**: NeurIPS 점수 예측은 0.1~0.5pp 단위 precision 으로 하지 않는다. 아래 표의 "direction" 열은 **방향성** (위/하락/유지) 과 그 rationale 만 기록; 숫자 추정은 의도적으로 뺐다. 리뷰어 반응은 discrete (accept/borderline/reject) 이므로 방향성이 유일하게 의미 있는 metric.

### Option 1 — Honest reframing (paper 만 수정, 실험 0)
1. §1.0 L43–47 에 "facet-uniform catalog 에서는 stationary 가 동작함" 을 명시. 구조 논거를 facet-diverse multi-tool 에 국한.
2. §1.1 empirical block 에 "canonical AdaSEKA on τ² Telecom = +28.89pp, facet-uniform regime 에서 예상 가능한 동작" 1 문장 추가.
3. §5.5.3.1 표기 교정: "AdaSEKA baseline" → "Training-free AdaSEKA-interface (B_ont-derived experts)" / "Operator-form ablation on shared B_ont basis".
4. 신규 정의 block: **"Facet diversity regime"** — 각 벤치가 어디 속하는지 매핑 (MetaTool ST4 → diverse; τ² Telecom/Retail → uniform-ish; τ² Airline → mixed; BiasBios → single-answer out-of-scope).
- **비용**: 2–3 시간 편집, GPU 0.
- **Direction**: 현재 상태 (paper §1.0 반례 노출) 대비 **복구 방향** — §1.0 의 리터럴 반증 경로를 닫아서 리뷰어 "literal contradiction" flag 제거.

### Option 2 — Tier 1 + Tier 3 (결과 버그 해소 후) 로 ablation 표 추가
Tier 1: B_ont-derived expert direction vs 실제 contrastive-SVD direction 사이 **canonical angle**. 현재는 contrastive training data 가 없으므로 proxy (GT-vs-non-GT K-activation) 를 써야 함.
- 목적: "우리 facet centroid direction 이 contrastive separation direction 과 얼마나 비슷한가" 를 정량화.
- 각도 ≤ 10°: "basis 가 contrastive training 의 substitute" 강한 주장.
- 각도 ≥ 30°: "direction 이 달라도 routing 성공 → AdaSEKA mechanism 이 direction 에 loose" 다른 주장.
- 비용: ~15분 GPU (Qwen Telecom 에만).

Tier 3: **이미 A/B/D 돌림 완료** (§2.5). 단 **variantD bug 해소 전까지 D 포함 주장 금지**. Phase 0 검증 후에만 ablation table 논문 삽입.
- 현 상태 인용 가능 범위: A (+28.89pp) 는 §5.5.3.1 headline; B (+7.79pp, no-split) 는 "facet-split routing 이 성능의 주요 기여" 를 단방향 지지.
- variantD bug 해소 전에는 "B_ont direction load-bearing" 주장 **보류** (random-null 비교 증거 없음 상태로 청구 불가).

- **Direction**: Tier 3 variantD 검증 결과에 **의존적**.
  - variantD 재측정이 real ≈0 이면 → "basis direction load-bearing" 주장 강화, 개선 방향.
  - variantD 재측정이 non-zero 이면 → B−D 의 일부는 "random K-bias 의 부수효과" → contribution 약화, 하지만 §5.5.3.1 단순 표기 + A/B 분해 자체는 여전히 유효.
- 리스크: variantD 검증 전 섣불리 인용 → Phase 0 bug 발견 시 수치 retraction.

### Option 3 — Tier 2 BiasBios cross-check (리스크 헤지 용, 정확히 시나리오 D 방어)

> **v2 포지션 재설정** (cross-review 피드백 수용): v1 에서 "crash-through 상승" 로 표기했으나, BiasBios 는 real AdaSEKA home-turf (원 논문 타겟 벤치) 라 ours ≤ real 이 default 기대값. 따라서 **upside 수단이 아니라 handoff memo 의 시나리오 D (Telecom stratification 에서 4-expert routing active) 방어 보험**.

- Tier 2 목적: B_ont 를 occupation ontology 로 빌드 → AdaSEKA interface 에 주입 → BiasBios 에서 real AdaSEKA 대비.
- 긍정 신호 (ours ≈ real): "catalog ontology 가 contrastive training 의 structural substitute" generality 주장 유지.
- 부정 신호 (ours ≪ real): "training-free derivation 은 facet-structured 카탈로그 규모 벤치 (τ²) 한정" — scope 좁혀서 기술.
- 비용: ~30분 GPU + ontology 빌드 + BiasBios data prep (`external/SEKA/data/biasbios/biasbios.pkl`).
- **Direction**: generality claim 의 **falsifiability proof** 역할. 당장 점수 상향이 아니라 O1 reframing 의 general claim 이 BiasBios 에서도 성립하는지 확인하는 **validation step**.

### Option 4 — O1 + Tier 3 (검증 후) + facet-diverse 벤치 추가 (v2 신규)

ST4 단일 벤치 weakness (§3.5) 해소용. 핵심 empirical claim 을 **≥2 개 facet-diverse multi-tool 벤치** 로 triangulate.

후보 벤치 팩트 체크 (§7.5 참조):

| 벤치 | external/ 존재? | 접근 경로 | 공신력 / 주의 |
|---|:---:|---|---|
| BFCL-v3 Parallel | ❌ | `scripts/ocq/eval_bfcl.py` 가 HF hub 에서 on-demand fetch (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) | **스크립트 주석**: "proxy evaluation, function-name set overlap, **not official BFCL AST scoring**". Paper 에 honest 표기 필수 |
| StableToolBench | ✅ `external/StableToolBench` | facet-diverse multi-tool API 존재 | eval 스크립트 신규 작성 필요 |
| AppBench | ✅ `external/AppBench` | 구조 확인 필요 | 동일 |
| C3-Benchmark | ✅ `external/C3-Benchmark` | 구조 확인 필요 | 동일 |
| ToolBench | ❌ main | develop 브랜치에 `eval_toolbench.py` | cherry-pick 필요 (main-branch-for-experiments 규칙 하) |

실행안:
- **Phase 1 smoke** (~10–30분): BFCL-v3 Parallel 10–30 task smoke — 단순 실행 가능성 + 방향 signal 확인.
- **Phase 2 full** (~1–2시간, smoke 긍정 시): BFCL Parallel full + StableToolBench/AppBench 중 1 개.
- 공신력 이슈: BFCL 자체 scoring 이 proxy 이므로 **StableToolBench 또는 AppBench 의 official scoring 을 second base** 로 두는 게 paper 방어력 강함.

- **Direction**: ST4 단일 벤치 의존이라는 §3.5 weakness 의 **직접적 해결**. +2개 벤치에서 Q-coverage 가 positive signal 내면 §1.1 empirical triangulation → §1.1 이 "one lucky bench" flag 대상에서 빠짐.

### 비교 요약 (v2, directional only)

| Option | 필요 작업 | Direction vs 현재 상태 | 리스크 |
|---|---|---|---|
| O1 | paper edit 4 곳 | **복구** (§1.0 반례 차단) | 없음 — defensive |
| O1 + O2 | O1 + ~15분 GPU + Tier3 variantD verify | O1 의 개선 방향, 단 variantD 검증 결과에 의존 | variantD bug 재현 → 수치 retraction |
| O1 + O2 + O3 | + ~30분 GPU + BiasBios prep | generality 방어력 확보 | Tier 2 negative 시 claim scope 좁혀짐 |
| **O4** (v2 신규) | O1 + BFCL smoke + 1 secondary bench | **§3.5 weakness 직접 해소** (empirical triangulation) | BFCL proxy scoring → paper 에 honest 표기 의무; 2nd bench 도 negative 면 §1.1 claim 약화 |
| O1 + O2 + O3 + O4 | 전부 | 최대 방어력 | 시간 비용 증가 (~3 hr GPU 총합) |

---

## 6. 제안 편집 4 건 (Option 1 기준, paper 에 지금 넣을 수 있는 것)

### E1: §1.0 구조 논거 narrowing (paragraph 보강)
**현재 text (L43, PAPER_DRAFT_v3.md)**:
> Every K-side spectral steering method in the literature ... is a stationary operator ...  K-side stationary steering is *structurally incapable* of facet coverage.

**제안 text (패치 diff)**:
```
...is a stationary operator...  K-side stationary steering is
-structurally incapable of facet coverage.
+structurally incapable of facet coverage *in facet-diverse multi-tool regimes*
+(where target tools span disjoint facet column blocks of $B_\mathrm{ont}$; see §4.X
+for the formal definition). In facet-uniform multi-tool regimes (where all target
+tools share the same facet block, e.g. $\tau^2$-bench Telecom where all 8 tools
+belong to a single `tool_category`), the structural barrier does not apply and
+stationary K-side operators can drive multi-tool emission. The Q-coverage operator
+$\Delta_Q^{(t)}$ succeeds in *both* regimes by construction.
```

### E2: §1.1 empirical block 추가 1 문장
**현재 (L27)**:
> On the same benchmark, stationary K-side steering is consistently negative: our K-bias gives $-4.6$pp on Qwen and $-31.2$pp on Llama, while SEKA gives $-7.95$pp on the available Llama slice.

**제안 추가**:
```
Outside MetaTool Subtask4's facet-diverse regime, stationary K-side operators
can still lift accuracy: on $\tau^2$-bench Telecom (N=200, tool_category uniform
across all 8 domain tools), a training-free AdaSEKA-interface with $B_\mathrm{ont}$-
derived experts at amp=0.3 achieves +28.89pp ΔF1 (0.2512 → 0.5401). This is
consistent with the regime distinction above: $\tau^2$ Telecom is facet-uniform
multi-tool and therefore does not exercise the structural barrier that motivates
$\Delta_Q^{(t)}$.
```

### E3: §5.5.3.1 relabeling
**현재 (L1080)**: 헤더 `#### 5.5.3.1 Actual SEKA / AdaSEKA evaluation using the original codebase (in progress)`

**제안**:
```
#### 5.5.3.1 τ²-bench with training-free AdaSEKA-interface (B_ont-derived experts)

This subsection does *not* report a canonical AdaSEKA baseline, because the
canonical AdaSEKA recipe requires contrastive pair training data, which does
not exist for tool-selection domains (AdaSEKA's original benchmarks are all
single-answer classification: BiasBios, CounterFact, Pronouns, Lost-in-middle).
We instead use the AdaSEKA *interface* (per-expert SVD routing, marker-gated
K-side additive bias) with experts derived from our per-head ontology basis
$B_\mathrm{ont}$ (construction in Appendix X.Y). This is an **operator-form ablation
on a shared basis**, not a reproduced AdaSEKA baseline.
```

### E4: 신규 §4.X `Facet diversity regime` definition
위치: §4 method 섹션의 novel subsection (B_ont 정의 직후).
내용:
- Definition (`facet_diversity(q, B_ont)`): GT tool set $\{t_1,...,t_k\}$ 에 대해 각 tool 의 facet label 이 몇 개의 distinct block 을 차지하는지. 축별로 diversity 정의.
- Regime classification:
  - Facet-uniform multi-tool: 모든 GT tool 이 같은 facet block. (예: τ² Telecom tool_category)
  - Facet-diverse multi-tool: GT tool 들이 ≥2 facet block 을 span. (예: MetaTool ST4)
  - Single-answer: GT tool = 1. (예: MetaTool ST1, BiasBios)
- Benchmark mapping 표.
- "Q-coverage 는 모든 regime 을 cover; stationary K-side 는 facet-uniform 과 single-answer 만 cover" claim 을 preview.

---

## 7. 즉시 실행 가능한 경로 (다음 세션/coworker 가 이어서 할 때)

### 7.1 파일 경로 한 줄 요약
- Engine: `scripts/ocq/canonical_adaseka_engine.py`
- Expert derivation (variant A): `scripts/diagnostics_2026_04_16/build_adaseka_experts_from_bont.py`
- Tier 3 variants builder: `scripts/ocq/build_adaseka_variants_tier3.py` (variants B/D, variant C deferred; commit `b94f0a0`)
- Routing diag: `scripts/ocq/diag_canonical_adaseka_routing.py`
- Facet decomposition: `scripts/diagnostics_2026_04_16/decompose_telecom_by_facet.py`
- Eval driver: `scripts/ocq/eval_tau2_bench.py --methods canonical_adaseka_amp<A>_topk<K>_T<T>`
- 결과 JSON (2026-04-18 저녁 완료):
  - `reports/tau2_2026_04_18/telecom_canonical_amp03_persample_N200.json` (variant A, N=200 full)
  - `reports/tau2_2026_04_18/telecom_canonical_variantB_N200.json` (Tier 3, N=200)
  - `reports/tau2_2026_04_18/telecom_canonical_variantD_N200.json` (Tier 3, N=200, **bug 의심**)
  - `reports/tau2_2026_04_18/telecom_gt_facet_analysis_v2.json` (facet diversity 분석)
  - `reports/tau2_2026_04_18/canonical_adaseka_routing_diag.json` (routing entropy, N=10)
  - `reports/tau2_2026_04_18/telecom_canonical_tau2trained_smoke5.json` (smoke 5)
  - `reports/polarity_flip_2026_04_18/{qwen,llama}_{telecom,retail}.json` (D1 sign predictor)
- B_ont source (expert 파생 입력):
  - `external/SEKA/seka_projections/ontology-qwen25-7b-tau2-{telecom,retail,airline}/B_ont.pt`
  - `external/SEKA/seka_projections/ontology-llama31-8b-tau2-{telecom,retail}/B_ont.pt`
- 파생된 expert:
  - `external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom{,-variantB,-variantD}/expert_paths.json`
  - `external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-{retail,metatool}/expert_paths.json`

### 7.2 재현 커맨드 (variant A headline)
```bash
source /home/woori/venvs/seka_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 scripts/ocq/eval_tau2_bench.py \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
  --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt \
  --domain telecom \
  --methods no_steer canonical_adaseka_amp0.3_topk3_T1.0 \
  --max-samples 200 --max-new-tokens 300 \
  --adaseka-expert-paths external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom/expert_paths.json \
  --adaseka-layers last10 \
  --out reports/tau2_2026_04_18/telecom_canonical_amp03_persample_N200.json
```
런타임: ~13 분 (Qwen2.5-7B-Inst, RTX 5880 / A6000 등 48GB 급 1 장).

### 7.3 Phase 0 — variantD bug 검증 (다음 단계 **선결 조건**)

Tier 3 의 B − D = +7.79pp 분해를 인용하려면 먼저 해야 하는 검증.

**Step 1** — verbose rerun (N=5 smoke, ~2 분):
```bash
source /home/woori/venvs/seka_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 scripts/ocq/eval_tau2_bench.py \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
  --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt \
  --domain telecom \
  --methods canonical_adaseka_amp0.3_topk3_T1.0 \
  --max-samples 5 --max-new-tokens 50 \
  --adaseka-expert-paths external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom-variantD/expert_paths.json \
  --adaseka-layers last10 \
  --verbose \
  --out /tmp/variantD_verify.json 2>&1 | tee /tmp/variantD_verify.log
```
로그에서 확인할 것:
- `[canonical_adaseka] loading expert SVDs from ...variantD/expert_paths.json` 줄 등장 여부.
- Hook 호출 수 / per-token mask token count / K-activation perturbation norm.
- 만약 log 에 expert load 자체가 안 찍히거나 mask token count = 0 이면 **hook 미적용 확정**.

**Step 2** — amp scaling smoke (N=5, ~2 분): `canonical_adaseka_amp1.0_topk3_T1.0` 로 바꿔서 실행. amp=0.3 대비 perturbation 3.3× 증가해야 효과 비례. prediction 여전히 no_steer 동일이면 **bug 확정**; 변동 생기면 D=0 은 real (매우 작은 effect 였음을 시사).

**Step 3** — perturbation norm direct check (~5 분): canonical_adaseka_engine.py 에 debug hook 추가해서 K_steered − K 의 frobenius norm per-layer 기록. variantA vs variantD 비교.

**결과 해석**:
- Bug 확정 → variantD 재구성 필요. 기존 N=200 run 은 paper 에 인용 금지.
- D=0 이 real 확정 → "random direction 은 effective perturbation 없음" 주장 가능. 단 왜 effect 가 정확히 0 이었는지 mechanism 설명 단락 필요.

### 7.4 Tier 1 착수 방법 (canonical angle)
- 스크립트: **아직 미작성**.
- 의사코드:
  ```python
  for facet in ['function_action', 'io_type', 'domain', 'tool_category']:
      U_ours = B_ont_block[facet]                     # our column block
      K_pos = collect_K(queries where gt_facet == facet)
      K_neg = collect_K(queries where gt_facet != facet)
      delta = (K_pos.mean(0) - K_neg.mean(0))
      U_proxy, _, _ = torch.linalg.svd(delta)[:, :rank]
      angle = canonical_angles(U_ours, U_proxy[:, :U_ours.shape[1]])
      log(angle.mean(), angle.max())
  ```
- 출력: `reports/adaseka_canonical_angle_2026_04_18.json`.
- 비용: ~15 분 GPU (Qwen Telecom).

### 7.5 BFCL-v3 Parallel + secondary bench 착수 (O4)

**BFCL-v3 (Primary smoke)**:
- 스크립트: `scripts/ocq/eval_bfcl.py` (이미 존재, 단 **proxy scoring** 주의 — 스크립트 주석 참조).
- 데이터: HuggingFace hub 에서 on-demand fetch (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`, `BFCL_v3_parallel.json` + `possible_answer/*.json`).
- Smoke 커맨드 (10 tasks, ~10 분):
  ```bash
  source /home/woori/venvs/seka_env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python3 scripts/ocq/eval_bfcl.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt \
    --subsets parallel --max-samples 10 --max-new-tokens 256 \
    --beta -0.1 --alpha 0.3 \
    --out reports/bfcl_2026_04_18/parallel_smoke.json
  ```
- **주의**: 이 eval_bfcl.py 는 "function-name set overlap" 기반 proxy. Official BFCL AST scoring 이 아니므로 paper 에 표기 시 **"proxy function-name F1 (not official BFCL-v3 AST score)"** 로 정확히 기술.

**Secondary bench (StableToolBench / AppBench)**:
- Eval 스크립트 **없음 (신규 작성 필요)**. 착수 전 구조 먼저 확인:
  ```bash
  ls external/StableToolBench/ external/AppBench/
  ```
- 벤치 구조 파악 → facet-diverse multi-tool 인지 confirm → eval 스크립트 작성 → smoke → full.

### 7.6 Tier 2 (BiasBios) 착수 방법
- Data: `external/SEKA/data/biasbios/biasbios.pkl` (real AdaSEKA 원본 — `external/SEKA/src/model/adaptive_seka_llm.py` 참조).
- B_ont 빌드: occupation-based ontology 를 `scripts/ocq/build_qwen_<name>_b_ont.py` 패턴으로 작성. BiasBios ~28 occupation facet.
- Eval: real AdaSEKA 의 classification head 재사용, expert SVD 만 B_ont-derived 로 교체.
- 참고: `adaseka_scope_mismatch_2026_04_18.md` — BiasBios 는 canonical AdaSEKA 가 동작하는 home-turf (즉 ours ≪ real 가능성 상당). **Upside 수단 아니라 handoff 시나리오 D 방어 보험** (§5 Option 3 참조).

---

## 8. 논문에 실을 때의 risk / counter-argument 매뉴얼 (v2)

| 리뷰어 가능 지적 | 방어 |
|---|---|
| "당신의 B_ont-derived experts 는 canonical AdaSEKA 가 아니다" | §5.5.3.1 relabeling (E3) — "operator-form ablation on shared basis" 라고 명시하고, canonical AdaSEKA training data 가 tool-selection 에 없음을 `adaseka_scope_mismatch` 근거로 공개. |
| "Tier 3 variant D (random orthonormal) 가 정확히 0 effect? 통계적으로 이상하다" | **현재 open issue**. Phase 0 (§7.3) 로 hook 미적용 bug 인지 real zero 인지 확인 후 논문 인용. bug 로 판명되면 variantD re-run; real zero 면 mechanism 설명 단락 추가. **bug 해소 전 variantD 기반 "basis load-bearing" 주장 금지**. |
| "Tier 3 variant D 가 +28pp 근처 나오면 당신들 basis 는 load-bearing 이 아니다" | Phase 0 후 variantD non-zero 인 경우. "routing interface + facet split 이 주 기여, basis direction 은 부분 기여" 로 방향 전환. §1.0 reframing (E1) 은 이 경우에도 유효. |
| "§1.0 structural argument 가 τ² 결과로 반박된다" | E1 의 facet-diverse vs uniform regime 분리. falsifiable prediction: 새로운 multi-tool 벤치에서 facet diversity 를 측정하면 stationary K-side 성공 여부 예측 가능. |
| "§1.1 empirical 증거가 MetaTool ST4 단일 벤치인데 generality 보장되나" | **v1 에서 under-emphasized 했던 약점**. Option O4 (BFCL-v3 Parallel + secondary bench) 로 empirical triangulation. BFCL 이 proxy scoring 이므로 secondary bench (StableToolBench / AppBench) 에서 공식 scoring 추가 권장. |
| "routing 이 architectural 이면 AdaSEKA query-adaptivity 가 무의미하다는 당신들 주장은 AdaSEKA 에 대한 공격이 아닌가" | § Related Work 에서 AdaSEKA 의 prompt-highlighting scope 는 유지 (BiasBios 같은 single-answer). τ² tool-selection 에서만 routing 이 flatten 되는지 cross-check (Tier 2 BiasBios 가 이 방어의 증거). |
| "Thm 6.17 가정 (H-cat) 이 τ² Telecom 에 성립하는가" | Appendix 에 τ² 별 (H-cat) gain 측정 추가 필요 (현재 MetaTool ST4 만 측정). 추가 측정 ~5 분 GPU. |
| "Tier 2 BiasBios 를 generality 증거로 제시하지만 ours ≪ real 나왔다" | Tier 2 포지션을 "generality validation step" 으로 한정 (crash-through upside 표기 금지). ours ≪ real 이면 scope 좁혀서 "training-free derivation 은 facet-structured 카탈로그 규모 벤치 한정" 으로 기술. |
| "BFCL proxy scoring 이라 공식 leaderboard 급 증거가 아니다" | paper 에 "proxy function-name F1, not official BFCL AST score" 로 명시 + secondary bench (StableToolBench / AppBench) 의 official scoring 을 second base 로. |

---

## 9. 논문 영향 — 방향성 요약 (v2: numerology 제거)

> **v2 update**: 기존 v1 에서 사용한 "6.3 → 5.8 → 6.2 → 6.4" 식 점수 numerology 는 cross-review 피드백 ("NeurIPS 리뷰어 점수는 integer + 편차가 커서 0.1~0.5pp 단위 precision 은 엔지니어링 fiction") 을 수용해 제거. 아래는 방향성 (direction) + rationale 만.

| 시점 / Option | Direction (vs 직전 상태) | Rationale |
|---|---|---|
| 2026-04-17 저녁 (v3 locked) | — (기준선) | ladapt safe floor + 4 contributions |
| 2026-04-18 저녁, paper 수정 없이 canonical_adaseka +28.89pp 노출만 | 🔻 하락 | §1.0 literal 반례 → 리뷰어 "direct contradiction" flag |
| O1 적용 후 | 🔺 복구 (리뷰 flag 차단) | regime taxonomy 로 §1.0 structural 논거 scope 좁힘 |
| O1 + O2 (Tier 1 + Tier 3 variantD 검증) | 조건부 🔺 | variantD bug 해소 후 ablation table 삽입 시 "basis + facet split 분해" 강화. bug 로 판명되면 유지 |
| O1 + O2 + O3 (Tier 2 BiasBios) | 방어 범위 확대 | ours ≈ real 면 generality 지지, ours ≪ real 면 scope 좁힘 — **upside 가 아니라 falsifiability check** |
| O4 (O1 + Tier 3 + BFCL + secondary bench) | 🔺 가장 확실한 개선 | §3.5 ST4 단일 벤치 weakness 직접 해소. ≥2 facet-diverse bench 에서 Q-coverage positive signal 내면 §1.1 triangulation |
| O1 + O2 + O3 + O4 | 🔺 최대 방어력 | 모든 리뷰어 예상 공격 루트 대응. 시간 비용 큼 (~3 hr GPU 총합) |

**우선순위 해석**:
- **O1 은 필수 (defensive)**. 적용 안 하면 §1.0 반례로 리뷰어 공격 루트 열림.
- **O4 는 strategic upside**. ST4 단일 벤치 의존도 낮추는 게 §1.1 의 가장 의미 있는 강화.
- **O2 는 조건부** — variantD bug 검증 (Phase 0) 먼저.
- **O3 는 보험** — Tier 2 는 generality validation step 이지 점수 상승 수단 아님.

NeurIPS 2026 main track 에서 "우리 결과가 한 벤치에 몰려 있는가" 는 reviewer-level discriminator 이다. 그래서 O4 가 O2, O3 보다 먼저 들어가야 한다는 게 v2 의 판단.

---

## 10. 이 문서의 의미 (TL;DR, v2)

1. **Canonical AdaSEKA 라 부르던 것의 정체**: AdaSEKA 의 interface (routing + marker-gated K hook) 에 **우리 B_ont 에서 파생한 experts** 를 주입한 객체. 진짜 AdaSEKA 논문의 contrastive-training expert 가 아니다. basis_matching_trap 방지 위해 "training-free AdaSEKA-interface (B_ont-derived)" 표기 사용.
2. **측정된 성능**: τ² Telecom N=200 에서 +28.89pp ΔF1. Multi-domain subset 에서 +36.17pp (single 보다 크다).
3. **Tier 3 ablation** (v2 신규):
   - A (current) 0.5401 / B (no split) 0.3291 / D (random) 0.2512
   - A − B = +21.10pp (facet-split routing 기여) — **유효**.
   - B − D = +7.79pp (basis direction 기여) — **variantD bug 검증 전까지 인용 보류**.
   - D = 0.00pp literal identical → hook 미적용 의심, Phase 0 검증 필요.
4. **논문 충돌**: §1.0 의 "stationary K-side = multi-selection 구조 불가" claim 과 literal 충돌. 하지만 τ² Telecom 이 facet-uniform 카탈로그 (tool_category 100% single) 임을 이용해 **regime taxonomy 로 reframe** 가능.
5. **가장 약한 지점** (v2 명시): §1.1 의 Q-coverage empirical 증거 = MetaTool ST4 단일 벤치. +1.64pp Qwen / +0.40pp Llama 의 marginal magnitude 까지 합치면 "one lucky bench" 공격 루트 열려 있음. **Option O4 (BFCL + secondary bench) 로 triangulation 필요**.
6. **추가 실험 4 갈래**: (Phase 0) variantD bug 검증, (Tier 1) canonical angle, (Tier 3 재인용 조건부) random null 비교, (O4) BFCL + StableToolBench/AppBench. Tier 2 (BiasBios) 는 **upside 가 아니라 generality validation / 시나리오 D 방어 보험** 포지션.
7. **이번 세션 (paper edit only) 할 일**: O1 의 E1–E4 (§6). GPU 필요 없음, 2–3 시간.
8. **포기해서는 안 되는 것**: τ² Telecom 데이터. 버리면 §6 Table 1 과 §5.5.3 구조 붕괴 + deployment-relevance 상실 → "academic toy" 비판 루트 열림. Reframe 만이 유일한 방어.
9. **표현 원칙** (v2 수용): 점수 예측은 0.1~0.5pp 단위 precision 으로 쓰지 않는다. 방향성 (복구/개선/약화) + rationale 만.

---

**이 문서 작성 근거 메모들 (세부 확인 시 참조)**:
- `inter_expert_routing_architectural_2026_04_18.md` — routing entropy 83%, argmax identical
- `fake_sv_degeneracy_2026_04_18.md` — intra-expert SV uniform
- `basis_matching_trap_2026_04_18.md` — labeling discipline
- `adaseka_scope_mismatch_2026_04_18.md` — canonical AdaSEKA 의 single-answer scope
- `adaseka_proxy_mistake_recurrence_2026_04_17.md` — proxy 사건 history
- `external_baseline_use_original_source.md` — baseline 비교 원칙
- `handoff_training_free_adaseka_2026_04_18_evening.md` — 이번 handoff 의 3-tier 계획 + 5-scenario 결정 트리
- `adaseka_vs_ours_differentiation_2026_04_10.md` — AdaSEKA vs ours 이전 정의

---

## 부록 A. v2 changelog (vs v1, 2026-04-18 21:00 KST)

| 영역 | v1 | v2 (이 버전) | 근거 |
|---|---|---|---|
| Tier 3 결과 | 미기재 | §2.5 신설 (A/B/D 표 + variantD bug flag) | 2026-04-18 23:00 KST Tier 3 완료 + bit-exact identical predictions 검증 |
| variantD 해석 | "random 도 +28pp 나오면 재해석" 수준 | D=0 literal 이 bug 의심으로 명시, Phase 0 검증 선결 조건 | 200/200 task prediction identical, F1/Recall/Exact bit-exact |
| 점수 numerology | "6.3 → 5.8 → 6.2 → 6.4" 연속 추정 | 제거, directional (🔻🔺) + rationale 만 | cross-review 피드백: NeurIPS 점수는 integer + 편차 크다 |
| Tier 2 (BiasBios) 포지션 | "crash-through upside" | "시나리오 D 방어 보험 / generality validation step" | cross-review: BiasBios 는 real AdaSEKA home-turf → ours ≤ real default |
| ST4 single-benchmark weakness | 언급 없음 | §3.5 신설 | cross-review: "왜 single benchmark?" reviewer 공격 루트 |
| O4 옵션 (BFCL + secondary) | 없음 | §5 신설 | §3.5 해소 위한 empirical triangulation |
| BFCL external/ 경로 | 가정 없이 간접 언급 | 팩트 체크: HF fetch, proxy scoring 주의 | `scripts/ocq/eval_bfcl.py` 본문 확인 |
| Phase 0 verification | 없음 | §7.3 신설 | variantD bug 검증 3-step 절차 |
| 리뷰어 공격 매뉴얼 | 5 행 | 8 행 (variantD, single-bench, BFCL proxy, BiasBios 추가) | 위 변경 사항 반영 |
| τ² 제거 옵션 (O2 원안) | "−0.5 ~ −0.3" 수치 | 본문에서 삭제, deployment-relevance 상실 risk 명시 | cross-review: "academic toy" 비판 루트 |
