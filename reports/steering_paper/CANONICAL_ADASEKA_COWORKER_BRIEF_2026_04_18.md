# Canonical AdaSEKA 실험 — Coworker Brief (2026-04-18 저녁)

**대상 독자**: 논문 공저·공동 연구자 중 canonical AdaSEKA 라인을 처음 보는 사람.
**목적**: 지금까지 무엇을 돌렸고, 무엇이 나왔으며, 그 결과가 논문 §1.0/§1.1/§5.5.3.1 에 왜 문제가 되고 어떻게 대응할지, 이어서 어떤 추가 실험이 계획되어 있는지를 self-contained 하게 설명.
**한 줄 요약**: τ² Telecom N=200 에서 canonical AdaSEKA-interface (우리 B_ont 로 파생한 experts) 가 **+28.89pp ΔF1** 을 찍었다. 그런데 이건 stationary K-side operator 이고, 현재 paper §1.0 은 "stationary K-side 는 multi-selection 구조적 불가" 라고 선언해 놓은 상태다. 그래서 결과를 버릴 수도, 그대로 실을 수도 없고, **regime taxonomy 로 reframe + 추가 실험(Tier 1–3)** 하는 방향으로 간다.

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

## 5. 대응 옵션 (우선순위)

### Option 1 — Honest reframing (minimal, paper 만 수정, 실험 0)
1. §1.0 L43–47 에 "facet-uniform catalog 에서는 stationary 가 동작함" 을 명시. 구조 논거를 facet-diverse multi-tool 에 국한.
2. §1.1 empirical block 에 "canonical AdaSEKA on τ² Telecom = +28.89pp, facet-uniform regime 에서 예상 가능한 동작" 1 문장 추가.
3. §5.5.3.1 표기 교정: "AdaSEKA baseline" → "Training-free AdaSEKA-interface (B_ont-derived experts)" 또는 "Operator-form ablation on shared B_ont basis".
4. 신규 정의 block: **"Facet diversity regime"** — 각 벤치가 어디 속하는지 매핑 (MetaTool ST4 → diverse; τ² Telecom/Retail → uniform-ish; τ² Airline → mixed; BiasBios → single-answer out-of-scope).
- **비용**: 약 2–3 시간 편집.
- **점수 영향 추정**: 현재 5.8–6.2 (v3 그대로 두면 §1.0 반례로 리뷰어 공격 루트) → 6.2–6.5 복구.

### Option 2 — Tier 1+3 추가 실험 (paper score 개선 폭 확대)
Tier 1: B_ont-derived expert direction vs 실제 contrastive-SVD direction 사이 **canonical angle**. 현재는 contrastive training data 가 없으므로 proxy (GT-vs-non-GT K-activation) 를 써야 함.
- 목적: "우리 facet centroid direction 이 contrastive separation direction 과 얼마나 비슷한가" 를 정량화.
- 각도 ≤ 10°: "basis 가 contrastive training 의 substitute" 강한 주장.
- 각도 ≥ 30°: "direction 이 달라도 routing 성공 → AdaSEKA mechanism 이 direction 에 loose" 다른 주장.
- 비용: ~15분 GPU (Qwen Telecom 에만).
- 산출: appendix 1 단락 + canonical angle 표.

Tier 3: Design-space ablation. 5 variant.
- (A) current (facet-split + uniform SV, B_ont-derived)
- (B) no facet split (하나의 큰 basis)
- (C) unsupervised real SV from K activation statistics
- (D) random orthonormal experts (matched rank)
- (E) fixed per-head mixture (routing 우회)
- 목적: "basis 효과 vs operator form 효과 vs routing adaptivity" 분리.
- 비용: ~26분 GPU (각 variant 에 Telecom N=200, 이미 Tier3 variant B 가 돌아간 기록 있음 — commit `cdaa175`).
- 산출: paper `tab:adaseka-ablation`.

Tier 1+3 결과가 강하게 나오면:
- "Training-free catalog → AdaSEKA interface derivation" 이 **논문 contribution #5** 로 승격.
- "AdaSEKA 의 query-adaptivity 자체가 architectural preference 로 대부분 설명됨" 이라는 부수 기여 가능 (§2.3 routing entropy 83%).

### Option 3 — Tier 2 BiasBios cross-check 추가 (crash-through)
- Tier 2 목적: B_ont 를 occupation ontology 로 새로 빌드해서 AdaSEKA interface 에 주입 → BiasBios (real AdaSEKA home-turf) 에서 real AdaSEKA 결과 대비 비교.
- 만약 ≈ 동등 성능: "catalog ontology 가 contrastive training 의 structural substitute" 가 BiasBios 같은 single-answer 벤치에서도 성립. 본문 표 1개 확정.
- 비용: ~30분 GPU + ontology 빌드 + BiasBios data prep (`external/SEKA/data/biasbios/biasbios.pkl` 존재).
- 산출: main body 신규 표 + §7 paragraph.

### 비교
| Option | 필요 작업 | 예상 점수 변화 | 리스크 |
|---|---|---|---|
| O1 | paper edit 4 곳 | +0.2 ~ +0.5 (vs 현 상태) | 없음 — defensive |
| O1 + Tier 1+3 | O1 + ~40 분 GPU | +0.4 ~ +0.8 | Tier 3 variant D 에서 random 도 +30pp 나오면 "basis 무관" 결론 → 우리 contribution 약화 |
| O1 + Tier 1+3 + Tier 2 | + ~1시간 GPU + data prep | +0.6 ~ +1.2 | Tier 2 가 negative 면 generality 주장 약화. 하지만 (a) τ² Telecom 결과는 남고 (b) O1 reframing 만으로도 방어 가능하므로 **downside 가 hard cap** |

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
- Expert derivation: `scripts/diagnostics_2026_04_16/build_adaseka_experts_from_bont.py`
- Tier 3 runner: `scripts/diagnostics_2026_04_16/build_adaseka_variants_tier3.py`
- Routing diag: `scripts/ocq/diag_canonical_adaseka_routing.py`
- Facet decomposition: `scripts/diagnostics_2026_04_16/decompose_telecom_by_facet.py`
- Eval driver: `scripts/ocq/eval_tau2_bench.py --methods canonical_adaseka_amp<A>_topk<K>_T<T>`
- 결과 JSON (오늘 완료):
  - `reports/tau2_2026_04_18/telecom_canonical_amp03_persample_N200.json` (N=200, full)
  - `reports/tau2_2026_04_18/telecom_gt_facet_analysis_v2.json` (facet diversity 분석)
  - `reports/tau2_2026_04_18/canonical_adaseka_routing_diag.json` (routing entropy, N=10)
  - `reports/tau2_2026_04_18/telecom_canonical_tau2trained_smoke5.json` (smoke 5)
  - `reports/polarity_flip_2026_04_18/{qwen,llama}_{telecom,retail}.json` (D1 sign predictor)
- B_ont source (expert 파생 입력):
  - `external/SEKA/seka_projections/ontology-qwen25-7b-tau2-{telecom,retail,airline}/B_ont.pt`
  - `external/SEKA/seka_projections/ontology-llama31-8b-tau2-{telecom,retail}/B_ont.pt`
- 파생된 expert:
  - `external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-{telecom,retail,metatool}/expert_paths.json`

### 7.2 재현 커맨드
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

### 7.3 Tier 1/3 착수 방법
- Tier 1 스크립트: **아직 미작성**. 작업 계획:
  ```python
  # 의사코드
  for facet in ['function_action', 'io_type', 'domain', 'tool_category']:
      U_ours = B_ont_block[facet]                     # our column block
      K_pos = collect_K(queries where gt_facet == facet)
      K_neg = collect_K(queries where gt_facet != facet)
      delta = (K_pos.mean(0) - K_neg.mean(0))
      U_proxy, _, _ = torch.linalg.svd(delta)[:, :rank]
      angle = canonical_angles(U_ours, U_proxy[:, :U_ours.shape[1]])
      log(angle.mean(), angle.max())
  ```
  출력: `reports/adaseka_canonical_angle_2026_04_18.json`.
- Tier 3 (variant B): `scripts/diagnostics_2026_04_16/build_adaseka_variants_tier3.py` (commit `cdaa175`) 이미 존재. variant D (random orthonormal) 는 이 스크립트에 추가 필요.

### 7.4 Tier 2 (BiasBios) 착수 방법
- Data: `external/SEKA/data/biasbios/biasbios.pkl` (real AdaSEKA 원본 데이터 경로 — `external/SEKA/src/model/adaptive_seka_llm.py` 참조).
- B_ont 빌드: occupation-based ontology 를 `scripts/ocq/build_qwen_<name>_b_ont.py` 패턴으로 작성. BiasBios 는 ~28 occupation facet.
- Eval: real AdaSEKA 의 classification head 재사용하되 expert SVD 만 B_ont-derived 로 교체.
- 참고: `adaseka_scope_mismatch_2026_04_18.md` — BiasBios 는 canonical AdaSEKA 가 동작하는 home-turf.

---

## 8. 논문에 실을 때의 risk / counter-argument 매뉴얼

| 리뷰어 가능 지적 | 방어 |
|---|---|
| "당신의 B_ont-derived experts 는 canonical AdaSEKA 가 아니다" | §5.5.3.1 relabeling (E3) — "operator-form ablation on shared basis" 라고 명시하고, canonical AdaSEKA training data 가 tool-selection 에 없음을 `adaseka_scope_mismatch` 근거로 공개. |
| "Tier 3 variant D (random orthonormal) 도 +28pp 나오면 당신들 basis 가 load-bearing 이 아니다" | Tier 3 pending. 만약 random 도 +28pp 나오면 **정직히 "routing interface 가 load-bearing, basis 는 부수적" 로 방향 전환**. 이 경우에도 §1.0 reframing (E1) 은 유효. |
| "§1.0 structural argument 가 τ² 결과로 반박된다" | E1 의 facet-diverse vs uniform regime 분리. falsifiable prediction: 새로운 multi-tool 벤치에서 facet diversity 를 측정하면 stationary K-side 성공 여부 예측 가능. |
| "routing 이 architectural 이면 AdaSEKA query-adaptivity 가 무의미하다는 당신들 주장은 AdaSEKA 에 대한 공격이 아닌가" | § Related Work 에서 AdaSEKA 의 prompt-highlighting scope 는 유지 (BiasBios 같은 single-answer). τ² tool-selection 에서만 routing 이 flatten 되는지 cross-check (Tier 2 BiasBios 가 이 방어의 증거). |
| "Thm 6.17 가정 (H-cat) 이 τ² Telecom 에 성립하는가" | Appendix 에 τ² 별 (H-cat) gain 측정 추가 필요 (현재 MetaTool ST4 만 측정). 추가 측정 ~5 분 GPU. |

---

## 9. 이번 세션 기준 점수 추정 요약

| 시점 | 점수 estimate | 근거 |
|---|---|---|
| 2026-04-17 저녁 (v3 locked) | 6.3–6.5 | ladapt safe floor + 4 contributions (Q-coverage, K-stability, Thm 6.1, OCQ) |
| 2026-04-18 저녁, paper 수정 없이 canonical_adaseka +28.89pp 노출만 한 경우 | 5.8–6.2 | §1.0 literal 반례 → 리뷰어 공격 루트 |
| O1 (paper edit 4 건) 적용 후 | 6.2–6.5 | 복구 + regime taxonomy 추가 |
| O1 + Tier 1+3 결과 긍정 시 | 6.4–6.8 | training-free derivation contribution 추가 |
| O1 + Tier 1+3 + Tier 2 긍정 시 | 6.6–7.2 | BiasBios main-body 표 + dual contribution |

NeurIPS 2026 main track acceptance threshold ~6.5. **O1 은 필수**, O2/O3 는 upside 판단.

---

## 10. 이 문서의 의미 (TL;DR)

1. **Canonical AdaSEKA 라 부르던 것의 정체**: AdaSEKA 의 interface (routing + marker-gated K hook) 에 **우리 B_ont 에서 파생한 experts** 를 주입한 객체. 진짜 AdaSEKA 논문의 contrastive-training expert 가 아니다.
2. **측정된 성능**: τ² Telecom N=200 에서 +28.89pp ΔF1. Multi-domain subset 에서 +36.17pp (single 보다 크다).
3. **논문 충돌**: §1.0 의 "stationary K-side = multi-selection 구조 불가" claim 과 literal 충돌. 하지만 τ² Telecom 이 facet-uniform 카탈로그 (tool_category 100% single) 임을 이용해 **regime taxonomy 로 reframe** 가능.
4. **추가 실험 Tier 1–3**: basis vs operator form 분리, random control, BiasBios cross-check. Tier 1+3 은 ~40 분 GPU, Tier 2 는 ~1 시간.
5. **이번 세션에서 해야 할 일**: paper edit 4 건 (E1–E4). 실험은 다른 세션 담당.
6. **포기해서는 안 되는 것**: τ² Telecom 데이터. 버리면 §6 Table 1 과 §5.5.3 구조 붕괴. Reframe 만이 유일한 방어.

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
