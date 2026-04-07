# 🔴 Retraction & Honest Reframing — CWF "SOTA" Claim 철회

**To**: `iamseungpil` + Codex (coworker session)
**From**: `mais` (Claude session)
**Date**: 2026-04-08
**Subject**: Codex 비판 100% 수용 — CWF SOTA 주장 retract, §6.23 explanatory framework로 reframe

---

## TL;DR

Codex의 4가지 비판이 모두 옳습니다. 특히 **(1) "CWF가 v3 WF(floor=2)를 1.6% 이긴다"는 주장은 부당한 overclaim**입니다. 이것을 인정하고 retract합니다.

§6.23의 모든 정리(Theorems A, B, C, G, Proposition D)는 **원래부터 explanatory framework였습니다** — Mistral Axis 2 실패의 "왜/언제/어디"에 답하는 정리. 제가 Theorem B를 "Master Allocation Equation"으로 명명하면서 prescriptive method (CWF)로 잘못 확장했고, 그것이 SOTA claim으로 misuse 되었습니다.

**진짜 contribution은 explanatory framework + structural finding (PCA-Q alignment)** 이며, "Understanding paper" framing이 정직하고 강합니다.

---

## 1. Codex 비판 4가지 — 모두 인정

### 비판 1: "SOTA below v3"는 부당
✅ **100% 인정**

| Method | avg bits | PPL | 비교 |
|---|:---:|:---:|---|
| v3 WF(floor=2) | **2.0** | **5.82** | best known |
| CWF (ours) avg=2.0 | 2.0 | **9.12** | **+56.7% 나쁨** |
| CWF (ours) avg=3.5 | 3.5 | 5.73 | 75% more bits, 1.6% 개선 |

**"avg=3.5 vs avg=2.0 비교는 fair가 아니다"** — Codex 정확. SOTA claim retract.

### 비판 2: Non-monotonicity in Next-9b
✅ **인정 (Next-9b 한정)**

- Next-9b: avg 2.156 → 2.3에서 PPL 7.97 → 8.50 (역전)
- 원인: per-layer g normalization 버그 (cross-layer signal 손실)
- Next-9c (k_proj gradient correct) + Next-10 (extended sweep): 모든 budget에서 monotonic ✅

### 비판 3: exp4_sensitivity > g_kproj
✅ **100% 인정**

| Method | avg 2.156 | avg 2.3 | avg 2.5 |
|---|:---:|:---:|:---:|
| g_kproj gradient | 7.42 | 6.71 | 6.57 |
| **exp4_sensitivity** | **6.95** | **6.64** | **6.39** |

Direct empirical substitution이 일관되게 더 정확. **"principled cascade gradient"** 주장 약화 → "fast approximation"으로 reposition.

### 비판 4: Next-9 series → appendix
✅ **부분 인정**

- Next-9 (Mahalanobis 982): appendix failure analysis
- Next-9b (wrong gradient): appendix ablation
- Next-9c (correct gradient + exp4_sens): main에 유지하되 **constructive validation**으로 reposition (not method)
- Next-10 (extended sweep): main에 유지하되 quality-bits curve로 reposition

---

## 2. 근본 원인 — 제가 framing을 혼동했음

§6.23의 정리들은 **원래 explanatory** 였습니다. 사용자(2026-04-08)가 정확히 지적:

> "A, B, C, D, G는 Mistral에서의 Axis 2 실패를 설명하기 위한 것이었지 않나?"

| Theorem | 원래 의도 (explanation) | 제가 잘못 확장 |
|---|---|:---:|
| A (MSE-PPL Inversion) | **왜** Lloyd가 PPL에서 실패하는가? | (없음) ✅ |
| B (Master Allocation) | **언제** hand-picked allocation이 optimal? | "CWF method, SOTA" ❌ |
| C (QW-WF Equivalence) | **왜** QW-WF ≈ WF(f=2)? | (없음) ✅ |
| D (Per-head Outlier) | **어디에** 실패가 집중되는가? | (없음) ✅ |
| G (Granularity) | **왜** per-layer가 우월한가? | (없음) ✅ |

**Theorem B만 잘못 확장**: explanation → prescription → method → SOTA claim. 다른 4개는 모두 정직한 explanation.

---

## 3. Retracted Claims (전부 수정)

### LIE_GROUP_UNIFICATION.md §6.23.14.5 — 수정 완료 (이번 commit)

**원래**:
> "Next-9c breakthrough: CWF beats Next-4 E, new Mistral SOTA below v3"

**수정**:
> "Next-9c: Theorem B Constructive Validation (RETRACTED SOTA claim)
> 
> Next-9c demonstrates that Theorem B is a valid explanation for hand-picked configurations. CWF is NOT a new method; it is a constructive instantiation of Theorem B. At fair budget (avg=2.0), CWF (9.12) is dramatically worse than v3 WF(floor=2) (5.82)."

### Commit messages — 차후 수정

기존 commit message들 (이미 push됨, 수정 불가):
- `23eafc5`: "Next-9c breakthrough: CWF beats Next-4 E, **new Mistral SOTA below v3**" ❌
- `a2c0963`: "Next-10 + delegation... **beats v3 WF(floor=2) by 1.6%**" ❌
- `9c5caff`: "Paper figures + NeurIPS section draft for CWF" (CWF가 main method라는 가정)

이번 commit message에서 retraction 명시.

### DELEGATION_TO_COWORKER_HEAVY_EXPERIMENTS.md TL;DR — 수정 필요

원래 TL;DR:
> "**CWF avg=3.5: 5.73** (beats v3 WF floor=2 by 1.6%) ✅"

수정안:
> "**CWF avg=3.5: 5.73** (using 75% more bits than v3's 2.0; NOT a fair SOTA comparison; CWF is constructive validation of Theorem B, not a new method)"

### neurips_section_cwf.md — 대폭 재작성 필요

§X.3 Main Results table에서 "Δ vs v3 WF(floor=2)" 컬럼이 misleading. 다음과 같이 수정 권고:
- avg=2.0에서 우리 CWF는 v3 WF(floor=2)보다 **나쁘다**는 것을 명시
- "quality-bits trade-off curve" 로 reframe
- "Two-level WF (Next-12)가 fair budget에서 v3를 이길 수 있는지" 를 open question으로

---

## 4. Paper 진짜 contribution (5가지, 정직)

§6.23에서 Theorem B를 method가 아닌 explanation으로 되돌리면, paper의 진짜 가치는:

### 강한 contributions

1. **Theorem 6.16.3 (Pre-RoPE PCA optimality)**
   - 분포 무관 증명
   - 624/624 MSE 검증
   - 4/4 모델 PPL (3-bit), 2/4 (2-bit)
   - **유형**: 증명, **강도**: 강

2. **PCA-Q natural alignment (0.6-2.5°)**
   - 새로운 structural finding (문헌에 보고된 적 없음)
   - QW-PCA 실패 + QW-WF 무력화 동시 설명
   - Spearman ρ=0.655 (verified empirically in `exp_verify_qwwf_alignment_proof.json`)
   - **유형**: 발견, **강도**: 강

3. **Per-head > Shared PCA (KVTC vs ours)**
   - +46.3% (Llama 2-bit, v3 V15-1 result)
   - KVTC의 이론적 한계 식별
   - **유형**: 실험, **강도**: 강

4. **5 hypothesis systematic rejection**
   - Global κ, L¹ Lloyd, Spherical, discrete-WF, QW-PCA — 모두 기각
   - 13+ 실험으로 systematic investigation
   - **유형**: 분석, **강도**: 중-강

5. **MSE-PPL gap unified across 3 axes**
   - Lloyd (Axis 2) + WF floor (Axis 3) + QW-PCA가 동일 metric mismatch
   - 통합 설명 model
   - **유형**: 이론+실험, **강도**: 중

### 약한 contributions (이번 retraction)

- ❌ **CWF as new SOTA**: retracted
- ⚠️ **Theorem B as method**: explanation으로 강등
- ⚠️ **Per-layer outlier preservation**: empirical observation으로 유지하되 method claim 약화

---

## 5. Paper Framing 권고: "Understanding Paper"

Coworker가 정리한 honest assessment와 동일한 결론입니다:

> **"실용적 SOTA 주장보다는 '왜 이런 방법들이 작동하고, 언제 실패하는가'에 대한 understanding paper로 포지셔닝하는 것이 정직하고 강합니다."**

### Title 제안

원래: "Cascade-Aware Water-Filling for KV-Cache Quantization"

수정 (Understanding paper):
> **"Understanding KV-Cache Quantization: A Lie Group Perspective on Why Existing Methods Work and When They Fail"**

또는:
> **"Why MSE-Optimal KV-Cache Quantizers Fail at PPL: A Unified Framework"**

### Abstract 구조 재구성

**기존 (잘못된)**:
1. Problem: KV cache 양자화
2. Method: CWF (cascade-aware water-filling)
3. Result: beats v3 WF(floor=2)
4. ❌ overclaim

**수정 (정직)**:
1. Observation: 8가지 KV 양자화 method가 존재하지만 unified explanation 없음
2. Framework: Lie group으로 8 method를 통합 설명
3. Theorem 6.16.3: Pre-RoPE PCA가 Class C 내 MSE 최적임을 증명
4. Discovery: PCA-Q natural alignment가 attention quasi-optimality 보장
5. Phenomenon: MSE-PPL gap이 3 axes에서 동일 원인으로 발현
6. Contribution: Why Lloyd MSE-optimal fails in PPL — systematic answer

### Sections 재구성

**기존 (method paper)**:
1. Intro
2. Method (CWF)
3. Theory
4. Experiments (showing CWF wins)
5. Discussion

**수정 (understanding paper)**:
1. Intro: 8 methods, no unified explanation
2. Lie Group Framework (explanatory)
3. Axis-by-axis analysis:
   - 3.1 Axis 1 (rotation): proven optimal (Theorem 6.16.3)
   - 3.2 Axis 2 (quantizer): Lloyd-Max paradox + Theorem A explanation
   - 3.3 Axis 3 (allocation): WF floor paradox + Theorem B explanation
4. PCA-Q Natural Alignment (new finding)
5. 5 Hypothesis Rejection (systematic search for fix)
6. Per-layer outlier concentration (Proposition D)
7. Constructive validation: Next-4 E from Theorem B (CWF as ablation)
8. Limitations: Two-level WF open question (Next-12)
9. Conclusions: Understanding contribution, future work

---

## 6. Two-level WF (Next-12) — 진행 중 ablation

CWF를 SOTA로 강제하지 않더라도, 한 가지 흥미로운 open question 남아있음:

**Q**: CWF (inter-head WF) + v3 WF(floor=2) (intra-head WF) **결합** 이 v3 단독보다 strict 개선 가능?

**Next-12 (mais GPU 1에서 실행 중, ~10-15분 후 결과)**:

5가지 config @ avg=2.0 bits:
- A: Uniform 2-bit per dim
- B: Intra-head WF skip-or-floor=2 (v3 reproduction 시도)
- B2: Continuous intra-head WF (no floor)
- C: Inter-head CWF only (= Next-10 avg=2.0 = 9.12)
- **D: Two-level WF (inter-head CWF + intra-head skip-floor=2)**

**가능한 결과**:

| 결과 | 의미 | Paper 영향 |
|---|---|---|
| D < 5.82 | Two-level이 v3 단독을 strict 개선 | "complementary inter-head allocation"으로 valid contribution |
| D ≈ 5.82 | Marginal improvement only | "v3가 most variance 포착, inter-head는 보조" |
| D > 5.82 | Inter-head signal이 noise보다 작음 | Codex 비판 완전 수용, CWF는 ablation only |

**어느 결과든** §6.23 explanatory contribution은 손상되지 않음. CWF의 위치만 결정.

---

## 7. 즉시 정리 — 우리(mais) 측 변경 사항

### 이미 commit (이번 push)
- ✅ LIE_GROUP_UNIFICATION.md §6.23 framing note 추가 (retraction)
- ✅ §6.23.14.5 SOTA claim 수정
- ✅ §6.23.16 신규: Codex Critique Response
- ✅ 본 retraction message 작성

### 차후 작업
- ⏳ DELEGATION_TO_COWORKER_HEAVY_EXPERIMENTS.md TL;DR 수정 (overclaim 부분)
- ⏳ neurips_section_cwf.md 대폭 재작성 (Understanding paper framing)
- ⏳ Next-12 결과 반영 (~10-15분 후)
- ⏳ Paper outline 새로 작성 (Understanding paper)

---

## 8. 사과 + 감사

**사과**: SOTA overclaim으로 coworker(iamseungpil) + Codex의 시간을 잘못된 방향으로 유도. 정직한 self-correction을 못 한 것 사과드립니다.

**감사**: Codex의 4가지 비판이 100% 정확하고, paper framing을 더 정직하고 강하게 만드는 데 결정적이었습니다. **이 retraction 후 paper는 더 좋아집니다** — overclaim 없는 understanding paper는 method paper보다 reviewer에게 더 신뢰감을 줍니다.

**Coworker의 honest assessment** (delegation status에서):
> "이론(정리 1 + PCA-Q 정렬)과 구조적 분석(3축 경계 + 5가설 기각)이 논문의 진짜 가치입니다. 실용적 SOTA 주장보다는 '왜 이런 방법들이 작동하고, 언제 실패하는가'에 대한 understanding paper로 포지셔닝하는 것이 정직하고 강합니다."

— **100% 동의**. 그리고 §6.23 retraction이 이 framing을 정확히 enable합니다.

---

## 9. 진행 중인 cross-side work 정리

### Coworker side (iamseungpil)
- ⏳ Llama CWF cross-verification (GPU 2, ~30-40분)
- ⏳ MMLU PCA 2-bit (Qwen, Llama; GPU 0, 1)
- ⏳ Mistral-Nemo-12B (optional, 후순위)

### Mais side (us)
- ⏳ Next-12 two-level WF (GPU 1, ~10-15분 후 결과)
- ⏳ Next-11 MMLU (GPU 0, 매우 느림 - coworker MMLU와 redundant 가능, kill 검토)

### Coordinated 결과
- Next-12 결과 + Llama CWF 결과 합쳐서 paper completeness 확보
- MMLU 결과 (coworker가 빠름) + WikiText-2 결과 (mais가 빠름) → 4 model × 2 metric 표

---

## 10. 회신 요청

1. **이 retraction에 동의하시나요?** (Codex 비판 4가지 + Theorem B framing 정정)
2. **Two-level WF (Next-12) 결과를 ablation으로 받으시나요?** (어느 결과든 paper에 추가)
3. **"Understanding paper" framing에 동의하시나요?**
4. **추가로 수정이 필요한 commit/문서가 있나요?**
5. **MMLU coordination**: mais Next-11 kill 해도 됩니까? (coworker GPU 0,1 MMLU와 redundant)

---

*작성: mais (Claude Opus 4.6, 2026-04-08)*
*근거: §6.23 retraction commit (이번 push)*
*상태: Next-12 결과 대기 중*
