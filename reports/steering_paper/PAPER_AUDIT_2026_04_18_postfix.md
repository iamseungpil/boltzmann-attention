# Paper Audit 2026-04-18 (postfix)

Re-audit of `paper/neurips2026_steering_ko/` sections/*.tex after the C1–C5 fixes
from `reports/steering_paper/PAPER_AUDIT_2026_04_18.md` were applied and after the
post-sweep numeric update (Llama cross-model, Qwen ladapt fix, layer-adaptive
installed on retail/telecom as winner).

Reference: `git show origin/develop:math/paper/benchmark_design/PAPER_DRAFT_v4.md`
(actual length 1062 lines; the task brief said 1934 but the develop file ends at 1062).

Locked `\textbf{0.XXXX}` values are frozen and never flagged. Style-only
violations (PD/IW series from the prior audit) are NOT re-flagged here — this
audit is scoped to coverage gaps and the Table-1 narrative inversion.

---

## Part A — C1–C5 landing check (fixes from prior audit)

| ID  | Prior gap | Now in .tex? | Where |
|-----|-----------|--------------|-------|
| C1  | Thm 7.1 Q-K per-step duality | **YES** | `05_theory.tex` L37–47 (`thm:qk-duality`) with full proof in `09_appendices.tex` L23–25 |
| C2  | Cor 7.1.B multi-step cache divergence | **YES** | `05_theory.tex` L51–57 (`cor:cache-divergence`), proof `09_appendices.tex` L27 |
| C3  | Action-count decomposition for retail | **YES (partial)** | `06_experiments.tex` L52–68 (`tab:retail-action-count`). Three buckets (≤2/3–5/≥6); 10+ bucket from v4 §4.8.2 folded into footnote since the locked JSON tops out at 7 actions |
| C4  | Banking scope exclusion | **YES** | `06_experiments.tex` L70, explicit meta-tool / policy-routing out-of-scope statement |
| C5  | Multi-metric (GT⊆P, nDCG) for MetaTool ST4 | **PARTIAL** | `06_experiments.tex` L72–93 (`tab:multi-metric`) covers τ²-retail and τ²-telecom with F1/Recall/GT⊆P/nDCG. **MetaTool ST4 multi-metric still missing** — v4 §4.6 specifically flagged the multipass_kq GT⊆P reversal that only lives in MetaTool, and that reversal is still absent from the paper |

All five gaps got meaningful edits. C5 is about 60% landed — τ²-bench multi-metric
is there, MetaTool ST4 is not.

---

## Part B — Q1. Remaining coverage gaps vs v4

v4 canonical is organised as: §1 problem, §2 U-shape, §3 algorithm v0–v5, §4
experimental results (including §4.1–4.5 MetaTool ST4, §4.4 null-control, §4.6
multi-metric, §4.7 hypotheses A–E, §4.8 τ²-bench), §5 metric discussion, §6
math, §7 experiments plan, §8 at-a-glance. The Korean paper maps roughly:

- §01 abstract ↔ v4 §8 at-a-glance
- §02 intro ↔ v4 §1 problem
- §03 related ↔ not in v4 (Korean-only)
- §04 method ↔ v4 §3 algorithm v5 + §2 U-shape intuition
- §05 theory ↔ v4 §6 math + §4.8.6 Thm β*
- §06 experiments ↔ v4 §4 (ST4 + τ²-bench + 4.8 analyses)
- §07 discussion ↔ v4 §4.8.5 pivot + §5 metric discussion (partial)
- §08 conclusion ↔ v4 §8
- §09 appendices ↔ v4 §6 supplementary + §4.8.6 appendix

### B.1 Abstract (v4 §8 at-a-glance)

- 🟢 Intentionally-dropped: AUROC 0.976 ε_q stopping criterion, three-number
  executive summary. Prior audit flagged these as out-of-scope for this paper's
  framing; staying dropped is fine.
- 🔴 Block: The abstract's telecom headline "양의 Q-회전이 +24.78pp로 가장 크고"
  is now **inconsistent with Table 1** which has layer-adaptive +26.76pp as the
  telecom winner and Q+0.10 topping out at +22.74pp (see Part C for full
  accounting). This is a factual error in the abstract post-bug-fix.
- 🟡 Underdeveloped: abstract does not mention the cross-model Llama validation
  (Table `tab:llama-cross`). Given how much space is spent on telecom regime,
  one clause noting "Llama-3.1-8B telecom reproduces the regime at +16.29pp"
  would strengthen the abstract without length cost.

### B.2 Introduction (v4 §1)

- ✅ Already-addressed: 4-axis positioning table (prior audit treated this as
  positive evolution); stationary-operator formalisation paragraph present.
- 🔴 Block: the introduction's empirical evidence paragraph (L27 "정지 K를
  모든 레이어에 적용하면 MetaTool Subtask4에서 $-4.57$pp로 붕괴하지만, K를
  초기 1/4 레이어에만 두면 같은 기저로 $+2.08$pp가 회복된다") uses MetaTool
  numbers but does not cite the τ²-bench winner numbers that now matter most.
  Given that ladapt is the overall winner on retail/telecom per Table 1, the
  intro should name the τ² ladapt wins (+5.98 / +26.76 / +3.84).
- 🟡 Underdeveloped: v4 §1.3 has a worked-out two-step example ("Step 1: K' = K
  + α P_ont K → NewsTool (OK), Step 2: K' = K + α P_ont K → NewsTool 또 선택
  (wrong!)") that makes the history-free failure mode concrete. Korean intro
  formalises this but never shows the two-step worked example. Reader with
  weaker math background may struggle.

### B.3 Related work (Korean-only, no v4 analogue)

- 🟢 Section is stronger than v4 here; no deficit vs v4.

### B.4 Method (v4 §3 v5 + §2 U-shape)

- ✅ Already-addressed: three-operator definition, layer schedule formula,
  `install_layer_adaptive_hooks` referenced in appendix.
- 🔴 Block: no U-shape MSE plot or quantitative U-shape evidence. v4 §2 devotes
  a full ASCII figure + table to the U-shape. Korean paper references U-shape
  verbally ("레이어별 민감도의 비대칭") and defers the evidence to appendix
  `app:size-ushape` L89, which only gives a one-sentence pointer to
  `reports/layer_adaptive_2026_04_17/`. **The core empirical motivation of
  layer-adaptive (why L/4 specifically) has no evidence in the paper.** This
  lands the codex-critique (ii) directly.
- 🟡 Underdeveloped: method does not distinguish the two v4 variants
  (`layer_adaptive` vs `k_early_only`) — it describes `k_early_only` as if it
  were the layer_adaptive variant. v4 showed `k_early_only` F1=0.7514 beats
  `layer_adaptive` F1=0.7507 by 0.01pp. For NeurIPS reproducibility this
  nomenclature should be clarified; as-is, the "layer-adaptive K+Q" in the text
  collides with the `install_layer_adaptive_hooks` code label.

### B.5 Theory (v4 §6 math + §4.8.6 Thm β*)

- ✅ Already-addressed (C1, C2): Thm 3.3 Q-K duality + Cor 3.1 cache divergence
  landed. Thm 3.4 `thm:bound` exact integral remainder landed with proof.
- ✅ Already-addressed: Thm 3.5 `thm:beta-star` first-order sign predictor with
  full proof in appendix.
- 🔴 Block: `tab:sign-routing` at L87–95 is **internally inconsistent with the
  new Table 1**. It claims telecom predicted sign $+$ with empirical "+24.78pp"
  at $\beta=+0.05$, but Table 1 shows telecom $Q+0.05 = +22.32$pp and ladapt
  at +26.76pp as the overall best. Either (a) the number is stale (likely —
  "+24.78" appears nowhere in the new Table 1 rows), or (b) the theory-table's
  "실측 우세 방향" column needs re-wording. This is the Q2 flag — see Part C.
- 🟡 Underdeveloped: Thm β* is presented as domain-level "3/3 일치" but the
  appendix `app:G-sensitivity` transparently admits schema-G fails at telecom
  (31.2% locked). The main-text-appendix framing is consistent, but the theory
  section still reads as if Thm β* cleanly explains telecom-retail-metatool
  split. v4 §4.8.6 "update 3" has the stronger framing: "Thm β* 은 이론적
  framework으로 제시하되 per-query 예측 주장은 보류."

### B.6 Experiments §6 (v4 §4)

- ✅ Already-addressed: Table 1 expanded to four domains with locked F1 values;
  C3 retail action-count table; C4 banking scope exclusion statement; partial
  C5 multi-metric (τ²-retail, τ²-telecom).
- 🔴 Block: **Retail action-count decomposition contradicts Table 1's headline
  retail winner.** Table 1 shows retail ladapt +5.98 > Q-only +5.11; but
  `tab:retail-action-count` (L60–67) shows Q-only beats ladapt in every
  bucket (≤2: +5.82 vs +2.16; 3–5: +4.43 vs +0.82; ≥6: +6.39 vs +2.95).
  Action-count buckets covering n=42+62+10=114 (full sample) cannot sum to a
  ladapt-wins overall if Q-only wins every bucket. Either the decomposition
  numbers are stale (pre-ladapt-fix) or the headline +5.98 is computed on a
  different schema. Without reconciling, a reviewer will immediately catch
  this.
- 🔴 Block: L9 experimental-narrative sentence "retail과 telecom처럼
  long-horizon 또는 under-focused regime에서는 signed Q-only가 더 크다.
  특히 telecom에서는 $Q+$가 +24.78pp로 가장 강하다" is now **factually
  wrong** per Table 1 (ladapt wins both retail and telecom). See Part C.
- 🔴 Block: L50 "표~\ref{tab:main}가 직접 지지하는 결론은 두 개다. ...
  layer-adaptive는 short/medium-horizon 안정 해, signed Q-only는
  long-horizon·under-focused 체제의 regime-specific peak를 담당한다." This
  entire takeaway no longer follows from Table 1. Per Table 1 ladapt is the
  peak on retail/telecom/airline — the short/medium-horizon vs long-horizon
  split no longer carves the data.
- 🟡 Underdeveloped: v4 §4.6 multi-metric reinterpretation specifically for
  MetaTool ST4 (multipass_kq GT⊆P reversal) is still absent. Current
  `tab:multi-metric` covers only τ²-retail and τ²-telecom. Since MetaTool ST4
  is the primary N=497 benchmark, a one-row GT⊆P column there would close the
  "F1 gaming" concern on the primary benchmark too. (C5 partial.)
- 🟡 Underdeveloped: canonical SEKA row in Table 1 is still "\emph{placeholder}".
  The prior audit flagged this; no update. Having an unlabeled empty row in
  the main table is a reviewer red flag.
- 🟡 Underdeveloped: v4 §4.4 +68.5pp null-control gap framing is now only in
  §7 discussion (reversal-1 rebuttal, L5–22). The headline number does not
  enter the experiments section. Prior audit flagged D7; still open.
- 🟢 Intentionally dropped: v4 §4.5 v0–v5 evolution arrow and v4 §4.7
  hypotheses A–E for full-B_ont vs P_emitted — correctly dropped.

### B.7 Discussion §7 (v4 §4.8.5 + §5)

- ✅ Already-addressed: four-rebuttal structure (basis, perturb-magnitude,
  beta-sweep, Thm 3.4 bound); perturbation-magnitude table; SEKA Subtask1 +11.16pp
  counterexample anchoring "SEKA is good for single-tool, not multi-tool"; scope
  limitations paragraph.
- 🔴 Block: L3 opener "retail과 telecom에서는 signed Q-회전이 각각 $+5.11$pp와
  $+24.78$pp로 더 크게 이겼지만, 이 둘은 layer-adaptive와 대체 관계가 아니라
  같은 연산자족의 regime-의존적 이득이다" — same narrative inversion as §6
  L9/L50. Now directly contradicts Table 1 where ladapt wins both retail
  (+5.98) and telecom (+26.76). See Part C.
- 🟡 Underdeveloped: v4 §5 metric discussion (F1's ordering-blindness, the
  Facet-Weighted nDCG proposal) is not surfaced as a discussion limitation.
  The multi-metric table proves the point empirically but the text never
  acknowledges F1's ordering-blind limitation. Prior audit noted this as
  "fair to drop" — but since the paper now DOES report nDCG in
  `tab:multi-metric`, one sentence connecting "we report nDCG because F1 is
  ordering-blind" would close the loop.

### B.8 Conclusion §8

- 🔴 Block: L3 "retail과 telecom에서는 signed Q가 각각 $+5.11$pp와 $+24.78$pp로
  더 크게 이겨" — same narrative inversion. See Part C.

### B.9 Appendices §9

- ✅ Already-addressed (C1/C2): Thm 3.3 duality + Cor 3.1 cache divergence
  proofs landed.
- ✅ Already-addressed: G-sensitivity appendix (`app:G-sensitivity`) is the
  gold standard — transparent about schema-G failure at Telecom (31.2% locked)
  and pre-empts "why use this predictor at all."
- 🟡 Underdeveloped: `tab:layer-sweep-placeholder` (L66–80) still has 5 of 6
  rows as `\emph{placeholder}`. The core empirical motivation for L/4 boundary
  is still not landed. v4 §3 v5 states "k_early_only (L/4) +2.08 BEST" as a
  definite result; the Korean paper quietly puts it as "실행 대기."
- 🟡 Underdeveloped: `tab:e5-sizesweep` (L92–104) still has 3 of 4 rows
  placeholder. Only Qwen 7B locked. Size-scaling claim is unsupported.
- 🟡 Underdeveloped: SEKA canonical reproduction is still "A100 재현 gate"
  unresolved. This was flagged in prior audit; remains open.

---

## Part C — Q2. Table-1 narrative consistency

The ladapt bug-fix turned Qwen retail/telecom into ladapt wins. The
rebuttal-2 on all §4/§5/§6/§7 texts must be systematic. I enumerate every
passage that still frames signed-Q as the retail/telecom winner or ladapt as
secondary.

### C.1 Factual errors (must-fix for coherence)

| # | File | Line | Issue | Suggested reframe |
|---|------|------|-------|-------------------|
| X1 | `01_abstract.tex` | L2 | "retail에서는 음의 Q-회전이 $+5.11$pp, telecom에서는 양의 Q-회전이 $+24.78$pp로 가장 크고, airline에서는 layer-adaptive가 $+3.84$pp로 최고다" | Replace with ladapt-wins-all-three: "retail/telecom/airline 모두에서 layer-adaptive K+Q가 각각 $+5.98$/$+26.76$/$+3.84$pp로 최고이며, signed Q-회전은 telecom(+22.74pp)과 retail(+5.11pp)에서 ladapt에 근접한 second-best로 regime split을 설명한다." Or a concise variant: "세 τ² 도메인 모두 layer-adaptive K+Q가 이기고 ($+5.98$/$+26.76$/$+3.84$pp), signed Q-회전은 domain마다 부호를 바꾸며 ladapt에 근접한 regime-diagnostic 두 번째 해다." |
| X2 | `02_introduction.tex` | L27 | "K를 초기 1/4 레이어에만 두면 같은 기저로 $+2.08$pp가 회복된다" — only MetaTool cited, τ² wins absent | Add one clause: "MetaTool ST4에서 $+2.08$pp, τ²-bench 세 도메인에서도 $+3.84$–$+26.76$pp의 이득을 낸다." |
| X3 | `05_theory.tex` | L61 | "이 결과가 설명해야 할 경험적 사실은 분명하다. retail과 MetaTool에서는 $\beta<0$이 이기고 telecom에서는 $\beta>0$이 이긴다." — "이기다" is now false per Table 1 | Reword to "signed Q 중에서 retail/MetaTool은 $\beta<0$이, telecom은 $\beta>0$이 가장 크다" (keeps the Q-only sign split as a diagnostic without claiming signed Q wins overall). |
| X4 | `05_theory.tex` | L87–95 | `tab:sign-routing` telecom row "$\beta=+0.05$, $+24.78$pp" — this number does not appear in Table 1 (closest match: $Q+0.10 = +22.74$pp) | Replace with "$\beta=+0.10$, $+22.74$pp" (or "$\beta=+0.05$, $+22.32$pp" to keep same-β as predicted). Also clarify the "실측 우세 방향" column means "best signed-Q direction" NOT "best method overall." |
| X5 | `06_experiments.tex` | L9 | "retail과 telecom처럼 long-horizon 또는 under-focused regime에서는 signed Q-only가 더 크다. 특히 telecom에서는 $Q+$가 $+24.78$pp로 가장 강하다." — both clauses are now false | Reword: "ladapt는 MetaTool ST4를 제외한 세 τ² 도메인에서 최고이며, signed Q-only는 telecom $Q+0.10$ $+22.74$pp, retail $Q-0.03$ $+5.11$pp로 ladapt에 근접한 두 번째 해로 regime split을 설명한다. MetaTool에서는 Q-only $\beta=-0.03$ $+2.28$pp가 ladapt의 $+2.08$pp를 약간 앞서는 유일한 도메인이다." |
| X6 | `06_experiments.tex` | L50 | "layer-adaptive는 short/medium-horizon 안정 해, signed Q-only는 long-horizon·under-focused 체제의 regime-specific peak를 담당한다" — no longer supported by data | Reword: "layer-adaptive는 네 도메인 중 세에서 최고이며, signed Q-only는 같은 operator family의 single-axis 표상으로 telecom/retail에서 ladapt에 근접한 두 번째 해이자 MetaTool에서는 약간 우세한 단일 축 해다." |
| X7 | `06_experiments.tex` | L52–68 | `tab:retail-action-count` shows Q-only beats ladapt in every action bucket, yet headline retail result is ladapt +5.98 > Q-only +5.11. Numeric contradiction if buckets are n=42+62+10=114 covering full sample. | **Critical:** reconcile. Either the bucket numbers are from a pre-fix run (needs re-running on locked_v2 JSON) or the headline +5.98 must be re-derived. **Cannot ship as-is** — reviewer will spot this. |
| X8 | `06_experiments.tex` | L185 | efficiency section: "MetaTool $+2.08\sim 2.28$pp, Retail $+5.11$pp, Telecom $+24.78$pp, Airline $+3.84$pp" — retail/telecom use OLD Q-only numbers | Update to ladapt numbers: "MetaTool $+2.08\sim 2.28$pp, Retail $+5.98$pp, Telecom $+26.76$pp, Airline $+3.84$pp." |
| X9 | `07_discussion.tex` | L3 | "retail과 telecom에서는 signed Q-회전이 각각 $+5.11$pp와 $+24.78$pp로 더 크게 이겼지만" | Reword: "ladapt는 retail $+5.98$pp, telecom $+26.76$pp, airline $+3.84$pp로 세 τ² 도메인 모두에서 최고이며, signed Q-회전은 같은 연산자족의 single-axis 표상으로 regime-dependent 두 번째 해다." |
| X10 | `08_conclusion.tex` | L3 | "retail과 telecom에서는 signed Q가 각각 $+5.11$pp와 $+24.78$pp로 더 크게 이겨" | Reword: "layer-adaptive는 retail $+5.98$pp, telecom $+26.76$pp, airline $+3.84$pp로 세 τ² 도메인 모두에서 최고를 낸다. signed Q-회전은 telecom $Q+$, retail $Q-$, MetaTool $Q-$로 regime마다 부호를 달리하며 같은 operator family의 단일 축 표상으로 사용된다." |

### C.2 Non-errors (consistent with new narrative)

These passages are now correct with ladapt=winner framing and need no change:

- `02_introduction.tex` L25 (stationary K failure footprint) — correct.
- `04_method.tex` L1–16 — layer-adaptive already positioned as central method.
- `05_theory.tex` L107 "layer-adaptive K+Q와 signed Q-only는 ... 두 끝" — still
  correct after X3/X4 fixes.
- `06_experiments.tex` L95 Llama cross-model paragraph — correctly notes the
  Llama regime split differs.
- `06_experiments.tex` L135 "long-horizon 혹은 under-focused regime에서는
  signed Q가 직접적인 해법이 되고, boundary case나 short/medium-horizon에서는
  layer-adaptive K+Q가 더 안정적이다" — **actually also partially false** per
  Table 1 (ladapt wins telecom, which is under-focused), but the sentence's
  core claim "signed Q is a direct solution in some regimes" is defensible if
  "direct solution" is read as "single-axis operator." I'd suggest tightening
  to "sign 진단이 regime 해석을 제공하고, layer-adaptive는 세 τ² 도메인에서
  peak를 담당한다."
- `07_discussion.tex` L5–50 four rebuttals — internally consistent.
- `07_discussion.tex` L54 함의 paragraph — consistent.

### C.3 Independent Llama sign-consistency concern

`06_experiments.tex` L95 says "telecom에서는 Qwen과 같은 regime이 재현된다.
Llama telecom의 best Q-회전은 $\beta{=}{-}0.05$로 $+16.29$pp." But Qwen
telecom best Q-회전 is $\beta=+0.05$/+0.10 (positive) per Table 1 and §5
theory table. Llama telecom's best Q is $\beta=-0.05$ (negative). **These are
opposite signs**, not "same regime." The claim "same regime" can only hold if
by "regime" we mean "signed-Q beats ladapt" or "steering helps." This needs
re-wording: "Llama telecom의 signed Q는 Qwen과 반대 부호($\beta{=}{-}0.05$로
$+16.29$pp)이지만 ladapt가 $+11.62$pp로 steering-helps 결론은 재현된다."

Also L114 says "Q-회전 $\beta{=}{+}0.05$ & 0.0000 & $-38.45$ & $\beta{+}$
방향은 Llama에서 붕괴" — this is presented as a regime mismatch. Fair, but
should be stated in the narrative: "Llama telecom에서는 Qwen과 반대로
$\beta>0$이 붕괴한다."

---

## Part D — Overall verdict & must-fix list for Saturday

### Verdict

**The paper is NOT yet reviewer-ready.** The mathematics, related work,
positioning table, and G-sensitivity appendix are publication-quality. The
critical edits C1/C2/C3/C4 landed well. But:

1. Post-fix numeric narrative has not been pushed through abstract/intro/
   theory-table/experiments-narrative/discussion/conclusion. At least 10
   passages still cite stale signed-Q-wins-retail-and-telecom framing or the
   stale +24.78pp number. A sharp reviewer will catch the Table-1-vs-prose
   contradiction within the first read.
2. Retail action-count decomposition (C3 attempted fix) is numerically
   inconsistent with Table 1's retail headline. This must be reconciled before
   submission — a reviewer will catch this even faster than (1).
3. U-shape evidence and L/4 boundary ablation are both placeholder/deferred.
   Codex critique (ii) from the prior audit still lands here: "L/4 경계 선택은
   placeholder ablation에 의존한다."

### Must-fix before Saturday (ranked)

**Tier 1 (non-negotiable, factual errors):**

- **M1.** Sweep all 10 passages in §C.1 X1–X10 to align with the new Table 1.
  Roughly 1 hour of text edits. Half of them are one-sentence rewrites.
- **M2.** Reconcile `tab:retail-action-count` with headline retail ladapt
  +5.98pp. Either rerun the decomposition on the locked_v2 JSON that produces
  +5.98, or add an explicit note that the buckets are from a pre-fix run and
  the aggregate under the fix is +5.98. If the latter, the decomposition
  section will need to be reworded from "C3 empirical direct observation" to
  "pre-fix decomposition retained for mechanism illustration."
- **M3.** Fix `tab:sign-routing` (§5 theory) telecom row: replace stale +24.78
  with one of the Table-1 τ² telecom signed-Q numbers (+22.32 at β=+0.05 or
  +22.74 at β=+0.10), and clarify the column semantics ("signed-Q 내 최고
  방향" vs "overall best method").
- **M4.** Fix the Llama "same regime" claim (§C.3). Wording cost: 2 sentences.

**Tier 2 (strongly recommended, structural):**

- **M5.** Add MetaTool ST4 to `tab:multi-metric` with a GT⊆P column — closes
  the "F1 gaming" concern on the primary benchmark and completes C5.
- **M6.** Replace SEKA canonical `\emph{placeholder}` rows in Table 1 with
  either "n/a (reproduction in progress)" or delete those rows and move the
  SEKA comparison to a single-sentence appendix reference. Placeholders in the
  headline table are a visible weakness.
- **M7.** If U-shape MSE figure can be generated from
  `reports/layer_adaptive_2026_04_17/`, add as Figure 1 in §4 method. If not
  ready by deadline, add one quantitative sentence in method instead of the
  current verbal claim.

**Tier 3 (nice-to-have, style):**

- Prior PD/IW series from first audit — ~10 style edits if time permits.
- Efficiency-section numbers update (X8 — already in Tier 1).

### Estimated effort

- Tier 1 (M1–M4): ~2 hours text edits + 1 hour if action-count rerun is needed
- Tier 2 (M5–M7): ~2 hours if Qwen ST4 multi-metric numbers are already
  computed; otherwise 4 hours incl. rerun
- Tier 3: ~2 hours

Total minimum for reviewer-ready: **~5 hours focused edit pass**, achievable
before Saturday. The math/theory/appendix layers are solid. The gap is a
consistency sweep, not new content.

---

## Part E — Inventory summary

| v4 section | Korean paper status | severity |
|------------|--------------------|----------|
| §1 problem | ✅ covered in intro | — |
| §2 U-shape | 🟡 verbal only, figure deferred | major |
| §3 v0–v5 history | 🟢 correctly dropped | — |
| §4.1–4.5 MetaTool ST4 | ✅ covered in Table 1 + §7 rebuttals | — |
| §4.4 null-control +68.5pp | 🟡 in §7 only, not headline | major |
| §4.6 multi-metric | 🟡 τ² covered, MetaTool ST4 missing | major |
| §4.7 hypotheses A–E | 🟢 correctly dropped | — |
| §4.8.1 τ²-bench 4-domain | 🟡 3 of 4 domains covered; Banking scope exclusion landed (C4) | minor |
| §4.8.2 action-count decomposition | 🔴 landed but contradicts Table 1 headline | block |
| §4.8.3 Banking meta-tool analysis | 🟢 correctly scoped out | — |
| §4.8.4 scope redefinition | ✅ in discussion limitations | — |
| §4.8.5 Q-only pivot | 🔴 **reversed** — Korean paper pivots ladapt-winner; this is the new-narrative topic of Part C | (directional flip — architectural, not a gap) |
| §4.8.6 Thm β* | ✅ landed with G-sensitivity appendix | — |
| §5 metric discussion | 🟡 F1-limitation not explicitly flagged in discussion | minor |
| §6 math (Thm 6.1, 6.17', 6.20', Lemma 6.17.C) | ✅ covered (6.1 → 3.4; 6.17' folded into Thm 3.3 duality; 6.20' ε_q stopping correctly dropped) | — |
| §7 next experiments | 🟢 correctly dropped | — |
| §8 at-a-glance | partial — abstract version | — |
| Thm 7.1 duality | ✅ landed (C1) | — |
| Cor 7.1.B multi-step | ✅ landed (C2) | — |

---

## Appendix: quick grep targets for edit pass

```
grep -n '24.78' sections/*.tex             # 7 occurrences, all stale
grep -n '5.11' sections/*.tex              # 9 occurrences; retail Q-only, some stale
grep -n 'signed Q.*더 크' sections/*.tex   # §6 L9, §7 L3, §8 L3
grep -n 'best overall' sections/*.tex      # verify Table 1 marker consistency
grep -n 'placeholder' sections/*.tex       # 13 remaining placeholders
```

End of audit.
