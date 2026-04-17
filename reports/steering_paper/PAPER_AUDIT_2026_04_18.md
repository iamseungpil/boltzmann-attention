# Paper Audit 2026-04-18

Audit of `paper/neurips2026_steering_ko/` against canonical reference
`origin/develop:math/paper/benchmark_design/PAPER_DRAFT_v4.md` (1062 lines).
Writing-style constraints from
`/home/v-seungplee/.claude/projects/-home-v-seungplee/memory/feedback_writing_style.md`
(no `\paragraph{}` leaders, no bullet lists in prose, implicit Intent→Hypothesis→Validation).
Locked `\textbf{0.XXXX}` values are frozen and never flagged for change.

---

## 1. Central-argument reading (≤150 words)

The Korean paper argues that stationary K-side steering (SEKA family) structurally fails at
multi-tool selection because the operator cannot encode emission history (Theorem 3.1), so its
ontology-aligned amplification turns into a `repeated_first_tool` bias. The authors propose
**layer-adaptive K+Q**: K confined to the first `L/4` layers to "imprint" the ontology
direction, Q-rotation on all layers to "cover" already-emitted facets. On top of this, they
frame **signed Q-rotation** as a regime diagnostic: the sign that helps a domain is the sign
of `\bar r_G - \bar r` (Theorem 3.3), explaining why retail/MetaTool prefer Q− while
telecom prefers Q+. The unifying claim is that layer-adaptive K+Q and signed Q-only are not
competing recipes but two regime-dependent members of the same operator family
`(α, β, ontology basis B)`. Method is training-free, ~2.6 MB per head, orthogonal to KV
quantisation.

---

## 2. Coverage gaps vs PAPER_DRAFT_v4

Legend: **block** = paper is misleading without this; **major** = weakens evaluation; **minor** = nice-to-have.

### 2.1 Present in v4 but MISSING or UNDERCOVERED in Korean paper

| Sev | v4 content | Korean paper status | Verdict |
|-----|-----------|---------------------|---------|
| block | v4 §4.8.5 "Q-only pivot": v4 re-frames the paper as *signed Q-only* primary, layer-adaptive secondary, citing **Thm 7.1 Q-K per-step duality + Cor 7.1.B multi-step divergence** (K-bias accumulates in KV-cache → long-horizon unstable). | Korean paper has the **opposite framing**: layer-adaptive = main method, signed Q = analysis tool. Neither Thm 7.1 (duality) nor Cor 7.1.B (cache-accumulation) appears. | **Restore as Q-K duality remark.** The central claim "같은 연산자족" is exactly what Thm 7.1 would anchor — without it, codex §5 critique ("same basis ≠ same operator family") lands. |
| block | v4 §4.8.2: Retail action-count decomposition — long-horizon (10+ actions) Q-only +10.7pp vs ladapt +1.3pp (**9× gap**). Mechanism = KV-cache accumulation. | Korean paper says "long-horizon 에서 signed Q가 더 직접적"(§exp:main) but gives NO numeric decomposition, no turn-length analysis. | **Restore the action-count table or cite the mechanism explicitly.** This is the load-bearing evidence for "regime-dependent, same family." |
| major | v4 §4.8.3: τ²-bench **Banking** full-domain fails at −5.99pp; after meta-tool filtering (n=13) recovers to +5.64pp with multipass ladapt. Scope exclusion is explicit. | Korean paper omits Banking entirely. §discussion limitations silent on meta-tool scope. | **Add one-sentence scope exclusion** ("meta-tool / policy routing out of scope; Banking full-domain not reported"). Without it, reviewer checks τ²-bench, sees 4 domains, asks why Banking is hidden. |
| major | v4 §4.6: multi-metric re-interpretation. **GT⊆Pred** metric flips ranking: multipass_kq is F1=4th but GT⊆P=**1st** (+4.02pp). "Recall-first vs Precision-first vs Balanced" method personality taxonomy. | Korean paper reports F1 only. No Exact, GT⊆P, avg_pred, or 3+pred counts. | **Restore the three-metric table (F1 / Exact / GT⊆P) for MetaTool ST4.** This is the cleanest defense against "F1 gaming" reviewer concern. |
| major | v4 §4.4 null-control: real-B +68.5pp gap vs random/shuffle with **α=0.3**. | Korean §discussion reports only K-side control (0.0 vs 0.685); omits the "+68.5pp directional specificity" framing. Also omits PCA-of-K as a pending control. | **Strengthen §discussion reversal-1 prose** — the +68.5pp gap at α=0.3 is the strongest single number for directional specificity. |
| major | v4 §5.3: Facet-Weighted nDCG metric proposal tied to ε_f measurements. | Absent. | Fair to drop for NeurIPS submission (proposed-only, no results), but discussion §limitations should acknowledge F1's ordering-blind limitation. |
| minor | v4 §3 hypotheses A–E explaining why full-B_ont Q beats P_emitted Q. | Absent. | Minor — the Korean paper doesn't attempt per-facet P_emitted, so not needed. |
| minor | v4 §8.2 "three key numbers" executive-summary framing (+2.08pp, +68.5pp, AUROC 0.976). | ε_q stopping / AUROC 0.976 entirely absent. | ε_q deploy predictor looks like a separate artifact; safe to omit for this paper's scope. |
| minor | v4 §6.3 ε_q stopping criterion AUROC 0.976. | Absent. | Same — out of this paper's scope. |

### 2.2 Present in Korean paper but NOT in v4 (evolution past v4)

| Korean §      | Content                                        | Status                   |
|----------------|------------------------------------------------|--------------------------|
| §theory Thm 3.4 (`thm:bound`) | Exact integral remainder bound for attention output; median LHS/RHS 2.36e-8 at Qwen L=13. | **Positive evolution** — tighter than v4's informal Thm 6.1; keep. |
| §appendices `app:G-sensitivity` | Three `G` variants (schema / logit-lens / generation-aggregate); schema-G fails telecom at 31.2% locked. | **Positive evolution** — v4 only hints at this; Korean paper's honesty about schema-G failure is exactly right. Keep. |
| §discussion Table `tab:e-perturb-mag` | Real-B perturbation **larger** than random/shuffle (621.3 vs 399.6 vs 292.0). | **Positive evolution** — directly rebuts "real-B works because it perturbs less." Keep. |
| §intro Table `tab:positioning` | 4-axis positioning table. | **Positive evolution** — v4 has this only as prose. Keep. |
| §appendices Prop `prop:ladapt-intuition` | Conditional role-separation under E1 (U-shape MSE). | Fine as conditional proposition; don't upgrade to theorem. |

### 2.3 v4 items deliberately dropped — "not important after all"

- v4 §3 v0..v5 history-style algorithm evolution table. Drop is correct — publication-quality paper shouldn't narrate its own git log.
- v4 §5.1–5.5 F1 metric discussion. Drop is correct — this is scratchpad-level, not paper material.
- v4 §4.7 hypotheses A–E for full-B vs P_emitted. Drop is correct — the Korean paper converged on full-B (B=BB⊤) and doesn't need to re-litigate.

---

## 3. Paper-digest violations

Principle: "never produce a flat 'we did A, B, C'. Deliver insight: what should the reader come away believing and why."

| # | Section | Line | Issue | One-sentence fix |
|---|---------|------|-------|------------------|
| PD1 | `02_introduction.tex` | L4 | Four-path taxonomy reads as enumeration ("네 갈래로 나뉜다"), not a lead. The insight (stationary ops can't represent emission state) is deferred to L25. | Move the one-big-idea ("static steering cannot encode multi-tool emission history") to the **first** sentence of §intro after the opener. |
| PD2 | `03_related_work.tex` | L5 | Paragraph catalogues four families but does not crystallise *what* the Korean paper adds beyond each. Reader finishes without a new belief. | End L5 with one sentence: "공통점은 정지성이고, 본 논문의 빈자리는 layer-schedule로 정지성을 푸는 것이다." |
| PD3 | `05_theory.tex` | L3 (section-opening) | Opens with "세 정리로 정당화한다" — a manifest, not an insight. | Replace with: "세 정리가 하나의 결론을 가리킨다 — 'K를 어디까지 허용할 것인가'가 연산자 설계의 중심 자유도다." |
| PD4 | `06_experiments.tex` | L3–L4 | §exp:main opens with "세 중심 주장을 검증한다" (manifest). | Replace with the already-present insight buried at L7: "핵심은 'K를 더 세게'가 아니라 'K를 어디까지'다." Hoist it to line 1. |
| PD5 | `07_discussion.tex` | L3 | Opens with a one-sentence summary that is actually good, but then L5–L24 becomes four rebuttals serialised as A→B→C→D. This is exactly the flat-list shape paper-digest warns against. | After L3, add a one-sentence bridge saying what the four rebuttals *collectively* prove ("gain이 basis 품질·perturb 크기·confidence boost·vacuous bound 중 어느 것으로도 환원되지 않는다"). Then the four paragraphs become evidence for that claim rather than a checklist. |
| PD6 | `01_abstract.tex` | L2 | Three domain numbers (retail +5.11, telecom +24.78, airline +3.84) reported flat. Reader doesn't know *why each is the number it is*. | Compress to one insight-carrying clause: "retail의 coverage regime에서는 Q−가, telecom의 under-focused regime에서는 Q+가, airline의 mixed boundary에서는 layer-adaptive가 이긴다 — 같은 기저 위에서 부호/스케줄만 바뀐다." |
| PD7 | `04_method.tex` | L13 | "본 방법론적 기여는 새로운 기저 자체가 아니라 연산자 배치에 있다" — this is the insight, but it appears in paragraph 4. | Move this sentence to L3 (section-opener) so §method begins with what it contributes. |
| PD8 | `05_theory.tex` | L81 "증명 사슬을 요약하면" | Summary list of which assumption each theorem uses. Useful information, but structured as an enumeration with no insight payload. | Collapse to one sentence: "네 정리는 모두 같은 연산자족 식 (2.1) 위에서 서로 다른 축을 진단한다 — history (Thm 3.1), sign (Thm 3.3), layer (Thm 3.4)." |

---

## 4. Iterative-writing violations

Style rules from `feedback_writing_style.md`: no bold `\paragraph{}` leaders, no bullet lists in prose, flowing topic-sentence transitions, no unsupported "따라서".

| # | Section | Line | Violation | One-sentence fix |
|---|---------|------|-----------|------------------|
| IW1 | `02_introduction.tex` | L31 | "본 논문의 기여는 네 가지로 요약된다. 첫째, ... 둘째, ... 셋째, ... 넷째, ..." — enumeration in prose. | Convert to three flowing sentences: the unfixable (Thm 3.1), the fix (layer schedule), the regime diagnostic (Thm 3.3). Drop "넷째" efficiency bullet into method §. |
| IW2 | `05_theory.tex` | L81 | "증명 사슬을 요약하면 다음과 같다" + list of what each theorem uses. Enumeration-style. | See PD8 — one sentence. |
| IW3 | `07_discussion.tex` | L3 L5 L23 L42 L49 | Opens each rebuttal with "첫 반론 / 둘째 반론 / 셋째 반론 / 넷째 반론" — mechanical connector pattern (NL5). | Vary the openers: "첫째로", "다음 반론은", "세 번째 가능성은", "마지막으로". Keep structure but break the `첫/둘/셋/넷` drumbeat. |
| IW4 | `06_experiments.tex` | L7 "가장 먼저 보이는 사실은 ..." and L9 "두 번째 관찰은 ..." | Same mechanical enumeration. | Use narrative transitions: "표 2의 첫 신호는 ... / 같은 표의 domain-split은 ..." |
| IW5 | `07_discussion.tex` | L54 "함의는 두 방향으로 확장된다. 하나는 ... 다른 하나는 ..." | Two-item enumeration. | Use one connected sentence: "정리 3.1이 모든 정지 연산자에 적용되므로 ITI/CAA/PASTA가 멀티-툴에 확장될 때 비슷한 반복 편향을 예측하며, 후속 연구는 layer-adaptive를 multi-turn stateful로 확장하거나 signed-Q 진단을 라벨-free proxy로 근사하는 두 방향에서 이어진다." |
| IW6 | `02_introduction.tex` | L25 "따라서" | The "따라서" is fine here — it follows from L20–24. No fix. | OK |
| IW7 | `05_theory.tex` | L23 "정리가 말하는 바는 분명하다. 멀티-툴에서 정지 K-only는 ..." | Implication sentence is good; "분명하다" is a mild overclaim (style rule 5). | "정리의 귀결은 명확하다" → replace with "결론은 단순하다" (avoid "명확" as intensifier). |
| IW8 | `06_experiments.tex` | L49 "표 2가 직접 지지하는 결론은 두 개다. SEKA류의 ... 따라서 논문의 메인 서사는 layer-adaptive이고, signed Q는 ..." | The "따라서" here is load-bearing but actually doesn't follow from the two-conclusion list — layer-adaptive wins only in airline. This is the codex critique. | Replace "논문의 메인 서사는 layer-adaptive이고" with a regime-split framing that does follow: "layer-adaptive는 short/medium-horizon 안정 해이고, signed Q는 regime-specific peak를 담당한다." Load-bearing. |
| IW9 | `04_method.tex` | L3 "먼저 기저와 세 연산자를 정의하고, 그 다음 레이어 스케줄의 직관과 비용을 서술한다." | Section-preview sentence — style rule discourages roadmap ("프리뷰") openers in favor of insight-first. | Delete. Section structure is readable without announcement. |
| IW10 | `09_appendices.tex` | L50 "세 가지 대안으로 정의해 robustness를 측정한다: (i) ... (ii) ... (iii) ..." | `(i)(ii)(iii)` inline enumeration in prose body. | Convert to connected sentence: "세 대안 — 도구 이름 단일 토큰, 스팬 평균, 정답 액션 스트링 전체 — 에서 진단 방향이 Airline을 제외한 세 도메인에서 유지되면 $\mathcal{G}$-민감도가 낮다는 의미다." |
| IW11 | `06_experiments.tex` | L49 "따라서 논문의 메인 서사는 ..." | Unsupported "따라서" — see IW8. | See IW8. |
| IW12 | `07_discussion.tex` | L42 "정리 3.3의 일차 진단도 전역 최대가 아니라 원점 근방의 개선 방향만을 예측한다는 점에서 이 좁은 구간 관측과 일관된다." | Connector "~다는 점에서 ... 일관된다" is translation-style (NL1). | "이 좁은 유효 구간은 정리 3.3이 원점 근방 부호만 예측한다는 해석과 부합한다." |
| IW13 | `06_experiments.tex` | L51 "우리는 이를 본문에서 *배포 알고리즘*이 아니라 *사후 해석용 진단*으로 사용한다. 이유는 명확하다." | "이유는 명확하다" — same overclaim pattern as IW7. | Delete "이유는 명확하다" and let the next sentence carry the justification directly. |

None of the files use `\paragraph{bold}` leaders — rule 1 is satisfied. Four `itemize` environments remain in `09_appendices.tex` section headers only, none in main body; rule 2 essentially satisfied.

---

## 5. Codex critique (verbatim)

GPT-5.4 via codex, read all 416 lines of concatenated sections, 300 words, Korean. Quoted as-delivered:

> 핵심 주장인 "멀티-툴에서 layer-adaptive K+Q와 signed Q는 대체재가 아니라 같은 연산자족의 regime-dependent 이득"은 흥미롭지만, 현재 원고는 이를 구조적으로 입증하지 못합니다.
>
> **(i) Unsupported claim**: 가장 큰 문제는 "같은 연산자족의 이득"이라는 해석이 실험적으로 증명되지 않았다는 점입니다. 본문은 두 방법이 같은 $P_{\mathrm{ont}}$를 공유한다는 사실을 제시하지만, 이것은 같은 기저를 쓴다는 뜻일 뿐 동일한 메커니즘적 가족이라는 결론으로 곧장 이어지지 않습니다. 실제 표는 오히려 signed Q-only가 MetaTool, retail, telecom에서 더 좋고, layer-adaptive는 airline에서만 최고임을 보여 줍니다. 이는 "비대체적 상보성"보다 "도메인별로 더 나은 별개 설계"로도 읽힙니다. 또한 정지 K 실패를 SEKA류 일반 한계로 확장하지만 canonical SEKA 수치가 placeholder이고 재현 실패를 스스로 인정해, 비교 기반 일반화가 성립하지 않습니다.
>
> **(ii) Evidence is thin**: layer-adaptive의 핵심인 $L/4$ 경계 선택은 아직 placeholder ablation에 의존합니다. 따라서 "초기 K, 후반 Q"가 원리인지 단지 한 하이퍼파라미터 우연인지 분리되지 않았습니다. `repeated_first_tool_rate` 역시 정량표가 아니라 화살표와 서술 중심이라, no-memory theorem의 관측적 footprint를 강하게 뒷받침하지 못합니다. signed Q 부호 진단도 offline, ground-truth-dependent proxy $\mathcal G$에 기대며, appendix에서 telecom에서 직접 실패한다고 적시합니다. 그럼에도 regime explanation의 핵심 지지축으로 사용됩니다.
>
> **(iii) Peer-review failure points**: placeholder 결과, 진행 중 ablation, 실패한 baseline 재현, 작은 일부 샘플 수(특히 airline N=50), 유의성/분산 보고 부재는 NeurIPS 심사에서 치명적입니다. 더 근본적으로, 이론은 "정지 K는 history-free"라는 자명한 성질과 "signed Q의 국소 도함수 부호"만 보일 뿐, 왜 layer-adaptive가 signed Q와 동일한 연산자적 원리로 묶여야 하는지까지는 논리적으로 닿지 않습니다. 현재 원고의 데이터는 "stationary K는 종종 나쁘고, Q-sign과 layer schedule이 과제별로 다르게 듣는다" 정도는 지지하지만, 제시된 중심 명제 전체를 통과시키기에는 증거 사슬이 아직 느슨합니다.

Three critique axes all converge on the same load-bearing gap: **the "same operator family" thesis is rhetorically asserted but not operator-level connected**. This is exactly the v4 content (Thm 7.1 Q-K per-step duality) that the Korean paper dropped — see §2.1 block-1 above.

---

## 6. Recommended edits, ranked

Format: `file | line-range | current (1-line abstract) | proposed (1-line abstract) | rationale`.

### Critical (do now — fixes block-level gaps)

| # | File | Lines | Current | Proposed | Rationale |
|---|------|-------|---------|----------|-----------|
| C1 | `sections/05_theory.tex` | insert after L33 (after Thm 3.2) | — | Add **Remark (Q-K per-step duality)**: under a single step, the effect of `K' = (I+αBB⊤)K` on logits equals the effect of `Q' = (I+αBB⊤)Q` (logits are bilinear in q,k; sharing BB⊤ makes α and β Q-side-of-K interchangeable at step T). | Restores v4 Thm 7.1 core content. Makes "same operator family" a derived fact, not a rhetorical parallel. Directly answers codex (i). |
| C2 | `sections/05_theory.tex` | insert after C1 | — | Add **Remark (multi-step cache accumulation)**: Thm 3.1 already forbids K from encoding history at step T+1. Under autoregressive decoding with KV-cache, K-bias persists across steps while Q-bias is recomputed; this asymmetry — same step-level identity, different multi-step stability — is why signed Q and layer-adaptive K+Q live in the same operator family but separate regime ends. | Delivers v4 Cor 7.1.B. Explains why duality at step T doesn't collapse into equivalence globally. Answers codex (i) structurally. |
| C3 | `sections/06_experiments.tex` | insert after L9 (after "telecom은 +24.78pp") | — | Add one-sentence action-count cite: "retail을 action 수로 분해하면 10+ 액션 구간에서 Q−가 +10.7pp, layer-adaptive는 +1.3pp로 9배 격차이며, K-bias의 KV-cache 누적(주석 C2)이 그 격차를 설명한다." | Restores v4 §4.8.2 evidence. Upgrades IW8's fix from rhetorical to empirical. |
| C4 | `sections/07_discussion.tex` | insert after L54 | — | Add one sentence: "Banking 도메인은 meta-tool 라우팅이 도구 선택이 아니라 정책 라우팅이므로 본 논문 범위 밖이며(§appendix), 이 scope 제한이 τ²-bench 3-도메인 보고의 근거다." | v4 §4.8.3 Banking exclusion needs explicit scope-statement so reviewers don't ask "why only 3 τ²-bench domains." |
| C5 | `sections/06_experiments.tex` | L46 table `tab:main` add one row | MetaTool ST4 section ends at `canonical SEKA` placeholder | Add columns **Exact**, **GT⊆P**, **avg_pred** to MetaTool ST4 rows (numbers exist in v4 §4.6). Keep the F1 column. | Restores multi-metric re-interpretation from v4. Frozen-bold preserved. Answers "F1 gaming" reviewer concern. |

### Desirable (do if time)

| # | File | Lines | Current | Proposed | Rationale |
|---|------|-------|---------|----------|-----------|
| D1 | `sections/06_experiments.tex` | L49 | "따라서 논문의 메인 서사는 layer-adaptive이고, signed Q는 왜 도메인마다 서로 다른 부호가 유리한지 설명하는 분석 축으로 배치하는 것이 자연스럽다." | "layer-adaptive는 short/medium-horizon 안정 해이고, signed Q는 regime-specific peak를 담당하며, 둘은 Q-K step-level duality(주석 C1)와 multi-step 누적 비대칭(주석 C2)에서 파생된 같은 연산자족의 두 끝이다." | See IW8/IW11. Load-bearing "따라서" replaced with operator-family argument. |
| D2 | `sections/02_introduction.tex` | L31 | "본 논문의 기여는 네 가지로 요약된다. 첫째 ... 넷째 ..." | Flowing three-sentence prose. See IW1. | Style rule 2 — avoid 나열식. |
| D3 | `sections/05_theory.tex` | L3 | "이 절은 본 논문의 layer-adaptive 설계를 세 정리로 정당화한다." | "네 정리가 한 결론을 가리킨다 — 'K를 어디까지 허용할 것인가'가 연산자 설계의 중심 자유도다." (네 정리 = add duality remark as lemma 3.1.5) | PD3 — insight-first section opener. |
| D4 | `sections/04_method.tex` | L3 | Section opens with "먼저 기저와 세 연산자를 정의하고..." | Section opens with "본 방법론적 기여는 새로운 기저 자체가 아니라 연산자 배치에 있다." | PD7 — insight-first. |
| D5 | `sections/07_discussion.tex` | L5 L23 L42 L49 | "첫 반론 / 둘째 반론 / 셋째 반론 / 넷째 반론" | Vary openers. See IW3. | NL5 mechanical pattern. |
| D6 | `sections/06_experiments.tex` | L68 footnote | "정리 3.1의 관측 가능한 예측은 정지 key-side에서 repeated_first_tool_rate이 no_steer보다 올라간다는 것이다." | Replace qualitative arrow table `tab:e3-mechanism` with locked numeric row ("`no_steer`: 0.XX, SEKA amp=1.0: 0.YY, 정지 K a=0.3: 0.ZZ, Q−: 0.WW, layer-adaptive: 0.VV"). If numbers not available, mark table-wide with `(N=497 full, numbers pending)` instead of per-cell arrows. | Codex (ii) lands directly on the arrow-style table. Quantifying this turns Thm 3.1 from "axiomatic" to "empirically footprinted." |
| D7 | `sections/07_discussion.tex` | L24 (after perturb-mag table) | Reversal-2 para ends without a quantitative hook to "+68.5pp." | Add one sentence: "real-B와 null의 F1 간격은 α=0.3에서 +68.5pp (real 0.685 vs null 0.000)로, '약해서 이긴다' 해석을 더 크게 벌린다." | §2.1 major-5 gap. Strengthens directional-specificity story. |

### Optional (note for later)

| # | File | Lines | Current | Proposed | Rationale |
|---|------|-------|---------|----------|-----------|
| O1 | `sections/01_abstract.tex` | L2 | Three domain numbers as flat list | Insight-carrying clause, see PD6 | Abstract reads as benchmark-sweep without PD6 fix. |
| O2 | `sections/09_appendices.tex` | L50 | "(i) (ii) (iii)" enumeration | Connected sentence, see IW10. | Style rule 2. |
| O3 | `sections/05_theory.tex` | L81 | "증명 사슬을 요약하면 다음과 같다" + list | One-sentence collapse, see PD8. | Style rule 2. |
| O4 | `sections/09_appendices.tex` | L52–73 `app:layer-sweep` | Placeholder table `tab:layer-sweep-placeholder` | Either fill (ablation running) or defer to supplementary. Current state puts "실행 대기" in the submission. | Codex (ii) — placeholder L/4 ablation is a NeurIPS-level weakness. If numbers won't land before submission, remove the table and state "layer-boundary ablation in supplementary." |
| O5 | `sections/09_appendices.tex` | L83–98 `tab:e5-sizesweep` | Three of four rows placeholder | Remove placeholder rows, report only 7B locked (0.7307, 0.7535, 0.7514) + single-sentence "1.5B/3B/14B 보조 스윕은 supplementary." | Same reasoning as O4 — placeholder in main appendix weakens submission. |
| O6 | `sections/05_theory.tex` | L23 L50 | "따라서 … 한다" / "따라서 … 일치한다" | Vary logical connectors ("그러므로", "이로부터") | NL5. |
| O7 | Throughout | — | Running-text `\emph{layer-adaptive K+Q steering}` | Define once in §intro, use `layer-adaptive K+Q` (without `\emph`) afterwards. | Micro-style; `\emph` at every mention looks nervous. |

---

## 7. Scope-of-change summary

The Korean paper is **structurally close to publication-ready** — the mathematics is tight, null-controls are honest, and the appendix G-sensitivity section is exemplary in acknowledging schema-G failure at telecom. Two classes of change matter most:

First, the central-argument phrase "같은 연산자족" currently floats. Critical edits C1–C2 restore Thm 7.1 + Cor 7.1.B from v4, which is where "same family" becomes a derived theorem rather than an assertion. Without this, codex-style reviewers will land on "same basis ≠ same family" and the paper's unifying pitch collapses.

Second, several load-bearing evidences were compressed out of the Korean paper: the action-count decomposition (C3), multi-metric GT⊆P re-interpretation (C5), Banking scope exclusion (C4), and real-B's +68.5pp null gap (D7). These exist verbatim in v4 §§4.4/4.6/4.8.2/4.8.3 and should be re-inserted as load-bearing numbers, not prose.

The style violations (PD/IW series) are all easily fixable and make the text read as insight-first prose rather than bulleted enumeration, in line with `feedback_writing_style.md`.

Locked `\textbf{0.XXXX}` values untouched. Auditor applied no edits.
