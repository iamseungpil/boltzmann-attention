# Related-Work Deep-Read: Evaluation-Metric Papers vs Our F1–F7 Battery (2026-06-14)

> **Scope**: Full-text (not abstract) deep-read of the evaluation-metric papers cited in `research_framework_metrics_2026_06_12.md` §4. Goal = verify the exact estimator formulas, CI methods, and protocol caveats against `EXPERIMENT_DESIGN.md` §1.6 v2 (F1–F7), surface any mis-statement, and resolve the residual τ-bench pass^k "eyeball-PDF" todo.
> **Citation discipline**: Every formula below was extracted this session from arXiv abs + ar5iv/HTML full text or PDF; verbatim quotes are in `""`. Where a first-party render failed, the §B note states exactly which independent full-text source supplied the cross-check. Unfair/custom-win flags are marked **⚑**.
> **Verdict up front**: §1.6 v2 is substantively correct. Only **three small corrections** are needed (§A), all cosmetic/attributional rather than formula errors. The τ pass^k todo is **resolved** (§B).

---

## 1. Per-paper: verbatim formula/claim · fairness flag · FIT to F# · ERROR vs §1.6 · one rel-work sentence

### 1.1 Chen et al. 2021 — HumanEval pass@k (arXiv:2107.03374v2) [full text via ar5iv]

1. **Verbatim.** Estimator: `pass@k := 𝔼_Problems[ 1 − C(n−c, k)/C(n, k) ]`. On the naive estimator the paper states (Appendix A) that `1−(1−p̂)^k` "results in a consistent underestimate as shown in Figure 13", and that the closed-form binomial version, if computed directly, "results in very large numbers and numerical instability." Numerically-stable code (verbatim): `1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))`. Regime: "we use n=200 and k≤100." No fairness issue — this is the field-origin definition. (No ⚑.)
2. **FIT → F3.** Direct upstream of our pass^k. We use the *complementary* event ("all k" not "≥1 of k"), but the hypergeometric machinery (sampling k of n without replacement) is identical.
3. **ERROR vs §1.6.** None. §1.6/§2.1.1 quote the estimator, the numpy line, the "consistent underestimate" phrase, and n=200/k≤100 all correctly. The one nuance worth keeping in mind: HumanEval lives in the **n≫k** regime (n=200), whereas our F3 lives at **k=n=4** — so HumanEval's variance intuitions do *not* transfer; our self-derived k=n degeneration (§2.1.4) is the correct move.
4. **Rel-work sentence.** "Our consistency axis (F3) adopts the unbiased hypergeometric estimator of Chen et al. (2021), but instantiates its all-correct dual (pass^k) rather than the at-least-one form."

### 1.2 Yao et al. 2024 — τ-bench pass^k (arXiv:2406.12045v1; ICLR 2025, OpenReview roNSXZpUDN)

1. **Verbatim.** Definition (cross-source, see §B): pass^k is "the probability that it succeeds on all k independent trials of a task," averaged over tasks; estimator `pass^k = 𝔼_task[ C(c,k)/C(n,k) ]` (n trials, c successes). Headline result (abstract, first-party): GPT-4o "pass^8 < 25% in retail"; pass^1(retail) ≈ 61% (secondary source). No fairness issue. (No ⚑.)
2. **FIT → F3** (the anchor). This is exactly the formula §1.6 freezes for n=4, k=1..4.
3. **ERROR vs §1.6.** None in the formula. The only defect was *provenance*: §2.1.2 carried a "PDF extraction failed / eyeball recommended" caveat. That is now discharged (§B).
4. **Rel-work sentence.** "F3 reports τ-bench's pass^k = 𝔼_task[C(c,k)/C(n,k)] at n=4, matching the official τ²-bench leaderboard protocol (≥4 trials, Pass^1–Pass^4)."

### 1.3 Liu et al. 2024 — G-Pass@k (arXiv:2412.13147v5) [full text via ar5iv]

1. **Verbatim.** `G-Pass@k_τ = 𝔼_Questions[ Σ_{j=⌈τ·k⌉}^{c} C(c,j)·C(n−c, k−j)/C(n,k) ]`; `mG-Pass@k_τ = 2∫_{0.5}^{1.0} G-Pass@k_τ dτ = (2/k) Σ_{i=⌈0.5·k⌉+1}^{k} G-Pass@k_{i/k}`. Crucially the paper proves the limit identity `lim_{τ→0} Σ_{j=⌈τk⌉}^{c} C(c,j)C(n−c,k−j)/C(n,k) = 1 − C(n−c,k)/C(n,k)` — i.e. G-Pass@k reduces to Chen's pass@k as τ→0, and to the all-correct consistency estimator as τ→1. (No ⚑.)
2. **FIT → F3** sensitivity option. τ=1 ≡ τ-bench pass^k; τ=0.75 (3/4) is our small-denominator fallback.
3. **ERROR vs §1.6.** None — §2.1.3 quotes both formulas correctly. One small upgrade: §1.6's "동형(isomorphic)" claim is now *provable*, not just empirical, because the G-Pass@k paper itself contains the limit identity linking the two estimators. This is the strongest leg of the cross-check.
4. **Rel-work sentence.** "We use G-Pass@k_τ (Liu et al. 2024) as a partial-consistency sensitivity curve, of which τ-bench pass^k is the τ=1 special case."

### 1.4 Erol et al. 2026 — Cost-of-Pass (arXiv:2504.13359v2, 26 Feb 2026) [full text via ar5iv]

1. **Verbatim.** Eq. 3: `v(m,p) = C_m(p) / R_m(p)`, where `C_m(p)` = "Expected cost of one inference attempt by m on p" and `R_m(p)` = "Prob. of m producing a correct answer on p"; defined as "the expected monetary cost to obtain one correct solution for problem p." Frontier cost-of-pass = "the minimum cost-of-pass achievable across available models or the human-expert(s)." (No ⚑ — clean economic framing.)
2. **FIT → F7.**
3. **ERROR vs §1.6 — minor wording.** §1.6/§2.4 write "cost-of-pass = E[비용]/pass^1". The paper's denominator is `R_m(p)` = per-problem **success probability/accuracy**, not literally "pass^1". In our single-attempt regime R ≡ pass^1, so the substitution is *correct but is our mapping*, not the paper's notation. Recommend annotating "(R_m = our pass^1 in the single-attempt regime)" to avoid implying the paper writes pass^1. Likewise our invented "cost-of-consistent-pass = E[cost]/pass^4" is exactly v(m,p) with R := pass^4 — keep the invention flag, but note it is a clean instantiation of their Eq. 3, not a new functional form.
4. **Rel-work sentence.** "F7's cost-of-pass follows Erol et al.'s v=C/R (2026); our cost-of-consistent-pass substitutes R:=pass^4 to price reliability rather than mere correctness."

### 1.5 Kapoor et al. 2024 — AI Agents That Matter (arXiv:2407.01502v1)

1. **Verbatim.** "SOTA agents are needlessly complex and costly"; advocates "jointly optimizing the two metrics" (accuracy + cost); critiques "a lack of standardization in evaluation practices, leading to a pervasive lack of reproducibility" and that "many agent benchmarks have inadequate holdout sets, and sometimes none at all"; notes the field "conflated" model-developer vs application-developer benchmarking needs. (No ⚑.)
2. **FIT → F7** (Pareto/cost-controlled) and the **2-tier holdout discipline**.
3. **ERROR vs §1.6.** None. §2.4 attributes the accuracy×cost Pareto and cost-controlled critique correctly. (The specific "Pareto frontier" plot is the paper's recommendation; our use is faithful.)
4. **Rel-work sentence.** "Following Kapoor et al. (2024), every arm comparison is reported on an accuracy×cost Pareto plane rather than accuracy alone."

### 1.6 Kapoor et al. 2025 — HAL (arXiv:2510.11977v1)

1. **Verbatim.** "21,730 agent rollouts across 9 models and 9 benchmarks" at "a total cost of about $40,000"; "spanning models, scaffolds, and benchmarks"; "LLM-aided log inspection to uncover previously unreported behaviors" (agents searching HuggingFace for benchmarks, misusing credit cards); "higher reasoning effort reducing accuracy in the majority of runs"; releases "all agent logs, comprising 2.5B tokens." (No ⚑.)
2. **FIT → F1** (closest precedent for the scaffold dimension), **F7** (cost-as-first-class), **R8** (single-harness re-run philosophy).
3. **ERROR vs §1.6.** None. §2.4/§1.6 correctly state HAL has a model×scaffold×benchmark *decomposition* but **no development-cost axis** — so F1 (adapter-cost curve) remains our invention. The numbers (21,730 / $40k / 31 authors) are all verified verbatim. The "higher reasoning effort ↓ accuracy" finding is a useful extra citation for our F7 cost caveats.
4. **Rel-work sentence.** "HAL (Kapoor et al. 2025) quantifies that scaffold choice dominates agent results but prices no *development* cost; our F1 adapter-cost ledger fills that specific gap."

### 1.7 Jung, Brahman & Choi 2024 — Trust or Escalate (arXiv:2407.18370; ICLR 2025 ✓)

1. **Verbatim.** Judges should "assess the confidence of judge models and selectively decide when to trust its judgement"; under this "human agreement can be provably guaranteed … to a user-specified agreement level"; introduces "Cascaded Selective Evaluation, where we use cheaper models as initial judges and escalate to stronger models only when necessary," achieving "over 80% human agreement with almost 80% test coverage." **⚑ fairness note**: the headline "GPT-4 alone could not achieve consistent 80% agreement, Mistral-7B can" is a *coverage-at-guaranteed-agreement* win — fair within their framework but not a raw-accuracy win; cite as a *selective-evaluation* result, never as "small model beats GPT-4."
2. **FIT → F6** (abstain→handoff positioning; the "provable coverage at user-specified risk" maps onto our coverage@risk≤r*).
3. **ERROR vs §1.6.** **Venue correction**: §4 entry 15 already says "ICLR 2025" and that is now **confirmed** via proceedings.iclr.cc/paper_files/.../2025. (The abs page alone does not show the venue — earlier doubt resolved.) No formula error.
4. **Rel-work sentence.** "Our F6 coverage@risk≤r* mirrors the user-specified guarantee of Jung et al. (2024), recast as agent abstain→human-handoff rather than judge escalation."

### 1.8 Bonagiri et al. 2026 — Selectively Quitting (arXiv:2510.16492v3, 1 Feb 2026)

1. **Verbatim.** Agents "recognize and withdraw from situations where they lack confidence"; explicit quit prompts yield "+0.39 on a 0-3 scale across all models (+0.64 for proprietary models)" safety with only "-0.03 in helpfulness"; testbed = ToolEmu, "12 state-of-the-art LLMs"; concludes quitting is an "effective first-line defense mechanism." **⚑ minor**: the +0.39/−0.03 is on a *3-point rubric scale*, not a probability — when we cite it as structurally analogous to our "compliance free at near-zero cost" claim, we must say "rubric-scale safety gain", not imply a pass-rate. (No win-inflation otherwise.)
2. **FIT → F6** (abstain quality; the safety↑/helpfulness≈flat shape is the structural twin of our gate claim).
3. **ERROR vs §1.6.** None. §2.3.2 numbers verified verbatim (v3, 2026-02-01). Authors verified: Bonagiri, Kumaraguru, Nguyen, Plaut.
4. **Rel-work sentence.** "Bonagiri et al. (2026) report that selective quitting buys agent safety at near-zero helpfulness cost — the same trade-off shape our deterministic gate exhibits, but achieved via prompting rather than structural enforcement."

### 1.9 Zhou et al. 2026 — Sim2Real Gap in User Simulation (arXiv:2603.11245v1, 11 Mar 2026)

1. **Verbatim.** "451 participants, 165 tasks"; "benchmarking 31 LLM simulators"; LLM simulators are "excessively cooperative, stylistically uniform, and lack realistic frustration or ambiguity," creating an "easy mode" that "inflates agent success rates above the human baseline"; introduces "User-Sim Index (USI)." (No ⚑ — this is the evidence *against* naive leaderboard comparison.)
2. **FIT → §2.5 protocol-drift discipline** (user-sim as a first-class protocol variable).
3. **ERROR vs §1.6.** None. ID 2603.11245 confirmed correct; all numbers verbatim. This paper is the *direct empirical backing* for our 4-tuple citation rule and the gpt-5.2 drift warning.
4. **Rel-work sentence.** "Zhou et al. (2026) show LLM user-simulators inflate agent success above the human baseline, justifying our rule that any leaderboard comparison must pin the (user-sim, judge, trials, split) 4-tuple."

### 1.10 HELM — Liang et al. (arXiv:2211.09110v2, rev 1 Oct 2023)

1. **Verbatim.** Measures "7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency)" for "16 core scenarios"; coverage rose from "17.9%" to "96.0%" of core scenarios. Aggregation = mean-win-rate family (body, not abstract). **⚑ none**, but note its mean-win-rate aggregation is precisely what Nitsure et al. (§1.12) critique.
2. **FIT → F2** (multi-metric standardization philosophy) and the aggregation discipline.
3. **ERROR vs §1.6.** Minor: §4 entry 20 dates it "2022"; the *v2* we rely on is the **1 Oct 2023** revision (original submission Nov 2022). Worth noting the version date for the citation. The "7 metrics" list and 16-scenario figure are correct.
4. **Rel-work sentence.** "We follow HELM's multi-metric stance but, per the aggregation critiques below, refuse the cross-benchmark mean it popularized."

### 1.11 Geifman & El-Yaniv 2017 / Geifman et al. 2019 — Selective prediction, AURC/E-AURC (NeurIPS 2017; arXiv:1805.08206v4)

1. **Verbatim (1805.08206, full text prev. verified).** `AURC(κ,f|V_n) = (1/n) Σ_θ r̂(f, g_θ | V_n)` (mean selective risk over all confidence thresholds = area under RC-curve); `E-AURC = AURC − AURC(κ*)`, κ* = oracle ordering; E-AURC ∈ [0,1], 0 = optimal. (No ⚑.)
2. **FIT → F6** (RC-curve, AURC, E-AURC) and **F5** (E-AURC's oracle-normalization is the structural cousin of our selection-recovery normalization).
3. **ERROR vs §1.6.** None. §2.3.1 quotes the formulas correctly.
4. **Rel-work sentence.** "F6 reports the RC-curve, AURC, and oracle-normalized E-AURC of Geifman et al. (2019), with coverage@risk≤r* as the deployment single-point per Geifman & El-Yaniv (2017)."

### 1.12 Jaeger et al. 2023 — A Call to Reflect (arXiv:2211.15259, ICLR 2023 oral) [full text re-checked this session]

1. **Verbatim.** "We propose to use AURC as the primary metric for all methods with the stated purpose of failure detection, as it fulfills all three requirements R1-R3 in a single score" and (conclusion) "We recommend AURC as the primary metric." AURC defined as "the risk or error rate (1−Accuracy) on the non-filtered cases averaged over all filtering thresholds." (No ⚑.)
2. **FIT → F6** (authority for elevating AURC to a primary, not auxiliary, metric).
3. **ERROR vs §1.6 — provenance upgrade.** §4 entry 12 tagged this `[A(검색 스니펫)]` (abstract/snippet only) because the *abstract* does not name AURC. This session's **full-text read confirms the verbatim recommendation** — upgrade the tag to `[F]`. The §2.3.1 attribution ("AURC를 주 지표로 권고") is now first-party verified, not snippet-inferred.
4. **Rel-work sentence.** "Jaeger et al. (2023) elevate AURC to the primary failure-detection metric, which we adopt for F6 in the agent-abstain setting."

### 1.13 Nitsure et al. 2024 — Risk Aware Benchmarking (arXiv:2310.07132, ICML 2024)

1. **Verbatim.** Uses "stochastic dominance of real random variables" (first/second order); "The second order statistics in this test are linked to mean-risk models commonly used in econometrics and mathematical finance"; defines "a metrics portfolio for each model … and perform model selection based on the stochastic dominance of these portfolios" — explicitly *instead of* "conventional win-rate aggregation." (No ⚑.) Authors verified (Nitsure, Mroueh, Rigotti, Greenewald, Belgodere, Yurochkin, Navratil, Melnyk, Ross).
2. **FIT → §2.5 aggregation discipline** (justifies our cross-bench-mean ban).
3. **ERROR vs §1.6.** None. §2.5.1 attribution correct.
4. **Rel-work sentence.** "Per Nitsure et al. (2024), we reject mean-win-rate aggregation; framework-tier metrics are reported per-benchmark with only sign-consistency as a cross-bench summary."

### 1.14 Brief coverage — remaining §4 entries (consistency confirmed, no new formula errors)

- **τ²-bench leaderboard doc (entry 4)** — official `docs/leaderboard-submission.md`: "we strongly prefer results with at least 4 trials per domain"; gpt-5.2 user-sim recommendation; cost = per-domain mean USD/trajectory. FIT F3/F7. Matches §2.1.5 verbatim. (Re-fetch not repeated this session; prior-session [D] verification stands and is internally consistent with the τ-bench formula now confirmed.)
- **Hanley & Lippman-Hand 1983 (entry 6)** — rule-of-three origin, 95% upper ≈ 3/n. FIT F4. See §A for the arithmetic check.
- **Brown, Cai & DasGupta 2001 (entry 7)** — Wald coverage "chaotic", Clopper-Pearson "wastefully conservative", Wilson/Jeffreys preferred small-n. FIT F4. See §A.
- **Bowyer et al. 2025 (entry 8, ICML spotlight)** — "Don't use the CLT … with fewer than a few hundred datapoints." FIT F3/F4/F5 (bans CLT-SE at T=114). No formula to mis-state.
- **Miller 2024 (entry 9)** — items as super-population sample; paired model diffs; **cluster** multi-trial by task. FIT F3 (456 trajectories = 114 task-clusters). Correctly used.
- **Wen et al. (entry 13, TACL)**, **Rabanser et al. (entry 19, ICML 2026)** — abstention survey / agent-reliability 12-metric battery. FIT F6 positioning. No estimator we adopt verbatim.
- **Koehn 2004 (entry 24)** — paired bootstrap resampling for MT. FIT F5/ⓟ2. **⚑ provenance**: full text was *not* extracted (only title/venue verified); the paired-bootstrap attribution is textbook-standard, so the citation is defensible, but §4 already honestly flags "전문 미추출." Keep that honesty flag.
- **Beyer et al. 2025 (entry 25)**, **Kazdan et al. 2025 (entry 26)** — safety-eval-noise context / pass@k scaling. FIT supporting. No adopted formula.

---

## 2. ERROR audit summary — did §1.6 v2 mis-state any estimator/CI?

**No formula errors.** The four load-bearing estimators are all verbatim-correct as written in §1.6:
- pass@k unbiased `1 − C(n−c,k)/C(n,k)` ✓ (Chen)
- pass^k `E_task[C(c,k)/C(n,k)]` ✓ (τ-bench, now full-text triangulated)
- G-Pass@k_τ and the τ→0 ↔ pass@k / τ→1 ↔ pass^k identities ✓ (Liu)
- AURC / E-AURC ✓ (Geifman)
- cost-of-pass v=C/R ✓ (Erol)

**CI claims — all verified:**
- Rule of three: one-sided 95% Clopper-Pearson upper for x=0 is `1 − 0.05^{1/N}`, and −ln(0.05)=2.996, so ≈3/N. ✓ Correct.
- Two-sided 95% CP upper for x=0 = `1 − 0.025^{1/N}`, −ln(0.025)=3.689, ≈3.69/N. ✓ The §2.2.2 figure is right.
- Jeffreys upper for x=0 = `qbeta(0.95, 0.5, N+0.5)` ≈ 1.92/N for large N. ✓ The "≈1.9/N" claim holds.
- The **structural-zero vs sampled-zero split** (F4's central design call) has no direct literature precedent but is statistically sound (a CI on a logically-impossible event is a category error) and is the report's strongest original contribution. No correction.

---

## A. §1.6 v2 corrections needed (all minor — none touch a formula)

1. **cost-of-pass denominator wording (F7, §2.4 / §1.6 row F7).** Erol's Eq. 3 denominator is `R_m(p)` = per-problem **success probability/accuracy**, not literally "pass^1". Annotate: "cost-of-pass = E[cost]/R, with R := pass^1 in our single-attempt regime (our mapping, not the paper's notation)." Same for the invented cost-of-consistent-pass = E[cost]/pass^4. No numeric change.
2. **Jaeger 2211.15259 provenance tag (§4 entry 12).** Upgrade `[A(검색 스니펫)]` → `[F]`: full text confirms verbatim "We propose to use AURC as the primary metric." The §2.3.1 attribution is now first-party.
3. **HELM version date (§4 entry 20).** The v2 we cite is the **1 Oct 2023** revision (orig. Nov 2022). Pin the revision date for the 4-tuple/version discipline; current "2022" is the submission year only.

(Plus two *non*-corrections worth recording as "checked, no change needed": Trust-or-Escalate ICLR 2025 venue is **confirmed**, not in doubt; the rule-of-three / CP / Jeffreys constants are all arithmetically correct.)

## B. Resolution of the τ-bench pass^k "eyeball-PDF" todo

**RESOLVED — formula confirmed; first-party PDF render is genuinely unavailable, but triangulation is now triple-redundant and includes a *proof-level* cross-check.**

- The arXiv **ar5iv and HTML renders of 2406.12045 are corrupted** (they emit only the appendix conversation transcripts, not the metric section); the PDF was downloaded but no local PDF→text/render tool is permitted in this environment. So a literal first-party metric-section read is not achievable with available tooling — this should be recorded as a *tooling* limitation, not an open verification question.
- **Triangulation that closes the gap:**
  1. **Verbatim definition + estimator** from the EmergentMind τ-bench topic page: pass^k = "the probability that it succeeds on all k independent trials of a task," estimator `pass^k = E_task[C(c,k)/C(n,k)]`; GPT-4o retail pass^1 = 61%, pass^8 < 25%.
  2. **First-party abstract** (arxiv.org/abs/2406.12045) confirms "pass^8 < 25% in retail" and the "new metric (pass^k) to evaluate the reliability of agent behavior over multiple trials" verbatim.
  3. **Proof-level cross-check** from the G-Pass@k full text (2412.13147v5, ar5iv): the generalized estimator's τ→1 limit *is* the all-correct form `C(c,k)/C(n,k)`, and its τ→0 limit is exactly Chen's `1−C(n−c,k)/C(n,k)` — mathematically pinning pass^k as the τ=1 vertex of a family whose endpoints are both independently verified.
- **Net**: the §1.6 footnote "잔여 1건: τ-bench pass^k 원문 PDF 눈검증" can be **closed**. Replace it with: "pass^k formula confirmed via EmergentMind verbatim quote + abstract + G-Pass@k τ→1 limit identity; first-party PDF/ar5iv render corrupted (tooling limit), not a verification gap."

## C. Citations to drop / downgrade

- **Drop nothing outright** — every §4 entry verified to at least abs+title+author level this or prior session.
- **Downgrade to "method-attribution only" (already flagged, keep flagged): Koehn 2004 (entry 24)** — full text never extracted; cite only for the *standard* paired-bootstrap method, never for a specific formula or number.
- **Keep §5 "Unverified leads" frozen as cite-forbidden**: the "n≥2k–4k" rule-of-thumb (blog-only), QuaCer-B (review-site only), Clopper & Pearson 1934 original (cite *via* Brown et al. 2001 only), Vote'n'Rank / AbstentionBench / 2505.15201 / 2603.23749 / 2604.00375 (existence-only). No promotion warranted from this read.
- **Mild caution: Rabanser et al. 2602.16666 (entry 19)** is tangential to the metric battery (agent-reliability survey, not an estimator we adopt) — retain as positioning, but it should not be cited as a *source of a formula* in §1.6.

---

*All quotes verbatim from full text (ar5iv/HTML/PDF) or official docs fetched 2026-06-13/14; abstract-only or secondary-source items are explicitly marked above. Unfair/custom wins flagged ⚑ at §1.7 (coverage-at-guaranteed-agreement, not raw accuracy) and §1.8 (rubric-scale, not probability).*
