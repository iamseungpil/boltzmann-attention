# Related work — load / working-memory / CoT-as-load-reduction (2026-07-05)

Source: deep-research DR1 (external-WM ↔ scale; 24/25 claims verified 3-0, 1 killed)
+ close reading of the two anchor papers (Abbe, Mirtaheri) and the complexity
hierarchy (Merrill & Sabharwal). For paper positioning: our thesis = a
*deterministic, structured* scaffold is a load-reduction operator that lets a
small tool-use agent match a larger one. DR2 (CoT-capability) and DR3
(test-time-compute / self-consistency vs scale) pending.

## 1. Established theory we USE (premise, not our claim)

CoT / scratchpad tokens = a serialized external working-memory / computation
register. Step-count sets computational capacity.

- **Merrill & Sabharwal 2024, "The Expressive Power of Transformers with CoT"
  (arXiv 2310.07923, ICLR).** Without CoT, decoder transformers are TC0-bounded
  and *provably cannot* do simple serial tasks (graph connectivity, finite-state
  simulation). With intermediate steps: **log** steps ≈ no gain, **linear** steps
  → recognize all regular languages, **polynomial** steps → *exactly* P (first
  exact class characterization). Hierarchy log<linear<poly is strict under
  standard conjectures (TC0≠NC1, L≠NL); needs projected pre-norm + log-precision.
- **Li et al. 2024, "CoT Empowers Transformers to Solve Inherently Serial
  Problems" (arXiv 2402.12875).** T intermediate steps → express any size-T
  circuit. (Already in our bib.)
- **Nye et al. 2021, "Show Your Work: Scratchpads" (arXiv 2112.00114).** Same
  model fails direct / succeeds step-by-step on addition & program execution.
  (Same-model improvement, not scale-substitution.)

## 2. ★ANCHOR + FOIL — Abbe et al., globality barrier & inductive scratchpad

**"How Far Can Transformers Reason? The Globality Barrier and Inductive
Scratchpad" (arXiv 2406.06467, NeurIPS 2024).** Close reading:

- **Globality degree (Def 2):** smallest k such that some size-k subset of input
  variables has non-trivial (n^{-O(1)}) mutual information with the label. High
  globality ⇒ you must attend to many tokens *jointly* to correlate with the
  answer.
- **Barrier (Thm 1):** high-globality distributions **cannot be weakly learned**
  by regular Transformers under poly-time GD. This is a **LEARNABILITY** limit,
  **independent of scale/depth** — NOT an expressivity limit. ("syllogisms cannot
  be composed on long chains.")
- **Three scratchpads:**
  - *Agnostic* (extra tokens, no step supervision) — **cannot** break the barrier
    (Thm 2).
  - *Educated* (supervised intermediate targets designed so each step
    (X,Y_<i)→Y_i has **low globality**) — **can** break it, but may fail OOD.
  - *Inductive* (Def 5; a state-transition g with s[i]=g(Q,s[i-1]), attention
    masking so each step sees only Q + previous state) — breaks the barrier **and**
    generalizes OOD (parity 30→50-55 bits; addition 4-10→20-26 digits, ~6×).
- **★The load-bearing point for us:** the barrier is broken by **STRUCTURE, not
  depth/params**; and that structure is **task-specific and human-designed** —
  "Human prior knowledge is required to identify the induction function g and
  state structure." The paper explicitly flags wanting a **"more 'generic'
  scratchpad"** but **offers none**. Validated only on synthetic tasks (cycle,
  parity, addition).

**How this positions our paper:**
- ANCHOR: proves *structured* externalization (not scale) is what breaks a
  scale/depth-independent barrier → grounds "deterministic scaffold = load-reduction
  operator, not arbitrary reasoning tokens."
- FOIL / whitespace: Abbe's structure is (a) *learned* (supervised intermediate
  targets), (b) *task-specific & human-designed per task*, (c) *synthetic tasks*.
  **Ours is (a) deterministic/external (the engine produces the structure, the
  model need not learn to), (b) domain-general (A2/gate_spec, transfers by
  A2-swap) — their explicit open "generic scratchpad" problem, (c) a real tool-use
  agent.** Abbe is the closest theoretical precedent AND the gap we fill.

## 3. ★Mirtaheri et al. — serial CoT vs parallel voting (self-consistency limit)

**"Let Me Think! A Long CoT Can Be Worth Exponentially Many Short Ones"
(arXiv 2505.21825, NeurIPS 2025).** Close reading:

- **Definitions:** *parallel scaling* = independent samples aggregated by
  **best-of-N / majority voting (self-consistency)**; *sequential scaling* = one
  long CoT.
- **Thm 1 (expressivity):** poly-length CoT implements any poly-time algorithm
  (incl. BFS/connectivity). **O(1)-length CoT ∈ TC0, and *aggregating many
  independent CoTs is STILL a TC0 circuit* → cannot solve (s,t)-connectivity.**
  ⇒ **parallel voting cannot add serial-computation depth.**
- **Thm 3 (bridge graphs):** majority vote needs **exp(Ω(d))** runs vs sequential
  **O(L)** — exponential separation. VQM: Ω(L) neighborhood queries needed for
  L-hop connectivity.
- **Caveats:** holds for *multi-hop reasoning*; "once sequential scale is large
  enough, parallel becomes more cost-effective (diminishing returns)." VQM is a
  heuristic abstraction.

**How this positions our paper (self-consistency / random-vs-systematic):**
- Backs the theoretical claim that **repetition/voting cannot substitute for
  serial depth** — one half of our "random vs systematic error" framing: for
  *inherently serial* residuals, voting is stuck in TC0.
- Complement (our angle): our comparison/selection residuals (max-of-N, ⋈) are
  themselves **IN TC0** — so they are NOT the serial-depth case; the open question
  is whether voting reduces the *execution noise* of a TC0 op (random error) vs a
  systematic bias. Mirtaheri bounds the serial side; our self-consistency probe
  will measure the TC0-reliability side.

## 4. External-memory systems (context/state, not parameter-scale)

- **MemGPT (2310.08560):** OS-tiered memory; exceeds context window (baselines 0%
  at nesting limits, MemGPT unaffected). For context, not scale.
- **Self-Notes (Lanchantin 2305.00833, NeurIPS 2023):** interleaved notes as
  working memory; OOD state-tracking 85.0% vs vanilla 24.4% vs scratchpad 11.6%
  (scratchpad's drop partly a GPT-2 context-window artifact). Requires supervised
  note-taking. Synthetic, small models.
- **Generative Agents (2304.03442):** external NL memory stream. NOTE: the
  memory-ablation "each component critically helps" claim was **REFUTED (0-3)** in
  DR1 verification — cite with care.
- **Retrieval (RETRO):** external store substitutes for *parameter scale* — but
  that is **knowledge**, not working-memory/reasoning; distinct from our claim.

## 5. Empirical WM-capacity limits vs scale

- **"Unable to Forget: Proactive Interference..." (2506.08184):** WM limits probed
  0.6B→600B+; **log-linear decline** in retrieval under interference across scale
  (i.e., a capacity axis distinct from context length). Supports "load evolves
  with scale."
- **CoThinker (2506.06843):** operationalizes Cognitive Load Theory for LLMs
  (bounded WM analogue; "context rot"); load-reduction via multi-agent
  coordination. Single preprint; medium confidence.
- **"Cognitive Load Limits in LLMs" (2509.19517):** monotone accuracy drop under
  extraneous load, but only on capable models (single-author preprint, n=200).

## 6. Whitespace (DR1 open questions) — our contribution

1. **No published work shows a SMALL model + external/structured WM MATCHING a
   LARGER model's parameter count on agentic multi-step tool-use tasks, holding
   the scaffold fixed** — the exact **iso-scaffold × cross-scale** comparison our
   paper runs (14B/32B, same scaffold). CONFIRMED absent.
2. Disentangling the scaffold's benefit into expressivity vs learnability
   (globality) vs WM-capacity — on *agent* tasks, not synthetic.
3. Mapping Abbe's "educated/inductive structure" onto a **domain-general**
   deterministic tool-use scaffold (their open "generic scratchpad").
4. Whether the load-degradation curves shrink with scale AND flatten under our
   scaffold (the crossover) — not established by current single-model evidence.

## 6b. DR2 (CoT-capability for small models) — salvaged (synth agent glitched; read directly)

DR2's search/verify worked (24 sources, 24 confirmed) but the final synthesis
returned placeholder text; recovered by reading the decisive sources.

- **★Sprague et al., "To CoT or not to CoT? CoT helps mainly on math and
  symbolic reasoning" (arXiv 2409.12183).** Meta-analysis of 100+ papers: **CoT
  gives strong gains PRIMARILY on math/logic; minimal elsewhere.** On MMLU,
  "directly generating the answer without CoT is almost identical accuracy unless
  the question involves symbolic operations." **Decisive:** "much of CoT's gain
  comes from improving symbolic execution, but it UNDERPERFORMS relative to using
  a symbolic SOLVER."
  → For us: (a) predicts our probe split — arithmetic max-of-N (symbolic → CoT
  should help) vs semantic ⋈ order-inference (non-symbolic → CoT should not);
  (b) **for the symbolic slice, a deterministic SOLVER beats CoT** — reinforces
  "deterministic scaffold (calc/gate), not learned CoT" for computation residuals.
- **REFUTED (0-3):** the claim that distillation *systematically* transfers a
  larger model's reasoning to a small one (distill > human-data), sourced to
  DeepSeek-R1 (2501.12948), did NOT survive verification. ⇒ "train the small model
  to serialize (CoT distillation)" is NOT a settled win — supports treating the
  learn-lever as an open question, not an assumed fix.
- **RIVAL — AgenticQwen (arXiv 2604.21590, 2026-04, post-cutoff):** small agentic
  models match larger via multi-round RL on synthetic data + explicit
  decision-tree structures. Closest "small-agentic == large" precedent, but a
  DIFFERENT mechanism: learned behavior-trees + domain-target RL, NOT a
  deterministic domain-general scaffold, and NO compliance / iso-scaffold ×
  cross-scale axis. Position against (rival for the headline, not our method).
- Training-to-reason precedents (for the learn-lever discussion): STaR
  (2203.14465), Distilling Step-by-Step (2212.08410). ReAct (2210.03629) =
  reason+act over tool state (prompted, large models).

## 6c. Our empirical probe (14B, one-shot vs CoT, load-vs-capability)

Isolated single-turn probes at 14B (Qwen2.5-14B-Instruct), temperature 0,
gpt-4.1 = 0. CoT budget 900 tok, truncation tracked (=0 for all rows below).
32B reference (make-or-break): variant GIVEN-SPEC 100%, GOAL 70%, genuine-⋈ 54%.

| primitive | one-shot | CoT | Δ | note |
|---|---|---|---|---|
| max-of-N (parallel arithmetic compare) | 77% | 93% | +17% | symbolic |
| joint-constraint (filter AND max) | 50% | 85% | +35% | symbolic (hardest) |
| operation-select (intent → tool) | 68% | 72% | +4% | semantic |

**★Key result & a corrected earlier error ([[08]]):** an earlier run reported CoT
Δ≈0 / −27% and nearly concluded "CoT cannot recover parallel comparison." Reading
the raw responses showed that was a **truncation artifact** (400-tok CoT budget cut
off many-candidate scans before the answer). With adequate budget (trunc=0), **CoT
strongly recovers the symbolic-comparison residuals** (+17%, +35%).

**Interpretation (theory-consistent):**
- The "serial vs parallel" framing I first drew was wrong; the user's point holds
  — parallel comparison IS serializable (running-max), and CoT executes it: it
  recovers max-of-N and joint-constraint.
- The real split is **SYMBOLIC vs SEMANTIC** (matches Sprague 2409.12183): CoT
  helps the symbolic/arithmetic primitives (+17/+35) and barely helps the semantic
  intent primitive (+4). Consistent with the earlier ⋈ (order-*matching*, semantic)
  not improving under CoT.
- Combined with Sprague's "symbolic SOLVER > CoT": for the symbolic slice, our
  deterministic scaffold (calc/gate) should beat even trained CoT; for the semantic
  slice (intent, ⋈ matching), neither CoT nor a solver helps — that is the
  scale/capability residual.

**Caveats:** probes are variant-level (within product), not the live cross-order ⋈;
a clean adequate-budget re-run of ⋈ order-inference is still pending. Tool:
`cot_probe5.py` on the remote.

### Self-consistency probe (does "repetition" overcome the error? random vs systematic)

Single one-shot (temp 0) vs K=5 majority vote (temp 0.8), 14B, `cot_probe6.py`:

| primitive | single | maj@5 | Δ |
|---|---|---|---|
| max-of-N | 77% | 77% | +0% |
| joint-constraint | 55% | 55% | +0% |
| operation-select | 68% | 68% | +0% |

**Self-consistency gives ZERO gain on all three.** A diversity check (8 samples
@temp 0.8) confirms this is NOT a no-sampling artifact: the model's output
distribution is **near-deterministic even at temp 0.8** — e.g., Laptop max-price:
all 8 samples pick the SAME WRONG id ($2729.32 over gold $2749.56), 8/8;
T-Shirt/Shoes: 8/8 correct. So the execution error is **systematic AND
high-confidence (≈zero sampling entropy)**, not stochastic.

**Conclusion (random-vs-systematic, empirically grounded):** repetition / majority
voting / self-consistency does NOT overcome the selection error — the model
reliably reproduces the same wrong answer. This matches Mirtaheri 2505.21825
(parallel voting stays in TC0, cannot add the missing computation). What DOES
help is (a) scope reduction (scaffold: GIVEN-SPEC 100% vs GOAL 67%) and (b)
serialization/CoT for the SYMBOLIC slice (probe5) — because CoT *changes the
computation* (explicit list-then-compare), not by resampling. The semantic slice
(intent, ⋈-matching) is recovered by none of these → the scale/capability residual.

## 6d. DR3 (test-time compute / self-consistency vs scale) — corroborates probe6

Verified report (24 confirmed, several refuted). Clean random-vs-systematic split
on TC0 selection/comparison, directly backing our probes.

**Random/stochastic error — recoverable by test-time compute (bounded):**
- **Snell et al. 2408.03314:** compute-optimal test-time allocation lets a small
  model beat a **14× larger** one (FLOPs-matched, >4× more efficient than raw
  BoN) — **but only on easy/moderate problems, low inference/pretrain ratio;
  hardest problems need scale.** = the execution-reliability (recoverable) vs
  capability (needs scale) boundary.
- **Large Language Monkeys, Brown et al. 2407.21787:** coverage scales over 4
  orders of magnitude — but converts to accuracy **only with a verifier**; without
  one, voting/RM **plateaus** (MATH/Llama-3-8B: coverage 82.9→98.4% but voting
  accuracy 40.5→41.4% flat over 100→10⁴ samples). ← our probe6 flat, at scale.

**Systematic error — repetition/voting CANNOT fix (our probe6, corroborated):**
- **Byerly & Khashabi, TACL 2026 (2411.01101)** [peer-reviewed, strongest]:
  self-consistency **actively DEGRADES** on long-context because systematic
  (position) bias → **correlated errors** → violates SC's independence; worse with
  **smaller models**. ← exactly our probe6 (correlated, voting fails).
- **Apple 2026, "Nine Judges, Two Effective Votes" (2605.29800):** 9 judges =
  **2.18 effective votes** (correlated errors on same items); "bottleneck is
  correlated judges, not aggregation." ← literature form of our 8/8-same-wrong.
- **Best-of-Majority, Di et al. 2510.03199:** majority voting has Ω(1) regret
  (plateau); BoN can degrade with N.
- **Zheng et al., ICLR 2024 (2309.03882):** LLM selection has systematic
  **selection bias** toward option IDs (permuting options swings accuracy ±15pts).
  **PriDe** corrects it **label-free at ~1.15× cost, no scale** (RStd 8.7→1.8,
  +2.6 acc). ← systematic error is fixable by *targeted deterministic correction*,
  NOT only scale — a precedent for "scaffold, not scale/voting."

**Net (DR3):** "reduce scope + repeat + aggregate" recovers the **stochastic**
component and can substitute for scale in a bounded easy-moderate regime, but a
**systematic (bias/correlated-error) residual persists that only scale/training —
or a targeted deterministic correction/verifier (= our scaffold) — removes.**
Refuted (do NOT use): self-certainty BoN making small match large (0-3).

**Whitespace (DR3 open questions) = our contribution:** no source studies the
exact "select 1-of-N TC0 already-given" task (our probes do); none combines
scope-reduction + debias + aggregation end-to-end to test small==large on
selection (our scaffold does); a per-instance stochastic-vs-systematic router is
open.

## 7. Candidate bib entries (verify before citing)

- 2310.07923 — Merrill & Sabharwal, Expressive Power of Transformers with CoT, ICLR 2024
- 2406.06467 — Abbe et al., Globality Barrier & Inductive Scratchpad, NeurIPS 2024
- 2505.21825 — Mirtaheri et al., Long CoT worth exponentially many short, NeurIPS 2025
- 2112.00114 — Nye et al., Show Your Work: Scratchpads, ICLR 2022
- 2305.00833 — Lanchantin et al., Learning to Reason and Memorize with Self-Notes, NeurIPS 2023
- 2310.08560 — Packer et al., MemGPT
- 2304.03442 — Park et al., Generative Agents (memory-ablation claim refuted — cite carefully)
- 2506.08184 — Unable to Forget: Proactive Interference reveals WM limits
- 2506.06843 — CoThinker (CLT for LLMs)
- 2509.19517 — Cognitive Load Limits in LLMs (weak; single-author)
- 2409.12183 — Sprague et al., To CoT or not to CoT? (CoT helps mainly math/symbolic; solver > CoT)
- 2604.21590 — AgenticQwen (RIVAL: small agentic == large via RL + decision-trees; post-cutoff, verify)
- 2203.14465 — Zelikman et al., STaR (Self-Taught Reasoner)
- 2212.08410 — Hsieh et al., Distilling Step-by-Step
- 2210.03629 — Yao et al., ReAct
- 2408.03314 — Snell et al., Scaling Test-Time Compute Optimally (>parameter scaling; bounded)
- 2407.21787 — Brown et al., Large Language Monkeys (coverage vs verifier plateau)
- 2411.01101 — Byerly & Khashabi, Self-Consistency Falls Short (TACL 2026; correlated errors)
- 2605.29800 — Apple, Nine Judges Two Effective Votes (correlated judges; post-cutoff verify)
- 2510.03199 — Di et al., Best-of-Majority (voting/BoN scaling limits)
- 2309.03882 — Zheng et al., LLMs Are Not Robust MC Selectors / PriDe (ICLR 2024; selection bias)
- (already in bib) 2402.12875 Li CoT-serial; 2305.15408 Feng; 2207.00729 Merrill parallelism tradeoff
