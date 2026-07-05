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
- (already in bib) 2402.12875 Li CoT-serial; 2305.15408 Feng; 2207.00729 Merrill parallelism tradeoff
