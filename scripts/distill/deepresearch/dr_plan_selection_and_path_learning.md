# Plan-Path Generation, Prioritization, Selection & Experience-Learning in LLM Agents
## SALVAGE SYNTHESIS — recovered from a killed deep-research run

> **Provenance / salvage note.** This report is reconstructed from the persisted agent
> transcripts of a `deep-research` fan-out that **completed its finder agents but was KILLED
> before final synthesis**. The fan-out was *planned* as ~6 angle searchers + source extractors
> (the orchestrator described it as "30 finder agents"), but only **8 agent transcripts persisted
> to disk** in the workflow directory. Of those:
> - **5 returned complete structured findings** (the four declared families + trajectory-offline-learning).
> - **1 was the lead/planner** agent (emitted the angle plan, not findings).
> - **1 (synthesis / neuro-symbolic angle) was interrupted ~1s before emitting its StructuredOutput**;
>   its raw web-search result lists were recovered from the transcript and are included below with a
>   "raw-hit, un-curated" caveat.
> - **1 (a source-extractor for the LATS paper) was killed immediately after tool-load** — no content.
>
> **Citation discipline:** every citation below literally appeared in a recovered transcript
> (either in a searcher's curated StructuredOutput or in a raw WebSearch result list). No citation
> has been invented. arXiv IDs that the searchers themselves flagged as unverified, or that appear
> only as raw search hits with future-dated (2026 / `26xx.*`) IDs, are explicitly marked
> **[ID unverified in transcripts]**. Claims whose source could not be tied to a specific recovered
> citation are marked **[source unverified in transcripts]**.

---

## How to read the central axis

The research question turns on one axis, which every family below is mapped onto:

> **INTERNALIZED** (prioritization baked into the LLM policy's weights via training) vs.
> **EXTERNAL** (prioritization performed by a deterministic search controller, a learned-but-separate
> value/critic store, or a retrieval-over-experience database that sits outside the frozen policy).

The thesis this report serves: *the LLM does finite, low-dimensional, abstract planning (learned
once, transferred zero-shot to new domains by swapping a config/ABox), while a deterministic machine
does exact execution, verification, and selection.* The synthesis section maps each prioritization
mechanism onto "should be learned-transferable LLM skill" vs "should be deterministic / swappable store."

---

## Family (1) — Search-based planning over LLM-generated steps (MCTS / tree search)

**Signal that drives prioritization:** node-selection scores in a tree search — typically **UCT**
(exploration + exploitation) over a **value estimate**, where the value is either **LLM self-evaluation**
or a learned value/PRM. The search *controller* (tree, selection, expansion, backprop) is an **EXTERNAL,
deterministic** mechanism; the *value signal* is usually **LLM-internalized (self-eval)** rather than a
separate learned store.

- **LATS — Language Agent Tree Search** (Zhou et al., ICML 2024, arXiv:2310.04406). The flagship
  MCTS-for-LLM-agent framework. The **same LLM** is action generator, LM-powered value function
  (prompted to output scalar future-reward estimates), and reflection mechanism; node selection uses
  **UCT over self-evaluated values + self-reflection**. The search controller (tree/UCT/backprop) is
  external and deterministic; the value signal is LLM-internalized — **no weight updates**; cross-trial
  learning is via in-context reflection / episodic memory. Reported to outperform ReAct, Reflexion, CoT,
  ToT, RAP on programming and web tasks; one searcher cited **94.4% pass@1 on HumanEval, 75.9 on
  HotPotQA**. *Canonical reference for "external search controller + LLM value signal."*

- **RAP — Reasoning with Language Model is Planning with World Model** (Hao et al., EMNLP 2023,
  arXiv:2305.14992; peer-reviewed at aclanthology.org/2023.emnlp-main.507). One LLM is repurposed as
  **both world model (predicts next state) and reasoning policy**; MCTS explores the reasoning/plan tree.
  Per-step reward = a **process/step-level blend of action log-probability, state confidence, LLM
  self-evaluation, and task-specific heuristics** — computed at inference time by the external search loop,
  not internalized. A precise instance of process-reward-driven branch ranking. Reported: LLaMA-33B+RAP
  beats CoT-on-GPT-4 by ~33% relative on plan generation.

- **ToolChain\*** (Zhuang et al., ICLR 2024, arXiv:2310.13227). The **tool-use / function-calling**
  member of this family, directly relevant to policy-gated agents. Formulates the **entire action space
  as a decision tree where each node is an API/function call**, then runs **A\* best-first search** with
  task-specific cost `g` (cost-so-far) + heuristic `h` (estimated future cost) to prune wrong-action
  branches and return the lowest-cost valid plan. The prioritization signal is an **explicit
  heuristic/cost function evaluated by an EXTERNAL deterministic controller (A\*)** — a clean contrast
  point for the deterministic-selector thesis. Reported +3.1% / +3.5% accuracy at 7.35× / 2.31× less time
  vs prior SOTA. Three-stage loop: select most-promising path → expand next actions → update cost
  functions.

- **Survey anchor — "Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey"**
  (2025, arXiv:2510.09988). Taxonomizes tree-search families (MCTS/UCT, A\*, beam, best-first; ToT, RAP,
  LATS, ToolChain\*, AlphaZero-style self-play) **against** reward designs (UCT, self-eval, learned value
  / PRM, outcome vs process). Best single scaffold tying family (1) to family (3).

- **ReST-MCTS\*** (Zhang et al., 2024, arXiv:2406.03816). Process-reward-guided tree search where a PRM
  supplies step-level value for node selection, **and successful trajectories are distilled back into the
  policy** — i.e., it sits on the boundary: external search-controller guidance *plus* internalization via
  self-training. Useful as the explicit external-vs-internalized contrast within search.

- **Raw-hit (un-curated, from interrupted synthesis agent):** **QLASS — Boosting Language Agent Inference
  via Q-Guided Stepwise Search** (arXiv:2502.02584 [ID unverified in transcripts]) and **ToolTree:
  Dual-Feedback MCTS + Bidirectional Pruning** (arXiv:2603.12740 [ID unverified in transcripts, future-dated])
  appeared as raw search results for tool-planning tree search but were never curated by a finder.

> *Note:* **Tree-of-Thoughts** and explicit **AlphaZero-style self-play** were named in the question and
> referenced inside the survey anchor, but **no dedicated ToT or AlphaZero citation was independently
> recovered** from the transcripts. **[source unverified in transcripts]** for any standalone ToT/AlphaZero claim.

---

## Family (2) — Learning from accumulated experience (memory, skill libraries, workflow memory)

**Signal that drives prioritization:** retrieval similarity + accumulated success/failure traces.
Across this entire family the **LLM policy stays frozen**; prioritization away from failing paths and
toward high-success paths lives in an **EXTERNAL, swappable store** (text reflections, NL insights,
code-skill libraries, induced workflows). This is the family most directly aligned with the thesis's
"domain-general frozen policy + domain-specific swappable experience store."

- **Reflexion — Language Agents with Verbal Reinforcement Learning** (Shinn et al., NeurIPS 2023,
  arXiv:2303.11366). Agent verbally reflects on outcome feedback (success/failure) and stores the
  self-critique in an **EXTERNAL episodic memory buffer** re-injected as context next trial — **no weight
  updates**. Frozen policy; prioritization lives entirely in a swappable text store. Reference point for
  sample efficiency of trial-based path prioritization (gains over a handful of trials).

- **ExpeL — LLM Agents Are Experiential Learners** (Zhao et al., AAAI 2024, arXiv:2308.10144). Distills
  accumulated trajectories (success/failure pairs gathered via Reflexion) into **natural-language INSIGHTS
  + a retrievable pool of successful trajectories** used as in-context exemplars. Reusable knowledge held
  in an external domain-specific store; base LLM untouched. Reports **cross-task transfer of extracted
  insights** — directly on the transfer question. (Also surfaced by the synthesis agent alongside
  **Synapse** as "retrieve similar past successful trajectories as exemplars.")

- **Voyager — An Open-Ended Embodied Agent with LLMs** (Wang et al., 2023, arXiv:2305.16291). The defining
  **skill-library** paper: verified executable programs (self-verification + environment feedback) stored
  in an ever-growing library **indexed by NL descriptions**; later tasks **retrieve and COMPOSE prior
  verified skills** rather than re-derive. Retrieval+composition controller is external; **GPT-4 is frozen
  (no fine-tuning)**. Compounding, compositional "prefer high-success paths" maps onto a value-store keyed
  by past success.

- **Agent Workflow Memory (AWM)** (Wang, Mao, Fried, Neubig, 2024, arXiv:2409.07429). Induces commonly
  reused **multi-step ROUTINES (workflows = abstracted action sequences / DAGs)** from past trajectories
  and selectively injects them; works **offline** (induced from training examples) and **online** (on the
  fly). Reported **+24.6% (Mind2Web) / +51.1% (WebArena) relative success with fewer steps**. The strongest
  match for *"abstract finite plan templates learned once and reused"*: the induced workflow is an external,
  swappable memory while the LLM stays fixed; highly applicable to tool-use / web agents.

- **Survey / situating sources:**
  - **"Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills"** (2025, arXiv:2512.16301
    [ID unverified in transcripts, future-dated]). Taxonomizes the three adaptation axes the thesis needs:
    **parametric post-training (DPO/RFT internalized) vs external memory vs skill libraries.**
  - **"Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers"** (2026,
    arXiv:2603.07670 [ID unverified in transcripts, future-dated]). Covers Voyager, ExpeL, AWM, JARVIS-1 /
    CRADLE (retrieve past trajectories as in-context plans, **keyed by environment similarity**), and
    Generative-Agents episodic streams (importance + embedding retrieval). Notably reports an **emerging
    consensus that distilling traces into abstract skills outperforms raw retrieval banks** — directly
    bearing on keeping a domain-general policy while the experience store stays domain-specific/replaceable.
  - **"A Benchmark for Procedural Memory Retrieval in Language Agents"** (2025, arXiv:2511.21730 [ID
    unverified in transcripts, future-dated]). Evaluates retrieving past procedures / successful plans from
    an experience DB — probes sample efficiency, transfer, and soundness of an external procedural-memory
    store keyed by prior success.
  - **Raw-hits (un-curated, from interrupted synthesis agent):** **Agent KB: Leveraging Cross-Domain
    Experience for Agentic Problem Solving** (arXiv:2507.06229 [ID unverified in transcripts]),
    **ProcMEM: Learning Reusable Procedural Memory via Non-Parametric PPO** (arXiv:2602.01869 [ID
    unverified, future-dated]), **LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems**
    (arXiv:2510.04851 [ID unverified]), and **"How Memory Management Impacts LLM Agents: experience-following
    behavior"** (arXiv:2505.16067 [ID unverified]). These appeared as raw search results on cross-domain
    experience reuse but were never curated by a finder.

---

## Family (2b) — Trajectory-level offline learning (DPO/RFT on success): INTERNALIZING the policy

**Signal that drives prioritization:** contrastive success > failure trajectory pairs, distilled into
**weights**. This is the **INTERNALIZED** counterpoint to family (2): prioritization is baked into the
policy, not held in an external store.

- **ETO — Trial and Error: Exploration-Based Trajectory Optimization** (Song et al., ACL 2024,
  arXiv:2403.02502). Cleanest trajectory-level internalization: SFT on expert trajectories → exploration
  collects the agent's own **FAILED** trajectories → forms contrastive (success > failure) pairs →
  **whole-trajectory DPO**. Prioritization (prefer high-success, avoid failing paths) is **baked into
  weights**. Reports strong OOD gains (~22% over SFT on ScienceWorld generalization); the searcher noted
  experiments **splicing sub-trajectories from different tasks to synthesize novel composite plans** —
  bears on cross-domain transfer vs domain-specific value stores.

- **Agent Q** (Putta et al., 2024, arXiv:2408.07199). Combines **guided MCTS over LLM steps + AI
  self-critique + iterative off-policy DPO**, so both successful and failed trajectories are distilled into
  the policy. Pairs an **EXTERNAL search controller (MCTS + value/self-eval)** with **INTERNALIZATION (DPO
  on resulting trajectories)** — lets you contrast which part stays external vs learned. **Caveat flagged by
  the searcher itself:** reported figures inconsistent (snippet says ~18.6% → 40.6%, ~9.3% over RFT; but
  "the original number is 18.6 → 81.7% pass-rate in the paper — verify exact figures against the PDF before
  citing"). **[figures unverified in transcripts]**

---

## Family (3) — Process Reward Models (PRM) & step-level value/critic functions

**Signal that drives prioritization:** a learned **step-level value / critic** that scores each
reasoning step or action; per-step PRM predictions act as **value estimates of reward-to-go** for the
base policy. PRMs are most often used as an **EXTERNAL verifier** to rank/select among candidates sampled
from a separate generator — but the *same* step-value signal can also be **internalized** via step-level
RL. Outcome (ORM, final-answer only) vs process (PRM, per-step) is the core distinction.

- **Let's Verify Step by Step** (Lightman et al., 2023, OpenAI, arXiv:2305.20050). Foundational
  outcome-vs-process paper. Trains a **PRM** (per-step feedback) vs an **ORM** (final-answer only); PRM
  substantially outperforms ORM and majority vote in **best-of-N reranking** on MATH (process-supervised
  model solves **78%** of a representative MATH subset). Releases **PRM800K** (800K human step labels). The
  PRM is an **EXTERNAL verifier that RANKS/SELECTS** among candidates from a separate generator — selection
  is offloaded to a learned-but-external scorer, not internalized. Process gives **more precise credit
  assignment + error localization** than outcome-only; also more interpretable/alignable. Active learning
  improves process-supervision efficiency.

- **Math-Shepherd** (Wang et al., 2024, arXiv:2312.08935 [ID unverified in transcripts — searcher
  rendered as html/2312.08935]). Canonical **automatic** step-value-without-human-labels method: a step's
  value = **Monte-Carlo-estimated probability that completions from that step reach the correct final
  answer (reward-to-go)**. Uses the PRM **both** as an EXTERNAL verifier (best-of-N / step-beam search)
  **and** as a dense reward to **INTERNALIZE via step-by-step PPO** — the canonical reference for the
  internal-vs-external distinction: same step-value signal, two deployment modes.

- **OmegaPRM** ("Improve Mathematical Reasoning by Automated Process Supervision," Luo et al., 2024,
  Google DeepMind, arXiv:2406.06592). Scales automatic step-value collection by combining MC estimation
  with **MCTS + binary search to locate the first erroneous step**, cutting annotation cost. Bridges family
  (1) and (3): the **MCTS controller is the external deterministic mechanism; the trained PRM is the learned
  value store** guiding node selection.

- **Scaling LLM Test-Time Compute Optimally...** (Snell et al., 2024, arXiv:2408.03314). Directly on family
  (4) trade-offs + synthesis: a learned process verifier used as an **EXTERNAL controller** for best-of-N
  reranking, beam search, and lookahead/tree search, and characterizes **when external search-against-a-
  verifier beats spending the same compute on a bigger/internalized policy.** Frames PRM per-step scores as
  reward-to-go value estimates; analyzes **compute-optimal allocation between generation and verification** —
  the core sample-efficiency / external-vs-internal trade-off.

- **DG-PRM — Dynamic and Generalizable Process Reward Modeling** (ACL 2025, arXiv:2507.17849). **Most
  directly relevant to the transfer concern.** Standard PRMs are heuristic and **BRITTLE across domains**.
  DG-PRM stores fine-grained **multi-dimensional reward criteria in an external "reward tree"** and
  **dynamically selects which criteria apply per step** (Pareto-dominance to pick positive/negative pairs),
  achieving strong **out-of-distribution generalization**. Concrete evidence that the **value/criteria store
  should be an external, swappable, domain-keyed structure** rather than a fixed scalar policy — matching
  "domain-general policy + swappable domain-specific value store."

- **ToolPRMBench — Evaluating and Advancing PRMs for Tool-using Agents** (2026, arXiv:2601.12294 [ID
  unverified in transcripts, future-dated]). Extends PRMs from math to **tool-use / function-calling**:
  step-level verifiers as critics scoring each action/sub-goal in long-horizon tool trajectories where
  sparse outcome reward is insufficient; includes **checklist-style sub-goal reward models** judging an
  action by its contribution to expected steps. The applicability dimension for policy-gated agents (cite
  with verification caveat — future-dated preprint).

- **Step-Level Q-Value Models** ("Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models,"
  2024, arXiv:2409.09345). Bridges (3) and (4) for tool-use agents: trains a **step-level Q-value model**
  scoring candidate actions so the agent prefers high-success branches; the **Q-store accumulates value
  estimates from experience and guides selection WITHOUT retraining the policy LLM** — concretely a
  domain-specific, swappable VALUE store keyed by estimated success, separate from a general policy. (The
  synthesis agent independently corroborated this with Wang et al. 2024 / Zhai et al. 2024 step-level value
  guidance for agent inference.)

- **Raw-hits (un-curated, from interrupted synthesis agent):**
  - **AgentPRM — Process Reward Models for LLM Agents via Step-Wise Promise and Progress** (arXiv:2511.08325
    [ID unverified, future-dated]; also ACM Web Conf 2026 listing). Re-defines PRM for agent tasks to capture
    **interdependence between sequential decisions and their contribution to the goal.**
  - **Principle Process Reward (PPR)** (OpenReview, no arXiv ID recovered [source unverified in transcripts]):
    RL approach unifying **principled step-level assessment + outcome verification**, with a Reward
    Normalization strategy calibrating outcome vs process rewards.
  - **TDRM — Smooth Reward Models with Temporal Difference** (arXiv:2509.15110 [ID unverified]).
  - These appeared only as raw search hits and were not curated; treat as leads, not verified findings.

---

## Family (4) — Success-rate / bandit / Bayesian-posterior path prioritization (provable improvement)

**Signal that drives prioritization:** accumulated success counts / Bayesian posterior over which
arms/paths succeed, with **regret guarantees**. The recurring finding: **soundness / provable
improvement comes from the EXTERNAL deterministic controller (bandit / PSRL algorithm), not from the
LLM.** The LLM supplies candidate actions/preferences; the external mechanism guarantees convergence.

- **Toward Efficient Exploration by Large Language Model Agents** (Arumugam et al., 2025,
  arXiv:2504.20997). Cleanest answer to family (4) and the synthesis. Implements **PSRL (Posterior
  Sampling for RL)** by outsourcing distinct algorithm steps to separate LLMs: an **Optimal-Sample-Policy
  LLM** selects actions given a sampled MDP hypothesis; a **Posterior-Updater LLM** updates beliefs over
  trajectories. **Retains PSRL's PROVABLE Bayesian regret / exploration guarantees that vanilla LLM agents
  lack.** The cleanest demonstration that prioritization/value machinery can be an **external Bayesian-
  posterior mechanism (the belief/experience store)** while the LLM is a swappable generator — exactly the
  "deterministic machine does selection, LLM does abstract steps" decomposition. (Evaluated on Bernoulli
  bandit, Combination Lock, Wordle, RiverSwim; base-LLM choice flips linear vs sublinear regret.)

- **Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents** (Xia et al., 2024/2025,
  arXiv:2407.01887). LLMs alone struggle to converge / exploit consistently; proposes **LEAD**, fusing
  classic dueling-bandit algorithms with the LLM. **LEAD INHERITS weak/strong-regret guarantees from the
  underlying bandit algorithm** — soundness from the **external** controller, not the LLM. Strong support
  for "provable path-prioritization lives in the external mechanism; the LLM supplies candidates."

- **TI-UCB — Convergence-Aware Online Model Selection with Time-Increasing Bandits** (2024,
  arXiv:2403.07213). A UCB-style bandit modeling an **increasing-then-converging** performance pattern with
  regret analysis — a **non-stationary** external success-rate prioritizer. Directly addresses the
  **non-stationarity** trade-off: success rates drift as the policy/skills improve, a caveat for any
  value-store keyed on past success.

- **Multi-Agent Conversational Bandit for Online Selection of User-Aligned LLM Responses** (2025,
  arXiv:2501.01849). Contextual/conversational bandit selecting among candidate LLM outputs with
  **near-optimal regret bounds** — a provably-improving external selector over LLM-generated candidates.
  (Framed at response-selection, not full plan-DAG granularity.)

> *Note:* No standalone classic Thompson-sampling tool-selection paper was independently curated; the
> synthesis agent's bandit search explicitly noted "the search results did not specifically contain
> detailed information about the Thompson sampling and bandit-based path selection aspects." The PSRL /
> dueling-bandit / UCB papers above carry the family-(4) load.

---

## KEY SYNTHESIS — mapping plan-path prioritization onto the thesis split

**Thesis:** LLM = finite, low-dimensional, **abstract** planning, learned once and transferred zero-shot
by **swapping a config/ABox**; a **deterministic machine** does exact execution, verification, and
**selection**. Below, each prioritization sub-function is assigned to *learned-transferable LLM skill* or
*deterministic / swappable store*, with the trade-off that justifies it. (All assignments are grounded in
the recovered families above.)

**A. Should be a LEARNED, domain-general LLM skill (internalize, transfer zero-shot):**
1. **Candidate plan-step generation / proposal** — the abstract "what kinds of steps exist and in what
   order." Every search family (LATS, RAP, ToolChain\*) uses the LLM as the *generator*; this is the
   transferable core. The thesis's "finite low-dimensional abstract planning" = the policy that proposes
   the skeleton, which families (1)+(2b) show transfers (ETO's sub-trajectory splicing; AWM's induced
   abstract workflows; ExpeL's cross-task insights).
2. **The *shape* of good plans** (decomposition, ordering priors) — internalizable via trajectory-level
   DPO/RFT (ETO, Agent Q) **only to the extent it is domain-general**. Caution: ETO/Agent Q internalize
   *domain-specific success preferences* into weights — which **fights** zero-shot transfer if over-fit.
   Keep the internalized part to domain-general step-shaping; push domain-specific success bias to the store.

**B. Should be a DETERMINISTIC external mechanism (the swappable, domain-specific half):**
3. **The search controller itself** — tree expansion, UCT/A\*, beam, backprop. Universally external and
   deterministic across the entire search family (LATS tree/UCT, ToolChain\* A\*, OmegaPRM/ReST-MCTS\* MCTS).
   Soundness and reproducibility come from here. **→ deterministic controller.**
4. **Exact selection among scored candidates** (best-of-N, top-K beam pruning) — the verifier/selector.
   Lightman/Snell show selection is offloaded to an external scorer; Snell shows this can beat a bigger
   internalized policy at fixed compute. **→ deterministic selector over an external value signal.**
5. **The value / success-rate store** keyed by past success — **the swappable, domain-specific component.**
   DG-PRM is the load-bearing evidence: a fixed scalar PRM is **brittle across domains**, but an **external
   "reward tree" of criteria that is dynamically selected per step generalizes OOD**. Step-Level Q-Value
   Models show the value store can accumulate and guide selection **without retraining the policy.** This is
   precisely "domain-general policy + domain-specific swappable value store." **→ external, swap on ABox change.**
6. **Provable path-prioritization / exploration** — bandit/PSRL. Arumugam (PSRL) and Xia (LEAD) show the
   **regret/soundness guarantee lives in the external algorithm**, with the LLM as a swappable generator.
   **→ external deterministic controller** when provable improvement is required.
7. **Experience DB / retrieval of past successful plans** — Reflexion / ExpeL / Voyager / AWM all keep the
   **policy frozen** and put accumulated success in an **external swappable store**. The "Memory for
   Autonomous LLM Agents" survey's emerging consensus — **distilled abstract skills > raw retrieval banks** —
   suggests the store should hold *abstracted* procedures (matching the thesis's "finite abstract plans"),
   keyed by environment/domain similarity (JARVIS-1/CRADLE pattern). **→ external, swappable.**

**C. Trade-offs that decide the split:**
- **Sample efficiency:** external experience/value stores (Reflexion-style, Q-value, PRM-reranking) improve
  in a *handful* of trials with **no weight updates**; internalization (DPO/RFT) needs many trajectories.
  Favor external for fast, cheap, swappable adaptation.
- **Non-stationarity:** keying a store on *past* success is fragile because success rates **drift** as the
  policy/skills improve (TI-UCB explicitly models increasing-then-converging performance). A static
  success-rate value store decays; prefer controllers that handle non-stationarity, or re-estimate on swap.
- **Transfer:** internalized domain-specific preferences (ETO/Agent Q) risk **anti-transfer**; the policy
  must stay domain-general. DG-PRM + per-domain reward-tree, and ExpeL/AWM external memory, give transfer by
  **swapping the store, not the policy** — the thesis's mechanism.
- **Soundness:** provable improvement (regret bounds, error-localization) is achievable **only in the
  external deterministic mechanism** (PSRL/LEAD regret inheritance; PRM step-level error localization;
  A\* cost optimality). A pure LLM self-eval value gives no guarantee. For **policy-gated tool-use agents**,
  this argues strongly for an external, auditable selector/verifier (ToolChain\* cost gating, ToolPRMBench
  step critics) rather than trusting an internalized policy to respect gates.

**Bottom line for the thesis:** generate (LLM, learned, transferable) → search & score (deterministic
controller + external value/PRM/bandit store, domain-specific & swappable on ABox change) → select & verify
(deterministic, sound, auditable). The literature recovered here repeatedly demonstrates that the **search
controller, the exact selector/verifier, and the success-keyed value/experience store can all be external
and swappable while the proposing policy stays frozen and domain-general** — which is exactly the
decomposition the thesis proposes. The principal hazards are (i) internalizing domain-specific success bias
into the policy (anti-transfer) and (ii) non-stationary success-rate stores going stale across swaps.

---

## Citations recovered (literally present in transcripts)

**Curated by finder agents (high confidence; arXiv IDs as they appeared):**
- Lightman et al. 2023 — *Let's Verify Step by Step* — arXiv:2305.20050 (OpenAI; PRM800K)
- Wang et al. 2024 — *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations* — arXiv:2312.08935 [ID unverified — appeared as html/2312.08935]
- Luo et al. 2024 — *OmegaPRM / Improve Mathematical Reasoning by Automated Process Supervision* — arXiv:2406.06592 (Google DeepMind)
- Snell et al. 2024 — *Scaling LLM Test-Time Compute Optimally...* — arXiv:2408.03314
- DG-PRM 2025 — *Dynamic and Generalizable Process Reward Modeling* — arXiv:2507.17849 (ACL 2025; aclanthology.org/2025.acl-long.212)
- *ToolPRMBench: Evaluating and Advancing PRMs for Tool-using Agents* — arXiv:2601.12294 [ID unverified, future-dated]
- Zhou et al. 2024 — *Language Agent Tree Search (LATS)* — arXiv:2310.04406 (ICML 2024)
- Hao et al. 2023 — *Reasoning with Language Model is Planning with World Model (RAP)* — arXiv:2305.14992 (EMNLP 2023; aclanthology.org/2023.emnlp-main.507)
- Zhuang et al. 2024 — *ToolChain\*: Efficient Action Space Navigation with A\* Search* — arXiv:2310.13227 (ICLR 2024)
- *Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey* (2025) — arXiv:2510.09988
- Zhang et al. 2024 — *ReST-MCTS\*: LLM Self-Training via Process Reward Guided Tree Search* — arXiv:2406.03816
- Shinn et al. 2023 — *Reflexion: Language Agents with Verbal Reinforcement Learning* — arXiv:2303.11366 (NeurIPS 2023)
- Zhao et al. 2024 — *ExpeL: LLM Agents Are Experiential Learners* — arXiv:2308.10144 (AAAI 2024)
- Wang et al. 2023 — *Voyager: An Open-Ended Embodied Agent with LLMs* — arXiv:2305.16291
- Wang, Mao, Fried, Neubig 2024 — *Agent Workflow Memory* — arXiv:2409.07429
- *Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills* (2025) — arXiv:2512.16301 [ID unverified, future-dated]
- *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers* (2026) — arXiv:2603.07670 [ID unverified, future-dated]
- *A Benchmark for Procedural Memory Retrieval in Language Agents* (2025) — arXiv:2511.21730 [ID unverified, future-dated]
- Song et al. 2024 — *Trial and Error: Exploration-Based Trajectory Optimization (ETO)* — arXiv:2403.02502 (ACL 2024)
- Putta et al. 2024 — *Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents* — arXiv:2408.07199 [reported figures unverified]
- *Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models* (2024) — arXiv:2409.09345
- Arumugam et al. 2025 — *Toward Efficient Exploration by Large Language Model Agents* — arXiv:2504.20997
- Xia et al. 2024/2025 — *Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents (LEAD)* — arXiv:2407.01887
- *Which LLM to Play? Convergence-Aware Online Model Selection with Time-Increasing Bandits (TI-UCB)* (2024) — arXiv:2403.07213
- *A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses* (2025) — arXiv:2501.01849

**Raw search hits (recovered from interrupted synthesis agent; NOT curated/verified — leads only):**
- QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search — arXiv:2502.02584 [ID unverified]
- AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress — arXiv:2511.08325 [ID unverified, future-dated]
- TDRM: Smooth Reward Models with Temporal Difference — arXiv:2509.15110 [ID unverified]
- Principle Process Reward (PPR) — OpenReview, no arXiv ID recovered [source unverified]
- Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving — arXiv:2507.06229 [ID unverified]
- ProcMEM: Learning Reusable Procedural Memory via Non-Parametric PPO — arXiv:2602.01869 [ID unverified, future-dated]
- LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems — arXiv:2510.04851 [ID unverified]
- How Memory Management Impacts LLM Agents (experience-following behavior) — arXiv:2505.16067 [ID unverified]
- ToolTree: Dual-Feedback MCTS + Bidirectional Pruning — arXiv:2603.12740 [ID unverified, future-dated]
- Survey: "A Survey of Process Reward Models: From Outcome Signals to Process Supervisions" — arXiv:2510.08049 [ID unverified]
- Survey: "Toward Large Reasoning Models: reinforced reasoning with LLMs" — PMC12546433 [secondary]
- *Scaling of Search and Learning: A Roadmap to Reproduce o1* — arXiv:2412.14135 [ID unverified, raw hit]

**Salvage caveats:**
- Several arXiv IDs are **future-dated (2026 / 25xx–26xx)** relative to the run date (2026-06-15). They are
  reproduced exactly as they appeared in the transcripts; treat all future-dated and "[ID unverified]" IDs
  as **unconfirmed** and re-verify before any external citation (per the project's arXiv-citation discipline).
- **Tree-of-Thoughts and AlphaZero-style self-play** were named in the brief and referenced inside the
  survey anchors but had **no independently recovered primary citation** — mark as [source unverified] if used.
- Two of the eight persisted transcripts contributed **no findings** (one source-extractor killed at
  tool-load; one lead/planner emitted only the angle plan). One synthesis agent was killed before curating
  its output; its raw hits are salvaged above but were never quality-filtered by the finder.
