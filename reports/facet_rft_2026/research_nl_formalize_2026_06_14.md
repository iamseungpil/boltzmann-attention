# NL → Formal-Specification ("LLM-as-formalizer") — latest (2024–2026) survey
> Grounds A2 thesis-core (NL policy → deterministic GATE_SPEC, distilled 7B compiler + deterministic replay verifier). Strict citation discipline: every paper below was opened on its arXiv abs page, title/authors/version verified, key results quoted verbatim. Unverified items are quarantined in §5. **Build BEYOND** the already-covered AgentSpec (`2503.18666`), NSVIF (`~2601.17789`), planning-formalizer survey (`2503.18971`) — those are NOT re-reported here.
> Date: 2026-06-14. Author: research agent. Do not treat as authority doc until merged by master (`EXPERIMENT_DESIGN.md §7`).

---

## 1. 핵심 답변 요약 (where the field is, 2026)

**The field has converged on one reliability recipe**, and it is exactly the shape of our A2 design — which is good for external validity and bad for naive novelty claims. The dominant 2024–2026 pattern for *any* NL→formal-artifact task (math, logic, LTL, PDDL, Rego, SQL) is:

> **generate candidate formalization(s) → run a SOUND external check → keep/repair/abstain → (optionally) aggregate surviving candidates.**

The "sound external check" instantiates per domain: a Lean/Isabelle compiler (autoformalization), a model-checker over sample traces (LTL), a solver/`VAL` validator (PDDL), OPA `opa test` + negative tests (Rego), or query execution (SQL). This is precisely our **deterministic replay verifier** as the fixed program with a swappable generator.

**The four techniques most transferable to OUR policy→GATE_SPEC compiler:**

1. **Cross-stage faithfulness detection** ("Do LLMs Game Formalization?", `2604.19459`). The single most important 2026 finding: *high compile/accept rate ≠ faithful spec*. Two distinct failure modes — **reactive axiom fabrication** (detectable by comparing across stages) vs. **silent premise mistranslation** (evades detection entirely). This directly augments our replay verifier, which catches over/under-deny but is blind to a spec that *replays correctly yet means the wrong thing*. → maps to A2 risk ② (verifier coverage).
2. **Draft-and-prune + survivor aggregation** ("Draft-and-Prune", `2603.17233`): draft multiple NL plans, condition program generation on them, **prune executable-but-contradictory/ambiguous formalizations**, majority-vote the survivors. This is the principled version of our K-sample + verifier-select + abstain (`A2_FRONTEND §3` inference row), with an explicit "executable but contradictory" prune our replay does not yet do.
3. **Ground-truth-free equivalence scoring** (VeriEquivBench, `2510.06296`): score a spec by *formally proving functional equivalence* (in Dafny) rather than matching a reference. Analog for the schema-expressivity ceiling: when no gold spec exists, equivalence-over-behavior is the metric — our replay-over-gold-trajectories is a weaker (trace-sampled) version of this.
4. **Knowledge-reasoning-fused distillation into 7B/32B** (StepFun-Formalizer, `2508.04440`): the *only* verified instance of a small-model formalizer reaching SOTA on a formalization benchmark, via dual-dataset distill (knowledge-rich examples + template-guided informal→formal reasoning traces) + SFT + RLVR. Direct template/precedent for our S0→S1→S2 ladder.

**Bottom line for novelty (§3):** the recipe is settled; our *combination* — swappable small-model generator + fixed deterministic verifier + verified-distillation + **business-policy/SOP domain with a runtime that enforces the spec** — is not occupied by any single paper. The closest analog by *task* (NL access-policy → executable Rego) is **Prose2Policy** (`2603.15799`), but it is prompt-pipeline frontier-only, no distillation, no transfer claim. The closest by *method* (small distilled formalizer) is StepFun, but it is math-only. Nobody combines them in the policy-enforcement regime.

---

## 2. Per-axis verified findings

### Axis 1 — Autoformalization (NL math/proof → Lean/Isabelle/Coq)

**ProofFlow** — *ProofFlow: A Dependency Graph Approach to Faithful Proof Autoformalization* — Cabral, Do, Yu, Tai, Feng, Shen — arXiv **`2510.15981` v1, 13 Oct 2025**.
- Method (verbatim): "ProofFlow first constructs a directed acyclic graph (DAG) to map the logical dependencies between proof steps. Then, it employs a novel lemma-based approach to systematically formalize each step as an intermediate lemma, preserving the logical structure."
- Result (verbatim): "achieving a ProofScore of 0.545, substantially exceeding baselines like full-proof formalization (0.123) … and step-proof formalization (0.072)." New benchmark = "184 undergraduate-level problems, manually annotated with … logical dependency graphs". ProofScore = composite of "syntactic correctness, semantic faithfulness, and structural fidelity."
- ⚠️ Fairness flag: **custom benchmark + custom metric** (ProofScore, 184 problems). The 0.545-vs-0.123 gap is on the authors' own metric; no third-party benchmark number. Treat the *direction* (structure-aware decomposition helps faithfulness) as the transferable insight, not the magnitude.
- **Transfer to non-math specs**: the DAG-of-dependencies idea maps onto our policy globality problem (A2 risk ③, long-policy cross-references) — formalize gate-by-gate as intermediate units with an explicit dependency graph rather than one-shot whole-policy dump.

**FormalAlign** — *FormalAlign: Automated Alignment Evaluation for Autoformalization* — arXiv **`2410.10135`, ICLR (accepted)**.
- Problem (verbatim framing): existing methods "can only verify the surface-form integrity of the autoformalized sequence via BLEU or by passing it to a formal language compiler," whereas FormalAlign "successfully detects the semantic misalignment of the autoformalized statement with the informal sequence." Method = "dual loss that combines a pair of mutually enhancing autoformalization and alignment tasks."
- Result (verbatim, from search summary — abstract-level): "achieves 3.19% higher performance on MiniF2F-Valid compared to baseline approaches." ⚠️ I verified the paper identity/abstract framing but did NOT open the results table; cite the 3.19% as **abstract-reported, table unverified**.
- **Transfer**: this is a *learned* faithfulness checker (compiler-pass ≠ aligned). Strong precedent that our replay verifier (behavioral, not semantic) needs a faithfulness companion — see §3(a).

**StepFun-Formalizer** — covered in Axis 5 (it's the small-model story).

**Lean Workbook** — *Lean Workbook: A large-scale Lean problem set formalized from natural language math problems* — Ying, Wu, Geng, Yuan, Lin, Chen — arXiv **`2406.03847` v1 6 Jun 2024 → v3 18 Jun 2025**. Method = "iteratively generates and filters synthetic data" → ~57k formal-informal pairs. ⚠️ Note: a search engine mis-attributed this ID to "FormalAlign"; it is NOT FormalAlign — corrected here. Relevance: precedent for **iterative generate-and-filter synthetic data engine** = our reverse-render data engine (`A2_FRONTEND §2`).

### Axis 2 — NL → formal SPEC / logic / temporal logic

**VLTL-Bench** — *Verifiable Natural Language to Linear Temporal Logic Translation: A Benchmark Dataset and Evaluation Suite* — English, Walker, Simon, Jha, Ewetz — arXiv **`2507.00877` v1 1 Jul 2025 → v2 18 Dec 2025**.
- Decomposition (verbatim): four steps — "lifting", "grounding", "translation", "verification" — and the benchmark "provides ground truths after each of these steps." Faithfulness check = "sample traces to validate the temporal logic expressions" across four state spaces.
- Critique it makes (verbatim): "current studies measure only the accuracy of the translation," overlooking verification; "reveals near-perfect performance on existing benchmarks" (i.e., prior NL→LTL benches are saturated/easy).
- ⚠️ Fairness flag: no per-model headline number in the abstract; evaluates nl2spec, NL2TL, NL2LTL, Lang2LTL, seq2seq. **Most transferable idea**: *ground-truth-after-each-substep* + *trace-based verification* — exactly our replay-over-trajectories, generalized. Our gate-spec pipeline has an implicit lifting/grounding/translation; VLTL-Bench argues you should be able to verify each.

**Do LLMs Game Formalization?** — *Do LLMs Game Formalization? Evaluating Faithfulness in Logical Reasoning* — Kim, Poiroux, Bosselut — arXiv **`2604.19459` v1, 21 Apr 2026**. ★ Most directly relevant single paper.
- Core finding (verbatim): "high compilation rates or accuracies should not be equated with faithful reasoning." Two failure modes: GPT-5 "fabricates axioms during proof generation, a reactive fallback detectable via cross-stage comparison"; DeepSeek-R1 "mistranslates premises during formalization, producing internally consistent outputs that evade detection entirely." Also: in unified generation "models prefer reporting failure over forcing proofs, even under prompting designed to encourage it" (i.e., abstention behavior is real).
- Fairness: observational study across frontier models, not a custom-win claim — credible. **This is our axis-④/faithfulness anchor.**

**NL→FOL, fine-tuned** — *Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs* — arXiv **`2509.22338`, Nov 2025**. Result (search-summary level, ⚠️ table unverified): "fine-tuned Flan-T5-XXL achieves 70% accuracy with predicate lists, outperforming GPT-4o and DeepSeek-R1-0528 with Chain-of-Thought." Relevance: **a small fine-tuned model beating frontier on a constrained formalization task when given the predicate list** — directly supports our "closed-schema + distilled small model can match frontier" thesis (A2 §4 prediction). ⚠️ "with predicate lists" = the schema is handed in, analogous to our guided/closed GATE_SPEC schema — fair to cite as supportive, but note the predicate-list crutch.

**Do LLMs Really Struggle at NL-FOL Translation?** — Brunello, Geatti, Mignani, Montanari, Saccomanno — arXiv **`2511.11816` v1, 14 Nov 2025**.
- Finding (verbatim): "state-of-the-art, dialogue-oriented LLMs demonstrate strong NL-FOL translation skills and a genuine grasp of sentence-level logic, whereas embedding-centric models perform markedly worse." Claims prior "struggle" narrative is partly an **evaluation-methodology artifact**; their protocol separates "genuine semantic-level logical understanding from superficial pattern recognition, memorization, and dataset contamination."
- **Caution for us**: a direct warning that EM/canonical-match metrics misrepresent formalization ability — echoes our own S0 finding that canonical-JSON EM "breaks on prose-field variation" (`A2_FRONTEND §3` S0 smoke). Use structure-EM, not surface EM. Already aligned.

### Axis 3 — NL → code / API / config / planning-domain (closest analogs)

**NL→PDDL, end-to-end agentic** — *End-to-end PDDL Planning with Hardcoded and Dynamic Agents* — La Malfa, Zhu, Marro, Bernardini, Wooldridge — arXiv **`2512.09629` v1 10 Dec 2025 → v2 8 May 2026**. ★ Closest *architecture* analog to a multi-stage NL-policy→spec compiler.
- Method (verbatim): orchestrator "receives a human specification written in natural language and converts it into a PDDL model, where the domain and problem are iteratively refined by sub-modules (agents)" addressing "time constraints and optimality, as well as ambiguities and contradictions." **Hardcoded agents** = "informed by logs and error traces with predefined objectives like fixing PDDL syntax and verifying temporal constraints"; **dynamic agents** adapt/revise the abstraction.
- Tools: Fast Downward, LPG, POPF, **VAL, uVAL** (deterministic validators in the loop). Models: GPT-4o/4.5-mini/5.4, Gemini-2.5/3-flash. Benchmarks: NaturalPlan, PlanBench, Sokoban, Blocksworld, Hanoi. Code public (github EmanueleLM/MultiAgentPlanning).
- ⚠️ No single headline accuracy number in the abstract — "significant step toward end-to-end planning." Treat as architectural precedent, not a metric win. **Transfer**: the **hardcoded-agent = deterministic-validator-feedback** split is exactly our fixed-verifier; their "ambiguity/contradiction resolution" sub-agents are what our reverse-render data engine must teach the 7B to do internally.

**Generating consistent PDDL domains** — Smirnov, Joublin, Ceravola, Gienger — arXiv **`2404.07751` v1, 11 Apr 2024**. Method = "automated consistency checking during the generation process." Honest limitation (verbatim): the checks "still can't guarantee absolute correctness … but they can serve as valuable source of feedback reducing the amount of correction efforts expected from a human in the loop." Domains: logistics, gripper, tyreworld, household, pizza. **Transfer**: realistic framing — generate-and-verify reduces but doesn't eliminate HITL; matches our abstain→HITL fallback.

**NL→Rego (policy-as-code)** — *Prose2Policy (P2P): A Practical LLM Pipeline for Translating Natural-Language Access Policies into Executable Rego* — Apple — arXiv **`2603.15799`, Mar 2026**. ★ Closest *task* analog to A2 (NL policy → executable enforcement code). See Axis 6.

**NL→SQL** — survey + execution-guided self-correction is mature. *SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction* (`2509.00581`), *CSC-SQL: Corrective Self-Consistency* (`2505.13271`), survey *Next-Generation Database Interfaces* (`2406.08426` v5). ⚠️ I did not open each abs page individually — cite as **trend evidence only** (search-summary level): execution-grounded validation + self-debug from error traces + corrective self-consistency is the standard NL→SQL reliability stack. The transferable bit: **execution feedback as the verifier** = our replay. Do not cite specific SQL numbers without verification.

### Axis 4 — Verification & faithfulness loops (the "replay verifier" analog — SOTA discipline)

The discipline, ranked by strength of guarantee:

1. **Formal equivalence (strongest)** — VeriEquivBench, `2510.06296` (Zeng, Che, Huang, Ye, Xu, Yuan, Fu; v1 7 Oct 2025 → v3 18 Apr 2026). Replaces reference-matching with "a formally grounded metric, the equivalence score" that "rigorously verifies the quality of generated specifications and code" by proving functional equivalence in Dafny. 2,389 problems. Verbatim sober conclusion: "generating formally verifiable code remains a profound challenge for state-of-the-art LLMs." **This is the ceiling of our replay idea**: replay-over-gold-trajectories ≈ trace-sampled equivalence; full equivalence proof is the unattainable-but-aspirational target.
2. **Trace/model-checking (strong)** — VLTL-Bench sample-trace verification (`2507.00877`); PDDL `VAL`/`uVAL` (`2512.09629`). = our replay over gold trajectories. **Exactly our regime.**
3. **Cross-stage comparison (catches fabrication)** — Do-LLMs-Game-Formalization (`2604.19459`): compare formalization stage vs proof stage to catch reactive fabrication; but silent mistranslation evades it.
4. **Learned alignment scorer (semantic)** — FormalAlign (`2410.10135`): a trained model that flags semantic misalignment a compiler-pass misses.
5. **Survivor aggregation + prune (self-consistency over formalizations)** — Draft-and-Prune (`2603.17233`): "prunes executable but contradictory or ambiguous formalizations, and aggregates predictions from surviving paths via majority voting." On AR-LSAT auto-formalization-only: "78.43% accuracy with GPT-4 and 78.00% with GPT-4o"; "100% on PrOntoQA and LogicalDeduction." ⚠️ Fairness: PrOntoQA/LogicalDeduction are near-saturated easy logic sets; the 100% is unremarkable; AR-LSAT (hard) is the real signal. Cite AR-LSAT, discount the 100%s.
6. **Abstention** — both Draft-and-Prune (prune-all → no answer) and Do-LLMs-Game (models "prefer reporting failure") show abstain-on-low-confidence is real and beneficial = our abstain→HITL row.

**Net SOTA discipline**: verifier-in-loop is *necessary but not sufficient* for faithfulness — a behavioral verifier (compile/execute/replay) is blind to silent semantic mistranslation. The 2026 frontier adds a **semantic/cross-stage faithfulness layer on top** of the behavioral check. Our replay verifier sits at level 2; we lack a level-3/4 companion.

### Axis 5 — Distillation / small-model formalizers

**StepFun-Formalizer** — *StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion* — Wu, Huang, Wan, Peng, Shang, Cao, Qi, Zhang, Du, Yan, Hu — arXiv **`2508.04440` v1 6 Aug 2025 → v3 26 Dec 2025**. ★ The one verified small-model formalizer SOTA.
- Sizes: **7B and 32B**. Method "ThinkingF" = dual dataset: "(1) distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates"; trained via **SFT and RLVR**.
- Result (verbatim): "StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models."
- ⚠️ Fairness: BEq@1 (back-translation/equivalence @1) is a defensible metric; "surpassing all prior" is a leaderboard claim on those two benches — credible but verify the leaderboard if load-bearing. **No isolated 7B-vs-32B-vs-frontier table reported in abstract** — the headline is the 32B; the 7B number is not in the abstract I read. Cite 32B SOTA; flag 7B-vs-frontier as **unverified at the per-size level**.
- **Direct map to S0→S2**: ThinkingF dual-dataset ≈ our (knowledge: real-domain teacher-compiled specs S1) + (reasoning: template-guided synthetic reverse-render S0); SFT+RLVR ≈ our S0 SFT + S2 on-policy DPO. **Strong precedent that the ladder shape works.**

**General distillation evidence** (⚠️ search-summary / blog level, NOT primary-verified — treat as soft): well-distilled 7B retains ~70–85% of 70B on agent tasks; ≤7B shows "persistent performance gaps." Do NOT cite these numbers in the paper without opening primaries; listed only to calibrate expectation that **7B-from-frontier on a closed-schema task is plausible but the gap is task-dependent**.

**Gap**: NO paper found that does *verified* distillation (verifier-filtered teacher traces → small student) for *spec/policy* generation specifically. StepFun uses RLVR (verifier reward) but for math. **This is open territory = our S1 contribution.**

### Axis 6 — Policy / compliance-specific (NL business-policy / access-policy → executable rules)

**Prose2Policy (P2P)** — Apple — arXiv **`2603.15799`, Mar 2026**. ★ Closest task analog.
- Pipeline (verbatim features): "policy detection, component extraction, schema validation, linting, compilation, automatic test generation and execution" → executable **Rego** (OPA). Goal "bridge … human-readable access requirements and machine-enforceable policy-as-code … emphasizing deployment reliability and auditability."
- Results (verbatim, on ACRE dataset): "95.3% compile rate for accepted policies, with automated testing achieving a 82.2% positive-test pass rate and a 98.9% negative-test pass rate."
- ⚠️ Fairness/scope flags: (i) **"compile rate for accepted policies"** — the "accepted" qualifier means abstained/rejected policies are excluded from the 95.3% denominator (selection effect — note this is the *honest-denominator* issue we know well from SOPBench). (ii) **auto-generated tests** are LLM-written, so 82.2/98.9 test-pass is partly self-graded unless tests are independently sound. (iii) Frontier-prompt pipeline, **no distillation, no small model, no transfer claim, no on-prem/sovereignty angle.**
- **This is the single closest published system to A2** and the most important one to position against. It validates the *task is real and useful* (Apple shipped a pipeline) while leaving our entire differentiation (distilled small model + fixed deterministic verifier as the program + sovereignty/air-gap + cross-domain transfer) untouched.

**ABAC/RBAC-from-NL** (verified identities, ⚠️ details search-summary level): *LLMs for ABAC Policy Mining* (`2511.18098`, Nov 2025); *Can LLMs Make (Personalized) Access Control Decisions?* (`2511.20284`, Nov 2025); *"Say What You Mean": NL Access Control with LLMs for IoT* / LACE (`2505.23835`, May 2025); LMN tool (`2502.12460`). Collectively: active 2025–2026 line on NL→machine-enforceable access policy. **Transfer**: confirms the access-control sub-domain is crowded; our domain (customer-service SOP → gate-spec for an agent runtime) is adjacent but distinct (procedural ordering + satisfier-tools + ask/terminal, not just allow/deny).

**Requirements-engineering NL→formal-spec**: survey *Formalising Software Requirements with LLMs* (`2506.14627` v1 17 Jun → v2 23 Jun 2025; 94 papers — ⚠️ abstract-level only, full taxonomy unread); related NL2Spec/Req2LTL line. The RE community frames the core challenges as **ambiguity, missing domain knowledge, contextual gaps, instability of LLM outputs** (search-summary) — identical to our A2 risk ① (implicit/world-knowledge clauses). NL2Spec's mitigation = sub-expression translation + human-in-loop disambiguation. **Transfer**: their ambiguity-surfacing UI ≈ our abstain→HITL.

### Axis 7 — Failure modes & open problems (verified, mapped to our risks)

| Failure mode | Verified source + verbatim | A2 risk |
|---|---|---|
| **Hallucinated predicates / fabricated axioms** | `2604.19459`: GPT-5 "fabricates axioms … a reactive fallback detectable via cross-stage comparison" | risk ②, ④ |
| **Silent semantic mistranslation (evades behavioral check)** | `2604.19459`: R1 "mistranslates premises … internally consistent outputs that evade detection entirely" | risk ② (replay blind spot) |
| **Compile/accept rate ≠ faithfulness** | `2604.19459`: "high compilation rates or accuracies should not be equated with faithful reasoning" | core |
| **Surface metric ≠ semantic ability** | `2511.11816`: prior "struggle" narrative partly an eval-methodology artifact; `2410.10135`: BLEU/compile-pass miss misalignment | EM→structure-EM (already done) |
| **Structure/globality loss in one-shot whole-input formalization** | `2510.15981`: full-proof baseline 0.123 vs DAG-decomposed 0.545 | risk ③ (long-policy globality) |
| **Schema/expressivity ceiling (no GT, when DSL can't express the clause)** | `2510.06296`: ground-truth-free equivalence needed precisely because reference specs don't exist; "remains a profound challenge" | risk ④ |
| **Ambiguity / missing world-knowledge in NL input** | RE survey + NL2Spec line (`2506.14627`); `2404.07751` (consistency checks can't "guarantee absolute correctness") | risk ① |
| **Small-model floor on hard formalization** | StepFun reports 32B headline, not 7B; general distill literature notes ≤7B "persistent gaps" (⚠️soft) | S0/S1 expectation calibration |

---

## 3. Mapped to A2 — adopt / novelty

### (a) Faithfulness checking BEYOND our replay verifier
Our replay verifier is a **level-2 behavioral check** (over/under-deny on gold trajectories). The 2026 SOTA says behavioral pass ≠ faithful (`2604.19459`). **Adopt two cheap additions:**
- **Cross-stage consistency** (from `2604.19459`): since our pipeline already produces an intermediate (NL → predicate prose → satisfier mapping), compare the model's own NL gloss of each gate against the source policy clause. Catches "reactive fabrication" of gates with no policy basis — a class our replay cannot see (a fabricated gate that happens to fire correctly on gold trajectories still passes replay).
- **Learned alignment scorer / round-trip** (FormalAlign `2410.10135` + our existing `t2_a2_roundtrip.py`): we already have round-trip (re-compile rendered NL → original spec) as a **data-cleaning** gate. FormalAlign's lesson: also run it as a **runtime faithfulness gate** on real (non-synthetic) policies where we lack the original spec — re-render the generated spec to NL and check entailment against the source policy. This is the analog of VLTL-Bench's "verify each substep."
- **Residual honesty**: even cross-stage + round-trip cannot catch *silent mistranslation that is internally consistent* (`2604.19459` R1 mode). Document this as the irreducible faithfulness gap → abstain on low round-trip agreement.

### (b) S1 verified-distillation
- **Adopt StepFun's ThinkingF dual-dataset split explicitly** (`2508.04440`): keep our two streams clean — (knowledge stream = real-domain teacher-compiled specs, replay-filtered) + (reasoning stream = template-guided synthetic reverse-render). StepFun's win came from *fusing* them, not either alone; our S0(synthetic-only) over-fit finding (`A2_FRONTEND §3` S0-v2 airline 0.876→0.528) is exactly the failure of reasoning-stream-only. **Their result is direct evidence S1 must mix real-domain knowledge in.**
- **Adopt Draft-and-Prune at S2/inference** (`2603.17233`): our K-sample+verifier-select should add the **"prune executable-but-contradictory"** step — a spec can replay-pass yet contain mutually contradictory gates (e.g., two gates with incompatible applies_to). Add a static contradiction check before majority/MBR selection.
- **Caveat on the headline claim**: StepFun reports 32B SOTA, not 7B-beats-frontier. Our pre-registered "7B+K+verifier ≥ frontier single-shot" (A2 §4) is *stronger* than anything verified in the literature for math; it is only credible because our task is **closed-schema + verifier-reducible-to-search** (the `2509.22338` "70% with predicate lists, beats GPT-4o" result is the closest support — small model + handed-in schema beats frontier). Keep the claim system-level (generator+verifier+select), never single-shot.

### (c) Schema-expressivity ceiling (risk ④)
- The field's answer is **VeriEquivBench's ground-truth-free equivalence** (`2510.06296`): when the target DSL can express it, prove equivalence; when it can't, the failure is *visible* (no equivalent spec provable). **Adopt the framing**: a clause the GATE_SPEC schema cannot express should produce a *detectable* abstention (round-trip will fail to reconstruct it), not a silently-wrong spec. Make the schema-ceiling a **measured abstention rate**, not a hidden error.
- This converts risk ④ from "unbounded silent failure" to "bounded, logged, HITL-routed" — the same move the RE survey (`2506.14627`) and `2404.07751` endorse (verify-reduces-but-doesn't-eliminate-HITL).

### What is GENUINELY NOVEL in our framing vs the field
The recipe (verifier-in-loop + faithfulness layer + generate-prune-aggregate) is **settled and we should claim no novelty there** — cite it as the established paradigm we instantiate. Our defensible novelty is the **specific 4-way combination, none of which co-occurs in any verified paper:**

1. **Swappable small-model generator + fixed deterministic verifier as the *program*** — VeriEquivBench/VLTL-Bench treat the verifier as an *evaluation* artifact; we treat it as the *permanent runtime contract* and the generator as replaceable. No surveyed paper makes the verifier the fixed product and the LLM the swappable part.
2. **Verified-distillation for *policy/spec* generation** — StepFun does verified-distill for *math*; Prose2Policy does *policy* but frontier-prompt-only, no distillation. The intersection (verified-distill × policy-spec) is empty in the verified set.
3. **Business-policy/SOP-enforcement domain** with procedural gates (auth-ordering, satisfier-tools, ask/terminal) — distinct from access-control allow/deny (Axis 6) and from math (Axes 1/5).
4. **Sovereignty / on-prem / air-gap motivation** for the small model — explicitly absent from every formalizer paper (all assume frontier API). This is our *raison d'être for training* (A2 §4) and is genuinely unoccupied.

**Risk to the novelty claim**: Prose2Policy (`2603.15799`) is close enough on task that reviewers will ask "why not just prompt a frontier model like P2P?" — our answer must lean on (3)+(4)+the transfer claim (LODO cross-domain), not on raw accuracy, because P2P's 95.3% compile rate looks strong (even with the honest-denominator caveat). **Position against P2P explicitly in related work.**

---

## 4. Verified bibliography / Unverified leads

### Verified (abs page opened, title/authors/version confirmed; results quoted)
- `2510.15981` v1 — ProofFlow (Cabral et al., 13 Oct 2025) — DAG/lemma autoformalization, ProofScore 0.545. ⚠️custom metric.
- `2410.10135` — FormalAlign (ICLR) — learned semantic-alignment scorer; 3.19% MiniF2F-Valid ⚠️table unverified.
- `2406.03847` v3 — Lean Workbook (Ying et al.) — ~57k iter-filtered NL↔Lean pairs. (NOT FormalAlign — search mislabel corrected.)
- `2507.00877` v2 — VLTL-Bench (English et al.) — lift/ground/translate/verify, trace verification.
- `2604.19459` v1 — Do LLMs Game Formalization? (Kim, Poiroux, Bosselut, 21 Apr 2026) — ★ fabrication vs silent-mistranslation; compile≠faithful.
- `2511.11816` v1 — Do LLMs Really Struggle at NL-FOL Translation? (Brunello et al.) — struggle = eval artifact.
- `2509.22338` — Advancing NL→FOL with fine-tuned LLMs — Flan-T5-XXL 70% w/ predicate lists > GPT-4o ⚠️table unverified.
- `2603.17233` v2 — Draft-and-Prune (Ni et al., 18 Mar 2026) — prune-contradictory + vote; AR-LSAT 78.43% GPT-4.
- `2512.09629` v2 — End-to-end PDDL Planning w/ Hardcoded & Dynamic Agents (La Malfa et al.) — orchestrator + VAL-in-loop. ⚠️no headline metric.
- `2404.07751` v1 — Generating consistent PDDL domains (Smirnov et al.) — consistency-check-in-generation, HITL-reducing.
- `2510.06296` v3 — VeriEquivBench (Zeng et al.) — ground-truth-free equivalence score (Dafny); 2,389 problems.
- `2508.04440` v3 — StepFun-Formalizer (Wu et al.) — ★ 7B/32B, ThinkingF (distill+RLVR); 32B BEq@1 40.5% FormalMATH-Lite, 26.7% ProverBench.
- `2603.15799` — Prose2Policy / P2P (Apple, Mar 2026) — ★ NL→Rego pipeline; 95.3% compile (accepted), 82.2%/98.9% test-pass. ⚠️honest-denominator + self-graded-tests.
- `2506.14627` v2 — Formalising Software Requirements with LLMs (survey, Beg et al.) — 94 papers. ⚠️abstract-level only.

### Unverified leads (identity seen in search; abs page NOT individually opened — DO NOT cite numbers without verifying)
- `2502.15795` — "Lean-ing on Quality: High-Quality Data Beats Diverse Multilingual Data in AutoFormalization."
- `2507.08665` — KELPS: verified multi-language autoformalization via semantic-syntactic alignment.
- `2509.09810` — Towards a Common Framework for Autoformalization.
- `2511.18098` — Harnessing LLMs for ABAC Policy Mining.
- `2511.20284` — Can LLMs Make (Personalized) Access Control Decisions?
- `2505.23835` — "Say What You Mean" / LACE: NL access control for IoT.
- `2502.12460` — LMN: machine-enforceable policies from NL access rules.
- `2509.00581` (SQL-of-Thought), `2505.13271` (CSC-SQL), `2406.08426` (NL→SQL survey) — NL→SQL self-correction; trend-only, no numbers verified.
- `2512.16814` — Grammar-Forced Translation of NL to Temporal Logic.
- `2507.03293` — LTLCrit / LogicGuard: temporal-logic LLM critic for embodied agents.
- General distillation 7B-retains-70-85%-of-70B claims — **blog/search-summary, not arXiv-verified; do not cite.**

### Citation-discipline notes
- Two arXiv IDs were search-mislabeled: `2406.03847` (=Lean Workbook, not FormalAlign — FormalAlign is `2410.10135`). Corrected above.
- Several "verbatim" results (`2410.10135` 3.19%, `2509.22338` 70%, `2508.04440` 7B-specific) are **abstract-summary level, results table not opened** — flagged inline; verify before they become load-bearing in the design doc.
- Fairness flags raised: ProofFlow (custom metric), Draft-and-Prune (saturated easy benches at 100%), Prose2Policy (honest-denominator + self-graded tests), StepFun (32B-not-7B headline), NL-FOL-finetuned (predicate-list crutch).
