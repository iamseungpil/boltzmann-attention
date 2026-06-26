# References — Paper 1 (verified 2026-06-26)

These are the **only** works cited in the manuscript. `\nocite{*}` was removed
from the LaTeX source; the bibliography now contains exactly these 29 entries.
Each was opened individually on arXiv and confirmed: exact title, first author,
year, and that the in-text use matches the paper's actual content.

(Scratch literature notes `refs_lit.md` / `refs_dr.md` are research-time
collections and are **not** the reference list — superseded by this file.)

| # | arXiv | First author | Title (verbatim) | Year |
|---|-------|--------------|------------------|------|
| 1 | 2207.00729 | W. Merrill & A. Sabharwal | The Parallelism Tradeoff: Limitations of Log-Precision Transformers | 2022 |
| 2 | 2305.15408 | G. Feng | Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective | 2023 |
| 3 | 2402.12875 | Z. Li | Chain of Thought Empowers Transformers to Solve Inherently Serial Problems | 2024 |
| 4 | 1911.01547 | F. Chollet | On the Measure of Intelligence | 2019 |
| 5 | 2401.11817 | Z. Xu | Hallucination is Inevitable: An Innate Limitation of Large Language Models | 2024 |
| 6 | 2509.04664 | A. T. Kalai | Why Language Models Hallucinate | 2025 |
| 7 | 2309.02427 | T. R. Sumers | Cognitive Architectures for Language Agents | 2023 |
| 8 | 2505.07087 | R. E. Wray | Applying Cognitive Design Patterns to General LLM Agents | 2025 |
| 9 | 2403.00810 | F. Zhu | Bootstrapping Cognitive Agents with a Large Language Model | 2024 |
| 10 | 2510.09355 | F. Yuan | NL2GenSym: Natural Language to Generative Symbolic Rules for SOAR Cognitive Architecture via Large Language Models | 2025 |
| 11 | 2402.01817 | S. Kambhampati | LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks | 2024 |
| 12 | 2211.10435 | L. Gao | PAL: Program-aided Language Models | 2022 |
| 13 | 2411.15124 | N. Lambert | Tülu 3: Pushing Frontiers in Open Language Model Post-Training | 2024 |
| 14 | 1807.06096 | N. Jansen | Safe Reinforcement Learning via Probabilistic Shields | 2018 |
| 15 | 2503.22738 | Z. Chen | ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning | 2025 |
| 16 | 2503.18666 | H. Wang | AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents | 2025 |
| 17 | 2402.00798 | Z. Li | Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents | 2024 |
| 18 | 2511.21689 | H. Su | ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration | 2025 |
| 19 | 2606.06976 | Y. Zhou | TRUST: Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning | 2026 |
| 20 | 2604.07036 | D. Piatrashyn | ReDAct: Uncertainty-Aware Deferral for LLM Agents | 2026 |
| 21 | 2410.10347 | J. Dekoninck | A Unified Approach to Routing and Cascading for LLMs | 2024 |
| 22 | 2506.07982 | V. Barres | τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment | 2025 |
| 23 | 2406.12045 | S. Yao | τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains | 2024 |
| 24 | 2503.08669 | Z. Li | SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints | 2025 |
| 25 | 2501.10132 | L. Zhong | ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario | 2025 |
| 26 | 2510.12838 | Q. Chen | A²FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning | 2025 |
| 27 | 2510.08439 | C. Qian | xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning | 2025 |
| 28 | 2504.18851 | H. Ross | When2Call: When (not) to Call Tools | 2025 |
| 29 | 2605.09252 | C.-E. Sun | LLM Agents Already Know When to Call Tools — Even Without Reasoning | 2026 |

## Notes on the verification pass (2026-06-26)

- **Primary-benchmark mis-citation fixed.** The manuscript's primary bench is
  τ²-bench (dual-control); it had been cited as `2406.12045`, which is the
  **τ-bench predecessor** (Yao et al.). Added the correct τ²-Bench entry
  (`2506.07982`, Barres et al.) and repointed the citation (τ-bench kept as the
  named predecessor).
- **TRUST claim corrected.** `2606.06976` ("TRUST", Yijin Zhou) is
  *uncertainty-aligned RL for tool-calling decisions*, not the "4-way
  Direct/Tool/Ask/Unable routing, 4B>30B" originally written. In-text rewritten
  to the paper's actual content.
- **ToolOrchestra numbers confirmed** against the paper's Table 16
  (Orchestrator-8B 80.2%@10.3¢ vs GPT-5 77.7%@31.3¢ on τ²).
- **Reduced from 229 padding entries to the directly-cited set.** `\nocite{*}`
  removed so the printed bibliography is exactly the cited set — no
  unverified/fabricated-id exposure.
- **§2.5 named-but-uncited rivals resolved.** Systems named in prose now carry
  verified citations: A²FM (2510.12838), xRouter (2510.08439), When2Call
  (2504.18851); the "tool-necessity linearly decodable, AUROC 0.89–0.96" claim
  is attributed to its real source (2605.09252, Sun et al.). **RTR** and
  **RITE** could not be found on arXiv (probable fabricated/garbled acronyms)
  and were **deleted** from the prose. **ARM2** (2510.08163) is real but is a
  multimodal reasoning-format model, not a router — removed from the routing
  list. Entry count 25 → 29.
