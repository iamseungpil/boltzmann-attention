# Deep-research references (89 total)

1. Qiu, Ye, Gao, Zou, Chen, Gui, Huang, Xue, Qiu, Zhao (2025). *Blueprint First, Model Second: A Framework for Deterministic LLM Workflow*. arXiv:2508.02721. — [offload-boundary] deterministic blueprint backbone, LLM only where judgment needed. [quantified results unverified]
2. (Authors n/a) (2025). *From REST to MCP: An Empirical Study of API Wrapping and Automated Server Generation for LLM Agents (AutoMCP)*. arXiv:2507.16044. — [offload-boundary] spec→server compilation 76%→94.2% with auto-repair, cuts glue.
3. (PayPal authors) (2025). *A Declarative Language for Building And Orchestrating LLM-Powered Agent Workflows*. arXiv:2512.19769. — [cost-tradeoff] declarative DSL, 76% faster modifications, config not code. [self-reported]
4. (Authors n/a) (2025). *Agentic AI: A Comprehensive Survey of Architectures, Applications, and Future Directions*. arXiv:2510.25445. — [offload-boundary] symbolic/hybrid dominate safety-critical; generate-then-validate pattern.
5. (Authors n/a) (2020). *Robotic Process Automation — A Systematic Literature Review and Assessment Framework*. arXiv:2012.11951. — [cost-tradeoff] RPA low build cost but fragile/costly maintenance (non-vendor anchor).
6. (Authors n/a) (2026). *Automatic End-to-End Data Integration using Large Language Models*. arXiv:2603.10547. — [offload-boundary] LLM auto-generates schema/value mappings, replaces glue. [unverified, future-dated]
7. (Authors n/a) (2025). *LRASGen: LLM-based RESTful API Specification Generation*. arXiv:2504.16833. — [offload-boundary] LLM generates OpenAPI specs, cuts config-authoring cost.
8. (Authors n/a) (2024). *AI-powered software testing tools: A systematic review and empirical assessment of their features and limitations*. arXiv:2409.00411. — [offload-boundary] honest counterweight on self-healing locator limits.
9. (Authors n/a) (2026). *SAGAI-MID: A Generative AI-Driven Middleware for Dynamic Runtime Interoperability*. arXiv:2603.28731. — [offload-boundary] LLM as self-healing runtime integration layer. [unverified, future-dated]
10. (Authors n/a) (2024). *GRAM: Generative Retrieval Augmented Matching of Data Schemas*. arXiv:2406.01876. — [offload-boundary] generative schema matching ~88.7% vs 75.3% prior.
11. (Authors n/a) (n.d.). *Matchmaker: Self-Improving Large Language Model Programs for Schema Matching*. OpenReview:18E2ZooCte. — [offload-boundary] zero-shot self-improving schema matching, confidence-gated.
12. (Authors n/a) (n.d.). *A Neuro-Symbolic Framework for Deterministic Reliability in AI-Assisted Structural Engineering: The SYNAPSE Architecture*. MDPI 2075-5309/16/3/534. — [offload-boundary] deterministic safety-critical math, 94% accuracy <2s.
13. ZenML (2025). *What 1,200 Production Deployments Reveal About LLMOps in 2025*. zenml.io blog. — [offload-boundary] safety logic moved out of prompts into infrastructure.
14. Anthropic (n.d.). *Building Effective AI Agents*. anthropic.com/research. — [offload-boundary] deterministic workflows vs flexible agents, prefer simplest.
15. Stack Overflow Blog (2025). *Reliability for Unreliable LLMs*. blog (2025-06-30). — [offload-boundary] determinism from acceptance/validation; gate state/money/trust.
16. Google Developers Blog (n.d.). *Production-Ready AI Agents: 5 Lessons from Refactoring a Monolith*. blog. — [formalize-granularity] shift NL contract to runtime-validated typed object.
17. Google ADK (n.d.). *OpenAPI tools*. google.github.io/adk-docs. — [offload-boundary] auto-generate tools from OpenAPI; endpoint change = spec edit.
18. Microsoft Learn (n.d.). *Declarative agent architecture (Copilot extensibility)*. MS Learn docs. — [cost-tradeoff] config-not-code; OpenAPI-bound, no multistep loops.
19. Microsoft Azure AI Foundry (n.d.). *Three Tiers of Agentic AI — and When to Use None of Them*. vendor. — [offload-boundary] deterministic supervisor + agentic specialists.
20. deepset (n.d.). *AI Agents and Deterministic Workflows: A Spectrum, Not a Binary Choice*. vendor. — [offload-boundary] partition by criticality, not task sophistication.
21. Deepchecks (n.d.). *How Prompt Updates Drive Most Incidents*. vendor. — [cost-tradeoff] prompt surface itself a high change-cost liability.
22. Praetorian (n.d.). *Deterministic AI Orchestration: A Platform Architecture for Autonomous Development*. blog. — [offload-boundary] mission-critical needs deterministic, auditable execution.
23. Ampersand (n.d.). *The Integration Debt Trap: Why Building Integrations In-House Breaks Down at Scale*. withampersand.com. — [cost-tradeoff] 160–480 hr/yr, $50k–$150k/yr per integration.
24. Airbyte (n.d.). *How to Build API Integrations That Don't Break*. airbyte.com. — [cost-tradeoff] one provider change propagates across all tenants.
25. Duvo (n.d.). *Why Every RPA Project Breaks (And How Agentic AI Fixes It)*. blog.duvo.ai. — [cost-tradeoff] 30–50% RPA failure, presentation-layer coupling.
26. Truto (n.d.). *Building Integrations In-House and Other Horror Stories*. vendor. — [cost-tradeoff] integrations silently stop syncing until customer notices.
27. denlava / DEV Community (2023). *Reducing API Test Brittleness: Strategies Against Minor Schema Changes*. DEV (2023-04-03). — [cost-tradeoff] hardcoded fields +70% test maintenance; schema-aware cuts 50–70%.
28. Towards Data Science (n.d.). *From Vibe Coding to Spec-Driven Development*. TDS. — [cost-tradeoff] spec as source of truth makes approach robust.
29. (Author n/a) (2026). *Spec-driven development case study with Claude Code*. blog (2026-05-12). — [cost-tradeoff] spec authority discipline. [partial identifier]
30. samchon (n.d.). *samchon/openapi — OpenAPI converters + LLM function-calling schema composer*. GitHub. — [formalize-granularity] validation feedback raises tool-call success 70%→98%.
31. Zhou et al. (2024). *Language Agent Tree Search (LATS)*. arXiv:2310.04406 / ICML 2024. — [search-internalization] external MCTS controller + LLM self-eval value, no weight updates.
32. Hao et al. (2023). *Reasoning with Language Model is Planning with World Model (RAP)*. arXiv:2305.14992 / EMNLP 2023. — [search-internalization] LLM as world model + policy, process-reward MCTS.
33. Zhuang et al. (2024). *ToolChain\*: Efficient Action Space Navigation with A\* Search*. arXiv:2310.13227 / ICLR 2024. — [search-internalization] external A\* cost/heuristic over API-call decision tree.
34. (Authors n/a) (2025). *Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey*. arXiv:2510.09988. — [search-internalization] taxonomizes tree-search families vs reward designs.
35. Zhang et al. (2024). *ReST-MCTS\*: LLM Self-Training via Process Reward Guided Tree Search*. arXiv:2406.03816. — [search-internalization] external search guidance plus internalization via self-training.
36. Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366 / NeurIPS 2023. — [learned-routing] external episodic memory, frozen policy, fast trial gains.
37. Zhao et al. (2024). *ExpeL: LLM Agents Are Experiential Learners*. arXiv:2308.10144 / AAAI 2024. — [learned-routing] distills traces into NL insights, cross-task transfer.
38. Wang et al. (2023). *Voyager: An Open-Ended Embodied Agent with LLMs*. arXiv:2305.16291. — [learned-routing] skill library indexed by NL, frozen GPT-4, compose skills.
39. Wang, Mao, Fried, Neubig (2024). *Agent Workflow Memory (AWM)*. arXiv:2409.07429. — [learned-routing] induces reusable abstract workflows, +24.6%/+51.1% relative.
40. (Authors n/a) (2025). *Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills*. arXiv:2512.16301. — [learned-routing] parametric vs external memory vs skill libraries. [unverified, future-dated]
41. (Authors n/a) (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers*. arXiv:2603.07670. — [learned-routing] distilled abstract skills > raw retrieval banks. [unverified, future-dated]
42. (Authors n/a) (2025). *A Benchmark for Procedural Memory Retrieval in Language Agents*. arXiv:2511.21730. — [learned-routing] evaluates retrieving past procedures from experience DB. [unverified, future-dated]
43. Song et al. (2024). *Trial and Error: Exploration-Based Trajectory Optimization (ETO)*. arXiv:2403.02502 / ACL 2024. — [search-internalization] whole-trajectory DPO internalizes success>failure into weights.
44. Putta et al. (2024). *Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents*. arXiv:2408.07199. — [search-internalization] MCTS + self-critique + off-policy DPO. [figures unverified]
45. (Authors n/a) (2024). *Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models*. arXiv:2409.09345. — [learned-routing] external Q-store guides selection without retraining policy.
46. Lightman et al. (2023). *Let's Verify Step by Step*. arXiv:2305.20050 / OpenAI; ICLR 2024. — [small-model-reasoning] PRM>ORM, MATH 78%, PRM800K; external verifier ranks candidates.
47. Wang et al. (2024). *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*. arXiv:2312.08935 / ACL 2024. — [small-model-reasoning] auto step-PRM, Mistral-7B GSM8K 89.1%; external + internalize modes.
48. Luo et al. (2024). *OmegaPRM / Improve Mathematical Reasoning by Automated Process Supervision*. arXiv:2406.06592 / Google DeepMind. — [small-model-reasoning] MCTS+binary search locates first error, scales PRM labels.
49. Snell et al. (2024). *Scaling LLM Test-Time Compute Optimally...*. arXiv:2408.03314. — [cost-tradeoff] external search-vs-verifier beats bigger policy at fixed compute, 14× larger.
50. (Authors n/a) (2025). *DG-PRM: Dynamic and Generalizable Process Reward Modeling*. arXiv:2507.17849 / ACL 2025. — [learned-routing] external reward-tree dynamically selected, OOD-generalizable; swappable store.
51. (Authors n/a) (2026). *ToolPRMBench: Evaluating and Advancing PRMs for Tool-using Agents*. arXiv:2601.12294. — [learned-routing] extends step-PRMs to tool-use, checklist sub-goal rewards. [unverified, future-dated]
52. Arumugam et al. (2025). *Toward Efficient Exploration by Large Language Model Agents*. arXiv:2504.20997. — [learned-routing] PSRL via LLMs, retains provable Bayesian regret guarantees.
53. Xia et al. (2024). *Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents (LEAD)*. arXiv:2407.01887. — [learned-routing] LEAD inherits regret guarantees from external bandit, not LLM.
54. (Authors n/a) (2024). *Which LLM to Play? Convergence-Aware Online Model Selection with Time-Increasing Bandits (TI-UCB)*. arXiv:2403.07213. — [learned-routing] non-stationary success-rate prioritizer with regret analysis.
55. (Authors n/a) (2025). *A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses*. arXiv:2501.01849. — [learned-routing] provably-improving external selector over LLM candidates.
56. (Authors n/a) (2025). *QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search*. arXiv:2502.02584. — [learned-routing] Q-guided stepwise search (raw hit). [unverified]
57. (Authors n/a) (2025). *AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress*. arXiv:2511.08325. — [learned-routing] PRM capturing decision interdependence (raw hit). [unverified, future-dated]
58. (Authors n/a) (2025). *TDRM: Smooth Reward Models with Temporal Difference*. arXiv:2509.15110. — [learned-routing] smoothed reward models (raw hit). [unverified]
59. (Authors n/a) (n.d.). *Principle Process Reward (PPR)*. OpenReview (no arXiv id). — [learned-routing] unifies step assessment + outcome verification (raw hit). [unverified]
60. (Authors n/a) (2025). *Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving*. arXiv:2507.06229. — [learned-routing] cross-domain experience reuse (raw hit). [unverified]
61. (Authors n/a) (2026). *ProcMEM: Learning Reusable Procedural Memory via Non-Parametric PPO*. arXiv:2602.01869. — [learned-routing] reusable procedural memory (raw hit). [unverified, future-dated]
62. (Authors n/a) (2025). *LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems*. arXiv:2510.04851. — [learned-routing] modular procedural memory (raw hit). [unverified]
63. (Authors n/a) (2025). *How Memory Management Impacts LLM Agents: experience-following behavior*. arXiv:2505.16067. — [learned-routing] memory management / experience-following (raw hit). [unverified]
64. (Authors n/a) (2026). *ToolTree: Dual-Feedback MCTS + Bidirectional Pruning*. arXiv:2603.12740. — [search-internalization] tool-planning tree search (raw hit). [unverified, future-dated]
65. (Authors n/a) (2025). *A Survey of Process Reward Models: From Outcome Signals to Process Supervisions*. arXiv:2510.08049. — [small-model-reasoning] PRM survey (raw hit). [unverified]
66. (Authors n/a) (n.d.). *Toward Large Reasoning Models: reinforced reasoning with LLMs*. PMC12546433. — [small-model-reasoning] reasoning-model survey (secondary). [unverified]
67. (Authors n/a) (2024). *Scaling of Search and Learning: A Roadmap to Reproduce o1*. arXiv:2412.14135. — [search-internalization] search+learning roadmap (raw hit). [unverified]
68. Guo et al. / IRNet (2019). *Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation (IRNet/SemQL)*. arXiv:1905.08205 / P19-1444. — [formalize-granularity] schema-link→SemQL→deterministic SQL; reference-emit, cross-domain transfer.
69. Dong & Lapata (2018). *Coarse-to-Fine Decoding for Neural Semantic Parsing*. ACL P18-1068. — [formalize-granularity] value-free sketch first then fill, shared across examples.
70. (Authors n/a) (2019). *Sketch-based semantic parsing*. arXiv:1909.00574. — [formalize-granularity] sketch-then-fill; value-filling is itself learned.
71. (Authors n/a) (2021). *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding*. EMNLP 2021.emnlp-main.779. — [constrained-decoding] T5-3B 68-71%→75.1%, execution error 12%→2%.
72. (Authors n/a) (2020). *RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL*. arXiv:1911.04942. — [formalize-granularity] schema-as-input relation-aware self-attention, zero-shot schema transfer.
73. (Authors n/a) (2017). *Execution-Guided Decoding for Text-to-SQL*. arXiv:1807.03100. — [constrained-decoding] deterministic execution engine filters faulty programs, WikiSQL 83.8%.
74. (Authors n/a) (2025). *Grammar-Constrained Decoding as Parser (form-not-meaning)*. ACL 2025.acl-industry.34. — [constrained-decoding] grammar guarantees form not meaning; CFG ceiling, needs solver. [semantic-improvement claim REFUTED]
75. Qin et al. (2023). *ToolLLM: Facilitating LLMs to Master 16000+ Real-World APIs*. arXiv:2307.16789. — [formalize-granularity] NL→API tool-plan setting (sourced, verification claim did not survive).
76. Patil et al. (2023). *Gorilla: Large Language Model Connected with Massive APIs*. arXiv:2305.15334. — [formalize-granularity] NL→API calling setting (sourced, verification claim did not survive).
77. Li et al. (2023). *API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs*. arXiv:2304.08244. — [formalize-granularity] NL→API benchmark (sourced, verification claim did not survive).
78. Tam et al. (2024). *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of LLMs*. arXiv:2408.02442 / EMNLP 2024 Industry. — [constrained-decoding] schema-JSON drops GSM8K ~26–73%; two-stage recovers.
79. (Authors n/a) (2025). *CRANE: Reasoning with Constrained LLM Generation*. arXiv:2502.09061 / ICML 2025. — [constrained-decoding] grammar augmentation + alternating constrained/free, +10pp GSM-sym/FOLIO.
80. Ye et al. (2024). *Guided Decoding distribution distortion (GAD)*. NeurIPS 2024. — [constrained-decoding] greedy token-masking + renormalization distorts distribution (KL-bias).
81. (Authors n/a) (n.d.). *JSONSchemaBench (in-schema reasoning-field-first)*. (benchmark). — [constrained-decoding] free-text reasoning field inside schema first recovers reasoning.
82. Chen et al. (2022). *Program of Thoughts (PoT): Disentangling Computation from Reasoning*. arXiv:2211.12588 / TMLR 2023. — [floor-measurement] program + deterministic interpreter offload, ~12% over CoT.
83. Gao et al. (2022). *PAL: Program-Aided Language Models*. arXiv:2211.10435 / ICML 2023. — [floor-measurement] PAL+Codex beats PaLM-540B+CoT by +15% on GSM8K; small+offload>540B.
84. Joren et al. (2025). *Sufficient Context: A New Lens on Retrieval-Augmented Generation*. ICLR 2025. — [floor-measurement] context-sufficiency autorater; small=reasoning-limited, large=info-limited.
85. Zhou et al. (2023). *Least-to-Most Prompting Enables Complex Reasoning in LLMs*. ICLR 2023. — [small-model-reasoning] SCAN 16%→≥99%; decomposition not scale-gated.
86. Hsieh et al. (2023). *Distilling Step-by-Step*. ACL 2023. — [small-model-reasoning] 770M T5 beats few-shot 540B PaLM with 80% less data.
87. Huang et al. (n.d.). *Large Language Models Cannot Self-Correct Reasoning Yet*. (ICLR). — [small-model-reasoning] internal self-correction degrades reasoning, external verify needed.
88. Zhang et al. (2024). *Small models need strong external verifiers*. ACL 2024. — [small-model-reasoning] small models fail self-verify on memorization-heavy substeps.
89. (Authors n/a) (2025). *T1: external-verifier scaling for small-model reasoning*. (2025). — [small-model-reasoning] small models need strong external verifier.
</content>
</invoke>
