# Deterministic vs. LLM-Learned Components in Production Agent Systems: The Maintenance/Change-Cost Tradeoff

> **SALVAGE SYNTHESIS.** This report was reconstructed *post-hoc* from a killed deep-research run.
> The fan-out completed its finder/extractor agents (the run was provisioned as a 30-agent
> fan-out; **20 agent transcripts persisted** in this workflow directory), but the harness was
> **KILLED before the final synthesis step ran**. This document is assembled by reading the
> persisted agent transcripts (`journal.jsonl` structured results + two transcripts whose
> results never reached the journal). **No new web searches were run.** Every citation below
> appears literally in the transcripts; claims whose source is ambiguous are flagged
> `[source unverified in transcripts]`.
>
> Recovery stats: **19 of 20 transcripts had usable findings**; **1 was thin/interrupted**
> (agent `ac34c60b26b5dae47`, the *Blueprint First* PDF extractor, was interrupted mid-fetch
> before emitting structured output — its partial WebFetch content is salvaged below). One of
> the 20 transcripts (`a2bd85fc8e7203581`) is the query-decomposer, not a finder; its angle
> taxonomy is reused as the report scaffold. Several extractors targeted the same sources, so
> the distinct-source count is lower than the agent count.

---

## Executive Summary

The recovered evidence converges on a single, well-supported thesis: **hand-written deterministic
integration glue is cheap to build but dominated by a recurring maintenance tail driven by
provider-side change**, while **config-/spec-driven architectures convert most "tool changed" events
from code rewrites into config edits**, and **LLMs are increasingly used both at build-time (to
generate the config/spec/connectors) and at runtime (to absorb schema drift and to formalize NL
into config-conformant calls)** — but only when wrapped in deterministic validation. Production
practice does **not** put the learned/deterministic boundary at "smart vs. dumb"; it puts it at
**accuracy-criticality**: money/auth/state-touching paths stay deterministic and configurable,
flexible/surface paths are delegated to the LLM. The config-driven core gets *both* high accuracy
*and* low change-cost, with a residual cost in **authoring and keeping the config authoritative** —
which LLMs can cut substantially (e.g., spec→server generation succeeding for 76% of tools,
rising to 94.2% with automated spec-repair) but not to zero.

---

## (1) Maintenance Cost & Brittleness of Deterministic Integrations

The strongest and most internally consistent block of evidence. Multiple independent extractors
(several from integration-platform vendors — flagged) report the same failure-mode taxonomy and
overlapping cost figures.

**Cost of ownership of hand-written connectors.**
- A single production SaaS integration costs **~200–500 hours of senior engineering to build** and
  **20–40 hours/quarter/integration in ongoing maintenance**; ~10 integrations represents "millions
  of dollars in capitalized cost" plus recurring opex. (blog, 2026-03-20; agent `a82082caa7a0b4847`)
- Annual per-integration maintenance is estimated at **$50,000–$150,000/year**, so a portfolio of
  50 integrations implies **$2.5M–$7.5M/year**; ongoing maintenance "consumes **10%–20% of a
  developer's time indefinitely** … one day a week"; "The cost of building integrations is not the
  first sprint. It is the permanent tail." (blog, 2026-03-10; agent `a0cbb61837f64f111`)
- Maintaining ~20 integrations that each change 2–3×/year yields 40–60 tasks/year ≈ **160–480
  engineering hours/year (~a quarter of one FTE)**; breaking changes occur in **~14.78% of
  real-world API releases** ("citing academic research"). (blog, 2026-03-09; agent `aca137db03260d7ce`)
- The Ampersand "integration debt" framing independently gives the same **160–480 hr/yr** and
  **$50k–$150k/yr** per integration. (vendor blog; agent `a860054a479df9199`)

**Why deterministic glue breaks: provider-side change, not consumer bugs.**
- The dominant failure mode is a **provider-side schema/behavioral change that still returns HTTP
  200 OK**, evading retries/circuit breakers; "The real failure is a provider-side change that
  slips past every retry loop and circuit breaker." (agent `aca137db03260d7ce`)
- Failures are often **silent** — "integrations silently stop syncing until a customer notices."
  (Truto, vendor; agent `a860054a479df9199`)
- One provider change **propagates across every customer connection**; a fix tested on one tenant
  doesn't generalize because of differing custom fields. (Airbyte, vendor; agent `a860054a479df9199`)
- Concrete churn cadence: **Salesforce 3 major releases/yr, ServiceNow quarterly, Jira continuously
  deprecating endpoints**; in-house efforts "predictably hit failure points within 18–24 months"
  along connection-scale, tenant-customization, and API-drift axes. (agent `a82082caa7a0b4847`)
- Vendor-specific, undocumented behaviors (HubSpot search endpoints lacking rate-limit headers,
  Microsoft Graph throttling individual requests inside a batch) make defensive glue expensive.
  (agent `a0cbb61837f64f111`)
- Root cause at the code level: **hardcoded field names / over-specified JSON structure** rather
  than schema-validated access; renaming `user_id`→`userId` breaks tightly coupled code. A blog
  reports a **~70% increase in test-maintenance time** from brittle hardcoded references; schema-aware
  validation **reduces maintenance overhead 50–70%**, and an accessor-function abstraction layer
  **cuts test updates 80–90%** for field renames. Recommended switch-point: when ">20% of schema
  changes break tests" or "test maintenance >30% of dev time." (blog, 2023-04-03; agents
  `a4c6152cb6e2b5ebe`, and corroborated by `a860054a479df9199`)

**RPA/iPaaS brittleness (quantified, but vendor-tilted).**
- Per **HfS Research** (cited via vendor blog), software licensing is only **25–30% of RPA TCO**;
  **70–75% goes to implementation, maintenance, and support**. RPA breaks because of screen-scraping
  / coordinate-based automation: a 3-year illustrative TCO is **€1.4M** (€250k licenses + €300k impl
  + €600k maint + €150k emergency + €100k training) = "560% of initial software cost," and an
  illustrative "20% weekly bot failure rate / 250+ hours weekly" managing failures. (blog,
  2025-10-25; agent `a0ba0ceb3d311ddcb`)
- A second source reports **30–50% RPA project failure rates** and maintenance consuming **30–50% of
  initial implementation budget annually**, attributing brittleness to presentation-layer coupling
  (a moved button breaks the bot). (Duvo, vendor; agent `a860054a479df9199`)
- **Academic anchor (non-vendor):** an arXiv systematic literature review of RPA (2012.11951, 2020;
  63 publications) characterizes RPA as **low implementation cost but costly, tedious maintenance and
  fragility when underlying applications/UIs change** — qualitative, no TCO percentages of its own.
  (agents `a26f2a3b55a2334fd`, snippet via `a860054a479df9199`)

> **Vendor-marketing flag:** the dollar/percentage figures in this section come predominantly from
> integration-platform and agentic-AI vendors whose conclusion is "don't build in-house." The
> *failure-mode taxonomy* (silent provider-side change, multi-tenant propagation, presentation-layer
> coupling) is concrete and corroborated across independent sources; the *specific TCO numbers*
> should be treated as directional, not audited. The arXiv RPA SLR is the cleanest non-vendor anchor.

---

## (2) Config-Driven / Declarative / Spec-Driven Architectures (Quantified Change-Cost Reduction)

This block directly tests "tool change = config edit, not code rewrite."

**Declarative agent DSL (production, quantified) — strongest sub-question-2 evidence.**
- A **PayPal** declarative DSL separates agent workflow spec from implementation, executing the same
  pipeline across Java/Python/Go and cloud/on-prem. Reported: **60% reduction in development time**,
  **3× deployment velocity**, **76% faster modifications**, **30% fewer steps**, workflows in
  **<50 lines of DSL vs 500+ lines imperative**, and "adding new tools or fine-tuning agent
  behaviors requires only pipeline specification changes, not code deployment." It separates a
  **deterministic orchestration layer** (sub-100ms overhead) from LLM calls. (arXiv, 2025-12-22;
  search agent `af42c3cd63e67c857`, verification agent `a3ff02d8a3b3a2c5e`)
  > These are **self-reported** numbers from the system's own authors — treat as vendor-grade.

**Spec→server compilation (the build-time half) — strongest empirical decoupling evidence.**
- **AutoMCP** (*From REST to MCP*, arXiv 2507.16044, 2025-07, rev. 2026-04) compiles OpenAPI 2.0/3.0
  specs into deployable MCP tool servers, eliminating hand-written glue (param parsing, header
  injection, request forwarding, response formatting). Quantified contrast: hand-written servers span
  **1.5k–704k SLOC with human maintainers** vs machine-generated **median ~0.7k SLOC, zero
  maintainers**, across 77 real-world APIs; it also auto-repairs spec defects. (search agent
  `af42c3cd63e67c857`)
- The verification pass on the same paper adds hard numbers: **baseline generation succeeds for 76%
  of sampled tools; automated spec-repair raises this to 94.2%** across 80 real-world OpenAPI
  contracts. Across **116 official MCP servers, 88.6% are fully/partially REST-backed and 92%
  implement tools as bare API wrappers** — i.e., most production tool integrations are thin
  spec-derived wrappers, not bespoke logic. MCP servers expose a **median 19% of available
  operations** following predictable patterns; spec-driven filtering/regrouping **cuts median tool
  count per API by one-third**. (verification agent `af26ecec7ecde2034`; primary source)

**OpenAPI-driven function calling in production frameworks.**
- **Google ADK** auto-generates callable tools from an OpenAPI v3.x spec: each operation yields a
  `RestApiTool`, `FunctionDeclaration`s are derived from `operationId`/parameters/request body, and
  cross-cutting config (auth) is applied once at the toolset level and propagated — "eliminates the
  need to manually define individual function tools for each API endpoint," so an endpoint change is
  a **spec edit, not a code rewrite**. (Google vendor docs; agents `a935db199be0b605b`, `aacb2868511451cba`)
- **samchon/openapi** converts Swagger v2.0 / OpenAPI v3.0 / v3.1 into LLM function-calling schemas
  with type-level validation, normalizing versions via a single intermediate format and targeting
  **multiple LLM vendors (OpenAI, Claude, Qwen, Llama)** — "one declarative spec → many model
  targets." Notably it reports that **deterministic validation feedback raises tool-call success
  from 70% (1st attempt) → 98% (2nd attempt with feedback), never failing by the 3rd**, and "98%+
  success rates in real-world LLM applications." (single-maintainer lib — verify maturity; agents
  `a27bbd146a713d8a9`, search snippet `af42c3cd63e67c857`)

**Config-as-data, with its limits.**
- **Microsoft 365 declarative agents**: "create sophisticated AI-powered solutions through
  configuration rather than custom code"; the configuration approach "**eliminates the need for
  custom infrastructure development and maintenance**." Developers control only a limited declarative
  surface (instructions, knowledge/data sources, API plugins) while Microsoft owns the
  orchestrator/model — a clean partition of change-surface (config) vs fixed core. **But** the
  declarative layer is **OpenAPI-bound** (poor fit for non-OpenAPI specs / on-prem APIs) and
  **cannot do chained/looped multistep operations** ("This architecture isn't suitable for complex
  multistep operations") — an explicit flexibility-for-simplicity trade. (MS Learn docs;
  verification agent `aa4344c1bcd734584`, search agent `af42c3cd63e67c857`)

**Spec-driven development discipline (practitioner).**
- "Spec-driven development decouples the specification (what/why) from the implementation (the
  code)," with the spec as the persistent source of truth across sessions and across different AI
  agents. Critically: "Keeping the specification as the source of truth … is what makes the approach
  robust" — patching code directly instead of updating specs degrades robustness; outdated specs
  produce broken builds when code is generated from them (a self-enforcing authority mechanism).
  (blog, 2026-05-12 + Towards Data Science; agents `aa71c7b6c1b3d1476`, search via `af42c3cd63e67c857`)

---

## (3) Neuro-Symbolic / Hybrid Patterns — Where Production Places the Boundary

The consistent finding: the boundary is drawn by **accuracy-criticality / reliability**, not by task
sophistication. Reliability-critical paths stay deterministic and configurable; flexible paths go to
the LLM.

**Aggregate production evidence.**
- **ZenML, "What 1,200 Production Deployments Reveal about LLMOps in 2025"**: across 1,200 real
  deployments, a recurring pattern is **using deterministic checks and traditional ML to validate,
  constrain, or gate LLM behavior**, with a systematic movement of **safety logic OUT of prompts and
  INTO infrastructure**. (search agent `a36b85b8bde449203`)
  The extractor on the same source (agent `a4a7343a449dcf1f0`, recovered — *not in journal*) pulled
  concrete production exemplars:
  - **Komodo Health** (healthcare): "the LLM has zero knowledge of authentication and authorisation,
    which are handled entirely by the APIs" — security-critical logic kept entirely outside the LLM.
  - **Databook "tool masking"**: a **configuration layer between agents and tool handlers** exposes
    only task-relevant fields ("a mask might only reveal the 3 fields relevant to a particular task"),
    reducing coupling/rework when upstream tools change — config-driven decoupling in practice.
  - **Ramp policy agent "autonomy slider"**: combines LLM decisions with **deterministic, configurable
    rules** (dollar limits, vendor blocklists, category restrictions) — the deterministic constraint
    surface is config-driven, not hardcoded.
  - **Reliability:** "architectural approaches like session tainting, dual-layer permissions, and
    API-based authorisation provide guarantees that prompt engineering cannot."
  - **Failure cost of NOT bounding loops:** **GetOnStack** costs "escalated from $127 weekly to
    $47,000 over four weeks due to an infinite conversation loop between agents," then six weeks
    building queues/circuit-breakers/cost controls.
  - "Better models shift where the engineering challenges lie, but they don't eliminate them" —
    config-driven harnesses/constraints/verification remain necessary regardless of model gains.

**Formalized boundary patterns.**
- **"Blueprint First, Model Second: A Framework for Deterministic LLM Workflow"** (arXiv 2508.02721,
  Aug 2025; Qiu et al.): a deterministic blueprint/workflow as the stable backbone, with the LLM
  invoked only where judgment is needed — "AI for discovery, deterministic code for execution." The
  blueprint "remain[s] stable even when underlying models or implementations change, supporting
  maintainability through configuration rather than code modification," and decoupling lets tools be
  substituted "without … modifying learned components." (search snippet via agent `a36b85b8bde449203`;
  partial PDF extraction via interrupted agent `ac34c60b26b5dae47` — **quantified results were NOT
  recovered before the kill** `[quantified results unverified in transcripts]`)
- **Microsoft Azure AI Foundry, "Three Tiers of Agentic AI"**: describes "**deterministic supervisor
  with agentic specialists**" (predictable routing on top, bounded LLM autonomy inside scopes) and
  "**LLM planning with deterministic execution**" — the LLM handles the "what," code handles the
  "how." (vendor, framework-agnostic heuristics; agent `a36b85b8bde449203`)
- **SYNAPSE** (MDPI, neuro-symbolic structural engineering): LLM for NL understanding/decomposition,
  **deterministic external algorithms for safety-critical calculations**, reaching **94% accuracy at
  <2s latency** — evidence the deterministic core need not sacrifice accuracy. (agent `a36b85b8bde449203`)
- **Agentic AI survey** (arXiv 2510.25445): symbolic/hybrid approaches dominate safety-critical
  domains (healthcare, robotics); pure-neural thrives in data-rich adaptive domains; recurrent
  **"LLM generates candidates, symbolic checker validates"** planner/checker pattern. (agent `a36b85b8bde449203`)
- **Praetorian** (engineering blog, product-flavored): mission-critical flows need deterministic
  same-input→same-output behavior; an LLM-only architecture "cannot solve hallucination/drift/
  context-poisoning," and deterministic execution gives auditability (every decision traces to a
  line of code) for compliance. (agent `a36b85b8bde449203`)

---

## (4) LLM as an Adaptation / Schema-Matching Layer

Evidence that LLMs can *absorb* tool/schema change (auto-mapping, self-healing, codegen) — with
accuracy caveats.

**Schema/entity matching.**
- **Matchmaker** (OpenReview 18E2ZooCte): a compositional, **self-improving zero-shot LLM program**
  (candidate generation → refinement → confidence scoring) for schema matching — absorbs schema
  differences without per-schema deterministic resolver code; confidence scoring is the drift/
  correctness lever. (agent `a5ab3e775f8f048ba`)
- **GRAM** (arXiv 2406.01876): generative retrieval-augmented schema matching, **~88.7% mean accuracy
  vs ~75.3% for prior learning-based methods**. The same agent flags pairing it with a GPT-4
  prompting study (**F1=0.58; majority voting cuts hallucination 24%→8%**) to show variance and the
  need for ensembling. `[The F1=0.58 / 24%→8% study is referenced but its title/URL is not separately
  recovered in the transcripts — source unverified in transcripts.]` (agent `a5ab3e775f8f048ba`)
- **Automatic End-to-End Data Integration using LLMs** (arXiv 2603.10547): an LLM (named "GPT-5.2" in
  the snippet) auto-generates schema mappings, value-normalization mappings, entity-matching training
  data, and conflict-resolution validation data — replacing hand-written integration glue. (preprint,
  verify benchmarks; agent `a5ab3e775f8f048ba`)

**Connector / spec codegen (build-time).**
- **LRASGen** (arXiv 2504.16833): LLM auto-generates **OpenAPI specs** (the declarative change-surface)
  from code/docs — directly addresses "how much can LLMs cut config-authoring cost." (agent `a5ab3e775f8f048ba`)
- **AutoMCP** (see §2): connector authoring absorbed by codegen, 76%→94.2% with auto-repair.
- **SAGAI-MID** (arXiv 2603.28731): generative-AI **runtime middleware** mediating between evolving
  services — LLM as self-healing integration layer rather than hard-coded connectors. (preprint,
  verify maturity; agent `a5ab3e775f8f048ba`)

**Self-healing — the honest counterweight.**
- A **systematic review of AI-powered software-testing tools** (arXiv 2409.00411) assesses
  self-healing test locators (the canonical "LLM absorbs UI/schema change" pattern) including
  **limitations**, explicitly offered as the counterweight to vendor "70–90% maintenance reduction"
  claims. (agent `a5ab3e775f8f048ba`)

> **Net read of §4:** LLM adaptation layers post strong accuracy (GRAM ~88.7%) but **non-trivial
> variance** (F1=0.58 without ensembling); they reduce authoring cost but introduce a *new* drift/
> correctness risk that must itself be gated deterministically (cf. confidence scoring, majority
> voting, validation feedback).

---

## (5) Where to Draw the Line — Industry Heuristics (change-frequency × accuracy-criticality)

- **Anthropic, "Building Effective AI Agents"** (canonical practitioner source): distinguishes
  deterministic **"workflows"** (LLMs+tools through predefined code paths; prioritize reliability/
  predictability) from **"agents"** (LLM directs control flow; more flexible, less reliable). Core
  guidance: find the simplest solution; add agentic complexity only when justified; prefer composable
  patterns over frameworks (which add debugging-hostile abstraction). (agent `aacb2868511451cba`)
- **deepset, "A Spectrum, Not a Binary Choice"**: partition by criticality — deterministic paths for
  critical operations, LLM autonomy where flexibility beats strict reliability; cites Harrison Chase's
  heuristic ("if the LLM can change control flow, it's an agent"). (vendor — sells Haystack; agent `aacb2868511451cba`)
- **Stack Overflow Blog, "Reliability for Unreliable LLMs"** (2025-06-30): "**real determinism comes
  not from generation but from acceptance/validation**" — engineer deterministic control loops
  (eval, retry, validation) around the LLM; **constrain/validate steps that touch state, money, or
  trust; allow looseness in presentation/exploration**. This is the operational boundary rule. (agent `aacb2868511451cba`)
- **Google Developers Blog, "5 Lessons from Refactoring a Monolith"**: specialized narrow-task agents
  beat one massive multi-step prompt; **shift the contract from fuzzy NL to a runtime-validated typed
  object (Pydantic)** to guarantee structural integrity and eliminate brittle custom parsing —
  directly supports "LLM does NL→config-conformant formalization, deterministic core validates."
  (agent `aacb2868511451cba`)
- **Deepchecks, "How Prompt Updates Drive Most Incidents"**: the LLM/prompt surface is itself a
  high-maintenance change-cost liability — "three words added to improve 'conversational flow'
  spiked structured-output error rates within hours and halted revenue workflows"; prompt changes
  are mostly incremental, undocumented, and accumulate untracked. (vendor eval tooling, concrete
  incident; agent `aacb2868511451cba`)

**Heuristic partition that emerges:** `stable + critical → deterministic (config-driven); volatile +
surface → LLM`. Critically, **both surfaces carry change-cost**: deterministic glue is brittle to
provider drift (§1), and the LLM/prompt surface is brittle to undocumented prompt edits (Deepchecks).
The win is moving the *critical* change-surface into **diffable, versionable, typed config** and
**validating the LLM surface deterministically**.

---

## KEY SYNTHESIS — A Cost-Aware Equilibrium

**Proposed split (supported by the recovered evidence):**

```
[ Generic deterministic core ]      ← stable, reusable runtime/orchestrator/validator
        + 
[ Declarative config = the change-surface ]   ← OpenAPI/JSON-Schema/DSL/typed contracts; diffable, versionable
        +
[ LLM at runtime: NL → config-conformant formalization ]   ← the "what", validated by the core
        +
[ LLM at build-time: generate/maintain config from tool docs/OpenAPI ]  ← cuts authoring cost
```

**Does making the deterministic part config-driven buy BOTH high accuracy AND low change-cost?**
The evidence says **yes, substantially** — with a measurable residual cost.

1. **Low change-cost is well-evidenced.** Tool change becomes a config edit, not a code rewrite:
   ADK/M365/samchon turn endpoint changes into spec edits; AutoMCP collapses 1.5k–704k SLOC of
   hand-maintained glue into machine-generated ~0.7k SLOC with *zero maintainers*; PayPal's DSL
   reports 76% faster modifications and <50 vs 500+ lines. The §1 baseline (the "permanent tail":
   10–20% of dev time, $50k–$150k/integration/yr, 160–480 hr/yr) is precisely what config-driven
   decoupling attacks.

2. **High accuracy is preserved when the core validates.** Deterministic validation feedback drives
   tool-call success 70%→98% (samchon); neuro-symbolic SYNAPSE hits 94% accuracy keeping safety-
   critical math deterministic; the dominant production pattern is "LLM proposes, deterministic
   layer validates/gates" (ZenML 1,200 deployments; Stack Overflow; planner/checker survey). So
   config-driven determinism does **not** force an accuracy sacrifice — provided the LLM's output is
   *checked against the config*, not trusted raw.

3. **Boundary placement is the lever, not model quality.** Money/auth/state paths stay deterministic
   and config-bounded (Komodo zero-auth-knowledge; Ramp autonomy slider; Databook tool masking); the
   GetOnStack $127→$47k loop is the cost of *omitting* deterministic bounds. "Better models shift but
   don't eliminate" the need for the harness.

**Residual cost of authoring/maintaining the config — and how much LLMs cut it.**
- The change-surface does **not vanish**; it relocates to the config/spec, which must be kept
  authoritative. Spec-driven discipline warns: patching code instead of the spec silently degrades
  robustness; the mitigation is making the spec generative-authoritative (stale spec → broken build).
- LLMs cut the authoring cost on both sides: build-time spec generation (LRASGen), spec→server
  compilation with auto-repair (AutoMCP **76%→94.2%**), and schema/value mapping generation (E2E data
  integration). But the residual is real: **AutoMCP still needs auto-repair to reach 94.2% (not
  100%)**; M365's declarative layer is **OpenAPI-bound and can't express multistep/looped logic**;
  LLM schema matching carries variance (GRAM ~88.7% but a referenced study at F1=0.58 needing majority
  voting). So LLMs reduce — but do not eliminate — config authoring/verification cost, and **whatever
  the LLM produces (config at build-time, formalized calls at runtime) must pass a deterministic
  check** to keep the accuracy guarantee.

**Bottom line.** A *generic deterministic core + declarative config as the sole change-surface +
LLM doing NL→config formalization (runtime) and config generation from docs (build-time)* is the
equilibrium the recovered production and academic evidence points to. It gets both high accuracy and
low change-cost **iff** (a) the config is typed/diffable/authoritative and (b) the core
deterministically validates every LLM output. The irreducible residual is config authorship/upkeep,
which LLMs cut by a large but bounded margin (best recovered number: spec-driven generation 76% →
94.2% with auto-repair), never to zero, and never without a deterministic acceptance gate.

---

## Citations Recovered (verifiable identifiers literally present in transcripts)

**arXiv / academic:**
- arXiv **2507.16044** — *From REST to MCP: An Empirical Study of API Wrapping and Automated Server
  Generation for LLM Agents* (AutoMCP). (publishDate 2025-07-21, rev 2026-04-06)
- arXiv **2512.19769** — *A Declarative Language for Building And Orchestrating LLM-Powered Agent
  Workflows* (PayPal). (2025-12-22)
- arXiv **2508.02721** — *Blueprint First, Model Second: A Framework for Deterministic LLM Workflow*,
  Qiu, Ye, Gao, Zou, Chen, Gui, Huang, Xue, Qiu, Zhao. (Aug 2025)
- arXiv **2510.25445** — *Agentic AI: A Comprehensive Survey of Architectures, Applications, and
  Future Directions.*
- arXiv **2012.11951** — *Robotic Process Automation — A Systematic Literature Review and Assessment
  Framework.* (2020-12-22)
- arXiv **2603.10547** — *Automatic End-to-End Data Integration using Large Language Models.* (preprint)
- arXiv **2504.16833** — *LRASGen: LLM-based RESTful API Specification Generation.*
- arXiv **2409.00411** — *AI-powered software testing tools: A systematic review and empirical
  assessment of their features and limitations.*
- arXiv **2603.28731** — *SAGAI-MID: A Generative AI-Driven Middleware for Dynamic Runtime
  Interoperability.* (preprint)
- arXiv **2406.01876** — *GRAM: Generative Retrieval Augmented Matching of Data Schemas.* (~88.7% acc)
- OpenReview **18E2ZooCte** — *Matchmaker: Self-Improving Large Language Model Programs for Schema
  Matching.*
- MDPI **2075-5309/16/3/534** — *A Neuro-Symbolic Framework for Deterministic Reliability in
  AI-Assisted Structural Engineering: The SYNAPSE Architecture.*

**Engineering blogs / vendor docs / practitioner:**
- ZenML — *What 1,200 Production Deployments Reveal About LLMOps in 2025*
  (zenml.io/blog/what-1200-production-deployments-reveal-about-llmops-in-2025). (2025-12-19)
- Anthropic — *Building Effective AI Agents* (anthropic.com/research/building-effective-agents).
- Stack Overflow Blog — *Reliability for Unreliable LLMs* (2025-06-30).
- Google Developers Blog — *Production-Ready AI Agents: 5 Lessons from Refactoring a Monolith.*
- Google ADK docs — *OpenAPI tools* (google.github.io/adk-docs/tools-custom/openapi-tools/).
- Microsoft Learn — *Declarative agent architecture* (Copilot extensibility).
- Microsoft Azure AI Foundry — *Three Tiers of Agentic AI — and When to Use None of Them.*
- deepset — *AI Agents and Deterministic Workflows: A Spectrum, Not a Binary Choice.*
- Deepchecks — *How Prompt Updates Drive Most Incidents.*
- Praetorian — *Deterministic AI Orchestration: A Platform Architecture for Autonomous Development.*
- Ampersand — *The Integration Debt Trap: Why Building Integrations In-House Breaks Down at Scale*
  (withampersand.com/blog/why-building-integrations-in-house-breaks-at-scale).
- Airbyte — *How to Build API Integrations That Don't Break* (airbyte.com/agentic-data/api-integrations).
- Duvo — *Why Every RPA Project Breaks (And How Agentic AI Fixes It)*
  (blog.duvo.ai/why-every-rpa-project-breaks-and-how-agentic-ai-fixes-it).
- Truto — *Building Integrations In-House and Other Horror Stories.*
- DEV Community (denlava) — *Reducing API Test Brittleness: Strategies Against Minor Schema Changes.*
  (2023-04-03)
- Towards Data Science — *From Vibe Coding to Spec-Driven Development.*
- (blog, 2026-05-12) spec-driven development case study with Claude Code (title/URL not separately
  recovered) `[partial identifier]`.
- GitHub **samchon/openapi** — OpenAPI converters + LLM function-calling schema composer.

**Referenced inside transcripts but not independently sourced (flagged):**
- HfS Research RPA TCO figure (25–30% licensing / 70–75% maintenance) — cited *via* the Duvo/[`a0ba0ceb`]
  vendor blogs, not directly. `[secondary citation]`
- "14.78% breaking changes across API releases" — attributed to "academic research" in a blog, no
  primary citation recovered. `[source unverified in transcripts]`
- GPT-4 schema-matching prompting study (F1=0.58; majority voting 24%→8% hallucination) — referenced
  by the schema-matching search agent without a recovered title/URL. `[source unverified in transcripts]`

---

*End of salvage synthesis. Coverage: 19/20 transcripts contributed; 1 (Blueprint-First PDF
extractor) interrupted with only partial WebFetch content recovered.*
