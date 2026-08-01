# Who Owns the Interpretation? A Boundary Criterion for Language Models in Ontology- and Database-Backed Agents

**Status: draft v0.1 (2026-08-01). Framework paper. The core mechanism (§7) is proposed and cost-analyzed but not yet evaluated end-to-end; §10 states exactly what is measured and what is not.**

---

## Abstract

Agents that act over a database under a written policy are usually built by putting a language model in front of deterministic machinery and then arguing, case by case, about which decisions the model should be allowed to make. We give a criterion instead, and we derive it from the classical conditions under which a database behaves like a logical theory at all. Reiter identified three: domain closure, unique names, and the closed-world assumption. We observe that these do not have the same status in a deployed agent. Domain closure and the closed-world assumption are *stipulations the policy is entitled to make* — if a policy enumerates the excluded merchants, that enumeration is the excluded set, and nothing outside the document contradicts it. The unique-name assumption is different: whether the string `Target - Eco Collection` in a transaction record denotes the entity the policy calls `Target` is not something any policy author has authority to declare, because it is a fact about language use rather than a rule. This asymmetry localizes the irreducible contribution of the language model. The deterministic layer owns the *theory* — the axioms and the inference over them. The language model owns the *interpretation* — the denotation of constants and the context-dependent application of predicates. Three mature fields reached this conclusion independently and long ago: record linkage has been probabilistic since 1969, ontology alignment has never been deterministic, and word-sense disambiguation saturates at annotator agreement. We then treat the case where interpretation is contested rather than merely uncertain, and here the asymmetry pays a second dividend. Because the obstacle is a missing declaration rather than missing information, it can be repaired by obtaining one: asking an interlocutor who holds authority over the contested question converts an open predicate into a closed one at run time, which is the same move the policy makes at authoring time. Authority is question-specific, so this yields a three-tier rule — ask where the interlocutor has standing, fall back where they do not, escalate the residue to the policy author. The classical semantics for disjunctive closed-world reasoning then enters in an unexpected role. Rather than deciding what to block, it decides **what to ask about**: the actions on which the readings disagree are exactly the set difference between enforcing under some reading and under every reading, and that difference is the agenda requiring a declaration. This assignment matters because the error tolerances differ — as an enforcement rule the criterion must be exact, since a wrong verdict blocks a legitimate action, whereas as a trigger a wrong verdict costs one unnecessary question, so a cheap over-approximation suffices and the worst-case complexity stops being load-bearing. It also repairs a known gap: frontier models detect ambiguity when prompted but almost never raise a clarifying question on their own, and computing the disagreement set in the engine makes the decision to ask deterministic rather than something the model must remember to do. An audit of six deployed guardrail frameworks finds that none of them can express the situation, because their policy languages are conjunctive or Horn by construction. We report measurements of how closed real tool schemas actually are — enumerated arguments make up 21–23% of parameters in user-submitted schemas, against 3% in synthetic benchmarks — and we state plainly that the enforcement mechanism itself remains to be evaluated.

---

## 1. Introduction

A deployed tool-use agent sits between three things: a natural-language request, a written domain policy, and a database. The engineering question that recurs at every design meeting is which decisions may be handed to deterministic code and which must be left to the model. The usual answers are procedural — a list of mechanisms that have worked, or a taxonomy of requirement types — and they do not compose, because they say nothing about *why* a given decision falls on one side.

We propose that the boundary is not a matter of taste and can be derived. The derivation starts from a question that was settled forty years ago in a different context: under what conditions does a database behave like a logical theory?

Reiter's answer names three assumptions. **Domain closure** says the individuals mentioned are all the individuals there are. **Unique names** says distinct names denote distinct individuals. The **closed-world assumption** says a relation holds only where the database says it does. With all three, a first-order theory has exactly one model, and that model is the database table [Reiter 1984, §3.1, Theorem 3.1].

Our observation is that these three do not survive contact with a deployed agent in the same way, and the pattern of survival is informative.

**Domain closure and the closed-world assumption are stipulations, and the policy is entitled to make them.** If a banking policy states that purchases at Target, Amazon, and Walmart are excluded from the elevated reward rate, then those are the excluded merchants. There is no external fact of the matter that the policy is failing to describe; the policy *constitutes* the rule. A merchant absent from the list is simply not excluded, and if the policy changes next quarter the axiom changes with it. Reiter anticipated exactly this. Asked when a domain should be treated as closed, he declined to give a test: the question, he wrote, is "clearly a pragmatic question" [Reiter 1980, p. 241], and he listed naturally closed domains — inventories, flight schedules — rather than a criterion.

**The unique-name assumption is not like this.** A transaction record carries `merchant_name = "Target - Eco Collection"`. The policy says `Target`. Are these the same entity for the purpose of the exclusion? No policy author declared an answer, and none could have: the question is whether two pieces of language pick out the same thing, which is a fact about usage, not a rule the author is empowered to lay down. In our banking domain the knowledge base makes the intended reading discoverable — it remarks that eco-labeled purchases *from the excluded general retailers* still earn the standard rate — but discovering that requires reading prose and deciding that one phrase covers another. That is a semantic judgment, and it is the one thing in the pipeline that no stipulation removes.

This gives the criterion. **The deterministic layer owns the theory; the language model owns the interpretation.** In logical terms an interpretation has exactly two parts — what the constants denote and where the predicates hold — and these correspond exactly to the two things practitioners already use language models for in this setting: resolving which entity a string refers to, and deciding whether a predicate applies in this context. Nothing else about the system requires a model. Once the interpretation is fixed, inference, counting, enforcement, and auditing are deterministic and should be.

The paper develops this in five steps. §3 establishes the asymmetry from primary sources. §4 states the split and its consequences for architecture. §5 shows that three separate research communities reached the same conclusion decades ago, which both supports the claim and bounds its novelty. §6–7 treat the harder case in which the interpretation is not merely uncertain but *contested*, and show that the classical repair for disjunctive closed-world reasoning applies, at a cost we compute. §8 audits what deployed systems currently do. §9 reports what we have measured, and §10 states what we have not.

**Contributions.**
1. A criterion for the language-model/deterministic boundary derived from the closure assumptions, together with the asymmetry argument that localizes the irreducible model contribution to interpretation (§3–4).
2. The observation that the two halves of an interpretation — constant denotation and context-dependent predicate application — correspond to three mature fields that independently abandoned deterministic solutions, which we use as convergent evidence and as a novelty bound (§5).
3. A treatment of *contested* interpretation as a three-tier rule keyed on **who holds authority over the contested question** — ask, else fall back, else escalate (§6.1) — together with the reassignment of minimal-model semantics from an enforcement rule to an **ask-trigger**: the disagreement set is the approval agenda, the error tolerance of a trigger is weaker than that of an enforcement rule, and computing it in the engine makes the decision to ask deterministic (§6.2). The fallback semantics is identified as EGCWA rather than GCWA, with a complexity analysis showing the practical regime is cheap (§6.3–7).
4. An audit of six deployed guardrail frameworks establishing that none can express epistemic disjunction in its policy language (§8).
5. Measurements of the closure of real tool schemas: enumerated parameters are 21–23% of user-submitted schemas against 3% of synthetic ones, with a median enumeration size of three to four (§9).

---

## 2. Setting

We consider an agent with three inputs: a user request in natural language, a policy document in natural-language prose, and a database with a fixed schema. The agent emits tool calls. A *gate* is deterministic code that inspects a proposed tool call and either permits it or refuses it with feedback.

We use one running example throughout, drawn from a banking domain in a public tool-agent benchmark. The database records transactions with a `merchant_name` field; the policy prose designates certain merchants as excluded from an elevated reward rate; the agent must compute rewards correctly. The domain contains 138 distinct merchant strings.

Two failure directions matter and they are not symmetric. **False enforcement** blocks an action the policy permits — the agent is prevented from doing something legitimate, and from the user's side the system is simply broken. **Under-enforcement** permits an action the policy forbids. In a compliance setting the second is the one the gate exists to prevent, but the first is the one a gate *introduces*, and it is the direction that a mistaken closure assumption produces.

---

## 3. The asymmetry

### 3.1 What Reiter established, from the primary text

The three assumptions appear together on the abstract page of the 1984 chapter, in a form close to how we have paraphrased them: domain closure as "the individuals occurring in the database are all and only the existing individuals," unique names as "individuals with distinct names are distinct," and the closed-world assumption as "the only possible instances of a relation are those implied by the database" [Reiter 1984, p. 191]. Section 3.1 gives a definition in *iff* form — a first-order theory is a relational theory of a relation exactly when it contains the domain-closure axiom, the unique-name axioms, the equality axioms, and the completion axioms — and Theorem 3.1 establishes the corresponding unique-model equivalence [p. 209–210].

It is important for our purposes that the conditions in that *iff* are conditions **on the syntax of a theory**: they ask whether the theory contains certain axioms. They do not ask whether a domain is in fact closed. Theorem 3.1 is a faithfulness result relating a proof-theoretic and a model-theoretic presentation, not a test of applicability. The chapter contains no procedure for checking that the assumptions hold of a domain; the closest it comes is a remark that a logical specification is "open for inspection" by a skeptical reader [p. 230] — that is, human inspection. Indeed §2.4 recasts domain closure and the closed-world assumption as the database's epistemic stance rather than as claims about the world, which removes the object of verification altogether.

Where the assumptions fail, Reiter's diagnosis is always internal to the theory. Adding disjunctive information to a relational theory makes it inconsistent [p. 214]; null values are mishandled because the closed-world assumption converts ignorance into falsity [p. 218]; and in §4.2.4 he rejects his own 1978 formulation as "unsuitable" on both grounds [p. 227–228]. But the prescription that follows is to *repair the axioms* — "some representation of the closed world assumption is necessary" [p. 228] — never to stop treating the domain as closed. The phrase "open world" does not occur anywhere in the chapter.

### 3.2 Why two assumptions are stipulable and one is not

A policy document is a legislative instrument, not a descriptive one. When it enumerates excluded merchants it is not attempting to report a pre-existing set and possibly failing; it is constituting the set. Domain closure and the closed-world assumption are therefore available to be adopted, and adopting them is not an epistemic risk but a choice with a known consequence: merchants the policy does not name are not excluded.

This choice is also the safe one. Enforcing only over named entities can under-enforce relative to an evaluator's expectations, but it cannot produce false enforcement, because every block traces to an explicit line of policy.

The unique-name assumption is not available on the same terms, for a structural reason. It concerns co-reference between symbols originating in *different* artifacts: a phrase written by a policy author and a string produced by a transaction system. Neither party has authority over the relation between them, and no third document declares it. The relation is a fact about how language is used in the domain, and it must be *judged*.

Reiter's 1980 paper carries the equality machinery corresponding to this assumption — equality axioms and the notion of an *E-saturated* database, in which the inequality between every pair of distinct constants is explicitly present. Our domain is not E-saturated: `Target` and `Target - Eco Collection` are distinct constants that, for the purpose of the exclusion, denote the same retailer.

### 3.3 The residual, stated precisely

Adopt the two stipulable assumptions. What remains is not a completeness problem — the list is complete by construction — but a **membership** problem: given an observed string, does it denote one of the named entities? A closed list does not close its own membership relation.

Our census of the banking domain locates this residual concretely. Of 138 merchant strings, nine have a leading token appearing in an exclusion context while the full string appears nowhere in the corpus, and three groups share a leading token across distinct merchants, two of them inside the exclusion context:

```
target -> {Target, Target - Eco Collection}
dell   -> {Dell,   Dell Technologies}
delta  -> {Delta Airlines, Delta Sky Club}      (outside exclusion context)
```

---

## 4. The split

**The deterministic layer owns the theory. The language model owns the interpretation.**

An interpretation of a first-order language assigns denotations to constants and extensions to predicates. These correspond to the two roles the model must play:

| Component | Question | Why not deterministic |
|---|---|---|
| Constant denotation | Which entity does this string refer to? | Co-reference across artifacts; no authority declares it (§3.2) |
| Predicate application | Does this predicate hold here? | The deciding context feature may itself be open |

Everything else belongs to the engine. Inference over the axioms, membership checks against a declared identification table, counting, enforcement, and audit are all closed operations, and they should be closed because closing them is what makes the system's behavior explicable.

Two qualifications keep the claim honest.

**Interpretation is not only identity.** Before any entity is resolved, the model must decide which predicate or tool a request concerns at all. This is formalization, and it is part of interpretation in the same sense — assigning meaning to symbols — but it is a distinct axis from co-reference and should not be collapsed into it.

**Context-dependence is not automatically open.** A predicate whose truth varies with context can still be deterministic if the deciding context feature is itself a closed field. `excluded_if(transaction_type == 'eco')` is engine work. The model is required only when the deciding feature is not readable from the schema.

**The architectural consequence.** If the model owns interpretation and the engine owns theory, then the interface between them is a *declaration*: the model states its interpretation, with grounds, as data; the engine consumes the declaration and reasons deterministically over it.

```
model    proposes an interpretation, with grounds
         "Target - Eco Collection denotes the policy's Target"    (identity claim)
   ↓     surfaced as a declaration, not applied silently
engine   consumes the declaration as data
         membership check · gate decision · ledger reconciliation
```

The model never performs inference; the engine never performs interpretation. This also explains a discipline that is otherwise merely prudent: interpretation cannot be verified, but it can be *surfaced*, and once surfaced it can be measured. A silent interpretation is an unauditable premise underneath a fully auditable proof.

**And it bounds the guarantee.** Everything the engine proves is relative to an interpretation it did not check. SATLM states the same limit for solver-aided reasoning, guaranteeing correctness only "with respect to the parsed specification" [Ye et al. 2023]. The engine has no means of detecting that its inputs were misinterpreted, because it takes them as true. §6 is our answer to this exposure.

---

## 5. Three fields already concluded this

The claim that co-reference and context-dependent predicate application resist deterministic solution is not new. It is the founding premise of three research communities, which is convergent support and also a bound on what we may claim.

**Record linkage / entity resolution.** Deciding whether two records describe the same entity has been formulated probabilistically since Fellegi and Sunter's 1969 theory, and the field has never had a deterministic solution. This is the unique-name assumption as an engineering discipline. *[grade: memory-based; citation to be verified before submission]*

**Ontology alignment and schema matching.** Deciding when symbols from two vocabularies denote the same thing has likewise always been heuristic or learned. *[grade: memory-based; to be verified]*

**Word-sense disambiguation.** Deciding which sense a word carries in context is the predicate half. Recent work shows that scale has pushed overall performance to the level of supervised systems — a Llama ladder rises from 56.3% at 1B to 81–82% at frontier scale — while genuinely fine-grained boundary cases remain flat at roughly 45%, and expert humans remain about nine points above the best model [Meconi et al. 2025]. The most striking result for our purposes is that the *same* model reaches up to 98% when allowed to describe the sense in free form rather than select from the inventory: the bottleneck is not comprehension but the forced alignment onto a finite label space.

This last point sharpens §4. The load our architecture must manage is not that the model fails to understand; it is that projecting an understood meaning onto a closed symbol set destroys information, and the destroyed part is not recoverable downstream.

---

## 6. When the interpretation is contested

§4 leaves an exposure: a wrong interpretation silently invalidates everything proved on top of it. In the important cases the model's uncertainty is not diffuse but *structured* — there are two defensible readings, and the choice between them is what is at stake.

`Target - Eco Collection` is such a case. Under one reading it denotes the policy's `Target` and the exclusion applies; under the other it is a distinct merchant and the exclusion does not. Both readings are defensible from the document.

Three responses are available, and they are not alternatives of equal standing.

- **Credulous** — block if the action is forbidden under *some* reading. This blocks on the authority of a single reading and is therefore a false-enforcement generator.
- **Skeptical** — block only if the action is forbidden under *every* reading. False enforcement becomes structurally impossible: any block is licensed by all readings simultaneously. The cost is under-enforcement in exactly the contested cases.
- **Ask** — put the contest to an interlocutor and enforce on the answer.

### 6.1 Asking is the stronger device, where authority is present

Skeptical enforcement is a *concession*: it discards the contested case because the system cannot settle it. Asking does not concede — it **settles** the contest, and then full enforcement resumes.

The reason asking is available at all follows from §3.2. The unique-name assumption is unstipulable because no party to the artifacts has authority over co-reference. But a conversation contains a party the documents do not: the interlocutor. Where that party holds authority over the contested question, asking **converts an open predicate into a closed one by obtaining the missing declaration**. This is the same move the policy makes for domain closure — someone with standing declares — relocated from authoring time to run time.

The qualification is the whole content of the rule, because authority is question-specific:

| Contested question | Does the present interlocutor have authority? | Response |
|---|---|---|
| "Which of these two transactions did you mean?" | **Yes** — the customer knows which purchase they made | **Ask** |
| "Is *Target - Eco Collection* covered by the *Target* exclusion?" | **No** — a customer has no standing over policy scope | **Skeptical**, and escalate |

This yields a three-tier rule rather than a binary choice:

1. **Ask**, when the interlocutor has authority over the contested question. Resolves the interpretation; enforcement then proceeds normally.
2. **Skeptical**, when they do not. Under-enforces, but cannot manufacture a false refusal.
3. **Escalate**, offline, to the party who *does* have authority — the policy author. A contested identity that recurs is a defect in the policy document, and the run-time record of skeptical fallbacks is precisely the defect report.

Tier 3 matters more than it looks. Skeptical enforcement is a *holding pattern*, not a resolution: it keeps the system safe while the ambiguity persists, and the ambiguity is supposed to be fixed upstream. A system that only ever falls back is one that never reports.

Asking is also not free, and the constraint is not compute. Every question spends a turn and some of the user's patience, and over-asking has a documented failure mode: CaMeL warns that frequent security prompts produce user fatigue, in which users become desensitized and approve reflexively — at which point the ask has become a rubber stamp and the guarantee is gone. Asking therefore has a budget, which is why tiers 2 and 3 exist rather than asking about everything.

The empirical case for tier 1 is strong and is not ours: clarification lifts task performance by roughly eight to twenty points with the model held fixed, across question answering, code generation, and agent planning. The obstacle is not capability but initiative — frontier models detect ambiguity at 60–80% when asked to, yet raise a clarifying question in almost no naturally occurring case, and on the tool-dialogue action of requesting a missing parameter GPT-4o scores 13.7 F1 against 44.8 for a fine-tuned small model. Knowing when to ask is a separate, learnable competence from knowing the answer, and it does not arrive with scale.

### 6.2 What the minimal-model machinery is actually for

There is a natural assumption to make at this point, and it is wrong: that the multiple-reading semantics should be the *enforcement* rule — block exactly what every reading forbids. That is coherent, but it spends the machinery on the one job it is worst suited to, and it inherits the under-enforcement cost for no reason.

The better use is as a **trigger**. Consider the two extremes as sets of actions:

```
credulous(P)  = forbidden under SOME reading
skeptical(P)  = forbidden under EVERY reading
```

Their difference, `credulous(P) \ skeptical(P)`, is exactly the set of actions the readings **disagree** about. That set is not a nuisance to be resolved by fiat in one direction or the other. It is the **agenda of things that require a declaration** — precisely the questions that should be put to whoever has authority (§6.1), and, when nobody present does, the queue that should be escalated to the policy author.

So the minimal-model semantics answers *what needs approval*, not *what to block*.

**Why this is the right assignment: the error tolerance is different.** As an enforcement rule, the criterion must be sound, because a wrong verdict blocks a legitimate action or admits a forbidden one — errors are harmful, so the computation must be exact, and exactness is what costs. As a trigger, a wrong verdict costs one unnecessary question, or one missed question that falls through to the tier-2 fallback — errors are merely *expensive*. A cheap, sound-but-incomplete over-approximation of disagreement is therefore entirely adequate, and the worst-case complexity of exact entailment (§7) stops being load-bearing. We keep §7 because the exact computation is affordable in this domain anyway, but the design no longer depends on it.

**This also repairs the initiative problem.** §6.1 established that clarification works and that models nonetheless almost never initiate it — detection at 60–80% when prompted, spontaneous asking at close to zero. Prompting a model to ask more is treating a structural gap as a behavioural one. If the disagreement set is computed by the engine, **the decision to ask becomes deterministic**: the model supplies the candidate readings, the engine detects that they diverge, and the engine raises the question. The model is never asked to notice that it is uncertain, which is the thing it demonstrably does not do. This is the same division as everywhere else in §4 — the model interprets, the engine decides what follows — applied to the act of asking.

**The composed rule.** Disagreement detection produces the candidate set; the authority test of §6.1 partitions it:

```
contested = credulous(P) \ skeptical(P)          ← engine computes
   ├── interlocutor has authority  → ASK, then enforce on the answer
   ├── interlocutor does not       → skeptical fallback (permit), and
   └──                                ESCALATE to the policy author
```

Tier 2 therefore covers only the residue that cannot be asked about, which is a far smaller set than the whole contested zone, and the under-enforcement it costs shrinks accordingly.

One limit worth stating: the trigger must be budgeted. If the contested set is large, asking about all of it recreates the fatigue problem of §6.1 from the other direction. In our domain it is small (§3.3), but ranking and a per-session budget are required in general, and we have not designed them.

### 6.3 The classical repair for the tier-2 fallback

The naive closed-world assumption is inconsistent on disjunctive information: given only `Pa ∨ Pb`, it derives `¬Pa` and `¬Pb`, contradicting the disjunction. Minker's Generalized Closed World Assumption repairs this by negating an atom only when it is absent from *every* minimal model [Minker 1982]. Its primary definition is proof-theoretic — negate `A` when every derivable disjunction `A ∨ K` has `K` already derivable — with the minimal-model characterization established as an equivalent.

A naming point matters here, because the wrong choice is the more expensive one. The semantics we want — *enforce only what holds in all minimal models* — is not GCWA but **EGCWA**, which is exactly `EGCWA(P) = MM(P)` [Eiter & Gottlob 1995, §4]. GCWA restricts its conclusions to atoms and negated atoms and is a proper fragment; its entailment problem for arbitrary formulas is not even known to be complete for the class that EGCWA is complete for, the best published upper bound being one level higher. The hierarchy is: naive CWA ⊂ GCWA ⊂ EGCWA ⊂ ECWA, with ECWA equivalent to propositional circumscription.

### 6.4 What the tier-2 fallback is, in one line

Skeptical enforcement is a **soundness-preserving under-approximation of the policy**. It never blocks something the policy permits under some defensible reading; it may permit something the policy forbids under some reading. Establishing this formally, and measuring the reduction in false enforcement that it buys and the under-enforcement that it costs, is the central empirical claim this framework licenses — and it is not yet done (§10).

---

## 7. Cost

The worst-case complexity of minimal-model reasoning is a reasonable thing to fear, and it is the wrong thing to fear here.

| Problem | Restriction | Class |
|---|---|---|
| EGCWA / ECWA / circumscription entailment | general | Π₂ᵖ-complete |
| `MM(T) ⊨ x`, positive | general | **coNP-complete — same as classical entailment** |
| CWA = GCWA = EGCWA | Horn | **P** |
| minimal model finding/checking | head-cycle-free | **P** |
| skeptical ASP | backdoor of size *k* to normality | **FPT, O(2ᵏn²)** |
| GCWA | first-order with function symbols | Π⁰₂-complete — **undecidable** |

Three observations make the practical case.

**Most enforcement conditions never touch the machinery.** If the enforced condition contains no negation, `MM(T) ⊨ F` iff `T ⊨ F` [Eiter & Gottlob 1993, §3.1]: minimal models are unnecessary and the cost is classical entailment, linear for Horn. Compliance conditions of the form "this must hold before that" fall here.

**Where negation is enforced, the parameter that matters is small.** With *K* disjunctive statements of at most *d* disjuncts, the whole set of minimal models is obtainable by enumerating at most `d^K` Horn least-model computations. In our banking domain, §3.3 gives **K = 2 or 3 and d = 2** — four to eight models. Policies are also static, so this is precomputation, not per-request work. *(This enumeration bound is our own derivation; for citation the published fixed-parameter result [Fichte & Szeider 2013] should be used instead.)*

**Three independent escape hatches exist if K grows.** Head-cycle-free programs admit polynomial minimal-model finding — and DLV detects this fragment at runtime; 2CNF theories drop a level to coNP; and backdoor or treewidth parameterizations give fixed-parameter tractability.

We add one caution against over-claiming. Competition results show structural second-level problems solved comfortably, but the organizers describe the non-head-cycle-free track as sparsely populated, random disjunctive instances still time out, and confirmed industrial uses are head-cycle-free and therefore first-level. The claim "disjunction is fine in practice" should be stated as conditional on that fragment. Two further traps: grounding a first-order policy over a large domain can blow up the propositional program even when its complexity class is benign, and minimal-model *checking* is coNP-complete, so a design that verifies a candidate model is worse than one that enumerates.

---

## 8. What deployed systems do

We audited six agent-guardrail frameworks, reading the policy-language grammar in each. The distinction that organizes the result:

- **D1, truth-functional disjunction**: `a OR b` evaluated over fully known concrete arguments. A single model settles it; no minimal-model machinery is involved.
- **D2, epistemic disjunction**: the policy itself is disjunctive and the engine does not know which disjunct holds. This is where §6 applies.

| System | Policy language | D1 | D2 | Default |
|---|---|---|---|---|
| AgentSpec | DSL + Python predicates | **no** — the grammar has no disjunction operator; conditions are conjunctions | no | deny-listed |
| Progent | DSL over JSON Schema | yes (`or`, `anyOf`) | no | default-deny |
| CaMeL | Python callbacks (DSL declined) | yes (host language) | no | allow-listed, escalates to user |
| GuardAgent | natural language → generated Python | no (attribute-value conjunctions) | no | deny-listed |
| NeMo Guardrails / Colang | dialogue-flow DSL | yes (`or when`) | no | open-world; the model fills gaps |
| Symbolic guardrails [Hong et al. 2026] | none — checks in tool implementations | not discussed | no | allow-listed |

**No system handles D2.** The terms *minimal model*, *skeptical*, *closed-world assumption*, *over-* and *under-approximation* do not appear in any of the six. Two systems implement relevant notions without naming them: Progent's default-deny is a closed-world semantics, and CaMeL's strict/normal interpreter modes are a choice between over- and under-approximating data-flow dependencies. Every framework assumes the policy has one determinate reading and evaluates a single model.

This is not an oversight so much as a consequence of language design: a policy language that is conjunctive or Horn by construction *cannot express* the situation, so the question never arises.

We must also record a prior claim on the problem statement. PolicyBank names our failure mode precisely — "imprecise quantifiers or exemplar lists that the agent interprets as exhaustive" — and reports the resulting rejection of legitimate edge cases. Its remedy is a structured natural-language memory with feedback, not a formal semantics; the paper contains no treatment of disjunction or minimal models. The correct framing of our contribution is therefore not that we identify the problem but that we give it an enforcement semantics with a guarantee, where prior work mitigates it heuristically.

---

## 9. Measurements

Two of these are ours, produced by deterministic scripts over public artifacts and reproducible; the third is from the literature.

**How closed are real tool schemas?** Parsing the Berkeley Function Calling Leaderboard v4 schemas:

| Source | Parameters with an enumeration | Median enumeration size |
|---|---|---|
| BFCL *live* (user-submitted) | **21.5–23.1%** | 3–4 |
| BFCL synthetic categories | **2.9–3.8%** | 3 |
| Schema-Guided Dialogue (categorical slots) | 24.7% | 3 |
| MultiWOZ 2.2 | 34.4% | 6 |

Two readings. Closed predicates are roughly a quarter of the surface and, when closed, are very small — the same answer from two fields twenty years apart. And synthetic benchmarks under-represent the closed fraction by nearly an order of magnitude, which matters for anyone calibrating a guardrail design on them.

**Are declared enumerations actually exhaustive?** Comparing BFCL gold answers against the declared `enum` for the corresponding parameter: in `simple_python`, all 48 gold values fall inside the declared set; in `live_simple`, 120 of 140 fall inside and the remaining 20 are the `"N/A"` sentinel used when an argument is omitted — **zero genuine violations**. Exhaustiveness is checkable, and here it checks out. We are not aware of a published instance of this check.

**Where the residual sits.** The banking census of §3.3: 138 merchant strings, nine with an exclusion-context head whose full string is absent from the corpus, three leading-token collision groups of which two are inside the exclusion context.

---

## 10. What is and is not established

**Established here.** The asymmetry among the three closure assumptions, read from primary sources (§3). The interpretation/theory split as its consequence (§4). The identification of the appropriate semantics for contested interpretation as EGCWA rather than GCWA, with the complexity regime computed (§6–7). The audit result that no deployed guardrail language expresses epistemic disjunction (§8). The schema-closure measurements (§9).

**Proposed but not evaluated.** The skeptical gate itself. We have not implemented it, not measured the false enforcement it removes, not measured the under-enforcement it introduces, and not run it end to end. The soundness-preserving-under-approximation property of §6.2 is stated but not proved. Every claim about what the mechanism *achieves* is therefore a design claim.

**Boundaries of the claim.** The criterion is derived for agents over a database under a written policy; we do not claim it transfers to open-domain assistants without such a document. The banking evidence is one domain. The complexity argument assumes a grounded propositional policy — first-order with function symbols is undecidable — and the small-*K* regime is measured in one domain only.

**Citations requiring verification before submission.** The record-linkage and ontology-alignment claims in §5 are stated from memory and flagged. Minker's 1982 text and Lifschitz's 1985 paper were not obtained; the GCWA definitions here rest on four independent secondary sources that agree, and the equivalences are attributed accordingly. Two attributions we have already had to correct and note here to prevent their recurrence: the reduction of GCWA to CWA on Horn theories is due to Shepherdson, not Minker, and Lifschitz's "Computing Circumscription" is a different paper that does not treat the closed-world assumption at all.

**A boundary we must not blur.** Closed predicates in description logics [Lutz et al. 2015] share our vocabulary but not our lineage: they *fix* a predicate's extension to the data rather than minimizing it, and their bibliographies cite none of the closed-world tradition. The term must be cited to avoid collision, and the two lines must not be joined.

---

## 11. Related work

**The closure assumptions.** Reiter established the three-assumption reconstruction [1984] and the equality and domain-closure results [1980]; our contribution is not the list but the observation that its members have different modal status in a deployed system, and the consequence for architecture.

**Disjunctive closed-world reasoning.** Minker's GCWA [1982] and its successors — EGCWA [Yahya & Henschen 1985], CCWA [Gelfond & Przymusinska 1986], ECWA [Gelfond et al. 1989], and the weakened WGCWA [Rajasekar et al. 1989] — supply the semantics; complexity is due to Eiter and Gottlob [1993, 1995] and Cadoli and Lenzerini [1990]. Answer-set programming absorbed the computational layer, and minimality is now invisible inside solvers. We contribute an application, not a semantics.

**Neurosymbolic allocation.** PAL and Program-of-Thought delegate execution to an interpreter; LLM-Modulo separates a generator from sound external critics; Logic-LM, SATLM, and Faithful CoT separate parsing from solving. What these leave open is the criterion — LLM-Modulo assumes a sound critic exists without saying when one can. AlphaGeometry comes closest to our conditions, arguing that a symbolic engine can own the part whose deduction closure is exhaustively derivable while the model owns the part with infinite branching, but presents this as a description of geometry rather than a general principle.

**Agent guardrails.** AgentSpec, Progent, CaMeL, GuardAgent, NeMo Guardrails, and the symbolic-guardrail study of Hong et al. are audited in §8. Hong et al. is the closest neighbor and must be conceded on one point: that symbolic offload can be applied broadly without sacrificing utility is already published, with artifacts, on the airline domain of the same benchmark family. Their study does not measure whether an enumeration is exhaustive, does not measure over-blocking, and reaches its classification through an informal match against a six-mechanism catalog whose inter-annotator agreement is not reported.

**The problem statement.** PolicyBank has priority on the failure mode itself (§8).

---

## 12. Conclusion

Asking which decisions a language model may make is the wrong question, because it has no principled answer. Asking which parts of a logical system a policy is entitled to stipulate has one, and it turns out to divide the work cleanly. A policy may declare what exists and what is true — those are its business. It cannot declare that two pieces of language refer to the same thing, because that is not a rule anyone is empowered to make. What is left over is exactly an interpretation: what the names denote and where the predicates hold. That is the model's job, and by the same argument nothing else is.

Two consequences follow that are not obvious from the starting point. Because interpretation cannot be verified, it must be declared and measured rather than applied silently — the model's semantic commitments belong in the record, not in the gaps. And because interpretations are sometimes contested rather than merely uncertain, the enforcement layer needs a semantics for disagreement; enforcing only what survives every reading is the one choice that cannot manufacture a false refusal. That semantics was worked out for databases in 1982, it is affordable at the scale contested identities actually occur, and no deployed agent framework can currently express the situation it addresses.

---

## References

*(Working list. Entries marked ‡ have been verified against primary text; † bibliography verified, body not obtained; ✗ flagged for verification before submission.)*

- ‡ Reiter, R. Equality and Domain Closure in First-Order Databases. *JACM* 27(2):235–249, 1980.
- ‡ Reiter, R. Towards a Logical Reconstruction of Relational Database Theory. In *On Conceptual Modelling*, Springer, 1984, pp. 191–238.
- † Minker, J. On Indefinite Databases and the Closed World Assumption. *CADE-6*, LNCS 138:292–308, 1982.
- † Yahya, A. & Henschen, L. Deduction in Non-Horn Databases. *JAR* 1(2):141–160, 1985.
- † Gelfond, M. & Przymusinska, H. Negation as Failure: Careful Closure Procedure. *AIJ* 30(3):273–287, 1986.
- † Gelfond, M., Przymusinska, H. & Przymusinski, T. On the Relationship Between Circumscription and Negation as Failure. *AIJ* 38(1):75–94, 1989.
- ‡ Eiter, T. & Gottlob, G. Propositional Circumscription and Extended Closed World Reasoning are Π₂ᵖ-complete. *TCS* 114(2):231–245, 1993.
- ‡ Eiter, T. & Gottlob, G. On the Computational Cost of Disjunctive Logic Programming. *AMAI* 15(3–4):289–323, 1995.
- ‡ Cadoli, M. & Lenzerini, M. The Complexity of Closed World Reasoning and Circumscription. *AAAI-90*.
- † Shepherdson, J.C. Negation in Logic Programming. In Minker (ed.), *Foundations of Deductive Databases and Logic Programming*, Morgan Kaufmann, 1988, pp. 19–88.
- † Lobo, J., Minker, J. & Rajasekar, A. *Foundations of Disjunctive Logic Programming*. MIT Press, 1992.
- † Fichte, J. & Szeider, S. Backdoors to Normality for Disjunctive Logic Programs. *AAAI 2013*.
- ‡ Meconi et al. Do Large Language Models Understand Word Senses? arXiv:2509.13905, 2025.
- ‡ Ye, X. et al. SatLM: Satisfiability-Aided Language Models. *NeurIPS 2023*.
- ‡ Lutz, C., Seylan, İ. & Wolter, F. Ontology-Mediated Queries with Closed Predicates. *IJCAI 2015*.
- ‡ Hong, Y. et al. Don't Make Models Guess Security and Safety: Symbolic Guardrails for Domain-Specific AI Agents. arXiv:2604.15579, 2026.
- ‡ PolicyBank. arXiv:2604.15505, 2026.
- ‡ Wang, H. et al. AgentSpec. arXiv:2503.18666. · ‡ Progent. arXiv:2504.11703. · ‡ Debenedetti, E. et al. CaMeL. arXiv:2503.18813. · ‡ GuardAgent. arXiv:2406.09187. · ‡ NeMo Guardrails. arXiv:2310.10501.
- ✗ Fellegi, I. & Sunter, A. A Theory for Record Linkage. *JASA*, 1969.
- ✗ Ontology alignment / schema matching survey — to be selected and verified.
