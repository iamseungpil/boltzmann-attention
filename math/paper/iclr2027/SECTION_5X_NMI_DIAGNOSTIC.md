# §5.X — Facet Orthogonality as a Dataset Diagnostic

*(Standalone draft for merge into PAPER_DRAFT_ICLR_v1.md. No OISA/AFOD
terminology — patent-pending and deliberately excluded. Uses only public
references: Bickel & Scheffer 2004, Vinh et al. 2010, Ranganathan 1933.)*

## Motivation

§5.6 (Phase C) reported that our position-pooled B_ont achieves the
same direction-specificity (attn_fro ratio 2.45) regardless of whether
catalog semantic content is preserved, perturbed within facet,
permuted across tool names, or replaced with random text. The F1
reframe concluded that semantic content is not load-bearing and the
K-subspace at catalog-specified token positions is the effective
ceiling.

An alternative mechanistic explanation for the same observation is that
our candidate "4-facet ontology" (function_action, io_type, domain,
tool_category) was **not actually 4-dimensional**. If the facets were
collinear to begin with, Gram-Schmidt over them would collapse to a
lower-rank span, and permuting within-facet labels would preserve span
by mathematical necessity — independently of whether semantic content
is load-bearing in principle.

We test this hypothesis with a Normalized Mutual Information
(NMI, Vinh et al. 2010) probe on seven public tool-selection
benchmarks spanning N=23 to N=618 unique tools.

## Protocol

For each benchmark, we compute four independent clusterings of the
tool catalog:

- **A. verb** — action verb extracted from tool description or name
  (search / get / translate / book / ...)
- **B. co-occurrence** — connected components on the graph whose
  edges connect tools that appear together in ≥ 2 ground-truth task
  annotations
- **C. parameter-signature Jaccard** — connected components on
  Jaccard ≥ 0.5 over required-argument name sets
- **D. domain / category** — explicit category label when available
  (StableToolBench `category_name`, AppBench `used_app`), otherwise
  NLP-inferred domain from description keywords

Labels are strings; we compute pairwise NMI as

$$\mathrm{NMI}(X, Y) = \frac{I(X; Y)}{\sqrt{H(X)\, H(Y)}} \in [0, 1].$$

Per-pair verdict uses the following thresholds (aligned with the
faceted-classification literature's informal independence criterion):
NMI < 0.3 → orthogonal, 0.3–0.5 → soft-orthogonal, > 0.5 → redundant.

## Benchmarks

| Benchmark | N_tools | N_queries | Scope |
|---|---:|---:|---|
| τ²-bench telecom | 43 | 2,285 | single domain |
| MetaTool | 388 | 995 + 497 | multi-domain (plugins) |
| StableToolBench G1 (tool-instruction) | 499 | 158 | multi-domain (RapidAPI) |
| TaskBench (HF) | 23 | 7,458 | multi-domain (ML subfields) |
| TaskBench (daily-life) | 40 | 4,318 | narrow daily-life APIs |
| AppBench | 25 | 801 | multi-app |
| C3-Bench | 618 | 256 | multi-domain (OpenAI schema) |

## Results

**Orthogonality counts (of 6 pairs per benchmark):**

| Benchmark | ORTH | soft | RED | Orthogonal axis (NMI) |
|---|:---:|:---:|:---:|---|
| τ²-bench telecom | 0 | 1 | 5 | — |
| **MetaTool** | **1** | 0 | 5 | verb × domain (0.185) |
| **StableToolBench** | **1** | 2 | 3 | verb × category_name (0.218) |
| **TaskBench HF** | **1** | 1 | 4 | verb × ml-subfield (0.144) |
| TaskBench daily-life | 0 | 0 | 6 | — |
| AppBench | 5 | 0 | 1 | *(degenerate, see caveat)* |
| **C3-Bench** | **1** | 2 | 3 | verb × domain (0.159) |

**Headline observation.** Four of four broad-scope multi-domain
benchmarks yield exactly one genuinely orthogonal pair, always
`verb × domain`, with NMI tightly bounded in
$[0.144, 0.218]$ — a 23 to 618 tool range, four independent dataset
authors, four different catalog-construction methodologies.

**Negative controls.** The two narrow-scope benchmarks (single-domain
τ²-telecom and narrow daily-life TaskBench) yield zero orthogonal
pairs, as predicted by the "multi-domain → orthogonal facet"
hypothesis. The AppBench outcome is measurement-artifact: its API
names use compound no-whitespace tokens (`findtrains`,
`reserverestaurant`) that our verb regex fails to split, collapsing
the verb clustering to a single class and producing trivially zero
NMI values.

**Consistent redundancies.** `B_cooccurrence × C_param_jaccard` lies
in $[0.76, 0.99]$ across all benchmarks, confirming that tools that
co-invoke in tasks share parameter schemas (shared entity types). In
practice these two views should be merged into a single axis.

## Implications

**For our prior Phase C result (§5.6).** τ²-telecom is the single
benchmark in our evaluation suite with zero orthogonal facet pairs.
Our "4-facet ontology" on this corpus therefore has effective
independent-axis count $\approx 1$, rather than 4. The previously
reported invariance of `attn_fro` under label permutation is
consistent with (but does not uniquely support) the F1 reframe:
the K-subspace ceiling hypothesis operates at one mechanistic level,
while facet-collinearity operates at another, and both yield the
same observable prediction on a single-domain dataset. Disentangling
the two requires a multi-facet dataset.

**For multi-facet ontology construction.** The `verb × domain`
decomposition is the empirically supported two-axis choice for any
downstream B_ont construction on multi-domain tool corpora. This is
the only decomposition that survives redundancy filtering on any of
the four multi-domain datasets we tested.

**For benchmark selection.** An NMI probe below the ~0.3 threshold
on at least one pair is a cheap (CPU, seconds) precondition for
claiming that a "multi-facet ontology" delivers gains on a given
corpus. Benchmarks that fail this test cannot meaningfully test the
multi-facet hypothesis at all.

## Caveats

1. **AppBench degeneracy** can be fixed with a camelCase splitter but
   was not in this pilot; we report it as measurement artifact.
2. **Small-N stability.** TaskBench HF (N=23) is the smallest
   orthogonal-pair benchmark; NMI at small N is known to be unstable
   (Vinh et al. 2010 §6). We flag this but do not treat it as
   confounding because three larger benchmarks (MetaTool 388,
   StableToolBench 499, C3-Bench 618) independently replicate.
3. **Single-pass labelling.** Our verb and domain extractors use
   rule-based NLP (regex + keyword voting). More sophisticated
   clustering (sentence-BERT + HDBSCAN, LLM-as-judge) might reveal
   additional orthogonal axes. Scope-wise, the rule-based probe is
   a lower bound on orthogonal-pair count.

## Artifact list (for reproducibility, will be moved to Appendix)

- NMI probe code: ~290 lines Python, CPU, ~30 seconds for full
  7-dataset sweep.
- Outputs: `reports/new_theorem_test/phase_f8_afod/nmi_*.json`,
  including labels and distributions per benchmark.
- All seven benchmarks are publicly available (links in Appendix A).

## Connection to §5.6 (Phase C) and §5.Y (F9 accuracy)

If the NMI probe gives zero orthogonal pairs on τ²-telecom (this
section) and our Phase C permutation test (§5.6) cannot distinguish
real from permuted B_ont on τ²-telecom, then the next natural test is
whether an NMI-verified orthogonal two-facet B_ont (verb × domain)
improves downstream tool-selection accuracy on a multi-domain corpus
(MetaTool). That is the subject of §5.Y (F9 accuracy experiment).
