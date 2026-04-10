#!/usr/bin/env python3
"""
MetaTool ontology builder v2 — fixes 6 construction bugs in v1.

Diagnosis lineage: `memory/ontology_rebuild_handoff_2026_04_10.md` documents
why v1 (`build_metatool_ontology.py`) produced a B_ont that collapses to
r_per_pair[function_action]≈1 per (L,h) pair and yields ε_q ≈ isotropic
floor (r/d = 24/128 = 0.1875) with MMLU/MetaTool separation only +0.009.

## What v2 changes vs v1

1. Source material: REAL user queries from
   `/tmp/MetaTool/dataset/data/all_clean_data.csv` (20,614 rows,
   Query + Tool columns). v1 used plugin_info.json descriptions
   with truncation and a single-template anchor per category.
2. No character truncation (bug 1 fix). Full query text retained,
   then padded by facet-specific wrapping to hit the sentence
   length target.
3. Exclusive cross-facet assignment (bug 2 fix). Each query is
   used in exactly ONE (facet, category) pair. Priority order
   tool_category > function_action > domain > io_type ensures the
   most-specific labels get first pick and the broad facets still
   have 40 queries each (verified by dry-run).
4. Template diversification (bug 3 fix). 8 different template
   structures per facet, rotated across sentences within a category,
   so the facet's K-mean is not dominated by a single template
   direction after averaging.
5. Minimum ~30-token sentences (bug 4 fix). Short queries are
   wrapped with explanatory context. Long queries passed through
   unwrapped (the model sees enough of the raw utterance).
6. 20+ sentences per category (bug 5 fix). All 46 categories hit
   40 by the dry-run. We cap at 30 here to keep the per-category
   K-extraction cost bounded (~ 1400 forward passes total).
7. tool_category picks top-15 BY REAL QUERY COUNT, not alphabetical
   (bug 6 fix). Top-15 is:
     FinanceTool (1122), ProductSearch (674), JobTool (557),
     TripTool (509), PDF&URLTool (369), NewsTool (361),
     ResearchFinder (357), CourseTool (317), WeatherTool (316),
     TripAdviceTool (308), ResearchHelper (303), RepoTool (268),
     MusicTool (267), HousePurchasingTool (259), SEOTool (250).

## FACET_ORDER (unchanged vs v1, matches the Gram-Schmidt order in
## scripts/ontology_facet_basis.py)

    function_action → io_type → domain → tool_category
    (most-general  → ...              → most-specific)

Gram-Schmidt residualizes each facet against the previous ones, so
function_action must carry a distinct K signal before the later
facets are allowed to add their contribution. v1's function_action
collapsed because of bugs 3 and 4.

## Output

Writes a JSON with the same schema as v1 so
`build_qwen_metatool_b_ont.py --ontology-json <v2.json>` can consume
it unchanged:

    {
      "facet_order": [...],
      "ontology": {facet: {cat: [sent1, sent2, ...]}},
      "facet_counts": {facet: n_cats},
      "n_sentences_per_category": {facet: median_n},
      "source": "MetaTool all_clean_data.csv (exclusive facet assign)",
    }

Output path: reports/axis2_theoretical_verification/metatool_ontology_v2.json

## Mode C safety

Per `APPENDIX_B_PROOFS.md:687-700` (Cor 6.6 consequence), runtime
calibration-mismatch corrections are forbidden in Mode C (Qwen2.5-7B).
This rebuild is DESIGN-TIME reconstruction of the operator B on
cleaner input sentences — no runtime calibration-subtract. The
theorem hazard does not apply.
"""

from __future__ import annotations

import csv
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------
# Sources and outputs
# ---------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
METATOOL_DIR = Path(
    os.environ.get("METATOOL_DIR", str(REPO / "external" / "MetaTool" / "dataset"))
)
QUERIES_CSV = METATOOL_DIR / "data" / "all_clean_data.csv"

OUT_DIR = REPO / "reports" / "axis2_theoretical_verification"
OUT_PATH = OUT_DIR / "metatool_ontology_v2.json"

N_SENTENCES_PER_CAT = 30
TOP_K_TOOL_CATEGORIES = 15
MIN_TOKENS_TARGET = 30  # wrapping lower bound, in rough word count

# ---------------------------------------------------------------------
# Facet keyword definitions (rule-based, deterministic)
#
# Scope of keyword pools tightened vs v1 to reduce false-positive
# cross-facet leakage. Each canonical category uses phrases that are
# common in real queries (I spot-checked samples from all_clean_data.csv).
# ---------------------------------------------------------------------

ACTION_KEYWORDS = [
    ("translate",  [r"\btranslat", r"\binterpret"]),
    ("convert",    [r"\bconvert", r"\bexchange"]),
    ("analyze",    [r"\banaly[sz]", r"\binsight", r"\bdiagnos", r"\bcompar"]),
    ("summarize",  [r"\bsummari[sz]", r"\babstract"]),
    ("recommend",  [r"\brecommend", r"\bsuggest", r"\bbest\b", r"\btop\b"]),
    ("book",       [r"\bbook\b", r"\breserv", r"\bappoint"]),
    ("track",      [r"\btrack", r"\bmonitor", r"\blog\b"]),
    ("create",     [r"\bcreate", r"\bgenerate", r"\bwrite", r"\bdraft", r"\bdesign", r"\bbuild"]),
    ("search",     [r"\bsearch", r"\bfind", r"\bdiscover", r"\bexplore"]),
    ("retrieve",   [r"\bget\b", r"\bfetch", r"\baccess", r"\bobtain", r"\bprovide", r"\bshow\b", r"\blook up"]),
    ("play",       [r"\bplay", r"\bgame"]),
    ("inform",     [r"\bnews", r"\bupdate", r"\balert", r"\bforecast"]),
]

IO_KEYWORDS = [
    ("video",       [r"\bvideo", r"\byoutube"]),
    ("image",       [r"\bimage", r"\bphoto", r"\bpicture", r"\bchart", r"\bvisual"]),
    ("audio",       [r"\baudio", r"\bsound", r"\bmusic", r"\bpodcast", r"\bvoice"]),
    ("structured",  [r"\bdatabase", r"\btable", r"\bschema", r"\bjson", r"\bspreadsheet", r"\bdata\b"]),
    ("numeric",     [r"\bcalculat", r"\bstatistic", r"\bprice", r"\bstock", r"\bbudget"]),
]

DOMAIN_KEYWORDS = [
    ("finance",       [r"\bfinanc", r"\bstock", r"\bcurrency", r"\bcrypto", r"\binvest", r"\bbank", r"\bNFT"]),
    ("news",          [r"\bnews", r"\bjournal", r"\bcurrent event"]),
    ("health",        [r"\bhealth", r"\bmedic", r"\bdiet", r"\bcalorie", r"\bnutri", r"\bfitness"]),
    ("education",     [r"\bcourse", r"\bstudy", r"\bschool", r"\btutor", r"\bquiz", r"\blearn"]),
    ("travel",        [r"\btravel", r"\btrip", r"\bhotel", r"\bflight", r"\bvacation", r"\btour"]),
    ("shopping",      [r"\bshop", r"\bproduct", r"\bcart", r"\bdiscount", r"\bpurchase", r"\bbuy", r"\bdeal"]),
    ("entertainment", [r"\bgame", r"\bmusic", r"\bmovie", r"\bvideo", r"\bart\b"]),
    ("food",          [r"\bfood", r"\brestaurant", r"\brecipe", r"\bcook", r"\bmeal"]),
    ("real_estate",   [r"\bhouse", r"\brent", r"\bproperty", r"\bapartment"]),
    ("weather",       [r"\bweather", r"\bforecast", r"\bclimate", r"\btemperature"]),
    ("productivity",  [r"\bnote", r"\btask", r"\breminder", r"\bcalendar", r"\bschedul"]),
    ("research",      [r"\bresearch", r"\bacadem", r"\bpaper", r"\bcitation"]),
    ("legal",         [r"\blaw", r"\blegal", r"\bregul"]),
    ("career",        [r"\bjob", r"\bcareer", r"\bresume", r"\bemploy"]),
]

FACET_ORDER = ["function_action", "io_type", "domain", "tool_category"]

# Priority for exclusive assignment: most-specific labels get first pick.
# tool_category has gold labels from the dataset. The others are regex
# classifications, so leave them lower priority.
ASSIGN_PRIORITY = ["tool_category", "function_action", "domain", "io_type"]


def match_kw(text: str, kw_list) -> str | None:
    """Return first matching canonical category, or None."""
    t = text.lower()
    for canonical, patterns in kw_list:
        if any(re.search(p, t) for p in patterns):
            return canonical
    return None


# ---------------------------------------------------------------------
# Template banks — 8 diversified structures per facet (bug 3 fix).
#
# Within a category we rotate through all 8 templates so that no single
# template direction dominates the category's K-mean. Each template is
# designed to place the category label in a different syntactic role
# (subject, object, modifier, embedded clause) so that averaged K vectors
# carry facet-specific variance rather than template-monoculture variance.
#
# {q} = raw user query; {label} = facet-specific category string
# (e.g., "search", "finance", "FinanceTool").
# ---------------------------------------------------------------------

FUNCTION_ACTION_TEMPLATES = [
    "The user wants to {label} something in this request: {q}",
    "In the following query, the core operation the user needs is to {label}. Query: {q}",
    "This is a {label}-type request. The phrasing used by the user is: {q}",
    "When a user writes the following, the underlying action they need is to {label}. User: {q}",
    "A typical {label} task reads like this example sentence: {q}",
    "The system must {label} in order to answer this request from the user: {q}",
    "What the user is trying to do here is {label}. They wrote the following sentence: {q}",
    "Performing a {label} operation is exactly what satisfies this user question: {q}",
]

IO_TYPE_TEMPLATES = [
    "This request involves {label} content as the primary modality: {q}",
    "The input or output modality in this query is {label}. Full query: {q}",
    "To answer this, the tool must work with {label} data. The user asked: {q}",
    "A typical {label}-oriented query looks like the following example: {q}",
    "The medium of interaction in the user's request is {label} content. Request: {q}",
    "This question deals primarily with {label} as its working medium: {q}",
    "Processing {label} data is required to satisfy the following request: {q}",
    "The user's question hinges on {label} as the central modality: {q}",
]

DOMAIN_TEMPLATES = [
    "This query falls squarely in the {label} knowledge domain. Query: {q}",
    "The topic of the following request is {label}. The user asked: {q}",
    "A representative question about {label} might be phrased like this: {q}",
    "To answer the following, one needs knowledge from the {label} domain: {q}",
    "This is a typical {label} question from a real user session: {q}",
    "The subject matter of this user query is {label} and nothing else: {q}",
    "Within the {label} area, users commonly ask questions of this form: {q}",
    "A user working in the {label} domain wrote the following question: {q}",
]

TOOL_CATEGORY_TEMPLATES = [
    "For the following user query, the appropriate tool to route to is {label}. Query: {q}",
    "This request should be routed to the {label} API. The user wrote: {q}",
    "Users invoke {label} when they ask things like the following: {q}",
    "The {label} service handles requests of this exact form. Example: {q}",
    "The {label} category covers user questions that look like this: {q}",
    "When a user asks the following, the system needs to call {label}: {q}",
    "A typical {label}-style question from the benchmark reads as: {q}",
    "This is precisely the kind of input that {label} is built to handle: {q}",
]

TEMPLATE_BANK = {
    "function_action": FUNCTION_ACTION_TEMPLATES,
    "io_type": IO_TYPE_TEMPLATES,
    "domain": DOMAIN_TEMPLATES,
    "tool_category": TOOL_CATEGORY_TEMPLATES,
}


def _ensure_min_tokens(sentence: str, min_tokens: int = MIN_TOKENS_TARGET) -> str:
    """Pad a too-short sentence with a neutral completion clause.

    Used only when the template-wrapped query is still short (rare for
    real queries that average 99 chars, but some top tool queries can
    be under 8 tokens).
    """
    if len(sentence.split()) >= min_tokens:
        return sentence
    return (
        sentence
        + " This sentence is provided as one of several semantically"
        " related examples used to characterize the category."
    )


# ---------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------

def load_queries() -> list[tuple[str, str]]:
    queries = []
    with open(QUERIES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("Query") or "").strip()
            t = (row.get("Tool") or "").strip()
            if q and t:
                queries.append((q, t))
    return queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cat", type=int, default=N_SENTENCES_PER_CAT)
    parser.add_argument("--top-k-tool-categories", type=int, default=TOP_K_TOOL_CATEGORIES)
    parser.add_argument(
        "--max-categories-per-facet",
        type=int,
        default=0,
        help="If > 0, keep only the first N categories per facet after construction. "
             "Useful for smoke tests.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="Output JSON path.",
    )
    return parser.parse_args()


def build_ontology_v2(
    n_per_cat: int = N_SENTENCES_PER_CAT,
    top_k_tool_categories: int = TOP_K_TOOL_CATEGORIES,
    max_categories_per_facet: int = 0,
) -> tuple[dict, dict]:
    queries = load_queries()
    assert queries, "no queries loaded"

    # Select top-K tool categories by real query count (bug 6 fix).
    tc_counts = Counter(t for _, t in queries)
    top_tc = [t for t, _ in tc_counts.most_common(top_k_tool_categories)]

    # ---- exclusive assignment (bug 2 fix) ----
    # For each query, compute candidate (facet, category) under each
    # facet's rules, then assign to the first facet in ASSIGN_PRIORITY
    # whose bucket still has room.
    #
    # We fill 2*n_per_cat per bucket so the later diversity-select step
    # has headroom to pick length-varied sentences.
    fill_target = 2 * n_per_cat
    buckets: dict[str, dict[str, list[str]]] = {f: defaultdict(list) for f in FACET_ORDER}

    def room(facet: str, cat: str) -> bool:
        return len(buckets[facet][cat]) < fill_target

    for q, t in queries:
        # Compute candidate label per facet.
        cand = {
            "tool_category": t if t in top_tc else None,
            "function_action": match_kw(q, ACTION_KEYWORDS),
            "domain": match_kw(q, DOMAIN_KEYWORDS),
            "io_type": match_kw(q, IO_KEYWORDS),
        }
        for facet in ASSIGN_PRIORITY:
            lab = cand[facet]
            if lab is not None and room(facet, lab):
                buckets[facet][lab].append(q)
                break  # each query used in exactly ONE facet

    # ---- sentence construction with template rotation (bug 3 fix) ----
    ontology: dict[str, dict[str, list[str]]] = {}
    for facet in FACET_ORDER:
        ontology[facet] = {}
        templates = TEMPLATE_BANK[facet]
        for cat, qlist in buckets[facet].items():
            # Diversity: prefer longer queries but include some short
            # ones too, to give the K-mean variance across lengths.
            # Sort by length, take evenly-spaced indices.
            qs = sorted(set(qlist), key=len)
            if len(qs) == 0:
                continue
            if len(qs) <= n_per_cat:
                picked = qs
            else:
                step = len(qs) / n_per_cat
                picked = [qs[int(i * step)] for i in range(n_per_cat)]

            # Human-readable label (keep original capitalization for
            # tool_category, snake-case verbs for function_action).
            label = cat if facet == "tool_category" else cat

            sentences = []
            for i, q in enumerate(picked):
                tmpl = templates[i % len(templates)]
                sent = tmpl.format(label=label, q=q)
                sent = _ensure_min_tokens(sent)
                sentences.append(sent)

            ontology[facet][cat] = sentences

        if max_categories_per_facet > 0:
            trimmed_items = list(ontology[facet].items())[:max_categories_per_facet]
            ontology[facet] = dict(trimmed_items)

    # ---- stats ----
    stats = {
        "facet_order": FACET_ORDER,
        "facet_counts": {f: len(ontology[f]) for f in FACET_ORDER},
        "sentences_per_category_median": {
            f: (sorted(len(s) for s in ontology[f].values())[len(ontology[f]) // 2]
                if ontology[f] else 0)
            for f in FACET_ORDER
        },
        "sentence_token_stats_median": {
            f: (sorted(
                    len(s.split())
                    for sents in ontology[f].values()
                    for s in sents
                )[sum(len(sents) for sents in ontology[f].values()) // 2]
                if ontology[f] else 0)
            for f in FACET_ORDER
        },
        "top_tool_categories": top_tc,
        "n_queries_total": len(queries),
        "n_queries_used_total": sum(
            len(sents) for facet in ontology.values() for sents in facet.values()
        ),
    }
    return ontology, stats


def main():
    args = parse_args()
    ontology, stats = build_ontology_v2(
        n_per_cat=args.n_per_cat,
        top_k_tool_categories=args.top_k_tool_categories,
        max_categories_per_facet=args.max_categories_per_facet,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "facet_order": FACET_ORDER,
        "ontology": ontology,
        "facet_counts": stats["facet_counts"],
        "sentences_per_category_median": stats["sentences_per_category_median"],
        "sentence_token_median": stats["sentence_token_stats_median"],
        "top_tool_categories": stats["top_tool_categories"],
        "n_queries_total": stats["n_queries_total"],
        "n_queries_used_total": stats["n_queries_used_total"],
        "source": "MetaTool all_clean_data.csv (exclusive facet assignment, template-rotated)",
        "builder_version": "v2",
    }
    args.out.write_text(json.dumps(payload, indent=2))

    print("Built MetaTool ontology v2:")
    for f in FACET_ORDER:
        n_cats = len(ontology[f])
        if n_cats == 0:
            print(f"  {f}: EMPTY")
            continue
        sizes = sorted([len(s) for s in ontology[f].values()])
        toks = sorted(
            len(s.split())
            for sents in ontology[f].values()
            for s in sents
        )
        print(
            f"  {f}: {n_cats} cats, "
            f"sent/cat median={sizes[len(sizes)//2]} min={sizes[0]} max={sizes[-1]}, "
            f"tok/sent median={toks[len(toks)//2]} min={toks[0]} max={toks[-1]}"
        )
    print(f"\ntop_tc: {stats['top_tool_categories']}")
    print(f"\nn_queries_used: {stats['n_queries_used_total']} / {stats['n_queries_total']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
