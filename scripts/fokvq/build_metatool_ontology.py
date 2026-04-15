#!/usr/bin/env python3
"""
Catalog-derived ontology builder for the MetaTool benchmark (ICLR 2024).

Constructs a 4-facet semantic ontology from the MetaTool tool catalog
that can be used as input to ontology_facet_basis.py to derive a
per-(layer, head) basis B_ont for both:
  (A) attention-time K-bias for tool selection accuracy
  (B) cache-time oc-FOKVQ categorical 1-bit quantization

**Source data**:
  /tmp/MetaTool/dataset/big_tool_des.json   — 47 high-level categories
  /tmp/MetaTool/dataset/plugin_info.json    — 390 plugin entries (name + desc)
  /tmp/MetaTool/dataset/plugin_des.json     — 199 short descriptions

**4 facets** (most-general → most-specific, matches FACET_ORDER convention):

  F1 = function_action  — what the tool DOES (action verb)
       Categories: search / retrieve / generate / translate / analyze /
                   recommend / book / play / track / convert
  F2 = io_type          — primary I/O modality
       Categories: text / structured / image / audio / video / numeric
  F3 = domain           — knowledge domain
       Categories: finance / news / health / education / entertainment /
                   travel / shopping / productivity / science / utilities
  F4 = tool_category    — MetaTool's own category label (47 values, but
                          we collapse to top-15 most populous to keep
                          basis dimension reasonable)

The ordering is critical for Gram-Schmidt residualization:
F1 (action) is most general — many tools do "search" across domains.
F4 (tool_category) is most specific — already a fine-grained label.

**Output**: a dict suitable for direct substitution into the
`ONTOLOGY` constant of ontology_facet_basis.py:

  {
    "function_action": {
      "search":     [sentence_1, sentence_2, ...],
      "generate":   [...],
      ...
    },
    "io_type": {
      "text":       [...],
      ...
    },
    ...
  }

Each facet's category is represented by a small set of canonical
sentences derived from MetaTool's actual tool descriptions plus a
synthetic completion frame to make K-extraction position-stable.

**Verb / domain / io_type assignment**:

We use a rule-based keyword matcher over each tool's
`description_for_model` field. Each tool gets exactly one (action,
io_type, domain) triple. Tools that match multiple actions/domains
go to the most-specific match per facet (the rule order encodes
specificity).

Output written to:
  reports/axis2_theoretical_verification/metatool_ontology.json
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------
# Sources and outputs
# ---------------------------------------------------------------------

METATOOL_DIR = Path("/tmp/MetaTool/dataset")
PLUGIN_INFO = METATOOL_DIR / "plugin_info.json"
BIG_TOOL_DES = METATOOL_DIR / "big_tool_des.json"
PLUGIN_DES = METATOOL_DIR / "plugin_des.json"

OUT_DIR = Path(
    "/home/woori/workspace_common/boltzmann-attention/reports/"
    "axis2_theoretical_verification"
)
OUT_PATH = OUT_DIR / "metatool_ontology.json"


# ---------------------------------------------------------------------
# Facet definitions (rule-based, deterministic)
# ---------------------------------------------------------------------

# F1: action verb. Each value is (canonical_name, [keyword regexes]).
# Order matters: first match wins. Place specific actions before generic.
ACTION_KEYWORDS = [
    ("translate",  [r"\btranslat", r"\binterpret", r"\bbabel"]),
    ("convert",    [r"\bconvert", r"\bexchange", r"\btransform"]),
    ("analyze",    [r"\banaly[sz]", r"\binsight", r"\bdiagnos"]),
    ("summarize",  [r"\bsummari[sz]", r"\babstract"]),
    ("recommend",  [r"\brecommend", r"\bsuggest", r"\btailored", r"\bpersonal"]),
    ("book",       [r"\bbook", r"\breserv", r"\bappoint"]),
    ("track",      [r"\btrack", r"\bmonitor", r"\bcount", r"\blog\b"]),
    ("create",     [r"\bcreate", r"\bgenerate", r"\bbuild", r"\bdesign", r"\bdraft", r"\bproduc"]),
    ("search",     [r"\bsearch", r"\bfind", r"\bdiscover", r"\bexplore", r"\bquery"]),
    ("retrieve",   [r"\bretriev", r"\bget\b", r"\bfetch", r"\baccess", r"\bobtain", r"\bprovide"]),
    ("play",       [r"\bplay", r"\bgame", r"\binteract"]),
    ("inform",     [r"\binform", r"\bnotif", r"\bupdat", r"\balert"]),
]

# F2: io type
IO_KEYWORDS = [
    ("video",       [r"\bvideo", r"\byoutube", r"\bfilm"]),
    ("image",       [r"\bimage", r"\bphoto", r"\bpicture", r"\bdiagram", r"\bchart", r"\bvisual", r"\bmeme", r"\bdraw", r"\bpaint"]),
    ("audio",       [r"\baudio", r"\bsound", r"\bmusic", r"\bpodcast", r"\bvoice", r"\bspeech"]),
    ("structured",  [r"\bdatabase", r"\btable", r"\bstructured", r"\bschema", r"\bjson", r"\bcsv", r"\bspreadsheet"]),
    ("numeric",     [r"\bnumeric", r"\bcalculat", r"\bmath", r"\bstatistic", r"\bprice", r"\bstock", r"\bcurrency", r"\bbudget"]),
    ("text",        [r".+"]),  # default fallback
]

# F3: domain
DOMAIN_KEYWORDS = [
    ("finance",       [r"\bfinanc", r"\bstock", r"\bcurrency", r"\bcrypto", r"\binvest", r"\bbank", r"\beconomic"]),
    ("news",          [r"\bnews", r"\bjournal", r"\bcurrent event"]),
    ("health",        [r"\bhealth", r"\bmedic", r"\bdiet", r"\bcalorie", r"\bnutri", r"\bfitness", r"\bdoctor"]),
    ("education",     [r"\beducation", r"\blearn", r"\bcourse", r"\bstudy", r"\bschool", r"\btutor", r"\bquiz"]),
    ("travel",        [r"\btravel", r"\btrip", r"\bhotel", r"\bflight", r"\bvacation", r"\btour"]),
    ("shopping",      [r"\bshop", r"\bproduct", r"\bcart", r"\bcoupon", r"\bdiscount", r"\bpurchase", r"\bbuy"]),
    ("entertainment", [r"\bentertain", r"\bgame", r"\bmusic", r"\bmovie", r"\bvideo", r"\bplay", r"\bart"]),
    ("food",          [r"\bfood", r"\brestaurant", r"\brecipe", r"\bcook", r"\bmeal"]),
    ("real_estate",   [r"\bhouse", r"\brent", r"\breal estate", r"\bproperty", r"\bapartment"]),
    ("weather",       [r"\bweather", r"\bforecast", r"\bclimate", r"\btemperature"]),
    ("productivity",  [r"\bnote", r"\btask", r"\breminder", r"\bcalendar", r"\bschedul"]),
    ("research",      [r"\bresearch", r"\bacadem", r"\bpaper", r"\bcitation"]),
    ("legal",         [r"\blaw", r"\blegal", r"\bregul", r"\bcompli"]),
    ("career",        [r"\bjob", r"\bcareer", r"\bresume", r"\bemploy"]),
    ("utility",       [r".+"]),  # fallback
]

FACET_ORDER = ["function_action", "io_type", "domain", "tool_category"]


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

def classify_tool(desc: str) -> dict:
    """Assign a (action, io_type, domain) triple to one tool description."""
    text = desc.lower()
    out = {}
    for facet, kw_list in [
        ("function_action", ACTION_KEYWORDS),
        ("io_type", IO_KEYWORDS),
        ("domain", DOMAIN_KEYWORDS),
    ]:
        for canonical, patterns in kw_list:
            if any(re.search(p, text) for p in patterns):
                out[facet] = canonical
                break
    return out


# ---------------------------------------------------------------------
# Sentence construction for Gram-Schmidt input
# ---------------------------------------------------------------------

# Each (facet, category) needs ≥3 representative sentences for stable
# K-extraction. Sentences must be syntactically diverse but
# semantically convergent on the category. We use templated sentences
# parameterized by sample tool descriptions, plus 2 hand-written
# canonical sentences per category that act as anchors.

ANCHOR_SENTENCES = {
    "function_action": {
        "translate":  [
            "This tool translates text between languages.",
            "Use this to interpret content from one language to another.",
        ],
        "convert":    [
            "This tool converts values between units or formats.",
            "Use this to transform one currency or measurement into another.",
        ],
        "analyze":    [
            "This tool analyzes data and surfaces insights.",
            "Use this to diagnose patterns in input data.",
        ],
        "summarize":  [
            "This tool summarizes long content into shorter form.",
            "Use this to create an abstract from a longer source document.",
        ],
        "recommend":  [
            "This tool recommends options based on user preferences.",
            "Use this to receive personalized suggestions.",
        ],
        "book":       [
            "This tool books reservations on the user's behalf.",
            "Use this to make an appointment or reserve a slot.",
        ],
        "track":      [
            "This tool tracks events or measurements over time.",
            "Use this to monitor something continuously.",
        ],
        "create":     [
            "This tool creates new content from scratch.",
            "Use this to generate or build a new artifact.",
        ],
        "search":     [
            "This tool searches a corpus and returns matching items.",
            "Use this to find relevant information quickly.",
        ],
        "retrieve":   [
            "This tool retrieves information from an external service.",
            "Use this to fetch and provide data on demand.",
        ],
        "play":       [
            "This tool runs an interactive game session.",
            "Use this to play a game with the user.",
        ],
        "inform":     [
            "This tool informs the user with notifications.",
            "Use this to send timely alerts and updates.",
        ],
    },
    "io_type": {
        "video":      [
            "This tool processes video content as input or output.",
            "Use this when working with video files or streams.",
        ],
        "image":      [
            "This tool processes image content as input or output.",
            "Use this when working with pictures, charts, or visual diagrams.",
        ],
        "audio":      [
            "This tool processes audio content as input or output.",
            "Use this when working with sound, music, or speech.",
        ],
        "structured": [
            "This tool processes structured data as input or output.",
            "Use this when working with tables, schemas, or database records.",
        ],
        "numeric":    [
            "This tool processes numeric data as input or output.",
            "Use this when working with calculations, prices, or statistics.",
        ],
        "text":       [
            "This tool processes text content as input or output.",
            "Use this when working with natural-language text.",
        ],
    },
    "domain": {
        "finance":       [
            "This tool operates in the financial services domain.",
            "Use this for banking, investing, or currency tasks.",
        ],
        "news":          [
            "This tool operates in the news and journalism domain.",
            "Use this for current-events queries.",
        ],
        "health":        [
            "This tool operates in the health and medical domain.",
            "Use this for fitness, nutrition, or clinical tasks.",
        ],
        "education":     [
            "This tool operates in the education and learning domain.",
            "Use this for studying, courses, or tutoring tasks.",
        ],
        "travel":        [
            "This tool operates in the travel and tourism domain.",
            "Use this for trip planning, hotels, or flights.",
        ],
        "shopping":      [
            "This tool operates in the shopping and e-commerce domain.",
            "Use this for product search, deals, or purchases.",
        ],
        "entertainment": [
            "This tool operates in the entertainment domain.",
            "Use this for games, music, movies, or art.",
        ],
        "food":          [
            "This tool operates in the food and dining domain.",
            "Use this for restaurants, recipes, or meal planning.",
        ],
        "real_estate":   [
            "This tool operates in the real-estate domain.",
            "Use this for renting, buying, or browsing properties.",
        ],
        "weather":       [
            "This tool operates in the weather and climate domain.",
            "Use this for forecasts and weather conditions.",
        ],
        "productivity":  [
            "This tool operates in the productivity and task-management domain.",
            "Use this for notes, reminders, or scheduling.",
        ],
        "research":      [
            "This tool operates in the academic research domain.",
            "Use this for searching papers or scholarly content.",
        ],
        "legal":         [
            "This tool operates in the legal and compliance domain.",
            "Use this for laws, regulations, or legal queries.",
        ],
        "career":        [
            "This tool operates in the careers and employment domain.",
            "Use this for jobs, resumes, or hiring tasks.",
        ],
        "utility":       [
            "This tool operates as a general-purpose utility.",
            "Use this for miscellaneous helper tasks.",
        ],
    },
}


def build_ontology(top_k_categories: int = 15) -> dict:
    with open(PLUGIN_INFO) as f:
        plugin_info = json.load(f)
    with open(BIG_TOOL_DES) as f:
        big_tool_des = json.load(f)

    # 1. Classify each plugin into (action, io, domain)
    classifications = []
    for plugin in plugin_info:
        desc = plugin.get("description_for_model") or plugin.get("description_for_human") or ""
        cls = classify_tool(desc)
        cls["plugin"] = plugin.get("name_for_model", "")
        cls["desc"] = desc
        classifications.append(cls)

    # 2. Pick top-K most populous tool categories from big_tool_des
    # (we use the 47 categories as F4 anchors)
    top_categories = sorted(big_tool_des.keys())[:top_k_categories]

    # 3. Build the ONTOLOGY dict in the format ontology_facet_basis.py expects
    ontology = {}
    for facet in FACET_ORDER:
        if facet == "tool_category":
            cats = top_categories
            anchors = {}
            for cat in cats:
                desc = big_tool_des[cat]
                # 2 anchor sentences per category, derived from the
                # category description and a generic frame
                anchors[cat] = [
                    f"This tool belongs to the {cat} category. {desc}",
                    f"Use the {cat} category for {desc.lower()[:80]}",
                    f"The {cat} tool family handles tasks in this category.",
                ]
            ontology[facet] = anchors
        else:
            ontology[facet] = {}
            for cat, sentences in ANCHOR_SENTENCES[facet].items():
                # Optionally enrich with sample plugin descriptions
                examples = [
                    p["desc"][:120]
                    for p in classifications
                    if p.get(facet) == cat
                ][:2]
                full = list(sentences) + [
                    f"For example: {ex}" for ex in examples if ex
                ]
                ontology[facet][cat] = full[:4]  # cap at 4 sentences per category

    return ontology, classifications


def main():
    ontology, classifications = build_ontology(top_k_categories=15)

    # Stats
    facet_counts = {f: len(ontology[f]) for f in FACET_ORDER}
    classif_action = Counter(c.get("function_action", "?") for c in classifications)
    classif_io = Counter(c.get("io_type", "?") for c in classifications)
    classif_domain = Counter(c.get("domain", "?") for c in classifications)

    print("Built MetaTool ontology:")
    for f in FACET_ORDER:
        print(f"  {f}: {facet_counts[f]} categories")
        for cat in list(ontology[f].keys())[:3]:
            print(f"    - {cat}: {len(ontology[f][cat])} sentences")
    print()
    print("Plugin classification distribution:")
    print(f"  function_action: {dict(classif_action.most_common(8))}")
    print(f"  io_type:         {dict(classif_io.most_common(6))}")
    print(f"  domain:          {dict(classif_domain.most_common(8))}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "facet_order": FACET_ORDER,
        "ontology": ontology,
        "facet_counts": facet_counts,
        "classification_distribution": {
            "function_action": dict(classif_action),
            "io_type": dict(classif_io),
            "domain": dict(classif_domain),
        },
        "n_plugins_classified": len(classifications),
        "source": "MetaTool plugin_info.json + big_tool_des.json",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote ontology to {OUT_PATH}")


if __name__ == "__main__":
    main()
