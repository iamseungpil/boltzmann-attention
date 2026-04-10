#!/usr/bin/env python3
"""
τ²-bench ontology builder.

Parses the Sierra Research τ²-bench tool definitions (Python class methods
decorated with ``@is_tool(ToolType.*)``) and emits a 4-facet ontology JSON
in the same shape consumed by ``build_qwen_metatool_b_ont.py``:

    {
        "facet_order": ["function_action", "io_type",
                        "domain", "tool_category"],
        "ontology": {
            "function_action": {cat: [sentence, ...], ...},
            "io_type":         {cat: [sentence, ...], ...},
            "domain":          {cat: [sentence, ...], ...},
            "tool_category":   {cat: [sentence, ...], ...},
        },
        ...
    }

Once the JSON is written, the B_ont build step is just the existing
``scripts/ocq/build_qwen_metatool_b_ont.py`` with two flag overrides —
NO separate ``build_tau2_b_ont.py`` is needed:

    python scripts/ocq/build_qwen_metatool_b_ont.py \\
        --ontology-json reports/axis2_theoretical_verification/tau2_retail_ontology.json \\
        --out external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt

Facet design (tau2)
-------------------
F1 = function_action  — verb from method name
        search / get / list / find / book / cancel / update / exchange /
        return / calculate / transfer / send
F2 = io_type          — primary argument/return semantics
        identifier / entity / query / transaction / confirmation / numeric
F3 = domain           — τ²-bench domain
        product / order / user / payment / inventory (retail), or
        flight / reservation / user / baggage / passenger / certificate
                                                                 (airline)
F4 = tool_category    — read from the ``@is_tool(ToolType.X)`` decorator
        read / write / generic / think

Facet order matches MetaTool's most-general → most-specific convention so
the same Gram-Schmidt residualization pipeline in
``ontology_facet_basis.build_facet_basis`` produces a clean basis.

CLI
---
    python scripts/ocq/build_tau2_ontology.py --domain retail \\
        --out reports/axis2_theoretical_verification/tau2_retail_ontology.json
    python scripts/ocq/build_tau2_ontology.py --domain airline \\
        --out reports/axis2_theoretical_verification/tau2_airline_ontology.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
TAU2_SRC = REPO / "external" / "tau2-bench" / "src" / "tau2" / "domains"

FACET_ORDER = ["function_action", "io_type", "domain", "tool_category"]


# ---------------------------------------------------------------------
# Facet rules
# ---------------------------------------------------------------------

# F1: action verb from the method name (order = specificity).
ACTION_KEYWORDS = [
    ("search",    [r"^search_", r"^find_", r"_search_"]),
    ("list",      [r"^list_"]),
    ("get",       [r"^get_"]),
    ("book",      [r"^book_", r"^create_", r"^make_"]),
    ("cancel",    [r"^cancel_"]),
    ("exchange",  [r"^exchange_"]),
    ("return",    [r"^return_"]),
    ("update",    [r"^update_", r"^modify_", r"^set_"]),
    ("send",      [r"^send_", r"^issue_"]),
    ("calculate", [r"^calculate$", r"^compute"]),
    ("transfer",  [r"^transfer_"]),
]

# F2: io_type — inferred from parameter + return-type heuristics.
# Input-side only; we use a coarse bucketing that is stable across domains.
IO_KEYWORDS_ARGS = [
    # (canonical, arg-name regexes)
    ("identifier",   [r"_id$", r"^id$", r"^reservation_id", r"^order_id",
                      r"^user_id", r"^product_id", r"^item_id"]),
    ("query",        [r"^query$", r"^name$", r"^email$", r"^expression$",
                      r"^summary$", r"^reason$"]),
    ("transaction",  [r"payment", r"amount", r"baggage", r"passenger",
                      r"cabin"]),
    ("entity",       [r"address", r"flight", r"airport", r"origin",
                      r"destination"]),
    ("confirmation", [r"confirm", r"accept", r"approve"]),
]
IO_FALLBACK = "entity"

# F3: domain category — per-tau2-domain.
DOMAIN_CATEGORIES = {
    "retail": {
        "product":   [r"product", r"item_type", r"list_all_product"],
        "order":     [r"order"],
        "user":      [r"user", r"address"],
        "payment":   [r"payment", r"gift", r"refund"],
        "inventory": [r"item", r"variant"],
    },
    "airline": {
        "flight":      [r"flight", r"airport"],
        "reservation": [r"reservation", r"book"],
        "user":        [r"user"],
        "baggage":     [r"bag"],
        "passenger":   [r"passenger"],
        "certificate": [r"certificate"],
    },
}
DOMAIN_FALLBACK = "utility"  # e.g. "calculate" / "transfer_to_human_agents"

# F4: tool_category from the decorator — ToolType values.
TOOLTYPE_MAP = {
    "READ":    "read",
    "WRITE":   "write",
    "GENERIC": "generic",
    "THINK":   "think",
}


# ---------------------------------------------------------------------
# tools.py parsing
# ---------------------------------------------------------------------

def parse_tau2_tools(tools_py: Path) -> List[dict]:
    """Return a list of dicts for each @is_tool-decorated method in
    the given tools.py file. Uses AST so decorators and docstrings are
    extracted without import-time side effects."""
    tree = ast.parse(tools_py.read_text())
    tools = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        tool_type = _is_tool_type(node)
        if tool_type is None:
            continue
        docstring = ast.get_docstring(node) or ""
        args = [a.arg for a in node.args.args if a.arg != "self"]
        tools.append({
            "name": node.name,
            "tool_type": tool_type,
            "docstring": docstring,
            "args": args,
        })
    return tools


def _is_tool_type(func: ast.FunctionDef) -> Optional[str]:
    """Return the ToolType string (e.g. 'READ') if func has an
    @is_tool(ToolType.X) decorator, else None."""
    for dec in func.decorator_list:
        # Decorator must be a Call: is_tool(ToolType.X)
        if not isinstance(dec, ast.Call):
            continue
        func_name = _dotted_name(dec.func)
        if func_name != "is_tool":
            continue
        if not dec.args:
            continue
        arg0 = dec.args[0]
        arg_name = _dotted_name(arg0)
        if arg_name and arg_name.startswith("ToolType."):
            return arg_name.split(".", 1)[1]
    return None


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

def classify_tool(tool: dict, domain: str) -> dict:
    name = tool["name"]
    args = [a.lower() for a in tool["args"]]
    docstring = tool["docstring"].lower()
    full_text = f"{name} {' '.join(args)} {docstring}"

    # F1: function_action
    action = _first_match(name, ACTION_KEYWORDS) or "other"

    # F2: io_type — vote across arg names, first matched category wins
    io = IO_FALLBACK
    for canonical, patterns in IO_KEYWORDS_ARGS:
        if any(re.search(p, a) for a in args for p in patterns):
            io = canonical
            break

    # F3: domain
    dom_cats = DOMAIN_CATEGORIES[domain]
    dom = DOMAIN_FALLBACK
    for canonical, patterns in dom_cats.items():
        if any(re.search(p, full_text) for p in patterns):
            dom = canonical
            break

    # F4: tool_category from decorator
    cat = TOOLTYPE_MAP.get(tool["tool_type"], "generic")

    return {
        "function_action": action,
        "io_type": io,
        "domain": dom,
        "tool_category": cat,
    }


def _first_match(name: str, rules: List[Tuple[str, List[str]]]) -> Optional[str]:
    lowered = name.lower()
    for canonical, patterns in rules:
        if any(re.search(p, lowered) for p in patterns):
            return canonical
    return None


# ---------------------------------------------------------------------
# Sentence construction
# ---------------------------------------------------------------------

ANCHOR_SENTENCES = {
    "function_action": {
        "search":    ["This tool searches a catalog and returns matching entries.",
                      "Use this to find items by attribute or query."],
        "list":      ["This tool lists all entries of a given type.",
                      "Use this to enumerate available options."],
        "get":       ["This tool retrieves details about a specific entity.",
                      "Use this to fetch the current state of a record."],
        "book":      ["This tool creates a new reservation or record.",
                      "Use this to commit a new entry to the system."],
        "cancel":    ["This tool cancels an existing record.",
                      "Use this to terminate a pending or active entry."],
        "exchange":  ["This tool exchanges one item for another.",
                      "Use this to swap an existing item with a replacement."],
        "return":    ["This tool initiates a return for delivered items.",
                      "Use this to send items back for refund."],
        "update":    ["This tool modifies fields on an existing record.",
                      "Use this to change attributes of an open entry."],
        "send":      ["This tool sends a confirmation or certificate.",
                      "Use this to deliver an artifact to the user."],
        "calculate": ["This tool evaluates a numeric expression.",
                      "Use this to compute arithmetic results."],
        "transfer":  ["This tool transfers the conversation to a human agent.",
                      "Use this to escalate when automated handling fails."],
        "other":     ["This tool performs a miscellaneous action.",
                      "Use this as a general-purpose helper."],
    },
    "io_type": {
        "identifier":   ["This tool primarily takes an identifier as input.",
                         "Use this to act on a record referenced by its id."],
        "query":        ["This tool primarily takes a free-text query as input.",
                         "Use this when searching by name or phrase."],
        "entity":       ["This tool primarily takes or returns a full entity object.",
                         "Use this when the payload is a structured record."],
        "transaction":  ["This tool primarily takes transaction data as input.",
                         "Use this when the operation involves payment or logistics."],
        "confirmation": ["This tool primarily takes a confirmation flag as input.",
                         "Use this when the user must approve a pending action."],
        "numeric":      ["This tool primarily operates on numeric values.",
                         "Use this for arithmetic or measurement tasks."],
    },
    "domain": {
        # retail
        "product":   ["This tool operates on product catalog data.",
                      "Use this for product lookup or category listing."],
        "order":     ["This tool operates on customer orders.",
                      "Use this for order lifecycle actions."],
        "user":      ["This tool operates on user account data.",
                      "Use this for user profile lookup or update."],
        "payment":   ["This tool operates on payment and refund flows.",
                      "Use this for money-movement actions."],
        "inventory": ["This tool operates on inventory item variants.",
                      "Use this for SKU-level lookups."],
        # airline
        "flight":      ["This tool operates on flight schedules and status.",
                        "Use this for flight search or status lookup."],
        "reservation": ["This tool operates on airline reservations.",
                        "Use this for booking or cancelling reservations."],
        "baggage":     ["This tool operates on baggage information.",
                        "Use this for baggage updates on a reservation."],
        "passenger":   ["This tool operates on passenger details.",
                        "Use this for passenger updates on a reservation."],
        "certificate": ["This tool operates on travel certificates.",
                        "Use this to issue or query certificates."],
        # shared
        "utility":     ["This tool performs a utility action not tied to a domain.",
                        "Use this for arithmetic, escalation, or generic helpers."],
    },
    "tool_category": {
        "read":    ["This tool only reads state and has no side effects.",
                    "Use this for idempotent lookups."],
        "write":   ["This tool writes or mutates state.",
                    "Use this only after explicit user confirmation."],
        "generic": ["This tool is a general-purpose helper.",
                    "Use this for utility operations not tied to data."],
        "think":   ["This tool is a reasoning aid and does not touch state.",
                    "Use this to structure the agent's internal deliberation."],
    },
}


def build_ontology(tools: List[dict], domain: str) -> Tuple[dict, List[dict]]:
    """Return (ontology_dict, per_tool_classifications)."""
    classifications = [
        {"name": t["name"], "docstring": t["docstring"][:200], **classify_tool(t, domain)}
        for t in tools
    ]

    # Bucket tools by (facet, category)
    buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for c in classifications:
        for f in FACET_ORDER:
            buckets[(f, c[f])].append(c)

    ontology: Dict[str, Dict[str, List[str]]] = {}
    for facet in FACET_ORDER:
        ontology[facet] = {}
        # Use only categories that actually appear in this domain, plus
        # any that have anchor sentences (to guarantee non-empty buckets).
        present = sorted({c[facet] for c in classifications})
        for cat in present:
            anchors = ANCHOR_SENTENCES.get(facet, {}).get(cat, [])
            if not anchors:
                anchors = [f"This tool belongs to the {cat} {facet}.",
                           f"Use this for {cat.replace('_', ' ')} operations."]
            # Enrich with up to 2 real tool docstring snippets from this bucket
            examples = []
            for entry in buckets[(facet, cat)][:2]:
                snippet = entry["docstring"].split("\n")[0].strip()
                if snippet:
                    examples.append(f"Example tool {entry['name']}: {snippet[:120]}")
            ontology[facet][cat] = (anchors + examples)[:4]

    return ontology, classifications


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["retail", "airline"], required=True)
    p.add_argument(
        "--tools-py", type=str, default="",
        help="Override path to tools.py; defaults to external/tau2-bench/"
             "src/tau2/domains/<domain>/tools.py",
    )
    p.add_argument(
        "--out", type=str, default="",
        help="Output JSON. Default: reports/axis2_theoretical_verification/"
             "tau2_<domain>_ontology.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    tools_py = Path(args.tools_py) if args.tools_py else (
        TAU2_SRC / args.domain / "tools.py"
    )
    out_path = Path(args.out) if args.out else (
        REPO / "reports" / "axis2_theoretical_verification"
        / f"tau2_{args.domain}_ontology.json"
    )

    print(f"[parse] {tools_py}", flush=True)
    tools = parse_tau2_tools(tools_py)
    print(f"[parse] {len(tools)} @is_tool-decorated methods", flush=True)

    ontology, classifications = build_ontology(tools, args.domain)

    # Stats
    print(f"\n[ontology] {args.domain}")
    for f in FACET_ORDER:
        cats = ontology[f]
        print(f"  {f}: {len(cats)} categories")
        for cat, sents in cats.items():
            print(f"    - {cat}: {len(sents)} sentences")

    counters = {f: Counter(c[f] for c in classifications) for f in FACET_ORDER}
    print("\n[classification distribution]")
    for f in FACET_ORDER:
        print(f"  {f}: {dict(counters[f])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "facet_order": FACET_ORDER,
        "ontology": ontology,
        "domain": args.domain,
        "facet_counts": {f: len(ontology[f]) for f in FACET_ORDER},
        "classification_distribution": {
            f: dict(counters[f]) for f in FACET_ORDER
        },
        "classifications": classifications,
        "n_tools": len(tools),
        "source": f"tau2-bench {args.domain}/tools.py",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
