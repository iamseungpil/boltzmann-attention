#!/usr/bin/env python3
"""F7 — Generate anchor sentences per graph node (class + tool).

Reads ontology_graph_{domain}.json (from extract_ontology_graph.py) and
produces an anchor-sentences JSON where each class + tool has 2-6 natural
sentences grounded in the actual Pydantic Field descriptions + tool
docstrings + relation verb templates.

These anchors are the inputs to build_ontology_structured_bont.py.

Output schema:
{
  "domain": "telecom",
  "per_class": {
      "Customer": {"sentences": [str, ...], "parents": [...], "depth": int}
  },
  "per_tool":  {
      "get_customer_by_id": {"sentences": [...], "verb": "get",
                              "tool_type": "READ",
                              "source_types": [...], "target_type": "..."}
  },
  "per_part_of": {
      "Customer::address": {"whole": "Customer", "part": "Address",
                             "via_field": "address",
                             "sentences": [...]}
  },
  "per_relation": {
      "book_reservation": {"sentences": [...], "source_types": [...],
                            "target_type": "Reservation"}
  }
}

The per_class / per_tool entries power Variant H (hierarchy) + Variant R
(relations).  per_part_of powers Variant C (composition-specific anchors
per whole-part pair).
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
TAU2_SRC = REPO / "external" / "tau2-bench" / "src" / "tau2" / "domains"

MAX_PER_CLASS = 6
MAX_PER_TOOL = 4
MAX_PER_PART = 3


def load_pydantic_field_descriptions(domain: str) -> Dict[str, List[dict]]:
    """For each class in the domain's data_model.py, extract [{field, type, description}]
    from the Field(description=...) kwarg.  Returns {class_name: [field_info, ...]}."""
    path = TAU2_SRC / domain / "data_model.py"
    tree = ast.parse(path.read_text())
    out: Dict[str, List[dict]] = defaultdict(list)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            fname = stmt.target.id
            desc = _extract_field_description(stmt.value)
            if desc:
                out[node.name].append({"name": fname, "desc": desc})
    return out


def _extract_field_description(val) -> str:
    """Pull the 'description' kwarg from a Field(...) call, if present."""
    if not isinstance(val, ast.Call):
        return ""
    fn = val.func
    fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
    if fname != "Field":
        return ""
    for kw in val.keywords:
        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    # positional: Field(default, description=...)?  Usually only kwarg, skip positional.
    return ""


def load_tool_docstrings(domain: str) -> Dict[str, str]:
    """For each @is_tool-decorated method in tools.py, return {name: docstring_first_para}."""
    path = TAU2_SRC / domain / "tools.py"
    tree = ast.parse(path.read_text())
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_tool = any(
            isinstance(dec, ast.Call)
            and (dec.func.id if isinstance(dec.func, ast.Name) else
                 (dec.func.attr if isinstance(dec.func, ast.Attribute) else "")) == "is_tool"
            for dec in node.decorator_list
        )
        if not is_tool:
            continue
        doc = ast.get_docstring(node) or ""
        first_para = doc.split("\n\n")[0].strip() if doc else ""
        out[node.name] = first_para
    return out


def class_anchor_sentences(class_name: str, fields: List[dict],
                            parents: List[str]) -> List[str]:
    """Produce 2-6 anchor sentences describing a class."""
    sentences: List[str] = []
    # Templated
    parent_hint = parents[0] if parents else ""
    if parent_hint and parent_hint not in {"BaseModel", "BaseModelNoExtra", "DB", "Enum"}:
        sentences.append(f"A {class_name} is a kind of {parent_hint}.")
    else:
        sentences.append(f"This describes a {class_name} entity in the domain.")
    sentences.append(f"The {class_name} record participates in downstream tool operations.")
    # Field-derived
    for field in fields[: MAX_PER_CLASS - len(sentences)]:
        desc = field["desc"].strip()
        if desc:
            # "<field>: <desc>" → make it sentence-like
            s = f"{class_name}.{field['name']}: {desc}"
            sentences.append(s[:200])
    return sentences[:MAX_PER_CLASS]


def tool_anchor_sentences(tool: dict, docstring: str) -> List[str]:
    """Produce 2-4 anchor sentences for a tool (relation)."""
    name = tool["tool_name"]
    verb = tool["verb"]
    src = tool.get("source_types") or []
    tgt = tool.get("target_type", "")
    sentences: List[str] = []
    # Templated
    src_str = " and ".join(src) if src else "a request"
    sentences.append(f"This tool {verb}s {src_str} to produce a {tgt}.")
    sentences.append(f"Invoke {name} when the user needs to {verb} a {tgt}.")
    # Docstring
    if docstring:
        first_line = docstring.split("\n")[0].strip()
        if first_line:
            sentences.append(first_line[:200])
    return sentences[:MAX_PER_TOOL]


def part_of_anchor_sentences(whole: str, part: str, via_field: str) -> List[str]:
    return [
        f"A {whole} contains a {part} accessed via its {via_field} attribute.",
        f"The {part} is a component of the {whole} record.",
        f"Operations on {whole} may read or modify its {part} via {via_field}.",
    ][:MAX_PER_PART]


def build_per_domain(domain: str, graph: dict) -> dict:
    fields_by_class = load_pydantic_field_descriptions(domain)
    docstrings = load_tool_docstrings(domain)

    per_class: Dict[str, dict] = {}
    classes = graph["nodes"]["classes"]
    # Depth from is-a edges (root = 0)
    parents_of = {c["name"]: c.get("parents", []) for c in classes}

    def depth(name: str, seen=None) -> int:
        seen = seen or set()
        if name in seen:
            return 0
        seen.add(name)
        parents = parents_of.get(name, [])
        if not parents or all(p in {"BaseModel", "BaseModelNoExtra", "DB", "Enum"} for p in parents):
            return 0
        return 1 + max(
            depth(p, seen) for p in parents
            if p not in {"BaseModel", "BaseModelNoExtra", "DB", "Enum"}
        )

    for c in classes:
        name = c["name"]
        fields = fields_by_class.get(name, [])
        parents = c.get("parents", [])
        per_class[name] = {
            "sentences": class_anchor_sentences(name, fields, parents),
            "parents": parents,
            "depth": depth(name),
            "kind": c["kind"],
        }

    per_tool: Dict[str, dict] = {}
    per_relation: Dict[str, dict] = {}
    for rel in graph["edges"]["relation"]:
        name = rel["tool_name"]
        doc = docstrings.get(name, "")
        sentences = tool_anchor_sentences(rel, doc)
        per_tool[name] = {
            "sentences": sentences,
            "verb": rel["verb"],
            "tool_type": rel["tool_type"],
            "source_types": rel["source_types"],
            "target_type": rel["target_type"],
        }
        per_relation[name] = {
            "sentences": sentences,
            "source_types": rel["source_types"],
            "target_type": rel["target_type"],
            "verb": rel["verb"],
        }

    per_part_of: Dict[str, dict] = {}
    for edge in graph["edges"]["part_of"]:
        key = f"{edge['whole']}::{edge['via_field']}"
        per_part_of[key] = {
            "whole": edge["whole"],
            "part": edge["part"],
            "via_field": edge["via_field"],
            "sentences": part_of_anchor_sentences(
                edge["whole"], edge["part"], edge["via_field"]
            ),
        }

    return {
        "domain": domain,
        "per_class": per_class,
        "per_tool": per_tool,
        "per_part_of": per_part_of,
        "per_relation": per_relation,
        "summary": {
            "n_class_sentences": sum(len(v["sentences"]) for v in per_class.values()),
            "n_tool_sentences": sum(len(v["sentences"]) for v in per_tool.values()),
            "n_part_sentences": sum(len(v["sentences"]) for v in per_part_of.values()),
            "n_relation_sentences": sum(len(v["sentences"]) for v in per_relation.values()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="reports/new_theorem_test/phase_f7/ontology_graph_all.json")
    ap.add_argument("--out-dir", default="reports/new_theorem_test/phase_f7")
    ap.add_argument("--domains", default="telecom,retail,airline")
    args = ap.parse_args()

    g_all = json.load(open(REPO / args.graph))
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in args.domains.split(","):
        graph = g_all["domains"][domain]
        out = build_per_domain(domain, graph)
        out_path = out_dir / f"anchor_sentences_{domain}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[{domain}] wrote {out_path}")
        print(f"  class sentences: {out['summary']['n_class_sentences']}")
        print(f"  tool sentences:  {out['summary']['n_tool_sentences']}")
        print(f"  part sentences:  {out['summary']['n_part_sentences']}")


if __name__ == "__main__":
    main()
