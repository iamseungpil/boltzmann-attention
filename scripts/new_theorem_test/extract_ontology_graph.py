#!/usr/bin/env python3
"""
F7 pilot: extract a REAL ontology graph (classes + is-a + part-of + relations)
from τ²-bench data_model.py + tools.py.

Output JSON schema:
{
  "domain": "telecom",
  "nodes": {
      "classes":   [{"name", "kind": "model"|"enum", "parent": <root>, "fields": [...]}],
      "tools":     [{"name", "tool_type", "args": [...], "return_type", "docstring"}]
  },
  "edges": {
      "is_a":    [{"child", "parent"}],            # Pydantic inheritance
      "part_of": [{"whole", "part", "via_field"}], # composition from Field(...)
      "relation":[{"source_types", "target_type",
                   "verb", "tool_type", "tool_name"}]  # one per @is_tool method
  },
  "summary": {"n_classes", "n_tools", "n_is_a", "n_part_of", "n_relation", "n_total_edges"}
}

Rank-budget check: if n_total_edges ∈ [20, 500], F7 build proceeds.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
TAU2_SRC = REPO / "external" / "tau2-bench" / "src" / "tau2" / "domains"

ACTION_VERBS = [
    "search", "find", "list", "get", "book", "make", "create",
    "cancel", "exchange", "return", "update", "modify", "set",
    "send", "issue", "calculate", "compute", "transfer",
    "suspend", "resume", "enable", "disable", "refuel",
    "remove", "add", "check", "verify",
]


# ---------------------------------------------------------------
# data_model.py parser
# ---------------------------------------------------------------

MODEL_ROOTS = {"BaseModelNoExtra", "BaseModel", "DB"}


def parse_data_model(path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """Return (classes, is_a_edges, part_of_edges)."""
    tree = ast.parse(path.read_text())
    classes: List[dict] = []
    is_a: List[dict] = []
    part_of: List[dict] = []

    known_model_names = set()
    known_enum_names = set()
    class_bases: Dict[str, List[str]] = {}

    # First pass: inventory classes + their direct bases
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_base_name(b) for b in node.bases]
        class_bases[node.name] = bases
        if any(b == "str" for b in bases) and any(b == "Enum" for b in bases):
            known_enum_names.add(node.name)
            continue

    # Iteratively mark a class as a "model" if any base is a root OR an already-known model
    changed = True
    while changed:
        changed = False
        for name, bases in class_bases.items():
            if name in known_model_names or name in known_enum_names:
                continue
            if any(b in MODEL_ROOTS or b in known_model_names for b in bases):
                known_model_names.add(name)
                changed = True

    # Second pass: extract fields + edges
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_base_name(b) for b in node.bases]
        kind = None
        parents: List[str] = []
        if node.name in known_enum_names:
            kind = "enum"
            parents = ["Enum"]
        elif node.name in known_model_names:
            kind = "model"
            # Use direct bases (filter to meaningful ones)
            parents = [b for b in bases
                       if b in MODEL_ROOTS or b in known_model_names]
            if not parents:
                parents = bases[:1] if bases else ["BaseModel"]
        else:
            continue

        for p in parents:
            is_a.append({"child": node.name, "parent": p})

        fields: List[dict] = []
        for stmt in node.body:
            # field: Type = Field(...)
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fname = stmt.target.id
                ftype = _ann_name(stmt.annotation)
                fields.append({"name": fname, "type": ftype})

                # part-of: field type references another known class
                referenced = _referenced_types(stmt.annotation)
                for ref in referenced:
                    if ref in known_model_names and ref != node.name:
                        part_of.append({
                            "whole": node.name,
                            "part": ref,
                            "via_field": fname,
                        })
                    elif ref in known_enum_names:
                        part_of.append({
                            "whole": node.name,
                            "part": ref,
                            "via_field": fname,
                        })
                    # Also detect "*_id" convention linking to a class
                    elif fname.endswith("_id") or fname.endswith("_ids"):
                        stem = fname[:-len("_ids")] if fname.endswith("_ids") else fname[:-len("_id")]
                        guess = stem.capitalize()
                        if guess in known_model_names and guess != node.name:
                            part_of.append({
                                "whole": node.name,
                                "part": guess,
                                "via_field": fname,
                            })

        classes.append({
            "name": node.name,
            "kind": kind,
            "parent": parent,
            "fields": fields,
        })

    # Dedup part_of
    seen = set()
    dedup = []
    for e in part_of:
        k = (e["whole"], e["part"], e["via_field"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(e)
    return classes, is_a, dedup


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _ann_name(node: ast.AST) -> str:
    """Flatten an annotation to a source-like string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ann_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_ann_name(node.value)}[{_ann_name(node.slice)}]"
    if isinstance(node, ast.Tuple):
        return ", ".join(_ann_name(e) for e in node.elts)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return "?"


def _referenced_types(node: ast.AST) -> List[str]:
    """Return all type names referenced in an annotation."""
    out: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.append(child.id)
    return out


# ---------------------------------------------------------------
# tools.py parser
# ---------------------------------------------------------------

def parse_tools(path: Path, known_model_names: set) -> Tuple[List[dict], List[dict]]:
    """Return (tools, relation_edges)."""
    tree = ast.parse(path.read_text())
    tools: List[dict] = []
    relations: List[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        tool_type = _is_tool_type(node)
        if tool_type is None:
            continue
        args = []
        for a in node.args.args:
            if a.arg == "self":
                continue
            t = _ann_name(a.annotation) if a.annotation else "Any"
            args.append({"name": a.arg, "type": t})
        ret_type = _ann_name(node.returns) if node.returns else "Any"
        docstring = ast.get_docstring(node) or ""
        verb = _extract_verb(node.name)

        tools.append({
            "name": node.name,
            "tool_type": tool_type,
            "args": args,
            "return_type": ret_type,
            "verb": verb,
            "docstring": docstring[:300],
        })

        # Relation edge
        src_types = []
        for a in args:
            refs = [r for r in re.findall(r"[A-Z]\w+", a["type"])
                    if r in known_model_names]
            src_types.extend(refs)
            # *_id convention
            if a["name"].endswith("_id"):
                stem = a["name"][:-3]
                guess = stem.capitalize()
                if guess in known_model_names:
                    src_types.append(guess)
        tgt_refs = [r for r in re.findall(r"[A-Z]\w+", ret_type)
                    if r in known_model_names]
        relations.append({
            "tool_name": node.name,
            "tool_type": tool_type,
            "verb": verb,
            "source_types": sorted(set(src_types)),
            "target_type": tgt_refs[0] if tgt_refs else ret_type,
        })
    return tools, relations


def _is_tool_type(func: ast.FunctionDef) -> Optional[str]:
    for dec in func.decorator_list:
        if isinstance(dec, ast.Call):
            fn = _base_name(dec.func)
            if fn == "is_tool" and dec.args:
                arg0 = dec.args[0]
                if isinstance(arg0, ast.Attribute):
                    return arg0.attr  # READ/WRITE/GENERIC/THINK
    return None


def _extract_verb(name: str) -> str:
    for v in ACTION_VERBS:
        if name.startswith(v + "_") or name == v:
            return v
    # fallback: first underscore-separated token
    return name.split("_", 1)[0]


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def extract_domain(domain: str) -> dict:
    data_model_path = TAU2_SRC / domain / "data_model.py"
    tools_path = TAU2_SRC / domain / "tools.py"

    classes, is_a, part_of = parse_data_model(data_model_path)
    known_models = {c["name"] for c in classes}
    tools, relations = parse_tools(tools_path, known_models)

    n_total = len(is_a) + len(part_of) + len(relations)
    return {
        "domain": domain,
        "nodes": {"classes": classes, "tools": tools},
        "edges": {
            "is_a": is_a,
            "part_of": part_of,
            "relation": relations,
        },
        "summary": {
            "n_classes": len(classes),
            "n_tools": len(tools),
            "n_is_a": len(is_a),
            "n_part_of": len(part_of),
            "n_relation": len(relations),
            "n_total_edges": n_total,
            "budget_gate_20_500": 20 <= n_total <= 500,
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["telecom", "retail", "airline"], default="telecom")
    p.add_argument("--out", default="")
    p.add_argument("--all", action="store_true",
                   help="Extract all three domains and write combined graph")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = REPO / "reports" / "new_theorem_test" / "phase_f7"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        combined = {"domains": {}, "summary": {}}
        total = 0
        for d in ["telecom", "retail", "airline"]:
            g = extract_domain(d)
            combined["domains"][d] = g
            total += g["summary"]["n_total_edges"]
            print(f"[{d}] classes={g['summary']['n_classes']} "
                  f"tools={g['summary']['n_tools']} "
                  f"is_a={g['summary']['n_is_a']} "
                  f"part_of={g['summary']['n_part_of']} "
                  f"relation={g['summary']['n_relation']} "
                  f"total={g['summary']['n_total_edges']}")
        combined["summary"] = {
            "n_total_edges": total,
            "budget_gate_20_500": 20 <= total <= 500,
        }
        print(f"\n[ALL] total edges = {total} — "
              f"gate {'PASS' if combined['summary']['budget_gate_20_500'] else 'FAIL'}")
        out_path = Path(args.out) if args.out else out_dir / "ontology_graph_all.json"
        out_path.write_text(json.dumps(combined, indent=2))
        print(f"\nwrote {out_path}")
        return

    g = extract_domain(args.domain)
    s = g["summary"]
    print(f"[{args.domain}]")
    print(f"  classes:    {s['n_classes']}")
    print(f"  tools:      {s['n_tools']}")
    print(f"  is_a:       {s['n_is_a']}")
    print(f"  part_of:    {s['n_part_of']}")
    print(f"  relation:   {s['n_relation']}")
    print(f"  total:      {s['n_total_edges']}")
    print(f"  budget gate [20, 500]: "
          f"{'PASS' if s['budget_gate_20_500'] else 'FAIL'}")

    out_path = Path(args.out) if args.out else out_dir / f"ontology_graph_{args.domain}.json"
    out_path.write_text(json.dumps(g, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
