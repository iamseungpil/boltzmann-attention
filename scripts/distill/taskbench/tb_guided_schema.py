#!/usr/bin/env python
"""grounded-copy v1: build per-domain JSON schema constraining tool-name slots
(task / task_links source·target) to the domain's valid tool ids. 나머지 필드는
비제약(자유 텍스트 steps·arguments) — TB_GROUNDED_COPY_V1_DESIGN.md §2.

Usage: python tb_guided_schema.py --tool_desc <domain>/tool_desc.json --out schema.json \
           [--dep temporal|resource]
(resource 형식은 출력에 task_links가 없음 — inference.py:186 vs :189)
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dep", default="temporal", choices=["temporal", "resource"])
    a = ap.parse_args()

    names = [t["id"] for t in json.load(open(a.tool_desc))["nodes"]]
    tool = {"type": "string", "enum": names}
    props = {
        "task_steps": {"type": "array", "items": {"type": "string"}},
        "task_nodes": {"type": "array", "items": {
            "type": "object",
            "properties": {"task": {"$ref": "#/$defs/ToolName"},
                           "arguments": {"type": "array"}},
            "required": ["task"]}},
    }
    required = ["task_steps", "task_nodes"]
    if a.dep == "temporal":
        props["task_links"] = {"type": "array", "items": {
            "type": "object",
            "properties": {"source": {"$ref": "#/$defs/ToolName"},
                           "target": {"$ref": "#/$defs/ToolName"}},
            "required": ["source", "target"]}}
        required.append("task_links")
    schema = {"type": "object", "properties": props, "required": required,
              "$defs": {"ToolName": tool}}
    json.dump(schema, open(a.out, "w"))
    print(f"[schema] {len(names)} tool names ({a.dep}) -> {a.out}")


if __name__ == "__main__":
    main()
