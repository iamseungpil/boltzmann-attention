#!/usr/bin/env python
"""grounded-copy v1: build per-domain JSON schema constraining tool-name slots
(task / task_links source·target) to the domain's valid tool ids. 나머지 필드는
비제약(자유 텍스트 steps·arguments) — TB_GROUNDED_COPY_V1_DESIGN.md §2.

Usage: python tb_guided_schema.py --tool_desc <domain>/tool_desc.json --out schema.json
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = [t["id"] for t in json.load(open(a.tool_desc))["nodes"]]
    tool = {"type": "string", "enum": names}
    schema = {
        "type": "object",
        "properties": {
            "task_steps": {"type": "array", "items": {"type": "string"}},
            "task_nodes": {"type": "array", "items": {
                "type": "object",
                "properties": {"task": {"$ref": "#/$defs/ToolName"},
                               "arguments": {"type": "array"}},
                "required": ["task"]}},
            "task_links": {"type": "array", "items": {
                "type": "object",
                "properties": {"source": {"$ref": "#/$defs/ToolName"},
                               "target": {"$ref": "#/$defs/ToolName"}},
                "required": ["source", "target"]}},
        },
        "required": ["task_steps", "task_nodes", "task_links"],
        "$defs": {"ToolName": tool},
    }
    json.dump(schema, open(a.out, "w"))
    print(f"[schema] {len(names)} tool names -> {a.out}")


if __name__ == "__main__":
    main()
