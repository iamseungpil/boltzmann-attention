#!/usr/bin/env python
"""P1a: TaskBench tool-graph -> native OpenAI function-calling 궤적.

설계 = NATIVE_FC_CONVERTER_DESIGN_2026_06_14.md v2 §3b.
- D3 병렬허용: DAG 위상-*레벨*별로 같은 레벨(상호 비의존) 노드 = 한 assistant 턴에 복수 tool_calls.
- 인자 = tool_nodes 값(verbatim copy = R1). 결과 = 합성(인자서 결정론 도출, 리스크#2).
- 도구명 = 실제명 유지(전역 alias 재번호는 fc_build_sft 단계).
- loss-mask: assistant 턴만(_supervise=True).

입력: TaskBench data.json(jsonl; tool_nodes/tool_links는 JSON 문자열) + tool_desc.json.
출력: {tools, messages, _meta} JSONL.

Usage: fc_convert_taskbench.py --data <data.json> --tool_desc <tool_desc.json> --out <out.jsonl> [--sample 5]
"""
import argparse, json, hashlib
from collections import defaultdict

TYPE_MAP = {"string": "string", "str": "string", "int": "integer", "integer": "integer",
            "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
            "date": "string", "list": "array", "array": "array"}


def load_records(p):
    txt = open(p, encoding="utf-8").read()
    if txt.lstrip()[:1] == "[":
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def maybe_json(x):
    return json.loads(x) if isinstance(x, str) else x


def build_schemas(tool_desc):
    nodes = tool_desc.get("nodes") if isinstance(tool_desc, dict) else tool_desc
    schemas = {}
    for t in nodes:
        props, req = {}, []
        for p in t.get("parameters", []):
            props[p["name"]] = {"type": TYPE_MAP.get(str(p.get("type", "string")).lower(), "string"),
                                "description": p.get("desc", "")}
            req.append(p["name"])
        schemas[t["id"]] = {"type": "function", "function": {
            "name": t["id"], "description": t.get("desc", ""),
            "parameters": {"type": "object", "properties": props, "required": req}}}
    return schemas


def dep_levels(nodes, links):
    """edge source->target 는 target이 source에 의존. 노드 레벨 = 1+max(소스 레벨). 레벨별 그룹 반환(인덱스)."""
    byname = defaultdict(list)
    for i, n in enumerate(nodes):
        byname[n["task"]].append(i)
    deps = {i: set() for i in range(len(nodes))}
    for l in links:
        s, t = l.get("source"), l.get("target")
        for ti in byname.get(t, []):
            for si in byname.get(s, []):
                if si != ti:
                    deps[ti].add(si)
    level = {}

    def lev(i, stack):
        if i in level:
            return level[i]
        if i in stack:          # cycle guard -> treat as root
            return 0
        if not deps[i]:
            level[i] = 0
            return 0
        level[i] = 1 + max(lev(d, stack | {i}) for d in deps[i])
        return level[i]

    for i in range(len(nodes)):
        lev(i, set())
    bylevel = defaultdict(list)
    for i in range(len(nodes)):
        bylevel[level[i]].append(i)
    return [bylevel[k] for k in sorted(bylevel)]


def synth_result(name, args):
    h = hashlib.md5((name + json.dumps(args, sort_keys=True)).encode()).hexdigest()[:8]
    return json.dumps({"status": "ok", "tool": name, "ref": h})


def convert(sample, schemas):
    instr = sample.get("instruction", "")
    nodes = maybe_json(sample.get("tool_nodes", "[]"))
    links = maybe_json(sample.get("tool_links", "[]"))
    if not nodes:
        return None
    # 모든 노드 도구가 schema에 있어야 변환(없으면 스킵=청정 분모)
    if any(n["task"] not in schemas for n in nodes):
        return None
    levels = dep_levels(nodes, links)
    seen, toolset = set(), []
    for n in nodes:
        if n["task"] not in seen:
            seen.add(n["task"]); toolset.append(schemas[n["task"]])
    messages = [{"role": "system", "content": "You are a tool-using assistant. Call the appropriate functions to fulfill the user request."},
                {"role": "user", "content": instr}]
    cid = 0
    for lvl in levels:
        tcs, results = [], []
        for i in lvl:
            n = nodes[i]
            args = {a["name"]: a.get("value") for a in n.get("arguments", [])}
            cid += 1
            tcid = "call_%d" % cid
            tcs.append({"id": tcid, "type": "function",
                        "function": {"name": n["task"], "arguments": json.dumps(args, ensure_ascii=False)}})
            results.append((tcid, synth_result(n["task"], args)))
        messages.append({"role": "assistant", "tool_calls": tcs, "_supervise": True})
        for tcid, res in results:
            messages.append({"role": "tool", "tool_call_id": tcid, "content": res})
    return {"tools": toolset, "messages": messages,
            "_meta": {"bench": "taskbench", "id": sample.get("id"), "type": sample.get("type"),
                      "n_nodes": len(nodes), "n_levels": len(levels)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="print N converted examples to stdout (no write if >0 and --dry)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    schemas = build_schemas(json.load(open(a.tool_desc, encoding="utf-8")))
    recs = load_records(a.data)
    out, n_skip = [], 0
    for r in recs:
        c = convert(r, schemas)
        if c is None:
            n_skip += 1
        else:
            out.append(c)
    # stats
    par = sum(1 for c in out if c["_meta"]["n_nodes"] > c["_meta"]["n_levels"])
    print("converted=%d skip=%d (no-schema/empty)  parallel-graphs=%d  tools_total=%d"
          % (len(out), n_skip, par, len(schemas)))
    if a.sample:
        for c in out[:a.sample]:
            print("\n=== %s (nodes=%d levels=%d) ===" % (c["_meta"]["id"], c["_meta"]["n_nodes"], c["_meta"]["n_levels"]))
            for m in c["messages"]:
                if m["role"] == "assistant":
                    print("  [assistant] " + " | ".join("%s(%s)" % (tc["function"]["name"], tc["function"]["arguments"]) for tc in m["tool_calls"]))
                elif m["role"] == "tool":
                    print("    [tool] " + m["content"][:60])
                else:
                    print("  [%s] %s" % (m["role"], str(m["content"])[:70]))
    if not a.dry:
        with open(a.out, "w", encoding="utf-8") as f:
            for c in out:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
