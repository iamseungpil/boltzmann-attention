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
import argparse, json, hashlib, re
from collections import defaultdict

NODE_REF = re.compile(r"<node-(\d+)>")

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
    # ★<node-K> 참조도 의존성으로(1-based) → threading한 출력이 항상 하류 호출보다 먼저 나오게 보장
    for i, n in enumerate(nodes):
        for m in NODE_REF.finditer(json.dumps(n.get("arguments"), ensure_ascii=False)):
            k = int(m.group(1)) - 1
            if 0 <= k < len(nodes) and k != i:
                deps[i].add(k)
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


def node_args(n, schema):
    """TaskBench arguments is dirty — 5 formats observed across domains:
      list-dict [{"name","value"}] (daily) / list-str ["example.mp4"] (mm·HF, dominant)
      / dict {..} / bare str / none. Bind bare values positionally to schema param
      names (R1 = verbatim copy; names recovered from tool_desc)."""
    raw = n.get("arguments")
    if not raw:
        return {}
    pnames = list(schema["function"]["parameters"]["properties"].keys())
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return {pnames[0] if pnames else "arg0": raw}
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):  # list-dict (may be mixed with bare values)
            out = {}
            for i, a in enumerate(raw):
                if isinstance(a, dict):
                    out[a.get("name", "arg%d" % i)] = a.get("value")
                else:
                    out[pnames[i] if i < len(pnames) else "arg%d" % i] = a
            return out
        return {(pnames[i] if i < len(pnames) else "arg%d" % i): v for i, v in enumerate(raw)}
    return {}


def synth_result(name, args, out_ref):
    """합성 결과 = 이 노드의 *출력 ref*를 담음(다운스트림 <node-N>이 이걸 fetch해 인자로 씀).
    ref = 궤적-고유·비-memorizable(md5) → 다운스트림 인자값의 유일 출처 = 이 출력 = fetch 강제(R1/R2)."""
    return json.dumps({"status": "ok", "tool": name, "result": out_ref})


def node_refs(sample, nodes):
    """노드별 출력 ref(궤적-고유). e.g. res_1a2b3c4d."""
    sid = str(sample.get("id"))
    return ["res_%s" % hashlib.md5((sid + "|" + str(i) + "|" + str(nodes[i].get("task", ""))).encode()).hexdigest()[:8]
            for i in range(len(nodes))]


def resolve_refs(val, refs):
    """인자값의 <node-K>(1-based) → 노드 (K-1) 출력 ref로 치환 = 출력→입력 threading."""
    if isinstance(val, str):
        def sub(m):
            k = int(m.group(1)) - 1
            return refs[k] if 0 <= k < len(refs) else m.group(0)
        return NODE_REF.sub(sub, val)
    if isinstance(val, list):
        return [resolve_refs(v, refs) for v in val]
    if isinstance(val, dict):
        return {k: resolve_refs(v, refs) for k, v in val.items()}
    return val


def convert(sample, schemas):
    instr = sample.get("instruction", "")
    nodes = maybe_json(sample.get("tool_nodes", "[]"))
    links = maybe_json(sample.get("tool_links", "[]"))
    if not isinstance(nodes, list) or not nodes:
        return None
    # malformed nodes (str instead of dict / no task) or unknown tool -> skip (clean denom)
    if any(not isinstance(n, dict) or "task" not in n for n in nodes):
        return None
    if any(n["task"] not in schemas for n in nodes):
        return None
    if not isinstance(links, list):
        links = []
    levels = dep_levels(nodes, links)
    seen, toolset = set(), []
    for n in nodes:
        if n["task"] not in seen:
            seen.add(n["task"]); toolset.append(schemas[n["task"]])
    refs = node_refs(sample, nodes)
    messages = [{"role": "system", "content": "You are a tool-using assistant. Call the appropriate functions to fulfill the user request."},
                {"role": "user", "content": instr}]
    cid = 0
    n_chain = 0
    for lvl in levels:
        tcs, results = [], []
        for i in lvl:
            n = nodes[i]
            args = node_args(n, schemas[n["task"]])
            # ★출력→입력 threading: <node-K> → 노드 K-1 출력 ref (fetch-to-obtain-arg 강제)
            resolved = {}
            for k, v in args.items():
                rv = resolve_refs(v, refs)
                if rv != v:
                    n_chain += 1
                resolved[k] = rv
            cid += 1
            tcid = "call_%d" % cid
            tcs.append({"id": tcid, "type": "function",
                        "function": {"name": n["task"], "arguments": json.dumps(resolved, ensure_ascii=False)}})
            results.append((tcid, synth_result(n["task"], resolved, refs[i])))
        messages.append({"role": "assistant", "tool_calls": tcs, "_supervise": True})
        for tcid, res in results:
            messages.append({"role": "tool", "tool_call_id": tcid, "content": res})
    return {"tools": toolset, "messages": messages,
            "_meta": {"bench": "taskbench", "id": sample.get("id"), "type": sample.get("type"),
                      "n_nodes": len(nodes), "n_levels": len(levels), "n_chain": n_chain}}


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
