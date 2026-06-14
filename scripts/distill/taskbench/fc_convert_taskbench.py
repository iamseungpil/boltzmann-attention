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
RES_REF = re.compile(r"res_[0-9a-f]{8}")

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


def dep_sources(nodes, links):
    """노드 index -> 의존성 source 노드 index 목록(self 제외·정렬). <node-N>은 링크로 해석(R 참조-규율: 자기참조 금지)."""
    byname = defaultdict(list)
    for i, n in enumerate(nodes):
        byname[n["task"]].append(i)
    src = {i: [] for i in range(len(nodes))}
    for l in links:
        s, t = l.get("source"), l.get("target")
        for ti in byname.get(t, []):
            for si in byname.get(s, []):
                if si != ti and si not in src[ti]:
                    src[ti].append(si)
    for i in src:
        src[i].sort()
    return src


def resolve_node_refs(val, srcs, refs):
    """<node-N> → 이 노드의 의존성 source 출력 ref로 치환. N 오름차순 rank를 srcs(정렬)에 매핑.
    ★self-reference 금지(srcs는 self 제외) — 미해결 시 원본 유지(threading 안 함)."""
    blob = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    ns = sorted({int(m.group(1)) for m in NODE_REF.finditer(blob)})
    if not ns or not srcs:
        return val, 0
    nmap = {n: refs[srcs[min(rank, len(srcs) - 1)]] for rank, n in enumerate(ns)}
    cnt = [0]

    def repl_str(x):
        return NODE_REF.sub(lambda m: (cnt.__setitem__(0, cnt[0] + 1) or nmap[int(m.group(1))])
                            if int(m.group(1)) in nmap else m.group(0), x)

    def walk(v):
        if isinstance(v, str):
            return repl_str(v)
        if isinstance(v, list):
            return [walk(z) for z in v]
        if isinstance(v, dict):
            return {k: walk(z) for k, z in v.items()}
        return v

    return walk(val), cnt[0]


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
    srcs = dep_sources(nodes, links)
    messages = [{"role": "system", "content": "You are a tool-using assistant. Call the appropriate functions to fulfill the user request."},
                {"role": "user", "content": instr}]
    cid = 0
    n_chain = 0
    for lvl in levels:
        tcs, results = [], []
        for i in lvl:
            n = nodes[i]
            args = node_args(n, schemas[n["task"]])
            # ★출력→입력 threading: <node-N> → 의존성 source 출력 ref (링크 기반·self 금지·fetch 강제)
            resolved = {}
            for k, v in args.items():
                rv, c = resolve_node_refs(v, srcs[i], refs)
                n_chain += c
                resolved[k] = rv
            cid += 1
            tcid = "call_%d" % cid
            tcs.append({"id": tcid, "type": "function",
                        "function": {"name": n["task"], "arguments": json.dumps(resolved, ensure_ascii=False)}})
            results.append((tcid, synth_result(n["task"], resolved, refs[i])))
        messages.append({"role": "assistant", "tool_calls": tcs, "_supervise": True})
        for tcid, res in results:
            messages.append({"role": "tool", "tool_call_id": tcid, "content": res})
    # ★정합성 post-check (결정론 drop): TaskBench 원본 비정합(no-link/cycle/dup-task)으로
    #   <node-N>이 못 풀리면(literal 잔존) 또는 res_가 산출 전 사용(forward/self-ref)되면 = 더러운 데이터 →
    #   억지 resolution 대신 trajectory 통째 drop(=fetch-to-obtain-arg 학습 오염 방지).
    produced = set()
    for m in messages:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                argstr = tc["function"]["arguments"]
                if "<node-" in argstr:
                    return None                      # no-link: 못 푼 literal 잔존
                for r in RES_REF.findall(argstr):
                    if r not in produced:
                        return None                  # cycle/dup: 산출 전 사용(forward/self-ref)
        elif m["role"] == "tool":
            produced.update(RES_REF.findall(str(m.get("content") or "")))
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
    print("converted=%d skip=%d (no-schema/empty/dirty-ref:no-link·cycle·dup)  parallel-graphs=%d  tools_total=%d"
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
