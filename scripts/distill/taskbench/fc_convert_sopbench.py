#!/usr/bin/env python
"""P1b: SOPBench FC 성공 rollout -> native OpenAI function-calling 궤적 (1:1 변환).

설계 = NATIVE_FC_CONVERTER_DESIGN_2026_06_14.md v2 §3a.
- 소스 = SOPBench/output/<domain>/ast_<teacher>-mode_fc-...json, evaluations[*].success=True 만.
- role 정규화: tool_call_id->tool / sender=='user'->user / 그 외->assistant.
- tools= (A1) = env/domains/<domain>/<domain>_assistant.py 의 `actions` (이미 OpenAI schema) + exit_conversation.
- 실제 도구명 유지(전역 alias 재번호는 fc_build_sft). loss-mask: assistant 턴(_supervise).
- 합성 없음: op순서·인자·결과 전부 실 rollout 보존 → R1=실행된 인자.

★ python 3.10+ 필요(SOPBench env가 match문 사용). 예: seka_env.
Usage: fc_convert_sopbench.py --rollout <file.json> --sopbench_root <SOPBench> --domain <d> --out <out.jsonl> [--sample N] [--dry]
"""
import argparse, json, importlib.util, os, sys


def load_actions(root, domain):
    path = os.path.join(root, "env", "domains", domain, "%s_assistant.py" % domain)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("%s_assistant" % domain, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, "actions")


def to_tools(actions):
    tools = []
    for a in actions:
        tools.append({"type": "function", "function": {
            "name": a["name"], "description": a.get("description", ""),
            "parameters": a.get("parameters", {"type": "object", "properties": {}})}})
    tools.append({"type": "function", "function": {
        "name": "exit_conversation",
        "description": "End the conversation when the request is complete or cannot be fulfilled.",
        "parameters": {"type": "object", "properties": {}}}})
    return tools


def norm_args(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def normalize(m, ctr):
    if m.get("tool_call_id"):
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": str(m.get("content", ""))}
    if m.get("sender") == "user":
        return {"role": "user", "content": m.get("content", "") or ""}
    # assistant
    out = {"role": "assistant", "content": m.get("content") or "", "_supervise": True}
    tcs = m.get("tool_calls") or []
    norm = []
    for tc in tcs:
        fn = tc.get("function", tc)
        ctr[0] += 1
        norm.append({"id": tc.get("id") or "call_%d" % ctr[0], "type": "function",
                     "function": {"name": fn.get("name"), "arguments": norm_args(fn.get("arguments", "{}"))}})
    if norm:
        out["tool_calls"] = norm
    return out


STATIC_PREFIXES = ("Here is all the information I can provide", "If you have completed my request")


def clean(raw):
    """SOPBench 스캐폴드 아티팩트 제거: 빈 assistant 턴·반복 정적-user·exit 후 종료."""
    out = []
    seen_static = False
    for mm in raw:
        r = mm["role"]
        if r == "assistant":
            if not mm.get("tool_calls") and not (mm.get("content") or "").strip():
                continue  # 빈 assistant 턴 제거(빈 출력 supervise 방지)
            out.append(mm)
            if mm.get("tool_calls") and any(tc["function"]["name"] == "exit_conversation" for tc in mm["tool_calls"]):
                break  # 종료
        elif r == "user":
            c = (mm.get("content") or "").lstrip()
            if any(c.startswith(p) for p in STATIC_PREFIXES):
                if seen_static:
                    continue  # 반복 정적-user 덤프 제거(정보는 첫 요청에 이미 존재)
                seen_static = True
            out.append(mm)
        else:  # tool
            out.append(mm)
    return out


def convert_task(task, tools, domain=None):
    inter = task["interactions"][0]
    prompt = inter.get("prompt", "")
    sys_msg = {"role": "system", "content": prompt if isinstance(prompt, str) else json.dumps(prompt)}
    ctr = [0]
    msgs = [sys_msg] + clean([normalize(m, ctr) for m in inter["interaction"]])
    # require >=1 assistant tool_call
    has_call = any(mm.get("tool_calls") for mm in msgs if mm["role"] == "assistant")
    if not has_call:
        return None
    ev = next((e for e in task["evaluations"] if e.get("success")), task["evaluations"][0])
    return {"tools": tools, "messages": msgs,
            "_meta": {"bench": "sopbench", "domain": task.get("domain") or domain,
                      "goal": ev.get("user_goal"), "should_succeed": ev.get("action_should_succeed"),
                      "n_calls": ctr[0]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--sopbench_root", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    actions = load_actions(a.sopbench_root, a.domain)
    tools = to_tools(actions)
    data = json.load(open(a.rollout, encoding="utf-8"))
    succ = [t for t in data if any(e.get("success") for e in t["evaluations"])]
    out, skip = [], 0
    for t in succ:
        c = convert_task(t, tools, a.domain)
        if c is None:
            skip += 1
        else:
            out.append(c)
    sT = sum(1 for c in out if c["_meta"]["should_succeed"])
    print("domain=%s tools=%d success=%d converted=%d skip=%d (should_T=%d should_F=%d)"
          % (a.domain, len(tools), len(succ), len(out), skip, sT, len(out) - sT))
    if a.sample:
        for c in out[:a.sample]:
            print("\n=== goal=%s should_succeed=%s calls=%d ===" % (c["_meta"]["goal"], c["_meta"]["should_succeed"], c["_meta"]["n_calls"]))
            for m in c["messages"]:
                if m["role"] == "assistant":
                    tc = " | ".join("%s(%s)" % (t["function"]["name"], t["function"]["arguments"][:50]) for t in m.get("tool_calls", []))
                    print("  [assistant] %s %s" % (str(m["content"])[:40], "-> " + tc if tc else ""))
                elif m["role"] == "tool":
                    print("    [tool] %s" % str(m["content"])[:50])
                elif m["role"] == "user":
                    print("  [user] %s" % str(m["content"])[:60])
                else:
                    print("  [system] %s" % str(m["content"])[:60])
    if not a.dry:
        with open(a.out, "w", encoding="utf-8") as f:
            for c in out:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
