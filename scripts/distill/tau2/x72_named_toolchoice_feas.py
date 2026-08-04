"""Can the server be made to emit one specific discoverable read, without a deny stub?

`require_tool_before` was demoted to advice because the only enforcement it had was a deny
stub committed as a tool output, and tau2 re-runs the environment to compare that output
byte for byte — three simulations died of it (C210/day6). A generation-side constraint has
no such problem: it produces no tool output at all, only the model's own call.

But naming the tool is not enough here. The discoverable tools are not in the `tools`
array — the array holds the dispatcher, and the tool that matters travels inside it as the
`agent_tool_name` argument. So forcing the read needs two things at once:

  T2  tool_choice names `unlock_discoverable_agent_tool`   → the call happens
  T3  ...and that call's schema admits exactly one value   → it is *this* read

T3 is the one that decides whether the lever is buildable, and it is a plain single-value
enum in the function schema, not a vLLM-specific extension. T1 is the baseline already in
production (T2_FORCE_ACTION), included so a server-side failure is distinguishable from a
feature gap.

Free: four short completions against the already-running server.
"""

import argparse
import json
import urllib.request

TARGET = "get_all_user_accounts_by_user_id_3847"

DISPATCH = {
    "type": "function",
    "function": {
        "name": "unlock_discoverable_agent_tool",
        "description": "Unlock a discoverable agent tool so it can be called.",
        "parameters": {
            "type": "object",
            "properties": {"agent_tool_name": {"type": "string"}},
            "required": ["agent_tool_name"],
        },
    },
}
PINNED = json.loads(json.dumps(DISPATCH))
PINNED["function"]["parameters"]["properties"]["agent_tool_name"] = {
    "type": "string", "enum": [TARGET]}

OTHER = {
    "type": "function",
    "function": {
        "name": "KB_search_bm25",
        "description": "Search the knowledge base.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    },
}

# A turn shaped like the ones that failed: the agent has just been told the account was not
# found, and in the run it answered with prose or another search instead of the read.
CONV = [
    {"role": "system", "content": "You are a bank agent. Use tools."},
    {"role": "user", "content": "My debit cards are all PIN locked, can you help?"},
    {"role": "assistant", "content": "Let me look up your cards."},
    {"role": "user", "content": "Thanks."},
    {"role": "assistant", "content": "I could not find the account."},
    {"role": "user", "content": "So what now?"},
]


def post(base, body, timeout=180):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def describe(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return "TEXT(호출 0): " + (msg.get("content") or "")[:70]
    out = []
    for tc in tcs:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass
        out.append(f"{fn.get('name')}({json.dumps(args, ensure_ascii=False)[:70]})")
    return " + ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    named = {"type": "function", "function": {"name": "unlock_discoverable_agent_tool"}}
    arms = [
        ("T0 tool_choice=auto (대조)", [DISPATCH, OTHER], "auto"),
        ("T1 tool_choice=required (현행 FORCE_ACTION)", [DISPATCH, OTHER], "required"),
        ("T2 named tool_choice", [DISPATCH, OTHER], named),
        ("T3 named + 단일값 enum (표적 고정)", [PINNED, OTHER], named),
    ]

    for label, tools, choice in arms:
        hits = 0
        seen = []
        for _ in range(args.n):
            body = {"model": args.model, "messages": CONV, "tools": tools,
                    "tool_choice": choice, "temperature": 0.0,
                    # required/named finish a whole tool-call JSON; a small cap truncates it
                    # into a misleading 400 (vLLM #19051/#36794).
                    "max_tokens": 1500}
            try:
                d = post(args.base, body)
                m = d["choices"][0]["message"]
                s = describe(m)
                if TARGET in s:
                    hits += 1
            except Exception as e:
                s = "ERR " + repr(e)[:110]
            seen.append(s)
        print(f"\n=== {label}")
        print(f"    표적({TARGET}) 적중 {hits}/{args.n}")
        for s in seen:
            print("      " + s)


if __name__ == "__main__":
    main()
