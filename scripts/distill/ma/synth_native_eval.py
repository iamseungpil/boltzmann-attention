#!/usr/bin/env python3
"""facet (3) keystone eval — NATIVE op-naming transfer (격리: formalize 출력만 채점·e2e 아님).

served 모델(hermes parser)에 {tools:[resolve_selection], messages:[system,user]} 보내고
emit된 resolve_selection tool_call을 파싱 → op/operand를 gold_op_ir(anchor_id 제외)와 결정론 매치.
= §21(op-IR)을 native 포맷서 재현하나 = §23E 다리 가부.

채점:
  recognition = op 라벨 일치 (NL→생성원 명명)
  operand     = op + among + attr + dir + set 일치 (anchor_id 제외; concrete 아님·intensional)
Usage: synth_native_eval.py --data heldout_native.jsonl --base http://localhost:PORT/v1 --model TAG --out out.json
"""
import argparse, json, sys
from collections import Counter, defaultdict
import urllib.request


def chat_toolcall(base, model, tools, messages, timeout=60):
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "tool_choice": "auto", "temperature": 0.0, "max_tokens": 256}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None, msg.get("content")  # 모델이 tool_call 안 함 (native 붕괴 신호)
    fn = tcs[0]["function"]
    try:
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    except Exception:
        args = {}
    return {"name": fn["name"], "args": args}, None


def _norm(d):
    """비교용 정규화 (anchor_id 제외·값 문자열화)."""
    if not isinstance(d, dict):
        return {}
    return {k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v))
            for k, v in d.items() if k != "anchor_id"}


def operand_match(emit, gold):
    return _norm(emit) == _norm({k: v for k, v in gold.items() if k != "anchor_id"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    by_op = defaultdict(lambda: {"n": 0, "recog": 0, "operand": 0, "no_call": 0})
    emit_ops = Counter()
    fails = []
    for r in rows:
        gold = r["_meta"]["gold_op_ir"]
        op = r["_meta"]["op"]
        tools = r["tools"]
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
        try:
            emit, content = chat_toolcall(a.base, a.model, tools, msgs)
        except Exception as e:
            emit, content = None, f"ERR {type(e).__name__}: {e}"
        s = by_op[op]; s["n"] += 1
        if emit is None:
            s["no_call"] += 1
            if len(fails) < 30:
                fails.append({"op": op, "reason": "no_tool_call", "content": str(content)[:120]})
            continue
        emit_ops[emit["args"].get("op", "?")] += 1
        recog = (emit["args"].get("op") == op)
        if recog:
            s["recog"] += 1
        if operand_match(emit["args"], gold):
            s["operand"] += 1
        elif len(fails) < 30:
            fails.append({"op": op, "emit": emit["args"], "gold": {k: v for k, v in gold.items() if k != "anchor_id"}})

    tot = {"n": 0, "recog": 0, "operand": 0, "no_call": 0}
    for op, s in by_op.items():
        for k in tot:
            tot[k] += s[k]
    def rate(s, k):
        return round(s[k] / s["n"], 3) if s["n"] else None
    summary = {"overall": {"n": tot["n"], "recognition": rate(tot, "recog"),
                           "operand_acc": rate(tot, "operand"), "no_tool_call": rate(tot, "no_call")},
               "by_op": {op: {"n": s["n"], "recognition": rate(s, "recog"),
                              "operand_acc": rate(s, "operand"), "no_call": s["no_call"]}
                         for op, s in sorted(by_op.items())},
               "emitted_op_dist": dict(emit_ops.most_common()),
               "model": a.model, "fails_sample": fails[:30]}
    json.dump(summary, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary["overall"], ensure_ascii=False))
    print("by_op:", json.dumps({k: v["recognition"] for k, v in summary["by_op"].items()}, ensure_ascii=False))
    print("emit_dist:", json.dumps(summary["emitted_op_dist"], ensure_ascii=False))


if __name__ == "__main__":
    main()
