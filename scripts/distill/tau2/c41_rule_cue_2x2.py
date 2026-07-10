#!/usr/bin/env python3
"""C41 — 2×2: 시스템 규칙 문장 × "나는 그 값이 없다" 큐 (short-turn 합성).

물음(사용자 2026-07-09): *"시스템 프롬프트로 된다는 거냐?"*
C38은 `규칙有+큐有 = 0.98` / `규칙無+큐有 = 0.40` 두 칸만 알았다. 0.98이 **규칙 덕인지 큐 덕인지 미분리**.

4 조건 × 지표 3종:
  gather  : 첫 행동이 lookup_*            (원하는 행동)
  fabricate: 첫 행동이 act_*/do_* 인데 인자값이 문맥(user 발화)에 없음   ← tau2의 실패 양식
  ask/text: tool_call 없음(사용자에게 되묻거나 설명)

★fabricate가 0에 가까우면 그 합성은 **tau2의 실패를 재현하지 못한다** = 학습 데이터로 무용(C38).

Run: python3 c41_rule_cue_2x2.py --base http://localhost:8140/v1 --models base --n 60
"""

import argparse
import json
import re
import urllib.request

NATIVE = "/home/woori/scratch/c38/heldout_native.jsonl"
NEUTRAL_SYS = "You are a tool-using assistant."
# cfbsynth user turn 끝의 결손 큐: "I don't have the id_xxxx." / "... the id."
CUE_RE = re.compile(r"\s*I don'?t have the [^.]*\.\s*$", re.I)


def chat(base, model, messages, tools, timeout=120):
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "tool_choice": "auto", "temperature": 0.0, "max_tokens": 128}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]


def load(n):
    out = []
    for line in open(NATIVE, encoding="utf-8"):
        out.append(json.loads(line))
        if len(out) >= n:
            break
    return out


def build(d, rule, cue):
    sys_msg = next(m for m in d["messages"] if m["role"] == "system")["content"]
    usr_msg = next(m for m in d["messages"] if m["role"] == "user")["content"]
    if not cue:
        stripped = CUE_RE.sub("", usr_msg).strip()
        if stripped == usr_msg.strip():           # 큐 패턴 불일치 → 이 샘플은 제외
            return None
        usr_msg = stripped
    return [{"role": "system", "content": sys_msg if rule else NEUTRAL_SYS},
            {"role": "user", "content": usr_msg}]


def classify(out, msgs):
    tcs = out.get("tool_calls") or []
    if not tcs:
        return "ask/text"
    fn = tcs[0]["function"]
    name = fn["name"]
    if name.startswith("lookup_"):
        return "gather"
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:
        args = {}
    ctx = " ".join(m["content"] for m in msgs).lower()
    for v in args.values():
        if isinstance(v, str) and v.lower() not in ctx:
            return "fabricate"          # 문맥에 없는 값을 consumer 인자로
    return "act(grounded)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--models", default="base")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    data = load(a.n)

    for model in a.models.split(","):
        print(f"\n######## {model}")
        print(f"{'조건':<22}{'gather':>9}{'fabricate':>12}{'act(grounded)':>16}{'ask/text':>11}{'n':>5}")
        for rule in (True, False):
            for cue in (True, False):
                from collections import Counter
                c = Counter()
                for d in data:
                    msgs = build(d, rule, cue)
                    if msgs is None:
                        continue
                    try:
                        out = chat(a.base, model, msgs, d["tools"])
                    except Exception:
                        continue
                    c[classify(out, msgs)] += 1
                n = sum(c.values()) or 1
                tag = f"규칙{'有' if rule else '無'}·큐{'有' if cue else '無'}"
                print(f"{tag:<22}{c['gather']/n:>9.2f}{c['fabricate']/n:>12.2f}"
                      f"{c['act(grounded)']/n:>16.2f}{c['ask/text']/n:>11.2f}{n:>5}", flush=True)
    print("\n판정: '규칙有·큐無' 가 높으면 → 짧은 문맥선 시스템 프롬프트가 실제로 작동(=tau2 실패는 load)")
    print("      '규칙有·큐無' 가 낮고 fabricate↑ 이면 → 0.98은 큐 덕. 그리고 이 조건이 비로소 tau2 실패를 재현")


if __name__ == "__main__":
    main()
