#!/usr/bin/env python3
"""C38 — cfbsynth 학습 체크포인트의 *in-dist* gather 정확도 (E6′ 선결 측정).

정본 doc: reports/facet_rft_2026/E6PRIME_GATHER_LEARN_DESIGN_2026_07_08.md §1
물음: C4의 학습 arm은 (H-nolearn) 애초에 못 배웠나, (H-transfer) 배웠는데 tau2로 안 넘어갔나.
처방이 정반대이므로 이걸 먼저 가른다.

held-out = seed 7 (학습 seed 0). per-traj 랜덤 id라 암기 불가.

3 조건 (전이 축을 분해):
  A  in-dist       : 학습 포맷 그대로 (텍스트 [SYSTEM]/[TOOLS] + hermes <tool_call>) · **규칙 문장 있음**
  B  native FC     : messages+tools (OpenAI FC) · 규칙 문장 있음
  C  native FC-rule: messages+tools · **규칙 문장 제거**  ← tau2에 가장 가까움 (T5 검정)

지표:
  gather-rate : 값이 문맥에 없을 때 lookup_* 도구를 부르는 비율   (핵심)
  copy-rate   : 값이 이미 있을 때 act_* 를 *실값*으로 부르는 비율  (퇴화 "항상 읽기" 방지 확인)

Run:
  python3 c38_indist_gather_eval.py --base http://localhost:8140/v1 \
      --models base,dpo,sft --cond A,B,C --n 60
"""

import argparse
import json
import re
import urllib.request

PAIRS = "/home/woori/scratch/c38/heldout_pairs.jsonl"
NATIVE = "/home/woori/scratch/c38/heldout_native.jsonl"
NEUTRAL_SYS = "You are a tool-using assistant."
TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def post(base, payload, timeout=120):
    req = urllib.request.Request(base.rstrip("/") + payload["_path"],
                                 data=json.dumps({k: v for k, v in payload.items()
                                                  if not k.startswith("_")}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen_completion(base, model, prompt):
    out = post(base, {"_path": "/completions", "model": model, "prompt": prompt,
                      "max_tokens": 128, "temperature": 0.0})
    return out["choices"][0]["text"]


def gen_chat(base, model, messages, tools):
    out = post(base, {"_path": "/chat/completions", "model": model, "messages": messages,
                      "tools": tools, "tool_choice": "auto", "max_tokens": 128, "temperature": 0.0})
    return out["choices"][0]["message"]


def parse_hermes(text):
    m = TC_RE.search(text or "")
    if not m:
        return None, None
    try:
        d = json.loads(m.group(1))
        return d.get("name"), d.get("arguments") or {}
    except Exception:
        return None, None


def is_lookup(name):
    return bool(name) and name.startswith("lookup_")


def is_act(name):
    return bool(name) and name.startswith(("act_", "do_"))


def load_pairs(n):
    g, c = [], []
    for line in open(PAIRS, encoding="utf-8"):
        d = json.loads(line)
        (g if d["_kind"] == "gather" else c).append(d)
    return g[:n], c[:n]


def load_native(n):
    out = []
    for line in open(NATIVE, encoding="utf-8"):
        out.append(json.loads(line))
        if len(out) >= n:
            break
    return out


def cond_A(base, model, gathers, copies):
    """학습 포맷 그대로: prompt 텍스트 → hermes 완성."""
    gr = sum(1 for d in gathers if is_lookup(parse_hermes(gen_completion(base, model, d["prompt"]))[0]))
    ok = 0
    for d in copies:
        nm, args = parse_hermes(gen_completion(base, model, d["prompt"]))
        real = parse_hermes(d["chosen"])[1]
        ok += int(is_act(nm) and args == real)
    return gr / max(len(gathers), 1), ok / max(len(copies), 1)


def cond_native(base, model, natives, strip_rule):
    """native FC. messages[0]=system(규칙) messages[1]=user → 첫 assistant 행동 예측."""
    gr = 0
    for d in natives:
        msgs = []
        for m in d["messages"][:2]:
            if m["role"] == "system":
                msgs.append({"role": "system",
                             "content": NEUTRAL_SYS if strip_rule else m["content"]})
            else:
                msgs.append({"role": m["role"], "content": m.get("content") or ""})
        try:
            out = gen_chat(base, model, msgs, d["tools"])
        except Exception:
            continue
        tcs = out.get("tool_calls") or []
        nm = tcs[0]["function"]["name"] if tcs else None
        gr += int(is_lookup(nm))
    return gr / max(len(natives), 1), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--models", default="base,dpo,sft")
    ap.add_argument("--cond", default="A,B,C")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    gathers, copies = load_pairs(a.n)
    natives = load_native(a.n)
    print(f"held-out: gather {len(gathers)} · copy {len(copies)} · native {len(natives)}\n", flush=True)

    conds = a.cond.split(",")
    print(f"{'model':<8}" + "".join(f"{'  cond '+c:>22}" for c in conds))
    for model in a.models.split(","):
        cells = []
        for c in conds:
            try:
                if c == "A":
                    g, cp = cond_A(a.base, model, gathers, copies)
                    cells.append(f"gather {g:.2f} copy {cp:.2f}")
                elif c == "B":
                    g, _ = cond_native(a.base, model, natives, strip_rule=False)
                    cells.append(f"gather {g:.2f}")
                else:
                    g, _ = cond_native(a.base, model, natives, strip_rule=True)
                    cells.append(f"gather {g:.2f}")
            except Exception as e:
                cells.append(f"ERR {type(e).__name__}")
            print(f"  [{model}/{c}] {cells[-1]}", flush=True)
        print(f"{model:<8}" + "".join(f"{x:>22}" for x in cells), flush=True)

    print("\n판정: A 높고 C 낮으면 → 규칙-프롬프트/포맷 조건부 학습(T5/T1) = H-transfer")
    print("      A도 낮으면 → 애초에 못 배움 = H-nolearn (데이터·손실 재설계)")


if __name__ == "__main__":
    main()
