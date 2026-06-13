#!/usr/bin/env python
"""ⓟ1 agent 격리 결정론 테스트 (2026-06-14 사용자 재조사):
det 런의 *실제 context*를 고정해 served 7B에 N회 반복 생성 → 동일성. user-sim 완전 배제 =
"같은 입력에 agent가 다른 답을 내나"의 엄밀 격리. enforce-eager+max-num-seqs1+seed0 serve 전제.

동일하지 않으면 = agent vLLM 커널 비결정(FP 비결합/atomic, seqs1이 못 고침·batch-invariant가 고침) 확정.
동일하면 = census의 8%는 다른 원인(컨텍스트 차이 등) → 재검토.

Usage: t2_agent_determinism.py --endpoint http://localhost:8351/v1 \
  --simdir <det results.json 폴더> --n 30 [--nctx 8]
"""
import argparse, json, urllib.request


def served_model(endpoint):
    with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def gen(endpoint, model, messages, max_tokens=256):
    payload = {"model": model, "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens, "seed": 0}
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def render(messages):
    """대화를 플랫 텍스트로 (tool 메시지 format 문제 회피·긴 실제 context)."""
    out = []
    for m in messages:
        r = m.get("role"); c = m.get("content")
        tc = m.get("tool_calls")
        if r == "assistant" and tc:
            acts = [t.get("function", {}).get("name", "") + "(" + str(t.get("function", {}).get("arguments", "")) + ")" for t in tc]
            out.append(f"Assistant: [calls {', '.join(acts)}]")
        elif r == "tool":
            out.append(f"Tool result: {c}")
        elif r == "user":
            out.append(f"User: {c}")
        elif r == "assistant":
            out.append(f"Assistant: {c}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--simdir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--nctx", type=int, default=8, help="테스트할 context(궤적) 수")
    a = ap.parse_args()
    model = served_model(a.endpoint)
    print(f"[agent-det] model={model} n_repeat={a.n}")
    d = json.load(open(f"{a.simdir}/results.json"))
    sims = d["simulations"]
    # 서로 다른 task의 trial0를 context로 — 다양한 실제 입력에서 반복 동일성
    seen, ctxs = set(), []
    for s in sims:
        tid = s["task_id"]
        if tid in seen:
            continue
        seen.add(tid)
        ms = s.get("messages", [])
        # 첫 assistant action 직전까지 = 깨끗한 prefix (그 지점서 모델이 행동 생성)
        cut = next((i for i, m in enumerate(ms) if m.get("role") == "assistant" and m.get("tool_calls")), len(ms))
        if cut < 2:
            continue
        ctxs.append((tid, ms[:cut]))
        if len(ctxs) >= a.nctx:
            break

    policy = sims[0].get("policy", "")
    n_ident = 0
    for tid, prefix in ctxs:
        sysmsg = {"role": "system", "content": policy + "\n\n# CONVERSATION SO FAR #\n" + render(prefix) +
                  "\n\nGenerate the assistant's next response."}
        msgs = [sysmsg, {"role": "user", "content": "(continue)"}]
        outs = [gen(a.endpoint, model, msgs) for _ in range(a.n)]
        uniq = len(set(outs))
        ident = uniq == 1
        n_ident += ident
        print(f"  task={tid}: {a.n}회 중 distinct={uniq} -> {'동일(결정론)' if ident else 'DIFFER(비결정)'}")
        if not ident:
            for o in sorted(set(outs))[:3]:
                print(f"     >> {o[:80]!r}")
    print(f"[agent-det 결론] {n_ident}/{len(ctxs)} context가 {a.n}회 완전동일 "
          f"= agent vLLM은 {'결정론(8% census는 다른 원인)' if n_ident==len(ctxs) else '비결정(커널 FP/atomic — batch-invariant 필요)'}")


if __name__ == "__main__":
    main()
