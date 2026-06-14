#!/usr/bin/env python
"""R1b ask-user augmentation (catastrophic-forgetting fix, 2026-06-14).

문제: SOPBench/TaskBench 학습데이터는 정보-upfront(user 첫 발화에 creds 다 줌) → 모델이
  "묻지 말고 바로 호출" 학습 → base의 ask-user 파국적 망각 → 인자값 날조(τ² fctbox 0.10/0.05<base 0.17).
처방(R1b): user 첫 발화의 값을 빼고 ask-then-provide로 분할 → "값 없으면 묻고, 준 값을 쓴다"를 학습.

변환(결정론 v1·템플릿):
  원본:  [sys, USER1(요청+값), ASST(tool_calls...), tool, ...]
  합성:  [sys, USER1'(값없는 generic 요청, goal서 도출),
              ASST_ask("...님 신원확인을 위해 <params>를 알려주시겠어요?"),
              USER2(<param>: <value> — 첫 tool-call 인자서 추출),
              ASST(원본 tool_calls...), tool, ...]
= R1b 직접 학습. 원본도 별도 유지(혼합)해 자연 어법 + ask 둘 다 노출.

Usage: fc_askuser_augment.py --in <conv.jsonl> --out <aug.jsonl> --frac 0.4 [--seed 42] [--sample N]
"""
import argparse, json, random

# 사용자-제공 인자(물어볼 값)일 만한 키 — 도구출력 파생값(order_id 등 lookup) 제외, 신원/입력류만
USER_PROVIDED_HINT = ("username", "identification", "password", "email", "first_name", "last_name",
                      "zip", "name", "phone", "dob", "birthday", "income", "asset", "license", "ssn",
                      "amount", "id_number", "member")


def goal_phrase(goal):
    return (goal or "complete my request").replace("_", " ")


def first_user_args(messages):
    """첫 assistant tool_call의 인자 중 user-제공류만 반환 (없으면 전체 인자)."""
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            try:
                args = json.loads(m["tool_calls"][0]["function"]["arguments"])
            except Exception:
                return None
            if not isinstance(args, dict):
                return None
            up = {k: v for k, v in args.items() if any(h in k.lower() for h in USER_PROVIDED_HINT)}
            return up or args
    return None


def augment(ex):
    msgs = ex["messages"]
    if not msgs or msgs[0]["role"] != "system":
        return None
    sys_msg = msgs[0]
    # 원본에서 첫 assistant(tool_call 보유) 인덱스
    first_a = next((i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")), None)
    if first_a is None:
        return None
    up = first_user_args(msgs)
    if not up:
        return None
    goal = ex["_meta"].get("goal")
    params = ", ".join(up.keys())
    provided = ", ".join("%s: %s" % (k, v) for k, v in up.items())
    new = [sys_msg,
           {"role": "user", "content": "Hi! I'd like to %s." % goal_phrase(goal)},
           {"role": "assistant", "content": "I can help with that. To proceed I'll need to verify your identity — could you provide your %s?" % params, "_supervise": True},
           {"role": "user", "content": "Sure — %s." % provided}]
    # 원본의 첫 assistant부터 끝까지 이어붙임 (tool_calls는 이제 user-제공 값에 grounded)
    new += msgs[first_a:]
    out = dict(ex)
    out["messages"] = new
    out["_meta"] = dict(ex["_meta"]); out["_meta"]["askuser_aug"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    aug, skip = [], 0
    for ex in rows:
        if rng.random() < a.frac:
            g = augment(ex)
            if g:
                aug.append(g)
            else:
                skip += 1
    print("input=%d  augmented=%d  skip(no-args/no-call)=%d  frac=%.2f" % (len(rows), len(aug), skip, a.frac))
    if a.sample:
        for g in aug[:a.sample]:
            print("\n=== aug goal=%s ===" % g["_meta"].get("goal"))
            for m in g["messages"][:6]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    print("  [assistant] CALL", [t["function"]["name"] for t in m["tool_calls"]])
                else:
                    print("  [%s] %s" % (m["role"], str(m.get("content"))[:90]))
    with open(a.out, "w", encoding="utf-8") as f:
        for g in aug:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
