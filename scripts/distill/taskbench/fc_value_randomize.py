#!/usr/bin/env python
"""R1b root-cause fix: 인자-값 randomization (2026-06-14).

문제: 모델이 placeholder 값(johndoe@example.com·John Doe·12345)을 *날조*. 원인 = 학습이
  assistant에게 *특정 값*(john_doe 등)을 emit하도록 supervise → 값을 *memorize* → τ²서 placeholder 날조.
처방: 각 궤적의 user-제공 식별값을 *포맷-보존 랜덤 토큰*으로 일관 치환(user 발화 + tool-call 인자 + tool 출력 모두).
  → 값이 매 궤적 달라 memorize 불가 → 모델은 *user 발화서 복사*해야 호출 가능 = R1b(컨텍스트서 복사) 강제.
  도구명 alias의 *값* 버전.

Usage: fc_value_randomize.py --in <conv.jsonl> --out <out.jsonl> [--seed 42] [--sample N]
"""
import argparse, json, random, string

AUTH_ARGS = ("username", "identification", "password", "email", "first_name", "last_name",
             "zip", "user_id", "phone", "dob", "license", "ssn", "member", "name", "id")


def rand_like(s, rng):
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(rng.choice(string.digits))
        elif ch.isalpha():
            out.append(rng.choice(string.ascii_lowercase) if ch.islower() else rng.choice(string.ascii_uppercase))
        else:
            out.append(ch)
    return "".join(out)


def first_auth_vals(msgs):
    """첫 assistant tool-call의 식별-인자 값들(=user 제공) 수집."""
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                continue
            if isinstance(args, dict):
                vals = set()
                for k, v in args.items():
                    if any(h in k.lower() for h in AUTH_ARGS) and isinstance(v, (str, int, float)):
                        sv = str(v)
                        if len(sv) >= 3 and not sv.isdigit() or (sv.isdigit() and len(sv) >= 4):
                            vals.add(sv)
                if vals:
                    return vals
    return set()


def multi_replace(text, repl):
    # 긴 값부터(부분 겹침 방지)
    for old in sorted(repl, key=len, reverse=True):
        text = text.replace(old, repl[old])
    return text


def randomize(ex, rng):
    msgs = ex["messages"]
    vals = first_auth_vals(msgs)
    if not vals:
        return ex
    repl = {v: rand_like(v, rng) for v in vals}
    for m in msgs:
        if m.get("role") == "system":
            continue  # 정책 텍스트 보존
        if isinstance(m.get("content"), str):
            m["content"] = multi_replace(m["content"], repl)
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = multi_replace(tc["function"]["arguments"], repl)
    ex["_meta"] = dict(ex.get("_meta", {}))
    ex["_meta"]["val_random"] = True
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    out, n_rand = [], 0
    for ex in rows:
        before = json.dumps(ex.get("_meta", {}))
        ex = randomize(ex, rng)
        if ex["_meta"].get("val_random"):
            n_rand += 1
        out.append(ex)
    print("input=%d randomized=%d (no-auth-val=%d)" % (len(rows), n_rand, len(rows) - n_rand))
    if a.sample:
        for ex in [e for e in out if e["_meta"].get("val_random")][:a.sample]:
            print("\n=== randomized goal=%s ===" % ex["_meta"].get("goal"))
            for m in ex["messages"][:5]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    print("  [assistant] CALL", [(t["function"]["name"], t["function"]["arguments"][:60]) for t in m["tool_calls"]])
                else:
                    print("  [%s] %s" % (m["role"], str(m.get("content"))[:80]))
    with open(a.out, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
