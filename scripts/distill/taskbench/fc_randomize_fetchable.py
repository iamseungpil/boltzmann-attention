#!/usr/bin/env python
"""v6: identity + tool-fetchable 값 randomization (R1/R2 fetch 강제, 2026-06-14).

근본원인(τ² 전수 autopsy): 모델이 **tool-fetchable 값**(order_id `#W0000000`·주소)을 fetch 안 하고
  placeholder *날조*. 원인 = SOPBench fetch-then-use 궤적(14.4%·예: get_provider_details→provider_id→
  submit_claim)의 **tool-출력 값이 randomize 안 됨 → 모델이 memorize로 우회** → τ²서 placeholder 날조.

처방(벤치-무관·thesis): tool-출력서 와서 이후 인자로 재사용되는 값을 **(출력+인자) 일관 randomize** →
  그 값의 유일한 in-context 출처 = tool 출력 → supervised 인자를 내려면 **getter 호출 선행 필수** =
  R1(복사·무날조)+R2(gather). 이미 전이 실증된 R1 도구-이름 grounding을 도구-출력-*값*으로 확장.
  ★provenance 보존: user 발화에도 있는 값은 user-제공(identity)이므로 fetchable로 randomize 안 함.

= fc_value_randomize.py(identity-only)의 상위호환. identity 패스 + fetchable 패스 둘 다 수행.

Usage: fc_randomize_fetchable.py --in sop_all.jsonl --out sop_rand2.jsonl [--seed 42] [--sample N]
"""
import argparse, json, random, re, string

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


def multi_replace(text, repl):
    for old in sorted(repl, key=len, reverse=True):
        if old:
            text = text.replace(old, repl[old])
    return text


# ---- identity pass (fc_value_randomize 동일) ----
def first_auth_vals(msgs):
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
                        if (len(sv) >= 3 and not sv.isdigit()) or (sv.isdigit() and len(sv) >= 4):
                            vals.add(sv)
                if vals:
                    return vals
    return set()


# ---- fetchable pass (신규) ----
def user_text(msgs):
    return " \n ".join(str(m.get("content") or "") for m in msgs if m.get("role") == "user")


def fetchable_vals(msgs):
    """tool 출력서 와서 이후 assistant tool_call 인자로 재사용 & user 발화엔 없는 값 = tool-fetchable.
    이것만 randomize하면 유일 출처=tool 출력 → fetch 강제."""
    ut = user_text(msgs)
    tool_text_upto = []  # 누적 tool 출력 텍스트(순서)
    seen_tool = ""
    cands = set()
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r == "tool":
            seen_tool += " \n " + str(m.get("content") or "")
        elif r == "assistant":
            for tc in m.get("tool_calls") or []:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    continue
                if not isinstance(args, dict):
                    continue
                for v in args.values():
                    if not isinstance(v, (str, int, float)):
                        continue
                    sv = str(v)
                    # 값-다움: len>=4 + (숫자/언더스코어/구두점 포함 = id/amount류), 출처=이전 tool 출력, user엔 없음
                    if len(sv) < 4:
                        continue
                    if sv in seen_tool and sv not in ut:
                        cands.add(sv)
    return cands


def randomize(ex, rng):
    msgs = ex["messages"]
    repl = {}
    for v in first_auth_vals(msgs):          # identity (user-제공)
        repl.setdefault(v, rand_like(v, rng))
    for v in fetchable_vals(msgs):           # tool-fetchable (도구-출력)
        repl.setdefault(v, rand_like(v, rng))
    if not repl:
        return ex, 0, 0
    n_id = len(first_auth_vals(msgs)); n_f = len(fetchable_vals(msgs))
    for m in msgs:
        if m.get("role") == "system":
            continue  # 정책 텍스트 보존
        if isinstance(m.get("content"), str):
            m["content"] = multi_replace(m["content"], repl)
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = multi_replace(tc["function"]["arguments"], repl)
    ex["_meta"] = dict(ex.get("_meta", {}))
    ex["_meta"]["val_random"] = True
    ex["_meta"]["fetchable_random"] = n_f > 0
    return ex, n_id, n_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    out = []
    n_any = n_fetch = tot_id = tot_f = 0
    for ex in rows:
        ex, nid, nf = randomize(ex, rng)
        if ex["_meta"].get("val_random"):
            n_any += 1
        if nf > 0:
            n_fetch += 1
        tot_id += nid; tot_f += nf
        out.append(ex)
    print("input=%d  randomized=%d  with-fetchable=%d (%.1f%%)" % (len(rows), n_any, n_fetch, 100*n_fetch/max(len(rows),1)))
    print("total identity-vals=%d  fetchable-vals=%d" % (tot_id, tot_f))
    if a.sample:
        for ex in [e for e in out if e["_meta"].get("fetchable_random")][:a.sample]:
            print("\n=== fetchable-random goal=%s ===" % ex["_meta"].get("goal"))
            for m in ex["messages"][:10]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    print("  [A] CALL", [(t["function"]["name"], t["function"]["arguments"][:70]) for t in m["tool_calls"]])
                elif m["role"] == "tool":
                    print("    [T]", str(m.get("content"))[:70])
                else:
                    print("  [%s] %s" % (m["role"], str(m.get("content"))[:70]))
    with open(a.out, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
