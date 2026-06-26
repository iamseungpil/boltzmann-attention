#!/usr/bin/env python
"""CFB-참조 추상 synth — P2b(gather-for-arg 2-hop) + P4(select-from-output) 의 *가장 추상적 구조*만.

근거: solo LoRA가 CFB 제외 → retail 연속홉 fetch(get_user_details→order_id→get_order_details)가
요구하는 P2b/P4 미학습 → order_id 날조(PRIMITIVE_COVERAGE_MATRIX:87-88). CFB 직접=표면매핑 위험
([[12-diversity-required]]) → CFB의 *추상 구조*를 합성. fc_randomize_fetchable.py(v6) 메커니즘 계승:
tool-출력→하류인자 값을 per-traj randomize → 유일 in-context 출처=getter 출력 → 인자 내려면 getter
선행 필수 = P2b(gather-for-arg)+P1(copy·무날조). 리스트 출력+선택 = P4.

resolve_selection synth과 *다른 stratum*: 거기=content-op(엔진 grounds·id emit 안 함) / 여기=flow
grounding(모델이 getter 출력서 id를 *복사*해 consumer 인자로 emit). 둘 다 합본해도 충돌 안 함(분리 규칙).

출력 = native-FC jsonl(tbnfc/synth_to_nativefc 동형):
  {tools:[getter..., consumer], messages:[system,user, A(getter call), T(list), (A/T 추가홉),
   A(consumer call=copied id)], _meta:{bench:"cfbsynth", hops, list_n, ...}}

Usage: synth_fetch_nativefc.py --out fetch_native.jsonl --n 4000 --seed 0 [--max_hops 3] [--max_list 5]
"""
import argparse
import json
import random
import string

SYS = ("You are a tool-using assistant. When an argument value is not given, obtain it by calling the "
       "tool that produces it, then copy the value from that tool's output. Never invent an id. If the "
       "producing tool returns several records, pick the one matching the user's description.")


def _rid(rng, n=8):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def _name(rng, pre):
    return f"{pre}_{''.join(rng.choice(string.ascii_lowercase) for _ in range(5))}"


def _val(rng):
    # diverse value surface: word / code / number-ish
    k = rng.random()
    if k < 0.34:
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 7)))
    if k < 0.67:
        return _rid(rng, rng.randint(5, 9)).upper()
    return str(rng.randint(100, 99999))


def make_traj(rng, max_hops, max_list):
    """Build one abstract P2b(+P4) trajectory.

    Chain of getters g0..g_{h-1} then a consumer. The id needed by the consumer is produced ONLY by the
    last getter's output (randomized). Earlier getters produce the key for the next getter (3-hop). The
    last getter may return a LIST -> P4 select by a discriminating attribute the user states.
    """
    hops = rng.randint(2, max_hops)          # total tool calls incl. consumer (>=2 = at least one fetch)
    n_getters = hops - 1
    # anonymized tool + field names (per-traj => structure-only learning)
    getters = [_name(rng, "lookup") for _ in range(n_getters)]
    consumer = _name(rng, "do")
    id_field = _name(rng, "id")              # the field the consumer's arg reads
    key_field = _name(rng, "key")
    attr_field = _name(rng, "attr")          # P4 discriminator
    consumer_arg = _name(rng, "arg")

    # user gives: an initial key (P1 from user) + a descriptive attr value (for P4 select)
    user_key = _val(rng)
    sel_val = _val(rng)
    list_n = rng.randint(1, max_list)        # 1 => pure P2b; >1 => P2b + P4 select
    target_idx = rng.randint(0, list_n - 1)
    target_id = _rid(rng)                    # the gold id (randomized -> must be copied from output)

    # tool schemas
    tools = []
    for i, g in enumerate(getters):
        keyname = key_field if i == 0 else f"{key_field}{i}"
        tools.append({"type": "function", "function": {
            "name": g, "description": f"Look up records by {keyname}. Returns matching record(s).",
            "parameters": {"type": "object", "properties": {
                keyname: {"type": "string", "description": "the lookup key"}}, "required": [keyname]}}})
    tools.append({"type": "function", "function": {
        "name": consumer, "description": f"Perform the action on the item identified by {consumer_arg}.",
        "parameters": {"type": "object", "properties": {
            consumer_arg: {"type": "string", "description": f"the {id_field} of the target item"}},
            "required": [consumer_arg]}}})
    rng.shuffle(tools)

    # user utterance (abstract, no id given)
    user = (f"Please {consumer.split('_')[0]} the item where {attr_field} is '{sel_val}'. "
            f"My {key_field} is {user_key}. I don't have the {id_field}.")

    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]

    # hop chain: each getter call -> tool output; key for next getter copied from prior output
    cur_key_name, cur_key_val = key_field, user_key
    for i, g in enumerate(getters):
        last = (i == n_getters - 1)
        # assistant calls getter with the current key (copied from user or prior output)
        messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": f"c{i}", "type": "function",
            "function": {"name": g, "arguments": json.dumps({cur_key_name: cur_key_val})}}]})
        if last:
            # last getter returns the records list (P4 target inside)
            recs = []
            for j in range(list_n):
                rec = {id_field: (target_id if j == target_idx else _rid(rng)),
                       attr_field: (sel_val if j == target_idx else _val(rng))}
                recs.append(rec)
            out = recs if list_n > 1 else recs[0]
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(out)})
        else:
            # intermediate getter returns next key (chain) -> forces P2b across hops
            nxt_name, nxt_val = f"{key_field}{i+1}", _val(rng)
            messages.append({"role": "tool", "tool_call_id": f"c{i}",
                             "content": json.dumps({nxt_name: nxt_val})})
            cur_key_name, cur_key_val = nxt_name, nxt_val

    # consumer call: arg = target_id COPIED from the last getter output (P2b/P1; P4 if list_n>1)
    messages.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": "cz", "type": "function",
        "function": {"name": consumer, "arguments": json.dumps({consumer_arg: target_id})}}]})
    messages.append({"role": "tool", "tool_call_id": "cz",
                     "content": json.dumps({"status": "ok", "tool": consumer})})

    return {"tools": tools, "messages": messages,
            "_meta": {"bench": "cfbsynth", "hops": hops, "list_n": list_n,
                      "primitive": "P2b" + ("+P4" if list_n > 1 else "")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_hops", type=int, default=3)
    ap.add_argument("--max_list", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from collections import Counter
    stats = Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for _ in range(a.n):
            ex = make_traj(rng, a.max_hops, a.max_list)
            stats[ex["_meta"]["primitive"]] += 1
            stats[f"hops{ex['_meta']['hops']}"] += 1
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[cfbsynth] wrote {a.n} -> {a.out}  dist={dict(stats)}")


if __name__ == "__main__":
    main()
