#!/usr/bin/env python
"""cfbsynth DPO 페어 — fetch-first prior(schema-example copy) 억제용 penalty 학습.

SFT(양성예시)는 copy reflex 못 죽임(실측 schema_copy 52). DPO는 *나쁜 행동을 벌점*으로 직접 누름.
verifier = provenance(GT-도출·reward model 불요): chosen=관측된 real id 복사 / rejected=스키마 예시값 복사.
도메인-일반(익명 도구·per-traj 랜덤 id·tau2 0). dpo_train.py 포맷 {prompt, chosen, rejected} 텍스트.

페어 2종/태스크:
  (gather)  prompt=[req·id없음]            chosen=getter 호출           rejected=consumer(id=예시값)  # fetch 대신 날조
  (copy)    prompt=[req+getter출력(real id)] chosen=consumer(id=real)   rejected=consumer(id=예시값)  # 관측무시·예시복사
차이가 id 토큰뿐 → margin이 copy prior를 정조준. tool_call=hermes 텍스트(생성 포맷 일치).

Run: cfbsynth_dpo_pairs.py --out pairs.jsonl --n 3000 --seed 0
"""
import argparse
import json
import random
import string

SYS = ("You are a tool-using assistant. When an argument value is not given, obtain it by calling "
       "the tool that produces it, then copy the value from that tool's output. Never invent an id "
       "or copy an example value from a tool's schema.")


def _rid(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def _tc(name, args):  # hermes tool_call 텍스트(Qwen 생성 포맷)
    return '<tool_call>\n{"name": "' + name + '", "arguments": ' + json.dumps(args) + '}\n</tool_call>'


def make(rng):
    getter = "lookup_" + _rid(rng, 4)
    consumer = "act_" + _rid(rng, 4)
    idfield = rng.choice(["id", "ref", "code", "number"])
    attr = rng.choice(["color", "name", "type", "label", "status"])
    val = rng.choice(["blue", "red", "alpha", "gamma", "open", "closed"])
    key = "key_" + _rid(rng, 3)
    keyval = _rid(rng, 5)
    example_id = rng.choice(["0000000", "00000", "EXAMPLE", "xxxxxxx", "123456"])  # 스키마 예시값(=rejected)
    n = rng.randint(2, 5)
    recs = [{idfield: _rid(rng, 7), attr: _rid(rng, 4)} for _ in range(n)]
    match = rng.randrange(n)
    recs[match][attr] = val
    real_id = recs[match][idfield]

    # 도구 스키마(예시값 포함=날조 미끼)
    tools = [
        {"type": "function", "function": {"name": getter, "description": "look up records by key",
         "parameters": {"type": "object", "properties": {key: {"type": "string"}}, "required": [key]}}},
        {"type": "function", "function": {"name": consumer, "description": "act on a record",
         "parameters": {"type": "object", "properties": {
             idfield: {"type": "string", "description": f"the record {idfield}, e.g. {example_id}"},
         }, "required": [idfield]}}},
    ]
    user = (f"Please {consumer} the record where {attr} is {val}. My {key} is {keyval}. "
            f"I don't have the {idfield}.")
    getter_out = json.dumps(recs)

    sys_user = f"[SYSTEM]\n{SYS}\n\n[TOOLS]\n{json.dumps(tools)}\n\n[USER]\n{user}"
    # pair 1: gather-first
    p_gather = sys_user + "\n\nWhat is your next tool call?"
    c_gather = _tc(getter, {key: keyval})
    r_gather = _tc(consumer, {idfield: example_id})
    # pair 2: copy real (after getter output)
    p_copy = (sys_user + "\n\n[ASSISTANT]\n" + c_gather + "\n\n[TOOL OUTPUT]\n" + getter_out
              + "\n\nWhat is your next tool call?")
    c_copy = _tc(consumer, {idfield: real_id})
    r_copy = _tc(consumer, {idfield: example_id})
    return [
        {"prompt": p_gather, "chosen": c_gather, "rejected": r_gather, "_kind": "gather"},
        {"prompt": p_copy, "chosen": c_copy, "rejected": r_copy, "_kind": "copy"},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000, help="총 페어 수(태스크당 2)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = []
    while len(out) < a.n:
        out.extend(make(rng))
    out = out[:a.n]
    with open(a.out, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(e["_kind"] for e in out)
    print(f"[cfbsynth_dpo] {len(out)} pairs -> {a.out} · kinds={dict(c)}")


if __name__ == "__main__":
    main()
