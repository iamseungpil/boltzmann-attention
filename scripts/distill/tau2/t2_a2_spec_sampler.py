#!/usr/bin/env python
"""P-A2-1 역방향 데이터엔진 1단: GATE_SPEC 문법-샘플러 (프로그램, 결정론).

spec을 먼저 무작위 합성 → frontier가 NL 정책으로 렌더 → (NL, spec) GT 완벽 쌍.
난이도 손잡이 = 게이트 수·predicate 유형 다양성·satisfier 도구 수. 가짜 도구 카탈로그
동반 생성(렌더러가 도구명을 알아야 함). 시드 고정(Date/random 미사용 — args로 받음).

Usage: t2_a2_spec_sampler.py --n 200 --seed 0 --out specs_synth.jsonl
출력 줄 = {"spec": {...}, "catalog": {...}, "domain_hint": "..."}
"""
import argparse, json, random

# predicate 유형 라이브러리 (실 정책서 귀납 — retail/airline 6게이트 + 일반화)
PRED_TYPES = [
    {"id": "AUTH_FIRST", "predicate": "authenticated user identity",
     "kind": "satisfier", "ask_tools": 1},
    {"id": "CONFIRM_WRITE", "predicate": "explicit user confirmation of the action",
     "kind": "ask"},
    {"id": "OWNER_SCOPE", "predicate": "target record belongs to the authenticated user",
     "kind": "terminal"},
    {"id": "ELIGIBILITY", "predicate": "action eligibility rules satisfied (status/time-window/tier)",
     "kind": "terminal", "db": True},
    {"id": "ARG_CONSTRAINT", "predicate": "argument composition within policy limits",
     "kind": "terminal", "db": True},
    {"id": "PREREQ_FETCH", "predicate": "required record fetched before mutation",
     "kind": "satisfier", "ask_tools": 1},
    {"id": "STATE_INVARIANT", "predicate": "resource not in a frozen/terminal state",
     "kind": "terminal", "db": True},
]
VERBS = ["create", "update", "cancel", "modify", "issue", "transfer", "adjust", "close", "renew"]
NOUNS = ["reservation", "order", "account", "policy", "claim", "subscription", "ticket", "loan"]
READS = ["get", "lookup", "find", "search", "list"]
ATTRS = ["id", "email", "name", "zip", "date_of_birth", "membership", "reason", "amount"]


def sample_catalog(rng, n_write, n_read):
    tools = {}
    for _ in range(n_write):
        nm = f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}"
        tools[nm] = {"type": "WRITE",
                     "required": rng.sample(ATTRS, rng.randint(1, 3))}
    for _ in range(n_read):
        nm = f"{rng.choice(READS)}_{rng.choice(NOUNS)}_details"
        tools[nm] = {"type": "READ", "required": [rng.choice(ATTRS)]}
    return tools


def build_spec(rng, catalog):
    writes = [t for t, c in catalog.items() if c["type"] == "WRITE"]
    reads = [t for t, c in catalog.items() if c["type"] == "READ"]
    user_scoped = writes + rng.sample(reads, min(len(reads), rng.randint(0, len(reads))))
    n_gates = rng.randint(2, 5)
    chosen = rng.sample(PRED_TYPES, min(n_gates, len(PRED_TYPES)))
    spec = {}
    for i, p in enumerate(chosen, 1):
        gid = f"G{i}_{p['id']}"
        gate = {"predicate": p["predicate"], "satisfiers": {}}
        if p["kind"] == "satisfier" and reads:
            sat = rng.sample(reads, min(len(reads), p.get("ask_tools", 1)))
            gate["satisfiers"] = {s: catalog[s]["required"] for s in sat}
            gate["ask"] = "obtain the prerequisite, then retry"
        elif p["kind"] == "ask":
            gate["ask"] = "list details and obtain user confirmation"
        else:
            gate["terminal"] = "deny: policy condition not met"
        applies = writes if p["id"] in ("CONFIRM_WRITE",) else user_scoped
        gate["applies_to"] = sorted(rng.sample(applies, max(1, len(applies) - rng.randint(0, 1))))
        if p.get("db"):
            gate["db_check"] = f"<{p['id'].lower()} predicate over args/record>"
        spec[gid] = gate
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    with open(a.out, "w") as wf:
        for k in range(a.n):
            cat = sample_catalog(rng, rng.randint(2, 6), rng.randint(2, 5))
            spec = build_spec(rng, cat)
            domain = f"{rng.choice(['retail','travel','banking','insurance','telecom','logistics'])}-{k}"
            wf.write(json.dumps({"id": k, "domain_hint": domain,
                                 "catalog": cat, "spec": spec}) + "\n")
    print(f"[spec-sampler] {a.n} specs (seed={a.seed}) -> {a.out}")


if __name__ == "__main__":
    main()
