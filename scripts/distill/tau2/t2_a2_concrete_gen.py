#!/usr/bin/env python
"""S1-diag 데이터: 구체-술어 합성 배치 (H4 = 구체성 갭 검증, 전과정 프로그램 — 렌더 비용 0).

S1 기각 부검의 1급 가설: 기존 합성 spec의 db_check/predicate가 전부 추상 플레이스홀더
("<eligibility predicate over args/record>") ↔ 실 spec(airline/telecom)은 구체 조건문
("gb_amount <= 2"·"bill.status == OVERDUE") — 모델이 구체 술어 작성을 학습한 적 없음.
처방: 샘플러가 구체 술어를 *프로그램으로 인스턴스화* + 정책 NL도 템플릿 산문으로 프로그램
렌더(조건이 산문에 등장 — QC 자동 충족). 스타일 다양성↓이지만 구체성 축을 분리 검증.

Usage: t2_a2_concrete_gen.py --n 60 --seed 8 --out specs/a2_s1d_concrete.jsonl
"""
import argparse, json, random, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_a2_spec_sampler import sample_catalog, NOUNS  # noqa: E402

STATUSES = ["Active", "Suspended", "Closed", "Pending", "Overdue", "Paid", "Draft", "Expired"]
UNITS = ["GB", "USD", "days", "items", "attempts"]


def concrete_gate(rng, kind, gid, applies, reads, catalog):
    noun = rng.choice(NOUNS).replace("_", " ")
    if kind == "ELIGIBILITY":
        st = rng.choice(STATUSES)
        pred = f"the {noun} status is {st}"
        db = f"record.status == '{st}'"
        term = f"deny: allowed only while the {noun} is {st} (the tool does not check this)"
        return {"predicate": pred, "satisfiers": {}, "terminal": term,
                "applies_to": applies, "db_check": db}
    if kind == "LIMIT":
        n = rng.choice([1, 2, 3, 5, 10, 30, 100])
        u = rng.choice(UNITS)
        field = rng.choice(["amount", "gb_amount", "quantity", "duration"])
        pred = f"{field} within the policy limit of {n} {u}"
        db = f"{field} <= {n}"
        term = f"deny: the maximum allowed {field} is {n} {u}"
        return {"predicate": pred, "satisfiers": {}, "terminal": term,
                "applies_to": applies, "db_check": db}
    if kind == "STATE":
        bad = rng.choice(["frozen", "disputed", "archived", "locked"])
        pred = f"the {noun} record is not {bad}"
        db = f"record.state != '{bad}'"
        term = f"deny: {bad} {noun} records cannot be modified"
        return {"predicate": pred, "satisfiers": {}, "terminal": term,
                "applies_to": applies, "db_check": db}
    if kind == "COMPOUND":
        st = rng.choice(STATUSES)
        n = rng.choice([7, 14, 30, 90])
        pred = (f"all {noun} obligations are {st} AND the request is within "
                f"{n} days of the original date")
        db = f"all(record.obligations.status == '{st}') and days_since(origin) <= {n}"
        term = (f"deny: requires every {noun} obligation {st} and a request "
                f"within {n} days — never override this")
        return {"predicate": pred, "satisfiers": {}, "terminal": term,
                "applies_to": applies, "db_check": db}
    # AUTH
    sat = rng.choice(reads)
    return {"predicate": "authenticated user identity",
            "satisfiers": {sat: catalog[sat]["required"]},
            "ask": "identify the user first, then retry",
            "applies_to": applies}


def render(rng, domain, catalog, spec):
    """프로그램 산문 렌더 — 모든 gated 도구명·satisfier·인자·구체 조건이 본문에 등장."""
    L = [f"# {domain.split('-')[0].title()} Service Policy", ""]
    L.append("You handle customer requests using the tools below. "
             "Deny anything that violates the rules in this document.")
    L.append("")
    for g, v in spec.items():
        title = g.split("_", 1)[1].replace("_", " ").title()
        L.append(f"## {title}")
        tools = ", ".join(v["applies_to"])
        if v.get("satisfiers"):
            sat, args = next(iter(v["satisfiers"].items()))
            arg_s = ", ".join(a.replace("_", " ") for a in args)
            L.append(f"Before running {tools}, you must establish {v['predicate']}: "
                     f"call {sat} with the customer's {arg_s}. If this has not been "
                     f"done, collect the {arg_s}, verify, then retry the request.")
        elif "terminal" in v:
            L.append(f"The operations {tools} are permitted only when {v['predicate']}. "
                     f"The backend does not verify this — if the condition fails, "
                     f"refuse the request.")
        else:
            L.append(f"Before {tools}: {v['predicate']} is required — "
                     f"{v.get('ask', 'obtain it, then retry')}.")
        L.append("")
    return "\n".join(L).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    kinds = ["AUTH", "ELIGIBILITY", "LIMIT", "STATE", "COMPOUND"]
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            cat = sample_catalog(rng, rng.randint(2, 5), rng.randint(2, 4))
            writes = [t for t, c in cat.items() if c["type"] == "WRITE"]
            reads = [t for t, c in cat.items() if c["type"] == "READ"]
            n_g = rng.randint(3, 5)
            ks = ["AUTH"] + rng.sample(kinds[1:], n_g - 1)
            spec = {}
            for j, k in enumerate(ks, 1):
                applies = sorted(rng.sample(writes, max(1, len(writes) - rng.randint(0, 1))))
                spec[f"G{j}_{k}"] = concrete_gate(rng, k, j, applies, reads, cat)
            dom = f"{rng.choice(['retail','telecom','banking','insurance','logistics'])}-c{i}"
            nl = render(rng, dom, cat, spec)
            f.write(json.dumps({"id": 1000 + i, "style": "concrete-program",
                                "qc": "program-render", "domain_hint": dom,
                                "catalog": cat, "spec": spec, "policy_nl": nl},
                               ensure_ascii=False) + "\n")
    print(f"[concrete-gen] {a.n} pairs (seed={a.seed}) -> {a.out}")


if __name__ == "__main__":
    main()
