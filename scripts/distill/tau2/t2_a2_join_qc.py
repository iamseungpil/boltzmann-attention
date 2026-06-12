#!/usr/bin/env python
"""P-A2-1 batch 조립 + 기계적 충실성 QC.

renders(part*.jsonl, id+style+policy_nl) × specs(jsonl) 조인 → 학습쌍 파일.
QC = 모든 gated 도구명·satisfier 도구·satisfier 인자가 정책 본문에 등장하는지
(렌더 전사 누락 검출 — applies_to 운반 실패는 학습 GT 오염이므로 fail은 DROP).

Usage: t2_a2_join_qc.py --specs specs_synth_b1.jsonl --renders part1 part2 part3 --out out.jsonl
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--renders", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = {}
    for l in open(a.specs):
        d = json.loads(l)
        specs[d["id"]] = d

    rows = []
    for p in a.renders:
        for l in open(p):
            rows.append(json.loads(l))

    kept, dropped = 0, 0
    with open(a.out, "w") as out:
        for r in rows:
            s = specs[r["id"]]
            nl = r["policy_nl"]
            nl_sp = nl.replace("_", " ")
            missing = []
            for g, gv in s["spec"].items():
                for t in gv.get("applies_to", []):
                    if t not in nl:
                        missing.append((g, t))
                for st, args in (gv.get("satisfiers") or {}).items():
                    if st not in nl:
                        missing.append((g, "SAT:" + st))
                    for arg in args:
                        if arg.replace("_", " ") not in nl_sp:
                            missing.append((g, "ARG:" + arg))
            if missing:
                dropped += 1
                print("QC-MISS", r["id"], r["style"], missing[:8])
                continue
            kept += 1
            out.write(json.dumps({"id": r["id"], "style": r["style"],
                                  "domain_hint": s["domain_hint"], "catalog": s["catalog"],
                                  "spec": s["spec"], "policy_nl": nl}) + "\n")
    print(f"[join-qc] kept={kept} dropped={dropped} -> {a.out}")


if __name__ == "__main__":
    main()
