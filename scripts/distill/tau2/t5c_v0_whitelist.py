#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T5-C V0 — disamb_sub_args 화이트리스트 정량 검증 (무료·기존 c51 데이터·GPU 불요).

c51_results.jsonl(400행): task,trial,idx,arg,gold,ncand,A,A_ok,B,B_ok
  A = 자유생성(orig 근사)·B = 후보 열거(서브콜 P-B 근사)·*_ok = gold 일치.
arg-type(key-token)별로:
  - A_ok(orig 근사 정답률) vs B_ok(서브콜 근사 정답률)
  - fix = A_ok0∧B_ok1 (서브콜이 고침) · break = A_ok1∧B_ok0 (서브콜이 정답을 깸=spurious 후보)
  - net = fix − break · GO: B_ok>A_ok ∧ fix>break ∧ n≥30 (§7 B5·보수)
편향 명기(B5-d): c51은 gold∈C 조건수집 → live(gold∉C 3.7%) 대비 과대추정 ⇒ 등재는 보수적으로.

usage: t5c_v0_whitelist.py --data c51_results.jsonl
"""
import argparse, json, math
from collections import defaultdict


def key_tokens(arg):
    toks = set()
    for t in str(arg).lower().split("_"):
        if t in ("id", "ids", "no", "num", "number", "code"):
            continue
        toks.add(t[:-1] if t.endswith("s") and len(t) > 3 else t)
    return toks or {str(arg).lower()}


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (c - m) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/woori/scratch/c51_results.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]

    by = defaultdict(list)          # token -> rows
    by_arg = defaultdict(list)      # exact arg -> rows
    for r in rows:
        toks = key_tokens(r["arg"])
        for t in toks:
            by[t].append(r)
        by_arg[r["arg"]].append(r)

    def _int(x):
        try:
            return int(x)
        except Exception:
            return 1 if x in (True, "True", "true") else 0

    print("=== per exact-arg ===")
    print("%-16s %4s %6s %6s %5s %6s %6s" % ("arg", "n", "A_ok", "B_ok", "fix", "break", "net"))
    for arg, rs in sorted(by_arg.items()):
        n = len(rs)
        a_ok = sum(_int(r["A_ok"]) for r in rs)
        b_ok = sum(_int(r["B_ok"]) for r in rs)
        fix = sum(1 for r in rs if not _int(r["A_ok"]) and _int(r["B_ok"]))
        brk = sum(1 for r in rs if _int(r["A_ok"]) and not _int(r["B_ok"]))
        print("%-16s %4d %6.3f %6.3f %5d %6d %+6d" % (arg, n, a_ok/n, b_ok/n, fix, brk, fix-brk))

    print("\n=== per key-token (화이트리스트 후보) ===")
    print("%-10s %4s %6s %6s %5s %6s %6s  %-7s %s" %
          ("token", "n", "A_ok", "B_ok", "fix", "break", "net", "netCIlo", "GO?"))
    go = []
    for tok, rs in sorted(by.items()):
        n = len(rs)
        a_ok = sum(_int(r["A_ok"]) for r in rs)
        b_ok = sum(_int(r["B_ok"]) for r in rs)
        fix = sum(1 for r in rs if not _int(r["A_ok"]) and _int(r["B_ok"]))
        brk = sum(1 for r in rs if _int(r["A_ok"]) and not _int(r["B_ok"]))
        # net>0의 신뢰하한 (fix를 성공, fix+break를 시행으로 본 Wilson·break-우세면 <0.5)
        trials = fix + brk
        ci_lo = wilson_lo(fix, trials) if trials else 0.0
        # GO: B_ok>A_ok ∧ fix>break ∧ n>=30 ∧ CIlo(fix/(fix+break))>0.5 (보수)
        ok = (b_ok > a_ok) and (fix > brk) and (n >= 30) and (ci_lo > 0.5)
        print("%-10s %4d %6.3f %6.3f %5d %6d %+6d  %6.3f  %s" %
              (tok, n, a_ok/n, b_ok/n, fix, brk, fix-brk, ci_lo, "GO" if ok else "-"))
        if ok:
            go.append(tok)
    print("\n★ GO 화이트리스트(disamb_sub_args 등재 권고):", go)
    print("  (보수 임계: fix/(fix+break) Wilson-하한 >0.5 ∧ n≥30 ∧ B>A. gold∈C 편향 → live 과대추정 감안)")


if __name__ == "__main__":
    main()
