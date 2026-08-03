"""How often would the [matches] line actually say something?

rev4.2 fixed the counter to phrase substring matching, because that is the operation
that produced the 195-vs-4 contrast. But the contrast came from one- and two-word
queries, and the live queries are mostly longer. If a six-word query never appears
verbatim in prose, the counter reports 0 for a search that returned eight documents —
"always zero" is as uninformative as "always true".

This measures, over every KB_search query the run actually issued:
  phrase  normalized query as a substring of the normalized document
  and_cw  every content word of the query present in the document (no threshold)

Both are deterministic and have the full corpus as denominator. The question is which
one is non-degenerate on the real query distribution.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

STOP = set(
    "a an the of for to in on at by with and or is are was were be been being do does did "
    "how what when where which who whom this that these those i you we they it my your our "
    "can could should would will shall may might must if then than there here about from "
    "into over under after before during any all each".split()
)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s or "").lower())).strip()


def load_docs():
    out = []
    for p in sorted(glob.glob(os.path.join(DOCS, "*"))):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                out.append(norm(fh.read()))
        except Exception:
            pass
    return out


def queries():
    qs = []
    for p in sorted(glob.glob(f"{SIM}/bank_ax33n_gpu*_20260803g.results.json.gz")):
        for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
            for m in s.get("messages") or []:
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    n = tc.get("name") or (tc.get("function") or {}).get("name")
                    if not n or not n.startswith("KB_search"):
                        continue
                    a = tc.get("arguments") or (tc.get("function") or {}).get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except Exception:
                            a = {}
                    q = (a or {}).get("query")
                    if q:
                        qs.append((n, q))
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    docs = load_docs()
    qs = queries()
    print(f"documents={len(docs)}  queries={len(qs)}\n")

    lens = collections.Counter(len(norm(q).split()) for _, q in qs)
    tot = sum(lens.values())
    print("=== 질의 단어 수 분포 ===")
    for band, sel in (("1-2", lambda k: k <= 2), ("3", lambda k: k == 3),
                      (">=4", lambda k: k >= 4)):
        n = sum(v for k, v in lens.items() if sel(k))
        print(f"  {band:>4} 단어: {n:4d}  ({n / tot:.0%})")

    rows = []
    for name, q in qs:
        nq = norm(q)
        cw = [w for w in nq.split() if w not in STOP]
        phrase = sum(1 for d in docs if nq and nq in d)
        and_cw = sum(1 for d in docs if cw and all(w in d for w in cw))
        rows.append((name, q, len(nq.split()), phrase, and_cw))

    def band(rs, key):
        c = collections.Counter()
        for r in rs:
            v = r[key]
            c["0" if v == 0 else "1-10" if v <= 10 else "11-50" if v <= 50 else ">50"] += 1
        return c

    print("\n=== 매칭 수 분포 (분모 = 전 코퍼스) ===")
    for label, key in (("phrase (현행 ④ 명세)", 3), ("and_cw (ⓐ 내용어 AND)", 4)):
        c = band(rows, key)
        n = len(rows)
        line = "  ".join(f"{k}={c[k]:3d}({c[k] / n:.0%})" for k in ("0", "1-10", "11-50", ">50"))
        print(f"  {label:26s} {line}")

    print("\n=== 표본 ===")
    for r in rows[: args.show]:
        print(f"  [{r[2]}단어] phrase={r[3]:3d} and_cw={r[4]:3d}  {r[1][:78]}")

    print("\n=== O6 보존 확인 ===")
    for q in ("transfer", "human agent"):
        nq = norm(q)
        cw = [w for w in nq.split() if w not in STOP]
        print(f"  {q!r:16s} phrase={sum(1 for d in docs if nq in d):4d} "
              f"and_cw={sum(1 for d in docs if all(w in d for w in cw)):4d}")


if __name__ == "__main__":
    main()
