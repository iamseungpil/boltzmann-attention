# -*- coding: utf-8 -*-
r"""x435 — **정책·KB 전 문서에서 표를 채운다** (사용자 지시 2026-08-20)

## 왜
지금까지의 빈칸 채우기는 **그 클래스의 자기 문서**(`doc_<family>_<class>_*`)만 봤다. 그런데 KB 에는
`bank_accounts_(general)` 47편 같은 **횡단 문서**가 따로 있고, 값이 거기 있을 수 있다.
그래서 후보 문서를 **KB 698편 전체**로 넓힌다.

## 문서 후보를 고르는 규칙 (판단 0)
그 클래스의 **이름 문자열이 등장하는 문서** 전부 + 파일명에 그 클래스 키가 든 문서 전부.
★고유명사 포함 검사이지 의미 판단이 아니다([[59]] 는 엔진이 **뜻을 뜯는 것**을 금지한다).
★속성어로 문서를 추리지 않는다 — 그건 어휘가 어긋나면 값을 통째로 놓친다(실측: `foreign_transaction_fee`
  ↔ 문서는 *"ATM withdrawals in a foreign currency"*).

## 절차 (x434 가 고른 최적점 = 문서 단위)
빈칸마다 후보 문서를 **하나씩** 물어 첫 값에서 멈춘다. 엔진은 **인용이 그 문서에 축자로 실재하고
값이 인용 안에 있는지**만 본다. 전 후보가 없다고 하면 그때 `absent` 로 확정한다(문서 목록을 함께 적는다).

사용: py -3 x435_fill_corpuswide.py [--port 8141] [--max-docs 14]
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x430_account_facts as FT  # noqa: E402
import x431_spec_selects as S  # noqa: E402

ATTRS = [x[0] for x in FT.ATTRS]
SYS = ("Answer ONLY from the text given. Reply with ONE JSON object: "
       "{\"value\": \"<verbatim>\", \"unit\": \"USD|percent|count|boolean|text\", "
       "\"quote\": \"<verbatim sentence>\"} if the text states it, otherwise {\"absent\": true}. "
       "Never paraphrase.")


def display_name(cls):
    """표 키 → 문서가 쓰는 이름( `green_account_(checking)` → `Green Account` )."""
    x = re.sub(r"\(.*?\)", "", str(cls).split("@")[0]).replace("_", " ").strip()
    return " ".join(w.capitalize() if w.islower() else w for w in x.split())


def load_corpus(docdir):
    out = []
    for f in sorted(os.listdir(docdir)):
        if not f.endswith(".json"):
            continue
        with io.open(os.path.join(docdir, f), encoding="utf-8") as fh:
            d = json.load(fh)
        out.append((f.replace(".json", ""), (d.get("title") or "") + ". " + (d.get("content") or "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--docdir", default=FT.DOCDIR)
    ap.add_argument("--max-docs", type=int, default=14, help="클래스당 물어볼 문서 상한(비용 경계)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with io.open(os.path.abspath(S.TBL), encoding="utf-8") as f:
        table = json.load(f)
    corpus = load_corpus(a.docdir)
    print("=" * 100)
    print("x435 · 코퍼스 전수 채우기 · 클래스 %d · 문서 %d" % (len(table), len(corpus)))
    print("=" * 100)

    filled = absent = skipped = 0
    tal = collections.Counter()
    for cls, row in table.items():
        if not isinstance(row, dict):
            continue
        nm = display_name(cls)
        key = str(cls).split("@")[0]
        cands = [(i, t) for i, t in corpus if key in i or (nm and nm.lower() in t.lower())]
        own = [x for x in cands if key in x[0]]
        extra = [x for x in cands if key not in x[0]]
        cands = (own + extra)[:a.max_docs]
        if not cands:
            continue
        blanks = [at for at in ATTRS if not (row.get(at) or {}).get("values")]
        tal["횡단 문서 있는 클래스"] += 1 if extra else 0
        for at in blanks:
            got = None
            for did, txt in cands:
                r = S.ask(a.port, SYS, "# Text\n%s\n\n# Question\nWhat is the %s?\n"
                          % (txt[:38000], at.replace("_", " ")), maxtok=300)
                if r.get("absent"):
                    continue
                val = str(r.get("value", "")).strip()
                q = " ".join(str(r.get("quote") or "").split())
                if val and q and S.cite_norm(q) in S.cite_norm(txt) and \
                        S.cite_norm(val).replace(" ", "") in S.cite_norm(q).replace(" ", ""):
                    got = {"values": [val], "conflict": False,
                           "unit": (r.get("unit") if r.get("unit") in
                                    ("USD", "percent", "count", "boolean", "text") else ""),
                           "cap": None,
                           "evidence": [{"value": val, "doc": did, "quote": q[:220]}]}
                    if key not in did:
                        tal["횡단 문서에서 회수"] += 1
                        print("  ★횡단 회수 %-24s %-28s ← %s" % (cls[:24], at, did[:56]))
                    break
            if got:
                row[at] = got
                filled += 1
            else:
                row[at] = {"values": [], "conflict": False, "evidence": [], "absent": True,
                           "searched_docs": [d for d, _t in cands]}
                absent += 1
        print("  %-30s 후보문서 %2d(자기 %2d·횡단 %2d) · 빈칸 %2d → 채움 %2d"
              % (cls[:30], len(cands), len(own), len(extra), len(blanks),
                 sum(1 for at in blanks if (row.get(at) or {}).get("values"))))
    print("\n채움 %d · 미기재 확정 %d · %s" % (filled, absent, dict(tal)))
    p = a.out or os.path.abspath(S.TBL).replace(".json", "_corpuswide.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
