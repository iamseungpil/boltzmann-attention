# -*- coding: utf-8 -*-
r"""x435 — **정책·KB 전 문서에서 표를 채운다** (사용자 지시 2026-08-20)

## 왜
지금까지의 빈칸 채우기는 **그 클래스의 자기 문서**(`doc_<family>_<class>_*`)만 봤다. 그런데 KB 에는
`bank_accounts_(general)` 47편 같은 **횡단 문서**가 따로 있고, 값이 거기 있을 수 있다.
그래서 후보 문서를 **KB 698편 전체**로 넓힌다.

## 문서 후보를 고르는 규칙 (판단 0)
**A3 문서 색인**(`t2_search.docs_for`)을 쓴다 — C410 이 이미 지어 둔 정본이다([[67]] 사본 금지).
초판은 내가 이름-포함 스캔을 새로 짰는데 그것이 바로 사본이었다(사용자 지적 2026-08-20).
색인은 `군 → 주어 → 문서 id` 이고 군마다 `_general_`(주어 없는 공통 문서)이 따로 있다.
후보 = 그 주어의 문서 + 그 군의 `_general_` + **`bank_accounts_bank_accounts` 군 공통 47편**.
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

import t2_search as TS  # noqa: E402
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


GROUP_OF = {"checking_accounts": "checking_accounts", "savings_accounts": "savings_accounts",
            "business_checking_accounts": "business_checking_accounts",
            "business_savings_accounts": "business_savings_accounts",
            "credit_cards": "credit_cards", "business_credit_cards": "business_credit_cards"}
CROSS = "bank_accounts_bank_accounts"


def index_candidates(a2, cls, family):
    """A3 색인이 주는 문서 id — 그 주어 + 군 공통 + 은행계좌 횡단 공통."""
    g = GROUP_OF.get(family)
    out = []
    if g:
        out += TS.docs_for(a2, g, subjects=[str(cls).split("@")[0]], general=True)
    out += TS.docs_for(a2, CROSS, subjects=None, general=True)
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


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
    # ★상한 기본값 = 0(무제한). 14 로 두었던 것은 **원칙이 아니라 내가 건 비용 경계**였고,
    #   문서당 1회 질의로 바꾸면서 비용이 `문서 × 속성` → `문서` 로 줄어 필요가 없어졌다.
    #   조용한 절단은 [[08]] 이 금지한다 — 실제로 물어본 문서 수를 클래스마다 찍는다.
    ap.add_argument("--max-docs", type=int, default=0, help="클래스당 문서 상한(0=무제한)")
    ap.add_argument("--out", default=None)
    # ★슬라이스 병렬(2026-08-20): 상한을 없애자 클래스당 문서가 66편이 되어 단일 프로세스로 6.5시간이다.
    #   클래스마다 독립이므로 **프로세스 3개**로 갈라 각자 자기 몫만 채우고 뒤에 합친다.
    #   (스레드로 루프를 다시 짜다 두 번 파일을 깨뜨렸다 — 구조를 안 건드리는 쪽을 택한다.)
    ap.add_argument("--slice", default="", help="i/n 형식. 예: 0/3")
    a = ap.parse_args()

    with io.open(os.path.abspath(S.TBL), encoding="utf-8") as f:
        table = json.load(f)
    corpus = load_corpus(a.docdir)
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        a2 = json.load(f)
    print("=" * 100)
    print("x435 · 코퍼스 전수 채우기 · 클래스 %d · 문서 %d" % (len(table), len(corpus)))
    print("=" * 100)

    filled = absent = skipped = 0
    tal = collections.Counter()
    _si, _sn = (a.slice.split("/") + ["1"])[:2] if a.slice else ("0", "1")
    _si, _sn = int(_si), max(1, int(_sn))
    for _k, (cls, row) in enumerate(list(table.items())):
        if not isinstance(row, dict) or (_k % _sn) != _si:
            continue
        key = str(cls).split("@")[0]
        bytext = dict(corpus)
        cands = [(i, bytext[i]) for i in index_candidates(a2, cls, row.get("_family")) if i in bytext]
        own = [x for x in cands if key in x[0]]
        extra = [x for x in cands if key not in x[0]]
        cands = (own + extra)[:a.max_docs] if a.max_docs else (own + extra)
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
        print("  %-30s 물어본문서 %3d(자기 %2d·횡단 %2d) · 빈칸 %2d → 채움 %2d"
              % (cls[:30], len(cands), len(own), len(extra), len(blanks),
                 sum(1 for at in blanks if (row.get(at) or {}).get("values"))))
    print("\n채움 %d · 미기재 확정 %d · %s" % (filled, absent, dict(tal)))
    suf = "_corpuswide%s.json" % (("_s%d" % _si) if a.slice else "")
    p = a.out or os.path.abspath(S.TBL).replace(".json", suf)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
