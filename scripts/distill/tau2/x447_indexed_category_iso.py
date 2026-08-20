# -*- coding: utf-8 -*-
r"""x447 — **A2 색인이 가리키는 문서만** 주면 범주를 옳게 정하나 (⒡안 격리 · 2026-08-20 밤)

## 왜 (사용자 지시)
*"우리는 선택에 필요한 문서들을 A2 A3 등에 인덱스로 명확하게 기술해야 한다."* ⇒ `catalog_arg_docs.spend_category`
선언 완료(범주별 KB 문서 id·제목 출처·gold 미참조). 이제 [[62]] 순서대로 **배선 전에 격리로** 잰다:
그 문서들만 앞에 놓았을 때 모델이 ⑴범주를 옳게 정하고 ⑵**KB 를 인용**하나.

## 배경 수치
    C570  fit 호출의 **77%** 가 KB 검색 **0회** 상태에서 난다 ⇒ 범주는 지금 **문서 없이** 정해진다
    C569  문서를 **110편 통째로** 줬을 때: 024 는 `operations` 를 **스스로 철회**했지만
          003·063 은 **손님 발화를 인용**해 실재 검산에 걸렸다(살아남은 주장 0/4)

## 팔
    A_none   문서 없음(현행 재현)
    B_index  **A2 색인이 가리키는 문서만**(전 범주 합집합·중복 제거) — 오늘 선언한 그것
검산: 인용이 **배달된 문서 안에** 실재하는가(정규화 부분문자열)뿐 — 뜻은 안 본다([[59]]ⓐ).

사용: py -3 x447_indexed_category_iso.py [--port 8141]   (⚠문서 경로 = 리모트)
"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x423_choice_isolation as I  # noqa: E402
import x431_spec_selects as X  # noqa: E402
import x437_declaration_isolation as P  # noqa: E402
import x430_account_facts as FT  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")


def index_docs():
    """A2 선언 → 문서 id 목록(전 범주 합집합). **우리가 고르지 않는다** — 선언을 읽을 뿐이다."""
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    decl = ((d.get("catalog_arg_docs") or {}).get("spend_category") or {})
    ids = []
    for k, v in decl.items():
        if k.startswith("_") or not isinstance(v, list):
            continue
        for x in v:
            if x not in ids:
                ids.append(x)
    out = []
    for did in ids:
        p = os.path.join(FT.DOCDIR, "%s.json" % did)
        if not os.path.exists(p):
            print("  ⚠선언된 문서를 못 찾음: %s" % did)
            continue
        dd = json.load(io.open(p, encoding="utf-8"))
        out.append((did, str(dd.get("title") or ""), str(dd.get("content") or "")))
    return out


def norm(t):
    return " ".join(str(t or "").split()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="idx1")
    a = ap.parse_args()
    ds = index_docs()
    blob = "\n\n".join("### %s — %s\n%s" % (i, t, b) for i, t, b in ds)
    print("=" * 100)
    print("x447 · A2 색인 문서 %d편 (%d자) 로 범주 결정 격리" % (len(ds), len(blob)))
    for i, t, _b in ds:
        print("     %-58s %s" % (i[:58], t[:60]))
    print("=" * 100)

    sys_a = ("Reply with ONE JSON object only: {\"spend_category\": \"<one of: travel, software, "
             "operations, media_advertising, green>\" or null}. Use null unless the customer's spending "
             "clearly belongs to one of those documented categories.")
    sys_b = (sys_a[:-1] + ", \"quote\": \"<a sentence copied word for word from the '# Documents' "
             "section that shows this spending qualifies>\"}. The quote MUST come from the documents, "
             "not from the customer. If no document sentence supports the category, set it to null.")

    seen, cs, rows = set(), [], []
    for c in I.cases(60):
        if c["arg"] != "card_type":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)
    for c in cs:
        said = " \n".join(P.turns(c))
        a_ans = X.ask(a.port, sys_a, "# What the customer said\n%s\n" % said[:5000], maxtok=200) or {}
        b_ans = X.ask(a.port, sys_b, "# Documents\n%s\n\n# What the customer said\n%s\n"
                      % (blob, said[:5000]), maxtok=400) or {}
        q = norm(b_ans.get("quote"))
        real = bool(q) and any(q in norm(b) for _i, _t, b in ds)
        rows.append({"task": c["task"], "trial": c["trial"],
                     "A_none": a_ans.get("spend_category"), "B_index": b_ans.get("spend_category"),
                     "quote": str(b_ans.get("quote") or "")[:220], "quote_real": real})
        print("  %-9s t%s  A_none=%-16s B_index=%-16s 인용실재=%s"
              % (c["task"], c["trial"], a_ans.get("spend_category"), b_ans.get("spend_category"), real))
        if b_ans.get("quote"):
            print("        인용: %s" % str(b_ans.get("quote"))[:160])
    p = os.path.abspath(os.path.join(REP, "x447_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n  ★인용 실재 %d/%d · 범주 유지 %d · 철회(None) %d"
          % (sum(1 for r in rows if r["quote_real"]), len(rows),
             sum(1 for r in rows if r["B_index"]), sum(1 for r in rows if not r["B_index"])))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
