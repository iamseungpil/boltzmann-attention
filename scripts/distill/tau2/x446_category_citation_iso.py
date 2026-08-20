# -*- coding: utf-8 -*-
r"""x446 — **인용을 요구하면 범주 주장이 달라지나** (⒠안 2부 · 2026-08-20 밤)

## 1부(x445)가 말한 것
`spend_category` 를 낸 sim **131 중 124(95%)** 는 그 호출 **이전에** 그 범주 키워드가 든 도구 출력을
**한 번도 받은 적이 없다** ⇒ 지금 그 주장들은 **문서 근거 없이** 나온다. ⒠(인용 요구)를 그대로 얹으면
95% 가 기본 요율로 강등되어 사실상 ⒟가 된다.

## 2부(여기)가 묻는 것
**문서를 앞에 놓고 인용을 요구하면** 모델의 범주 주장이 달라지나 — 특히 024(트럭)에서
`operations` 주장을 **거두나**. KB 는 답을 갖고 있다(C568):
    `doc_..._business_gold_rewards_card_003` = *"What Qualifies as Operations Spend?"*
    자격 목록에 차량 구매가 없고, 자매 문서는 *"equipment billed as physical goods"* 를 제외로 적는다.

## 팔
    A_now    현행 — 손님 발화만 주고 `spend_category` 를 받는다
    B_cite   같은 발화 + **그 계열 카드 문서 전부** + *"자격을 보이는 축자 인용을 같이 내라.
             못 내면 `spend_category` 를 비워라"*
검산: 인용이 문서에 **실재하는가**(정규화 부분문자열)뿐 — 뜻은 안 본다([[59]]ⓐ).

사용: py -3 x446_category_citation_iso.py [--port 8141]   (⚠문서 경로 = 리모트)
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
FAMS = {"business": "business_credit_cards", "personal": "credit_cards"}


def docs_for(family):
    """그 계열의 카드 문서 전부 — 우리가 고르지 않는다(파일명 접두사만)."""
    out = []
    import glob
    for p in sorted(glob.glob(os.path.join(FT.DOCDIR, "doc_%s_*.json" % family))):
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        out.append((d.get("id") or os.path.basename(p),
                    str(d.get("title") or ""), str(d.get("content") or "")))
    return out


def norm(t):
    return " ".join(str(t or "").split()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="cite1")
    a = ap.parse_args()

    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] != "card_type":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    sys_a = ("Reply with ONE JSON object only: {\"spend_category\": \"<one of: travel, software, "
             "operations, media_advertising, green>\" or null}. Use null unless the customer's spending "
             "clearly belongs to one of those documented categories.")
    sys_b = (sys_a[:-1] + ", \"quote\": \"<a verbatim sentence from the documents below that shows this "
             "spending qualifies for that category>\"}. If you cannot quote such a sentence, set "
             "spend_category to null.")

    print("=" * 100)
    print("x446 · 인용 요구가 범주 주장을 바꾸나 · 사례 %d" % len(cs))
    print("=" * 100)
    rows = []
    for c in cs:
        said = " \n".join(P.turns(c))
        fam = "business_credit_cards" if "business" in said.lower() else "credit_cards"
        ds = docs_for(fam)
        blob = "\n\n".join("### %s — %s\n%s" % (i, t, b) for i, t, b in ds)
        a_ans = X.ask(a.port, sys_a, "# What the customer said\n%s\n" % said[:5000], maxtok=200) or {}
        b_ans = X.ask(a.port, sys_b, "# Documents\n%s\n\n# What the customer said\n%s\n"
                      % (blob[:60000], said[:5000]), maxtok=400) or {}
        q = norm(b_ans.get("quote"))
        real = bool(q) and any(q in norm(b) for _i, _t, b in ds)
        rows.append({"task": c["task"], "trial": c["trial"], "family": fam,
                     "A_now": a_ans.get("spend_category"),
                     "B_cite": b_ans.get("spend_category"),
                     "quote": str(b_ans.get("quote") or "")[:200], "quote_real": real,
                     "n_docs": len(ds)})
        print("  %-9s t%s (%s·문서 %d)  A=%-18s B=%-18s 인용실재=%s"
              % (c["task"], c["trial"], fam.split("_")[0], len(ds),
                 a_ans.get("spend_category"), b_ans.get("spend_category"), real))
        if b_ans.get("quote"):
            print("        인용: %s" % str(b_ans.get("quote"))[:150])
    p = os.path.abspath(os.path.join(REP, "x446_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
