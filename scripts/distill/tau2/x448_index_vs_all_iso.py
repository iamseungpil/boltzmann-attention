# -*- coding: utf-8 -*-
r"""x448 — **전체 전달 ↔ A2 색인 전달**을 한 프롬프트에서 가른다 (2026-08-21·격리·무료)

## 왜 (사용자 지시 축자)
*"그냥 전체 문서 내용 검색해서 주는거랑, 인덱스로 필요한 문서만 주는 걸 비교해서 계산해야 하는거
아닌가?"* / *"A2 A3 에서 하는 것은 어떤 서브에이전트에서 어떤 결정을 위해 필요한 문서가 뭔지
정확하게 기술하는 것만이다."* / *"지금까지 하던 방식으로 하면 안되나?"*

## 무엇이 문제였나 (이 프로브가 고치는 것)
C569(110편 전달·`x446`)와 C571(색인 12편 전달·`x447`)은 **서로 다른 스크립트·다른 프롬프트**로 났다.
0/4 ↔ 4/4 의 대비가 **색인 때문인지 프롬프트 때문인지 가를 수 없다**. 여기서는 네 팔이
**같은 프롬프트·같은 계약·같은 정규화·같은 손님 발화**를 쓰고 **재료만 다르다**.

## 팔 (전달 관용구 — 기존 격리 방식 그대로. 도구 루프 없음)
    Z_none    문서 없음                                   현행 재현(C570: 결정 시점에 KB 가 없다)
    W_all     카드 계열 문서 **전부**                      x446 관용구 — *"그냥 전체 주는 것"*
    B_index   **A2 색인이 가리킨 문서만**                   x447 관용구 — 사용자가 요구한 것
    N_sham    같은 편수·같은 계열의 **다른** 문서            부정통제([[57]]) — *"양"* 인가 *"무엇"* 인가
  ⚠엔진은 문서를 **읽어 전달만** 한다. 해석·선택·순위 0([[59]]). 정책 문서 읽기가 우리 층 몫인 것은
    확정된 경계다(`t2_search.py` §경계 · C405ⓔ 사용자 결정).
  ⚠A2 색인의 출처는 **문서 제목**뿐이고 gold 를 보지 않았다([[23]]). 오라클 팔(`x417.doc_slice` 계열)은
    여기 없다 — 있으면 상한 표시를 반드시 붙일 것.

## 채점 (닫힌 술어만·[[59]]ⓐ)
    correct              참조 라벨과 일치 (진단용·[[69]]·gold 카드에서 유도)
    quote_real           인용이 **그 팔이 받은 재료** 안에 실재하는가(형태 정규화 부분문자열)
    quote_from_customer  인용이 **손님 발화**에서 왔는가 — C569 가 걸린 실패 양식
  참조: 003→travel(gold `Silver Rewards Card`=travel 4%) · 024→None(gold `Business Bronze`) ·
        063→None(gold `Silver Rewards Card`). ⛔pass 가 아니다.

사용: (리모트) py x448_index_vs_all_iso.py --port 8141 [--arms Z_none,W_all,B_index,N_sham]
"""
import argparse
import glob
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

import x423_choice_isolation as I       # noqa: E402  케이스 원천(카드축)
import x431_spec_selects as X           # noqa: E402  ask · cite_norm 정본
import x437_declaration_isolation as P  # noqa: E402  손님 턴 추출
import x447_indexed_category_iso as IX  # noqa: E402  A2 선언 읽기(사본 금지·[[67]])
import x446_category_citation_iso as W  # noqa: E402  계열 전체 읽기(사본 금지·[[67]])
import x430_account_facts as FT         # noqa: E402  DOCDIR

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
FAMS = ("credit_cards", "business_credit_cards")

# 참조 라벨 = gold 카드에서 유도(진단용). A2 저작에는 쓰이지 않았다([[23]]).
REF = {"003": "travel", "024": None, "063": None}

SYS = ("You are the module that decides ONE thing for a Rho-Bank support agent: which documented "
       "spending category the customer's spending belongs to. Reply with ONE JSON object only: "
       "{\"spend_category\": \"<one of: travel, software, operations, media_advertising, green>\" "
       "or null, \"quote\": \"<one sentence copied word for word from the '# Documents' section "
       "that shows this spending qualifies>\"}. The quote MUST come from the documents, never from "
       "the customer. If no document sentence supports a category, set spend_category to null.")


def all_docs():
    """카드 계열 문서 **전부**. 우리가 고르지 않는다 — 파일명 접두사만(x446 관용구)."""
    out = []
    for fam in FAMS:
        out.extend(W.docs_for(fam))
    return out


def sham_docs(declared_ids, n):
    """부정통제 — 선언 **밖**에서 같은 계열·같은 편수. 규칙 하나(이름 순)."""
    out = []
    for p in sorted(glob.glob(os.path.join(FT.DOCDIR, "doc_*credit_card*.json"))):
        did = os.path.basename(p)[:-5]
        if did in declared_ids:
            continue
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        out.append((d.get("id") or did, str(d.get("title") or ""), str(d.get("content") or "")))
        if len(out) >= n:
            break
    return out


def blob(ds):
    return "\n\n".join("### %s — %s\n%s" % (i, t, b) for i, t, b in ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="iva1")
    ap.add_argument("--arms", default="Z_none,W_all,B_index,N_sham")
    ap.add_argument("--maxchars", type=int, default=90000,
                    help="팔 본문 상한. 넘으면 **잘렸다고 인쇄한다**(조용한 절단 금지)")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    declared = IX.index_docs()
    dec_ids = {d[0] for d in declared}
    everything = all_docs()
    sham = sham_docs(dec_ids, len(declared))
    mats = {"Z_none": [], "W_all": everything, "B_index": declared, "N_sham": sham}

    print("=" * 100)
    print("x448 · 한 프롬프트·재료만 다름 · 팔 %s" % ",".join(arms))
    for k in arms:
        ds = mats[k]
        print("   %-8s %3d편  %7d자%s" % (k, len(ds), len(blob(ds)),
                                          "  ⚠절단됨" if len(blob(ds)) > a.maxchars else ""))
    print("=" * 100)

    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] != "card_type":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    rows = []
    for c in cs:
        said = " \n".join(P.turns(c))[:5000]
        ref = REF.get(str(c["task"]).split("_")[-1], "?")
        print("\n%s t%s   참조=%s" % (c["task"], c["trial"], ref))
        for arm in arms:
            ds = mats[arm]
            body = blob(ds)
            cut = len(body) > a.maxchars
            body = body[:a.maxchars]
            head = ("# Documents\n%s\n\n" % body) if ds else ""
            ans = X.ask(a.port, SYS, head + "# What the customer said\n%s\n" % said, maxtok=400) or {}
            cat = ans.get("spend_category")
            cat = str(cat).strip().lower() if cat else None
            q = X.cite_norm(ans.get("quote"))
            real = bool(q) and q in X.cite_norm(body)
            from_cust = bool(q) and q in X.cite_norm(said)
            rows.append({"task": c["task"], "trial": c["trial"], "arm": arm, "ref": ref,
                         "cat": cat, "quote": str(ans.get("quote") or "")[:240],
                         "quote_real": real, "quote_from_customer": from_cust,
                         "n_docs": len(ds), "chars": len(body), "truncated": cut,
                         "correct": (cat == ref) if ref != "?" else None})
            print("   %-8s cat=%-16s 참조일치=%-5s 인용실재=%-5s 손님인용=%-5s%s"
                  % (arm, cat, (cat == ref) if ref != "?" else "-", real, from_cust,
                     "  ⚠절단" if cut else ""))
            if ans.get("quote"):
                print("        인용: %s" % str(ans.get("quote"))[:150])

    p = os.path.abspath(os.path.join(REP, "x448_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n" + "=" * 100)
    print("%-8s %-9s %-9s %-9s %-8s" % ("팔", "참조일치", "인용실재", "손님인용", "범주유지"))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        print("%-8s %-9s %-9s %-9s %-8s"
              % (arm,
                 "%d/%d" % (sum(1 for r in rs if r["correct"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["quote_real"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["quote_from_customer"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["cat"]), len(rs))))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
