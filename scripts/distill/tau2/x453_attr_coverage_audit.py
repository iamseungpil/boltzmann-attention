# -*- coding: utf-8 -*-
r"""x453 — **선언된 속성 목록의 완결성 감사** (2026-08-21·무료·G1)

## 왜 (사용자 지적으로 방향 정정)
내가 *"속성 목록을 우리가 적어 둔 것이 근본 문제"* 라고 했는데 **틀렸다**. 사용자 축자:
*"속성 목록을 적어두는게 왜 문제인가? 우리 비용 측면에서 미리 DAG 나 속성을 명시하는게
결정론적으로 이득이면 하면 된다."* — 맞다. [[05]] 는 A2 를 가변부로 두고 도메인-특화 **내용**을
허용하며(엔진만 도메인-일반), [[23]] 이 금지하는 것은 **출처가 gold 인 것**이다. 선언은 1회 비용·
결정론·감사 가능이고 매 런 발견보다 싸다.

진짜 결함 둘:
  ⑴ 16종 목록이 코퍼스를 얼마나 덮는지 **한 번도 재지 않았다** — 빠진 칸을 *"문서 미기재"* 로 읽었다
     (`x452` 직전 전수: 채움 346 · absent 550).
  ⑵ **추가 절차가 없다** — 지금 내가 필요한 속성을 골라 넣으면 그건 gold 를 보고 고른 것이다([[23]]).

## 미리 못 박은 선택 규칙 (결과를 보기 전에 고정)
    · 문서에게 *"값을 명시하는 속성을 전부 대라"* 고 묻는다 — LLM formalize·엔진은 담기만([[59]])
    · 검산 = 닫힌 술어 둘: 인용이 문서에 실재 · 값이 인용 안에 실재 (정본 `t2_search.quote_in`)
    · **서로 다른 클래스 `--minclasses`(기본 5) 이상**에서 명시된 속성만 후보로 채택 — 빈도뿐
    · 속성마다 문서 id + 축자 인용을 남긴다. 못 대면 넣지 않는다
    ⛔gold·태스크·실패 사례는 보지 않는다. 무엇이 어느 태스크를 고치는지는 **채택 후에** 본다

계열 목록은 A2 선언(`catalog_arg_families`)에서 읽는다([[71]] 계약 2항).

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x453_attr_coverage_audit.py --port 8140
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

import x430_account_facts as FT         # noqa: E402  DOCDIR·ATTRS(현행 선언)
import x431_spec_selects as X           # noqa: E402  ask 정본
import x452_conditional_facts as C      # noqa: E402  선언 읽기·검산(사본 금지·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))

# ★선택 기준은 **빈도가 아니라 정책이 요구하는 것**이다(사용자 정정 축자: *"gold 보고 고른다가
#   아니라. 정책에서 필요한 것을 고른다 이다."*). 그래서 문서에 두 가지를 함께 묻는다 —
#   ⑴어떤 속성에 값을 명시하나 ⑵그중 **계좌를 열거나 유지하는 요건·자격**으로 말하는 것은 무엇인가.
#   판단은 LLM 이 하고 엔진은 인용 실재만 본다([[59]]). 빈도는 **보고용**이지 채택 기준이 아니다.
SYS = ("Answer ONLY from the document. Reply with ONE JSON object with two lists:\n"
       "{\"attributes\": [{\"name\": \"<short snake_case name>\", \"value\": \"<verbatim>\", "
       "\"quote\": \"<verbatim sentence or table row containing it>\"}],\n"
       " \"requirements\": [{\"name\": \"<short snake_case name of the attribute it constrains>\", "
       "\"requirement\": \"<verbatim>\", \"quote\": \"<verbatim sentence stating it>\"}]}\n"
       "`attributes` = every attribute of the account this document states a concrete value for. "
       "`requirements` = only those the document states as a condition for opening the account, "
       "keeping it, or qualifying for its benefits. Use the document's own wording for names. "
       "Never paraphrase: each value and requirement must appear inside its own quote. "
       "Use empty lists if the document states none.")


def slug(name):
    """속성 이름 정규화 — **형태만**(소문자·비영숫자→`_`). 뜻은 안 본다([[59]])."""
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--minclasses", type=int, default=5)
    ap.add_argument("--maxdocs", type=int, default=0, help="0=전부 (연습용 제한)")
    ap.add_argument("--out", default="x453_attr_coverage.json")
    a = ap.parse_args()

    fams = C.declared_families()
    byc = C.docs_by_class(fams)
    have = {n for n, _al in FT.ATTRS}
    print("=" * 96)
    print("x453 · 선언 계열 %d · 클래스 %d · 현행 선언 속성 %d종 · 채택선 = 클래스 %d 이상"
          % (len(fams), len(byc), len(have), a.minclasses))
    print("=" * 96)

    seen_classes = collections.defaultdict(set)     # attr -> {class}  (값을 명시)
    req_classes = collections.defaultdict(set)      # attr -> {class}  ★정책이 **요건**이라 말함
    example = {}                                    # attr -> (class, value, quote)
    req_example = {}                                # attr -> (class, requirement, quote)
    rejected = ndocs = 0
    for cls in sorted(byc):
        docs = byc[cls]
        if a.maxdocs:
            docs = docs[:a.maxdocs]
        for did, text in docs:
            ndocs += 1
            body = " ".join(text.split())[:12000]
            got = X.ask(a.port, SYS, "# Document %s\n%s\n" % (did, body), maxtok=900) or {}
            for it in (got.get("attributes") or []):
                if not isinstance(it, dict):
                    continue
                nm, v, q = slug(it.get("name")), str(it.get("value") or ""), str(it.get("quote") or "")
                if not (nm and v and q):
                    continue
                if not (C.contained(q, body) and C.contained(v, q)):
                    rejected += 1
                    continue
                seen_classes[nm].add(cls)
                example.setdefault(nm, (cls, v, " ".join(q.split())[:180]))
            for it in (got.get("requirements") or []):
                if not isinstance(it, dict):
                    continue
                nm = slug(it.get("name"))
                rq = str(it.get("requirement") or "")
                q = str(it.get("quote") or "")
                if not (nm and rq and q):
                    continue
                if not (C.contained(q, body) and C.contained(rq, q)):
                    rejected += 1
                    continue
                req_classes[nm].add(cls)
                req_example.setdefault(nm, (cls, rq, " ".join(q.split())[:180]))
        print("  %-30s 누적 속성 %d종" % (cls[:30], len(seen_classes)))

    # ★채택 = **정책이 요건이라고 말한 속성**(빈도 무관) ∪ 값이 널리 명시된 속성(보조).
    #   기준을 결과 보기 **전에** 못 박았다 — gold·태스크·실패 사례는 보지 않는다([[23]]).
    #   사용자 축자: *"gold 를 정책이나 KB 에서 골라낼 수 있으면, 한번의 설정으로 비용을 줄일 수 있다."*
    ranked = sorted(seen_classes.items(), key=lambda kv: -len(kv[1]))
    req_ranked = sorted(req_classes.items(), key=lambda kv: -len(kv[1]))
    names = {n for n, _s in req_ranked} | {n for n, s in ranked if len(s) >= a.minclasses}
    adopt = [(n, seen_classes.get(n) or req_classes.get(n) or set()) for n in sorted(names)]
    new = [(n, s) for n, s in adopt if n not in have]
    missing_now = [n for n in sorted(have) if n not in seen_classes]

    print("\n" + "=" * 96)
    print("문서 %d편 · 검산 탈락 %d · 관측된 속성 %d종 · 채택(≥%d 클래스) %d종 · **현행에 없는 것 %d종**"
          % (ndocs, rejected, len(seen_classes), a.minclasses, len(adopt), len(new)))
    print("\n★[정책이 **요건**이라고 말한 속성] — 채택의 1차 기준")
    for n, s in req_ranked[:20]:
        c, rq, q = req_example[n]
        print("  %-32s %2d 클래스  %-14s %s" % (n[:32], len(s), str(rq)[:14], q[:58]))
    print("\n[채택되었는데 현행 선언에 없는 속성] — 이것이 목록의 결손이다")
    for n, s in new[:30]:
        c, v, q = example[n]
        print("  %-34s %2d 클래스  예: %s=%s  “%s”" % (n[:34], len(s), c[:14], str(v)[:14], q[:60]))
    print("\n[현행 선언에 있는데 한 번도 관측 안 된 속성]")
    print("  " + (", ".join(missing_now) if missing_now else "(없음)"))

    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"minclasses": a.minclasses, "n_docs": ndocs, "rejected": rejected,
                   "observed": {n: sorted(s) for n, s in seen_classes.items()},
                   "example": {n: {"class": c, "value": v, "quote": q}
                               for n, (c, v, q) in example.items()},
                   "requirements": {n: sorted(s) for n, s in req_classes.items()},
                   "req_example": {n: {"class": c, "requirement": r, "quote": q}
                                   for n, (c, r, q) in req_example.items()},
                   "adopt": [n for n, _s in adopt], "new_vs_declared": [n for n, _s in new],
                   "declared_never_seen": missing_now}, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
