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


def locate(q, lines, maxwin=8):
    """검산이 끝난 인용이 문서 **어디**인지 — (시작 줄, 창 크기). 못 찾으면 (None, None).

    사용자 지시(2026-08-21): *"문서 전체적으로 깨끗하게 읽고, 인용해야할 문서 id 및 필요하면
    위치까지 A3 에 기록하라."* 위치가 있으면 격리 서브에 **절 단위로** 넘길 수 있다 — 실측
    부피가 클래스 3개 전량 ≈ 36,870자인데 x448 은 15,503자에서 되고 90,000자에서 절단돼 실패했다.

    엔진은 **찾기만** 한다: 정본 검산(`t2_search.quote_in`)으로 창을 1줄부터 넓혀 가장 좁은
    창을 취한다. 뜻은 안 본다([[59]]).
    """
    n = len(lines)
    for k in range(1, maxwin + 1):
        for i in range(0, max(1, n - k + 1)):
            if C.contained(q, "\n".join(lines[i:i + k])):
                return i, k
    return None, None


def section_of(lines, i):
    """그 줄을 감싸는 소제목 — 마크다운 헤딩 한 줄(형태만·`^#{1,6}\s`)."""
    for j in range(min(i, len(lines) - 1), -1, -1):
        if re.match(r"\s*#{1,6}\s+\S", lines[j] or ""):
            return " ".join((lines[j] or "").split())
    return ""


def slug(name):
    """속성 이름 정규화 — **형태만**(소문자·비영숫자→`_`). 뜻은 안 본다([[59]])."""
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--minclasses", type=int, default=5)
    ap.add_argument("--maxdocs", type=int, default=0, help="0=전부 (연습용 제한)")
    ap.add_argument("--out", default="x453_attr_coverage.json")
    ap.add_argument("--only", default="", help="선언 계열 중 이번 감사에서 볼 것(범위 제한·재정의 아님)")
    a = ap.parse_args()

    # ★계열은 A2 선언에서 온다. `--only` 는 **이번 런의 범위**만 좁힌다 — 선언을 바꾸지 않는다.
    #   G1 판정에 필요한 계열만 먼저 보고, 갈리면 그때 전수로 넓힌다(안 갈리면 전수 감사는 낭비다).
    fams = C.declared_families()
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        unknown = want - set(fams)
        if unknown:
            raise SystemExit("선언에 없는 계열: %r" % sorted(unknown))
        fams = [f for f in fams if f in want]
    byc = C.docs_by_class(fams)
    have = {n for n, _al in FT.ATTRS}
    print("=" * 96)
    print("x453 · 선언 계열 %d · 클래스 %d · 현행 선언 속성 %d종 · 채택선 = 클래스 %d 이상"
          % (len(fams), len(byc), len(have), a.minclasses))
    print("=" * 96)

    seen_classes = collections.defaultdict(set)     # attr -> {class}  (값을 명시)
    req_classes = collections.defaultdict(set)      # attr -> {class}  ★정책이 **요건**이라 말함
    # ★문서 id 색인 (2026-08-21·사용자 지시 *"정확한 100% 문서 링크를 shell 로 읽어와서 격리해야
    #   한다. bm25 는 approximate 이다"*): 이 스캔은 이미 **문서 단위**로 돌면서 어느 문서가 그
    #   축의 값을 말하는지 인용까지 검산한다 — 그런데 `did` 를 안 남겨서 그 색인을 버리고 있었다.
    #   남기면 *속성 → 정확한 문서 id* 가 **같은 비용으로** 나오고, 그것이 격리 서브에 **검색 없이**
    #   넘길 재료의 선언이 된다([[71]] 계약 3항). ⚠제목 규칙으로는 구멍이 난다 — 실측: gold_account
    #   14편 중 제목에 interest/APY 가 있는 것은 3편인데, 값은 `specifications and requirements`
    #   처럼 제목이 축을 안 말하는 문서에도 있다. 인용이 검산된 문서만 담는다.
    cites = collections.defaultdict(list)           # attr -> [{doc,class,line,span,section,quote}]
    attr_docs = collections.defaultdict(set)        # attr -> {doc_id}  ★인용이 검산된 문서만
    class_docs = collections.defaultdict(set)       # class -> {doc_id} (스캔 분모)
    example = {}                                    # attr -> (class, value, quote)
    req_example = {}                                # attr -> (class, requirement, quote)
    rejected = ndocs = 0
    for cls in sorted(byc):
        docs = byc[cls]
        if a.maxdocs:
            docs = docs[:a.maxdocs]
        for did, text in docs:
            ndocs += 1
            class_docs[cls].add(did)
            body = " ".join(text.split())[:12000]
            # ★위치 기록용 **원문 줄** (검산은 종전 `body` 그대로 — 거동 보존).
            #   실측: 문서 최대 7,878자라 12,000 절단에 걸리는 문서는 **0편**이다.
            raw_lines = (text or "").split("\n")
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
                _li, _lk = locate(q, raw_lines)
                cites[nm].append({"doc": did, "class": cls, "axis": "value", "value": v,
                                  "line": _li, "span": _lk,
                                  "section": section_of(raw_lines, _li) if _li is not None else "",
                                  "quote": " ".join(q.split())[:300]})
                attr_docs[nm].add(did)
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
                _li, _lk = locate(q, raw_lines)
                cites[nm].append({"doc": did, "class": cls, "axis": "requirement", "value": rq,
                                  "line": _li, "span": _lk,
                                  "section": section_of(raw_lines, _li) if _li is not None else "",
                                  "quote": " ".join(q.split())[:300]})
                attr_docs[nm].add(did)
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

    # ★산출물을 **인쇄보다 먼저** 쓴다(2026-08-21): 1차 실행은 28 클래스를 다 돌고 나서
    #   보고 루프의 `KeyError` 로 죽어 **JSON 을 통째로 잃었다**. 보고는 부수적이고 데이터가 본체다.
    payload = {"minclasses": a.minclasses, "n_docs": ndocs, "rejected": rejected,
               "families": fams,
               "observed": {n: sorted(s) for n, s in seen_classes.items()},
               "example": {n: {"class": c, "value": v, "quote": q}
                           for n, (c, v, q) in example.items()},
               "requirements": {n: sorted(s) for n, s in req_classes.items()},
               "req_example": {n: {"class": c, "requirement": r, "quote": q}
                               for n, (c, r, q) in req_example.items()},
               "attr_docs": {n: sorted(v) for n, v in attr_docs.items()},
               "cites": {n: v for n, v in cites.items()},
               "n_cites": sum(len(v) for v in cites.values()),
               "n_cites_unlocated": sum(1 for v in cites.values() for c in v if c["line"] is None),
               "class_docs": {c: sorted(v) for c, v in class_docs.items()},
               "adopt": [n for n, _s in adopt], "new_vs_declared": [n for n, _s in new],
               "declared_never_seen": missing_now}
    _p0 = os.path.join(REP, a.out)
    with io.open(_p0, "w", encoding="utf-8") as _f0:
        json.dump(payload, _f0, ensure_ascii=False, indent=1)
    print("\n[산출물 선기록] → %s" % _p0)

    print("\n" + "=" * 96)
    print("문서 %d편 · 검산 탈락 %d · 관측된 속성 %d종 · 채택(≥%d 클래스) %d종 · **현행에 없는 것 %d종**"
          % (ndocs, rejected, len(seen_classes), a.minclasses, len(adopt), len(new)))
    print("\n★[정책이 **요건**이라고 말한 속성] — 채택의 1차 기준")
    for n, s in req_ranked[:20]:
        c, rq, q = req_example[n]
        print("  %-32s %2d 클래스  %-14s %s" % (n[:32], len(s), str(rq)[:14], q[:58]))
    print("\n[채택되었는데 현행 선언에 없는 속성] — 이것이 목록의 결손이다")
    for n, s in new[:30]:
        # ★요건에만 등장한 축은 `example` 에 없다 — 1차 실행이 여기서 `KeyError` 로 죽어
        #   **JSON 산출물을 통째로 잃었다**(2026-08-21). 두 사전 중 있는 쪽을 쓴다.
        c, v, q = example.get(n) or req_example.get(n) or ("?", "?", "")
        print("  %-34s %2d 클래스  예: %s=%s  “%s”" % (n[:34], len(s), c[:14], str(v)[:14], q[:60]))
    print("\n[현행 선언에 있는데 한 번도 관측 안 된 속성]")
    print("  " + (", ".join(missing_now) if missing_now else "(없음)"))

    # ⛔말미의 두 번째 쓰기는 **삭제했다**(2026-08-21). 같은 경로에 구 payload 를 다시 써서 위의 선기록을
    #   덮었고, 그래서 완주한 런의 산출물엔 `families` 가 **없었다** — 선기록이 살아남는 경우가
    #   *"보고 루프가 죽었을 때"* 뿐이었다는 뜻이다. 쓰기는 한 자리만 둔다.
    return 0


if __name__ == "__main__":
    sys.exit(main())
