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
NL = chr(10)

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


def all_docs_by_class(fams):
    """코퍼스 **전수**를 (클래스 키 → [(doc_id, text)]) 로 나눈다 (2026-08-21).

    사용자 지시: *"문서 전체적으로 깨끗하게 읽고"*. 종전 스캔은 선언된 계열
    (`catalog_arg_families.account_class`)만 읽어서 **선언 밖 149편을 통째로 건너뛰었다** —
    그런데 이 축에 정작 필요한 **공용 APY 정책**이 거기 있다:
        doc_bank_accounts_bank_accounts_(general)_012  Linked Checking Account APY Boosts …
        doc_bank_accounts_bank_accounts_(general)_045  Credit Card APY Bonuses: Stacking Policy
        doc_bank_accounts_bank_accounts_(general)_046  Linked Checking Account APY Boost: Selection Policy
    즉 색인을 저작하면서 **스태킹 규칙 문서를 안 보고** 있었다. 카드 계열도 마찬가지로 빠져 있었다.

    선언된 계열은 **종전 키 그대로**(계열 내 클래스) 두어 기존 산출물과 비교 가능하게 하고,
    나머지는 파일명 규약으로 키를 만든다(`doc_<key>_<NNN>.json` → `<key>`·형태만·[[59]]).
    """
    byc = C.docs_by_class(fams)
    taken = {did for cl in byc for did, _t in byc[cl]}
    for f in sorted(os.listdir(FT.DOCDIR)):
        if not f.endswith(".json"):
            continue
        did = f[:-5]
        if did in taken:
            continue
        key = re.sub(r"_\d+$", "", did[4:] if did.startswith("doc_") else did)
        with io.open(os.path.join(FT.DOCDIR, f), encoding="utf-8") as fh:
            d = json.load(fh)
        byc[key].append((did, (d.get("title") or "") + ". " + (d.get("content") or "")))
    return byc


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


def char_span(lines, i, k):
    """줄 인덱스 → **(문자 오프셋, 길이)**. 문서 원문 기준(줄바꿈 1자 포함).

    사용자 지시(2026-08-21): *"문서 id, 오프셋과 읽을 길이를 지정해주면 되지 않을까?"* — 절
    이름보다 이쪽이 그대로 **읽기 명세**가 된다. 선언이 `(doc, offset, length)` 를 주면 엔진은
    자를 뿐 해석하지 않는다([[59]]).
    """
    off = sum(len(l) + 1 for l in lines[:i])
    return off, len(NL.join(lines[i:i + k]))


def section_span(lines, i):
    """그 줄을 감싸는 **절**의 (오프셋, 길이) — 가장 가까운 앞선 헤딩부터 같은/상위 레벨 직전까지.

    인용 한 줄만 넘기면 문맥이 없다(`Tier 1 APY: 3.0%` 만으로는 계층인지 모른다). 절은 문서가
    스스로 그은 경계라 우리가 자르는 것이 아니다. 헤딩이 없으면 문서 전체.
    """
    head = None
    for j in range(min(i, len(lines) - 1), -1, -1):
        m = re.match(r"\s*(#{1,6})\s+\S", lines[j] or "")
        if m:
            head = (j, len(m.group(1)))
            break
    if head is None:
        return 0, len(NL.join(lines))
    start, lvl = head
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"\s*(#{1,6})\s+\S", lines[j] or "")
        if m and len(m.group(1)) <= lvl:
            end = j
            break
    return char_span(lines, start, end - start)


def section_of(lines, i):
    r"""그 줄을 감싸는 소제목 — 마크다운 헤딩 한 줄(형태만·`^#{1,6}\s`)."""
    for j in range(min(i, len(lines) - 1), -1, -1):
        if re.match(r"\s*#{1,6}\s+\S", lines[j] or ""):
            return " ".join((lines[j] or "").split())
    return ""


def cite_row(did, cls, axis, value, q, lines, text, li, lk):
    """한 인용의 **읽기 명세** — 문서 id · 인용 범위 · 감싸는 절 범위 · 검산 결과.

    `read = (doc, section_off, section_len)` 하나면 격리 서브에 넘길 재료가 확정된다.
    엔진은 자른 범위가 그 인용을 **실제로 담는지** 다시 본다(닫힌 술어·정본 `quote_in`).
    """
    row = {"doc": did, "class": cls, "axis": axis, "value": str(value)[:80],
           "line": li, "span": lk, "quote": " ".join(q.split())[:300],
           "quote_off": None, "quote_len": None,
           "section": "", "section_off": None, "section_len": None, "span_ok": None}
    if li is None:
        return row
    qo, ql = char_span(lines, li, lk)
    so, sl = section_span(lines, li)
    row.update({"quote_off": qo, "quote_len": ql,
                "section": section_of(lines, li), "section_off": so, "section_len": sl,
                "span_ok": bool(C.contained(q, (text or "")[so:so + sl]))})
    return row


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
    ap.add_argument("--all-docs", action="store_true", default=True,
                    help="코퍼스 전수(선언 밖 문서 포함·기본 ON) — 공용 정책이 선언 밖에 있다")
    ap.add_argument("--declared-only", dest="all_docs", action="store_false",
                    help="선언된 계열만(종전 거동)")
    # ★샤딩 (2026-08-21): 전수 698편은 한 프로세스로 3~5시간이다. 클래스 키를 **정렬 후
    #   나머지 연산**으로 갈라 두 GPU 에 태우면 절반이 된다. 분할이 결정론이라 합집합이 곧 전수고,
    #   합치기는 사전 union 하나다(`x453_merge_shards.py`). 샤드마다 `--out` 을 달리 준다.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
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
    byc = all_docs_by_class(fams) if (a.all_docs and not a.only) else C.docs_by_class(fams)
    if a.of > 1:
        _keys = sorted(byc)
        byc = {k: byc[k] for i, k in enumerate(_keys) if i % a.of == a.shard}
    have = {n for n, _al in FT.ATTRS}
    print("=" * 96)
    print("x453 · 선언 계열 %d · 클래스 %d · 문서 %d편%s%s · 현행 선언 속성 %d종 · 채택선 = 클래스 %d 이상"
          % (len(fams), len(byc), sum(len(v) for v in byc.values()),
             " (전수)" if (a.all_docs and not a.only) else " (선언 계열만)",
             ("  샤드 %d/%d" % (a.shard, a.of)) if a.of > 1 else "",
             len(have), a.minclasses))
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
    # ★"기능별로 문서 빠지지 않게"(사용자 지시 2026-08-21)를 **검증 가능하게** 만드는 두 계기:
    #   ⑴ 읽었는데 아무 축도 못 건진 문서 ⑵ 검산에서 떨어진 (문서·축) 짝.
    #   이 둘이 없으면 색인의 구멍이 조용히 남는다 — 빠진 것을 세지 않으면 "전수"는 말뿐이다.
    doc_yield = {}                                  # doc_id -> 채택된 (값+요건) 수
    rejected_detail = []                            # [{doc,name,why,quote}]  검산 탈락 전수
    attr_docs = collections.defaultdict(set)        # attr -> {doc_id}  ★인용이 검산된 문서만
    class_docs = collections.defaultdict(set)       # class -> {doc_id} (스캔 분모)
    example = {}                                    # attr -> (class, value, quote)
    req_example = {}                                # attr -> (class, requirement, quote)
    rejected = ndocs = retried = 0
    empty_after_retry = []                          # 두 번 물어도 빈 답 — 진짜 구멍 후보
    for cls in sorted(byc):
        docs = byc[cls]
        if a.maxdocs:
            docs = docs[:a.maxdocs]
        for did, text in docs:
            ndocs += 1
            class_docs[cls].add(did)
            doc_yield[did] = 0
            body = " ".join(text.split())[:12000]
            # ★위치 기록용 **원문 줄** (검산은 종전 `body` 그대로 — 거동 보존).
            #   실측: 문서 최대 7,878자라 12,000 절단에 걸리는 문서는 **0편**이다.
            raw_lines = (text or "").split("\n")
            # ★잘림은 **빈 답과 구분이 안 된다** (2026-08-21·사용자 지시 *"어차피 런타임에 모든
            #   문서 읽을려면 비용이 많이 든다. 이번 한번에 빠짐 없이 기록하라"*): `ask` 는 JSON
            #   파싱 실패에도 `{}` 를 돌려주므로, 속성이 촘촘한 문서에서 900 토큰에 잘리면
            #   *"이 문서엔 아무것도 없다"* 로 조용히 기록된다. 한도를 올리고, **그래도 빈 답이면
            #   한 번 더 크게** 물어 본다 — 재질의 수를 세어 구멍을 수치로 남긴다.
            _q = "# Document %s\n%s\n" % (did, body)
            got = X.ask(a.port, SYS, _q, maxtok=2400) or {}
            if not got:
                retried += 1
                got = X.ask(a.port, SYS, _q, maxtok=4000) or {}
                if not got:
                    empty_after_retry.append(did)
            for it in (got.get("attributes") or []):
                if not isinstance(it, dict):
                    continue
                nm, v, q = slug(it.get("name")), str(it.get("value") or ""), str(it.get("quote") or "")
                if not (nm and v and q):
                    continue
                if not (C.contained(q, body) and C.contained(v, q)):
                    rejected += 1
                    rejected_detail.append({"doc": did, "name": nm, "axis": "value",
                                            "why": ("quote not in doc" if not C.contained(q, body)
                                                    else "value not in quote"),
                                            "value": v[:60], "quote": " ".join(q.split())[:160]})
                    continue
                seen_classes[nm].add(cls)
                doc_yield[did] = doc_yield.get(did, 0) + 1
                _li, _lk = locate(q, raw_lines)
                cites[nm].append(cite_row(did, cls, "value", v, q, raw_lines, text, _li, _lk))
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
                    rejected_detail.append({"doc": did, "name": nm, "axis": "requirement",
                                            "why": ("quote not in doc" if not C.contained(q, body)
                                                    else "requirement not in quote"),
                                            "value": rq[:60], "quote": " ".join(q.split())[:160]})
                    continue
                req_classes[nm].add(cls)
                doc_yield[did] = doc_yield.get(did, 0) + 1
                _li, _lk = locate(q, raw_lines)
                cites[nm].append(cite_row(did, cls, "requirement", rq, q, raw_lines, text, _li, _lk))
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
               "shard": a.shard, "of": a.of, "classes_in_shard": sorted(byc),
               "families": fams,
               "observed": {n: sorted(s) for n, s in seen_classes.items()},
               "example": {n: {"class": c, "value": v, "quote": q}
                           for n, (c, v, q) in example.items()},
               "requirements": {n: sorted(s) for n, s in req_classes.items()},
               "req_example": {n: {"class": c, "requirement": r, "quote": q}
                               for n, (c, r, q) in req_example.items()},
               "attr_docs": {n: sorted(v) for n, v in attr_docs.items()},
               "cites": {n: v for n, v in cites.items()},
               "per_doc_yield": doc_yield, "n_retried": retried,
               "empty_after_retry": empty_after_retry,
               "docs_with_no_attrs": sorted(k for k, v in doc_yield.items() if not v),
               "rejected_detail": rejected_detail,
               "n_cites": sum(len(v) for v in cites.values()),
               "n_cites_unlocated": sum(1 for v in cites.values() for c in v if c["line"] is None),
               "n_cites_span_bad": sum(1 for v in cites.values() for c in v
                                       if c.get("span_ok") is False),
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
    _zero = sorted(k for k, v in doc_yield.items() if not v)
    print("\n[읽었는데 아무 축도 못 건진 문서] %d / %d편 — 여기가 색인의 구멍 후보다"
          % (len(_zero), len(doc_yield)))
    print("  " + (", ".join(_zero[:12]) if _zero else "(없음)"))
    print("[검산 탈락 상세] %d건 (문서·축·사유를 산출물에 전수 기록)" % len(rejected_detail))
    print("[빈 답 재질의] %d건 · 두 번 물어도 빈 답 %d편" % (retried, len(empty_after_retry)))
    _nc = sum(len(v) for v in cites.values())
    _nl = sum(1 for v in cites.values() for c in v if c["line"] is None)
    _nb = sum(1 for v in cites.values() for c in v if c.get("span_ok") is False)
    _sl = [c["section_len"] for v in cites.values() for c in v if c.get("section_len")]
    print("[읽기 명세] 인용 %d건 · 위치 못 잡음 %d · 절 범위 검산 실패 %d · 절 길이 중앙 %d자"
          % (_nc, _nl, _nb, (sorted(_sl)[len(_sl) // 2] if _sl else 0)))
    print("\n[현행 선언에 있는데 한 번도 관측 안 된 속성]")
    print("  " + (", ".join(missing_now) if missing_now else "(없음)"))

    # ⛔말미의 두 번째 쓰기는 **삭제했다**(2026-08-21). 같은 경로에 구 payload 를 다시 써서 위의 선기록을
    #   덮었고, 그래서 완주한 런의 산출물엔 `families` 가 **없었다** — 선기록이 살아남는 경우가
    #   *"보고 루프가 죽었을 때"* 뿐이었다는 뜻이다. 쓰기는 한 자리만 둔다.
    return 0


if __name__ == "__main__":
    sys.exit(main())
