# -*- coding: utf-8 -*-
r"""t2_search — **검색 에이전트의 결정론 부분** (2026-08-10·원장 C405).

## 왜

x237: A3 선언 축자와 파일명 규약을 다 보여 줘도 필요한 문서 회수가 **최고 3/8** 이다. 모델은
코퍼스에 없는 문구로 grep 한다(`'no overdraft fees'` ↔ 실제 *"Overdrafts incur a fee of $0.00"*).
x236: 도구를 `shell` 하나로 줄이면 **8/8** 이 그것을 쓴다 — 도구 선택은 *빼기*로 닫힌다(C404).
⇒ **어느 문서를 읽을지는 결정론으로 정한다.** 문구를 짓게 하지 않는다.

## 분담 (이 모듈이 하는 것과 안 하는 것)

  LLM        무엇이 필요한가(주어군·축) · 문서의 유효 구간 형식화        ← **여기 없음**(호출부)
  이 모듈    A3 링크 → 문서 id → **파일 읽기** · 만료 제외 · 인용 추출   ← 결정론
  모델       남은 것 중 선택                                            ← 끝까지 모델

## 경계 (C405⒠·사용자 결정)

  고객 DB 읽기 = 모델 몫(대화마다 달라지는 도메인 행동) — 여기서 하지 않는다.
  정책 문서 읽기 = 우리 층 가능(대화와 무관한 고정 상수) — A3 가 빌드 시점에 한 그 일과
  **시점만 다르다**.

⚠엔진은 문서 **내용을 해석하지 않는다** — 읽고, 링크로 고르고, 준 구간으로 거를 뿐이다([[59]]).
⚠**모르면 안 뺀다**: 유효 구간이 없는 문서는 남긴다([[25]]).
⚠**시야 = 링크 커버리지**([[50]] ADB). `coverage()` 가 그 수를 인쇄한다.
"""
import glob
import json
import os
import re

__all__ = ["linked_docs", "read_docs", "drop_expired", "coverage", "as_material",
           "docs_for", "declared_windows", "index_coverage", "material_for",
           "corpus_from_env"]

_DATE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})")


def _rows(a2):
    return ((a2 or {}).get("policy_ontology") or {}).get("rows") or []


def linked_docs(a2, subjects=None, axes=None):
    """A3 가 링크한 문서 id 집합. 주어·축으로 좁힐 수 있다 (없으면 전부)."""
    out = []
    for r in _rows(a2):
        if subjects and r.get("subject") not in subjects:
            continue
        if axes and r.get("axis") not in axes:
            continue
        d = (r.get("source") or {}).get("doc")
        if d and d not in out:
            out.append(d)
    return out


def _ontology(a2):
    return ((a2 or {}).get("policy_ontology") or {})


def docs_for(a2, group, subjects=None, general=True):
    """A3 **문서 색인**(`doc_index`)에서 그 문서군의 문서 id — 주어로 좁힐 수 있다.

    색인은 빌드 시점에 파일명에서 유도해 적어 둔 것이다(x244). 엔진은 **적힌 것을 읽기만**
    하고 이름을 뜯지 않는다([[59]]). 주어를 안 주면 그 군 전체이고, `general` 이면 주어 없는
    공통 문서(`_general_`)를 함께 준다 — 070/071 의 프로모션 고지가 거기 산다.

    ⚠고르는 것은 여전히 모델이다. 이 함수는 **어느 파일을 읽을지**만 정한다(⛔0 ③).
    """
    idx = (_ontology(a2).get("doc_index") or {}).get(group) or {}
    out = []
    for s in (sorted(idx) if subjects is None else list(subjects)):
        if s == "_general_":
            continue
        for d in (idx.get(s) or ()):
            if d not in out:
                out.append(d)
    if general:
        for d in (idx.get("_general_") or ()):
            if d not in out:
                out.append(d)
    return out


def declared_windows(a2, doc_ids=None):
    """A3 가 선언한 **유효 구간** — `{문서: (시작, 끝)}`. 적히지 않은 문서는 안 들어온다.

    구간은 빌드 시점에 **LLM 이 문서 축자에서** 형식화하고 엔진이 인용 실재를 검산해 적은
    것이다(x242). 여기서는 조회만 한다 — 모르면 없고, 없으면 `drop_expired` 가 **안 뺀다**([[25]]).
    """
    want = set(doc_ids) if doc_ids else None
    out = {}
    for r in (_ontology(a2).get("doc_windows") or ()):
        d = r.get("doc")
        if not d or (want is not None and d not in want):
            continue
        out[d] = (r.get("from"), r.get("to"))
    return out


def index_coverage(a2, doc_dir):
    """색인 커버리지 = 이 에이전트의 시야([[50]] ADB). (색인 수, 코퍼스 수, 비율)."""
    idx = _ontology(a2).get("doc_index") or {}
    n = len({d for subs in idx.values() for v in subs.values() for d in v})
    total = len(glob.glob(os.path.join(doc_dir, "doc_*.json")))
    return n, total, (n / total if total else 0.0)


def coverage(a2, doc_dir):
    """링크 커버리지 = 이 에이전트의 시야. (링크 수, 코퍼스 수, 비율)."""
    linked = set(linked_docs(a2))
    total = len(glob.glob(os.path.join(doc_dir, "doc_*.json")))
    return len(linked), total, (len(linked) / total if total else 0.0)


def corpus_from_env(env):
    """환경이 **이미 메모리에 들고 있는** 문서 → `{id: 본문}` (없으면 빈 dict).

    ★왜 이 경로인가 ([[05]]): 코퍼스 경로를 엔진에 박으면 그 순간 도메인-특화가 순증한다.
      하네스는 `KnowledgeBase.documents` 를 만들어 도구에 넘겨 두므로(`environment.py`
      `build_tools(variant, db, knowledge_base, …)`) **같은 것을 그대로 읽으면** 된다 —
      새 I/O 도 새 상수도 없다. 찾지 못하면 빈 dict 이고, 그러면 검색 에이전트는 침묵한다.
    ⚠속성 이름은 하네스 사정이라 몇 가지를 시도한다. 못 찾는 것은 **조용한 성공보다 낫다**.
    """
    seen = []
    for holder in (env, getattr(env, "tools", None), getattr(env, "user_tools", None)):
        if holder is None:
            continue
        for attr in ("knowledge_base", "kb", "_knowledge_base", "_kb"):
            kb = getattr(holder, attr, None)
            docs = getattr(kb, "documents", None)
            if isinstance(docs, dict) and docs:
                for k, v in docs.items():
                    seen.append((str(k), str(getattr(v, "content", v) or "")))
                return dict(seen)
    return {}


def read_docs(doc_ids, doc_dir=None, corpus=None):
    """문서 id → {id: 본문}. 없는 id 는 **조용히 건너뛰지 않고** 표시한다.

    `corpus` 를 주면 그것(=환경이 든 문서)에서 읽고, 없으면 `doc_dir` 에서 읽는다.
    """
    out, missing = {}, []
    for d in doc_ids:
        if corpus is not None:
            if d in corpus:
                out[d] = str(corpus[d] or "")
            else:
                missing.append(d)
            continue
        p = os.path.join(doc_dir or "", "%s.json" % d)
        if not os.path.exists(p):
            missing.append(d)
            continue
        try:
            o = json.load(open(p, encoding="utf-8"))
        except Exception:
            missing.append(d)
            continue
        out[d] = str(o.get("content") or "")
    return out, missing


def drop_expired(docs, spans, now):
    """만료 제외. `spans` = {doc_id: (from, to)} — **LLM 이 형식화한 것**을 받는다.

    반환: (남은 docs, 뺀 것 [(doc, from, to)]). 구간이 없는 문서는 **남긴다**.
    비교는 `YYYY-MM-DD` 문자열 순서 — 날짜 산수뿐이고 내용 해석은 없다.
    """
    keep, dropped = {}, []
    n = (_DATE.match(str(now)) or [None])
    now_s = str(now)[:10] if _DATE.match(str(now)) else None
    for d, c in docs.items():
        sp = (spans or {}).get(d)
        if not sp or not now_s:
            keep[d] = c
            continue
        f, t = (str(sp[0])[:10] if sp[0] else None), (str(sp[1])[:10] if sp[1] else None)
        if (f and now_s < f) or (t and now_s > t):
            dropped.append((d, f, t))
        else:
            keep[d] = c
    return keep, dropped


def material_for(a2, group, doc_dir=None, now=None, per_doc=400, windowed="general",
                 corpus=None):
    """검색 에이전트의 결정론부 **전체 체인**: 색인 → 읽기 → 만료 제거 → 축자 재료.

    ## 왜 이 모양인가 (x243·n=8 — 재고 결정론을 **줄인** 결과)

        S1 축별 문장 + 활성 프로모션    8/8
        S2 제품 문서 **전문** + 프로모션 8/8   ← 축 선별이 필요 없다
        S3 문서 **앞 400자** + 프로모션  8/8   ← 그래서 기본 `per_doc=400`
        S4 문서만(프로모션 없음)        0/8   ← 유효창이 본체다

    ⇒ 축 링크도 축 형식화도 짓지 않는다(⛔0 ③). 엔진이 하는 일은 **읽고·비교하고·자르기**뿐이고,
      고르는 것은 끝까지 모델이다.

    ## 무엇을 싣나

      · 그 문서군의 문서(A3 `doc_index` 가 적어 둔 id)
      · **효력 있는** 유효창 문서(A3 `doc_windows`) — 프로모션 고지는 제품군이 아니라
        `_general_` 쪽에 살기 때문에 군만 읽으면 놓친다(070/071 실측).
      · 뺀 것은 **이유와 함께** 남긴다(C327 — 조용히 빼지 않는다).

    `windowed`: **기본 `"general"`** = `_general_` 풀의 효력 있는 문서만 · `"all"` = 전부 ·
    `"none"` = 안 싣는다(부정 통제).

    ★기본이 `"general"` 인 이유 (x248·071 실물·n=8): 두 축 모두 **8/8** 인데 `"all"`(다른 상품군의
      효력 있는 고지까지)은 **7/8·4/8 로 떨어진다** — 답이 섞인다(`Sky Blue Gold Saver Account`).
      *더하기는 해롭다*(C404)의 또 한 사례이므로 **좁힌 판을 기본**으로 둔다.
    `corpus` 를 주면 디스크 대신 **환경이 든 문서**에서 읽는다(`corpus_from_env`).

    ⚠엔진은 문서 내용을 해석하지 않는다([[59]]). ⚠구간을 모르는 문서는 **남긴다**([[25]]).
    """
    ids = list(docs_for(a2, group))
    if windowed != "none":
        pool = set(declared_windows(a2))
        if windowed == "general":
            idx = _ontology(a2).get("doc_index") or {}
            gen = {d for subs in idx.values() for d in (subs.get("_general_") or ())}
            pool &= gen
        for d in sorted(pool):
            if d not in ids:
                ids.append(d)
    read, missing = read_docs(ids, doc_dir, corpus)
    keep, dropped = drop_expired(read, declared_windows(a2, read), now)
    return as_material(keep, dropped, per_doc=per_doc), {
        "linked": len(ids), "read": len(read), "missing": missing,
        "kept": len(keep), "dropped": [d for d, _f, _t in dropped]}


def as_material(docs, dropped=(), per_doc=1200):
    """결정점에 실을 재료 — 문서 **축자**와, 뺀 것의 **이유**(C327: 조용히 빼지 않는다)."""
    parts = []
    for d, c in docs.items():
        parts.append("[%s]\n%s" % (d, " ".join(str(c).split())[:per_doc]))
    if dropped:
        parts.append("Excluded as out of date (their stated period does not include today): "
                     + "; ".join("%s (%s–%s)" % (d, f or "?", t or "?") for d, f, t in dropped))
    return "\n\n".join(parts)
