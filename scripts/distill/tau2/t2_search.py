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

__all__ = ["linked_docs", "read_docs", "drop_expired", "coverage", "as_material"]

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


def coverage(a2, doc_dir):
    """링크 커버리지 = 이 에이전트의 시야. (링크 수, 코퍼스 수, 비율)."""
    linked = set(linked_docs(a2))
    total = len(glob.glob(os.path.join(doc_dir, "doc_*.json")))
    return len(linked), total, (len(linked) / total if total else 0.0)


def read_docs(doc_ids, doc_dir):
    """문서 id → {id: 본문}. 없는 id 는 **조용히 건너뛰지 않고** 표시한다."""
    out, missing = {}, []
    for d in doc_ids:
        p = os.path.join(doc_dir, "%s.json" % d)
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


def as_material(docs, dropped=(), per_doc=1200):
    """결정점에 실을 재료 — 문서 **축자**와, 뺀 것의 **이유**(C327: 조용히 빼지 않는다)."""
    parts = []
    for d, c in docs.items():
        parts.append("[%s]\n%s" % (d, " ".join(str(c).split())[:per_doc]))
    if dropped:
        parts.append("Excluded as out of date (their stated period does not include today): "
                     + "; ".join("%s (%s–%s)" % (d, f or "?", t or "?") for d, f, t in dropped))
    return "\n\n".join(parts)
