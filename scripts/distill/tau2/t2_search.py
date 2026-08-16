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
import sys

import t2_subcall as SC   # 단발-격리 서브 정본(2026-08-14 리팩토링)

__all__ = ["linked_docs", "read_docs", "drop_expired", "coverage", "as_material",
           "docs_for", "declared_windows", "index_coverage", "material_for",
           "corpus_from_env", "action_index_note", "formalize_groups", "decide_from_docs", "to_iso"]

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


def action_index_note(a2):
    """A3 `action_index` 를 **한 번** 인쇄할 문구로 — 엔진은 고르지 않는다(나열만·[[62]] ④).

    무엇을 넣나: 행동을 기술하는 문서의 **제목**과 그 문서가 대는 **도구 이름**. 43줄이다.
    왜 이 크기인가(x319·n=24·블록별 8·8·8 — 잡음 바닥 ±4 밖):
        도움 없음 **10/24** → **action 문서 제목 43줄 24/24**
        도구 설명 91종 23/24 · **이름만 91종 16/24**
    ⇒ 표면화는 듣고, **의미를 담은 것이 이름보다 낫다**. 그리고 가장 싼 것(43줄)이 가장 좋다 —
      698 문서를 읽을 필요도, 91 설명을 실을 필요도 없다(사용자 지시 *"비용이 최소가 되게"*).

    선언이 없으면 **빈 문자열**(거동 변화 0). 문면은 A3 `action_index_text`(엔진 리터럴 0·[[05]]).
    """
    po = _ontology(a2)
    rows = po.get("action_index") or []
    head = po.get("action_index_text")
    if not rows or not head:
        return ""
    out = [head]
    for r in rows:
        t = " ".join(str(r.get("title") or "").split())
        tools = ", ".join(r.get("tools") or [])
        out.append("- %s%s" % (t, (" [%s]" % tools) if tools else ""))
    return "\n".join(out)


def corpus_from_env(env):
    """환경이 **이미 메모리에 들고 있는** 문서 → `{id: 본문}` (없으면 빈 dict).

    ★왜 이 경로인가 ([[05]]): 코퍼스 경로를 엔진에 박으면 그 순간 도메인-특화가 순증한다.
      하네스는 `KnowledgeBase.documents` 를 만들어 도구에 넘겨 두므로(`environment.py`
      `build_tools(variant, db, knowledge_base, …)`) **같은 것을 그대로 읽으면** 된다 —
      새 I/O 도 새 상수도 없다. 찾지 못하면 빈 dict 이고, 그러면 검색 에이전트는 침묵한다.
    ★**이름 추측 → 구조 탐색**으로 바꿨다 (2026-08-11·라이브 실패 2회째): 첫 판은
      `knowledge_base`·`kb` 같은 속성 이름을 찍었는데 **툴킷은 KB 를 안 들고 있다** — 들고 있는
      것은 파이프라인이고(`_kb_pipeline`), 문서는 그 안의 `state["documents"]` /
      `state["doc_content_map"]` 에 산다(`tau2/knowledge/pipeline.py`). 라이브에서
      *"환경에서 문서를 못 찾음"* 8회로 죽었다. 하네스 내부 이름에 매이지 않도록 **모양으로**
      찾는다: 얕은 깊이에서 `{id: 본문}` 또는 `[{id,text}]` 를 지닌 것을 집는다.
    ⚠깊이 5·폭 60 으로 묶는다(순환·거대 객체 방어). 실제 경로는 `tools → _kb_pipeline →
      state → doc_content_map` 로 **4단**이라 3단 제한은 못 닿았다(첫 판의 두 번째 실패).
    ⚠못 찾으면 빈 dict — 침묵이 조용한 성공보다 낫다.
    """
    def _as_map(v):
        if isinstance(v, dict) and v:
            ks = list(v)[:3]
            if all(isinstance(k, str) and str(k).startswith("doc_") for k in ks):
                first = v[ks[0]]
                if isinstance(first, str):
                    return {str(k): str(x or "") for k, x in v.items()}
                if hasattr(first, "content"):
                    return {str(k): str(getattr(x, "content", "") or "") for k, x in v.items()}
        if isinstance(v, list) and v and isinstance(v[0], dict):
            f = v[0]
            key = "text" if "text" in f else ("content" if "content" in f else None)
            if key and "id" in f:
                return {str(d.get("id")): str(d.get(key) or "") for d in v if d.get("id")}
        return None

    seen, queue, depth = set(), [(env, 0)], 0
    while queue:
        obj, d = queue.pop(0)
        if obj is None or d > 5 or id(obj) in seen:
            continue
        seen.add(id(obj))
        got = _as_map(obj)
        if got:
            return got
        holders = []
        if isinstance(obj, dict):
            holders = list(obj.values())[:40]
        else:
            for attr in dir(obj) if d else ("tools", "user_tools", "environment", "db"):
                if attr.startswith("__"):
                    continue
                try:
                    holders.append(getattr(obj, attr))
                except Exception:
                    continue
        for h in holders[:60]:
            if isinstance(h, (dict, list)) or hasattr(h, "__dict__"):
                queue.append((h, d + 1))
    return {}


def to_iso(value, fmts=("%m/%d/%Y", "%Y-%m-%d")):
    """선언된 형식의 날짜 문자열 → `YYYY-MM-DD` (못 읽으면 None).

    ★왜 필요한가 (2026-08-11 라이브·세 번째 死배선): `formalize_now` 는 A2 가 선언한 형식
      (`%m/%d/%Y`)으로 돌려주는데 `drop_expired` 는 ISO 문자열 순서로 비교한다. 그대로 넘기면
      `_DATE` 정규식이 안 맞아 `now_s=None` → **아무것도 안 빠진다**. 즉 성공한 것처럼 보이면서
      엔진의 유일한 일이 죽는다 — 로그에는 `뺀 것 0` 으로만 남는다.
    ⚠날짜 산수뿐이고 도메인 해석은 없다([[59]]). 못 읽으면 None → 그러면 안 뺀다([[25]]).
    """
    import datetime as _dt
    s = str(value or "").strip()
    if not s:
        return None
    for f in fmts:
        try:
            return _dt.datetime.strptime(s, f).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


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


def formalize_groups(agent, la, UserMessage, spec, texts, groups):
    """① 손님의 말 → **문서군 목록**(요청 하나당 하나). 닫힌 집합(A3 `doc_index` 키)이라 엔진이 검증한다([[22]]).

    ★필요성 (x248·071 실물): 군을 잘못 고르면 재료가 통째로 어긋난다. 첫 판은 요청을 문맥
      **뒤**에 놓아 savings 요청에도 checking 군을 골랐고, 재료가 겹친 덕에 답만 우연히 맞았다
      (= 조용한 오류). 요청을 **머리**에 두니 두 축 다 맞았다 — 그래서 여기서도 요청이 앞이다.
    ⚠못 고르면 **None** → 검색 에이전트는 침묵한다(모르면 안 뺀다·[[25]]).
    ⚠엔진은 답이 키 집합의 원소인지만 본다. 군 이름의 뜻은 모른다([[59]]).
    """
    tpl = (spec or {}).get("group_prompt")
    if not (tpl and agent is not None and la is not None and texts and groups):
        return []
    names = sorted(groups)
    listing = "\n".join("  %s" % g for g in names)
    ask = "\n---\n".join(str(t) for t in texts)[-6000:]
    try:
        raw = " ".join(SC.sub_generate(agent, la, UserMessage, tpl.format(groups=listing, text=ask), "doc_group_formalize").split())
    except Exception as e:
        print("[T2_DOCGROUP] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return []
    # ★**복수로 받는다** (2026-08-11·사용자 지시 *"축별로 분리해서 결정점을 나누게 하라"*):
    #   071 은 요청이 둘이다(사업자 체킹 하나 + 사업자 세이빙 하나). 하나만 고르게 하면 나머지
    #   축은 영영 못 다루고, 실제로 라이브에서 개인 체킹을 골라 손님이 **이미 가진** 계좌를
    #   추천했다. x252 는 축을 하나씩 물었을 때 두 축 다 8/8 이었다 — 라이브를 그 형태로 되돌린다.
    #   ⚠고르는 것은 여전히 LLM 이다. 엔진은 **답에 등장한 순서대로** 담기만 한다(순위 계산 0).
    lower = raw.lower()
    out = sorted((g for g in names if g and g.lower() in lower), key=lambda g: lower.find(g.lower()))
    # 더 긴 이름이 짧은 이름을 품으면(`business_checking_accounts` ⊃ `checking_accounts`)
    # 짧은 쪽은 **그 자리에서 나온 것이 아니다** — 문자열 포함일 뿐이라 뺀다.
    out = [g for g in out if not any(o != g and g.lower() in o.lower() for o in out)]
    print("[T2_DOCGROUP] raw=%r → %s" % (raw[:80], ", ".join(out) or "군 집합 밖 = 침묵"),
          file=sys.stderr, flush=True)
    return out


def sub_records(agent, la, UserMessage, text, keys):
    """도구 출력 **한 덩이 → 레코드 JSON**(LLM formalize). 엔진은 뽑지 않는다([[59]] 강화판).

    ★왜 (2026-08-17·사용자 지시 *"절대 결정론기에서 … 어떠한 정규식도 쓰면 안된다"*):
      `t2_resolve.parse_records`(정규식 줄 파서)와 `_ref_verify_deny` 의 `field:` 추출이
      살아 있는 마지막 두 자리였다. 판정(상한 비교·상점 대조)은 엔진에 남지만 **입력은
      서브가 만든다**.
    ★전제는 측정됐다(x345·det n=1·표집 24/24): 격리 서브가 레코드 값을 **축자로 복원**한다.

    계약: 모델이 JSON 을 낸다 · 엔진은 **각 값이 원문에 실재하는지만** 확인하고(`in`),
      통과 못 한 값은 **버린다**(조용한 오답 금지). 못 만들면 **빈 목록** → 호출부는 종전처럼
      아무 판정도 하지 않는다(fail-open).
    ⚠정규식 0 — JSON 경계는 `find`/`rfind`.
    """
    if not (agent is not None and la is not None and text and keys):
        return []
    body = ("Below is one tool output.\n\n" + str(text)[:6000]
            + "\n\nReturn the records it contains as a JSON array of objects with exactly "
              "these keys: " + ", ".join(keys)
            + ". Copy every value VERBATIM from the text - do not compute, convert, or infer. "
              "If a key is absent for a record, set it to null. Reply with JSON only.")
    try:
        raw = str(SC.sub_generate(agent, la, UserMessage, body, "sub_records") or "")
    except Exception as e:
        print("[T2_SUB_RECORDS] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return []
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        rows = json.loads(raw[i:j + 1])
    except Exception:
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        keep = {}
        for k in keys:
            v = r.get(k)
            if v is None:
                continue
            if str(v) in text:                 # ★존재확인만 (추출 0·C45 동형)
                keep[k] = v
        if keep:
            out.append(keep)
    print("[T2_SUB_RECORDS] %d행 중 원문 검증 통과 %d행" % (len(rows or []), len(out)),
          file=sys.stderr, flush=True)
    return out


def sub_requirements(agent, la, UserMessage, spec, user_text):
    """손님 발화 → **요구 인용 목록**(LLM formalize). 엔진은 뽑지 않는다([[59]]).

    ★왜 (x343·n=24=8×3·블록 편차 0): 이 다음 단계(`decide_from_docs`)가 문서+후보줄만 받으면
      `Gold Account` **24/24 오답**이고, 손님 요구를 축자로 받으면 **24/24 정답**이며, 무관한
      요구를 주면 **0/24**(부정통제 통과). 요구는 재료가 아니라 **판별자**다.

    계약: 모델이 **인용을 낸다**(문면 = A2 `requirement_prompt`·도메인 어휘 0) · 엔진은
      그 인용의 **원문 존재만** 확인한다(호출부 `in` 연산·C45 동형). 파싱·정규식 0 —
      JSON 경계는 `find`/`rfind` 문자열 연산으로만 잡는다.
    ⚠못 만들면 **빈 목록**(침묵) — 종전 거동으로 떨어진다.
    """
    tpl = (spec or {}).get("requirement_prompt")
    if not (tpl and agent is not None and la is not None and user_text):
        return []
    try:
        body = tpl.format(messages=str(user_text)[:8000])
        raw = str(SC.sub_generate(agent, la, UserMessage, body, "sub_requirement") or "")
    except Exception as e:
        print("[T2_SUB_REQUIREMENT] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return []
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        out = json.loads(raw[i:j + 1])
    except Exception:
        return []
    return [x for x in out if isinstance(x, str)] if isinstance(out, list) else []


def decide_from_docs(agent, la, UserMessage, spec, material, ask):
    """③ 남은 것 중 **고르기** — 격리 문맥(요청 + 재료)뿐. 대화 잔여물은 한 글자도 없다.

    ★구성은 측정된 그대로다(x248·n=8·두 축 8/8): 요청이 머리, 재료가 몸, 질문이 꼬리.
    ⚠엔진은 답을 **고르지 않고 나르기만** 한다 — argmax 도 최댓값도 없다(⛔0 ④).
    ⚠빈 답이면 None → 아무 말도 덧붙이지 않는다.
    """
    tpl = (spec or {}).get("doc_decide_prompt")
    if not (tpl and agent is not None and la is not None and material):
        return None
    try:
        kw = dict((k, v) for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
                  if "tool" not in k)
        body = tpl.format(ask=str(ask)[:3000], material=material)
        sub = SC.sub_generate(agent, la, UserMessage, body, "doc_decide")
        raw = " ".join(str(sub or "").split())
    except Exception as e:
        print("[T2_DOCDECIDE] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return None
    out = raw.strip().strip("*.").strip() or None
    print("[T2_DOCDECIDE] → %r" % (out,), file=sys.stderr, flush=True)
    return out


def as_material(docs, dropped=(), per_doc=1200):
    """결정점에 실을 재료 — 문서 **축자**와, 뺀 것의 **이유**(C327: 조용히 빼지 않는다)."""
    parts = []
    for d, c in docs.items():
        parts.append("[%s]\n%s" % (d, " ".join(str(c).split())[:per_doc]))
    if dropped:
        parts.append("Excluded as out of date (their stated period does not include today): "
                     + "; ".join("%s (%s–%s)" % (d, f or "?", t or "?") for d, f, t in dropped))
    return "\n\n".join(parts)
