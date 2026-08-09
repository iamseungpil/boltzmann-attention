# -*- coding: utf-8 -*-
"""원장이 돌아오면 거기서 기계로 나오는 수를 낸다 — **전사는 LLM, 계산은 엔진**.

왜 필요한가 (101/102 전수 부검·`TASK_101_102_DETERMINISM_FORENSIC_2026_08_07`):
`get_referrals_by_user`가 29행을 돌려줘도 **아무도 세지 않는다**. 무엇을 제출할 수 있는지를 정하는
수는 창_잔여(=상한 − 최근 N일 내 건수)인데 **19/22 trial에서 언급조차 되지 않았다**. 모델이 행을
세는 일 자체는 대체로 맞다(계수 35✓/8✗). 빠진 것은 세는 능력이 아니라 **세라는 계기**다.

★분담 (사용자 지시 2026-08-07: *"패턴 매칭은 사용하지 말라. LLM이 formalize 하고 계산만 엔진이 한다"*):

    LLM   도구가 돌려준 원장 텍스트 → **구조화된 행**(A2가 이름을 준 키만)      = 해석
    엔진  그 행들 위의 산수(창 안 건수·잔여·그룹별 누적)                        = 이론

초안은 엔진이 정규식으로 `key: value` 블록을 뜯었다. 그것은 엔진이 도메인 텍스트를 읽는 것
(=엔진-formalize)이고 [[03b]]가 금지한다 — 출력 형식이 바뀌면 조용히 틀리고, 무엇보다 우리 명제
([[10]] 선택기·검증기=결정론 / 생성·해석=LLM)를 깨뜨린다. 그래서 파서를 **전부 제거**했다.

**하지 않는 것**(의도적):
  · 유형별 **연간 상한**을 말하지 않는다. 상한은 원장에 없고 정책 문서에 있다. 여기서 선언하면
    A2가 상품 수만큼 두꺼워지고([[50]] ADB) 문서가 바뀌면 조용히 틀린다. 상한은 모델이 **인용과
    함께** 가져오고 `[SOURCE]`가 인용 실재성을 검증한다([[22]]). 실측이 그 분담을 지지한다 —
    상한을 문서 회수 **후** 말하면 2✓/1✗, 문서 **없이** 말하면 0✓/5✗.
  · 추천하지 않는다. 무엇을 고를지는 모델 몫이다([[05]] Q2). 문구는 *사실*로만 쓰고, 이 계수가
    상한이 **아니라고** 명시한다 — 라이브에서 모델이 자기가 센 7을 상한으로 삼아
    *"The limit for the Sky Blue Account is 7"* 이라 말했다(실제 8).

은행 어휘 0: 키 이름·창 상수·문구·프롬프트 문장은 전부 A2가 준다.
"""

import datetime
import hashlib
import json
import re
import sys

__all__ = ["specs_for", "formalize_rows", "formalize_now", "formalize_limits",
           "formalize_thresholds", "window_and_tally", "earliest_age",
           "exhausted_text", "ineligible_text", "facts_text",
           "parse_rows", "parse_pairs", "parse_scalar"]


# ── 모델 응답 → 값 : 순수 파서 3종 (`FACT_DAG_DESIGN_2026_08_08.md` §2c `shape`) ──────────
# 왜 꺼내 두나: 파생-사실 DAG의 `formalize` 노드가 **같은 검증**을 걸어야 하는데, 새로 쓰면
# 두 벌이 되고 갈린다(`t2_precedence.py` 첫 주석이 같은 병을 적어 두었다 — 같은 술어가 두 벌이면
# T1 사실 모순이 이 층에서 다시 생긴다). LLM 호출 밖으로 꺼내면 **모델 없이 검정**도 된다.
# 거동은 그대로다 — 아래 세 `formalize_*`가 이 함수들을 부르고, 로직은 옮긴 것뿐이다.

def parse_rows(raw, keys):
    """`shape="rows"` — 리스트 · **선언된 키가 전부 있는 행만** 채택."""
    m = re.search(r"\[.*\]", str(raw or ""), re.S)
    if not m:
        return []
    try:
        rows = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict):
            keep = {k: r[k] for k in keys if r.get(k) not in (None, "")}
            if len(keep) == len(keys):
                out.append(keep)
    return out


def parse_pairs(raw, field, hay):
    """`shape="pairs"` — 딕트 · `int` · **인용 실재 검증**. 반환 `(값, 거절 수, 모델이 준 수)`.

    엔진은 인용문의 **뜻을 읽지 않는다** — 회수된 텍스트에 그 문자열이 있는지만 본다([[59]]).
    """
    m = re.search(r"\{.*\}", str(raw or ""), re.S)
    if not m:
        return {}, 0, 0
    try:
        got = json.loads(m.group(0))
    except Exception:
        return {}, 0, 0
    out, rejected = {}, 0
    for k, v in (got.items() if isinstance(got, dict) else []):
        try:
            num = int(str((v or {}).get(field)).strip())
            quote = " ".join(str((v or {}).get("quote") or "").split())
        except Exception:
            rejected += 1
            continue
        if num <= 0 or len(quote) < 12 or quote not in hay:
            rejected += 1          # 인용이 회수된 텍스트에 없다 = 채택하지 않는다
            continue
        out[str(k)] = (num, quote)
    return out, rejected, (len(got) if isinstance(got, dict) else 0)


def parse_scalar(raw, fmts):
    """`shape="scalar"` — 첫 토큰 · 선언된 형식으로 파싱되면 그 문자열, 아니면 None."""
    s = str(raw or "").strip()
    cand = s.split()[0].strip('".,') if s.split() else ""
    return cand if _date(cand, fmts) else None


def _fam(n):
    """접미사(`_3847`)를 뗀 base 이름 — 선언은 base로 적히고 호출은 접미사가 붙는다."""
    s = str(n or "")
    i = s.rfind("_")
    return s[:i] if i > 0 and s[i + 1:].isdigit() else s


def specs_for(a2, tool_name):
    """★가족 이름으로 맞춘다. 발견형 도구는 `..._3847`처럼 접미사가 붙어 호출되므로
    정확 일치로 두면 task_100의 `get_all_user_accounts_by_user_id_3847`이 선언과 안 붙는다."""
    fam = _fam(tool_name)
    return [s for s in ((a2 or {}).get("ledger_metrics") or [])
            if s.get("trigger_tool") in (tool_name, fam)
            or _fam(s.get("trigger_tool")) == fam]


def formalize_rows(agent, la, UserMessage, text, spec):
    """도구 출력 → 행 리스트. **모델이 전사한다.** 실패/불가 = [] (미개입).

    엔진이 텍스트에서 하는 일은 없다. 모델이 낸 JSON을 받아 **선언된 키만** 남긴다.
    """
    if agent is None or la is None or not text:
        return []
    keys = [k for k in (spec.get("row_keys") or []) if k]
    if not keys:
        return []
    tpl = spec.get("formalize_prompt")
    if not tpl:
        return []
    prompt = tpl.format(keys=", ".join(keys), text=str(text)[:12000])
    try:
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="ledger_rows_formalize", **kw)
        raw = getattr(sub, "content", None) or ""
    except Exception:
        return []
    return parse_rows(raw, keys)


def formalize_now(agent, la, UserMessage, texts, spec):
    """대화가 말하는 '오늘' → 날짜 문자열. **모델이 읽는다.** 못 정하면 None = 창 계산 미개입.

    엔진이 도구 결과에서 날짜를 정규식으로 뽑는 길(`t2_resolve._current_time_str`)이 이미 있지만
    그것이 [[59]]가 금지한 바로 그 형태다 — 문면이 바뀌면 조용히 틀리고, 분담을 뒤집는다.
    여기서는 텍스트를 **모델에게 주고** 형식만 받는다.
    """
    tpl = (spec or {}).get("now_prompt")
    if not (tpl and agent is not None and la is not None and texts):
        return None
    # ★한 번 정해지면 그 sim 내내 같다 — 다시 물어 봐야 같은 답이고, 실패한 자리에서 다시 물으면
    #   같은 이유로 또 실패한다. 성공을 재사용한다(호출도 아낀다).
    memo = getattr(agent, "_t2_ledger_now", None)
    if memo:
        return memo
    # ★머리 + 꼬리 (2026-08-07·x128 배제 진단). 구판은 꼬리 8개만 줬다. 실측: 이 환경은 "오늘"을
    #   **대화 첫머리**에 한 번 말하고(`The current time is …`) 그 뒤로는 안 말한다. 그래서 조회가
    #   대화 중반에 일어나면 꼬리 발췌에 그 문장이 **구조적으로 못 들어간다** — 절단도 아니고 모델이
    #   못 읽은 것도 아니다(꼬리에 날짜가 들어간 경우엔 읽어 냈다). 머리를 함께 준다.
    #   위치로만 고르고 내용은 보지 않는다([[59]] — 어느 문장이 날짜인지 판정하는 것은 모델 몫).
    head, tail = list(texts[:3]), list(texts[-8:])
    seen, sel = set(), []
    for t in head + tail:
        if id(t) not in seen:
            seen.add(id(t))
            sel.append(t)
    prompt = tpl.format(text="\n---\n".join(str(t)[:1500] for t in sel))
    try:
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="ledger_now_formalize", **kw)
        raw = (getattr(sub, "content", None) or "").strip()
    except Exception:
        return None
    cand = parse_scalar(raw, spec.get("date_formats") or ["%m/%d/%Y"])
    if not cand:
        return None
    try:
        agent._t2_ledger_now = cand
    except Exception:
        pass
    return cand


def formalize_limits(agent, la, UserMessage, texts, spec):
    """회수된 문서가 **말한 상한**을 모델이 뽑는다 — `{그룹: (상한, 인용문)}`.

    왜 여기까지 오나 (원장 C302·C304): 우리는 유형별 **누계**를 정확히 셈해 넘기고, 문구가
    *"These are counts of what has been used. They are NOT the limits"* 라고 못 박는다. 그런데
    상한과의 비교를 **아무도 하지 않는다** — 우리는 안 하기로 했고(상한은 원장에 없다) 모델은
    실제로 안 했다. 그 결과 잔여 0인 유형이 제출된다(def_k t0: `Light Green` 사용 3/상한 3).

    분담은 그대로다([[52]]·[[10]]): **해석=LLM**(어느 문서의 어느 문장이 이 유형의 상한인가),
    **산수=엔진**(상한 − 누계). 엔진은 문서를 읽지 않는다([[59]]) — 모델이 낸 JSON만 받는다.
    인용문은 **존재 검증용**이다(엔진이 뜻을 읽지 않고 회수된 텍스트에 실재하는지만 본다 —
    `t2_source`가 하는 것과 같은 종류의 확인이고 [[22]] 따름정리가 요구하는 근거-우선 계약이다).

    ⚠상한을 A2에 박지 않는다([[50]] ADB): 상품 수만큼 두꺼워지고 문서가 바뀌면 조용히 틀린다.
    ⚠**못 찾으면 비운다.** 모르는 유형은 결과에서 빠지고, 그 유형에 대해서는 아무 말도 안 한다.
    """
    return _formalize_pairs(agent, la, UserMessage, texts, spec,
                            key="limit_prompt", field="limit",
                            memo_attr="_t2_ledger_limits",
                            call_name="ledger_limits_formalize")


def formalize_thresholds(agent, la, UserMessage, texts, spec):
    """회수된 문서가 말한 **최소 기간 요건**을 모델이 뽑는다 — `{그룹: (일수, 인용문)}`.

    `formalize_limits`와 같은 계약이고 축만 다르다(연간 상한 → 관계기간 문턱). 나눈 이유는
    두 수가 서로 다른 문서 절에서 오고, 하나만 나온 경우에도 그것만으로 판정이 되기 때문이다.

    실측 근거(task_100): 손님 관계기간은 **65일**이고 우리가 이미 정확히 셈해 넘긴다. 그런데
    `World Blue`는 *"maintained checking account status with Rho-Bank for at least 90 days"* 라
    문서가 말하는데 **아무도 65와 90을 맞대지 않아** 그대로 제출됐다(런 i). 해석=LLM·산수=엔진.
    """
    return _formalize_pairs(agent, la, UserMessage, texts, spec,
                            key="threshold_prompt", field="min_days",
                            memo_attr="_t2_ledger_thresholds",
                            call_name="ledger_thresholds_formalize")


def formalize_case_facts(agent, la, UserMessage, texts, spec, wanted):
    """**손님이 대화에서 말한 수**를 축 이름으로 받는다 — `{축: (값, 인용문)}`.

    왜 필요한가 (2026-08-09·C342 후속·사용자 지적 *"자격 상한 산수로 필터해야 하는 것 아닌가"*):
    자격 기준 중 둘은 피연산자가 **DB가 아니라 대화**에 있다 — 피추천 사업체가 얼마를 예치할지,
    설립한 지 얼마나 됐는지. A3 는 그 문턱을 이미 인용과 함께 들고 있는데(`qualifying_deposit_usd`
    등) 맞댈 상대가 없어서 우리는 그 축을 **표시만 하고 거르지 않았다**. 실측 결과 그것이
    task_099 의 실패다: 거르지 않으면 통과 집합의 최고액이 `Beige` 500(예치 요건 100000·손님은
    30000)이고, 걸면 최고액이 `World Blue` 300 = gold 가 된다.

    분담은 다른 형식화와 같다([[52]]·[[10]]): **해석=LLM**(대화의 어느 문장이 그 수인가),
    **산수·검증=엔진**(인용이 대화에 실재하는지 + 문턱과의 비교). 엔진은 대화를 파싱하지 않는다.
    ★묻는 항목(`wanted`)은 **A3 축 설명 그대로** 실어 보낸다 — 엔진에도 프롬프트에도 도메인
      어휘를 새로 쓰지 않는다(도메인 내용은 출처가 있는 A3 한 곳에만 산다).
    ⚠못 찾으면 비운다. 그 축은 **거르지 않는다** — 모르는 것을 탈락으로 바꾸지 않는다.
    """
    if not wanted:
        return {}
    # ★키를 **골격으로** 준다 (2026-08-09 라이브 실측·자기정정): 항목 목록을 발췌 **뒤**에
    #   붙이기만 했더니 모델이 목록을 무시하고 자기 키를 지어냈다 —
    #   `current_holdings_in_Cobalt_Blue_Account=15000` · `bonus_value_for_..._card=75` 처럼
    #   **묻지도 않은 값**을 냈고, 요청한 축 이름은 그 런에서 **0회** 나왔다. 즉 추출은
    #   발화하는데 결과가 필터에 닿지 않는 **조용한 사망**이었다(단위검정은 순수함수만 봤다).
    #   ⇒ ⒜ 목록을 발췌 **앞**에 놓고(A2 `{items}`) ⒝ 채울 골격을 그대로 보여 주고
    #     ⒞ 엔진이 **요청한 키만 받는다**(집합 원소 검사·의미 판단 0·[[22]]).
    names = [k for k, _v in wanted]
    items = ("\n".join("  %s  — %s" % (k, v) for k, v in wanted)
             + "\n\nFill in this exact skeleton, dropping any item the conversation does not "
               "state:\n{"
             + ", ".join('"%s": {"value": <integer>, "quote": "<exact sentence>"}' % k
                         for k in names)
             + "}\nDo not invent other keys. Do not report anything that is not in this list.")
    out = _formalize_pairs(agent, la, UserMessage, texts, spec,
                           key="case_facts_prompt", field="value",
                           memo_attr="_t2_case_facts",
                           call_name="case_facts_formalize", extra=items)
    keep = {k: v for k, v in (out or {}).items() if k in set(names)}
    if out and not keep:
        print("[T2_LEDGER] case_facts: 요청 밖 키만 왔다 %s — 전부 버린다"
              % sorted(out)[:4], file=sys.stderr, flush=True)
    return keep


def formalize_objective(agent, la, UserMessage, texts, spec):
    """손님이 **무엇을 최대화해 달라는지** 한 구절로 — 인용과 함께.

    왜 이것만 형식화하나 (x156 실측): 5/5 를 낸 깨끗한 문맥은 *표 + 정제된 사실 + 고정 질문* 이고,
    **손님 발화를 그대로 실은 구성은 0/5** 였다(`only user`). 그러니 대화를 통째로 나르면 안 된다.
    그렇다고 목적을 우리가 정할 수도 없다 — 같은 계열 안에서도 갈린다(`내가 받는 보너스 최대` vs
    `둘이 받는 합산 최대`). ⇒ **목적 한 구절만** LLM 이 뽑고 엔진은 인용 실재만 본다([[22]]).

    ⚠못 뽑으면 침묵한다 — 목적을 모르면 재도출 질문 자체가 성립하지 않는다.
    """
    tpl = (spec or {}).get("objective_prompt")
    if not (tpl and agent is not None and la is not None and texts):
        return None
    memo = getattr(agent, "_t2_objective", None)
    if memo is not None:
        return memo
    hay = " ".join("\n".join(str(t) for t in texts).split())
    sel, used = [], 0
    for t in reversed(list(texts)):
        s = str(t)[:3000]
        if used + len(s) > 90000:
            break
        sel.append(s)
        used += len(s)
    sel.reverse()
    try:
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        try:
            um = UserMessage(role="user", content=tpl.format(text="\n---\n".join(sel)))
        except TypeError:
            um = UserMessage(content=tpl.format(text="\n---\n".join(sel)))
        raw = getattr(la.generate(model=agent.llm, tools=None, messages=[um],
                                  call_name="objective_formalize", **kw), "content", None) or ""
    except Exception as e:
        print("[T2_LEDGER] objective 실패: %r" % (e,), file=sys.stderr, flush=True)
        return None
    m = re.search(r"\{.*\}", str(raw), re.S)
    if not m:
        return None
    try:
        got = json.loads(m.group(0))
    except Exception:
        return None
    obj = " ".join(str(got.get("objective") or "").split())
    quote = " ".join(str(got.get("quote") or "").split())
    ok = bool(obj) and len(quote) >= 12 and quote in hay
    print("[T2_LEDGER] objective=%r 인용실재=%s" % (obj[:70], ok),
          file=sys.stderr, flush=True)
    if not ok:
        return None
    try:
        agent._t2_objective = obj
    except Exception:
        pass
    return obj


def rederive_choice(agent, la, UserMessage, spec, table, facts, asked, allowed):
    """**같은 모델에게 깨끗한 문맥으로 다시 묻는다** — 그리고 그 답만 돌려준다.

    ★근거 (2026-08-09·x154·유료 0·로컬 32B·gold 무참조): task_099 는 라이브 **12 sim 전수 실패**
      이고 제출물 **10/12 가 손님이 이미 보유한 계좌**였다. 절제가 길을 하나씩 닫았다 —
        · 같은 표를 궤적 안에 더 실어도                     **0/5**  (현행 라이브)
        · *"보유한 것이 정답이라는 뜻은 아니다"* 반증 문구    **0/5**  (앵커는 떼지만 argmax 를 틀린다)
        · **깨끗한 문맥**(표 + 손님 사실 + 질문)만 주면        **5/5**
        · 그 답을 **궤적 안으로 되돌려 넣으면**               **5/5**  ← 표가 있든 없든 같다
      ⇒ 결손은 정보가 아니라 **부하**다([[45]]). 처방은 더 보여주는 것이 아니라 **판단하는 자리를
        바꾸는 것**이다.

    분담 ([[52]]·[[10]]·[[05]] Q2): **고르는 것은 두 번 다 모델**이다. 엔진이 하는 일은 셋뿐 —
      ⒜ 검증된 재료로 문맥을 조립하고(정책 상수 = A3 인용 · 손님 사실 = 인용 실재 확인된 형식화)
      ⒝ 손님의 질문은 **그의 말 그대로** 싣고(우리가 해석하지 않는다)
      ⒞ 돌아온 답이 **목록의 원소인지**만 본다(집합 검사·의미 판단 0·[[22]]).
    ⚠원소가 아니면 **침묵**한다. 목록 밖의 이름을 우리가 권위 있게 옮기면 그것이 날조 통로가 된다
      (C107 실물: 날조 인자를 막는 게이트가 날조 *도구명*을 제조했다 — 게이트도 부작용이 있다).
    ⚠이 문장은 *"우리가 골랐다"* 가 아니라 *"별도 분석이 이것을 고른다"* 로 나간다 — 실제로 그렇다.
    """
    tpl = (spec or {}).get("rederive_prompt")
    # ⚠`asked`(목적 구절)가 **비어 있는 것이 측정된 구성**이다 — x158 n=10: 099 는 목적을 실으면
    #   0/10(전부 카드), 빼면 10/10. 가드가 `asked` 를 요구하면 그 구성이 라이브에서 **조용히
    #   조기 반환**된다. 실제로 유료 런 `bank_rederive_20260809k`(6 sim)이 발화 0 으로 돌았다
    #   (2026-08-09·로그 `[T2_REDERIVE]` 0회). 필수 재료는 표와 목록뿐이다.
    asked = asked or ""
    if not (tpl and agent is not None and la is not None and table and allowed):
        return None
    memo = dict(getattr(agent, "_t2_rederive", None) or {})
    key = hashlib.sha1(("\n".join((table, facts or "", asked))).encode("utf-8")).hexdigest()[:16]
    if key in memo:                       # 같은 재료면 다시 묻지 않는다(턴마다 호출되는 자리다)
        return memo[key]
    prompt = tpl.format(table=table, facts=facts or "", asked=asked)
    try:
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="rederive_choice", **kw)
        raw = " ".join(str(getattr(sub, "content", None) or "").split())
    except Exception as e:
        print("[T2_REDERIVE] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return None
    # 집합 검사만 한다 — 가장 긴 일치를 고른다(부분 문자열이 다른 이름에 먹히지 않게).
    hit = sorted((a for a in allowed if a and a.lower() in raw.lower()), key=len, reverse=True)
    out = hit[0] if hit else None
    print("[T2_REDERIVE] raw=%r → %s" % (raw[:80], out or "목록 밖 = 침묵"),
          file=sys.stderr, flush=True)
    memo[key] = out
    try:
        agent._t2_rederive = memo
    except Exception:
        pass
    return out


def formalize_objective_axis(agent, la, UserMessage, spec, texts, axes):
    """손님이 **무엇을 최대화해 달라는지**를 A2 **축 이름 하나**로 — 닫힌 집합이라 엔진이 검증한다.

    왜 형식화인가 ([[23]]): 순위 축을 A2 에 **정적으로 선언하면** 그 태스크의 목적을 우리가
    구워 넣는 것이고, 출처가 정책도 환경도 아니게 된다. 목적은 **손님의 말**에 있으므로
    해석은 LLM 이 하고([[52]]), 엔진은 결과가 **A2 축 집합의 원소인지**만 본다([[22]] 닫힌 술어).

    ★필요성 (x187 실측): 축을 이름으로 지목해 주면 14B/task_100 이 `Q2` **8/8** 인데, 스스로
      해석하게 두면 `Q3` **0/8** 이다(32B 는 둘이 같다). ⇒ **축 해석은 스케일이 사는 결손**이고
      이 형식화가 그 자리를 메운다.
    ⚠이 값은 **재도출 문맥에 싣지 않는다** — 목적 구절을 실으면 해로웠다(x158: 099 10/10 → 0/10).
      쓰이는 곳은 엔진의 **재계산 축**뿐이다(`mismatch_value`).
    ⚠못 고르면 **None** — 그러면 D1c 를 하지 않는다. 모르는 것을 기준으로 재질의하지 않는다.
    """
    tpl = (spec or {}).get("objective_axis_prompt")
    if not (tpl and agent is not None and la is not None and texts and axes):
        return None
    memo = getattr(agent, "_t2_obj_axis", None)
    if memo is not None:
        return memo or None
    names = list(axes)
    listing = "\n".join("  %s — %s" % (k, axes[k]) for k in names)
    # ★2026-08-09 라이브 부검(런 r·C374): 구판은 전체를 이어 붙인 뒤 **꼬리 6000자**만 봤다.
    #   그 창에는 KB 문서만 들어온다 — 목적을 말하는 문장은 손님의 **첫 발화**에 있고, 결정점은
    #   턴 24~26 이다. 실측: 두 sim 다 손님의 목적 문장이 창 밖(099 총 28,523자·100 총 15,044자)
    #   이라 서브가 `NONE` 을 냈고, 그 결과 **순위(`runners`)가 빈 채로 나가고 D1c 가 아예 안 돌았다**
    #   (=x191 의 `B_rank`·x192 의 재질의가 라이브에서 죽어 있었다). 같은 형태의 실패를 이미 두 번
    #   고쳤다 — `now_prompt`(머리+꼬리·2026-08-07) · `_formalize_pairs`(항목별 절단+총예산·2026-08-08).
    #   ⇒ 세 번째로 같은 것을 고치지 않도록 **선택기를 한 자리로 합친다**([[55]] 배관 먼저).
    hay = "\n---\n".join(_excerpt(texts))
    try:
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        try:
            um = UserMessage(role="user", content=tpl.format(axes=listing, text=hay))
        except TypeError:
            um = UserMessage(content=tpl.format(axes=listing, text=hay))
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="objective_axis_formalize", **kw)
        raw = " ".join(str(getattr(sub, "content", None) or "").split())
    except Exception as e:
        print("[T2_OBJ_AXIS] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return None
    hit = sorted((a for a in names if a and a.lower() in raw.lower()), key=len, reverse=True)
    out = hit[0] if hit else None
    print("[T2_OBJ_AXIS] raw=%r → %s" % (raw[:60], out or "축 집합 밖 = 침묵"),
          file=sys.stderr, flush=True)
    try:
        agent._t2_obj_axis = out or ""
    except Exception:
        pass
    return out


def subject_kinds(rows, field):
    """A3 가 **행에 적어 둔 종류**를 주어별로 모은다 — 엔진은 그 값의 뜻을 모른다.

    종류는 A3 행이 이미 인용하고 있는 **출처 문서군**에서 빌드 시점에 유도된다(`x203`).
    엔진이 문서 id 를 뜯으면 그것이 도메인 패턴매칭이므로([[59]]) 여기서는 **선언된 필드를
    읽기만** 한다.

    ⚠한 주어가 여러 종류에 걸치면 **뺀다** — 강제하지 않는다(§4b). 빠진 주어는 종류 필터에
      걸리지 않으므로 표에 그대로 남는다(모름 ≠ 탈락·[[25]]).
    """
    if not (rows and field):
        return {}
    seen = {}
    for r in rows:
        s, k = r.get("subject"), r.get(field)
        if not (s and k):
            continue
        seen.setdefault(str(s).strip(), set()).add(str(k).strip())
    return dict((s, sorted(v)[0]) for s, v in seen.items() if len(v) == 1)


def formalize_kind(agent, la, UserMessage, spec, texts, kinds):
    """손님이 **어떤 종류의 상품**을 말하는지 — A3 가 들고 있는 종류 이름 하나로.

    ## 왜 이것이 필요한가 (x201·격리·n=8·32B)

    098 의 통과 표에는 개인 체킹 5 + 사업자 카드 6 + 카드 3 이 함께 실린다. 손님은 친구가
    **계좌를 여는** 이야기를 하는데 모델은 카드의 단일 최대 수(referrer 300)를 집는다:

        A_iso  (현행 표)                     0/8   ← `Business Platinum Rewards Card`
        E_hint (표 + 한 줄로 무엇을 묻는지)   **0/8**   ← 전달로는 안 된다
        F_kind (종류로 거른 표)               8/8
        G_llm  (LLM 이 종류 선택 → 엔진 필터) **8/8**  (종류 선택 8/8 정확)

    ⛔0 ②를 지켰다 — **전달 팔을 먼저 재서 실패**했기에 필터가 정당하다([[62]]).

    ## 경계

    종류의 *해석* 은 LLM 이 한다(손님의 말 → 종류 이름). 엔진은 그 답이 **A3 종류 집합의
    원소인지**만 보고, 그 종류가 아닌 행을 뺀다 — 답을 고르지 않는다([[22]]·[[52]]).
    거른 뒤에도 다섯 행이 남고 그중 무엇을 고를지는 여전히 모델 몫이다.

    ⚠못 고르면 **None** → 아무것도 거르지 않는다(종전 거동). 모르는 것으로 행을 빼지 않는다.
    """
    tpl = (spec or {}).get("kind_prompt")
    if not (tpl and agent is not None and la is not None and texts and kinds):
        return None
    memo = getattr(agent, "_t2_kind", None)
    if memo is not None:
        return memo or None
    names = sorted(set(kinds))
    listing = "\n".join("  %s" % k for k in names)
    hay = "\n---\n".join(_excerpt(texts))
    try:
        kw = dict((k, v) for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
                  if "tool" not in k)
        try:
            um = UserMessage(role="user", content=tpl.format(kinds=listing, text=hay))
        except TypeError:
            um = UserMessage(content=tpl.format(kinds=listing, text=hay))
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="kind_formalize", **kw)
        raw = " ".join(str(getattr(sub, "content", None) or "").split())
    except Exception as e:
        print("[T2_KIND] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        return None
    hit = sorted((k for k in names if k and k.lower() in raw.lower()), key=len, reverse=True)
    out = hit[0] if hit else None
    print("[T2_KIND] raw=%r → %s" % (raw[:60], out or "종류 집합 밖 = 안 거른다"),
          file=sys.stderr, flush=True)
    try:
        agent._t2_kind = out or ""
    except Exception:
        pass
    return out


def restrict_to_kind(axis_maps, kinds_by_subject, kind):
    """그 종류가 **아닌 것으로 확인된** 주어만 뺀다 — 종류를 모르는 주어는 남긴다([[25]])."""
    if not (kind and kinds_by_subject):
        return axis_maps, []
    drop = set(s for s, k in kinds_by_subject.items() if k != kind)
    if not drop:
        return axis_maps, []
    out = dict((ax, dict((s, v) for s, v in (m or {}).items() if s not in drop))
               for ax, m in (axis_maps or {}).items())
    return out, sorted(drop)


def _rank_by(rows, axis_map, exclude=()):
    """통과 집합(**구조체**)을 그 축의 값으로 내림차순 — `[(주어, 값)]`. 값이 없는 주어는 뺀다.

    ⚠입력은 `eligible_text(..., as_rows=True)` 가 준 `[(주어, bits)]` 다. **문자열을 되파싱하지
      않는다** — 표를 만든 것도 우리이므로 애초에 구조체로 들고 다닌다([[59]] 규율의 정신).
    """
    out = []
    for s, _bits in (rows or ()):
        if s in exclude:
            continue
        v = (axis_map or {}).get(s)
        if v is None:                        # 그 축 값이 없는 주어는 순위에서 뺀다
            continue
        try:
            n = _num(v[0] if isinstance(v, (list, tuple)) else v)
        except Exception:                    # 수로 못 읽는 값도 순위에서 뺀다(모름≠0)
            continue
        if n is not None:
            out.append((s, n))
    return sorted(out, key=lambda kv: (-kv[1], kv[0]))


def decided_text(spec, choice, rows, operands, axis, axis_map, runners_n=3):
    """★결정 블록 = **지목 + 근거 + 순위** (규격서 `ANCHOR_SLOT_SPEC_2026_08_09` §5b·`B_rank`).

    ★왜 이 모양인가 (x190·x191 실측·2모델×2태스크·n=8): 같은 답인데 **근거를 붙여야 채택된다**.
      · `B_min`  지목 한 줄(제3자 보고체)                      → **전 셀 0/8**
      · `B_ops`  + 피연산자 + 선택된 행의 상수                 → 32B 8/8 · 14B/100 3/8
      · `B_rank` + **상위 N위 순위**                            → **2모델×2태스크 4/4셀 8/8**·날조 0
      그리고 후속 질문(*"두 번째로 좋은 것은"*)도 순위를 실어야 닫힌다(0/8 → 8/8).

    ⚠**근거는 신뢰를 사지 검증을 사지 않는다**: 근거를 붙이면 채택률이 오르지만 읽는 쪽이
      검산해서 채택하는 것이 아니다 — x185 에서 **틀린** 지목에 올바른 숫자를 붙이자 14B 가
      4/8 → 0/8 로 **더 확실히** 틀렸다. 그러므로 이 블록을 쓰는 배치는 **엔진 재계산
      검증**(`reask_pick`)을 함께 켜야 한다. 블록만 세게 만드는 것은 위험을 키운다.

    분담: 고르는 것은 여전히 모델(`rederive_choice`)이고, 여기서 엔진이 하는 일은 **조립뿐**이다
    ([[05]] Q2). 표는 우리가 만든 것이라 줄을 꺼내는 것은 자기 형식 파싱이다([[59]] 무관).
    ⚠축이 없으면 순위 없이 나간다 — 모르는 것을 지어내지 않는다.
    """
    tpl = (spec or {}).get("decided_text")
    if not (tpl and choice):
        return ""
    row = next((_row_line(s, b).strip() for s, b in (rows or ()) if s == choice), "")
    runners = ""
    if axis and axis_map:
        rest = _rank_by(rows, axis_map, exclude=(choice,))[:max(0, int(runners_n))]
        if rest:
            runners = "; ".join("%s (%s=%s)" % (s, axis, _num(v)) for s, v in rest)
    return tpl.format(choice=choice, operands=(operands or "").strip(),
                      row=row, runners=runners)


def mismatch_value(rows, axis_map, choice):
    """★D1c — 엔진이 **재계산해 불일치만 탐지**한다. 답은 돌려주지 않는다.

    반환: `None`(일치·판정 불가) 또는 `(고른 값, 최댓값)`. **이름은 반환하지 않는다** —
    이름을 돌려주면 그것이 지목이 되어 [[05]] Q2 *"고르는 것은 모델"* 을 넘는다([[52]]:
    집행 규칙이 아니라 **질문 트리거**).

    ★근거 (x192·2모델×2태스크×2정렬·n=8): 격리 서브가 틀리던 **3셀 전부 `no_reask` 0/8**.
      **무내용 재시도는 3셀 전부 0/8**([[57]] 부정 통제) — 효과는 재시도가 아니라 **정보**다.
      **값만 되돌리면 8/8**이고 이름을 말한 상한(`reask_name`)과 **동일**하다.
    ⚠정직한 단서: 값이 유일하면 사실상 지목과 같다. **형식상 보존이지 실질 보존인지는
      동점 사례로 갈라야 한다**(미측정·규격서 §6).
    """
    rank = _rank_by(rows, axis_map)
    if not rank or not choice:
        return None
    got = dict(rank).get(choice)
    if got is None:
        return None
    best = rank[0][1]
    return None if got >= best else (got, best)


def _excerpt(texts, per=3000, budget=90000):
    """형식화에 줄 발췌를 **위치로만** 고른다 — 모든 형식화가 공유하는 한 자리.

    ★발췌는 **꼬리가 아니다** (2026-08-08·lim_n 위치 실측). 상한을 말하는 텍스트가 인덱스
      2·15·17·19처럼 흩어져 있는데 구판은 `texts[-12:]`만 봤다 — 결정점으로 옮겨도 꼬리이면
      앞쪽(2번)을 잃고, 대화가 길어지면 뒤쪽도 같은 방식으로 밀려난다.
    ★같은 형태의 실패를 **세 번** 겪었다: `now_prompt`(오늘 날짜가 대화 머리에 있었다·2026-08-07)
      · `_formalize_pairs`(2026-08-08) · `formalize_objective_axis`(라이브 런 r·C374 — 목적을
      말하는 손님 첫 발화가 꼬리 6000자 밖이라 서브가 `NONE` 을 냈고 순위·D1c 가 통째로 죽었다).
      ⇒ 선택기를 **함수 하나**로 합친다. 다음 형식화는 이 자리를 쓰면 같은 실패를 안 겪는다.
    ⚠고르는 기준은 **위치·길이뿐**이다. 어느 텍스트가 무엇을 담았는지 판정하는 것은 모델 몫이고,
      엔진이 내용으로 고르면 [[59]] 위반이다.
    """
    sel, used = [], 0
    for t in reversed(list(texts or ())):  # 최신부터 담고, 예산이 남으면 앞쪽도 들어온다
        s = str(t)[:per]
        if used + len(s) > budget:
            break                          # 잘려 나가는 쪽은 **가장 오래된 것**이어야 한다
        sel.append(s)
        used += len(s)
    sel.reverse()                          # 다시 시간 순서로(읽는 쪽이 대화 순서를 본다)
    return sel


def _formalize_pairs(agent, la, UserMessage, texts, spec, key, field, memo_attr, call_name,
                     extra=""):
    """`{그룹: (정수, 인용문)}` 형태를 모델에게 받는 공용 절차 — 인용 실재만 엔진이 확인한다."""
    tpl = (spec or {}).get(key)
    if not (tpl and agent is not None and la is not None and texts):
        return {}
    memo = getattr(agent, memo_attr, None)
    if memo is not None:
        return memo
    # ★자리는 **A2가 정한다**: 템플릿에 `{items}` 가 있으면 거기 넣는다(발췌 앞에 둘 수 있다).
    #   없으면 구판대로 뒤에 잇는다 — 기존 선언 3종의 거동은 그대로다.
    _body = "\n---\n".join(_excerpt(texts))
    try:
        prompt = tpl.format(text=_body, items=extra)
    except (KeyError, IndexError):
        prompt = tpl.format(text=_body) + (extra or "")
    try:
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name=call_name, **kw)
        raw = getattr(sub, "content", None) or ""
    except Exception:
        return {}
    hay = " ".join("\n".join(str(t) for t in texts).split())
    # ⚠거동 델타 **한 건**(의도·값 아님): 구판은 응답에 `{...}`가 아예 없으면 **인쇄 없이** 돌아갔다.
    #   이제는 `model gave 0` 줄이 찍힌다 — §5의 규율("성공만 인쇄하면 못 찾았다와 안 돌았다가
    #   같아 보인다")이 요구하는 방향이고, **반환 값은 구판과 동일**(빈 dict)이다.
    out, rejected, _given = parse_pairs(raw, field, hay)
    # ★빈손도 찍는다 (2026-08-08·lim_n 라이브). 성공만 찍으면 "못 찾았다"와 "안 돌았다"가
    #   구분되지 않는다 — 오늘 `seen=N`으로 배운 것과 같은 형태다.
    # 이름까지 찍는다 — 개수만으로는 *"소진된 유형을 못 뽑은 것"* 과 *"뽑았는데 소진이 아닌 것"* 이
    # 구분되지 않는다(dp_p에서 추출 2회·발화 0회가 정확히 그 모호함이었다).
    print("[T2_LEDGER] %s: model gave %d, accepted %d, rejected %d · %s"
          % (field, _given, len(out), rejected,
             ", ".join("%s=%s" % (k, v[0]) for k, v in sorted(out.items())) or "(none)"),
          file=sys.stderr, flush=True)
    # ★★빈손은 **기억하지 않는다**. 구판은 `{}`도 메모해 그 sim 내내 재시도가 없었다.
    #   원장 read는 대화 앞쪽(턴 10~12)에서 일어나는데 상품 문서는 그보다 **뒤에** 회수된다 —
    #   즉 상한이 도착하기 전에 물어보고 그 빈 답을 영구 고정한 셈이었다. 비었으면 다음 read에서
    #   다시 묻는다(문서가 그 사이에 들어왔을 수 있다).
    if out:
        try:
            setattr(agent, memo_attr, out)
        except Exception:
            pass
    return out


def ineligible_text(days, thresholds, spec):
    """경과일과 인용된 문턱을 맞대어 **아직 못 되는 그룹**만 말한다. 추천하지 않는다."""
    tpl = (spec or {}).get("ineligible_text")
    if not (tpl and thresholds and days is not None):
        return ""
    blocked, ok = [], []
    for g, (need, _q) in sorted(thresholds.items()):
        (blocked if int(days) < need else ok).append("%s needs %d" % (g, need))
    if not blocked:
        return ""
    return tpl.format(days=int(days), blocked="; ".join(blocked), ok="; ".join(ok) or "(none)")


def formalize_subject_align(agent, la, UserMessage, spec, groups, subjects):
    """원장 그룹 이름이 **A3 주어 중 어느 것을 가리키는가** — 해석은 LLM, 검사는 엔진.

    ## 왜 이것이 필요한가 (2026-08-09·C376·전수 실측)

    원장(도구 출력)은 `Navy Blue Account` 로 말하고 A3 주어는 `Navy Blue` 다. 두 소비자가
    **정확 일치**로 맞대므로 그 접미사 하나가 둘 다 무력화한다 — 라이브 전수 계량:

      · `unmatched_text`  발화 **77회 · A3 주어와 정확일치 0개**. 지목된 7 그룹 중 **5개는
        접미사만 떼면 A3 에 있고**, 나머지 2개도 접두사(`Business `) 차이다. 즉 *"허용치가
        원장에 없다"* 는 함의는 **관측 77회 전부에서 거짓**이었고, 매번 조사 지시를 동반했다.
      · `exhausted_text` 발화 **0회**. `tally.get(g, 0)` 이 언제나 0 이라 `used >= lim` 이
        성립할 수 없다 ⇒ **엔진이 한도 소진을 말한 적이 한 번도 없다.** 099 는 바로 그
        판정(`Hunter Green` 사용 9회 대 연간 한도)에 걸려 있는데 우리는 자료를 쥐고 침묵했다.

    ⇒ [[25]]: 우리 출력이 유일한 근거원인데 오도했다. 고쳐야 한다.

    ## 왜 정규화가 아니라 형식화인가 ([[59]]·[[22]]·C316)

    엔진이 `" Account"` 를 떼면 그것이 곧 도메인 텍스트 패턴매칭이다 — 금지선이다. 그리고
    두 표기가 같은 것을 가리키는지는 **열린 술어**라 엔진이 판정할 자격이 없다. 그래서 분담은
    다른 형식화와 같다: **해석 = LLM**(어느 주어인가), **검사 = 엔진**(그 답이 A3 주어 집합의
    원소인가). 답이 원소가 아니면 그 그룹은 **정렬되지 않은 채로 남는다** — 그러면 종전대로
    `unmatched_text` 가 그것만 말한다. 모르는 것을 맞다고 바꾸지 않는다.

    반환: `{원장 그룹 이름: A3 주어}` (정렬된 것만). 못 고른 그룹은 **키가 없다**.
    """
    tpl = (spec or {}).get("subject_align_prompt")
    if not (tpl and agent is not None and la is not None and groups and subjects):
        return {}
    subs = sorted(str(s) for s in subjects if s)
    gs = sorted(str(g) for g in groups if g)
    # ★기억은 **내용에 묶는다** (2026-08-09 자기감사·`rederive_choice` 와 같은 규율). 결정점은
    #   원장이 자라는 동안 여러 턴 호출된다 — 그룹 하나만 보이던 이른 턴의 답을 sim 내내
    #   재사용하면 **뒤에 들어온 그룹은 영영 정렬되지 않는다**(조용한 사망·[[24]] 계보).
    #   재료가 그대로면 다시 묻지 않고, 자라면 다시 묻는다.
    memo = dict(getattr(agent, "_t2_subj_align", None) or {})
    key = hashlib.sha1(("\n".join(subs + ["\x00"] + gs)).encode("utf-8")).hexdigest()[:16]
    if key in memo:
        return memo[key]
    prompt = tpl.format(subjects="\n".join("  " + s for s in subs),
                        groups="\n".join("  " + g for g in gs))
    out, called = {}, False
    try:
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="subject_align_formalize", **kw)
        called = True
        raw = str(getattr(sub, "content", None) or "")
        m = re.search(r"\{.*\}", raw, re.S)
        got = json.loads(m.group(0)) if m else {}
        # ★엔진의 몫은 여기까지다 — **양쪽 다 집합 원소인지**만 본다(의미 판단 0).
        subset, gset = set(subs), set(gs)
        for g, s in (got.items() if isinstance(got, dict) else ()):
            if g in gset and isinstance(s, str) and s in subset:
                out[g] = s
    except Exception as e:
        print("[T2_SUBJ_ALIGN] 호출 실패(무발화): %r" % (e,), file=sys.stderr, flush=True)
        out, called = {}, False
    print("[T2_SUBJ_ALIGN] %d/%d 그룹 정렬 %s"
          % (len(out), len(gs), sorted(out.items())[:4]), file=sys.stderr, flush=True)
    # ★**실패는 기억하지 않는다**: 예외로 죽은 호출을 캐시하면 일시적 실패 하나가 그 sim 전체의
    #   영구 침묵이 된다. 모델이 정직하게 `{}` 를 낸 것(=called)은 답이므로 기억한다.
    if called:
        memo[key] = out
        try:
            agent._t2_subj_align = memo
        except Exception:
            pass
    return out


def align_tally(tally, align):
    """정렬된 그룹은 **A3 주어 이름으로** 옮겨 담고, 못 고른 그룹은 원래 이름으로 남긴다.

    엔진은 이름을 만들지 않는다 — `align` 의 값은 이미 A3 주어 집합의 원소로 검사된 것뿐이다.
    같은 주어로 두 그룹이 정렬되면 **합산**한다(원장 표기가 갈렸을 뿐 같은 상품이므로).
    """
    if not tally:
        return {}, {}
    aligned, left = {}, {}
    for g, n in tally.items():
        s = (align or {}).get(g)
        if s:
            aligned[s] = aligned.get(s, 0) + int(n)
        else:
            left[g] = n
    return aligned, left


def exhausted_text(tally, limits, spec):
    """누계와 상한을 맞대어 **남은 자리가 없는 그룹**만 말한다. 산수뿐이고 추천은 하지 않는다."""
    tpl = (spec or {}).get("exhausted_text")
    if not (tpl and limits):
        return ""
    gone, left = [], []
    for g, (lim, _q) in sorted(limits.items()):
        used = int(tally.get(g, 0))
        (gone if used >= lim else left).append("%s %d/%d" % (g, used, lim))
    if not gone:
        return ""
    return tpl.format(exhausted="; ".join(gone), remaining="; ".join(left) or "(none)")


def _num(v):
    """`(값, 인용)` 도 값 그대로도 받는다 — A3 조회와 원장 산수가 같은 자리에서 만난다."""
    return int(v[0] if isinstance(v, (tuple, list)) else v)


def _row_line(subject, bits):
    """통과 집합 한 줄의 **유일한** 생성 지점 — 읽는 쪽은 이 문자열을 되파싱하지 않는다."""
    return "  %s: %s" % (subject, ", ".join(bits))


def _live_criteria(crit, days, tally, axis_maps, stated):
    """지금 **실제로 걸 수 있는** 기준만 추린다 — 피연산자가 있는 축만.

    `eligible_text` 에서 **축자 그대로 추출**한 것이다(거동 불변). `verified_subjects` 가 같은
    판정을 두 번 쓰지 않게 하려고 함수로 뺐다 — 복제하면 둘이 갈리고, 갈린 목록이 오늘
    C377 결함의 형태였다.
    """
    live = []
    for c in crit or ():
        ax, src, rel = c.get("axis"), c.get("operand"), c.get("compare")
        m = (axis_maps or {}).get(ax) or {}
        if not (ax and rel and m):
            continue
        if src == "tally":
            # ⚠원장을 **안 읽었으면** 거르지 않는다. 빈 dict 를 0 으로 세면 *"올해 아무것도
            #   안 썼다"* 는 주장이 되는데 그건 우리가 확인한 사실이 아니다([[25]]).
            if tally is not None:
                live.append((m, rel, "tally", None))
        elif src == "days":
            if days is not None:
                live.append((m, rel, "scalar", int(days)))
        elif src == "stated":
            v = (stated or {}).get(ax)
            if v is not None:
                live.append((m, rel, "scalar", _num(v)))
    return live


def eligible_text(days, tally, axis_maps, spec, stated=None, as_rows=False):
    """자격을 **엔진이 걸러** 통과한 후보만, 그 상품에 기록된 정책 상수와 함께 말한다. 산수뿐이다.

    ★근거 (2026-08-08·C337·x150 절제 실측): 모델은 argmax 를 완벽히 하고 **자격 필터를 못 한다**.
      같은 표를 주고 물으면 0/5(전부 자격 미달 상품을 고른다)인데, **자격 미달 행을 미리 뺀 표**를
      주면 **5/5** 다. 축 이름을 문장으로 풀어써도(0/5), *"먼저 자격을 따지고 그 다음 최고를
      고르라"* 고 명시적으로 분해해 줘도(0/5) 안 움직인다 — 등대 §1.4 F2b(계산형 기준·thinking
      무효)의 독립 재현이고, 거기 처방이 *"형식화(LLM)→결정론 실행(filter)"* 이다.
    ⇒ 전체 표를 표면화하는 대신 **통과 집합**을 준다. 고르는 것은 여전히 모델이다([[05]] Q2).

    ★남은 축을 **같이 싣는다** (2026-08-08 자기정정·§ 아래 실측): 통과 집합을 *이름과 보너스만*
      으로 줄이면 099가 닫히지 않는다 — 그 sim 은 관계기간이 2년이라 문턱이 아무도 못 거르고,
      정답을 가르는 것은 예치 하한·회사 연령이다(A3 실측: `True Blue` 350 이 `World Blue` 300 보다
      크다 ⇒ 보너스만 주면 argmax 가 오히려 오답을 가리킨다). x149 A_clean(5/5)·x150 P2(5/5)가
      쓴 표에는 그 축들이 **들어 있었다** ⇒ 재현하려면 같이 실어야 한다. 축 목록·문턱 축 이름은
      전부 A2 선언이고 엔진에 도메인 어휘는 없다.

    ★기준은 **A2 `eligible.criteria` 선언**이고 엔진에 축 이름이 없다. 피연산자 출처는 셋 —
      `days`(계좌 원장) · `tally`(추천 원장) · `stated`(손님이 대화에서 말한 수·LLM 형식화).
      2026-08-09 자기정정: 처음엔 원장에서 온 둘만 걸고 나머지는 **표시만** 했는데, 그것이
      task_099 를 못 닫은 이유였다 — 그 sim 은 관계기간 2년이라 문턱이 아무도 못 거르고,
      가르는 축은 예치 하한이다(`Beige` 500 은 100000 을 요구하고 손님은 30000). 거르면
      최고액이 `World Blue` 300 = gold 가 된다. *"자격 산수로 거른다"* 는 원래 계약이고,
      피연산자가 DB 밖에 있다는 것은 거르지 않을 이유가 아니라 **형식화할 이유**다([[52]]).
    문서에 그 기준이 없는 주어는 **거르지 않는다** — 모르는 것을 탈락으로 바꾸지 않는다.
    피연산자를 못 구한 축도 **거르지 않는다**. 남은 축은 값 그대로 실어 모델이 보게 한다.

    ⚠**닫을 수 없으면 침묵한다**: 걸 수 있는 기준이 하나도 없으면 이 문장은 나가지 않는다.
      문턱을 못 걸고 만든 '통과 집합'은 통과 집합이 아니라 **전체 표에 통과 도장을 찍은 것**이고,
      실측상 그 상태의 최고액은 정확히 오답(`World Blue Balance` 300)이다.
    ⚠정렬은 **이름순**이다. 보너스순으로 내리면 첫 줄이 곧 답이 되어 우리가 argmax 까지 해 버린다
      ([[05]] Q2: 고르는 것은 모델). 측정된 조건(P2 표)도 이름순이었다.
    """
    tpl = (spec or {}).get("eligible_text")
    cfg = (spec or {}).get("eligible") or {}
    show = list(cfg.get("show_axes") or ())
    crit = list(cfg.get("criteria") or ())
    if not (tpl and show and crit):
        return ""
    # 피연산자: 원장에서 온 것(경과일·누계)과 대화에서 온 것(`stated`)을 한 자리에서 받는다.
    # 축마다 피연산자가 **있을 때만** 거른다 — 없는 축은 거르지 않는다(모름 ≠ 탈락).
    live = _live_criteria(crit, days, tally, axis_maps, stated)
    # ★거를 수 있는 기준이 하나도 없으면 침묵한다 — 통과 도장을 찍는 것이 되기 때문이다.
    if not live:
        return ""
    subs = set()
    for a in show:
        subs |= set((axis_maps or {}).get(a) or {})
    ok = []
    for s in sorted(subs):
        drop = False
        for m, rel, kind, val in live:
            th = m.get(s)
            if th is None:                      # 그 상품엔 그 기준이 문서에 없다 = 안 거른다
                continue
            lhs = int((tally or {}).get(s, 0)) if kind == "tally" else val
            th = _num(th)
            if (rel == "ge" and lhs < th) or (rel == "le" and lhs > th) \
                    or (rel == "lt" and lhs >= th) or (rel == "gt" and lhs <= th):
                drop = True
                break
        if drop:
            continue
        bits = ["%s=%s" % (a, _num((axis_maps or {}).get(a, {})[s]))
                for a in show if s in ((axis_maps or {}).get(a) or {})]
        if bits:
            ok.append((s, bits))
    if not ok:
        return ""
    if as_rows:                               # 통과 집합을 **구조체로** 돌려준다(파싱 금지)
        return ok
    return tpl.format(eligible="\n".join(_row_line(s, bits) for s, bits in ok))


def unmatched_text(tally, limits, spec):
    """원장에 있는데 **상한 행이 없는** 그룹을 이름과 함께 말한다. 집합 뺄셈뿐이다.

    왜 필요한가 (2026-08-08·C327): 구판은 그런 그룹을 **조용히 뺐다**. 실측에서 7건짜리 그룹
    하나가 그렇게 사라졌고, 모델 쪽에서 보면 그 그룹은 *검사를 통과한 것*과 구별되지 않는다 —
    침묵이 "문제 없음"으로 읽힌다. 코퍼스 전수로 확인해 보니 그 이름은 문서에 **다른 표기로만**
    있었다(원장 표기는 상한과 무관한 제목에 1회 등장할 뿐이다).

    ⚠**엔진이 이름을 맞추지 않는다.** 두 표기가 같은 것을 가리키는지는 의미 판단이라 모델 몫이고
      ([[22]]·C316), `_a3_map`이 주어를 정규화하지 않는 이유와 같은 규율이다. 엔진이 할 수 있는
      말은 *우리가 이 그룹을 판정하지 못했다*는 사실뿐이며, 어느 쪽인지(다른 표기인가·아직 회수
      안 된 문서인가)는 문구가 **양쪽을 다 열어 둔 채** 모델에게 넘긴다.
    """
    tpl = (spec or {}).get("unmatched_text")
    if not (tpl and tally and limits):
        return ""
    miss = sorted(g for g in tally if g not in limits)
    if not miss:
        return ""
    return tpl.format(unmatched="; ".join("%s %d" % (g, int(tally[g])) for g in miss))


def _date(s, fmts):
    for f in fmts:
        try:
            return datetime.datetime.strptime(str(s).split()[0], f).date()
        except Exception:
            pass
    return None


def window_and_tally(rows, spec, now=None):
    """(창_잔여, 창_안_건수, 그룹별 건수). 순수 산수 — 문자열 해석 없음.

    날짜 형식은 A2가 선언한 목록으로만 읽는다(값 변환이지 도메인 텍스트 파싱이 아니다).
    기준일을 모르면 창은 None = 미개입.
    """
    gf, df = spec.get("group_field"), spec.get("date_field")
    fmts = spec.get("date_formats") or ["%m/%d/%Y"]
    tally = {}
    for r in rows:
        g = r.get(gf)
        if g:
            tally[g] = tally.get(g, 0) + 1
    ref = _date(now, fmts) if now else None
    if not (ref and df and spec.get("window_days") and spec.get("window_max") is not None):
        return None, None, tally
    days = int(spec["window_days"])
    inwin = 0
    for r in rows:
        d = _date(r.get(df), fmts)
        if d is not None and 0 <= (ref - d).days <= days:
            inwin += 1
    return max(0, int(spec["window_max"]) - inwin), inwin, tally


def window_history(rows, spec):
    """각 기록을 **그 앞의 기록들**과 맞대어 센다 — 만들어질 당시 이미 창 한도였던 것을 말한다.

    ## 왜 (2026-08-09·C379·task_010)

    기존 `window_text` 는 *"지금부터 몇 건 더 가능한가"* 만 말한다 — **앞을 보는 문장**이다.
    그런데 010 의 손님이 묻는 것은 뒤다: *"you're not actually telling me **why** either one
    didn't pay out."* 실측 궤적에서 에이전트는 상태(REJECTED/IN_PROGRESS)까지는 우리 문장으로
    알았는데 **이유를 못 찾아 이관으로 끝냈다**(두 trial 다 제출 0).

    이유는 산수로 나온다. 원장 축자: `10/20` · `10/22` · **`10/25`(REJECTED)** · `11/05`
    (IN_PROGRESS), A2 선언 `window_days=9` · `window_max=2`.
      · `10/25` 앞 9일 안에 `10/20`·`10/22` **둘** ⇒ 만들어질 당시 이미 한도였다.
      · `11/05` 앞 9일 안에 **0건** ⇒ 한도가 아니었다.
    두 기록의 상태가 **둘 다 이 산수와 맞는다.**

    ⚠**인과는 말하지 않는다.** 엔진이 하는 말은 *"만들어질 당시 앞 9일에 N건이 있었다"* 까지고,
      *"그래서 거절됐다"* 는 모델이 문서와 맞대어 판단할 몫이다([[25]]·[[22]]). 우리가 인과를
      단정하면 그것이 곧 날조 통로다.
    반환: 창 한도에 이미 닿아 있던 기록이 없으면 **빈 문자열**(말할 것이 없으면 말하지 않는다).
    """
    tpl = (spec or {}).get("window_history_text")
    df, gf = (spec or {}).get("date_field"), (spec or {}).get("group_field")
    days, mx = (spec or {}).get("window_days"), (spec or {}).get("window_max")
    fmts = (spec or {}).get("date_formats") or ["%m/%d/%Y"]
    if not (tpl and df and rows and days and mx is not None):
        return ""
    dated = []
    for r in rows:
        d = _date(r.get(df), fmts)
        if d is not None:
            dated.append((d, str(r.get(gf) or "").strip()))
    if len(dated) < 2:
        return ""
    dated.sort()
    hit = []
    for i, (d, name) in enumerate(dated):
        prior = sum(1 for d0, _n0 in dated[:i] if 0 <= (d - d0).days <= int(days))
        if prior >= int(mx):
            hit.append("%s (%s): %d before it within %d days" % (name, d, prior, int(days)))
    if not hit:
        return ""
    return tpl.format(crowded="; ".join(hit), days=int(days), max=int(mx))


def status_breakdown(rows, spec):
    """원장 행을 **선언된 상태 필드로** 묶어 센다 — 세기만 하고 뜻은 말하지 않는다.

    ## 왜 (2026-08-09·C378·task_010)

    손님 축자: *"I got four friends to sign up, but I only received bonuses for two."* 원장은
    그 답을 이미 들고 있다 — 4행 중 둘은 완료, 하나는 진행 중, **하나는 거절**이고 gold 는
    그 거절된 행의 상품을 다시 제출하는 것이다. 그런데 A2 `row_keys` 가 `date` 와 그룹 필드
    둘만 선언해서 **엔진은 상태를 아예 읽지 않았다.** 실패 궤적은 이미 완료된 행의 상품을
    제출했다(런 s·t1). 통과 궤적(2026-08-04)은 정확히 이 구분을 말로 한 뒤에 통과했다.

    ⚠**상태 값을 엔진이 알지 못한다.** 어떤 값이 있는지·무엇을 뜻하는지 하나도 모른 채,
      모델이 전사해 온 값으로 묶기만 한다(도메인 어휘 0). *왜 그 상태인가* 는 원장이 아니라
      문서에 있고, 문구가 그것을 명시해 모델을 문서로 보낸다([[22]]·[[25]]).
    ⚠상태가 하나뿐이면 침묵한다 — 나눌 것이 없으면 말할 것도 없다(발화 예산·Δspurious).
    """
    tpl = (spec or {}).get("status_text")
    sf, gf = (spec or {}).get("status_field"), (spec or {}).get("group_field")
    if not (tpl and sf and rows):
        return ""
    by = {}
    for r in rows:
        st = r.get(sf)
        if not st:
            continue
        by.setdefault(str(st), []).append(str(r.get(gf) or "").strip())
    if len(by) < 2:
        return ""
    parts = []
    for st in sorted(by):
        names = [n for n in by[st] if n]
        parts.append("%s %d%s" % (st, len(by[st]), (" — " + ", ".join(sorted(set(names))))
                                  if names else ""))
    return tpl.format(total=len(rows), breakdown="; ".join(parts))


def earliest_age(rows, spec, now=None):
    """(가장 이른 날짜, 오늘까지 경과일). 관계 기간(tenure) 같은 값이 여기서 나온다.

    task_100이 이 형태다: 추천 자격이 **첫 체킹 계좌 개설일로부터 며칠**인가로 갈린다
    (정책 축자: *"The tenure threshold is measured from when you opened your **very first**
    checking account with us"*). 계좌 행은 이미 도구가 돌려주고, 날짜 빼기는 산수다.

    ⚠**문턱은 여기서 말하지 않는다.** 상품별 일수는 원장이 아니라 문서에 있고, 코퍼스가 그것을
    상품마다 다른 문형으로 쓴다(Hunter Green만 *"minimum relationship duration of 60 days"* 로
    축자 확인됨·World Blue는 그 문형이 없다). 문턱을 A2에 박으면 못 찾은 값을 지어내게 된다 —
    상한과 같은 규율로, 문턱은 모델이 **인용과 함께** 가져오고 `[SOURCE]`가 실재성을 검증한다.
    """
    df = spec.get("age_field")
    fmts = spec.get("date_formats") or ["%m/%d/%Y"]
    ref = _date(now, fmts) if now else None
    if not (df and ref):
        return None, None
    ds = [_date(r.get(df), fmts) for r in rows]
    ds = [d for d in ds if d is not None]
    if not ds:
        return None, None
    first = min(ds)
    return first, (ref - first).days


def facts_text(rows, spec, now=None):
    """A2 문구에 값을 채운 블록(없으면 빈 문자열). 엔진은 이름과 수만 채운다([[05]] Q2)."""
    if not rows or not spec.get("text"):
        return ""
    remain, inwin, tally = window_and_tally(rows, spec, now)
    tally_s = " | ".join("%s %d" % (k, v) for k, v in
                         sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))
    win_s = ""
    if remain is not None and spec.get("window_text"):
        win_s = spec["window_text"].format(days=spec.get("window_days"), used=inwin,
                                           max=spec.get("window_max"), remaining=remain)
    age_s = ""
    first, days = earliest_age(rows, spec, now)
    if days is not None and spec.get("age_text"):
        age_s = spec["age_text"].format(since=first, days=days)
    out = spec["text"].format(total=len(rows), tally=tally_s, window=win_s, age=age_s)
    return out if out.startswith("\n") else "\n" + out


if __name__ == "__main__":                       # 자기검정 — 산수만(전사는 모델 몫이라 여기서 못 돈다)
    spec = {"trigger_tool": "T", "row_keys": ["d", "g"], "date_field": "d", "group_field": "g",
            "date_formats": ["%m/%d/%Y"], "window_days": 9, "window_max": 2,
            "window_text": "within {days}d: {used} used, {remaining} of {max} left",
            "text": "[COMPUTED FACTS] rows={total} | {tally} | {window}"}
    rows = [{"d": "11/%02d/2025" % (i + 1), "g": "G%d" % (i % 2)} for i in range(1, 6)]
    # 날짜 11/02~11/06, 기준 11/14 → 경과 12·11·10·9·8일. 창(≤9일)에 드는 것은 11/05·11/06 둘뿐.
    rem, inw, tal = window_and_tally(rows, spec, now="11/14/2025")
    assert (inw, rem) == (2, 0), (inw, rem)      # 경계(정확히 9일)는 **포함**
    assert tal == {"G1": 3, "G0": 2}, tal
    assert window_and_tally(rows, spec, now=None)[0] is None      # 기준일 모르면 미개입

    # ── 파서 3종 (`shape`) — **거절 경로까지** 본다. 채택만 검정하면 "안 걸러졌다"를 못 본다 ──
    assert parse_rows('noise [{"d":"1","g":"A"},{"d":"2"}] tail', ["d", "g"]) == [{"d": "1", "g": "A"}]
    assert parse_rows("설명만 있고 리스트가 없다", ["d"]) == []          # 블록 없음 = 빈손
    _hay = "the annual limit is 3 per year and nothing else"
    _raw = ('{"A":{"limit":3,"quote":"the annual limit is 3 per year"},'
            ' "B":{"limit":2,"quote":"a sentence that never appeared"},'
            ' "C":{"limit":0,"quote":"the annual limit is 3 per year"},'
            ' "D":{"limit":2,"quote":"too short"}}')
    _got, _rej, _given = parse_pairs(_raw, "limit", _hay)
    assert _given == 4 and len(_got) == 1 and _rej == 3, (_given, _got, _rej)
    assert _got["A"][0] == 3                                       # B=인용 부재 · C=0 · D=너무 짧음
    assert parse_pairs("no json here", "limit", _hay) == ({}, 0, 0)
    assert parse_scalar('"11/14/2025" 라고 답함', ["%m/%d/%Y"]) == "11/14/2025"
    assert parse_scalar("2025-11-14", ["%m/%d/%Y", "%Y-%m-%d"]) == "2025-11-14"   # 선언된 둘째 형식
    assert parse_scalar("2025-11-14", ["%m/%d/%Y"]) is None         # 선언 안 된 형식 = 미채택
    assert parse_scalar("", ["%m/%d/%Y"]) is None
    print("t2_ledger self-test OK ·", facts_text(rows, spec, now="11/14/2025").strip())
