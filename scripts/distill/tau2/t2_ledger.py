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
    if not (tpl and agent is not None and la is not None and table and asked and allowed):
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


def _formalize_pairs(agent, la, UserMessage, texts, spec, key, field, memo_attr, call_name,
                     extra=""):
    """`{그룹: (정수, 인용문)}` 형태를 모델에게 받는 공용 절차 — 인용 실재만 엔진이 확인한다."""
    tpl = (spec or {}).get(key)
    if not (tpl and agent is not None and la is not None and texts):
        return {}
    memo = getattr(agent, memo_attr, None)
    if memo is not None:
        return memo
    # ★발췌는 **꼬리가 아니다** (2026-08-08·lim_n 위치 실측). 상한을 말하는 텍스트가 인덱스
    #   2·15·17·19처럼 흩어져 있는데 구판은 `texts[-12:]`만 봤다 — 결정점으로 옮겨도 꼬리이면
    #   앞쪽(2번)을 잃고, 대화가 길어지면 뒤쪽도 같은 방식으로 밀려난다. 오늘 아침 `now_prompt`가
    #   정확히 같은 형태로 실패했다(오늘 날짜가 대화 머리에 있었다).
    #   ⇒ **위치로만** 고른다: 전부 주되 각각 절단하고 총량에 상한을 둔다. 어느 텍스트가 상한을
    #      담았는지 판정하는 것은 모델 몫이고, 엔진이 내용으로 고르면 [[59]] 위반이다.
    _per = 3000
    _budget = 90000
    _sel, _used = [], 0
    for t in reversed(list(texts)):       # 최신부터 담고, 예산이 남으면 앞쪽도 들어온다
        s = str(t)[:_per]
        if _used + len(s) > _budget:
            break                         # 잘려 나가는 쪽은 **가장 오래된 것**이어야 한다
        _sel.append(s)
        _used += len(s)
    _sel.reverse()                        # 다시 시간 순서로(읽는 쪽이 대화 순서를 본다)
    # ★자리는 **A2가 정한다**: 템플릿에 `{items}` 가 있으면 거기 넣는다(발췌 앞에 둘 수 있다).
    #   없으면 구판대로 뒤에 잇는다 — 기존 선언 3종의 거동은 그대로다.
    _body = "\n---\n".join(_sel)
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


def eligible_text(days, tally, axis_maps, spec, stated=None):
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
    live = []
    for c in crit:
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
            ok.append("  %s: %s" % (s, ", ".join(bits)))
    if not ok:
        return ""
    return tpl.format(eligible="\n".join(ok))


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
