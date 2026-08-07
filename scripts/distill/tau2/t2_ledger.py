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
           "exhausted_text", "ineligible_text", "facts_text"]


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
    m = re.search(r"\[.*\]", raw, re.S)
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
    cand = raw.split()[0].strip('".,') if raw.split() else ""
    if not _date(cand, (spec.get("date_formats") or ["%m/%d/%Y"])):
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


def _formalize_pairs(agent, la, UserMessage, texts, spec, key, field, memo_attr, call_name):
    """`{그룹: (정수, 인용문)}` 형태를 모델에게 받는 공용 절차 — 인용 실재만 엔진이 확인한다."""
    tpl = (spec or {}).get(key)
    if not (tpl and agent is not None and la is not None and texts):
        return {}
    memo = getattr(agent, memo_attr, None)
    if memo is not None:
        return memo
    prompt = tpl.format(text="\n---\n".join(str(t)[:4000] for t in texts[-12:]))
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
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        got = json.loads(m.group(0))
    except Exception:
        return {}
    hay = " ".join("\n".join(str(t) for t in texts).split())
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
    # ★빈손도 찍는다 (2026-08-08·lim_n 라이브). 성공만 찍으면 "못 찾았다"와 "안 돌았다"가
    #   구분되지 않는다 — 오늘 `seen=N`으로 배운 것과 같은 형태다.
    print("[T2_LEDGER] %s: model gave %d, accepted %d, rejected %d"
          % (field, len(got) if isinstance(got, dict) else 0, len(out), rejected),
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
    print("t2_ledger self-test OK ·", facts_text(rows, spec, now="11/14/2025").strip())
