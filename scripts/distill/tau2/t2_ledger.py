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

__all__ = ["specs_for", "formalize_rows", "window_and_tally", "facts_text"]


def specs_for(a2, tool_name):
    return [s for s in ((a2 or {}).get("ledger_metrics") or [])
            if s.get("trigger_tool") == tool_name]


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
    out = spec["text"].format(total=len(rows), tally=tally_s, window=win_s)
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
