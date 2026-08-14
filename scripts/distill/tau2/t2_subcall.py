# -*- coding: utf-8 -*-
"""격리 서브 호출·계약 파싱·근거 검산의 **정본 라이브러리** (2026-08-14·사용자 지시).

왜 이 파일이 있나: 같은 관용구가 엔진에 반복 복제돼 왔다 — 실측 인벤토리(2026-08-14):
  · 단발 격리 서브 호출(la.generate)                 39곳 / 8파일
  · `UserMessage(role=..)` try/except TypeError      50곳 / 8파일
  · JSON 계약 파싱(re.search + json.loads)           10곳 / 5파일
  · 근거(grounding) 검산                             4벌 구현 / 23 사용처
사본마다 미묘하게 달라 결함이 **사본별로** 생겼다(예: FIX-13 1차 검산이 `9.50↔9.5` 를 놓침 —
`_val_grounded` 는 이미 형식-불문 수치 매칭을 했는데 재사용하지 않고 다시 짰다).

계약(전 채널 공통·[[10]]·[[22]]):
  · 판단(무엇을 고를지·값이 무엇인지)은 **전부 LLM 몫** — 이 라이브러리는 전송·파싱·닫힌-술어
    검산만 한다. argmax·정답 산출 없음([[62]]).
  · 검산은 닫힌 술어 둘뿐: 이름의 집합-소속 · 값의 코퍼스-실재(substring/수치/날짜 형식-불문).
  · 실패는 항상 조용한 폴백 신호(None/빈값) — 호출부 종전 거동 보존.

신규 채널 규칙: 단발 격리 서브가 필요하면 **이 파일의 함수를 부른다**. 인라인 la.generate 를
새로 쓰면 `test_subcall_canonical.py`(사본-수 래칫)가 잡는다.

⚠범위 밖(의도적): `t2_scaffold_get._sub_*` 계열은 **다회전 getter-루프 서브**(도구 실행 동반)라
계약이 다르다 — 그 가족은 이미 한 파일에 모여 있고 여기 정본은 **단발(tools=None)** 만 담당한다.
`t2_gate_patch` 의 메인-루프 재생성(la.generate(tools=self.tools))도 서브가 아니므로 범위 밖.
"""
import json
import re
import sys


def make_user_message(UserMessage, content):
    """UserMessage 생성 관용구의 정본 (시그니처 차이 흡수·50곳 사본 대체)."""
    try:
        return UserMessage(role="user", content=content)
    except TypeError:
        return UserMessage(content=content)


def sub_generate(agent, la, UserMessage, prompt, call_name, temperature=None):
    """단발 격리 서브 호출의 정본 (39곳 사본 대체).

    자체 메시지 리스트 하나·tools=None·llm_args 의 tool 계열 키 제거. 반환 = 텍스트('' = 실패).
    예외는 삼키고 '' — 호출부가 폴백을 결정한다(조용한 거동 변경 금지).
    """
    if agent is None or la is None or UserMessage is None:
        return ""
    try:
        um = make_user_message(UserMessage, prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        if temperature is not None:
            kw["temperature"] = temperature
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name=call_name, **kw)
        return str(getattr(sub, "content", None) or "")
    except Exception as e:
        print("[T2_SUBCALL] %s 실패(폴백): %r" % (call_name, e), file=sys.stderr, flush=True)
        return ""


def parse_contract(txt, key=None):
    """서브 응답에서 JSON 계약 하나를 꺼낸다 (10곳 사본 대체).

    key 를 주면 그 키를 담은 최외곽 객체를 요구한다. 반환 dict | None(=폴백).
    도메인 텍스트 스캔이 아니다 — **우리가 요구한 계약**의 회수다([[59]] 허용역).
    """
    t = str(txt or "")
    pat = r"\{.*%s.*\}" % re.escape('"%s"' % key) if key else r"\{.*\}"
    m = re.search(pat, t, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if key is not None and (not isinstance(obj, dict) or key not in obj):
        return None
    return obj if isinstance(obj, dict) else None


def val_grounded(val, corpus_texts, kind=None):
    """값 하나의 코퍼스-실재 검산 정본 — `t2_scaffold_get._val_grounded` 위임(재구현 금지).

    수치는 형식-불문($·%·, 흡수·9.50≡9.5)·날짜도 형식-불문. FIX-13 1차가 이걸 재사용하지 않고
    substring 만 봐서 `9.50↔9.5` 를 놓쳤다 — 그 실수의 재발 방지 지점이 이 위임이다.
    """
    from t2_scaffold_get import _val_grounded
    return _val_grounded(val, corpus_texts, kind=kind)


def recent_tool_text(msgs, cap=4000, scope="recent"):
    """성공 도구결과 축자(근거 코퍼스·msgs 기반). scope="recent"=직전 손님 발화 이후 · "all"=전체.

    orch 가 있는 자리는 `t2_scaffold_get._corpus_texts(orch, ["ledger"])` 를 쓰라 — 그쪽이
    전체-원장 코퍼스의 정본이다. 이 함수는 msgs 만 있는 resolve-계열 전용.

    ★scope 가 인자다 (2026-08-14·x311 사전등록): 075 착수는 근거를 **직전 턴으로 자르면 0/8**,
      **대화 전체로 넓히면 8/8** 이다(C_GEN75 ↔ D_GEN75F). 075 의 `user_id` 는 훨씬 앞 턴의
      도구 결과에서 나왔고 그 뒤 손님 발화가 여러 번 있어 'recent' 창 밖이었다.
      073(직전 턴에 감사 결과가 있는 형)은 어느 쪽이든 8/8 이라 넓혀도 잃는 게 없다.
    """
    ms = list(msgs or [])
    start = 0
    if scope != "all":
        last_user = max([i for i, m in enumerate(ms)
                         if getattr(m, "role", None) == "user"] or [-1])
        start = last_user + 1
    out = []
    for m in ms[start:]:
        if getattr(m, "role", None) != "tool":
            continue
        if getattr(m, "error", False):
            continue
        c = str(getattr(m, "content", "") or "").strip()
        if c and not c.lstrip().startswith("Error"):
            out.append(c)
    txt = "\n".join(out)
    return txt[-int(cap):] if cap and len(txt) > int(cap) else txt


def grounded_calls(calls, corpus_texts, names):
    """제안 호출 목록의 닫힌-술어 검산 (FIX-13 검산부의 정본).

    ① 도구명 ∈ names(실재 레지스트리·엔진이 이름을 짓지 않는다)
    ② 제안의 모든 값이 코퍼스에 실재(`val_grounded` — 수치·날짜 형식-불문)
    탈락 = 목록에서 제외 → 전부 탈락이면 호출부가 폴백한다. 판단 0([[25]] 집행 장치).

    ★[[59]] 경계 (2026-08-14 사용자 질의 — 코드에서 바로 답 받도록 박제):
      금지는 *"문자열을 본다"* 가 아니라 **방향**이다. 금지 = 엔진 패턴이 **도메인 텍스트에서
      값을 뜯어** 판단을 만든다(폐기된 `parse_records`형). 허용 = **LLM 이 낸 값**의 원문 실재를
      엔진이 확인한다(C45 provenance·quote_grounding·WRITE-GROUNDING 계보).
      이 함수는 후자다 — 값의 출처는 전부 서브(LLM)이고, 엔진 출력은 **불리언뿐**이며 통과분은
      LLM 산출의 부분집합이다. 엔진이 값을 만들어내는 경로가 없다.
      ⚠한계(의도적): 잡는 것은 **날조**(코퍼스에 없는 값)뿐이다. *근거에 있으나 틀린 값*(다른
      행의 금액을 집는 등)은 못 잡고, **잡으려 들면 그때부터 위반**이다(엔진이 정답 재도출 =
      gold 프로그램 재작성·[[62]]). 그 층은 LLM 이 본다 — 근거를 동봉하면 메인이 실제로
      걸러낸다(x310 B_CITE_W 순응 0/8).
    """
    ok = []
    for c in (calls or []):
        if not isinstance(c, dict):
            continue
        tool = str(c.get("tool") or c.get("name") or "")
        if tool not in (names or set()):
            continue
        vals = [v for k, v in c.items()
                if k not in ("tool", "name") and v not in (None, "")]
        if not vals or not all(val_grounded(v, corpus_texts) for v in vals):
            continue
        ok.append(c)
    return ok
