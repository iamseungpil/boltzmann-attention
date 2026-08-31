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
import hashlib
import os
import json
import re
import sys


def make_user_message(UserMessage, content):
    """UserMessage 생성 관용구의 정본 (시그니처 차이 흡수·50곳 사본 대체)."""
    try:
        return UserMessage(role="user", content=content)
    except TypeError:
        return UserMessage(content=content)


def _record_subcall(call_name, prompt, out, err=None, cached=False):
    """서브가 **무엇을 받았고 무엇을 냈나**를 사이드카에 남긴다 (2026-08-24 신설·기록만).

    ★왜 필요한가 — 없어서 하루를 태웠다:
      · 원장 C508⒥ 축자: *"라이브는 **실제로 실린 인용 축자를 어디에도 안 남긴다**(사이드카
        `sub_requirement` 0건) — 개수만 안다. **다음 런 전 필수 수리**."*
      · 2026-08-24 서브-오답 워크플로(에이전트 26)의 유일한 실질 약점도 같은 것이었다:
        *"이 런에는 서브의 실제 프롬프트 문자열을 남긴 계기가 **없다**. 입력 주장은 코드 +
        실효 A2 + 분기 마커로 세운 것이지 **포획된 축자로 세운 것이 아니다**."*
      ⇒ [[76]] 진단 순서 ①(*"서브가 무엇을 받았나 — 라이브 입력 ↔ 격리 입력 축자 대조"*)이
        요구하는 재료가 라이브에 **없었다**. 그래서 서브가 틀렸을 때 ⒜자격 없이 배선된 것인지
        ⒝라이브에서 다른 것을 받는 것인지를 **원리상 가릴 수 없었다**.

    거동 불변 — 기록뿐이고 반환값·호출 순서를 건드리지 않는다. `T2_FB_SIDECAR` 미설정이면
    `t2_fbsidecar.record` 가 스스로 no-op 이다(그 모듈 설계 제약 2). 본문은 `T2_FB_SIDECAR_TEXT=1`
    일 때만 저장되고 4000자에서 잘린다 — **프롬프트는 머리(=요청)가 앞에 오므로 그 창으로 족하다**.
    응답은 짧으므로 meta 에 머리 600자를 함께 싣는다(길이·해시는 항상 남는다).
    """
    try:
        import t2_fbsidecar as _sc
        _o = "" if out is None else str(out)
        _sc.record("subcall", prompt, None,
                   call_name=str(call_name),
                   prompt_len=len(str(prompt or "")),
                   out_len=len(_o),
                   out_head=_o[:600],
                   cached=bool(cached),   # 재사용인지 실제 호출인지 가른다([[25]])
                   err=(None if err is None else repr(err)[:200]))
    except Exception:
        pass                                     # 기록 실패가 런을 깨면 안 된다


def sub_generate(agent, la, UserMessage, prompt, call_name, temperature=None):
    """단발 격리 서브 호출의 정본 (39곳 사본 대체).

    자체 메시지 리스트 하나·tools=None·llm_args 의 tool 계열 키 제거. 반환 = 텍스트('' = 실패).
    예외는 삼키고 '' — 호출부가 폴백을 결정한다(조용한 거동 변경 금지).

    ★2026-08-24: 여기 한 자리에 `_record_subcall` 을 달아 **서브 호출 전량**(호출부 35곳)이
      입력·출력을 사이드카에 남기게 했다. 사본을 35개 만들지 않는 이유는 [[67]] 그대로다.
    """
    if agent is None or la is None or UserMessage is None:
        return ""
    try:
        um = make_user_message(UserMessage, prompt)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        if temperature is not None:
            kw["temperature"] = temperature
        # ★같은 물음을 두 번 묻지 않는다 (2026-08-31·`T2_SUBCALL_CACHE`·기본 ON).
        #   실측(x697 라이브·바이트 동일 지문): `intent_operator_formalize` 6회 중 **5회가
        #   프롬프트까지 동일**했다 — prompt 239 · gen 4,108 · reason 18,220B · content 40B.
        #   그 중복만 **16,432토큰 = 그 런 전체 생성의 50%** 다. 창이 *손님 발화 마지막 6개*
        #   라서, 손님이 말하지 않은 턴에는 프롬프트가 글자 하나 안 바뀐다.
        #   ⚠**정보 손실 0인 조건에서만** 쓴다: `temperature==0` 일 때만 캐시한다(그때 응답은
        #     결정론이고, 실제로 reason 바이트까지 동일했다). 온도가 있으면 재표집이 의미이므로
        #     캐시하지 않는다 — 닫힌 술어 하나([[22]]).
        #   ⚠범위는 **이 에이전트(=이 sim)** 다. 프로세스 전역이면 sim 간 오염이 된다.
        #   ⚠사이드카에는 `cached=True` 로 남긴다 — 포렌식이 *부른 것*과 *재사용한 것*을
        #     구별할 수 있어야 한다([[25]]).
        _t = kw.get("temperature")
        _key = None
        if os.environ.get("T2_SUBCALL_CACHE", "1") == "1":
            try:
                if _t is not None and float(_t) == 0.0:
                    _key = (str(call_name),
                            hashlib.sha1(str(prompt).encode("utf-8", "replace")).hexdigest())
            except Exception:
                _key = None
        if _key is not None:
            _c = getattr(agent, "_t2_subcall_cache", None)
            if _c is None:
                _c = {}
                try:
                    setattr(agent, "_t2_subcall_cache", _c)
                except Exception:
                    _c = None
            if _c is not None and _key in _c:
                out = _c[_key]
                print("[T2_SUBCALL] cache hit call=%s (같은 프롬프트 재질의 생략 · %d자)"
                      % (call_name, len(prompt or "")), file=sys.stderr, flush=True)
                _record_subcall(call_name, prompt, out, cached=True)
                return out
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name=call_name, **kw)
        out = str(getattr(sub, "content", None) or "")
        if _key is not None:
            _c = getattr(agent, "_t2_subcall_cache", None)
            if isinstance(_c, dict):
                _c[_key] = out
        _record_subcall(call_name, prompt, out)
        return out
    except Exception as e:
        print("[T2_SUBCALL] %s 실패(폴백): %r" % (call_name, e), file=sys.stderr, flush=True)
        # ★실패도 남긴다 — 빈 반환이 *안 불렀다* 인지 *부르고 실패했다* 인지 가려야 한다([[25]]).
        _record_subcall(call_name, prompt, "", err=e)
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


def _leaf_values(call):
    """제안 호출에서 **잎 값**만 모은다 — 중첩 `arguments` 를 푼다.

    ★2026-08-14 실측 결함: x311 이 계약을 `{"tool", "arguments": {...}}` 로 일반화했는데
    검산부는 평면 구조만 알아서 `c.items()` 가 **dict 하나**를 값으로 내놨다. dict 의 문자열
    형태는 코퍼스에 있을 리 없으므로 **우리 형식을 지킨 제안은 100% 기각**됐다(072 실물:
    `제안 N건 → 통과 0건` ×8·t7290/t7291 양쪽). 073 이 통과한 자리는 모델이 우리 형식을
    **어기고** 평면으로 답한 경우였다 — 계약과 검산기가 서로 다른 모양을 보고 있었다.
    푸는 것뿐이고 판단은 0이다."""
    out = []

    def walk(v):
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 in ("tool", "name"):
                    continue
                walk(v2)
        elif isinstance(v, (list, tuple)):
            for v2 in v:
                walk(v2)
        elif v not in (None, ""):
            out.append(v)

    for k, v in (call or {}).items():
        if k in ("tool", "name"):
            continue
        walk(v)
    return out


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
        vals = _leaf_values(c)
        if not vals or not all(val_grounded(v, corpus_texts) for v in vals):
            continue
        ok.append(c)
    return ok
