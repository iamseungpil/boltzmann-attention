"""P1 — A2가 선언한 **선행 read**를 생성-측에서 한 번 고정한다.

`require_tool_before`는 `open_bank_account` 앞에 `get_all_user_accounts_by_user_id`를 요구하지만
**강제하지 못한다.** deny 스텁이 tool 출력이라 tau2 replay가 깨끗한 env에서 재실행해 내용을 정확
비교하고, 불일치하면 sim 전체가 죽기 때문이다(C210/day6 3건 실측). 그래서 사후 권고로 강등됐고,
N97에서 그 권고의 실발화는 **1회**였다. gold가 관문을 요구한 신규 실패 79 sim 중 **53이 끝내
호출하지 않았다.**

생성-측 제약은 그 반론이 닿지 않는다 — **tool 출력을 만들지 않으므로** replay가 비교할 것이 없다.
다만 이름만 지정해서는 안 된다는 것이 실측으로 나왔다(`x72` · 서버 n=3):

    tool_choice=auto            → 호출 0 · 순수 산문
    tool_choice=required        → KB_search_bm25 (현행 T2_FORCE_ACTION의 한계)
    named tool_choice (디스패처) → unlock(agent_tool_name="AccountLookupTool")   ← 내부 이름 날조
    named + 단일값 enum          → unlock(agent_tool_name="get_all_..._3847")    ← 3/3 적중

discoverable 도구는 `tools` 배열에 없고 디스패처 인자로 실려가므로, **그 인자를 단일값 enum으로
고정**해야 표적에 닿는다. 평범한 JSON schema이고 vLLM 확장이 아니다.

★rev3(2026-08-05·스모크 실측 후 재배선). 초판은 **방금 생성된 메시지의 tool_calls**를 순회해
표적을 찾았는데, 재생성을 실제로 부르는 것은 `T2_FORCE_ACTION`(= 호출이 0일 때)이라 창이 열리지
않았다 — 라이브 발화 0. 그리고 위 x72가 증명한 상황이 정확히 그 무호출 경우다. 그래서 표적을
**sim-범위 상태**로 옮겼다:

    M(sim) = A2 선언 선행 read − 이 sim에서 실행된 도구
    pin ⟸ M ≠ ∅ ∧ 수요 신호(⒜의존도구 시도 ∨ ⒝우리 층 결손통지 ∨ ⒞레코드 부재 오류)
          ∧ 피의존 ≥2 ∧ read 접두 ∧ 레지스트리 유일 해소 ∧ 아직 미고정(1회/sim)

수요 게이트와 피의존 제한이 둘 다 필요하다 — 조건 없이 sim-범위로 두면 **194/194 sim에서** 발화하고
선언 순서로 고르면 조준이 **gold 적중 0**이다(전수 분할 검증 = `x74_pin_target_precount.py`).
채택안은 trial 0에서 고르고 trial 1에서 확인했다(발화 16·적중 69%).

★[[05]] 경계
  · 표적은 **A2 기존 선언 두 키**에서만 온다 — gold를 보지 않는다([[03b]]).
  · 접미사는 **env 레지스트리 기계도출**([[23]] opex 0) — A2 순증 0.
  · ⒞의 `not found`는 tau2 **env 공통 관용구**다(4도메인 전부) — 도메인 데이터가 아니다.
  · **read만 고정한다.** write 이름 고정은 이 모듈이 거부한다([[14]] "read(discovery)만 강제").
  · 고정은 **재생성 1회**뿐이고 그 뒤 무엇을 할지는 모델 몫이다.
  · 스키마를 안전하게 만들 수 없으면 **고정하지 않는다**(이름만 지정하면 날조를 부르므로).
"""

import os
import re

_SUFFIX = re.compile(r"_\d{3,4}$")
# read만 허용한다. 이 접두 밖의 도구는 고정 대상이 아니다(엔진 리터럴이 아니라 동사 패턴).
_READ_PREFIX = ("get_", "list_", "check_", "search_", "view_", "read_", "fetch_")


def enabled():
    return os.environ.get("T2_PIN_READ") == "1"


def _fam(n):
    return _SUFFIX.sub("", n or "")


def _called_fams(messages):
    out = set()
    for m in messages or []:
        for tc in (getattr(m, "tool_calls", None) or []):
            n = getattr(tc, "name", None)
            if n:
                out.add(_fam(n))
            a = getattr(tc, "arguments", None)
            if isinstance(a, dict):
                for k in ("agent_tool_name", "discoverable_tool_name", "user_tool_name"):
                    if a.get(k):
                        out.add(_fam(a[k]))
    return out


def _declarations(a2):
    """(의존 도구, [선행 read]) — A2의 **두 키를 모두** 읽는다, 선언 순서대로.

    rev2까지는 `require_tool_before`만 봤는데, N97 라이브에서 거부 15회는 전부 다른 키
    (`scaffold_get_tools[].requires_reads`)에서 나왔다. 한 키만 보는 것이 P1의 2차 간극이었다.
    """
    out = []
    for dep, reads in ((a2 or {}).get("require_tool_before") or {}).items():
        out.append((_fam(dep), [_fam(r) for r in (reads or [])]))
    for e in ((a2 or {}).get("scaffold_get_tools") or []):
        if isinstance(e, dict) and e.get("requires_reads"):
            dep = e.get("tool") or e.get("name") or ""
            out.append((_fam(dep), [_fam(r) for r in e["requires_reads"]]))
    return out


def _refcount(decls):
    """선행 read별 **피의존 수** — 몇 개의 선언이 이 read를 요구하는가.

    막고 있는 경로가 하나뿐인 read는 "모델이 무엇을 하려는가"의 증거가 약하다. 전수 분할
    검증에서 이 제한이 조준을 57%→69%(out-of-sample)로 올리고 오조준을 21→10으로 줄였다
    (`x74_pin_target_precount.py` · 설계서 §1.7). A2에서 세는 수라 gold와 무관하다.
    """
    c = {}
    for _dep, reads in decls:
        for r in reads:
            c[r] = c.get(r, 0) + 1
    return c


def _demand(messages, decls):
    """이 sim에서 관측된 **수요 신호**가 가리키는 선행 read 집합.

    셋 중 하나라도 서면 그 read가 수요된 것으로 본다(설계서 §1.7):
      ⒜ 그 read를 요구하는 의존 도구를 이미 시도했다        — 호출 이력(닫힘)
      ⒝ 우리 층이 그 결손을 이미 통지했다                   — `[READ-FIRST]`
      ⒞ 레코드를 못 찾았다는 환경 오류를 받았다              — `not found`(tau2 env 공통 관용구)
    수요가 없으면 고정하지 않는다. 무제한으로 두면 194/194 sim에서 발화한다(§1.7).
    """
    all_reads = {r for _d, reads in decls for r in reads}
    # ⒜는 **에이전트가** 시도한 것만 본다(손님이 실행한 도구는 수요 신호가 아니다).
    agent_calls = _called_fams([m for m in (messages or [])
                                if getattr(m, "role", None) == "assistant"])
    out = set()
    for dep, reads in decls:
        if dep in agent_calls:                              # ⒜
            out |= set(reads)
    for m in messages or []:
        # ⒝⒞는 **tool 출력**만 본다 — 엔진 통지와 환경 오류다. 모델 산문을 신호로 쓰면
        # 그것이 바로 폐기된 산문-파싱이다(설계서 §3-1·[[16]] P2-b).
        if getattr(m, "role", None) != "tool":
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        if "[READ-FIRST]" in c:                             # ⒝
            out |= {r for r in all_reads if r in c}
        if "not found" in c.lower():                        # ⒞
            out |= all_reads
    return out


def pinned_already(orch):
    return bool(getattr(orch, "_t2_pin_read_done", False))


def mark_pinned(orch):
    """고정이 **실제로 적용된 뒤에만** 캡을 소모한다(스키마 조립 실패는 소모 아님)."""
    try:
        orch._t2_pin_read_done = True
    except Exception:
        pass


def _resolve(orch, base):
    """base → 접미사 포함 실명. 레지스트리에서 **유일**할 때만."""
    try:
        import t2_callable_hint as _CH
        pairs = _CH.resolve(orch, [base])
        return pairs[0][1] if pairs else None
    except Exception:
        return None


def pin_for(orch, am, a2, messages):
    """이번 재생성에서 고정할 (도구, 인자, 값). 조건 미충족이면 None.

    **sim-범위**다 — 방금 생성된 메시지에 호출이 있든 없든 성립한다. rev2까지는 `am.tool_calls`를
    순회해서 표적을 찾았는데, 재생성을 실제로 부르는 것은 `T2_FORCE_ACTION`(= 호출이 0일 때)이라
    창이 열리지 않았다. 그리고 x72가 3/3으로 증명한 상황이 바로 그 무호출 경우였다(설계서 §0).

        M(sim) = A2가 선언한 선행 read − 이 sim에서 실행된 도구
        pin ⟸ M ≠ ∅ ∧ 수요 신호 ∧ 피의존≥2 ∧ read 접두 ∧ 레지스트리 유일 해소 ∧ 아직 미고정
    """
    if not enabled() or pinned_already(orch):
        return None
    ep = ((a2 or {}).get("eplan") or {})
    unlock = ep.get("unlock_tool")
    decls = _declarations(a2)
    if not (decls and unlock):
        return None
    # 방금 생성된 호출도 실행 집합에 넣는다 — 그 턴에 이미 부른 read를 다시 고정하지 않도록.
    called = _called_fams(list(messages or []) + [am])
    demanded = _demand(messages, decls)
    rc = _refcount(decls)

    ranked, seen = [], set()
    for i, (_dep, reads) in enumerate(decls):
        for j, r in enumerate(reads):
            if r in seen or r in called or r not in demanded:
                continue
            if rc.get(r, 0) < 2:
                continue                      # 피의존 1 = 증거가 약하다(설계서 §1.7)
            if not r.startswith(_READ_PREFIX):
                continue                      # write는 고정하지 않는다([[14]])
            seen.add(r)
            ranked.append((-rc[r], i, j, r))
    ranked.sort()
    for _rc, _i, _j, r in ranked:
        full = _resolve(orch, r)
        if full:                              # 유일 해소만 — 모호하면 다음 후보로
            return (unlock, "agent_tool_name", full)
    return None


class _PinnedTool:
    """`openai_schema`만 바꿔 끼우는 얇은 shim. 실행 경로는 원본이 그대로 진다.

    tau2는 `[tool.openai_schema for tool in tools]`로만 스키마를 읽으므로(llm_utils:389)
    이 속성 하나면 충분하고, 원본 Tool은 건드리지 않는다.
    """

    def __init__(self, tool, schema):
        self._t = tool
        self._schema = schema

    @property
    def openai_schema(self):
        return self._schema

    def __getattr__(self, k):
        return getattr(self._t, k)


def tools_with_pin(tools, tool_name, arg_name, value):
    """해당 도구의 `arg_name`을 단일값 enum으로 고정한 도구 목록. 실패하면 None."""
    try:
        import copy
        out, hit = [], False
        for t in tools or []:
            if getattr(t, "name", None) != tool_name:
                out.append(t)
                continue
            sch = copy.deepcopy(t.openai_schema)
            props = (((sch.get("function") or {}).get("parameters") or {})
                     .get("properties") or {})
            if arg_name not in props:
                return None                    # 인자가 없으면 고정 불가 = 고정하지 않는다
            props[arg_name] = {"type": "string", "enum": [value]}
            out.append(_PinnedTool(t, sch))
            hit = True
        return out if hit else None
    except Exception:
        return None


def choice(tool_name):
    return {"type": "function", "function": {"name": tool_name}}


def selftest():
    class _T:
        def __init__(self, name, props):
            self.name = name
            self._p = props

        @property
        def openai_schema(self):
            return {"type": "function",
                    "function": {"name": self.name,
                                 "parameters": {"type": "object", "properties": dict(self._p)}}}

    class _TC:
        def __init__(self, name, args):
            self.name, self.arguments = name, args

    class _AM:
        def __init__(self, tcs):
            self.tool_calls = tcs

    class _M:                        # assistant 메시지
        def __init__(self, tcs):
            self.tool_calls, self.role, self.content = tcs, "assistant", None

    class _TM:                       # tool 메시지(엔진 통지·env 오류)
        def __init__(self, content):
            self.content, self.tool_calls, self.role = content, [], "tool"

    class _Orch:
        pass

    import t2_pin_read as M
    M._resolve = lambda orch, base: {
        "get_all_user_accounts_by_user_id": "get_all_user_accounts_by_user_id_3847",
        "get_bank_account_transactions": "get_bank_account_transactions_9173",
        "check_card_application_fit": "check_card_application_fit_5512"}.get(base)
    # 관문 = 3개 선언이 의존(피의존 3) · fit = 1개만(피의존 1) · close_ = write
    a2 = {"require_tool_before": {"open_bank_account": ["get_all_user_accounts_by_user_id"],
                                  "submit_referral": ["get_all_user_accounts_by_user_id"],
                                  "apply_for_credit_card": ["check_card_application_fit"],
                                  "pay_credit_card": ["close_bank_account"]},
          "scaffold_get_tools": [{"tool": "get_interest_correction",
                                  "requires_reads": ["get_all_user_accounts_by_user_id",
                                                     "get_bank_account_transactions"]}],
          "eplan": {"unlock_tool": "unlock_discoverable_agent_tool"}}
    os.environ["T2_PIN_READ"] = "1"
    GATE = ("unlock_discoverable_agent_tool", "agent_tool_name",
            "get_all_user_accounts_by_user_id_3847")

    # ★핵심 회귀: 호출이 **하나도 없는** 재생성에서 열려야 한다(rev2가 못 열던 그 창).
    am0 = _AM([])
    hist_c = [_TM("Error: Account 'rp65a7b3c4' not found")]
    assert M.pin_for(_Orch(), am0, a2, hist_c) == GATE
    print("  ok   무호출 재생성 + 레코드부재 오류 → 관문으로 고정")

    hist_b = [_TM("Error: [READ-FIRST] missing: get_all_user_accounts_by_user_id")]
    assert M.pin_for(_Orch(), am0, a2, hist_b) == GATE
    print("  ok   우리 층의 결손 통지도 수요 신호")

    hist_a = [_M([_TC("call_discoverable_agent_tool", {"agent_tool_name": "open_bank_account_4821"})])]
    assert M.pin_for(_Orch(), am0, a2, hist_a) == GATE
    print("  ok   의존 도구를 시도했으면 그 선행 read로 고정")

    assert M.pin_for(_Orch(), am0, a2, []) is None
    print("  ok   수요 신호가 없으면 무발화 (194/194 발화 방지)")

    hist_done = hist_c + [_M([_TC("call_discoverable_agent_tool",
                                  {"agent_tool_name": "get_all_user_accounts_by_user_id_3847"})])]
    assert M.pin_for(_Orch(), am0, a2, hist_done) is None
    print("  ok   이미 읽었으면 무발화")

    # fit은 피의존 1 — 수요가 있어도 고정 대상이 아니다(오조준 9건이 여기서 사라졌다)
    a2_fit = {"require_tool_before": {"apply_for_credit_card": ["check_card_application_fit"]},
              "eplan": {"unlock_tool": "unlock_discoverable_agent_tool"}}
    assert M.pin_for(_Orch(), am0, a2_fit, [_TM("not found")]) is None
    print("  ok   피의존 1인 선행 read는 제외 (§1.7)")

    a2_w = {"require_tool_before": {"pay_credit_card": ["close_bank_account"],
                                    "x": ["close_bank_account"]},
            "eplan": {"unlock_tool": "unlock_discoverable_agent_tool"}}
    assert M.pin_for(_Orch(), am0, a2_w, [_TM("not found")]) is None
    print("  ok   선행이 write면 고정하지 않는다 ([[14]] read만)")

    o = _Orch()
    assert M.pin_for(o, am0, a2, hist_c) == GATE
    M.mark_pinned(o)
    assert M.pin_for(o, am0, a2, hist_c) is None
    print("  ok   1회/sim 캡 — 적용된 뒤에만 소모")

    os.environ["T2_PIN_READ"] = "0"
    assert M.pin_for(_Orch(), am0, a2, hist_c) is None
    os.environ["T2_PIN_READ"] = "1"
    print("  ok   플래그 OFF면 무발화")

    tools = [_T("unlock_discoverable_agent_tool", {"agent_tool_name": {"type": "string"}}),
             _T("KB_search_bm25", {"query": {"type": "string"}})]
    pinned = M.tools_with_pin(tools, "unlock_discoverable_agent_tool", "agent_tool_name",
                              "get_all_user_accounts_by_user_id_3847")
    sch = [t.openai_schema for t in pinned if t.name == "unlock_discoverable_agent_tool"][0]
    assert sch["function"]["parameters"]["properties"]["agent_tool_name"]["enum"] == \
        ["get_all_user_accounts_by_user_id_3847"]
    assert tools[0].openai_schema["function"]["parameters"]["properties"]["agent_tool_name"] == \
        {"type": "string"}, "원본 오염"
    print("  ok   enum 고정 + 원본 Tool 불변")

    assert M.tools_with_pin(tools, "unlock_discoverable_agent_tool", "no_such_arg", "x") is None
    assert M.tools_with_pin(tools, "no_such_tool", "agent_tool_name", "x") is None
    print("  ok   인자·도구 부재면 고정 포기(이름만 지정 안 함)")
    print("PASS (9/9)")


if __name__ == "__main__":
    selftest()
