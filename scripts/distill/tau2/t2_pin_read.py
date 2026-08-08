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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_precedence as prec                      # noqa: E402  — 선행 선언의 유일한 입구

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
    """(의존 도구, [선행 read]) — **`t2_precedence.declarations`가 유일한 입구다**, 선언 순서대로.

    rev2까지는 `require_tool_before`만 봤는데, N97 라이브에서 거부 15회는 전부 다른 키
    (`scaffold_get_tools[].requires_reads`)에서 나왔다. 한 키만 보는 것이 P1의 2차 간극이었다.
    rev3(2026-08-08): 그 **두 키를 여기서 직접 읽던 것을 걷어냈다** — 선행 관계의 원천은
    A3 인덱스로 옮기는 중이고(설계서 §1d.1), 소비자가 각자 키를 읽으면 원천이 다시 갈린다.
    보던 범위는 **그대로**다(두 출처) ⇒ 거동 변화 0, `x144 --verify`가 등가를 강제한다.
    """
    return prec.declarations(a2, (prec.SRC_REQUIRE_BEFORE, prec.SRC_REQUIRES_READS))


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


def _demand(messages, decls, declared=()):
    """이 sim에서 관측된 **수요 신호**가 가리키는 선행 read 집합.

    넷 중 하나라도 서면 그 read가 수요된 것으로 본다(설계서 §1.7):
      ⒜ 그 read를 요구하는 의존 도구를 이미 시도했다        — 호출 이력(닫힘)
      ⒝ 우리 층이 그 결손을 이미 통지했다                   — `[READ-FIRST]`
      ⒞ 레코드를 못 찾았다는 환경 오류를 받았다              — `not found`(tau2 env 공통 관용구)
      ⒟ **우리 요건 큐가 이 sim에서 그 read를 이름으로 요구했다** — 원천(2026-08-08·C330)
    수요가 없으면 고정하지 않는다. 무제한으로 두면 194/194 sim에서 발화한다(§1.7).

    ★왜 ⒟를 더했나 — 전수 재현(라이브 2 sim에 이 게이트를 그대로 돌림)에서 ⒜⒝⒞가
      **셋 다 원리적으로 못 뜬다**는 것이 나왔다. 표적 read는 나머지 조건을 전부 통과하는데
      (피의존 4·read 접두·미실행·레지스트리 유일 해소) 수요 하나에서 막혀 **핀이 시도조차
      되지 않았다**(라이브 발화 0):
        ⒜ 이 계열의 의존 도구는 **손님이 실행**한다. 이 함수는 의도적으로 assistant 호출만
           세므로(손님 실행을 신호로 쓰면 이미 늦다) 집합에 영영 들어오지 않는다.
        ⒝ 찾는 문자열 `[READ-FIRST]`가 **현 피드백 어휘에 없다**(우리는 `[ORDER]`·
           `[CHECK-FIRST]`·`[PRE-ACTION-KB]`를 쓴다). 게다가 여기서 훑는 것은 커밋된
           tool 메시지인데 우리 통지는 **비커밋 생성-채널**이라 `messages`에 나타나지 않는다
           — 문자열을 고쳐도 채널이 어긋난 채다(궤적 태그 스캔 0건으로 확인).
        ⒞ tau2 실제 문구는 `No records found in '...'` 이고 `"not found"` 부분문자열에
           **안 걸린다**. "env 공통 관용구" 가정이 이 문형에서 틀렸다.
      ⇒ 프록시를 늘리는 대신 **원천**을 받는다. 요건 큐(`t2_dominance.requirements_for`)가
        내는 `reads:` 요건의 `satisfiers`가 곧 *"우리가 지금 이 read를 요구하고 있다"* 이다.
      ⚠대가: 조준이 넓어진다. 남는 억제는 `피의존≥2`·1회/sim·미실행 조건뿐이므로 §1.7이
        경고한 과발화를 `x74` 분할 검증으로 다시 재야 한다([[57]] 부정통제).
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
    out |= {r for r in all_reads if r in set(declared or ())}   # ⒟
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


def _mutates(orch, name):
    """이 도구가 상태를 바꾸는가 — **레지스트리에 묻는다**(이름 접두로 짐작하지 않는다).

    판정 불가면 True(안전측=강제하지 않음). write 강제 금지의 근거는 이름이 아니라 이 성질이다:
    틀린 read는 한 턴 낭비지만 틀린 write는 되돌릴 수 없고 replay가 재실행·바이트 대조한다.
    """
    env = getattr(getattr(orch, "_t2_orch", None), "environment", None) or \
        getattr(orch, "environment", None)
    if env is None or not name:
        return True
    for tk in (getattr(env, "tools", None), getattr(env, "user_tools", None)):
        if tk is None:
            continue
        try:
            if hasattr(tk, "has_tool") and tk.has_tool(name):
                try:
                    return bool(tk.tool_mutates_state(name))
                except Exception:
                    return True
        except Exception:
            continue
    return True


def pin_for(orch, am, a2, messages):
    """이번 재생성에서 고정할 (도구, 인자, 값). 조건 미충족이면 None.

    ★C331(2026-08-08·사용자 지시 *"꼭 실행해야 하는 선행 의존성은 DAG로 하기로 했지 않나,
      그대로 실행하게 하라"*): **표적을 더 이상 짐작하지 않는다.** 선행 그래프가 매 턴
      *"지금 할 일"*을 satisfier **하나**로 특정해 준다(게이트도 마찬가지다 —
      GB1→`verify_identity`, GB3→`get_referrals_by_user`, reads:→그 read). gate_patch가 그
      머리를 `_t2_demanded_step`에 심고, 여기서는 그것을 그대로 쓴다.

    그래서 옛 규칙 셋을 **지웠다**. 전부 표적을 짐작하던 시절의 조준 보조물이었다:
      · 수요 신호 프록시 3종 — 이 계열에서 원리적으로 못 뜬다는 것이 전수로 확인됐다(C330)
      · `피의존≥2`          — 여러 선언이 요구하면 진짜겠지, 라는 짐작. 큐가 이름을 주니 불요
      · 1회/sim 캡          — 잘못 쏠까 봐 둔 것. 표적이 확정이면 **끝날 때까지** 유지가 맞다.
        실제로 이번 실패가 그 형태였다: 권고 턴(24·28)이 전부 무호출인데 캡이 있으면 거기서
        못 건다. 멈춤 조건은 캡이 아니라 **그 호출이 실제로 나왔는가**다.

    남는 배제는 성질뿐이다:
      · 이미 실행됐다 → 해제(자연스러운 종료)
      · **상태를 바꾼다** → 강제하지 않는다(등대 §1.5 write 강제 금지). 이름 접두가 아니라
        레지스트리의 `mutates_state`로 판정한다 — 이름은 규약이고 성질은 사실이다.
      · 이름을 부를 수 없다 → 고정 포기(에이전트 도구도 아니고 레지스트리서 유일 해소도 안 되면
        손님이 실행하는 단계이거나 모호한 것이다. 손님 모델의 디코딩은 우리가 못 건드린다).

    반환 형태 둘:
      · 에이전트 일반 도구      → `(이름, None, None)`      = 지목만(날조할 인자가 없다)
      · discoverable(디스패처)  → `(unlock, 인자, 실명)`     = 지목 + 단일값 enum
    """
    if not enabled():
        return None
    step = getattr(orch, "_t2_demanded_step", None)
    if not step:
        return None
    # 방금 생성된 호출도 실행 집합에 넣는다 — 그 턴에 이미 부른 단계를 다시 고정하지 않도록.
    if _fam(step) in _called_fams(list(messages or []) + [am]):
        return None
    if _mutates(orch, step):
        return None
    own = {getattr(t, "name", None) for t in (getattr(orch, "tools", None) or [])}
    if step in own:                       # 에이전트가 직접 부를 수 있는 도구 = 지목만으로 충분
        return (step, None, None)
    unlock = ((a2 or {}).get("eplan") or {}).get("unlock_tool")
    full = _resolve(orch, step)
    if unlock and full:                   # discoverable = 안쪽 이름을 enum으로 함께 고정
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
    """해당 도구의 `arg_name`을 단일값 enum으로 고정한 도구 목록. 실패하면 None.

    ★`arg_name=None` = **스키마를 건드릴 것이 없다**(2026-08-08·C331). 표적이 에이전트의
      일반 도구면 이름 지목만으로 유일하게 정해진다 — 안쪽 이름을 인자로 실어 나르는
      디스패처가 아니라서 날조할 자리가 없다. 그 경우 도구 목록을 그대로 돌려주고
      지목(`choice`)만 건다.
    """
    if arg_name is None:
        return list(tools or []) if any(getattr(t, "name", None) == tool_name
                                        for t in (tools or [])) else None
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
        "get_all_user_accounts_by_user_id": "get_all_user_accounts_by_user_id_3847"}.get(base)
    a2 = {"eplan": {"unlock_tool": "unlock_discoverable_agent_tool"}}
    os.environ["T2_PIN_READ"] = "1"
    GATE = ("unlock_discoverable_agent_tool", "agent_tool_name",
            "get_all_user_accounts_by_user_id_3847")

    class _Reg:                      # env 레지스트리 대역 — 성질은 여기서 읽는다
        def __init__(self, muts):
            self._m = dict(muts)

        def has_tool(self, n):
            return n in self._m

        def tool_mutates_state(self, n):
            return self._m[n]

    class _Env:
        def __init__(self, muts):
            self.tools = _Reg(muts)
            self.user_tools = None

    MUTS = {"get_all_user_accounts_by_user_id": False, "verify_identity": False,
            "close_bank_account": True}

    def _orch(step=None, own=(), muts=None):
        o = _Orch()
        o.environment = _Env(MUTS if muts is None else muts)
        o.tools = [_T(n, {}) for n in own]
        if step:
            o._t2_demanded_step = step
        return o

    am0 = _AM([])

    # ★핵심: 그래프가 특정한 단계를 **호출 0인 턴**에서 고정한다(옛 프록시·조준 규칙 없음).
    assert M.pin_for(_orch("get_all_user_accounts_by_user_id"), am0, a2, []) == GATE
    print("  ok   그래프가 준 단계를 그대로 고정(discoverable = 지목 + 단일값 enum)")

    assert M.pin_for(_orch(), am0, a2, []) is None
    print("  ok   그래프가 단계를 특정하지 않으면 무발화")

    # 에이전트 일반 도구는 **지목만**으로 유일하다 — 날조할 인자가 없다
    assert M.pin_for(_orch("verify_identity", own=("verify_identity",)), am0, a2, []) \
        == ("verify_identity", None, None)
    print("  ok   일반 도구는 지목만(게이트 단계도 강제 대상 — read 전용 아님)")

    done = [_M([_TC("call_discoverable_agent_tool",
                    {"agent_tool_name": "get_all_user_accounts_by_user_id_3847"})])]
    assert M.pin_for(_orch("get_all_user_accounts_by_user_id"), am0, a2, done) is None
    print("  ok   그 호출이 실제로 나오면 해제(멈춤 조건 = 캡이 아니라 실행)")

    # ★write 배제의 근거는 이름이 아니라 **레지스트리가 말하는 성질**이다
    assert M.pin_for(_orch("close_bank_account"), am0, a2, []) is None
    print("  ok   상태를 바꾸는 단계는 고정하지 않는다(등대 §1.5·이름 접두 아님)")

    assert M.pin_for(_orch("unknown_tool", muts={}), am0, a2, []) is None
    print("  ok   부를 수 없는 이름(손님 실행·미해소)은 고정 포기")

    os.environ["T2_PIN_READ"] = "0"
    assert M.pin_for(_orch("get_all_user_accounts_by_user_id"), am0, a2, []) is None
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

    assert M.tools_with_pin(tools, "KB_search_bm25", None, None) is not None
    assert M.tools_with_pin(tools, "no_such_tool", None, None) is None
    print("  ok   arg_name=None = 스키마 무변경(일반 도구 지목 경로)")
    print("PASS (10/10)")


if __name__ == "__main__":
    selftest()
