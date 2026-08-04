"""P1 — A2가 선언한 **선행 read**를 생성-측에서 한 번 고정한다.

`require_tool_before`는 `open_bank_account` 앞에 `get_all_user_accounts_by_user_id`를 요구하지만
**강제하지 못한다.** deny 스텁이 tool 출력이라 tau2 replay가 깨끗한 env에서 재실행해 내용을 정확
비교하고, 불일치하면 sim 전체가 죽기 때문이다(C210/day6 3건 실측). 그래서 사후 권고로 강등됐고,
N97에서 그 권고의 실발화는 **1회**였다. gold가 이 read를 요구한 신규 실패 59 sim 중 **40이 끝내
호출하지 않았다.**

생성-측 제약은 그 반론이 닿지 않는다 — **tool 출력을 만들지 않으므로** replay가 비교할 것이 없다.
다만 이름만 지정해서는 안 된다는 것이 실측으로 나왔다(`x72` · 서버 n=3):

    tool_choice=auto            → 호출 0 · 순수 산문
    tool_choice=required        → KB_search_bm25 (현행 T2_FORCE_ACTION의 한계)
    named tool_choice (디스패처) → unlock(agent_tool_name="AccountLookupTool")   ← 내부 이름 날조
    named + 단일값 enum          → unlock(agent_tool_name="get_all_..._3847")    ← 3/3 적중

discoverable 도구는 `tools` 배열에 없고 디스패처 인자로 실려가므로, **그 인자를 단일값 enum으로
고정**해야 표적에 닿는다. 평범한 JSON schema이고 vLLM 확장이 아니다.

★[[05]] 경계
  · 표적은 **A2 기존 선언**(`require_tool_before`)에서만 온다 — gold를 보지 않는다([[03b]]).
  · 접미사는 **env 레지스트리 기계도출**([[23]] opex 0) — A2 순증 0.
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

    조건: 방금 생성된 호출 중 A2 `require_tool_before`가 선행 read를 요구하는 것이 있고,
    그 read가 아직 미실행이며, 레지스트리에서 유일하게 해소되고, **read 접두**를 가질 것.
    """
    if not enabled():
        return None
    rb = ((a2 or {}).get("require_tool_before") or {})
    ep = ((a2 or {}).get("eplan") or {})
    unlock = ep.get("unlock_tool")
    if not (rb and unlock):
        return None
    called = _called_fams(messages)
    for tc in (getattr(am, "tool_calls", None) or []):
        eff = _fam(getattr(tc, "name", None))
        a = getattr(tc, "arguments", None)
        if isinstance(a, dict):
            inner = (a.get("agent_tool_name") or a.get("discoverable_tool_name")
                     or a.get("user_tool_name"))
            if inner:
                eff = _fam(inner)
        for need in (rb.get(eff) or []):
            if _fam(need) in called:
                continue
            if not _fam(need).startswith(_READ_PREFIX):
                continue                      # write는 고정하지 않는다([[14]])
            full = _resolve(orch, need)
            if full:
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

    class _M:
        def __init__(self, tcs):
            self.tool_calls = tcs

    import t2_pin_read as M
    M._resolve = lambda orch, base: {"get_all_user_accounts_by_user_id":
                                     "get_all_user_accounts_by_user_id_3847"}.get(base)
    a2 = {"require_tool_before": {"open_bank_account": ["get_all_user_accounts_by_user_id"],
                                  "pay_credit_card": ["close_bank_account"]},
          "eplan": {"unlock_tool": "unlock_discoverable_agent_tool"}}
    os.environ["T2_PIN_READ"] = "1"

    am = _AM([_TC("call_discoverable_agent_tool", {"agent_tool_name": "open_bank_account_4821"})])
    p = M.pin_for(None, am, a2, [])
    assert p == ("unlock_discoverable_agent_tool", "agent_tool_name",
                 "get_all_user_accounts_by_user_id_3847"), p
    print("  ok   선행 read 미실행 → 그 read로 고정")

    hist = [_M([_TC("call_discoverable_agent_tool",
                    {"agent_tool_name": "get_all_user_accounts_by_user_id_3847"})])]
    assert M.pin_for(None, am, a2, hist) is None
    print("  ok   이미 읽었으면 무발화")

    am_w = _AM([_TC("pay_credit_card", {})])
    assert M.pin_for(None, am_w, a2, []) is None
    print("  ok   선행이 write면 고정하지 않는다 ([[14]] read만)")

    os.environ["T2_PIN_READ"] = "0"
    assert M.pin_for(None, am, a2, []) is None
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
    print("PASS (6/6)")


if __name__ == "__main__":
    selftest()
