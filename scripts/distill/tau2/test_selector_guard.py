# -*- coding: utf-8 -*-
r"""**도구-선택자 슬롯 가드가 치환 경로에 닿는가** (R3·2026-08-24).

## 무엇을 못 박는가

`T2_GROUND` 는 날조로 판정된 인자를 문맥의 값으로 **제자리 치환**한다. 그 판정은
`_first_fab_call` 이 하는데, 여기에는 *도구-선택자 슬롯 예외*가 **없었다** — 예외는
`_prov_scan_args`(= `T2_PROVENANCE` 전용이고 어느 런처도 그 변수를 export 하지 않는
死경로)에만 있었다. 결과: 뱅킹 코퍼스 치환 **371건이 전부 선택자 키**(`agent_tool_name` 365 ·
`discoverable_tool_name` 6)였고 **정답 도구명을 낸 것은 0건** — 우변은 전부 대문자 인명이었다
(`CARLOS RODRIGUEZ` 91 · `LIANG JINHAI` 86 …). 반면 리테일에서는 같은 기구가
`address1`/`zip`/`address2`/`item_ids` 에 **78건 정상** 발화한다.

⇒ 수리는 ⑴ 가드를 **치환하는 경로에서 닿게** 하고 ⑵ '선택자인가'를 **env 스키마에서 도출**한다
(4-이름 하드코딩 튜플 아님). 이 검정이 고정하는 것:

  ⒜ **도출이 도메인-일반**이다 — 이름을 모르는 합성 스키마(`target_tool_name`)도 잡고,
     페이로드(`arguments`) 파라미터를 가진 도구가 없는 리테일에서는 집합이 **∅** 이다.
  ⒝ **양성대조(수리 전 결함 재현)** — `selectors=∅`(종전 기본값)이면 `agent_tool_name` 이
     날조로 잡히고 `_grounded_candidates` 가 **고객 이름**을 유일 후보로 내며 실제로 치환된다.
  ⒞ **수리 후** — 같은 입력에 도출 집합을 넘기면 fab 은 None 이 되어 치환 경로에 못 들어간다.
  ⒟ **부정통제 1(리테일 78건 보존)** — 리테일 데이터 인자(`address1`/`zip`/`item_ids`)는
     수리 후에도 그대로 잡히고 그대로 치환된다.
  ⒠ **부정통제 2(뱅킹 데이터 인자 보존)** — 선택자가 아닌 `user_id`/`transaction_id` 날조는
     수리 후에도 그대로 잡힌다. 즉 이 수리가 파는 것은 **도구명 슬롯 하나**뿐이다.
  ⒡ **배선 래칫(死경로 재발 방지)** — 소스의 모든 `_first_fab_call(` 호출이 `selectors=` 를
     넘기는지 확인한다. 이 결함의 본체가 *"가드는 있는데 호출되는 경로에 없었다"* 였다.

실행: `PYTHONIOENCODING=utf-8 py -3 test_selector_guard.py`
"""
import io
import json
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_gate_patch as G                                          # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


# ─────────────────────────── 최소 shim (엔진이 읽는 면만) ───────────────────────────
class Tool(object):
    """`openai_schema` 만 읽히는 tau2 Tool 의 최소 대역."""

    def __init__(self, name, props):
        self.name = name
        self.openai_schema = {"type": "function", "function": {
            "name": name,
            "parameters": {"type": "object", "properties": props}}}


class TC(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        self.id = name


class AM(object):
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class Msg(object):
    def __init__(self, role, content, error=False):
        self.role = role
        self.content = content
        self.error = error


class Holder(object):
    def __init__(self, tools):
        self.tools = tools


S = {"type": "string"}
I = {"type": "integer"}
OPT_S = {"anyOf": [{"type": "string"}, {"type": "null"}]}

# tau2 banking 형상 (발견형 디스패처 + 잠금 도구 + 평범한 read)
BANK_TOOLS = [
    Tool("call_discoverable_agent_tool", {"agent_tool_name": S, "arguments": S}),
    Tool("unlock_discoverable_agent_tool", {"agent_tool_name": S}),
    Tool("give_discoverable_user_tool", {"discoverable_tool_name": S, "arguments": S}),
    Tool("get_user_information_by_id", {"user_id": S}),
]

# tau2 retail 형상 (페이로드 파라미터를 가진 도구가 하나도 없다)
RETAIL_TOOLS = [
    Tool("modify_user_address", {"user_id": S, "address1": S, "address2": S,
                                 "city": S, "state": S, "country": S, "zip": S}),
    Tool("modify_pending_order_items", {"order_id": S, "item_ids": {"type": "array"},
                                        "new_item_ids": {"type": "array"},
                                        "payment_method_id": S}),
    Tool("get_order_details", {"order_id": S}),
]


def main():
    print("\n[⒜] 선택자 술어가 **스키마에서** 도출되는가 (도메인-일반)")
    sel_bank = G.selector_args_of(Holder(BANK_TOOLS))
    check(sel_bank == frozenset({"agent_tool_name", "discoverable_tool_name"}),
          "banking 도출 = {agent_tool_name, discoverable_tool_name} · 실제 %s" % sorted(sel_bank))

    sel_retail = G.selector_args_of(Holder(RETAIL_TOOLS))
    check(sel_retail == frozenset(),
          "retail 도출 = ∅ (페이로드 파라미터를 가진 도구 0) · 실제 %s" % sorted(sel_retail))

    # 하드코딩 목록이 아니라 **구조**로 잡는다 — 엔진이 이름을 모르는 합성 도메인
    synth = [Tool("dispatch_capability", {"target_tool_name": S, "arguments": S})]
    check(G.selector_args_of(Holder(synth)) == frozenset({"target_tool_name"}),
          "이름을 모르는 합성 디스패처(`target_tool_name`)도 구조로 도출된다(전이)")

    # 정수 형제는 라우팅 후보가 아니다 → 문자열 슬롯이 유일하면 도출
    withnum = [Tool("dispatch_with_timeout", {"target_tool_name": S, "timeout_s": I,
                                              "arguments": S})]
    check(G.selector_args_of(Holder(withnum)) == frozenset({"target_tool_name"}),
          "비-문자열 형제(정수)는 후보가 아니다")

    # Optional[str] 도 문자열로 센다 → 문자열 슬롯 2개 = **기권**(모르면 안 뺀다)
    ambi = [Tool("dispatch_ambiguous", {"target_tool_name": S, "note": OPT_S, "arguments": S})]
    check(G.selector_args_of(Holder(ambi)) == frozenset(),
          "라우팅 슬롯이 유일하지 않으면 **기권**한다(종전 거동 유지)")

    check(G.selector_args_of(Holder([])) == frozenset(), "도구 0 = ∅ (예외 0)")
    check(G.selector_args_of(object()) == frozenset(), "스키마를 못 얻으면 ∅ (예외 0)")

    # ── 뱅킹 실물 형상: 지어낸 도구명 + 문맥에는 고객 레코드만 ──
    fake_tool = "update_debit_card_order_9834"
    record = {"user_id": "f7d3a82c91", "name": "CARLOS RODRIGUEZ",
              "account_id": "chk_b4d92f7c28"}
    msgs = [Msg("user", "please update my debit card order"),
            Msg("tool", json.dumps(record))]
    ctx = G._ctx_from_messages(msgs)
    inner = {"account_id": "chk_b4d92f7c28"}
    call = TC("call_discoverable_agent_tool",
              {"agent_tool_name": fake_tool, "arguments": json.dumps(inner)})
    am = AM([call])

    print("\n[⒝] 양성대조 — 수리 전(가드 미도달 = selectors ∅) 결함 재현")
    fab0 = G._first_fab_call(am, ctx, G.DEFAULT_ARG_HINTS)
    check(fab0 is not None and fab0[1] == "agent_tool_name",
          "선택자 슬롯이 '날조 데이터 인자'로 잡힌다 · 실제 %r" % (fab0[1] if fab0 else None,))
    cands = G._grounded_candidates("agent_tool_name", fake_tool, msgs, lenient=True)
    check(cands == ["CARLOS RODRIGUEZ"],
          "치환 후보가 **고객 이름 하나**로 확정된다(=치환 조건 성립) · 실제 %r" % (cands,))
    if fab0 and len(cands) == 1:
        tc0 = TC(call.name, dict(call.arguments))
        did = G._subst_arg_value(tc0, "agent_tool_name", fake_tool, cands[0])
        check(did and tc0.arguments["agent_tool_name"] == "CARLOS RODRIGUEZ",
              "실제로 치환된다 = 371/371 오작동 기전 그대로 · 결과 %r"
              % (tc0.arguments.get("agent_tool_name"),))

    print("\n[⒞] 수리 후 — 도출 집합을 넘기면 치환 경로에 들어가지 않는다")
    fab1 = G._first_fab_call(am, ctx, G.DEFAULT_ARG_HINTS, selectors=sel_bank)
    check(fab1 is None, "선택자 슬롯만 남은 호출은 fab 없음 · 실제 %r" % (fab1[1] if fab1 else None,))
    check(call.arguments["agent_tool_name"] == fake_tool,
          "원 호출은 손대지 않는다(엔진이 도구를 고르지 않음) — env 가 이름을 판정한다")

    # 잠금 도구(페이로드 형제 없음)의 같은 슬롯도 보호된다
    unlock = AM([TC("unlock_discoverable_agent_tool", {"agent_tool_name": fake_tool})])
    check(G._first_fab_call(unlock, ctx, G.DEFAULT_ARG_HINTS) is not None,
          "양성대조: 잠금 도구의 선택자도 수리 전에는 잡혔다")
    check(G._first_fab_call(unlock, ctx, G.DEFAULT_ARG_HINTS, selectors=sel_bank) is None,
          "수리 후: 페이로드 형제가 없는 잠금 도구의 같은 슬롯도 보호된다")

    print("\n[⒟] 부정통제 1 — 리테일 정상 발화 78건(address1/zip/address2/item_ids) 보존")
    r_rec = {"user_id": "sara_doe_496", "address": {
        "address1": "742 Evergreen Terrace", "address2": "Suite 5",
        "city": "Springfield", "zip": "77243"}}
    r_msgs = [Msg("user", "change my address, user sara_doe_496"),
              Msg("tool", json.dumps(r_rec))]
    r_ctx = G._ctx_from_messages(r_msgs)
    for arg, bad, want in (("address1", "123 Fake Blvd", "742 Evergreen Terrace"),
                           ("address2", "Apt 999", "Suite 5"),
                           ("zip", "99999", "77243")):
        r_call = TC("modify_user_address", {"user_id": "sara_doe_496", arg: bad})
        r_am = AM([r_call])
        fab = G._first_fab_call(r_am, r_ctx, G.DEFAULT_ARG_HINTS, selectors=sel_retail)
        c = G._grounded_candidates(arg, bad, r_msgs, lenient=True)
        ok_sub = len(c) == 1 and G._subst_arg_value(r_call, arg, bad, c[0])
        check(fab is not None and fab[1] == arg and ok_sub
              and r_call.arguments[arg] == want,
              "retail %s: 수리 후에도 잡고 그대로 치환한다 → %r"
              % (arg, r_call.arguments.get(arg)))

    # item_ids(리스트 원소) 도 그대로
    i_rec = {"order_id": "#W2378156", "items": [{"item_id": "1656367028"}]}
    i_msgs = [Msg("user", "swap an item in order #W2378156"), Msg("tool", json.dumps(i_rec))]
    i_ctx = G._ctx_from_messages(i_msgs)
    i_call = TC("modify_pending_order_items",
                {"order_id": "#W2378156", "item_ids": ["9999999999"]})
    fab_i = G._first_fab_call(AM([i_call]), i_ctx, G.DEFAULT_ARG_HINTS, selectors=sel_retail)
    check(fab_i is not None and fab_i[1] == "item_ids",
          "retail item_ids: 수리 후에도 잡힌다 · 실제 %r" % (fab_i[1] if fab_i else None,))

    print("\n[⒠] 부정통제 2 — 뱅킹의 **데이터** 인자 날조는 그대로 잡힌다")
    plain = AM([TC("get_user_information_by_id", {"user_id": "zz9999zzz"})])
    check(G._first_fab_call(plain, ctx, G.DEFAULT_ARG_HINTS, selectors=sel_bank) is not None,
          "선택자가 아닌 user_id 날조는 수리 후에도 잡힌다")
    plain_ok = AM([TC("get_user_information_by_id", {"user_id": "f7d3a82c91"})])
    check(G._first_fab_call(plain_ok, ctx, G.DEFAULT_ARG_HINTS, selectors=sel_bank) is None,
          "문맥에 있는 실값은 여전히 통과한다(over-block 0)")
    mixed = AM([TC("call_discoverable_agent_tool",
                   {"agent_tool_name": fake_tool, "card_id": "card123456"})])
    fab_m = G._first_fab_call(mixed, ctx, G.DEFAULT_ARG_HINTS, selectors=sel_bank)
    check(fab_m is not None and fab_m[1] == "card_id",
          "같은 호출에서 선택자는 건너뛰고 **다음 데이터 인자**를 계속 스캔한다 · 실제 %r"
          % (fab_m[1] if fab_m else None,))

    print("\n[⒡] `_prov_scan_args` 도 같은 도출 집합을 쓴다(사본 0·死경로였던 쪽)")
    keys_none = [k for k, _ in G._prov_scan_args(call)]
    keys_sel = [k for k, _ in G._prov_scan_args(call, selectors=sel_bank)]
    check("agent_tool_name" not in keys_none and "agent_tool_name" not in keys_sel,
          "선택자는 어느 쪽으로도 검사 대상에 안 들어간다 · %r / %r" % (keys_none, keys_sel))
    keys_empty = [k for k, _ in G._prov_scan_args(call, selectors=frozenset())]
    check("agent_tool_name" in keys_empty,
          "∅ 를 명시로 넘기면 예외가 꺼진다 = 집합이 유일한 스위치(하드코딩 아님)")
    check(G._provenance_deny(TC("call_discoverable_agent_tool",
                                {"agent_tool_name": fake_tool,
                                 "arguments": json.dumps(inner)}),
                             ctx, selectors=sel_bank) is None,
          "_provenance_deny 도 선택자 때문에 반려하지 않는다")

    print("\n[⒢] 배선 래칫 — 치환 경로에서 가드가 다시 죽지 않게")
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    calls = [ln.strip() for ln in src.splitlines()
             if "_first_fab_call(" in ln and not ln.lstrip().startswith("#")
             and "def _first_fab_call" not in ln]
    missing = [ln for ln in calls if "selectors=" not in ln]
    check(len(calls) >= 7 and not missing,
          "`_first_fab_call` 호출 %d개 전부 selectors= 를 넘긴다 · 누락 %r" % (len(calls), missing))
    check("_SELECTOR_FALLBACK" in src and src.count("_SELECTOR_FALLBACK") == 2,
          "4-이름 튜플은 오프라인 폴백 1곳에서만 참조된다(권위 = 스키마 도출)")

    print("\n%s" % ("PASS" if not FAIL else "FAIL(%d): " % len(FAIL) + " · ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
