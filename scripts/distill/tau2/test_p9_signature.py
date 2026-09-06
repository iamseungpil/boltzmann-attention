# -*- coding: utf-8 -*-
"""★P9 — `give_discoverable_user_tool` 서명 선언이 **서버 실물과 일치**한다 (2026-09-06).

무엇이 있었나. A2 가 이 도구의 허용 인자를 `["discoverable_tool_name"]` 하나로 선언했고,
`T2_TOOL_SIGNATURE` 가 `arguments` 를 실은 호출을 반려했다. 그런데 서버는 그 인자를 **받는다**:

    tools.py:533-534
      def give_discoverable_user_tool(self, discoverable_tool_name: str,
                                      arguments: str = "{}") -> str:
    tools.py:551-570 (본체)
      args_dict = json.loads(arguments)
      sig = inspect.signature(method)
      for arg_name in args_dict:
          if arg_name not in sig.parameters and arg_name != "self":
              return f"Error: Unexpected parameter: {arg_name}"

⇒ 서버가 그 인자를 파싱해 **실제 사용자-도구 시그니처와 대조**한다. 우리보다 정확하다.

⚠선언의 출처(정책 인용)는 **실재한다** — `prompts/components/additional_instructions.md` 축자
  *"Use the `give_discoverable_user_tool(discoverable_tool_name)` function"*.
  틀린 것은 관측이 아니라 **추론**이다: 그 문장은 «설명만 말고 실제로 불러라» 이지 인자 목록의
  전수 규정이 아니다. `task_005` 센티널과 같은 형상(관측 맞음·추론 틀림).

⊖해악 실측(Q3.8 280 sim): 반려 1,140 중 **143건이 이 하나**(고유 조합 1개). 발화 43 sim
  통과율 **28%**(전체 44.6%). `task_015` 사망 · `task_022` **130 스텝·2h46m 폭주**.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(name, ok, detail=""):
    if not ok:
        FAILS.append(name)
    print("%-34s %s%s" % (name, "ok" if ok else "FAIL", (" | " + str(detail)[:80]) if detail else ""))
    return ok


def _sig(layer):
    p = os.path.join(HERE, "a2", "banking_knowledge.%s.json" % layer)
    d = json.load(io.open(p, encoding="utf-8"))
    return (d.get("tool_signatures") or {}).get("give_discoverable_user_tool")


def t_declares_both(sig=None):
    """선언이 서버가 받는 두 인자를 담는다."""
    for layer in ("specific", "gate"):
        s = sig if sig is not None else _sig(layer)
        check("A_%s_has_arguments" % layer, "arguments" in (s or []), s)
        check("A_%s_has_toolname" % layer, "discoverable_tool_name" in (s or []), s)


def t_predicate_lets_it_through(sig=None):
    """술어가 그 호출을 통과시킨다 — 순수 함수로 직접 검산."""
    sys.path.insert(0, HERE)
    import t2_signature as S
    a2 = {"tool_signatures": {"give_discoverable_user_tool": sig if sig is not None else _sig("gate")}}
    real = {"discoverable_tool_name": "get_referral_link", "arguments": '{"user_id":"x"}'}
    v = S.signature_violation("give_discoverable_user_tool", real, a2, force=True)
    check("B_real_call_allowed", v is None, v or "")
    # ★레버가 죽지 않았는지 — 진짜 엉뚱한 키는 여전히 잡아야 한다
    bogus = dict(real, nonsense_key="x")
    v2 = S.signature_violation("give_discoverable_user_tool", bogus, a2, force=True)
    check("B_bogus_key_still_denied", v2 is not None and "nonsense_key" in (v2 or ""), (v2 or "")[:70])


if __name__ == "__main__":
    t_declares_both()
    t_predicate_lets_it_through()

    # ── ★부정통제 ([[57]]) — 구판 선언으로 돌리면 **실패해야** 한다 ──────────────
    print("\n--- 부정통제: 구판 선언(인자 1개)으로 재검정 ---")
    before = len(FAILS)
    OLD = ["discoverable_tool_name"]
    t_declares_both(OLD)
    t_predicate_lets_it_through(OLD)
    neg = len(FAILS) - before
    if neg > 0:
        print("부정통제 OK — 되돌리면 %d 항목이 실패한다 (이 검정은 무의미하지 않다)" % neg)
        del FAILS[before:]
    else:
        FAILS.append("NEGATIVE_CONTROL_VACUOUS")
        print("⛔부정통제 실패 — 되돌려도 통과한다. 이 검정은 아무것도 안 지킨다.")

    print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (P9 계약 + 부정통제)")
    sys.exit(1 if FAILS else 0)
