# -*- coding: utf-8 -*-
"""읽기 루틴 검정 — 남은 조회가 여럿이면 그 집합으로 채널을 좁힌다 (2026-08-18).

사용자 지시 축자: *"남은게 3개면 3개 도구를 루틴으로 연속으로 하면 되지 않나?"* · *"안내는 3개 다 하고,"*

핀으로 박는 것:
  ① 남은 것이 **전부 read** 이고 잠겨 있으면 → **잠금해제** 도구로, 잠긴 이름들을 함께 지목
     (050 에서 그 넷 = 정책 Step 2 의 *"You MUST check ALL of the following"* 그대로:
      대기기간·분쟁 이력·미수령 교체카드·납부이력)
  ② 전부 read 이고 전부 풀려 있으면 → 호출 도구로, 그 집합을 지목
  ③ **write 가 하나라도 섞이면 침묵** (§1.5 Q5 쓰기 강제 금지) — 050 의 제출 단계가 그 경우다
  ④ 절차가 없거나 선언이 없으면 침묵(미선언 도메인 거동 불변)
  ⑤ 엔진에 **도메인 도구 이름 리터럴 0** — 이름은 **A3**(L3 `<domain>.specific.json`)
     `dispatcher_role_check` 에서 읽는다([[05]] 1)

⚠오프라인·모델 호출 0. 절차 선언은 **실물 A3**(`banking_knowledge.specific.json`)를 쓴다
(대역 선언으로 통과시키지 않는다).
"""
import io
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                       encoding="utf-8"))

READS = {"get_user_dispute_history_7291", "get_pending_replacement_orders_5765",
         "get_credit_limit_increase_history_4829", "get_payment_history_6183",
         "get_credit_card_accounts_by_user"}


class _Fn(object):
    def __init__(self, kind):
        self.__tool_type__ = kind


class _TK(object):
    def __init__(self):
        self.tools = {n: _Fn("ToolType.READ") for n in READS}
        self.tools["submit_credit_limit_increase_request_7392"] = _Fn("ToolType.WRITE")
        self.tools["approve_credit_limit_increase_5847"] = _Fn("ToolType.WRITE")


class _Env(object):
    def __init__(self):
        self.tools = _TK()


class _Agent(object):
    def __init__(self):
        self._t2_orch = types.SimpleNamespace(environment=_Env())


class _M(object):
    """assistant 호출 + 성공 tool 응답 한 쌍 (executed 로 세어지려면 둘 다 필요)."""
    def __init__(self, role, tcs=(), mid=None, content=""):
        self.role, self.content, self.id, self.error = role, content, mid, False
        self.tool_call_id = mid
        self.tool_calls = list(tcs)


def _call(name, i, inner=None):
    args = {"agent_tool_name": name}
    if inner:
        args["arguments"] = json.dumps(inner)
    return types.SimpleNamespace(id="c%d" % i, name="call_discoverable_agent_tool",
                                 arguments=args, requestor="assistant")


def _hist(pairs):
    """[(효력 도구명, i)] → 성공 호출로 채운 메시지 목록."""
    out = []
    for nm, i in pairs:
        out.append(_M("assistant", [_call(nm, i)]))
        out.append(_M("tool", (), mid="c%d" % i, content="ok"))
    return out


def _unlock(names, i0=90):
    out = []
    for k, n in enumerate(names):
        tc = types.SimpleNamespace(id="u%d" % (i0 + k), name="unlock_discoverable_agent_tool",
                                   arguments={"agent_tool_name": n}, requestor="assistant")
        out.append(_M("assistant", [tc]))
        out.append(_M("tool", (), mid="u%d" % (i0 + k), content="Tool unlocked: %s" % n))
    return out


def main():
    bad = 0
    ag = _Agent()

    # 컷 A — 제출만 끝난 자리: ready = 조회 둘, 아직 잠금
    hA = _hist([("submit_credit_limit_increase_request_7392", 1)])
    pin = GP._read_routine_pin(ag, A2, hA)
    print("① 전부 read·잠김 → %r" % (pin,))
    if not pin or pin[0] != "unlock_discoverable_agent_tool":
        print("   FAIL — 잠긴 이름을 두고 호출을 지목하면 048 livelock 으로 간다"); bad += 1
    elif sorted(pin[2]) != ["get_credit_limit_increase_history_4829", "get_payment_history_6183",
                            "get_pending_replacement_orders_5765", "get_user_dispute_history_7291"]:
        print("   FAIL — 남은 조회 집합이 아니다: %r" % (pin[2],)); bad += 1

    # 컷 B — 같은 자리에서 둘 다 unlock 된 상태
    pin = GP._read_routine_pin(ag, A2, hA + _unlock(
        ["get_user_dispute_history_7291", "get_pending_replacement_orders_5765",
         "get_credit_limit_increase_history_4829", "get_payment_history_6183"]))
    print("② 전부 read·해제됨 → %r" % (pin,))
    if not pin or pin[0] != "call_discoverable_agent_tool" or len(pin[2]) != 4:
        print("   FAIL — 풀린 뒤에는 호출 집합으로 좁혀야 한다"); bad += 1

    # 컷 B2 — 넷 중 둘을 실제로 부른 뒤: 집합에서 **빠진다**(루틴이 줄어든다)
    pin = GP._read_routine_pin(ag, A2, hA + _unlock(
        ["get_user_dispute_history_7291", "get_pending_replacement_orders_5765",
         "get_credit_limit_increase_history_4829", "get_payment_history_6183"])
        + _hist([("get_credit_limit_increase_history_4829", 20),
                 ("get_payment_history_6183", 21)]))
    print("②b 둘 부른 뒤 남은 집합 → %r" % (pin[2] if pin else None,))
    if not pin or sorted(pin[2]) != ["get_pending_replacement_orders_5765",
                                     "get_user_dispute_history_7291"]:
        print("   FAIL — 부른 것이 집합에서 안 빠진다(루틴이 안 줄어든다)"); bad += 1

    # 컷 C — 아무것도 안 한 자리: ready 에 write(제출)가 있다
    pin = GP._read_routine_pin(ag, A2, _hist([("get_credit_card_accounts_by_user", 5)]))
    print("③ write 섞임 → %r" % (pin,))
    if pin is not None:
        print("   FAIL — 쓰기를 강제했다(§1.5 Q5 위반)"); bad += 1

    # 컷 D — 절차 선언이 없는 A2
    pin = GP._read_routine_pin(ag, {"procedures": []}, hA)
    print("④ 절차 미선언 → %r" % (pin,))
    if pin is not None:
        print("   FAIL — 선언 없이 발화했다"); bad += 1

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i = src.find("def _read_routine_pin(")
    body = src[i:i + 2600]
    lit = [n for n in ("call_discoverable_agent_tool", "unlock_discoverable_agent_tool",
                       "agent_tool_name") if '"%s"' % n in body]
    print("⑤ 함수 안 도메인 리터럴: %s" % (lit or "없음"))
    if lit:
        print("   FAIL — 도구 이름을 엔진에 박았다([[05]] 1)"); bad += 1

    print("\n%s" % ("test_read_routine PASS" if not bad else "test_read_routine FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
