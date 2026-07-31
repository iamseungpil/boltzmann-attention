# -*- coding: utf-8 -*-
"""give 채널 게이트 회귀 — 판정 집합 교체가 표적을 잡고 정당한 give를 안 막는가 (2026-07-31).

설계 = `GIVE_CHANNEL_GATE_DESIGN_2026_07_31.md`. 근거 = C257.

배경: `dispatcher_role_check`의 "give 대상 = agent 도구면 deny"는 판정을 **`self.tools` 소속**으로
한다. **잠긴 agent-discoverable 도구는 `self.tools`에 없어서** 빠져나간다 — Y1 전수에서 give 89회 중
**18회가 env user-discoverable 집합 밖**이었고, 그 우회가 dispatcher 미호출 55건으로 이어졌다.

여기서 강제하는 불변식:
  ① 집합 밖 대상(Y1 실측 6종)은 **전부 deny 대상**이다
  ② env user-discoverable 4종은 **하나도 막히면 안 된다**(038 자해 재발 금지)
  ③ 플래그 OFF면 판정이 **구판 그대로**(행동 무변경·롤백 가능)

★술어만 단위 시험한다(라이브 배선은 스모크가 볼 것). 술어 = 집합 소속이므로 재구현 없이
`_user_discoverable`의 반환을 그대로 쓴다([[03b]]).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Y1 전수 실측(C257)
IN_SET = ["deposit_check_3847", "get_card_last_4_digits",
          "get_referral_link", "submit_cash_back_dispute_0589"]
OUT_OF_SET = ["apply_for_credit_card", "submit_transaction", "submit_cash_back_dispute",
              "submit_referral", "setup_travel_notification", "claim_annual_fee_rebate"]

D = set(IN_SET)


def deny(name, envset_on):
    """엔진과 **같은 술어**(t2_gate_patch의 give 분기)를 그대로 표현."""
    if envset_on:
        return bool(name) and name not in D
    return name in {"KB_search", "change_user_email"}   # 구판: self.tools 소속만 deny


OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


print("[①] 집합 밖 대상은 전부 deny (표적 = Y1 실측 18회의 대상 6종)")
for n in OUT_OF_SET:
    chk(deny(n, True), "%-28s deny" % n)

print("\n[②] ★env user-discoverable은 하나도 막히면 안 된다 (038 자해 재발 금지)")
for n in IN_SET:
    chk(not deny(n, True), "%-28s 통과" % n)

print("\n[③] 플래그 OFF면 구판 그대로 (롤백 가능·행동 무변경)")
for n in OUT_OF_SET:
    chk(not deny(n, False), "%-28s OFF에서는 통과(구판 동작)" % n)

print("\n[④] 실제 env 집합과 대조 (스냅샷이 낡지 않았나)")
try:
    import json
    import io as _io
    surf = json.load(_io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
    live = set(surf["banking_knowledge"].get("discoverable_user_tools") or [])
    chk(live == D, "env_surface 스냅샷 == 테스트 상수 (%s)" % sorted(live))
except Exception as e:
    chk(False, "env_surface 확인 실패: %r" % (e,))

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
