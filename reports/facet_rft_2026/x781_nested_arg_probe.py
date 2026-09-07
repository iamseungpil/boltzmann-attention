# -*- coding: utf-8 -*-
"""x781 — T2_CHOICE_GROUND / write_arg_enum 의 인자 판독이 디스패처 중첩 인자를 보는가.
격리: 실제 궤적(task_055 turn 63)의 호출 모양 그대로 만들어 엔진 헬퍼 두 개를 부른다.
판정 대상은 엔진 함수뿐이고 프롬프트/모델은 쓰지 않는다([[78]])."""
import sys, os, json
sys.path.insert(0, r"C:\workspace\ba-frft\scripts\distill\tau2")
import t2_gate_patch as G


class TC:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


# 축자: bank_k8143long3_20260904_0839 / task_055 / messages[63]
tc = TC("call_discoverable_agent_tool",
        {"agent_tool_name": "open_bank_account_4821",
         "arguments": '{"user_id": "224959b99e", "account_type": "savings", "account_class": "Gold Account"}'})

print("exact_tool_name =", G._exact_tool_name(tc))
ad = G._args_dict(tc)
print("args_dict keys  =", sorted(ad.keys()))
print("args_dict.get('account_class') =", repr(ad.get("account_class")))
print("args_dict.get('account_type')  =", repr(ad.get("account_type")))

# choice_grounding 술어 재현 (t2_gate_patch.py:15487-15491 축자 구조)
spec = {"tool": "open_bank_account_4821", "arg": "account_class"}
v = str(ad.get(spec["arg"]) or "").strip()
print("CHOICE_GROUND _v_cg =", repr(v), "-> fires?" , bool(v))

# 비교: write_arg_grounding 은 같은 자리에서 unwrap 한다 (:2116-2129)
inner = {}
for vv in ad.values():
    if isinstance(vv, str) and vv.strip().startswith("{"):
        try:
            j = json.loads(vv)
            if isinstance(j, dict):
                inner.update(j)
        except Exception:
            pass
print("WAG inner.get('account_class') =", repr(inner.get("account_class")))
