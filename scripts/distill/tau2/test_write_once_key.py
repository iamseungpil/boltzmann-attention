# -*- coding: utf-8 -*-
r"""test_write_once_key - `write_once_keys` 래칫 (t7378 task_074#s361454 수리).

실물: `T2_DUP_WRITE` 는 그 런에서 **켜져 있었는데도** 재적용을 못 막았다. 가드의 키가
`_mut_key_of`(이름+인자 전체)라 같은 계좌에 14.5 를 적용한 뒤 30.0 을 적용하는 것이 **다른 키**
였기 때문이다. 정책은 도구 설명 축자로 "may only be called ONCE per checking account per
customer interaction" 이라고 **계좌**를 유일성 키로 말한다.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP                                              # noqa: E402

FAIL = []


def chk(name, cond, extra=""):
    if cond:
        print("  ok   %s%s" % (name, ("  " + extra) if extra else ""))
    else:
        FAIL.append(name)
        print("  FAIL %s%s" % (name, ("  " + extra) if extra else ""))


class TC(object):
    def __init__(self, name, args, tid="x"):
        self.name = name
        self.arguments = args
        self.id = tid


A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                       encoding="utf-8"))


def credit(acct, amount):
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": "apply_checking_account_credit_5829",
               "arguments": json.dumps({"account_id": acct, "amount": amount,
                                        "credit_type": "fee_refund"})})


print("[1] 선언이 세 사본에 있다")
for rel in ("banking_knowledge.gate.json", "banking_knowledge.specific.json",
            os.path.join("split", "banking_knowledge.core.json")):
    p = os.path.join(HERE, "a2", rel)
    d = json.load(io.open(p, encoding="utf-8"))
    chk("%s 에 write_once_keys" % os.path.basename(rel), bool(d.get("write_once_keys")))
    if d.get("write_once_keys"):
        chk("%s: 키가 계좌다" % os.path.basename(rel),
            "account_id" in (d["write_once_keys"][0].get("keys") or []))

print("")
print("[2] 실물 시나리오 - 같은 계좌·다른 금액")
a = credit("chk_ar72c5d8e3_2", 14.5)
b = credit("chk_ar72c5d8e3_2", 30.0)
ka, kb = GP._once_key_of(a, A2), GP._once_key_of(b, A2)
chk("좁힌 키가 나온다", bool(ka), str(ka))
chk("같은 계좌면 **같은 키**", ka == kb)
chk("종전 키는 서로 다르다(그래서 못 막았다)", GP._mut_key_of(a) != GP._mut_key_of(b))

print("")
print("[3] 경계")
c = credit("chk_ar72c5d8e3_3", 4.75)
chk("다른 계좌면 다른 키", GP._once_key_of(c, A2) != ka)
other = TC("call_discoverable_agent_tool",
           {"agent_tool_name": "get_bank_account_transactions_9173",
            "arguments": json.dumps({"account_id": "chk_ar72c5d8e3_2"})})
chk("접두가 안 맞는 도구는 None", GP._once_key_of(other, A2) is None)
chk("선언이 없으면 None (fail-open)", GP._once_key_of(a, {}) is None)
chk("a2 가 None 이어도 안 터진다", GP._once_key_of(a, None) is None)

print("")
print("[4] 문면 - 좁힌 키로 막을 때는 '같은 인자'라고 말하지 않는다 ([[25]])")
chk("전용 문면이 있다", hasattr(GP, "_DUP_WRITE_ONCE_FB"))
if hasattr(GP, "_DUP_WRITE_ONCE_FB"):
    t = GP._DUP_WRITE_ONCE_FB
    chk("'same arguments' 를 말하지 않는다", "same arguments" not in t)
    chk("ONCE per target 을 말한다", "ONCE per target" in t)
    chk("자리표시자 두 개", "{at}" in t and "{result}" in t)

print("")
print("RESULT: %s" % ("PASS" if not FAIL else "FAIL (%d) %s" % (len(FAIL), FAIL[:3])))
sys.exit(0 if not FAIL else 1)
