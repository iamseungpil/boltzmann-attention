# -*- coding: utf-8 -*-
r"""test_write_amount_evidence - 크레딧 금액이 **우리 도구가 계산한 총액**인지 검사 (t7378 s626729).

실물: 우리 비교기가 `14.5 · 4.75 · 3.7` 을 내고 반환문이 축자로 *"use it as the credit amount"*
라고 했는데, 모델은 손님이 말한 `9.00 · 1.50 · 1.50` 을 제출했다(손님은 `_err` 행만 세었다).
반경(8런·쓰기 58건): 막힐 것 4건 전부 WRONGARG · 오차단 0.
"""
import io, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_gate_patch as GP                                              # noqa: E402

FAIL = []
def chk(name, cond, extra=""):
    if cond: print("  ok   %s%s" % (name, ("  " + extra) if extra else ""))
    else:
        FAIL.append(name); print("  FAIL %s%s" % (name, ("  " + extra) if extra else ""))

class TC(object):
    def __init__(s, name, args): s.name = name; s.arguments = args; s.id = "x"
class M(object):
    def __init__(s, role, content): s.role = role; s.content = content

ACCT = "chk_ar72c5d8e3_2"
AUDIT = ("ATM withdrawals whose net charge does NOT match ... The signed total of the differences "
         "listed above, computed by this tool, is 14.5 (a negative difference lowers it). That "
         "signed total is the net correction for THIS account - use it as the credit amount.")
MSGS = [M("tool", "Accounts: %s, chk_ar72c5d8e3_3" % ACCT), M("tool", AUDIT)]

def credit(amount):
    return TC("call_discoverable_agent_tool",
              {"agent_tool_name": "apply_checking_account_credit_5829",
               "arguments": json.dumps({"account_id": ACCT, "amount": amount,
                                        "credit_type": "fee_refund"})})

print("[1] 토큰 확장 - {arg:NAME} 과 숫자 표기")
ex = GP._wev_expand(["computed by this tool, is {arg:amount}"], {"amount": 14.5}, ACCT)
chk("여러 표기를 만든다", any("is 14.5" in t for t in ex) and any("is 14.50" in t for t in ex), str(sorted(ex)[:3]))
chk("{id} 도 계속 된다", GP._wev_expand(["for {id}"], {}, ACCT) == ["for %s" % ACCT])
chk("인자가 없으면 토큰을 버린다(오차단 회피)", GP._wev_expand(["is {arg:nope}"], {"amount": 1}, ACCT) == [])

print("")
print("[2] 선언이 세 사본에 있다")
specs_by_layer = {}
for rel in ("banking_knowledge.gate.json", "banking_knowledge.specific.json",
            os.path.join("split", "banking_knowledge.core.json")):
    d = json.load(io.open(os.path.join(HERE, "a2", rel), encoding="utf-8"))
    sp = [s for s in (d.get("write_evidence_specs") or [])
          if (s.get("applies_when") or {}).get("prefix") == "apply_checking_account_credit"]
    specs_by_layer[rel] = sp
    chk("%s 에 크레딧 금액 스펙" % os.path.basename(rel), len(sp) == 1)
vals = [json.dumps(v, sort_keys=True, ensure_ascii=False) for v in specs_by_layer.values()]
chk("세 사본이 축자 동일 ([[24]])", len(set(vals)) == 1)

SPECS = specs_by_layer["banking_knowledge.gate.json"]

print("")
print("[3] 실물 판정")
chk("우리가 계산한 14.5 는 통과", GP._wev_deny_msgs(MSGS, credit(14.5), SPECS) is None)
chk("표기가 달라도 통과 (14.50)", GP._wev_deny_msgs(MSGS, credit(14.50), SPECS) is None)
d9 = GP._wev_deny_msgs(MSGS, credit(9.0), SPECS)
chk("손님이 말한 9.0 은 막힌다", d9 is not None)
if d9:
    chk("문면이 고칠 방법을 댄다 ([[64]])", "signed total" in d9 and ACCT in d9)
chk("옛 오답 5.0 도 막힌다", GP._wev_deny_msgs(MSGS, credit(5.0), SPECS) is not None)

print("")
print("[4] 오차단 회피")
chk("감사 결과가 없으면 계좌 id 부재로 skip 되지 않는다(막는다)",
    GP._wev_deny_msgs([M("tool", "Accounts: %s" % ACCT)], credit(14.5), SPECS) is not None)
other = TC("call_discoverable_agent_tool",
           {"agent_tool_name": "get_bank_account_transactions_9173",
            "arguments": json.dumps({"account_id": ACCT})})
chk("다른 도구는 이 스펙이 안 건드린다", GP._wev_deny_msgs(MSGS, other, SPECS) is None)

print("")
print("RESULT: %s" % ("PASS" if not FAIL else "FAIL (%d) %s" % (len(FAIL), FAIL[:3])))
sys.exit(0 if not FAIL else 1)
