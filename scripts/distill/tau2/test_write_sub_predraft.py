# -*- coding: utf-8 -*-
"""사전(pre-draft) sync 서브 배선 검정 (`T2_WRITE_SUB=2`·사용자 지시 2026-08-14).

왜 사전 자리인가: 종전 배선은 **초안을 본 뒤**에만 말할 수 있어 옳은 시점의 27%만 잡았고,
나머지에 끼어들면 그 턴 호출을 버리는데 그중 **33%가 버리면 안 되는 것**(write 23%·새 read 10%)
이었다(감사 25 sim·101 호출). 사전 자리에는 버릴 초안이 없다.

검정 4칸(모두 순수 함수 수준 — 라이브 루프는 스모크로 별도 확인):
  T1 근거가 바뀌면 서브를 부르고 제안을 work 에 얹는다
  T2 Δspurious: **같은 근거**면 두 번째부터 침묵([[57]] 인자-변화)
  T3 Δspurious: 이미 성공 실행된 도구는 후보 이름에서 빠진다(중복 write 방지·U1 계열)
  T4 플래그 미설정(또는 =1)이면 사전 자리는 **아무 것도 안 한다**(종전 경로 보존)
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_subcall as SC                                           # noqa: E402
import t2_resolve as R                                            # noqa: E402

BASIS1 = ("net correction of THIS account = $9.50 for account chk_kj93a7b2e1_1")
BASIS2 = BASIS1 + "\nsecond audit line: = $1.50 for account chk_kj93a7b2e1_3"
NAMES = {"apply_checking_account_credit_5829"}
A2 = {"write_initiation": {"instructions": "I", "answer_format": "F",
                           "delivery_template": "CALLS {calls} BASIS {basis}",
                           "basis_max_chars": 4000, "temperature": 0},
      "eplan": {"unlock_tool": "unlock_x", "dispatch_tool": "call_x", "list_tool": "list_x"}}
FAILS = []


class M(object):
    def __init__(self, role, content="", tool_calls=None, error=False, id=None):
        self.role, self.content, self.tool_calls = role, content, tool_calls
        self.error, self.id = error, id


class TC(object):
    def __init__(self, name, arguments, id="c1"):
        self.name, self.arguments, self.id = name, arguments, id


class UserMessage(object):
    def __init__(self, content="", role="user"):
        self.role, self.content = role, content


class LA(object):
    calls = 0
    def __init__(self, payload):
        self.payload = payload
    def generate(self, model=None, tools=None, messages=None, call_name=None, **kw):
        LA.calls += 1
        return type("S", (), {"content": self.payload})()


class Agent(object):
    llm, llm_args, tools = "m", {}, []


PROP = ('{"calls": [{"tool": "apply_checking_account_credit_5829", '
        '"account_id": "chk_kj93a7b2e1_1", "amount": 9.50}]}')


def chk(name, cond, extra=""):
    print("%-4s %s %s" % ("PASS" if cond else "FAIL", name, extra))
    if not cond:
        FAILS.append(name)


def predraft(agent, msgs, la):
    """라이브 사전 자리와 **같은 순서**의 축소판(배선 로직 동형 — 라이브는 t2_gate_patch)."""
    if os.environ.get("T2_WRITE_SUB") != "2":
        return None
    basis = SC.recent_tool_text(msgs, 4000)
    sig = hash(basis)
    if not basis or sig == getattr(agent, "_t2_write_basis", None):
        return None
    agent._t2_write_basis = sig
    done = R._executed_dispatch_names(msgs, A2)
    return R.sub_write_proposal(agent, la, UserMessage, msgs, A2, NAMES - set(done))


os.environ["T2_WRITE_SUB"] = "2"
ag = Agent()
msgs1 = [M("user", "fix the fees"), M("tool", BASIS1, id="t1")]
r1 = predraft(ag, msgs1, LA(PROP))
chk("T1_basis_arrival_delivers", bool(r1) and "apply_checking_account_credit_5829" in r1,
    (r1 or "None")[:48])

r2 = predraft(ag, msgs1, LA(PROP))
chk("T2_same_basis_silent", r2 is None)

# T3: 그 도구가 이미 성공 실행됐으면 후보에서 빠져 → 제안 탈락(중복 write 방지)
ag2 = Agent()
msgs3 = [M("user", "fix the fees"), M("tool", BASIS2, id="t2"),
         M("assistant", "", [TC("call_x", {"agent_tool_name": "apply_checking_account_credit_5829"},
                                id="c9")]),
         M("tool", "Credit applied successfully!", id="c9")]
r3 = predraft(ag2, msgs3, LA(PROP))
chk("T3_executed_tool_excluded", r3 is None)

# T4: 플래그 미설정 → 사전 자리 무발화
os.environ["T2_WRITE_SUB"] = "1"
ag3 = Agent()
chk("T4_flag_off_no_predraft", predraft(ag3, msgs1, LA(PROP)) is None)

print("=" * 60)
print("FAILS:", FAILS or "none")
sys.exit(1 if FAILS else 0)
