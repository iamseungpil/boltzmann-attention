# -*- coding: utf-8 -*-
"""완료-주장 **격리 검증 서브** 검정 (`T2_CLAIM_VERIFY`·기본 OFF·2026-08-18).

사용자 지시 축자: *"LLM 격리로 env 정책과 실행한 도구, 현재 실행했다고 주장하는 도구를 참 거짓으로
판단하게 별도의 검증 에이전트 돌리면 되는거 아닌가? 이건 LLM 이 잘 할 수 있다."* ·
*"환불했다고 주장하기 전에 … 사용자에게 출력하기 전에 캐치해서 검증하는 서브 에이전트"*

무엇을 못 박나:
  ① 서브가 **거짓**이라 하면 그 주장은 미입증으로 돌아온다 (t7318 073: 조회 도구로 환급 주장)
  ② 서브가 **참**이라 하면 그대로 둔다
  ③ 참인데 지목(`did`)이 **원장 밖**이면 판정을 버린다 — 모르면 막지 않는다([[25]])
  ④ 파싱 실패·템플릿 미선언 → 종전 거동(빈 목록)
  ⑤ 플래그 OFF 면 **아무 일도 없다**(바이트 동일)
  ⑥ 서브에 들어가는 것은 **원장과 주장뿐** — 대화 잔여물 0([[65]])

⚠오프라인. 서브콜은 대역으로 갈아 끼우고 모델을 부르지 않는다.
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

try:                                   # 로컬엔 tau2 가 없다 — 없으면 헬퍼가 import 에서 빠져나가
    import tau2.agent.llm_agent        # 부정 대조가 전부 거짓 통과한다(C543ⓕ 계열).
except Exception:
    for _n in ("tau2", "tau2.agent", "tau2.agent.llm_agent",
               "tau2.data_model", "tau2.data_model.message"):
        sys.modules.setdefault(_n, types.ModuleType(_n))
    sys.modules["tau2.data_model.message"].UserMessage = object
    sys.modules["tau2"].agent = sys.modules["tau2.agent"]
    sys.modules["tau2.agent"].llm_agent = sys.modules["tau2.agent.llm_agent"]

import t2_gate_patch as GP
import t2_subcall as SC

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = {"get_atm_fee_discrepancies", "get_all_user_accounts_by_user_id_3847",
          "transfer_to_human_agents"}
CLAIM = [{"kind": "record_update", "what": "applied the fee refunds to all three accounts",
          "tool": "get_atm_fee_discrepancies"}]
FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


class _FakeSC(object):
    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def sub_generate(self, agent, la, UM, body, tag, temperature=None):
        self.seen.append(body)
        return self.answer

    parse_contract = staticmethod(SC.parse_contract)


def run(answer, claims=None, spec=None, flag="1"):
    real_env = os.environ.get("T2_CLAIM_VERIFY")
    os.environ["T2_CLAIM_VERIFY"] = flag
    real = GP.__dict__.get("_SC_for_test")
    import t2_subcall as _real_mod
    fake = _FakeSC(answer)
    sys.modules["t2_subcall"] = fake
    try:
        out = GP._claim_verify_false("agent", spec if spec is not None else SPEC,
                                     claims if claims is not None else CLAIM, LEDGER)
        return out, fake
    finally:
        sys.modules["t2_subcall"] = _real_mod
        if real_env is None:
            os.environ.pop("T2_CLAIM_VERIFY", None)
        else:
            os.environ["T2_CLAIM_VERIFY"] = real_env


A2 = json.load(io.open(os.path.join(HERE, "a2", "base", "shared.json"), encoding="utf-8"))
SPEC = {"verify_question": (A2.get("claim_audit") or {}).get("verify_question")}


def main():
    chk(bool(SPEC["verify_question"]), "L1 에 verify_question 이 선언돼 있다")

    out, fake = run('{"true": false, "did": ""}')
    chk(len(out) == 1, "서브가 거짓이라 하면 미입증으로 돌아온다 (t7318 073)")

    out, _ = run('{"true": true, "did": "get_atm_fee_discrepancies"}')
    chk(out == [], "서브가 참이라 하면 그대로 둔다")

    out, _ = run('{"true": true, "did": "apply_checking_account_credit_5829"}')
    chk(out == [], "참인데 지목이 원장 밖이면 판정을 버린다(막지 않는다·[[25]])")

    out, _ = run("판정을 못 하겠다")
    chk(out == [], "파싱 실패 → 종전 거동")

    out, _ = run('{"true": false}', spec={})
    chk(out == [], "템플릿 미선언 → 종전 거동")

    out, _ = run('{"true": false, "did": ""}', flag="0")
    chk(out == [], "플래그 OFF 면 아무 일도 없다")

    _, fake = run('{"true": false, "did": ""}')
    body = fake.seen[0] if fake.seen else ""
    chk("get_atm_fee_discrepancies" in body and "applied the fee refunds" in body,
        "서브는 원장과 주장을 받는다")
    chk("customer" not in body.split("An agent is about to tell the customer")[-1].lower()
        or "###" not in body,
        "서브에 대화 잔여물이 실리지 않는다([[65]])")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
