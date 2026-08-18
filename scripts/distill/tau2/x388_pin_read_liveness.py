# -*- coding: utf-8 -*-
"""x388 — **기존 레버 생존 확인**: `T2_PIN_READ_STEPS`(read 강제)가 왜 후보 0을 내는가.

사용자 지시(2026-08-18 축자): *"새로운 레버들 짓지 말라. 이미 거의 모든 레버는 다 지어져 있고,
A2 A3 구조도 이미 거의 모두 있다. 엔진도 다 있다. 기존 코드나 기존 레버들이 살아 있는지 확인하고
연결하라."*

## 관측 (t7315 · task_050 · 수리 후)
  · `[T2_PROC_ABSENT] surface credit_limit_increase quiet>=3 done=2 of 7` — 절차 엔진은 **살아 있다**
  · `[T2_PIN_READ_STEPS] no unique read target (0)` **×2** — read 강제 레버는 발화하는데 **후보가 0**
  · gold 액션 3~6(`get_user_dispute_history_7291`·`get_pending_replacement_orders_5765`)은
    t7315·t7314 ctl·t7314 treat **세 팔 전부 MISS**

## 이 프로브가 가르는 것 (결정론·모델 호출 0)
  ⒜ `render_state(...)['ready_tools']` 가 **비어 있나** (절차 상태 계산 쪽 문제)
  ⒝ 아니면 목록은 있는데 `_is_read_tool(env, name)` 이 전부 False 인가 (분류 쪽 문제)
     — discoverable 도구는 **잠겨 있는 동안 env 레지스트리에 없다**. 그러면 read 강제 레버는
       *자기가 만들어진 바로 그 도구들*에 대해 구조적으로 침묵한다.

⚠판정이 아니라 **분기 확인**이다. 실행은 오프라인이고 유료 0.
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import collections

import t2_procedure as PROC

HERE = os.path.dirname(os.path.abspath(__file__))
A2P = os.path.join(HERE, "a2", "banking_knowledge.specific.json")

# t7315 task_050 이 실제로 실행한 도구(궤적 축자 · 접미사 포함 이름 그대로)
EXECUTED = collections.Counter({
    "verify_identity": 2,
    "get_user_information_by_name": 1,
    "get_current_time": 1,
    "log_verification": 1,
    "get_credit_card_accounts_by_user": 1,
    "check_cli_eligibility": 2,
    "unlock_discoverable_agent_tool": 3,
    "get_credit_limit_increase_history_4829": 1,
    "get_payment_history_6183": 1,
    "submit_credit_limit_increase_request_7392": 1,
    "KB_search_bm25": 4,
})
UNLOCKED = {"get_credit_limit_increase_history_4829", "get_payment_history_6183",
            "submit_credit_limit_increase_request_7392"}
ARGS_BY_TOOL = {"get_payment_history_6183": [{"credit_card_account_id": "cc_584f9c5d00_gold",
                                              "months": 12}]}


def main():
    a2 = json.load(io.open(A2P, encoding="utf-8"))
    procs = a2.get("procedures") or []
    proc = next((p for p in procs if p.get("id") == "credit_limit_increase"), None)
    if proc is None:
        print("FAIL — credit_limit_increase 선언이 없다"); return 1

    act = PROC.active_procedures(procs, EXECUTED)
    print("① 활성 절차: %s" % [p.get("id") for p in act])

    st = PROC.render_state(proc, EXECUTED, UNLOCKED, None, args_by_tool=ARGS_BY_TOOL)
    print("② render_state:")
    for k in sorted(st):
        v = str(st[k])
        print("     %-14s %s" % (k, v[:200]))

    cand = [t.strip() for t in str(st.get("ready_tools") or "").split(",") if t.strip()]
    print("③ ready_tools 후보 %d개: %s" % (len(cand), cand))
    if not cand:
        print("   ⇒ ⒜ 분기: 절차 상태가 후보를 못 낸다 — `_is_read_tool` 이전에 이미 0이다.")
    else:
        print("   ⇒ ⒝ 분기 후보: 목록은 있다. 남은 물음은 `_is_read_tool` 이 이 이름들을 read 로 "
              "보느냐이고, 그것은 **라이브 env 가 있어야** 답한다(잠긴 discoverable 은 레지스트리에 "
              "없을 수 있다).")

    # 인자 요건이 상태에 반영되는가 (months=12 는 정책 밖: one_of [6,3])
    node = next((n for n in (proc.get("nodes") or []) if n.get("id") == "payment_history"), None)
    done_ok = PROC.is_done(node, EXECUTED, args_by_tool=ARGS_BY_TOOL) if hasattr(PROC, "is_done") else None
    print("④ payment_history(months=12) 완료 판정: %r  (정책 one_of=%s)"
          % (done_ok, (node or {}).get("param_requirement", {}).get("one_of")))

    print("⑤ decision 노드의 선행: %s"
          % [n.get("id") for n in (proc.get("nodes") or []) if n.get("id") == "decision"][:1])
    dec = next((n for n in (proc.get("nodes") or []) if n.get("id") == "decision"), None)
    if dec:
        print("     requires=%s" % dec.get("requires"))
        try:
            print("     unmet(approve_credit_limit_increase_5847)=%s"
                  % PROC.unmet_nodes(proc, "approve_credit_limit_increase_5847", EXECUTED))
        except Exception as e:
            print("     unmet 호출 실패: %r" % (e,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
