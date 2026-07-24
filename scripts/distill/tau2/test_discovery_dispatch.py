#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISCOVERY-DISPATCH 게이트 감지 predicate selftest (2026-07-25·C151).
직접호출된 discoverable 도구(suffixed name·dispatcher 미경유)를 감지 → deny+프로토콜 지시로
에이전트가 unlock→call_discoverable 재발행(compliance 게이트·reroute 아님·[[05]]/[[10]]).
근거(C149/C150): 등록(agent_discoverable_tools CALLED)은 call_discoverable_agent_tool 내부에서만·
직접호출은 평가 리플레이가 스킵→미등록. 이 predicate = t2_gate_patch dd_fb 감지와 동일 로직.
Run: py -3 test_discovery_dispatch.py"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                     encoding="utf-8"))["eplan"]


def is_direct_discoverable(name, spec=SPEC):
    """dispatch_tool 선언 도메인서 이름이 suffixed(_NNNN)이고 dispatcher/unlock/list가
    아니면 직접호출 discoverable(=deny 대상). 순수·도메인일반(스펙 ABox + 구조 suffix)."""
    if not name or not spec.get("dispatch_tool"):
        return False
    safe = {spec.get("dispatch_tool"),
            spec.get("unlock_tool", "unlock_discoverable_agent_tool"),
            spec.get("list_tool", "list_discoverable_agent_tools")}
    return name not in safe and bool(re.search(r"_\d{3,4}$", name))


CASES = {
    # 직접호출된 discoverable → deny(True)
    "get_all_user_accounts_by_user_id_3847": True,
    "get_closure_reason_history_8293": True,
    "apply_credit_card_account_flag_6147": True,
    "get_user_dispute_history_7291": True,
    # dispatcher/프로토콜 도구 = safe(False)
    "call_discoverable_agent_tool": False,
    "unlock_discoverable_agent_tool": False,
    "list_discoverable_agent_tools": False,
    # base/scaffold_get 도구(suffix 없음) = False
    "verify_identity": False,
    "get_current_time": False,
    "log_verification": False,
    "KB_search_bm25": False,
    "get_reward_discrepancies": False,
    "check_card_closure_eligibility": False,
}


def main():
    ok = True
    for nm, exp in CASES.items():
        got = is_direct_discoverable(nm)
        good = got == exp
        ok &= good
        print("[%s] %-42s direct_disc=%s (exp %s)"
              % ("PASS" if good else "FAIL", nm, got, exp))
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
