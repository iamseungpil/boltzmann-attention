#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""READ_DEDUP loop-break selftest (C123·rall19 043 실측) — tau2 필요(리모트 seka_env).
동일 (name,args) 소형-출력 read가 K회(기본 3) 실행되면 크기 무관 스텁+redirect.
Run: PYTHONPATH=src:$REPO/scripts/distill/tau2 python test_read_dedup_loopbreak.py"""
import os
import sys
from types import SimpleNamespace as NS

os.environ["T2_READ_DEDUP"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tau2.orchestrator.orchestrator import BaseOrchestrator  # noqa: E402
from tau2.data_model.message import ToolMessage  # noqa: E402
import t2_gate_patch as G  # noqa: E402

SHORT = "./doc_credit_cards_credit_card_account_logistics_009.md:- checking_account_id (string, required)"
EXECUTED = {"n": 0}


def fake_exec(self, tcs):
    EXECUTED["n"] += len(tcs)
    return [ToolMessage(id=t.id, role="tool", requestor="assistant", error=False,
                        content=SHORT) for t in tcs]


BaseOrchestrator._t2_orig_exec = fake_exec
G._install_regen_exec()


def tc(i):
    return NS(name="shell", arguments={"command": "grep -r 'checking_account_id' ."},
              id="c%d" % i, requestor="assistant")


def main():
    # ★2026-08-03 수리: C241 U1'(2026-07-30) 이후 실효-write 술어가 **도메인 A2에서** 어휘를
    #   읽는다 — `environment` 없는 페이크 self는 a2=None ⇒ 모든 도구가 write로 판정되어
    #   dedup 카운터 자체가 안 돌았고, 이 검정은 그 뒤로 조용히 FAIL이었다(레버 회귀가 아니라
    #   **검정의 부패**). 도메인을 실제로 물려 원래 의도대로 잰다.
    slf = NS(agent=None, tools=None,
             environment=NS(domain_name="banking_knowledge",
                            _is_mutating_tool=lambda n: False))
    ok = True
    outs = []
    for i in range(5):
        res = BaseOrchestrator._execute_tool_calls(slf, [tc(i)])
        outs.append(res[0].content)
    stubbed = ["[DUPLICATE-READ]" in c for c in outs]
    # K=3: 1~3회차 실행·4회차부터 스텁
    good = stubbed == [False, False, False, True, True] and EXECUTED["n"] == 3
    print("[%s] loopbreak_small_output: stub_pattern=%s executed=%d"
          % ("PASS" if good else "FAIL", stubbed, EXECUTED["n"]))
    ok &= good
    good = "grep" not in outs[3] or "PLAIN WORDS" in outs[3]
    good = "PLAIN WORDS" in outs[3]  # shell+grep-in-args → redirect 힌트 포함
    print("[%s] loopbreak_redirect_args_scan: redirect=%s" % ("PASS" if good else "FAIL", good))
    ok &= good
    # 1:1·순서 보장(크래시 픽스 invariant): 스텁 턴도 결과 길이=호출 길이
    res = BaseOrchestrator._execute_tool_calls(slf, [tc(9)])
    good = len(res) == 1 and res[0].id == "c9"
    print("[%s] loopbreak_result_pairing" % ("PASS" if good else "FAIL",))
    ok &= good
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
