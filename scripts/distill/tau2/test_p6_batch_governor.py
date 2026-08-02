#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6 — 단일-메시지 **병렬 배치**를 반복 거버너가 세는가 (검정 케이스만·AX32 설계서 §P6).

동기(018 실측·029형 사촌): 한 assistant 메시지가 give를 6발 병렬로 쏜다. 반복 거버너/REPEAT_CAP은
**턴 간** 반복을 전제로 설계됐으므로, 같은 메시지 안의 병렬 콜이 계수되는지가 미검이었다.
이 검정은 처방이 아니라 **관측 확정**이다 — 결과가 어느 쪽이든 그대로 박제한다.

측정 3종(모두 한 번의 `_execute_tool_calls` 호출 = 한 메시지):
  ⒜ 동일 (name,args) 6발  → seen 카운터가 배치 안에서 누적되어 K(3)회 초과분이 스텁되는가
  ⒝ **서로 다른** 인자 6발 → 스텁 0(= 거버너는 동일성 기반이라 '많이 부름' 자체는 안 센다)
  ⒞ REPEAT_CAP=8         → 배치 내 계수가 캡 채널로도 이어지는가

⚠tau2 필요(리모트 seka_env). Run:
  PYTHONPATH=src:$REPO/scripts/distill/tau2 python test_p6_batch_governor.py"""
import os
import sys
from types import SimpleNamespace as NS

os.environ["T2_READ_DEDUP"] = "1"
os.environ.setdefault("T2_REPEAT_CAP", "8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tau2.orchestrator.orchestrator import BaseOrchestrator  # noqa: E402
from tau2.data_model.message import ToolMessage  # noqa: E402
import t2_gate_patch as G  # noqa: E402

OUT = "Found 1 record(s) in 'x':\n\n1. Record ID: r1\n   f: v\n"
EXECUTED = {"n": 0}


def fake_exec(self, tcs):
    EXECUTED["n"] += len(tcs)
    return [ToolMessage(id=t.id, role="tool", requestor="assistant", error=False, content=OUT)
            for t in tcs]


BaseOrchestrator._t2_orig_exec = fake_exec
G._install_regen_exec()

ok = True


def chk(c, m):
    global ok
    ok &= bool(c)
    print(("[PASS] " if c else "[FAIL] ") + m)


def tc(i, arg="same"):
    return NS(name="get_user_information_by_id", arguments={"user_id": arg},
              id="b%d" % i, requestor="assistant")


def main():
    # ★환경 필수(C241 U1'·2026-07-30): 실효-write 술어는 도메인 A2에서 어휘를 읽는다 —
    #   `environment` 없는 페이크는 **모든 도구를 write로 판정**해 dedup 카운터가 아예 안 돈다
    #   (그 상태로 재면 "배치를 안 센다"는 **가짜 음성**이 나온다·test_read_dedup_loopbreak가
    #   같은 이유로 조용히 FAIL 중이었다).
    slf = NS(agent=None, tools=None,
             environment=NS(domain_name="banking_knowledge",
                            _is_mutating_tool=lambda n: False))
    # ⒜ 동일 호출 6발을 **한 메시지**로
    EXECUTED["n"] = 0
    res = BaseOrchestrator._execute_tool_calls(slf, [tc(i) for i in range(6)])
    stub = ["[DUPLICATE-READ]" in (r.content or "") for r in res]
    chk(len(res) == 6, "결과 1:1 (배치 6 → 6)")
    print("       stub_pattern=%s executed=%d" % (stub, EXECUTED["n"]))
    chk(any(stub), "★배치 내 동일 콜이 계수된다(같은 메시지 안에서 스텁 발생) — 018 우려 해소")
    chk(EXECUTED["n"] <= 3, "K=3 초과분은 실행되지 않음: executed=%d" % EXECUTED["n"])
    # ⒝ 서로 다른 인자 6발 = 반복 아님 → 스텁 0(설계된 거동: 거버너는 동일성 기반)
    EXECUTED["n"] = 0
    res2 = BaseOrchestrator._execute_tool_calls(slf, [tc(10 + i, "u%d" % i) for i in range(6)])
    stub2 = ["[DUPLICATE-READ]" in (r.content or "") for r in res2]
    chk(not any(stub2) and EXECUTED["n"] == 6,
        "★서로 다른 인자 6발 = 스텁 0(‘한 메시지 6발’ 자체는 거버너의 표적이 아니다)")
    # ⒞ 캡 채널: 같은 키를 계속 두들기면 CAP 문구로 승격되는가(배치 계수의 연속성)
    for i in range(3):
        BaseOrchestrator._execute_tool_calls(slf, [tc(100 + i)])
    r3 = BaseOrchestrator._execute_tool_calls(slf, [tc(200)])[0]
    chk("[REPEAT-CAP]" in (r3.content or "") or "IDENTICAL call" in (r3.content or ""),
        "캡/에스컬레이션 채널로 승격(K=%s): %s" % (os.environ.get("T2_REPEAT_CAP"),
                                                (r3.content or "")[-60:].replace("\n", " ")))
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
