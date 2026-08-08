# -*- coding: utf-8 -*-
"""회귀 검정 (C341): 완료-주장은 **모델이 지목한 호출**로 대조한다 — `kind` 라벨이 아니라.

무엇을 막는 검정인가 (실측·run f 사이드카): `log_verification` 이 양쪽 sim 에서 **실제로
실행됐는데**, 모델이 그 주장의 kind 를 `record_update` 로 선언했고 A2 는 그 도구를 `verify`
아래 두므로 색인이 빗나가 우리가 결정 턴마다 *"the conversation ledger shows NO such event:
record_update: logged verification record"* 를 내보냈다. **맞는 행동을 했는데 틀렸다고 말한
것**이고, 우리 출력이 이 대화의 유일한 근거원이라 그 자체로 오염이다([[25]]).

권위는 실행 원장이고 kind 는 해석이다([[52]]) ⇒ 주장이 호출을 지목하면 엔진은 **이름 집합
대조**만 한다([[22]] 의미 판단 0). 지목이 없으면 구판(kind 색인)으로 떨어져 거동이 보존된다.

오프라인 전용. 실행: py -3 test_claim_tool_index.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G                                      # noqa: E402

FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


# A2 event_map 을 흉내 낸다 — 요점은 **kind 와 도구의 배치가 어긋난다**는 것이다.
EMAP = {"verify": ["verify_identity", "log_verification"],
        "record_update": ["update_"],
        "write": "__effective_write__"}
LEDGER = {"verify_identity", "log_verification"}


def main():
    # ── 실물 재현: kind 는 빗나갔고 지목은 맞다 ──────────────────────────────
    real = [{"kind": "record_update", "what": "logged verification record",
             "tool": "log_verification"}]
    chk(G._claim_unbacked(real, EMAP, LEDGER, []) == [],
        "kind 가 빗나가도 지목한 호출이 원장에 있으면 미입증이 아니다  ← run f 거짓 발화")

    # ── 음성 통제: 레버가 죽지 않았다 ([[57]]) ───────────────────────────────
    lie = [{"kind": "verify", "what": "checked the knowledge base", "tool": "KB_search"}]
    chk(len(G._claim_unbacked(lie, EMAP, LEDGER, [])) == 1,
        "지목한 호출이 원장에 없으면 그대로 잡는다 (레버 생존)")

    # ── 구판 경로 보존: 지목이 없으면 kind 색인 ─────────────────────────────
    old = [{"kind": "record_update", "what": "logged verification record"}]
    chk(len(G._claim_unbacked(old, EMAP, LEDGER, [])) == 1,
        "지목이 없으면 구판(kind 색인) 그대로 (거동 보존)")
    old_ok = [{"kind": "verify", "what": "verified identity"}]
    chk(G._claim_unbacked(old_ok, EMAP, LEDGER, []) == [],
        "지목이 없고 kind 가 맞으면 구판대로 통과")

    # ── discoverable 접미사 정규화 ─────────────────────────────────────────
    suf = [{"kind": "verify", "what": "read the accounts",
            "tool": "call_discoverable_agent_tool_3847"}]
    chk(G._claim_unbacked(suf, EMAP, {"call_discoverable_agent_tool"}, []) == [],
        "`_NNNN` 접미사가 붙은 지목도 같은 이름으로 본다")

    # ── 빈 지목은 지목이 아니다 ────────────────────────────────────────────
    blank = [{"kind": "record_update", "what": "x", "tool": "   "}]
    chk(len(G._claim_unbacked(blank, EMAP, LEDGER, [])) == 1,
        "빈 문자열 지목은 미지목으로 본다(구판 경로)")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
