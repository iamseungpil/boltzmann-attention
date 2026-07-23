#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""무료 오프라인 selftest (모델 불요) — T2_HAVE_VALUE 검출기 _have_value_reask_fb 논리 검증.
C115 have-value→act 일반레버. 실제 shipped A2 spec(banking_knowledge.gate.json)로 fire/no-fire 전수.
Run: py -3 test_have_value_reask.py   (통과=모든 케이스 PASS·[[08]] 검출 술어 격리 검증)"""
import json
import os
import sys
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_gate_patch as G  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
SPECS = A2["have_value_reask"]

# ── 합성 메시지 빌더 (tau2 객체 모사: role/content/tool_calls/error/id) ──
def um(c):
    return NS(role="user", content=c, tool_calls=None, error=False, id=None)


def am(c, tool_calls=None):
    return NS(role="assistant", content=c, tool_calls=tool_calls, error=False, id=None)


def tm(c, error=False, id=None):
    return NS(role="tool", content=c, tool_calls=None, error=error, id=id)


def tc(name, args):
    return NS(name=name, arguments=args, id="x")


PROD_OUT = ("Card information retrieved successfully.\n\nExecuted: get_card_last_4_digits\n"
            "Arguments: {\n  \"credit_card_account_id\": \"cc_01f21c9970_gold\"\n}\n"
            "Last 4 digits of card: 1652")
ERR_OUT = ("Error: Tool 'get_card_last_4_digits' has not been given to you by the agent. "
           "The agent must first use `give_discoverable_user_tool`.")

# producer 성공출력 + 이전 assistant 재요청이 있는 committed 히스토리(idx55 동형)
BASE_HIST = [
    um("I need to dispute 8 charges."),
    am("It seems there was an error: I need the last 4 digits of the credit card."),  # 이전 재요청
    tm(ERR_OUT),                                        # producer 에러(marker 없음=미포착 검증)
    am("To proceed with filing the disputes, I need the correct last 4 digits."),     # 이전 재요청
    tm(PROD_OUT),                                        # producer 성공출력(marker 실재)
    um("Got it - the last 4 digits are 1652. Please file all 8 disputes now."),
]

# 현재 턴(am): last-4 재요청·write 미시도
AM_REASK = am("I apologize. It seems we need to ensure we have the correct last 4 digits of your card.")
AM_NEUTRAL = am("Sure, let me proceed with the disputes right away.")
# 현재 턴: W(file_dispute) 호출 중
AM_FILES = am("Filing now.", tool_calls=[tc("call_discoverable_agent_tool",
              {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
               "arguments": {"transaction_id": "txn_1", "card_last_4_digits": "1652"}})])
# 현재 턴: producer 재호출(user측 디스패처)
AM_RECALL = am("Let me re-run the lookup.", tool_calls=[tc("call_discoverable_user_tool",
               {"user_tool_name": "get_card_last_4_digits",
                "arguments": {"credit_card_account_id": "cc_01f21c9970_gold"}})])

RESULTS = []


def check(name, expect_fire, cur_am, hist=None, must_contain=None):
    h = BASE_HIST if hist is None else hist
    fb = G._have_value_reask_fb(cur_am, h, SPECS)
    fired = fb is not None
    ok = (fired == expect_fire)
    if ok and expect_fire and must_contain:
        ok = all(s in fb for s in must_contain)
    RESULTS.append((name, ok, fired, fb))
    print("[%s] %s  fired=%s%s" % ("PASS" if ok else "FAIL", name, fired,
          ("" if not fb else "  fb=%r" % (fb[:90] + "..."))))


# 1) FIRE: 산문 재요청 + producer 성공출력 + 이전 재요청 + W 미시도 → 값(1652) 인용
check("prose-reask fires w/ value", True, AM_REASK, must_contain=["1652", "HAVE-VALUE"])
# 2) NO-FIRE: producer 성공출력 없음(에러출력만) → 미실재
hist_noprod = [m for m in BASE_HIST if "Executed" not in str(m.content)]
check("no producer output -> silent", False, AM_REASK, hist=hist_noprod)
# 3) NO-FIRE: 이전 assistant 재요청 이력 없음(정당한 첫 질문)
hist_noprior = [um("dispute"), tm(PROD_OUT), um("last 4 are 1652")]
check("no prior reask -> silent", False, AM_REASK, hist=hist_noprior)
# 4) NO-FIRE: 현재 턴이 재요청 아님(중립 진행)
check("am not reasking -> silent", False, AM_NEUTRAL)
# 5) NO-FIRE: 현재 턴이 W(file_dispute) 호출 중
check("am files W -> silent", False, AM_FILES)
# 6) FIRE: 현재 턴이 producer 재호출(user측) → 값 인용
check("producer re-call fires", True, AM_RECALL, must_contain=["1652"])
# 7) value_pattern 미실재 시 valclause 생략도 정상 문구(빈 producer 값)
prod_noval = PROD_OUT.replace("Last 4 digits of card: 1652", "Last 4 digits retrieved.")
hist_noval = [m if "Executed" not in str(m.content) else tm(prod_noval) for m in BASE_HIST]
G_fb = G._have_value_reask_fb(AM_REASK, hist_noval, SPECS)
ok7 = G_fb is not None and "1652" not in G_fb and "  ." not in G_fb and "{val" not in G_fb
RESULTS.append(("value-free wording clean", ok7, G_fb is not None, G_fb))
print("[%s] value-free wording clean (no dangling)  fb=%r" % ("PASS" if ok7 else "FAIL",
      None if not G_fb else G_fb[:90] + "..."))

nfail = sum(1 for _, ok, _, _ in RESULTS if not ok)
print("\n%d/%d PASS" % (len(RESULTS) - nfail, len(RESULTS)))
sys.exit(1 if nfail else 0)
