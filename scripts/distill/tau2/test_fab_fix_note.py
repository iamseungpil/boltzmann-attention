# -*- coding: utf-8 -*-
"""P4 검정(2026-08-21·[[64]]·t7335 halfB 079) — FAB_STRIP 차단 문면에 해소-read 도구명이 실린다.

무엇을 고정하나:
  ⑴ 차단 문면 두 칸([[64]]): 무엇이 틀렸나(인자·값) + 무엇을 하면 풀리나(해소-read 도구명)
  ⑵ 도구명은 엔진 리터럴이 아니라 **선언 기계 도출** — a2["arg_source_reads"] 우선,
     폴백 a2["relations"]["by_tool"][eff]["requires"](C586)
  ⑶ 선언이 비면 인자·값 지목만 남는다(지어내지 않는다 — test_keep_deny_body ⒟ 동형)
  ⑷ 실제 banking A2 로 079 실패 경로 재현: freeze_debit_card 의 날조 card_id 차단 문면에
     get_debit_cards_by_account_id_7823 이 실린다
  ⑸ A2 2사본(specific/gate) json-등가 · FAB 블록이 _fab_fix_note 를 실제로 부른다(배선 생존)

오프라인 전용(LLM·서버 불요). 실행: py -3 test_fab_fix_note.py
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

import t2_gate_patch as GP                                         # noqa: E402
from gate_interpreter import load_domain_a2                        # noqa: E402

OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


print("[⑴⑵ 합성 A2 — 인자 선언 우선·폴백·빈 선언]")
_a2 = {"arg_source_reads": {"card_id": ["read_cards_XX"], "_note": "주석은 걸러진다"},
       "relations": {"by_tool": {"freeze_debit_card": {"requires": ["read_fallback_YY"]}}}}
n = GP._fab_fix_note([("freeze_debit_card", [("card_id", "dbc_fake123")])], _a2)
chk("무엇이 틀렸나: 인자·값 지목", "card_id='dbc_fake123'" in n, n)
chk("무엇을 하면 풀리나: 인자 선언의 read", "read_cards_XX" in n, n)
chk("인자 선언이 있으면 폴백 미사용", "read_fallback_YY" not in n)
n2 = GP._fab_fix_note([("freeze_debit_card", [("mystery_arg", "zz993311")])], _a2)
chk("인자 선언 없으면 by_tool.requires 폴백", "read_fallback_YY" in n2, n2)
n3 = GP._fab_fix_note([("unknown_tool", [("mystery_arg", "zz993311")])], {})
chk("선언 전무 = 인자·값 지목만(지어내지 않는다)",
    "mystery_arg='zz993311'" in n3 and "first read the real value" not in n3, n3)
chk("빈 입력 = 빈 문자열", GP._fab_fix_note([], _a2) == "" and GP._fab_fix_note(None, None) == "")

print("[⑷ 실제 banking A2 — 079 실패 경로]")
a2 = load_domain_a2("banking_knowledge")
chk("arg_source_reads 로드", bool((a2 or {}).get("arg_source_reads")))
n4 = GP._fab_fix_note([("freeze_debit_card", [("card_id", "dbc_cr89a2b3c4_ev")])], a2)
chk("079: card_id 차단 문면에 해소-read 도구명",
    "get_debit_cards_by_account_id_7823" in n4, n4)
chk("079: 해소 순서(계좌 목록 read 선행)",
    n4.find("get_all_user_accounts_by_user_id_3847") < n4.find("get_debit_cards_by_account_id_7823"))
n5 = GP._fab_fix_note([("order_debit_card", [("account_id", "Evergreen Account")])], a2)
chk("079: 클래스명-as-ID 차단 문면에 계좌 목록 read",
    "get_all_user_accounts_by_user_id_3847" in n5, n5)
chk("영어 문면(C125)·값 지목", "Evergreen Account" in n5 and "record read in this conversation" in n5)

print("[⑸ 배선·사본 동기]")
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
_i = SRC.find('os.environ.get("T2_FAB_STRIP") == "1"')
_seg = SRC[_i:_i + 3000]
chk("FAB 블록이 _fab_fix_note 를 부른다(배선 생존·[[67]] 0단계)",
    _i >= 0 and "_fab_fix_note(_fab_bad, a2)" in _seg)
chk("구판 무지목 노트 상수 문자열이 단독으로 남아있지 않다",
    "were not processed.]\")" not in _seg)
_sp = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                        encoding="utf-8"))["arg_source_reads"]
_gt = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                        encoding="utf-8"))["arg_source_reads"]
chk("arg_source_reads 2사본 json-등가([[24]])", _sp == _gt)
chk("선언 read 는 전부 env 레지스트리에 실재(mutates=false)", (lambda: (
    lambda surf: all(
        r in surf and not surf[r]["mutates"]
        for k, v in _sp.items() if not k.startswith("_") for r in v)
)(json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"),
                    encoding="utf-8"))["banking_knowledge"]["tools"]))())

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
