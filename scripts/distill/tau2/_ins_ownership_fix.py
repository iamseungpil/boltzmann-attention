# -*- coding: utf-8 -*-
"""일회성 — FIX-7 소유권-정정 deny 키 2종 A2 3사본 삽입.

격리 근거: x298_ownership_deny_probe.py (3셀 n=8·사전등록) — A_LIVE 1/8 · B_OWN 6/8 ·
D_BARE 0/8. 문면은 **측정한 그 문자열 축자**([[03b]]: 측정한 문구 = 출시할 문구).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

# x298 B_OWN 축자(프로브의 b_own() 문자열 그대로 · {name}/{matches} 만 치환자)
FB = ("Error: '{name}' is not a tool that can be given to the customer - there is no "
      "customer-side tool by that name on file. The tool on file that matches what you asked "
      "for is one of YOUR OWN agent tools: {matches}. The customer cannot run it and "
      "describing the steps to them does not execute it - unlock it and call it yourself.")
ARGS = ["discoverable_tool_name", "user_tool_name"]
NOTE = ("2026-08-13 FIX-7. 실물 = t7277 075 엔진로그 118행: 착수 시도가 "
        "give_discoverable_user_tool(discoverable_tool_name='open_account') 였는데 우리 문구는 "
        "feedback_not_discoverable(\"unlock_ 은 적용 안 된다\"+레지스트리 45개)로 **채널이 어긋난** "
        "말을 했고, 그 이름이 사실 에이전트 자신의 도구(open_bank_account_4821)라는 사실을 말하지 "
        "않아 수동 안내로 접혔다. 격리 x298(3셀 n=8·사전등록): A_LIVE 1/8 · B_OWN 6/8 · D_BARE 0/8. "
        "술어는 닫힘([[22]]): 손님-측 인자 키(user_tool_channel_args) ∧ 이름 토큰이 에이전트 "
        "레지스트리와 겹침(_tok_overlap·기계·판단 0). 겹침 없으면 침묵(fail-open)·무엇을 호출할지는 "
        "모델 몫([[62]] ③④). 문면은 측정 축자 그대로([[03b]]).")

entries = []
for rel in PATHS:
    p = os.path.join(HERE, rel)
    j = json.load(io.open(p, encoding="utf-8"))
    d = j.get("discoverable_name_check")
    if d is None:
        print("MISSING discoverable_name_check in %s" % rel)
        sys.exit(1)
    d["feedback_user_tool_is_agents"] = FB
    d["user_tool_channel_args"] = list(ARGS)
    d["_note_feedback_user_tool_is_agents"] = NOTE
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(d)
    print("updated %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
