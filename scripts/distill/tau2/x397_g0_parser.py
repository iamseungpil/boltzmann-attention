# -*- coding: utf-8 -*-
"""x397 G0-1 — parse_tool / plan_names 왕복 검정. 정답 형태 응답을 주입해 검출률을 잰다."""
import io, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import x395_compliance_iso as X

T = "get_customer_by_id"

NEXT_CASES = [
 ("F1 순수 JSON", "{\"tool\": \"get_customer_by_id\", \"arguments\": {\"id\": \"c_1\"}}"),
 ("F2 ```json 펜스", "```json\n{\"tool\": \"get_customer_by_id\", \"arguments\": {}}\n```"),
 ("F2b ``` 펜스(언어없음)", "```\n{\"tool\": \"get_customer_by_id\", \"arguments\": {}}\n```"),
 ("F3 앞뒤 산문+JSON", "Sure, the next step is to look up the customer.\n{\"tool\": \"get_customer_by_id\", \"arguments\": {\"id\": \"c_1\"}}\nLet me know if you need more."),
 ("F3b 앞 산문에 중괄호", "I will call the tool {tool} now.\n{\"tool\": \"get_customer_by_id\", \"arguments\": {}}"),
 ("F4 홑따옴표 JSON", "{tool: get_customer_by_id, arguments: {id: c_1}}"),
 ("F5 중첩 {tool:{name}}", "{\"tool\": {\"name\": \"get_customer_by_id\", \"arguments\": {}}}"),
 ("F6 JSON 여러 개", "{\"tool\": \"get_customer_by_id\", \"arguments\": {}}\n{\"tool\": \"verify_identity\", \"arguments\": {}}"),
 ("F7 펜스 안 산문 뒤 JSON", "Reasoning: policy says verify first.\n```json\n{\"tool\": \"get_customer_by_id\", \"arguments\": {}}\n```"),
 ("F8 name 키(tool 없음)", "{\"name\": \"get_customer_by_id\", \"arguments\": {}}"),
 ("F9 function 래핑", "{\"function\": {\"name\": \"get_customer_by_id\", \"arguments\": {}}}"),
 ("F10 개행/들여쓰기 pretty", "{\n  \"tool\": \"get_customer_by_id\",\n  \"arguments\": {\n    \"id\": \"c_1\"\n  }\n}"),
 ("F11 뒤 산문에 중괄호", "{\"tool\": \"get_customer_by_id\", \"arguments\": {}}\nNote: arguments {} may vary."),
 ("F12 XML/함수호출 문법", "<tool_call>{\"name\": \"get_customer_by_id\", \"arguments\": {}}</tool_call>"),
 ("F13 400자 절단(꼬리 잘림)", ("Let me think about this carefully. " * 11) + "{\"tool\": \"get_customer_by_id\", \"argum"),
]

PLAN_CASES = [
 ("P1 순수 JSON", "{\"plan\": [{\"tool\": \"get_customer_by_id\"}, {\"tool\": \"verify_identity\"}]}"),
 ("P2 펜스", "```json\n{\"plan\": [{\"tool\": \"get_customer_by_id\"}]}\n```"),
 ("P3 앞뒤 산문", "Here is the plan.\n{\"plan\": [{\"tool\": \"get_customer_by_id\"}]}\nDone."),
 ("P4 홑따옴표", "{plan: [{tool: get_customer_by_id}]}"),
 ("P5 문자열 배열", "{\"plan\": [\"get_customer_by_id\", \"verify_identity\"]}"),
 ("P6 JSON 여러 개", "{\"plan\": [{\"tool\": \"get_customer_by_id\"}]}\n{\"plan\": [{\"tool\": \"x\"}]}"),
 ("P7 최상위 배열", "[{\"tool\": \"get_customer_by_id\"}]"),
 ("P8 name 키", "{\"plan\": [{\"name\": \"get_customer_by_id\"}]}"),
]

print("== G0-1a parse_tool 왕복 (정답 = %s) ==" % T)
ok = 0
for label, s in NEXT_CASES:
    try:
        nm, obj = X.parse_tool(s)
    except Exception as e:
        nm, obj = "EXC:" + str(e)[:60], None
    hit = (nm == T)
    ok += hit
    print("  %-6s %-24s pred=%-24s said_only=%s" % ("OK" if hit else "MISS", label, str(nm)[:24], (not nm)))
print("  검출률 %d/%d = %.3f" % (ok, len(NEXT_CASES), ok / float(len(NEXT_CASES))))

print("\n== G0-1b plan_names 왕복 ==")
ok2 = 0
for label, s in PLAN_CASES:
    try:
        pl = X.plan_names(s)
    except Exception as e:
        pl = ["EXC:" + str(e)[:60]]
    hit = bool(pl) and pl[0] == T
    ok2 += hit
    print("  %-6s %-24s plan=%s" % ("OK" if hit else "MISS", label, str(pl)[:70]))
print("  검출률 %d/%d = %.3f" % (ok2, len(PLAN_CASES), ok2 / float(len(PLAN_CASES))))

print("\n== G0-1c 기권/거부 형태가 said_only 로 세어지는지 ==")
for label, s in [("R1 tool:null(지시대로)", "{\"tool\": null, \"reason\": \"need more info\"}"),
                 ("R2 순수 산문 되묻기", "I need the account id before I can proceed. Could you confirm it?"),
                 ("R3 빈 응답", "")]:
    nm, _ = X.parse_tool(s)
    print("  %-24s pred=%-10s said_only=%s" % (label, str(nm), (not nm)))
