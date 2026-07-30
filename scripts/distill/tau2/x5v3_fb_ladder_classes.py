# -*- coding: utf-8 -*-
"""X5 v3 — 오류 클래스 2종 확장 (논문③ §9 한계 해소: admissible-set이 진짜 이득을 내는 클래스인가).

v2는 클래스 1종(존재하지 않는 도구명)만 봤고 거기서는 '정답=재검색'이라 등급3(admissible-set)이
구성상 무용했다. v3는 admissible-set이 자연스러운 처방인 두 클래스를 본다:
  (A) 결핍 인자 — 필수 파라미터 누락(선언 목록이 곧 후보 집합)
  (B) 타입/enum 위반 — 값이 선언된 허용 집합 밖(enum 목록이 곧 후보 집합)
v2와 동일 규율: 스키마를 프롬프트에 나열하지 않고 **KB 문서 안에만** 두고, 반복-실패 진입,
문맥 부하, 32B·시드 6, 응답 원문 덤프.
"""
import json, re, urllib.request

URL = "http://localhost:8140/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
SEEDS = 6

SYS = (
    "You are a Rho-Bank service agent. Agent tools available:\n"
    "  search_knowledge_base(query: str)\n"
    "  get_credit_card_accounts(customer_id: str)\n"
    "  call_discoverable_agent_tool(agent_tool_name: str, arguments: str)\n"
    "Parameter schemas for specialized tools are documented in the knowledge base, not here.\n"
    "Reply with exactly one line: CALL <tool_name>(<args>)")

# 스키마가 적힌 문서는 *아직 검색되지 않았다* (검색하면 얻을 수 있음)
KB_SEEN = (
    "1. Filing a Dispute - Customer Guidance\n   ID: doc_disp_003\n"
    "   Content: Customers may dispute a posted transaction within 60 days. The agent files the "
    "dispute on the customer's behalf using the internal dispute tool. Disputes are reviewed "
    "within 10 business days.\n")

CASES = {
    "missing_arg": dict(
        gold_param="reason_code",
        ctx=[
            {"role": "user", "content": "There's a charge I don't recognize - txn_77aa12 on my Gold card. Please dispute it."},
            {"role": "assistant", "content": "CALL get_credit_card_accounts(customer_id='cust_5521')"},
            {"role": "tool", "content": "Accounts: [{'credit_card_account_id': 'cc_584f9c5d00_gold', 'type': 'Gold Rewards Card'}]"},
            {"role": "assistant", "content": "CALL search_knowledge_base(query='dispute a transaction')"},
            {"role": "tool", "content": KB_SEEN},
            {"role": "assistant", "content": "CALL call_discoverable_agent_tool(agent_tool_name='file_dispute_4829', "
                                             "arguments='{\"credit_card_account_id\": \"cc_584f9c5d00_gold\", \"transaction_id\": \"txn_77aa12\"}')"},
            {"role": "tool", "content": "Error: Missing required parameter."},
            {"role": "assistant", "content": "CALL call_discoverable_agent_tool(agent_tool_name='file_dispute_4829', "
                                             "arguments='{\"credit_card_account_id\": \"cc_584f9c5d00_gold\", \"transaction_id\": \"txn_77aa12\", \"note\": \"unrecognized\"}')"},
        ],
        fb={
            1: "Error: Missing required parameter.",
            2: ("Error: Missing required parameter for 'file_dispute_4829' - one parameter declared "
                "in that tool's documented schema has no value, and 'note' is not a declared "
                "parameter."),
            3: ("Error: Missing required parameter for 'file_dispute_4829'. Declared parameters: "
                "credit_card_account_id, transaction_id, reason_code."),
            4: "Error: Missing required parameter 'reason_code' for 'file_dispute_4829'.",
            5: ("Error: call CALL call_discoverable_agent_tool(agent_tool_name='file_dispute_4829', "
                "arguments='{\"credit_card_account_id\": \"cc_584f9c5d00_gold\", \"transaction_id\": "
                "\"txn_77aa12\", \"reason_code\": \"unauthorized\"}')"),
        },
        leak="reason_code"),
    "enum_violation": dict(
        gold_param="unauthorized",
        ctx=[
            {"role": "user", "content": "Dispute txn_77aa12 on my Gold card - I never made that purchase."},
            {"role": "assistant", "content": "CALL search_knowledge_base(query='dispute a transaction')"},
            {"role": "tool", "content": KB_SEEN},
            {"role": "assistant", "content": "CALL call_discoverable_agent_tool(agent_tool_name='file_dispute_4829', "
                                             "arguments='{\"credit_card_account_id\": \"cc_584f9c5d00_gold\", \"transaction_id\": \"txn_77aa12\", "
                                             "\"reason_code\": \"fraudulent charge\"}')"},
            {"role": "tool", "content": "Error: Invalid value for parameter 'reason_code'."},
            {"role": "assistant", "content": "CALL call_discoverable_agent_tool(agent_tool_name='file_dispute_4829', "
                                             "arguments='{\"credit_card_account_id\": \"cc_584f9c5d00_gold\", \"transaction_id\": \"txn_77aa12\", "
                                             "\"reason_code\": \"not_recognized\"}')"},
        ],
        fb={
            1: "Error: Invalid value for parameter 'reason_code'.",
            2: ("Error: Invalid value for parameter 'reason_code' - the value must be one of the "
                "codes enumerated in that tool's documented schema, not free text."),
            3: ("Error: Invalid value for parameter 'reason_code'. Allowed values: unauthorized, "
                "duplicate, incorrect_amount, service_not_received."),
            4: "Error: Invalid value for parameter 'reason_code'; use 'unauthorized'.",
            5: ("Error: call CALL call_discoverable_agent_tool(agent_tool_name='file_dispute_4829', "
                "arguments='{\"credit_card_account_id\": \"cc_584f9c5d00_gold\", \"transaction_id\": "
                "\"txn_77aa12\", \"reason_code\": \"unauthorized\"}')"),
        },
        leak="unauthorized"),
}


def gen(msgs, seed, max_tokens=240):
    body = {"model": MODEL, "messages": msgs, "temperature": 0.0,
            "max_tokens": max_tokens, "seed": seed}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def classify(key, out, case):
    o = out.strip()
    if "search_knowledge_base" in o:
        return "search"          # 스키마 문서를 찾으러 감 = 올바른 다음 수
    if key == "missing_arg":
        if re.search(r"reason_code", o):
            return "gold"
        return "invent" if "call_discoverable" in o else "other"
    else:
        m = re.search(r"reason_code\"?\s*:\s*\"?([A-Za-z_ ]+)", o)
        if m and m.group(1).strip() == "unauthorized":
            return "gold"
        return "invent" if m else "other"


dumps = []
print("=== X5 v3: error-class expansion (32B, %d seeds) ===" % SEEDS, flush=True)
for key, case in CASES.items():
    for grade in (1, 2, 3, 4, 5):
        cnt = {"gold": 0, "search": 0, "invent": 0, "other": 0}
        parrot = 0
        for seed in range(SEEDS):
            msgs = [{"role": "system", "content": SYS}] + case["ctx"] + [
                {"role": "tool", "content": case["fb"][grade]},
                {"role": "user", "content": "Continue. Reply with exactly one CALL line."}]
            out = gen(msgs, seed)
            k = classify(key, out, case)
            cnt[k] += 1
            if k == "gold" and case["leak"].lower() in case["fb"][grade].lower():
                parrot += 1
            if seed < 2:
                dumps.append((key, grade, seed, k, out.strip()[:220]))
        rec = cnt["gold"] + cnt["search"]
        print("%-16s g%d  recovery=%d/%d (gold=%d search=%d invent=%d other=%d) parrot=%d"
              % (key, grade, rec, SEEDS, cnt["gold"], cnt["search"], cnt["invent"],
                 cnt["other"], parrot), flush=True)

print("")
print("=== verbatim dumps (seed 0-1) ===")
for key, grade, seed, k, txt in dumps:
    print("--- %s g%d s%d [%s] ---" % (key, grade, seed, k))
    print(txt.replace("\n", " ")[:200])
