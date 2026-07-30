# -*- coding: utf-8 -*-
"""X5 v2 — 피드백 사다리 재설계판 (v1 천장 실패 교정·X_FREE_TRACK §6 v2 설계 이행).

v1 대비 변경:
  (a) 도구 이름을 **명시 목록으로 주지 않는다** — KB 검색으로만 알 수 있는 discoverable 상태
      (실 banking 하네스 구조 모사: unlock→call 2단·이름은 문서 안에)
  (b) **반복-실패 진입**: 같은 오류를 2회 한 뒤의 상태에서 피드백 제공(ITS 사다리 실발동 지점)
  (c) **문맥 부하**: 앞선 대화·도구 출력을 채워 긴 문맥(실 궤적 근사)
  (d) **7B arm**(실패 원천) + 32B 대조
  (e) 회복 판정 강화: 올바른 dispatcher 2단 호출 형태 + 인자까지
측정: recovery / parroting(축자 복사) / 2차-일반화(등급1만).
"""
import json, re, urllib.request

MODELS = {"32B": ("http://localhost:8140/v1/chat/completions",
                  "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")}
N_SEED = 4

# ── 실 하네스 구조 모사: discoverable 도구는 KB 문서 안에 이름이 있음 ──
KB_DOC = (
    "1. Internal: Card Replacement Procedure\n   ID: doc_cards_replacement_014\n"
    "   Content: When a customer reports a damaged card, the agent must first verify the account "
    "record, then submit the replacement using the internal replacement tool. Requests are "
    "processed within 7-10 business days. Replacement requires the card account id and a reason "
    "code from {damaged, lost, stolen}.\n")
# 정답 도구 이름은 *다른* 문서에만 등장 → 첫 검색으론 못 얻음(실 궤적의 발견 부하)
KB_DOC2 = (
    "1. Internal: Replacement Tooling Reference\n   ID: doc_cards_tooling_041\n"
    "   Content: Use submit_card_replacement_request_8814(credit_card_account_id, reason_code) "
    "to file the replacement. Do not use the generic dispute tools for replacements.\n")

SYS = (
    "You are a Rho-Bank service agent. You have these agent tools:\n"
    "  search_knowledge_base(query: str)\n"
    "  get_credit_card_accounts(customer_id: str)\n"
    "  unlock_discoverable_agent_tool(agent_tool_name: str)\n"
    "  call_discoverable_agent_tool(agent_tool_name: str, arguments: str)\n"
    "Specialized internal tools are NOT listed: their exact names appear inside knowledge-base "
    "documents. To use one you must unlock it by its exact full name, then call it.\n"
    "Reply with exactly one line: CALL <tool_name>(<args>)")

# 반복-실패 문맥: 검색 1회(이름 없는 문서) → 발명 이름 2회 시도 → env 거부 2회
def build_context():
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "Hi, my Gold card is physically damaged - I need a replacement."},
        {"role": "assistant", "content": "CALL get_credit_card_accounts(customer_id='cust_5521')"},
        {"role": "tool", "content": "Accounts: [{'credit_card_account_id': 'cc_584f9c5d00_gold', "
                                    "'type': 'Gold Rewards Card', 'status': 'active'}]"},
        {"role": "assistant", "content": "CALL search_knowledge_base(query='damaged card replacement')"},
        {"role": "tool", "content": KB_DOC},
        {"role": "assistant", "content": "CALL unlock_discoverable_agent_tool("
                                         "agent_tool_name='submit_card_replacement')"},
        {"role": "tool", "content": "Error: Unknown discoverable tool 'submit_card_replacement'."},
        {"role": "assistant", "content": "CALL unlock_discoverable_agent_tool("
                                         "agent_tool_name='request_card_replacement')"},
        {"role": "tool", "content": "Error: Unknown discoverable tool 'request_card_replacement'."},
        {"role": "assistant", "content": "CALL unlock_discoverable_agent_tool("
                                         "agent_tool_name='card_replacement_request')"},
    ]

GOLD_NAME = "submit_card_replacement_request_8814"
FB = {
    1: "Error: Unknown discoverable tool 'card_replacement_request'.",
    2: ("Error: Unknown discoverable tool 'card_replacement_request' - the name you are unlocking "
        "was not found in any document you have retrieved; discoverable tool names must be copied "
        "verbatim from a knowledge-base document."),
    3: ("Error: Unknown discoverable tool 'card_replacement_request'. The name must come from a "
        "retrieved document; you have retrieved doc_cards_replacement_014 only, which does not "
        "contain a tool name. Search again with different plain words (the tooling reference for "
        "this action is a separate document)."),
    4: ("Error: Unknown discoverable tool 'card_replacement_request'. The correct tool name is in "
        "doc_cards_tooling_041; retrieve that document."),
    5: ("Error: Unknown discoverable tool. Use "
        "unlock_discoverable_agent_tool(agent_tool_name='%s')." % GOLD_NAME),
}

SECOND_TURN = (
    "Also my wife's Silver card was lost last week - please file a replacement for that one too "
    "(account cc_99b1_silver).")
FB1_SECOND = "Error: Unknown discoverable tool."


def gen(url, model, msgs, seed=None, max_tokens=220):
    body = {"model": model, "messages": msgs, "temperature": 0.0, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def classify(out):
    """recovery 유형: gold-name(정답 이름 도달) / search(재검색=올바른 다음 수) / invent(새 발명) / other"""
    o = out.strip()
    if GOLD_NAME in o:
        return "gold"
    if re.search(r"search_knowledge_base", o):
        return "search"
    m = re.search(r"agent_tool_name\s*=\s*['\"]([^'\"]+)", o)
    if m and m.group(1) != GOLD_NAME:
        return "invent"
    return "other"


rows = []
for tag, (url, model) in MODELS.items():
    for grade in (1, 2, 3, 4, 5):
        cnt = {"gold": 0, "search": 0, "invent": 0, "other": 0}
        parrot = 0
        second_gold = 0
        for seed in range(N_SEED):
            msgs = build_context() + [
                {"role": "tool", "content": FB[grade]},
                {"role": "user", "content": "Continue. Reply with exactly one CALL line."}]
            out = gen(url, model, msgs, seed=seed)
            k = classify(out)
            cnt[k] += 1
            if k == "gold" and GOLD_NAME in FB[grade]:
                parrot += 1
            # 2차: 같은 클래스 새 인스턴스 + 등급1만
            msgs2 = msgs + [
                {"role": "assistant", "content": out.strip().split("\n")[0][:200]},
                {"role": "tool", "content": "Tool unlocked: %s" % GOLD_NAME},
                {"role": "assistant", "content": "CALL call_discoverable_agent_tool("
                                                 "agent_tool_name='%s', arguments='{\"credit_card_account_id\": "
                                                 "\"cc_584f9c5d00_gold\", \"reason_code\": \"damaged\"}')" % GOLD_NAME},
                {"role": "tool", "content": "Replacement request submitted."},
                {"role": "user", "content": SECOND_TURN},
                {"role": "assistant", "content": "CALL unlock_discoverable_agent_tool("
                                                 "agent_tool_name='submit_lost_card_replacement')"},
                {"role": "tool", "content": FB1_SECOND},
                {"role": "user", "content": "Continue. Reply with exactly one CALL line."}]
            out2 = gen(url, model, msgs2, seed=seed)
            if classify(out2) in ("gold", "search"):
                second_gold += 1
        rows.append((tag, grade, cnt, parrot, second_gold))
        print("%s g%d  gold=%d search=%d invent=%d other=%d | parrot=%d | 2nd-ok=%d/%d"
              % (tag, grade, cnt["gold"], cnt["search"], cnt["invent"], cnt["other"],
                 parrot, second_gold, N_SEED), flush=True)

print("")
print("=== aggregate (recovery = gold or search; parroting = gold when name was fed) ===")
for tag, _ in MODELS.items():
    for g in (1, 2, 3, 4, 5):
        r = next(x for x in rows if x[0] == tag and x[1] == g)
        rec = r[2]["gold"] + r[2]["search"]
        print("%s grade %d: recovery %d/%d  gold-direct %d  parroting %d  2nd-ok %d/%d"
              % (tag, g, rec, N_SEED, r[2]["gold"], r[3], r[4], N_SEED))
