# -*- coding: utf-8 -*-
"""x397 G0-3 — 자명 문항 바닥효과 검정. A_min 과 같은 형식·같은 SYS·같은 도구목록으로 묻는다.
   V_proc = 정책 절차줄 포함(A_min 과 동일 형식) · V_bare = 절차줄 없이 요청만.
   게이트: EMIT(도구 이름을 내놓은 비율) >= 0.90"""
import os, sys, json, threading, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import x395_compliance_iso as X

PORT = 8141
docs = X.load_docs(); TOOLS = X.tool_universe(docs)

ITEMS = [
 ("I1", "제 당좌계좌 chk_a1b2c3d4 의 거래내역을 보여주세요.",
  "get_bank_account_transactions_9173",
  "- [Internal: Retrieving Bank Account Transaction Hi] Tool: get_bank_account_transactions_9173(account_id)"),
 ("I2", "제 사용자 ID 는 usr_k9m2p7 입니다. 제 명의의 모든 계좌를 보여주세요.",
  "get_all_user_accounts_by_user_id_3847",
  "- [Internal: Retrieving User Accounts] Use get_all_user_accounts_by_user_id_3847 to list the accounts."),
 ("I3", "제 직불카드 dbc_7f3e9a 를 지금 동결해 주세요. 방금 지갑을 잃어버렸습니다.",
  "freeze_debit_card_3892",
  "- [Internal: Freezing a Debit Card] Use freeze_debit_card_3892 to freeze the card."),
 ("I4", "지난주에 동결했던 제 직불카드 dbc_7f3e9a 를 다시 해제해 주세요. 카드를 찾았습니다.",
  "unfreeze_debit_card_3893",
  "- [Internal: Unfreezing a Debit Card] Use unfreeze_debit_card_3893 to unfreeze the card."),
 ("I5", "제 사용자 ID 는 usr_k9m2p7 입니다. 제가 가진 신용카드 계좌들을 보여주세요.",
  "get_credit_card_accounts_by_user",
  "- [Internal: Retrieving Credit Card Accounts] Use get_credit_card_accounts_by_user to list them."),
 ("I6", "제 당좌계좌 chk_a1b2c3d4 에 연결된 직불카드 목록을 보여주세요.",
  "get_debit_cards_by_account_id_7823",
  "- [Internal: Retrieving Debit Cards] Tool: get_debit_cards_by_account_id_7823(account_id)"),
 ("I7", "제 사용자 ID 는 usr_k9m2p7 입니다. 제가 지금까지 제기한 분쟁 이력을 보여주세요.",
  "get_user_dispute_history_7291",
  "- [Internal: Reviewing Dispute History] Use get_user_dispute_history_7291 to review prior disputes."),
 ("I8", "제 신용한도 상향 요청 이력을 보여주세요. 제 신용카드 계좌 ID 는 cc_5t8w1x 입니다.",
  "get_credit_limit_increase_history_4829",
  "- [Internal: Credit Limit Increase History] Tool: get_credit_limit_increase_history_4829(credit_card_account_id)"),
]

Q = ("\n\n# 질문\n지금 시점에서 **다음에 호출할 도구 하나**를 정하라. "
     "JSON 하나로만 답하라: {\"tool\": \"<이름>\", \"arguments\": {…}}")

jobs = []
for iid, ask, gold, proc in ITEMS:
    tools = "# 호출 가능한 도구(정책 문서가 정의한 것)\n" + ", ".join(TOOLS) + "\n\n"
    base = tools + ("# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n지금까지 호출한 도구: (없음)\n"
                    "대화에 등장한 레코드 id: (없음)\n\n" % ask)
    for var, body in (("V_proc", base + "# 정책 절차(축자)\n" + proc + Q),
                      ("V_bare", base + Q)):
        for k in range(3):
            jobs.append({"iid": iid, "gold": gold, "var": var, "k": k,
                         "temp": 0.0 if k == 0 else 0.7,
                         "msgs": [{"role": "system", "content": X.SYS},
                                  {"role": "user", "content": body}]})

lock, out = threading.Lock(), []
def work():
    while True:
        with lock:
            if not jobs: return
            j = jobs.pop(0)
        try: txt = X.call(PORT, j["msgs"], j["temp"])
        except Exception as e: txt = "ERROR " + str(e)[:200]
        nm, _ = X.parse_tool(txt)
        r = dict(iid=j["iid"], var=j["var"], k=j["k"], pred=nm, gold=j["gold"],
                 emit=bool(nm), exact=(bool(nm) and nm == j["gold"]), raw=txt[:300])
        with lock:
            out.append(r)
            print("  %-3s %-7s k=%d emit=%-5s exact=%-5s pred=%s" % (r["iid"], r["var"], r["k"], r["emit"], r["exact"], str(nm)[:44]))
ths = [threading.Thread(target=work) for _ in range(4)]
[t.start() for t in ths]; [t.join() for t in ths]

print("\n## G0-3 요약")
for var in ("V_proc", "V_bare"):
    rs = [r for r in out if r["var"] == var]
    print("  %-7s n=%d  EMIT=%.3f (%d/%d)  EXACT=%.3f (%d/%d)"
          % (var, len(rs), sum(r["emit"] for r in rs)/float(len(rs)), sum(r["emit"] for r in rs), len(rs),
             sum(r["exact"] for r in rs)/float(len(rs)), sum(r["exact"] for r in rs), len(rs)))
print("\n## 문항별")
for iid, ask, gold, proc in ITEMS:
    cells = []
    for var in ("V_proc", "V_bare"):
        rs = [r for r in out if r["var"] == var and r["iid"] == iid]
        cells.append("%s emit %d/%d exact %d/%d" % (var, sum(r["emit"] for r in rs), len(rs), sum(r["exact"] for r in rs), len(rs)))
    print("  %-3s %-40s %s" % (iid, gold[:40], " | ".join(cells)))
print("\n## emit=False 원문")
for r in out:
    if not r["emit"]: print("  %s %s k=%d :: %s" % (r["iid"], r["var"], r["k"], r["raw"][:220]))
print("\n## exact=False 이면서 emit=True 인 것(오지목)")
for r in out:
    if r["emit"] and not r["exact"]: print("  %s %s k=%d pred=%s gold=%s" % (r["iid"], r["var"], r["k"], r["pred"], r["gold"]))
o = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x397_g0_trivial.json"
open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("\n원자료: %s" % o)
