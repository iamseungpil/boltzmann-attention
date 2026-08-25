# -*- coding: utf-8 -*-
"""A2 두 정본 층에 **선언된 열거값** 여섯 칸을 추가한다([[24]] 양쪽 동일).

출처는 도구 사용법 문서(tools.py 독스트링) 축자 하나뿐이다 — gold 도 env 오류문도 아니다([[23]]).
엔진은 소속 판정과 명단 반환뿐이고 어느 값이 옳은지는 모델이 정한다([[62]]③④·[[22]] 닫힌 술어).
"""
import json, io, sys

FB = ("`{val}` is not one of the values `{arg}` accepts. "
      "Use exactly one of these, copied verbatim: {candidates}.")

WHY = ("★2026-08-25 신설. 왜(t7354 라이브 실측·정본 `t2_forensic.action_diff` 로 짝지어 봄): "
       "085 grpA1 t0 의 유일한 미달 행 `file_debit_card_transaction_dispute_6281` 이 "
       "gold 와 **인자 표기에서만** 갈린다 — `transaction_type` gold='atm_withdrawal' ↔ "
       "got='ATM Withdrawal' · `dispute_category` gold='atm_cash_discrepancy' ↔ got='ATM Error / "
       "cash not received (partial dispense)' · `pin_compromised` gold='no' ↔ got='No'(대문자) · "
       "`card_action` gold='keep_active' ↔ got='None'. 040 은 같은 계열이 신용 도구에서 난다 — "
       "`card_action` gold='keep_active' ↔ got='dispute' · `resolution_requested` "
       "gold='full_refund' ↔ got='Refund'. 그 넷은 env 가 `Error: Invalid <arg>. Must be one of: "
       "[...]` 로 되튕기고 모델은 msg38→msg82 까지 **22 메시지를 재시도에 태운다**. "
       "⚠출처 = 도구 사용법 문서 축자(아래 `_source_`) 하나. ⚠엔진은 고르지 않는다.")

NEW = [
    # ── 직불 분쟁 도구
    dict(arg="transaction_type",
         prefix="file_debit_card_transaction_dispute",
         values=["pin_purchase", "signature_purchase", "online_purchase", "atm_withdrawal",
                 "atm_deposit", "recurring_payment", "person_to_person"],
         source=("transaction_type (string): Type of transaction. Must be one of: 'pin_purchase', "
                 "'signature_purchase', 'online_purchase', 'atm_withdrawal', 'atm_deposit', "
                 "'recurring_payment', 'person_to_person'")),
    dict(arg="pin_compromised",
         prefix="file_debit_card_transaction_dispute",
         values=["yes_shared", "yes_observed", "no", "unknown"],
         source=("pin_compromised (string): Whether the customer's PIN may have been compromised. "
                 "Must be one of: 'yes_shared', 'yes_observed', 'no', 'unknown'")),
    dict(arg="card_action",
         prefix="file_debit_card_transaction_dispute",
         values=["keep_active", "freeze_pending_investigation", "close_and_reissue"],
         source=("card_action (string): Action to take on the card. Must be one of: 'keep_active', "
                 "'freeze_pending_investigation', 'close_and_reissue'")),
    # ── 신용 분쟁 도구
    dict(arg="card_action",
         prefix="file_credit_card_transaction_dispute",
         values=["keep_active", "cancel_and_reissue"],
         source=("card_action (string): Flag indicating the card's status. Must be one of: "
                 "'keep_active' (card remains active, dispute only), 'cancel_and_reissue' "
                 "(card is being cancelled and replaced). This is for record-keeping only and "
                 "does NOT order a replacement card.")),
    dict(arg="resolution_requested",
         prefix="file_credit_card_transaction_dispute",
         values=["full_refund", "partial_refund"],
         source=("resolution_requested (string): Resolution being requested. Must be one of: "
                 "'full_refund', 'partial_refund'")),
]

DC_VALUES = ["unauthorized_transaction", "atm_cash_discrepancy", "atm_deposit_not_credited",
             "duplicate_charge", "incorrect_amount", "goods_services_not_received",
             "recurring_charge_after_cancellation", "card_present_fraud",
             "card_not_present_fraud"]
DC_SOURCE = ("dispute_category (string): Category of the dispute. Must be one of: "
             "'unauthorized_transaction', 'atm_cash_discrepancy', 'atm_deposit_not_credited', "
             "'duplicate_charge', 'incorrect_amount', 'goods_services_not_received', "
             "'recurring_charge_after_cancellation', 'card_present_fraud', "
             "'card_not_present_fraud'")


def spec(d):
    return {
        "applies_to": "call_discoverable_agent_tool",
        "applies_when": {"arg": "agent_tool_name", "prefix": d["prefix"]},
        "arg": d["arg"],
        "values": d["values"],
        "feedback": FB,
        "_source_": d["source"],
        "_note_": WHY,
    }


def find_holder(o):
    if isinstance(o, dict):
        if "write_arg_enum" in o:
            return o
        for v in o.values():
            r = find_holder(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_holder(v)
            if r is not None:
                return r
    return None


for f in ("a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"):
    raw = io.open(f, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    d = json.loads(raw)
    holder = find_holder(d)
    ens = holder["write_arg_enum"]
    # ⑴ 직불 dispute_category 는 **이미 있는 칸**에 값 목록만 얹는다(불리언 선언과 같은 자리).
    for s in ens:
        if (s.get("arg") == "dispute_category"
                and (s.get("applies_when") or {}).get("prefix") == "file_debit_card_transaction_dispute"):
            if not s.get("values"):
                s["values"] = DC_VALUES
                s["feedback"] = FB
                s["_source_dispute_category_"] = DC_SOURCE
    # ⑵ 없는 칸만 추가한다(중복 방지).
    have = {(x.get("arg"), (x.get("applies_when") or {}).get("prefix")) for x in ens}
    added = 0
    for n in NEW:
        if (n["arg"], n["prefix"]) in have:
            continue
        ens.append(spec(n))
        added += 1
    out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
    io.open(f, "w", encoding="utf-8", newline="").write(out.replace("\n", nl))
    print("%s  specs=%d  added=%d  nl=%r" % (f, len(ens), added, nl))
