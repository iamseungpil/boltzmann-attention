# -*- coding: utf-8 -*-
"""A2 두 정본 층의 `write_rules` 에 **책임 한도 표**를 선언한다([[24]] 양쪽 동일).

출처 = 정책 문서 본문 축자 하나뿐 — gold 도 env 오류문도 아니다([[23]]).
엔진은 검색도 순위도 하지 않는다: 선언된 문장을 결정점에 그대로 싣는다.
"""
import json, io

TEXT = ("Maximum liability by reporting timing: reported within 2 business days of the "
        "statement = $50; reported within 60 days of the statement = $500; reported after "
        "60 days = unlimited liability, and the tool takes -1 for unlimited.")

NOTE = ("★2026-08-25 신설. 격리 x538(085·창 5·n4): 창 그대로 **12/20** ↔ 이 표를 결정점에 놓으면 "
        "**20/20** ↔ 같은 길이의 무관한 문장 **12/20**([[57]] 통과). 합성도 쟀다 — x538b 에서 "
        "이미 실려 있는 '가장 이른 중복' 문장과 **함께** 실었을 때 `B_both` **20/20** 으로 "
        "서로 죽이지 않는다([[19]]). 왜 필요한가(t7354 라이브 실측): 085 의 유일한 미달 행에서 "
        "표기 아홉이 닫히고 나면 남는 칸이 `customer_max_liability_amount` 하나이고 제출값이 "
        "**'0'** 이다 — 표 안에 없는 값이다. 격리의 A_asis 는 '30000'·'50000' 을 8/20 냈다"
        "(라이브의 '0' 을 축자로 재현하지는 못했다 — 그 단서를 남긴다). "
        "출처 = 문서 축자: \"- Reported within 2 business days of statement: Maximum liability "
        "$50 / - Reported within 60 days of statement: Maximum liability $500 / - Reported "
        "after 60 days: Unlimited liability - customer may not recover funds\" "
        "(doc_bank_accounts_bank_accounts_(general)_031 'Internal: Filing a Debit Card "
        "Transaction Dispute') + 도구 문서 \"customer_max_liability_amount (number): ... "
        "Use -1 for unlimited liability.\" ⚠엔진은 어느 티어가 옳은지 판정하지 않는다 — "
        "신고 시점을 읽는 것은 모델 몫이다([[62]]③④).")


def find_holder(o):
    if isinstance(o, dict):
        if "write_rules" in o:
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
    h = find_holder(d)
    rules = h["write_rules"]
    have = any(str(r.get("text") or "").startswith("Maximum liability by reporting") for r in rules)
    if not have:
        rules.append({"applies_to": "file_debit_card_transaction_dispute",
                      "text": TEXT, "_note_": NOTE})
    out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
    io.open(f, "w", encoding="utf-8", newline="").write(out.replace("\n", nl))
    print("%s  write_rules=%d  added=%s" % (f, len(rules), not have))
