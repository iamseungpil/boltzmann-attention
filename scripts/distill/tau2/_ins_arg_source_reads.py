# -*- coding: utf-8 -*-
"""일회성 — P4(2026-08-21·[[64]]·t7335 halfB 079): `arg_source_reads` A2 저작 2사본 삽입.

무엇: FAB_STRIP 차단 노트의 해소-read 지목용 **인자명 → 원천-read 도구 목록** 선언.
소비자 = t2_gate_patch._fab_fix_note 하나(선언 나열만·선택 0·엔진 도메인 리터럴 0).
출처([[23]]): 전부 env 레지스트리(`a2/env_surface.json` desc 축자)와 우리 로컬 궤적의 env 출력
관측(bank_hve2e9_base_20260723: 'users' 레코드에 user_id 필드 실재). tasks/gold 무참조.
목록 순서 = 해소 순서(앞 read 출력이 뒤 read 인자를 준다·예: 계좌 목록 → checking id → 카드 목록).
층 = L3 specific(도메인-특화 선언) + gate.json 동기([[24]] 양방향·test_a2_three_layer ①).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"]

ASR = {
 "_note": ("2026-08-21 P4([[64]]·t7335 halfB 079 [26]/[32] 접힘): FAB_STRIP 차단 노트의 해소-read "
           "지목용 인자명→원천-read 맵. 1회 저작([[72]])·출처는 전부 env 레지스트리 desc 축자"
           "(a2/env_surface.json)와 로컬 궤적의 env 출력 관측([[23]]·tasks/gold 무참조). 소비자 = "
           "t2_gate_patch._fab_fix_note 하나(나열만·선택 0). 목록 순서 = 해소 순서. FAB_STRIP 이 "
           "잡는 값은 정의상 ctx 부재(손님 발화도 ctx 포함)이므로 '손님이 준 값' 케이스는 여기 "
           "도달하지 않는다 — 해소가 read 인 것이 맞다. name/customer_name/full_name 계열은 넣지 "
           "않는다: 이름은 레코드-파생이 아니라 손님-제공 값이라 해소가 read 가 아니고(PROV/regen "
           "관할·074/079 'John Doe' 차단이 그 경로), 잘못 지목하면 [[25]] 위반이 된다."),
 "account_id": ["get_all_user_accounts_by_user_id_3847"],
 "_note_account_id": ("env desc 축자: get_all_user_accounts_by_user_id_3847 'Retrieve all accounts "
                      "(checking, savings, credit cards) for a customer.' — 계좌 id 는 이 목록 "
                      "레코드에서 복사. checking/source/destination 변형도 같은 목록이 원천."),
 "checking_account_id": ["get_all_user_accounts_by_user_id_3847"],
 "source_account_id": ["get_all_user_accounts_by_user_id_3847"],
 "destination_account_id": ["get_all_user_accounts_by_user_id_3847"],
 "card_id": ["get_all_user_accounts_by_user_id_3847", "get_debit_cards_by_account_id_7823"],
 "_note_card_id": ("env desc 축자: get_debit_cards_by_account_id_7823 'Retrieve all debit cards "
                   "associated with a checking account. Returns card details including status, issue "
                   "reason, and expiration date.' · 'account_id (string): The checking account ID' — "
                   "checking id 가 선행 입력이므로 계좌 목록 read 를 앞에 둔다. 079 실패 경로가 "
                   "정확히 이 두 read 부재였다(오테이블 ×3·클래스명-as-ID ×2)."),
 "credit_card_account_id": ["get_credit_card_accounts_by_user"],
 "_note_credit_card_account_id": ("env desc 축자: get_credit_card_accounts_by_user 'Get all credit "
                                  "card accounts for a user.'"),
 "transaction_id": ["get_bank_account_transactions_9173", "get_credit_card_transactions_by_user"],
 "_note_transaction_id": ("env desc 축자: get_bank_account_transactions_9173 'Retrieve the "
                          "transaction history for a bank account.' · get_credit_card_transactions_"
                          "by_user 'Get all credit card transactions for a user.' — 은행/카드 거래 "
                          "양쪽이 transaction_id 원천이라 둘 다 나열([[70]] 절충 공개: 어느 쪽인지는 "
                          "write 종류에 달렸고 그 선택은 모델 몫·read 는 비변이라 과다 나열 무해)."),
 "user_id": ["get_user_information_by_name", "get_user_information_by_email"],
 "_note_user_id": ("env desc 축자: get_user_information_by_name 'Get the information ... for a user "
                   "by their name.' — 로컬 궤적 관측(bank_hve2e9_base_20260723)으로 출력 레코드에 "
                   "user_id 필드 실재 확인('user_id: 224959b99e'). by_id 는 user_id 를 이미 "
                   "요구하므로 원천이 될 수 없어 제외(기계 도출: args 에 user_id 없는 users 조회)."),
 "email": ["get_user_information_by_name", "get_user_information_by_id"],
 "address": ["get_user_information_by_name", "get_user_information_by_id"],
 "phone": ["get_user_information_by_name", "get_user_information_by_id"],
 "phone_number": ["get_user_information_by_name", "get_user_information_by_id"],
 "shipping_address": ["get_user_information_by_name", "get_user_information_by_id"],
 "_note_contact_fields": ("env desc 축자: get_user_information_by_* 'Get the information (date of "
                          "birth, email, phone number, address) for a user ...' — 연락처류 인자의 "
                          "레코드 원천은 users 레코드. 손님이 다른 값을 새로 주는 경우는 ctx 에 "
                          "실재하므로 FAB_STRIP 에 잡히지 않는다(위 _note 참조)."),
}

for rel in PATHS:
    p = os.path.join(HERE, rel)
    j = json.load(io.open(p, encoding="utf-8"))
    if "arg_source_reads" in j:
        print("already present in %s — overwrite" % rel)
    j["arg_source_reads"] = ASR
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("updated %s" % rel)

a = json.load(io.open(os.path.join(HERE, PATHS[0]), encoding="utf-8"))["arg_source_reads"]
b = json.load(io.open(os.path.join(HERE, PATHS[1]), encoding="utf-8"))["arg_source_reads"]
print("2사본 json-등가:", a == b)
