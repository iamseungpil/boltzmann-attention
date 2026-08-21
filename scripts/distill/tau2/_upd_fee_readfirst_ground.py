# -*- coding: utf-8 -*-
"""일회성 — P3: get_atm_fee_discrepancies 입력 출처 검산 (3사본·2026-08-21).

근거 = t7335 074 실측(HALFB §task_074): 거래 read 0회인 채 ① @last 참조 날조 ×4(→env
[ARGS-FORMAT] deny — "read 가 없다"는 상류 결손 무지목) ② 거래행 통짜 날조(txn12345/54321/
67890/98765 + 가공 fee/amount/network) → comparator 가 날조 입력을 성실히 판정 → discrepancy
에코 → 고객 보고 세탁([[25]] 역방향).

기존 기구가 왜 못 덮었나([[55]] 우리 배관 먼저 — 조사 확정):
  ① READ-FIRST(T2_SG_REQREADS·엔진 게이트 실재·go_stack ON): 이 도구에 `requires_reads`
     **미선언** — check_cli_eligibility 死배선 4호와 동형의 A2 공백.
  ② grounded_params(T2_PROD_BIND·엔진 실재·go_stack ON): 자매 comparator
     get_reward_discrepancies(ratefix)는 C211/F6a 로 transaction_id 를 선언해 발명-행
     비-에코를 이미 닫았는데, 이 도구에는 **미선언**.
  ③ ground.array_fields(T2_SG_GROUND): 원소별 {value, source} 인용 검산 축이라 source
     필드가 없는 이 행 스키마(transaction_id/fee_amount/withdrawal_amount/network)에는
     원리상 적용 불가 — 이 축의 결핍이 아니라 스키마 불일치.
⇒ 수리 = 엔진 신설 0·**기존 두 기구의 A2 선언을 채운다**(레버 신설 금지 준수).

검정 = test_comparator_read_first.py (선언·selector 부정통제·F6a op 거동).
"""
import copy
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]

RR = ["get_bank_account_transactions"]
RR_FB = (
    "Error: [READ-FIRST] this audit only judges fee lines that were READ in this "
    "conversation, and the required transaction read is missing: {missing}. "
    "get_bank_account_transactions is a DISCOVERABLE tool whose REAL name carries a numeric "
    "suffix - find its full suffixed name in the knowledge base (search the base name), "
    "unlock it, and call it with the checking account's id copied from the accounts listing "
    "(an account's display class name like 'Purple Account' is NOT an account id). Only rows "
    "copied from that read's output can be judged - rows typed from memory or with invented "
    "ids are dropped as unverified. Then call this tool again.")
RR_NOTE = (
    "2026-08-21 P3 (t7335 074 실측). 이 comparator 는 check_cli_eligibility 와 달리 "
    "requires_reads 가 없어, 거래 read 0회인 채 @last 참조 날조([4]×4)·행 통짜 날조([42])로 "
    "진행해도 READ-FIRST 가 한 번도 안 걸렸다(死배선 4호 동형·A2 선언 공백 = 커버리지 구멍). "
    "값의 출처 = 이 도구 자신의 description 축자('First retrieve the account's transactions, "
    "then call this ONCE PER ACCOUNT')와 params.transactions 계약('copied from the records') "
    "— gold 무참조([[23]]). 이름은 접미사 없이 쓴다(_eff_tool_name 이 _NNNN 을 지우고 대조). "
    "엔진 게이트 = 기존 T2_SG_REQREADS(신설 0).")
GP = {"transaction_id": {"producer_contains": ["record id:"]}}
GP_NOTE = (
    "2026-08-21 P3 (t7335 074 실측·HALFB §task_074): 날조 거래행이 출처 검산 없이 판정을 "
    "받아 discrepancy 로 에코되고 고객 보고까지 세탁됐다. 신설 아님 — 자매 comparator "
    "get_reward_discrepancies(ratefix)의 C211/F6a grounded_params.transaction_id 동형 선언을 "
    "이 도구에 채운다(발명-행 = 결핍 강등 → 판정 제외·비-에코·P4 지목 경로 합류). selector 를 "
    "테이블명이 아니라 'record id:'(env DB 레코드 덤프의 기계 포맷 — _byref_records/src0 "
    "안전판이 이미 쓰는 닫힌 술어)로 두는 근거: 이 도구의 read 는 디스패처 경유라 "
    "__tool_outputs blob 이 테이블 혼합이고, 우리 comparator 반환문에는 'Record ID:' 라인이 "
    "없어 자기 출력 에코(txn54321 (charged ...))로는 재접지가 성립하지 않는다(부정통제 = "
    "test_comparator_read_first). ±([[70]] 공개): 사는 것 = 날조 행의 판정·에코 차단. 파는 것 "
    "= main 이 그 계좌 거래를 안 읽고 서브 fetch 에만 의존하는 호출은 행 전체 unverified 기권 "
    "→ read 후 재호출 1왕복 추가(description 이 이미 main 선독을 계약으로 명시하므로 계약 밖 "
    "경로만 비싸진다). 효과 주장은 A/B 전 금지 — 본 선언은 단측 관찰 기반.")
ROW_FIELDS = ["transaction_id", "fee_amount", "withdrawal_amount", "network"]
ROW_NOTE = (
    "2026-08-21 P3: row_fields 는 fetch_formalize 모드가 소비하지 않는다(_sub_formalize/"
    "_sub_inject 전용) — 여기 선언하는 유일 효과는 _split_missing_fields 의 출처 분류: "
    "transaction_id 등 레코드-유래 필드가 결핍(P4b 강등 포함)일 때 abstain 문구가 '레코드에서 "
    "읽어 재호출하라'(이행 가능 지시)로 나가게 한다(C275 ⑤ 모순-지시 방지·C278 §2c).")

entries = []
for rel in PATHS:
    p = os.path.join(HERE, rel)
    j = json.load(io.open(p, encoding="utf-8"))
    hit = None
    for t in j.get("scaffold_get_tools") or []:
        if t.get("name") == "get_atm_fee_discrepancies":
            hit = t
            break
    if hit is None:
        print("MISSING in %s" % rel)
        sys.exit(1)
    hit["requires_reads"] = copy.deepcopy(RR)
    hit["requires_reads_feedback"] = RR_FB
    hit["_note_requires_reads"] = RR_NOTE
    hit["grounded_params"] = copy.deepcopy(GP)
    hit["_note_grounded_params"] = GP_NOTE
    iso = hit.get("isolate")
    if isinstance(iso, dict):
        iso["row_fields"] = list(ROW_FIELDS)
        iso["_note_row_fields"] = ROW_NOTE
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
        f.write("\n")
    entries.append(hit)
    print("updated %s" % rel)

print("3사본 json-등가:", entries[0] == entries[1] == entries[2])
