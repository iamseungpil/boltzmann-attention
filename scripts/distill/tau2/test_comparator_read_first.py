# -*- coding: utf-8 -*-
"""P3 — comparator 입력 출처 검산 검정 (2026-08-21 · t7335 074 실측 · HALFB §task_074).

결함(수리 전): get_atm_fee_discrepancies 가 거래 read 0회·통짜 날조 행(txn12345/54321/…)에
discrepancy 판정을 부여해 고객 보고까지 세탁됐다. 원인 = 엔진 결손이 아니라 **A2 선언 공백**:
requires_reads(READ-FIRST 게이트)와 grounded_params(C211/F6a 발명-행 강등)가 자매 도구
(check_cli_eligibility·get_reward_discrepancies ratefix)에는 있고 이 도구에만 없었다.

검정 축:
  ① 선언 실재·3사본 동기([[24]]) — requires_reads/feedback/grounded_params/row_fields
  ② 선언 불변식 — 이름이 문면에 축자·접미사 없음(test_requires_reads_wired ①②③ 동형)
  ③ selector 의미 — 진짜 덤프만 후보·날조 id 불검출·실재 id 검출(C211 test_f6a 동형)
  ④ 자기-에코 부정통제 — 우리 comparator 자신의 반환문(날조 id 에코 포함)은 'record id:'
     selector 에 **불일치** = 2차 호출 재접지 불가(P2 에코-그라운딩과 같은 family 의 차단)
  ⑤ F6a op 거동 — transaction_id 결핍(강등) 행은 판정 제외+계상·비-에코, 정상 행은 판정
  ⑥ 양성 대조 — 선언을 지운 사본에서 ①이 실제로 잡힌다

오프라인 전용(유료 X·[[09]]). 실행: py -3 test_comparator_read_first.py
"""
import copy
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_compute as TC  # noqa: E402

PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]
TOOL = "get_atm_fee_discrepancies"

PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def load_tool(rel):
    j = json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))
    return next((t for t in j.get("scaffold_get_tools") or [] if t.get("name") == TOOL), None)


def audit(t):
    """test_requires_reads_wired ①②③ 동형(이 도구 국소)."""
    bad = []
    fb = t.get("requires_reads_feedback")
    rr = t.get("requires_reads") or []
    if fb and not rr:
        bad.append("문면만 있고 requires_reads 없음(죽은 문면)")
    for r in rr:
        if fb and r not in fb:
            bad.append("%r 가 문면에 없음" % r)
        if re.search(r"_\d+$", str(r)):
            bad.append("%r 에 _NNNN 접미사" % r)
    return bad


def main():
    tools = {rel: load_tool(rel) for rel in PATHS}

    # ── ① 선언 실재·3사본 동기 ──────────────────────────────────────────────────
    print("[선언] requires_reads·grounded_params·row_fields 실재 + 3사본 동기([[24]])")
    t0 = tools[PATHS[0]]
    check("①: 3사본 모두 도구 실재", all(t is not None for t in tools.values()))
    check("①: requires_reads = ['get_bank_account_transactions']",
          t0.get("requires_reads") == ["get_bank_account_transactions"])
    check("①: grounded_params.transaction_id.producer_contains = ['record id:']",
          (t0.get("grounded_params") or {}).get("transaction_id", {})
          .get("producer_contains") == ["record id:"])
    check("①: isolate.row_fields 가 op 참조 필드를 덮는다",
          set((t0.get("isolate") or {}).get("row_fields") or [])
          >= {"transaction_id", "fee_amount", "withdrawal_amount", "network"})
    for k in ("requires_reads", "requires_reads_feedback", "grounded_params"):
        check("①: 3사본 '%s' 동일" % k,
              all((t or {}).get(k) == t0.get(k) for t in tools.values()))

    # ── ② 선언 불변식 ──────────────────────────────────────────────────────────
    print("[불변식] 이름 축자·접미사 없음(死배선 4호 재발 방지)")
    bad = audit(t0)
    check("②: 위반 0 — %s" % ("없음" if not bad else " · ".join(bad)), not bad)

    # ── ③ selector 의미 (C211 test_f6a 동형) ──────────────────────────────────
    print("[selector] 진짜 덤프만 후보·날조 id 불검출")
    dump = ("found 3 record(s) in a table:\n"
            "1. record id: btxn_63306834d5ba\n   amount: 2.00\n   description: non-rho atm fee\n"
            "2. record id: btxn_aa11\n   amount: 100.00\n   description: atm withdrawal\n")
    sels = [s.lower() for s in t0["grounded_params"]["transaction_id"]["producer_contains"]]
    outs = {"call_discoverable_agent_tool": dump}
    cands = [t for t in outs.values() if any(s in t for s in sels)]
    check("③: DB 덤프가 후보로 잡힌다", len(cands) == 1)
    check("③: 실재 id 는 후보 안에 실재(통과 방향)",
          any("btxn_63306834d5ba" in t for t in cands))
    check("③: 날조 id(074 실물 txn54321)는 후보 어디에도 없음(강등 방향)",
          not any("txn54321" in t for t in cands))

    # ── ④ 자기-에코 부정통제 ────────────────────────────────────────────────────
    print("[부정통제] 우리 comparator 반환문(날조 id 에코)은 selector 불일치 = 재접지 불가")
    detail = t0["detail_item_template"].format(id="txn54321", actual=1.50, expected=0.00,
                                               delta=1.50)
    own_out = str(t0["return_template"]).replace("{details}", detail)
    check("④: 에코 반환문에 날조 id 실재(에코 자체는 남는다 — 074 [45] 재현)",
          "txn54321" in own_out)
    check("④: 그 반환문은 'record id:' selector 에 불일치 → 후보가 못 된다",
          not any(s in own_out.lower() for s in sels))

    # ── ⑤ F6a op 거동 — 강등 행 판정 제외·비-에코, 정상 행 판정 ─────────────────────
    print("[F6a] transaction_id 결핍(P4b 강등) 행 = 판정 제외+계상·비-에코")
    rows = [
        # 정상 행: Bluest non_rho 기대 $2.00 ↔ 부과 $3.50 → discrepant
        {"transaction_id": "btxn_63306834d5ba", "fee_amount": 3.50,
         "withdrawal_amount": 100.0, "network": "non_rho"},
        # 강등 행(날조 → PROD_BIND 가 id 를 None 으로): 판정 제외
        {"transaction_id": None, "fee_amount": 1.50,
         "withdrawal_amount": 60.0, "network": "non_rho"},
    ]
    ctx = {"account_class": "Bluest Account", "transactions": rows}
    res = TC.apply_op(t0["op"], ctx)
    st = ctx.get("_sg_stats") or {}
    check("⑤: 정상 행만 판정(out_ids=['btxn_63306834d5ba'])", res == ["btxn_63306834d5ba"])
    check("⑤: None 비-에코(발명-행이 discrepant 로 안 나간다)", None not in res)
    check("⑤: skipped=1 + missing_fields['transaction_id']=1 계상",
          st.get("skipped") == 1
          and (st.get("missing_fields") or {}).get("transaction_id") == 1)

    # ── ⑤b row_fields 분류 — 결핍 문구가 '레코드에서 읽어 재호출' 로 나가는 축 ────────
    import t2_scaffold_get as SG
    rec, sub = SG._split_missing_fields({"transaction_id": 1}, t0.get("isolate"))
    check("⑤b: transaction_id 는 record-유래로 분류(이행 가능 지시 경로·C275 ⑤ 방지)",
          rec == {"transaction_id": 1} and sub == {})

    # ── ⑥ 양성 대조 ────────────────────────────────────────────────────────────
    print("[양성 대조] 선언을 지우면 검정이 잡는다")
    probe = copy.deepcopy(t0)
    probe.pop("requires_reads", None)
    check("⑥: requires_reads 삭제 사본에서 위반 검출", bool(audit(probe)))

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:")
        for x in FAIL:
            print("  - " + x)
        sys.exit(1)
    print("ALL PASS — 074 경로(무-read 호출·날조 행) 봉쇄 선언이 실재·동기·의미 합치.")


if __name__ == "__main__":
    main()
