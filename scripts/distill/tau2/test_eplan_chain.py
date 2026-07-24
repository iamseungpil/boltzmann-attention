#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-PLAN intent-chain coverage selftest (2026-07-24 피벗·C133·write_tools 확장).
chain_gap: 신호매칭+집합차(결정론). 합성 + rall23 043 실궤적 replay(미완 사슬 검출).
Run: py -3 test_eplan_chain.py"""
import gzip
import json
import os
import sys
from types import SimpleNamespace as NS

os.environ.setdefault("T2_EPLAN", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_eplan_patch as E  # noqa: E402

SPEC = E.load_eplan_spec("banking_knowledge")


def um(c):
    return NS(role="user", content=c, tool_calls=None, error=False, id=None)


def call(disp_name, ok=True, cid="x"):
    return NS(role="assistant", content=None, error=False, id=None,
              tool_calls=[NS(name="call_discoverable_agent_tool", id=cid,
                             arguments={"agent_tool_name": disp_name, "arguments": "{}"})])


def res(cid, error=False):
    return NS(role="tool", content=("ok" if not error else "Error"), tool_calls=None,
              error=error, id=cid)


def main():
    ok = True
    # ① close 신호 + write 0 → 전 필수 missing (C148: close는 required_writes서 제외=강제 안 함·
    #    finalize_writes 게이트로만·retention-accept시 close 억제). 필수 write=log_reason+apply_flag.
    msgs = [um("I'd like to close my Platinum Rewards Card.")]
    g = E.chain_gap(msgs, SPEC)
    good = (g is not None and "close_credit_card_account" not in g["missing_writes"]
            and "apply_credit_card_account_flag" in g["missing_writes"]
            and "log_credit_card_closure_reason" in g["missing_writes"]
            and len(g["missing_reads"]) == 3)
    print("[%s] close_signal_prereqs_missing_no_close" % ("PASS" if good else "FAIL"))
    ok &= good
    # ② close 신호 + apply_flag(retention) 실행 → apply_flag missing서 빠짐·executed_count↑
    msgs = [um("close my credit card"),
            call("apply_credit_card_account_flag_6147", ok=True, cid="c1"), res("c1")]
    g = E.chain_gap(msgs, SPEC)
    good = (g is not None and "apply_credit_card_account_flag" not in g["missing_writes"]
            and g["executed_count"] == 1)
    print("[%s] retention_executed_removed (exec=%s)" % ("PASS" if good else "FAIL", g and g["executed_count"]))
    ok &= good
    # ③ apply_flag 실행이 ERROR → 여전히 missing(성공만 계수)
    msgs = [um("close my card"), call("apply_credit_card_account_flag_6147", cid="c2"), res("c2", error=True)]
    g = E.chain_gap(msgs, SPEC)
    good = g is not None and "apply_credit_card_account_flag" in g["missing_writes"]
    print("[%s] errored_write_still_missing" % ("PASS" if good else "FAIL"))
    ok &= good
    # ④ 신호 없음(dispute 태스크) → chain None(qty 폴백)
    g = E.chain_gap([um("I need to dispute 8 charges")], SPEC)
    good = g is None
    print("[%s] no_close_signal_none" % ("PASS" if good else "FAIL"))
    ok &= good
    # ⑤ 전 필수 완료 → None (gap 없음=종료)
    full = [um("close my account")]
    cid = 0
    for t in ["get_closure_reason_history_1", "get_pending_replacement_orders_2",
              "get_user_dispute_history_3", "close_credit_card_account_4",
              "log_credit_card_closure_reason_5", "apply_credit_card_account_flag_6"]:
        cid += 1
        full += [call(t, cid=str(cid)), res(str(cid))]
    g = E.chain_gap(full, SPEC)
    good = g is None
    print("[%s] all_done_none" % ("PASS" if good else "FAIL"))
    ok &= good
    # ⑥ directive 리마인더 문구
    g = E.chain_gap([um("close my platinum card")], SPEC)
    r = E.chain_reminder(g)
    good = ("Do NOT end" in r and "apply_credit_card_account_flag" in r
            and "get_closure_reason_history" in r)  # C148: close 제거·retention write가 대신
    print("[%s] directive_names_missing" % ("PASS" if good else "FAIL"))
    ok &= good

    # ⑦ rall23 043 실궤적 replay: 미완 사슬(close 안 됨) → gap 검출
    gz = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                      "bank_rall23b_20260724.results.json.gz")
    if os.path.exists(gz):
        d = json.load(gzip.open(gz))
        sim = next((s for s in d["simulations"] if str(s.get("task_id")) == "task_043"), None)
        if sim:
            M = [NS(role=m.get("role"), content=m.get("content"), error=m.get("error"),
                    id=m.get("id"), tool_calls=[NS(name=t.get("name"), id=t.get("id"),
                    arguments=t.get("arguments")) for t in (m.get("tool_calls") or [])])
                 for m in sim["messages"]]
            g = E.chain_gap(M, SPEC)
            # rall23 043: close 미실행 → gap 있어야 함
            good = g is not None and len(g["missing_writes"]) >= 1
            print("[%s] rall23_043_incomplete_chain_detected (missing_writes=%s)"
                  % ("PASS" if good else "FAIL", g and g["missing_writes"]))
            ok &= good
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
