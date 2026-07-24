#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BRANCH-REGROUND selftest (2026-07-24·C146). resurface_doc(focused 정책문서 추출)
+ branch_reground_reminder(read=이름/조건write=문서·dedup). 합성 + rall25a 043 실궤적.
전부 결정론(LLM 0). Run: py -3 test_branch_reground.py"""
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

# 합성 KB 출력(번호목록 형식·doc 헤더=번호줄+다음줄 ID:) — 2 doc 블록, 하나만 대상 명시
KB_OUT = (
    "1. Generic account info\n"
    "   ID: doc_generic_001\n"
    "   Score: 5.0\n"
    "   Content: This document explains general account basics with no special tools.\n"
    "2. Retention Protocol\n"
    "   ID: doc_retention_003\n"
    "   Score: 9.9\n"
    "   Content: ## Step 4 Retention. For 2+ year customers with an annual fee concern,\n"
    "   waive the fee using apply_special_flag_6147 with flag_type='annual_fee_waived'.\n")


def tool_msg(content, cid="t1"):
    return NS(role="tool", content=content, tool_calls=None, error=False, id=cid)


def asst_msg(content):
    return NS(role="assistant", content=content, tool_calls=None, error=False, id=None)


def main():
    ok = True

    # ① resurface_doc: 대상 도구(suffix strip) 명시 블록 추출·focused(전체 아님)
    doc = E.resurface_doc([tool_msg(KB_OUT)], "apply_special_flag")
    good = doc is not None and "apply_special_flag" in doc and "annual_fee_waived" in doc \
        and "Generic account info" not in doc and len(doc) < len(KB_OUT)
    print("[%s] resurface_focused_block (len=%s)" % ("PASS" if good else "FAIL", doc and len(doc)))
    ok &= good

    # ② resurface_doc: 대상 명시 문서 없음 → None
    good = E.resurface_doc([tool_msg(KB_OUT)], "nonexistent_tool") is None
    print("[%s] resurface_none_when_absent" % ("PASS" if good else "FAIL"))
    ok &= good

    # ③ resurface_doc: tool-role만 읽음(assistant 출력의 도구명은 무시)
    good = E.resurface_doc([asst_msg(KB_OUT)], "apply_special_flag") is None
    print("[%s] resurface_tool_role_only" % ("PASS" if good else "FAIL"))
    ok &= good

    # ④ branch_reground_reminder: read=이름만(문서 없음)·조건write=정책문서 첨부
    chain = {"missing_reads": ["get_user_dispute_history"],
             "missing_writes": ["apply_special_flag"], "executed_count": 3, "phrase": "P."}
    r = E.branch_reground_reminder(chain, [tool_msg(KB_OUT)], SPEC)
    good = ("get_user_dispute_history" in r and "[POLICY" in r
            and "annual_fee_waived" in r  # 정책문서 본문 첨부됨
            and "read (unlock+call): get_user_dispute_history" in r)
    print("[%s] reminder_read_name_write_doc" % ("PASS" if good else "FAIL"))
    ok &= good

    # ⑤ 문서 없는 write → simple_w(이름만)·POLICY 블록 0
    chain2 = {"missing_reads": [], "missing_writes": ["plain_write_tool"],
              "executed_count": 1, "phrase": None}
    r2 = E.branch_reground_reminder(chain2, [tool_msg(KB_OUT)], SPEC)
    good = "plain_write_tool" in r2 and "[POLICY" not in r2
    print("[%s] write_without_doc_name_only" % ("PASS" if good else "FAIL"))
    ok &= good

    # ⑥ dedup: 두 write가 같은 doc 블록 → POLICY 블록 1회·라벨 2개
    kb2 = KB_OUT.replace("apply_special_flag_6147",
                         "apply_special_flag_6147 and close_special_5")
    chain3 = {"missing_reads": [], "phrase": None, "executed_count": 0,
              "missing_writes": ["apply_special_flag", "close_special"]}
    r3 = E.branch_reground_reminder(chain3, [tool_msg(kb2)], SPEC)
    good = r3.count("[POLICY") == 1 and "apply_special_flag" in r3 and "close_special" in r3
    print("[%s] dedup_same_doc_one_block (blocks=%d)"
          % ("PASS" if good else "FAIL", r3.count("[POLICY")))
    ok &= good

    # ⑦b for_finalize=True: finalize_writes(close) 제외·"종결 前 선행 먼저" 프레이밍
    chain_f = {"missing_reads": ["get_user_dispute_history"],
               "missing_writes": ["apply_special_flag", "close_credit_card_account"],
               "executed_count": 3, "phrase": None}
    rf = E.branch_reground_reminder(chain_f, [tool_msg(KB_OUT)], SPEC, for_finalize=True)
    good = ("close_credit_card_account" not in rf.split("[POLICY")[0]  # 남은목록서 close 제외
            and "STOP" in rf and "before you close" in rf.lower()
            and "get_user_dispute_history" in rf and "annual_fee_waived" in rf)
    print("[%s] for_finalize_excludes_close" % ("PASS" if good else "FAIL"))
    ok &= good

    # ⑦ rall25a 043 실궤적: apply_flag 조건write=doc_003(annual_fee_waived) 첨부·dispute=이름만
    gz = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                      "bank_rall25a_20260724.results.json.gz")
    if os.path.exists(gz):
        d = json.load(gzip.open(gz))
        sim = next((s for s in d["simulations"] if str(s.get("task_id")) == "task_043"), None)
        M = [NS(role=m.get("role"), content=m.get("content"), error=m.get("error"),
                id=m.get("id"), tool_calls=[NS(name=t.get("name"), id=t.get("id"),
                arguments=t.get("arguments")) for t in (m.get("tool_calls") or [])])
             for m in sim["messages"]]
        # close 직전 컷
        cut = []
        for m in M:
            if any("close_credit_card_account" in str(getattr(t, "arguments", ""))
                   for t in (m.tool_calls or [])):
                break
            cut.append(m)
        g = E.chain_gap(cut, SPEC)
        r = E.branch_reground_reminder(g, cut, SPEC)
        good = ("apply_credit_card_account_flag" in "".join(g["missing_writes"])
                and "annual_fee_waived" in r         # doc_003 재부각됨
                and "get_user_dispute_history" in r  # dispute=read 이름
                and "Retention Protocol" in r)       # 실제 정책문서 추출 확증
        print("[%s] real_043_apply_flag_doc_resurfaced (reminder_len=%d)"
              % ("PASS" if good else "FAIL", len(r)))
        ok &= good

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
