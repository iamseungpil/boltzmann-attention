#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-PLAN 지속 구동 결정 로직 selftest (2026-07-24 피벗·EPLAN_PERSISTENT_DRIVER_DESIGN).
drive_decision 순수 함수: budget K·progress-guard·plan충족 종료 전수.
Run: py -3 test_eplan_persist.py"""
import os
import sys

os.environ.setdefault("T2_EPLAN", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_eplan_patch as E  # noqa: E402

D = E.drive_decision


def main():
    ok = True
    cases = [
        # (name, drives, K, exec_now, last_exec, has_gap, expect)
        ("first_drive_with_gap", 0, 4, 1, -1, True, "hold"),
        ("no_gap_terminates", 0, 4, 3, -1, False, "terminate"),
        ("budget_exhausted", 4, 4, 5, 4, True, "terminate"),
        ("progress_continues", 1, 4, 3, 2, True, "hold"),          # exec 2→3 진전 → 계속
        ("no_progress_releases", 1, 4, 2, 2, True, "release"),      # exec 2→2 정체 → 놓아줌
        ("regress_releases", 2, 4, 1, 3, True, "release"),          # exec 감소 → 놓아줌
        ("gap_closed_midway", 2, 4, 5, 3, False, "terminate"),      # 진전 있어도 gap 닫히면 종료
        ("last_budget_slot_holds", 3, 4, 4, 2, True, "hold"),       # drives 3<4·진전 → 마지막 hold
    ]
    for name, dr, K, en, le, hg, exp in cases:
        got = D(dr, K, en, le, hg)
        st = "PASS" if got == exp else "FAIL"
        ok &= (got == exp)
        print("[%s] %s -> %s (want %s)" % (st, name, got, exp))
    # 시나리오: 043형 — 4턴 구동, 매턴 1 write 진전, K=4 소진까지 hold, 이후 terminate
    print("--- scenario: 043 chain drive (progress each turn) ---")
    drives, last, K = 0, -1, 4
    seq = [1, 2, 3, 4, 5]  # executed writes after each turn
    holds = 0
    for turn, en in enumerate(seq):
        dec = D(drives, K, en, last, en < 5)  # gap until 5 writes done
        print("  turn %d exec=%d -> %s" % (turn, en, dec))
        if dec == "hold":
            holds += 1
            drives += 1
            last = en
    good = holds == 4  # 4번 구동(K), 5번째=gap 닫힘 terminate
    print("[%s] scenario_drives_to_completion (holds=%d)" % ("PASS" if good else "FAIL", holds))
    ok &= good
    # ── directive 리마인더(v1.4·C116 처방화) ──
    print("--- directive reminder ---")
    r = E.cp5_gap_reminder(8, 3, [], {"txn_a", "txn_b", "txn_c"}, "file the dispute for each remaining record")
    checks = [("names_done", "txn_a" in r and "txn_b" in r),
              ("states_remaining", "5 remain" in r),
              ("active_not_passive", "Do NOT end" in r and "re-check with the user" not in r),
              ("names_action", "file the dispute" in r)]
    for nm, c in checks:
        st = "PASS" if c else "FAIL"
        ok &= c
        print("[%s] directive_%s" % (st, nm))
    r2 = E.cp5_gap_reminder(2, 0, ["W1", "W3"], set(), None)
    good = "W1" in r2 and "read their details" in r2 and "Do not end" in r2
    print("[%s] directive_unexamined_read_first" % ("PASS" if good else "FAIL"))
    ok &= good

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
