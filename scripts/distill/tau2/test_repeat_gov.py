#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""반복 거버너 동등성 검정 — 전 단에서 **레거시 문자열과 바이트 동일**(L1·CONSOLIDATION §5)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import t2_repeat_gov as RG   # noqa: E402

# ── 레거시 축자 (t2_gate_patch.py HEAD에서 복사 — 이 사본이 검정의 기준) ────────
L_STUB = ("[DUPLICATE-READ] This exact call (same tool, same arguments) was "
          "already executed earlier in this conversation; its full output is "
          "shown above and has not changed. Refer to that output instead of "
          "re-reading.")
L_DIG = ("[DUPLICATE-READ] This exact call was already executed earlier; its "
         "output was COMPACTED from view to save space and has not changed. Do "
         "NOT re-run it. If a tool needs that data, pass it BY REFERENCE as "
         "@last:%s instead of re-reading.")
L_REDIR = (" Do NOT repeat this exact search. If you are looking up a discoverable "
           "tool, note that a bare function-name query matches no document text — "
           "search PLAIN WORDS describing the action/step (the everyday words a policy "
           "document would use), not the tool's function name. If you already have the "
           "information you need, proceed to the next step instead of searching again.")
L_ESC = (" You have now issued this IDENTICAL call %d times and the result "
         "has not changed once — repeating it again cannot produce new "
         "information. Change what you do: use DIFFERENT search words, or "
         "act on the information you already have, or ask the customer. Do "
         "not issue this same call again.")
L_CAP = (" [REPEAT-CAP] This identical call has now been issued %d times and is "
         "no longer being executed. Stop this line of action: state to the "
         "customer what you could not resolve, or take a DIFFERENT action. "
         "This has been recorded as an unresolved blocker.")


def legacy(tool, n, search, dig, capk):
    body = (L_DIG % tool) if dig else L_STUB
    redir = L_REDIR if search else ""
    if capk and capk > 3 and n >= capk:
        esc = L_CAP % n
    elif n >= 3:
        esc = L_ESC % n
    else:
        esc = ""
    return body + redir + esc, (n >= 3), bool(capk and capk > 3 and n >= capk)


ok = fail = 0


def chk(name, cond):
    global ok, fail
    ok, fail = ok + (1 if cond else 0), fail + (0 if cond else 1)
    print(("  ✓ " if cond else "  ✗ ") + name)


CASES = [  # (tool, n_rep, is_search, digested, cap_k)
    ("KB_search", 1, True, False, 0),          # 첫 스텁 + redirect
    ("get_user_information_by_id", 1, False, False, 0),   # 비검색 첫 스텁
    ("KB_search", 3, True, False, 0),          # esc 진입
    ("get_user_information_by_id", 5, False, False, 0),   # 비검색 esc
    ("KB_search", 8, True, False, 8),          # 캡
    ("KB_search", 12, True, False, 8),         # 캡 이후
    ("get_credit_card_transactions_by_user", 2, False, True, 0),   # 다이제스트
    ("KB_search", 4, True, True, 8),           # 다이제스트+검색+esc
    ("KB_search", 8, True, True, 8),           # 다이제스트+캡
    ("x", 3, False, False, 3),                 # ★cap_k=3은 무효(K>3 규율) → esc여야
]
for tool, n, srch, dig, capk in CASES:
    g = RG.ladder(tool, n, srch, dig, capk)
    l = legacy(tool, n, srch, dig, capk)
    chk("%s n=%d s=%d d=%d K=%s → 바이트 동일" % (tool[:24], n, srch, dig, capk), g == l)

# 캡 규율: K<=3이면 캡 미발동(레거시 `_cap > 3` 동일)
g = RG.ladder("x", 10, False, False, 3)
chk("K=3 → capped=False (K>3 규율)", g[2] is False)
# error 플래그 경계
chk("n=2 → error=False", RG.ladder("x", 2, False, False, 0)[1] is False)
chk("n=3 → error=True", RG.ladder("x", 3, False, False, 0)[1] is True)

print("\n%d PASS · %d FAIL" % (ok, fail))
sys.exit(1 if fail else 0)
