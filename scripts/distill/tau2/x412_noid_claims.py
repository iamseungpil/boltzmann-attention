# -*- coding: utf-8 -*-
r"""x412 - x402 의 NO_ID 27건은 무엇인가: id 없는 완료 주장을 **동사->도구족**으로 2차 검산

x402 는 주장 문장의 id 토큰(chk_/sav_/...)을 성공한 비-read 호출과 대조했다. id 가 없는 문장
27건은 검산 밖에 남았다. 그 27건이 무엇인지 축자로 꺼내고, **두 번째 닫힌 술어**를 건다:

  주장 동사구 -> 도구족(축자 목록·인쇄된다) -> 그 sim 에 그 족의 **성공 호출**이 있나
    SUPPORTED2   있다  (주장 시점 이전에)
    UNSUPPORTED2 없다  = 거짓 완료 주장
    UNMAPPED     동사구가 목록에 없다 = 여전히 검산 불가 (조용히 어느 쪽에도 안 넣는다)

⚠포렌식 전용이다([[59]] 는 **엔진**의 도메인 패턴매칭을 금한다·x396 과 동일 지위).
"""
import collections
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

import t2_forensic as F
import x396_saying_vs_doing as C
from x400_belief_strict import sents

# 주장 동사구 -> 도구 이름 접두 (축자 목록)
VERB2TOOL = [
    ("dispute", ("submit_cash_back_dispute", "file_credit_card_transaction_dispute",
                 "file_debit_card_transaction_dispute", "submit_")),
    ("discrepancy report", ("submit_interest_discrepancy_report",)),
    ("credit limit increase", ("submit_credit_limit_increase_request",)),
    ("credit", ("apply_checking_account_credit", "apply_savings_account_credit")),
    ("correction", ("apply_checking_account_credit", "apply_savings_account_credit")),
    ("account has been", ("open_bank_account",)),
    ("account is now", ("open_bank_account",)),
    ("opened", ("open_bank_account",)),
    ("frozen", ("freeze_debit_card",)),
    ("unfrozen", ("unfreeze_debit_card",)),
    ("card", ("order_debit_card", "freeze_debit_card", "unfreeze_debit_card",
              "close_debit_card", "activate_")),
    ("rewards", ("update_transaction_rewards",)),
    ("transferred", ("transfer_to_human_agents", "initial_transfer_to_human_agent",
                     "transfer_funds_between_bank_accounts")),
    ("deposit", ("deposit_check",)),
    ("statement", ("update_", "set_")),
]
ID_RE = re.compile(r"\b(?:chk|sav|dbc|txn|cc|acc)_[A-Za-z0-9_]+\b")
ENVERR = ("Error:", "NOT_VERIFIED", "not been given", "Unknown", "Invalid", "cannot be")
READ_HINT = ("get_", "list_", "search_", "find_", "check_", "read_", "fetch_", "retrieve_", "KB_")


def main():
    print("=" * 116)
    print("x412 · id 없는 완료 주장 27건 — 무엇이고, 2차 검산에서 참인가")
    print("동사구 -> 도구족 (축자 목록):")
    for v, t in VERB2TOOL:
        print("   %-22s -> %s" % (v, ", ".join(t)[:78]))
    print("=" * 116)

    rows = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            msgs = sim.get("messages") or []
            R = {m["id"]: " ".join(str(m.get("content") or "").split())
                 for m in msgs if m.get("role") == "tool" and m.get("id")}
            # 성공한 비-read 호출 (메시지 순번과 함께)
            succ = []
            for i, m in enumerate(msgs):
                for tc in (m.get("tool_calls") or []):
                    a = F.argsof(tc)
                    nm = str(F.inner_name(a) or F.nameof(tc))
                    b = R.get(tc.get("id"), "")
                    if not b or any(p in b for p in ENVERR):
                        continue
                    if any(nm.startswith(h) for h in READ_HINT):
                        continue
                    succ.append((i, nm))
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant" or not (m.get("content") or ""):
                    continue
                for s in sents(" ".join(str(m["content"]).split())):
                    if not C.DONE_RE.search(s) or ID_RE.search(s):
                        continue
                    low = s.lower()
                    fam = None
                    for v, pre in VERB2TOOL:
                        if v in low:
                            fam = (v, pre)
                            break
                    if fam is None:
                        code, ev = "UNMAPPED", ""
                    else:
                        hit = [(j, n) for j, n in succ
                               if any(n.startswith(p) for p in fam[1]) and j < i]
                        code = "SUPPORTED2" if hit else "UNSUPPORTED2"
                        ev = (hit[0][1] if hit else "그 족 성공호출 0")
                    rows.append({"task": F.task_id(sim), "trial": sim.get("trial"),
                                 "code": code, "verb": (fam[0] if fam else "-"),
                                 "ev": ev, "s": s[:120]})

    print("\n## 총 %d문장 (id 없는 완료 주장)" % len(rows))
    for k, v in collections.Counter(r["code"] for r in rows).most_common():
        print("   %-14s %2d  (%.0f%%)" % (k, v, 100.0 * v / max(len(rows), 1)))

    for code in ("UNSUPPORTED2", "SUPPORTED2", "UNMAPPED"):
        sub = [r for r in rows if r["code"] == code]
        if not sub:
            continue
        print("\n### %s (%d) — 축자" % (code, len(sub)))
        for r in sub:
            print("  %-9s t%-2s [%s] %s" % (r["task"], r["trial"], r["verb"][:18], r["s"]))
            if code != "UNMAPPED":
                print("            %s-> %s" % (" " * 12, r["ev"]))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x412_noid_claims.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


sys.exit(main())
