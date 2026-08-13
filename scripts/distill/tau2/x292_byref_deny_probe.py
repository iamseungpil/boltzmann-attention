# -*- coding: utf-8 -*-
r"""x292 — FIX-4 문면 격리: BYREF deny 가 틀린 출처 도구를 지목하는 오유도 제거 검증.

배경(t7274w 전수 정독): 072/073/074 전부 msg02 에서 get_atm_fee_discrepancies 를
`@last:get_credit_card_transactions_by_user`(틀린 출처)로 호출 → 현행 deny *"call that tool
first, then reference it"* 가 **그 틀린 도구 호출을 지시** → 074 는 실제로 그 도구를 호출하고
credit-card 경로로 표류(전패). [[64]]: 거부는 "무엇을 하면 풀리나"를 **옳게** 담아야 한다.

셀 2 (n=8·074 t0 msg02 직후 문맥·문면 리터럴 = 출시 예정 축자):
  A_CUR  현행 deny 4줄(라이브 축자) — 오유도 재현 대조
  B_NEW  신 deny 4줄(출시 예정 축자: 참조 가능한 출력 열거(지금은 없음)+틀린 도구 지시 제거)

계기: 다음 어시스턴트 턴의 tool_calls 에 get_credit_card_transactions_by_user 등장 = BAD.
판정(사전 고정): A_CUR BAD ≥6/8 ∧ B_NEW BAD ≤2/8 → FIX-4 출시. A_CUR ≤2/8 → 오유도가
다음-턴 단위 재현 안 됨 = 보류·재설계. 중간(3~5) → n=16 재측정 1회.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x292_byref_deny_probe.py [N]
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402

TAG = "bank_t7274w_a_20260813w"
TASK = "task_074"
WRONG = "get_credit_card_transactions_by_user"

CUR = ("Error: [BYREF] no committed non-error output of '%s' found in this conversation "
       "— call that tool first, then reference it" % WRONG)
NEW = ("Error: [BYREF] no committed non-error output of '%s' exists in this conversation "
       "(no record-read tool has returned records yet). Reference only the output of a "
       "record-read tool that has ALREADY returned records here. If the records you need "
       "have not been read yet, first call the tool that reads THOSE records (see this "
       "tool's parameter description for what it expects), then reference ITS output."
       % WRONG)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    # 컷 = 첫 BYREF 오표적 호출 어시스턴트 턴 직후(그 턴 포함).
    #   인자 저장 형태 2종(function.arguments / tc.arguments·dict 포함) 모두 검사 — 1차
    #   실행 "컷 없음"의 원인이 이 형태 차이였다.
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        blob = "".join(str((tc.get("function") or {}).get("arguments") or "")
                       + str(tc.get("arguments") or "") + str(tc)
                       for tc in (m.get("tool_calls") or []))
        if WRONG in blob:
            cut = i + 1
            break
    if cut is None:
        print("컷 없음")
        return
    ncalls = len(msgs[cut - 1].get("tool_calls") or [])
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    print("074 t%s cut=%d · deny %d줄 · n=%d · URL=%s\n" % (
        sim.get("trial"), cut, ncalls, n,
        os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, deny in (("A_CUR", CUR), ("B_NEW", NEW)):
        base = B.render(msgs[:cut], ours)
        # render 는 [user] ASK 를 덧붙인다 — x292 는 ASK 불필요·deny 만 잇는다
        base = base[:base.rfind("\n[user] ")]
        body = base + "".join("\n[tool] %s" % deny for _ in range(ncalls))
        bad = good_read = other = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            names = [str((tc.get("function") or {}).get("name") or "")
                     + str((tc.get("function") or {}).get("arguments") or "")
                     for tc in (r.get("tool_calls") or [])]
            blob = " ".join(names)
            if WRONG in blob:
                bad += 1
                cnt["bad:" + (names[0].split("{")[0] if names else "?")] += 1
            elif "get_bank_account_transactions" in blob or "bank_account" in blob:
                good_read += 1
            else:
                other += 1
                first = (names[0].split("{")[0] if names
                         else ("(text)" if r.get("content") else "(empty)"))
                cnt["other:" + first] += 1
        print("%-6s BAD(오표적) %d/%d · bank-read %d · 기타 %d · %s" % (
            label, bad, n, good_read, other, dict(cnt)))
    print("\n※ 판정(사전 고정): A_CUR ≥6/8 ∧ B_NEW ≤2/8 → FIX-4 출시. A_CUR ≤2/8 → 보류."
          " 중간(3~5) → n=16 1회.")


if __name__ == "__main__":
    main()
