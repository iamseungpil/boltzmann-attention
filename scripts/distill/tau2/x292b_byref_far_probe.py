# -*- coding: utf-8 -*-
r"""x292b — FIX-4 원거리 창 격리: BYREF deny 의 오유도가 **첫 read 시점**에 실현되는가.

배경(t7276 074 전수 정독): msg02 오표적 BYREF deny(4회) 후 검증 절차로 회복 — 그러나
검증 완료 직후 **msg27 첫 read 가 정확히 deny 가 지목한 `get_credit_card_transactions_by_user`**
(그 도구명이 궤적에 등장한 유일 출처 = deny 문면). x292(next-turn 창)는 A_CUR 7/24 로 기각
됐지만 실제 실현 창은 first-read — [[18]] 정보-맞춤 재격리.

셀 2 (n=8·t7276 074 msg26(log_verification 성공) 직후 컷·문면 리터럴 = x292 와 동일 축자):
  A_CUR  msg03~06 deny = 현행 문면 그대로
  B_NEW  msg03~06 deny 를 신 문면으로 치환(참조 가능 출력 열거+오지시 제거)

계기: 다음 어시스턴트 턴 tool_calls 에 get_credit_card_transactions_by_user = BAD.
판정(사전 고정): A_CUR ≥6/8 ∧ B_NEW ≤2/8 → FIX-4 출시(관측 창 = first-read 로 정정).
  A_CUR ≤2/8 → msg27 은 확률적 사건 = FIX-4 기각 유지. 중간(3~5) → n=16 재측정 1회.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x292b_byref_far_probe.py [N]
"""
import collections
import copy
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
from x292_byref_deny_probe import CUR, NEW, WRONG                 # noqa: E402

TAG = "bank_t7276_a_20260813x"
TASK = "task_074"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    # 컷 = log_verification 성공 tool 메시지 직후
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and "Verification logged successfully" in str(
                m.get("content") or ""):
            cut = i + 1
            break
    if cut is None:
        print("컷 없음(log_verification)")
        return
    deny_idx = [i for i, m in enumerate(msgs[:cut])
                if m.get("role") == "tool" and "[BYREF]" in str(m.get("content") or "")]
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    print("074 t%s cut=%d · deny %d개 치환 · n=%d · URL=%s\n" % (
        sim.get("trial"), cut, len(deny_idx),
        n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, deny in (("A_CUR", CUR), ("B_NEW", NEW)):
        arm = copy.deepcopy(msgs[:cut])
        for i in deny_idx:
            arm[i]["content"] = deny
        body = B.render(arm, ours)
        body = body[:body.rfind("\n[user] ")]      # render 의 ASK 제거 — 이어질 턴만 측정
        bad = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            names = [str((tc.get("function") or {}).get("name") or tc.get("name") or "")
                     + str((tc.get("function") or {}).get("arguments")
                           or tc.get("arguments") or "")
                     for tc in (r.get("tool_calls") or [])]
            blob = " ".join(names)
            if WRONG in blob:
                bad += 1
                cnt["bad"] += 1
            else:
                first = (names[0].split("{")[0] if names
                         else ("(text)" if r.get("content") else "(empty)"))
                cnt[first] += 1
        print("%-6s BAD(오표적 read) %d/%d · %s" % (label, bad, n, dict(cnt)))
    print("\n※ 판정(사전 고정): A_CUR ≥6/8 ∧ B_NEW ≤2/8 → FIX-4 출시(first-read 창)."
          " A_CUR ≤2/8 → 기각 유지. 중간(3~5) → n=16 1회.")


if __name__ == "__main__":
    main()
