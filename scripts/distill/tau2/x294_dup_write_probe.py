# -*- coding: utf-8 -*-
r"""x294 — 중복-write 격리: 동일 인자 credit 재실행이 재현되는가·deny 문면이 멈추는가.

배경(실측 2런): t7274w 073 chk_3 동일 credit 2회 · t7276 072 msg93→msg99 **두 계좌 모두
동일 인자 재실행**(DB 발산). FIX-5 의 "do not credit the same lines twice" 산문은 무력
([[63]]: 지시는 안 듣는다) — 닫힌 술어(동일 도구+동일 인자 write 가 이미 성공) deny 후보.
선례 = 출시된 DUPLICATE-READ deny(동형·읽기판).

셀 2 (n=8·t7276 072 중복-재실행 어시스턴트 턴 직전 컷):
  A_ASIS  문맥 그대로 → 중복 재실행 재현율
  B_DENY  + [assistant 중복 호출] + [tool] DUPLICATE-WRITE deny(출시 예정 축자) →
          그 다음 턴이 같은 write 를 **또** 내는가(BAD) / 영수증으로 답하는가(GOOD)

판정(사전 고정): A_ASIS ≥6/8(재현) ∧ B_DENY BAD ≤2/8 → T2_WRITE_DEDUP 출시.
  A_ASIS ≤2/8 → 재현 실패 = 보류(라이브 확률 사건). 중간(3~5) → n=16 재측정 1회.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x294_dup_write_probe.py [N]
"""
import collections
import json
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

TAG = "bank_t7276_a_20260813x"
TASK = "task_072"

DENY = ("Error: [DUPLICATE-WRITE] this exact write (same tool, same arguments) already "
        "succeeded earlier in this conversation - its receipt is shown above and the account "
        "has already been credited. Do NOT execute it again: doing so would apply the same "
        "change twice. If the customer is asking whether it was done, answer from the earlier "
        "receipt instead of re-executing.")


def credit_sig(tc):
    a = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or ""
    s = a if isinstance(a, str) else json.dumps(a, sort_keys=True)
    return s if "apply_checking_account_credit" in (str(tc) or "") else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    # 중복 턴 = 앞서 성공한 credit 호출과 동일 시그니처를 다시 내는 첫 어시스턴트 턴
    seen, dup_turn, dup_calls = set(), None, []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        sigs = [s for s in (credit_sig(tc) for tc in (m.get("tool_calls") or [])) if s]
        if any(s in seen for s in sigs):
            dup_turn = i
            dup_calls = [tc for tc in (m.get("tool_calls") or []) if credit_sig(tc)]
            break
        seen.update(sigs)
    if dup_turn is None:
        print("중복 턴 없음")
        return
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    base = B.render(msgs[:dup_turn], ours)
    base = base[:base.rfind("\n[user] ")]
    print("072 t%s dup_turn=%d · dup_calls=%d · n=%d · URL=%s\n" % (
        sim.get("trial"), dup_turn, len(dup_calls), n,
        os.environ.get("T2_PROBE_URL", "localhost:8140")))
    # A_ASIS: 중복 재실행 재현율
    for label in ("A_ASIS", "B_DENY"):
        body = base
        if label == "B_DENY":
            calls_line = "[assistant calls] " + ", ".join(
                "call_discoverable_agent_tool(%s)" % str(
                    tc.get("arguments") or "")[:160] for tc in dup_calls)
            body = base + "\n" + calls_line + "".join(
                "\n[tool] %s" % DENY for _ in dup_calls)
        bad = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            sigs = [s for s in (credit_sig(tc) for tc in (r.get("tool_calls") or [])) if s]
            isdup = any(s in seen for s in sigs)
            if isdup:
                bad += 1
                cnt["re-credit"] += 1
            else:
                nm = ""
                for tc in (r.get("tool_calls") or []):
                    nm = str(tc.get("name") or (tc.get("function") or {}).get("name") or "")
                    break
                cnt[nm or ("(text)" if r.get("content") else "(empty)")] += 1
        print("%-7s 중복재실행 %d/%d · %s" % (label, bad, n, dict(cnt)))
    print("\n※ 판정(사전 고정): A_ASIS ≥6/8 ∧ B_DENY ≤2/8 → T2_WRITE_DEDUP 출시."
          " A_ASIS ≤2/8 → 보류. 중간(3~5) → n=16 1회.")


if __name__ == "__main__":
    main()
