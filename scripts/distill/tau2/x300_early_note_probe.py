# -*- coding: utf-8 -*-
r"""x300 — 첫 접힘 시점 소유권 note 격리: 발화 **시점**이 레버인가 (075 pass 표적).

배경(x298 ↔ x299 대조·같은 문면): 접힘 텍스트 **1회** 시점 컷 = 6/8 · **2회 누적** 후 컷 =
0/8. 문면은 같다 — 갈린 것은 시점뿐([[64]] C413 동형: 이름 없는 접힘 문구가 3회↑ 나온 sim 은
6/6 실패). 라이브(t7278 075)에서 claimprov 는 turn13 저가치 주장에 **cap 1/sim 을 전소**하고
(코드 주석 t2_gate_patch:9652 가 이 양식을 이미 문서화) 결정적 접힘 turn29~31 엔 아무 note 도
가지 않았다.

셀 3 (n=8·컷 = t7278 075 **첫 접힘 산문 직후**(msg29+1)·도구 호출 없는 산문 턴):
  A_NONE   note 없음                        — 라이브 재현(그 다음 턴도 접힘 예상)
  B_NOTE   소유권 note(FIX-7 문면 축자·reminder 형태로 대화에 삽입)
  D_GEN    일반 촉구 note(도구 이름·소유권 사실 **없이** "약속한 일을 지금 하라")
           — 부정통제: 소유권 사실이 인자인지 vs 촉구 자체가 인자인지 귀속([[57]])

계기: 그 다음 턴이 open_bank_account_4821 을 unlock/call 하는가.
판정(사전 고정): B_NOTE ≥6/8 ∧ A_NONE ≤2/8 ∧ D_GEN ≤3/8 → **시점 레버 출시**(소유권 note 를
  첫 접힘 턴에 발화 — claimprov 전용 예산/조기 창). D_GEN ≥6/8 → 촉구만으로 충분(문면 단순화).
  B_NOTE ≤2/8 → 시점도 못 연다 = L1 학습행 확정.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x300_early_note_probe.py [N]
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
from x297_registry_hygiene_probe import TARGET                    # noqa: E402

TAG = "bank_t7278_b_20260813z"
TASK = "task_075"

# FIX-7 출시본 문면을 note 형태로(축자·{name}=모델이 안내한 외부 절차 대신 도구 사실만)
B_NOTE = ("Note: [CLAIM-PROVENANCE] tool ownership — opening a bank account is done by a tool "
          "in YOUR OWN tool list, not the customer's: %s. The customer cannot run it, and "
          "describing the steps to them does not execute it - unlock it and call it yourself."
          % TARGET)
D_GEN = ("Note: [CLAIM-PROVENANCE] earlier in this conversation you told the customer you "
         "would take care of this, but the conversation ledger shows it was never actually "
         "executed. Do the promised work now, and do not end your involvement by describing "
         "the work as done or under way.")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and "Visit the Rho-Bank" in str(m.get("content") or ""):
            cut = i + 1
            break
    if cut is None:
        print("첫 접힘 산문 없음")
        return
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    base = B.render(msgs[:cut], ours)
    base = base[:base.rfind("\n[user] ")] if "\n[user] " in base else base
    print("075 cut=%d(첫 접힘 직후) · n=%d · URL=%s\n" % (
        cut, n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, note in (("A_NONE", None), ("B_NOTE", B_NOTE), ("D_GEN", D_GEN)):
        body = base + (("\n[system] " + note) if note else "")
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                # ⚠mx=400 재검 (2026-08-14): 이 모델은 content 를 먼저 뱉고 tool_call 을
                #   잇는다. 400 이면 산문이 긴 팔에서 **호출 직전 절단**(finish=length)돼
                #   '(text)' 로 기록된다 — x295b 에서 0/8→7/8 로 뒤집힌 그 결함이다.
                #   음성 팔일수록 산문이 길어 **처치와 상관된 인공물**이 되므로 전 팔 재측정.
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            blob = " ".join(str(t) for t in (r.get("tool_calls") or []))
            if TARGET in blob:
                hit += 1
                cnt["target"] += 1
            else:
                first = ""
                for t in (r.get("tool_calls") or []):
                    first = str(t.get("name") or (t.get("function") or {}).get("name") or "")
                    break
                cnt[first or ("(text)" if r.get("content") else "(empty)")] += 1
        print("%-7s target %d/%d · %s" % (label, hit, n, dict(cnt)))
    print("\n※ 판정(사전 고정): B_NOTE ≥6 ∧ A_NONE ≤2 ∧ D_GEN ≤3 → 시점 레버 출시."
          " D_GEN ≥6 → 촉구만으로 충분. B_NOTE ≤2 → L1 학습행 확정.")


if __name__ == "__main__":
    main()
