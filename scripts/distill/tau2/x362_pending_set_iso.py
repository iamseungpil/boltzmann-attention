# -*- coding: utf-8 -*-
r"""x362 — **019 가족: 대기 집합에 discoverable 도구를 넣으면 `give` 를 지목하는가**.

## 왜 (오늘 코드 포렌식·설계서 §2c 동렬)

C500: 019 가족은 격리 24/24 ↔ 라이브 **0**. `t2_gap` 사다리 5칸이 그 0 을 **재현하지 못했다**
⇒ 결손은 모델이 아니라 **우리 층**이다([[55]] 0단계). 코드에서 자리를 찾았다:

    _uacts = {t for t in a2["action_tools"] if _exec_side(t) == "user"}   # ★정적 7개
    ...
    if _utgt in _upending:        # ← 손님-실행 안내는 여기서만 나간다

019 의 gold 도구 `submit_cash_back_dispute_0589` 는 **런타임에 건네지는 discoverable 도구**라
그 정적 목록에 **영원히 없다**. 라이브 로그 축자가 그대로다:

    pending_agent=[… 'give_discoverable_user_tool' …] formalized_target=call_discoverable_agent_tool

⇒ 형식화기가 **에이전트 직접 호출**을 지목하고, 손님-실행 분기는 침묵하며, `give` 는 아무도
   요구하지 않는다.

## 무엇을 재는가

라이브와 **같은 프롬프트**(`t2_resolve.formalize_intent_tool` 축자)로, 후보 집합만 바꿔 지목을 본다.

    A_REF     현행 집합(A2 `action_tools` 7개)                    ← 라이브 재현(기대: CALL)
    B_DISC    + **env 가 이 대화에서 표면화한 discoverable 도구**  ← 지목이 옮겨가는가
    D_EARLY   조사 **전** 컷(msg 20)에서 B 구성                    ← (1차 통제·아래 ⚠)
    D_OTHER   **다른 태스크의 손님 발화**(024 카드 요청) + B 집합     ← ★진짜 부정통제

⚠1차 실행에서 `D_EARLY` 가 B 와 같은 답(DISPUTE)을 냈다. 진단: 손님은 **이른 컷 이전에 이미**
  분쟁을 말한다 ⇒ 그 컷은 *뒤집힐 수 없는 통제*였다(x335·x336 부류). 의도 지목은 **준비 상태와
  무관**하므로 통제는 *의도 자체가 없는 발화*로 세워야 한다 — 그것이 `D_OTHER` 다.

⚠[[66]]: 무엇을 지목할지는 **LLM** 이 정한다. 이 프로브도 구현안도 엔진이 규칙을 넣지 않는다 —
  바뀌는 것은 **후보 집합의 출처**(정적 선언 → env 가 표면화한 닫힌 집합)뿐이다.

## 판정 (사전 고정)

    B 에서 DISPUTE 또는 GIVE 지목 ≥6/8  ∧  D_EARLY ≤2  → 수리한다(대기 집합에 discovered 추가)
    B ≈ A (둘 다 CALL)                                   → 후보 집합이 원인이 아니다 ⇒ 다른 자리
    D_EARLY ≥3                                           → 계기 무효

실행: T2_PROBE_URL=http://localhost:8141/... python x362_pending_set_iso.py [k] [nb]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                           # noqa: E402
import t2_probe as P                                              # noqa: E402

TAG, TASK = "bank_t7301_probe2_20260816d", "task_019"
CUT, EARLY = 39, 20
DISC = "submit_cash_back_dispute_0589"          # env 가 대화에서 표면화한 도구(궤자 축자)
A2P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2",
                   "banking_knowledge.settings.json")
MARKS = {"DISPUTE": DISC, "GIVE": "give_discoverable_user_tool",
         "CALL": "call_discoverable_agent_tool", "NONE": "none"}

# ★라이브 축자(`t2_resolve.formalize_intent_tool`) — 케이스 규칙 0([[66]]·judge6 사고 기록)
PROMPT = ("The user is talking to a customer-service agent. Based ONLY on what the user asked, "
          "which ONE of these tools must the agent CALL to fulfill the request? "
          "Reply with the exact tool name, or 'none' if none applies.\n"
          "Tools: {tools}\n"
          "User said:\n- {users}\n"
          'Reply JSON only: {{"tool": "<name or none>"}}')


def users_upto(sim, cut, last=6):
    out = []
    for m in (sim.get("messages") or [])[:cut]:
        if m.get("role") == "user":
            c = " ".join(str(m.get("content") or "").split())
            if c:
                out.append(c[:300])
    return out[-last:]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    a2 = json.load(io.open(A2P, encoding="utf-8"))
    acts = sorted(a2.get("action_tools") or [])
    if not acts:
        print("A2 action_tools 없음 — 중단(계기 결함)")
        return 1
    sims = [s for s in F.sims(TAG) if F.task_id(s) == TASK]
    if not sims:
        print("019 궤적 없음 — 중단")
        return 1
    sim = sims[0]
    u_cut = users_upto(sim, CUT)
    u_early = users_upto(sim, EARLY)
    if not u_cut or not u_early or not u_other:
        print("손님 발화를 못 찾음 — 중단(계기 결함)")
        return 1
    # ★진짜 부정통제용: 다른 태스크(024·카드 신청)의 손님 발화 — 분쟁 의도가 **없다**
    o_sims = [s2 for s2 in F.sims("bank_t7305_treataux_20260817a") if F.task_id(s2) == "task_024"]
    u_other = users_upto(o_sims[0], 12) if o_sims else []
    body = lambda tools, users: PROMPT.format(tools=", ".join(tools),      # noqa: E731
                                              users="\n- ".join(users))
    print("x362 · %s/%s · cut=%d · 현행 후보 %d개 · discovered=%r"
          % (TAG, TASK, CUT, len(acts), DISC))
    print("판정(사전 고정): B 에서 DISPUTE|GIVE ≥6/8 ∧ D_EARLY ≤2 → 대기집합 수리 · B≈A → 후보 "
          "집합이 원인 아님 · D_EARLY ≥3 → 계기 무효\n")
    site = {"tag": TAG, "task": TASK, "cut": CUT, "sim": None, "base": ""}
    P.run("x362", site,
          [("A_REF", body(acts, u_cut)),
           ("B_DISC", body(sorted(set(acts) | {DISC}), u_cut)),
           ("D_EARLY", body(sorted(set(acts) | {DISC}), u_early)),
           ("D_OTHER", body(sorted(set(acts) | {DISC}), u_other))],
          MARKS,
          "B 에서 DISPUTE|GIVE ≥6/8 ∧ D_EARLY ≤2 → 수리 · B≈A → 다른 자리 · D_EARLY ≥3 → 무효",
          "", None, k, nb, det=True, names=sorted(set(acts) | {DISC, "none"}))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
