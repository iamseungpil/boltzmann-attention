# -*- coding: utf-8 -*-
r"""x338 — **도구 지급 단계(give) 미실행** 격리. 019 가족(6 태스크)의 축을 가른다. `t2_probe` 정본 호출.

## 사건 (t7301·019·2026-08-16)

gold 는 2단이다: 에이전트가 `give_discoverable_user_tool` 로 손님에게 분쟁 도구를 **건네고** →
**손님이** `submit_cash_back_dispute_0589` 를 호출한다. 실측은 `give` **0회** · dispute **0/4**.

그런데 무지가 아니다 — 궤적 축자:
  · msg 36 (에이전트): *"We need to use the `submit_cash_back_dispute_0589` tool, which is a
    **user-discoverable tool**. **You will need to run this tool yourself**…"* ⇒ 이름도, 채널도 안다.
  · msg 38 (env): *"Tool … **has not been given to you by the agent. The agent must first use
    `give_discoverable_user_tool`** to give this tool to you."* ⇒ **정답 절차를 그대로 들었다.**
  · 그래도 안 부른다.

⇒ **P3(절차 무지)가 아니라 P4(방출) 계열** 가설. C489 동형(이름 18/24 ↔ 실행 2/24).
이 프로브가 그 가설을 격리로 검정한다.

## 셀 4 (컷 = 019 msg 39 = env 가 정답을 말한 **직후** · 도구 **전체** 바인딩)

    A_REF     라이브 축자 그대로              ← 이 자리에서 스스로 내는가(기준선)
    B_QUOTE   + env 문장 **축자 재인용**       ← 표면화만으로 닫히나([[66]] 인용-근거)
    C_DEMAND  + 열거 없는 행동 촉구 한 줄      ← 격리 상한(라이브는 null·C492 — 여기선 천장만 잰다)
    D_EARLY   **조사 전 컷**(msg 20)           ← ★부정통제: 아직 낼 자리가 아니다

⚠지표는 **말이 아니라 방출**이다 — `t2_probe._count` 가 `tool_calls` 까지 본다(2026-08-16 수리).
⚠도구는 **환경 스키마 전체**를 묶는다. 하나만 묶으면 그게 지목이다(x322: 24/24 → 0/24).
⚠[[62]] ③: 무엇을 부를지는 모델이 정한다. 엔진은 인용만 한다.

## 판정 (사전 고정 · 잡음 ±4 ⇒ 차 ≥5 만 인용)

    A_REF GIVE ≥18                → 격리에선 낸다 ⇒ 라이브 결손은 **부하/우리 문구**(P1·P3 축)
    A_REF GIVE ≤6 ∧ B ≥18         → **표면화로 닫힌다**(가족 6개의 값싼 레버)
    A≈B ∧ C ≥18                   → 촉구가 필요 = **P4(끝맺음)** 확정 ⇒ 학습 축([[13]])
    A≈B≈C 전부 낮음               → 그 자리에서 **아무도 못 연다** ⇒ 다른 축 재탐색
    D_EARLY ≥6                    → 부정통제 실패(아무 데서나 낸다) ⇒ 이 프로브 무효

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x338_give_step_iso.py [k] [nb]
"""
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
import t2_gap as G                                                # noqa: E402

TAG, TASK = "bank_t7301_probe2_20260816d", "task_019"
CUT, EARLY = 39, 20
MARKS = {"GIVE": "give_discoverable_user_tool", "DISPUTE": "submit_cash_back_dispute"}
DEMAND = "\n[instruction] Carry out the next step of this request now."
ASK = "\n[instruction] Do the next step."


def env_quote(sim, cut):
    """env 가 정답 절차를 말한 문장을 **궤적에서 축자로** 되찾는다(내가 쓰지 않는다)."""
    for m in reversed((sim.get("messages") or [])[:cut]):
        c = " ".join(str(m.get("content") or "").split())
        if m.get("role") == "tool" and "give_discoverable_user_tool" in c:
            return c[:400]
    return ""


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.sims(TAG) if F.task_id(s) == TASK)
    site = P.site(TAG, TASK, CUT)
    early = P.site(TAG, TASK, EARLY)
    quote = env_quote(sim, CUT)
    if not quote:
        print("env 문장을 궤적에서 못 찾음 — 중단(계기 결함)")
        return 1
    tools = G.schemas()
    if not tools:
        print("도구 스키마 없음 — 중단")
        return 1
    print("x338 · %s/%s · cut=%d(부정통제 %d) · 도구 %d개 바인딩\nenv 축자: %s\n"
          % (TAG, TASK, CUT, EARLY, len(tools), quote[:200]))

    P.run("x338", site,
          [("A_REF", ""), ("B_QUOTE", "[note] " + quote), ("C_DEMAND", "[note] " + quote + DEMAND)],
          MARKS,
          "A≥18 → 격리선 냄(부하/문구) · A≤6∧B≥18 → **표면화로 닫힘** · A≈B∧C≥18 → **P4 확정**(학습 축) · "
          "전부 낮음 → 다른 축 · D_EARLY≥6 → 통제 실패",
          ASK, None, k, nb, tools=tools)
    print("\n── 부정통제(조사 전 컷) ──")
    P.run("x338-neg", early, [("A_REF", ""), ("D_EARLY", "")], MARKS,
          "이 자리에서 GIVE 를 내면 아무 데서나 내는 것 = 프로브 무효", ASK, None, k, nb, tools=tools)


if __name__ == "__main__":
    sys.exit(main() or 0)
