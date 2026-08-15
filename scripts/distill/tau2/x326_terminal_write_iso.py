# -*- coding: utf-8 -*-
r"""x326 — **끝맺음(terminal write)이 왜 떨어지나**: 부하인가 능력인가.

## 사건 (t7295·75 sim per-step 재분류)

미호출 gold 가 있는 실패 sim 34 중 **28(82%)** 이 **마지막 gold 액션**을 빠뜨렸다
(균일 가정 기대값 0.56 ⇒ +26pp 편중). 채택 결손(이름이 눈앞에 노출됐는데 안 쓴 80건)을
직독하면 안 쓴 도구가 **언제나 마지막 write** 다:

    073  신원확인·계좌조회·거래조회·`get_atm_fee_discrepancies` 다 함 → `apply_checking_account_credit_5829` **안 함**
    050  자격·이력·결제이력·`submit_credit_limit_increase_request_7392` 다 함 → `approve_...` **안 함**
    072  같은 도구를 **unlock 까지** 하고 → 안 부름. 손님은 *"thanks! ###STOP###"* 로 만족.

⇒ *"못 찾아서 헤맨다"* 가 아니라 **조사를 끝내고 실행만 빠진다**. 그 결손이 **부하**(궤적이
길어 결정을 못 함)인지 **능력**(그 자리에서 무엇을 할지 모름)인지 여기서 가른다([[62]] ①).

## 셀 4 (컷 = 073#0 의 `get_atm_fee_discrepancies` 반환 **직후** = msg 50)

    A_MIN     손님 발화 + 마지막 도구 출력만(어시스턴트 잔여물 제거)  ← 정보-맞춘 최소
    B_TRAJ    라이브 축자 전문                                       ← 실제 궤적 부하
    C_STATE   B + **중립 상태 표면화** 한 줄(도구 이름 0·지목 0)      ← 표면화로 닫히나
    D_EARLY   조사 **완료 전** 컷(msg 30)                            ← ★부정통제

⚠**D_EARLY 가 이 프로브의 생명이다.** 그 자리의 옳은 다음 수는 조회이지 크레딧이 아니다.
거기서도 크레딧을 대면 모델은 "태스크를 보고 늘 그 도구를 대는" 것이고 A·B·C 수치는 **무의미**하다
(2026-08-15 §7 의 오류와 같은 부류 — 한 축만 보고 결론 직행).
⚠[[62]] ③④: 어떤 팔도 도구를 **지목하지 않는다**. 후보 목록도 주지 않는다 — 모델이 이미
본 것(궤적)만 쓴다. 엔진이 순위·최댓값·정답을 내지 않는다.
⚠탈출구 금지(2026-08-15 §7-6 교훈): *"or ASK"* 류 한 단어 도피를 넣지 않는다.

## 판정 (사전 고정 · 블록별 병기 · n=24=8×3 · 잡음 바닥 ±4·C483)

    D_EARLY ≥ 8                 → **프로브 무효**(항상 그 도구를 댄다). 아래를 읽지 말 것.
    A_MIN ≥18 ∧ B_TRAJ ≤ 6      → **부하**. 레버 = 문맥 축소/표면화(결정론 실행 아님)
    A_MIN ≈ B_TRAJ ≥18          → 모델은 **안다** ⇒ 결손은 *이름 대기* 가 아니라 **실행 전이**
                                   ⇒ 이 컷은 틀렸다. 컷을 "행동 직전"이 아니라 **행동 시점**으로 옮겨 재측정
    C_STATE ≥18 ∧ B ≤14         → 상태 한 줄로 닫힌다(최소 개입)
    A_MIN ≤ 6 ∧ B ≤ 6           → **능력/경계**. 전달로는 안 산다

실행(리모트·8141·[[30]] 포트 분리):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x326_terminal_write_iso.py [블록크기] [블록수]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

TAG, TASK = "bank_t7295_b_20260815n", "task_073"
CUT, EARLY_CUT = 50, 30
RIGHT = "apply_checking_account_credit"
# 중립 상태 표면화 — **무엇을 하라고 말하지 않는다**. 시스템에 아직 기록이 없다는 사실뿐이다.
STATE = ("[note] No change has been recorded in the system for this customer's request so far in "
         "this conversation.")
ASK = ("\n[instruction] Do NOT call any tool yet. Reply with ONE line only: the name of the tool "
       "you would call next, nothing else.")


def minimal(sim, cut):
    """정보-맞춘 최소: 손님 발화 전부 + **마지막** 도구 출력. 어시스턴트 산문은 제거.

    ⚠저작 0 — 궤적에서 **골라내기만** 한다. 문장을 짓지 않는다([[59]])."""
    users, tools = [], []
    for m in (sim.get("messages") or [])[:cut]:
        c = " ".join(str(m.get("content") or "").split())
        if not c:
            continue
        if m.get("role") == "user":
            users.append("[user] " + c[:900])
        elif m.get("role") == "tool" and not m.get("error"):
            tools.append("[tool] " + c[:1200])
    return "\n".join(users + tools[-1:])


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.scored(TAG)
               if F.task_id(s) == TASK and s.get("trial") == 0)
    traj = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    early = "\n".join([B.HEAD, "", B.transcript(sim, EARLY_CUT)])
    mini = "\n".join([B.HEAD, "", minimal(sim, CUT)])

    arms = (("A_MIN", mini + ASK),
            ("B_TRAJ", traj + ASK),
            ("C_STATE", traj + "\n\n" + STATE + ASK),
            ("D_EARLY", early + ASK))
    print("x326 · %s/%s · cut=%d(early=%d) · 최소 %d자 / 궤적 %d자 · %d×%d블록\n"
          % (TAG, TASK, CUT, EARLY_CUT, len(mini), len(traj), k, nb))
    res = {}
    for label, body in arms:
        blocks = []
        for b in range(nb):
            h = 0
            for i in range(k):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, 60)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                out = " ".join(str(r.get("content") or "").split())
                ok = RIGHT in out
                h += ok
                print("    [%s b%d %02d] %s %s" % (label, b + 1, i, "HIT" if ok else "-",
                                                   out[:60]), flush=True)
            blocks.append(h)
        res[label] = (sum(blocks), blocks)
        print("%-10s %d/%d · 블록 %s\n" % (label, sum(blocks), k * nb, blocks))
    print("판정(사전 고정): D_EARLY≥8 → **프로브 무효** · A≥18∧B≤6 → 부하 · "
          "A≈B≥18 → 모델은 안다(컷을 행동 시점으로 옮겨 재측정) · C≥18∧B≤14 → 상태 한 줄로 닫힘 · "
          "A≤6∧B≤6 → 능력/경계")
    print("측정치: " + " · ".join("%s=%d%s" % (a, v[0], v[1]) for a, v in res.items()))


if __name__ == "__main__":
    main()
