# -*- coding: utf-8 -*-
r"""x330 — 끝맺음: **이름 대기**가 아니라 **실제 tool_call 방출**로 채점한다.

## 왜 (x326 의 직접 후속)

x326(같은 컷·n=24)은 *"다음에 부를 도구 **이름**을 한 줄로"* 를 물었고 **21/24** 가 맞혔다
(부정통제 `D_EARLY` 0/24 로 유효). 그런데 **라이브에서는 그 호출이 나가지 않았다**
(073 은 unlock 도 없이 종료 · 072 는 unlock 만 하고 끝).

⇒ 결손은 *무엇을 할지 모름*이 아니라 **아는 것을 실행으로 옮기지 못함**일 수 있다([[46]]
knowing–doing 축). x326 은 그것을 가를 수 없다 — **말하기**만 쟀기 때문이다.
여기서는 도구를 **바인딩**하고(`tool_choice=auto`) **실제 `tool_calls` 방출**로 채점한다.

## 셀 4 (컷 = 073#0 msg 50 · n=24=8×3)

    A_NAME   x326 의 `B_TRAJ` 재현 — "이름 한 줄"      ← 말하기 기준선(≈21/24 나와야 계기 정상)
    B_EMIT   같은 문맥 · **도구 바인딩** · 지시 없음    ← ★실행 기준. 무엇을 하라고 말하지 않는다
    C_EMIT_ASK B_EMIT + "다음 한 수를 **실행**하라"     ← 지시하면 넘어오나(전달 아닌 촉구)
    D_EARLY  조사 완료 **전** 컷 + 바인딩              ← ★부정통제(여기서 방출하면 무효)

⚠[[62]] ③④: 어느 팔도 **도구를 지목하지 않는다**. 후보 목록은 바인딩된 스키마 전체이고
  그것은 라이브와 같은 조건이다. 엔진은 순위·최댓값·정답을 내지 않는다.
⚠스키마는 **환경에서 기계 추출**한다(`Tool.openai_schema`) — 저작 0([[59]]).
⚠탈출구 문구 금지(2026-08-15 §7-6).

## 판정 (사전 고정 · 잡음 바닥 ±4·C483)

    D_EARLY 방출 ≥ 8              → **프로브 무효**
    A_NAME ≥18 ∧ B_EMIT ≤ 6       → ★**knowing–doing 확정**: 알지만 실행으로 안 옮긴다
    A_NAME ≈ B_EMIT ≥18           → 실행도 된다 ⇒ 라이브 실패는 **이 컷 밖**(대화 부하·종결 압력)
    C_EMIT_ASK − B_EMIT ≥ 5       → **촉구가 산다**(전달이 아니라 요구가 레버)
    C ≈ B                         → 촉구도 무효 ⇒ 남는 것은 궤적 부하

실행(리모트·**비어 있는 GPU 포트**로): T2_PROBE_URL=http://localhost:8140/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x330_terminal_write_emit.py [블록크기] [블록수]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from loguru import logger                                          # noqa: E402
logger.remove()

from x216_read_and_offset import chat                              # noqa: E402
import t2_forensic as F                                            # noqa: E402
import x313_bailout_iso as B                                       # noqa: E402
from tau2.registry import registry                                 # noqa: E402

TAG, TASK = "bank_t7295_b_20260815n", "task_073"
CUT, EARLY_CUT = 50, 30
RIGHT = "apply_checking_account_credit"
NAME_ASK = ("\n[instruction] Do NOT call any tool yet. Reply with ONE line only: the name of the "
            "tool you would call next, nothing else.")
DO_ASK = "\n[instruction] Carry out the next step of this request now."


def schemas():
    """환경이 든 도구의 OpenAI 스키마 — 기계 추출(저작 0)."""
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="no_knowledge")
    out = []
    for t in env.get_tools():
        try:
            out.append(t.openai_schema)
        except Exception:
            continue
    return out


def emitted(msg):
    """방출된 tool_calls 안에 그 도구가 있나(래퍼 안쪽 인자까지 문자열로 훑는다)."""
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function") or {}
        if RIGHT in (str(f.get("name", "")) + str(f.get("arguments", ""))):
            return True
    return False


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK and s.get("trial") == 0)
    traj = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    early = "\n".join([B.HEAD, "", B.transcript(sim, EARLY_CUT)])
    tools = schemas()
    print("x330 · %s/%s · cut=%d(early=%d) · 본문 %d자 · 도구 %d개 · %d×%d블록\n"
          % (TAG, TASK, CUT, EARLY_CUT, len(traj), len(tools), k, nb))
    if not tools:
        print("도구 스키마 추출 실패 — 중단(계기 결함)"); return 1

    arms = (("A_NAME", traj + NAME_ASK, None, False),
            ("B_EMIT", traj, tools, True),
            ("C_EMIT_ASK", traj + DO_ASK, tools, True),
            ("D_EARLY", early, tools, True))
    res = {}
    for label, body, tl, by_call in arms:
        blocks = []
        for b in range(nb):
            h = 0
            for i in range(k):
                try:
                    m = chat(body, tl, 0.0 if i == 0 else 0.7, 300)
                except Exception as e:
                    m = {"content": "ERR %s" % type(e).__name__}
                ok = emitted(m) if by_call else (RIGHT in str(m.get("content") or ""))
                h += ok
                shown = (",".join((tc.get("function") or {}).get("name", "")
                                  for tc in (m.get("tool_calls") or []))
                         or " ".join(str(m.get("content") or "").split())[:50])
                print("    [%s b%d %02d] %s %s" % (label, b + 1, i, "HIT" if ok else "-",
                                                   shown[:60]), flush=True)
            blocks.append(h)
        res[label] = (sum(blocks), blocks)
        print("%-11s %d/%d · 블록 %s\n" % (label, sum(blocks), k * nb, blocks))
    print("판정(사전 고정): D≥8 → 무효 · A≥18∧B≤6 → **knowing–doing 확정** · "
          "A≈B≥18 → 실행도 됨(라이브 실패는 이 컷 밖) · C−B≥5 → 촉구가 산다 · C≈B → 촉구 무효")
    print("측정치: " + " · ".join("%s=%d%s" % (a, v[0], v[1]) for a, v in res.items()))


if __name__ == "__main__":
    sys.exit(main() or 0)
