# -*- coding: utf-8 -*-
r"""x332 — **모델이 스스로 세고 체크하게** 하면 실행으로 넘어오나 (사용자 제안 2026-08-15).

## 제안 축자
*"몇 개를 완수해야 되는지 **말하게 하고**, 완수한 것을 체크리스트로 체크하고 계속 남은 갯수를
알려주는 건 어떤가?"*

## 왜 이 형태가 규율에 맞나
- **엔진이 세지 않는다.** 무엇이 요구사항인지 판정하는 것은 의도 해석이고 그것은 LLM 몫이다([[66]]).
  여기서 엔진은 *"세어서 표시하고 남은 것을 처리하라"* 는 **요구만** 한다.
- **도구를 지목하지 않는다.** x322 실측: 우리 지목이 24/24 → **0/24** 로 파괴했다.
- ⚠**선행 반증 있음**: x327 에서 *계수 표면화*(엔진이 횟수를 세어 줌)는 **완전히 무효**였다
  (C 0/24 ↔ D_NEG4 0/24). 그것은 **엔진이 센 것**이고 이번은 **모델이 세는 것**이라 다르지만,
  *"세면 될 것"* 이라는 기대는 이미 한 번 틀렸다 — 그래서 잰다.

## 셀 4 (컷 = 073#0 msg 50 · 도구 바인딩 · n=24=8×3)

    A_BASE      지시 없음                                   ← 기준선(≈0~2)
    B_SELFLIST  **모델이 세고·체크하고·남은 것을 처리**        ← 제안
    C_ASK       단순 촉구(x330 `C_EMIT_ASK` 재현)            ← 대조(≈11~13)
    D_EARLY     조사 완료 **전** 컷 + B 와 같은 지시           ← ★부정통제

## 판정 (사전 고정 · 잡음 바닥 ±4·C483)

    D_EARLY ≥ 8      → **프로브 무효**(지시가 아무 데서나 write 를 유발한다)
    B − A ≥ 5        → 자기-체크리스트가 산다
    B − C ≥ 5        → **단순 촉구보다 낫다**(제안 채택 근거)
    B ≈ C            → 추가 이득 없음 ⇒ 더 싼 촉구를 쓴다([[62]] 최소)
    B < C − 5        → 오히려 해롭다(세느라 실행을 미룬다)

★**온도 분리 병기 의무**(2026-08-15 발견): temp 0.0 은 블록당 1회다. 오늘 실측에서
**요구가 있는 팔만 temp0 에서 뒤집혔다**(요구 6/6 ↔ 무요구 0/12) — 즉 **결정론적 모드**가
바뀌는지와 **꼬리만 넓어지는지**는 다른 사건이다. 이 프로브도 둘을 갈라 보고한다.

실행: T2_PROBE_URL=http://localhost:8140/v1/chat/completions PYTHONPATH=…/tau2-bench/src \
  python x332_selfcount_checklist_iso.py [블록크기] [블록수]
"""
import os
import re
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

TAG, TASK, CUT, EARLY_CUT = "bank_t7295_b_20260815n", "task_073", 50, 30
RIGHT = "apply_checking_account_credit"
ASK = "\n[instruction] Carry out the next step of this request now."
# 모델이 센다 — 엔진은 항목이 무엇인지도, 몇 개인지도 말하지 않는다([[66]]).
SELFLIST = ("\n[instruction] First state how many separate things this customer has asked for in "
            "total. Then list each one and mark it DONE or NOT DONE based on what has actually "
            "been recorded by a tool in this conversation. Then state how many are still NOT DONE, "
            "and handle the remaining ones now.")


def schemas():
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="no_knowledge")
    out = []
    for t in env.get_tools():
        try:
            out.append(t.openai_schema)
        except Exception:
            continue
    return out


def emitted(msg):
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
    print("x332 · %s/%s · cut=%d · 본문 %d자 · 도구 %d · %d×%d블록\n"
          % (TAG, TASK, CUT, len(traj), len(tools), k, nb))
    arms = (("A_BASE", traj), ("B_SELFLIST", traj + SELFLIST),
            ("C_ASK", traj + ASK), ("D_EARLY", early + SELFLIST))
    res = {}
    for label, body in arms:
        blocks, t0h, t0n = [], 0, 0
        for b in range(nb):
            h = 0
            for i in range(k):
                try:
                    m = chat(body, tools, 0.0 if i == 0 else 0.7, 500)
                except Exception as e:
                    m = {"content": "ERR %s" % type(e).__name__}
                ok = emitted(m)
                h += ok
                if i == 0:
                    t0n += 1
                    t0h += ok
                shown = (",".join((tc.get("function") or {}).get("name", "")
                                  for tc in (m.get("tool_calls") or []))
                         or " ".join(str(m.get("content") or "").split())[:50])
                print("    [%s b%d %02d] %s %s" % (label, b + 1, i, "HIT" if ok else "-",
                                                   shown[:60]), flush=True)
            blocks.append(h)
        res[label] = (sum(blocks), blocks, t0h, t0n)
        print("%-11s %d/%d · 블록 %s · **temp0 %d/%d**\n"
              % (label, sum(blocks), k * nb, blocks, t0h, t0n))
    print("판정(사전 고정): D≥8 → 무효 · B−A≥5 → 체크리스트가 산다 · B−C≥5 → 촉구보다 낫다 · "
          "B≈C → 추가 이득 없음(더 싼 쪽) · B<C−5 → 해롭다")
    print("측정치: " + " · ".join("%s=%d%s temp0=%d/%d" % (a, v[0], v[1], v[2], v[3])
                                  for a, v in res.items()))


if __name__ == "__main__":
    sys.exit(main() or 0)
