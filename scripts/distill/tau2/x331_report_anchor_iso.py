# -*- coding: utf-8 -*-
r"""x331 — 실행 대신 **보고**하는 잔여: 자기가 쓴 보고문에 정박한 것인가.

## 왜 (x330 의 직접 후속)

x330(같은 컷·n=24·도구 바인딩): 이름 **18/24** ↔ 실행 **2/24**(부정통제 0/24). 촉구를 넣으면
**11/24**. 남는 절반의 정체를 원자료로 읽으면 **모양이 하나다** — 미실행 22건이 전부 보고문이고
(*"I have reviewed the transactions…"* · *"Based on the information retrieved, there are …"*),
촉구 팔의 잔여 13건도 같은 문장으로 시작한다. 1건은 **unlock 만 하고 멈췄다**(072 라이브와 동형).

가설: **문맥에 이미 자기가 쓴 보고문이 쌓여 있어서** 다음 수도 보고가 된다(자기-정박).
[[63]] 대로 **제거로만** 잰다 — 무엇도 더하지 않는다.

## 셀 4 (컷 = 073#0 msg 50 · n=24=8×3 · 도구 바인딩 · 지시 없음)

    A_BASE     x330 `B_EMIT` 재현 (라이브 축자)                  ← 기준선(≈2 나와야 계기 정상)
    B_NOPROSE  **어시스턴트 산문 전부 제거**(손님 발화·도구 출력 유지) ← 자기-정박 제거
    C_LASTONLY **직전 어시스턴트 보고 1개만 제거**                 ← 최소 개입판
    D_ASK      x330 `C_EMIT_ASK` 재현(촉구)                      ← 대조 기준(≈11)

⚠제거만 한다 — 문장을 짓지 않고, 도구를 지목하지 않으며, 무엇이 남았는지 알려주지도 않는다.
⚠도구 출력은 **전부 남긴다**. x326 의 `A_MIN`(0/24)은 어시스턴트 산문과 함께 *계산된 요약*까지
  날려서 정보가 빈약해진 프로브였다 — 여기서는 그 요약의 출처인 `get_atm_fee_discrepancies`
  **도구 출력이 그대로 남는다**(073 은 msg 49 가 그 출력이다). 같은 함정을 되풀이하지 않는다.

## 판정 (사전 고정 · 잡음 바닥 ±4·C483)

    A_BASE ≥ 8                → **재현 실패** = 계기 이상. 아래를 읽지 말 것.
    B_NOPROSE − A_BASE ≥ 5    → ★**자기-정박이 원인**. 레버 = 우리 층이 보고문을 되먹이지 않기
    C_LASTONLY − A_BASE ≥ 5   → **직전 보고 하나**로 충분(더 작은 개입)
    B ≈ C ≈ A                 → 자기-정박 **아님** ⇒ 잔여는 다른 곳(다음 프로브로)
    D_ASK 로 11 부근 재현      → 계기 정상 확인용

실행(비어 있는 GPU 포트): T2_PROBE_URL=http://localhost:8140/v1/chat/completions \
  PYTHONPATH=…/tau2-bench/src python x331_report_anchor_iso.py [블록크기] [블록수]
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

TAG, TASK, CUT = "bank_t7295_b_20260815n", "task_073", 50
RIGHT = "apply_checking_account_credit"
DO_ASK = "\n[instruction] Carry out the next step of this request now."


def transcript(sim, cut, drop_assistant=False, drop_last_assistant=False):
    """축자 전사 — **제거만** 한다(문장 생성 0). `B.transcript` 와 같은 형식을 유지한다."""
    rows = []
    for m in (sim.get("messages") or [])[:cut]:
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        if role == "tool":
            if m.get("error") or not c:
                continue
            rows.append(("tool", "[tool] " + c[:1200]))
        elif role in ("user", "assistant") and c:
            rows.append((role, "[%s] %s" % (role, c[:900])))
    if drop_assistant:
        rows = [r for r in rows if r[0] != "assistant"]
    elif drop_last_assistant:
        for i in range(len(rows) - 1, -1, -1):
            if rows[i][0] == "assistant":
                rows.pop(i)
                break
    return "\n".join(t for _r, t in rows)


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
    head = B.HEAD + "\n\n"
    base = head + transcript(sim, CUT)
    noprose = head + transcript(sim, CUT, drop_assistant=True)
    lastonly = head + transcript(sim, CUT, drop_last_assistant=True)
    tools = schemas()
    print("x331 · %s/%s · cut=%d · 본문 base %d자 / noprose %d자 / lastonly %d자 · 도구 %d · %d×%d\n"
          % (TAG, TASK, CUT, len(base), len(noprose), len(lastonly), len(tools), k, nb))
    if not tools or len(noprose) < 200:
        print("계기 이상 — 중단"); return 1

    arms = (("A_BASE", base), ("B_NOPROSE", noprose),
            ("C_LASTONLY", lastonly), ("D_ASK", base + DO_ASK))
    res = {}
    for label, body in arms:
        blocks = []
        for b in range(nb):
            h = 0
            for i in range(k):
                try:
                    m = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
                except Exception as e:
                    m = {"content": "ERR %s" % type(e).__name__}
                ok = emitted(m)
                h += ok
                shown = (",".join((tc.get("function") or {}).get("name", "")
                                  for tc in (m.get("tool_calls") or []))
                         or " ".join(str(m.get("content") or "").split())[:50])
                print("    [%s b%d %02d] %s %s" % (label, b + 1, i, "HIT" if ok else "-",
                                                   shown[:60]), flush=True)
            blocks.append(h)
        res[label] = (sum(blocks), blocks)
        print("%-11s %d/%d · 블록 %s\n" % (label, sum(blocks), k * nb, blocks))
    print("판정(사전 고정): A≥8 → 재현 실패(계기 이상) · B−A≥5 → **자기-정박이 원인** · "
          "C−A≥5 → 직전 보고 하나로 충분 · B≈C≈A → 자기-정박 아님 · D≈11 → 계기 정상")
    print("측정치: " + " · ".join("%s=%d%s" % (a, v[0], v[1]) for a, v in res.items()))


if __name__ == "__main__":
    sys.exit(main() or 0)
