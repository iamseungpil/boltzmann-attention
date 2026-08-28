# -*- coding: utf-8 -*-
r"""x588 - **빠진 변이(MISSING)** 를 per-step 으로 귀속한다 (모델 0 · 무료).

## 왜 (2026-08-28 밤)

`x587` 이 t7376/t7372 를 훑자 구조가 드러났다: **금액 WRONGARG 는 소수파**다.
t7376 의 18 실패 중 금액 변이가 있는 것은 1 건이고 **17 건은 `missing` 뿐**이다.
즉 지배적 실패형은 *"틀린 값을 썼다"* 가 아니라 ***"그 변이를 아예 안 했다"*** 이다.

## 어떻게 (닫힌 술어 · 추측 0)

빠진 gold 변이마다 **실효 도구 이름**(`call_discoverable_agent_tool` 이면 `agent_tool_name`)을
꺼내 궤적을 훑는다:

    NEVER_IN_CONTEXT    그 이름이 궤적 어디에도 없다            -> 발견 자체가 안 됨
    DELIVERED_NOT_CALLED 이름이 `role=tool` 본문에 배달됐는데 호출 0 -> **부하**(x583 형)
    CALLED_THEN_ERROR    호출했는데 결과가 `Error:`             -> 우리 층/env 가 막음
    CALLED_OTHER_ARGS    호출은 했는데 인자가 다름               -> 값 문제(x587 관할)

배달 시점(msg_i)과 첫 호출 시점을 함께 찍는다. **모든 행이 sim # msg 로 짚힌다.**
"""
import json
import re
import sys

sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")
import t2_forensic as F


def eff_name(e):
    a = e.get("args") or {}
    inner = a.get("arguments")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except Exception:
            inner = {}
    n = a.get("agent_tool_name") or (inner or {}).get("agent_tool_name")
    return n or e.get("name")


def scan(sim):
    msgs = sim.get("messages") or []
    d = F.mutation_diff(sim, F.mutating_tools(), tag=None) or {}
    out = []
    for e in (d.get("missing") or ()):
        if not isinstance(e, dict):
            continue
        nm = eff_name(e)
        if not nm:
            continue
        delivered = called = err_at = None
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and delivered is None:
                if nm in str(m.get("content") or ""):
                    delivered = i
            for tc in (m.get("tool_calls") or []):
                a = tc.get("arguments")
                a = a if isinstance(a, str) else json.dumps(a or {}, ensure_ascii=False)
                if nm == str(tc.get("name") or "") or nm in a:
                    if "unlock" in str(tc.get("name") or ""):
                        continue
                    if called is None:
                        called = i
                    tid = tc.get("id")
                    for j in range(i + 1, len(msgs)):
                        mj = msgs[j]
                        if mj.get("role") != "tool" or mj.get("id") != tid:
                            continue
                        if str(mj.get("content") or "").lstrip().startswith("Error:"):
                            err_at = j
                        break
        if called is None:
            verdict = "DELIVERED_NOT_CALLED" if delivered is not None else "NEVER_IN_CONTEXT"
        elif err_at is not None:
            verdict = "CALLED_THEN_ERROR"
        else:
            verdict = "CALLED_OTHER_ARGS"
        out.append({"tool": nm, "delivered_at": delivered, "called_at": called,
                    "error_at": err_at, "verdict": verdict})
    return out


def main(argv=None):
    for tag in ((argv or sys.argv[1:]) or ["bank_t7376_treat_20260828"]):
        try:
            sims = F.sims(tag)
        except Exception as ex:
            print("(못 읽음) %s : %r" % (tag, ex)); continue
        print("#" * 112)
        print("# %s" % tag)
        print("#" * 112)
        for s in sims:
            if (s.get("reward_info") or {}).get("reward") == 1.0:
                continue
            rows = scan(s)
            if not rows:
                continue
            print("%s" % F.simtag(s))
            for x in rows:
                print("   %-22s 판정 %-21s 배달 msg[%s] · 호출 msg[%s] · 오류 msg[%s]"
                      % (x["tool"][:22], x["verdict"], x["delivered_at"], x["called_at"], x["error_at"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
