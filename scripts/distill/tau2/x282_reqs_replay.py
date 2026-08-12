# -*- coding: utf-8 -*-
r"""x282 — p런 010 0/4 부검: `requirements_for(submit_referral)` 오프라인 격리 재생.

관측(사이드카 전수): 12 sim 전부 마지막 [ORDER] 푸시가 CANNOT-YET 이고, 끝까지 남는 조건이
`get_all_user_accounts_by_user_id has not been called` — 그런데 궤적은 그 도구를 디스패처
(`call_discoverable_agent_tool(agent_tool_name="..._3847")`)로 msg 18~20 에 실행했다.

가설 두 갈래(이 프로브가 가른다):
  H1 검사기 결함 — 전체 prefix 를 줘도 `_reqs` 가 안 빈다 (이름공간/크레딧 버그).
  H2 라이브 상태 결함 — 검사기는 전체 정보에서 비운다. 그러면 라이브(unified_regen 시점)의
     state.messages 가 커밋된 궤적과 달랐던 것 — regen 경로를 판다.

방법: results.json 의 각 010 sim 에 대해 prefix K=0..N 마다
  executed_K = _executed_tool_names 동형 재계산(정확명·실패마커 동형)
  requirements_for(a2, [], "submit_referral", executed=executed_K)
를 부르고, 미충족 요건 id 집합이 K 에 따라 언제 어떻게 변하는지 인쇄한다.
전부 기존 결정 로직 재사용 — 신규 판정 0 ([[62]] 해당 없음·진단 전용).

실행(리모트): python3 x282_reqs_replay.py [tag] [task_id] [target]
기본: bank_judge6p_b_20260813p task_010 submit_referral
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x266_decide_ask_axis import a2 as _a2                          # noqa: E402
import t2_dominance as _DOM                                          # noqa: E402

SIMS = os.environ.get("X282_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")


def exact_name(tc):
    """dict 판 `_exact_tool_name` — 디스패처(call_*)는 인자의 레지스트리 실명."""
    nm = str(tc.get("name") or "")
    if nm.startswith("call_"):
        a = tc.get("arguments") or {}
        raw = a.get("arguments")
        inner = (a.get("agent_tool_name") or a.get("user_tool_name")
                 or a.get("discoverable_tool_name") or "")
        if not inner and isinstance(raw, str):
            try:
                inner = (json.loads(raw) or {}).get("agent_tool_name") or ""
            except Exception:
                inner = ""
        if inner:
            return str(inner)
    return nm


def executed_upto(msgs, k, marks):
    """dict 판 `_executed_tool_names` — prefix `msgs[:k]` 기준 실행-성공 정확명 집합."""
    ok, pending = set(), {}
    for m in msgs[:k]:
        for tc in (m.get("tool_calls") or []):
            pending[tc.get("id")] = exact_name(tc)
        if m.get("role") == "tool":
            nm = pending.get(m.get("id") or m.get("tool_call_id"))
            txt = str(m.get("content") or "").lstrip()
            failed = (m.get("error") or txt.startswith("Error:")
                      or any(txt.startswith(x) for x in marks))
            if nm and not failed:
                ok.add(nm)
    return ok


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_judge6p_b_20260813p"
    want = sys.argv[2] if len(sys.argv) > 2 else "task_010"
    target = sys.argv[3] if len(sys.argv) > 3 else "submit_referral"
    a2 = _a2()
    marks = tuple(a2.get("failure_markers") or ())
    d = json.load(io.open(os.path.join(SIMS, tag, "results.json"), encoding="utf-8"))
    for si, s in enumerate(d["simulations"]):
        if s["task_id"] != want:
            continue
        msgs = s.get("messages") or []
        print("=" * 96)
        print("%s sim#%d trial=%s msgs=%d target=%s" % (want, si, s.get("trial"), len(msgs), target))
        prev = None
        for k in range(len(msgs) + 1):
            done = executed_upto(msgs, k, marks)
            reqs = _DOM.requirements_for(a2, [], target, executed=done)
            sig = tuple(sorted(r["id"] for r in reqs))
            if sig != prev:
                names = {"%s->%s" % (r["id"], ",".join(r.get("satisfiers") or [])) for r in reqs}
                print("  K=%-3d outstanding=%d  %s" % (k, len(reqs), sorted(names) or "∅ (GO 가능)"))
                prev = sig
        # 최종 prefix 에서의 done 집합(가족명 병기)
        done = executed_upto(msgs, len(msgs), marks)
        fam = sorted({re.sub(r"_\d+$", "", x) for x in done})
        print("  done(exact)=%s" % sorted(done))
        print("  done(fam)  =%s" % fam)


if __name__ == "__main__":
    main()
