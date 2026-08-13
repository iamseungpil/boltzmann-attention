# -*- coding: utf-8 -*-
r"""x283b — 출시 문면 축자 검증: 레지스트리-폴백 문장(REG_FB)이 C_STEP2 와 같은 효과인가.

사전등록: `PROBE_X283_X285_DESIGN_2026_08_13.md` §x283b (2026-08-13·실행 전 기입).

x283 C_STEP2(=KB-출처절 문면+실명)는 071 t1/t3 에서 8/8 로 체인을 열었다. 출시 경로는
회수-실패 폴백이라 KB-출처절이 거짓이 되므로 출처절만 레지스트리로 바꾼 REG_FB 를 쓴다
(`t2_resolve.DISCOVERY_STEP2_REG_FB` — 이 스크립트는 그 상수를 **import 해서** 잰다·[[03b]]
두 벌 금지). 문턱(사전 고정): **071 t1·t3 각각 ≥6/8**. t0(2/8였던 문맥)은 정보용.

실행(리모트·GPU0=8140 전용 — 사용자 지시 2026-08-13):
  python x283b_regfb_verbatim.py [N]
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
import t2_resolve as R                                            # noqa: E402
from x266_decide_ask_axis import a2 as _a2                        # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    a2 = _a2()
    unlock = (a2.get("eplan") or {}).get("unlock_tool")
    e_reg = R.DISCOVERY_STEP2_REG_FB.format(name=P.LOCKED, unlock=unlock)
    sims = [s for s in X.load(P.TAG)
            if s["task_id"] == "task_071" and (s.get("reward_info") or {}).get("reward") != 1]
    print("071 실패 궤적 %d개 · n=%d · URL=%s" % (
        len(sims), n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    print("E_REG 축자: %s\n" % e_reg)
    for sim in sims:
        cut = P.pick_cut(sim)
        if cut is None:
            print("  (컷 없음 — 건너뜀 t%s)" % sim.get("trial"))
            continue
        tools = U.tools_of(sim)
        body = P.build(sim, cut, e_reg)
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 200)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[P.hit(r)] += 1
        print("  t%-2s E_REG 문맥 %6d자 · HIT %d/%d" % (sim.get("trial"), len(body), c["HIT"], n))
    print("\n※ 문턱(사전 고정): t1·t3 각각 ≥6/8 → 폴백 문면 확정. 미달 → 문면 재론(개정 1회만).")


if __name__ == "__main__":
    main()
