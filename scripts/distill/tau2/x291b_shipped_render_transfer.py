# -*- coding: utf-8 -*-
r"""x291b — (B2) 출시본 축자 전이 확인: 실제 도구 렌더 출력이 C_CALC 효과를 재현하는가.

배경: x291 C_CALC 8/8 은 **합산 표**(수기)였다. 출시본 (B2)는 스태킹 미규정([[23]])이라
**축별 2열 분리** + 파이썬-repr 렌더 — 문면이 다르므로 x287b 교훈(문면 리터럴=출시본 축자)
대로 전이를 재확인한다. 사전등록 문턱: ≥6/8 출시 확정 · ≤2/8 이면 렌더/2열이 효과를 죽인
것 — (B2) 보류·표 형식 포렌식 · 3~5 → n=16 1회.

셀 1 (n=8·계기 = x291 동일 FINAL 줄 feefree):
  C_SHIP  x291 C_CALC 와 같은 위치에, 수기 표 대신 **A2 정본 op 실행 + return_template 렌더
          축자**를 주입 (docs 는 B_DOCS 와 동일 — 출시 후 라이브에서 문서·도구가 공존하므로).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x291b_shipped_render_transfer.py [N]
"""
import io
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
import x291_checking_pick_iso as B                                # noqa: E402
from t2_compute import apply_op                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def shipped_text():
    """A2 정본에서 op 로드 → 075 사용 패턴 실행 → return_template 렌더 축자."""
    a = json.load(io.open(os.path.join(HERE, "a2/banking_knowledge.specific.json"),
                          encoding="utf-8"))
    e = next(t for t in a["scaffold_get_tools"]
             if t["name"] == "get_checking_atm_fee_totals")
    res = apply_op(e["op"], {"months": 3, "withdrawals_per_month": 6,
                             "withdrawal_amount": 350})
    return e["return_template"].format(result=res)   # 라이브 _render_scalar 동형(str 포맷)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(B.TAG) if s["task_id"] == B.TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    cut = next(i + 1 for i, m in enumerate(msgs)
               if m.get("role") == "user" and "$350" in str(m.get("content") or ""))
    docs_b = B.all_docs("checking_accounts", B.checking_classes())
    tools = U.tools_of(sim)
    P.TAG = B.TAG
    ours = P.our_lines(sim)
    calc = shipped_text()
    body = B.render(msgs[:cut], ours, docs_b, calc)
    print("075 t%s cut=%d · 렌더 %d자 · n=%d · URL=%s\n" % (
        sim.get("trial"), cut, len(calc), n,
        os.environ.get("T2_PROBE_URL", "localhost:8140")))
    hits = purples = nofmt = 0
    for i in range(n):
        try:
            r = chat(body, tools, 0.0 if i == 0 else 0.7, 700)
        except Exception as e:
            r = {"content": "ERR %s" % type(e).__name__}
        h, pu, fmt = B.final_pick(r.get("content"))
        hits += h
        purples += pu
        nofmt += (not fmt)
        if i < 2 or not h:                     # [[08]] 정독용: 초반 2건+미스 전건 스니펫
            t = " ".join(str(r.get("content") or "(no content)").split())
            print("  [%d]%s %s" % (i, "" if h else " MISS", t[-260:]))
    print("C_SHIP  문맥 %6d자 · feefree %d/%d · purple %d · FINAL줄부재 %d" % (
        len(body), hits, n, purples, nofmt))
    print("\n※ 사전등록: ≥6/8 → (B2) 출시 확정 · ≤2/8 → 보류·표 형식 포렌식 · 3~5 → n=16 1회.")


if __name__ == "__main__":
    main()
