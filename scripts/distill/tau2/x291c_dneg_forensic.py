# -*- coding: utf-8 -*-
r"""x291c — D_NEG 이상치 포렌식(리뷰 지적 ③·[[08]]): 무관 문서 팔의 8건이 실제 무엇을 골랐나.

x291 실측: A_LIVE purple 7/8 ↔ D_NEG purple 0/8 — 무관(business) 문서를 얹기만 해도 Purple
픽이 전멸했다. A_LIVE 의 Purple 이 함정 소화가 아니라 문맥-민감 불안정 픽일 가능성을 가리는
정독(판정 자체는 불변·결론 문장 보강용). 전건 FINAL 줄 로그.

실행(리모트·8141): T2_PROBE_URL=... python x291c_dneg_forensic.py [N]
"""
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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(B.TAG) if s["task_id"] == B.TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    cut = next(i + 1 for i, m in enumerate(msgs)
               if m.get("role") == "user" and "$350" in str(m.get("content") or ""))
    docs_b = B.all_docs("checking_accounts", B.checking_classes())
    docs_n = B.all_docs("business_checking_accounts", B.business_classes())[:len(docs_b)]
    tools = U.tools_of(sim)
    P.TAG = B.TAG
    ours = P.our_lines(sim)
    body = B.render(msgs[:cut], ours, docs_n)
    print("D_NEG 재측정·전건 로그 · 문맥 %d자 · n=%d\n" % (len(body), n))
    for i in range(n):
        try:
            r = chat(body, tools, 0.0 if i == 0 else 0.7, 700)
        except Exception as e:
            r = {"content": "ERR %s" % type(e).__name__}
        t = " ".join(str(r.get("content") or "(no content)").split())
        fline = "(FINAL 줄 부재)"
        for ln in reversed(str(r.get("content") or "").splitlines()):
            if "FINAL" in ln.upper():
                fline = " ".join(ln.split())
                break
        print("  [%02d] %s" % (i, fline[:140]))
        print("       … %s" % t[-200:])


if __name__ == "__main__":
    main()
