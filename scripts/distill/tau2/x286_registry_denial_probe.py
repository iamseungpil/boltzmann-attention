# -*- coding: utf-8 -*-
r"""x286 — 레지스트리 부재-단정 반박이 대체-추천을 막는가 (098 t2형).

사전등록: `PROBE_X283_X285_DESIGN_2026_08_13.md` §x286.

실행(리모트·GPU 유휴): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x286_registry_denial_probe.py [N]
"""
import collections
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

TAG = "bank_judge6p_a_20260813p"
TASK = "task_098"
TRIAL = 2
B_REG = ("Error: [CLAIM-PROVENANCE] your reply asserts 'Blue Account' is not listed - but "
         "the account registry DOES list blue_account (checking accounts). Do not substitute "
         "a different account for the one the customer named; if its details were not "
         "retrieved, say plainly what you could not retrieve.")
NEG = "Note: please continue helping the customer with their referral request."


def score(resp):
    c = str(resp.get("content") or "").lower()
    blue = "blue account" in c
    subst = ("light blue" in c) or ("green fee" in c)
    return "HIT" if (blue and not subst) else "MISS"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK and s.get("trial") == TRIAL]
    if not sims:
        print("궤적 없음")
        return
    sim = sims[0]
    # cut = 부재-단정 초안 턴 — "does not seem to be explicitly listed" 발화 자리
    cut = None
    draft = None
    for i, m in enumerate(sim["messages"]):
        c = str(m.get("content") or "")
        if m.get("role") == "assistant" and "not seem to be explicitly listed" in c:
            cut, draft = i, " ".join(c.split())
            break
    if cut is None:
        print("컷 지점 없음 — 시점 재선정 1회 규칙 적용 필요")
        return
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    print("trial=%s cut=%d · n=%d · URL=%s\n" % (
        TRIAL, cut, n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, inject in (("A_LIVE", None), ("B_REG", B_REG), ("D_NEG", NEG)):
        out = []
        for i, m in enumerate(sim["messages"][:cut]):
            r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
            tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
                   for tc in (m.get("tool_calls") or [])]
            if tcs:
                out.append("[%s calls] %s" % (r, ", ".join(tcs)))
            if c:
                out.append("[%s] %s" % (r, c[:700]))
            for t in ours.get(i, ()):
                out.append("[system] %s" % t[:1100])
        # regen 국면 재현: 초안이 있었고, (B/D) 우리 노트가 붙는다
        out.append("[assistant-draft] %s" % draft[:700])
        if inject:
            out.append("[system] %s" % inject)
        out.append("[system] Rewrite your reply to the customer now.")
        body = "\n".join(out)
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            cnt[score(r)] += 1
        print("%-7s 문맥 %6d자 · HIT %d/%d" % (label, len(body), cnt["HIT"], n))
    print("\n※ 판정: B≥6/8 ∧ A≤2/8 → 부재-단정 반박 레버(claimprov 술어 확장·닫힌 검사)."
          "\n  t0형(단정 없는 조용한 대체)은 범위 밖 — 이겨도 098 기대 +1/4.")


if __name__ == "__main__":
    main()
