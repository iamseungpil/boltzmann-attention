# -*- coding: utf-8 -*-
r"""x285 — GB2 recovery "exactly this message" 가 공존-내용을 지우는가 (p런 010 t1 실물).

사전등록: `PROBE_X283_X285_DESIGN_2026_08_13.md` §x285.

## 근거 (P_RUN_FORENSIC_2026_08_13.md §1.4)

b#2(010 t1) msg34: 모델 초안이 정답(*"9-day window has passed ... you should be able to re-submit
the Platinum Rewards Card referral now"*)을 담은 그 턴에 GB2 deny 의 recovery *"Send the user
exactly this message now: 'TRANSFER NOTICE: ...'"* 가 발화 → regen 커밋은 **NOTICE 한 줄뿐**
(정답 소멸). [[64]]: 거부가 해법을 담되 **다른 내용을 지우게 말해선 안 된다**.

## 팔 (n=8 · 문턱 사전 고정)

  A_EXACT  현행 recovery 축자
  B_COEX   "Your reply must include this exact sentence - you may keep your other content
            around it: ..." (공존형)
  C_NEG    recovery 문장 없이 deny 본문만(fail-closed 유지 대조·[[64]])

계기 두 개를 **둘 다** 센다(§1.3 모트 — 게이트 자신의 역효과 계측):
  ⓐ NOTICE 문장 포함(게이트 요건 충족)   ⓑ 재제출-가능 내용 생존(window/re-submit/Platinum)

판정(사전 고정): B 성공 = ⓐ≥7/8 ∧ ⓑ≥6/8 ∧ (A 에서 ⓑ≤2/8). 승리 시 GB2 recovery 문면만
교체(A2 정본 층·[[24]] 사본 전수 확인 + gate.json 동기).

실행(리모트·GPU 유휴 확인 후): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x285_gb2_coexist_probe.py [N]
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

TAG = "bank_judge6p_b_20260813p"
TASK = "task_010"
TRIAL = 1
NOTICE = "TRANSFER NOTICE: Would you like to be transferred to a human agent?"
DENY_HEAD = ("Error: [POLICY GATE GB2_NOTICE_BEFORE_TRANSFER] blocked by policy gate: the "
             "pre-transfer notice (ask-first) has been communicated to the user not "
             "established. Recovery: (1) do NOT retry this tool now; ")
A_EXACT = (DENY_HEAD + "(2) the transfer is blocked only because the exact notice has not "
           "been communicated yet. Send the user exactly this message now: \"%s\" - even if "
           "the customer already asked for or agreed to the transfer (their agreement stays "
           "valid; this notice is a required disclosure, not asking again). Send it at most "
           "ONCE; after it is on the record, CALL transfer_to_human_agents as a tool call "
           "with an appropriate summary - do not repeat the notice." % NOTICE)
B_COEX = (DENY_HEAD + "(2) the transfer is blocked only because the exact notice has not "
          "been communicated yet. Your reply must include this exact sentence - you may "
          "keep your other content (answers, confirmations) around it: \"%s\" - even if "
          "the customer already asked for or agreed to the transfer (their agreement stays "
          "valid; this notice is a required disclosure, not asking again). Include it at "
          "most ONCE; after it is on the record, CALL transfer_to_human_agents as a tool "
          "call with an appropriate summary - do not repeat the notice." % NOTICE)
C_NEG = (DENY_HEAD + "(2) the pre-transfer notice must be communicated to the user before "
         "the transfer can run.")


def score(resp):
    c = str(resp.get("content") or "")
    lo = c.lower()
    a = NOTICE.lower()[:40] in lo
    b = ("platinum" in lo) and ("re-submit" in lo or "resubmit" in lo or "window" in lo)
    return a, b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK and s.get("trial") == TRIAL]
    if not sims:
        print("궤적 없음")
        return
    sim = sims[0]
    # cut = 정답-초안 직전 user 턴(msg33) 다음 — user 가 "창 지났으면 재제출" 을 물은 자리.
    cut = None
    for i, m in enumerate(sim["messages"]):
        c = str(m.get("content") or "").lower()
        if m.get("role") == "user" and ("re-submit" in c or "resubmit" in c) and "window" in c:
            cut = i + 1
    if cut is None:
        print("컷 지점 없음 — 설계 §x285 의 시점 재선정 1회 규칙에 따라 재선정 필요")
        return
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    print("trial=%s cut=%d · n=%d · URL=%s\n" % (
        TRIAL, cut, n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, recov in (("A_EXACT", A_EXACT), ("B_COEX", B_COEX), ("C_NEG", C_NEG)):
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
        # 모델이 transfer 를 시도했고 우리가 deny 로 답한 국면을 재현한다
        out.append("[assistant calls] transfer_to_human_agents")
        out.append("[tool] %s" % recov)
        body = "\n".join(out)
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            a, b = score(r)
            cnt["notice"] += int(a)
            cnt["content"] += int(b)
            cnt["both"] += int(a and b)
        print("%-8s 문맥 %6d자 · NOTICE %d/%d · 내용생존 %d/%d · 동시 %d/%d" % (
            label, len(body), cnt["notice"], n, cnt["content"], n, cnt["both"], n))
    print("\n※ 판정: B = NOTICE≥7/8 ∧ 내용생존≥6/8 ∧ (A 내용생존≤2/8) → GB2 recovery 문면 교체"
          "([[24]] 정본+gate.json 동기). A 가 이미 내용을 살리면 라이브 소멸은 다른 자리 — 재수사.")


if __name__ == "__main__":
    main()
