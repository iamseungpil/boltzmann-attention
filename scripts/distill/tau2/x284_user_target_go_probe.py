# -*- coding: utf-8 -*-
r"""x284 — user-측 표적에 GO 1문장이 지시를 낳는가 (p런 010 0/4).

사전등록: `PROBE_X283_X285_DESIGN_2026_08_13.md` §x284.

## 근거 (P_RUN_FORENSIC_2026_08_13.md §1)

p런 010 은 선행 read 완료(msg 19~21) 후에도 에이전트가 "고객이 submit_referral 을 실행하라"는
지시를 끝내 발화하지 않았다(4/4). 우리 층은 CANNOT-YET 를 제때 떨어뜨렸지만 **긍정 해제(GO)
신호가 구조적으로 없다** — `cleared()` 는 모델이 직접 시도했다 보류된 표적만 부활시키고,
user-측 도구는 모델이 시도 자체를 안 한다. 전달량-성적 단조 역상관(k 2/4 침묵 → m 1/4 부분 →
p 0/4 완전 전달)이 방향을 가리킨다.

## 팔 (n=8 · 문턱 ≥6/8 사전 고정)

  A_LIVE  read 완료 직후 턴의 우리 병합 푸시([ACTION] 소유권+상태표) 그대로
  B_GO    A + DEFAULT_CLEARED 동형 1문장(t2_dominance 축자 형식·[[03b]])
  C_NONE  우리 푸시 전면 제거(k런 침묵 최고성적 재현 대조)
  D_NEG   A + 무내용 강조(길이 통제·[[57]])

통과(HIT) = 다음 어시스턴트 발화가 고객에게 submit_referral 실행을 지시
(`submit_referral` 실명 언급 ∨ "re-submit"+"Platinum" 동시 출현 — 닫힌 문자열 술어).

판정(사전 고정): B≥6/8 ∧ A≤2/8 → GO 부재 근인 확정 → 라이브 수리 = 요건 공집합 **전이 시**
cleared() 1회(결정론 상태전이·의도분류 0). C 가 B 와 동급이면 처방은 GO 추가가 아니라 **철회**([[65]]).

실행(리모트·GPU 유휴 확인 후): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x284_user_target_go_probe.py [N]
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
TOOL = "submit_referral"
# ★DEFAULT_CLEARED 축자가 아니다 — 그 문면은 "Do '{target}' now, **as a real tool call**" 이라
#   user-측 도구엔 자기모순이 된다(에이전트가 부르면 우리가 deny 하는 그 호출을 명령).
#   아래는 그 형식의 user-run 변형이고, 프로브가 이기면 **출시도 이 변형을 cleared() 옆에
#   신설**하는 것이다([[03b]] 측정문면=출시문면 — 두 벌이 아니라 이것이 출시본).
GO = ("[ORDER] Earlier '%s' was held back because the prior read(s) were missing. "
      "That now holds. It is the CUSTOMER who runs '%s' - tell them in this reply to "
      "run it now with their details (user_id, account_type), then confirm the result."
      % (TOOL, TOOL))
NEG = "Note: please make sure to help the customer with their request."


def reads_done_cut(sim):
    """계좌조회 디스패치의 tool 결과 **다음** user 턴 자리 — 요건이 막 공집합이 된 지점."""
    done = None
    pend = {}
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            nm = (tc.get("function") or {}).get("name") or tc.get("name") or ""
            if nm == "call_discoverable_agent_tool" and "get_all_user_accounts" in str(
                    tc.get("arguments") or {}):
                pend[tc.get("id")] = i
        if m.get("role") == "tool" and (m.get("id") in pend or m.get("tool_call_id") in pend):
            done = i
            break
    if done is None:
        return None
    for j in range(done + 1, len(sim["messages"])):
        if sim["messages"][j].get("role") == "user" and str(
                sim["messages"][j].get("content") or "").strip():
            return j + 1
    return None


def hit(resp):
    c = str(resp.get("content") or "").lower()
    if TOOL in c:
        return "HIT"
    if ("re-submit" in c or "resubmit" in c) and "platinum" in c:
        return "HIT"
    return "MISS"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG)
            if s["task_id"] == TASK and (s.get("reward_info") or {}).get("reward") != 1]
    print("실패 궤적 %d개 · n=%d · URL=%s" % (
        len(sims), n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    print("GO 축자: %s\n" % GO)
    grand = collections.Counter()
    P.TAG = TAG                              # our_lines 가 이 태그의 사이드카를 읽게
    for sim in sims:
        cut = reads_done_cut(sim)
        if cut is None:
            print("  (read 완료 지점 없음 — 건너뜀 trial=%s)" % sim.get("trial"))
            continue
        tools = U.tools_of(sim)
        ours = P.our_lines(sim)
        # cut 직전 턴들에 나갔던 우리 문장(병합 푸시)이 A_LIVE 의 실물이다
        for label, mode in (("A_LIVE", "live"), ("B_GO", "go"),
                            ("C_NONE", "none"), ("D_NEG", "neg")):
            out = []
            for i, m in enumerate(sim["messages"][:cut]):
                r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
                tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
                       for tc in (m.get("tool_calls") or [])]
                if tcs:
                    out.append("[%s calls] %s" % (r, ", ".join(tcs)))
                if c:
                    out.append("[%s] %s" % (r, c[:700]))
                if mode != "none":
                    for t in ours.get(i, ()):
                        out.append("[system] %s" % t[:1100])
            if mode == "go":
                out.append("[system] %s" % GO)
            elif mode == "neg":
                out.append("[system] %s" % NEG)
            body = "\n".join(out)
            cnt = collections.Counter()
            for i in range(n):
                try:
                    r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                cnt[hit(r)] += 1
            grand[label] += cnt["HIT"]
            print("  t%-2s %-7s 문맥 %6d자 · HIT %d/%d" % (
                sim.get("trial"), label, len(body), cnt["HIT"], n))
    print("\n합계: " + " · ".join("%s %d" % (k, v) for k, v in sorted(grand.items())))
    print("※ B≥6/8 ∧ A≤2/8 → GO 부재 근인 → cleared() 전이-발화 수리. C≈B → 철회가 처방([[65]]).")


if __name__ == "__main__":
    main()
