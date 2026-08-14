# -*- coding: utf-8 -*-
r"""x298 — 소유권-정정 deny 격리: 손님에게 넘기려는 날조 도구명을 '네 도구'로 되돌리는가.

배경(t7277 075 엔진 로그 118행 실물): 모델의 착수 시도는 **`give_discoverable_user_tool
(discoverable_tool_name='open_account')`** — 자기서비스 접힘의 *도구 형태*(손님에게 넘김).
라이브 deny 는 `feedback_not_discoverable`(*"unlock_discoverable_agent_tool does not apply"*
+ 레지스트리 45개)로 **채널이 어긋난 문구**를 냈다: 모델은 give_ 를 불렀는데 문구는 unlock_
얘기를 하고, **그 이름이 사실은 에이전트 자신의 도구**(open_bank_account_4821)라는 말은 없다.
x297(초안을 unlock_ 로 오가정) A_FULL 1/8·B_TOK 2/8 = 그 컷의 계기 무효 — 초안 교체 재격리.

선례: x287b(근거확인 deny 가 접힘을 A0/8→B8/8 돌파)·claimprov ownership 노트(*"in YOUR OWN
tool list, not the customer's"*)는 이미 이 도메인에서 발화 중인 문형.

셀 3 (n=8·075 log_verification 성공 직후 컷·초안 = 라이브 축자 give_ 호출):
  A_LIVE  라이브 deny 축자(45개 목록·unlock_ 문구) — 재현 대조
  B_OWN   소유권 정정 deny: 손님-도구 부재 사실 + **토큰-겹침 에이전트 도구명**(기계 필터·
          x297 검증: 'open_account' → open_bank_account_4821 단일) + 손님은 실행 못 한다는 사실
  D_BARE  부재 단정만(목록·소유권 없음) — 부정통제

계기: 다음 턴이 open_bank_account_4821 을 unlock 또는 call 하는가(자기 도구 착수).
판정(사전 고정): A_LIVE ≤2/8 ∧ B_OWN ≥6/8 → 소유권-정정 deny 출시(도메인-일반: give_ 대상이
  손님 레지스트리에 없고 에이전트 레지스트리에 토큰-일치가 있으면 그 사실을 말한다).
  A_LIVE ≥6/8 → 라이브 문구도 충분(접힘 딴 원인). D_BARE ≥3/8 → 계기 오염. 중간 → n=16 1회.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x298_ownership_deny_probe.py [N]
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
import x291_checking_pick_iso as B                                # noqa: E402
from x297_registry_hygiene_probe import REGISTRY, tok_filter, FAB, TARGET, TAG, TASK  # noqa: E402

DRAFT = ("[assistant calls] give_discoverable_user_tool"
         "({'discoverable_tool_name': '%s'})" % FAB)

A_LIVE = ("Error: '%s' is not one of the discoverable tools in this domain, so it cannot be "
          "unlocked - unlock_discoverable_agent_tool does not apply to it. If it is a tool you "
          "already have, call it directly. Do not search the knowledge base for a suffixed "
          "version of this name: there is none. The discoverable tools that DO exist in this "
          "domain are: %s. If one of these performs the action, unlock and call that exact "
          "name." % (FAB, ", ".join(sorted(REGISTRY))))

D_BARE = ("Error: '%s' is not a discoverable tool in this domain." % FAB)


def b_own(match):
    return ("Error: '%s' is not a tool that can be given to the customer - there is no "
            "customer-side tool by that name on file. The tool on file that matches what you "
            "asked for is one of YOUR OWN agent tools: %s. The customer cannot run it and "
            "describing the steps to them does not execute it - unlock it and call it "
            "yourself." % (FAB, ", ".join(match)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and "Verification logged successfully" in str(
                m.get("content") or ""):
            cut = i + 1
            break
    if cut is None:
        print("컷 없음")
        return
    match = tok_filter(FAB, REGISTRY)
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    base = B.render(msgs[:cut], ours)
    base = base[:base.rfind("\n[user] ")]
    print("075 t%s cut=%d · 소유권 매치=%s · n=%d · URL=%s\n" % (
        sim.get("trial"), cut, match, n,
        os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, deny in (("A_LIVE", A_LIVE), ("B_OWN", b_own(match)), ("D_BARE", D_BARE)):
        body = base + "\n" + DRAFT + "\n[tool] " + deny
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                # ⚠mx=400 재검 (2026-08-14): 이 모델은 content 를 먼저 뱉고 tool_call 을
                #   잇는다. 400 이면 산문이 긴 팔에서 **호출 직전 절단**(finish=length)돼
                #   '(text)' 로 기록된다 — x295b 에서 0/8→7/8 로 뒤집힌 그 결함이다.
                #   음성 팔일수록 산문이 길어 **처치와 상관된 인공물**이 되므로 전 팔 재측정.
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            blob = " ".join(str(tc) for tc in (r.get("tool_calls") or []))
            if TARGET in blob:
                hit += 1
                cnt["target(%s)" % ("call" if "call_discoverable" in blob else "unlock")] += 1
            else:
                first = ""
                for tc in (r.get("tool_calls") or []):
                    first = str(tc.get("name") or (tc.get("function") or {}).get("name") or "")
                    break
                cnt[first or ("(text)" if r.get("content") else "(empty)")] += 1
        print("%-7s target %d/%d · %s" % (label, hit, n, dict(cnt)))
    print("\n※ 판정(사전 고정): A_LIVE ≤2/8 ∧ B_OWN ≥6/8 → 소유권-정정 deny 출시."
          " A_LIVE ≥6/8 → 딴 원인. D_BARE ≥3/8 → 계기 오염. 중간(3~5) → n=16 1회.")


if __name__ == "__main__":
    main()
