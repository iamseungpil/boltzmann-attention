# -*- coding: utf-8 -*-
r"""x299 — 접힘-누적 문맥에서의 소유권 deny + 강제-도구 (t7278 075 실패 지점 재격리).

배경: FIX-7 은 t7278 075 turn31 에 **정확히 발화**(사이드카 실물)했는데 모델은 또 수동 안내를
냈다. x298(6/8)과 갈린 유일한 차이 = **컷 위치**: x298 은 log_verification 직후(접힘 텍스트 1회)
였고 라이브 실패 지점은 그 **뒤**(접힘 안내가 문맥에 누적된 상태·[[64]] C413 동형 — 이름 없는
접힘 문구가 3회↑ 나온 sim 은 6/6 실패). 즉 x298 은 낙관 편향 컷이었다([[08]] 자기교정).

셀 4 (n=8·컷 = t7278 075 msg31 직전 = 접힘 텍스트(msg29)+GO(msg30) 포함·초안=라이브 give_ 호출):
  A_LIVE   구 deny(45개 목록·unlock 문구)          — 재현 대조
  B_OWN    FIX-7 소유권 deny(출시본 축자)           — 라이브 재현(실패 예상)
  C_FORCE  FIX-7 소유권 deny + **tool_choice=required**(산문 턴 금지·T2_UNLOCK_NAME 선례)
  D_BARE   부재 단정만 + required                   — 강제만으로 되는지(부정통제·귀속)

계기: 그 턴이 open_bank_account_4821 을 unlock/call 하는가.
판정(사전 고정): C_FORCE ≥6/8 ∧ B_OWN ≤2/8 ∧ **D_BARE ≤2/8** → 소유권 경로에 force_required
  출시(문면+강제의 결합이 필요). C_FORCE ≥6 ∧ D_BARE ≥6 → 강제만으로 충분(문면 불요·FIX-7
  유지 여부 재검토). C_FORCE ≤2 → 접힘-누적은 문면·강제로 안 닫힘 = 학습행(L1 확정).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x299_fold_context_probe.py [N]
"""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x216_read_and_offset as R                                  # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402
from x297_registry_hygiene_probe import REGISTRY, FAB, TARGET      # noqa: E402
from x298_ownership_deny_probe import A_LIVE, D_BARE, b_own, DRAFT  # noqa: E402

TAG = "bank_t7278_b_20260813z"
TASK = "task_075"
MATCH = ["open_bank_account_4821"]


def chat_tc(prompt, tools, temp, mx, tool_choice="auto"):
    """x216.chat 동형 + tool_choice 노출(강제-도구 팔 측정용)."""
    body = {"model": R.MODEL, "temperature": temp, "max_tokens": mx,
            "messages": [{"role": "user", "content": prompt}]}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(R.URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    # 컷 = 마지막 user 발화(GO/확인) 직후 = 라이브가 접힌 그 턴의 입력
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and "set it up" in str(m.get("content") or "").lower():
            cut = i + 1
    if cut is None:
        for i, m in enumerate(msgs):
            if m.get("role") == "user":
                cut = i + 1
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    base = B.render(msgs[:cut], ours)
    base = base[:base.rfind("\n[user] ")] if "\n[user] " in base else base
    folds = sum(1 for m in msgs[:cut] if m.get("role") == "assistant"
                and "Visit the Rho-Bank" in str(m.get("content") or ""))
    print("075 cut=%d · 접힘 텍스트 %d회 누적 · n=%d · URL=%s\n" % (
        cut, folds, n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    arms = (("A_LIVE", A_LIVE, "auto"), ("B_OWN", b_own(MATCH), "auto"),
            ("C_FORCE", b_own(MATCH), "required"), ("D_BARE", D_BARE, "required"))
    for label, deny, tc in arms:
        body = base + "\n" + DRAFT + "\n[tool] " + deny
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat_tc(body, tools, 0.0 if i == 0 else 0.7, 400, tc)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            blob = " ".join(str(t) for t in (r.get("tool_calls") or []))
            if TARGET in blob:
                hit += 1
                cnt["target"] += 1
            else:
                first = ""
                for t in (r.get("tool_calls") or []):
                    first = str(t.get("name") or (t.get("function") or {}).get("name") or "")
                    first += ":" + str(t.get("arguments") or
                                       (t.get("function") or {}).get("arguments") or "")[:38]
                    break
                cnt[first or ("(text)" if r.get("content") else "(empty)")] += 1
        print("%-8s [%s] target %d/%d · %s" % (label, tc, hit, n, dict(cnt)))
    print("\n※ 판정(사전 고정): C_FORCE ≥6 ∧ B_OWN ≤2 ∧ D_BARE ≤2 → 소유권 경로 force_required"
          " 출시. C_FORCE ≥6 ∧ D_BARE ≥6 → 강제만으로 충분. C_FORCE ≤2 → 학습행(L1).")


if __name__ == "__main__":
    main()
