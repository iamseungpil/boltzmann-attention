# -*- coding: utf-8 -*-
r"""x302 — kind-map 소유권 명명 격리: 073 credit 착수 축 (마지막 남은 칸).

배경(t7284 073 실물·산술 축은 이미 닫힘): 도구가 세 계좌 **$9.50/$9.00/$1.50 = gold** 를
냈는데도 credit write 가 안 나갔다. claimprov 실물(turn 73):
  `record_update: apply ATM fee corrections` → **owner split unknown=1** →
  도구명 없는 일반 문구(=x300 D_GEN 형·격리 **0/8**)가 나갔다.
FIX-8 토큰 매치는 "apply" 1개뿐이라 문턱(2) 미달로 **정당하게 침묵**했다(fail-open).

가용 자원: A2 `claim_prov.event_map` 이 이미 **kind→도구 접두**를 선언한다(원장 대조용) —
`record_update: ["__effective_write__","update_transaction_rewards","update_",
"apply_statement_credit","apply_checking_account_credit"]`. 이 선언 + env 레지스트리 교집합이
곧 후보 이름이다(A2 선언·기계 사실·도메인 텍스트 파싱 0·[[59]] 무결).
⚠ x300 은 **이름 1개**만 측정했다 — 이름 3개는 미측정이다(x287b: 8개 8/8 ↔ 31개 5/8 =
목록 크기가 효과 변수). 그래서 이 프로브가 필요하다([[62]]).

셀 4 (n=8·컷 = 073 이 "적용하겠다" 약속 후 미실행인 그 턴):
  A_LIVE   라이브 문구 축자(도구명 없는 일반 촉구)      — 재현 대조
  B_KIND   소유권 note + **kind-map 교집합 이름 전부**   — 출시 후보
  C_ONE    소유권 note + **credit 도구 1개만**           — x300 동형(상한 참조)
  D_NONE   note 없음                                      — 부정통제

계기: 다음 턴이 `apply_checking_account_credit_5829` 를 unlock/call 하는가.
판정(사전 고정): A_LIVE ≤2/8 ∧ D_NONE ≤2/8 ∧ **B_KIND ≥6/8** → kind-map 명명 출시.
  B_KIND ≤2/8 ∧ C_ONE ≥6/8 → **목록 크기가 인자** → 좁히는 규칙(토큰 겹침 최대 1개) 설계 후 재측정.
  둘 다 ≤2/8 → 이 축은 문면으로 안 열림 → 배치(FIX-10 동형·write 착수 서브) 검토.

실행(리모트·8141·**유료 런 종료 후**): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x302_kindmap_ownership_probe.py [N]
"""
import collections
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

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402
from x297_registry_hygiene_probe import REGISTRY                  # noqa: E402

TAG = "bank_t7284_b_20260814f"
TASK = "task_073"
TARGET = "apply_checking_account_credit_5829"
CLAIM = "apply ATM fee corrections"
A2P = "a2/banking_knowledge.gate.json"

NOTE_T = ("Note: [CLAIM-PROVENANCE] tool ownership — what you promised (%s) is done by a tool "
          "in YOUR OWN tool list, not the customer's: %s. The customer cannot run it, and "
          "describing the correction to them does not execute it - unlock it and call it "
          "yourself." % (CLAIM, "%s"))
A_LIVE = ("Error: [CLAIM-PROVENANCE] earlier in this conversation you told the customer you "
          "WOULD do the following, but the conversation ledger shows it was never actually "
          "executed: record_update: %s. You are about to end your involvement (resign or "
          "transfer) with these promises unfulfilled — that abandons the customer's request. "
          "Do the promised work NOW by calling the real tools." % CLAIM)


def kind_matches(kind="record_update"):
    """A2 event_map[kind] 접두 ∩ env 레지스트리 (A2 선언·기계 교집합·판단 0)."""
    a = json.load(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), A2P),
                          encoding="utf-8"))
    pre = ((a.get("claim_prov") or {}).get("event_map") or {}).get(kind) or []
    pre = [p for p in (pre if isinstance(pre, list) else [pre]) if not p.startswith("__")]
    return sorted({n for n in REGISTRY if any(n.startswith(p) for p in pre)})


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    # 컷 = 마지막 fee 도구 결과(net 산출) 직후 = 착수해야 할 그 지점
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and "net correction" in str(m.get("content") or ""):
            cut = i + 1
    if cut is None:
        print("컷 없음(net correction)")
        return
    names = kind_matches()
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    base = B.render(msgs[:cut], ours)
    base = base[:base.rfind("\n[user] ")] if "\n[user] " in base else base
    print("073 cut=%d(net 산출 직후) · kind-map 이름 %s · n=%d · URL=%s\n" % (
        cut, names, n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    arms = (("A_LIVE", A_LIVE), ("B_KIND", NOTE_T % ", ".join(names)),
            ("C_ONE", NOTE_T % TARGET), ("D_NONE", None))
    for label, note in arms:
        body = base + (("\n[system] " + note) if note else "")
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                # ⚠mx 는 1500 이상이어야 한다 (2026-08-14 실측): 이 모델은 content 를
                #   먼저 뱉고 tool_call 을 잇는데, 500 이면 `finish_reason=length` 로
                #   **호출 직전에 잘려** 전건이 '(text)' 로 기록된다. 같은 본문이 1500 에선
                #   도구를 부른다 — 산문이 긴 팔일수록 더 잘리므로 **처치와 상관된 인공물**이다.
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            blob = " ".join(str(t) for t in (r.get("tool_calls") or []))
            if TARGET in blob:
                hit += 1
                cnt["target"] += 1
            else:
                first = ""
                for t in (r.get("tool_calls") or []):
                    first = str(t.get("name") or "")
                    if "agent_tool_name" in str(t.get("arguments") or ""):
                        m2 = re.search(r'"agent_tool_name":\s*"([^"]+)"',
                                       str(t.get("arguments")))
                        if m2:
                            first += ":" + m2.group(1)
                    break
                cnt[first or ("(text)" if r.get("content") else "(empty)")] += 1
        print("%-8s target %d/%d · %s" % (label, hit, n, dict(cnt)))
    print("\n※ 판정(사전 고정): A_LIVE ≤2 ∧ D_NONE ≤2 ∧ B_KIND ≥6 → kind-map 명명 출시."
          " B_KIND ≤2 ∧ C_ONE ≥6 → 목록 크기가 인자(좁히는 규칙 후 재측정)."
          " 둘 다 ≤2 → 배치(write 착수 서브) 검토.")


if __name__ == "__main__":
    main()
