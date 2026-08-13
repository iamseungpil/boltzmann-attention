# -*- coding: utf-8 -*-
r"""x301b — 라이브 컷 스펙-에코 격리: 형식화 시점에 param 스펙을 재제시하면 C_DUP 효과가
라이브 문맥에서도 재현되는가 (t7282 073 실물 컷).

배경: x301 C_DUP 8/8(스펙을 프롬프트 본문에 제시) ↔ t7282 라이브 = A_CUR 동형(오답 그대로·
"paired withdrawal" 문맥 등장 0 — 문면이 도구 스키마 속에 묻혀 형식화 시점에 미도달).
x298↔x299 가 실증한 "시점이 인자" 원리의 형식화판. 컷 = 라이브 첫 인라인-JSON 형식화 호출 턴.

셀 3 (n=8·계기 = 재생성된 호출 인자의 rho 정오·duplicate_of 유무):
  A_ASIS  컷 그대로 재생성 — 라이브 오답 재현 대조
  B_ECHO  + [해당 호출 + deny: "re-issue following this spec:" + A2 param 스펙 축자]
  D_OTHER + [동일 deny 형식 + **무관 도구**(get_checking_atm_fee_totals) 스펙 동수] — 통제

판정(사전 고정): A_ASIS rho ≤2/8 ∧ B_ECHO rho ≥6/8 ∧ D_OTHER ≤2/8 → **스펙-에코 deny 출시**
  (도메인-일반: A2 `formalize_echo` 선언 도구의 첫 호출 1회 deny + param 스펙 재제시·fail-open).
  B_ECHO ≤2/8 → 라이브 문맥에선 재제시로도 안 닫힘 → 격리-서브 formalize 경로로.
  D_OTHER ≥3/8 → 무효(형식 효과).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x301b_spec_echo_live_probe.py [N]
"""
import collections
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

TAG = "bank_t7282_b_20260814d"
TASK = "task_073"
TOOL = "get_atm_fee_discrepancies"
OTHER = "get_checking_atm_fee_totals"
A2P = "a2/banking_knowledge.specific.json"
# 계좌1 형식화 대상 rho fee 라인(원장 실측): btxn_kj07s5t6u7v9 (RHO-BANK $400 인출 위 $3)
RHO_FEE = "btxn_kj07s5t6u7v9"
RHO_WD = "btxn_kj07s5t6u7v8"


def spec_of(name):
    a = json.load(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), A2P),
                          encoding="utf-8"))
    e = next(t for t in a["scaffold_get_tools"] if t["name"] == name)
    p = e.get("params") or {}
    return "transactions: %s" % p.get("transactions", "") if name == TOOL else \
        "\n".join("%s: %s" % (k, v) for k, v in p.items())


def judge(msg):
    """재생성 호출에서 계좌1 rho fee 라인의 network + duplicate_of 사용 여부."""
    for tc in (msg.get("tool_calls") or []):
        if TOOL not in str(tc.get("name") or "") + str(tc):
            continue
        a = tc.get("arguments") or {}
        a = json.loads(a) if isinstance(a, str) else a
        inner = a.get("transactions")
        try:
            rows = json.loads(inner) if isinstance(inner, str) else (inner or [])
        except Exception:
            return None, None, "parse-fail"
        if not isinstance(rows, list):
            return None, None, "byref/str"
        net = None
        for r in rows:
            if str(r.get("transaction_id")) in (RHO_FEE, RHO_WD):
                net = str(r.get("network") or "").lower()
        dup = any(r.get("duplicate_of") for r in rows if isinstance(r, dict))
        return (net == "rho"), dup, "net=%s rows=%d" % (net, len(rows))
    return None, None, "(no call)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    cut = live_call = None
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            if TOOL in str(tc.get("name") or "") and "transaction_id" in str(tc):
                cut, live_call = i, tc
                break
        if cut is not None:
            break
    if cut is None:
        print("컷 없음(인라인 형식화 호출)")
        return
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    base = B.render(msgs[:cut], ours)
    base = base[:base.rfind("\n[user] ")] if "\n[user] " in base else base
    call_line = "[assistant calls] %s(%s)" % (TOOL, str(live_call.get("arguments"))[:1500])
    deny_t = ("Error: re-issue this call following the tool's transactions specification "
              "exactly:\n%s")
    print("073 cut=%d(첫 인라인 형식화) · n=%d · URL=%s\n" % (
        cut, n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    arms = (("A_ASIS", None), ("B_ECHO", deny_t % spec_of(TOOL)),
            ("D_OTHER", deny_t % spec_of(OTHER)))
    for label, deny in arms:
        body = base if deny is None else base + "\n" + call_line + "\n[tool] " + deny
        rho = dup = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 1400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            a, b, info = judge(r)
            rho += 1 if a else 0
            dup += 1 if b else 0
            cnt[info[:24]] += 1
            print("  [%s %02d] rho=%s dup=%s %s" % (label, i, a, b, info[:60]))
        print("%-8s rho %d/%d · dup %d/%d · %s\n" % (label, rho, n, dup, n, dict(cnt)))
    print("※ 판정(사전 고정): A_ASIS ≤2 ∧ B_ECHO ≥6 ∧ D_OTHER ≤2 → 스펙-에코 deny 출시."
          " B_ECHO ≤2 → 격리-서브 formalize 경로. D_OTHER ≥3 → 무효.")


if __name__ == "__main__":
    main()
