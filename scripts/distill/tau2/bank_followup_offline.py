# -*- coding: utf-8 -*-
"""T2_FOLLOWUP_REQUIRED 오프라인 검증 — 최종 10 sim replay (무료·[[09]]·라이브 前).
발화 조건(엔진과 동일): 사임 ∧ get_reward_discrepancies 호출됨 ∧ give_discoverable_user_tool 미호출 ∧ 1/sim.
기대: 실패 4 sim 중 give=0인 3개서 발화 / pass 6 sim 발화 수 = over-block 게이지.
사용: py -3 bank_followup_offline.py
"""
import gzip, json, io, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
T, F = "get_reward_discrepancies", "give_discoverable_user_tool"

for tag in ("ctl", "dreq"):
    with gzip.open(os.path.join(B, f"bank_{tag}_20260716_2230.results.json.gz"), "rt", encoding="utf-8") as f:
        d = json.load(f)
    print("#" * 64)
    for si, s in enumerate(d["simulations"]):
        msgs = s.get("messages") or []
        called = set()
        fired_at = []
        fired = False
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            tcs = [tc.get("name") for tc in (m.get("tool_calls") or [])]
            resign = (not tcs) and isinstance(m.get("content"), str) and m["content"].strip()
            if resign and not fired and T in called and F not in called:
                fired_at.append(i)
                fired = True
            called.update(n for n in tcs if n)
        rw = (s.get("reward_info") or {}).get("reward")
        print(f"{tag} sim{si} reward={rw}  ★발화 {len(fired_at)} @msg{fired_at}"
              f"  (T호출={'Y' if T in called else 'N'} F호출={'Y' if F in called else 'N'})")
