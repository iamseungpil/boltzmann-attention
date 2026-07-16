# -*- coding: utf-8 -*-
"""nt=4 fup 실패 2건 per-step 전수 포렌식 ([[08]] (4) 원문 정독).
매 스텝: 발화자·도구호출(인자 전문)·tool 반환(전문)·텍스트(요지) — 원인 확정용.
사용: py -3 bank_nt4_fail_perstep.py [sim번호들…]
"""
import gzip, json, io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

with gzip.open(os.path.join(B, "bank_fup_20260716_nt4.results.json.gz"), "rt", encoding="utf-8") as f:
    d = json.load(f)

sims = [int(x) for x in sys.argv[1:]] or [0, 1]
for si in sims:
    s = d["simulations"][si]
    print("=" * 74)
    print(f"# fup(nt=4) sim{si}  reward={(s.get('reward_info') or {}).get('reward')} "
          f"종료={s.get('termination_reason')}")
    for i, m in enumerate(s.get("messages") or []):
        role = m.get("role")
        c = m.get("content")
        for tc in (m.get("tool_calls") or []):
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            print(f"[{i:2d}] {role:9s} CALL {tc.get('name')}")
            print(f"       args = {json.dumps(args, ensure_ascii=False)[:400]}")
        if isinstance(c, str) and c.strip():
            tag = role
            if role == "tool":
                tag = "TOOL←"
            print(f"[{i:2d}] {tag:9s}: {c[:420]}".replace("\n", " "))
    print()
