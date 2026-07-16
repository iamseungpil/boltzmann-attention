# -*- coding: utf-8 -*-
"""nt=4 ctl/fup 전수 포렌식 ([[08]]·설계문 §4 계측).
sim별: 종료사유·실패 유형 분류(검증벽/완료날조/기타)·FOLLOWUP 발화 지점·regen 후 give emit 여부·
사용자 제출 수·over-block 후보(pass 궤적 접촉).
사용: py -3 bank_nt4_forensic.py
"""
import gzip, json, io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
T, F = "get_reward_discrepancies", "give_discoverable_user_tool"
GOLD_T = {"txn_f093f96e2001", "txn_580773a8649e", "txn_d398545ca1a2", "txn_37b5b8e67a5e"}


def names(m):
    return [tc.get("name") for tc in (m.get("tool_calls") or [])]


for tag in ("ctl", "fup"):
    with gzip.open(os.path.join(B, f"bank_{tag}_20260716_nt4.results.json.gz"), "rt", encoding="utf-8") as f:
        d = json.load(f)
    print("#" * 72)
    print(f"# {tag} (nt=4)")
    for si, s in enumerate(d["simulations"]):
        msgs = s.get("messages") or []
        rw = (s.get("reward_info") or {}).get("reward")
        term = s.get("termination_reason")
        agent_seq, submitted, fu_feedback_idx = [], set(), []
        called = set()
        for i, m in enumerate(msgs):
            role = m.get("role")
            if role == "assistant":
                for n in names(m):
                    if n:
                        agent_seq.append(n)
                        called.add(n)
            if role == "user":
                for tc in (m.get("tool_calls") or []):
                    a = tc.get("arguments") or {}
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except Exception:
                            a = {}
                    inner = a.get("arguments")
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except Exception:
                            inner = {}
                    tid = (inner or {}).get("transaction_id")
                    if tid:
                        submitted.add(tid)
            c = m.get("content")
            if role == "user" and isinstance(c, str) and "[FOLLOW-UP]" in c:
                fu_feedback_idx.append(i)
        # 실패 유형 분류 (구조 이벤트)
        if rw == 1.0:
            kind = "PASS"
        elif T not in called:
            kind = "①검증벽/미도달" if "get_credit_card_transactions_by_user" not in called else "①b 거래읽고 T 미호출"
        elif F not in called:
            kind = "②완료날조형(T후 give 0)"
        else:
            kind = f"③give까지 갔으나 제출 {len(submitted)}/4"
        print(f"\n--- sim{si} reward={rw} 종료={term}  ★유형={kind}")
        print(f"  agent({len(agent_seq)}): {agent_seq}")
        print(f"  제출 {len(submitted)}/4 | 누락 {sorted(GOLD_T - submitted)}")
        if fu_feedback_idx:
            i0 = fu_feedback_idx[0]
            after = [n for m in msgs[i0:i0 + 3] if m.get("role") == "assistant" for n in names(m)]
            print(f"  ★FOLLOWUP 피드백 @msg{fu_feedback_idx} → regen 직후 emit: {after or '(텍스트)'}")
