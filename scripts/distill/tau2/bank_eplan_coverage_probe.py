# -*- coding: utf-8 -*-
"""bank_eplan_coverage_probe.py — 지배 레버(coverage/reach) 오프라인 실증 (C101 (b)·2026-07-17).

목적: t2_eplan_patch의 *라이브 스캐폴드 메커니즘*(PlanLedger + coverage_gap + (a)파서)이
banking transaction-level under-action(surface된 거래 다 dispute 안 함)을 잡는지 실 궤적서 실증.
[[08]] 순서: 공유 ABox/엔진 바꾸기 前 메커니즘부터 무료 실증.

- listed   = 거래목록 출력서 (a)파서로 추출한 transaction_id (엔진 _extract_entity_ids·entity_key=transaction_id)
- executed = dispute write(call_discoverable_agent_tool·agent_tool_name=file_*_dispute)의 nested transaction_id
- required = gold dispute transaction_id (action_checks)
- coverage_gap = required − executed → under-action. 그중 surfaced(∈listed)=coverage(정보보유·리마인더표적) vs 미surface=reach.

사용: py bank_eplan_coverage_probe.py
"""
import json, glob, re, sys, io, os
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_eplan_patch as E

fam = lambda n: re.sub(r"_\d+$", "", str(n))
def nd(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return x if isinstance(x, dict) else {}

DISPUTE_WRITE = ("file_credit_card_transaction_dispute", "file_debit_card_transaction_dispute",
                 "submit_cash_back_dispute")
DISPATCH = "call_discoverable_agent_tool"
SPEC = {"entity_key": "transaction_id"}   # transaction-level coverage (C101 (b): account_id 아님)

def probe():
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))
    n_sim = 0; n_gap = 0
    gap_tids = 0; gap_surfaced = 0; gap_unsurfaced = 0
    listed_empty_sims = 0
    per_reason = Counter()
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue                                  # 실패 sim만
            if tuple(ri.get("reward_basis") or []) != ("DB",):
                continue
            msgs = s.get("messages") or []
            res = {m.get("id"): m for m in msgs if m.get("role") == "tool" and m.get("id")}
            led = E.PlanLedger(SPEC)
            # required (gold disputes)
            gold = {}
            for ac in (ri.get("action_checks") or []):
                a = ac.get("action") or {}
                atn = nd(a.get("arguments")).get("agent_tool_name", "")
                if fam(atn) in DISPUTE_WRITE:
                    ga = nd(nd(a.get("arguments")).get("arguments"))
                    tid = str(ga.get("transaction_id") or "")
                    if tid:
                        gold[tid] = ga
            if not gold:
                continue
            # listed (거래목록 출력) + executed (dispute write nested tid) — 라이브 ledger 경로
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    nm = tc.get("name") or ""
                    args = nd(tc.get("arguments"))
                    tm = res.get(tc.get("id"))
                    out = str(tm.get("content")) if (tm and not tm.get("error")) else ""
                    # (a) 파서: 거래목록 출력서 listed 갱신
                    if out and "transaction_id" in out:
                        led.listed |= E._extract_entity_ids(out, "transaction_id")
                    # dispute write → executed
                    if nm == DISPATCH and fam(args.get("agent_tool_name", "")) in DISPUTE_WRITE:
                        inner = nd(args.get("arguments"))
                        tid = str(inner.get("transaction_id") or "")
                        ok = tm is not None and not tm.get("error")
                        if tid and ok:
                            led.note_write("dispute", tid)
            # seed required as replan (coverage_gap = replan 기준)
            led.set_replan([{"intent_class": "dispute", "entity": tid} for tid in gold])
            gaps = E.coverage_gap(led)
            n_sim += 1
            if not led.listed:
                listed_empty_sims += 1
            if gaps:
                n_gap += 1
                for g in gaps:
                    gap_tids += 1
                    if g["entity"] in led.listed:
                        gap_surfaced += 1; per_reason["surfaced_not_disputed(COVERAGE)"] += 1
                    else:
                        gap_unsurfaced += 1; per_reason["not_surfaced(REACH)"] += 1
    print("=== 지배 레버 coverage-probe (t2_eplan_patch ledger·실 실패 sim·DB-basis) ===")
    print("  실패 sim(gold dispute有): %d" % n_sim)
    print("  listed 채워진 sim: %d (%.0f%%)  ← (a)파서 작동 실증 (기존 ∅였음)" %
          (n_sim - listed_empty_sims, 100 * (n_sim - listed_empty_sims) / max(n_sim, 1)))
    print("  coverage_gap>0 sim (미제출 dispute 有): %d (%.0f%%)  ← under-action 검출" %
          (n_gap, 100 * n_gap / max(n_sim, 1)))
    print("  gap transaction 총 %d:" % gap_tids)
    print("    surfaced-not-disputed (COVERAGE·정보보유→리마인더 표적): %d (%.0f%%)" %
          (gap_surfaced, 100 * gap_surfaced / max(gap_tids, 1)))
    print("    not-surfaced         (REACH·발견실패→FIND-enumerate 표적): %d (%.0f%%)" %
          (gap_unsurfaced, 100 * gap_unsurfaced / max(gap_tids, 1)))
    print("  판정: coverage_gap이 under-action 검출 + surfaced 분율=리마인더가 잡는 몫(C94 COVERAGE/FIND와 대조).")

if __name__ == "__main__":
    probe()
