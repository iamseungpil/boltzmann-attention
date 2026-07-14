# -*- coding: utf-8 -*-
"""compute 키스톤 사정권 실측 (2026-07-14·§14.8·forensic 종착).
id-correct dispute 쌍(agent가 올바른 transaction_id로 제출한 dispute)의 필드를 gold와 대조 →
compute-필드(liability·provisional_credit·partial_refund·card_action) 오답률 + compute-closability
(오답필드가 전부 compute면 compute 엔진이 fail→pass). = 키스톤 compute의 진짜 slice.

전제(§14.7 확정): banking dispute 실패의 decidable slice = 인자의 computed 필드(§7·frontier-irreducible).
reference-filter(⋈4%)·act-gate(refuted) 아님. 이 스크립트가 그 slice를 정량."""
import json, glob, re
from collections import Counter
import bank_filter_repro as B

fam = lambda nm: re.sub(r"_\d+$", "", str(nm))
COMPUTE = {"customer_max_liability_amount", "provisional_credit_eligible",
           "eligible_for_provisional_credit", "partial_refund_amount", "card_action"}


def norm(v):
    s = str(v).strip().lower(); m = re.sub(r"[$,]", "", s)
    try:
        return round(float(m), 2)
    except Exception:
        pass
    if s in ("true", "yes", "y"):
        return True
    if s in ("false", "no", "n"):
        return False
    return s


def hard_set(data):
    per = {}
    for d in data.values():
        for s in d["simulations"]:
            r = (s.get("reward_info") or {}).get("reward")
            if r is None:
                continue
            t = str(s["task_id"]); per.setdefault(t, [0, 0]); per[t][1] += 1
            if r == 1.0:
                per[t][0] += 1
    return {t for t, p in per.items() if p[1] >= 10 and p[0] / p[1] <= 0.10}


def main():
    data = {}
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        try:
            data[f] = json.load(open(f, encoding="utf-8"))
        except Exception:
            pass
    hard = hard_set(data)
    present = Counter(); mism = Counter(); cat = Counter(); pairs = 0
    for d in data.values():
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            acalls = {}
            for m in (s.get("messages") or []):
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") == "call_discoverable_agent_tool" \
                            and "transaction_dispute" in fam(B.nd(tc.get("arguments")).get("agent_tool_name")):
                        a = B.nd(B.nd(tc.get("arguments")).get("arguments"))
                        tid = str(a.get("transaction_id") or "")
                        if tid:
                            acalls.setdefault(tid, a)
            for ac in (ri.get("action_checks") or []):
                a = ac.get("action") or {}
                if "transaction_dispute" not in fam(B.nd(a.get("arguments")).get("agent_tool_name", "")):
                    continue
                ga = B.nd(B.nd(a.get("arguments")).get("arguments")); tid = str(ga.get("transaction_id") or "")
                if not tid or tid not in acalls:
                    continue                                   # id-correct 쌍만
                aa = acalls[tid]; pairs += 1
                wrong = set()
                for k, gv in ga.items():
                    if k == "transaction_id":
                        continue
                    present[k] += 1
                    if norm(aa.get(k)) != norm(gv):
                        mism[k] += 1; wrong.add(k)
                if not wrong:
                    cat["pass(전필드정확)"] += 1
                elif wrong <= COMPUTE:
                    cat["compute만(엔진이 닫음)"] += 1
                elif wrong & COMPUTE:
                    cat["혼합(compute+other)"] += 1
                else:
                    cat["noncompute만(compute무관)"] += 1

    print("=== id-correct dispute 쌍 %d · 필드 오답률(★=compute) ===" % pairs)
    for k, _ in present.most_common():
        star = " ★" if k in COMPUTE else ""
        print("  %-32s %4d/%4d  %5.1f%%%s" % (k, mism[k], present[k], 100 * mism[k] / max(present[k], 1), star))
    print("\n=== compute-closability 분해 ===")
    for k in ["pass(전필드정확)", "compute만(엔진이 닫음)", "혼합(compute+other)", "noncompute만(compute무관)"]:
        print("  %-24s %4d (%.1f%%)" % (k, cat[k], 100 * cat[k] / max(pairs, 1)))
    slice_ = cat["compute만(엔진이 닫음)"]
    print("\n★compute slice = %d (id-correct의 %.1f%% · 전체 gold-dispute 3904의 %.1f%%)"
          % (slice_, 100 * slice_ / max(pairs, 1), 100 * slice_ / 3904))
    print("  주도=liability(51%% 오답·lookup_table)·provisional_credit(bool_expr)·card_action. §8-1 gold-blind 재현율이 실효 상한.")


if __name__ == "__main__":
    main()
