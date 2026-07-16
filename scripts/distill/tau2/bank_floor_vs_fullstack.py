# -*- coding: utf-8 -*-
"""bank_floor_vs_fullstack.py — 43 floor 태스크별 base floor(32B) vs outer/inner full-stack 대조 (2026-07-17).

사용자: "outer inner full stack으로 바꾼거 하고 이전에 floor 측정한 43개 태스크별로 비교."
floor(nt=1·태스크당 1 sim)마다:
  - floor 실측 pass(reward=1) / fail.
  - fail이면 미충족 gold-write의 원인(A1 REACH·A2 COVERAGE·B1 compute·B2 ref·B3 F3·B4 judg·B5 gather).
  - full-stack 닫힘 tier: HARD(A1선택술어+B1compute+B2ref=결정론) 만이면 CLOSE-hard·
    +SOFT(A2 coverage)·+B3(F3스킬)·잔여(B4/B5=경계/user).
per-task 표 + 요약(full-stack이 floor-fail 중 몇 개를 어느 tier로 닫나).
"""
import json, gzip, re, sys, io, os, argparse
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

fam = lambda n: re.sub(r"_\d+$", "", str(n or ""))
_READ = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
_PROC = re.compile(r"(^log_|_verification$|^kb_|^shell$|discoverable|transfer_to_human|give_|unlock_)", re.I)
isw = lambda n: bool(fam(n)) and not _READ.match(fam(n)) and not _PROC.search(fam(n))
_TXN = re.compile(r"\b((?:txn|btxn|chk|dbc|ccord|dcord|clsr|cli|card|acct|ca)_[0-9a-fA-F]{6,})\b")

def nd(x):
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception: return {}
    return x if isinstance(x, dict) else {}

def load_abox():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "banking_knowledge.gate.json")
    return json.load(open(p, encoding="utf-8"))

def entity_of(args):
    for k in ("transaction_id", "card_id", "account_id", "credit_card_account_id", "checking_account_id"):
        if args.get(k): return str(args[k])
    return ""

def field_op(field, fo):
    if field in set(fo.get("judgment", [])): return "B4_judg"
    if field in set(fo.get("compute", [])): return "B1_compute"
    if field in set(fo.get("id_ref", [])) or field.endswith("_id"): return "B2_ref"
    if field in set(fo.get("enum", [])): return "B3_F3"
    return "B5_gather"

# tier: 결정론 HARD 닫힘 = {A1(선택술어 조건), B1, B2} · SOFT = A2 · 스킬 = B3 · 경계 = B4,B5
HARD = {"A1_REACH", "B1_compute", "B2_ref"}
SOFT = {"A2_COVERAGE"}
SKILL = {"B3_F3"}
BOUND = {"B4_judg", "B5_gather"}

def sim_causes(s, fo):
    ri = s.get("reward_info") or {}
    called = set(); subs = {}; surfaced = set()
    for m in (s.get("messages") or []):
        if m.get("role") in ("tool", "user"):
            surfaced |= set(_TXN.findall(str(m.get("content"))))
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            if nm == "call_discoverable_agent_tool":
                outer = nd(tc.get("arguments")); tfam = fam(outer.get("agent_tool_name", ""))
                called.add(tfam); ia = nd(outer.get("arguments")); e = entity_of(ia)
                if e: subs[(tfam, e)] = ia
            elif nm: called.add(fam(nm))
    causes = []
    for ac in (ri.get("action_checks") or []):
        a = ac.get("action") or {}; outer = nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "") or a.get("name", "")
        if not isw(atn): continue
        met = ac.get("action_reward"); met = met if met is not None else (1.0 if ac.get("action_match") else 0.0)
        if float(met) >= 1.0: continue
        tf = fam(atn); ga = nd(outer.get("arguments")); ent = entity_of(ga)
        if tf not in called:
            causes.append("A2_COVERAGE" if (ent and ent in surfaced) else "A1_REACH")
        else:
            wrong = [k for k, gv in ga.items() if k != "transaction_id" and str(subs.get((tf, ent), {}).get(k)) != str(gv)]
            ops = [field_op(k, fo) for k in wrong] or ["B5_gather"]
            rank = {"B1_compute": 0, "B2_ref": 1, "B3_F3": 2, "B4_judg": 3, "B5_gather": 4}
            causes.append(min(ops, key=lambda o: rank[o]))
    return causes

def tier_of(causes):
    cs = set(causes)
    if not cs: return "blind(pure-DB)"
    if cs <= HARD: return "CLOSE-hard(결정론)"
    if cs <= (HARD | SOFT): return "+soft(coverage)"
    if cs <= (HARD | SOFT | SKILL): return "+B3(F3스킬)"
    return "잔여(경계·user)"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); a = ap.parse_args()
    fo = load_abox().get("field_ops") or {}
    op = gzip.open if a.results.endswith(".gz") else open
    d = json.load(op(a.results, "rt", encoding="utf-8"))
    rows = []
    for s in d.get("simulations", []):
        t = s.get("task_id"); ri = s.get("reward_info") or {}
        rew = ri.get("reward")
        if rew == 1.0:
            rows.append((t, "PASS", "-", [])); continue
        causes = sim_causes(s, fo)
        rows.append((t, "FAIL", tier_of(causes), causes))
    rows.sort(key=lambda r: r[0])
    print("=== 43 floor 태스크별: base floor(32B) vs full-stack 닫힘 ===")
    print("%-10s %-5s %-20s %s" % ("task", "floor", "full-stack tier", "미충족 원인"))
    tier_ct = Counter(); fail_n = 0
    for t, res, tier, causes in rows:
        cc = dict(Counter(causes)) if causes else ""
        print("%-10s %-5s %-20s %s" % (t, res, tier if res == "FAIL" else "", cc))
        if res == "FAIL":
            fail_n += 1; tier_ct[tier] += 1
    print("\n=== 요약 (floor-fail %d개의 full-stack 닫힘 tier) ===" % fail_n)
    for k, v in tier_ct.most_common():
        print("  %-22s %d (%.0f%%)" % (k, v, 100*v/max(fail_n, 1)))
    passn = sum(1 for r in rows if r[1] == "PASS")
    print("\nfloor PASS: %d/%d = %.1f%%" % (passn, len(rows), 100*passn/max(len(rows), 1)))
    hard = tier_ct.get("CLOSE-hard(결정론)", 0)
    print("full-stack HARD로 닫을 floor-fail: %d개 → 상한 pass = (%d+%d)/%d = %.1f%% (결정론만)"
          % (hard, passn, hard, len(rows), 100*(passn+hard)/max(len(rows), 1)))

if __name__ == "__main__":
    main()
