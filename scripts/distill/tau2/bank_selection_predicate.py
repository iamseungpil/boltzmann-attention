# -*- coding: utf-8 -*-
"""bank_selection_predicate.py — 게이트1: ABox 선택술어 정확도 (❹ B-tier·2026-07-16).

❹: write 타깃의 84.9%가 discovery-surface+선택술어 필요. dispute는 C78 reference_filter 81.9% 검증됨.
미검증 = card/account-ops(close/freeze/unfreeze/order card·apply credit·interest). 질문:
  discovery된 후보 엔티티 중 gold 타깃이 gold-free 술어로 유일 선택되나?
분류(per write family·per sim):
  UNIQUE     : 후보 엔티티 1개=선택 자명(trivial)
  CUST-MENTION: 후보 다수·gold가 user 발화에 명시(고객 지목)
  LINKABLE   : 후보 다수·gold가 disputed txn/문제 엔티티에 link(cascade 술어)
  AMBIGUOUS  : 후보 다수·gold-free 신호 없음(선택술어 부재=경계/ASK)
로컬 무료·per-family. 엔티티타입=family가 쓰는 id 필드(ABox action_tools 관례)."""
import json, glob, re, sys, io
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
fam = lambda n: re.sub(r"_\d+$", "", str(n))
READ = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
PROC = re.compile(r"(^log_|_verification$|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_)", re.I)

# family → 선택 엔티티 필드 (도메인지식·측정용)
def ent_field(tf):
    if "dispute" in tf or "transaction" in tf:
        return "transaction_id"
    if "card" in tf:
        return "card_id"
    if "account" in tf or "interest" in tf or "credit" in tf:
        return "account_id"
    return None

# 엔티티타입별 후보 정규식 (discovery서 등장한 그 타입 id)
CAND_RX = {
    "transaction_id": re.compile(r"\b([bc]?txn_[0-9a-f]+)\b", re.I),
    "card_id": re.compile(r"\b((?:cc|dbc)_[0-9a-z_]+)\b", re.I),
    "account_id": re.compile(r"\b((?:chk|sav|cc)_[0-9a-z_]+)\b", re.I),
}

by_fam = defaultdict(Counter)
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        if ri.get("reward") in (None, 1.0): continue
        if tuple(ri.get("reward_basis") or []) != ("DB",): continue
        if str(s.get("termination_reason")) == "too_many_errors": continue
        utext = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") == "user").lower()
        disc_text = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") == "tool").lower()
        # gold disputed txn (cascade link 판정용)
        disputed = set()
        for ac in (ri.get("action_checks") or []):
            a = ac.get("action") or {}; outer = Nd(a.get("arguments"))
            if "dispute" in fam(outer.get("agent_tool_name", "")):
                t = Nd(outer.get("arguments")).get("transaction_id")
                if t: disputed.add(str(t).lower())
        # gold write 타깃 per family
        goldw = defaultdict(set)
        for ac in (ri.get("action_checks") or []):
            a = ac.get("action") or {}; outer = Nd(a.get("arguments")); atn = outer.get("agent_tool_name", "")
            if not atn or "arguments" not in outer: continue
            tf = fam(atn)
            if READ.match(tf) or PROC.search(tf): continue
            ef = ent_field(tf)
            if not ef: continue
            eid = Nd(outer.get("arguments")).get(ef)
            if eid: goldw[tf].add(str(eid).lower())
        # scope-all 신호: 고객이 "전부/all/both/N개" 범위로 지시(→act-on-all·선택술어 불필요·[[08]] per-case 근거)
        scope_all = bool(re.search(r"\b(all|both|every|each|all of them|all three|all four|all my|both of|three of|freeze them|close all)\b", utext))
        for tf, gold in goldw.items():
            ef = ent_field(tf); rx = CAND_RX.get(ef)
            if not rx: continue
            cands = set(m.lower() for m in rx.findall(disc_text))
            # gold이 후보의 다수(≥절반) 커버 = act-on-all 패턴
            gold_in_c = gold & cands
            act_all = scope_all and len(gold_in_c) >= 1 and len(gold) >= max(1, len(cands) // 2)
            for g in gold:
                if g not in cands:
                    by_fam[tf]["(gold∉discovery)"] += 1; continue
                if len(cands) <= 1:
                    by_fam[tf]["UNIQUE(후보1·자명)"] += 1
                elif g in utext:
                    by_fam[tf]["CUST-MENTION(고객 지목·raw)"] += 1
                elif act_all:
                    by_fam[tf]["SCOPE-ALL(전부 지시)"] += 1
                elif ef == "card_id" and any(t in disc_text for t in disputed):
                    by_fam[tf]["LINKABLE(disputed txn cascade)"] += 1
                else:
                    by_fam[tf]["AMBIGUOUS(선택술어 부재)"] += 1

print("=== 게이트1: card/account-ops 선택술어 정확도 (후보 다수 중 gold 유일선택 가능성) ===")
agg = Counter()
for tf, c in sorted(by_fam.items(), key=lambda x: -sum(x[1].values())):
    tot = sum(c.values())
    if tot < 30: continue
    selk = ["UNIQUE(후보1·자명)", "CUST-MENTION(고객 지목·raw)", "SCOPE-ALL(전부 지시)", "LINKABLE(disputed txn cascade)"]
    sel = sum(c[k] for k in selk)
    print("\n  %s (n=%d):" % (tf, tot))
    for k, v in c.most_common():
        print("     %-32s %5d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
    print("     ★선택가능(UNIQUE+MENTION+SCOPE+LINK) = %.1f%% · AMBIGUOUS = %.1f%%"
          % (100 * sel / max(tot, 1), 100 * c["AMBIGUOUS(선택술어 부재)"] / max(tot, 1)))
    for k, v in c.items(): agg[k] += v
tot = sum(agg.values())
sel = sum(agg[k] for k in ["UNIQUE(후보1·자명)", "CUST-MENTION(고객 지목·raw)", "SCOPE-ALL(전부 지시)", "LINKABLE(disputed txn cascade)"])
print("\n=== 전 card/account-ops 종합 (n=%d) ===" % tot)
for k, v in agg.most_common():
    print("  %-32s %6d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
print("\n  ★종합 gold-free 선택가능 = %.1f%% · AMBIGUOUS(경계) = %.1f%% · gold∉discovery = %.1f%%"
      % (100 * sel / max(tot, 1), 100 * agg["AMBIGUOUS(선택술어 부재)"] / max(tot, 1),
         100 * agg["(gold∉discovery)"] / max(tot, 1)))
print("  판정: 선택가능 높으면 Track A 실현(선택술어 존재)·AMBIGUOUS 높으면 술어부재=ASK/경계.")
