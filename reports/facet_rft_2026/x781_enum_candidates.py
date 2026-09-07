# -*- coding: utf-8 -*-
"""x781b — T2_WRITE_ARG_ENUM 이 켜졌다면 이 군에서 어떤 후보 명단을 냈고,
실측 인자값이 그 집합 안이었나. 엔진 정본 함수만 부른다(사본 0·[[67]])."""
import sys, json
sys.path.insert(0, r"C:\workspace\ba-frft\scripts\distill\tau2")
import t2_gate_patch as G

A2 = json.load(open(r"C:\workspace\ba-frft\scripts\distill\tau2\a2\banking_knowledge.gate.json",
                   encoding="utf-8"))
di = (A2.get("policy_ontology") or {}).get("doc_index") or {}
spec = [s for s in A2["write_arg_enum"]
        if (s.get("applies_when") or {}).get("prefix") == "open_bank_account"][0]
gmap = spec["group_map"]

# (task, account_type, 실측 account_class, gold account_class)
CASES = [
    ("055", "savings", "Gold Account", "Silver Plus Account"),
    ("056", "business_savings", "Emerald Saver Account", "Silver Plus Saver Account"),
    ("066", "savings", "Green Account (savings)", "Green Account"),
    ("066", "checking", "Evergreen Account", "Evergreen Account"),
]
for task, gval, got, gold in CASES:
    grp = gmap.get(gval)
    subs = di.get(grp) or {}
    names = G._display_slugs(subs)
    print("--- task_%s  group_arg=%r -> group=%r  후보 %d" % (task, gval, grp, len(names)))
    print("    got  =%r  in_set=%s" % (got, got in names))
    print("    gold =%r  in_set=%s" % (gold, gold in names))
    if got not in names:
        print("    => deny 문면의 candidates 에 실릴 명단:", ", ".join(sorted(names))[:600])

print("\n=== 군별 후보 명단 축자 ===")
for grp in ("savings_accounts", "checking_accounts", "business_savings_accounts"):
    subs = di.get(grp) or {}
    print(grp, "슬러그:", sorted(G._subject_keys(subs)))
    print(grp, "표시명:", sorted(G._display_slugs(subs)))
