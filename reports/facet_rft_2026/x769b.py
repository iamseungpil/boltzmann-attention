# -*- coding: utf-8 -*-
import io, json, os, sys
HERE = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, HERE)
import t2_gate_patch as G
sys.stdout.reconfigure(encoding="utf-8")
a2 = G._domain_a2("banking_knowledge")
surf = json.load(io.open(os.path.join(HERE,"a2","env_surface.json"),encoding="utf-8"))["banking_knowledge"]["tools"]
axes = {}
for r in G._policy_facts(a2):
    axes.setdefault(str(r.get("axis") or ""), 0)
    axes[str(r.get("axis") or "")] += len(r.get("sources") or [])
TARGETS = ["customer_max_liability_amount","eligible_for_provisional_credit","provisional_credit_given",
           "provisional_credit_issued","provisional_credit_amount","issue_noticed_date","discovery_date",
           "disputed_amount","dispute_category","transaction_type","card_design","current_holdings",
           "status","rewards_earned","credit_limit"]
for t in ["file_credit_card_transaction_dispute_4829","file_debit_card_transaction_dispute_6281","submit_cash_back_dispute_0589"]:
    print("=== %s args=%r" % (t, sorted(surf[t].get("args") or [])))
print()
print("=== 표적 필드가 ① 그 도구의 선언 인자인가 ② A3 축인가 ===")
dargs = set(surf["file_debit_card_transaction_dispute_6281"].get("args") or [])
cargs = set(surf["file_credit_card_transaction_dispute_4829"].get("args") or [])
for f in TARGETS:
    print("  %-34s debit_arg=%-5s credit_arg=%-5s A3축=%-5s (A3 인용 %d)"
          % (f, f in dargs, f in cargs, f in axes, axes.get(f,0)))
print()
print("=== A3 축 중 'liab'/'provisional'/'date' 를 담은 이름 ===")
for k in sorted(axes):
    kl=k.lower()
    if "liab" in kl or "provisional" in kl or "date" in kl or "noticed" in kl:
        print("   %-46s 인용 %d" % (k, axes[k]))
