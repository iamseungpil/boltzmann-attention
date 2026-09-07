# -*- coding: utf-8 -*-
import io, json, os, sys
HERE = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, HERE); import t2_gate_patch as G
sys.stdout.reconfigure(encoding="utf-8")
a2 = G._domain_a2("banking_knowledge")
WANT = ["eligible_for_provisional_credit","issue_noticed_date","purchase_date","transaction_date",
        "contacted_merchant","police_report_filed","written_statement_provided","card_action",
        "resolution_requested","dispute_reason","card_last_4_digits","user_id","transaction_id",
        "account_id","card_id","full_name","phone","address"]
for r in G._policy_facts(a2):
    ax = str(r.get("axis") or "")
    if ax in WANT:
        for s in (r.get("sources") or []):
            q = str(s.get("quote") or "").strip()
            if q: print("[%s] subj=%s doc=%s\n    %s\n" % (ax, r.get("subject"), s.get("doc"), q[:600]))
