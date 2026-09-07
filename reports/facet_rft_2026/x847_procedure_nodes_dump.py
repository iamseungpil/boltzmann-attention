import json
A2 = "/home/woori/scratch/repo_rep1/scripts/distill/tau2/a2/banking_knowledge.gate.json"
d = json.load(open(A2, encoding="utf-8"))
for p in (d.get("procedures") or []):
    if p.get("id") != "credit_card_closure_retention": continue
    print("키:", sorted(p))
    for k, v in p.items():
        if k.startswith("_"): continue
        print("\n-- %s --" % k)
        print(json.dumps(v, ensure_ascii=False)[:1200])
