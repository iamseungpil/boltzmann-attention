# -*- coding: utf-8 -*-
"""x851 — LB7 배달 재료 검산 (GPU 0). 제목이 실제로 붙는가."""
import json, os
A2 = "/home/woori/scratch/repo_rep3/scripts/distill/tau2/a2/banking_knowledge.gate.json"
DD = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
po = (json.load(open(A2, encoding="utf-8")).get("policy_ontology") or {})
idx = po.get("doc_index") or {}
degen = [g for g, subs in idx.items()
         if not [k for k in subs if k != "_general_"]]
print("퇴화 축(계열이 _general_ 뿐): %s" % degen)
for g in degen:
    ids = []
    for s, dl in (idx.get(g) or {}).items():
        ids += [d for d in (dl or ()) if d]
    lines, titled = [], 0
    for d in ids:
        t = ""
        try:
            with open(os.path.join(DD, d + ".json"), encoding="utf-8") as fh:
                t = str(json.load(fh).get("title") or "")
        except Exception:
            t = ""
        if t: titled += 1
        lines.append(("%s  %s" % (d, t)) if t else d)
    blob = "\n".join(lines)
    print("\n== %s ==" % g)
    print("  문서 %d · 제목 붙은 것 %d · 배달 크기 %d자" % (len(ids), titled, len(blob)))
    print("  표본:")
    for l in lines[:4]: print("    " + l[:110])
    tgt = "doc_bank_accounts_bank_accounts_(general)_042"
    print("  x829 표적 포함? %s" % ("예" if any(tgt in l for l in lines) else "아니오"))
