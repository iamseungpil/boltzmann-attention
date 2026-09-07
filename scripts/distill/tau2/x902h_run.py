# -*- coding: utf-8 -*-
"""x902h — 라이브 발화 실증: 회수분 메시지 본문에 각 레버의 문면이 실제로 들어갔나."""
import sys, json, gzip, re
sims = json.loads(gzip.open(sys.argv[1], "rt", encoding="utf-8").read())
MARK = {
    "ARG_LABEL":      "is what the records above give as",
    "HAVE-VALUE":     "[HAVE-VALUE]",
    "VALUE-ACQUIRE":  "[VALUE-ACQUIRE]",
    "REF-VERIFY":     "[REF-VERIFY]",
    "WRITE-EVIDENCE": "[WRITE-EVIDENCE]",
    "PROCEDURE(forbid_when)": "[PROCEDURE] the retention protocol",
    "PROVENANCE":     "[PROVENANCE]",
    "NL-NUM":         "[NL-NUM]",
    "DUPLICATE-WRITE": "[DUPLICATE-WRITE]",
    "READ-ALL":       "[READ-ALL]",
}
for k, mk in MARK.items():
    tot = 0
    pas = 0
    tasks = set()
    for s in sims:
        n = sum(1 for m in s["msgs"] if isinstance(m.get("content"), str) and mk in m["content"])
        if n:
            tot += n
            tasks.add(s["task"])
            if (s["reward"] or 0) >= 1.0:
                pas += n
    print("%-24s 라이브 발화 %4d · 통과 sim 발화 %3d · 태스크 %s"
          % (k, tot, pas, sorted(tasks)[:12]))
