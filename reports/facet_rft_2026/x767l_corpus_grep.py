# -*- coding: utf-8 -*-
import json, io, sys, re
d = json.load(io.open("reports/facet_rft_2026/x767_corpus_snapshot.json", encoding="utf-8"))
docs = d["docs"]; titles = d["titles"]
pats = sys.argv[1:]
for pat in pats:
    hits = [(k, v) for k, v in sorted(docs.items()) if pat in v]
    print("### PAT %r -> %d docs" % (pat, len(hits)))
    for k, v in hits:
        for line in v.splitlines():
            if pat in line:
                print("  [%s] %s" % (k, line.strip()))
    print()
