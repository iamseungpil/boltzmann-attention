# -*- coding: utf-8 -*-
"""x766-c: [TAG] 발화 리터럴 <-> 근처 A2 키 리터럴 연관(읽기 전용·후보 제시용).

★이 스크립트는 *후보*만 낸다. 확정은 소스 정독으로 한다([[77]]).
"""
import glob
import json
import os
import re
import sys

ENG = r"C:\workspace\ba-frft\scripts\distill\tau2"
KEYS = json.load(open(os.path.join(ENG, "_ep_work", "x766", "keys_bank.json"), encoding="utf-8"))
TAGS = json.load(open(os.path.join(ENG, "_ep_work", "x766", "want_tags.json"), encoding="utf-8"))
W = 150

files = sorted(glob.glob(os.path.join(ENG, "t2_*.py"))) + [os.path.join(ENG, "gate_interpreter.py")]
res = {}
for t in TAGS:
    hits = []
    for path in files:
        if not os.path.exists(path):
            continue
        lines = open(path, encoding="utf-8").read().split("\n")
        for i, L in enumerate(lines):
            if L.lstrip().startswith("#"):
                continue
            if ("[%s]" % t) not in L:
                continue
            lo, hi = max(0, i - W), min(len(lines), i + W)
            near = {}
            for j in range(lo, hi):
                if lines[j].lstrip().startswith("#"):
                    continue
                for k in KEYS:
                    if ('"%s"' % k) in lines[j] or ("'%s'" % k) in lines[j]:
                        d = abs(j - i)
                        if k not in near or d < near[k]:
                            near[k] = d
            hits.append({"file": os.path.basename(path), "line": i + 1,
                         "near": sorted(near.items(), key=lambda kv: kv[1])[:6]})
    res[t] = hits

json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for t, hits in res.items():
    print("### %s  (%d sites)" % (t, len(hits)))
    for h in hits[:4]:
        print("   %s:%d  %s" % (h["file"], h["line"], h["near"]))
