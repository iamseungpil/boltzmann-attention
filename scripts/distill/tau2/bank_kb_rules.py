# -*- coding: utf-8 -*-
"""KB 문서(tool content)서 정책 규칙 추출: provisional credit 자격·apy 테이블·reward rate.
발명 금지([[05]])·KB 원문 근거([[08]])."""
import json, glob, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

KW = {
    "provisional": re.compile(r"provisional credit", re.I),
    "apy_table": re.compile(r"(apy|annual percentage yield).{0,60}(tier|balance|%)", re.I),
    "reward": re.compile(r"(reward|points).{0,40}(per|rate|earn|\$)", re.I),
}
seen = {k: set() for k in KW}
snips = {k: [] for k in KW}
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        for m in (s.get("messages") or []):
            if m.get("role") != "tool":
                continue
            c = str(m.get("content"))
            # doc id 추출
            docids = re.findall(r"doc_[a-z0-9_()]+", c)
            for k, rx in KW.items():
                if rx.search(c):
                    for did in docids:
                        if did not in seen[k] and len(snips[k]) < 6:
                            # 규칙 주변 스니펫
                            mt = rx.search(c)
                            i = mt.start()
                            snips[k].append((did, c[max(0, i - 60):i + 320].replace("\n", " ")))
                            seen[k].add(did)
for k in KW:
    print("\n========== %s ==========" % k)
    for did, sn in snips[k][:5]:
        print("[%s]" % did)
        print("  ", sn[:360])
