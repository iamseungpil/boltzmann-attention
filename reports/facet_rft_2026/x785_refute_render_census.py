# -*- coding: utf-8 -*-
"""x785 — 반증 프로브 ④: 로컬 fb 사이드카 전수에서 «이름이 빈 문면»([[84]])과
«WORK-INCOMPLETE 계수»의 태스크별 실발화를 센다. 읽기 전용.
"""
import gzip, glob, json, os, re, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIRS = {}
for ln in open(os.path.join(ROOT, "x769_pairs.txt"), encoding="utf-8"):
    p = ln.split()
    if len(p) >= 3:
        PAIRS.setdefault(p[0], set()).add(p[1])

NONE_PAT = re.compile(r"None: None|: None\b|None \(tool")
WI_PAT = re.compile(r"\[WORK-INCOMPLETE\][^\n]*?(\d+) item\(s\)[^\n]*?and (\d+) you have")

none_tasks = collections.defaultdict(set)
wi_rows = []
opscope = collections.defaultdict(set)
files = 0
for p in sorted(glob.glob(os.path.join(ROOT, "sim_results", "fb_*.jsonl.gz"))):
    tag = os.path.basename(p)[3:-9]
    files += 1
    for ln in gzip.open(p, "rt", encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        txt = o.get("text") or ""
        if not txt:
            continue
        st = str(o.get("simtag") or "")
        tid = st.split("#")[0]
        if NONE_PAT.search(txt):
            none_tasks[tid].add((tag, o.get("turn")))
        m = WI_PAT.search(txt.replace("\n", " "))
        if m:
            wi_rows.append((tid, tag, o.get("turn"), m.group(1), m.group(2)))
        if "[OPERATOR-SCOPE]" in txt:
            opscope[tid].add((tag, o.get("turn")))

print("fb files scanned:", files)
FAIL = set()
for tag, ts in PAIRS.items():
    FAIL |= ts
print("\n=== [[84]] 이름-빈 문면 («None») 실발화 태스크 ===")
print("총 태스크:", len(none_tasks))
inf = sorted(t for t in none_tasks if t in FAIL)
print("실패 46 명단과 교집합 (%d):" % len(inf))
for t in inf:
    print("   %-9s turns=%s" % (t, sorted(none_tasks[t], key=lambda x: (x[0], x[1]))[:6]))
print("실패 명단 밖 (%d): %s" % (len([t for t in none_tasks if t not in FAIL]),
                                sorted(t for t in none_tasks if t not in FAIL)))

print("\n=== [WORK-INCOMPLETE] 계수 실발화 ===")
for r in sorted(set(wi_rows)):
    print("   task=%-9s tag=%-38s turn=%-4s items=%s acted=%s" % r)

print("\n=== [OPERATOR-SCOPE] 실발화 태스크 ===")
for t in sorted(opscope):
    print("   %-9s %s" % (t, sorted(opscope[t], key=lambda x: (x[0], x[1]))[:8]))
