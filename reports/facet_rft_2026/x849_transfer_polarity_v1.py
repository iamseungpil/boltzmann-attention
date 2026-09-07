# -*- coding: utf-8 -*-
r"""x849 — `transfer_to_human_agents` 한 점에 몰린 레버들의 극성·조건 표 (오프라인)

x848: 이 도구가 스택 전체에서 가장 다투는 표적(192 sim). 다섯 태그가 서로 반대를 지시한다.
여기서는 그 문면을 **정규화해 전수로** 뽑는다 — 몇 종류의 서로 다른 말이 있고, 각각 몇 sim 인가.
"""
import json, gzip, os, re, glob, collections

TARGET = "transfer_to_human_agents"
ALIASES = [TARGET, "transfer to a human", "transfer the customer", "hand this conversation off",
           "request_human_agent_transfer"]
ROOTS = ["/home/woori/scratch/logs", "/home/woori/scratch/x768"]
TAG = re.compile(r"\[([A-Z][A-Z0-9 _\-']{2,30})\]")
SPLIT = re.compile(r"(?<=[.!?])\s+|\s+[-–—]\s+|;\s+|\n")
NEG = ["do not", "don't", "cannot", "can not", "must not", "never", "skip ",
       "blocked", "forbid", "prohibit", "does not apply", "not by you", "no longer", "without"]
POS = ["next:", "must ", "you must", "call ", "required", "require ", "proceed", "use "]

def polarity(c):
    c = c.lower()
    n = min([c.find(m) for m in NEG if m in c] or [10**9])
    p = min([c.find(m) for m in POS if m in c] or [10**9])
    if n == p == 10**9: return None
    return "FORBID" if n <= p else "REQUIRE"

def norm(t):
    t = re.sub(r"'[^']{1,60}'", "'X'", t)
    t = re.sub(r"\b\d+\b", "N", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:230]

per = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # tag->pol->text
sims = collections.defaultdict(lambda: collections.defaultdict(set))                  # tag->pol->sims
files = set()
for r in ROOTS:
    files |= set(glob.glob(os.path.join(r, "**", "fb_*.jsonl*"), recursive=True))
for f in sorted(files):
    if os.path.getsize(f) > 40 * 1024 * 1024: continue
    op = gzip.open if f.endswith(".gz") else open
    try:
        with op(f, "rt", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln: continue
                try: rr = json.loads(ln)
                except Exception: continue
                st = str(rr.get("simtag") or "")
                text = str(rr.get("text") or rr.get("feedback") or rr.get("msg") or "")
                if not (st and text): continue
                if not any(a in text for a in ALIASES): continue
                m = TAG.search(text[:90])
                if not m: continue
                tag = m.group(1).strip()
                for clause in SPLIT.split(text):
                    if not any(a in clause for a in ALIASES): continue
                    pol = polarity(clause)
                    if not pol: continue
                    per[tag][pol][norm(clause)] += 1
                    sims[tag][pol].add((os.path.basename(f), st))
    except Exception:
        continue

rows = sorted(per, key=lambda t: -(len(sims[t]["REQUIRE"]) + len(sims[t]["FORBID"])))
print("== `%s` 를 지목한 레버 전수 ==" % TARGET)
print("  %-20s %8s %8s   %s" % ("태그", "REQUIRE", "FORBID", "sim(합)"))
for t in rows:
    r_, f_ = len(sims[t]["REQUIRE"]), len(sims[t]["FORBID"])
    print("  %-20s %8d %8d   %d" % (t, r_, f_, r_ + f_))
print("\n== 문면 (태그별 · 극성별 · 상위 2종) ==")
for t in rows[:9]:
    print("\n-- %s --" % t)
    for pol in ("REQUIRE", "FORBID"):
        for txt, n in per[t][pol].most_common(2):
            print("   [%-7s %4d] %s" % (pol, n, txt))
