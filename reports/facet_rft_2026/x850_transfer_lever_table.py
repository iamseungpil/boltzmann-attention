# -*- coding: utf-8 -*-
r"""x850 — `transfer_to_human_agents` 레버 표 (극성을 **구조**에서 뽑는다)

x849 의 실패: 표면 부정어로 극성을 정하면 뒤집힌다. `[PROTOCOL] You are about to use X, but
nothing you retrieved defines it` 은 **막는 말**인데 "use" 때문에 REQUIRE 로 읽혔고,
`[FOLLOW-UP] ... was never called, so the customer has NOT actually been transferred` 는
**밀는 말**인데 부정어 때문에 FORBID 로 읽혔다. 부정어는 *상태 서술*이지 명령이 아니다.

⇒ 극성은 사이드카 `kind` 에서 온다 — 그건 **무슨 일이 일어났는가**의 구조적 기록이다:
   tool-deny  = 그 호출이 실제로 거부됐다      (HARD BLOCK)
   route      = 다른 곳으로 돌렸다             (REDIRECT)
   reminder-* = 문면을 주입했다                (SOFT PUSH)
"""
import json, gzip, os, re, glob, collections

TARGET = "transfer_to_human_agents"
ROOTS = ["/home/woori/scratch/logs", "/home/woori/scratch/x768"]
TAG = re.compile(r"\[([A-Z][A-Z0-9 _\-']{2,30})\]")
KINDMAP = {"tool-deny": "BLOCK", "route": "REDIRECT",
           "reminder-user": "PUSH", "reminder-assistant": "PUSH", "prompt": "PUSH"}

def norm(t):
    t = re.sub(r"'[^']{1,60}'", "'X'", t)
    t = re.sub(r"\b\d+\b", "N", t)
    return re.sub(r"\s+", " ", t).strip()[:200]

cell = collections.defaultdict(lambda: collections.defaultdict(set))     # tag -> pol -> sims
txts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
kinds = collections.Counter()
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
                st = str(rr.get("simtag") or ""); text = str(rr.get("text") or rr.get("feedback") or "")
                if not (st and text and TARGET in text): continue
                m = TAG.search(text[:90])
                if not m: continue
                tag = m.group(1).strip(); kind = str(rr.get("kind") or "")
                kinds[kind] += 1
                pol = KINDMAP.get(kind, "OTHER:" + kind)
                cell[tag][pol].add((os.path.basename(f), st))
                txts[tag][pol][norm(text)] += 1
    except Exception:
        continue

POLS = ["BLOCK", "REDIRECT", "PUSH"]
rows = sorted(cell, key=lambda t: -sum(len(cell[t][p]) for p in cell[t]))
print("kind 분포: %s\n" % dict(kinds.most_common()))
print("== `%s` 를 지목한 레버 × 실제 효과 (sim 수) ==" % TARGET)
print("  %-24s %7s %9s %6s  %s" % ("태그", "BLOCK", "REDIRECT", "PUSH", "그 외"))
for t in rows:
    other = sum(len(v) for k, v in cell[t].items() if k not in POLS)
    print("  %-24s %7d %9d %6d  %d"
          % (t, len(cell[t]["BLOCK"]), len(cell[t]["REDIRECT"]), len(cell[t]["PUSH"]), other))
print("\n== 대표 문면 (태그 × 효과) ==")
for t in rows[:8]:
    for p in POLS:
        if not cell[t][p]: continue
        top = txts[t][p].most_common(1)[0]
        print("\n  [%s / %s · %d sim]" % (t, p, len(cell[t][p])))
        print("    %s" % top[0][:190])
