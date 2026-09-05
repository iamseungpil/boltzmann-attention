#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x792 - 한 sim 의 **우리 층 발화 타임라인** (sidecar fb_ + run log 레버 마커). 2026-09-05.

사용:  python x792_flipB_sidecar_timeline.py <tag> <task_id> <seed> [max_turn] [maxchars]
⛔ 판정하지 않는다. 찍기만 한다.
"""
import io, json, re, sys, gzip, os

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

LOGS = "/home/woori/scratch/logs"


def opentext(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


def main():
    tag, task, seed = sys.argv[1], sys.argv[2], sys.argv[3]
    MAXT = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    MAXC = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    simtag = "%s#s%s" % (task, seed)
    print("== SIDECAR %s  simtag=%s  (turn<=%s)" % (tag, simtag, MAXT))
    fb = None
    for cand in ("%s/fb_%s.jsonl" % (LOGS, tag), "%s/fb_%s.jsonl.gz" % (LOGS, tag)):
        if os.path.exists(cand):
            fb = cand
            break
    rows = []
    if fb:
        for line in opentext(fb):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("simtag") != simtag:
                continue
            t = o.get("turn")
            if isinstance(t, int) and t > MAXT:
                continue
            rows.append(o)
    else:
        print("  (사이드카 없음)")
    rows.sort(key=lambda o: (o.get("turn") if isinstance(o.get("turn"), int) else -1))
    for o in rows:
        k = o.get("kind")
        if k == "subcall":
            print("  t%-3s %-19s %-28s out=%s" % (o.get("turn"), k, o.get("call_name"),
                                                  str(o.get("out_head"))[:110].replace("\n", " ")))
        else:
            print("  t%-3s %-19s ch=%-12s len=%s :: %s" % (
                o.get("turn"), k, o.get("channel"), o.get("len"),
                str(o.get("text"))[:MAXC].replace("\n", " \n ")))
    print("== LOG MARKERS %s  [sim=%s]" % (tag, simtag))
    lg = None
    for cand in ("%s/%s.log" % (LOGS, tag), "%s/%s.log.gz" % (LOGS, tag)):
        if os.path.exists(cand):
            lg = cand
            break
    if not lg:
        print("  (로그 없음)")
        return
    pat = re.compile(r"\[sim=" + re.escape(simtag) + r"\]")
    mk = re.compile(r"\[(T2_[A-Z0-9_]+)\]|\[T2_LEVER\]\s+(T2_[A-Z0-9_]+)")
    from collections import Counter
    cnt = Counter()
    lines = []
    for line in opentext(lg):
        if not pat.search(line):
            continue
        ms = mk.findall(line)
        if not ms:
            continue
        names = [a or b for a, b in ms]
        cnt.update(names)
        lines.append((names, line.strip()[:MAXC]))
    print("  마커 집계: %s" % cnt.most_common(40))
    for names, l in lines[:80]:
        print("   | %s" % l)


if __name__ == "__main__":
    main()
