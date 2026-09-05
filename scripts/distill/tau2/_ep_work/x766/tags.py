# -*- coding: utf-8 -*-
"""x766-b: 사이드카 발화 태그 집계(도메인별·시기별). 읽기 전용."""
import collections
import glob
import gzip
import json
import os
import re
import sys

SR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
TAG = re.compile(r"\[([A-Z][A-Z0-9 _\-]{2,40})\]")
POL = re.compile(r"POLICY GATE ([A-Z0-9_]+)")


def scan(files):
    tags = collections.Counter()
    kinds = collections.Counter()
    sims = set()
    tag_sims = collections.defaultdict(set)
    tag_kind = collections.defaultdict(collections.Counter)
    routes = collections.Counter()
    subcalls = collections.Counter()
    for f in files:
        try:
            fh = gzip.open(f, "rt", encoding="utf-8")
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                k = r.get("kind")
                kinds[k] += 1
                st = r.get("simtag") or r.get("sim")
                sims.add((os.path.basename(f), st))
                if k == "route":
                    routes[(r.get("agent"), r.get("outcome"))] += 1
                elif k == "subcall":
                    subcalls[r.get("call_name")] += 1
                if k not in ("reminder-user", "reminder-assistant", "tool-deny", "speak-prohibit"):
                    continue
                t = r.get("text") or ""
                seen = set()
                for m in TAG.finditer(t):
                    g = m.group(1)
                    if g.startswith("POLICY GATE"):
                        continue
                    seen.add(g)
                for m in POL.finditer(t):
                    seen.add("POLICY_GATE:" + m.group(1))
                if not seen:
                    seen.add("(NOTAG)")
                for g in seen:
                    tags[g] += 1
                    tag_kind[g][k] += 1
                    tag_sims[g].add((os.path.basename(f), st))
        except Exception as e:
            print("ERR", f, e, file=sys.stderr)
    return tags, kinds, len(sims), {t: len(v) for t, v in tag_sims.items()}, tag_kind, routes, subcalls


def report(label, files):
    tags, kinds, nsim, tsims, tkind, routes, subcalls = scan(files)
    print("### %s  files=%d sims=%d rows=%d" % (label, len(files), nsim, sum(kinds.values())))
    print("kinds:", dict(kinds))
    print("%-34s %8s %6s  %s" % ("TAG", "rows", "sims", "kinds"))
    for t, v in tags.most_common(200):
        if t == "(NOTAG)":
            continue
        print("%-34s %8d %6d  %s" % (t, v, tsims.get(t, 0), dict(tkind[t])))
    print("-- route agents:", routes.most_common(20))
    print("-- subcalls:", subcalls.most_common(20))
    print()


allb = [f for f in sorted(glob.glob(os.path.join(SR, "fb_bank_*.jsonl.gz")))
        if "airline" not in f and "retail" not in f]
recent = [f for f in allb if os.path.getmtime(f) > 0 and
          any(d in os.path.basename(f) for d in ("20260830", "20260831", "20260901", "20260902",
                                                 "20260903", "20260904", "20260905"))]
report("BANKING all", allb)
report("BANKING recent(0830~0905)", recent)
report("RETAIL t7391", [os.path.join(SR, "fb_bank_t7391_retail_20260829.jsonl.gz")])
report("AIRLINE t7390", [os.path.join(SR, "fb_bank_t7390_airline_20260829.jsonl.gz")])
