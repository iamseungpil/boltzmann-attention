# -*- coding: utf-8 -*-
"""x766-h: 엄격판 축자 검산 — **우리 층에서만 나올 수 있는** 문자열만 프로브로 쓴다.

x766-g 의 결함: A2 가 정책 문서를 인용한 조각은 KB 전달문에도 그대로 나타나 **키 귀속이 오염**된다
(`catalog_attrs` 749행이 실은 KB 문서였고, `enum_priority` 905행은 `gates` 의 문면이었다).
⇒ 프로브를 «[TAG] 를 포함하거나 'Error:'/'Note:' 로 시작하는 피드백 문면» 으로 제한하고,
   두 키가 공유하는 조각은 **양쪽 모두에서 제외**한다(귀속 불가는 세지 않는다).
읽기 전용.
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

ENG = r"C:\workspace\ba-frft\scripts\distill\tau2"
A2 = os.path.join(ENG, "a2")
SR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
DOM = sys.argv[1]
FILES = {"banking": "banking_knowledge.gate.json", "retail": "retail.gate.json",
         "airline": "airline.gate.json"}
d = json.load(open(os.path.join(A2, FILES[DOM]), encoding="utf-8"))
PH = re.compile(r"\{[^}]*\}|%[sd]")
TAGGED = re.compile(r"\[[A-Z][A-Z0-9 _\-]{2,40}\]")


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            if str(k).startswith("_"):
                continue
            for r in strings(v):
                yield r
    elif isinstance(o, list):
        for v in o:
            for r in strings(v):
                yield r


probes = {}
for k, v in d.items():
    if k.startswith("_"):
        continue
    out = []
    for s in strings(v):
        if not (TAGGED.search(s) or s.startswith("Error:") or s.startswith("Note:")):
            continue
        for frag in PH.split(s):
            frag = frag.strip()
            if len(frag) >= 30 and " " in frag and frag not in out:
                out.append(frag[:100])
    probes[k] = out[:60]

own = {}
for k, fr in probes.items():
    for f in fr:
        own.setdefault(f, set()).add(k)
shared = {f for f, ks in own.items() if len(ks) > 1}
for k in probes:
    probes[k] = [f for f in probes[k] if f not in shared]

if DOM == "banking":
    SKIP = {"fb_bank_x607_A_control_20260829.jsonl.gz", "fb_bank_x607_C_undeclared_20260829.jsonl.gz",
            "fb_bank_x612_D_suppfix_20260829.jsonl.gz", "fb_bank_x613_E_suppfix_only_20260830.jsonl.gz"}
    files = [f for f in sorted(glob.glob(os.path.join(SR, "fb_bank_*.jsonl.gz")))
             if "airline" not in f and "retail" not in f and os.path.basename(f) not in SKIP]
elif DOM == "retail":
    files = [os.path.join(SR, "fb_bank_t7391_retail_20260829.jsonl.gz"),
             os.path.join(SR, "fb_bank_t7391_retail_smoke_20260829.jsonl.gz")]
else:
    files = [os.path.join(SR, "fb_bank_t7390_airline_20260829.jsonl.gz"),
             os.path.join(SR, "fb_bank_t7390_airline_smoke_20260829.jsonl.gz")]

rows = collections.Counter()
sims = collections.defaultdict(set)
kinds = collections.defaultdict(collections.Counter)
for f in files:
    for line in gzip.open(f, "rt", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("text") or ""
        if not t:
            continue
        for k, fr in probes.items():
            for p in fr:
                if p in t:
                    rows[k] += 1
                    sims[k].add((os.path.basename(f), r.get("simtag") or r.get("sim")))
                    kinds[k][r.get("kind")] += 1
                    break

print("### %s  files=%d   (shared/ambiguous fragments dropped: %d)" % (DOM, len(files), len(shared)))
print("%-30s %6s %8s %6s %8s  %s" % ("KEY", "probe", "rows", "sims", "tool-deny", "kinds"))
order = sorted(probes, key=lambda k: -rows[k])
for k in order:
    if not probes[k]:
        continue
    print("%-30s %6d %8d %6d %8d  %s"
          % (k, len(probes[k]), rows[k], len(sims[k]), kinds[k].get("tool-deny", 0),
             str(dict(kinds[k]))[:52]))
print()
print("ZERO (probe exists, never delivered):", sorted(k for k in probes if probes[k] and rows[k] == 0))
print("NO-FEEDBACK-PROSE (이 검사 대상 아님):", sorted(k for k in probes if not probes[k]))
