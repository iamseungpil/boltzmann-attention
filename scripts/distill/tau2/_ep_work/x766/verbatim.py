# -*- coding: utf-8 -*-
"""x766-g: A2 키의 **자기 문자열**이 사이드카 발화에 축자로 나타났는가.

[[77]] 축자 검산: 근접 추정도 태그 추정도 아니고, 그 키가 저작한 문자열 그 자체를
사이드카 text 에서 substring 으로 찾는다. 찾히면 그 키는 **모델에게 실제로 갔다**.
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

DOM = sys.argv[1]                       # banking | retail | airline
FILES = {"banking": "banking_knowledge.gate.json", "retail": "retail.gate.json",
         "airline": "airline.gate.json"}
d = json.load(open(os.path.join(A2, FILES[DOM]), encoding="utf-8"))

# 후보 문자열: 키가 저작한 산문 중 길이 30+ 인 조각(플레이스홀더 앞까지 자름)
PH = re.compile(r"\{[^}]*\}|%[sd]")


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            if str(k).startswith("_note") or str(k).startswith("_quote"):
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
    seen = []
    for s in strings(v):
        for frag in PH.split(s):
            frag = frag.strip()
            if len(frag) >= 34 and " " in frag:
                seen.append(frag[:90])
    # 중복 제거 + 상위 40개
    uniq = []
    for f in seen:
        if f not in uniq:
            uniq.append(f)
    probes[k] = uniq[:40]

# 사이드카 스캔
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

hit_rows = collections.Counter()
hit_sims = collections.defaultdict(set)
hit_kind = collections.defaultdict(collections.Counter)
example = {}
nrows = 0
for f in files:
    try:
        for line in gzip.open(f, "rt", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text") or ""
            if not t:
                continue
            nrows += 1
            for k, fr in probes.items():
                for p in fr:
                    if p in t:
                        hit_rows[k] += 1
                        hit_sims[k].add((os.path.basename(f), r.get("simtag") or r.get("sim")))
                        hit_kind[k][r.get("kind")] += 1
                        example.setdefault(k, (os.path.basename(f), p))
                        break
    except Exception as e:
        print("ERR", f, e, file=sys.stderr)

print("### %s  files=%d  text-bearing rows=%d" % (DOM, len(files), nrows))
print("%-30s %5s %8s %6s  %-28s %s" % ("KEY", "probe", "rows", "sims", "kinds", "verbatim example"))
rows = []
for k in probes:
    rows.append((hit_rows[k], k))
rows.sort(reverse=True)
for n, k in rows:
    ex = example.get(k, ("", ""))[1][:56]
    print("%-30s %5d %8d %6d  %-28s %s" % (k, len(probes[k]), n, len(hit_sims[k]),
                                           str(dict(hit_kind[k]))[:28], ex))
print()
print("ZERO-VERBATIM keys (probe>0 but never in any utterance):",
      sorted(k for n, k in rows if n == 0 and probes[k]))
print("NO-PROBE keys (키가 산문을 저작하지 않음 = 이 검사로는 판정 불가):",
      sorted(k for k in probes if not probes[k]))
