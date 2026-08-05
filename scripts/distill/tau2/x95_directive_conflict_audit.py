# -*- coding: utf-8 -*-
"""Do our own messages tell the model opposite things?

`task_048` was read as "we named the tool and it still would not act". The log says
otherwise. The procedure message gave the exact registry name and said not to search; the
agent then called unlock with the bare name; `T2_UNLOCK_NAME` refused it and said *"You do
NOT know the suffix from memory - search the knowledge base NOW"*; the agent searched three
times; `[T2_SEARCH_EXHAUST] nudge stubs=8` followed. The agent obeyed both instructions.
They contradicted each other.

That is not a judgement about the model, so it should not be diagnosed by reading
trajectories one at a time. Two passes, both closed:

  static   every feedback string we can send, classified by the directive it issues, with
           opposing classes reported as pairs. We author these strings, so the check is
           complete over them — the marker table is printed so it can be audited.
  dynamic  per run, which levers actually fired together in the same simulation. A pair
           that is opposed in the static pass and co-fires in the dynamic pass is a
           contradiction the model really received.

Free: A2, the engine source, and run logs.

  usage: x95_directive_conflict_audit.py [log_glob]
"""

import collections
import glob
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOGS = sys.argv[1] if len(sys.argv) > 1 else "*_20260805j"

# 지시 분류 — 우리가 쓴 문구이므로 표지도 우리가 선언한다(감사 가능하도록 전부 출력한다).
MARKERS = [
    ("SEARCH",     [r"search the knowledge base", r"search it before", r"Query with plain words",
                    r"search for the transfer", r"Search for the transfer"]),
    ("NO_SEARCH",  [r"Do not search", r"do not search", r"You do not need to search"]),
    ("UNKNOWN_NAME", [r"You do NOT know the suffix", r"you do not know its full suffixed name"]),
    ("KNOWN_NAME", [r"the name above is complete", r"with exactly that name",
                    r"exactly those names"]),
    ("CALL_NOW",   [r"call it now", r"unlock and call", r"Do that step before continuing",
                    r"Do them before continuing"]),
    ("DO_NOT_ACT", [r"do not proceed", r"Do NOT invent", r"do not update"]),
]
OPPOSED = [("SEARCH", "NO_SEARCH"), ("UNKNOWN_NAME", "KNOWN_NAME")]


def classify(text):
    out = set()
    for name, pats in MARKERS:
        for p in pats:
            if re.search(p, text or ""):
                out.add(name)
                break
    return out


def a2_strings(domain="banking_knowledge"):
    import gate_interpreter as GI
    d = GI.load_domain_a2(domain) or {}
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + "[%d]" % i)
        elif isinstance(o, str) and len(o) > 60:
            # `_`로 시작하는 키는 **주석**이라 모델에 전달되지 않는다 — 감사 대상이 아니다.
            if not path.rsplit("/", 1)[-1].startswith("_"):
                out.append((path, o))
    walk(d, "A2")
    return out


def engine_strings():
    """엔진에 박힌 피드백 문자열 — `Error: [` 로 시작하는 것만(전달되는 형태)."""
    out = []
    for f in ("t2_gate_patch.py", "t2_repeat_gov.py", "t2_scaffold_get.py"):
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        src = io.open(p, encoding="utf-8").read()
        for m in re.finditer(r'"((?:Error: |\[)[^"]{60,})"', src):
            out.append((f, m.group(1)))
    return out


print("① 정적 감사 — 우리가 보낼 수 있는 문구의 지시 분류")
items = [("A2:" + p, s) for p, s in a2_strings()] + [("ENG:" + f, s) for f, s in engine_strings()]
by_class = collections.defaultdict(list)
for src, s in items:
    for c in classify(s):
        by_class[c].append((src, s))
for c, lst in sorted(by_class.items()):
    print("  %-14s %d개" % (c, len(lst)))
    for src, s in lst[:4]:
        print("      %-52s %s" % (src[:52], s[:88].replace("\n", " ")))
print()
print("  ★상반 쌍")
conflict_src = collections.defaultdict(set)
def slot(src):
    """같은 슬롯의 **대안 문구**는 상호배타다 — 엔진이 하나만 고르므로 모순이 아니다."""
    return re.sub(r"/feedback(_[a-z]+)?$", "/feedback", src)


for a, b in OPPOSED:
    ai = [(s, x) for s, x in by_class.get(a, [])]
    bi = [(s, x) for s, x in by_class.get(b, [])]
    _aslots = {slot(s) for s, _ in ai}
    ai = [(s, x) for s, x in ai if slot(s) not in {slot(y) for y, _ in bi}] or ai
    bi = [(s, x) for s, x in bi if slot(s) not in _aslots]
    if ai and bi:
        print("    **%s ↔ %s** — %d개 vs %d개" % (a, b, len(ai), len(bi)))
        for src, _ in ai:
            conflict_src[a].add(src)
        for src, _ in bi:
            conflict_src[b].add(src)
        for src, s in ai[:3]:
            print("        [%s] %s | %s" % (a, src[:46], s[:74]))
        for src, s in bi[:3]:
            print("        [%s] %s | %s" % (b, src[:46], s[:74]))
if not conflict_src:
    print("    없음")

print()
print("② 동적 — 같은 런에서 실제로 함께 발화한 레버")
LOGDIR = "/home/woori/scratch/logs"
files = sorted(glob.glob(os.path.join(LOGDIR, "bank_smk_gpu*%s.log" % LOGS))) or \
        sorted(glob.glob(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "sim_results", "*%s.log.gz" % LOGS)))
if not files:
    print("  로그 없음(리모트에서 돌릴 것): %s" % LOGS)
else:
    for f in files:
        op = gzip.open if f.endswith(".gz") else io.open
        with op(f, "rt", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        marks = collections.Counter(re.findall(r"\[(T2_[A-Z_]+)\]", txt))
        print("  %-40s %s" % (os.path.basename(f),
                              dict(sorted(marks.items(), key=lambda x: -x[1])[:8])))
        both = [k for k in ("T2_UNLOCK_NAME", "T2_PROCEDURE", "T2_PROC_ABSENT",
                            "T2_SEARCH_EXHAUST") if marks.get(k)]
        if "T2_UNLOCK_NAME" in both and ("T2_PROCEDURE" in both or "T2_PROC_ABSENT" in both):
            print("      ★모순 동시발화: UNLOCK_NAME(검색하라) + 절차 문구(검색하지 마라)"
                  " — UNLOCK_NAME %d회 · SEARCH_EXHAUST %d회"
                  % (marks.get("T2_UNLOCK_NAME", 0), marks.get("T2_SEARCH_EXHAUST", 0)))
