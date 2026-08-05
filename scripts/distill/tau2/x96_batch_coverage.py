# -*- coding: utf-8 -*-
"""Which tasks does everything we built this arc actually touch?

Twelve tasks were fixed one at a time, each against its own census. That says nothing
about the other eighty-five, and the next run should start with the ones most likely to
show the change rather than with whatever order a handoff happened to list. So this
re-scores every simulation against the declarations as they stand now and reports, per
task, how many of this arc's checks would have something to say.

Each axis is the same closed predicate the live lever uses — the point is not a new
measurement but one place that runs all of them:

  procedure   an entered procedure with an unmet step and quiet turns (x86)
  transcribe  a payload contradicting, inventing, or under-filling its records (x90/F27/F31)
  transfer    a protocol tool used without its document, or gold's transfer never made (x93/x92)
  withdrawn   a settled row nobody submitted, counted only on clean inputs (x94)

`covered` is not a prediction that the task will pass. It says our checks would speak
there — whether the model then acts is exactly what the run measures.

Free: persisted trajectories and the declarations.

  usage: x96_batch_coverage.py [arm] [top_n]
"""

import collections
import glob
import gzip
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

import t2_procedure as P          # noqa: E402
import t2_transcribe as T         # noqa: E402
from t2_scaffold_get import _parse_record_dump as PARSE   # noqa: E402
from x50_says_not_does import ARMS, SIM                   # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
K = int(os.environ.get("X96_K", "3"))
DOCS = (os.environ.get("T2_KB_DOCS_DIR")
        or "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents")


def a2():
    import gate_interpreter as GI
    return GI.load_domain_a2("banking_knowledge") or {}


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


def args_of(x):
    a = x.get("arguments") if isinstance(x, dict) else None
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def docs_naming(tools):
    out = collections.defaultdict(set)
    for f in glob.glob(os.path.join(DOCS, "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        c = d.get("content") or ""
        for t in tools:
            if t in c:
                out[t].add(d.get("id"))
    return out


A = a2()
PROCS = A.get("procedures") or []
TRS = {k: v for k, v in (A.get("transcription_check") or {}).items()
       if not k.startswith("_") and isinstance(v, dict)}
WDS = A.get("withdrawn_row_check") or {}
RDB = (A.get("require_doc_before") or {}).get("tools") or []
T2D = docs_naming(RDB) if RDB else {}
IDP = re.compile(str(WDS.get("id_pattern") or r"\btxn_[0-9a-f_]+\b"))

score = collections.defaultdict(collections.Counter)
for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        tid = s.get("task_id")
        sc = score[tid]
        sc["sim"] += 1
        sc["pass"] += 1 if (s.get("reward_info") or {}).get("reward") == 1.0 else 0
        recs, pend, executed = {}, {}, collections.Counter()
        quiet = collections.Counter()
        seen_docs, settled, submitted, first_use = set(), set(), set(), {}
        clean_call = None
        for m in s.get("messages") or []:
            if m.get("role") == "tool":
                nm = pend.pop(m.get("id") or m.get("tool_call_id"), None)
                c = str(m.get("content") or "")
                err = c.lstrip().startswith("Error:") or bool(m.get("error"))
                if nm and not err:
                    executed[nm] += 1
                try:
                    rows = PARSE(c)
                except Exception:
                    rows = []
                for r in rows:
                    for sp in TRS.values():
                        rid = r.get(sp.get("id_key"))
                        if rid:
                            recs[str(rid)] = r
                for did in {x for v in T2D.values() for x in v}:
                    if did and did in c:
                        seen_docs.add(did)
                if nm == WDS.get("settle_tool") and not err and \
                        (m.get("id") or m.get("tool_call_id")) == clean_call:
                    settled |= set(IDP.findall(c))
                continue
            for tc in (m.get("tool_calls") or []):
                nm = inner(args_of(tc)) or tc.get("name")
                pend[tc.get("id")] = nm
                if nm in T2D and nm not in first_use:
                    first_use[nm] = set(seen_docs)
                if nm == WDS.get("submit_tool"):
                    for v in args_of(tc).values():
                        if isinstance(v, str):
                            submitted |= set(IDP.findall(v))
                sp = TRS.get(tc.get("name"))
                if sp:
                    bad = T.mismatches(sp, args_of(tc), recs)
                    unk = T.unknown_ids(sp, args_of(tc), recs)
                    fs = T.field_sources(sp, args_of(tc), recs, sp.get("require_fields") or ())
                    if bad:
                        sc["transcribe_mismatch"] += 1
                    if unk:
                        sc["transcribe_unknown"] += 1
                    if fs:
                        sc["transcribe_fieldsrc"] += 1
                    clean_call = tc.get("id") if not (bad or unk) else clean_call
            if m.get("role") != "assistant":
                continue
            called = {inner(args_of(tc)) or tc.get("name") for tc in (m.get("tool_calls") or [])}
            for pr in P.active_procedures(PROCS, executed):
                nodes = {t for n in (pr.get("nodes") or []) for t in (P._tools_of(n) or [])}
                if called & nodes:
                    quiet[pr.get("id")] = 0
                    continue
                quiet[pr.get("id")] += 1
                if quiet[pr.get("id")] >= K and P.next_step(pr, executed)[0]:
                    sc["procedure_absent"] += 1
                    quiet[pr.get("id")] = 0
        for nm, had in first_use.items():
            if T2D.get(nm) and not (had & T2D[nm]):
                sc["doc_unread"] += 1
        if settled - submitted:
            sc["withdrawn"] += 1

AXES = ["procedure_absent", "transcribe_mismatch", "transcribe_unknown",
        "transcribe_fieldsrc", "doc_unread", "withdrawn"]
rows = []
for tid, sc in score.items():
    hit = sum(1 for a in AXES if sc[a])
    rows.append((hit, sum(sc[a] for a in AXES), tid, sc))
rows.sort(reverse=True)

print("arm %s · 태스크 %d · K=%d" % (ARM, len(score), K))
print("%-12s %-5s %-6s %s" % ("task", "축", "건", "  ".join("%-10s" % a[:10] for a in AXES)))
for hit, tot, tid, sc in rows[:TOP]:
    print("%-12s %-5d %-6d %s   (sim %d·pass %d)"
          % (tid, hit, tot, "  ".join("%-10d" % sc[a] for a in AXES), sc["sim"], sc["pass"]))
print()
print("★다음 런 앞자리 권고(축 수 → 건수 순):")
print("   " + ",".join(t for _, _, t, _ in rows[:12]))
