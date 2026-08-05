# -*- coding: utf-8 -*-
"""Fill the pre-registered verdict table from a run, instead of reading it by hand.

Every judgement in this arc that turned out wrong was assembled by eye from a log and a
results file — seven of them in one session ([[55]]). The pre-registration says exactly
which numbers decide each indicator, so the filling can be mechanical, and mechanical is
the point: the table comes out the same whoever runs it, and it can be built and checked
against an old run while the new one is still going.

Indicators are the ones registered for smoke m:

  A sidecar     the non-committed feedback file exists and carries more than one channel
  B 048 conflict  UNLOCK_NAME fires vs. duplicate searches that followed it
  C 028 chain     the unlock-hint fires, and unlock calls actually happen
  D 022 field     the field-source surface fires
  E protocol      require-doc surface fires; procedure firing on 032/035 is predicted zero
  F side effects  context deaths, replay errors, over-block
  G pass          per task, reported as [D] and never used to attribute

  usage: x97_smoke_verdict.py <tag>            e.g. 20260805m
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

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260805m"
LOGD = "/home/woori/scratch/logs"
SIMD = "/home/woori/scratch/tau2-bench/data/simulations"
REPO_SIM = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def read_logs():
    txt = []
    for f in sorted(glob.glob(os.path.join(LOGD, "bank_smk_gpu*_%s.log" % TAG))):
        txt.append(io.open(f, encoding="utf-8", errors="ignore").read())
    if not txt:
        for f in sorted(glob.glob(os.path.join(REPO_SIM, "bank_smk_gpu*_%s.log.gz" % TAG))):
            txt.append(gzip.open(f, "rt", encoding="utf-8", errors="ignore").read())
    return "\n".join(txt)


def read_sims():
    out = []
    for f in sorted(glob.glob(os.path.join(SIMD, "bank_smk_gpu*_%s" % TAG, "results.json"))):
        out.extend(json.load(open(f, encoding="utf-8")).get("simulations") or [])
    if not out:
        for f in sorted(glob.glob(os.path.join(REPO_SIM, "bank_smk_gpu*_%s.results.json.gz" % TAG))):
            out.extend(json.load(gzip.open(f, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def read_sidecar():
    for p in (os.path.join(LOGD, "fb_%s.jsonl" % TAG),
              os.path.join(REPO_SIM, "fb_%s.jsonl.gz" % TAG)):
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else io.open
            ch = collections.Counter()
            n = 0
            with op(p, "rt", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    try:
                        ch[json.loads(line).get("channel")] += 1
                        n += 1
                    except Exception:
                        pass
            return p, n, ch
    return None, 0, collections.Counter()


log = read_logs()
sims = read_sims()
if not sims:
    raise SystemExit("결과 없음 — 아직 진행 중이거나 태그가 틀렸다: %s" % TAG)
marks = collections.Counter(re.findall(r"\[(T2_[A-Z_]+)\]", log))
by_task = {s.get("task_id"): s for s in sims}


def dup_searches(task):
    s = by_task.get(task)
    if not s:
        return None
    q = collections.Counter()
    for m in s.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            if tc.get("name") == "KB_search_bm25":
                a = tc.get("arguments")
                a = a if isinstance(a, dict) else json.loads(a or "{}")
                q[str(a.get("query"))] += 1
    return sum(v - 1 for v in q.values() if v > 1)


def calls(task, name_frag):
    s = by_task.get(task)
    if not s:
        return None
    return sum(1 for m in (s.get("messages") or [])
               for tc in (m.get("tool_calls") or []) if name_frag in (tc.get("name") or ""))


path, n_side, ch = read_sidecar()
stubs = max([int(x) for x in re.findall(r"SEARCH_EXHAUST\] nudge stubs=(\d+)", log)] or [0])
ctx_death = sum(1 for s in sims if s.get("termination_reason") == "context_window_exceeded")

print("# 스모크 %s 판정 (자동)" % TAG)
print()
print("| # | 지표 | 결과 |")
print("|---|---|---|")
print("| **A** | 사이드카 | %s · 항목 %d · **채널 %d종** %s |"
      % (os.path.basename(path) if path else "**없음**", n_side, len(ch), dict(ch) or ""))
print("| **B** | 048 지시 모순 | `UNLOCK_NAME` %d회 · `SEARCH_EXHAUST` 최대 stubs **%d** · 048 중복질의 **%s** |"
      % (marks.get("T2_UNLOCK_NAME", 0), stubs, dup_searches("task_048")))
print("| **C** | 028 체인 | `unlock-hint` %d회 · 028 unlock 호출 **%s** |"
      % (log.count("[T2_FOLLOWUP] unlock-hint"), calls("task_028", "unlock")))
print("| **D** | 022 필드출처 | `T2_FIELD_SOURCE` **%d회** |" % marks.get("T2_FIELD_SOURCE", 0))
print("| **E** | 프로토콜 | `T2_REQUIRE_DOC` **%d회** · `T2_PROCEDURE` %d · `T2_PROC_ABSENT` %d |"
      % (marks.get("T2_REQUIRE_DOC", 0), marks.get("T2_PROCEDURE", 0),
         marks.get("T2_PROC_ABSENT", 0)))
print("| **F** | 부작용 | ctx 사망 **%d** · replay ValueError %d · 400/schema %d |"
      % (ctx_death, log.count("ValueError"),
         len(re.findall(r"BadRequest|invalid_request", log, re.I))))
rew = ", ".join("%s=%s" % (t.replace("task_", ""), (by_task[t].get("reward_info") or {}).get("reward"))
                for t in sorted(by_task))
print("| **G** | pass([D]) | %s |" % rew)
print()
print("기타 발화: %s" % dict(sorted(marks.items(), key=lambda x: -x[1])[:10]))
