# -*- coding: utf-8 -*-
"""What did we actually tell this simulation, in order, and did any two of it disagree?

Until the sidecar was on, this could not be asked. Feedback goes to the working buffer and
not to `state.messages`, so a failing run left no record of what the scaffold said — which
is why `task_048` was diagnosed three times as the model's fault before the fourth reading
found our own two messages telling it to search and not to search.

Now the sidecar keys each line by the first user utterance's hash, so the instructions can
be joined back to the simulation and read as a sequence. This prints, per failing task:

  what gold wanted that never happened
  every instruction we sent, in turn order, with its channel
  contradictions among the instructions this simulation actually received

The contradiction classes are the ones `x95` declares — the difference is that x95 audits
what we *could* send and this audits what one run *did* receive.

  usage: x98_instruction_trace.py <tag> [task ...]
"""

import collections
import glob
import gzip
import hashlib
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

from x95_directive_conflict_audit import MARKERS, OPPOSED, classify  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260805m"
WANT = set(sys.argv[2:])
LOGD = "/home/woori/scratch/logs"
SIMD = "/home/woori/scratch/tau2-bench/data/simulations"


def sims():
    out = []
    for f in sorted(glob.glob(os.path.join(SIMD, "bank_smk_gpu*_%s" % TAG, "results.json"))):
        out.extend(json.load(open(f, encoding="utf-8")).get("simulations") or [])
    return out


def fingerprint(sim):
    for m in sim.get("messages") or []:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return hashlib.sha1(m["content"].strip().encode("utf-8")).hexdigest()[:12]
    return None


def sidecar():
    p = os.path.join(LOGD, "fb_%s.jsonl" % TAG)
    by = collections.defaultdict(list)
    if not os.path.exists(p):
        return by
    for line in io.open(p, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("text"):
            by[r.get("sim")].append(r)
    for v in by.values():
        v.sort(key=lambda r: r.get("turn") or 0)
    return by


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


SC = sidecar()
for s in sims():
    tid = s.get("task_id")
    if (WANT and tid not in WANT) or (s.get("reward_info") or {}).get("reward") == 1.0:
        continue
    fp = fingerprint(s)
    recs = SC.get(fp, [])
    called = collections.Counter()
    for m in s.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            called[inner(tc.get("arguments")) or tc.get("name")] += 1
    gold = collections.Counter()
    for c in (s.get("reward_info") or {}).get("action_checks") or []:
        a = c.get("action") or {}
        gold[inner(a.get("arguments")) or a.get("name")] += 1
    missing = [k for k in gold if k not in called]

    print("=" * 78)
    print("== %s  reward=%s  term=%s  fp=%s  지시 %d건"
          % (tid, (s.get("reward_info") or {}).get("reward"), s.get("termination_reason"),
             fp, len(recs)))
    print("   gold인데 미호출: %s" % (missing or "없음"))
    seen = collections.Counter()
    for r in recs:
        cls = classify(r.get("text") or "")
        for c in cls:
            seen[c] += 1
        print("   t%-3s %-16s %-22s %s"
              % (r.get("turn"), r.get("channel", "")[:16], ",".join(sorted(cls)) or "-",
                 re.sub(r"\s+", " ", r["text"])[:96]))
    conf = [(a, b) for a, b in OPPOSED if seen.get(a) and seen.get(b)]
    print("   ★이 sim이 **실제로 받은** 모순: %s"
          % (", ".join("%s↔%s (%d vs %d)" % (a, b, seen[a], seen[b]) for a, b in conf) or "없음"))
