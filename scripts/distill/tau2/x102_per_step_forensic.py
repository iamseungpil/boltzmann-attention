# -*- coding: utf-8 -*-
"""One line per step: what was called, what came back, what we said, and whether gold moved.

The task-level reads keep landing on "it stopped doing the procedure", which is a summary,
not a cause. What decides a prescription is the step where the run left gold's path and what
was on the screen at that step — so this walks the messages in order and annotates each one:

  GOLD    this call satisfies a gold action (by tool name)
  ERR     the environment returned an error
  fb:…    what our layer said at that point (sidecar, channel-tagged)

and then prints, per simulation, the last step at which gold moved and everything after it —
the segment where the run was already lost, which is where prescriptions have to bite.

The sidecar clock is `len(messages)`, so instructions join to steps directly.

  usage: x102_per_step_forensic.py <tag> [task ...]
"""

import collections
import glob
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260805n"
WANT = set(sys.argv[2:])
LOGD = os.environ.get("T2_LOGD", "/home/woori/scratch/logs")
SIMD = os.environ.get("T2_SIMD", "/home/woori/scratch/tau2-bench/data/simulations")
AFTER = int(os.environ.get("X102_AFTER", "0"))     # 0 = 전체, N = 마지막 gold 이후만


def inner(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


def sims():
    out = []
    for f in sorted(glob.glob(os.path.join(SIMD, "bank_smk_gpu*_%s" % TAG, "results.json"))):
        out.extend(json.load(io.open(f, encoding="utf-8")).get("simulations") or [])
    return out


def fingerprint(sim):
    for m in sim.get("messages") or []:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return hashlib.sha1(m["content"].strip().encode("utf-8")).hexdigest()[:12]
    return None


SC = collections.defaultdict(lambda: collections.defaultdict(list))
p = os.path.join(LOGD, "fb_%s.jsonl" % TAG)
if os.path.exists(p):
    for line in io.open(p, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("text"):
            SC[r.get("sim")][r.get("turn") or 0].append(
                (str(r.get("channel"))[:10], " ".join(str(r["text"]).split())))

for s in sims():
    tid = s.get("task_id")
    if (WANT and tid not in WANT) or (s.get("reward_info") or {}).get("reward") == 1.0:
        continue
    ms = s.get("messages") or []
    fb = SC.get(fingerprint(s), {})
    gold = collections.Counter()
    for c in ((s.get("reward_info") or {}).get("action_checks") or []):
        a = c.get("action") or {}
        gold[inner(a.get("arguments")) or a.get("name")] += 1
    ri = s.get("reward_info") or {}
    print("\n" + "=" * 90)
    print("== %s reward=%s basis=%s term=%s · %d steps · gold %s"
          % (tid, ri.get("reward"), (ri.get("reward_breakdown") or {}) and
             list((ri.get("reward_breakdown") or {}).keys()), s.get("termination_reason"),
             len(ms), dict(gold)))

    done, last_gold, pend = collections.Counter(), -1, {}
    lines = []
    for i, m in enumerate(ms):
        role = m.get("role")
        txt = " ".join(str(m.get("content") or "").split())
        marks = []
        for tc in (m.get("tool_calls") or []):
            nm = inner(tc.get("arguments")) or tc.get("name")
            args = json.dumps(tc.get("arguments"), ensure_ascii=False)[:90]
            g = ""
            if nm in gold:
                done[nm] += 1
                g = " GOLD"
                last_gold = i
            marks.append("call %s%s %s" % (nm, g, args))
            pend[tc.get("id")] = nm
        if role == "tool":
            err = txt.lstrip().startswith("Error:")
            marks.append("%s← %s" % ("ERR " if err else "", txt[:110]))
        lines.append((i, role, txt[:110], marks, fb.get(i) or []))

    start = 0 if not AFTER else max(0, last_gold)
    print("   마지막 gold 이동 = step %d / %d · 미달성 gold: %s"
          % (last_gold, len(ms), [k for k in gold if not done.get(k)]))
    for i, role, txt, marks, fbs in lines:
        if i < start:
            continue
        head = "  [%03d %-9s]" % (i, role)
        if marks:
            print("%s %s" % (head, marks[0]))
            for x in marks[1:]:
                print("%s %s" % (" " * len(head), x))
        else:
            print("%s %s" % (head, txt))
        for ch, t in fbs:
            print("%s   fb:%-10s %s" % (" " * len(head), ch, t[:130]))
