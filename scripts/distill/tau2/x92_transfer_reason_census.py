# -*- coding: utf-8 -*-
"""Is the transfer reason a code the policy defines, and does an exhausted search ever reach one?

The transfer tool's own docstring says the reason enum lives in the knowledge base and
must be looked up before calling. Two failures follow from ignoring that. `task_014`
transferred with a code that exists but is not the one the situation calls for — which
tier applies is a judgement, and stays the model's ([[22]]). `task_012` never transferred
at all: it searched four times, found nothing, invented an in-app procedure and closed the
conversation, while gold transfers with the code for an unsuccessful search.

Membership is closed and countable; tier choice is not. So this measures two things:

  invented    a reason code that appears nowhere in the policy document
  no_escalate simulations where gold transfers but the agent never called the tool

and the gate is the same as always: would refusing an invented code ever refuse gold?

Free: persisted trajectories, the policy document, gold.

  usage: x92_transfer_reason_census.py [arm]
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

from x50_says_not_does import ARMS, SIM  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
TOOL = "transfer_to_human_agents"
ARG = "reason"


def policy_codes():
    """정책 문서가 정의한 코드 집합 — 파일에서 읽고, 못 읽으면 gold에서 유추하지 않는다."""
    docs = os.environ.get("T2_KB_DOCS_DIR") or (
        "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    out = set()
    for f in glob.glob(os.path.join(docs, "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        c = d.get("content") or ""
        if "Transfer Reason Codes" not in (d.get("title") or "") and "reason code" not in c.lower():
            continue
        # 표 형식 `| code | when |` 의 첫 칸만 — 문서 포맷 전사이지 판단이 아니다
        for m in re.finditer(r"^\|\s*([a-z][a-z0-9_]{6,})\s*\|", c, re.M):
            out.add(m.group(1))
    return out


def args_of(x):
    a = x.get("arguments") if isinstance(x, dict) else None
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


codes = policy_codes()
if not codes:
    raise SystemExit("정책 문서에서 코드 집합을 못 읽었다 — T2_KB_DOCS_DIR 확인(추측하지 않는다)")
print("정책이 정의한 이관 코드 %d종" % len(codes))

tally = collections.Counter()
bad, gold_bad, noesc = [], [], []
for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        tally["sim"] += 1
        sid = "%s/t%s" % (s.get("task_id"), s.get("trial"))
        called = False
        for m in s.get("messages") or []:
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") != TOOL:
                    continue
                called = True
                tally["call"] += 1
                r = str(args_of(tc).get(ARG) or "")
                if r and r not in codes:
                    tally["invented"] += 1
                    if len(bad) < 10:
                        bad.append((sid, r))
        gold_calls = [c for c in ((s.get("reward_info") or {}).get("action_checks") or [])
                      if (c.get("action") or {}).get("name") == TOOL]
        for c in gold_calls:
            r = str(args_of(c.get("action") or {}).get(ARG) or "")
            if r and r not in codes:
                tally["gold_invented"] += 1
                if len(gold_bad) < 8:
                    gold_bad.append((sid, r))
        if gold_calls and not called:
            tally["no_escalate"] += 1
            if len(noesc) < 10:
                noesc.append((sid, str(args_of(gold_calls[0].get("action") or {}).get(ARG) or "")))

print("arm %s · sim %d · 이관 호출 %d" % (ARM, tally["sim"], tally["call"]))
print("  정책에 없는 코드로 이관 **%d건**" % tally["invented"])
for sid, r in bad:
    print("    %-16s %s" % (sid, r))
print()
print("  gold은 이관하는데 **에이전트가 한 번도 안 부른 sim = %d**" % tally["no_escalate"])
for sid, r in noesc:
    print("    %-16s gold reason=%s" % (sid, r))
print()
print("  ★게이트 — gold 자신이 정책 밖 코드를 쓴 횟수 = **%d**" % tally["gold_invented"])
for sid, r in gold_bad:
    print("    %-16s %s" % (sid, r))
print("  판정: %s" % ("멤버십 검사 등재 가능 (오차단 0)" if not tally["gold_invented"]
                      else "**등재 금지** — gold이 문서 밖 코드를 쓴다"))
