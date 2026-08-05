# -*- coding: utf-8 -*-
"""Does the agent reach for a protocol's tool without having read the protocol?

`task_035` is a credit-bureau incident and the agent called the purchase-decline
protocol's tool instead. `task_032` is a purchase decline and the agent went straight to
the standard transfer. Both are the same failure: the wrong protocol, or none. Which
situation the conversation is in is an open question ([[22]]) and stays the model's — but
a closed one sits next to it, and the transfer tool's own docstring states it:

    "The proper transfer reason enum can be found in the knowledge base:
     search it before calling this tool to select the proper applicable reason."

So: was the document that defines this tool ever retrieved before the tool was used? The
retrieval history is closed, and the document that names a tool is a fact of the corpus,
not a judgement about the customer.

  used_unread   the tool was called and its own document was never retrieved
  gold_unread   gold does the same — which would make a deny wrong

Free: persisted trajectories and the document corpus.

  usage: x93_protocol_doc_census.py [arm]
"""

import collections
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x50_says_not_does import ARMS, SIM  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
DOCS = (os.environ.get("T2_KB_DOCS_DIR")
        or "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents")

# 프로토콜을 대표하는 도구들 — 각각 자기 문서에서만 나온다(전용 도구)
TOOLS = ["initial_transfer_to_human_agent_0218", "initial_transfer_to_human_agent_1822",
         "emergency_credit_bureau_incident_transfer_1114", "transfer_to_human_agents"]


def tool_to_docs():
    """도구 이름 → 그 이름을 담은 문서 id 집합. 코퍼스 사실이지 판단이 아니다."""
    out = collections.defaultdict(set)
    for f in glob.glob(os.path.join(DOCS, "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        c = d.get("content") or ""
        for tool in TOOLS:
            if tool in c:
                out[tool].add(d.get("id"))
    return out


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


t2d = tool_to_docs()
if not t2d:
    raise SystemExit("문서 코퍼스를 못 읽었다 — T2_KB_DOCS_DIR 확인")
print("도구 → 문서 지도")
for k, v in sorted(t2d.items()):
    print("  %-46s %s" % (k, sorted(v)))
print()

tally = collections.Counter()
ex, gex = [], []
for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        tally["sim"] += 1
        sid = "%s/t%s" % (s.get("task_id"), s.get("trial"))
        seen_docs, first_use = set(), {}
        for m in s.get("messages") or []:
            if m.get("role") == "tool":
                c = str(m.get("content") or "")
                for docs in t2d.values():
                    for did in docs:
                        if did and did in c:
                            seen_docs.add(did)
            for tc in (m.get("tool_calls") or []):
                nm = inner(tc.get("arguments")) or tc.get("name")
                if nm in t2d and nm not in first_use:
                    first_use[nm] = set(seen_docs)
        for nm, had in first_use.items():
            tally["use"] += 1
            if not (had & t2d[nm]):
                tally["used_unread"] += 1
                if len(ex) < 12:
                    ex.append((sid, nm))
        gold_names = {inner((c.get("action") or {}).get("arguments"))
                      or (c.get("action") or {}).get("name")
                      for c in ((s.get("reward_info") or {}).get("action_checks") or [])}
        for nm in gold_names & set(t2d):
            if nm not in first_use:
                continue
            if not (first_use[nm] & t2d[nm]):
                tally["gold_unread"] += 1
                if len(gex) < 8:
                    gex.append((sid, nm))

print("arm %s · sim %d · 프로토콜 도구 최초사용 %d" % (ARM, tally["sim"], tally["use"]))
print("  **문서를 안 읽고 쓴 경우 %d건**" % tally["used_unread"])
for sid, nm in ex:
    print("    %-16s %s" % (sid, nm))
print()
print("  ★게이트 — gold이 요구한 도구인데 문서 미열람 상태로 쓴 경우 = **%d**" % tally["gold_unread"])
for sid, nm in gex:
    print("    %-16s %s" % (sid, nm))
print("  판정: %s" % ("표면화 등재 가능" if tally["used_unread"]
                      else "표적 없음 — 만들지 않는다"))
