# -*- coding: utf-8 -*-
"""x902 — gate-core 의미-패턴(A1) 감사: 실물 회수분에 술어를 직접 먹여 오발/누락을 잰다."""
import sys, os, json, gzip, re, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G

CORP = sys.argv[1]


class M(object):
    __slots__ = ("role", "content", "tool_calls", "error", "id", "tool_call_id", "requestor")

    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.error = bool(d.get("error"))
        self.id = d.get("id")
        self.tool_call_id = d.get("tool_call_id")
        self.requestor = d.get("requestor") or "assistant"
        self.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]


class TC(object):
    __slots__ = ("name", "arguments", "requestor")

    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = d.get("arguments")
        self.requestor = d.get("requestor") or "assistant"


class Orch(object):
    def __init__(self, msgs):
        self._m = msgs

    def get_messages(self):
        return self._m


sims = json.loads(gzip.open(CORP, "rt", encoding="utf-8").read())
for s in sims:
    s["M"] = [M(d) for d in s["msgs"]]
print("corpus sims=%d  msgs=%d" % (len(sims), sum(len(s["M"]) for s in sims)))
print("tags:", sorted({s["tag"] for s in sims}))
print("rewards: pass=%d fail=%d" % (sum(1 for s in sims if (s["reward"] or 0) >= 1.0),
                                    sum(1 for s in sims if (s["reward"] or 0) < 1.0)))
json.dump(True, open(os.devnull, "w"))
