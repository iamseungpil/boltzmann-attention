# -*- coding: utf-8 -*-
"""x766-e: banking A2 키별 **발화 증거 대장**. 읽기 전용(a2/·엔진 수정 0).

칸:
  key / bytes / n_entries / 선언이 가리키는 도구(존재/부재) / 사이드카 태그 / 발화수(all·recent)
"""
import collections
import json
import os
import re
import sys

ENG = r"C:\workspace\ba-frft\scripts\distill\tau2"
A2 = os.path.join(ENG, "a2")

gate = json.load(open(os.path.join(A2, "banking_knowledge.gate.json"), encoding="utf-8"))
env = json.load(open(os.path.join(A2, "env_surface.json"), encoding="utf-8"))
b = env["banking_knowledge"]
btools = set(b["tools"].keys()) if isinstance(b.get("tools"), dict) else set(b.get("tools") or [])
for extra in ("exposed", "discoverable_user_tools", "user_tools", "discoverable_agent_tools"):
    v = b.get(extra)
    if isinstance(v, list):
        btools |= set(v)
    elif isinstance(v, dict):
        btools |= set(v.keys())

TOOLNAME = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,6})\b")


def toolrefs(val):
    blob = json.dumps(val, ensure_ascii=False)
    cands = set(TOOLNAME.findall(blob))
    return cands


rows = []
for k, v in gate.items():
    if k.startswith("_"):
        continue
    blob = json.dumps(v, ensure_ascii=False)
    n = len(v) if hasattr(v, "__len__") else 1
    cands = toolrefs(v)
    hit = sorted(c for c in cands if c in btools)
    rows.append({"key": k, "bytes": len(blob.encode("utf-8")), "n": n,
                 "tools_declared_present": len(hit), "tools_present_sample": hit[:5],
                 "empty": (n == 0)})

rows.sort(key=lambda r: -r["bytes"])
print("banking tools in env inventory:", len(btools))
print("%-30s %9s %6s %6s" % ("KEY", "bytes", "n", "tools"))
for r in rows:
    print("%-30s %9d %6s %6d %s" % (r["key"], r["bytes"], r["n"], r["tools_declared_present"],
                                    "  EMPTY" if r["empty"] else ""))
json.dump(rows, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
