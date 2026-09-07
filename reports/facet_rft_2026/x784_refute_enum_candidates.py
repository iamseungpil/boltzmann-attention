# -*- coding: utf-8 -*-
"""x784 — 반증 프로브 ③: T2_WRITE_ARG_ENUM 을 **무장했을 때** gold account_class 가
후보 명단에 드는가. 엔진 정본 헬퍼(`_display_slugs`)를 그대로 호출한다(사본 금지 [[67]]).
읽기 전용.
"""
import json, os, sys

ENG = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, ENG)
import t2_gate_patch as G

A2 = json.load(open(os.path.join(ENG, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
DI = ((A2.get("policy_ontology") or {}).get("doc_index") or {})
print("doc_index groups:", len(DI))

spec = None
for s in (A2.get("write_arg_enum") or []):
    if s.get("arg") == "account_class":
        spec = s
        break
print("spec group_map:", spec.get("group_map"))

CAND = {}
for gval, grp in (spec.get("group_map") or {}).items():
    subs = DI.get(grp) or {}
    names = G._display_slugs(subs)
    CAND[gval] = names
    print("\n[%s -> %s] n=%d" % (gval, grp, len(names)))
    print("   ", names)

# gold (account_type, account_class) 짝 — x782 census 재사용 대신 직접 재수집
import gzip, glob, collections
TASKS = {}
for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "sim_results", "*.results.json.gz"))):
    try:
        d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    except Exception:
        continue
    for t in (d.get("tasks") or []):
        if t.get("id") and t["id"] not in TASKS:
            TASKS[t["id"]] = t

print("\n=== gold (account_type, account_class) × 후보 명단 소속 ===")
miss = []
tot = 0
for tid in sorted(TASKS):
    for a in ((TASKS[tid].get("evaluation_criteria") or {}).get("actions") or []):
        ar = a.get("arguments") or {}
        if a.get("name") == "call_discoverable_agent_tool":
            raw = ar.get("arguments")
            if isinstance(raw, str):
                try:
                    ar = json.loads(raw)
                except Exception:
                    ar = {}
        if "account_class" not in ar:
            continue
        at, ac = ar.get("account_type"), ar.get("account_class")
        names = CAND.get(at, [])
        ok = ac in names
        tot += 1
        if not ok:
            miss.append((tid, a.get("action_id"), at, ac))
        print("  %-9s %-10s type=%-18s class=%-28r in_candidates=%s"
              % (tid, a.get("action_id"), at, ac, ok))
print("\nGOLD OUTSIDE CANDIDATE LIST: %d / %d" % (len(miss), tot))
for m in miss:
    print("   ", m)
