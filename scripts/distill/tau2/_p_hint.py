# -*- coding: utf-8 -*-
"""`digit` 힌트를 켜면 **무엇이 더 막히는가** — t7354 한 런 전수(폭발 반경 측정·[[66]]).

술어는 엔진의 그것을 그대로 쓴다(사본 금지·[[67]]): `_hint_hit` + `_ctx_has` 대용으로
'값이 이전 메시지 어디에도 문자열로 없는가'. 판단 0.
"""
import sys, json, collections
sys.path.insert(0, ".")
import t2_forensic as F
import t2_gate_patch as G

TAGS = ["bank_t7354_grpA1_20260825", "bank_t7354_grpA2_20260825",
        "bank_t7354_grpA3_20260825", "bank_t7354_grpA4_20260825",
        "bank_t7354_grpB1_20260825", "bank_t7354_grpB2_20260825"]
OLD = G.DEFAULT_ARG_HINTS
NEW = tuple(set(OLD) | {"digit"})

added = collections.Counter()
rows = []
for tag in TAGS:
    try:
        sims = F.sims(tag, ".results.json.gz")
    except Exception:
        continue
    for s in sims:
        tid = str(s.get("task_id"))
        ms = s.get("messages") or []
        ctx = []
        for i, m in enumerate(ms):
            c = str(m.get("content") or "")
            for tc in (m.get("tool_calls") or []):
                for k, v in G._prov_scan_args(tc):
                    for val in G._flatten(v):
                        sv = str(val).strip()
                        if len(sv) < 4:
                            continue
                        was = G._hint_hit(k, OLD)
                        now = G._hint_hit(k, NEW)
                        if was or not now:
                            continue          # 이미 보던 것 = 변화 아님
                        blob = "\n".join(ctx)
                        if sv.lower() in blob.lower():
                            continue          # 문맥에 있으면 통과 = 변화 아님
                        added[(tid, k)] += 1
                        rows.append((tag, tid, i, k, sv))
            ctx.append(c)
print("== `digit` 힌트 추가로 **새로 막히는** 인자 ==")
for (tid, k), n in added.most_common():
    print("  %-10s %-28s %d건" % (tid, k, n))
print("총 %d건 · 태스크 %d개" % (sum(added.values()), len({t for t, _ in added})))
for r in rows[:12]:
    print("   ", r)
