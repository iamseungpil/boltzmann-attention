# -*- coding: utf-8 -*-
"""x783 — 반증 프로브 ②: 로컬 회수분에서 **모델이 실제로 실은 인자**를 뽑는다(읽기 전용).
sim 핀은 x769_pairs.txt 축자를 따른다.
"""
import gzip, json, os, sys, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIRS = {}
for ln in open(os.path.join(ROOT, "x769_pairs.txt"), encoding="utf-8"):
    p = ln.split()
    if len(p) >= 3:
        PAIRS[p[1]] = (p[0], p[2])

WANT = sys.argv[1:] or ["task_071", "task_079", "task_085", "task_101", "task_007",
                        "task_066", "task_061", "task_051", "task_054", "task_015",
                        "task_078", "task_041", "task_010"]

for tid in WANT:
    tag, sid = PAIRS.get(tid, ("?", "?"))
    p = os.path.join(ROOT, "sim_results", "%s.results.json.gz" % tag)
    if not os.path.exists(p):
        print("\n##### %s  tag=%s  results.json.gz NOT LOCAL" % (tid, tag))
        continue
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    sims = [s for s in d.get("simulations", []) if s.get("task_id") == tid]
    print("\n##### %s  tag=%s  sims=%d  pin=%s" % (tid, tag, len(sims), sid[:8]))
    for s in sims:
        print("  sim id=%s reward=%s" % (str(s.get("id"))[:8],
                                         (s.get("reward_info") or {}).get("reward")))
        if str(s.get("id"))[:8] != sid[:8]:
            print("   (핀 불일치 — 이 sim 은 x769_pairs 핀이 아니다)")
        for i, m in enumerate(s.get("messages") or []):
            for tc in (m.get("tool_calls") or []):
                nm = tc.get("name")
                ar = tc.get("arguments")
                if isinstance(ar, str):
                    try:
                        ar = json.loads(ar)
                    except Exception:
                        ar = {"_raw": ar}
                inner = (ar or {}).get("agent_tool_name") or (ar or {}).get("user_tool_name") \
                    or (ar or {}).get("discoverable_tool_name")
                sub = (ar or {}).get("arguments")
                if isinstance(sub, str):
                    try:
                        sub = json.loads(sub)
                    except Exception:
                        pass
                shown = sub if isinstance(sub, dict) else ar
                eff = inner or nm
                print("   [%d] name=%-32s eff=%-40s args=%s"
                      % (i, nm, eff, json.dumps(shown, ensure_ascii=False)[:320]))
