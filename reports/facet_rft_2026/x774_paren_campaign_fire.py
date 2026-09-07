# -*- coding: utf-8 -*-
"""x774 — 캠페인(ours viewmax2 · 97) 회수분에서 T2_SIBLING_PAREN 술어 발화 실측.
엔진 사본 0: t2_gate_patch.sibling_paren_arg 를 그대로 임포트. CPU · 모델 0 · GPU 0.
"""
import os, sys, gzip, json, glob, datetime
sys.path.insert(0, r"C:\workspace\ba-frft\scripts\distill\tau2")
os.environ.setdefault("T2_IMPORT_ONLY", "1")
from t2_gate_patch import sibling_paren_arg, _exact_tool_name  # noqa

SR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"

class TC(object):
    def __init__(self, name, args):
        self.name = name; self.arguments = args
        self.function = type("F", (), {"name": name, "arguments": args})()

def load(p):
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        return json.load(f)

def ts_of(sim):
    for k in ("start_time", "timestamp", "created_at", "end_time"):
        v = sim.get(k)
        if v: return str(v)
    return ""

best = {}   # task -> (ts, tag, sim)
files = sorted(glob.glob(os.path.join(SR, "bank_*.results.json.gz")))
for p in files:
    tag = os.path.basename(p).replace(".results.json.gz", "")
    try: d = load(p)
    except Exception: continue
    sims = d.get("simulations") or d.get("results") or []
    if not isinstance(sims, list): continue
    for s in sims:
        if not isinstance(s, dict): continue
        t = s.get("task_id") or s.get("task") or ""
        t = str(t)
        ts = ts_of(s)
        if not (ts >= "2026-09-03T13" and ts <= "2026-09-05T03:30"): continue
        cur = best.get(t)
        if cur is None or ts > cur[0]:
            best[t] = (ts, tag, s)

def rw(s):
    r = s.get("reward_info") or {}
    v = r.get("reward", s.get("reward"))
    try: return float(v)
    except Exception: return None

fires = {}
for t, (ts, tag, s) in sorted(best.items()):
    msgs = s.get("messages") or []
    hits = []
    for m in msgs:
        if (m.get("role") or "") != "assistant": continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            nm = fn.get("name") or tc.get("name")
            ar = fn.get("arguments", tc.get("arguments"))
            o = TC(nm, ar)
            try: sp = sibling_paren_arg(o)
            except Exception: sp = None
            if sp: hits.append(sp)
    if hits: fires[t] = (rw(s), tag, hits)

print("campaign tasks=%d pass=%d fail=%d" % (
    len(best), sum(1 for t,(a,b,s) in best.items() if (rw(s) or 0) >= 1.0),
    sum(1 for t,(a,b,s) in best.items() if (rw(s) or 0) < 1.0)))
print("SIBLING_PAREN fires on %d tasks" % len(fires))
for t, (r, tag, hits) in sorted(fires.items()):
    print("  %-10s reward=%s tag=%s n=%d" % (t, r, tag, len(hits)))
    for h in hits[:4]:
        print("      tool=%s arg=%s val=%r token=%r" % (h[0], h[1], str(h[2])[:70], h[3]))
