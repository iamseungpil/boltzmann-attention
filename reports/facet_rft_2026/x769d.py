# -*- coding: utf-8 -*-
import glob, gzip, json, os, sys, collections, re
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
WR = ("file_credit_card_transaction_dispute", "file_debit_card_transaction_dispute", "submit_cash_back_dispute")
def eff(tc):
    nm = str(tc.get("name") or "")
    if nm.startswith("call_"):
        ar = tc.get("arguments") or {}
        if isinstance(ar, str):
            try: ar = json.loads(ar)
            except Exception: ar = {}
        inner = ar.get("agent_tool_name") or ar.get("user_tool_name") or ar.get("discoverable_tool_name") or ""
        if inner: return re.sub(r"_\d+$", "", str(inner))
    return re.sub(r"_\d+$", "", nm)
LO, HI = "2026-09-03T14", "2026-09-05T04"
best = {}
for p in glob.glob(os.path.join(ROOT, "*.results.json.gz")):
    try: d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    except Exception: continue
    for s in (d.get("simulations") or []):
        st = str(s.get("start_time") or "")
        if not (LO <= st <= HI): continue
        t = s.get("task_id")
        if not t: continue
        if t not in best or st > best[t][0]: best[t] = (st, os.path.basename(p), s)
out = []
for t, (st, tag, s) in sorted(best.items()):
    c = collections.Counter()
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            e = eff(tc)
            if e in WR: c[e] += 1
    r = (s.get("reward_info") or {}).get("reward")
    out.append((t, r, dict(c), s.get("id"), tag))
print("로컬 회수 태스크 %d  (pass %d)" % (len(out), sum(1 for x in out if (x[1] or 0) >= 1.0)))
print()
print("=== 분쟁 write 호출 실측 ===")
tot = collections.Counter(); nt = 0
for t, r, c, sid, tag in out:
    if c:
        nt += 1; tot.update(c)
        print("  %-10s reward=%-5s %r" % (t, r, c))
print("\n태스크 %d · 총계 %r" % (nt, dict(tot)))
