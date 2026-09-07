# -*- coding: utf-8 -*-
"""x769c - 캠페인 창(09-03T14 ~ 09-05T03) 재구성 + 분쟁 write 도구 호출 실측. 세기만."""
import glob, gzip, json, os, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
LO, HI = "2026-09-03T14", "2026-09-05T03"
best = {}
for p in glob.glob(os.path.join(ROOT, "*.results.json.gz")):
    try: d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    except Exception: continue
    for s in (d.get("simulations") or []):
        st = str(s.get("start_time") or "")
        if not (LO <= st <= HI + "\uffff"): continue
        t = s.get("task_id")
        if not t: continue
        cur = best.get(t)
        if cur is None or st > cur[0]:
            best[t] = (st, os.path.basename(p), s)
print("태스크 %d" % len(best))
rew = {t: ((v[2].get("reward_info") or {}).get("reward")) for t, v in best.items()}
npass = sum(1 for t in rew if (rew[t] or 0) >= 1.0)
print("pass %d · fail %d" % (npass, len(best) - npass))
json.dump({t: [v[0], v[1], v[2].get("id")] for t, v in best.items()},
          open(r"C:\Users\승원\AppData\Local\Temp\claude\C--workspace\7fc6c1a1-f227-4592-be9c-44f0ba6cffac\scratchpad\campaign_pins.json","w"), indent=0)
# 분쟁 write 도구 호출
WR = ("file_credit_card_transaction_dispute", "file_debit_card_transaction_dispute", "submit_cash_back_dispute")
rows = []
for t, (st, tag, s) in sorted(best.items()):
    cnt = collections.Counter()
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = str(tc.get("name") or "")
            args = tc.get("arguments") or {}
            eff = nm
            if isinstance(args, dict):
                for k in ("tool_name","name","tool"):
                    if isinstance(args.get(k), str): eff = args[k]; break
            for w in WR:
                if eff == w or eff.startswith(w): cnt[w] += 1
    rows.append((t, rew.get(t), tag, dict(cnt)))
print()
print("=== 분쟁 write 호출이 있는 태스크 (실측) ===")
for t, r, tag, c in rows:
    if c: print("  %-10s reward=%-5s %-46s %r" % (t, r, tag[:46], c))
print()
tot = collections.Counter()
for _, _, _, c in rows: tot.update(c)
print("총계:", dict(tot))
