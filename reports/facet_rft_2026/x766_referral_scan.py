# x766d — for each task_010 sim in the named bundles, list every submit_referral tool_call
# with its requestor + args, plus message key inventory. Read-only.
import gzip, json, os

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
BUNDLES = [
    "bank_010ctl_20260904_0007.results.json.gz",
    "bank_010treat_20260903_2152.results.json.gz",
    "bank_g97151p11_viewmax2_20260903_1924.results.json.gz",
    "bank_night2p1_t3prime_20260901_2341.results.json.gz",
    "bank_x723_t3B_viewscale_max_20260901_1106.results.json.gz",
]

for fn in BUNDLES:
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p):
        print("MISSING %s" % fn); continue
    with gzip.open(p, "rb") as fh:
        obj = json.load(fh)
    for s in obj["simulations"]:
        if s.get("task_id") != "task_010":
            continue
        msgs = s.get("messages") or []
        print("=" * 100)
        print("%s | seed=%s trial=%s reward=%s nmsgs=%d term=%s"
              % (fn, s.get("seed"), s.get("trial"), (s.get("reward_info") or {}).get("reward"),
                 len(msgs), s.get("termination_reason")))
        keys = set()
        for m in msgs:
            keys |= set(m.keys())
        print("  message keys: %s" % sorted(keys))
        # turn field sanity
        mism = [i for i, m in enumerate(msgs) if m.get("turn_idx") != i]
        print("  turn_idx != index at: %s" % (mism[:10] if mism else "NONE (turn_idx == message index for all)"))
        for i, m in enumerate(msgs):
            for c in (m.get("tool_calls") or []):
                nm = c.get("name")
                if "referral" in str(nm):
                    print("  [msg %d] role=%s requestor=%s name=%s args=%s"
                          % (i, m.get("role"), c.get("requestor"), nm,
                             json.dumps(c.get("arguments"), ensure_ascii=False)))
        # reward detail
        ri = s.get("reward_info") or {}
        db = ri.get("db_check") or ri.get("reward_breakdown") or {}
        print("  reward_info keys: %s" % sorted(ri.keys()))
        print("  db_check: %s" % json.dumps(db, ensure_ascii=False)[:1500])
