# x766h — inventory every env-document occurrence of `submit_referral` that reached the agent
# as a tool observation (KB retrieval output), across the 010 bundles. Read-only.
import gzip, json, os, re

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
BUNDLES = [
    "bank_010ctl_20260904_0007.results.json.gz",
    "bank_010treat_20260903_2152.results.json.gz",
    "bank_g97151p11_viewmax2_20260903_1924.results.json.gz",
    "bank_night2p1_t3prime_20260901_2341.results.json.gz",
    "bank_x723_t3B_viewscale_max_20260901_1106.results.json.gz",
]

seen = {}
for fn in BUNDLES:
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p):
        continue
    with gzip.open(p, "rb") as fh:
        obj = json.load(fh)
    for s in obj["simulations"]:
        if s.get("task_id") != "task_010":
            continue
        for i, m in enumerate(s.get("messages") or []):
            if m.get("role") != "tool":
                continue
            c = m.get("content") or ""
            for k in [mo.start() for mo in re.finditer("submit_referral", c)]:
                # find the enclosing doc id, searching backwards for "ID: doc_"
                back = c[:k]
                mid = None
                mm = list(re.finditer(r"ID: (doc_[^\s]+)", back))
                if mm:
                    mid = mm[-1].group(1)
                snip = c[max(0, k - 500): k + 500]
                key = (mid, snip[:200])
                if key in seen:
                    continue
                seen[key] = True
                print("=" * 100)
                print("%s | seed=%s | msg %d | doc=%s" % (fn, s.get("seed"), i, mid))
                print(snip)
