# x766e — for every task_010 sim in the campaign+control bundles: find each [ACTION] feedback
# entry in the fb sidecar, then print the message at that same index and the following two.
import gzip, json, os

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
PAIRS = [
    ("bank_010ctl_20260904_0007", "campaign ctl (viewmax2, engine_sha a208c8e0)"),
    ("bank_010treat_20260903_2152", "campaign treat (viewmax2_actdemand, engine_sha c9076674)"),
    ("bank_g97151p11_viewmax2_20260903_1924", "campaign g97 (viewmax2, engine_sha a208c8e0)"),
    ("bank_night2p1_t3prime_20260901_2341", "prior PASS (t3prime, no provenance line)"),
    ("bank_x723_t3B_viewscale_max_20260901_1106", "prior PASS (t3B viewscale_max, engine_sha bd9dee5d)"),
]

def clip(s, n=700):
    s = s or ""
    return s if len(s) <= n else s[:n] + "  …[clipped %d chars]" % (len(s) - n)

for base, label in PAIRS:
    rp = os.path.join(ROOT, base + ".results.json.gz")
    fp = os.path.join(ROOT, "fb_" + base + ".jsonl.gz")
    print("\n" + "#" * 110)
    print("# %s  — %s" % (base, label))
    if not os.path.exists(rp):
        print("  MISSING results"); continue
    with gzip.open(rp, "rb") as fh:
        obj = json.load(fh)
    sims = [s for s in obj["simulations"] if s.get("task_id") == "task_010"]
    fbs = []
    if os.path.exists(fp):
        with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    fbs.append(json.loads(line))
                except Exception:
                    pass
    else:
        print("  (no fb sidecar)")
    for s in sims:
        seed = s.get("seed")
        msgs = s.get("messages") or []
        print("\n  --- seed=%s trial=%s reward=%s nmsgs=%d term=%s"
              % (seed, s.get("trial"), (s.get("reward_info") or {}).get("reward"),
                 len(msgs), s.get("termination_reason")))
        tag = "task_010#s%s" % seed
        acts = [o for o in fbs if o.get("simtag") == tag and "[ACTION]" in (o.get("text") or "")]
        print("      [ACTION] fb entries: turns=%s  fb_sim_ids=%s"
              % ([o.get("turn") for o in acts], sorted({o.get("sim") for o in acts})))
        for o in acts:
            t = int(o.get("turn"))
            head = (o.get("text") or "").split("\n")[0]
            print("      * [ACTION] at fb turn=%d :: %s" % (t, clip(head, 400)))
            for j in (t, t + 1, t + 2):
                if 0 <= j < len(msgs):
                    m = msgs[j]
                    tcs = [{"name": c.get("name"), "requestor": c.get("requestor"),
                            "arguments": c.get("arguments")} for c in (m.get("tool_calls") or [])]
                    print("        [msg %d] role=%s tool_calls=%s" % (j, m.get("role"),
                          json.dumps(tcs, ensure_ascii=False) if tcs else "-"))
                    if m.get("content"):
                        print("           content: %s" % clip(m["content"].strip(), 700).replace("\n", "\n           "))
                else:
                    print("        [msg %d] OUT OF RANGE (nmsgs=%d)" % (j, len(msgs)))
