# x766 — P10 (D16) trajectory strand: locate every recovered result bundle containing task_010.
# Read-only scan. No prompt authoring. [[71]] materials come from declared files only.
import gzip, json, os, sys, io

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
NEEDLE = b'"task_010"'

hits = []
files = sorted(f for f in os.listdir(ROOT) if f.endswith(".json.gz"))
for fn in files:
    p = os.path.join(ROOT, fn)
    try:
        with gzip.open(p, "rb") as fh:
            blob = fh.read()
    except Exception as e:
        continue
    if NEEDLE not in blob:
        continue
    try:
        obj = json.loads(blob.decode("utf-8", "replace"))
    except Exception as e:
        hits.append((fn, "PARSE_FAIL:%s" % e, None))
        continue
    sims = obj.get("simulations") if isinstance(obj, dict) else None
    if sims is None:
        hits.append((fn, "NO_SIMULATIONS_KEY keys=%s" % (list(obj)[:12] if isinstance(obj, dict) else type(obj)), None))
        continue
    rows = []
    for i, s in enumerate(sims):
        tid = s.get("task_id")
        if tid != "task_010":
            continue
        rw = s.get("reward_info", {}) or {}
        rows.append(dict(
            idx=i,
            sim_id=s.get("id"),
            trial=s.get("trial"),
            reward=rw.get("reward"),
            nmsgs=len(s.get("messages") or []),
            term=s.get("termination_reason"),
            seed=s.get("seed", "<absent>"),
        ))
    if rows:
        info = obj.get("info", {}) or {}
        hits.append((fn, info, rows))

for fn, info, rows in hits:
    if rows is None:
        print("== %-60s  %s" % (fn, info))
        continue
    meta = ""
    if isinstance(info, dict):
        gi = info.get("git_commit") or info.get("engine_sha") or ""
        ts = info.get("timestamp") or ""
        agent = json.dumps(info.get("agent_info", {}), ensure_ascii=False)[:160]
        meta = "git=%s ts=%s agent=%s" % (gi, ts, agent)
    print("== %s" % fn)
    print("   info: %s" % meta)
    for r in rows:
        print("   task_010 idx=%s sim_id=%s trial=%s reward=%s nmsgs=%s term=%s seed=%s"
              % (r["idx"], r["sim_id"], r["trial"], r["reward"], r["nmsgs"], r["term"], r["seed"]))
print("\nTOTAL FILES SCANNED: %d ; FILES WITH task_010: %d" % (len(files), len([h for h in hits if h[2]])))
