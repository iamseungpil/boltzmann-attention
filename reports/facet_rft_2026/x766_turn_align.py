# x766f — test whether the fb sidecar `turn` equals the index of the regenerated assistant
# message, by comparing the pre-regen draft (kind=reminder-assistant) against the persisted
# message at index turn, turn-1, turn+1.
import gzip, json, os

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
CASES = [
    ("bank_010ctl_20260904_0007", 373753),
    ("bank_night2p1_t3prime_20260901_2341", 626729),
    ("bank_x723_t3B_viewscale_max_20260901_1106", 626729),
]

for base, seed in CASES:
    with gzip.open(os.path.join(ROOT, base + ".results.json.gz"), "rb") as fh:
        obj = json.load(fh)
    sim = [s for s in obj["simulations"]
           if s.get("task_id") == "task_010" and s.get("seed") == seed][0]
    msgs = sim.get("messages") or []
    fbs = []
    with gzip.open(os.path.join(ROOT, "fb_" + base + ".jsonl.gz"), "rt",
                   encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    fbs.append(json.loads(line))
                except Exception:
                    pass
    tag = "task_010#s%d" % seed
    print("\n#### %s seed=%s nmsgs=%d" % (base, seed, len(msgs)))
    for i, m in enumerate(msgs):
        print("   idx %2d role=%-9s len(content)=%s tool_calls=%d"
              % (i, m.get("role"), len(m.get("content") or ""), len(m.get("tool_calls") or [])))
    for o in fbs:
        if o.get("simtag") != tag or o.get("kind") != "reminder-assistant":
            continue
        t = int(o.get("turn"))
        draft = o.get("text") or ""
        print("   >> reminder-assistant turn=%d draft_len=%d  (declared len=%s)"
              % (t, len(draft), o.get("len")))
        for j in (t - 1, t, t + 1):
            if 0 <= j < len(msgs):
                c = msgs[j].get("content") or ""
                print("      cmp idx %d role=%s len=%d identical=%s prefix_match=%s"
                      % (j, msgs[j].get("role"), len(c), c == draft,
                         (c[:80] == draft[:80]) if c and draft else False))
