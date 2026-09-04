# x766c — dump feedback sidecar entries verbatim for one sim tag (read-only).
import gzip, json, os, sys

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
fn = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else None
sub = sys.argv[3] if len(sys.argv) > 3 else None   # substring filter on text

with gzip.open(os.path.join(ROOT, fn), "rt", encoding="utf-8", errors="replace") as fh:
    for ln, line in enumerate(fh):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if tag and o.get("simtag") != tag:
            continue
        txt = o.get("text") or ""
        if sub and sub not in txt:
            continue
        print("=" * 100)
        print("[line %d] kind=%s sim=%s turn=%s channel=%s call_name=%s len=%s sha=%s cached=%s"
              % (ln, o.get("kind"), o.get("sim"), o.get("turn"), o.get("channel"),
                 o.get("call_name"), o.get("len"), o.get("sha"), o.get("cached")))
        print(txt)
