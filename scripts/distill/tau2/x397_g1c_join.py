# -*- coding: utf-8 -*-
r"""x397 G1-c join - exposure vs hit, and B_full truncation audit."""
import io, json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import t2_forensic as F
import x397_argexposure as G
import x397_g1c_report as R

ARMS = ["A_min", "B_tail4", "B_tail8", "B_tail16", "B_tail32", "B_full"]
RES = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x395_compliance_iso.json"

docs, TOOLS, cases = G.build_cases(12)
res = json.load(io.open(RES, encoding="utf-8"))
hit = collections.defaultdict(lambda: [0, 0])
for r in res:
    if r["mode"] != "next":
        continue
    k = (r["task"], r["tool"], r["arm"])
    hit[k][0] += 1 if r["hit_exact"] else 0
    hit[k][1] += 1

print("## message counts / B_full truncation (maxmsg=60, tool result cut=400)")
for c in cases:
    msgs = c["sim"].get("messages") or []
    ntool = sum(1 for m in msgs if m.get("role") == "tool")
    longcut = sum(1 for m in msgs if m.get("role") == "tool" and len(" ".join(str(m.get("content") or "").split())) > 400)
    print("  %-9s %-38s msgs=%-3d tool_msgs=%-3d cut>400=%-3d B_full_truncated=%s"
          % (c["task"], c["tool"][:38], len(msgs), ntool, longcut, len(msgs) > 60))

print("")
print("## exposure(ID args) vs hit_exact (n=3 per cell)")
print("%-9s %-38s %s" % ("task", "tool", " ".join("%-13s" % a for a in ARMS)))
pairs = collections.defaultdict(list)
for c in cases:
    ga = G.gold_args_for(c["sim"], c["tool"])
    P = G.build_prompts(c, TOOLS)
    vals = [v for v in R.gold_values(ga) if v[3] == "ID"]
    cells = []
    for a in ARMS:
        p = P[a]
        e = sum(1 for (k, s, v, cl) in vals if R.present(p, s, v)) / float(max(1, len(vals)))
        h = hit[(c["task"], c["tool"], a)]
        cells.append("%-13s" % ("e=%.2f h=%d/%d" % (e, h[0], h[1])))
        pairs[a].append((e, h[0] / float(max(1, h[1]))))
    print("%-9s %-38s %s" % (c["task"], c["tool"][:38], " ".join(cells)))

print("")
print("## per-arm: mean ID-exposure vs mean hit, and within-arm corr(exposure,hit) over 12 targets")
allpts = []
for a in ARMS:
    xs = [p[0] for p in pairs[a]]
    ys = [p[1] for p in pairs[a]]
    allpts += pairs[a]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    print("  %-9s exposure=%.2f  hit=%.2f  r=%.2f" % (a, mx, my, r))
xs = [p[0] for p in allpts]
ys = [p[1] for p in allpts]
mx = sum(xs) / len(xs)
my = sum(ys) / len(ys)
sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
print("  POOLED n=%d  r=%.2f" % (len(allpts), cov / (sx * sy)))

print("")
print("## split by exposure: hit rate when ID args fully exposed vs not (pooled over arms)")
hi = [y for x, y in allpts if x >= 0.999]
lo = [y for x, y in allpts if x < 0.999]
print("  exposed=1.00: n=%d hit=%.2f | exposed<1.00: n=%d hit=%.2f"
      % (len(hi), sum(hi) / len(hi), len(lo), sum(lo) / len(lo)))

print("")
print("## distractor mass: distinct known tool names appearing in the BODY (outside the tool list)")
for a in ARMS + ["C_neg"]:
    tot = []
    for c in cases:
        P = G.build_prompts(c, TOOLS)
        body = P[a].split("\n\n", 1)[1]
        tot.append(len(set(t for t in TOOLS if t in body)))
    print("  %-9s mean=%.1f [%d-%d]" % (a, sum(tot) / float(len(tot)), min(tot), max(tot)))
