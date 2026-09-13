# -*- coding: utf-8 -*-
"""Lever census for the LB code base: every sidecar lever (advice/deny/block/tool/inject/fold/regen)
joined to the reward of the simulation it fired in, pooled over every arm in out_lb, paired within
task (fired sims vs quiet sims of the same task and arm), gold never read.

  python lb_lever_census.py <out_lb dir> <base sim_results dir> <out prefix>

Writes <out>.json (everything) and <out>.txt (the tables). Read-only over the inputs.
"""
import collections, glob, gzip, json, os, re, sys

O, D, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SKIP = (".infra_void", ".run1", ".partial", ".first", ".pre_", ".taint", ".nt2", "x818cloud")
SKIP_PRE = {"smk", "probe", "dx", "dbg"}
TAG = re.compile(r"\[([A-Z][A-Z_ -]{2,30})\]")


def load(p):
    return json.load(gzip.open(p, "rt", encoding="utf-8"))


def base_scores():
    out = {}
    for p in glob.glob(D + "/bank_x806_base_nt4_task_*.results.json.gz"):
        if "x818cloud" in p:
            continue
        t = p.split("task_")[-1].split(".")[0]
        s = load(p)["simulations"]
        out[t] = (sum(1 for x in s if (x.get("reward_info") or {}).get("reward") == 1.0), len(s))
    return out


def lever_keys(row):
    k, src, lb = row.get("kind"), str(row.get("source") or ""), str(row.get("lb") or "")
    text = str(row.get("text") or "")
    tag = TAG.search(text)
    tag = tag.group(1) if tag else ""
    keys = []
    if k in ("lb-advice", "lb-deny", "lb-block", "lb-release"):
        keys.append("%s:%s:%s" % (k[3:], lb or "?", src.split(":")[0] or tag or "?"))
        if tag:
            keys.append("%s:tag:%s" % (k[3:], tag))
    elif k == "lb-tool":
        keys.append("tool:LB2:%s" % (src or "?"))
    elif k in ("lb-inject", "lb-fold", "lb-regen", "lb-tools", "lb-ask"):
        keys.append(k[3:])
    return keys


base = base_scores()
sims = []            # (arm, task, sim_id, reward, set(keys), n_rows)
arm_task = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
join = collections.Counter()
for rp in sorted(glob.glob(O + "/*_task_*.results.json.gz")):
    b = os.path.basename(rp)
    if any(x in b for x in SKIP) or b.startswith("bank_x806"):
        continue
    pre, t = b.rsplit("_task_", 1)[0], b.split("_task_")[-1].split(".")[0]
    if pre in SKIP_PRE:
        continue
    try:
        d = load(rp)
    except Exception:
        join["bad_results"] += 1
        continue
    ss = d.get("simulations") or []
    if any(x.get("termination_reason") == "infrastructure_error" for x in ss):
        join["infra_arm_task"] += 1
        continue
    side = collections.defaultdict(list)
    sp = O + "/fb_%s_task_%s.jsonl.gz" % (pre, t)
    if os.path.exists(sp):
        for line in gzip.open(sp, "rt", encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            side[str(r.get("sim"))].append(r)
    else:
        join["no_sidecar_file"] += 1
    for s in ss:
        rw = 1 if (s.get("reward_info") or {}).get("reward") == 1.0 else 0
        rows = side.get(str(s.get("id")), [])
        keys = set(k for r in rows for k in lever_keys(r))
        sims.append((pre, t, s.get("id"), rw, keys, len(rows)))
        arm_task[pre][t][0] += rw
        arm_task[pre][t][1] += 1
        join["sims"] += 1
        join["sims_with_rows"] += 1 if rows else 0

# ---- per-lever: pooled and within (arm, task) pairing
levers = sorted({k for _, _, _, _, ks, _ in sims for k in ks})
by_cell = collections.defaultdict(list)          # (arm, task) -> [(rw, keys)]
for pre, t, _, rw, ks, _ in sims:
    by_cell[(pre, t)].append((rw, ks))
table = []
for L in levers:
    fw = fl = qw = ql = 0
    pf = pq = 0.0
    npairs = better = worse = 0
    arms = set()
    for (pre, t), cell in by_cell.items():
        f = [rw for rw, ks in cell if L in ks]
        q = [rw for rw, ks in cell if L not in ks]
        if f:
            arms.add(pre)
        fw += sum(f); fl += len(f) - sum(f); qw += sum(q); ql += len(q) - sum(q)
        if f and q:
            npairs += 1
            rf, rq = sum(f) / len(f), sum(q) / len(q)
            pf += rf; pq += rq
            better += rf > rq
            worse += rf < rq
    nf, nq = fw + fl, qw + ql
    table.append(dict(lever=L, fired=nf, fired_win=fw, quiet=nq, quiet_win=qw,
                      fired_pct=round(100.0 * fw / nf, 1) if nf else None,
                      quiet_pct=round(100.0 * qw / nq, 1) if nq else None,
                      pairs=npairs, paired_delta_pp=round(100.0 * (pf - pq) / npairs, 1) if npairs else None,
                      better=better, worse=worse, arms=sorted(arms)))

# ---- per-arm vs base on the same tasks (4-sim cells only)
arm_rows = []
for pre, tasks in arm_task.items():
    ours = base_sum = n = 0
    per = {}
    for t, (p, m) in tasks.items():
        if m != 4 or base.get(t, (0, 0))[1] != 4:
            continue
        ours += p; base_sum += base[t][0]; n += 1
        per[t] = [p, base[t][0]]
    if n:
        arm_rows.append(dict(arm=pre, tasks=n, ours=ours, base=base_sum, delta=ours - base_sum, per_task=per))
arm_rows.sort(key=lambda r: -r["tasks"])

# ---- task pooled: our sims over every arm vs base
pooled = collections.defaultdict(lambda: [0, 0])
for pre, t, _, rw, _, _ in sims:
    pooled[t][0] += rw; pooled[t][1] += 1

json.dump(dict(join=join, levers=table, arms=arm_rows, base=base,
               pooled={t: v for t, v in pooled.items()}), open(OUT + ".json", "w"), ensure_ascii=False, indent=1)
with open(OUT + ".txt", "w", encoding="utf-8") as f:
    w = lambda *a: print(*a, file=f)
    w("JOIN", dict(join))
    w("\n== ARMS vs base (same tasks, 4-sim cells) ==")
    for r in arm_rows:
        w("%-8s tasks %3d  ours %3d  base %3d  delta %+d" % (r["arm"], r["tasks"], r["ours"], r["base"], r["delta"]))
    w("\n== LEVERS (pooled over arms; paired within arm x task) ==")
    w("%-48s %6s %6s | %6s %6s | %5s %7s %4s %4s  arms" % ("lever", "fired", "win%", "quiet", "win%", "pairs", "dpp", "bet", "wor"))
    for r in sorted(table, key=lambda r: -r["fired"]):
        w("%-48s %6d %6s | %6d %6s | %5d %7s %4d %4d  %s" % (
            r["lever"], r["fired"], r["fired_pct"], r["quiet"], r["quiet_pct"], r["pairs"],
            r["paired_delta_pp"], r["better"], r["worse"], ",".join(r["arms"][:12]) + ("…" if len(r["arms"]) > 12 else "")))
print(open(OUT + ".txt", encoding="utf-8").read())
