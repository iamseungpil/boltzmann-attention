"""Evidence-(B), VERIFIED shipped schema (remote-confirmed 2026-06-01) — run from a SOPBench
clone root (e.g. /home/woori/scratch/SOPBench).

Shipped file `output/<domain>/ast_<model>-mode_...-shuffle_False.json` = a LIST of records;
each record = {domain, setup, task, interactions, evaluations, statistics} where
  record["task"]["action_should_succeed"] -> bool
  record["task"]["user_goal"]             -> goal name
  record["evaluations"]                   -> list of eval dicts, each with "success" + 5
                                             sub-checks (no_tool_call_error,
                                             constraint_not_violated, database_match,
                                             action_called_correctly, dirgraph_satisfied)

Align records across model files by (user_goal, occurrence-index within that file's goal);
for each should_succeed=True instance ask: did ANY model ever get success==True? Instances
never passed by any model (with enough coverage) are candidate "impossible" tasks (strong
corroboration, NOT proof — the decisive test is oracle-replay = evidence A).

RUN (clone root):  python scripts/offline_crosscheck.py --domain bank --out /tmp/xc.json
"""
import argparse, glob, json, os, collections

SUB = ["no_tool_call_error", "constraint_not_violated", "database_match",
       "action_called_correctly", "dirgraph_satisfied"]


def rec_passed(rec):
    for e in rec.get("evaluations", []) or []:
        if isinstance(e, dict) and e.get("success"):
            return True
    return False


def rec_fail_counter(rec):
    c = collections.Counter()
    for e in rec.get("evaluations", []) or []:
        if isinstance(e, dict):
            for s in SUB:
                if s in e and not e[s]:
                    c[s] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bank")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_models", type=int, default=5)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join("output", args.domain, "ast_*.json")))
    print("ast files:", len(files))

    agg = collections.defaultdict(lambda: {"any": False, "seen": 0,
                                           "fail": collections.Counter(), "passers": []})
    scanned = 0
    for f in files:
        model = os.path.basename(f)[len("ast_"):].split("-mode_")[0]
        try:
            recs = json.load(open(f))
        except Exception as e:
            print("  skip", f, e)
            continue
        if not isinstance(recs, list):
            continue
        per_goal = collections.Counter()
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            t = rec.get("task", {})
            if t.get("action_should_succeed") is not True:
                continue
            g = t.get("user_goal")
            idx = per_goal[g]; per_goal[g] += 1
            scanned += 1
            c = agg[(g, idx)]; c["seen"] += 1
            if rec_passed(rec):
                c["any"] = True
                if model not in c["passers"]:
                    c["passers"].append(model)
            else:
                c["fail"] += rec_fail_counter(rec)

    print("should_succeed=True instance-records scanned:", scanned)
    print("distinct (goal,idx) instances:", len(agg))

    never = {k: v for k, v in agg.items() if not v["any"] and v["seen"] >= args.min_models}
    never_goals = sorted({k[0] for k in never})
    passed_goals = sorted({k[0] for k, v in agg.items() if v["any"]})

    print("NEVER passed by any model (seen>=%d): %d instances / %d goals"
          % (args.min_models, len(never), len(never_goals)))
    for k in sorted(never):
        v = never[k]
        print("   %s idx%d seen=%d fail=%s"
              % (k[0], k[1], v["seen"], dict(v["fail"].most_common(3))))
    print("goals passed by >=1 model: %d" % len(passed_goals))
    print("  ", passed_goals)

    if args.out:
        ser = {"%s|%d" % (g, j): {"seen": v["seen"], "any": v["any"],
                                  "passers": v["passers"], "fail": dict(v["fail"])}
               for (g, j), v in agg.items()}
        json.dump({"domain": args.domain, "scanned": scanned, "n_instances": len(agg),
                   "never_passed": len(never), "never_goals": never_goals,
                   "passed_goals": passed_goals, "instances": ser},
                  open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
