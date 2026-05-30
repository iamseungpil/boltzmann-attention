#!/usr/bin/env python3
"""induce_observation_triggers.py — ABox for a DETERMINISTIC executor (design §15.11).

The hierarchical planner emits an abstract step; the executor must pick the CONCRETE
tool. A step is underdetermined (apply_targeted_fix -> {enable_roaming, refuel_data}); the
disambiguator is the OBSERVED STATE from diagnostic reads. We declare that as ontology
(no executor training needed):

  observation_triggers(read, field, value, fix_tool)   state predicate -> tool
  arg_source(fix_tool, param, read.field)              tool arg <- prior read output field

Both are induced (domain-general) from teacher SUCCESS trajectories: the read-result
fields immediately preceding each GT fix-write, and the provenance of each write's args.
Output: induced/obs_triggers_<domain>.json. The executor = step_realizes_tool^-1 (candidate
tools) x observation_triggers (pick by state match) x arg_source (fill args).

Usage:
  python scripts/distill/induce_observation_triggers.py --domain telecom
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

ID_KEY_RE = re.compile(r"(^|\.)(id|.*_id)$", re.I)
ID_VAL_RE = re.compile(r"^[A-Z]{1,3}\d{2,}$")  # P1002, D1002, C1001, B1003...
# free-text / instance attributes that are NOT generalizable state predicates
NONSTATE_KEY_RE = re.compile(r"name|address|email|phone|date|period|_at$|descr|title|"
                             r"street|city|zip|first|last|number|amount|price|cost", re.I)
DATE_VAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _is_id_field(key, val):
    return bool(ID_KEY_RE.search(key)) or bool(ID_VAL_RE.match(str(val)))


def _is_state_field(key, val):
    """Keep only low-cardinality categorical/boolean STATE (roaming_enabled, status),
    drop instance IDs, names, addresses, dates, free text, amounts."""
    if _is_id_field(key, val) or NONSTATE_KEY_RE.search(key):
        return False
    s = str(val)
    if " " in s or DATE_VAL_RE.match(s) or len(s) > 16:
        return False
    return True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fix_coverage import is_read, _shipped_files  # noqa: E402
from procedure_scorecard import gt_agent_actions  # noqa: E402

DEFAULT_TAU2 = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"


def reward_of(s):
    return (s.get("reward_info") or {}).get("reward", 0) or 0


def flatten(obj, prefix="", out=None, depth=0):
    """Flatten dict/list to {dotted_key: scalar_str}, scalars only, depth<=3."""
    if out is None:
        out = {}
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}{k}." if prefix == "" else f"{prefix}{k}.", out, depth + 1) \
                if isinstance(v, (dict, list)) else out.__setitem__(f"{prefix}{k}", _scalar(v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            if isinstance(v, (dict, list)):
                flatten(v, f"{prefix}{i}.", out, depth + 1)
            else:
                out[f"{prefix}{i}"] = _scalar(v)
    return out


def _scalar(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float, str)):
        s = str(v)
        return s if len(s) <= 40 else None
    return None


def parse_tool_content(c):
    try:
        d = json.loads(c) if isinstance(c, str) else c
    except Exception:
        return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2)
    ap.add_argument("--shipped-dir", default=DEFAULT_TAU2 + "/data/tau2/results/final")
    ap.add_argument("--min-support", type=int, default=4)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--min-precision", type=float, default=0.8,
                    help="P(tool|pred): keep only state predicates that discriminate the tool")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = json.load(open(os.path.join(args.tau2_root, "data", "tau2", "domains",
                                        args.domain, "tasks.json")))
    gt_writes = set()
    for t in tasks:
        for a in gt_agent_actions(t):
            gt_writes.add(a["name"])

    # counts
    trig = defaultdict(Counter)     # fix_tool -> Counter("read|field=value")
    pred_tool = defaultdict(Counter)  # pred -> Counter(fix_tool)  (for precision/discrimination)
    fix_n = Counter()               # fix_tool -> occurrences (with a preceding read)
    argsrc = defaultdict(Counter)   # (fix_tool, param) -> Counter("read.field")
    arg_n = Counter()               # (fix_tool, param) -> occurrences

    for f in _shipped_files(args.domain, args.shipped_dir):
        try:
            sims = json.load(open(f)).get("simulations", [])
        except Exception:
            continue
        for s in sims:
            if reward_of(s) < 0.999:
                continue
            id2name = {}
            recent_fields = {}     # field-path -> value (from reads so far)
            recent_by_read = []    # list of (read_name, {field:value})
            last_read = None
            last_read_fields = {}
            for m in s.get("messages") or []:
                r = m.get("role")
                if r == "assistant":
                    for tc in m.get("tool_calls") or []:
                        id2name[tc.get("id")] = tc.get("name")
                        name = tc.get("name")
                        if not name or tc.get("requestor", "assistant") != "assistant":
                            continue
                        if not is_read(name) and name in gt_writes:
                            # FIX write: record trigger from last read's fields
                            if last_read is not None and last_read_fields:
                                fix_n[name] += 1
                                for fld, val in last_read_fields.items():
                                    if val is None or not _is_state_field(fld, val):
                                        continue  # keep only categorical/boolean STATE
                                    pred = f"{last_read}|{fld}={val}"
                                    trig[name][pred] += 1
                                    pred_tool[pred][name] += 1
                            # arg_source: provenance of this write's args
                            wargs = tc.get("arguments") or {}
                            for p, v in wargs.items():
                                sv = _scalar(v)
                                if sv is None:
                                    continue
                                # find a prior read field carrying same value
                                for rname, flds in recent_by_read:
                                    hit = [fk for fk, fv in flds.items() if fv == sv]
                                    if hit:
                                        arg_n[(name, p)] += 1
                                        argsrc[(name, p)][f"{rname}.{hit[0]}"] += 1
                                        break
                        last_read = None  # reset between writes
                elif r == "tool":
                    caller = id2name.get(m.get("id"))
                    if caller and is_read(caller):
                        d = parse_tool_content(m.get("content"))
                        flds = flatten(d) if d is not None else {}
                        last_read = caller
                        last_read_fields = flds
                        recent_by_read.append((caller, flds))
                        recent_by_read[:] = recent_by_read[-8:]

    # discriminative state->tool: keep preds where this tool dominates the followers
    # (precision = P(tool | pred)), not just frequent. Excludes shared/ambiguous state.
    observation_triggers = {}
    for tool, ctr in trig.items():
        n = fix_n[tool]
        sig = []
        for pred, c in ctr.most_common():
            tot = sum(pred_tool[pred].values())
            prec = c / tot if tot else 0
            if c >= args.min_support and prec >= args.min_precision:
                sig.append({"pred": pred, "support": c,
                            "precision": round(prec, 2), "recall": round(c / n, 2)})
        if sig:
            observation_triggers[tool] = {"n": n, "triggers": sig[:8]}

    arg_source = {}
    for (tool, p), ctr in argsrc.items():
        src, c = ctr.most_common(1)[0]
        n = arg_n[(tool, p)]
        if c >= args.min_support and c / n >= args.min_frac:
            arg_source.setdefault(tool, {})[p] = {"source": src, "support": c, "frac": round(c / n, 2)}

    out = args.out or os.path.join(
        "reports/facet_rft_2026/phase4_distill/induced",
        f"obs_triggers_{args.domain}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"domain": args.domain,
               "observation_triggers": observation_triggers,
               "arg_source": arg_source},
              open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[{args.domain}] observation_triggers: {len(observation_triggers)} tools | "
          f"arg_source: {len(arg_source)} tools")
    for tool, d in list(observation_triggers.items())[:8]:
        top = d["triggers"][0] if d["triggers"] else {}
        print(f"    {tool:30} <= {top.get('pred','')} (prec {top.get('precision','')})")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
