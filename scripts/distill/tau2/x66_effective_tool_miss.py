"""Re-measure the missed gold actions at the level of the tool that was actually meant.

Every gold action in this domain is wrapped: `call_discoverable_agent_tool` carrying the
real tool in `agent_tool_name`, and `unlock_discoverable_agent_tool` before it. x50 and
x54 compare at the wrapper level, so a gold call the agent answered with a *different*
discoverable tool still counts as "same family, wrong argument" — which is why the
mismatch tally there is topped by the key `agent_tool_name` (157). That number is not an
argument error; it is the wrapper hiding a tool-selection error behind an argument name.

This unwraps both sides and asks, per missed gold action:

  never-unlocked   the tool was never unlocked, so it could not be called through the
                   dispatcher at all — and gold counts the unlock itself as an action
  never-called     unlocked or not, the effective tool never appears in an agent call
  wrong-args       the effective tool was called; these argument keys differ
  direct-call      the tool was called under its own name rather than through the
                   dispatcher — scored as a miss even though the work may have happened

The last class is the one to keep honest about: it is an artifact of how the benchmark
registers calls (C149/C150), not a failure of the agent to act, and it must not be
counted with the rest.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM, fam, norm_args  # noqa: E402

WRAPPER = {"call_discoverable_agent_tool", "unlock_discoverable_agent_tool",
           "give_discoverable_user_tool", "call_discoverable_user_tool"}


def effective(name, args):
    """The tool a wrapper call is really about, or the name itself when it is not one."""
    if name in WRAPPER:
        inner = (args.get("agent_tool_name") or args.get("discoverable_tool_name")
                 or args.get("user_tool_name"))
        if inner:
            return fam(inner), norm_args(args.get("arguments"))
    return fam(name), args


def agent_actions(sim):
    """(wrapper_name, effective_name, effective_args) for every assistant tool call."""
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            args = norm_args(a)
            eff, eargs = effective(n, args)
            out.append((n, eff, eargs))
    return out


def where(tool, by_role, blob):
    """Which channel put the tool name in front of the agent, if any."""
    if tool in by_role["tool"]:
        return "in-kb-output"      # retrieval handed it over; the failure is downstream
    if tool in by_role["assistant"]:
        return "own-prose-only"    # it named the tool itself but never called it
    return "unseen" if tool not in blob else "elsewhere"


def load(pattern):
    out = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        print(f"  read {os.path.basename(p)}")
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97", choices=sorted(ARMS))
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    sims = load(ARMS[args.arm])
    cls = collections.Counter()
    by_tool = collections.Counter()
    argkeys = collections.Counter()
    wrapper_of_miss = collections.Counter()
    per_sim = collections.Counter()
    lines = []

    for s in sims:
        if (s.get("reward_info") or {}).get("reward") == 1.0:
            continue
        acts = agent_actions(s)
        unlocked = {e for w, e, _ in acts if w == "unlock_discoverable_agent_tool"}
        called_via = {e for w, e, _ in acts if w in WRAPPER}
        called_direct = {e for w, e, _ in acts if w not in WRAPPER}
        seen = collections.defaultdict(list)
        for _, e, ea in acts:
            seen[e].append(ea)

        # Did the tool's name ever reach the transcript at all? This is what separates a
        # retrieval failure from a selection failure: a name the agent never saw cannot
        # have been declined, and a name it did see and skipped is a different defect.
        # The suffix-stripped name is a substring of the suffixed one, so this matches
        # the tool wherever it appeared — search output, policy text, or the agent's prose.
        blob = json.dumps(s.get("messages") or [], ensure_ascii=False)
        # Where it surfaced decides which step failed. A name that arrived in a tool
        # result was handed to the agent by retrieval — anything after that is the
        # agent's. A name that only ever appears in the agent's own prose was never
        # retrieved, and treating the two alike hides which half is broken.
        by_role = collections.defaultdict(str)
        for m in s.get("messages") or []:
            by_role[m.get("role") or "?"] += json.dumps(m, ensure_ascii=False)

        for c in (s.get("reward_info") or {}).get("action_checks") or []:
            if c.get("action_match"):
                continue
            g = c.get("action") or {}
            if g.get("requestor") != "assistant":
                continue
            geff, gargs = effective(g.get("name"), norm_args(g.get("arguments")))
            wrapper_of_miss[g.get("name")] += 1

            if g.get("name") == "unlock_discoverable_agent_tool" and geff not in unlocked:
                k = "never-unlocked-" + where(geff, by_role, blob)
            elif (g.get("name") in WRAPPER and geff in called_direct
                    and geff not in called_via):
                # Only a *wrapped* gold action can be missed this way. A plain tool called
                # under its own name is not a dispatch artifact — it is simply the call,
                # and its arguments are what decide the match.
                k = "direct-call"
            elif geff in called_via or geff in called_direct:
                diff = [key for key, v in gargs.items()
                        if not any(a.get(key) == v for a in seen[geff])]
                k = "wrong-args" if diff else "same-args-still-missed"
                for key in diff:
                    argkeys[key] += 1
            else:
                k = "never-called-" + where(geff, by_role, blob)

            cls[k] += 1
            by_tool[(k, geff)] += 1
            per_sim[(s["task_id"], s.get("trial"), k)] += 1
            lines.append(f"  {s['task_id']}/t{s.get('trial')}  {k:22} {geff}")

    print(f"\n=== {args.arm}: missed assistant-side gold actions, unwrapped ===")
    tot = sum(cls.values())
    for k, v in cls.most_common():
        print(f"  {k:24} {v:4}  ({100*v/tot:.0f}%)")
    print(f"  {'TOTAL':24} {tot:4}")

    print("\n  [gold wrapper the miss was written as]")
    for k, v in wrapper_of_miss.most_common():
        print(f"    {k:36} {v}")

    print("\n  [sims touched by each class]")
    sims_by = collections.Counter()
    for (t, tr, k) in per_sim:
        sims_by[k] += 1
    for k, v in sims_by.most_common():
        print(f"    {k:24} {v} sims")

    if argkeys:
        print("\n  [differing argument keys, effective level]")
        for k, v in argkeys.most_common(12):
            print(f"    {k:36} {v}")

    print("\n  [top effective tools per class]")
    for k in [c for c, _ in cls.most_common()]:
        top = [(t, v) for (kk, t), v in by_tool.most_common() if kk == k][:6]
        print(f"    {k}: " + ", ".join(f"{t}({v})" for t, v in top))

    if args.detail:
        print()
        print("\n".join(lines))


if __name__ == "__main__":
    main()
