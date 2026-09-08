#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replay persisted simulations through the LB engines and count where each one would fire.

Costs no GPU and no money: it reads results.json(.gz) files and, for every assistant turn in them,
builds the Turn state that turn had and runs evaluate(). What it measures is **firing**, not effect -
the trajectory is counterfactual, because a real deny would have changed everything after it. An
engine that fires nowhere here is dead wiring; an engine that fires on a passing simulation is a
candidate for over-blocking. Both are worth knowing before spending an engine hour.

  python lb_replay.py <results.json.gz ...> [--domain banking_knowledge] [--verbose]

Sub-call kinds (LB4 claims, LB2 select) are skipped and counted separately: they need a model.
"""

import argparse
import collections
import glob
import gzip
import io
import json
import os
import sys

import lb_a2
from lb_coordinator import Turn, evaluate, resolve, fam


class Call(object):
    __slots__ = ("name", "arguments", "id")

    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = d.get("arguments") or {}
        self.id = d.get("id")


class Msg(object):
    __slots__ = ("role", "content", "tool_calls", "id", "error", "requestor")

    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [Call(c) for c in (d.get("tool_calls") or [])]
        self.id = d.get("id") or d.get("tool_call_id")
        self.error = bool(d.get("error"))
        self.requestor = d.get("requestor")


def load(path):
    opener = gzip.open if path.endswith(".gz") else io.open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def state_before(a2, messages, upto):
    """The executed counter and unlocked set as of message `upto` (exclusive)."""
    probe = Turn(a2, [], Msg({}))
    dispatch = a2.get("dispatch") or {}
    executed, unlocked, pending = collections.Counter(), set(), {}
    for m in messages[:upto]:
        for c in m.tool_calls:
            pending[c.id] = probe.name_of(c)
            if c.name == dispatch.get("unlock_tool"):
                unlocked.add(probe.named(c))
        if m.role == "tool":
            name = pending.pop(m.id, None)
            text = str(m.content or "").lstrip()
            failed = m.error or any(text.startswith(k) for k in a2.get("failure_markers") or [])
            if name and not failed:
                executed[name] += 1
    return executed, unlocked


def replay_sim(a2, sim, corpus):
    """[(turn_index, Finding)] for every assistant turn of one simulation."""
    messages = [Msg(m) for m in sim.get("messages") or []]
    out = []
    for i, m in enumerate(messages):
        if m.role != "assistant":
            continue
        executed, unlocked = state_before(a2, messages, i)
        turn = Turn(a2, messages[:i], m, executed=executed, unlocked=unlocked, corpus=corpus)
        for lb, findings in evaluate(turn).items():
            for f in findings:
                out.append((i, f))
        for c in resolve(evaluate(turn), turn).conflicts:
            out.append((i, c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    a2 = lb_a2.load(a.domain)
    if a2 is None:
        raise SystemExit("no a2/%s.lb.json" % a.domain)
    corpus = {}
    d = os.environ.get("LB_DOCS_DIR")
    if d and os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith((".md", ".txt", ".json")):
                corpus[os.path.splitext(f)[0]] = io.open(os.path.join(d, f), encoding="utf-8", errors="replace").read()

    paths = [p for pat in a.files for p in sorted(glob.glob(pat))]
    per_task, by_source, conflicts = collections.OrderedDict(), collections.Counter(), collections.Counter()
    for p in paths:
        try:
            data = load(p)
        except Exception as e:
            print("skip %s: %r" % (os.path.basename(p), e), file=sys.stderr)
            continue
        for sim in data.get("simulations") or []:
            task, reward = sim.get("task_id"), (sim.get("reward_info") or {}).get("reward")
            row = per_task.setdefault(task, {"sims": 0, "pass": 0, "fire": collections.Counter()})
            row["sims"] += 1
            row["pass"] += 1 if reward == 1.0 else 0
            deny_turns = set()
            for i, item in replay_sim(a2, sim, corpus):
                if isinstance(item, dict):
                    conflicts[(item["winner"][0], tuple(l[0] for l in item["losers"]))] += 1
                    continue
                if item.primitive == "deny":
                    deny_turns.add(i)
                row["fire"][item.lb] += 1
                by_source[(item.lb, item.source.split(":")[0], item.primitive)] += 1
                if a.verbose:
                    print("  %s msg%-3d %s %s %s %s" % (task, i, item.lb, item.primitive, item.source,
                                                        (item.order or (item.facts or [""])[0])[:90]))
            row.setdefault("deny_turns", []).append(len(deny_turns))

    print("\ntask      pass  denied turns/sim   LB1 LB2 LB3 LB4 LB5 LB7")
    for task, row in sorted(per_task.items()):
        f, dt = row["fire"], row.get("deny_turns") or [0]
        print("%-9s %d/%-3d %-18s %3d %3d %3d %3d %3d %3d"
              % (task, row["pass"], row["sims"], ",".join(str(x) for x in dt),
                 f["LB1"], f["LB2"], f["LB3"], f["LB4"], f["LB5"], f["LB7"]))
    print("\nfirings by engine, rule and primitive")
    for (lb, source, prim), n in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
        print("  %-4s %-22s %-8s %d" % (lb, source, prim, n))
    silent = [lb for lb in ("LB1", "LB2", "LB3", "LB4", "LB5", "LB7")
              if not any(row["fire"][lb] for row in per_task.values())]
    print("\nsilent engines (no firing on any replayed turn): %s" % (", ".join(silent) or "none"))
    print("conflicts (winner <- losers): %s" % (dict(("%s<-%s" % (w, ",".join(l)), n) for (w, l), n in conflicts.items()) or "none"))
    print("note: sub-call kinds (LB4 claims, LB2 select) do not run here - they need a model.")
    print("note: firing is counterfactual - after a real deny the conversation would have diverged.")


if __name__ == "__main__":
    main()
