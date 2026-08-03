"""Per task, per step: where the run failed, and which layer could have caught it.

The question this exists to answer is not "what broke" but "whose job was it" — the A2
declaration layer, or the engine. Getting that wrong is expensive in both directions:
authoring A2 for something the environment already states is opex we do not need, and
reaching for the engine on something only the policy knows is a domain-specific scaffold,
which is forbidden ([[05]]).

So each missed action is attributed by a stated rule, in the order [[23]] fixes:

  engine · env-derivable   the trigger is computable from the environment surface and the
                           transcript alone — the tool is in the registry, its name came
                           back in a search result, and it was never unlocked. No new A2.
  A2 candidate             the engine would need a domain fact it does not have. This is
                           only a candidate: the fact must be traced to policy or KB prose
                           before it may be authored, and if it is knowable only from gold
                           there is no lever and it stays open.
  open · LLM               choosing among several admissible values (which enum, which of
                           three eligible cards). Not closed under variation, so not
                           scaffold ([[22]]).

The A2 column is measured, not assumed: a tool counts as declared only if it appears in
the files that drive engine behaviour (settings, specific), not in `env_surface.json`,
which is a machine-derived enumeration of the environment and costs nothing.
"""

import argparse
import collections
import glob
import gzip
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM, norm_args  # noqa: E402
from x66_effective_tool_miss import WRAPPER, agent_actions, effective, where  # noqa: E402

A2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
DRIVING = ["banking_knowledge.settings.json", "banking_knowledge.specific.json"]
SURFACE = "env_surface.json"


def read(*names):
    out = ""
    for n in names:
        p = os.path.join(A2, n)
        if os.path.isfile(p):
            out += io.open(p, encoding="utf-8").read()
    return out


def attribute(cls, tool, driving, surface):
    """Which layer could have closed this miss, by the rule in the docstring."""
    if cls.endswith("in-kb-output") and tool in surface:
        return "engine · env-derivable"
    if cls.endswith("unseen"):
        return "engine · retrieval" if tool in surface else "A2 candidate"
    if cls.endswith("own-prose-only"):
        return "engine · env-derivable"
    if cls == "wrong-args":
        return "open · LLM" if tool not in driving else "A2 candidate"
    return "A2 candidate"


def load(pattern):
    out = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def classify_misses(s):
    """Every missed assistant-side gold action, with its class and decisive step."""
    acts = agent_actions(s)
    unlocked = {e for w, e, _ in acts if w == "unlock_discoverable_agent_tool"}
    via = {e for w, e, _ in acts if w in WRAPPER}
    direct = {e for w, e, _ in acts if w not in WRAPPER}
    seen = collections.defaultdict(list)
    for _, e, ea in acts:
        seen[e].append(ea)
    blob = json.dumps(s.get("messages") or [], ensure_ascii=False)
    by = collections.defaultdict(str)
    for m in s.get("messages") or []:
        by[m.get("role") or "?"] += json.dumps(m, ensure_ascii=False)

    out = []
    for c in (s.get("reward_info") or {}).get("action_checks") or []:
        if c.get("action_match"):
            continue
        g = c.get("action") or {}
        ge, ga = effective(g.get("name"), norm_args(g.get("arguments")))
        if g.get("requestor") != "assistant":
            out.append((f"user-side · {'실행했으나 인자 불일치' if any(1 for m in s.get('messages') or [] if m.get('role')=='user' and any((tc.get('name') or '')==g.get('name') for tc in m.get('tool_calls') or [])) else '손님이 실행 안 함'}", ge, []))
            continue
        if g.get("name") == "unlock_discoverable_agent_tool" and ge not in unlocked:
            cls = "never-unlocked-" + where(ge, by, blob)
            diff = []
        elif ge in via or ge in direct:
            diff = [k for k, v in ga.items() if not any(a.get(k) == v for a in seen[ge])]
            cls = "wrong-args" if diff else "same-args"
        else:
            cls = "never-called-" + where(ge, by, blob)
            diff = []
        out.append((cls, ge, diff))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the per-task listing to this file")
    args = ap.parse_args()

    driving, surface = read(*DRIVING), read(SURFACE)
    sims = load(ARMS[args.arm])
    front = {s["task_id"] for s in load(ARMS["A"])} | {s["task_id"] for s in load(ARMS["B4"])}

    per_task = collections.defaultdict(list)
    for s in sims:
        per_task[s["task_id"]].append(s)

    lines = []
    verdict = collections.Counter()
    vd_by_group = collections.defaultdict(collections.Counter)
    task_verdict = {}

    for t in sorted(per_task):
        grp = "front32" if t in front else "new65"
        head = f"── {t}  [{grp}]"
        lines.append(head + " " + "─" * max(0, 68 - len(head)))
        tv = collections.Counter()
        for s in sorted(per_task[t], key=lambda x: x.get("trial") or 0):
            ri = s.get("reward_info") or {}
            if ri.get("reward") == 1.0:
                lines.append(f"   t{s.get('trial')}  PASS")
                continue
            misses = classify_misses(s)
            n_gold = len(ri.get("action_checks") or [])
            lines.append(f"   t{s.get('trial')}  FAIL  [{s.get('termination_reason')}]  "
                         f"gold {n_gold} · miss {len(misses)}")
            agg = collections.Counter()
            for cls, tool, diff in misses:
                v = ("user-side" if cls.startswith("user-side")
                     else attribute(cls, tool, driving, surface))
                verdict[v] += 1
                vd_by_group[grp][v] += 1
                tv[v] += 1
                agg[(cls, v, tool, tuple(diff))] += 1
            for (cls, v, tool, diff), k in agg.most_common():
                d = f" · 틀린 키 {list(diff)}" if diff else ""
                lines.append(f"        {k:2}x  {cls:28} {tool:38} → {v}{d}")
        if tv:
            task_verdict[t] = (grp, tv.most_common(1)[0][0], sum(tv.values()))
        lines.append("")

    print(f"=== {args.arm}: 귀속 집계 ===")
    tot = sum(verdict.values())
    for v, n in verdict.most_common():
        print(f"  {v:26} {n:4}  ({100*n/tot:.0f}%)")
    print(f"  {'합계':26} {tot:4}")
    print("\n  [그룹별]")
    for grp in ("front32", "new65"):
        g = vd_by_group[grp]
        gt = sum(g.values()) or 1
        print(f"    {grp}: " + " · ".join(f"{v} {n}({100*n/gt:.0f}%)" for v, n in g.most_common()))

    print("\n  [태스크 단위 지배 귀속]")
    dom = collections.defaultdict(collections.Counter)
    for t, (grp, v, n) in task_verdict.items():
        dom[grp][v] += 1
    for grp in ("front32", "new65"):
        print(f"    {grp}: " + " · ".join(f"{v} {n}태스크" for v, n in dom[grp].most_common()))

    if args.out:
        io.open(args.out, "w", encoding="utf-8").write("\n".join(lines))
        print(f"\n태스크별 per-step 목록 → {args.out}  ({len(lines)} 줄)")


if __name__ == "__main__":
    main()
