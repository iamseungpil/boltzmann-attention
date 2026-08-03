"""Does the agent fail because it lacks information, or because it narrates instead of acting?

The B4 read-through concluded the dominant defect is not retrieval — it is turning an
action into a sentence while already holding everything the action needs (§3 of the
2026-08-04 handoff). That was reached by reading trajectories. This puts a decidable
predicate under it and runs it over every failed simulation in both arms, so the claim
can be checked rather than trusted.

For each gold action the agent was supposed to emit (requestor=assistant) and did not:

  NEVER / ARG      the tool was never called at all, or called with different arguments
  mentioned        the tool's name appears in the agent's own prose — it knew the name
  info-in-context  every gold argument value already appears earlier in the transcript,
                   so the call was writable at that point without any further retrieval
  claimed-done     the agent asserted somewhere that the work was completed

"Says instead of does" is NEVER ∧ info-in-context: the agent had what the call needed
and produced text instead. `claimed-done` on top of that is the harmful form, because
the customer is told a write happened that did not.

Sims that fail with no agent-side miss at all are listed separately — their failure is
DB state or a user-executed action, and this predicate says nothing about them.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")

ARMS = {
    "A":  "bank_ax33n_gpu*_20260803g",
    "B4": "bank_b4_gpu*_20260803h",
}

# Assertions that a write already happened. Kept narrow and past-tense on purpose:
# "I will file the dispute" is a plan, "the dispute has been filed" is a false report.
DONE = re.compile(
    r"(has|have) been (successfully )?(filed|submitted|processed|updated|created|closed|"
    r"issued|applied|transferred|completed|credited|reversed|logged|generated|sent)"
    r"|i (have|already) (filed|submitted|processed|updated|created|logged|applied|sent)"
    r"|(is|are) now (filed|submitted|updated|processed|active|complete)",
    re.I)


def load(pattern):
    files = sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz"))
    if not files:
        raise SystemExit(f"no runs matched {SIM}/{pattern}.results.json.gz")
    out = []
    for p in files:
        print(f"  read {os.path.basename(p)}")
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def norm_args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def fam(name):
    """Drop the 4-digit discoverable-tool suffix so `x_0589` and `x` compare equal."""
    return re.sub(r"_\d{3,4}$", "", name or "")


def calls(sim):
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
            out.append((n, args))
            # A dispatcher call carries the real tool name and its arguments inside.
            inner = args.get("agent_tool_name") or args.get("user_tool_name") or \
                args.get("tool_name")
            if inner:
                ia = args.get("arguments")
                out.append((inner, norm_args(ia) if ia is not None else {}))
    return out


def agent_text(sim):
    return "\n".join(m.get("content") or "" for m in sim.get("messages") or []
                     if m.get("role") == "assistant" and isinstance(m.get("content"), str))


def transcript_before_end(sim):
    """Everything the agent could read: all message bodies and all tool outputs."""
    parts = []
    for m in sim.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        for tc in m.get("tool_calls") or []:
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            parts.append(a if isinstance(a, str) else json.dumps(a or {}))
    return "\n".join(parts).lower()


def values_present(gold_args, hay):
    """Are all gold argument values already somewhere in the transcript?

    Booleans and very short values are excluded: 'true' or '5' appearing somewhere
    proves nothing. If nothing substantive is left to check, the action is not
    counted as reachable (conservative — it cannot inflate the claim).
    """
    checked = 0
    for v in (gold_args or {}).values():
        if isinstance(v, bool) or v is None:
            continue
        s = str(v).strip().lower()
        if len(s) < 4:
            continue
        checked += 1
        if s not in hay:
            return False, checked
    return (checked > 0), checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    ap.add_argument("--detail", action="store_true", help="one line per missed action")
    args = ap.parse_args()

    for arm in args.arms.split(","):
        print(f"\n{'=' * 76}\n[{arm}]  {ARMS[arm]}\n{'=' * 76}")
        sims = load(ARMS[arm])
        fails = [s for s in sims if ((s.get("reward_info") or {}).get("reward") or 0.0) != 1.0]

        tally = collections.Counter()
        no_miss, rows = [], []
        for s in sorted(fails, key=lambda x: (x["task_id"], x.get("trial") or 0)):
            key = f"{s['task_id']}/t{s.get('trial')}"
            cl = calls(s)
            names = {fam(n) for n, _ in cl}
            text = agent_text(s)
            hay = transcript_before_end(s)
            claimed = bool(DONE.search(text))

            missed = [c for c in (s.get("reward_info") or {}).get("action_checks") or []
                      if not c.get("action_match")
                      and (c.get("action") or {}).get("requestor") == "assistant"]
            if not missed:
                no_miss.append((key, claimed))
                tally["에이전트-측 MISS 없음(sim)"] += 1
                continue

            tally["MISS 있는 sim"] += 1
            for c in missed:
                g = c.get("action") or {}
                gname, gargs = g.get("name"), norm_args(g.get("arguments"))
                called = fam(gname) in names
                reach, nchk = values_present(gargs, hay)
                mentioned = fam(gname).lower() in text.lower()
                kind = "ARG" if called else "NEVER"
                tally[f"missed:{kind}"] += 1
                if kind == "NEVER":
                    tally[f"NEVER·정보보유={reach}"] += 1
                    if reach:
                        tally["★말로 바꿈(NEVER∧정보보유)"] += 1
                        if claimed:
                            tally["★★그 위에 완료 주장까지"] += 1
                    if mentioned:
                        tally["NEVER인데 도구명 발화"] += 1
                rows.append((key, gname, kind, reach, nchk, mentioned, claimed))

        print(f"\n실패 sim {len(fails)} / 완주 {len(sims)}")
        for k, v in sorted(tally.items()):
            print(f"  {k:34s} {v}")

        print(f"\n--- 에이전트-측 MISS 0인 실패 {len(no_miss)}건 (DB 상태·user 실행) ---")
        print("  " + ", ".join(f"{k}{'!' if c else ''}" for k, c in no_miss))
        print("  (! = 그럼에도 완료를 주장한 sim)")

        if args.detail:
            print("\n--- 놓친 gold 행동 전수 ---")
            print(f"  {'sim':14s} {'kind':6s} {'정보보유':6s} {'발화':4s} {'완료주장':5s} tool")
            for key, gname, kind, reach, nchk, mentioned, claimed in rows:
                print(f"  {key:14s} {kind:6s} {str(reach):6s} {str(mentioned):4s} "
                      f"{str(claimed):5s} {gname}  (검사값 {nchk})")


if __name__ == "__main__":
    main()
