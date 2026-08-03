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

SIM_REMOTE = ("/home/woori/workspace_common/boltzmann-attention-pi/"
              "reports/facet_rft_2026/sim_results")
# The persisted gz live in the repo, so this reads on either side without editing the
# constant — the hard-coded path was one of the four instrument defects of 2026-08-04.
SIM_LOCAL = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "reports", "facet_rft_2026", "sim_results"))
SIM = SIM_REMOTE if os.path.isdir(SIM_REMOTE) else SIM_LOCAL

ARMS = {
    "A":  "bank_ax33n_gpu*_20260803g",
    "B4": "bank_b4_gpu*_20260803h",
    # The full-97 sweep of 2026-08-04. `main` is the LPT-scheduled block, `batch_NN` the
    # shared reserve; the glob takes both so the arm grows as the reserve drains.
    "N97": "bank_n97_gpu*_20260804",
}

# Wrappers, not actions: they carry another tool's name in their arguments, so without
# excluding them every dispatcher call looks like a call gold never asked for.
DISPATCH = {"call_discoverable_agent_tool", "unlock_discoverable_agent_tool",
            "give_discoverable_user_tool", "think", ""}

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
            inner = (args.get("agent_tool_name") or args.get("discoverable_tool_name")
                     or args.get("user_tool_name") or args.get("tool_name"))
            if inner:
                ia = args.get("arguments")
                out.append((inner, norm_args(ia) if ia is not None else {}))
    return out


def user_calls(sim):
    """What the customer executed themselves.

    These sit in role=user messages with requestor=user. Without reading them, a gold
    action the customer performed with the wrong arguments is indistinguishable from
    one they never performed at all — and those have opposite causes: the first is the
    agent steering them wrong, the second is the agent never getting them to act.
    """
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != "user":
            continue
        for tc in m.get("tool_calls") or []:
            out.append(((tc.get("name") or ""), norm_args(tc.get("arguments"))))
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


def variants(v):
    """Surface forms the same value can legitimately take in a transcript.

    62000 is written '62,000' and '$62,000'; a date is punctuated several ways.
    Without this the check reports 'the model did not have it' for values it plainly
    had, which would understate reachability.
    """
    s = str(v).strip().lower()
    out = {s}
    # bool is an int in Python, and `float("true")` raises — the same numbers-only blind
    # spot the 08-04 read-through found in the grounding check was in this instrument too.
    if (isinstance(v, (int, float)) and not isinstance(v, bool)) \
            or re.fullmatch(r"-?\d+(\.\d+)?", s):
        n = str(v)
        out |= {n, f"{float(s):,.0f}" if "." not in s else n, n.rstrip("0").rstrip(".")}
        try:
            out.add(f"{int(float(s)):,d}")
        except Exception:
            pass
    out |= {re.sub(r"[^a-z0-9]", "", s), re.sub(r"[_\-]", " ", s)}
    return {x for x in out if len(x) >= 4}


def values_present(gold_args, hay):
    """How much of this call was already writable from the transcript?

    Returns (present, checked). Free-text arguments the agent is meant to compose
    (summary, notes, descriptions) are excluded — they are never in context verbatim
    and counting them would make every action look unreachable.
    """
    present = checked = 0
    for k, v in (gold_args or {}).items():
        if isinstance(v, bool) or v is None:
            continue
        if k in ("summary", "notes", "note", "description", "message", "reason_details"):
            continue
        s = str(v).strip().lower()
        if len(s) < 4:
            continue
        checked += 1
        if any(x in hay for x in variants(v)):
            present += 1
    return present, checked


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
        # Per-action counts are dominated by a few tasks that miss the same call 8-17
        # times (040, 041). Judging from them would be reading one task as if it were
        # the run. Every action-level count below is paired with a sim-level one.
        per_sim_class = collections.Counter()
        argkeys = collections.Counter()
        no_miss, rows = [], []
        for s in sorted(fails, key=lambda x: (x["task_id"], x.get("trial") or 0)):
            key = f"{s['task_id']}/t{s.get('trial')}"
            cl = calls(s)
            names = {fam(n) for n, _ in cl}
            text = agent_text(s)
            hay = transcript_before_end(s)
            claimed = bool(DONE.search(text))

            checks = (s.get("reward_info") or {}).get("action_checks") or []
            missed = [c for c in checks if not c.get("action_match")
                      and (c.get("action") or {}).get("requestor") == "assistant"]
            if not missed:
                # The agent emitted every action that was its to emit and still failed.
                # Two very different things hide here: a customer-executed action that
                # never happened because the tool was never handed over, and a DB that
                # diverged from gold while every gold action matched (over-action).
                user_missed = [c for c in checks if not c.get("action_match")
                               and (c.get("action") or {}).get("requestor") == "user"]
                uc = user_calls(s)
                # The handover argument is `discoverable_tool_name`; reading any other
                # key here silently reports every sim as never having handed anything
                # over (caught 2026-08-04 doing exactly that).
                handed = {fam(a.get("discoverable_tool_name") or a.get("agent_tool_name")
                              or a.get("user_tool_name") or a.get("tool_name") or "")
                          for n, a in cl if "give" in (n or "") or "unlock" in (n or "")}
                executed, ungiven, detail = [], [], []
                for c in user_missed:
                    g = c.get("action") or {}
                    gname, gargs = g.get("name"), norm_args(g.get("arguments"))
                    tried = [a for n, a in uc if fam(n) == fam(gname)]
                    if tried:
                        best = min(tried, key=lambda e: sum(
                            1 for k in set(gargs) | set(e) if gargs.get(k) != e.get(k)))
                        diff = [k for k in sorted(set(gargs) | set(best))
                                if gargs.get(k) != best.get(k)]
                        executed.append((gname, diff))
                        detail.append(f"{fam(gname)}[{','.join(diff[:3])}]")
                    else:
                        ungiven.append((gname, fam(gname) in handed))
                        detail.append(f"{fam(gname)}(미실행"
                                      f"{'·인계함' if fam(gname) in handed else '·미인계'})")
                gold_all = {fam((c.get("action") or {}).get("name")) for c in checks}
                extra = sorted({fam(n) for n, _ in cl} - gold_all - DISPATCH)
                sub = ("★user 실행·인자 불일치(안내가 틀림)" if executed else
                       "user 미실행" if ungiven else
                       "DB만 불일치(gold 행동 전부 일치)")
                no_miss.append((key, claimed, sub, "; ".join(detail), extra))
                tally[f"MISS없음·{sub}"] += 1
                per_sim_class[f"MISS없음 — {sub}"] += 1
                continue

            tally["MISS 있는 sim"] += 1
            sim_kinds = set()
            for c in missed:
                g = c.get("action") or {}
                gname, gargs = g.get("name"), norm_args(g.get("arguments"))
                called = fam(gname) in names
                if called:
                    # Which argument did it get wrong? Compare against the closest
                    # call of that tool, the same way the census picks its witness.
                    tried = [a for n, a in cl if fam(n) == fam(gname)]
                    if tried:
                        best = min(tried, key=lambda e: sum(
                            1 for k in set(gargs) | set(e) if gargs.get(k) != e.get(k)))
                        for k in sorted(set(gargs) | set(best)):
                            if gargs.get(k) != best.get(k):
                                argkeys[k] += 1
                pres, nchk = values_present(gargs, hay)
                # Name-reachability is the reach-scan predicate (`ax33g_reach_scan`):
                # did the model ever see the tool's name? Argument-reachability is
                # the stronger question of whether the call was writable. They are
                # different measures and the gap between them is the point.
                name_reach = fam(gname).lower() in hay
                mentioned = fam(gname).lower() in text.lower()
                kind = "ARG" if called else "NEVER"
                tally[f"missed:{kind}"] += 1
                level = ("ALL" if nchk and pres == nchk else
                         "NONE" if not pres else "PARTIAL") if nchk else "n/a(검사값 0)"
                if kind == "NEVER":
                    tally[f"NEVER·인자보유 {level}"] += 1
                    tally[f"NEVER·이름보유={name_reach}"] += 1
                    if level == "ALL":
                        tally["★말로 바꿈(NEVER∧인자 ALL)"] += 1
                        if claimed:
                            tally["★★그 위에 완료 주장까지"] += 1
                    if mentioned:
                        tally["NEVER인데 도구명 발화"] += 1
                sim_kinds.add("말로 바꿈" if kind == "NEVER" and level == "ALL" else kind)
                rows.append((key, gname, kind, f"{pres}/{nchk}", level, mentioned, claimed))

            if "말로 바꿈" in sim_kinds:
                per_sim_class["★말로 바꿈 포함"] += 1
            elif sim_kinds == {"NEVER"}:
                per_sim_class["NEVER만(정보도 없음)"] += 1
            elif sim_kinds == {"ARG"}:
                per_sim_class["ARG만(인자 오류)"] += 1
            else:
                per_sim_class["ARG+NEVER 혼합"] += 1

        print(f"\n실패 sim {len(fails)} / 완주 {len(sims)}")
        print("\n  [sim 단위 — 판정은 이 표로]")
        for k, v in per_sim_class.most_common():
            print(f"  {k:34s} {v}")
        print("\n  [행동 단위 — 소수 태스크가 지배하므로 참고용]")
        for k, v in sorted(tally.items()):
            print(f"  {k:34s} {v}")
        print("\n  [틀린 인자 키 (ARG 케이스)]")
        for k, v in argkeys.most_common(10):
            print(f"  {k:34s} {v}")

        print(f"\n--- 에이전트-측 MISS 0인 실패 {len(no_miss)}건 ---")
        for key, c, sub, detail, extra in no_miss:
            print(f"  {key:16s} {sub:30s} {detail}{'  [완료 주장]' if c else ''}"
                  + (f"\n  {'':16s} gold-밖 호출: {', '.join(extra[:5])}" if extra else ""))

        if args.detail:
            print("\n--- 놓친 gold 행동 전수 ---")
            print(f"  {'sim':16s} {'kind':6s} {'인자':7s} {'수준':8s} {'발화':6s} {'완료주장':6s} tool")
            for key, gname, kind, ratio, level, mentioned, claimed in rows:
                print(f"  {key:16s} {kind:6s} {ratio:7s} {level:8s} {str(mentioned):6s} "
                      f"{str(claimed):6s} {gname}")


if __name__ == "__main__":
    main()
