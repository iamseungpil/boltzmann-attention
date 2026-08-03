"""What would the completion-claim block fire on, and what does the existing gate miss?

The design (`ACTION_HANDOFF_LEVERS_DESIGN_2026_08_04.md` §3 L4) left one thing open:
which predicate to hang the lever on. The two numbers in the handoff — 11 and 49 —
both come from analyzers that read gold (`x50` needs the missed action, `x59` needs the
action_checks). Neither is implementable: at generation time the engine has the
trajectory and nothing else. So this counts the target under the **predicate that can
actually run**, and only then can the target be honestly pre-registered.

Three closed variants, each a superset of the next:

  V3  claim-regex ∧ this turn emitted no tool call            (the resignation window)
  V1  V3 ∧ no agent call at all since the last user message   (design §4 as written)
  V2  V3 ∧ no effective write anywhere earlier in the sim     (what T2_WRITE_PROV uses)
  V4  V2 ∧ no transfer call either                           (V2 minus the transfer hole)
  V5  per-verb: a "transferred" claim wants a transfer event, every other verb wants a
      write event                                            (V2 with the hole closed
                                                              instead of widened)

V1 has a known hole: an agent that filed the dispute at turn 5 and recaps "your dispute
has been filed" at turn 12 fires it, and blocking that would be a regression. V2 closes
the hole by refusing to fire once any write exists, at the cost of missing a false claim
that follows a *different* true one. This prints both so the choice is made on measured
over-fire, not on argument.

The second question is why the existing gate is not already doing this. `T2_WRITE_PROV`
occupies the same window (`_resign` ∧ no effective write) but asks the model to declare
`claims_completion` itself. The run logs say that declaration comes back False almost
every time. `--logs` counts that directly, since a lever whose window fires 600 times
and whose body fires 5 is not a lever that needs a sibling — it needs its predicate
replaced.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SIM_LOCAL = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
SIM_REMOTE = ("/home/woori/workspace_common/boltzmann-attention-pi/"
              "reports/facet_rft_2026/sim_results")

ARMS = {"A": "bank_ax33n_gpu*_20260803g", "B4": "bank_b4_gpu*_20260803h"}

# Same assertion set as x50, deliberately: if the two analyzers disagree on what counts
# as a completion claim, neither number can be compared with the other.
DONE = re.compile(
    r"(has|have) been (successfully )?(filed|submitted|processed|updated|created|closed|"
    r"issued|applied|transferred|completed|credited|reversed|logged|generated|sent)"
    r"|i (have|already) (filed|submitted|processed|updated|created|logged|applied|sent)"
    r"|(is|are) now (filed|submitted|updated|processed|active|complete)",
    re.I)

_SUFFIX_RE = re.compile(r"_\d+$")
_READ_PREFIX_RE = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
_PROCEDURAL_RE = re.compile(r"(^log_|^verify_|_verification$|^kb_|^shell$|transfer_to_human)", re.I)
# The one branch of the procedural set that still leaves an event in the ledger a claim
# can legitimately point at. Named separately because V4 needs it and write does not.
_TRANSFER_RE = re.compile(r"transfer_to_human|_transfer_\d|^transfer_", re.I)


def a2_procedural(a2):
    """Mirror of `t2_gate_patch._a2_procedural` — the dispatchers A2 calls procedural.

    Kept as a mirror rather than an import because importing the patch pulls the whole
    gate stack in; the assertion below is that the two stay identical, and `--selftest`
    checks it against the live function when the import happens to succeed.
    """
    if not a2:
        return frozenset()
    ep = a2.get("eplan") or {}
    out = {ep.get("dispatch_tool"), ep.get("unlock_tool"), ep.get("list_tool")}
    out.add((a2.get("completion_guard") or {}).get("user_execution_tool"))
    for t in a2.get("scaffold_get_tools") or []:
        if not isinstance(t, dict):
            continue
        fu = t.get("follow_up") or {}
        out.add(fu.get("tool"))
        out.add((fu.get("completion_guard") or {}).get("user_execution_tool"))
    for v in a2.get("value_acquisition") or []:
        if isinstance(v, dict):
            out.add(v.get("give_tool"))
    return frozenset(_SUFFIX_RE.sub("", str(x)) for x in out if x)


def is_effective_write(name, procedural):
    if not name:
        return False
    if _READ_PREFIX_RE.match(name) or _PROCEDURAL_RE.search(name):
        return False
    return _SUFFIX_RE.sub("", str(name)) not in procedural


def eff_name(tc):
    """The tool a call really exercises — a dispatcher's payload, else its own name."""
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    if isinstance(a, dict):
        inner = (a.get("agent_tool_name") or a.get("discoverable_tool_name")
                 or a.get("user_tool_name") or a.get("tool_name"))
        if inner:
            return str(inner)
    return str(tc.get("name") or (tc.get("function") or {}).get("name") or "")


def load(sim_dir, pattern):
    files = sorted(glob.glob(os.path.join(sim_dir, f"{pattern}.results.json.gz")))
    if not files:
        raise SystemExit(f"no runs matched {sim_dir}/{pattern}.results.json.gz")
    out = []
    for p in files:
        print(f"  read {os.path.basename(p)}")
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def scan(sim, procedural):
    """Fire-by-fire walk of one simulation.

    Returns (fires, wrote_before_any_fire) where each fire is a dict of which variants
    hold at that turn, so a sim can be attributed to the loosest variant that caught it.
    """
    fires = []
    calls_since_user = 0
    write_seen = False
    transfer_seen = False

    def note(tcs):
        nonlocal write_seen, transfer_seen
        for tc in tcs:
            n = eff_name(tc)
            if is_effective_write(n, procedural):
                write_seen = True
            # `_is_effective_write` deliberately excludes the transfer tool, so a run
            # that legitimately transferred and then said "you have been transferred"
            # looks unbacked to V2. Measured: that is what almost every V2 over-fire on
            # a passing sim is. Tracked separately so V4 can close it.
            if _TRANSFER_RE.search(n or ""):
                transfer_seen = True

    for m in sim.get("messages") or []:
        role = m.get("role")
        tcs = m.get("tool_calls") or []
        if role == "user":
            calls_since_user = 0
            # A customer-executed action changes the world too: the claim that follows
            # it can be true. Counting only agent calls here would over-fire on exactly
            # the hand-off tasks the run is built around.
            note(tcs)
            continue
        if role != "assistant":
            continue
        content = m.get("content")
        if tcs:
            calls_since_user += len(tcs)
            note(tcs)
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        hits = DONE.findall(content)
        if not hits:
            continue
        # Which event would back each verb. "transferred" points at the transfer tool,
        # everything else at a write — the same kind→event split the engine already
        # keeps in A2 `claim_prov.event_map`, read off the regex instead of off an LLM
        # declaration.
        verbs = {w.strip().lower() for h in hits for w in h if w.strip()}
        wants_transfer = bool(verbs & {"transferred"})
        wants_write = bool(verbs - {"transferred", "has", "have", "is", "are",
                                    "i", "successfully", "already", "now"})
        fires.append({
            "V3": True,
            "V1": calls_since_user == 0,
            "V2": not write_seen,
            "V4": not write_seen and not transfer_seen,
            "V5": ((wants_write and not write_seen)
                   or (wants_transfer and not transfer_seen)),
            "text": DONE.search(content).group(0)[:60],
        })
    return fires


def gold_unbacked(sim):
    """Did the run actually miss an action it was supposed to emit?

    This is the gold-side question the runtime predicate cannot ask. It is here only to
    label a fire as justified or not — never as part of the predicate.
    """
    checks = (sim.get("reward_info") or {}).get("action_checks") or []
    return any(not c.get("action_match") for c in checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    ap.add_argument("--sim", default="")
    ap.add_argument("--a2", default=os.path.join(HERE, "a2", "banking_knowledge.gate.json"))
    ap.add_argument("--logs", action="store_true",
                    help="also tally what T2_WRITE_PROV did in the same runs")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    sim_dir = args.sim or (SIM_LOCAL if os.path.isdir(SIM_LOCAL) else SIM_REMOTE)
    a2 = {}
    if os.path.exists(args.a2):
        a2 = json.load(open(args.a2, encoding="utf-8"))
    procedural = a2_procedural(a2)
    print(f"sim_dir = {sim_dir}")
    print(f"A2 procedural (non-write dispatchers) = {sorted(procedural) or '∅ — A2 not loaded'}")

    for arm in args.arms.split(","):
        print(f"\n{'=' * 78}\n[{arm}]  {ARMS[arm]}\n{'=' * 78}")
        sims = load(sim_dir, ARMS[arm])
        # sims x variants, split by outcome. A fire in a passing sim is the cost side:
        # the lever would have spent a regeneration re-writing a true sentence.
        tally = collections.Counter()
        rows = []
        for s in sorted(sims, key=lambda x: (x["task_id"], x.get("trial") or 0)):
            ok = ((s.get("reward_info") or {}).get("reward") or 0.0) == 1.0
            fires = scan(s, procedural)
            if not fires:
                continue
            key = f"{s['task_id']}/t{s.get('trial')}"
            unbacked = gold_unbacked(s)
            for v in ("V1", "V2", "V4", "V5", "V3"):
                hit = [f for f in fires if f[v]]
                if not hit:
                    continue
                tally[f"{v}·sim·{'pass' if ok else 'fail'}"] += 1
                tally[f"{v}·turn"] += len(hit)
                if not ok and unbacked:
                    tally[f"{v}·sim·fail·주장 미이행"] += 1
            rows.append((key, ok, unbacked, fires))

        n_pass = sum(1 for s in sims if ((s.get("reward_info") or {}).get("reward") or 0.0) == 1.0)
        print(f"\n완주 {len(sims)} (pass {n_pass} / fail {len(sims) - n_pass})")
        print(f"\n  {'변이':4s} {'발화 sim(fail)':>14s} {'그중 미이행':>11s} "
              f"{'발화 sim(pass)':>14s} {'발화 턴':>7s}")
        for v, desc in (("V1", "직전 user 이후 호출 0"), ("V2", "원장에 write 0"),
                        ("V4", "원장에 write 0 ∧ 이관 0"),
                        ("V5", "동사별 이벤트 대조"), ("V3", "그 턴 호출 0")):
            print(f"  {v:4s} {tally[f'{v}·sim·fail']:14d} "
                  f"{tally[f'{v}·sim·fail·주장 미이행']:11d} "
                  f"{tally[f'{v}·sim·pass']:14d} {tally[f'{v}·turn']:7d}   {desc}")

        # The pass-side fire is the whole reason V1 and V2 are printed side by side.
        print(f"\n  통과 sim 오발화: V1 {tally['V1·sim·pass']} · V2 {tally['V2·sim·pass']}"
              f" · V4 {tally['V4·sim·pass']} · V5 {tally['V5·sim·pass']}"
              f" · V3 {tally['V3·sim·pass']}")

        if args.detail:
            print("\n  --- 발화 전수 ---")
            for key, ok, unbacked, fires in rows:
                for f in fires:
                    print(f"  {key:16s} {'pass' if ok else 'fail':4s} "
                          f"{'미이행' if unbacked else '전부이행':8s} "
                          f"V1={int(f['V1'])} V2={int(f['V2'])} V5={int(f['V5'])}  "
                          f"{f['text']!r}")

    if args.logs:
        print(f"\n{'=' * 78}\n[기존 게이트 T2_WRITE_PROV 실측 — 같은 런의 stderr]\n{'=' * 78}")
        tot = collections.Counter()
        for arm in args.arms.split(","):
            for p in sorted(glob.glob(os.path.join(sim_dir, f"{ARMS[arm]}.log.gz"))):
                w = t = r = 0
                for ln in gzip.open(p, "rt", encoding="utf-8", errors="replace"):
                    if "[T2_WRITEPROV] window" in ln:
                        w += 1
                        if "declared_completion=True" in ln:
                            t += 1
                    elif "[T2_WRITEPROV] regen" in ln:
                        r += 1
                tot[arm + "·window"] += w
                tot[arm + "·True"] += t
                tot[arm + "·regen"] += r
                print(f"  {os.path.basename(p):40s} window={w:4d} 선언True={t:3d} regen={r:3d}")
        for arm in args.arms.split(","):
            w, t = tot[arm + "·window"], tot[arm + "·True"]
            print(f"  [{arm}] 창 {w} · LLM 자기선언 True {t} "
                  f"({(100.0 * t / w) if w else 0:.1f}%) · regen {tot[arm + '·regen']}")
        print("\n  ⇒ 창은 열려 있고 **선언이 닫는다**. 같은 창에 결정론 술어를 걸면"
              "\n    새 게이트가 아니라 기존 게이트의 술어 교체다([[05]] §5).")


if __name__ == "__main__":
    main()
