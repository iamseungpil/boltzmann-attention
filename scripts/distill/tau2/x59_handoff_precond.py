"""Do the hand-off preconditions have anything to fire on, and do the levers overlap?

Two questions the design cannot be run without answering, both free.

§1 L5-b's target. The lever says "no tool that was unlocked and then never called".
   If every failing transfer sim never unlocked anything, the predicate is satisfied
   vacuously and the lever fires nowhere — an empty target (C263). The looser form is
   J = (tools named in documents the run actually retrieved) − (tools it used), where
   "used" must include handing the tool to the customer, or a correct hand-off is
   counted as an unused tool (`_tool_given`, t2_gate_patch.py:1256).

§2 Overlap. The lever targets sum to more sims than there are failures, and the design
   attributes results to cause-class counts. If one sim is targeted by three levers,
   that attribution does not hold: fixing it with one lever lowers all three counts.
   This prints the sim x lever matrix so the pre-registration can be written against
   what is actually disjoint.
"""

import argparse
import collections
import glob
import gzip
import json
import re

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")
ARMS = {"A": "bank_ax33n_gpu*_20260803g", "B4": "bank_b4_gpu*_20260803h"}

TRANSFER = "transfer_to_human_agents"
DISCOVERABLE = re.compile(r"\b([a-z][a-z_]{6,}_\d{4})\b")
DONE = re.compile(
    r"(has|have) been (successfully )?(filed|submitted|processed|updated|created|closed|"
    r"issued|applied|transferred|completed|credited|reversed|logged|generated|sent)"
    r"|i (have|already) (filed|submitted|processed|updated|created|logged|applied|sent)", re.I)


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def fam(n):
    return n or ""


def walk(sim):
    """Every agent call in order, with the tool a wrapper carries resolved."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name") or ""
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            args = norm(a)
            inner = (args.get("agent_tool_name") or args.get("discoverable_tool_name")
                     or args.get("user_tool_name") or "")
            out.append((i, n, inner, args))
    return out


def retrieved_tool_names(sim, upto):
    """Tool names appearing in documents the run actually received before step `upto`."""
    names = set()
    for m in (sim.get("messages") or [])[:upto]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            names |= set(DISCOVERABLE.findall(m["content"]))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    ap.add_argument("--tasks", default="", help="e.g. task_032,task_033,task_035")
    args = ap.parse_args()
    only = {t for t in args.tasks.split(",") if t}

    matrix, detail = [], []
    for arm in args.arms.split(","):
        for p in sorted(glob.glob(f"{SIM}/{ARMS[arm]}.results.json.gz")):
            for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
                if only and s["task_id"] not in only:
                    continue
                ri = s.get("reward_info") or {}
                if (ri.get("reward") or 0.0) == 1.0:
                    continue
                cl = walk(s)
                msgs = s.get("messages") or []

                unlocked = {inner for _, n, inner, _ in cl if "unlock" in n and inner}
                given = {inner for _, n, inner, _ in cl if "give" in n and inner}
                called = {inner for _, n, inner, _ in cl if "call" in n and inner}
                direct = {n for _, n, _, _ in cl}
                used = called | given | direct
                tstep = next((i for i, n, _, _ in cl if n == TRANSFER), None)
                kb_before = sum(1 for i, n, _, _ in cl
                                if n.startswith("KB_search") and (tstep is None or i < tstep))

                # L5-b as written, and the looser J form.
                b_written = sorted(unlocked - used)
                seen = retrieved_tool_names(s, tstep if tstep is not None else len(msgs))
                j = sorted(seen - used)

                # L5-c: transfer whose reason/summary differ from gold.
                l5c = False
                for c in ri.get("action_checks") or []:
                    a = c.get("action") or {}
                    if not c.get("action_match") and a.get("name") == TRANSFER:
                        l5c = True
                # L4: a completion claim with no matching call anywhere.
                claim = any(m.get("role") == "assistant" and isinstance(m.get("content"), str)
                            and DONE.search(m["content"]) for m in msgs)
                miss_assist = [c for c in ri.get("action_checks") or []
                               if not c.get("action_match")
                               and (c.get("action") or {}).get("requestor") == "assistant"]
                miss_user = [c for c in ri.get("action_checks") or []
                             if not c.get("action_match")
                             and (c.get("action") or {}).get("requestor") == "user"]
                l4 = claim and bool(miss_assist or miss_user)
                l1 = any((c.get("action") or {}).get("name") == "apply_for_credit_card"
                         for c in ri.get("action_checks") or [] if not c.get("action_match"))

                key = f"{arm} {s['task_id']}/t{s.get('trial')}"
                matrix.append({
                    "sim": key,
                    "L1": l1, "L4": l4,
                    "L5a": tstep is not None and kb_before == 0,
                    "L5b": bool(b_written), "L5bJ": bool(j), "L5c": l5c,
                })
                detail.append((key, kb_before, sorted(unlocked), sorted(called),
                               sorted(given), tstep, b_written, j[:6]))

    print("=== §1 이관 전제 실측 ===")
    print(f"{'sim':22s} {'KB전':>4s} {'unlock':>6s} {'call':>5s} {'give':>5s} "
          f"{'transfer':>8s}  L5b(해금-미호출)  J(회수-미사용)")
    for k, kb, u, c, g, t, bw, j in detail:
        print(f"{k:22s} {kb:4d} {len(u):6d} {len(c):5d} {len(g):5d} "
              f"{('step %s' % t) if t is not None else '-':>8s}  {len(bw):>2d} {bw[:2]}  "
              f"{len(j):>2d} {j[:3]}")

    nb = sum(1 for m in matrix if m["L5b"])
    nj = sum(1 for m in matrix if m["L5bJ"])
    print(f"\n  L5-b(명세 그대로) 발화 sim: {nb} / {len(matrix)}"
          + ("   ⇒ ★표적 공집합(C263)" if nb == 0 else ""))
    print(f"  L5-b(J 형태) 발화 sim   : {nj} / {len(matrix)}")

    print("\n=== §2 sim × 레버 표적 매트릭스 ===")
    cols = ["L1", "L4", "L5a", "L5bJ", "L5c"]
    print(f"{'sim':22s} " + " ".join(f"{c:>5s}" for c in cols) + "   합")
    hist = collections.Counter()
    for m in matrix:
        hit = [c for c in cols if m[c]]
        hist[len(hit)] += 1
        print(f"{m['sim']:22s} " + " ".join(f"{'●' if m[c] else '·':>5s}" for c in cols)
              + f"   {len(hit)}")
    print("\n  레버 개수별 sim 수:", dict(sorted(hist.items())))
    for c in cols:
        print(f"  {c:5s} 표적 {sum(1 for m in matrix if m[c])}")
    print(f"  어떤 레버도 안 닿는 sim: {hist[0]}")


if __name__ == "__main__":
    main()
