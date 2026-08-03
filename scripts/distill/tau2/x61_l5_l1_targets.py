"""What do the hand-off preconditions fire on — and what do they cost on runs that pass?

`x59` counted lever targets over failures only. That answers half the question. A
precondition that fires on 3 failures and 9 passes is not a lever, it is a regression,
and the design's own counter-metric list says so (§5: "이관 자체를 막지 않았는가"). L4
died on exactly this asymmetry once the implementable predicate was measured (`x60`),
so the remaining levers get the same treatment before any of them is written.

Every predicate here is the one the engine can actually evaluate at generation time —
trajectory facts and the call's own arguments, no gold, no reward.

  L5-a  a transfer call with zero KB_search before it
  L5-c  a transfer whose `summary` names none of the tools the run actually called
  L5-b′ a transfer whose `summary` states no retrieval count at all

Reported per arm as (fires on failing sims, fires on passing sims). The second column
is the price.
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
TRANSFER = "transfer_to_human_agents"

# Wrappers, search and bookkeeping: naming one of these in a summary says nothing about
# what was attempted, so they cannot satisfy L5-c. Mirrors the engine's own
# `_PROCEDURAL_RE` plus the A2-declared dispatchers, so the count here is the count the
# implemented predicate will produce — the two must not drift.
NOISE = re.compile(r"(^log_|^verify_|_verification$|^kb_|^shell$|transfer_to_human"
                   r"|^think$|^(call|unlock|give|list)_discoverable)", re.I)
# Any digit-bearing phrase about documents/results — the weakest possible form of
# "I read k of them", so L5-b′'s target is not inflated by demanding a fixed wording.
COUNT_RE = re.compile(r"\b\d+\s+(of\s+\d+\s+)?(document|doc|result|record|article|polic)", re.I)


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def walk(sim):
    out = []
    for m in sim.get("messages") or []:
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
            out.append((n, inner, args))
    return out


def quotes_attempt(summary, attempted):
    """Does the summary name something the run actually called?

    Matching is on the tool's word stem, not the exact identifier: an agent writing
    "I filed the dispute" has cited `submit_cash_back_dispute_0589` in the only way a
    customer-facing sentence can. Requiring the identifier verbatim would make the
    predicate fire on every well-written summary.
    """
    s = (summary or "").lower()
    if not s:
        return False
    for name in attempted:
        stem = re.sub(r"_\d{3,4}$", "", name).lower()
        words = [w for w in stem.split("_") if len(w) >= 4]
        if words and all(w in s for w in words):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    ap.add_argument("--sim", default="")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    sim_dir = args.sim or (SIM_LOCAL if os.path.isdir(SIM_LOCAL) else SIM_REMOTE)
    print(f"sim_dir = {sim_dir}")

    for arm in args.arms.split(","):
        print(f"\n{'=' * 78}\n[{arm}]  {ARMS[arm]}\n{'=' * 78}")
        sims = []
        for p in sorted(glob.glob(os.path.join(sim_dir, f"{ARMS[arm]}.results.json.gz"))):
            print(f"  read {os.path.basename(p)}")
            sims.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])

        t = collections.Counter()
        rows = []
        for s in sorted(sims, key=lambda x: (x["task_id"], x.get("trial") or 0)):
            ok = ((s.get("reward_info") or {}).get("reward") or 0.0) == 1.0
            side = "pass" if ok else "fail"
            cl = walk(s)
            t[f"sim·{side}"] += 1
            tidx = next((i for i, (n, _, _) in enumerate(cl) if n == TRANSFER), None)
            if tidx is None:
                t[f"이관 없음·{side}"] += 1
                continue
            t[f"이관 있음·{side}"] += 1
            kb_before = sum(1 for n, _, _ in cl[:tidx] if n.startswith("KB_search"))
            attempted = {(inner or n) for n, inner, _ in cl[:tidx]
                         if not NOISE.match(inner or n)}
            summary = str(cl[tidx][2].get("summary") or "")

            a_fire = kb_before == 0
            c_fire = not quotes_attempt(summary, attempted)
            b_fire = not COUNT_RE.search(summary)
            for k, fire in (("L5a", a_fire), ("L5c", c_fire), ("L5b′", b_fire)):
                if fire:
                    t[f"{k}·{side}"] += 1
            rows.append((f"{s['task_id']}/t{s.get('trial')}", side, kb_before,
                         len(attempted), a_fire, c_fire, b_fire, summary[:90]))

        print(f"\n  완주 {t['sim·pass'] + t['sim·fail']} "
              f"(pass {t['sim·pass']} / fail {t['sim·fail']}) · "
              f"이관 호출 sim = fail {t['이관 있음·fail']} / pass {t['이관 있음·pass']}")
        print(f"\n  {'레버':6s} {'표적(fail)':>10s} {'대가(pass)':>10s}   술어")
        for k, desc in (("L5a", "이관 전 KB_search 0회"),
                        ("L5c", "summary가 호출한 도구를 하나도 인용 안 함"),
                        ("L5b′", "summary에 회수 계수 진술 없음")):
            print(f"  {k:6s} {t[k + '·fail']:10d} {t[k + '·pass']:10d}   {desc}")
        print("\n  ※ 대가 열 = 통과한 sim에서 같은 술어가 발화한 수 = 그만큼 정당한 이관을 막는다.")

        if args.detail:
            print(f"\n  {'sim':16s} {'':4s} {'KB전':>4s} {'시도':>4s}  a c b  summary")
            for key, side, kb, na, a_f, c_f, b_f, sm in rows:
                print(f"  {key:16s} {side:4s} {kb:4d} {na:4d}  "
                      f"{int(a_f)} {int(c_f)} {int(b_f)}  {sm!r}")


if __name__ == "__main__":
    main()
