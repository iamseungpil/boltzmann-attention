"""The grounding check is asymmetric: it audits the numbers and waves the booleans through.

`ACTION_HANDOFF_LEVERS_DESIGN §3 L1″-b` calls this ours to fix, and 024/t0 is the
witness — `min_credit_limit=40000` was dropped as an invented constraint while
`business="true"` and `invited="false"` went straight into the filter. The reason is in
A2, not in the engine: every parameter declared under `ground.intent_fields` is numeric,
so no boolean has a cue to be checked against.

The asserted `false` is not a neutral default. `invited=false` is a claim that the
customer was not invited, and it deletes the invitation-only cards from the catalog
before the customer ever sees them. The measured fill was `false` 18 / null 7 / true 0.

This counts, over both arms, how often a boolean was asserted with nothing in the
customer's own words to support it — which is the target L1″-b would act on — and
splits it by whether the run passed, because the same expansion on a passing run is the
price.
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

FIT = "check_card_application_fit"

# Cue vocabulary taken from the tool's own parameter descriptions in A2 — the same place
# the numeric cues came from. Nothing here is read off gold: `invited` is described as
# "(Diamond Elite invitation)", `premium_subscriber` as "the bank's premium
# subscription", and so on.
CUES = {
    "business": ["business", "company", "llc", "corporate", "my shop", "my store"],
    "invited": ["invit", "invitation", "diamond elite"],
    "needs_purchase_protection": ["purchase protection", "protection"],
    "premium_subscriber": ["premium", "subscription", "subscriber"],
}


def truthy_assert(v):
    """Was the parameter *asserted* rather than left unknown?

    None and "" are the honest unknown. Both `true` and `false` are assertions — the
    whole point is that a `false` the customer never said is as invented as a `true`.
    """
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "false", "1", "0", "yes", "no")


def user_text_before(sim, upto):
    return " ".join(str(m.get("content") or "") for m in (sim.get("messages") or [])[:upto]
                    if m.get("role") == "user").lower()


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
            sims.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])

        fill = collections.Counter()
        ungrounded = collections.Counter()
        sims_hit = {"pass": set(), "fail": set()}
        rows = []
        for s in sorted(sims, key=lambda x: (x["task_id"], x.get("trial") or 0)):
            ok = ((s.get("reward_info") or {}).get("reward") or 0.0) == 1.0
            side = "pass" if ok else "fail"
            key = f"{s['task_id']}/t{s.get('trial')}"
            for i, m in enumerate(sim_msgs := (s.get("messages") or [])):
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    if (tc.get("name") or "") != FIT:
                        continue
                    a = tc.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except Exception:
                            a = {}
                    a = a if isinstance(a, dict) else {}
                    utext = user_text_before(s, i)
                    bad = []
                    for p, cues in CUES.items():
                        v = a.get(p)
                        fill[f"{p}={str(v).lower() if v is not None else 'null'}"] += 1
                        if not truthy_assert(v):
                            continue
                        if not any(re.search(re.escape(c), utext) for c in cues):
                            ungrounded[p] += 1
                            bad.append(f"{p}={v}")
                    if bad:
                        sims_hit[side].add(key)
                        rows.append((key, side, bad))
            _ = sim_msgs

        print("\n  [불리언 인자 채움 분포]")
        for k, v in sorted(fill.items()):
            print(f"    {k:38s} {v}")
        print("\n  [손님이 말한 적 없는데 단언된 불리언 — L1″-b 표적]")
        for k, v in ungrounded.most_common():
            print(f"    {k:38s} {v}회")
        print(f"\n  발화 sim: fail {len(sims_hit['fail'])} · pass {len(sims_hit['pass'])}")

        if args.detail:
            for key, side, bad in rows:
                print(f"    {key:16s} {side:4s} {'; '.join(bad)}")


if __name__ == "__main__":
    main()
