"""
build_dpo_pairs.py — §3 Rung1.5 ②: DPO preference pairs for the gather-act ORDERING.

Reads a readiness-gate SFT jsonl (from build_tbox_planner_sft.py --scratchpad). 2-token terminal
format (v2): `ready=false; <tool>` (gather) or `ready=true; preconds_verified=<t/f>; permitted=<t/f>;
<ACT|STOP>` (terminal). For each step it emits {prompt, chosen, rejected} where `rejected` is the
ORDERING-VIOLATING alternative — the negative signal that positive-only SFT cannot give:

  chosen = `ready=false; <tool>`                                    -> rejected = `...preconds_verified=true; permitted=true; ACT`  (premature ACT)
  chosen = `ready=true; preconds_verified=true; permitted=true; ACT`  -> rejected = `...permitted=false; STOP`                       (over-refuse)
  chosen = `ready=true; ...; STOP`                                  -> rejected = `...preconds_verified=true; permitted=true; ACT`  (under-refuse / premature)

So DPO teaches: do NOT ACT while ready=false (incomplete gather); do NOT flip a true gate to STOP;
do NOT ACT when a gate is false. Pairs are GT-derived (no reward model).

RUN:  python scripts/build_dpo_pairs.py --sft lodo_train_s1_scratch.jsonl --out dpo_s1.jsonl
"""
import argparse
import json
import re


def make_rejected(chosen: str):
    """Return the ordering-violating alternative for a readiness-gate target, or None to skip."""
    c = chosen.strip()
    low = c.lower()
    # 2-token format (build_tbox_planner_sft v2): ready=true; preconds_verified=<t/f>; permitted=<t/f>; <ACT|STOP>
    if low.startswith("ready=false"):
        # gather/establish step -> the violation is to ACT prematurely (claim both gates pass)
        return "ready=true; preconds_verified=true; permitted=true; ACT"
    if low.startswith("ready=true"):
        if re.search(r"\bact\b", low):            # should_T terminal (ACT) -> over-refuse (flip permitted)
            return "ready=true; preconds_verified=true; permitted=false; STOP"
        if re.search(r"\bstop\b", low):           # STOP terminal -> under-refuse / premature ACT
            return "ready=true; preconds_verified=true; permitted=true; ACT"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="readiness-gate SFT jsonl (build_tbox_planner_sft --scratchpad)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = nskip = 0
    from collections import Counter
    kind = Counter()
    with open(args.out, "w", encoding="utf-8") as w:
        for line in open(args.sft, encoding="utf-8"):
            if not line.strip():
                continue
            e = json.loads(line)
            msgs = e["messages"]
            prompt = msgs[0]["content"]
            chosen = msgs[1]["content"]
            rejected = make_rejected(chosen)
            if rejected is None or rejected.strip() == chosen.strip():
                nskip += 1
                continue
            kind[e.get("target_kind", "?")] += 1
            w.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                                "domain": e.get("domain"), "goal": e.get("goal"),
                                "target_kind": e.get("target_kind")}) + "\n")
            n += 1
    print(f"[dpo] {n} pairs -> {args.out} (skipped {nskip}); by kind: {dict(kind)}")


if __name__ == "__main__":
    main()
