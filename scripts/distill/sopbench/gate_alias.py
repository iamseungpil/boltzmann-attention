"""
gate_alias.py — G-alias ZERO-TRAIN learnability gate (TASK_CONSTRAINT_DESIGN §8.5.★).

Before spending a retrain on the alias / source-3 setting, check it is LEARNABLE: can a STRONG
model already produce the correct next-operator decisions from the (possibly aliased, NL-only)
prompts? Rationale (project lesson "zero-train first, n!=1"): arm-3v2/arm-4a sit at should_T
~2-4/48 on the EASIER named setting; aliasing raises difficulty. If even a 32B teacher floors
out on alias+source3, the tool DESCRIPTIONS / POLICY lack the signal a 7B would need -> fix the
descriptions before retraining. If the strong model scores high, the skill is real and worth
distilling into the 7B (TBox).

This is OFFLINE + teacher-forced: it reuses the SFT jsonl produced by build_tbox_planner_sft.py
(each row = {prompt, target, target_kind}); the prompt already carries the GT history, so we
score per-STEP next-operator accuracy (exactly what SFT optimizes), NOT a live rollout. No
SOPBench env import needed — only an OpenAI-compatible endpoint.

Run (compare the three regimes; build each jsonl first with the matching flags):
  python scripts/build_tbox_planner_sft.py --domain bank --out g                       # s1 named
  python scripts/build_tbox_planner_sft.py --domain bank --out g --source 3            # s3 named
  python scripts/build_tbox_planner_sft.py --domain bank --out g --alias --source 3    # s3 ALIAS
  for f in g/sft_tbox_bank.jsonl g/sft_tbox_bank_s3.jsonl g/sft_tbox_bank_alias_s3.jsonl; do
     python scripts/gate_alias.py --sft $f --model <strong> --base_url http://localhost:8000/v1
  done
Gate (pre-registered): alias+s3 first-step next-op accuracy >= 0.6 on a strong model => LEARNABLE.
"""
import argparse
import json
import re

from openai import OpenAI


def shown_tokens(prompt: str):
    """The tool tokens displayed to the model = the leading 'name' of each '- {name}: ...' line
    in the TOOLS block. Matches the planner's copy-grounded selection space (aliases when on)."""
    toks = []
    in_tools = False
    for line in prompt.splitlines():
        if line.startswith("TOOLS"):
            in_tools = True
            continue
        if in_tools:
            if line.startswith("- "):
                m = re.match(r"- ([^:]+):", line)
                if m:
                    toks.append(m.group(1).strip())
            elif line.strip() and not line.startswith("- ") and not line.startswith(" "):
                break          # left the TOOLS block
    return toks


def pick(raw: str, toks):
    """Copy-grounded de-ref of the model output to one shown token or STOP (mirrors _plan_v2)."""
    low = raw.strip().lower()
    if "stop" in low or "refuse" in low or "exit_conversation" in low:
        return "STOP"
    hits = [t for t in toks if t and t in raw]
    if hits:
        return max(hits, key=len)
    first = low.split()[0].strip(".,:\"'") if low.split() else ""
    for t in toks:
        if t.lower() == first:
            return t
    return "STOP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="SFT jsonl from build_tbox_planner_sft.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--max", type=int, default=0, help="cap #rows (0 = all)")
    ap.add_argument("--first_only", action="store_true",
                    help="score only the FIRST step of each task (the critical 'gather right check' call)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.sft, encoding="utf-8") if l.strip()]
    if args.first_only:
        seen, keep = set(), []
        for r in rows:
            key = (r.get("domain"), r.get("goal"))
            # first row per (domain,goal) in file order ~= first step (build emits steps in order)
            if key not in seen:
                seen.add(key)
                keep.append(r)
        rows = keep
    if args.max:
        rows = rows[:args.max]

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    from collections import Counter
    tot = Counter()
    ok = Counter()
    for r in rows:
        prompt = r["messages"][0]["content"]
        gold = r["messages"][1]["content"]
        kind = r.get("target_kind", "?")
        toks = shown_tokens(prompt)
        resp = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, top_p=0.01, max_tokens=24)
        out = resp.choices[0].message.content or ""
        chosen = pick(out, toks)
        correct = (chosen == gold) or (gold == "STOP" and chosen == "STOP")
        tot[kind] += 1
        tot["ALL"] += 1
        if correct:
            ok[kind] += 1
            ok["ALL"] += 1

    print(f"\n=== G-alias gate: {args.sft} | model={args.model} | "
          f"{'first-step' if args.first_only else 'per-step'} ===")
    for k in ["ALL", "GOAL", "establish", "STOP"]:
        if tot[k]:
            print(f"  {k:10} {ok[k]:4}/{tot[k]:<4} = {ok[k]/tot[k]:.3f}")
    a = ok["ALL"] / tot["ALL"] if tot["ALL"] else 0.0
    print(f"  -> next-op accuracy {a:.3f}  "
          f"({'LEARNABLE (>=0.6)' if a >= 0.6 else 'WEAK SIGNAL (<0.6): strengthen descriptions/policy'})")


if __name__ == "__main__":
    main()
