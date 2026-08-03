"""Why does the agent name exactly one card, and why does it assert unverified booleans?

M1 and M2a are described but not explained. This reads the sentence that introduces the
losing card in every M1 failure, and every argument the fit tool was called with, so the
cause is taken from the text rather than guessed.

Two candidate causes are checked explicitly, because both would be ours rather than the
model's:

  instruction-shaped   the card name appears inside an explanation of how to call the
                       apply tool — an example value in a usage instruction, not an
                       answer to "which cards qualify". We added user-tool argument
                       guidance; it would produce exactly this shape.
  boolean default      unknown booleans arriving as "false" while unknown numbers arrive
                       as null. For `invited`, false is not absence of a constraint — it
                       asserts the customer was not invited and deletes the answer.
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

# Telling the customer how to run the tool, as opposed to telling them which card fits.
HOWTO = re.compile(
    r"(you can use the|please use the|to apply for|run the|use the following|"
    r"here are the arguments|arguments you need|for example|e\.g\.|`card_type`|"
    r"card_type:)", re.I)


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    args = ap.parse_args()

    shapes = collections.Counter()
    argfill = collections.Counter()
    print("=== M1: 오답 카드를 처음 꺼낸 문장 ===")
    for arm in args.arms.split(","):
        for p in sorted(glob.glob(f"{SIM}/{ARMS[arm]}.results.json.gz")):
            for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
                msgs = s.get("messages") or []
                gold = applied = None
                for c in (s.get("reward_info") or {}).get("action_checks") or []:
                    a = c.get("action") or {}
                    if a.get("name") == "apply_for_credit_card":
                        gold = norm(a.get("arguments")).get("card_type")
                if not gold:
                    continue
                for m in msgs:
                    if m.get("role") == "user":
                        for tc in m.get("tool_calls") or []:
                            if (tc.get("name") or "") == "apply_for_credit_card":
                                applied = norm(tc.get("arguments")).get("card_type")

                # How every fit call filled its parameters.
                for m in msgs:
                    for tc in m.get("tool_calls") or []:
                        if "fit" not in (tc.get("name") or ""):
                            continue
                        a = norm(tc.get("arguments"))
                        for k, v in a.items():
                            if v is None:
                                argfill[(k, "null")] += 1
                            elif str(v).lower() in ("true", "false"):
                                argfill[(k, str(v).lower())] += 1
                            else:
                                argfill[(k, "value")] += 1

                if not applied or applied == gold:
                    continue
                for i, m in enumerate(msgs):
                    if m.get("role") != "assistant" or not isinstance(m.get("content"), str):
                        continue
                    if applied.lower() not in m["content"].lower():
                        continue
                    txt = " ".join(m["content"].split())
                    shape = "지시형(도구 사용법)" if HOWTO.search(txt[:400]) else "답변형(추천/비교)"
                    shapes[shape] += 1
                    print(f"\n  {arm} {s['task_id']}/t{s.get('trial')}  step {i}  [{shape}]")
                    print(f"    gold={gold} · 손님 신청={applied}")
                    print(f"    {txt[:260]}")
                    break

    print("\n=== 문장 형태 집계 ===")
    for k, v in shapes.most_common():
        print(f"  {k:20s} {v}")

    print("\n=== fit 인자를 어떻게 채웠나 (전 호출) ===")
    keys = sorted({k for k, _ in argfill})
    print(f"  {'인자':26s} {'null':>6s} {'true':>6s} {'false':>6s} {'값':>6s}")
    for k in keys:
        print(f"  {k:26s} {argfill[(k,'null')]:6d} {argfill[(k,'true')]:6d} "
              f"{argfill[(k,'false')]:6d} {argfill[(k,'value')]:6d}")


if __name__ == "__main__":
    main()
