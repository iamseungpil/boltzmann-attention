"""What the notice toll cost, in turns.

The GB2 gate opens only when the agent has uttered a fixed sentence, judged by
`notice_sent_in` = first 48 characters of the normalized notice as a substring.
Paraphrases do not count. This measures how many turns the run spent producing
utterances that meant "shall I transfer you" but missed that prefix.

This is the baseline for the rev4 ① change (§5 1급 지표 1번). It must be run
against the OLD notice text, which is what the recorded run was played under —
pass --notice to re-run it against any other wording.

Decidable definitions (no semantic judgement):
  transfer-intent utterance  an assistant message whose normalized text contains
                             one of INTENT_MARKERS
  hit                        that utterance carries the compelled prefix48
  miss                       it does not — a turn spent without opening the gate
  permanently blocked        a simulation with >=1 intent utterance, 0 hits, and
                             no transfer_to_human_agents call
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import notice_norm  # noqa: E402

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"

OLD_NOTICE = ("TRANSFER NOTICE: I have checked the knowledge base and there are no further "
              "actions I can take for this request. Would you like to be transferred to a "
              "human agent?")

INTENT_MARKERS = (
    "transfer notice",
    "transferred to a human agent",
    "transfer you to a human agent",
    "transfer you to a human",
    "connect you with a human agent",
    "would you like to be transferred",
)


def sims(tag):
    for p in sorted(glob.glob(f"{SIM}/bank_ax33n_gpu*_{tag}.results.json.gz")):
        for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
            yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260803g")
    ap.add_argument("--notice", default=OLD_NOTICE)
    ap.add_argument("--prefix", type=int, default=48)
    args = ap.parse_args()

    key = notice_norm(args.notice)[: args.prefix]
    print(f"compelled prefix{args.prefix} = {key!r}\n")

    per_sim, rows = {}, []
    for s in sims(args.tag):
        sid = (s.get("task_id"), s.get("trial"))
        hits = misses = 0
        called = False
        # Only utterances before the toll is paid count as spent turns. Once the
        # prefix has landed or the transfer has gone through, later mentions of
        # transferring are confirmations, not attempts.
        paid = False
        for m in s.get("messages") or []:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                n = tc.get("name") or (tc.get("function") or {}).get("name")
                if n == "transfer_to_human_agents":
                    called = True
                    paid = True
            c = m.get("content")
            if not isinstance(c, str) or not c:
                continue
            nc = notice_norm(c)
            if not any(mk in nc for mk in INTENT_MARKERS):
                continue
            if key and key in nc:
                hits += 1
                paid = True
            elif not paid:
                misses += 1
                rows.append((s.get("task_id"), s.get("trial"), " ".join(c.split())[:90]))
        per_sim[sid] = {"hits": hits, "misses": misses, "called": called,
                        "reward": (s.get("reward_info") or {}).get("reward") or 0.0}

    intents = sum(v["hits"] + v["misses"] for v in per_sim.values())
    misses = sum(v["misses"] for v in per_sim.values())
    hits = sum(v["hits"] for v in per_sim.values())
    miss_sims = [k for k, v in per_sim.items() if v["misses"]]
    blocked = [k for k, v in per_sim.items()
               if (v["hits"] + v["misses"]) and not v["hits"] and not v["called"]]

    print(f"이관-의도 발화 : {intents}")
    print(f"  prefix 일치  : {hits}")
    print(f"  미일치(낭비) : {misses}" + (f"  ({misses / intents:.0%})" if intents else ""))
    print(f"미일치 발생 sim: {len(miss_sims)} / {len(per_sim)}")
    print(f"끝내 못 맞춘 sim(이관 호출 0): {len(blocked)}  {sorted(blocked)}")

    worst = sorted(per_sim.items(), key=lambda kv: -kv[1]["misses"])[:6]
    print("\n최악 sim (미일치 수):")
    for (t, tr), v in worst:
        if not v["misses"]:
            break
        print(f"  {t}/t{tr}  miss={v['misses']} hit={v['hits']} called={v['called']} rew={v['reward']}")

    print("\n미일치 발화 표본:")
    for r in rows[:8]:
        print(f"  {r[0]}/t{r[1]}: {r[2]}")

    band = collections.Counter(min(v["misses"], 5) for v in per_sim.values() if v["misses"])
    print("\n분포(미일치/ sim):", dict(sorted(band.items())))


if __name__ == "__main__":
    main()
