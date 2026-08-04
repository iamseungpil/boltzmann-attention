"""What did the match-count note actually say when it ran?

Registering `T2_MATCH_COUNT` again is only worth it if the line carries information. The
lever exists to make "195 matched, 8 shown" distinguishable from "4 matched, all 4 shown"
— the second is a boundary certificate, the first is an instruction to narrow. A note that
mostly says "no document contains all of these words" carries neither.

B4 is the only arm that ran it, so the distribution is read there, from the transcripts
rather than the driver log: the log can repeat a line the model never saw, and what
matters is what reached the conversation.

Counting rule, stated because two counts of this disagreed: one occurrence per regex match
of `matches: …` inside a role=tool message body, deduplicated by nothing — a message
carrying three notes counts three. Driver-log lines are deliberately excluded: the log can
hold a line the model never saw, and only what reached the conversation can have acted.

Classification is by the emitting branch of `t2_match_count.note`, matched on a marker that
is unique to each — the first version of this script tested for "all" and silently folded
the partial form into the certificate, because every branch says "all of these words":

  0건     `matches: no document contains all of these words; K shown by ranking`
          → starts with "no document"
  전부표시  `matches: N documents contain all of these words; all N shown`
          → contains "; all "          ← the certificate: everything that matched was seen
  부분표시  `matches: N documents contain all of these words; K shown (M not shown)`
          → contains "not shown)"      ← the narrow-your-query signal
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import SIM  # noqa: E402

NOTE = re.compile(r"matches: [^\n]{0,90}")


def classify(note):
    """분기별 **고유 표지**로 가른다 — 세 문구가 전부 "all of these words"를 담고 있어서
    "all" 포함 여부로 가르면 부분표시가 인증으로 접힌다(초판 실측 149 = 126 + 23)."""
    if "no document" in note:
        return "0건 — 이 낱말 전부를 담은 문서 없음"
    if "not shown)" in note:
        return "부분 표시 — 좁히라는 신호"
    if "; all " in note:
        return "전부 표시 — 경계 인증 성립"
    return "미분류(문구 변경?)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="bank_b4_gpu*_20260803h")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{SIM}/{args.glob}.results.json.gz"))
    if not files:
        raise SystemExit(f"no runs matched {args.glob}")

    kind = collections.Counter()
    sims_with = set()
    examples = {}
    total_sims = 0
    for p in files:
        for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
            total_sims += 1
            for m in s.get("messages") or []:
                if m.get("role") != "tool" or not isinstance(m.get("content"), str):
                    continue
                for note in NOTE.findall(m["content"]):
                    k = classify(note)
                    kind[k] += 1
                    sims_with.add((s["task_id"], s.get("trial")))
                    examples.setdefault(k, note.strip())

    total = sum(kind.values())
    print(f"{args.glob} · sim {total_sims} · 주석 {total}건 / {len(sims_with)} sim\n")
    for k, n in kind.most_common():
        print(f"  {k:<28} {n:>4}  ({n / max(1, total):.0%})")
        print(f"      예: {examples[k]}")
    cert = kind["전부 표시 — 경계 인증 성립"]
    print(f"\n경계 인증 비율 = {cert}/{total} = {cert / max(1, total):.0%}")
    print("등재 판정 기준(설계서 §4): 인증 비율이 과반이면 등재, 아니면 문구·연산 수정이 선결.")


if __name__ == "__main__":
    main()
