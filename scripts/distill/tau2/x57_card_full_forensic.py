"""Every card-application simulation, traced to why the customer applied for what they did.

The pair in task_003 suggested one mechanism — the agent narrows to a shortlist that
excludes the right card, and the customer can only apply for a card that was named.
This checks that against all of them, in both arms, and splits the branches the pair
could not show:

  gold in eligible / excluded / unverified   whether our own filter dropped the answer,
                                             which would be a different failure entirely
  named-set and first-named                  what the agent put in front of the customer
  asked-for-list                             whether the customer had to demand the list
  applied                                    what the customer actually did

A failure where the right card was never eligible is our filter's fault. One where it
was eligible but never named is the shortlist. One where it was named and still not
chosen is neither, and needs its own reading.
"""

import argparse
import ast
import collections
import glob
import gzip
import json
import re

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")
ARMS = {"A": "bank_ax33n_gpu*_20260803g", "B4": "bank_b4_gpu*_20260803h"}

ASK_LIST = re.compile(
    r"\b(list|which (cards|ones|of your)|what (cards|options)|options (that|do)|"
    r"compare|best option|top recommendation|other (cards|options))\b", re.I)


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def parse_fit(text):
    """The eligible / excluded / unverified buckets as the tool actually reported them."""
    i = text.find("{'eligible'")
    if i < 0:
        return None
    try:
        d = ast.literal_eval(text[i:])
    except Exception:
        # Fall back to reading the bucket names positionally rather than guessing.
        out = {}
        for b in ("eligible", "excluded", "unverified"):
            m = re.search(rf"'{b}': \[(.*?)\](?=, '(?:eligible|excluded|unverified|note)')",
                          text, re.S)
            out[b] = re.findall(r"'card': '([^']+)'", m.group(1)) if m else []
        return out
    return {b: [c.get("card") for c in (d.get(b) or []) if isinstance(c, dict)]
            for b in ("eligible", "excluded", "unverified")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    args = ap.parse_args()

    rows = []
    for arm in args.arms.split(","):
        for p in sorted(glob.glob(f"{SIM}/{ARMS[arm]}.results.json.gz")):
            for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
                msgs = s.get("messages") or []
                gold = None
                for c in (s.get("reward_info") or {}).get("action_checks") or []:
                    a = c.get("action") or {}
                    if a.get("name") == "apply_for_credit_card":
                        gold = norm(a.get("arguments")).get("card_type")
                if not gold:
                    continue

                buckets, applied, act = None, None, len(msgs)
                for i, m in enumerate(msgs):
                    if m.get("role") == "tool" and isinstance(m.get("content"), str):
                        b = parse_fit(m["content"])
                        if b and buckets is None:
                            buckets = b
                    if m.get("role") == "user":
                        for tc in m.get("tool_calls") or []:
                            if (tc.get("name") or "") == "apply_for_credit_card":
                                applied = norm(tc.get("arguments")).get("card_type")
                                act = min(act, i)
                if buckets is None:
                    rows.append({"arm": arm, "sim": f"{s['task_id']}/t{s.get('trial')}",
                                 "pass": ((s.get("reward_info") or {}).get("reward") or 0) == 1.0,
                                 "where": "fit 미호출", "named": [], "first": None,
                                 "asked": 0, "gold": gold, "applied": applied})
                    continue

                where = next((b for b in ("eligible", "excluded", "unverified")
                              if gold in (buckets.get(b) or [])), "표에 없음")
                allc = [c for b in buckets.values() for c in b]
                named, first = [], None
                for m in msgs[:act]:
                    if m.get("role") != "assistant" or not isinstance(m.get("content"), str):
                        continue
                    for c in allc:
                        if c.lower() in m["content"].lower() and c not in named:
                            named.append(c)
                            if first is None:
                                first = c
                asked = sum(1 for m in msgs[:act] if m.get("role") == "user"
                            and isinstance(m.get("content"), str) and ASK_LIST.search(m["content"]))
                rows.append({
                    "arm": arm, "sim": f"{s['task_id']}/t{s.get('trial')}",
                    "pass": ((s.get("reward_info") or {}).get("reward") or 0.0) == 1.0,
                    "where": where, "elig": len(buckets.get("eligible") or []),
                    "named": named, "first": first, "asked": asked,
                    "gold": gold, "applied": applied,
                })

    for r in sorted(rows, key=lambda x: (x["sim"], x["arm"])):
        ok = "PASS" if r["pass"] else "FAIL"
        print(f"{r['arm']:3s} {r['sim']:14s} {ok}  gold={r['gold']} → {r['where']}"
              f" · 적격 {r.get('elig', '-')}")
        print(f"    에이전트 언급 {len(r['named'])}장 (첫={r['first']}) {r['named']}")
        print(f"    손님 목록요구 {r['asked']}회 · 신청={r['applied']}"
              f"{'  ★gold 미언급' if r['gold'] not in r['named'] else ''}")

    print("\n=== gold가 어느 버킷에 있었나 × 통과 ===")
    t = collections.Counter()
    for r in rows:
        t[(r["where"], r["pass"])] += 1
    for w in sorted({k[0] for k in t}):
        p, f = t[(w, True)], t[(w, False)]
        print(f"  {w:10s} 통과 {p} / 실패 {f}")

    print("\n=== 실패 사유 분해 ===")
    d = collections.Counter()
    for r in rows:
        if r["pass"]:
            continue
        if r["where"] == "excluded":
            d["우리 필터가 gold를 배제"] += 1
        elif r["where"] in ("표에 없음", "fit 미호출"):
            d["gold가 표에 아예 없음/fit 미호출"] += 1
        elif r["gold"] not in r["named"]:
            d["★적격이었으나 에이전트가 언급 안 함"] += 1
        elif r["applied"] is None:
            d["언급했으나 손님이 신청 자체를 안 함"] += 1
        else:
            d["언급했는데 손님이 다른 카드 신청"] += 1
    for k, v in d.most_common():
        print(f"  {k:34s} {v}")


if __name__ == "__main__":
    main()
