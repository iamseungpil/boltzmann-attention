#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""039 wrong-pick 격리 프로브 (2026-07-23·유저 지시 "부하 문제 아닌가? 격리 실험하라").
등대 §1.4 load(s)=p_iso−p_traj>0 판정 — 정보-맞춘 격리([[08]]): rall19 treat 039.0의
결정 시점 정보(유저 8건 서술 + 57건 거래 목록 [+전체 대화 프리픽스])를 그대로 제시하고
8건의 transaction_id 선택만 격리 질의. 채점=gold 8(ordered·per-item).
변형: A_minimal(서술+목록만) / B_fullctx(결정 직전 64-msg 플랫 transcript).
Run(remote): seka python probe_039_join_iso.py --base http://localhost:8140/v1 --n 8
"""
import argparse
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ART = json.load(open(os.path.join(HERE, "probe_039_join_artifacts.json"), encoding="utf-8"))
GOLD = ART["gold"]

ASK = ("Based on the customer's list of 8 disputed items and the transaction history above, "
       "identify the transaction_id for EACH of the customer's 8 items, in the same order "
       "(items 1-8). Answer with ONLY a JSON array of 8 transaction_id strings, nothing else.")


def flat_prefix():
    lines = []
    for m in ART["prefix"]:
        r = m.get("role")
        c = str(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            lines.append("[%s TOOL_CALL] %s %s" % (r, tc.get("name"),
                                                   str(tc.get("arguments"))[:400]))
        if c:
            lines.append("[%s] %s" % (r, c))
    return "\n".join(lines)


def build(variant):
    if variant == "A_minimal":
        u = ("CUSTOMER'S DISPUTE LIST:\n" + ART["user_list"] + "\n\nTRANSACTION HISTORY:\n"
             + ART["txn_listing"] + "\n\n" + ASK)
    else:
        u = ("Below is a bank-agent conversation transcript (including tool outputs).\n\n"
             + flat_prefix() + "\n\n" + ASK)
    return [{"role": "system", "content": "You are a precise banking assistant."},
            {"role": "user", "content": u}]


def call(base, msgs, temp, model):
    r = requests.post(base + "/chat/completions", json={
        "model": model, "messages": msgs, "temperature": temp, "max_tokens": 500}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def score(txt):
    ids = re.findall(r"txn_[0-9a-f]{6,}", txt)
    seen = list(dict.fromkeys(ids))[:8]
    per = [i < len(seen) and seen[i] == GOLD[i] for i in range(8)]
    return {"n_ids": len(seen), "ordered_correct": sum(per), "per_item": per,
            "set_overlap": len(set(seen) & set(GOLD)), "item3_ok": (len(seen) > 2 and seen[2] == GOLD[2]),
            "picked_41735": "txn_41735bd2d06d" in seen, "ids": seen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    print("GOLD:", GOLD)
    for variant in ("A_minimal", "B_fullctx"):
        msgs = build(variant)
        runs = [("greedy", 0.0)] + [("s%d" % i, 0.7) for i in range(a.n)]
        agg = []
        for tag, t in runs:
            try:
                s = score(call(a.base, msgs, t, a.model))
            except Exception as e:
                print("%s %s ERROR %r" % (variant, tag, e))
                continue
            agg.append(s)
            print("%s %s ordered=%d/8 set=%d/8 item3=%s pick41735=%s"
                  % (variant, tag, s["ordered_correct"], s["set_overlap"],
                     s["item3_ok"], s["picked_41735"]))
        if agg:
            print("== %s SUMMARY: mean_set=%.2f/8 item3_ok=%d/%d 41735=%d/%d"
                  % (variant, sum(x["set_overlap"] for x in agg) / len(agg),
                     sum(x["item3_ok"] for x in agg), len(agg),
                     sum(x["picked_41735"] for x in agg), len(agg)))


if __name__ == "__main__":
    main()
