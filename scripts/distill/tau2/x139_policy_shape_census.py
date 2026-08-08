# -*- coding: utf-8 -*-
"""x139 — 정책 문장이 **(주어, 축, 값)으로 떨어지는가** 전수 인구조사 (유료 0 · LLM 0).

왜: `A3_POLICY_ONTOLOGY_DESIGN_2026_08_08.md` §8이 *"조건절 정책은 A3 밖"* 이라고만 적어 두었다.
그 비율을 모르면 **A3가 정책의 몇 %를 덮는지 모른 채** *"수십 행이면 된다"* 고 말하는 셈이다.
이 프로브가 §9-3을 닫는다.

가르는 축(문장 단위):
  단순 — 수치가 있고 **조건 표지가 없다** ⇒ `(주어, 축, 값)` 한 행으로 떨어진다
  조건 — `if/unless/except/provided/only if/when/subject to/must also/in addition` 등이 붙는다
  계층 — `first N … thereafter`, `up to … then`, 등급/구간이 나뉜다
  관계 — 주어가 **둘 이상**(예: 추천인 ∧ 피추천인)이라 한 주어에 못 붙는다

⚠**분석 도구이지 엔진이 아니다**([[59]]는 엔진을 규율한다). 여기서 나온 문자열은 A3·엔진에
들어가지 않는다. 그리고 표지는 **내가 고른 것**이라 분류는 근사다 — 그래서 **표본을 함께 인쇄**해
per-case로 읽게 한다([[08]]).

usage: x139_policy_shape_census.py --docs <dir> [--sample 4]
"""

import argparse
import collections
import glob
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NUM = re.compile(r"\b\d[\d,]*\b")
COND = re.compile(r"\b(if|unless|except|provided that|only if|only when|when the|subject to|"
                  r"must also|in addition|otherwise|but not|does not apply|waived)\b", re.I)
TIER = re.compile(r"\b(first \w+|thereafter|per tier|tier \d|up to .{0,20}then|"
                  r"whichever|beyond the|after the first)\b", re.I)
REL = re.compile(r"\b(referred (person|business|friend|customer)|the person you refer|"
                 r"both .{0,20}and|each party)\b", re.I)
# 정책적 수치를 담은 문장만 본다 — 순수 서술·인사말 제외
AXIS = re.compile(r"\b(days?|limit|maximum|minimum|per year|per calendar year|bonus|"
                  r"deposit|spend|purchases|tenure|duration)\b", re.I)


def sentences(text):
    t = str(text or "").replace("\r", "")
    parts = re.split(r"(?<=[.!?])\s+|\n(?=[-*#|])|\n{2,}", t)
    return [" ".join(p.split()) for p in parts if p and p.strip()]


def classify(s):
    if REL.search(s):
        return "관계(주어 둘 이상)"
    if TIER.search(s):
        return "계층(구간·등급)"
    if COND.search(s):
        return "조건절"
    return "단순"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--sample", type=int, default=4)
    a = ap.parse_args()

    cnt = collections.Counter()
    samples = collections.defaultdict(list)
    docs = sorted(glob.glob(os.path.join(a.docs, "*.json")))
    n_sent = 0
    for p in docs:
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        for s in sentences(d.get("content")):
            n_sent += 1
            if not (NUM.search(s) and AXIS.search(s)):
                continue
            k = classify(s)
            cnt[k] += 1
            if len(samples[k]) < a.sample:
                samples[k].append((d.get("id", "?"), s))

    tot = sum(cnt.values())
    print("문서 %d개 · 문장 %d개 · **정책 수치 문장 %d개**\n" % (len(docs), n_sent, tot))
    print("| 형태 | 건수 | 비율 | A3 (주어,축,값) 한 행으로? |")
    print("|---|---|---|---|")
    for k in ("단순", "조건절", "계층(구간·등급)", "관계(주어 둘 이상)"):
        c = cnt.get(k, 0)
        ok = "**예**" if k == "단순" else "아니오"
        print("| %s | %d | %.0f%% | %s |" % (k, c, 100.0 * c / tot if tot else 0, ok))
    print("\n⇒ **A3가 한 행으로 덮는 비율 = %.0f%%**" % (100.0 * cnt.get("단순", 0) / tot if tot else 0))

    for k in ("단순", "조건절", "계층(구간·등급)", "관계(주어 둘 이상)"):
        if not samples[k]:
            continue
        print("\n── %s 표본 ──" % k)
        for did, s in samples[k]:
            print("   [%s] %s" % (did[-24:], s[:150]))
    print("\n⚠표지는 내가 고른 것이라 분류는 근사다 — 위 표본을 per-case로 읽고 판정할 것([[08]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
