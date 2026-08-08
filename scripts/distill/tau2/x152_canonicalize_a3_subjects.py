# -*- coding: utf-8 -*-
"""x152 — A3 주어 **이름 통일**(사용자 지시 2026-08-08: *"상품 이름을 통일하라"*). 유료 0·오프라인.

## 왜

A3는 한 상품을 **여러 표기로** 들고 있고 축이 그 표기들로 **갈려** 있다:

    World Blue International Checking   referrer_tenure_days = 90     (…world_blue_001)
    World Blue Balance                  referrer_bonus_usd   = 300    (…world_blue_007)
    Cobalt Blue                         referrer_tenure_days = 60     (…cobalt_blue_005)
    Cobalt Blue Business Checking       referrer_bonus_usd   = 150    (…cobalt_blue_001)

그래서 자격 필터를 주어 단위로 돌리면 보너스 쪽 표기가 **문턱 없이 통과**한다 — 65일 손님에게
`World Blue`(문턱 90)를 최고액으로 추천하는 문장을 우리가 직접 만들어 준다. 런 f 의 오답이 그것이다.

## 통일 근거 ([[23]] 출처 = **정책 문서 축자 + env DB**, gold 무참조)

⒜ **문서 본문이 같은 상품이라고 말한다.** `…world_blue_007` 제목은 *"Earning Interest on Your
   World Blue Balance"* 이고 본문은 *"referring other international businesses **to World Blue**"* ·
   *"Referral bonuses for **World Blue** members"* 다. 즉 `Balance` 는 상품명이 아니라 **잔액**이고,
   x140 이 제목의 명사구를 주어로 뽑으면서 상품명처럼 굳힌 것이다(추출 결함).
⒝ **원천에 통일된 표기가 없다.** 같은 상품을 문서가 꼬리말만 바꿔 부른다 —
   *"Navigating Currency Conversions on **World Blue**"* / *"Getting Started with **World Blue
   International Checking**"*, *"**Cobalt Blue**: Payments and Transfers Explained"* /
   *"Getting Started with **Cobalt Blue Business Checking**"* / *"**Cobalt Blue Account**: ATM Fees…"*.
   DB `accounts.level` 도 갈린다(`Navy Blue`·`Cobalt Blue` 는 꼬리말 없이, `Green Account`·
   `Light Blue Account` 는 붙여서). ⇒ **일관된 것은 머리 어구뿐**이다.
⒞ 그래서 정본 = **머리 어구**. 이것은 A2 `name_rules` 가 이미 선언한 규칙과 **같은 것**이다
   (*"the identity is the leading name phrase; trailing product-form words … describe the form of the
   product, not which product it is"*). 새 지식이 아니라 그 규칙을 A3 에 **적용**하는 것이다.

## 규율

· **표를 만들지 않는다**([[50]] ADB·C333). 꼬리 형태어 목록은 **형태어 수**에만 비례하고 상품이
  늘어도 안 자란다. 유사어 쌍 표는 오배정을 조용히 박제하므로 만들지 않는다.
· **머리 어구는 축자 일치**여야 한다 — `Green` 과 `Green Fee-Free` 는 다른 상품이고 여기서도 다르다.
  `Light Green` ↔ `Light Blue` 를 붙였던 C316 재발이 구조적으로 불가능하다(머리가 다르다).
· 형태어는 **checking 계열 꼬리말만**이다. `Rewards Card` 류는 건드리지 않는다 —
  `Green Rewards Card` 를 `Green` 으로 접으면 **다른 상품과 합쳐진다**.
· **가족을 넘는 병합은 거부**한다. 통일 결과 두 주어가 같은 이름이 되는데 **출처 문서 가족이 다르면**
  적용하지 않고 인쇄한다(같은 이름·다른 상품일 수 있고, 그 판단은 여기서 할 수 없다).
· **축 값 충돌은 거부**한다. 합쳐진 주어의 같은 축에 다른 값이 있으면 적용하지 않고 인쇄한다([[25]]).
· 인용(`source.quote`)은 **원문 그대로 둔다** — 근거는 문서의 말이지 우리 표기가 아니다([[22]]).

usage:  py -3 x152_canonicalize_a3_subjects.py            # 건식(무엇이 바뀌는지만)
        py -3 x152_canonicalize_a3_subjects.py --apply    # specific.json 갱신
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

A2 = os.path.join(HERE, "a2", "banking_knowledge.specific.json")

# 꼬리 형태어 — **checking 계열만**. 코퍼스 제목 실측에서 온 것이고(위 ⒝) 상품 수에 안 자란다.
# 긴 것부터 본다(`Business Checking` 이 `Checking` 보다 먼저 걸려야 한다).
FORMS = ("International Checking", "Business Checking", "Checking Account", "Checking",
         "Account", "Balance")


def head(subject):
    """머리 어구 = 꼬리 형태어를 뗀 것. 뗄 게 없으면 그대로. 빈 이름은 만들지 않는다."""
    t = str(subject or "").strip()
    for f in FORMS:
        if t.endswith(" " + f):
            cand = t[: -(len(f) + 1)].strip()
            return cand if cand else t
    return t


def family(row):
    return re.sub(r"_\d+$", "", str((row.get("source") or {}).get("doc") or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--path", default=A2)
    a = ap.parse_args()

    doc = json.load(io.open(a.path, encoding="utf-8"))
    rows = doc["policy_ontology"]["rows"]

    fam_of = {}
    for r in rows:
        fam_of.setdefault(str(r.get("subject")), set()).add(family(r))

    proposed = {s: head(s) for s in fam_of}
    renames = {s: h for s, h in proposed.items() if h != s}

    # ── 관문 1: 가족을 넘는 병합인가 ─────────────────────────────────────────
    by_new = collections.defaultdict(set)
    for s, h in proposed.items():
        by_new[h].add(s)
    blocked_fam = set()
    for h, olds in sorted(by_new.items()):
        if len(olds) < 2:
            continue
        fams = set()
        for s in olds:
            fams |= fam_of[s]
        if len(fams) > 1:
            print("⛔ 가족을 넘는 병합 — 거부: %r ← %s" % (h, sorted(olds)))
            for s in sorted(olds):
                print("     %-40s %s" % (s, sorted(fam_of[s])))
            blocked_fam |= olds

    # ── 관문 2: 병합 후 같은 축에 다른 값이 생기나 ───────────────────────────
    vals = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        s = str(r.get("subject"))
        if s in blocked_fam:
            continue
        vals[proposed[s]][r.get("axis")].add(r.get("value"))
    blocked_val = set()
    for h, ax in sorted(vals.items()):
        for axis, vs in sorted(ax.items()):
            if len(vs) > 1:
                print("⛔ 병합 시 축 값 충돌 — 거부: %r / %s → %s" % (h, axis, sorted(vs)))
                blocked_val |= {s for s in by_new[h]}

    skip = blocked_fam | blocked_val
    applied = {s: h for s, h in renames.items() if s not in skip}

    print()
    print("=== 이름 통일 (적용 %d건 / 거부 %d건) ===" % (len(applied), len(skip)))
    for s in sorted(applied):
        n = sum(1 for r in rows if str(r.get("subject")) == s)
        print("  %-40s → %-28s (행 %d)" % (s, applied[s], n))
    merged = {h: sorted(v) for h, v in by_new.items() if len(v) > 1 and not (v & skip)}
    if merged:
        print()
        print("=== 합쳐지는 표기 ===")
        for h, olds in sorted(merged.items()):
            print("  %-28s ← %s" % (h, olds))

    print()
    print("주어 %d → %d" % (len(fam_of), len({proposed.get(s, s) if s not in skip else s
                                              for s in fam_of})))

    if not a.apply:
        print("\n(건식 — 파일은 그대로. 적용하려면 --apply)")
        return 0

    for r in rows:
        s = str(r.get("subject"))
        if s in applied:
            r["subject"] = applied[s]
    with io.open(a.path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("\n적용 완료: %s" % a.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
