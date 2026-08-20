# -*- coding: utf-8 -*-
r"""x429 — **수치 술어로 다시 잰 카탈로그 상한** (사용자 지시 2026-08-20: *"상한 다시 재라"*)

## 왜
x428 의 20% 는 술어 언어가 **낱말 포함**이라 나온 값이다 — 그 언어는 `$0 overdraft` 와
`$35 overdraft` 를 못 가른다. 그런데 이 태스크들의 제약은 전부 **수치 비교·부정**이다:
*"absolutely no overdraft fees"* · *"2–3 ATM withdrawals … about $80 each"* ·
*"limit as high as $100,000"* · *"net of the $450 annual fee"*. 그러니 20% 는 상한이 아니라
**상한의 하한**이었다. 여기서 언어를 올려 다시 잰다.

## 자 — 사실 표를 만들지 않고 상한을 묻는 법
*"gold 를 다른 후보 전부와 가르는 수치 술어가 **존재하는가**"* 만 물으면 표가 필요 없다.

    후보 c 의 사실 = 배달 본문에서 그 이름이 나온 **모든** 자리(축자·앞뒤 %d자)
    손님이 말한 속성어 w 에 대해  NUM(c, w) = 사실 안에서 w 근처(±%d자)에 있는 수들
    gold 가 other 와 **분리 가능** ⟺ ∃w ∈ 손님속성어 : NUM(gold,w) ≠ NUM(other,w)
    gold 가 **모든** other 와 분리 가능 ⟺ 수치 술어 언어로 유일하게 지목 가능

신탁이므로 w 는 경쟁자마다 가장 유리한 것을 고른다 — **상한**이다.
같이 내는 것: ⒜ 낱말 언어(x428 재현) ⒝ 수치 언어 ⒞ 수치 언어인데 **손님이 말한 수**만 쓰는 판(더 엄격).

★사실·속성어는 전부 배달 본문과 손님 발화에서 **형태로만** 뽑는다. gold 는 상한 정의와 채점에만([[23]]).

## ⛔결과: **이 계기는 무효다 — 수치 인용 금지** (2026-08-20 부정통제)
gold 를 다른 후보로 바꿔 같은 것을 물었더니:

    수치 술어 : gold 10/10 = 100%   ↔  **가짜 27/30 = 90%**
    엄격 술어 : gold  8/10 =  80%   ↔  **가짜 19/30 = 63%**

⇒ *"gold 를 가르는 술어가 존재한다"* 는 **아무 후보에나 성립한다**. 서로 다른 문서 두 개는 언제나
어떤 낱말 근처의 어떤 수에서 다르다. 이 정식화는 상한을 재지 못한다.
그리고 낱말 판본도 슬라이스를 바꾸자 **20%(x428·첫 등장 ±900자) → 50%(여기·모든 등장 ±2600자)** 로
움직였다 — 같은 언어인데 값이 배로 뛴다 ⇒ **두 수치 다 인용 금지**.

★남는 교훈: 상한은 *"가를 수 있나"* 가 아니라 *"손님 제약이 **어느 속성·방향·문턱**으로 읽히나"* 를
  알아야 정해진다. 그 읽기가 곧 formalize 이므로 **기계 대용치로는 못 잰다**. 정직한 길은 C462 전범 —
  **정책 축자로 사실표·제약 스펙을 저작**(gold 미접촉)한 뒤 그 스펙이 gold 를 고르는지 보는 것뿐이다.

사용: py -3 x429_numeric_ceiling.py
"""
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

import x423_choice_isolation as I  # noqa: E402
import x426_free_gates as G  # noqa: E402
import x427_catalog_minimal as CM  # noqa: E402
import x428_catalog_ceiling as C8  # noqa: E402

PAD = 2600          # 후보 사실 슬라이스 길이(등장마다)
NEAR = 90           # 속성어 근처로 볼 문자 수
RE_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")


def facts_all(name, docs, pad=PAD):
    """그 이름이 나온 **모든** 자리의 축자(첫 자리만 보던 x427 의 한계를 푼다)."""
    buf = []
    for d in docs:
        for m in re.finditer(re.escape(name), d):
            buf.append(d[max(0, m.start() - 140):m.end() + pad])
    return " ".join(buf)


def nums_near(text, w, near=NEAR):
    low = text.lower()
    out = set()
    for m in re.finditer(re.escape(w), low):
        seg = text[max(0, m.start() - near):m.end() + near]
        for x in RE_NUM.finditer(seg):
            try:
                out.add(float(x.group(0).replace(",", "")))
            except Exception:
                pass
    return frozenset(out)


def customer_nums(said):
    out = set()
    for m in RE_NUM.finditer(said):
        try:
            out.add(float(m.group(0).replace(",", "")))
        except Exception:
            pass
    return out


def main():
    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] not in CM.ARGS:
            continue
        k = (c["task"], c["trial"], c["arg"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    print("=" * 108)
    print("x429 · 수치 술어 상한 (LLM 0) · 사례 %d · 사실 슬라이스 = 모든 등장 ±%d자" % (len(cs), PAD))
    print("=" * 108)
    rows, tal = [], collections.Counter()
    for c in cs:
        docs = G.delivered(c["sim"], c["msg_i"])
        names = CM.catalog_names(docs, c["gold"])
        facts = {n: facts_all(n, docs) for n in names}
        facts = {n: v for n, v in facts.items() if v}
        if c["gold"] not in facts:
            facts[c["gold"]] = facts_all(c["gold"], docs) or " "
        said = G.customer_said(c["sim"], c["msg_i"])
        cw = sorted(C8.words(said))
        cnum = customer_nums(said)
        others = [n for n in facts if n != c["gold"]]
        gf = facts[c["gold"]]

        # ⒜ 낱말 언어 (x428 재현)
        gw = {t for t in cw if t in gf.lower()}
        surv_w = [n for n, v in facts.items() if all(t in v.lower() for t in gw)] if gw else list(facts)
        ok_w = len(surv_w) == 1 and surv_w[0] == c["gold"]

        # ⒝ 수치 언어 — 경쟁자마다 가르는 속성어가 있나
        sep, unsep = 0, []
        gnum = {w: nums_near(gf, w) for w in cw}
        for o in others:
            of = facts[o]
            found = None
            for w in cw:
                if not gnum[w] and not nums_near(of, w):
                    continue
                if gnum[w] != nums_near(of, w):
                    found = w
                    break
            if found:
                sep += 1
            else:
                unsep.append(o)
        ok_n = (sep == len(others))

        # ⒞ 손님이 **말한 수**만 쓰는 엄격판
        sep2 = 0
        for o in others:
            of = facts[o]
            hit = False
            for w in cw:
                a, b = gnum[w] & cnum, nums_near(of, w) & cnum
                if a != b and (a or b):
                    hit = True
                    break
            if hit:
                sep2 += 1
        ok_s = (sep2 == len(others))

        tal["낱말"] += ok_w
        tal["수치"] += ok_n
        tal["수치·손님수만"] += ok_s
        rows.append({"task": c["task"], "trial": c["trial"], "arg": c["arg"], "gold": c["gold"],
                     "n_cand": len(facts), "word": ok_w, "num": ok_n, "num_strict": ok_s,
                     "unseparated": unsep[:5]})
        print("  %-9s t%s %-14s 후보 %2d · 속성어 %3d | 낱말 %s · 수치 %s(%d/%d) · 엄격 %s | 못가른 예 %s"
              % (c["task"], c["trial"], c["arg"], len(facts), len(cw),
                 "✅" if ok_w else "  ", "✅" if ok_n else "  ", sep, len(others),
                 "✅" if ok_s else "  ", ", ".join(unsep[:2]) or "-"))
    n = len(rows)
    print()
    for k in ("낱말", "수치", "수치·손님수만"):
        print("  ★상한(%s 술어): **%d/%d = %.0f%%**" % (k, tal[k], n, 100.0 * tal[k] / n if n else 0))
    print("  ⚠분리 가능 ≠ 모델이 고른다. 이건 **언어의 표현력 상한**이지 성적이 아니다.")
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x429_numeric_ceiling.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
