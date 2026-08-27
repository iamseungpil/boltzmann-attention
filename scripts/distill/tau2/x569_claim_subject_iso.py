# -*- coding: utf-8 -*-
r"""x569 — 주장의 수치를 **그 주체의 문서 블록**과 맞대면 갈리는가 (유료 0).

## 결함 (t7365 `task_016#s1567` · 2026-08-27)

016 의 마지막 한 칸은 도구 인자가 아니라 **에이전트 문장 속 숫자**다. 손님이 찍는 도구이고,
손님은 에이전트가 말한 수를 쓴다 — msg[45] 축자: *"thanks for **confirming the $150 remaining**"*.

에이전트가 그 대화에서 댄 수:
    turn 38  *"the conditions … for the Silver Rewards Card … you will receive a **$75** bonus"*
    turn 46  *"has indeed met the **$500** spending threshold with the recent $150 purchase"*
이 카드의 요건은 **$750** 이고 그 문서는 이미 와 있다.

우리 검사(`t2_source.unsourced_claims`)는 그것을 통과시킨다 — 그 독스트링이 이유를 적어 뒀다:
    *"숫자만 대조하면 같은 숫자가 **다른 항목**으로 코퍼스에 있어 거짓 통과할 수 있다 …
      한 줄 단위 근접성이지 의미 대조가 아니다."*
$75·$500·$150 은 전부 **다른 제품 문서**에 실재한다(이 대화에 `spend at least $N` 문서가 다섯 개
와 있고 제품마다 수가 다르다).

## 팔 — 값은 말하지 않는다

    A_asis    그대로                                  ← 재현 게이트(다른 제품의 수가 나와야 한다)
    B_subj    *"그 수는 당신이 말한 그 제품의 문서에 없다"*   ← **숫자 0개**
    N_len     길이만 맞춘 무관 문장([[57]])

## 채점 — 닫힌 술어 · gold 무참조([[23]])

회수된 KB 응답을 **문서 블록(`ID: doc_…`)** 으로 자르고, 주체를 언급한 블록의 수 집합과
다른 블록의 수 집합을 만든다. 답이 어느 쪽 수를 쓰는지 인쇄한다. 프롬프트에는 안 들어간다.

사용: PYTHONPATH=. py -3 x569_claim_subject_iso.py --port 8140 [--wiring-only]
"""
import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                             # noqa: E402
import x559_016_row_pick_iso as X559                                # noqa: E402
import x567_numeric_arg_census as X567                             # noqa: E402

NL = chr(10)
SUBJ = "Silver Rewards Card"
ASK = (NL + NL + "Write your next reply to the customer. Say what the referred person must do "
       "for this card's referral bonus, with the figure the records give. Two sentences at most.")


def blocks(ms, upto, subj):
    """회수된 문서를 블록으로 자르고 주체 언급 여부로 가른다 — `ID: doc_…` 경계·대소문자 접기."""
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", subj) if len(t) > 3]
    mine, other = set(), set()
    for m in ms[:upto]:
        if m.get("role") != "tool":
            continue
        c = " ".join(str(m.get("content") or "").split())
        for blk in re.split(r"(?=ID: doc_)", c):
            low = blk.lower()
            tgt = mine if (toks and all(t in low for t in toks)) else other
            for n in re.findall(r"\$(\d[\d,]*(?:\.\d\d)?)", blk):
                # ⚠`rstrip(".0")` 은 **문자를** 깎는다 — 750 을 75 로, 1500 을 15 로 만든다.
                #   정규화는 정본(`x567.digits`)을 쓴다.
                tgt.add(X567.digits(n))
    return mine, other - mine


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tag", default="bank_t7365_hard0_20260827")
    ap.add_argument("--sim", default="task_016#s1567")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    sims = [s for s in F.scored(a.tag) if F.simtag(s) == a.sim]
    if not sims:
        print("그 sim 이 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2
    ms = sims[0].get("messages") or []
    # 결정점 = 에이전트가 이 카드의 요건을 **수와 함께** 말하는 첫 자리
    w = next((i for i, m in enumerate(ms)
              if m.get("role") == "assistant"
              and SUBJ.lower() in str(m.get("content") or "").lower()
              and re.search(r"\$\d", str(m.get("content") or ""))), None)
    if w is None:
        print("그런 발화가 없다", file=sys.stderr)
        return 2
    mine, other = blocks(ms, w, SUBJ)
    live = " ".join(str(ms[w].get("content") or "").split())
    print("# x569 — 결정점 msg[%d]" % w)
    print("   라이브가 그 자리에서 한 말: %s" % live[:190])
    print("   주체 블록의 수: %s" % sorted(mine)[:12])
    print("   다른 제품 블록의 수: %s" % sorted(other)[:12])
    say = (NL + NL + "Error: [SOURCE] the figure in that sentence is not one the retrieved "
           "document for that card gives. Figures for a card come only from the document that "
           "names that card - other cards' documents are on the page too. Re-read the one for "
           "this card and restate it.")
    if a.wiring_only:
        print("--- B_subj 문면 ---")
        print("   " + " ".join(say.split()))
        leak = [n for n in mine if n in say]
        print("   ⚠누출: %s" % (leak or "없음"))
        return 0

    base = X559.render(ms, w)
    adds = {"A_asis": "", "B_subj": say,
            "N_len": NL + NL + ("[note] " + "the documents retrieved so far remain current and "
                                "complete for this customer. " * 5)[:len(say)]}
    print()
    print("%-8s %-5s %-56s %s" % ("팔", "temp", "답에 나온 수", "판정"))
    print("-" * 100)
    tally = collections.defaultdict(lambda: [0, 0])
    for nm in ("A_asis", "B_subj", "N_len"):
        body = base + adds[nm] + ASK
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, body, 150, tp)).split())
                except Exception as e:
                    print("%-8s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                got = [X567.digits(g) for g in re.findall(r"\$(\d[\d,]*(?:\.\d\d)?)", rep)]
                ok = any(g in mine for g in got)
                bad = [g for g in got if g in other and g not in mine]
                tally[nm][1] += 1
                if ok:
                    tally[nm][0] += 1
                print("%-8s %-5s %-56s %s"
                      % (nm, tp, (", ".join("$" + g for g in got) or "수 없음")[:56],
                         ("주체 문서의 수" if ok else
                          ("다른 제품의 수 $%s" % bad[0] if bad else "-"))))
    print()
    print("## 판정 (주체 문서의 수를 쓴 비율)")
    for nm in ("A_asis", "B_subj", "N_len"):
        print("   %-8s %d/%d" % (nm, tally[nm][0], tally[nm][1]))
    print()
    print("⚠A_asis 가 이미 옳으면 결손이 아니다([[62]] 2b). N_len 이 같으면 길이다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
