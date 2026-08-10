# -*- coding: utf-8 -*-
r"""x235 — **task_070 은 어느 칸에서 무너지는가** (LLM 단독 사다리 · 유료 0 · 엔진 0).

## 왜 (사용자 지시 2026-08-10)

> *"일단 LLM 만으로 어디서 문제가 생기는지부터 확인하라."*

설계서(`TASK_070_071_DESIGN_2026_08_09`)의 §4 단계 1 이 요구하는 관문이다. 표를 지어 주기 전에
**모델 단독으로 어디까지 가는지**를 본다. A3 빌드(M1)는 이 사다리에서 드러난 결손 칸에만
정당화된다(⛔0 ②·③).

## 사다리 — 아래로 갈수록 우리가 더 많이 해 준다

  R0_ASK      손님 요구 + **후보 이름 8개**만                 ← 부정 통제(재료 없이 맞히면 무효)
  R1_DOCS     + 문서 **원문 문장**(제품×축, 축자)             ← 값 추출·필터·선택 전부 LLM
  R2_TABLE    + 우리가 정리한 **값 표**(8행 4축)              ← 추출을 빼 준다
  R3_PROMO    + 활성 프로모션 문서 **축자**                   ← 우선순위 재료를 준다
  R4_EXPIRED  R3 + **만료된 프로모션까지** 함께               ← 만료를 스스로 빼는가(M2 필요성)
  R5_FILTER   요구를 충족하는 행만 남긴 표                    ← 천장(엔진이 걸러 준 상태)

무너지는 첫 칸이 결손이다. R1 에서 되면 M1(A3 빌드)은 **불필요**하고, R2 에서만 되면 결손은
**추출**이며, R3 가 필요하면 결손은 **유효창**이다.

⚠값과 문장은 전부 **문서에서** 온다(x234 추출·인용 실재 확인). gold 는 채점에만([[23]]).
⚠요구사항은 태스크 정의 **축자**를 쓴다 — 내가 고쳐 쓰지 않는다.

실행: python x235_070_staircase.py [N]
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                            # noqa: E402

DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
GOLD = "Sky Blue"
ASK = ("Which ONE business checking account should I open? "
       "Answer with the account name only.")
AX = {
    "ATM rebate per month": re.compile(r"(?:rebat|cap on ATM fee rebates)", re.I),
    "overdraft fee": re.compile(r"overdraft", re.I),
    "minimum balance": re.compile(r"minimum balance", re.I),
    "APY": re.compile(r"\bAPY\b", re.I),
}
NUM = re.compile(r"\d")


def docs(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(DOM, "documents", pattern))):
        try:
            out.append(json.load(open(p, encoding="utf-8")))
        except Exception:
            pass
    return out


def axis_sentences():
    """제품별 축 문장을 **원문 그대로** 모은다 (값을 우리가 파싱하지 않는다)."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in docs("doc_business_checking_accounts_*.json"):
        m = re.match(r"doc_business_checking_accounts_(.+?)_\d+$", d.get("id", ""))
        if not m:
            continue
        prod = m.group(1)
        for line in str(d.get("content") or "").split("\n"):
            s = " ".join(line.split()).lstrip("-| ").strip()
            if not s or not NUM.search(s):
                continue
            for ax, rx in AX.items():
                if rx.search(s) and s not in [x for x, _ in per[prod][ax]]:
                    per[prod][ax].append((s[:150], d["id"]))
    return per


def promos():
    """날짜 구간을 가진 프로모션 문서 (활성/만료를 우리가 판정하지 않고 **원문**을 준다)."""
    act, exp = [], []
    for d in docs("doc_bank_accounts_bank_accounts_(general)_*.json"):
        c = str(d.get("content") or "")
        if "11/01" in c and "11/30" in c:
            act.append((d["id"], " ".join(c.split())[:700]))
        elif "10/12" in c and "11/12" in c:
            exp.append((d["id"], " ".join(c.split())[:700]))
    return act, exp


def task_requirements():
    """손님 요구를 **태스크 정의 축자**로."""
    p = os.path.join(DOM, "tasks", "task_070.json")
    o = json.load(open(p, encoding="utf-8"))
    o = o[0] if isinstance(o, list) and o else o
    return " ".join(((o.get("user_scenario") or {}).get("instructions") or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    per = axis_sentences()
    prods = sorted(per)
    act, exp = promos()
    req = task_requirements()
    print("제품 %d · 활성 프로모션 문서 %d · 만료 %d · 요구문 %d자"
          % (len(prods), len(act), len(exp), len(req)))

    names = "Candidate business checking accounts: " + ", ".join(
        p.replace("_", " ").title() for p in prods)
    doc_block = []
    for p in prods:
        lines = []
        for ax in AX:
            for s, i in per[p][ax][:2]:
                lines.append("  - %s  [%s]" % (s, i))
        doc_block.append("%s:\n%s" % (p.replace("_", " ").title(), "\n".join(lines)))
    doc_block = "Policy document sentences on record:\n" + "\n".join(doc_block)

    # R2 표 — 값은 x234 가 문서에서 뽑은 것(같은 문장에서 온다)
    tbl = {"beige": ("50", "0", "250,000", "-"), "cobalt_blue": ("15", "0", "-", "0.5"),
           "hunter_green": ("20", "0", "-", "1.0"), "lime_green": ("25", "0", "-", "1.5"),
           "navy_blue": ("10", "0", "-", "0.5"), "sky_blue": ("15", "0", "-", "1.25"),
           "true_blue": ("30", "25", "25,000", "2.0"), "world_blue": ("100", "30", "-", "-")}
    table = "Values on record (from the sentences above):\n" + "\n".join(
        "  %-14s ATM rebate/mo $%s · overdraft $%s · minimum balance %s · APY %s%%"
        % (p.replace("_", " ").title(), t[0], t[1],
           ("$" + t[2]) if t[2] != "-" else "not stated", t[3])
        for p, t in sorted(tbl.items()))
    keep = ("Rows that meet every stated requirement:\n"
            + "\n".join("  %s" % p.replace("_", " ").title()
                        for p in ("hunter_green", "lime_green", "sky_blue")))
    pr_act = "\n".join("[%s] %s" % (i, c) for i, c in act)
    pr_exp = "\n".join("[%s] %s" % (i, c) for i, c in exp)

    arms = [("R0_ASK", "\n\n".join([req, names])),
            ("R1_DOCS", "\n\n".join([req, names, doc_block])),
            ("R2_TABLE", "\n\n".join([req, names, table])),
            ("R3_PROMO", "\n\n".join([req, names, table, pr_act])),
            ("R4_EXPIRED", "\n\n".join([req, names, table, pr_act, pr_exp])),
            # ★핵심 대조 (사용자 지시 2026-08-10): 추천 계열에서 필요했던 것은 **필터 하나**였다.
            #   070 이 그와 같은지 가른다 — `R5` 는 **필터만**(프로모션 없음), `R6` 은 필터+유효창.
            #   R5 가 높으면 **새 레버는 필요 없다**. R5 가 낮고 R6 만 높을 때에만 유효창이
            #   새 기전으로 정당화된다. ⚠태스크마다 A3 를 채우는 것은 떠먹이기다([[50]] ADB·
            #   사용자 지시) — 공통 기전으로 닫히는지를 먼저 본다.
            ("R5_FILTER_ONLY", "\n\n".join([req, keep, table])),
            ("R6_FILTER_PROMO", "\n\n".join([req, keep, table, pr_act])),
            # ★A3 빌드(M1)가 필요한가를 가르는 칸 — **문서 원문 + 활성 프로모션**만 준다.
            #   높으면 값 정리는 불필요하고 남는 기전은 **유효창 하나**다(떠먹이기 회피).
            ("R7_DOCS_PROMO", "\n\n".join([req, names, doc_block, pr_act]))]
    for name, body in arms:
        print("   %-11s %6d자" % (name, len(body)))
    for name, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                t = chat(body + "\n\n" + ASK, None, 0.0 if i == 0 else 0.7, 24).get("content", "")
            except Exception as e:
                t = "ERR %s" % type(e).__name__
            c[" ".join(str(t).split())[:40]] += 1
        hit = sum(v for k, v in c.items()
                  if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(GOLD),
                                  str(k).strip(), re.I))
        print("  %-11s gold %d/%d   %s" % (name, hit, n, c.most_common(3)))
    print("\n※ 읽는 법 — R0 는 낮아야 한다(부정 통제). 처음으로 높아지는 칸이 **결손의 자리**다."
          "\n  R1 에서 되면 A3 빌드는 불필요하고, R2 에서만 되면 결손은 **추출**,"
          "\n  R3 가 필요하면 **유효창**, R4 가 무너지면 **만료 구분**이 결손이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
