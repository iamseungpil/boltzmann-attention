# -*- coding: utf-8 -*-
"""무엇이 선택을 망치는가 — 프롬프트 절제(ablation) 격리([[18]] 확장·x149 후속).

x149 가 두 진단을 냈다: 099 는 **부하**(깨끗한 표 5/5 → 실제 문맥 0/5), 100 은 **격리해도 실패**
(표에 tenure=90 이 있는데 65일 손님에게 World Blue 를 5/5 로 고른다). 이 스크립트는 그 두 실패의
**원인 문구**를 각각 절제로 찾는다.

  100 계열(표·질문을 바꾼다) — 능력/표현 축
    P0 baseline        정리된 표 + 사실 + 질문
    P1 axis-rename     `referrer_tenure_days` → 뜻이 문장으로 드러나는 이름
    P2 prefiltered     자격 미달 행을 **미리 제거**한 표(순수 argmax 능력만 남긴다)
    P3 two-step        "먼저 자격 되는 것을 나열하고, 그 중 최고를 고르라"(분해)
    P4 fact-rephrase   같은 사실을 다른 문장으로

  099 계열(문맥 블록을 뺀다) — 부하 축
    Q0 baseline        실제 궤적 문맥 전체
    Q1 −history        추천 이력(도구 출력) 제거
    Q2 −accounts       계좌 목록(도구 출력) 제거
    Q3 user-only       손님 발화만 남김(에이전트 자기 발화 제거 = 자기-정박 제거)
    Q4 tail6           마지막 6 메시지만

유료 0(로컬 vllm). 실행: py -3 x150_choice_ablation.py [TAG] [N]
"""
import gzip
import json
import os
import re
import sys
import urllib.request
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
Q = X.QUESTION


def ask(prompt, temp):
    return X.ask(prompt, temp)


def msgs_of(tag, task):
    p = os.path.join(SIMS, tag + ".json.gz")
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    sim = next(s for s in d["simulations"] if s["task_id"] == task)
    cut = len(sim["messages"])
    for i, m in enumerate(sim["messages"]):
        if any(tc.get("name") == "submit_referral" for tc in (m.get("tool_calls") or [])):
            cut = i
            break
    return sim["messages"][:cut]


def render(ms):
    out = []
    for m in ms:
        c = " ".join(str(m.get("content") or "").split())
        tc = ", ".join(str(t.get("name")) for t in (m.get("tool_calls") or []))
        if tc:
            c = (c + " [called: %s]" % tc).strip()
        if c:
            out.append("%s: %s" % (str(m.get("role")).upper(), c[:900]))
    return "\n".join(out[-40:])


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_remeas_20260808f"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    clean = X.a3_table_clean()

    # ── 100 계열 ────────────────────────────────────────────────────────────
    f100 = X.FACTS["task_100"]
    p100 = {}
    p100["P0 baseline"] = clean + "\n\n" + f100 + "\n\n" + Q
    p100["P1 axis-rename"] = (clean.replace(
        "referrer_tenure_days",
        "min_days_the_referring_customer_must_already_have_banked_here")
        + "\n\n" + f100 + "\n\n" + Q)
    keep = []
    for ln in clean.splitlines():
        m = re.search(r"referrer_tenure_days=(\d+)", ln)
        if ln.startswith("  ") and m and int(m.group(1)) > 65:
            continue
        keep.append(ln)
    p100["P2 prefiltered"] = "\n".join(keep) + "\n\n" + f100 + "\n\n" + Q
    p100["P3 two-step"] = (clean + "\n\n" + f100 + "\n\nFirst work out which accounts this "
                           "customer is allowed to refer for at all, then among those pick the "
                           "one with the largest bonus for the customer. " + Q)
    p100["P4 fact-rephrase"] = (clean + "\n\nThe customer has banked with us for 65 days. The "
                                "business being referred is brand new and will deposit about "
                                "$31,000.\n\n" + Q)

    # ── 099 계열 ────────────────────────────────────────────────────────────
    ms = msgs_of(tag, "task_099")
    f099 = X.FACTS["task_099"]

    def _drop(pred):
        return [m for m in ms if not pred(m)]

    def _isout(m, needle):
        return m.get("role") == "tool" and needle in str(m.get("content") or "")

    p099 = {}
    p099["Q0 baseline"] = ("Here is a customer-service conversation so far.\n\n" + render(ms)
                           + "\n\n" + clean + "\n\n" + f099 + "\n\n" + Q)
    p099["Q1 -history"] = ("Here is a customer-service conversation so far.\n\n"
                           + render(_drop(lambda m: _isout(m, "in 'referrals'")))
                           + "\n\n" + clean + "\n\n" + f099 + "\n\n" + Q)
    p099["Q2 -accounts"] = ("Here is a customer-service conversation so far.\n\n"
                            + render(_drop(lambda m: _isout(m, "Accounts for user")))
                            + "\n\n" + clean + "\n\n" + f099 + "\n\n" + Q)
    p099["Q3 user-only"] = ("Here is a customer-service conversation so far.\n\n"
                            + render([m for m in ms if m.get("role") == "user"])
                            + "\n\n" + clean + "\n\n" + f099 + "\n\n" + Q)
    p099["Q4 tail6"] = ("Here is a customer-service conversation so far.\n\n" + render(ms[-6:])
                        + "\n\n" + clean + "\n\n" + f099 + "\n\n" + Q)

    res = collections.OrderedDict()
    for task, arms in (("task_100", p100), ("task_099", p099)):
        for label, prompt in arms.items():
            answers = []
            for i in range(n):
                try:
                    answers.append(ask(prompt, 0.0 if i == 0 else 0.7))
                except Exception as e:
                    answers.append("ERR %r" % (e,))
            res[(task, label)] = answers

    print()
    for (task, label), answers in res.items():
        gold = X.GOLD[task]
        hit = sum(1 for a in answers if gold.lower() in str(a).lower())
        top = collections.Counter(re.sub(r"\s+", " ", a)[:24] for a in answers).most_common(2)
        print("%-9s %-16s gold=%-12s %d/%d   %s" % (task, label, gold, hit, len(answers), top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
