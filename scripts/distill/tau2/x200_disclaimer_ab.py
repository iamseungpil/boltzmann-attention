# -*- coding: utf-8 -*-
r"""x200 — task_010: **우리 꼬리말이 답을 막는가** (격리 A/B · 유료 0 · 엔진 변경 0).

## 왜 이 프로브인가 ([[55]] 우리 문구부터)

y010 사이드카 실측: `상태별세기×4` · `창산수×4` 가 **실제로 나갔다**. 그런데 두 sim 다 실패했고
궤적은 이렇게 흘렀다 —

  에이전트: (우리 문장을 거의 축자로) *"COMPLETE 2 · IN_PROGRESS 1 · REJECTED 1(Platinum)"*
  손님    : *"그건 **왜**를 답하지 않는다 — 무엇이 거절을 일으켰나?"*
  에이전트: 상담원 이관.

그런데 우리가 보낸 창산수 문장이 바로 그 *왜* 였다:
*"Platinum Rewards Card (2025-10-25): 2 before it within 9 days. The rolling-window allowance on
record is 2 in 9 days."* 그리고 정책 문서는 축자로 *"REJECTED — the user has too many referral
processes going on"* 이라고 말한다.

⇒ 가설: **우리 꼬리말이 모델에게 그 계산을 쓰지 말라고 지시했다.** 두 문장 다 끝에
*"it does not say why any record carries the status it carries"* 를 달고 있다. 우리는 날조를
막으려 썼는데([[25]]), 모델은 그대로 따라 *"이유는 모른다"* 로 갔다 — [[55]] 의 문구-모순 계열.

## 팔 (n 회 · 손님의 **실제 후속 요구**를 그대로 씀)

  OLD     원장 + 현행 두 문장(부정 꼬리말) → 이유를 대나, 이관하나
  NEW     원장 + 꼬리말만 고친 두 문장    → 계산을 쓰나
  NEWDOC  NEW + **정책 문서 축자**(상태 정의)  → 문서가 닿으면 닫히나
  OLDDOC  OLD + 정책 문서 축자             → 문서만으로 닫히나 (꼬리말 효과의 귀속)
  D_null  두 문장 없이 원장만              → 부정 통제

⚠**우리는 답을 문장에 넣지 않는다.** 어느 팔에서도 *"Platinum 이 거절된 이유는 창이다"* 라고
  말해 주지 않는다 — 계산과 문서를 주고 **연결은 모델이 한다**([[62]]).

채점: ⒜ 이관/모름으로 끝나지 않고 ⒝ 이유를 창/건수로 짚는가. 둘 다 문자열 검사이므로
`--show` 로 원문을 반드시 눈으로 확인할 것([[08]]).

실행: python x200_disclaimer_ab.py [N] [--show]
"""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

# 라이브 원장(궤적 축자·x197 과 같은 출처)
ROWS = [
    {"date": "10/20/2025", "referred_account_type": "Bronze Rewards Card", "referral_status": "COMPLETE"},
    {"date": "10/22/2025", "referred_account_type": "Gold Rewards Card", "referral_status": "COMPLETE"},
    {"date": "11/05/2025", "referred_account_type": "Silver Rewards Card", "referral_status": "IN_PROGRESS"},
    {"date": "10/25/2025", "referred_account_type": "Platinum Rewards Card", "referral_status": "REJECTED"},
]
LEDGER = "Found 4 record(s) in 'referrals':\n" + "\n".join(
    "%d. referral_id: %s\n   referred_account_type: %s\n   referral_status: %s\n   date: %s"
    % (i + 1, "-", r["referred_account_type"], r["referral_status"], r["date"])
    for i, r in enumerate(ROWS))

# 손님의 **실제** 후속 요구 (y010 궤적 축자)
ASK = ("Thanks, but that doesn't really answer the \"why.\" Why is the Platinum Rewards Card "
       "referral marked REJECTED - what specifically caused it to be rejected? If you can tell me "
       "the specific reason, I'll know what to do next.")

# 정책 문서 축자 (doc_credit_cards_credit_cards_(general)_001)
DOC = ("From the knowledge base, document \"Understanding Credit Card Referral Statuses\":\n"
       "- COMPLETE - the referred person has successfully opened a new account and met the "
       "criteria to get the referral bonus\n"
       "- IN_PROGRESS - the referred person has successfully opened a new account and is in "
       "progress to meet the criteria for the referral bonus\n"
       "- REJECTED - the user has too many referral processes going on\n"
       "- REJECTED: Do not retry immediately; review the user's existing referral activity "
       "before advising next steps.")

# ── 후보 꼬리말 (A2 에 넣기 **전에** 여기서 잰다) ───────────────────────────────
#   고치는 것은 꼬리말뿐이다. 앞부분(세기·산수)은 한 글자도 안 바꾼다.
NEW_STATUS_TAIL = (" This is a count of what the records say, not a judgement: the records carry "
                   "the status but not its definition. The knowledge base has a document that "
                   "defines what each status value means - retrieve it before explaining a status.")
NEW_WINDOW_TAIL = (" The rolling-window allowance on record is {max} in {days} days. This is "
                   "arithmetic on the dates, so a record that already had {max} inside its window "
                   "is a fact you may state; what that fact implies for the record's status is in "
                   "the document that defines the statuses - retrieve it and say which applies.")


def sentences(mode):
    """mode: 'live' = 지금 A2 그대로 · 'oldwin' = 창 문장을 **옛 꼬리말로 되돌린 것** ·
    'bothnew' = 상태 문구까지 바꾼 것(기록용 — 회귀가 금지한다).

    ⚠2026-08-10: `test_status_breakdown` 이 *"상태 문구에 검색 지시를 넣지 말 것"* 을 못박고
      있다(v010 실측: 그 지시대로 **상태 낱말로 검색**해 이유를 못 찾았다). 그래서 실제 적용은
      **창 산수 문장 하나**뿐이고, 이 프로브는 그 좁힌 변경만으로 효과가 나는지 가른다.
    """
    a2 = load_domain_a2("banking_knowledge")
    sp = a2["ledger_metrics"][0]
    st = LG.status_breakdown(ROWS, sp)
    wh = LG.window_history(ROWS, sp)
    if mode == "live":
        return (st + wh).strip()
    if mode == "oldwin":
        wh0 = (wh.split(" This is arithmetic on the dates")[0]
               + " This says how many records already fell inside the window when each of these "
                 "was made - it does not say why any record carries the status it carries.")
        return (st + "\n" + wh0).strip()
    st2 = st.split(" This is a count of what the records say")[0] + NEW_STATUS_TAIL
    return (st2 + "\n" + wh).strip()


def ask(prompt, n_tokens=220, temp=0.0):
    body = {"model": MODEL, "temperature": temp, "max_tokens": n_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


ESCAPE = ("transfer", "human agent", "cannot determine", "can't determine", "unable to determine",
          "not able to determine", "do not have", "don't have", "no specific reason",
          "not specified", "unclear")
CAUSE = ("too many", "rolling", "within 9 days", "9-day", "9 day", "allowance", "limit of 2",
         "two other referrals", "2 referrals", "window")


def main():
    n = 8
    show = "--show" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
    arms = [
        ("OLD", sentences("oldwin"), False),          # 되돌린 옛 꼬리말
        ("WINONLY", sentences("live"), False),        # ★실제 적용된 변경 — 창 문장 하나
        ("BOTHNEW", sentences("bothnew"), False),     # 상태 문구까지 (회귀가 금지·참고용)
        ("OLDDOC", sentences("oldwin"), True),
        ("WINDOC", sentences("live"), True),
        ("D_null", "", False),
    ]
    print("=" * 100)
    print("task_010 꼬리말 A/B  (n=%d · %s)" % (n, MODEL))
    print("=" * 100)
    print("\n[OLD = 되돌린 옛 꼬리말]\n%s\n\n[WINONLY = 적용본]\n%s\n"
          % (sentences("oldwin"), sentences("live")))
    out = {}
    for label, block, with_doc in arms:
        c = collections.Counter()
        texts = []
        for i in range(n):
            p = LEDGER + ("\n\n" + block if block else "")
            if with_doc:
                p += "\n\n" + DOC
            p += ("\n\nThe customer asks:\n%s\n\nAnswer the customer in two or three sentences."
                  % ASK)
            try:
                t = ask(p, temp=0.0 if i == 0 else 0.7)
            except Exception as e:
                t = "ERR %s" % type(e).__name__
            texts.append(t)
            lo = t.lower()
            gave = any(k in lo for k in CAUSE)
            fled = any(k in lo for k in ESCAPE)
            c["이유O" if gave else "이유X"] += 1
            c["이관O" if fled else "이관X"] += 1
        out[label] = dict(c)
        print("  %-8s 이유 %d/%d · 이관 %d/%d" % (label, c["이유O"], n, c["이관O"], n))
        if show:
            for t in texts[:2]:
                print("      | " + t[:300])
    json.dump(out, open(os.environ.get("T2_X200_OUT", "x200_out.json"), "w"), indent=1)
    print("\n※ 가설이 맞다면 OLD 는 이관이 많고 NEW 는 이유가 는다. NEW 가 OLD 와 같으면"
          "\n  꼬리말은 원인이 아니고(가설 사망) 다른 자리를 봐야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
