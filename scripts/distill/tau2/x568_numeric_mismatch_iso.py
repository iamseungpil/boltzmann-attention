# -*- coding: utf-8 -*-
r"""x568 — **불일치만** 말하면 금액이 옳게 가는가 (유료 0 · 016 의 마지막 한 칸).

## 자리 (t7365 · `task_016#s1567` msg[43])

    찍은 것: submit_transaction{user_id: friend_user_5839, credit_card_type: 'Silver Rewards Card',
                                merchant_name: 'Best Buy', category: 'Shopping', **amount: 150**}
    gold   : 같은 네 인자 · **amount '750'**

다섯 중 넷이 정확하다. `150` 은 날조가 아니라 **다른 제품의 문서**에서 왔다 —
msg[3] 축자 *"Earn a referral bonus of $150 for each referred **logistics business**…"*.
이 카드의 요건 문서는 msg[37] 에 이미 왔다(*"spend at least $750 within 60 days"*).

## 무엇을 말하나 — **불일치 사실뿐. 값은 말하지 않는다**

    A_asis   그대로                                        ← 재현 게이트(150 이 나와야 한다)
    B_mism   *"그 수는 그 주체의 기록이 주는 `<축>` 값이 아니다"*   ← **숫자 0개**
    N_len    길이만 맞춘 무관 문장([[57]])

축 이름(`qualifying_spend`)은 A3 선언에서 오고, 주체는 **모델이 자기 인자에 넣은 값**이다.
⛔엔진이 750 을 말하면 그것이 gold 프로그램 재작성이다([[62]]) — 이 프로브는 그 선을 지킨다.

## 채점 — 닫힌 술어 · **프롬프트에 안 들어간다**

A3 의 그 (주체·축) 행이 주는 수 집합을 채점에만 쓴다. 답이 그 수를 쓰면 통과.
⚠주체 정합은 라이브에선 LLM 몫이다(`formalize_subject_align`·C376). 이 프로브는 채점을 위해
  토큰 겹침으로 A3 행을 고른다 — **계기의 편의이지 엔진 규칙이 아니다**.

사용: PYTHONPATH=. py -3 x568_numeric_mismatch_iso.py --port 8140 [--wiring-only]
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

import gate_interpreter as GI                                       # noqa: E402
import t2_dominance as DOM                                          # noqa: E402
import t2_forensic as F                                             # noqa: E402
import t2_gate_patch as G                                           # noqa: E402
import x559_016_row_pick_iso as X559                                # noqa: E402
import x564_arg_producer_census as X564                             # noqa: E402
import x567_numeric_arg_census as X567                              # noqa: E402

NL = chr(10)
ASK = (NL + NL + "What is the very next tool call you make? Reply with one line only, "
       "in the form `tool_name {\"arg\": \"value\"}`. Nothing else.")


def a3_for(rows, subject_value, arg):
    """그 주체의 A3 행 — 채점용. 토큰 겹침으로 고른다(계기 편의·본문 주석 참조)."""
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", str(subject_value)) if len(t) > 3]
    best = []
    for r in rows:
        s = str(r.get("subject") or "").lower()
        if toks and all(t in s for t in toks):
            best.append(r)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tag", default="bank_t7365_hard0_20260827")
    ap.add_argument("--sim", default="task_016#s1567")
    ap.add_argument("--arg", default="amount")
    ap.add_argument("--subject-arg", default="credit_card_type")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    a2 = GI.load_domain_a2("banking_knowledge") or {}
    facts = G._policy_facts(a2)
    sims = [s for s in F.scored(a.tag) if F.simtag(s) == a.sim]
    if not sims:
        print("그 sim 이 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2
    ms = sims[0].get("messages") or []

    spot = None
    for i, m in enumerate(ms):
        for tc in (m.get("tool_calls") or ()):
            ar = DOM._args_dict(X564._TC(tc)) or {}
            if a.arg in ar and a.subject_arg in ar:
                spot = (i, str(ar[a.arg]), str(ar[a.subject_arg]),
                        F.inner_name(F.argsof(tc)) or F.nameof(tc))
                break
        if spot:
            break
    if not spot:
        print("그 인자를 든 호출이 없다", file=sys.stderr)
        return 2
    i, bad, subj, tool = spot
    rows = a3_for(facts, subj, a.arg)
    axes = sorted({str(r.get("axis")) for r in rows})
    # ★채점은 **이 대화에 실제로 도착한 요건 문서의 수**로 한다 — gold 를 열지 않는다([[23]]).
    #   그 문서는 이 sim 의 msg[37] 이고 축자는 *"must be approved and spend at least $750 within
    #   60 days of account opening"* 이다. 고정 포맷 한 규칙(`spend at least $N`)으로만 읽는다.
    #   ⚠좁히지 않으면 공허하다 — 이 대화에는 `spend at least $N` 문서가 **다섯 개** 와 있고
    #     제품마다 수가 다르다(3,000 · 500 · 750 · 2,250 · 1,500). 그것이 이 자리의 난이도다.
    #     그래서 **모델이 자기 인자에 넣은 주체**를 언급한 문서로만 좁힌다.
    #   ⚠**문서 블록 단위로** 자른다. KB 응답 하나에 문서가 여럿 들어 있어 메시지 단위로 보면
    #     다섯 제품의 수가 전부 한 봉투에 담긴다. 경계는 env 가 찍는 `ID: doc_…` 이다.
    stoks = [t for t in re.findall(r"[A-Za-z]+", subj) if len(t) > 3]
    want, other = [], []
    for m2 in ms[:i]:
        if m2.get("role") != "tool":
            continue
        c2 = " ".join(str(m2.get("content") or "").split())
        for blk in re.split(r"(?=ID: doc_)", c2):
            low = blk.lower()
            # 제품 이름은 본문 제목보다 **문서 id** 에 확실히 있다
            #   (`doc_credit_cards_silver_rewards_card_011`) — 대소문자를 접는다.
            mine = stoks and all(t.lower() in low for t in stoks)
            for n2 in re.findall(r"spend at least \$?(\d[\d,]*(?:\.\d+)?)", blk):
                tgt = want if mine else other
                if n2 not in tgt:
                    tgt.append(n2)
    print("# x568 — msg[%d] %s · 주체 `%s`" % (i, tool, subj))
    print("   모델이 넣은 값: %s" % bad)
    print("   A3 행 %d · 축 %s" % (len(rows), axes[:6]))
    print("   채점용 기대 수 — 주체를 언급한 문서: %s · 다른 제품 문서: %s" % (want, other))
    if not want:
        print("이 대화에 그 요건 문서가 안 왔다 — 채점할 것이 없다([[25]])", file=sys.stderr)
        return 2

    # ★축은 **엔진이 고를 수 없는 판단**이다([[22]]) — 라이브에서 그것을 대는 것은 모델 몫이고
    #   여기서는 문면에서 아예 뺀다. 우리가 말하는 것은 불일치 사실 하나뿐이다.
    say = (NL + NL + "Error: [SOURCE] the number you are about to pass as '%s' is not a value the "
           "records on file give for this subject. The document that states this subject's "
           "requirement is above in this conversation - take the number from there and re-issue "
           "the call." % (a.arg,))
    if a.wiring_only:
        print("--- B_mism 문면 ---")
        print("   " + " ".join(say.split()))
        leak = [w for w in want if w in say]
        print("   ⚠문면에 기대값 누출: %s" % (leak or "없음"))
        print("--- 문맥 %d자 ---" % len(X559.render(ms, i)))
        return 0

    base = X559.render(ms, i)
    adds = {"A_asis": "", "B_mism": say,
            "N_len": NL + NL + ("[note] " + "the records gathered so far in this conversation "
                                "remain current and complete. " * 5)[:len(say)]}
    print()
    print("%-8s %-5s %-58s %s" % ("팔", "temp", "다음 호출", "판정"))
    print("-" * 100)
    tally = collections.defaultdict(lambda: [0, 0])
    for nm in ("A_asis", "B_mism", "N_len"):
        body = base + adds[nm] + ASK
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, body, 110, tp)).split())
                except Exception as e:
                    print("%-8s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                got = re.findall(r"[\"']?%s[\"']?\s*:\s*[\"']?(\d[\d,.]*)" % re.escape(a.arg), rep)
                ok = any(X567.digits(g) in {X567.digits(w) for w in want} for g in got)
                tally[nm][1] += 1
                if ok:
                    tally[nm][0] += 1
                print("%-8s %-5s %-58s %s"
                      % (nm, tp, rep[:58],
                         ("기록의 수 %s" % got[0]) if ok else
                         ("옛 값 그대로" if any(X567.digits(g) == X567.digits(bad) for g in got)
                          else ("다른 수 %s" % got[0] if got else "-"))))
    print()
    print("## 판정 (기록이 주는 수를 쓴 비율)")
    for nm in ("A_asis", "B_mism", "N_len"):
        print("   %-8s %d/%d" % (nm, tally[nm][0], tally[nm][1]))
    print()
    print("⚠A_asis 가 이미 옳으면 결손이 아니다([[62]] 2b). N_len 이 같으면 길이다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
