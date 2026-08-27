# -*- coding: utf-8 -*-
r"""x571 — 기록에 없는 수를 말한 자리에서, **그 사실만** 말하면 고치는가 (유료 0).

## 결손 (t7365 `task_016#s1567` msg[42])

    msg[41] tool       `No records found in 'credit_card_transaction_history'`
    msg[42] assistant  *"Currently, your friend has made transactions totaling **$350**.
                         To reach the $500 threshold, they need to make additional purchases
                         totaling **$150**."*
    msg[43] user       `submit_transaction{… amount: 150}`      ← 손님이 그 수를 그대로 쓴다

`$350` 은 손님의 **기록에 대한 주장**인데 레코드 덤프에는 없다(정책 산문에는 있다).
코퍼스(8 런·기록 관련 문장의 통화 수치 892): dump 39% · **policy 30%** · user 23% · none 8% ·
부호표 9 태스크 23 sim **reward 1.0 = 0**(`x570`).

## 팔 — 값도, 지시도 넣지 않는다

    A_asis   그대로                                       ← 재현 게이트(기록에 없는 수가 나와야 한다)
    B_say    *"그 문장이 기록에 대해 대는 수가 기록에 없다"*      ← **숫자 0 · 무엇을 말하라는 지시 0**
    N_len    길이만 맞춘 무관 문장([[57]])

⛔물음(ASK)은 **중립**이어야 한다. 오늘 `x569` 에서 나는 물음에 *"with the figure the records
  give"* 를 넣었고, 그 지시 자체가 처치라서 A_asis 가 5/5 로 옳게 나왔다 — 결손이 사라졌다.
  여기서는 *"손님에게 보낼 다음 답을 써라"* 만 묻는다.

## 채점 — 닫힌 술어

답에서 **기록에 관한 문장**의 통화 수치를 뽑아, 그 수가 레코드 덤프에 있는지 본다
(정본 `x570.record_claim_sources` 와 같은 술어). 엔진은 어느 수가 옳은지 모른다.

사용: PYTHONPATH=. py -3 x571_record_claim_iso.py --port 8140 [--wiring-only]
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
import x567_numeric_arg_census as X567                              # noqa: E402
import x570_prose_figure_census as X570                             # noqa: E402

NL = chr(10)
ASK = NL + NL + "Write your next reply to the customer."


def dumps_before(ms, upto):
    return [" ".join(str(m.get("content") or "").split())
            for m in ms[:upto]
            if m.get("role") == "tool" and "Record ID:" in str(m.get("content") or "")]


def bad_figs(text, dumps):
    """기록에 관한 문장의 수 중 **덤프에 없는** 것."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", " ".join(str(text or "").split())):
        if not X570.REC_WORD.search(sent):
            continue
        for g in X570.figs(sent):
            if not any(X567.in_text(g, d) for d in dumps):
                out.append(g)
    return out


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
    src = X570.record_claim_sources(ms)
    spot = next((i for i, k, g, _t in src if k != "dump"), None)
    if spot is None:
        print("덤프 밖 기록주장이 없다", file=sys.stderr)
        return 2
    dumps = dumps_before(ms, spot)
    live = " ".join(str(ms[spot].get("content") or "").split())
    print("# x571 — 결정점 msg[%d] · 그 앞 레코드 덤프 %d개" % (spot, len(dumps)))
    print("   라이브가 그 자리에서 한 말: %s" % live[:200])
    print("   그 발화의 덤프 밖 수: %s" % bad_figs(live, dumps))
    say = (NL + NL + "Error: [SOURCE] your reply states a figure for what this customer has "
           "actually done, and the records retrieved in this conversation do not carry that "
           "figure. Only the records say what has happened; the policy pages say what is "
           "required. Say only what the records show.")
    if a.wiring_only:
        print("--- B_say 문면 ---")
        print("   " + " ".join(say.split()))
        print("   ⚠문면의 숫자: %s" % (re.findall(r"\d+", say) or "없음"))
        print("--- 물음 ---%s" % ASK)
        print("--- 문맥 %d자 ---" % len(X559.render(ms, spot)))
        return 0

    base = X559.render(ms, spot)
    adds = {"A_asis": "", "B_say": say,
            "N_len": NL + NL + ("[note] " + "the information gathered so far in this "
                                "conversation remains current. " * 6)[:len(say)]}
    print()
    print("%-8s %-5s %-52s %s" % ("팔", "temp", "덤프 밖 수", "판정"))
    print("-" * 100)
    tally = collections.defaultdict(lambda: [0, 0])
    for nm in ("A_asis", "B_say", "N_len"):
        body = base + adds[nm] + ASK
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, body, 220, tp)).split())
                except Exception as e:
                    print("%-8s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                bad = bad_figs(rep, dumps)
                tally[nm][1] += 1
                if not bad:
                    tally[nm][0] += 1
                print("%-8s %-5s %-52s %s"
                      % (nm, tp, (", ".join("$" + b for b in bad) or "없음")[:52],
                         "기록 밖 수 없음" if not bad else "-"))
    print()
    print("## 판정 (기록에 없는 수를 **안 말한** 비율)")
    for nm in ("A_asis", "B_say", "N_len"):
        print("   %-8s %d/%d" % (nm, tally[nm][0], tally[nm][1]))
    print()
    print("⚠A_asis 가 이미 깨끗하면 결손이 아니다([[62]] 2b). N_len 이 같으면 길이다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
