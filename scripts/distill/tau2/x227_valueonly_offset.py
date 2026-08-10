# -*- coding: utf-8 -*-
r"""x227 — **결정 턴을 값만으로 보내면, 채택은 얼마나 오르고 무엇을 파는가** (유료 0 · 엔진 0).

## 왜 (x226 + 사용자 승인 2026-08-10)

x226: 문맥(40,856~112,729자)을 **한 글자도 안 지우고** 블록 **앞 문구만** 떼면 098·099·100
전부 **8/8**. 앞 문구는 결정과 무관한 절차 지시(`Error: [ACTION] 'submit_referral' is run by
the CUSTOMER…`)다. x225: 그 위에 **뒤 꼬리**(CLAIM-PROVENANCE 등)까지 붙은 현행은 **0/8**.

⇒ 규칙 E(메인 채널 = 값만·C392)를 결정 턴에 적용하는 것이 후보 레버다. **그런데 그 지시들은
공짜로 생긴 게 아니다** — 손님이 `submit_referral` 을 직접 실행하게 만들고, 에이전트가 자기가
했다고 말하거나 외부 사이트로 보내는 것을 막으려고 넣은 것이다. §1.3: **부작용 없는 레버는 없다.**
그래서 여기서는 **한 프로브에서 둘 다** 잰다.

## 팔 (문맥 = 결정점까지의 전사 · 문구는 라이브 축자에서 자르기만)

  A_LIVE    앞문구 + 블록 + 뒤꼬리      ← 현행 라이브 재현
  B_VALUE   **블록(값)만**              ← 후보 레버
  C_NOPRE   블록 + 뒤꼬리 (앞문구만 제거)
  D_NOTAIL  앞문구 + 블록 (꼬리만 제거)
  E_NONE    앞문구 + 뒤꼬리, **블록 없음**  ← 부정 통제

## 두 계기

  NAME    *"어느 계좌를 써야 하나 — 이름만"* → **채택**(gold 정확 일치)
  REPLY   *"이제 손님에게 답하라"* → **상쇄**를 센다:
            · self_claim  = 자기가 제출했다/하겠다고 말함      (위반·over-action 대리지표)
            · external    = 웹사이트·앱·지점으로 보냄          (위반)
            · hands_over  = `submit_referral` 을 손님이 이 대화에서 실행하라고 말함 (정상)

⚠REPLY 채점은 **문자열 지표**라 대리다([[M]]). 판정 근거로 팔마다 실제 답변 2건을 인쇄한다.
⚠어떤 문장도 새로 쓰지 않는다 — 라이브 축자에서 자르기만 한다. gold 는 채점에만([[23]]).

실행: python x227_valueonly_offset.py [N]
"""
import collections
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
from x219_adoption import ASK, BLOCK_SIG, GOLD                    # noqa: E402
from x225_runnerup_removal import live_turn, pick_bounded         # noqa: E402

REPLY_ASK = "Now write your reply to the customer."
SELF = re.compile(r"\bI(?:'ve| have| will| am|'m)?\s+(?:already\s+)?(?:go(?:ne|ing) ahead and\s+)?"
                  r"(?:submit|submitted|submitting|process|processed|filed)\b", re.I)
EXTERNAL = re.compile(r"\b(website|web site|portal|mobile app|online banking|branch|"
                      r"sign in to your account|log in to your account)\b", re.I)
HANDS = re.compile(r"submit_referral", re.I)


def grade_reply(t):
    return {"self_claim": bool(SELF.search(t)),
            "external": bool(EXTERNAL.search(t)),
            "hands_over": bool(HANDS.search(t))}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    m = max(4, n // 2 + 2)          # REPLY 는 토큰이 커서 조금 적게
    out = {}
    for task in ("task_098", "task_099", "task_100"):
        gold = GOLD[task]
        blk, tail = live_turn(gold)
        got = pick_bounded(task)
        if not blk or not got:
            print("\n%s — 재료가 없다. 건너뛴다." % task)
            continue
        kb, tag, trial, ctx = got
        i = blk.find(BLOCK_SIG)
        pre, val = blk[:i].rstrip(), blk[i:].strip()
        tl = ("\n" + "\n".join(tail)) if tail else ""
        arms = [("A_LIVE", ctx + "\n\n" + pre + "\n" + val + tl),
                ("B_VALUE", ctx + "\n\n" + val),
                ("C_NOPRE", ctx + "\n\n" + val + tl),
                ("D_NOTAIL", ctx + "\n\n" + pre + "\n" + val),
                ("E_NONE", ctx + "\n\n" + pre + tl)]
        print("\n" + "=" * 96)
        print("%s  %s t%s · KB %d · 문맥 %d자 · 앞문구 %d자 · 값 %d자 · 꼬리 %d자 · gold=%r"
              % (task, tag, trial, kb, len(ctx), len(pre), len(val), len(tl), gold))
        for name, body in arms:
            print("   %-9s 앞문구 %s · 값 %s · 꼬리 %s · %6d자"
                  % (name, "O" if pre and pre[:60] in body else "X",
                     "O" if BLOCK_SIG in body else "X",
                     "O" if tail and tail[0][:60] in body else "X", len(body)))

        for name, body in arms:
            c = collections.Counter()
            for i2 in range(n):
                try:
                    t = chat(body + "\n\n" + ASK, None, 0.0 if i2 == 0 else 0.7, 24).get(
                        "content", "")
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                c[" ".join(str(t).split())[:40]] += 1
            hit = sum(v for k, v in c.items()
                      if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold),
                                      str(k).strip(), re.I))
            out["%s/%s/NAME" % (task, name)] = [hit, n]
            print("  NAME  %-9s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))

        for name, body in arms:
            agg = collections.Counter()
            gold_seen, shown = 0, []
            for i2 in range(m):
                try:
                    t = chat(body + "\n\n" + REPLY_ASK, None, 0.0 if i2 == 0 else 0.7,
                             220).get("content", "") or ""
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                g = grade_reply(t)
                for k, v in g.items():
                    agg[k] += 1 if v else 0
                if re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(gold), t):
                    gold_seen += 1
                if len(shown) < 2:
                    shown.append(" ".join(t.split())[:160])
            out["%s/%s/REPLY" % (task, name)] = {"n": m, "gold": gold_seen, **dict(agg)}
            print("  REPLY %-9s gold언급 %d/%d · self_claim %d · external %d · hands_over %d"
                  % (name, gold_seen, m, agg["self_claim"], agg["external"], agg["hands_over"]))
            for s in shown:
                print("        · %s" % s)

    json.dump(out, open(os.environ.get("T2_X227_OUT", "x227_out.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\n※ 읽는 법 — B_VALUE 가 NAME 을 사면서 REPLY 의 hands_over 를 팔면 그것이 이 레버의"
          "\n  가격이다. self_claim·external 이 B 에서 늘면 절차 지시를 결정 턴 **밖**으로 옮기는"
          "\n  구성(다음 턴에 붙이기)이 필요하다는 뜻이지, 지시를 없애도 된다는 뜻이 아니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
