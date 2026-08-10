# -*- coding: utf-8 -*-
r"""x228 — **액션 발화를 격리 서브가 지으면, 지시 없이도 소유권이 서는가** (유료 0 · 엔진 0).

## 왜 (x227 + 사용자 설계 2026-08-10)

x227(098): 결정 턴에서 지시를 빼면 **채택 0/8 → 8/8**, 그런데 *"`submit_referral` 은 손님이
이 대화에서 실행한다"* 를 말하는 비율이 **3/6 → 0/6**. 지시는 없애면 안 되고 **자리를 옮겨야**
한다.

> 사용자: *"각 결정점과 action 을 하나로 묶고 거기에 적절한 지시만 최소로."*
> *"그 지시가 메인에 있으면 다시 각 서브에 혼잡을 일으킨다 — 지시는 그 서브 안에서 정리하라."*

⇒ **액션도 기능 서브로 내린다.** 그리고 두 번째 형태도 같이 잰다: 그 지시를 **문장이 아니라
도구 소유자 표기(사실)**로 바꾸면 되는가(규칙 E: 메인 채널 = 값·사실만).

## 팔 (계기는 전부 REPLY — *"이제 손님에게 답하라"*)

  A_MAIN_DIR      메인 전 문맥 + 값 + **지시문**        ← 현행 라이브
  B_MAIN_VAL      메인 전 문맥 + 값                     ← 지시 뺀 것(x227 재현)
  C_MAIN_TOOLTAB  메인 전 문맥 + 값 + **소유자 표기**
  D_SUB_DIR       **손님 발화 + 값**(격리) + 지시문      ← 액션 서브
  E_SUB_VAL       격리 + 값                             ← 부정 통제(지시 없음)
  F_SUB_TOOLTAB   격리 + 값 + 소유자 표기

계기: `hands_over`(submit_referral 을 손님이 실행하라고 넘김) · `self_claim`(자기가 했다) ·
`external`(웹사이트·앱으로 보냄) · `gold`(결정 값을 그대로 말하는가).

⚠소유자 표기는 **새 주장이 아니다** — 라이브 지시문이 이미 말하는 소유자·인자를 표 형태로 옮긴
것뿐이다(문구 신설 0). ⚠REPLY 채점은 문자열 대리지표라 [[M]]·팔마다 실물 2건을 인쇄한다.

실행: python x228_action_agent.py [N]
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
from x219_adoption import BLOCK_SIG, GOLD, render                 # noqa: E402
from x223_subq_isolation import decision_cut, user_turns          # noqa: E402
from x225_runnerup_removal import live_turn, pick_bounded         # noqa: E402
import x219_adoption as X219                                      # noqa: E402

REPLY_ASK = "Now write your reply to the customer."
TOOLTAB = ("Tool ownership on record:\n"
           "  Tools you call:          KB_search_dense · KB_search_bm25 · get_referrals_by_user ·"
           " give_discoverable_user_tool\n"
           "  Tools the CUSTOMER runs in this chat:  submit_referral(user_id, account_type)")
SELF = re.compile(r"\bI(?:'ve| have| will| am|'m)?\s+(?:already\s+)?(?:go(?:ne|ing) ahead and\s+)?"
                  r"(?:submit|submitted|submitting|process|processed|filed)\b", re.I)
EXTERNAL = re.compile(r"\b(website|web site|portal|mobile app|online banking|branch|"
                      r"sign in to your account|log in to your account)\b", re.I)
HANDS = re.compile(r"submit_referral", re.I)


def sims_for(task):
    """격리 팔에 쓸 손님 발화를 얻기 위해 같은 sim 의 원본 메시지도 돌려준다."""
    import glob
    import gzip
    best = None
    for pat in X219.PATS:
        for p in sorted(glob.glob(pat)):
            try:
                f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") \
                    else open(p, encoding="utf-8")
                d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for s in (d.get("simulations") or []):
                if not isinstance(s, dict) or s.get("task_id") != task:
                    continue
                if (s.get("reward_info") or {}).get("reward") == 1:
                    continue
                msgs = s.get("messages") or []
                if len(msgs) < 8:
                    continue
                cut = decision_cut(msgs)
                body = render(msgs[:cut])
                kb = body.count("Score:")
                if kb == 0 or len(body) > 120000:
                    continue
                if best is None or kb > best[0]:
                    best = (kb, body, msgs, cut)
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    out = {}
    for task in ("task_098", "task_099", "task_100"):
        gold = GOLD[task]
        blk, _tail = live_turn(gold)
        got = sims_for(task)
        if not blk or not got:
            print("\n%s — 재료가 없다. 건너뛴다." % task)
            continue
        kb, ctx, msgs, cut = got
        i = blk.find(BLOCK_SIG)
        pre, val = blk[:i].rstrip(), blk[i:].strip()
        us = "\n".join("[user] %s" % u for u in user_turns(msgs, cut))
        arms = [("A_MAIN_DIR", ctx + "\n\n" + pre + "\n" + val),
                ("B_MAIN_VAL", ctx + "\n\n" + val),
                ("C_MAIN_TOOLTAB", ctx + "\n\n" + val + "\n\n" + TOOLTAB),
                ("D_SUB_DIR", us + "\n\n" + pre + "\n" + val),
                ("E_SUB_VAL", us + "\n\n" + val),
                ("F_SUB_TOOLTAB", us + "\n\n" + val + "\n\n" + TOOLTAB)]
        print("\n" + "=" * 96)
        print("%s · KB %d · 메인 %d자 · 손님발화 %d자 · 앞지시 %d자 · 값 %d자 · gold=%r · n=%d"
              % (task, kb, len(ctx), len(us), len(pre), len(val), gold, n))
        for name, body in arms:
            print("   %-14s 지시 %s · 표기 %s · 값 %s · %6d자"
                  % (name, "O" if pre and pre[:60] in body else "X",
                     "O" if TOOLTAB[:30] in body else "X",
                     "O" if BLOCK_SIG in body else "X", len(body)))
        for name, body in arms:
            agg, gold_seen, shown = collections.Counter(), 0, []
            for k in range(n):
                try:
                    t = chat(body + "\n\n" + REPLY_ASK, None, 0.0 if k == 0 else 0.7,
                             220).get("content", "") or ""
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                agg["hands_over"] += 1 if HANDS.search(t) else 0
                agg["self_claim"] += 1 if SELF.search(t) else 0
                agg["external"] += 1 if EXTERNAL.search(t) else 0
                if re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(gold), t):
                    gold_seen += 1
                if len(shown) < 2:
                    shown.append(" ".join(t.split())[:150])
            out["%s/%s" % (task, name)] = {"n": n, "gold": gold_seen, **dict(agg)}
            print("  %-14s gold %d/%d · hands_over %d · self_claim %d · external %d"
                  % (name, gold_seen, n, agg["hands_over"], agg["self_claim"], agg["external"]))
            for s in shown:
                print("        · %s" % s)
    json.dump(out, open(os.environ.get("T2_X228_OUT", "x228_out.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\n※ 읽는 법 — D/F 가 A 를 이기면 *'액션도 서브로'* 가 성립한다."
          "\n  C·F 가 D 만큼 나오면 지시문을 **사실(소유자 표기)로 대체**할 수 있다."
          "\n  E(격리·지시 없음)는 낮아야 한다 — 높으면 격리만으로 되는 것이라 지시가 불필요하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
