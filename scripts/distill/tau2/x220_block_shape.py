# -*- coding: utf-8 -*-
r"""x220 — 결정 블록이 **어떤 모양으로 나가야 채택되나** (격리 A/B · 유료 0 · 엔진 0).

## 왜 (x219 + 사이드카 실물)

x219: 같은 문맥·같은 블록인데 **블록만 끝에 붙이면 0/8 → 8/8**. 그래서 라이브가 실제로 어떻게
붙이는지 열어 봤더니, 블록은 결정 지점(`reminder-user`)에 붙긴 하는데 **혼자가 아니다** —
한 메시지가 이렇게 시작한다:

    Error: [ACTION] 'submit_referral' is run by the CUSTOMER, not by you. There is no
    agent-side procedure to look up for running it, so do not search for one and do not
    transfer for this. …

그리고 블록의 **마지막 줄**은 이렇다:

    The next best ones it ranked, in order: Light Blue (referrer_bonus_usd=30); Green …

⚠098 의 세 실패가 전부 제출한 것이 **`Light Blue`** — 우리가 적어 준 **차순위 1번**이다.

## 팔 (라이브 메시지 실물을 조각내며 가른다 · 문맥·질문 고정)

  G_LIVE       라이브 메시지 **그대로**                      ← 재현되어야 한다
  H_NOERR      − 맨 앞 `Error: [ACTION]…` 지시
  I_NORUNNERS  − `The next best ones it ranked…` 줄
  J_BOTH       − 둘 다
  K_ANSWERONLY `It answers: X` 한 줄만
  L_NONE       블록 없음                                     ← 부정 통제

⚠라이브 문구를 **우리가 새로 쓰지 않는다** — 사이드카에서 꺼낸 실물을 조각낼 뿐이다.

실행: python x220_block_shape.py [N]
"""
import collections
import glob
import gzip
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
from x219_adoption import pick, render, ASK                      # noqa: E402

FBS = "/home/woori/scratch/logs/fb_bank_alllevers_20260810.jsonl"
SIG = "A separate check was run on the policy constants"


def live_msgs():
    """사이드카에서 **결정 블록을 담은 리마인더 실물**을 답별로 모은다."""
    out = {}
    for ln in open(FBS, encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = o.get("text") or ""
        if SIG not in t:
            continue
        m = re.search(r"It answers: ([^.\n]+)", t)
        if m:
            out.setdefault(m.group(1).strip(), t)
    return out


def cut_err(t):
    """맨 앞 `Error: …` 지시 블록을 뺀다 (첫 빈 줄/첫 우리 문장 전까지)."""
    lines = t.split("\n")
    keep = [l for l in lines if not l.startswith("Error:")]
    return "\n".join(keep).strip()


def cut_runners(t):
    return "\n".join(l for l in t.split("\n")
                     if not l.startswith("The next best ones it ranked")).strip()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    msgs_by_answer = live_msgs()
    print("라이브 리마인더 %d종: %s" % (len(msgs_by_answer), sorted(msgs_by_answer)))
    out = {}
    for task, gold in (("task_098", "Blue"), ("task_099", "World Blue")):
        live = msgs_by_answer.get(gold)
        if not live:
            print("\n%s — 그 답의 라이브 리마인더를 못 찾았다." % task)
            continue
        got = pick(task, gold)
        if not got:
            print("\n%s — 혼잡한 실패 sim 을 못 찾았다." % task)
            continue
        kb, tag, trial, msgs = got
        ctx = render(msgs)
        ans = re.search(r"It answers: ([^.\n]+)", live)
        only = "A separate check was run. It answers: %s." % ans.group(1).strip()
        arms = [("G_LIVE", live), ("H_NOERR", cut_err(live)),
                ("I_NORUNNERS", cut_runners(live)),
                ("J_BOTH", cut_runners(cut_err(live))),
                ("K_ANSWERONLY", only), ("L_NONE", "")]
        print("\n" + "=" * 96)
        print("%s  %s trial=%s · KB %d · 문맥 %d자 · gold=%r · n=%d"
              % (task, tag, trial, kb, len(ctx), gold, n))
        for name, blk in arms:
            print("   %-13s %5d자 · Error지시 %d · runners %d"
                  % (name, len(blk), blk.count("Error:"),
                     blk.count("The next best ones it ranked")))
        for name, blk in arms:
            c = collections.Counter()
            for i in range(n):
                p = ctx + (("\n\n" + blk) if blk else "") + "\n\n" + ASK
                try:
                    t = chat(p, None, 0.0 if i == 0 else 0.7, 24).get("content", "")
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                c[" ".join(str(t).split())[:40]] += 1
            hit = sum(v for k, v in c.items()
                      if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold),
                                      str(k).strip(), re.I))
            out["%s/%s" % (task, name)] = [hit, n]
            print("  %-13s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X220_OUT", "x220_out.json"), "w"), indent=1)
    print("\n※ G_LIVE 가 낮고 H 나 I 가 높으면 그 조각이 범인이다."
          "\n  I_NORUNNERS 가 살리면 **우리가 적어 준 차순위**가 오답을 만든 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
