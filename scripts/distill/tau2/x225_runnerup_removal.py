# -*- coding: utf-8 -*-
r"""x225 — **차순위 목록을 안 적으면 되는가** (격리 A/B · 유료 0 · 엔진 0).

## 왜 (x224 사실 + 사용자 지시 2026-08-10)

x224 가 잰 사실 — 블록 메시지는 답으로 끝나지 않는다. `It answers: X` 뒤에 중앙 **429자**가
더 붙고 그 내용이 **차순위 목록**이다. 그리고 098 의 실패 답 `Light Blue` 는 그 목록의
**첫 항목**이고, 010 실패에 얽힌 `Silver Rewards Card` 도 그 목록에 있다.
(⚠099 는 예외 — 실패 답 `Navy Blue` 는 목록 밖이고 KB 스니펫에서 온 이름이다.)

> 사용자: *"차순위 답을 안 적으면 안 되나? 차순위 답 때문에 문제가 생기는 거 아닌가?"*

## 팔 (문맥 = **결정점까지의 전사** + 라이브에서 그 턴에 실제로 나간 것들)

  A_CUR     현행 그대로 (블록 + 차순위 + 뒤따르는 절차 지시)   ← 재현
  B_NORUN   **차순위 목록만** 삭제                              ← 사용자 제안
  C_NOTAIL  차순위 + **뒤따르는 절차 지시**까지 삭제
  D_NONE    블록 없음                                           ← 부정 통제
  E_ONLY    블록만(차순위 포함)                                 ← 천장·계기 검사

**세 태스크 전부** 잰다 — 098 만이 아니라 지금 통과하는 099·100 도. 레버는 하나를 사면 하나를
판다(§1.3): 차순위 삭제가 통과하던 곳을 깨는지 **같은 프로브에서** 본다.

⚠블록·주입 문구는 라이브에서 나간 축자 그대로다. 지우기만 하고 아무 문장도 새로 쓰지 않는다
(규칙 E: 메인 채널 = 값만). gold 는 채점에만 쓴다([[23]]).
⚠팔마다 담긴 것(블록·차순위·꼬리·자수)을 먼저 인쇄한다(C395′ 규칙).

실행: python x225_runnerup_removal.py [N]
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
from x219_adoption import ASK, BLOCK_SIG, GOLD, render            # noqa: E402
import x219_adoption as X219                                      # noqa: E402

X219.PATS = X219.PATS + [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "../../../reports/facet_rft_2026/sim_results/*.json.gz")]
FB_GZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../../../reports/facet_rft_2026/sim_results/"
                     "fb_bank_alllevers_20260810.jsonl.gz")
FB_RAW = "/home/woori/scratch/logs/fb_bank_alllevers_20260810.jsonl"

MAXCHARS = 120000
RUNNERS = re.compile(r"\n?The next best ones it ranked, in order: [^\n]*\n?")
ANSWER_SIG = "It answers:"
DECISION_SIG = "Accounts for user"


def fb_rows():
    if os.path.exists(FB_RAW):
        return [json.loads(l) for l in open(FB_RAW, encoding="utf-8", errors="replace")
                if l.strip()]
    return [json.loads(l) for l in gzip.open(FB_GZ, "rt", encoding="utf-8") if l.strip()]


def live_turn(gold):
    """그 답을 낸 사이드카 sim 의 **블록 턴 전체**를 순서대로 (블록 메시지, 뒤따르는 주입들)."""
    by = collections.defaultdict(list)
    for r in fb_rows():
        by[(r["sim"], r["turn"])].append(r)
    for (_sim, _turn), rs in sorted(by.items()):
        idx = [j for j, r in enumerate(rs) if BLOCK_SIG in r["text"]]
        if not idx:
            continue
        blk = rs[idx[0]]["text"]
        m = re.search(r"It answers: ([^.\n]+)", blk)
        if not m or m.group(1).strip().rstrip(".") != gold:
            continue
        tail = [r["text"] for r in rs[idx[0] + 1:]
                if r["kind"] in ("reminder-user", "tool-deny")]
        return blk, tail
    return None, []


def pick_bounded(task):
    best, seen = None, set()
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
                if (s.get("reward_info") or {}).get("reward") == 1 or s.get("id") in seen:
                    continue
                seen.add(s.get("id"))
                msgs = s.get("messages") or []
                if len(msgs) < 8:
                    continue
                cut = len(msgs)
                for i, m in enumerate(msgs):
                    if m.get("role") == "tool" and DECISION_SIG in str(m.get("content") or ""):
                        cut = i + 1
                        break
                body = render(msgs[:cut])
                kb = body.count("Score:")
                if kb == 0 or len(body) > MAXCHARS:
                    continue
                if best is None or kb > best[0]:
                    best = (kb, os.path.basename(p).split(".")[0], s.get("trial"), body)
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    out = {}
    for task in ("task_098", "task_099", "task_100"):
        gold = GOLD[task]
        blk, tail = live_turn(gold)
        got = pick_bounded(task)
        if not blk or not got:
            print("\n%s — 블록(%s)·문맥(%s) 중 하나가 없다. 건너뛴다."
                  % (task, "O" if blk else "X", "O" if got else "X"))
            continue
        kb, tag, trial, ctx = got
        blk_norun = RUNNERS.sub("\n", blk).rstrip()
        tail_txt = ("\n" + "\n".join(tail)) if tail else ""
        arms = [("A_CUR", ctx + "\n\n" + blk + tail_txt),
                ("B_NORUN", ctx + "\n\n" + blk_norun + tail_txt),
                ("C_NOTAIL", ctx + "\n\n" + blk_norun),
                ("D_NONE", ctx + tail_txt),
                ("E_ONLY", blk)]
        print("\n" + "=" * 96)
        print("%s  %s t%s · KB %d · 결정점까지 %d자 · gold=%r · n=%d"
              % (task, tag, trial, kb, len(ctx), gold, n))
        print("  [차순위 줄] %s" % (re.search(RUNNERS, blk).group(0).strip()[:150]
                                    if RUNNERS.search(blk) else "(없음)"))
        print("  [뒤따르는 주입] %d건 %d자" % (len(tail), len(tail_txt)))
        for name, body in arms:
            print("   %-9s 블록 %s · 차순위 %s · 꼬리 %s · %6d자"
                  % (name, "O" if BLOCK_SIG in body else "X",
                     "O" if RUNNERS.search(body) else "X",
                     "O" if tail and tail[0][:60] in body else "X", len(body)))
        for name, body in arms:
            c = collections.Counter()
            for i in range(n):
                try:
                    t = chat(body + "\n\n" + ASK, None, 0.0 if i == 0 else 0.7, 24).get(
                        "content", "")
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                c[" ".join(str(t).split())[:40]] += 1
            hit = sum(v for k, v in c.items()
                      if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold),
                                      str(k).strip(), re.I))
            out["%s/%s" % (task, name)] = [hit, n]
            print("  %-9s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X225_OUT", "x225_out.json"), "w"), indent=1)
    print("\n※ 읽는 법 — D_NONE 은 낮아야 하고(부정 통제) E_ONLY 는 높아야 한다(계기)."
          "\n  B_NORUN 이 A_CUR 보다 높으면 차순위가 범인이다. **단 셋 다 봐야 한다** —"
          "\n  098 에서 올라도 099·100 에서 내려가면 그것이 이 레버의 값이다(§1.3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
