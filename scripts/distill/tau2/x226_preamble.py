# -*- coding: utf-8 -*-
r"""x226 — **블록 앞에 붙은 문구가 문제인가** (격리 A/B · 유료 0 · 엔진 0).

## 왜 (사용자 질문 2026-08-10)

> *"거리랑 상관없이, 블록 앞에 붙은 문구들이 문제를 일으키는 건가?"*

x224 사실: 블록은 혼자 나가지 않는다 — 같은 메시지에서 **앞에 220~1,879자**가 먼저 온다
(`Error: [ACTION] 'submit_referral' is run by the CUSTOMER…` 절차 지시 · 상태별 세기 ·
창 산수). x225 의 `E_ONLY` 는 **그 앞 문구를 포함한 메시지 전문**이었고 세 태스크 전부
**8/8** 이었다 ⇒ 앞 문구만으로는 지지 않는다. 그러나 **긴 문맥과 함께일 때** 기여하는지는
안 쟀다. 여기서 그것만 가른다.

## 팔 (문맥 = 결정점까지의 전사 · x225 와 같은 사례)

  A_CUR      문맥 + 블록 메시지 **전문**(앞 문구 포함)     ← x225 재현
  B_NOPRE    문맥 + **값 부분만**(앞 문구 삭제)
  C_PREONLY  문맥 + **앞 문구만**(값 삭제)                 ← 기여 분리·부정 통제
  D_ISO      블록 **값 부분만**, 문맥 없음                  ← 천장

⚠아무 문장도 새로 쓰지 않는다 — 라이브 축자에서 **자르기만** 한다(규칙 E).
gold 는 채점에만 쓴다([[23]]). 팔마다 담긴 것을 먼저 인쇄한다(C395′ 규칙).

실행: python x226_preamble.py [N]
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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    out = {}
    for task in ("task_098", "task_099", "task_100"):
        gold = GOLD[task]
        blk, _tail = live_turn(gold)
        got = pick_bounded(task)
        if not blk or not got:
            print("\n%s — 재료가 없다. 건너뛴다." % task)
            continue
        kb, tag, trial, ctx = got
        i = blk.find(BLOCK_SIG)
        pre, val = blk[:i].rstrip(), blk[i:].strip()
        arms = [("A_CUR", ctx + "\n\n" + blk),
                ("B_NOPRE", ctx + "\n\n" + val),
                ("C_PREONLY", ctx + "\n\n" + pre),
                ("D_ISO", val)]
        print("\n" + "=" * 96)
        print("%s  %s t%s · KB %d · 문맥 %d자 · 앞문구 %d자 · 값 %d자 · gold=%r · n=%d"
              % (task, tag, trial, kb, len(ctx), len(pre), len(val), gold, n))
        print("  [앞 문구 첫 200자] %s" % " ".join(pre.split())[:200])
        for name, body in arms:
            print("   %-10s 앞문구 %s · 값 %s · %6d자"
                  % (name, "O" if pre and pre[:60] in body else "X",
                     "O" if BLOCK_SIG in body else "X", len(body)))
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
            out["%s/%s" % (task, name)] = [hit, n]
            print("  %-10s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X226_OUT", "x226_out.json"), "w"), indent=1)
    print("\n※ 읽는 법 — B_NOPRE 가 A_CUR 보다 높으면 **앞 문구가 기여**한 것이고,"
          "\n  같으면 앞 문구는 무관하다(그러면 남는 변수는 문맥 자체다)."
          "\n  C_PREONLY 는 낮아야 한다 — 높으면 값 없이도 맞힌다는 뜻이라 다른 팔이 무의미해진다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
