# -*- coding: utf-8 -*-
r"""x231 — **어느 문장 하나가 값을 옮기는가**: 문장 하나 + 값 (유료 0 · 엔진 0).

## 왜 (사용자 지시 2026-08-10)

> *"지시 중에 어떤 게 바꾸는지 정확하게 확인하라. '손님 도구다' 때문인지 다른 문구 때문인지.
> 문구 + 값만으로 실험하면 나오지 않나?"*

x229 는 **leave-one-out**(전문에서 하나만 빼기)이었다 — 40,856~112,729자 문맥 위에서 한 문장을
빼는 것이라 신호가 잡음에 묻혔다(FULL 자체가 4/8·7/8로 흔들렸다). **leave-one-in** 이 옳다:
바닥을 깨끗하게 두고 **문장 하나만 얹는다**.

## 두 바닥에서 같은 사다리를 올린다

  BARE 바닥 = **값만**(문맥 0)          ← 사용자가 말한 형태. 가장 예민하다.
  CTX  바닥 = **문맥 + 값**             ← 라이브에 가까운 조건

  각 바닥에서:  NONE(값만) · +S0 · +S1 · … · FULL(문장 전부)

문장 하나를 얹었을 때 값이 무너지면 **그 문장이 범인**이다. 여러 문장이 각각 무너뜨리면
범인은 문장의 내용이 아니라 **지시가 있다는 사실**이다.

⚠새 문장을 쓰지 않는다 — 라이브 축자를 문장 단위로 자르기만 한다. gold 는 채점에만([[23]]).
⚠BARE 에서 안 무너지고 CTX 에서만 무너지면, 그 문장은 **혼자서는 무해하고 문맥과 결합할 때만**
해로운 것이다 — 그 구별이 이 프로브의 값어치다.

실행: python x231_one_sentence_in.py [N]
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

SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def run(body, gold, n):
    c = collections.Counter()
    for k in range(n):
        try:
            t = chat(body + "\n\n" + ASK, None, 0.0 if k == 0 else 0.7, 24).get("content", "")
        except Exception as e:
            t = "ERR %s" % type(e).__name__
        c[" ".join(str(t).split())[:40]] += 1
    hit = sum(v for kk, v in c.items()
              if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold), str(kk).strip(), re.I))
    return hit, c


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    out = {}
    for task in ("task_098", "task_100", "task_099"):
        gold = GOLD[task]
        blk, _tail = live_turn(gold)
        got = pick_bounded(task)
        if not blk or not got:
            print("\n%s — 재료가 없다. 건너뛴다." % task)
            continue
        kb, tag, trial, ctx = got
        i = blk.find(BLOCK_SIG)
        pre, val = blk[:i].rstrip(), blk[i:].strip()
        sents = [s.strip() for s in SPLIT.split(pre) if s.strip()]
        print("\n" + "=" * 96)
        print("%s  %s t%s · 문맥 %d자 · 앞문구 %d문장 · gold=%r · n=%d"
              % (task, tag, trial, len(ctx), len(sents), gold, n))
        for j, s in enumerate(sents):
            print("   S%-2d %s" % (j, " ".join(s.split())[:150]))
        ladder = ([("NONE", "")] + [("+S%d" % j, s) for j, s in enumerate(sents)]
                  + [("FULL", pre)])
        for base_name, base in (("BARE", ""), ("CTX", ctx + "\n\n")):
            print("  --- 바닥 = %s" % base_name)
            for label, add in ladder:
                body = base + ((add + "\n" + val) if add else val)
                hit, c = run(body, gold, n)
                out["%s/%s/%s" % (task, base_name, label)] = [hit, n]
                print("   %-5s %-5s gold %d/%d   %s"
                      % (base_name, label, hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X231_OUT", "x231_out.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\n※ 읽는 법 — 각 바닥에서 NONE 이 천장이다. 어떤 +S{i} 가 NONE 보다 뚜렷이 낮으면"
          "\n  그 문장이 범인이고, 여러 문장이 각각 낮추면 범인은 **지시의 존재**다."
          "\n  BARE 무해 + CTX 유해 = 그 문장은 **문맥과 결합할 때만** 해롭다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
