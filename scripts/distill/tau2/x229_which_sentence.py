# -*- coding: utf-8 -*-
r"""x229 — **그 지시의 어느 문장이 값을 옮기는가** (문장 단위 leave-one-out · 유료 0 · 엔진 0).

## 왜 (사용자 질문 2026-08-10)

> *"'손님의 도구다'라는 말 하나가 문제인가? 그건 결정에 전혀 영향을 미치지 않아야 하는 것 아닌가?"*

논리적으로는 그렇다. 그런데 실측의 **지문**이 다른 것을 가리킨다 —

  · 100: 지시 있으면 `Hunter Green **Business Checking**`, 없으면 `Hunter Green` (x227)
  · 099: 지시 있으면 `Navy Blue **Business Checking**` (x225·x227)

즉 **결정이 바뀌는 게 아니라 요구되는 출력의 종류가 바뀐 것처럼** 보인다 — 정책 행 이름을 말하는
과제에서 **도구 인자를 채우는 과제**로. 그 지시의 마지막 문장이 축자로
`Arguments of submit_referral: user_id, account_type.` 이다.

가설을 문장 단위로 가른다. **문장 하나를 뺀 나머지 전부**를 놓고 재서, 뺐을 때 값이 돌아오면
그 문장이 범인이다(leave-one-out). 새 문장은 쓰지 않는다 — 라이브 축자에서 빼기만 한다.

  FULL      앞문구 전문 + 값          ← 재현
  −S{i}     앞문구에서 i번째 문장만 뺀 것 + 값
  NONE      앞문구 없이 값만          ← 천장(=x227 B_VALUE)

⚠팔마다 남은 문장 수·자수를 먼저 인쇄한다. 채점은 정확 일치.

실행: python x229_which_sentence.py [N]
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
        sents = [s for s in SPLIT.split(pre) if s.strip()]
        print("\n" + "=" * 96)
        print("%s  %s t%s · 문맥 %d자 · 앞문구 %d자 = %d문장 · gold=%r · n=%d"
              % (task, tag, trial, len(ctx), len(pre), len(sents), gold, n))
        for j, s in enumerate(sents):
            print("   S%-2d %s" % (j, " ".join(s.split())[:150]))
        arms = [("FULL", pre + "\n" + val)]
        for j in range(len(sents)):
            keep = "\n".join(s for k, s in enumerate(sents) if k != j)
            arms.append(("-S%d" % j, (keep + "\n" + val) if keep.strip() else val))
        arms.append(("NONE", val))
        for name, tailbody in arms:
            body = ctx + "\n\n" + tailbody
            c = collections.Counter()
            for k in range(n):
                try:
                    t = chat(body + "\n\n" + ASK, None, 0.0 if k == 0 else 0.7, 24).get(
                        "content", "")
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                c[" ".join(str(t).split())[:40]] += 1
            hit = sum(v for kk, v in c.items()
                      if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold),
                                      str(kk).strip(), re.I))
            out["%s/%s" % (task, name)] = [hit, n]
            print("  %-5s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X229_OUT", "x229_out.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\n※ 읽는 법 — FULL 이 낮고 NONE 이 높은 것이 전제다(x227 재현)."
          "\n  어떤 −S{i} 가 NONE 수준으로 올라오면 **그 문장 하나가 범인**이고,"
          "\n  어느 하나를 빼도 안 오르면 범인은 **문장이 아니라 지시의 존재 자체**다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
