# -*- coding: utf-8 -*-
r"""x230 — **답을 어디서 읽는가**: 표식 값으로 출처를 추적한다 (유료 0 · 엔진 0).

## 왜

x229: 앞문구를 **문장 단위로 하나씩 빼도** 값이 안 돌아온다(100: 최대 5/8·전부 빼면 8/8).
⇒ 범인은 특정 문장이 아니다. 그리고 오답의 **모양**이 지문을 남긴다 — `Hunter Green
**Business Checking**` · `Navy Blue **Business Checking**` 은 대화의 문서·기록이 쓰는 표기이고,
098 의 `Light Blue` 는 그 궤적의 **첫 KB 스니펫**이다.

⇒ 가설: 지시가 붙으면 모델이 답을 **우리 블록이 아니라 대화에서 읽는다**(출처 전환).
결정 기준이 바뀌는 것이 아니다.

## 어떻게 (표식)

블록이 지목하는 이름을 **대화 어디에도 없는 표식**으로 바꾼다(`Zephyr Teal`). 그러면 답의
출처가 **문자로 갈린다** —

  · 표식을 말한다  → 우리 블록에서 읽었다
  · 대화의 이름을 말한다 → 대화에서 읽었다(재도출)

  M_DIR    문맥 + 앞문구(지시) + **표식 블록**
  M_VAL    문맥 + **표식 블록**만
  M_NOCTX  **표식 블록**만 (문맥 없음)      ← 계기 검사(표식을 읽을 수 있는가)

⚠**표식은 측정 도구다. 라이브에 나가지 않는다** — 가짜 값을 내보내는 것은 [[25]] 위반이다.
⚠gold 는 여기서 안 쓴다. 세는 것은 *"표식이냐 대화의 이름이냐"* 뿐이다.

실행: python x230_marker_provenance.py [N]
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

MARK = "Zephyr Teal"


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
        # 블록 안의 gold 이름만 표식으로 치환 (차순위·근거 행 표기는 그대로 둔다)
        val_m = re.sub(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(gold), MARK, val)
        n_sub = len(re.findall(re.escape(MARK), val_m))
        in_ctx = len(re.findall(re.escape(MARK), ctx))
        print("\n" + "=" * 96)
        print("%s  %s t%s · 문맥 %d자 · 표식 치환 %d회 · 표식이 문맥에 %d회(0이어야 한다)"
              % (task, tag, trial, len(ctx), n_sub, in_ctx))
        arms = [("M_DIR", ctx + "\n\n" + pre + "\n" + val_m),
                ("M_VAL", ctx + "\n\n" + val_m),
                ("M_NOCTX", val_m)]
        for name, body in arms:
            c = collections.Counter()
            mark_hit = 0
            for k in range(n):
                try:
                    t = chat(body + "\n\n" + ASK, None, 0.0 if k == 0 else 0.7, 24).get(
                        "content", "") or ""
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                if re.search(re.escape(MARK), t, re.I):
                    mark_hit += 1
                c[" ".join(str(t).split())[:40]] += 1
            out["%s/%s" % (task, name)] = [mark_hit, n]
            print("  %-8s 표식 %d/%d   %s" % (name, mark_hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X230_OUT", "x230_out.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\n※ 읽는 법 — M_NOCTX 가 높아야 계기가 산다(표식을 읽을 수 있다)."
          "\n  M_VAL 높음 + M_DIR 낮음 = **지시가 출처를 대화로 옮긴다**(가설 확증)."
          "\n  둘 다 낮으면 모델은 애초에 블록에서 안 읽는 것이고, 채택 이득의 설명이 달라진다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
