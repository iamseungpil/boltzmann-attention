# -*- coding: utf-8 -*-
r"""x223 — **손님의 말 + 서브에이전트 결과만으로 답하면 되는가** (유료 0 · 엔진 0).

## 왜 (사용자 지시 2026-08-10)

> *"메인 컨텍스트에서 질문에 답할 때 사용자의 메시지와 서브에이전트 메시지만 추려서
> 질문에 격리해서 해도 되지 않나?"*

x222 는 **오답 정박어를 지우는** 절제였다 — 그건 그 시행이 낸 오답을 알아야 지을 수 있으므로
라이브에 옮길 수 없다(진단 전용). 여기서는 **옮길 수 있는 형태 하나만** 잰다:
결정 턴의 답을 **{손님 발화 + 우리 블록}** 격리에서 짓는다. 이것은 라이브에서 오답을 몰라도
그대로 성립하고, C397(온톨로지 격리 100%)·규칙 E(메인 채널=값만)와 같은 계열이다.

## 팔 (전부 같은 질문으로 끝난다)

  T_FULL    실제 실패 문맥 전부 + 블록          ← 라이브 재현(기준선)
  S_ALLQ    **손님 발화 전부** + 블록           ← 사용자 설계(넓은 격리)
  S_ASKQ    **손님의 첫 요청 1건** + 블록        ← 사용자 설계(좁은 격리)
  N_NOSUB   손님 발화 전부, **블록 없음**       ← 부정 통제(오답이 나와야 한다)
  N_ONLY    **블록만**                          ← 천장·계기 검사

⚠어느 팔도 답을 강요하지 않는다 — 블록은 라이브에서 나간 문구 그대로이고, 우리가 고르는 것은
**무엇을 문맥에 담느냐**뿐이다. gold 는 채점에만 쓴다([[23]]).
⚠팔마다 담긴 것(블록·KB 수·손님 발화 수·자수)을 먼저 인쇄한다(C395′ 규칙).

채점 = 정확 일치(`Blue` 가 `Light Blue` 에 걸린다).

실행: python x223_subq_isolation.py [N]
"""
import collections
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
from x219_adoption import live_blocks as live_blocks_remote       # noqa: E402

import x219_adoption as X219                                      # noqa: E402

# 로컬(미러)에서도 돌게 — 리모트 경로가 없으면 repo 영속본을 본다
X219.PATS = X219.PATS + [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "../../../reports/facet_rft_2026/sim_results/*.json.gz")]

FB_GZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../../../reports/facet_rft_2026/sim_results/"
                     "fb_bank_alllevers_20260810.jsonl.gz")


def live_blocks():
    out = live_blocks_remote()
    if out or not os.path.exists(FB_GZ):
        return out
    for ln in gzip.open(FB_GZ, "rt", encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = o.get("text") or ""
        if BLOCK_SIG in t:
            m = re.search(r"It answers: ([^.\n]+)", t)
            if m:
                out.setdefault(m.group(1).strip(), t.strip())
    return out


MAXCHARS = 120000          # 창(44,672 토큰) 안에 드는 문맥만 T_FULL 로 세운다
CTRL = re.compile(r"###[A-Z_]+###")


def pick_bounded(task):
    """그 태스크의 **실패** sim 중 창에 드는 것 가운데 가장 혼잡한 것."""
    import glob
    best = None
    seen = set()
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
                key = (s.get("id"),)
                if key in seen:
                    continue
                seen.add(key)
                body = render(msgs)
                kb = body.count("Score:")
                if kb == 0 or len(body) > MAXCHARS:
                    continue
                if best is None or kb > best[0]:
                    best = (kb, os.path.basename(p).split(".")[0], s.get("trial"), msgs)
    return best
# 결정점 = 계좌 원장이 돌아온 자리 (블록은 그 직후에 나간다)
DECISION_SIG = "Accounts for user"


def decision_cut(msgs):
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and DECISION_SIG in str(m.get("content") or ""):
            return i + 1
    return len(msgs)


def user_turns(msgs, cut=None):
    """**결정점 이전**의 손님 발화만. 제어 토큰은 뺀다(C395′ ⒞ 재발 방지).

    결정 이후의 마무리 발화에는 그 시행이 이미 제출한 **오답이 들어 있다** — 라이브의 결정
    시점에는 존재하지 않는 문장이므로 담으면 안 된다.
    """
    cut = len(msgs) if cut is None else cut
    out = []
    for m in msgs[:cut]:
        if m.get("role") != "user":
            continue
        c = " ".join(CTRL.sub(" ", str(m.get("content") or "")).split())
        if c:
            out.append(c)
    return out


def audit(name, body, n_user):
    return ("   %-8s 블록 %s · KB %2d · 손님발화 %2d · %6d자"
            % (name, "O" if BLOCK_SIG in body else "X", body.count("Score:"),
               n_user, len(body)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    blocks = live_blocks()
    print("라이브 결정 블록 %d종: %s" % (len(blocks), sorted(blocks)))
    out = {}
    for task in ("task_098", "task_099", "task_100"):
        gold = GOLD[task]
        blk = blocks.get(gold)
        got = pick_bounded(task)
        if not blk or not got:
            print("\n%s — 블록(%s) 또는 실패 sim 을 못 찾았다. 건너뛴다."
                  % (task, "O" if blk else "X"))
            continue
        kb, tag, trial, msgs = got
        full = render(msgs)
        if len(full) > MAXCHARS:
            print("\n%s — 문맥 %d자 > 창(%d자). T_FULL 을 못 세우므로 건너뛴다."
                  % (task, len(full), MAXCHARS))
            continue
        cut = decision_cut(msgs)
        us = user_turns(msgs, cut)
        allq = "\n".join("[user] %s" % u for u in us)
        lastq = "[user] %s" % us[0] if us else ""      # 손님의 **요청**(첫 발화)
        arms = [("T_FULL", full + "\n\n" + blk, len(us)),
                ("S_ALLQ", allq + "\n\n" + blk, len(us)),
                ("S_ASKQ", lastq + "\n\n" + blk, 1),
                ("N_NOSUB", allq, len(us)),
                ("N_ONLY", blk, 0)]
        print("\n" + "=" * 96)
        print("%s  %s t%s · KB %d · %d자 · 손님발화 %d · gold=%r · n=%d"
              % (task, tag, trial, kb, len(full), len(us), gold, n))
        print("  [손님의 첫 요청] %s" % (us[-1][:200] if us else "(없음)"))
        for name, body, nu in arms:
            print(audit(name, body, nu))
        for name, body, _ in arms:
            c = collections.Counter()
            for i in range(n):
                p = body + "\n\n" + ASK
                try:
                    t = chat(p, None, 0.0 if i == 0 else 0.7, 24).get("content", "")
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                c[" ".join(str(t).split())[:40]] += 1
            hit = sum(v for k, v in c.items()
                      if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold),
                                      str(k).strip(), re.I))
            out["%s/%s" % (task, name)] = [hit, n]
            print("  %-8s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X223_OUT", "x223_out.json"), "w"), indent=1)
    print("\n※ 읽는 법 — N_NOSUB 0/8(부정 통제)·N_ONLY 8/8(계기)이 전제다."
          "\n  S_ALLQ·S_ASKQ 가 T_FULL 보다 높으면 *'손님 말 + 서브 결과만 추려 격리'* 가 성립한다."
          "\n  S 가 N_ONLY 보다 낮으면 정박은 **손님 발화 자체**에 남아 있는 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
