# -*- coding: utf-8 -*-
r"""x224 — **블록은 실제로 어디에 놓이는가** (포렌식 · 유료 0 · 엔진 0 · 모델 호출 0).

## 왜 (사용자 지시 2026-08-10)

> *"질문과 서브에이전트 블록 사이의 거리가 문제인가? 사실부터 확인해달라."*

레버도 절제도 짓기 전에 **사실**을 인쇄한다. 세 가지 거리를 각각 잰다 —

  ⑴ **블록 → 생성점**: 블록이 나간 뒤 그 턴에 우리가 **더 붙인 것**이 있는가.
     (같은 메시지 안에서 블록 **뒤에 붙은 글자수**, 같은 턴의 **뒤따르는 주입**)
  ⑵ **블록 안**: 한 메시지에 지시·세기·산수가 섞여 있다면 `It answers:` 는 그 안 어디인가.
  ⑶ **손님의 요청 → 블록**: 답해야 할 질문이 문맥의 어디에 있고 블록에서 얼마나 먼가.
     (전사 기준 자수 — 손님의 첫 실질 요청부터 결정점까지)

⚠사이드카 `sim` 은 해시라 전사와 시행별 대응이 안 된다(HANDOFF §10). ⑴⑵는 사이드카에서,
⑶은 전사에서 각각 재고 **합치지 않는다**.

실행: python x224_block_distance.py [태그]
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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
SIMS = os.path.join(ROOT, "reports/facet_rft_2026/sim_results")
BLOCK_SIG = "A separate check was run on the policy constants"
ANSWER_SIG = "It answers:"
DECISION_SIG = "Accounts for user"


def rows(tag):
    p = os.path.join(SIMS, "fb_%s.jsonl.gz" % tag)
    return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]


def part1_2(tag):
    by = collections.defaultdict(list)
    for r in rows(tag):
        by[(r["sim"], r["turn"])].append(r)
    print("\n⑴⑵ 사이드카 — 블록이 든 턴을 통째로 편다 (한 턴에 나간 것 전부·순서대로)")
    print("%-13s %5s  %-18s %7s  %s" % ("sim", "turn", "kind", "len", "블록 위치"))
    tail_after_answer, tail_after_block, follow = [], [], []
    for (sim, turn), rs in sorted(by.items()):
        if not any(BLOCK_SIG in r["text"] for r in rs):
            continue
        for j, r in enumerate(rs):
            i = r["text"].find(BLOCK_SIG)
            if i < 0:
                where = "-"
            else:
                a = r["text"].find(ANSWER_SIG)
                where = ("앞 %d자 · 블록 뒤 %d자 · `It answers:` 뒤 %d자"
                         % (i, len(r["text"]) - i - len(BLOCK_SIG),
                            (len(r["text"]) - a - len(ANSWER_SIG)) if a >= 0 else -1))
                tail_after_block.append(len(r["text"]) - i - len(BLOCK_SIG))
                if a >= 0:
                    tail_after_answer.append(len(r["text"]) - a - len(ANSWER_SIG))
                follow.append(sum(len(x["text"]) for x in rs[j + 1:]))
            print("%-13s %5d  %-18s %7d  %s" % (sim, turn, r["kind"], r["len"], where))
    def stat(v):
        return "n=%d 중앙 %d 최소 %d 최대 %d" % (
            len(v), sorted(v)[len(v) // 2] if v else -1, min(v) if v else -1,
            max(v) if v else -1)
    print("\n  · 같은 메시지에서 **블록 뒤에 붙은 글자수**: %s" % stat(tail_after_block))
    print("  · 같은 메시지에서 **`It answers:` 뒤 글자수**: %s" % stat(tail_after_answer))
    print("  · 같은 턴에서 **블록 뒤에 더 나간 주입 글자수**: %s" % stat(follow))


def part3(task):
    print("\n⑶ 전사 — 손님의 요청은 어디에 있고 결정점까지 얼마나 먼가 (%s·실패 sim)" % task)
    print("%-28s %2s %8s %8s %8s  %s"
          % ("run", "t", "전체자", "요청위치", "요청→결정", "손님의 첫 실질 요청"))
    for p in sorted(glob.glob(os.path.join(SIMS, "*.json.gz"))):
        try:
            d = json.load(gzip.open(p, "rt", encoding="utf-8"))
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
            # 문맥을 글자 위치로 재구성
            pos, ask_at, dec_at, ask_txt = 0, None, None, ""
            for m in msgs:
                c = " ".join(str(m.get("content") or "").split())
                if ask_at is None and m.get("role") == "user" and len(c) > 80:
                    ask_at, ask_txt = pos, c
                if dec_at is None and m.get("role") == "tool" and DECISION_SIG in c:
                    dec_at = pos + len(c)
                pos += len(c) + 1
            if ask_at is None:
                continue
            print("%-28s %2s %8d %8s %8s  %s"
                  % (os.path.basename(p).split(".")[0][:28], s.get("trial"), pos,
                     ask_at, (dec_at - ask_at) if dec_at else "-", ask_txt[:70]))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_alllevers_20260810"
    print("### %s" % tag)
    part1_2(tag)
    for task in ("task_098", "task_099", "task_100"):
        part3(task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
