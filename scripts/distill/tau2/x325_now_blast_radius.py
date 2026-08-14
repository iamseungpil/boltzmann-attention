# -*- coding: utf-8 -*-
"""x325 — **`now` 전달 수정의 영향 반경**(사용자 지시 2026-08-15).

물음: `T2_SEARCH_AGENT` 가 `now 미확정` 으로 침묵한 자리를 살리면 **어느 태스크의 pass 가
움직일 수 있나**. 그리고 **움직이면 안 되는** 태스크(이미 전달·이미 통과)는 어디인가([[57]]
부작용 계측 = Δspurious 감시 목록).

세는 것(전부 우리 로그 프로토콜·도메인 판단 0·[[59]]):
  BLOCKED    `now 미확정` 침묵 — 이 수정이 여는 자리
  DELIVERED  재료가 실제로 나간 자리(`group=…`)
  AXIS-DONE  설계된 침묵(요청 축 처리 완료) — 이 수정과 무관
  NO-DOCS    코퍼스에서 문서를 못 찾음 — 다른 병
  (줄 없음)  검색 에이전트가 그 태스크에서 **아예 안 열림** — 이 수정과 무관

판정 등급:
  ★열림-완전   BLOCKED>0 ∧ DELIVERED=0   재료가 한 번도 안 갔다 → 최대 기대
  ☆열림-부분   BLOCKED>0 ∧ DELIVERED>0   일부만 갔다
  ―무관        BLOCKED=0
  ⚠불변의무    이미 통과 중 → 거동이 바뀌면 그게 회귀다

사용: py x325_now_blast_radius.py <log> [<log>...]   (성적은 같은 태그의 결과에서 읽는다)
"""
import collections
import io
import os
import re
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402

SIM = re.compile(r"\[sim=(task_\d+)#[^\]]*\]\s*\[T2_SEARCH_AGENT\]\s*(.*)")


def classify(rest):
    if rest.startswith("group="):
        return "DELIVERED"
    if "now 미확정" in rest:
        return "BLOCKED"
    if "모두 처리됨" in rest or "축 처리 완료" in rest:
        return "AXIS-DONE"
    if "문서를 못 찾음" in rest:
        return "NO-DOCS"
    return "OTHER"


def tag_of(log):
    return os.path.basename(log).replace(".log", "")


def main(logs):
    per = collections.defaultdict(collections.Counter)
    score = {}
    for log in logs:
        for ln in io.open(log, encoding="utf-8", errors="replace"):
            m = SIM.search(ln)
            if m:
                per[m.group(1)][classify(m.group(2))] += 1
        try:
            for s in F.sims(tag_of(log)):
                ri = s.get("reward_info") or {}
                if ri.get("reward") is None:
                    continue
                t = F.task_id(s)
                n, p = score.get(t, (0, 0))
                score[t] = (n + 1, p + (1 if ri.get("reward") == 1.0 else 0))
        except Exception as e:
            print("성적 로드 실패 %s: %r" % (tag_of(log), e))

    print("%-10s %8s %10s %10s %8s  %s" %
          ("task", "BLOCKED", "DELIVERED", "AXIS-DONE", "pass", "판정"))
    print("-" * 78)
    tot = collections.Counter()
    for t in sorted(set(per) | set(score)):
        c = per[t]
        n, p = score.get(t, (0, 0))
        if c["BLOCKED"] and not c["DELIVERED"]:
            verdict = "★열림-완전"
        elif c["BLOCKED"]:
            verdict = "☆열림-부분"
        elif not c:
            verdict = "―미개시"
        else:
            verdict = "―무관"
        if n and p == n:
            verdict += " ⚠불변의무"
        tot[verdict.split(" ")[0]] += 1
        print("%-10s %8d %10d %10d %5d/%-3d  %s"
              % (t, c["BLOCKED"], c["DELIVERED"], c["AXIS-DONE"], p, n, verdict))
    print("-" * 78)
    print("요약:", dict(tot))
    op = [t for t in sorted(per) if per[t]["BLOCKED"]]
    sims = sum(score.get(t, (0, 0))[0] for t in op)
    won = sum(score.get(t, (0, 0))[1] for t in op)
    print("영향권 태스크 %d개 · 그 안의 sim %d · 현재 통과 %d" % (len(op), sims, won))


if __name__ == "__main__":
    main(sys.argv[1:] or ["/home/woori/scratch/logs/bank_t7295_a_20260815n.log",
                          "/home/woori/scratch/logs/bank_t7295_b_20260815n.log"])
