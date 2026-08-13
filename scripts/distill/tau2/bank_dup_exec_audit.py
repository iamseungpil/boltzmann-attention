# -*- coding: utf-8 -*-
"""중복 성공 실행 ↔ 우리 레버 발화 인접성 감사 ([[57]] 부정통제·U1 수리 前 실측).

073 t0 에서 **같은 인자의 성공 write 가 4회** 나갔고, 그 사이마다 우리 사이드카에
`[DISCOVERY-REQUIRED]`/`[DISCOVERY-STEP2]`/`[ACTION-REQUIRED]` 가 있었다. 이것이 그 sim
한 건의 우연인지, 아니면 우리 레버가 **완료 상태를 되감는** 일반 결함인지 센다.

세는 것 (sim 별):
  · dup   = 같은 (도구,인자) 조합의 **성공** 실행 2회+ → 그 반복 횟수 합
  · lever = 직전 성공 실행과 이 반복 사이 턴에 위 3문구 중 하나가 사이드카에 있는 반복 수
  · other = 그 사이에 우리 문구가 없던 반복 수 (= 모델 자발 반복 · **부정통제**)

사용(사이드카 있는 리모트): py bank_dup_exec_audit.py [tag ...]
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ⚠stdout 래핑은 **하지 않는다** — 아래 모듈이 이미 감싼다. 두 번 감싸면 앞 래퍼가 회수되며
#   버퍼를 닫아 `I/O operation on closed file` 로 죽는다(2026-08-14 실측).
from bank_fail_forensic_all import (jload, fb_for, label, nameof, argsof, norm)  # noqa: E402

MARKS = ("[DISCOVERY-REQUIRED]", "[DISCOVERY-STEP2]", "[ACTION-REQUIRED]")
ALL = ["bank_t7285_a_20260814g", "bank_t7285_b_20260814g",
       "bank_t7286_a_20260814h", "bank_t7286_b_20260814h"]


def write_names(d):
    """이 런의 gold 채점표가 **write** 로 표시한 도구 이름 집합(벤치 메타데이터·분석 전용).

    읽기 중복은 DB 를 안 깨지만 write 중복은 최종 상태를 깨서 태스크를 떨어뜨린다 — 둘을
    같은 칸에 세면 신호가 희석된다(2026-08-14: 전체 151건 중 자발 71% 인데, 이 분리 없이는
    write 중복의 귀속을 못 본다).
    """
    out = set()
    for s in d.get("simulations", []):
        for ck in ((s.get("reward_info") or {}).get("action_checks") or []):
            if ck.get("tool_type") != "write":
                continue
            a = ck.get("action") or {}
            out.add(label(a.get("name"), a.get("arguments") or {}))
            out.add(a.get("name"))
    return {n for n in out if n}


def main(tags):
    tot = collections.Counter()
    for tag in tags:
        d = jload(tag)
        fb = fb_for(tag)
        wnames = write_names(d)
        for s in d.get("simulations", []):
            msgs = s.get("messages") or []
            res = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
            simtag = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            # 우리 문구가 있던 턴
            lever_turns = sorted({r.get("turn") for r in (fb.get(simtag) or [])
                                  if any(k in (r.get("text") or "") for k in MARKS)
                                  and r.get("turn") is not None})
            seen = collections.defaultdict(list)      # (label,args) → 성공 실행 msg 인덱스들
            rows = []
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                for tc in (m.get("tool_calls") or []):
                    r = res.get(tc.get("id")) or {}
                    if r.get("error") or str(r.get("content") or "").lstrip().startswith("Error:"):
                        continue
                    key = (label(nameof(tc), argsof(tc)), norm(argsof(tc)))
                    if seen[key]:
                        prev = seen[key][-1]
                        near = [t for t in lever_turns if prev < t <= i]
                        isw = key[0] in wnames or nameof(tc) in wnames
                        rows.append((key[0], prev, i, bool(near), isw))
                    seen[key].append(i)
            dup = len(rows)
            lever = sum(1 for r in rows if r[3])
            wrows = [r for r in rows if r[4]]
            if dup:
                print("%-28s %-12s dup=%-3d 우리문구-사이=%-3d 자발=%-3d ★write중복=%d(우리 %d)" % (
                    simtag, "reward=%s" % (s.get("reward_info") or {}).get("reward"),
                    dup, lever, dup - lever, len(wrows), sum(1 for r in wrows if r[3])))
                for nm, a, b, near, isw in (wrows or rows)[:6]:
                    print("    %-3s %-42s %d→%d %s" % ("W" if isw else "r", nm[:42], a, b,
                                                       "★우리문구" if near else "자발"))
            tot["dup"] += dup
            tot["lever"] += lever
            tot["wdup"] += len(wrows)
            tot["wlever"] += sum(1 for r in wrows if r[3])
            tot["sims"] += 1
    print("-" * 84)
    print("합계: sim %d · 중복 성공 실행 %d · 그중 직전에 우리 문구 %d (%.0f%%) · 자발 %d" % (
        tot["sims"], tot["dup"], tot["lever"],
        100.0 * tot["lever"] / tot["dup"] if tot["dup"] else 0, tot["dup"] - tot["lever"]))
    print("★write 중복만: %d · 그중 우리 문구 직전 %d (%.0f%%) · 자발 %d" % (
        tot["wdup"], tot["wlever"],
        100.0 * tot["wlever"] / tot["wdup"] if tot["wdup"] else 0, tot["wdup"] - tot["wlever"]))


if __name__ == "__main__":
    main([a for a in sys.argv[1:]] or ALL)
