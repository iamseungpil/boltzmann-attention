# -*- coding: utf-8 -*-
r"""x413 - 태스크별 **궤적 정독** 덤퍼 (사용자 지시 2026-08-19: 통계 말고 궤적으로 실패 시작지점)

한 sim 을 사람이 읽을 수 있는 사건열로 편다. 집계 0·판정 0 — 읽기 위한 것이다.
  U:  손님 발화 (머리 160자)
  A:  어시스턴트 산문 (머리 160자)
  >>  도구 호출 + 인자 요약
  <-  그 호출의 결과 (머리 160자·길이 표시)
  ##  gold 액션 목록(match 표시)

사용: py -3 x413_trace_read.py task_040 [task_085 ...]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F
import x396_saying_vs_doing as C


def head(s, n=160):
    return " ".join(str(s or "").split())[:n]


def dump(sim):
    rw = (sim.get("reward_info") or {}).get("reward")
    print("=" * 118)
    print("### %s  trial %s  reward=%s  term=%s"
          % (F.task_id(sim), sim.get("trial"), rw, F.term_reason(sim)))
    print("## gold")
    for g in C.gold_rows(sim):
        print("   [%s] %-44s %s" % ("O" if g["match"] else "X", g["name"][:44],
                                    head(json.dumps(g["args"], ensure_ascii=False), 90)))
    msgs = sim.get("messages") or []
    R = {}
    for m in msgs:
        if m.get("role") == "tool" and m.get("id"):
            R[m["id"]] = m
    print("## 사건열")
    n = 0
    for i, m in enumerate(msgs):
        role = m.get("role")
        c = m.get("content")
        if role == "user" and c:
            print("%3d U: %s" % (i, head(c)))
        elif role == "assistant":
            if c:
                print("%3d A: %s" % (i, head(c)))
            for tc in (m.get("tool_calls") or []):
                a = F.argsof(tc)
                nm = str(F.nameof(tc))
                inner = F.inner_name(a)
                lbl = nm + ("(%s)" % inner if inner else "")
                ar = head(json.dumps(a, ensure_ascii=False, default=str), 110)
                print("%3d >> %-52s %s" % (i, lbl[:52], ar))
                tm = R.get(tc.get("id"))
                if tm is not None:
                    body = " ".join(str(tm.get("content") or "").split())
                    print("      <- (%6d자) %s" % (len(body), head(body, 150)))
                n += 1
    print("## 호출 %d회 · 메시지 %d" % (n, len(msgs)))


def main():
    want = [a for a in sys.argv[1:] if a.startswith("task_")]
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if want and F.task_id(sim) not in want:
                continue
            dump(sim)
    return 0


sys.exit(main())
