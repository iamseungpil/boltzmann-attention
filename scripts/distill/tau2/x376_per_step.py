#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""x376 — **per-step 궤적 추적**: 한 sim 을 turn 단위로 펼쳐 *첫 실패 지점*과 그 자리의
우리 층 개입을 나란히 본다(사용자 지시 2026-08-18 *"per step 궤적 추적하여 원인과 수정 레버 확정"*).

## 무엇을 조인하는가 (셋을 turn 으로 붙인다)

  · **궤적**(`results.json`) — 무엇을 말하고 무엇을 불렀는가
  · **trace**(`trace_<tag>.jsonl`) — **어느 기구가** 그 turn 에 떴는가(turn 보유 99.3%·2026-08-18 정본화)
  · **사이드카**(`fb_<tag>.jsonl`) — 우리가 그 turn 에 **무슨 문장을** 넣었는가(turn 보유 100%)

## 판정 규율

  ⚠**gold 대조는 방향만**: `action_match` 는 소수점 표기로 무너진다(C486) ⇒ *어느 gold 액션이
    아예 없는가*(도구 이름 수준)까지만 읽고 인자 일치는 주장하지 않는다.
  ⚠**원인 이름은 DF 코드로**(`t2_levers.CAUSES`) — 새 이름을 만들지 않는다([[48]]).
  ⚠**레버는 여기서 확정하지 않는다**: 이 스크립트는 *어디서·무엇이* 어긋났는지까지 낸다.
    처방은 ⛔0 대로 **격리로 결손을 먼저 재고** 나서 고른다.

사용: python x376_per_step.py <tag> <task_id> [--full]
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F  # noqa: E402

TAG = sys.argv[1]
TASK = sys.argv[2]
FULL = "--full" in sys.argv
SUF = ".results.json.gz"
NOISE = {"T2_A2_VARIANT", "T2_LEVER", "T2_SG_TRACE", "T2_FB_VIEW", "T2_AXIS"}


def main():
    sim = next(s for s in F.scored(TAG, SUF) if F.task_id(s) == TASK)
    key = F.simtag(sim)
    rw = (sim.get("reward_info") or {}).get("reward")
    print("=" * 96)
    print("x376  %s / %s   reward=%s  종료=%s  steps=%d"
          % (TAG, key, rw, F.term_reason(sim), len(sim.get("messages") or [])))
    print("=" * 96)

    gold = F.gold_actions(sim)
    gnames = []
    for a in gold:
        n = (a.get("action", {}) or {}).get("name") or a.get("name") or ""
        ar = (a.get("action", {}) or {}).get("arguments") or a.get("arguments") or {}
        inner = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                 or ar.get("discoverable_tool_name") or "")
        gnames.append(str(inner or n))
    done = [F.nameof(tc) for _m, tc in F.calls(sim)]
    done_inner = []
    for _m, tc in F.calls(sim):
        n = F.nameof(tc)
        a = F.argsof(tc)
        done_inner.append(str(F.inner_name(a) or n))
    print("\n[gold 액션 %d] %s" % (len(gnames), ", ".join(gnames) or "(없음)"))
    miss = [g for g in gnames if g not in done_inner]
    print("[호출된 것] %s" % ", ".join(sorted(set(done_inner))))
    print("[★한 번도 안 부른 gold 도구] %s" % (", ".join(sorted(set(miss))) or "없음"))

    marks = collections.defaultdict(list)
    for d in F.trace(TAG):
        if d.get("sim") != key:
            continue
        t = d.get("turn")
        m = d.get("mark")
        if isinstance(t, int) and m and m not in NOISE:
            marks[t].append((m, str(d.get("line") or "")))
    side = collections.defaultdict(list)
    for d in F.sidecar_rows(TAG):
        if d.get("sim") != key:
            continue
        t = d.get("turn")
        if isinstance(t, int):
            side[t].append((d.get("channel"), str(d.get("text") or "")))

    print("\n%-5s %-9s %s" % ("turn", "역할", "내용 / 호출 / 우리 층"))
    print("-" * 96)
    turn = 0
    for m in (sim.get("messages") or []):
        role = str(m.get("role"))
        if role == "assistant":
            turn += 1
        tcs = m.get("tool_calls") or []
        txt = " ".join(str(m.get("content") or "").split())
        if role in ("user", "assistant") and (txt or tcs):
            print("%-5d %-9s %s" % (turn, role, txt[:110] if txt else "(도구만)"))
        for tc in tcs:
            print("%-5s %-9s → CALL %s" % ("", "", F.label(F.nameof(tc), F.argsof(tc))[:90]))
        if role == "tool" and FULL:
            print("%-5s %-9s ← %s" % ("", "tool", txt[:110]))
        if role == "assistant":
            for mk, ln in marks.get(turn, []):
                print("%-5s %-9s ⟦%s⟧ %s" % ("", "", mk, ln[len(mk) + 3:][:88]))
            for ch, tx in side.get(turn, []):
                print("%-5s %-9s ⟪우리 문장·%s⟫ %s" % ("", "", ch, " ".join(tx.split())[:80]))
    print("-" * 96)
    print("⚠gold 대조는 **도구 이름 수준**까지만(C486) · 처방은 격리로 결손을 잰 뒤에(⛔0)")


if __name__ == "__main__":
    main()
