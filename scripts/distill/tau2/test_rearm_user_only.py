#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`T2_REARM_USER_ONLY`(A-3′) 래칫 — **구현이 측정한 것과 같은가**를 잠근다.

## 왜 이 형태인가 ([[76]] *격리 = 서브다* · [[78]] 격리→라이브 배선)

A-3′ 의 부호표는 `x553_rearm_role_split.py` 가 **정본 술어를 두 번 부르는 방식**으로 냈다:
assistant 메시지의 content 를 None 으로 만들고 창을 0 부터 열어 *"user 발화 · 전 접두"* 를
흉내 냈다. 이제 그 처방이 플래그로 라이브에 들어갔다. 그러면 잠글 것은 효과가 아니라
**동치성**이다 — 플래그 ON 의 결과가 x553 이 잰 팔과 **발화마다 같아야** 한다. 다르면 우리가
잰 것과 다른 것을 켠 것이고, 그때 부호표는 아무 것도 보증하지 않는다.

    ①  ON(원본 메시지·실제 served_at)  ==  OFF(assistant 지운 메시지·served_at 0)   ← 발화마다
    ②  OFF(원본·실제 served_at)        ==  라이브 로그가 남긴 관측 신규 계열        ← 바이트 불변
    ③  부호표가 결정을 안 뒤집는다      기준선 kill 27/84 · 통과 sim 3 · **반증 잔존 1**

재료는 **실제 궤적**이다 — 영속 로그에 `[T2_SEARCH_REARM]` 이 있는 태그 전량(합성 0).

실행: PYTHONIOENCODING=utf-8 py -3 test_rearm_user_only.py [--tags GLOB]
"""
import argparse
import fnmatch
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_forensic as F                                              # noqa: E402
import x553_rearm_role_split as X                                    # noqa: E402

FAIL = []


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


def flagged(on, fn, *a, **k):
    prev = os.environ.get("T2_REARM_USER_ONLY")
    if on:
        os.environ["T2_REARM_USER_ONLY"] = "1"
    else:
        os.environ.pop("T2_REARM_USER_ONLY", None)
    try:
        return fn(*a, **k)
    finally:
        os.environ.pop("T2_REARM_USER_ONLY", None)
        if prev is not None:
            os.environ["T2_REARM_USER_ONLY"] = prev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="*")
    a = ap.parse_args(argv)

    import gate_interpreter as GI
    po = (GI.load_domain_a2("banking_knowledge") or {}).get("policy_ontology") or {}
    if not po.get("doc_index"):
        print("A2 doc_index 없음 — 잠글 것이 없다")
        return 2

    files = {}
    for p in F.all_result_files():
        files.setdefault(F.tag_of_file(p), p)

    judged = kill = mismatch = repro_bad = 0
    kill_r1 = alive = 0
    for tag in sorted(files):
        if not fnmatch.fnmatch(tag, a.tags):
            continue
        try:
            text = F.log_text(tag)
        except Exception:
            continue
        if "[T2_SEARCH_REARM]" not in (text or ""):
            continue
        try:
            by = {F.simtag(s): s for s in F.sims(files[tag])}
        except Exception:
            continue
        for ev in X.firings_in_log(text):
            sim = by.get(ev["sim"])
            if sim is None or ev["served_at"] is None or ev["fire_turn"] is None:
                continue
            raw = (sim.get("messages") or [])[: int(ev["fire_turn"])]
            allm = [X._M(m.get("role"), m.get("content")) for m in raw]
            useronly = [X._M(m.role, m.content if m.role == "user" else None) for m in allm]
            obs = set(ev["new"])
            asis = flagged(False, X.replay, po, allm, ev["group"], ev["served"], ev["served_at"])
            if not (obs and obs <= asis):
                repro_bad += 1
                continue
            judged += 1
            on = flagged(True, X.replay, po, allm, ev["group"], ev["served"], ev["served_at"])
            sim_a3p = flagged(False, X.replay, po, useronly, ev["group"], ev["served"], 0)
            if on != sim_a3p:
                mismatch += 1
                if mismatch <= 3:
                    print("   [mismatch] %s %s ON=%s ↔ 측정팔=%s"
                          % (tag, ev["sim"], sorted(on), sorted(sim_a3p)))
            if not on:
                kill += 1
                if (sim.get("reward_info") or {}).get("reward") == 1.0:
                    kill_r1 += 1
                    if X.gold_write_after(sim, ev["fire_turn"]):
                        alive += 1

    print("§1 구현 == 측정한 팔 (발화마다)")
    chk(judged > 0, "판정 가능한 발화가 있다", judged)
    chk(mismatch == 0, "ON 결과가 x553 이 잰 팔과 **전부 일치**", "불일치 %d / %d" % (mismatch, judged))

    print("\n§2 플래그 OFF = 바이트 불변")
    chk(repro_bad <= 1, "OFF 재생이 라이브 관측을 재현(재현 실패 %d 허용 1)" % repro_bad, repro_bad)

    print("\n§3 부호표가 결정을 뒤집지 않는가")
    # ⚠절대 수는 **런이 쌓이면 움직인다**(2026-08-26 저녁 판정 78 → 밤 84). 그래서 잠그는 것은
    #   숫자가 아니라 **결정을 뒤집는 조건**이다. 기준선(x553·2026-08-26 밤·발화 91 / 판정 84):
    #   A-3 67 kill ↔ **A-3′ 27 kill** · 통과 sim 3 · **반증 잔존 1**.
    print("   지금: 판정 %d · kill %d(%.0f%%) · 통과 sim %d · 반증 잔존 %d"
          % (judged, kill, 100.0 * kill / max(1, judged), kill_r1, alive))
    chk(alive <= 1, "반증을 견디는 순손실이 **1건 이하**", alive)
    chk(kill_r1 <= 3, "통과 sim 에서 죽는 발화 3건 이하", kill_r1)
    chk(0.20 <= kill / max(1.0, float(judged)) <= 0.50,
        "죽는 비율이 기준선 대역(20~50%) 안", "%.0f%%" % (100.0 * kill / max(1, judged)))

    print("\n%s  (%d 실패)" % ("FAIL" if FAIL else "ALL OK", len(FAIL)))
    for m in FAIL:
        print("  - %s" % m)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
