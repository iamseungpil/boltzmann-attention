#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x785 — `_enum_items` 영향 범위 전수 census + 대안 술어 오프라인 A/B (2026-09-06).

★왜 P9 처럼 「그냥 고치면」 안 되나
  P9 는 **사실 오류**였다(서버가 받는 인자를 우리가 없다고 우겼다) — 고치면 잃는 게 0.
  P11 은 다르다. `_enum_items`(t2_eplan_patch.py:107)는 **진짜 다건 요청도 잡는다**.
  없애면 coverage 실패(고객이 세 건을 말했는데 한 건만 처리)가 돌아올 수 있다.
  ⇒ 바꾸기 전에 **누구를 사고 누구를 파는지** 회수분으로 재야 한다([[70]]).

★확정된 결함 (2026-09-06 실측 · 함수를 직접 돌려 확인)
    "date of birth, current email, phone number, or mailing address"   -> 4   오발
    "Please dispute the charges at Starbucks, Amazon, and Netflix"     -> 0   누락
  검증 **필드 이름** 나열을 「기록 4건」으로 세고, 진짜 3건 분쟁은 못 센다. 술어가 거꾸로 돈다.
  그 4가 `multi_entity_hint`(:283) -> `discovery_L1`(:343) -> E-PLAN L1 deny 로 갔고
  `task_004`(과거 4/4 통과)가 이관을 못 해 0.0 이 됐다.

★이 프로브가 답하는 것
  ① `_enum_items>=3` 이 회수분에서 **몇 sim** 에 나오나 · 그중 **몇이 통과**했나
     (통과 sim 에서 나오면 그 술어를 바꿀 때 그 sim 들이 위태롭다)
  ② 발화시킨 **문장이 누구 발화**인가 — user 면 진짜 요청일 수 있고, **assistant** 면
     우리가 모델 자신의 나열을 세는 것이다(task_004 가 그 경우)
  ③ 대안 술어(선언 구동)로 바꾸면 발화가 어떻게 달라지나 — **같은 코퍼스에 두 술어를 동시에**
     돌려 교차표를 낸다

★대안 술어 (선언 구동 · gold 무참조)
  A2 `eplan.entity_key`(banking = "transaction_id")를 가진 **실물 id 가 도구 출력에 몇 개** 나왔나.
  필드 이름("date of birth")도, 상점 이름("Starbucks")도 id 가 아니라 안 걸린다.
  ⇒ [[22]] 닫힌 술어이고 [[52]] 「엔진=계산」에 맞는다. 의미 판단이 없다.

⛔이 프로브는 **읽기만** 한다. 엔진을 고치지 않는다.
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")
import t2_eplan_patch as E                                            # noqa: E402

SIM = "/home/woori/scratch/tau2-bench/data/simulations"
Q38 = "/home/woori/scratch/x768/q38_sims.txt"
A2P = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2/a2/banking_knowledge.gate.json"

A2 = json.load(open(A2P, encoding="utf-8"))
EKEY = ((A2.get("eplan") or {}).get("entity_key")) or "transaction_id"
# 선언된 id 의 모양은 **선언에서** 온다 — 엔진 리터럴 0. 도구 출력에서 그 키의 값을 센다.
ID_IN_JSON = re.compile(r'"%s"\s*:\s*"([^"]{4,})"' % re.escape(EKEY))
ID_IN_TEXT = re.compile(r'\b%s\b[:=]?\s*([A-Za-z0-9_\-]{4,})' % re.escape(EKEY))


def alt_entity_count(messages, upto):
    """대안 술어: `entity_key` 의 **실물 값**이 도구 출력에 몇 개 나왔나(고유)."""
    ids = set()
    for m in messages[:upto]:
        if (m.get("role") or "") != "tool":
            continue
        c = str(m.get("content") or "")
        ids |= set(ID_IN_JSON.findall(c)) | set(ID_IN_TEXT.findall(c))
    return len(ids)


def main():
    rw = {}
    for ln in open(Q38, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 4:
            rw[(p[0], p[1])] = 1 if (p[3] not in ("None", "") and float(p[3]) >= 1) else 0
    rows = []
    for tag in sorted({t for t, _ in rw}):
        f = os.path.join(SIM, tag, "results.json")
        if not os.path.exists(f):
            continue
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for s in (r.get("simulations") or []):
            key = (tag, s.get("task_id"))
            if key not in rw:
                continue
            ms = s.get("messages") or []
            fired, first = 0, None
            for i, m in enumerate(ms):
                n = E._enum_items(str(m.get("content") or ""))
                if n >= E._ENUM_MIN:
                    fired += 1
                    if first is None:
                        first = (i, m.get("role"), n, str(m.get("content") or "")[:110])
            alt = alt_entity_count(ms, first[0] if first else len(ms))
            rows.append((tag, s.get("task_id"), rw[key], fired, first, alt))
    # ── 보고 ────────────────────────────────────────────────────────────────
    hit = [r for r in rows if r[3] > 0]
    print("Q38 sim %d · `_enum_items>=3` 발화 sim %d (%.0f%%)" % (len(rows), len(hit), 100 * len(hit) / max(len(rows), 1)))
    print("  그중 통과 %d / 실패 %d   (전체 통과율 %.0f%%)"
          % (sum(r[2] for r in hit), len(hit) - sum(r[2] for r in hit),
             100 * sum(r[2] for r in rows) / max(len(rows), 1)))
    print()
    byrole = collections.Counter(r[4][1] for r in hit if r[4])
    print("★첫 발화를 만든 메시지의 role:", dict(byrole))
    print("   (assistant = **모델 자신의 나열**을 우리가 세고 있다는 뜻 · task_004 가 그 경우)")
    print()
    bytask = collections.Counter(r[1] for r in hit)
    print("발화 태스크 %d 종 · 상위:" % len(bytask))
    for t, n in bytask.most_common(12):
        ps = [r[2] for r in hit if r[1] == t]
        print("   %-11s sim %2d · 통과 %d" % (t, n, sum(ps)))
    print()
    print("=== 교차표: 구 술어(_enum_items>=3) × 대안(entity_key id >= 3) ===")
    tab = collections.Counter()
    for tag, task, ok, fired, first, alt in rows:
        tab[(fired > 0, alt >= 3)] += 1
    print("   %-22s %6s %6s" % ("", "대안 O", "대안 X"))
    print("   %-22s %6d %6d" % ("구 술어 O", tab[(True, True)], tab[(True, False)]))
    print("   %-22s %6d %6d" % ("구 술어 X", tab[(False, True)], tab[(False, False)]))
    print()
    only_old = [r for r in rows if r[3] > 0 and r[5] < 3]
    print("★구 술어만 잡는 %d sim (= 대안으로 바꾸면 **발화가 사라진다**)" % len(only_old))
    print("   그중 통과 %d — 이 수가 [[70]] 파는 것의 상한이다" % sum(r[2] for r in only_old))
    for tag, task, ok, fired, first, alt in only_old[:8]:
        rl = first[1] if first else "?"
        tx = (first[3] if first else "").replace("\n", " ")
        print("   %-11s rw=%d role=%-9s %s" % (task, ok, rl, tx[:78]))


if __name__ == "__main__":
    main()
