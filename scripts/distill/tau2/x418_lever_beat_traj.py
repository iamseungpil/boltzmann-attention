# -*- coding: utf-8 -*-
r"""x418 - 레버 발화 카운트 **두 번째 신호**: 궤적(도구 결과 본문)에 남은 우리 층 효과 문자열

## 왜 (2026-08-19 · x44 계기 교정)
`x44_lever_coverage.py` 는 **stderr 태그**만 센다. 그런데 상당수 레버는 아무것도 인쇄하지 않고
**도구 결과 본문을 바꾼다**(예: `[GROUNDING WARNING]` 은 t7328 로그에 0회지만 t7326 궤적에는 실재).
⇒ x44 의 "ON·무발화" 목록에는 **진짜 dark** 와 **조용한 레버**가 섞여 있다. 이 계기가 후자를 건진다.

효과 문자열 목록은 축자이고 인쇄된다(해석 0). 궤적 = `sim_results/*.results.json.gz`.
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F
import x396_saying_vs_doing as C

# 우리 층이 궤적에 남기는 효과 문자열(축자) -> 레버 이름
EFFECT = [
    ("[GROUNDING WARNING]", "T2_GROUND / grounding drop"),
    ("could not be verified against the account records", "T2_GROUND"),
    ("the customer never mentioned this kind of requirement", "intent_fields"),
    ("NOT_VERIFIED", "verify_identity 게이트"),
    ("[READ-FIRST]", "T2_SG_REQREADS"),
    ("blocked by a policy gate", "gate _BLOCK_NOTE"),
    ("Satisfy the gate requirement", "gate _BLOCK_NOTE"),
    ("[POLICY_QA]", "T2_FN_ISOLATE wrap"),
    ("NOTHING was actually checked", "T2_ZERO_ROW"),
    ("An empty result means", "T2_RETURN_EMPTY"),
    ("Do NOT conclude that nothing fits", "eligible-empty note"),
    ("re-read the exact value", "T2_GROUND 지시문"),
    ("[quote-pin]", "T2_QUOTE_PIN"),
    ("Tool unlocked:", "env unlock"),
    ("COMPACTED from view", "T2_VIEW_COMPACT"),
    ("was NOT executed", "gate deny"),
]


def main():
    print("=" * 100)
    print("x418 · 궤적에 남은 우리 층 효과 문자열 (t7326 40 sim)")
    print("효과 문자열 목록(축자):")
    for s, lv in EFFECT:
        print("   %-52s -> %s" % (s[:52], lv))
    print("=" * 100)
    cnt = collections.Counter()
    sims = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    n = 0
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            n += 1
            t = F.task_id(sim)
            body = " ".join(" ".join(str(m.get("content") or "").split())
                            for m in (sim.get("messages") or []) if m.get("role") == "tool")
            for s, lv in EFFECT:
                c = body.count(s)
                if c:
                    cnt[s] += c
                    sims[s] += 1
                    per[s][t] += c
    print("\n%-52s %8s %8s" % ("효과 문자열", "총발화", "sim수"))
    for s, lv in EFFECT:
        print("%-52s %8d %8d   %s" % (s[:52], cnt[s], sims[s], lv))
    print("\n총 %d sim" % n)
    print("\n## 태스크별 (발화가 있는 것만)")
    for s, lv in EFFECT:
        if not cnt[s]:
            continue
        print("  %-40s %s" % (s[:40], " ".join("%s×%d" % (k.replace("task_", ""), v)
                                               for k, v in per[s].most_common(8))))
    return 0


sys.exit(main())
