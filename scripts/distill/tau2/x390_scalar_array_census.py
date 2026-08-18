# -*- coding: utf-8 -*-
"""x390 — `T2_SCALAR_ARRAY` 의 **표적이 이 도메인에 존재하는가** (무료·결정론·LLM 0).

사용자 지시(2026-08-18): 목적불명 레버 넷을 리뷰한 뒤 — *"켜고 계수 돌려라"*.

## 왜 세는가 ([[62]] ①)
`T2_SCALAR_ARRAY` 는 **단수 이름 인자에 배열이 온 것**을 표면화한다(A2 문면 축자:
*"You passed a list of {n} values in `{field}`, which takes a single value. Issue one call per
value."*). 술어는 닫혀 있고 도메인 리터럴 0이다. 그런데 **이 도메인에서 그 결함이 몇 번 나는지**
측정이 없다 — 표적이 0이면 켜도 무발화이고, 있으면 그만큼 값이 있다.

## 방법
영속 궤적(리모트 `data/simulations/*/results.json`)의 **에이전트 호출 인자**를 정본 술어
`t2_axis_levers.scalar_array_note` 로 그대로 통과시킨다(사본 금지·[[67]]). 중첩 JSON 문자열도
그 함수가 푼다. 세는 것은 **발화했을 자리**이지 pass 가 아니다.

usage: py -3 x390_scalar_array_census.py <sim_dir> [<sim_dir> ...]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_axis_levers as AX

TPL = "{field}|{n}"          # 계수용 — 문면은 A2 몫이고 여기선 자리만 센다


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: x390_scalar_array_census.py <sim_dir> ...")
        return 2
    hits = collections.Counter()
    per_task = collections.Counter()
    sims = calls = 0
    for d in dirs:
        p = os.path.join(d, "results.json")
        if not os.path.exists(p):
            continue
        try:
            data = json.load(io.open(p, encoding="utf-8", errors="replace"))
        except Exception as e:
            print("  (건너뜀 %s: %r)" % (d, e))
            continue
        for s in data.get("simulations") or []:
            sims += 1
            tid = s.get("task_id")
            for m in s.get("messages") or []:
                for tc in (m.get("tool_calls") or []):
                    if str(tc.get("requestor") or "assistant") != "assistant":
                        continue
                    calls += 1
                    a = tc.get("arguments") or {}
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except Exception:
                            continue
                    if not isinstance(a, dict):
                        continue
                    note = AX.scalar_array_note(a, TPL)
                    if note:
                        hits[note] += 1
                        per_task[tid] += 1
    print("sim %d · 에이전트 호출 %d" % (sims, calls))
    print("표적(단수 이름에 배열) 총 %d건 · %d 태스크" % (sum(hits.values()), len(per_task)))
    for k, v in hits.most_common(15):
        print("   %-46s %d" % (k, v))
    if per_task:
        print("태스크별:", dict(per_task.most_common(10)))
    if not hits:
        print("⇒ 표적 0 — 이 코퍼스에서는 켜도 무발화다(레버의 값은 여기서 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
