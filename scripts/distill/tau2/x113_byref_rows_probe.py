# -*- coding: utf-8 -*-
"""도구가 본 행은 몇 개였나 — `@last:` 참조를 **엔진 함수 그대로** 오프라인 재현한다.

023 실측(2026-08-06): 거래 원장은 60행·2024-11~2025-10 **매월** 있는데, `check_rebate_qualification`은
*"9 of 12 windows had NO input records"* 를 냈고 살아남은 창은 #0·#1·#2뿐이었다. 원인 후보는 셋이었다 —
ⓐ뷰 압축이 커밋본을 덮어쓴다 ⓑ덤프 파서에 상한이 있다 ⓒ창 매핑이 어긋난다.
ⓐ·ⓑ는 코드로 배제된다(압축은 `work` 사본에만·파서는 무상한). 남은 ⓒ를 **엔진 함수로 직접** 확인한다:
커밋된 덤프 텍스트를 `_parse_record_dump`로 파싱하고, `_month_window_index`로 창을 매겨
`last_complete` 선택(`k//12 == k_asof//12 - 1`)이 실제로 어느 행을 남기는지 센다.

  usage:  x113_byref_rows_probe.py [task_023] [--trial 0]
"""

import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from x109_task_dossier import load_sims                      # noqa: E402
import t2_scaffold_get as SG                                 # noqa: E402
import t2_compute as CP                                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TASK = next((a for a in sys.argv[1:] if a.startswith("task_")), "task_023")
TRIAL = int(sys.argv[sys.argv.index("--trial") + 1]) if "--trial" in sys.argv else 0


def main():
    sims = [s for s in load_sims()
            if s["task_id"] == TASK and (s.get("trial") or 0) == TRIAL]
    if not sims:
        print("sim 없음: %s t%s" % (TASK, TRIAL))
        return
    s = sims[0]
    print("== %s t%s (%s) ==" % (TASK, TRIAL, s["_src"]))

    dump = None
    for m in s.get("messages") or []:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and "credit_card_transaction_history" in c:
            dump = c                                   # `@last:` 규약 = 마지막 비에러 출력
    if dump is None:
        print("거래 덤프 없음")
        return

    rows = SG._parse_record_dump(dump)
    print("커밋된 덤프 %d자 → **파싱 행 %d개**" % (len(dump), len(rows)))
    print("  필드 예시: %s" % json.dumps(rows[0], ensure_ascii=False)[:220])

    # 인자는 궤적에서 그대로 읽는다(추정 금지).
    args = None
    for m in s.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            if "rebate" in str(tc.get("name") or ""):
                a = tc.get("arguments")
                args = json.loads(a) if isinstance(a, str) else a
                break
        if args:
            break
    anchor = (args or {}).get("account_opening_date")
    as_of = (args or {}).get("as_of_date")
    print("  인자: account_opening_date=%s · as_of_date=%s · monthly_threshold=%s"
          % (anchor, as_of, (args or {}).get("monthly_threshold")))

    date_fields = [f for f in ("posting_date", "transaction_date", "date") if f in rows[0]]
    print("  덤프의 날짜 필드: %s" % (date_fields or "없음"))
    for df in date_fields:
        ks = [CP._month_window_index(anchor, r.get(df)) for r in rows]
        ok = [k for k in ks if k is not None]
        k_asof = CP._month_window_index(anchor, as_of)
        yy = (k_asof // 12 - 1) if k_asof is not None else None
        sel = [k for k in ok if yy is not None and k // 12 == yy]
        print("\n  [%s] 창 인덱스 산출 %d/%d행 · as_of 창 k=%s · last_complete 연도 yy=%s"
              % (df, len(ok), len(rows), k_asof, yy))
        print("    연도별 행 수: %s" % dict(sorted(collections.Counter(k // 12 for k in ok).items())))
        print("    선택 연도 안의 월 창: %s"
              % dict(sorted(collections.Counter(k % 12 for k in sel).items())))
        missing = [i for i in range(12) if i not in {k % 12 for k in sel}]
        print("    ⇒ 비어 있는 창: %s" % (missing or "없음"))
    replay_op(rows, args)


def replay_op(rows, args):
    """A2가 선언한 rebate 스펙을 **그대로** 오프라인 평가한다 — 행이 어디서 사라지는지 본다."""
    import json as _j
    a2p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    a2 = _j.load(io.open(a2p, encoding="utf-8"))
    spec = None
    for t in (a2.get("scaffold_get_tools") or []):
        if "rebate" in str(t.get("name") or ""):
            spec = t
            break
    if not spec:
        print("\n  (A2에 rebate 선언 없음)")
        return
    print("\n== A2 선언 재현 ==")
    print("  이름: %s" % spec.get("name"))
    op = spec.get("op")
    print("  op 트리 키: %s" % _j.dumps(op, ensure_ascii=False)[:600])
    ctx = dict(args or {})
    ctx["transactions"] = rows
    try:
        out = CP.apply_op(op, ctx)
        print("  결과: %s" % _j.dumps(out, ensure_ascii=False)[:400])
    except Exception as e:
        print("  평가 예외: %r" % (e,))
    print("  ctx 사이드채널: _expected_groups=%s" % (ctx.get("_expected_groups"),))


if __name__ == "__main__":
    main()
