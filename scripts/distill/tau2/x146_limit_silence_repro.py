# -*- coding: utf-8 -*-
"""C324 근거: 상한 대조가 **만들어질 수 있었는데 나가지 못했다**를 오프라인으로 확증한다.

배경 — 2026-08-08 라이브 2 sim 중 하나는 상한·문턱 문장이 **0회** 나갔다(사이드카 귀속으로
확정: 그 sim의 `T2_LIMIT_REDUCE` beat 0). 그래서 두 가설이 남았다.
  ⓐ 산수가 빈 문자열을 냈다(값 부재·이름 불일치 등) → 문장 자체가 없었다
  ⓑ 문장은 만들어지는데 **발화 경로가 없었다**
이 스크립트는 궤적에 남은 원장 출력과 repo의 A3 온톨로지로 **실제 엔진 함수를 그대로 돌려**
ⓐ를 배제한다. 유료 런 불요·LLM 불요([[09]]).

읽는 것은 전부 tracked 정본이다:
  · 궤적   `reports/facet_rft_2026/sim_results/<TAG>.json.gz`
  · 선언   `a2/<domain>.settings.json`  (ledger_metrics — 필드명·문구를 여기서만 읽는다)
  · 상수   `a2/<domain>.specific.json`  (policy_ontology)
도메인 어휘는 이 파일에 없다 — 필드명도 축 이름도 선언에서 가져온다([[05]]).

실행: py -3 x146_limit_silence_repro.py [TASK_ID]
"""
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

import t2_ledger as LG                                        # noqa: E402
import t2_factdag as FD                                       # noqa: E402
from gate_interpreter import load_domain_a2                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GZ = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                  "sim_results", "bank_stage1b_20260808.json.gz")
DOMAIN = "banking_knowledge"
AXIS = "annual_referral_limit"          # 대조할 축(= A3 축 이름·선언과 같은 어휘)
NOW = "11/14/2025"                      # 그 sim의 get_current_time 반환(궤적 사실)


def _rows_from_output(text, spec):
    """도구 출력 블록을 선언된 필드만 뽑아 행으로. **분석 전용**(엔진 경로 아님).

    엔진은 이 일을 하지 않는다 — 라이브에서는 LLM이 전사하고([[52]] 해석은 LLM),
    여기서는 그 전사를 대신할 뿐이다. 필드명은 A2 선언에서만 온다.
    """
    keys = [k for k in (spec.get("group_field"), spec.get("date_field"),
                        spec.get("age_field")) if k]
    out = []
    for blk in re.split(r"\n\s*\d+\.\s", text or "")[1:]:
        row = {}
        for k in keys:
            m = re.search(re.escape(k) + r":\s*(.+)", blk)
            if m:
                row[k] = m.group(1).strip()
        if row:
            out.append(row)
    return out


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "task_101"
    a2 = load_domain_a2(DOMAIN)
    specs = a2.get("ledger_metrics") or []
    spec = next((s for s in specs if s.get("exhausted_text")), None)
    if spec is None:
        print("상한 대조 선언 없음 — 중단")
        return 1
    trig = spec.get("trigger_tool")

    d = json.load(gzip.open(os.path.normpath(GZ), "rt", encoding="utf-8"))
    sim = next(s for s in d["simulations"] if s.get("task_id") == task)
    ids = {tc.get("id"): tc.get("name")
           for m in sim["messages"] for tc in (m.get("tool_calls") or [])}
    text = ""
    for m in sim["messages"]:
        if m.get("role") == "tool" and ids.get(m.get("id")) == trig:
            text = m.get("content") or ""
            break
    if not text:
        print("%s: 원장 도구(%s) 출력 없음 — 이 sim은 대상 아님" % (task, trig))
        return 0

    rows = _rows_from_output(text, spec)
    remain, inwin, tally = LG.window_and_tally(rows, spec, now=NOW)
    lims = FD._a3_map((a2.get("policy_ontology") or {}).get("rows") or (), {"axis": AXIS})

    print("== %s ==" % task)
    print("원장 행수      : %d" % len(rows))
    print("그룹별 누계    : %s" % ", ".join("%s %d" % kv for kv in sorted(tally.items())))
    print("A3 상한 주어수 : %d" % len(lims))
    print("창             : remain=%s inwin=%s (window=%s/%s)"
          % (remain, inwin, spec.get("window_days"), spec.get("window_max")))

    out = LG.exhausted_text(tally, lims, spec)
    print("\n-- exhausted_text (엔진 함수 그대로) --")
    print(out if out else "(빈 문자열 = 침묵)")

    # 표기 축 점검: 원장 그룹이 A3에 없으면 그 그룹은 **판정 자체를 못 받는다**(C319 축).
    unknown = [g for g in tally if g not in lims]
    print("\nA3에 없는 원장 그룹: %s" % (unknown or "(없음)"))

    # 손님이 실제로 실행한 값과 대조 — 소진 그룹을 골랐는가.
    picked = []
    for m in sim["messages"]:
        for tc in (m.get("tool_calls") or []):
            for v in (tc.get("arguments") or {}).values():
                if isinstance(v, str) and v in lims:
                    picked.append(v)
    gone = {g for g, (cap, _q) in lims.items() if int(tally.get(g, 0)) >= cap}
    print("실행된 값       : %s" % (sorted(set(picked)) or "(없음)"))
    print("그중 소진 그룹  : %s" % (sorted(set(picked) & gone) or "(없음)"))

    print("\n판정: %s" % ("문장은 만들어진다 ⇒ 침묵의 원인은 **발화 경로**(가설ⓑ)"
                          if out else "문장이 만들어지지 않는다 ⇒ 가설ⓐ"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
