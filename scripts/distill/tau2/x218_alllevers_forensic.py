# -*- coding: utf-8 -*-
r"""x218 — `bank_alllevers_20260810` **실패 6건 전수 부검** (유료 0 · 궤적 정독).

집계에서 결론으로 직행하지 않는다([[08]]). 실패마다 **어디서 갈렸는지**를 궤적에서 찾는다 —

  ⒜ gold 액션별 충족/미충족과 **실제로 무엇을 했는지**
  ⒝ 우리 기구가 그 sim 에서 **무엇을 발화했는지**(`T2_DIAG`·`T2_KIND`·`T2_REDERIVE`·피연산자)
  ⒞ 결정 블록·진단 문장이 **실제로 나갔는지**(사이드카 채널)
  ⒟ 에이전트의 마지막 두 발화(무엇을 근거로 답했는지)

실행 (리모트): python x218_alllevers_forensic.py [tag]
"""
import collections
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOGS = os.environ.get("T2_LOGS", "/home/woori/scratch/logs")
SIMS = os.environ.get("T2_SIMS", "/home/woori/scratch/tau2-bench/data/simulations")
MARKS = ("T2_DIAG", "T2_KIND", "T2_REDERIVE", "T2_OBJ_AXIS", "T2_LEDGER] value",
         "queued to view", "transcription returned 0", "T2_DEMANDED_STEP")
FB = (("결정블록", "A separate check was run on the policy constants"),
      ("진단", "A separate check was run on the records and the policy definitions"),
      ("상태별세기", "grouped by the status each record carries"),
      ("창산수", "Date arithmetic on the records above"),
      ("상태정의", "policy document that defines these status values"),
      ("통과표", "Policy constants on record"))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_alllevers_20260810"
    res = os.path.join(SIMS, tag, "results.json")
    d = json.load(open(res, encoding="utf-8"))

    # 마크는 sim 단위 합계로만 나온다(시행 번호가 없음) — 그 한계를 인쇄한다
    marks = collections.defaultdict(collections.Counter)
    tr = os.path.join(LOGS, "trace_%s.jsonl" % tag)
    if os.path.exists(tr):
        for ln in open(tr, encoding="utf-8", errors="replace"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            l = str(o.get("line") or "")
            for m in MARKS:
                if m in l:
                    marks[o.get("sim")][l[:150]] += 1
    fb = collections.defaultdict(collections.Counter)
    fbp = os.path.join(LOGS, "fb_%s.jsonl" % tag)
    if os.path.exists(fbp):
        for ln in open(fbp, encoding="utf-8", errors="replace"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            t = str(o.get("text") or o.get("body") or "")
            for name, sig in FB:
                if sig in t:
                    fb[o.get("sim") or o.get("task") or "?"][name] += 1

    print("=" * 100)
    print("%s — 실패 전수 부검" % tag)
    print("=" * 100)
    for s in d["simulations"]:
        ri = s.get("reward_info") or {}
        if ri.get("reward") == 1:
            continue
        t = s["task_id"]
        print("\n" + "-" * 100)
        print("%s trial=%s  %s" % (t, s.get("trial"), s.get("termination_reason")))
        for c in (ri.get("action_checks") or []):
            a = c.get("action") or {}
            print("   gold %-28s met=%-5s %s"
                  % (a.get("name"), c.get("action_match"),
                     json.dumps(a.get("arguments"), ensure_ascii=False)[:80]))
        calls = []
        for m in s["messages"]:
            for tc in (m.get("tool_calls") or []):
                f = tc.get("function") or tc
                ar = f.get("arguments")
                ar = ar if isinstance(ar, str) else json.dumps(ar, ensure_ascii=False)
                calls.append("%s(%s)" % (f.get("name"), ar[:60]))
        print("   호출 %d개, 마지막 6: %s" % (len(calls), calls[-6:]))
        last = [" ".join(str(m.get("content") or "").split()) for m in s["messages"]
                if m.get("role") == "assistant" and (m.get("content") or "").strip()]
        for x in last[-2:]:
            print("   말미: %s" % x[:260])
        print("   기구(태스크 합계): %s"
              % dict(collections.Counter(
                  re.sub(r"\[|\]", "", k.split("]")[0]) for k in marks.get(t, {}))))
        print("   사이드카(태스크 합계): %s" % dict(fb.get(t, {})))
    print("\n※ 마크·사이드카는 **시행 번호가 없어 태스크 합계**다 — 시행별 귀속은 못 한다.")
    print("※ 무엇이 갈렸는지는 위 gold 충족/미충족 + 실제 호출 + 말미로 판단한다([[08]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
