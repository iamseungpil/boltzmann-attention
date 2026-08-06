# -*- coding: utf-8 -*-
"""우리 판정 행 vs gold가 요구한 행 vs 실제 제출된 행 — 분쟁군 전수 대조.

020에서 확인된 것(2026-08-06): `get_reward_discrepancies`가 `txn_a8f1c2d3e404`를 불일치로
**적극 판정**했고(coverage 26/26·기권 0), 손님이 그 여분을 제출해 db_match가 깨졌다. 검산하면
그 행의 상점(Microsoft)은 정책의 **제외 목록**에 있어 표준 1.0%가 맞다 — 즉 여분은 모델의 과잉이
아니라 **우리 산식의 오판정**이다([[25]] 정본 오염).

한 건을 보고 계열 전체를 단정하면 안 되므로([[08]]) 전수로 센다. 세 집합을 태스크·trial마다 낸다:

  ENGINE  우리 도구 출력이 "recorded X, correct Y"로 지목한 거래 id
  GOLD    태스크 evaluation_criteria가 요구한 분쟁 거래 id
  DONE    원장에 실제로 제출된 분쟁 거래 id(손님·에이전트 양 채널)

  ENGINE − GOLD = 우리가 만든 여분(오판정 후보)   GOLD − ENGINE = 우리가 놓친 행
  GOLD − DONE   = 완결성 잔여(후보 K)             DONE − GOLD = 과제출

  usage:  x110_engine_row_audit.py [--tag 20260806]
"""

import collections
import glob
import gzip
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x109_task_dossier import load_sims, load_tasks, eff          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 우리 엔진 출력의 고정 포맷 전사(env 포맷 전사 계보 — 판단 0)
ENGINE_ROW = re.compile(r"(txn_[0-9a-zA-Z]+)\s*\(recorded\s+([0-9.,]+)\s*points?,\s*correct\s+([0-9.,]+)")
COVERAGE = re.compile(r"\[coverage\]\s*(\d+)\s+of\s+(\d+)\s+rows")
DISPUTE = "submit_cash_back_dispute"


def txn_of(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return None
    if not isinstance(a, dict):
        return None
    inner = a.get("arguments")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except Exception:
            inner = {}
    if isinstance(inner, dict) and inner.get("transaction_id"):
        return str(inner["transaction_id"])
    return str(a["transaction_id"]) if a.get("transaction_id") else None


def gold_rows(task):
    out = []
    for a in ((task or {}).get("evaluation_criteria") or {}).get("actions") or []:
        args = a.get("arguments") or {}
        blob = json.dumps(args, ensure_ascii=False)
        if DISPUTE not in blob:
            continue
        m = re.search(r"(txn_[0-9a-zA-Z]+)", blob)
        if m:
            out.append(m.group(1))
    return out


def main():
    sims = load_sims()
    tasks = load_tasks()
    rows = []
    for s in sims:
        eng, done, cov = [], [], []
        for m in s.get("messages") or []:
            if m.get("role") == "tool":
                c = str(m.get("content") or "")
                if "correct" in c and "recorded" in c:
                    eng += [g[0] for g in ENGINE_ROW.findall(c)]
                cov += COVERAGE.findall(c)
            for tc in (m.get("tool_calls") or []):
                if DISPUTE in str(eff(tc) or "") and m.get("role") in ("user", "assistant"):
                    t = txn_of(tc)
                    if t:
                        done.append(t)
        tsk = (tasks.get(s["task_id"]) or (None, None))[1]
        gold = gold_rows(tsk)
        if not (eng or gold):
            continue
        rows.append((s["task_id"], s.get("trial"), (s.get("reward_info") or {}).get("reward"),
                     set(eng), set(gold), set(done), cov, s["_src"]))

    print("== 분쟁군 전수 대조 (ENGINE=우리 판정 · GOLD=요구 · DONE=제출) ==")
    print("  대상 sim %d개\n" % len(rows))
    extra_tot = miss_tot = short_tot = over_tot = 0
    for tid, tr, rw, eng, gold, done, cov, src in sorted(rows):
        extra = eng - gold
        miss = gold - eng
        short = gold - done
        over = done - gold
        extra_tot += len(extra)
        miss_tot += len(miss)
        short_tot += len(short)
        over_tot += len(over)
        print("%-10s t%-2s reward=%-4s ENGINE %2d GOLD %2d DONE %2d | 여분 %-2d 누락 %-2d 미제출 %-2d 과제출 %-2d %s"
              % (tid, tr, rw, len(eng), len(gold), len(done), len(extra), len(miss),
                 len(short), len(over), "" if not cov else "cov=%s" % ",".join("%s/%s" % c for c in cov[:3])))
        if extra:
            print("      ⚠우리가 만든 여분: %s" % ", ".join(sorted(extra)))
        if miss:
            print("      ⚠우리가 놓친 행:   %s" % ", ".join(sorted(miss)))
    print("\n== 합계 ==")
    print("  여분(ENGINE−GOLD) %d · 누락(GOLD−ENGINE) %d · 미제출(GOLD−DONE) %d · 과제출(DONE−GOLD) %d"
          % (extra_tot, miss_tot, short_tot, over_tot))
    print("  ⚠여분·누락은 **우리 산식**의 문제이고, 미제출은 완결성(레버)의 문제다 — 처방이 다르다.")


if __name__ == "__main__":
    main()
