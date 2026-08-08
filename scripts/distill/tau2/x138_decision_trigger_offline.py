# -*- coding: utf-8 -*-
"""x138 — `FACT_AT_DECISION` **단계 0**: 트리거·해결·판정을 궤적 위에서 재생한다 (유료 0·LLM 0).

정본 = `FACT_AT_DECISION_DESIGN_2026_08_08.md` §6 단계 0.
게이트 = **발화 지점 목록 + 이름 불일치 건수 + 과차단 후보 0**.

무엇을 하나:
  1. 선언된 행동(`submit_referral`)의 호출을 전수 수집하고 `operand_arg`(`account_type`) 값을 뽑는다.
     — **구조화된 인자**라 산문 해석이 아니다([[59]] 대상 아님).
  2. 그 값이 `doc_minimums`의 키와 **문자열로 정확히 같은지** 본다(§3: 정확 일치만 판정·아니면 침묵).
     `doc_minimums`는 x135 오라클(항목별 질의 2패스 합집합)을 쓴다 — **이 프로브는 LLM을 부르지 않는다.**
  3. 결과를 셋으로 가른다: **판정 가능**(정확 일치) / **이름 불일치 침묵** / **오라클 없음**(그 sim 미측정).

⚠이 프로브는 tenure(경과일)를 계산하지 않는다 — 단계 0의 게이트가 묻는 것은 *"트리거가 어디서
몇 번 뜨고, 이름이 얼마나 맞는가"* 이고, 그 둘은 LLM 없이 답할 수 있다. 실제 미달 판정은 단계 1이다.

usage: x138_decision_trigger_offline.py [--tool submit_referral] [--arg account_type]
"""

import argparse
import collections
import glob
import gzip
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
SIMS = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
SHORT = {"bank_stack_dp_20260808p": "dp", "bank_stack_lim_20260808n": "lim"}


def oracle_by_sim(axis="threshold"):
    """x135 결과 → `{(short, task, idx): {키: 수}}`. 없으면 그 sim은 '오라클 없음'."""
    p = os.path.join(SIMS, "x135_excerpt_arms_%s.json.gz" % axis)
    out = {}
    if not os.path.exists(p):
        return out
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    for rec in d["sims"]:
        short, task, idx = rec["spec"].split(":")
        m = {}
        for pair in rec["oracle"]["pairs"]:
            k, v = pair.rsplit("=", 1)
            m[k] = int(v)
        out[(short, task, int(idx))] = m
    return out


def triggers(sim, tool, arg):
    """그 sim에서 선언 행동의 호출 전부 → `[값, …]`(순서 보존)."""
    out = []
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            a = a or {}
            nm = str(tc.get("name") or "")
            eff = a.get("agent_tool_name") if (nm.startswith("call_") and a.get("agent_tool_name")) else nm
            if eff == tool and arg in a:
                out.append(str(a[arg]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="submit_referral")
    ap.add_argument("--arg", default="account_type")
    a = ap.parse_args()

    orc = oracle_by_sim()
    n_trig = n_judge = n_mismatch = n_nooracle = 0
    mismatch_pairs = collections.Counter()
    judged = collections.Counter()

    for p in sorted(glob.glob(os.path.join(SIMS, "bank_stack_*2026080[78]*.json.gz"))):
        dirname = os.path.basename(p).replace(".json.gz", "")
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        rows = []
        for i, sim in enumerate(d.get("simulations") or []):
            task = sim.get("task_id")
            vals = triggers(sim, a.tool, a.arg)
            if not vals:
                continue
            key = (SHORT.get(dirname, "?"), task, i)
            keys = orc.get(key)
            for v in vals:
                n_trig += 1
                if keys is None:
                    n_nooracle += 1
                    rows.append((task, i, v, "오라클 없음"))
                elif v in keys:
                    n_judge += 1
                    judged[v] += 1
                    rows.append((task, i, v, "판정 가능 (문턱 %d)" % keys[v]))
                else:
                    n_mismatch += 1
                    near = [k for k in keys if k.split()[0] == v.split()[0]]
                    mismatch_pairs[(v, near[0] if near else "(유사 키 없음)")] += 1
                    rows.append((task, i, v, "이름 불일치 → 침묵 (오라클: %s)"
                                 % (near[0] if near else "해당 상품 없음")))
        if rows:
            print("\n== %s" % dirname)
            for t, i, v, verdict in rows:
                print("   %-9s sim%d  %-38s %s" % (t, i, v, verdict))

    print("\n" + "=" * 96)
    print("트리거 **%d건** · 판정 가능 %d · **이름 불일치 침묵 %d** · 오라클 없음 %d"
          % (n_trig, n_judge, n_mismatch, n_nooracle))
    if mismatch_pairs:
        print("\n★이름 불일치 짝(인자 값 ↔ 오라클 키):")
        for (v, k), c in mismatch_pairs.most_common():
            print("   %2d×  %-38s ↔ %s" % (c, v, k))
    cov = 100.0 * n_judge / (n_judge + n_mismatch) if (n_judge + n_mismatch) else 0
    print("\n오라클이 있는 트리거 중 판정 가능 비율 = **%.0f%%**" % cov)
    print("⚠단계 0 게이트는 여기까지다 — 실제 미달 판정(tenure 대조)은 단계 1이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
