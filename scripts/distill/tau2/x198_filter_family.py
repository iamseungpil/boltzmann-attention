# -*- coding: utf-8 -*-
r"""x198 — **필터가 레버일 수 있는 태스크**를 `tasks.json` 구조 서명으로 고른다 (유료 0).

## 무엇을 고르나

098/099/100 의 공통 모양은 하나다: **gold 액션의 인자 값이 정책 표의 주어(A3 subject)** 다.
즉 *"후보 중 하나를 고르라"* 는 태스크다. 그 계열에서만 *"자격 없는 후보를 빼 주면 모델이
맞힌다"*(x197 `A_ver` 8/8 · x150 `미달 행 뺀 표 5/5`)가 성립할 수 있다.

## ★이 스크립트가 **못** 하는 것 (넘어가면 안 되는 선)

`tasks.json` 은 *"모델이 필터에서 실패하는가"* 를 말하지 않는다. 그것은 **격리 프로브**로만
나온다(x197). 여기서 나오는 것은 **후보 목록**이고, 확정은 태스크마다 따로다.
또 하나: *"우리 필터가 걸릴 수 있는가"* 는 **A3 커버리지** 문제다 — 098 이 진 이유가 정확히
그것이었다(`Gold Years` 는 예치 문턱이 A3 에 없어 안 걸러졌다). 그래서 후보마다 **커버리지도
같이 센다**.

gold 는 **표적 선정·진단에만** 쓴다([[23]]·N97 §0 규율). A2/A3 에 넣지 않는다.

## 출력

  §1 후보 태스크 — gold 인자가 A3 주어인 것 (그 주어·인자 이름과 함께)
  §2 후보별 **A3 커버리지** — 그 계열 후보 전체가 기준 축 값을 갖고 있는가
     (안 갖고 있으면 우리 필터는 그 행을 **못 거른다** = 098 의 실패 형태)
  §3 이미 아는 것과의 대조 — 통과/실패 census 와 겹쳐 본다

실행: python x198_filter_family.py [--tasks <tasks.json>]
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from gate_interpreter import load_domain_a2                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
TASKS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
CENSUS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                      "bank_task_census.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    a = ap.parse_args()

    a2 = load_domain_a2("banking_knowledge") or {}
    po = a2.get("policy_ontology") or {}
    rows = po.get("rows") or []
    subjects = {str(r.get("subject", "")).strip() for r in rows if r.get("subject")}
    by_subj = collections.defaultdict(set)
    for r in rows:
        by_subj[str(r.get("subject", "")).strip()].add(r.get("axis"))
    sp = next((x for x in a2.get("ledger_metrics", []) if x.get("eligible_text")), {})
    crit = [c.get("axis") for c in ((sp.get("eligible") or {}).get("criteria") or ())]

    tasks = json.load(open(a.tasks, encoding="utf-8"))
    print("태스크 %d · A3 주어 %d · 자격 기준 축 %s" % (len(tasks), len(subjects), crit))

    # ── §1 gold 인자가 A3 주어인 태스크 ─────────────────────────────────────
    fam = []
    for t in tasks:
        acts = ((t.get("evaluation_criteria") or {}).get("actions") or [])
        hits = []
        for act in acts:
            for k, v in (act.get("arguments") or {}).items():
                if isinstance(v, str):
                    s0 = v.strip()
                    # 원장 표기(`… Account`)도 같은 주어를 가리킬 수 있다 — 접미사 차이는
                    # 여기서만(분석) 관용한다. 엔진은 이 정규화를 하지 않는다([[59]]).
                    for cand in (s0, s0.replace(" Account", "").strip()):
                        if cand in subjects:
                            hits.append((act.get("name"), k, cand))
                            break
        if hits:
            # ★서명이 과다 선택한다 (2026-08-09 표본 확인): 제품 이름이 gold 에 나온다고
            #   다 선택형이 아니다 — 048 은 카드 4장 **해지** 플로우인데 gold 24액션 중간에
            #   이름이 스칠 뿐이다. 002 는 gold 1액션짜리 순수 추천이고 098/099/100 과 같다.
            #   ⇒ **gold 액션 수**를 같이 싣고, 짧은 것만 선택형 후보로 표시한다.
            fam.append((t["id"], hits, len(acts)))

    print("\n§1 gold 인자가 **정책 표 주어**인 태스크 — %d개" % len(fam))
    for tid, hits, na in sorted(fam, key=lambda z: z[2]):
        seen = sorted({h[2] for h in hits})
        args = sorted({"%s.%s" % (h[0], h[1]) for h in hits})
        mark = "★선택형" if na <= 6 else "  (긴 플로우)"
        print("  %-9s gold%-3d %s %-40s %s" % (tid.replace("task_", ""), na, mark,
                                               ", ".join(args)[:40], ", ".join(seen)[:40]))

    # ── §2 후보 계열의 A3 커버리지 ──────────────────────────────────────────
    print("\n§2 자격 기준 축의 A3 커버리지 — **없는 행은 우리 필터가 못 거른다**(098 의 실패 형태)")
    for ax in crit:
        have = {s for s in subjects if ax in by_subj[s]}
        miss = sorted(subjects - have)
        print("  %-26s 있음 %3d / %3d   없음 %d%s"
              % (ax, len(have), len(subjects), len(miss),
                 ("  예: " + ", ".join(miss[:6])) if miss else ""))

    # ── §3 census 와 대조 ───────────────────────────────────────────────────
    try:
        cen = {r["id"]: r for r in json.load(open(CENSUS, encoding="utf-8"))}
    except Exception:
        cen = {}
    if cen:
        print("\n§3 census 대조 (전수 런 기준)")
        st = collections.Counter()
        for tid, _h, _na in fam:
            r = cen.get(tid) or {}
            n, p = r.get("n") or 0, r.get("pass") or 0
            st["전승" if (n and p == n) else ("전패" if n and p == 0 else "혼합/미측정")] += 1
        print("  후보 %d개 중: %s" % (len(fam), dict(st)))
        never = [t for t, _h, _na in fam if (cen.get(t) or {}).get("pass") == 0]
        print("  전패 후보: %s" % ", ".join(x.replace("task_", "") for x in never))

    print("\n※ 이것은 **후보 목록**이다. '필터가 레버'는 태스크마다 격리 프로브(x197)로 확정한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
