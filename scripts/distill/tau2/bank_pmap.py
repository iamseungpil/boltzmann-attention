# -*- coding: utf-8 -*-
r"""bank_pmap — 97 태스크를 **P1~P6 레버 축으로 기계 분류**(2026-08-16·사용자 지시).

## 무엇을 하고 무엇을 못 하나 (먼저 정직하게)

여기서 나오는 것은 **사전 지도(prior)** 이지 귀속이 아니다.

  · **할 수 있는 것** — 태스크 정의와 census 집계만으로 *어느 P 가 걸릴 법한가* 를 센다.
    근거는 셋뿐이고 전부 기계값이다: gold 액션의 **형태**(반복/개수) · **문서 의존**(kb·n_docs) ·
    **실행 여부**(`fail_wrote`).
  · **할 수 없는 것** — *어느 P 가 실제 병목인가* 는 **궤적**이 있어야 한다. 오늘 실물로 두 번
    당했다: 069 는 자격 축인 줄 알았으나 **벤치 결함**이었고, 070 은 유효창 축인 줄 알았으나
    다른 축에서 이미 탈락이었다. ⇒ **지도는 표적 후보를 좁히는 데만** 쓰고, 표적으로 확정하기
    전에 궤적·문서 대조를 반드시 한다([[62]]·[[23]]).

## 규칙 (전부 닫힌 술어 · 도메인 판단 0)

    P5 completion   같은 gold 도구를 **3회 이상** 반복 호출  → 열거 완결(coverage)
    P4 emission     `fail_wrote == 0`(한 번도 write 안 함)   → 발견/끝맺음 = 학습 축
    P1 delivery     `kb == True` 이고 `n_docs >= 8`          → 문서를 읽어야 답이 정해진다
    P3 verification 인자에 **id/금액**이 실린 write 가 있다   → 형식·출처 검증
    P6 compute      금액·요율·이자 낱말이 시나리오에 있다     → 계산형 기준
    P2 removal      유효창(프로모션) 문서군에 걸린다          → 만료 제거

⚠한 태스크가 **여러 P** 를 받는다(실패는 층으로 온다·per-step 85.2%가 다층·C491).
⚠`P6/P2` 는 시나리오 낱말에 기대므로 가장 약하다 — 후보로만 읽어라.
"""
import collections
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                      "bank_task_taxonomy_20260810.json")
MONEY = re.compile(r"\b(APY|cash ?back|interest|rebate|fee|reward|%|\$)", re.I)
WINDOW = re.compile(r"promo|promotion|expire|valid|active from", re.I)


def classify(t, scen=""):
    ps = []
    acts = t.get("acts") or []
    rep = collections.Counter(acts)
    if rep and max(rep.values()) >= 3:
        ps.append("P5")
    if t.get("n", 0) and t.get("fail_wrote", 0) == 0 and t.get("rate", 0) == 0:
        ps.append("P4")
    if t.get("kb") and t.get("n_docs", 0) >= 8:
        ps.append("P1")
    if any(a in ("call_discoverable_agent_tool", "call_discoverable_user_tool",
                 "apply_for_credit_card", "submit_transaction") for a in acts):
        ps.append("P3")
    if MONEY.search(scen):
        ps.append("P6")
    if WINDOW.search(scen):
        ps.append("P2")
    return ps or ["미분류"]


def main():
    d = json.load(io.open(CENSUS, encoding="utf-8"))
    scen = {}
    p = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
    if os.path.exists(p):
        raw = json.load(io.open(p, encoding="utf-8"))
        for t in (raw if isinstance(raw, list) else raw.get("tasks", [])):
            scen[str(t.get("id"))] = json.dumps(t.get("user_scenario") or {}, ensure_ascii=False)

    rows, cnt, multi = [], collections.Counter(), collections.Counter()
    for t in d:
        ps = classify(t, scen.get(t["id"], ""))
        rows.append((t["id"], t["rate"], t["n"], t.get("fail_wrote", 0), ps))
        for x in ps:
            cnt[x] += 1
        multi[len(ps)] += 1

    print("97 태스크 P-지도 (사전 지도·귀속 아님)\n")
    print("축별 태스크 수:", dict(cnt.most_common()))
    print("한 태스크가 받은 축 개수 분포:", dict(sorted(multi.items())))
    print("시나리오 텍스트 %d/97 확보%s\n" % (len(scen), "" if scen else " (원격에서 실행하면 P6/P2 가 채워진다)"))

    print("== 0% 태스크의 축 조합 (표적 선정용) ==")
    combo = collections.Counter("+".join(ps) for i, r, n, w, ps in rows if r == 0)
    for c, k in combo.most_common(12):
        print("  %-22s %d" % (c, k))

    print("\n== 표적 2·3 확인 ==")
    for tid in ("task_019", "task_020", "task_022", "task_027", "task_028", "task_029", "task_024"):
        for i, r, n, w, ps in rows:
            if i == tid:
                print("  %s rate=%.0f%% wrote=%d/%d → %s" % (i, 100 * r, w, n, "+".join(ps)))

    print("\n== 축별 명부(0% 만) ==")
    for ax in ("P5", "P4", "P1", "P6", "P2", "P3"):
        ids = [i.replace("task_", "") for i, r, n, w, ps in rows if r == 0 and ax in ps]
        print("  %-3s %2d: %s" % (ax, len(ids), " ".join(ids)))


if __name__ == "__main__":
    main()
