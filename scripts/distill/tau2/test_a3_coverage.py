# -*- coding: utf-8 -*-
r"""회귀 — A3 **커버리지**가 필터를 살린다 (C385·C387 · GPU 0 · 순수함수).

`test_objective_sum.py` 를 대체한다. 그 파일은 C384 에서 **철회된 기구**(`combine_axes`·
`formalize_objective_axes`·`verified_subjects`·`rederive_by_axis`)를 부르다 import 시점에
죽어 있었다. 살아남은 검정 두 개(표시 축의 조회 가능성·단일 축 문구)만 여기로 옮기고,
098 을 실제로 가르는 자리 — **A3 가 정책 문서의 값을 갖고 있는가** — 를 못박는다.

무엇을 막는가 —
 ⒜ **조용한 되돌림**: `Gold Years` 의 예치 문턱($1,000)이 A3 에서 다시 사라지는 것. 그 값이
    없으면 예치 $600 손님에게 통과 표가 `Gold Years`(합산 125)를 남기고, 그게 정확히 098 의
    라이브 오답이었다(C385: `A_iso` 0/8 ↔ `A_ver` 8/8).
 ⒝ **날조**: 카드 주어에 예치 문턱이 생기는 것. 카드는 `qualifying_spend_usd` 를 갖고 예치
    개념이 **없다** — 빈칸이 아니라 없는 것이다([[25]]·[[62]]).
 ⒞ **출처 없는 값**: 새 행이 `source.doc` + 축자 인용 없이 들어오는 것([[23]]).
 ⒟ **조용히 죽는 순위**: 표에 보여 주는 축을 A3 에서 조회 못 하는 것.

⚠이 파일은 **엔진을 시험하지 않는다** — 엔진은 이번에 한 줄도 바뀌지 않았다. 시험 대상은
  A2/A3 데이터([[05]] 가변부)와 그것이 기존 필터에 미치는 효과다.

실행: py -3 test_a3_coverage.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
import t2_factdag as FD                                         # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

FAILED = []


def chk(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + (("  — " + str(extra)) if extra else ""))
    if not cond:
        FAILED.append(label)


def main():
    a2 = load_domain_a2("banking_knowledge") or {}
    rows = (a2.get("policy_ontology") or {}).get("rows") or []
    spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get("eligible_text")), None)
    chk(spec is not None and bool(rows), "A2/A3 가 실린다 (%d행)" % len(rows))
    if spec is None:
        return 1
    cfg = spec.get("eligible") or {}
    show = list(cfg.get("show_axes") or ())
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in show}
    dep_ax = next((c["axis"] for c in (cfg.get("criteria") or ())
                   if c.get("operand") == "stated" and c.get("compare") == "ge"), None)

    print("\n§1 표에 보여 주는 축은 전부 조회 가능하다 (못 하면 순위가 조용히 죽는다)")
    resolvable = {(n.get("params") or {}).get("axis")
                  for n in (a2.get("derived") or ()) if n.get("op") == "a3_map"}
    missing = [ax for ax in show if ax not in resolvable]
    chk(not missing, "show_axes ⊆ a3_map 노드", missing or "빠짐 없음")
    chk("ONE of the names" in (spec.get("objective_axis_prompt") or ""),
        "단일 축 문구가 그대로다 (099/100 3/3 을 세운 구성)")

    print("\n§2 정책 문서가 말하는 예치 문턱이 A3 에 있다 (C387·빌더 누락 복구)")
    chk(dep_ax == "qualifying_deposit_usd", "≥ 방향 대화-피연산자 축", dep_ax)
    dep = maps.get(dep_ax) or {}
    for subj, val in (("Gold Years", 1000), ("Light Blue", 500), ("Light Green", 100),
                      ("Blue", 500), ("Green Fee-Free", 300)):
        got = LG._num(dep[subj]) if subj in dep else None
        chk(got == val, "%-16s 예치 문턱 %s" % (subj, val), got)
    for r in rows:
        if r.get("axis") == dep_ax:
            src = r.get("source") or {}
            chk(bool(src.get("doc")) and bool(src.get("quote")),
                "%-16s 행이 출처+축자 인용을 들고 있다" % r.get("subject"))

    print("\n§3 ⛔카드에는 예치 문턱이 없다 (있으면 날조다)")
    spend = {r.get("subject") for r in rows if r.get("axis") == "qualifying_spend_usd"}
    chk(bool(spend), "지출 문턱을 갖는 주어가 존재한다 (%d)" % len(spend))
    bad = sorted(spend & set(dep))
    chk(not bad, "지출 축을 갖는 주어에 예치 축이 붙지 않았다", bad or "없음")

    print("\n§4 098 의 자리 — 예치 $600 손님에게 통과 표가 무엇을 남기나")
    tbl = LG.eligible_text(400, {}, maps, spec, {dep_ax: 600}) or ""
    named = {l.strip().split(":")[0].strip() for l in tbl.splitlines() if l.startswith("  ")}
    chk("Blue" in named, "gold 인 `Blue` 는 남는다 (문턱 500 ≤ 600)")
    for s in ("Gold Years", "Purple", "Dark Green", "Bluest"):
        chk(s not in named, "%-12s 는 빠진다 (문턱 > 600)" % s)
    # 표 안에서 합산 최댓값이 **하나**이고 그것이 gold 인지 — 우리가 argmax 를 하지 않으므로
    # 이것은 엔진의 출력이 아니라 **표의 성질**에 대한 진술이다([[62]]).
    m1, m2 = maps.get("referrer_bonus_usd") or {}, maps.get("referred_bonus_usd") or {}
    tot = {s: LG._num(m1[s]) + LG._num(m2[s]) for s in named if s in m1 and s in m2}
    chk(bool(tot), "표에 두 보너스가 다 있는 주어가 있다 (%d)" % len(tot))
    if tot:
        best = max(tot.values())
        top = sorted(s for s, v in tot.items() if v == best)
        chk(top == ["Blue"], "합산 최댓값 주어가 `Blue` 하나다", "%s = %s" % (top, best))

    print("\n§5 음성 통제 — 채우기 전 상태였다면 이 검정이 실패해야 한다")
    dep_old = {k: v for k, v in dep.items() if k not in ("Gold Years", "Light Blue", "Light Green")}
    maps_old = dict(maps, **{dep_ax: dep_old})
    old = LG.eligible_text(400, {}, maps_old, spec, {dep_ax: 600}) or ""
    old_named = {l.strip().split(":")[0].strip() for l in old.splitlines() if l.startswith("  ")}
    chk("Gold Years" in old_named, "구판에서는 `Gold Years` 가 표에 남았다 (=라이브 오답의 원인)")
    tot_old = {s: LG._num(m1[s]) + LG._num(m2[s]) for s in old_named if s in m1 and s in m2}
    chk(tot_old and max(tot_old, key=tot_old.get) != "Blue",
        "구판 표의 합산 최댓값은 gold 가 아니었다", max(tot_old, key=tot_old.get) if tot_old else "-")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
