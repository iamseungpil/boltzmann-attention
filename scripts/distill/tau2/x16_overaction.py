# -*- coding: utf-8 -*-
"""X16 — over-action 계측 O1·O2·O3 + Z7 오프라인 검증 (2026-07-31·무료).

설계 = `DECLFIRST_LIVE_WIRING_DESIGN_2026_07_31.md` §4-b·§5-Z7.

★왜 필요한가(Y1 포렌식): Y1 32태스크 중 **24개가 DB-basis**이고 `db_match`는 **최종 DB 전체 해시**라
**정답지에 없는 write를 벌한다**. flip 9건 중 **3건(020·023·027)은 채점된 action이 두 trial에서
동일한데 DB만 갈렸다** — 원인 후보가 **여분 행동**이다. 그래서 arm③이 pass를 올릴 때 그 경로가
**과행동 억제**인지 운인지 가르려면 이 계측이 **먼저** 있어야 한다.

지표(전부 닫힌 술어·**gold 미참조**):
  O1  실효-write 호출 수 / 런        (`t2_gate_patch._is_effective_write` 재사용·A2 전달)
  O2  미선언 write 비율               (봉투 필요 → Y1 궤적엔 봉투가 없으므로 **N/A**로 보고)
  O3  중복·재실행 write 수            (같은 실효-이름 + 같은 인자의 2회차 이상)

★Z7의 진짜 시험은 ③이다: **flip 3건에서 PASS/fail trial의 O1·O3 차이가 실제로 보이는가.**
안 보이면 "이 계측으로는 그 flip을 설명 못 한다"는 뜻이고, 그건 **한계로 기록**한다([[08]]).

용법: py -3 x16_overaction.py <results.json> [--z7]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as GP                                            # noqa: E402

_A2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "banking_knowledge.gate.json")


def load_a2():
    with open(_A2, encoding="utf-8") as f:
        return json.load(f)


def eff_name(name, args):
    """dispatcher 호출이면 **내부 도구명**이 실효 이름이다(외피는 절차 도구)."""
    if isinstance(args, dict):
        for k in ("agent_tool_name", "discoverable_tool_name"):
            if args.get(k):
                return str(args[k])
    return name


def sim_metrics(sim, a2):
    """한 sim의 O1·O3 (+ write 목록)."""
    seen = set()
    writes, dupes = [], []
    for m in sim.get("messages") or []:
        if m.get("role") not in ("assistant", "user"):
            continue
        for tc in m.get("tool_calls") or []:
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            a = a if isinstance(a, dict) else {}
            en = eff_name(tc.get("name"), a)
            if not GP._is_effective_write(en, a2):
                continue
            key = (en, json.dumps(a, sort_keys=True, ensure_ascii=False)[:400])
            writes.append(en)
            if key in seen:
                dupes.append(en)
            seen.add(key)
    return {"O1": len(writes), "O3": len(dupes),
            "writes": Counter(writes), "dupes": Counter(dupes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--z7", action="store_true", help="flip 3건 대조까지 수행")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    a2 = load_a2()
    with open(args.results, encoding="utf-8") as f:
        d = json.load(f)
    by = defaultdict(dict)
    for s in d.get("simulations") or []:
        by[s.get("task_id")][s.get("trial")] = s

    print("=== Z7-① O1 정의 점검 — 무엇이 write로 세어지나 ===")
    allw = Counter()
    for tr in by.values():
        for s in tr.values():
            allw.update(sim_metrics(s, a2)["writes"])
    print("  write로 계상된 실효-이름 상위:")
    for n, c in allw.most_common(12):
        print("     %-42s %d" % (n, c))
    print("  ⚠`give_…`·`unlock_…`·`KB_search` 류가 여기 있으면 술어가 잘못된 것이다(절차·읽기).")

    print("\n=== Z7-② O3 중복 판정 표본 ===")
    ex = 0
    for t, tr in sorted(by.items()):
        for k, s in sorted(tr.items()):
            mm = sim_metrics(s, a2)
            if mm["O3"] and ex < 5:
                print("  %s trial %s: O1=%d O3=%d · 중복 %s"
                      % (t, k, mm["O1"], mm["O3"], dict(mm["dupes"])))
                ex += 1

    print("\n=== 전 태스크 O1/O3 (trial 0 → trial 1) ===")
    print("%-10s %-6s %-14s %-14s" % ("task", "basis", "trial0 O1/O3", "trial1 O1/O3"))
    for t, tr in sorted(by.items()):
        ks = sorted(tr)
        cells = []
        for k in ks:
            mm = sim_metrics(tr[k], a2)
            rw = 1 if ((tr[k].get("reward_info") or {}).get("reward") or 0) >= 1 else 0
            cells.append("%d/%d %s" % (mm["O1"], mm["O3"], "PASS" if rw else "fail"))
        b = "+".join((tr[ks[0]].get("reward_info") or {}).get("reward_basis") or [])
        print("%-10s %-6s %-14s %-14s" % (t, b, cells[0], cells[1] if len(cells) > 1 else "-"))

    if args.z7:
        print("\n" + "=" * 74)
        print("★Z7-③ 진짜 시험 — action은 같은데 DB만 갈린 flip 3건에서 O1·O3가 보이나")
        print("=" * 74)
        seen_diff = 0
        for t in ("task_020", "task_023", "task_027"):
            tr = by.get(t) or {}
            if len(tr) < 2:
                continue
            rows = []
            for k in sorted(tr):
                s = tr[k]
                mm = sim_metrics(s, a2)
                rw = 1 if ((s.get("reward_info") or {}).get("reward") or 0) >= 1 else 0
                rows.append((k, rw, mm))
            p = [r for r in rows if r[1] == 1][0]
            f = [r for r in rows if r[1] == 0][0]
            dO1, dO3 = f[2]["O1"] - p[2]["O1"], f[2]["O3"] - p[2]["O3"]
            visible = (dO1 != 0 or dO3 != 0)
            seen_diff += visible
            print("\n  %s  PASS(trial %s) O1=%d O3=%d   vs   fail(trial %s) O1=%d O3=%d"
                  % (t, p[0], p[2]["O1"], p[2]["O3"], f[0], f[2]["O1"], f[2]["O3"]))
            print("     ΔO1=%+d · ΔO3=%+d → %s" % (dO1, dO3, "**차이 보임**" if visible
                                                   else "차이 없음(이 계측으로 설명 불가)"))
            only_f = f[2]["writes"] - p[2]["writes"]
            only_p = p[2]["writes"] - f[2]["writes"]
            if only_f:
                print("     fail에만 있는 write: %s" % dict(only_f))
            if only_p:
                print("     PASS에만 있는 write: %s" % dict(only_p))
        print("\n  ⇒ Z7-③: 3건 중 **%d건**에서 O1/O3 차이가 보인다." % seen_diff)
        print("     보이지 않는 건은 **DB 해시의 다른 축**(등록 경로·값 미세차)이므로 이 계측의")
        print("     한계로 기록한다 — 설계서 §4-b 말미 경고와 일치.")


if __name__ == "__main__":
    main()
