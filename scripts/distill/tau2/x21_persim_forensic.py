# -*- coding: utf-8 -*-
"""X21 — **완료된 sim 단위** per-step 포렌식 (2026-07-31·무료).

`y1_forensic.py`는 **두 trial이 다 끝난 태스크**만 본다(항상-실패 / flip 판정이 목적). 진행 중인
런에는 완주 태스크가 없어 아무것도 못 낸다. 이 도구는 **완료된 sim 각각**을 본다 — 중간 포렌식용.

채점 규약은 **추측하지 않는다**: C245에서 소스 직독으로 축자 재구현하고 저장된 `action_match`와
1080/1080 일치로 검증된 `x12_action_fail_exact`의 `preds`/`classify`를 **그대로 재사용**한다([[03b]]).

★[[08]] 규율 — 이 도구는 **분류**를 낸다. "왜 그 도구를 안 불렀나"는 분류가 아니라 궤적 정독으로만
  말할 수 있다. 그리고 진행 중 런의 완료분은 **빨리 끝난 sim에 치우쳐 있다**(완료 순서 편향).

산출:
  ① sim별: reward · db_match · 종료사유 · gold action별 실패 분류
  ② 태스크별 집계(같은 태스크의 여러 trial이 **같은 방식으로** 실패하는가 = 안정적 결손인가)
  ③ 실패 분류 분포 + 종료사유 교차표

용법: py -3 x21_persim_forensic.py <results.json> [<results.json> ...] [--task task_007]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x12_action_fail_exact as X12  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sims_of(d):
    return d.get("simulations") or []


def gold_actions(sim):
    """gold action 목록 — 저장된 reward_info에서 읽는다(우리가 만들지 않는다)."""
    ri = sim.get("reward_info") or {}
    out = []
    for chk in (ri.get("action_checks") or []):
        a = chk.get("action") or {}
        out.append({"name": a.get("name"), "arguments": a.get("arguments") or {},
                    "compare_args": a.get("compare_args"),
                    "matched": bool(chk.get("action_match"))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--task", default="")
    ap.add_argument("--args", action="store_true",
                    help="★인자 수준 대조 — 분류(TOP_VALUE 등)만으로는 원인을 말할 수 없다. "
                         "gold 값 vs 모델이 실제로 넣은 값을 나란히 찍는다.")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = []
    for p in args.results:
        if not os.path.exists(p):
            print("  ⚠없음: %s" % p)
            continue
        for s in sims_of(load(p)):
            tid = s.get("task_id")
            if args.task and tid != args.task:
                continue
            ri = s.get("reward_info") or {}
            msgs = s.get("messages") or []
            plist = X12.preds(msgs)
            fails = []
            for g in gold_actions(s):
                if g["matched"]:
                    continue
                cls, detail = X12.classify(g, plist)
                same = [a for n, a in plist if n == g["name"]]
                fails.append((cls, g["name"], detail, g["arguments"], same))
            rows.append({
                "task": tid, "trial": s.get("trial"),
                "reward": ri.get("reward"), "db": (ri.get("db_check") or {}).get("db_match"),
                "term": (s.get("termination_reason") or "?"),
                "turns": len(msgs), "calls": len(plist), "fails": fails,
                "n_gold": len(gold_actions(s)),
            })

    if not rows:
        print("완료 sim 0건 — 아직 수확할 것이 없다.")
        return

    print("=" * 100)
    print("① sim별 (완료 %d건) — ⚠진행 중 런이면 **빨리 끝난 sim 편향**([[08]])" % len(rows))
    print("=" * 100)
    print("  %-10s %-6s %-7s %-6s %-24s %-6s %s"
          % ("task", "trial", "reward", "db", "종료사유", "호출", "실패 분류"))
    for r in sorted(rows, key=lambda x: (str(x["task"]), str(x["trial"]))):
        cls = ", ".join("%s(%s)" % (f[0], f[1]) for f in r["fails"][:3]) or "-"
        if len(r["fails"]) > 3:
            cls += " …+%d" % (len(r["fails"]) - 3)
        print("  %-10s %-6s %-7s %-6s %-24s %-6s %s"
              % (r["task"], r["trial"], r["reward"], r["db"], str(r["term"])[:24], r["calls"], cls))

    if args.args:
        print("\n" + "=" * 100)
        print("①-b ★인자 수준 대조 — gold가 요구한 값 vs 모델이 실제로 넣은 값")
        print("=" * 100)
        for r in sorted(rows, key=lambda x: (str(x["task"]), str(x["trial"]))):
            if not r["fails"]:
                continue
            print("\n  [%s trial %s]" % (r["task"], r["trial"]))
            for cls, name, det, gargs, same in r["fails"]:
                print("    %-16s %s" % (cls, name))
                # 분류기가 준 차이-종류(x12 `sub_kind`)를 그대로 — 재구현하지 않는다([[03b]])
                if isinstance(det, dict):
                    print("        차이: %s" % json.dumps(det, ensure_ascii=False)[:200])
                print("        gold: %s" % json.dumps(gargs, ensure_ascii=False)[:200])
                for i, pa in enumerate(same[:2]):
                    print("        pred[%d]: %s" % (i, json.dumps(pa, ensure_ascii=False)[:200]))
                if not same:
                    print("        pred: **그 도구를 한 번도 부르지 않았다**")

    print("\n" + "=" * 100)
    print("② 태스크별 — 같은 태스크의 여러 trial이 **같은 방식으로** 실패하는가")
    print("=" * 100)
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for t in sorted(by):
        rs = by[t]
        sigs = {tuple(sorted(f[0] for f in r["fails"])) for r in rs}
        rw = [r["reward"] for r in rs]
        verdict = ("trial %d개 · reward %s · " % (len(rs), rw)
                   + ("**분류 동일**(안정적 결손)" if len(sigs) == 1 and len(rs) > 1
                      else ("**분류 상이**(원인이 흔들림)" if len(rs) > 1 else "trial 1건(판정 보류)")))
        print("  %-10s %s" % (t, verdict))

    print("\n" + "=" * 100)
    print("③ 실패 분류 분포 · 종료사유 교차표")
    print("=" * 100)
    cc = Counter(f[0] for r in rows for f in r["fails"])
    print("  실패 분류:", cc.most_common() or "(없음)")
    tt = Counter((r["term"], "pass" if (r["reward"] or 0) >= 1 else "fail") for r in rows)
    for k, n in sorted(tt.items()):
        print("    %-26s %-5s %d" % (k[0], k[1], n))
    dbm = Counter((r["db"], "pass" if (r["reward"] or 0) >= 1 else "fail") for r in rows)
    print("  db_match × 결과:", dict(dbm))


if __name__ == "__main__":
    main()
