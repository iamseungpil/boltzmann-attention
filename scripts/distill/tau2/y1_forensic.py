# -*- coding: utf-8 -*-
"""Y1 전수 per-step 포렌식 — 항상-실패 17건의 원인 · flip 9건의 성패 갈림 지점 (2026-07-31).

사용자 지시: "fail 17건과 flip 9건 모두 전수 정밀 per-step 포렌식 · 실패 원인을 밝히고
flip은 성공과 실패의 원인도 정확히 밝혀라".

채점 규약은 **추측하지 않는다** — C245에서 소스 직독으로 축자 재구현한 `x12_action_fail_exact`의
`preds`/`matches`/`classify`를 **그대로 재사용**한다(그 재구현은 저장된 `action_match`와
1080/1080 일치로 검증됐다).

산출:
  ① 항상-실패(두 trial 다 fail) — gold action별 실패 분류 + **두 trial의 분류가 같은가**
     (같으면 **안정적 결손** = 진짜 표적 / 다르면 실패 원인 자체가 흔들림)
  ② flip — **어느 gold action이 갈렸는가** + 통과 trial에서 그것을 어떻게 충족했고
     실패 trial에서 무엇을 했는가(호출 유무·인자 차이) + 두 trial의 행동 요약 대조

용법: py -3 y1_forensic.py <results.json> [--task task_007]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x12_action_fail_exact as X12                                   # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sim_actions(sim):
    """gold action별 (매치 여부, 분류, 상세)."""
    ri = sim.get("reward_info") or {}
    plist = X12.preds(sim.get("messages") or [])
    out = []
    for ac in ri.get("action_checks") or []:
        g = ac.get("action") or {}
        matched = bool(ac.get("action_match"))
        cls, det = ("MATCH", {}) if matched else X12.classify(g, plist)
        out.append({"id": g.get("action_id"), "name": g.get("name"),
                    "cmp": g.get("compare_args"), "args": g.get("arguments") or {},
                    "matched": matched, "cls": cls, "det": det})
    return out, plist


def call_summary(plist):
    c = Counter(n for n, _ in plist)
    return c


def inner_name(args):
    """dispatcher 호출의 내부 도구명(있으면) — 행동 요약용."""
    for k in ("agent_tool_name", "discoverable_tool_name"):
        if isinstance(args, dict) and args.get(k):
            return str(args[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--task", default="")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    d = load(args.results)
    by_task = defaultdict(dict)
    for s in d.get("simulations") or []:
        r = (s.get("reward_info") or {}).get("reward")
        by_task[s.get("task_id")][s.get("trial")] = s

    fails, flips = [], []
    for t, tr in sorted(by_task.items()):
        rs = [(k, (v.get("reward_info") or {}).get("reward")) for k, v in sorted(tr.items())]
        vals = [1 if (x or 0) >= 1 else 0 for _, x in rs]
        if len(vals) < 2:
            continue
        (flips if len(set(vals)) > 1 else (fails if sum(vals) == 0 else [])).append(t)

    if args.task:
        fails = [args.task] if args.task in fails else []
        flips = [args.task] if args.task in flips else []

    # ── ① 항상-실패 ────────────────────────────────────────────────────────
    print("=" * 90)
    print("① 항상-실패 %d건 — gold action별 실패 원인 (두 trial 대조)" % len(fails))
    print("=" * 90)
    stable = Counter()
    cls_all = Counter()
    for t in fails:
        tr = by_task[t]
        per = {}
        for k, s in sorted(tr.items()):
            acts, plist = sim_actions(s)
            per[k] = (acts, plist, s)
        a0 = per[sorted(per)[0]][0]
        a1 = per[sorted(per)[1]][0]
        sig0 = tuple((a["id"], a["cls"]) for a in a0)
        sig1 = tuple((a["id"], a["cls"]) for a in a1)
        same = sig0 == sig1
        stable["동일" if same else "상이"] += 1
        print("\n--- %s  (gold action %d개) %s" % (t, len(a0), "· 두 trial 원인 동일" if same
                                                   else "· ⚠trial마다 원인 다름"))
        for i, a in enumerate(a0):
            b = a1[i] if i < len(a1) else {"cls": "?", "det": {}}
            cls_all[a["cls"]] += 1
            cls_all[b["cls"]] += 1
            det = a["det"] or {}
            extra = ""
            if a["cls"] == "NAME_ABSENT":
                extra = "  ← 그 외부 도구를 한 번도 안 부름"
            elif a["cls"] == "TOP_VALUE":
                extra = "  키=%s" % det.get("keys")
            elif a["cls"] == "NESTED_VALUE":
                extra = "  내부=%s" % det.get("inner")
            elif a["cls"] == "PRED_EXTRA_KEY":
                extra = "  여분키=%s" % det.get("keys")
            print("   %-8s %-34s t0=%-14s t1=%-14s%s"
                  % (a["id"], (a["name"] or "")[:34], a["cls"], b["cls"], extra))
        for k in sorted(per):
            acts, plist, s = per[k]
            top = ", ".join("%s×%d" % (n, c) for n, c in call_summary(plist).most_common(4))
            inners = [x for x in (inner_name(a) for _, a in plist) if x]
            print("     trial %s: 호출 %d회 [%s]%s · 종료=%s"
                  % (k, len(plist), top,
                     " · 내부도구 " + ",".join(sorted(set(inners))[:4]) if inners else "",
                     s.get("termination_reason")))

    print("\n항상-실패 요약: 두 trial 원인 %s" % dict(stable))
    print("  실패 분류 분포(두 trial 합산): %s" % dict(cls_all.most_common()))

    # ── ② flip ────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("② flip %d건 — 성공/실패가 갈린 지점" % len(flips))
    print("=" * 90)
    for t in flips:
        tr = by_task[t]
        ks = sorted(tr)
        info = {}
        for k in ks:
            s = tr[k]
            acts, plist = sim_actions(s)
            rw = 1 if ((s.get("reward_info") or {}).get("reward") or 0) >= 1 else 0
            info[k] = dict(acts=acts, plist=plist, rw=rw, s=s)
        p = [k for k in ks if info[k]["rw"] == 1][0]
        f = [k for k in ks if info[k]["rw"] == 0][0]
        print("\n--- %s   PASS=trial %s · fail=trial %s" % (t, p, f))
        ap_, af = info[p]["acts"], info[f]["acts"]
        for i, a in enumerate(af):
            b = ap_[i] if i < len(ap_) else {"matched": None, "cls": "?"}
            if a["matched"] == b["matched"]:
                continue
            det = a["det"] or {}
            print("   ★갈린 gold: %s %s" % (a["id"], a["name"]))
            print("      실패 trial: %s %s" % (a["cls"], json.dumps(det, ensure_ascii=False)[:160]))
            print("      통과 trial: 매치")
            # 실패 trial이 그 도구를 부르긴 했나
            same_calls = [(n, ar) for n, ar in info[f]["plist"] if n == a["name"]]
            print("      실패 trial의 동일 도구 호출 %d회%s"
                  % (len(same_calls),
                     "" if not same_calls else " · 인자 예: "
                     + json.dumps(same_calls[0][1], ensure_ascii=False)[:120]))
            gold_args = a["args"]
            print("      gold 인자(비교 대상 %s): %s"
                  % (a["cmp"], json.dumps({k: v for k, v in gold_args.items()
                                           if a["cmp"] is None or k in (a["cmp"] or [])},
                                          ensure_ascii=False)[:160]))
        for k in ks:
            plist = info[k]["plist"]
            top = ", ".join("%s×%d" % (n, c) for n, c in call_summary(plist).most_common(4))
            inners = [x for x in (inner_name(ar) for _, ar in plist) if x]
            print("      trial %s(%s): 호출 %d회 [%s]%s · %s초 · 종료=%s"
                  % (k, "PASS" if info[k]["rw"] else "fail", len(plist), top,
                     " · 내부 " + ",".join(sorted(set(inners))[:4]) if inners else "",
                     round(info[k]["s"].get("duration") or 0),
                     info[k]["s"].get("termination_reason")))

    print("\n⚠[[08]]: 분류는 **채점 규약 축자 재구현**(C245·1080/1080 검증)에 기반한다. "
          "행동의 '이유'(왜 그 도구를 안 불렀나)는 분류가 아니라 **궤적 정독**으로만 말할 수 있다.")


if __name__ == "__main__":
    main()
