#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""newstack_perstep_census.py — 신스택 이득/부작용 per-step 귀속 (직접실행·[[08]]).

개입은 생성-레벨(silent)이라 transcript에 안 보임 → 마커는 stderr 로그(sim-id 없음).
귀속 = 마커의 (tool, val)을 sim의 write와 값-매칭. nt 불일치(신 nt=1 vs COMP nt=4)라
sim-비율 직접비교는 무효 → 태스크-단위 + v25e(nt=4·COMP 짝) 별도.

산출:
  §A 마커 census(레버별 발화 수)
  §B 레버 발화-sim ∩ 통과 (귀속 후 그 sim 통과했나 = 이득 개연)
  §C Δspurious: b78c2 OVER_ACTION(gold 없는 write) 율 vs COMP
  §D cap-hit sim(=deny 무한루프 방지장치 발동) 통과 여부 (알려진 부작용 t27/t103류)
usage: newstack_perstep_census.py --new <gz> --log <stderr.gz> --comp <comp.gz> --tasks <tasks.json>
"""
import argparse, gzip, json, re, sys
from collections import Counter, defaultdict

WR = ("modify", "exchange", "return", "cancel")


def load_gz(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_log(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def args_of(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def sim_writes(sim):
    """sim의 전 write (tool, args, ok)."""
    res = {m.get("id"): m for m in (sim.get("messages") or []) if m.get("role") == "tool"}
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            if any(w in nm for w in WR):
                tm = res.get(tc.get("id"))
                ok = tm is not None and not tm.get("error")
                out.append((nm, args_of(tc.get("arguments")), ok))
    return out


def sim_vals(sim):
    """sim이 언급한 모든 order_id/값 집합(마커 귀속용)."""
    vals = set()
    for m in sim.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            vals |= set(re.findall(r"#W\d+", c))
        for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            for v in args_of(tc.get("arguments")).values():
                for x in (v if isinstance(v, list) else [v]):
                    vals.add(str(x))
    return vals


def gold_writes(task):
    return [(x.get("name"), args_of(x.get("arguments")))
            for x in ((task.get("evaluation_criteria") or {}).get("actions") or [])
            if x.get("requestor", "assistant") == "assistant" and any(w in (x.get("name") or "") for w in WR)]


def reward(s):
    return (s.get("reward_info") or {}).get("reward")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args()
    new = load_gz(a.new)["simulations"]
    comp = load_gz(a.comp)["simulations"]
    tasks = {str(t["id"]): t for t in load_gz(a.tasks)}
    log = load_log(a.log)

    # ── §A 마커 census
    mk = Counter()
    for pat, name in [(r"\[T2_DISAMB\] fired", "DISAMB"), (r"\[T2_GROUND\] substituted", "GROUND"),
                      (r"\[T2_PROV\] regen fired", "PROV_regen"), (r"\[T2_PROV\] rescue", "PROV_rescue"),
                      (r"\[T2_EPLAN\] L1 deny", "EPLAN_L1"), (r"\[T2_EPLAN\] L2 deny", "EPLAN_L2"),
                      (r"\[T2_EPLAN\] walk gap", "EPLAN_walk"), (r"\[T2_EPLAN\] deny cap", "EPLAN_cap"),
                      (r"\[T2_PRINCIPLE", "P2"), (r"\[T2_NLNUM", "NLNUM"), (r"\[T2_CALCX", "CALCX")]:
        mk[name] = len(re.findall(pat, log))
    print("== §A 마커 census (신스택 b78c2·전 발화) ==")
    for k, v in mk.most_common():
        if v:
            print("  %-14s %d" % (k, v))

    # ── 마커 → sim 귀속 (val 매칭): DISAMB/PROV/EPLAN-L2 는 val 있음
    def markers_with_val(pat):
        return re.findall(pat, log)
    disamb_vals = markers_with_val(r"\[T2_DISAMB\] fired tool=\S+ arg=\S+ val=(\S+)")
    prov_vals = markers_with_val(r"\[T2_PROV\] regen fired tool=\S+ arg=\S+ val=(.+)")
    l2_sibs = re.findall(r"\[T2_EPLAN\] L2 deny: unexamined siblings ([^\n]+)", log)
    l2_vals = set()
    for s in l2_sibs:
        l2_vals |= set(re.findall(r"#W\d+", s))

    # sim별 val 인덱스
    byval = defaultdict(list)
    for s in new:
        for v in sim_vals(s):
            byval[v].append(s)

    def attribute(vals):
        hit = set()
        for v in vals:
            v = v.strip()
            for s in byval.get(v, []):
                hit.add((str(s.get("task_id")), s.get("trial")))
        return hit

    print("\n== §B 레버 발화-sim ∩ 통과 (귀속·nt=1 신스택) ==")
    print("  주의: 귀속은 val-매칭(다른 sim에 같은 order_id면 과대). 통과=이득 개연이지 인과 아님.")
    for name, vals in [("DISAMB", disamb_vals), ("PROV_regen", prov_vals),
                       ("EPLAN_L2", list(l2_vals))]:
        sims = attribute(vals)
        if not sims:
            continue
        rmap = {(str(s.get("task_id")), s.get("trial")): reward(s) for s in new}
        p = sum(1 for k in sims if (rmap.get(k) or 0) >= 1)
        print("  %-12s 귀속 sim %d · 그중 통과 %d (%.0f%%)" % (name, len(sims), p, 100 * p / max(len(sims), 1)))

    # ── §C Δspurious: OVER_ACTION (gold 없는 tool 실행)
    def over_action_rate(sims):
        n = ov = 0
        exsims = []
        for s in sims:
            task = tasks.get(str(s.get("task_id")))
            if not task:
                continue
            gtools = {g[0] for g in gold_writes(task)}
            execd = [(nm, ar) for nm, ar, ok in sim_writes(s) if ok]
            extra = [nm for nm, ar in execd if nm not in gtools]
            n += 1
            if extra:
                ov += 1
                exsims.append((str(s.get("task_id")), s.get("trial"), extra))
        return n, ov, exsims
    nn, nov, nex = over_action_rate(new)
    cn, cov, _ = over_action_rate(comp)
    print("\n== §C Δspurious (OVER_ACTION = gold-없는 write 실행) ==")
    print("  신스택 b78c2: %d/%d (%.1f%%)  |  COMP: %d/%d (%.1f%%)" %
          (nov, nn, 100 * nov / max(nn, 1), cov, cn, 100 * cov / max(cn, 1)))
    print("  ★부작용 후보(신스택 over-action sim): %s" %
          ", ".join("t%s.%s%s" % (t, tr, ex) for t, tr, ex in nex[:20]))

    # ── §D cap-hit 부작용 (deny 무한루프 방지장치 = 알려진 t27/t103류)
    # cap-hit sim 귀속: L2 sibling val이 있는 sim 중 실패한 것 (근사)
    print("\n== §D E-PLAN cap-hit / L2-deny 통과 여부 (t27/t103 부작용 계열) ==")
    l2sims = attribute(list(l2_vals))
    rmap = {(str(s.get("task_id")), s.get("trial")): reward(s) for s in new}
    l2fail = [k for k in l2sims if (rmap.get(k) or 0) < 1]
    print("  L2-deny 귀속 sim %d · 그중 실패 %d — 실패 sim: %s" %
          (len(l2sims), len(l2fail), ", ".join("t%s.%s" % k for k in sorted(l2fail)[:25])))
    print("  (cap=19회 발동 = deny 무한루프 방지 실작동·per-case는 t27/t103 정독 doc 참조)")


if __name__ == "__main__":
    main()
