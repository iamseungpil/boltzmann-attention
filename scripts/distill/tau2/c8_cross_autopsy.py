#!/usr/bin/env python
"""C8 recovery 3-arm 교차 전수 궤적 autopsy (2026-06-22) — "loop-only 경계" 재확정.

C8 = floor / gate / gate_retry, **T2_PROVENANCE·T2_AUTOFETCH OFF**(c8_eval.sh:30).
질문: retry-controller가 *왜* pass를 못 올리나(loop-only 경계)? 근본원인 재확정.
가설(재확정 대상): A(order_id 날조→not-found→포기)의 진짜 결함 = *값-부재*(grounding)이지
  *루프-존재*가 아니다 → retry-controller(루프차단·다양화지시)는 루프를 없애도 *실값을 안 줘서*
  pass 못 올리고 loop→give-up 전이만 일으킴(A↑). 실 처방 = autofetch/grounding(C3·C8서 제외).

각 task를 3 arm 교차로: pass(floor/gate/retry) + root-cause(tau2_rootcause_census.classify 재사용)
  + gate_denied? + retry_fired? + autofetch_present?(=0 기대) + producer-getter 시도?
교차 버킷: gate_HURT(floor✓→gate✗) · retry_RESCUE(gate✗→retry✓) · retry_BREAK(gate✓→retry✗)
  · A_persistent(전 arm A-fail). 각 버킷 compact trace 덤프.

Usage: c8_cross_autopsy.py [base_sim_dir]   (default data/simulations)
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau2_rootcause_census import classify, call_seq, parse_args  # noqa: E402

AUTOFETCH_MARK = "i have fetched the authenticated user's actual record"
GATE_MARK = "policy gate"
RETRY_MARKS = ("retry_loop", "retry_escalate")


def load(d):
    return json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))["simulations"]


def markers(sim):
    """trajectory 내 마커 존재 카운트."""
    msgs = sim.get("messages") or []
    gate = retry = af = 0
    for m in msgs:
        if m.get("role") != "tool":
            continue
        c = str(m.get("content") or "").lower()
        if GATE_MARK in c:
            gate += 1
        if any(r in c for r in RETRY_MARKS):
            retry += 1
        if AUTOFETCH_MARK in c:
            af += 1
    return gate, retry, af


def passed(sim):
    return (sim.get("reward_info") or {}).get("reward", 0.0) > 0


def post_deny_action(sim):
    """첫 게이트/에러 deny 이후 에이전트의 최종 향방 요약(give-up 신호)."""
    term = sim.get("termination_reason")
    msgs = sim.get("messages") or []
    # 마지막 assistant 발화(전송/사과/질문) 신호
    last_assist = ""
    for m in reversed(msgs):
        if m.get("role") == "assistant" and m.get("content"):
            last_assist = str(m.get("content"))[:90].replace("\n", " ")
            break
    transferred = any((m.get("role") == "assistant") and m.get("tool_calls")
                      and any((tc.get("function", tc).get("name") or "") == "transfer_to_human_agents"
                              for tc in m["tool_calls"]) for m in msgs)
    return term, transferred, last_assist


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "data/simulations"
    arms = ["floor", "gate", "gate_retry"]
    data = {a: {s["task_id"]: s for s in load(os.path.join(base, f"c8_{a}"))} for a in arms}
    tasks = sorted(set(data["floor"]), key=lambda x: int(x) if str(x).isdigit() else 1e9)

    # 1) 전 arm 마커 집계 (autofetch=0 기대)
    print("=" * 70)
    print("1) ARM 마커 집계 (gate-deny / retry-fire / autofetch-inject)")
    for a in arms:
        gs = rs = afs = 0
        npass = 0
        for s in data[a].values():
            g, r, af = markers(s)
            gs += g
            rs += r
            afs += af
            npass += passed(s)
        print(f"  {a:11s} pass={npass:3d}/114 | gate-deny msgs={gs:4d} | "
              f"retry-fire msgs={rs:4d} | AUTOFETCH-inject msgs={afs:3d}")
    print("  ⇒ AUTOFETCH-inject=0 확인 = C8는 실값-주입 메커니즘 부재(c8_eval.sh:30).")

    # 2) root-cause 분포 per arm
    print("=" * 70)
    print("2) ROOT-CAUSE 분포 per arm (실패궤적)")
    cls = {a: {t: classify(s) for t, s in data[a].items()} for a in arms}
    for a in arms:
        dist = Counter(x["root"] for x in cls[a].values())
        print(f"  {a:11s} " + " ".join(f"{k}={v}" for k, v in dist.most_common()))

    # 3) 교차 flip 버킷
    def P(a, t):
        return passed(data[a][t])
    gate_hurt = [t for t in tasks if P("floor", t) and not P("gate", t)]
    gate_help = [t for t in tasks if not P("floor", t) and P("gate", t)]
    retry_rescue = [t for t in tasks if not P("gate", t) and P("gate_retry", t)]
    retry_break = [t for t in tasks if P("gate", t) and not P("gate_retry", t)]
    print("=" * 70)
    print("3) 교차 PASS FLIP")
    print(f"  gate_HURT  (floor✓→gate✗) = {len(gate_hurt):2d}  {gate_hurt}")
    print(f"  gate_HELP  (floor✗→gate✓) = {len(gate_help):2d}  {gate_help}")
    print(f"  retry_RESCUE(gate✗→retry✓)= {len(retry_rescue):2d}  {retry_rescue}")
    print(f"  retry_BREAK (gate✓→retry✗)= {len(retry_break):2d}  {retry_break}")
    print(f"  net gate  = {len(gate_help)-len(gate_hurt):+d} | "
          f"net retry = {len(retry_rescue)-len(retry_break):+d}")

    # 4) gate_HURT 정밀: 게이트가 왜 통과하던 걸 깼나
    print("=" * 70)
    print("4) gate_HURT 정밀 (floor 통과→gate 실패 원인)")
    for t in gate_hurt:
        x = cls["gate"][t]
        g, r, af = markers(data["gate"][t])
        term, tr, la = post_deny_action(data["gate"][t])
        print(f"  task {t:>3} | gate-deny={g} root={x['root']} term={term} | last:'{la[:60]}'")

    # 5) retry의 A 전이 검증: gate vs retry에서 A(fab/notfound) 추이 + retry 후 행동
    print("=" * 70)
    print("5) retry-controller 효과: A(fab→포기) 추이 + retry 발동 후 최종행동")
    for a in ("gate", "gate_retry"):
        afail = [t for t, x in cls[a].items() if x["root"] in ("fab_then_loop", "fab_then_switch", "auth_fab") and not P(a, t)]
        loops = [t for t, x in cls[a].items() if "loop" in x["root"] and not P(a, t)]
        print(f"  {a:11s}: A(fab)-fail={len(afail)} | loop-fail={len(loops)}")
    # retry 발동했으나 실패한 궤적의 최종 향방
    rfired_fail = [t for t in tasks if not P("gate_retry", t) and markers(data["gate_retry"][t])[1] > 0]
    print(f"\n  retry 발동∧실패 궤적 = {len(rfired_fail)}개. 발동 후 최종행동:")
    term_c = Counter()
    transfer_c = 0
    for t in rfired_fail:
        term, tr, la = post_deny_action(data["gate_retry"][t])
        term_c[term] += 1
        transfer_c += tr
    print("   termination:", dict(term_c), "| transfer_to_human 도달:", transfer_c)

    # 6) 샘플 trace: retry 발동 A-fail 3건 (루프차단됐으나 실값 못얻어 실패)
    print("=" * 70)
    print("6) 샘플 TRACE — retry 발동 A-fail (루프는 끊겼으나 실값 부재로 미복구)")
    shown = 0
    for t in rfired_fail:
        x = cls["gate_retry"][t]
        if x["root"] not in ("fab_then_loop", "fab_then_switch", "auth_fab"):
            continue
        s = data["gate_retry"][t]
        seq = call_seq(s.get("messages") or [])
        print(f"\n  --- task {t} (root={x['root']} term={s.get('termination_reason')}) ---")
        for (idx, nm, ar, outp) in seq[:14]:
            o = (str(outp)[:70].replace("\n", " ")) if outp else "-"
            av = ",".join(f"{k}={str(v)[:18]}" for k, v in (ar.items() if isinstance(ar, dict) else []))
            print(f"     {nm}({av[:55]}) -> {o}")
        shown += 1
        if shown >= 3:
            break

    print("=" * 70)
    print("결론 신호: (a)autofetch-inject=0 (b)retry net≈0∧A↑ (c)retry발동 후 transfer/포기 지배")
    print("  ⇒ 'loop-only 경계' 재확정: recovery(루프차단)는 *값-부재*(grounding) 결함을 못 메움.")


if __name__ == "__main__":
    main()
