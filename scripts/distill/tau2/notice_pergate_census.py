#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""notice_pergate_census.py — NOTICE-PERGATE 착수 게이트 (무료·오프라인·NEXT_LEVER_GEN §1.4/§1.5 O).

① 재현 동일성: retail 단일-notice에서 per-gate violations_of_sim ≡ 구 first-notice 공식
   (comp gz 456 전수 대조 — per-gate 평가가 스칼라 의미론과 동일함의 회귀 증명).
② G8 가상-deny 전수: passing sim의 cancel_pending_order 시도에서 G8(환불-고지)이
   deny했을 횟수 → over-block 상한. GO 조건: passing-cancel deny 낮음(>2%면 A2 미부착·보고만)
   ∧ 표적(t57형·failing)서 deny 실재.

usage: PYTHONIOENCODING=utf-8 py -3 notice_pergate_census.py --results <gz>
"""
import argparse, gzip, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_compliance import domain_constants, violations_of_sim, _args_of  # noqa: E402
from gate_interpreter import GateInterpreter  # noqa: E402

# G8 초안 = test_notice_gate.py G8 그대로 (§1.3 표적)
G8 = {
    "id": "G8_REFUND_NOTICE", "kind": "notice",
    "notice_text": "Per policy, the refund for a cancelled order always goes back to the "
                   "original payment method used for that order.",
    "applies_to": ["cancel_pending_order"],
}


def load_sims(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)["simulations"]


def old_g4(sim, transfer_msg):
    """구 first-notice 공식 replica (2026-06-13 정합화 의미론 그대로)."""
    msgs = sim.get("messages") or []
    res_by_id = {m["id"]: m for m in msgs if m.get("role") == "tool" and m.get("id")}
    transfer_executed = notice_sent = False
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        mc = m.get("content")
        if isinstance(mc, str) and transfer_msg and transfer_msg in mc:
            notice_sent = True
        for tc in (m.get("tool_calls") or []):
            res = res_by_id.get(tc.get("id"))
            content = (res or {}).get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            if "POLICY GATE" in content:
                continue
            if tc.get("name") == "transfer_to_human_agents" and res is not None \
                    and not res.get("error"):
                transfer_executed = True
    return transfer_executed and not notice_sent


def g8_virtual(sim):
    """G8 가상-deny replay: cancel 시도 시점에 환불-고지문이 이전 어시스턴트 발화에
    있었는가(순서-반영·엔진 per-gate 의미론 동형). 반환 (deny수, cancel시도수, 실행성공수)."""
    msgs = sim.get("messages") or []
    res_by_id = {m["id"]: m for m in msgs if m.get("role") == "tool" and m.get("id")}
    sent = False
    denies = attempts = executed = 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        mc = m.get("content")
        if isinstance(mc, str) and G8["notice_text"] in mc:
            sent = True
        for tc in (m.get("tool_calls") or []):
            if not GateInterpreter._gate_applies(G8, tc.get("name"), _args_of(tc)):
                continue
            attempts += 1
            if not sent:
                denies += 1
            res = res_by_id.get(tc.get("id"))
            content = (res or {}).get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            if res is not None and not res.get("error") and "POLICY GATE" not in content:
                executed += 1
    return denies, attempts, executed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    sims = load_sims(a.results)
    C = domain_constants("retail")

    # ① 재현 동일성 (single-notice: per-gate ≡ first-notice)
    mism = 0
    for s in sims:
        v = violations_of_sim(s, C)
        if v["g4"] != old_g4(s, C["TRANSFER_MSG"]):
            mism += 1
            print("  MISMATCH task %s trial %s" % (s.get("task_id"), s.get("trial")))
    print("① 재현 동일성: per-gate g4 vs 구 first-notice 공식 — mismatch %d/%d %s"
          % (mism, len(sims), "(100%% 재현)" if mism == 0 else "**회귀**"))

    # ② G8 가상-deny census
    n_pass = n_fail = 0
    pass_deny_sims, fail_deny_sims = [], []
    tot = {"pass": [0, 0, 0], "fail": [0, 0, 0]}
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        ok = r >= 1
        n_pass += ok
        n_fail += (not ok)
        d, at, ex = g8_virtual(s)
        key = "pass" if ok else "fail"
        tot[key][0] += d
        tot[key][1] += at
        tot[key][2] += ex
        if d:
            (pass_deny_sims if ok else fail_deny_sims).append((s.get("task_id"), s.get("trial"), d))
    print("\n② G8 가상-deny (comp %d sims: pass=%d fail=%d)" % (len(sims), n_pass, n_fail))
    print("  PASSING: deny-sims=%d/%d (%.1f%% = over-block 상한) · denies=%d attempts=%d executed=%d"
          % (len(pass_deny_sims), n_pass, 100.0 * len(pass_deny_sims) / max(n_pass, 1),
             tot["pass"][0], tot["pass"][1], tot["pass"][2]))
    print("  FAILING: deny-sims=%d/%d · denies=%d attempts=%d executed=%d (표적 t57형 실재 확인)"
          % (len(fail_deny_sims), n_fail, tot["fail"][0], tot["fail"][1], tot["fail"][2]))
    print("  passing deny sims:", sorted(set(t for t, _, _ in pass_deny_sims)))
    print("  failing deny sims:", sorted(set(t for t, _, _ in fail_deny_sims)))
    thr = 2.0
    rate = 100.0 * len(pass_deny_sims) / max(n_pass, 1)
    print("\n판정(설계 착수 게이트): over-block %.1f%% %s %.0f%% → %s"
          % (rate, ">" if rate > thr else "<=", thr,
             "A2 미부착·보고만" if rate > thr else "G8 A2 부착 GO"))


if __name__ == "__main__":
    main()
