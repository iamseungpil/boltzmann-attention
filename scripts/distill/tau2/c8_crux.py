#!/usr/bin/env python
"""C8 crux 확인 — A-fab 실패가 producer(get_user_details)를 *부르지 않고* placeholder order_id를
날조하는가 (= '날조-FIRST' 근본·grounding 결함). + task16(fatima) gold order 대조.
Usage: c8_crux.py [base_sim_dir]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau2_rootcause_census import classify, call_seq  # noqa: E402

PLACE = re.compile(r"#W0+\d{1,3}\b")  # #W0000001 류 placeholder


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "data/simulations"

    def load(a):
        return {s["task_id"]: s for s in json.load(open(f"{base}/c8_{a}/results.json"))["simulations"]}

    arms = {a: load(a) for a in ("floor", "gate", "gate_retry")}

    def calls(s):
        return call_seq(s.get("messages") or [])

    print("=== CRUX: A-fab 실패궤적이 producer(get_user_details)를 부르나? ===")
    for a, sims in arms.items():
        fabfail = [t for t, s in sims.items()
                   if classify(s)["root"] in ("fab_then_loop", "fab_then_switch", "auth_fab")
                   and (s.get("reward_info") or {}).get("reward", 0) <= 0]
        authed = called_gud = fab_ph = authed_no_gud = 0
        for t in fabfail:
            seq = calls(sims[t])
            names = [n for _, n, _, _ in seq]
            has_auth = any(n.startswith("find_user_id") for n in names) and any(
                o and re.match(r'^"?[a-z]+_[a-z]+_\d+"?$', str(o).strip())
                for _, n, _, o in seq if n.startswith("find_user_id"))
            gud = "get_user_details" in names
            ph = any(PLACE.search(str(v)) for _, _, ar, _ in seq
                     for v in (ar.values() if isinstance(ar, dict) else []))
            authed += has_auth
            called_gud += gud
            fab_ph += ph
            authed_no_gud += (has_auth and not gud)
        print(f"  {a:11s} fab-fail={len(fabfail):2d} | 인증성공={authed:2d} | "
              f"get_user_details호출={called_gud:2d} | #W000000x날조={fab_ph:2d} | "
              f"인증✓∧producer미호출={authed_no_gud:2d}")

    print("\n=== task 16 (fatima_johnson_7581) gold order_id 대조 ===")
    res = json.load(open(f"{base}/c8_gate_retry/results.json"))
    t16 = [t for t in res["tasks"] if str(t.get("id")) == "16"]
    if t16:
        t = t16[0]
        ec = t.get("evaluation_criteria") or {}
        for a in (ec.get("actions") or []):
            ar = a.get("arguments") or {}
            ids = [(k, v) for k, v in ar.items() if "order" in k.lower() or k.lower().endswith("id")]
            print(f"  gold {a.get('name')}: {ids}")
        ui = (t.get("user_scenario") or {}).get("instructions") or {}
        print("  scenario:", str(ui)[:220])


if __name__ == "__main__":
    main()
