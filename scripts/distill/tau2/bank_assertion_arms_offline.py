# -*- coding: utf-8 -*-
"""assertion-provenance 2-arm 오프라인 검증 (무료·[[09]]·라이브 前 필수).

t019g/t019d 실궤적을 replay하며 **엔진 판정 로직만** 재현해 각 arm의 발화 지점을 센다.
(regen 결과는 라이브서만 알 수 있음 — 여기서 재는 것은 *발화 여부/지점*과 over-block 후보.)

검증 대상:
  arm discovery-required : data_source 읽음 ∧ producer 미호출 ∧ 사임 → fire
  arm self-declaration   : (LLM 선언 필요 = 오프라인 재현 불가) → *발화 기회 지점*만 카운트
사용: py -3 bank_assertion_arms_offline.py
"""
import gzip, json, io, sys, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

A2P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "banking_knowledge.gate.json")
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def _tool_names(m):
    return [tc.get("name") or (tc.get("function") or {}).get("name")
            for tc in (m.get("tool_calls") or [])]


def run(path, tag, a2):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    aps = a2.get("analysis_producers") or []
    print("#" * 72)
    print("#", tag)
    for si, sim in enumerate(data.get("simulations") or []):
        msgs = sim.get("messages") or []
        called = set()
        fires, chances = [], 0
        fired = False
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            tcs = _tool_names(m)
            resign = (not tcs) and isinstance(m.get("content"), str) and m["content"].strip()
            if resign:
                chances += 1
                if not fired:
                    for sp in aps:
                        ds, pr = sp.get("data_source"), sp.get("producer")
                        if ds in called and pr and pr not in called:
                            fires.append(i)
                            fired = True      # 상한 1/sim
                            break
            called.update(n for n in tcs if n)
        print(f"  sim{si}: 사임(발화기회) {chances} | ★discovery-required 발화 {len(fires)} @msg{fires} "
              f"| producer 실호출 {'get_reward_discrepancies' in called}")


with open(A2P, encoding="utf-8") as f:
    A2 = json.load(f)
print("A2 analysis_producers:", json.dumps(A2.get("analysis_producers"), ensure_ascii=False))
print("A2 assertion_operands:", json.dumps(A2.get("assertion_operands"), ensure_ascii=False))
run(os.path.join(BASE, "bank_t019g_20260716.results.json.gz"), "t019g (ASK게이트 arm)", A2)
run(os.path.join(BASE, "bank_t019d_20260716.results.json.gz"), "t019d (대조)", A2)
