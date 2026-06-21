#!/usr/bin/env python
"""fetch-first 실패 *메커니즘* 궤적 전수 census (FETCHFIRST_PROMPT_FAILURE_MECHANISM §3).

failcensus_deep는 sim-수준(A_notfound)까지. 이건 *왜 그 id가 틀렸나*를 궤적 읽어 분류:
  M1M4_no_gather : 날조 id를 쓰기 전에 producer(get_user_details 등)를 *안 부름* = 규칙무시/prior.
       └ schema_copy 하위: 날조값이 스키마-example 패턴(#W0000000류) = prior-override 직접 proxy(M4).
  M2_gather_wrong: producer는 불렀으나 그 출력에 *없는* id 사용 = 2-hop threading 실패(capability).
  grounded_other : 쓴 id는 grounded인데 실패(=B operand/기타·fetch-first 아님).
M1 vs M4 최종 구분은 P-restate arm(rule-echo)이 함 — 여기선 no_gather + schema_copy율로 prior 강도 측정.

Run: PY c4_prompt_mechanism.py <sim_dir1> [sim_dir2 ...]
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

# 소비 도구(id 사용) / 생산 도구(id 출처). 분석 도구라 도메인 리터럴 OK(엔진 아님).
CONSUMERS = {"get_order_details", "modify_pending_order_items", "modify_pending_order_address",
             "modify_pending_order_payment", "exchange_delivered_order_items",
             "return_delivered_order_items", "cancel_pending_order",
             "get_reservation_details", "cancel_reservation", "update_reservation_flights",
             "update_reservation_passengers", "update_reservation_baggages"}
PRODUCERS = {"get_user_details", "find_user_id_by_email", "find_user_id_by_name_zip"}
ID_KEYS = ("order_id", "reservation_id", "id")  # 소비 인자 중 식별자
PLACEHOLDER = re.compile(r"#W0+\d{0,3}\b|example\.com|johndoe|jane_doe|ABC123|XYZ789", re.I)


def _args(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def _grounded_text(msgs, upto):
    """upto 이전 tool 출력 + user 메시지의 원문 concat(정규화·substring 매치용).
    ★정규식 토큰화는 'W2378156.'(마침표글루)로 idv와 불일치하는 버그 → 정규화 substring으로 교체."""
    t = " ".join(str(m.get("content") or "") for m in msgs[:upto]
                 if m.get("role") in ("tool", "user"))
    return t.lower().replace("#", "")


def _grounded(idv, text):
    v = idv.lower().replace("#", "").strip().strip(".,;:")
    return len(v) >= 4 and v in text


def classify(s):
    msgs = s.get("messages", [])
    called_producer = False
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            if nm in PRODUCERS:
                called_producer = True
            if nm in CONSUMERS:
                a = _args(tc)
                idv = next((str(a[k]) for k in a if any(idk in k.lower() for idk in ID_KEYS)), None)
                if not idv:
                    continue
                if _grounded(idv, _grounded_text(msgs, i)):
                    continue  # grounded(정규화 substring) → fetch-first 위반 아님
                # 날조 id 발견
                if PLACEHOLDER.search(idv):
                    return "M1M4_no_gather", "schema_copy"
                if not called_producer:
                    return "M1M4_no_gather", "generic_fab"
                return "M2_gather_wrong", "threading"
    return "grounded_other", "-"


def census(sim_dir):
    with open(os.path.join(sim_dir, "results.json"), encoding="utf-8") as f:
        sims = json.load(f).get("simulations", [])
    cat = Counter(); sub = Counter(); ex = defaultdict(list); nfail = 0
    for s in sims:
        if (s.get("reward_info") or {}).get("reward", 1) >= 1:
            continue
        nfail += 1
        c, sb = classify(s)
        cat[c] += 1; sub[(c, sb)] += 1
        if len(ex[c]) < 5:
            ex[c].append(s.get("task_id"))
    return {"dir": os.path.basename(sim_dir), "nfail": nfail, "cat": cat, "sub": sub, "ex": ex}


def main():
    for d in sys.argv[1:]:
        try:
            r = census(d)
        except Exception as e:
            print(f"SKIP {d}: {e}"); continue
        nf = r["nfail"] or 1
        print(f"\n=== {r['dir']} (nfail={r['nfail']}) ===")
        for c in ("M1M4_no_gather", "M2_gather_wrong", "grounded_other"):
            n = r["cat"].get(c, 0)
            print(f"  {c:18s} {n:3d} ({n/nf:.2f})  ex={r['ex'].get(c, [])[:5]}")
        print("  sub:", dict(r["sub"]))
        # M-MECH-ROW for grep
        print(f"M_MECH_ROW {r['dir']} nfail={r['nfail']} "
              f"no_gather={r['cat'].get('M1M4_no_gather',0)} "
              f"schema_copy={r['sub'].get(('M1M4_no_gather','schema_copy'),0)} "
              f"gather_wrong={r['cat'].get('M2_gather_wrong',0)} "
              f"grounded_other={r['cat'].get('grounded_other',0)}")


if __name__ == "__main__":
    main()
