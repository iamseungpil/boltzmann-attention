#!/usr/bin/env python3
"""C50 - R1 위계 검증기 (호출가능성 + GET-chain 2-hop + DISAMBIGUATE) vs R0.

정본 doc: reports/facet_rft_2026/C49_ASK_HIERARCHY_NOT_BAN_2026_07_09.md

위계 (먼저 되는 데서 멈춤):
  FIND        : gold_x 가 {발화 U 도구출력}에 실재
  GET         : producer(x) P 가 *지금 호출 가능* (P의 입력키가 문맥에 있거나, 그 입력키도 GET-chain으로 도달가능·깊이2)
  DISAMBIGUATE: 문맥에 x-형 후보가 2개 이상 (⋈ 지점) -> 열거해 고르게
  INFER       : (본 프로브 미분류)
  ASK         : 위 전부 불가 (최후)

R0 검증기: producer 존재하면 ASK 금지 (C44·too strong)
R1 검증기: 위 위계. ASK 위반 = 더 높은 갈래가 지금 가능한데 ASK 함.

두 모집단:
  FAB   : 원 궤적 날조 지점 (retail 30)
  DEADEND : producer 있으나 입력키 없어 R0가 GET강제하는 막다른길 (retail 7 + 유사)
"""
import gzip
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/home/woori/scratch")
import c47_dprime as D  # noqa: E402
from e11a_isolated_probe import find_violations  # noqa: E402

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"

# 인자 -> producer 도구 + 그 도구의 필수 입력키 (retail)
# 입력키 __auth__ = find_user/get_user 전력이면 충족 · 나머지는 문맥에 그 키값이 있어야
PRODUCER = {
    "new_item_ids": ("get_product_details", "product_id"),
    "item_ids": ("get_order_details", "order_id"),
    "payment_method_id": ("get_user_details", "__auth__"),
    "order_id": ("get_user_details", "__auth__"),
    "address1": ("get_order_details", "order_id"),
}
# 입력키의 출처 (GET-chain 재귀용): 그 키를 어떻게 얻나
KEY_SOURCE = {
    "product_id": None,          # 도구 출력(get_order_details)의 아이템에 들어있음 -> 문맥의존
    "order_id": ("get_user_details", "__auth__"),   # user 조회하면 주문목록 나옴
}
ID_IN_CTX = {
    "product_id": re.compile(r"\bproduct_id\b|\b\d{10}\b"),
    "order_id": re.compile(r"w\d{7}"),
}


def norm(x):
    return D.norm(x)


def authed(sim, idx):
    return any(mm.get("role") == "assistant" and any(
        t.get("name") in ("find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details")
        for t in (mm.get("tool_calls") or [])) for mm in sim["messages"][:idx])


def key_present(sim, idx, inp):
    pre = D.prefix_txt(sim, idx)
    if inp == "__auth__":
        return authed(sim, idx)
    pat = ID_IN_CTX.get(inp)
    return bool(pat and pat.search(pre))


def get_callable(sim, idx, key, depth=2):
    """producer(key)를 지금 호출 가능한가 (GET-chain 재귀·깊이 depth)."""
    prod = PRODUCER.get(key)
    if not prod:
        return False
    tool, inp = prod
    if key_present(sim, idx, inp):
        return True
    # 입력키가 없다 -> 그 입력키를 GET-chain으로 얻을 수 있나
    if depth > 0:
        src = KEY_SOURCE.get(inp)
        if src:
            _, inp2 = src
            if key_present(sim, idx, inp2):
                return True
    return False


def candidate_count(sim, idx, key):
    """문맥(도구출력)에 x-형 후보가 몇 개."""
    tool_txt = " ".join(str(m.get("content")) for m in sim["messages"][:idx] if m.get("role") == "tool")
    if key in ("new_item_ids", "item_ids"):
        pat = r"\b\d{10}\b"
    elif key == "payment_method_id":
        pat = r"(?:credit_card|gift_card|paypal)_\d+"
    elif key == "address1":
        pat = r"\d+ [A-Z][a-z]+ (?:Street|Avenue|Drive|Lane|Road)"
    else:
        return 0
    return len(set(re.findall(pat, tool_txt)))


def r1_label(sim, idx, key, gold_value):
    """R1 위계가 판정하는 이 지점의 *정답 갈래*."""
    if gold_value is not None and norm(gold_value) in D.prefix_txt(sim, idx):
        # 값이 문맥에 있음 -> FIND. 단 후보 여럿이면 DISAMBIGUATE
        return "DISAMBIGUATE" if candidate_count(sim, idx, key) >= 2 else "FIND"
    if get_callable(sim, idx, key):
        return "GET"
    if candidate_count(sim, idx, key) >= 2:
        return "DISAMBIGUATE"
    return "ASK"


def r0_says_ask_banned(key):
    return key in PRODUCER   # producer 존재하면 R0는 ASK 금지


def main():
    sims = json.load(gzip.open(SIM + "fl32b_floor_retail_t4.results.json.gz"))["simulations"]
    V = find_violations(sims, 40)

    # 각 결정점에서 R1 정답 갈래 + R0 판정 대조
    rows = []
    for sim, idx, tc, key, val, want in V:
        gold, gv = D.gold_label(sim, tc, key, idx)
        r1 = r1_label(sim, idx, key, gv)
        getc = get_callable(sim, idx, key)
        rows.append({"task": str(sim.get("task_id")), "trial": sim.get("trial"), "arg": key,
                     "gold": gold, "R1정답": r1, "GET호출가능": getc,
                     "후보수": candidate_count(sim, idx, key)})

    print("=== R1 위계가 배정한 정답 갈래 (날조 지점 40) ===")
    print(" ", dict(Counter(r["R1정답"] for r in rows)))
    print("\n=== gold(값거처) x R1(갈래) 교차 ===")
    cm = Counter((r["gold"], r["R1정답"]) for r in rows)
    for k, v in sorted(cm.items(), key=lambda x: -x[1]):
        print("   %-10s -> %-14s : %d" % (k[0], k[1], v))

    # ★R0 vs R1 이 갈리는 지점: gold=GET 불가능인데 R0는 GET강제
    print("\n=== ★R0 막다른길 (GET 호출불가인데 R0는 ASK금지=GET강제) ===")
    dead = [r for r in rows if not r["GET호출가능"] and r0_says_ask_banned(r["arg"]) and r["R1정답"] in ("ASK", "DISAMBIGUATE")]
    print("   %d 건" % len(dead))
    for r in dead:
        print("   t%s tr%s %-18s gold=%s R1=%s 후보=%d" % (r["task"], r["trial"], r["arg"], r["gold"], r["R1정답"], r["후보수"]))

    print("\n=== DISAMBIGUATE 지점 (후보 2+·⋈ 처방) ===")
    dis = [r for r in rows if r["R1정답"] == "DISAMBIGUATE"]
    for r in dis:
        print("   t%s tr%s %-18s gold=%s 후보=%d" % (r["task"], r["trial"], r["arg"], r["gold"], r["후보수"]))


if __name__ == "__main__":
    main()
