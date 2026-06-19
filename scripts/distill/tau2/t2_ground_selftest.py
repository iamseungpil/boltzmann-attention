#!/usr/bin/env python
"""t2_resolve_patch grounding 로직 로컬 셀프테스트 (GPU/tau2 불요·원격 e2e 전 검증).

spec-driven _ground + resolve_op_tau2가 retail(map·평면 options·anchor ⋈)과 airline(list·explode/unnest·
available 술어)을 *같은 코드*로 처리하는지 + fetch-vs-ask 라우팅 신호(producer_present)를 확인.

Run (로컬): py -3 scripts/distill/tau2/t2_ground_selftest.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import t2_resolve_patch as P  # noqa: E402


def _load(domain):
    with open(os.path.join(_HERE, "a2", f"{domain}.grounding.json"), encoding="utf-8") as f:
        return json.load(f)


def _check(name, got, exp):
    ok = got == exp
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: got={got!r} exp={exp!r}")
    return ok


def main():
    fails = 0

    # ── retail: variants(map) + order items(list) anchor ⋈ ──
    rspec = _load("retail")
    retail_outs = [
        # 최근→과거 순(=_tool_outputs 반환순). get_order_details, get_product_details.
        {"order_id": "O1", "items": [{"item_id": "I2", "product_id": "P1"}]},
        {"product_id": "P1", "variants": {
            "I1": {"options": {"color": "black", "size": "M"}, "available": True},
            "I2": {"options": {"color": "white", "size": "M"}, "available": True},
            "I3": {"options": {"color": "black", "size": "L"}, "available": False},
        }},
    ]
    cat, anchor, present = P._ground(retail_outs, rspec)
    print("retail:")
    fails += not _check("producer_present", present, True)
    fails += not _check("anchor(order item of P1)", anchor, "I2")
    fails += not _check("catalog size", len(cat), 3)
    # substitute: anchor=I2(white,M), set color→black, keep size → I1(black,M) available
    rid = P.resolve_op_tau2({"op": "substitute", "set": {"color": "black"}}, cat, anchor_id=anchor)
    fails += not _check("substitute color→black keep size", rid, "I1")
    # filter among → unique
    rid = P.resolve_op_tau2({"op": "filter", "among": {"color": "white", "size": "M"}}, cat, anchor_id=anchor)
    fails += not _check("filter white,M", rid, "I2")

    # ── airline: search_direct_flight(list) + explode(unnest) by cabin + available 술어 seats>0 ──
    aspec = _load("airline")
    airline_outs = [
        [
            {"flight_number": "HAT1", "scheduled_departure_time_est": "06:00:00", "date": "2024-05-15",
             "prices": {"business": 1200, "economy": 230, "basic_economy": 87},
             "available_seats": {"business": 4, "economy": 6, "basic_economy": 0}},
            {"flight_number": "HAT2", "scheduled_departure_time_est": "10:00:00", "date": "2024-05-15",
             "prices": {"business": 1100, "economy": 200, "basic_economy": 90},
             "available_seats": {"business": 2, "economy": 0, "basic_economy": 5}},
        ],
    ]
    cat, anchor, present = P._ground(airline_outs, aspec)
    print("airline:")
    fails += not _check("producer_present", present, True)
    fails += not _check("anchor(none·no anchor_source)", anchor, None)
    fails += not _check("catalog size (2 flights × 3 cabins)", len(cat), 6)
    # cheapest economy AVAILABLE: HAT2/eco(200) seats0=unavail → HAT1/eco(230)
    rid = P.resolve_op_tau2({"op": "argmin", "attr": "price", "among": {"cabin": "economy"}}, cat)
    fails += not _check("argmin price among cabin=economy (avail)", rid, "HAT1")
    # cheapest basic_economy AVAILABLE: HAT1/be seats0=unavail → HAT2/be(90)
    rid = P.resolve_op_tau2({"op": "argmin", "attr": "price", "among": {"cabin": "basic_economy"}}, cat)
    fails += not _check("argmin price among cabin=basic_economy (avail)", rid, "HAT2")

    # ── fetch routing 신호: 후보 컨테이너 부재 → producer_present False ──
    cat, anchor, present = P._ground([{"user_id": "U1", "name": "x"}], rspec)
    print("fetch-routing:")
    fails += not _check("no candidate container → producer_present", present, False)
    fails += not _check("no candidate → catalog None", cat, None)

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAIL(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
