# -*- coding: utf-8 -*-
"""X596 — airline·retail A2 완결 저작 (2026-08-29·사용자 지시 *"A2. A3 저작 완료하라"*).

★왜 한 번에 저작하나 ([[72]] 사용자 지시 2026-08-21): *"1회 오프라인 저작이 매 런 발견보다 싸면
완결된 A2/A3 를 그냥 만든다"* · *"찔끔 넣고 '아직 안 갈렸다' 반복이 가장 비싼 경로"*.
범위는 태스크가 아니라 **코퍼스**다([[05]] 전이).

⛔출처 규율([[23]]): 값의 출처는 **env 레지스트리**와 **도메인 정책 산문**뿐이다. gold·task 파일은
   읽지 않았다. 키마다 `_note_<키>` 에 축자 출처를 적는다 — 못 대면 넣지 않는다.
   재료 = `a2/env_surface_airline_retail.json`(도구·인자·mutates·정책 전문·리모트 레지스트리 실측)

⛔死선언 금지: 넣기 전에 **엔진 리더의 계약**을 확인했다. 확인 못 한 키는 넣지 않는다
   (오늘 판정에서 읽기 0곳인 DEAD 4키가 나온 것이 그 이유다).

용법:
  py -3 x596_author_a2.py --dry     # 무엇이 바뀌는지만 출력
  py -3 x596_author_a2.py --apply   # settings/specific 기록(gate.json 재생성은 x18 --emit)
"""
import argparse
import io
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2 = os.path.join(_HERE, "a2")


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def dump(p, o):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(o, ensure_ascii=False, indent=1))


# ─────────────────────────────────────────────────────────────────────────────
# 저작 내용. (파일, 키) -> (값, 주석)
# 주석은 [[23]] 의무이므로 값과 **같은 자리에서** 쓴다 — 나중에 붙이면 안 붙는다.
# ─────────────────────────────────────────────────────────────────────────────

ENV_ERR_NOTE = (
    "출처 = env 축자. `tau2-bench/src/tau2/environment/environment.py:452` 가 도구 예외를 "
    "`resp = f\"Error: {e}\"` 로 감싼다 — 이 도메인의 도구 실패는 **전부** 이 접두사로 나온다"
    "(airline/retail `tools.py` 의 ValueError 23종 전수 확인). banking 이 표지를 넷 더 가진 것은 "
    "예외를 안 쓰고 문자열로 실패를 알리는 경로가 따로 있기 때문이고, 여기에는 그 경로가 없다. "
    "미선언이면 실패한 호출이 '실행됨'으로 잡혀 게이트가 거짓 충족된다"
    "(`t2_gate_patch.py:3537` 독스트링). gold 무참조."
)

AIRLINE_SETTINGS = {
    "action_tools": (
        ["book_reservation", "cancel_reservation", "send_certificate",
         "update_reservation_baggages", "update_reservation_flights",
         "update_reservation_passengers"],
        "출처 = env 레지스트리 실측(`MUTATES_STATE_ATTR`). airline 도구 14종 중 "
        "`mutates=True` 인 것이 정확히 이 6개다. 정책 축자가 같은 집합을 부른다: "
        "*\"Before taking any actions that update the booking database (booking, modifying "
        "flights, editing baggage, changing cabin class, or updating passenger information), "
        "you must list the action details and obtain explicit user confirmation (yes)\"*. "
        "⚠banking `_note_action_tools` 가 박제한 함정 재발 방지 — 목록이 빠지면 "
        "손님-실행/행동 분기가 통째로 조용해진다(x369: 58태스크 영향). gold 무참조."
    ),
    "failure_markers": (["Error:"], ENV_ERR_NOTE),
    "calc_tool": (
        "calculate",
        "출처 = env 레지스트리(airline `calculate(expression)` 실재). 엔진"
        "(`t2_gate_patch.py:12844` T2_NLNUM_PROV)은 통화기호+숫자 패턴만 보고 도메인 어휘가 0이며, "
        "재발행 문구가 이 도구명을 지목한다. 정책이 금액을 다루도록 요구한다 — 축자: "
        "*\"Each extra baggage is 50 dollars\"* · *\"The travel insurance is 30 dollars per "
        "passenger\"* · *\"the user is required to pay for the difference\"*. gold 무참조."
    ),
}

RETAIL_SETTINGS = {
    "action_tools": (
        ["cancel_pending_order", "exchange_delivered_order_items",
         "modify_pending_order_address", "modify_pending_order_items",
         "modify_pending_order_payment", "modify_user_address",
         "return_delivered_order_items"],
        "★2026-08-29 수리: `modify_user_address` 가 빠져 있었다. 출처 = env 레지스트리 실측"
        "(`MUTATES_STATE_ATTR`) — retail 변이 도구는 **7개**인데 선언은 6개였다. 빠진 도구는 "
        "행동 도구로 취급되지 않아 확인-게이트·손님/에이전트 실행 분기가 그 도구에 대해 "
        "조용했다. banking `_note_action_tools` 가 같은 종류의 누락(`call_discoverable_user_tool`)을 "
        "이미 박제했고 그때 1,618회가 3원소로 고정돼 있었다. gold 무참조."
    ),
    "failure_markers": (["Error:"], ENV_ERR_NOTE),
}

PLAN = [
    ("airline.settings.json", AIRLINE_SETTINGS),
    ("retail.settings.json", RETAIL_SETTINGS),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    for fn, adds in PLAN:
        dom = fn.split(".")[0]
        # ⛔[[24]] 양방향 규칙: 정본(settings/specific)만 고치면 등가 게이트가 FAIL 하고
        #   레거시 직접-read 사이트가 옛 값을 쓴다. `gate.json` 은 **같은 편집에서** 함께 간다.
        #   ⚠`x18 --emit` 은 쓰지 않는다 — 그것은 gate.json → 분할 방향이라 이 편집을 되돌리고,
        #   덤으로 손으로 놓인 banking 키들을 재정규화한다(2026-08-29 에 한 번 밟았다).
        targets = [os.path.join(_A2, fn), os.path.join(_A2, dom + ".gate.json")]
        print("=== %s (+ %s.gate.json)" % (fn, dom))
        changed = []
        for k, (val, note) in adds.items():
            if load(targets[0]).get(k) == val and load(targets[1]).get(k) == val:
                continue
            changed.append((k, load(targets[1]).get(k), val))
        for k, b, v in changed:
            print("   %-18s %s -> %s" % (k, json.dumps(b, ensure_ascii=False)[:55],
                                         json.dumps(v, ensure_ascii=False)[:105]))
        if not changed:
            print("   변경 없음")
        if a.apply and changed:
            for p in targets:
                d = load(p)
                for k, (val, note) in adds.items():
                    d[k] = val
                    d["_note_" + k] = note
                dump(p, d)
            print("   기록함 (두 파일)")


if __name__ == "__main__":
    main()
