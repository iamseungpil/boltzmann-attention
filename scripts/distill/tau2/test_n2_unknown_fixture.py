"""미기입이 안전한 폴백이 아니라는 것을 픽스처로 고정한다 (설계서 §2.4b).

rev2는 "ASK 예산이 소진되면 인자를 빼고 호출한다"로 충돌을 해소했다. 재리뷰가 하류 거동을 물었고,
확인해 보니 **두 하류가 모두 빈 값을 false로 되돌린다** — 미기입은 false 단정과 같은 결과다.

    env  `apply_for_credit_card(rho_bank_subscription: bool = False)`   (tools.py:4409)
    우리 `catalog_filter`: `if row.get("invite_only") and not ctx.get("invited")`

우리 쪽은 고칠 수 있고 고쳤다(T2_UNKNOWN_UNVERIFIED). env 쪽은 못 고치므로 폴백이 ASK-또는-보류여야
한다. 이 테스트는 그 두 사실이 나중에 조용히 뒤집히지 않도록 박아 둔다 — 특히 우리 쪽 수정이
**명시적 false는 그대로 배제**한다는 것(문서화된 판단은 보존)을 함께 고정한다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_compute as C  # noqa: E402

SPEC = {"op": "catalog_filter", "table": [
    {"card": "Diamond Elite Card", "invite_only": True, "annual_fee": 495.0,
     "source": "doc diamond_*"},
    {"card": "Gold Rewards Card", "annual_fee": 0.0, "source": "doc gold_*"}]}


def buckets(ctx, flag):
    os.environ["T2_UNKNOWN_UNVERIFIED"] = "1" if flag else "0"
    r = C.apply_op(SPEC, dict(ctx))
    return {k: [(x.get("card") if isinstance(x, dict) else x) for x in (r.get(k) or [])]
            for k in ("eligible", "excluded", "unverified")}


def main():
    off = buckets({}, False)
    assert off["excluded"] == ["Diamond Elite Card"], off
    print("  ok   OFF: 미지가 배제로 접힌다 (종전 거동·이것이 결함이었다)")

    on = buckets({}, True)
    assert on["unverified"] == ["Diamond Elite Card"] and not on["excluded"], on
    print("  ok   ON : 미지 → unverified (카드가 살아남고 선택은 모델에게)")

    exp = buckets({"invited": False}, True)
    assert exp["excluded"] == ["Diamond Elite Card"], exp
    print("  ok   ON : **명시적 false는 그대로 배제** (문서화된 판단 보존)")

    yes = buckets({"invited": True}, True)
    assert "Diamond Elite Card" in yes["eligible"], yes
    print("  ok   ON : 초대받았으면 eligible")

    # env 쪽 사실 — 코드가 아니라 서명을 박제한다(리모트 파일이라 여기선 상수로 고정)
    print("  note env apply_for_credit_card(rho_bank_subscription: bool = False)"
          " → 미기입 = false 실행 (그래서 폴백은 ASK-또는-보류)")
    print("PASS (4/4)")


if __name__ == "__main__":
    main()
