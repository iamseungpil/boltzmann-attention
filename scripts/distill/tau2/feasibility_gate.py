#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""feasibility_gate.py -- operand-level precondition gate (decidable·domain-general·[[05]]).

structural controller(plan_execute_orch)가 못 잡는 operand 오류를 execute 시점에 차단:
  - WRONG_STATUS       : action의 applies-status != 실제 order status (delivered<->pending)
  - PARTIAL_CANCEL     : cancel은 whole-order만 -> 부분취소 의도 = 불가능
  - PRODUCT_SWAP       : modify/exchange의 new item은 같은 product의 variant여야 함 (luggage->coat 불가)
  - REFUND_CARD        : refund/modify-payment 결제수단 in {order 원결제} U {gift card} (DB 5/5 검증)
INFEASIBLE -> block + fallback(ASK / 대안 action). 엔진=도메인-일반, 지식=ACTION_SPEC(ABox) + ctx(db 조회).
ctx 주입으로 오프라인 단위테스트 가능(db 불요). 대상 fail: t10 t34 t57 t63 / t66 t85 t14 t51 t53.
"""
# ACTION_SPEC = ABox(retail 인스턴스). (action -> (intent_class, applies_status, batchable))
ACTION_SPEC = {
    "modify_pending_order_items":    ("item_change", "pending",   True),
    "exchange_delivered_order_items":("item_change", "delivered", True),
    "return_delivered_order_items":  ("item_return", "delivered", True),
    "modify_pending_order_address":  ("address",     "pending",   False),
    "cancel_pending_order":          ("cancel",      "pending",   False),
    "modify_pending_order_payment":  ("payment",     "pending",   False),
}


def check_write(action, args, ctx):
    """반환 list[str] of INFEASIBLE 사유 (빈 = 실행가능). ctx = 조회 함수 dict."""
    reasons = []
    oid = args.get("order_id")
    spec = ACTION_SPEC.get(action)
    # 1. status feasibility (action이 요구하는 status와 실제 status 불일치)
    st = ctx["order_status"](oid)
    if spec and st is not None and st != spec[1]:
        reasons.append("WRONG_STATUS:%s needs %s, order=%s" % (action, spec[1], st))
    # 2. partial-cancel 불가 (cancel은 whole-order)
    if action == "cancel_pending_order" and args.get("item_ids"):
        reasons.append("PARTIAL_CANCEL_IMPOSSIBLE:cancel is whole-order only")
    # 3. product-swap 불가 (new item은 같은 product의 variant)
    if action in ("modify_pending_order_items", "exchange_delivered_order_items"):
        for old, new in zip(args.get("item_ids") or [], args.get("new_item_ids") or []):
            po, pn = ctx["product_of"](old), ctx["product_of"](new)
            if po is not None and pn is not None and po != pn:
                reasons.append("PRODUCT_SWAP_IMPOSSIBLE:%s(%s)->%s(%s) different product" % (old, po, new, pn))
    # 4. refund/payment 카드 소유 (원결제 U gift)
    if action == "modify_pending_order_payment":
        pm = args.get("payment_method_id")
        allowed = set(ctx["order_payment_ids"](oid)) | set(ctx["user_giftcards"](oid))
        if pm and pm not in allowed:
            reasons.append("REFUND_CARD_NOT_OWNED:%s not in order-original U gift" % pm)
    return reasons


# ---- 오프라인 단위테스트 (db 불요·stub ctx·gpt-4.1 0) ----
if __name__ == "__main__":
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    fails = []
    def ck(name, cond):
        print("  [%s] %s" % ("ok" if cond else "FAIL", name));
        if not cond: fails.append(name)

    # t34: 부분취소(cancel with items) on pending order -> PARTIAL_CANCEL
    ctx34 = {"order_status": lambda o: "pending", "product_of": lambda i: None,
             "order_payment_ids": lambda o: [], "user_giftcards": lambda o: []}
    r = check_write("cancel_pending_order", {"order_id": "W1", "item_ids": ["a"]}, ctx34)
    print("t34 partial-cancel:", r); ck("PARTIAL_CANCEL 검출", any("PARTIAL_CANCEL" in x for x in r))

    # t66: luggage->coat via modify (different product) -> PRODUCT_SWAP
    prod = {"lug1": "P_LUGGAGE", "coat1": "P_COAT"}
    ctx66 = {"order_status": lambda o: "pending", "product_of": lambda i: prod.get(i),
             "order_payment_ids": lambda o: [], "user_giftcards": lambda o: []}
    r = check_write("modify_pending_order_items", {"order_id": "W1", "item_ids": ["lug1"], "new_item_ids": ["coat1"]}, ctx66)
    print("t66 product-swap:", r); ck("PRODUCT_SWAP 검출", any("PRODUCT_SWAP" in x for x in r))

    # same-product variant swap -> FEASIBLE (빈)
    prodv = {"v1": "P_TSHIRT", "v2": "P_TSHIRT"}
    ctxv = {"order_status": lambda o: "pending", "product_of": lambda i: prodv.get(i),
            "order_payment_ids": lambda o: [], "user_giftcards": lambda o: []}
    r = check_write("modify_pending_order_items", {"order_id": "W1", "item_ids": ["v1"], "new_item_ids": ["v2"]}, ctxv)
    print("variant swap (feasible):", r); ck("같은-product variant = 통과", r == [])

    # t63/t10: refund to non-owned card -> REFUND_CARD
    ctx63 = {"order_status": lambda o: "pending", "product_of": lambda i: None,
             "order_payment_ids": lambda o: {"gift_card_1"}, "user_giftcards": lambda o: {"gift_card_1"}}
    r = check_write("modify_pending_order_payment", {"order_id": "W1", "payment_method_id": "credit_9999"}, ctx63)
    print("t63 wrong-refund-card:", r); ck("REFUND_CARD 검출", any("REFUND_CARD" in x for x in r))
    # owned card -> feasible
    r = check_write("modify_pending_order_payment", {"order_id": "W1", "payment_method_id": "gift_card_1"}, ctx63)
    ck("소유 카드 = 통과", r == [])

    # t85: exchange_delivered on PENDING order -> WRONG_STATUS
    ctx85 = {"order_status": lambda o: "pending", "product_of": lambda i: None,
             "order_payment_ids": lambda o: [], "user_giftcards": lambda o: []}
    r = check_write("exchange_delivered_order_items", {"order_id": "W1", "item_ids": ["a"], "new_item_ids": ["b"]}, ctx85)
    print("t85 wrong-status:", r); ck("WRONG_STATUS 검출", any("WRONG_STATUS" in x for x in r))

    print("\n=== 결과 ===")
    if fails: print("FAIL:", fails); raise SystemExit(1)
    print("ALL PASS -- feasibility gate가 t34/t66/t63/t10/t85 계열 불가능-op 차단·정상op 통과.")
