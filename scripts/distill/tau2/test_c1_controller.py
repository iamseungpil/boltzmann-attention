#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_c1_controller.py — C1 controller 순수로직 오프라인 단위테스트 (db·32B·gpt-4.1 0).

Phase-0 plan_probe가 산출한 *실제* plan 3건을 controller에 통과시켜 결정론 회복을 검증:
  t71  BATCH_SPLIT → batch-merge 로 1콜 병합 (core 회복)
  t109 wrong-action(pending에 exchange_delivered) → status-fix remap (core 회복)
  t111 fabricated/wrong order drop → provenance, 단 ⋈로 누락된 gold 주문은 *회복 불가*(scale 잔여)
db 불요: status_fn·valid_oids 를 stub 주입. [[05]] controller=도메인-일반·ACTION_SPEC(ABox)만 참조.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from plan_execute_orch import controller, ACTION_SPEC

FS = frozenset
_fail = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _fail.append(name)


# ── t71: 한 주문 품목변경 2콜 분할 → batch-merge 1콜 ─────────────────────────
# plan_probe 실측 plan(t71): modify_items x2 on W5270061 + address + payment
plan71 = [("modify_pending_order_items", "W5270061", FS({"7453605304"})),
          ("modify_pending_order_items", "W5270061", FS({"2492465580"})),
          ("modify_pending_order_address", "W5270061", FS()),
          ("modify_pending_order_payment", "W5270061", FS())]
status71 = lambda oid: "pending"          # W5270061 = pending
valid71 = {"W5270061"}
norm71, fix71 = controller(plan71, valid71, status_fn=status71)
mi71 = [c for c in norm71 if c[0] == "modify_pending_order_items"]
print("t71 (batch-split → merge):")
check("modify_items 1콜로 병합", len(mi71) == 1, f"got {len(mi71)} calls")
check("병합 콜에 두 품목 union", len(mi71) == 1 and mi71[0][2] == FS({"7453605304", "2492465580"}))
check("batch_merge 발화=1", fix71.get("batch_merge") == 1, str(fix71))

# ── t109: pending 주문에 exchange_delivered → status-fix → modify_pending ───
plan109 = [("modify_pending_order_address", "W1603792", FS()),
           ("exchange_delivered_order_items", "W1603792", FS({"6501071631"}))]
status109 = lambda oid: "pending"         # W1603792 = pending (gold이 modify_pending 사용)
valid109 = {"W1603792"}
norm109, fix109 = controller(plan109, valid109, status_fn=status109)
acts109 = {c[0] for c in norm109}
print("t109 (wrong-action by status → remap):")
check("exchange_delivered → modify_pending_order_items 로 remap",
      "modify_pending_order_items" in acts109 and "exchange_delivered_order_items" not in acts109, str(acts109))
check("status_fix 발화=1", fix109.get("status_fix") == 1, str(fix109))

# ── t111: 틀린 주문(W3964602 ∉ 유저주문) drop, 단 ⋈로 누락된 W9810810은 회복불가 ──
plan111 = [("modify_pending_order_items", "W3730488", FS({"2913673670"})),
           ("exchange_delivered_order_items", "W3964602", FS({"4965355367"})),
           ("modify_pending_order_address", "W3730488", FS())]
status111 = lambda oid: "pending"
valid111 = {"W3730488", "W9810810"}       # 유저 실제 주문 (W3964602 = 타-유저/날조)
norm111, fix111 = controller(plan111, valid111, status_fn=status111)
oids111 = {c[1] for c in norm111}
print("t111 (provenance drop + ⋈-miss 회복불가):")
check("W3964602(무근거) drop", "W3964602" not in oids111, str(oids111))
check("provenance_drop 발화=1", fix111.get("provenance_drop") == 1, str(fix111))
check("⋈로 누락된 W9810810은 여전히 부재(결정론 불가·scale 잔여)",
      "W9810810" not in oids111, "controller가 없는 계획을 만들어내면 안 됨")

# ── [[05]] 가드: ACTION_SPEC(ABox)만이 도메인 지식·controller 로직에 retail 리터럴 0 ──
print("[[05]] 가드:")
check("ACTION_SPEC = 6 retail action (ABox)", len(ACTION_SPEC) == 6)
check("모든 spec = (intent_class, status, batchable)", all(len(v) == 3 for v in ACTION_SPEC.values()))

print("\n=== 결과 ===")
if _fail:
    print(f"FAIL {len(_fail)}: {_fail}")
    raise SystemExit(1)
print("ALL PASS — controller 결정론 회복(batch/status/provenance) 검증·⋈-miss는 scale 잔여로 정직 분리.")
