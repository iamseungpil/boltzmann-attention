#!/usr/bin/env python
"""tau2 A2 adapter v1: retail policy.md -> deterministic gates (BENCH_PORTFOLIO §3.5 ②).

Gate 4종 (policy.md 수동 컴파일 — front-end 자동화의 GT):
  G1 AUTH_FIRST     user-scoped 도구는 인증(find_user_id_* 성공) 선행 필수
                    ("even when the user already provides the user id")
  G2 CONFIRM_WRITE  WRITE 도구는 직전 user 턴이 명시적 확인(yes)일 때만
  G3 SINGLE_USER    인증 유저 외 타 유저/타 유저 주문 대상 호출 deny
  G4 TRANSFER_MSG   transfer 후 고정 문구 송신 (post-hoc 체크 헬퍼만 — pre-call 게이트 불가)

★스코핑 발견 (2026-06-12): per-action 인자규칙(~100줄: status·reason enum·동일상품·
잔액·환불처)은 tools.py가 전부 자체 집행(raise) — 게이트 가치는 대화-수준 G1-G3에만 있음.
SOPBench 대비 어댑터 비용이 더 작다 (정책 136줄 -> 게이트 ~40줄).

검증 (Guard-2 동형): --validate 가 gold actions를 replay, G1+G3 over-deny=0 확인.
(G2는 gold에 user 발화 없어 replay 불가 — live 전용. G4는 post-hoc.)

Run: cd /home/woori/scratch/tau2-bench && PYTHONPATH=src \
  /home/woori/venvs/seka_env/bin/python $REPO/scripts/distill/tau2/t2_gate.py --validate
"""
import argparse, json, re

AUTH_TOOLS = {"find_user_id_by_name_zip", "find_user_id_by_email"}
WRITE_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}
# user-scoped READ: 제품 카탈로그(list/get_product/get_item)·calculate는 user-무관이라 제외
USER_SCOPED = WRITE_TOOLS | {"get_user_details", "get_order_details"}
CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|sure|confirm|confirmed|correct|proceed|go ahead|ok(ay)?|sounds good|"
    r"please do|that works|do it)\b", re.I)
TRANSFER_MSG = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."


class RetailGate:
    """대화-수준 결정론 게이트. 도구 실행 *전* check(), 실행 *후* observe()."""

    def __init__(self, db=None, enable_g2=True):
        self.db = db  # RetailDB (G3 order->user resolve용; None이면 G3 order 체크 skip)
        self.enable_g2 = enable_g2
        self.auth_user = None

    def observe(self, tool_name, args, result, ok=True):
        """도구 실행 결과로 상태 갱신 (find 성공 -> 인증 확립)."""
        if ok and tool_name in AUTH_TOOLS and isinstance(result, str) and result:
            self.auth_user = result

    def _order_owner(self, order_id):
        if self.db is None:
            return None
        order = self.db.orders.get(order_id)
        return order.user_id if order else None

    def check(self, tool_name, args, last_user_msg=None):
        """returns (allowed: bool, gate: str|None, reason: str|None)"""
        # G1: 인증 선행
        if tool_name in USER_SCOPED and self.auth_user is None:
            return False, "G1_AUTH_FIRST", (
                "authenticate the user first via find_user_id_by_email or "
                "find_user_id_by_name_zip (required even if the user gave a user id)")
        # G3: 단일-유저 범위
        if self.auth_user is not None:
            uid = args.get("user_id")
            if uid and uid != self.auth_user:
                return False, "G3_SINGLE_USER", f"user_id {uid} != authenticated {self.auth_user}"
            oid = args.get("order_id")
            if oid:
                owner = self._order_owner(oid)
                if owner is not None and owner != self.auth_user:
                    return False, "G3_SINGLE_USER", f"order {oid} belongs to {owner}"
        # G2: 쓰기-전-확인 (live 전용 — last_user_msg=None이면 skip)
        if self.enable_g2 and tool_name in WRITE_TOOLS and last_user_msg is not None:
            if not CONFIRM_RE.search(last_user_msg):
                return False, "G2_CONFIRM_WRITE", (
                    "list the action details and obtain explicit user confirmation (yes) first")
        return True, None, None


def check_transfer_msg(messages_after_transfer):
    """G4 post-hoc: transfer_to_human_agents 호출 후 고정 문구 송신 여부."""
    return any(TRANSFER_MSG in (m or "") for m in messages_after_transfer)


def validate(domain="retail"):
    """Guard-2 동형: gold actions replay -> G1+G3 over-deny=0 검증 (G2 off)."""
    import importlib
    mod = importlib.import_module(f"tau2.domains.{domain}.environment")
    env = mod.get_environment()
    db = env.tools.db
    tasks = mod.get_tasks(None)

    over_deny, no_auth_gold, n_actions = [], [], 0
    for t in tasks:
        gate = RetailGate(db=db, enable_g2=False)
        actions = t.evaluation_criteria.actions if t.evaluation_criteria else None
        if not actions:
            continue
        if not any(a.name in AUTH_TOOLS for a in actions):
            no_auth_gold.append(t.id)  # gold가 인증 act 없이 시작 -> G1 전제 점검용
        for a in actions:
            n_actions += 1
            ok, g, why = gate.check(a.name, a.arguments or {})
            if not ok:
                over_deny.append((t.id, a.name, g, why))
            # replay: 실행됐다 치고 상태 갱신 (find는 db에서 실답 resolve)
            if a.name in AUTH_TOOLS:
                uid = _resolve_find(db, a.name, a.arguments or {})
                gate.observe(a.name, a.arguments, uid)
    print(f"[validate] tasks={len(tasks)} gold_actions={n_actions} "
          f"OVER_DENY={len(over_deny)} no_auth_gold={len(no_auth_gold)}")
    for row in over_deny[:20]:
        print("  OVER:", row)
    if no_auth_gold:
        print("  no-auth-gold task ids:", no_auth_gold[:20])
    return len(over_deny)


def _resolve_find(db, name, args):
    for uid, u in db.users.items():
        if name == "find_user_id_by_email":
            if u.email.lower() == args.get("email", "").lower():
                return uid
        else:
            n, a = u.name, u.address
            if (n.first_name.lower() == args.get("first_name", "").lower()
                    and n.last_name.lower() == args.get("last_name", "").lower()
                    and a.zip == args.get("zip", "")):
                return uid
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--domain", default="retail")
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(1 if validate(a.domain) else 0)
