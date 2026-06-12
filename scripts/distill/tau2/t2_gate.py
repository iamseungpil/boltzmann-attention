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
    """Guard-2 동형 2-pass 검증 (G2는 gold에 user 발화 없어 live 전용).

    ⚠️ gold actions는 DB-state 보상에 필요한 액션만 담아 인증 READ가 생략될 수 있음
    (46/114 실측) — 단순 replay의 G1 deny는 observed-proxy 아티팩트라 검증으로 무효.
    Pass A (G1 순서): gold에 auth가 *있는* 태스크에서 auth가 user-scoped보다 선행하는지.
    Pass B (G3 스코프): GT 유저(gold 인자의 order owner·user_id 합의)로 auth 시드 후
                        타-유저 deny=0 확인. GT 합의 실패(2-유저 태스크)는 별도 보고.
    """
    import importlib
    mod = importlib.import_module(f"tau2.domains.{domain}.environment")
    env = mod.get_environment()
    db = env.tools.db
    tasks = mod.get_tasks(None)

    g1_violation, g3_over, multi_user, no_auth_gold = [], [], [], []
    n_actions = 0
    for t in tasks:
        actions = (t.evaluation_criteria.actions if t.evaluation_criteria else None) or []
        if not actions:
            continue
        n_actions += len(actions)
        # GT 유저 유도: user_id 인자 + order_id -> db owner 의 합의
        gt = set()
        for a in actions:
            args = a.arguments or {}
            if args.get("user_id"):
                gt.add(args["user_id"])
            if args.get("order_id"):
                owner = db.orders.get(args["order_id"])
                if owner is not None:
                    gt.add(owner.user_id)
        if len(gt) > 1:
            multi_user.append((t.id, sorted(gt)))
            continue
        gt_user = gt.pop() if gt else None

        has_auth = any(a.name in AUTH_TOOLS for a in actions)
        if not has_auth:
            no_auth_gold.append(t.id)
        # Pass A: auth 포함 gold에서 순서 위반 (auth 이전의 user-scoped 호출)
        if has_auth:
            gate = RetailGate(db=db, enable_g2=False)
            for a in actions:
                ok, g, why = gate.check(a.name, a.arguments or {})
                if not ok and g == "G1_AUTH_FIRST":
                    g1_violation.append((t.id, a.name))
                if a.name in AUTH_TOOLS:
                    gate.observe(a.name, a.arguments,
                                 _resolve_find(db, a.name, a.arguments or {}))
        # Pass B: auth 시드 후 G3 over-deny
        gate = RetailGate(db=db, enable_g2=False)
        gate.auth_user = gt_user
        for a in actions:
            ok, g, why = gate.check(a.name, a.arguments or {})
            if not ok:
                g3_over.append((t.id, a.name, g, why))

    print(f"[validate] tasks={len(tasks)} gold_actions={n_actions}")
    print(f"  PassA G1-ordering violations = {len(g1_violation)} (expect 0)")
    print(f"  PassB G3 over-deny           = {len(g3_over)} (expect 0)")
    print(f"  gold-without-auth (bench property, live G1 무관) = {len(no_auth_gold)}")
    print(f"  multi-user GT (단일-유저 정책과 긴장 — 점검 대상) = {len(multi_user)}")
    for row in (g1_violation + g3_over)[:20]:
        print("  FAIL:", row)
    for row in multi_user[:10]:
        print("  MULTI:", row)
    return len(g1_violation) + len(g3_over)


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
