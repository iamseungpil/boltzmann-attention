#!/usr/bin/env python
"""tau2 retail 바인딩 + GT-validate (A2-구동·키스톤 후).

★엔진 = `gate_interpreter.py`(GateInterpreter·벤치-일반·도메인 분기 0). 이 파일은
  (1) retail A2(`a2/retail.gate.json`) 바인딩 + 호환 export(AUTH_TOOLS 등·분석도구용),
  (2) GT replay 검증(validate·Guard-2 동형),
  (3) deprecated 호환 alias RetailGate(외부 호출자 없음·안전망).
도구셋/게이트는 전부 A2서 *도출* — 코드 하드코딩 리터럴 폐기(2026-06-21 키스톤).

Gate 4 kind (policy.md → A2 컴파일):
  auth(G1)      user-scoped 도구는 인증(satisfier 성공) 선행 필수
  confirm(G2)   write 도구는 직전 user 턴이 명시 확인(yes)일 때만 (live 전용)
  ownership(G3) 인증 유저 외 타 유저/타 주문 대상 호출 deny
  notice(G4)    transfer 후 고정 문구 송신 (live 전용)

검증 (Guard-2 동형): --validate 가 gold actions를 replay, auth+ownership over-deny=0 확인.

Run: cd /home/woori/scratch/tau2-bench && PYTHONPATH=src \
  /home/woori/venvs/seka_env/bin/python $REPO/scripts/distill/tau2/t2_gate.py --validate
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import (  # noqa: E402
    CONFIRM_RE, GateInterpreter, auth_satisfier_tools, load_domain_a2,
    render_recovery as _render_gate, resolvers_from_env)

_DOMAIN = "retail"
_A2 = load_domain_a2(_DOMAIN) or {"gates": []}
_GATES = _A2["gates"]

# ── A2-도출 호환 export (구 module-global 리터럴 폐기 — 전부 A2서 도출) ──
GATE_SPEC = {g["id"]: g for g in _GATES}                       # 구 dict 호환 (A2 유래)
AUTH_TOOLS = auth_satisfier_tools(_GATES)                      # auth satisfier 도구
WRITE_TOOLS = {t for g in _GATES if g.get("kind") == "confirm" for t in g["applies_to"]}
USER_SCOPED = {t for g in _GATES if g.get("kind") == "auth" for t in g["applies_to"]}
TRANSFER_MSG = next((g.get("notice_text") for g in _GATES if g.get("kind") == "notice"), "")
# CONFIRM_RE = gate_interpreter.CONFIRM_RE (re-export)


def render_recovery(gate_name, spec, detail=""):
    """구 시그니처 호환 wrapper → 엔진 render_recovery(spec, detail)."""
    return _render_gate(spec, detail)


def _db_resolvers(db):
    """retail db 기반 ownership resolver (validate/compat용·env 없을 때).
    resolve_owner([target_arg, _, owner_field], args) → db.orders[val].owner_field."""
    def resolve_owner(path, args):
        val = args.get(path[0])
        if not val or db is None:
            return None
        rec = db.orders.get(val)
        return getattr(rec, path[2], None) if rec is not None else None
    return {"resolve_owner": resolve_owner}


class RetailGate:
    """deprecated 호환 alias — GateInterpreter(retail A2) 래퍼. (외부 호출자 없음·안전망)"""

    def __init__(self, db=None, enable_g2=True):
        self.db = db
        self._gi = GateInterpreter(_GATES, resolvers=_db_resolvers(db), enable_g2=enable_g2)

    @property
    def auth_user(self):
        return self._gi.auth_user

    @auth_user.setter
    def auth_user(self, v):
        self._gi.auth_user = v

    def observe(self, tool_name, args, result, ok=True):
        self._gi.observe(tool_name, args, result, ok)

    def check(self, tool_name, args, last_user_msg=None, transfer_msg_sent=None):
        return self._gi.check(tool_name, args, last_user_msg=last_user_msg,
                              transfer_msg_sent=transfer_msg_sent)


def check_transfer_msg(messages_after_transfer):
    """G4 post-hoc: transfer 호출 후 고정 문구 송신 여부."""
    return any(TRANSFER_MSG in (m or "") for m in messages_after_transfer)


def validate(domain="retail"):
    """Guard-2 동형 2-pass 검증 (G2는 gold에 user 발화 없어 live 전용).

    Pass A (auth 순서): gold에 auth가 *있는* 태스크에서 auth가 user-scoped보다 선행하는지.
    Pass B (ownership 스코프): GT 유저로 auth 시드 후 타-유저 deny=0 확인.
    """
    import importlib
    mod = importlib.import_module(f"tau2.domains.{domain}.environment")
    env = mod.get_environment()
    db = env.tools.db
    tasks = mod.get_tasks(None)
    gates = (load_domain_a2(domain) or {"gates": []})["gates"]
    auth_tools = auth_satisfier_tools(gates)
    resolvers = resolvers_from_env(env)
    # GT 유저 도출 = A2 ownership 게이트(owner_field + resolver_path) 구동 — 도메인-일반(retail order·airline reservation 동일).
    own_gate = next((g for g in gates if g.get("kind") == "ownership"), {})
    owner_field = own_gate.get("owner_field", "user_id")
    resolver_path = own_gate.get("resolver_path")
    resolve_owner = resolvers.get("resolve_owner")

    kind_by_id = {g.get("id"): g.get("kind") for g in gates}
    g1_violation, g3_over, multi_user, no_auth_gold = [], [], [], []
    g5_precond = []  # preconditions gold-replay deny = 도구 precondition과 정확-정렬→benign(gold action-set 대안)
    n_actions = 0
    for t in tasks:
        actions = (t.evaluation_criteria.actions if t.evaluation_criteria else None) or []
        if not actions:
            continue
        n_actions += len(actions)
        gt = set()
        for a in actions:
            args = a.arguments or {}
            if args.get(owner_field):
                gt.add(args[owner_field])
            if resolver_path and resolve_owner and args.get(resolver_path[0]):
                owner = resolve_owner(resolver_path, args)
                if owner is not None:
                    gt.add(owner)
        if len(gt) > 1:
            multi_user.append((t.id, sorted(gt)))
            continue
        gt_user = gt.pop() if gt else None

        has_auth = any(a.name in auth_tools for a in actions)
        if not has_auth:
            no_auth_gold.append(t.id)
        if has_auth:
            gate = GateInterpreter(gates, resolvers=resolvers, enable_g2=False)
            for a in actions:
                ok, g, why = gate.check(a.name, a.arguments or {})
                if not ok and g and g.startswith("G1"):
                    g1_violation.append((t.id, a.name))
                if a.name in auth_tools:
                    gate.observe(a.name, a.arguments,
                                 _resolve_find(db, a.name, a.arguments or {}))
        gate = GateInterpreter(gates, resolvers=resolvers, enable_g2=False)
        gate.auth_user = gt_user
        for a in actions:
            ok, g, why = gate.check(a.name, a.arguments or {})
            if ok:
                continue
            if kind_by_id.get(g) == "preconditions":
                # gold replay서 preconditions deny = 도구 precondition과 정확-정렬(false-block 불가).
                # gold action-set이 대안(예: 같은 주문에 exchange+modify 동시기재)을 담을 때 발생 = benign.
                g5_precond.append((t.id, a.name, why))
            else:
                g3_over.append((t.id, a.name, g, why))

    print(f"[validate] domain={domain} tasks={len(tasks)} gold_actions={n_actions}")
    print(f"  PassA auth-ordering violations = {len(g1_violation)} (expect 0)")
    print(f"  PassB ownership over-deny      = {len(g3_over)} (expect 0)")
    print(f"  G5 precond gold-replay deny    = {len(g5_precond)} (benign·도구-정렬·gold 대안 action)")
    print(f"  gold-without-auth (bench property, live auth 무관) = {len(no_auth_gold)}")
    print(f"  multi-user GT (단일-유저 정책과 긴장 — 점검 대상) = {len(multi_user)}")
    for row in (g1_violation + g3_over)[:20]:
        print("  FAIL:", row)
    for row in g5_precond[:10]:
        print("  G5-precond:", row[0], row[1])
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
    ap.add_argument("--render", action="store_true", help="생성 메시지 미리보기")
    a = ap.parse_args()
    if a.render:
        gates = (load_domain_a2(a.domain) or {"gates": []})["gates"]
        for g in gates:
            print(f"--- {g['id']} ({g['kind']})\n{_render_gate(g)}")
    if a.validate:
        raise SystemExit(1 if validate(a.domain) else 0)
