#!/usr/bin/env python
"""GateInterpreter — 벤치-일반 결정론 게이트 엔진 (FIXED·절대 도메인 분기 0).

도구멤버십·정책은 전부 A2(`a2/<domain>.gate.json`)서 *읽고*, 집행 로직(유한 gate-kind)만 고정.
authority = GATE_INTERPRETER_UNIFICATION_DESIGN_2026_06_18 + A2_DRIVEN_SCAFFOLD_KEYSTONE_IMPL_2026_06_21.

유한 gate-kind closure (layer-B): auth / confirm / ownership / notice  (preconditions=SOP는 후속).
새 도메인 = `a2/<domain>.gate.json`만 컴파일 → 게이트·메시지·autofetch가 따라옴. 코드 수정 0.

⛔ 이 파일에 `if domain`/`if bench`/도구명 하드코딩 = 0 (검증: A2_DRIVEN..IMPL §4).
"""
import json
import os
import re

CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|sure|confirm|confirmed|correct|proceed|go ahead|ok(ay)?|sounds good|"
    r"please do|that works|do it)\b", re.I)

# deny-message 우선순위 (원 RetailGate 의미 보존: notice→auth→ownership→confirm).
_KIND_PRIORITY = {"notice": 0, "auth": 1, "ownership": 2, "confirm": 3, "preconditions": 4}

_A2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")


def load_domain_a2(domain):
    """a2/<domain>.gate.json 로드. 없으면 None(=게이트 비활성)."""
    path = os.path.join(_A2_DIR, f"{domain}.gate.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_recovery(gate, detail=""):
    """R3-side 불변 템플릿: 게이트 spec(A2 산출물) -> 복구 메시지. 도메인 문자열 없음."""
    head = f"blocked by policy gate: {gate['predicate']} not established"
    if gate.get("note"):
        head += f" ({gate['note']})"
    if detail:
        head += f" [{detail}]"
    if gate.get("terminal"):
        return f"{head}. This cannot be satisfied — {gate['terminal']}."
    steps = ["(1) do NOT retry this tool now"]
    if gate.get("satisfiers"):
        asks = " OR ".join(", ".join(v) for v in gate["satisfiers"].values())
        calls = " or ".join(f"{t}({', '.join(v)})" for t, v in gate["satisfiers"].items())
        steps += [f"(2) ask the user for: {asks}",
                  f"(3) call {calls} with that info",
                  "(4) once it succeeds, retry the original action"]
    elif gate.get("ask"):
        steps += [f"(2) {gate['ask']}",
                  "(3) once this is done, retry the original action"]
    return f"{head}. Recovery: " + "; ".join(steps)


class GateState:
    def __init__(self):
        self.auth_user = None       # 인증 확립된 user id
        self.notice_sent = False    # notice 고정문구 송신 여부(=transfer_msg_sent)


class GateInterpreter:
    """대화-수준 결정론 게이트. 실행 *전* check(), 실행 *후* observe().

    gates    : A2 list (도메인 swap = 이 데이터만 교체).
    resolvers: 엔진 제공 결정론 lookup. ownership용 resolve_owner(resolver_path, args)->owner|None.
    """

    def __init__(self, gates, resolvers=None, enable_g2=True):
        self.gates = sorted(gates or [], key=lambda g: _KIND_PRIORITY.get(g.get("kind"), 9))
        self.resolvers = resolvers or {}
        self.enable_g2 = enable_g2
        self.state = GateState()

    # ── 호환 프로퍼티 (분석도구가 .auth_user 직접 참조) ──
    @property
    def auth_user(self):
        return self.state.auth_user

    @auth_user.setter
    def auth_user(self, v):
        self.state.auth_user = v

    def observe(self, tool_name, args, result, ok=True):
        """satisfier 도구 성공 → 인증 확립 (kind=auth·satisfiers 키)."""
        if not ok:
            return
        for g in self.gates:
            if g.get("kind") == "auth" and tool_name in (g.get("satisfiers") or {}):
                if isinstance(result, str) and result:
                    self.state.auth_user = result

    def _resolve_owner(self, gate, args):
        """ownership: 직접 owner_field 인자 또는 resolver_path 도출 owner. (owner|None)."""
        owner_field = gate.get("owner_field", "user_id")
        # (a) 직접: 호출 인자에 owner_field가 있으면 그 값
        direct = args.get(owner_field)
        if direct:
            return direct
        # (b) 간접: resolver_path[target_arg, producer, owner_field] → 엔진 lookup
        path = gate.get("resolver_path")
        fn = self.resolvers.get("resolve_owner")
        if path and fn and args.get(path[0]):
            return fn(path, args)
        return None

    def check(self, tool_name, args, last_user_msg=None, transfer_msg_sent=None):
        """returns (allowed, gate_id|None, reason|None).
        last_user_msg=None → confirm skip(replay). transfer_msg_sent=None → notice skip."""
        args = args or {}
        for g in self.gates:
            if tool_name not in g.get("applies_to", []):
                continue
            kind = g.get("kind")

            if kind == "notice":
                if transfer_msg_sent is False:
                    return False, g["id"], render_recovery(g)

            elif kind == "auth":
                if self.state.auth_user is None:
                    return False, g["id"], render_recovery(g)

            elif kind == "ownership":
                if self.state.auth_user is not None:
                    owner = self._resolve_owner(g, args)
                    if owner is not None and owner != self.state.auth_user:
                        return False, g["id"], render_recovery(
                            g, detail=f"target owner {owner} != authenticated {self.state.auth_user}")

            elif kind == "confirm":
                if self.enable_g2 and last_user_msg is not None:
                    if not CONFIRM_RE.search(last_user_msg):
                        return False, g["id"], render_recovery(g)

        return True, None, None


def resolvers_from_env(env):
    """env(tau2 environment)서 ownership resolver 구성 — 도메인-일반(도구명/필드는 A2 resolver_path).

    resolve_owner(resolver_path=[target_arg, producer_tool, owner_field], args) -> owner|None
      = producer_tool(target_arg=args[target_arg]) 호출(read-only getter·error budget 무소비) → owner_field 읽기.
    """
    tools = getattr(env, "tools", None)

    def resolve_owner(path, args):
        target_arg, producer_tool, owner_field = path[0], path[1], path[2]
        val = args.get(target_arg)
        if not val or tools is None:
            return None
        fn = getattr(tools, producer_tool, None)
        if fn is None:
            return None
        try:
            out = fn(**{target_arg: val})
        except Exception:
            return None
        # pydantic obj 또는 dict 모두 지원
        if isinstance(out, dict):
            return out.get(owner_field)
        return getattr(out, owner_field, None)

    return {"resolve_owner": resolve_owner}


def auth_satisfier_tools(gates):
    """A2 gates서 auth satisfier 도구 집합 도출 (= 구 AUTH_TOOLS·호환 export)."""
    s = set()
    for g in (gates or []):
        if g.get("kind") == "auth":
            s |= set((g.get("satisfiers") or {}).keys())
    return s
