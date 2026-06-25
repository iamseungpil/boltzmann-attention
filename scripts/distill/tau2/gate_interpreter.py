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

# deny-message 우선순위 (원 RetailGate 의미 보존: notice→auth→ownership→confirm→preconditions).
_KIND_PRIORITY = {"notice": 0, "auth": 1, "ownership": 2, "confirm": 3, "preconditions": 4, "select_confirm": 5}

# 이미-행동(intermediate) status 토큰 — 정확-매칭 allow에 없어도 "use other tool"이 아니라
# "already acted, do not retry"로 steer해야 하는 상태(예: "pending (item modified)", "return requested").
_ACTED_TOKENS = ("modified", "requested", "cancelled", "canceled")

_A2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")


def pick_steer(gate, status):
    """preconditions deny 시 *현재 status 값*에 맞는 방향지시 선택 (도메인-일반·status는 A2 사실).

    ★blanket cross-tool 유도 금지: "pending (item modified)"는 'pending' 포함하지만 *이미-행동* 상태라
    'use modify' 유도가 틀림 → _acted 우선. (리뷰#3 버그 픽스.)
    """
    smap = (gate.get("steer_by_status_class") or {})
    s = (status or "").lower()
    if any(tok in s for tok in _ACTED_TOKENS):
        return smap.get("_acted", "")
    for key, msg in smap.items():
        if key.startswith("_"):
            continue
        if key.lower() in s:
            return msg
    return ""


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
        self.presented_select = False  # select_confirm: 후보집합 1회 제시 여부(중복방지)


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

    def _present_candidates(self, gate):
        """select_confirm: owned-entity 전체 후보를 명시 선택지로 제시 (도메인-일반·Probe-B 형식).
        >1 후보일 때만(disambiguation 필요·1개=false-friction 회피). A2:
        user_producer/orders_field=인증user의 후보 id 목록 · detail_producer/present_fields=각 후보 표시."""
        rf = self.resolvers.get("resolve_field")
        fr = self.resolvers.get("fetch_record")
        if not rf or not fr:
            return None
        ua = gate.get("user_id_arg", "user_id")
        ids = rf([ua, gate.get("user_producer"), gate.get("orders_field")], {ua: self.state.auth_user})
        if not ids or not isinstance(ids, (list, tuple)) or len(ids) <= 1:
            return None
        id_arg = gate.get("detail_id_arg", "order_id")
        fields = gate.get("present_fields") or []
        lines = []
        for cid in ids:
            rec = fr(gate.get("detail_producer"), id_arg, cid)
            if not isinstance(rec, dict):
                continue
            shown = {f: rec.get(f) for f in fields}
            lines.append(f"- {cid}: {json.dumps(shown, default=str, ensure_ascii=False)}")
        if not lines:
            return None
        head = gate.get("message", "DISAMBIGUATION CHECK — verify the target id matches the customer's request.")
        return head + "\n" + "\n".join(lines)

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

            elif kind == "preconditions":
                # write 실행 *전* target record의 status를 read-only resolver로 읽어 허용집합 membership 검사.
                # 못 읽으면(인자 부재·lookup 실패) deny 안 함 = false-block 회피(리뷰#2/R4).
                fn = self.resolvers.get("resolve_field")
                for chk in (g.get("checks") or []):
                    if tool_name not in (chk.get("applies_to") or []):
                        continue
                    path = chk.get("resolver_path")
                    if not fn or not path or not args.get(path[0]):
                        continue
                    cur = fn(path, args)
                    if cur is None:
                        continue
                    if cur not in (chk.get("allow") or []):
                        steer = pick_steer(g, cur)
                        return False, g["id"], (
                            f"[precondition] {tool_name} not permitted: this order's status is '{cur}' "
                            f"(required: {chk.get('allow')}). {steer} "
                            f"Do NOT retry {tool_name} on this order.").strip()

            elif kind == "select_confirm":
                # 절차-offload(측정 arm·[[05]] Q3=yes·flag-gated): 결정점서 owned-entity 후보집합을
                # 1회 명시 제시(Probe-B 형식) → 모델이 *재확인/재선택*(select=모델 몫·동결 아님).
                if self.state.auth_user is not None and not self.state.presented_select:
                    msg = self._present_candidates(g)
                    if msg:
                        self.state.presented_select = True
                        return False, g["id"], msg

        return True, None, None


def resolvers_from_env(env):
    """env(tau2 environment)서 결정론 read-only resolver 구성 — 도메인-일반(도구명/필드는 A2 resolver_path).

    resolve_field(path=[target_arg, producer_tool, field], args) -> value|None
      = producer_tool(target_arg=args[target_arg]) 호출(read-only getter·error budget 무소비) → field 읽기.
    ownership=owner_field 읽기, preconditions=status 읽기 = 동일 메커니즘(field만 다름). resolve_owner=호환 alias.
    ★status는 write 후 변하므로 캐시 금지(매 호출 fresh read·read-only라 안전).
    """
    tools = getattr(env, "tools", None)

    def resolve_field(path, args):
        target_arg, producer_tool, field = path[0], path[1], path[2]
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
            return out.get(field)
        return getattr(out, field, None)

    def fetch_record(producer, id_arg, id_val):
        """select_confirm 후보 표시용: producer(id_arg=id_val) → 전체 record(dict). read-only."""
        if tools is None or not id_val or not producer:
            return None
        fn = getattr(tools, producer, None)
        if fn is None:
            return None
        try:
            out = fn(**{id_arg: id_val})
        except Exception:
            return None
        if isinstance(out, dict):
            return out
        for m in ("model_dump", "dict"):
            f = getattr(out, m, None)
            if callable(f):
                try:
                    return f()
                except Exception:
                    pass
        return None

    return {"resolve_owner": resolve_field, "resolve_field": resolve_field, "fetch_record": fetch_record}


def candidate_summary(resolvers, gate, uid):
    """REPLAY-SAFE 후보 제시: owned-entity 후보를 *읽기-응답에 덧붙일* clean 요약으로 (deny 아님).
    읽기 tool은 evaluation replay서 skip되므로 content 증강이 안전(write-deny=replay 깨짐과 대조).
    >1 후보일 때만(disambiguation 필요). gate=select_confirm A2(producer-map). 도메인-일반."""
    rf = resolvers.get("resolve_field"); fr = resolvers.get("fetch_record")
    if not rf or not fr or not uid:
        return None
    ua = gate.get("user_id_arg", "user_id")
    ids = rf([ua, gate.get("user_producer"), gate.get("orders_field")], {ua: uid})
    if not ids or not isinstance(ids, (list, tuple)) or len(ids) <= 1:
        return None
    id_arg = gate.get("detail_id_arg", "order_id")
    fields = gate.get("present_fields") or []
    label = gate.get("present_label", "entity")  # A2: 도메인 명칭(order/reservation…)
    lines = []
    for cid in ids:
        rec = fr(gate.get("detail_producer"), id_arg, cid)
        if not isinstance(rec, dict):
            continue
        # ★도메인-일반: A2가 정한 present_fields를 *그대로* dump (필드 구조 해석 0·grep retail=0).
        shown = {f: rec.get(f) for f in fields} if fields else rec
        lines.append(f"- {cid}: {json.dumps(shown, default=str, ensure_ascii=False)}")
    if not lines:
        return None
    return (f"\n\n[DISAMBIGUATION NOTE — this customer's full {label} list]\n" + "\n".join(lines) +
            f"\nBefore any write, pick the {id_arg} matching the customer's request "
            "by comparing the fields above to what the customer described.")


def auth_satisfier_tools(gates):
    """A2 gates서 auth satisfier 도구 집합 도출 (= 구 AUTH_TOOLS·호환 export)."""
    s = set()
    for g in (gates or []):
        if g.get("kind") == "auth":
            s |= set((g.get("satisfiers") or {}).keys())
    return s
