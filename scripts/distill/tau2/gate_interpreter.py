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
_KIND_PRIORITY = {"notice": 0, "auth": 1, "ownership": 2, "confirm": 3, "preconditions": 4,
                  "constraints": 4.5, "select_confirm": 5, "exhaust_before_escalate": 6}

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
        # kind=exhaust_before_escalate (측정 arm·E1 Phase B). 후보 엔티티=DB 결정론 enumerate.
        self.inspected = set()      # 모델이 실제로 조회한 엔티티 id


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
        """satisfier 도구 성공 → 인증 확립 (kind=auth·satisfiers 키).
        + kind=exhaust_before_escalate: 후보 엔티티 집합 수집 / 검사(read) 이력 기록. 전부 A2-구동."""
        if not ok:
            return
        args = args or {}
        for g in self.gates:
            if g.get("kind") == "auth" and tool_name in (g.get("satisfiers") or {}):
                if isinstance(result, str) and result:
                    self.state.auth_user = result

            elif g.get("kind") == "exhaust_before_escalate":
                insp = g.get("inspect") or {}
                if tool_name == insp.get("tool"):
                    v = args.get(insp.get("arg"))
                    if v:
                        self.state.inspected.add(str(v))

    def _exhaust_remaining(self, gate):
        """후보 엔티티 = 인증 사용자에 대한 **DB 결정론 enumerate**(resolve_field·read-only).
        enumerate 불가(미인증/리졸버 없음) → 빈 집합 = 보수적 OFF(미발화)."""
        uid = self.state.auth_user
        rf = (self.resolvers or {}).get("resolve_field")
        src = gate.get("entity_source") or {}
        path = src.get("resolver_path")
        if not uid or not rf or not path:
            return set()
        ids = rf(path, {src.get("user_id_arg", "user_id"): uid})
        if not isinstance(ids, (list, tuple, set)):
            return set()
        return {str(i) for i in ids} - self.state.inspected

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

            elif kind == "exhaust_before_escalate":
                # 측정 arm(E1 Phase B): escalate 도구를, 후보 엔티티를 다 *읽기 전에는* deny.
                # 강제하는 행동 = 읽기(멱등·무해)뿐. 행동(write) 선택은 여전히 모델 몫.
                rem = self._exhaust_remaining(g)
                if rem:
                    return False, g["id"], render_recovery(
                        g, detail=f"{len(rem)} not yet inspected: {', '.join(sorted(rem)[:5])}")

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

            elif kind == "constraints":
                # operation-semantic 정책 불변식(순수-args로 decidable·env-mirror). 무효 write 사전차단+steer.
                # 엔진=general op{disjoint·equal_len}·필드=A2(retail 0). 둘 다 env가 이미 강제=false-block 0.
                # (member_of/payment=파생필드 [[05]] 위험으로 미포함·정적 deprioritize·CONSTRAINT_GATE_DESIGN §accounting.)
                for chk in (g.get("checks") or []):
                    if tool_name not in (chk.get("applies_to") or []):
                        continue
                    op = chk.get("op")
                    f1, f2 = chk.get("fields", [None, None])
                    v1, v2 = args.get(f1), args.get(f2)
                    if op == "disjoint":
                        if v1 and v2 and (set(map(str, v1)) & set(map(str, v2))):
                            return False, g["id"], chk.get("steer")
                    elif op == "equal_len":
                        if v1 is not None and v2 is not None and len(v1) != len(v2):
                            return False, g["id"], chk.get("steer")

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


def nested_candidate_summary(output_record, spec):
    """REPLAY-SAFE operand 후보 제시(L2 item / L3 variant): read 응답 record 안의 nested
    list/dict를 *명시 choice-set*으로 (deny 아님·읽기증강). order.items·product.variants류.
    spec(A2): {nested_field, id_field, fields, label}. 도메인-일반 — 필드 구조 해석 0·grep retail=0.
    >1 후보일 때만(disambiguation 필요·단일이면 noise)."""
    if not isinstance(output_record, dict):
        return None
    nested = output_record.get(spec.get("nested_field"))
    if isinstance(nested, dict):
        items = list(nested.values())
    elif isinstance(nested, list):
        items = nested
    else:
        return None
    if len(items) <= 1:
        return None
    id_field = spec.get("id_field", "id")
    fields = spec.get("fields") or []
    label = spec.get("label", "option")
    lines = []
    for rec in items:
        if not isinstance(rec, dict):
            continue
        cid = rec.get(id_field)
        shown = ({f: rec.get(f) for f in fields} if fields
                 else {k: v for k, v in rec.items() if k != id_field})
        lines.append(f"- {id_field}={cid}: {json.dumps(shown, default=str, ensure_ascii=False)}")
    if not lines:
        return None
    return (f"\n\n[OPERAND DISAMBIGUATION — every {label} with its {id_field}]\n" + "\n".join(lines) +
            f"\nWhen the action needs a {id_field}, copy the EXACT {id_field} above for the {label} "
            "the customer described (match by the fields shown). Never guess, invent, or carry an "
            f"{id_field} from a different {label}.")


def compute_facts(record, specs):
    """REPLAY-SAFE 결정론 집계 주입(calc_NL offload·measurement arm): read record서 A2-spec aggregate 계산.
    엔진=general op{count_where·count·sum·lookup}만·nested_field/cond_field/item_field=A2 calc_specs(retail 필드 0).
    모델이 *틀리는* 산술/집계(available 필터·총액)를 결정론 계산·주입 → 보고는 모델(report-conversion 측정).
    반환=주입 텍스트(파싱가능 'label: value' = report-conversion census 마커)·없으면 None."""
    if not isinstance(record, dict):
        return None
    out = []
    for sp in (specs or []):
        op = sp.get("op")
        label = sp.get("label", "value")
        nf = sp.get("nested_field")
        coll = record.get(nf) if nf else record
        items = list(coll.values()) if isinstance(coll, dict) else (coll if isinstance(coll, list) else None)
        val = None
        if op == "count_where" and items is not None:
            cf, cv = sp.get("cond_field"), sp.get("cond_value")
            val = sum(1 for it in items if isinstance(it, dict) and it.get(cf) == cv)
        elif op == "count" and items is not None:
            val = len(items)
        elif op == "sum" and items is not None:
            itf = sp.get("item_field")
            try:
                val = round(sum(float(it.get(itf, 0)) for it in items if isinstance(it, dict)), 2)
            except (TypeError, ValueError):
                val = None
        elif op == "lookup":
            val = record.get(sp.get("field"))
        if val is not None:
            out.append(f"- {label}: {val}")
    if not out:
        return None
    return ("\n\n[COMPUTED FACTS — deterministic; when you report any of these, use these EXACT values]\n"
            + "\n".join(out))


def auth_satisfier_tools(gates):
    """A2 gates서 auth satisfier 도구 집합 도출 (= 구 AUTH_TOOLS·호환 export)."""
    s = set()
    for g in (gates or []):
        if g.get("kind") == "auth":
            s |= set((g.get("satisfiers") or {}).keys())
    return s


def _try_json(s):
    if isinstance(s, dict):
        return s
    try:
        import json as _j
        return _j.loads(s)
    except Exception:
        return None


def observe_tools(gates):
    """observe()가 상태를 갱신해야 하는 도구 집합 (auth satisfier + exhaust의 entity_source/inspect).
    도메인-일반: 이름은 전부 A2서 읽음."""
    s = auth_satisfier_tools(gates)
    for g in (gates or []):
        if g.get("kind") == "exhaust_before_escalate":
            t = (g.get("entity_source") or {}).get("tool")
            if t:
                s.add(t)
            t = (g.get("inspect") or {}).get("tool")
            if t:
                s.add(t)
    return s
