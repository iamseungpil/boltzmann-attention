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


def notice_norm(s):
    """★C213/G1 (day8 032 [S]·경계정본 §3-1): notice 판정 정규화 — **닫힌 연산만**
    (소문자화·공백 압축·영숫자/공백 외 제거). 유사도/의미 매칭 금지(열린 술어 재도입 방지)."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s or "").lower())).strip()


def notice_sent_in(texts, notice_text, prefix=48):
    """★C213/G1: notice-송신 **공용 술어**(게이트 GB2·EPLAN grant·compliance 측정층 일원화 —
    032=", Sofia" 1토큰 개인화가 전문-일치를 영구 불충족·EPLAN(prefix48)과 술어 불일치 [S]).
    판정=정규화 후 앞 prefix자 부분문자열. notice_text 부재=None(판단 불가)."""
    key = notice_norm(notice_text)[:prefix]
    if not key:
        return None
    return any(key in notice_norm(t) for t in texts if isinstance(t, str))


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
    """A2 로드 = **3층 병합**. 없으면 None(=게이트 비활성).

    ★층 정의(2026-07-31 사용자 지시): capex/opex를 정직하게 계상하려면 "새 도메인마다 새로
    써야 하는 내용"이 **파일 경계로** 갈려야 한다.
      L1 `a2/base/shared.json`        — 벤치마크 공통·수정 0으로 그대로 씀 → 새 도메인 비용 **0**
      L2 `a2/<domain>.settings.json`  — **구조는 동일**하고 값만 도메인별 → 템플릿 채우기
      L3 `a2/<domain>.specific.json`  — 그 도메인에만 있는 도구·규칙 → **저작 + 엔진 접속**

    병합 순서 L1 → L2 → L3 (뒤가 앞을 덮는다) ⇒ 기존 동작 불변.
    `<domain>.gate.json`은 레거시 read-site(105곳) 호환용 **생성물**이며, 분리 파일이 있으면
    그쪽이 정본이다. 둘이 갈라지지 않는지는 `x18_a2_three_layer.py --verify`가 강제한다.
    정본 설계 = `A2_THREE_LAYER_SPLIT_DESIGN_2026_07_31.md`.
    """
    def _read(p):
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    merged = {}
    base = _read(os.path.join(_A2_DIR, "base", "shared.json"))
    if base:
        merged.update({k: v for k, v in base.items() if not k.startswith("_")})
    settings = _read(os.path.join(_A2_DIR, f"{domain}.settings.json"))
    specific = _read(os.path.join(_A2_DIR, f"{domain}.specific.json"))
    if settings is None and specific is None:
        dom = _read(os.path.join(_A2_DIR, f"{domain}.gate.json"))   # 분리 전 체크아웃 대비
        if dom is None:
            return None
        merged.update(dom)
        return merged
    for part in (settings, specific):
        if part:
            merged.update(part)
    _compose_claim_audit(merged)
    _resolve_a3_refs(merged)
    _spread_name_rules(merged)
    return merged


def _spread_name_rules(merged):
    """최상위 `name_rules` 를 `derived` 의 formalize 노드 params 에 심는다 — **적재 시점 한 곳**.

    왜 여기인가 (2026-08-08·C335): 이름 대조 규칙이 필요한 노드가 둘이다 — 주어 하나를 고르는
    `focus`(shape term)와, 원장 행의 그룹 필드를 A3 주어로 옮겨 쓰는 전사(shape rows·`align_field`).
    노드마다 같은 문장을 적으면 **두 벌이 되어 갈린다**(`_resolve_a3_refs` 가 같은 이유로 여기 있다).
    A2 는 한 번 쓰고, 노드가 이미 자기 값을 지녔으면 그것을 존중한다.
    """
    rules = merged.get("name_rules")
    if not rules:
        return
    for n in (merged.get("derived") or []):
        if n.get("op") != "formalize":
            continue
        p = dict(n.get("params") or {})
        p.setdefault("name_rules", rules)
        n["params"] = p


def _resolve_a3_refs(merged):
    """A2 안의 `{"a3": [축, 주어]}` 참조를 **A3 온톨로지의 값**으로 바꾼다.

    왜 적재 시점 한 곳인가: 같은 정책 상수가 두 A2 선언에 **각각 리터럴로** 박혀 있었고(A3 행까지
    하면 원천 셋), 값이 갈리면 어느 것이 쓰였는지 나중에 귀속이 안 된다. 소비자마다 A3를 읽게 하면
    그 코드가 소비자 수만큼 생긴다 — **여기서 한 번 푼다.** 그러면 A2를 읽는 모든 코드가
    **손대지 않고** 같은 값을 본다.

    ⚠**문자열 매칭을 하지 않는다**(사용자 지적 2026-08-08). 온톨로지는 형식화된 포맷이고 참조는
    그 안의 키다 ⇒ **정확 일치만**. 특히 대소문자를 접으면 안 된다 — 설계서 §9-4가 표기마다 행을
    따로 두라고 한 그 표기들이 **한 칸으로 뭉개지고 하나가 조용히 사라진다**.
    ⚠못 찾으면 **죽는다**. 조용히 기본값을 쓰면 그 순간 원천이 다시 둘이 된다.
    """
    onto = merged.get("policy_ontology") or {}
    rows = onto.get("rows") or []
    if not rows:
        return
    table = {}
    for r in rows:
        table.setdefault((r.get("axis"), r.get("subject")), r.get("value"))

    def walk(node):
        if isinstance(node, dict):
            ref = node.get("a3")
            if isinstance(ref, list) and len(ref) == 2 and len(node) == 1:
                key = (ref[0], ref[1])
                if key not in table:
                    raise KeyError("A3 참조를 풀 수 없다: %r" % (ref,))
                return table[key]
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    for k in list(merged):
        if k == "policy_ontology":
            continue
        merged[k] = walk(merged[k])


def _compose_claim_audit(merged):
    """★L1 산문 + L2 결합 → 구 `claim_prov`/`completion_guard`(2026-07-31 승격).

    두 키는 [[23]] 감사에서 **정책 근거가 없다**고 확정된 유이한 것인데, gold 경유도 아니었다 —
    담고 있는 것이 banking 사실이 아니라 **도메인-일반 무결성 원리**("한 일을 했다고 말했으면
    실행 원장과 대조한다")여서다. 산문은 L1(`base/shared.json`·새 도메인 비용 0), 이 도메인의
    결합 5개(kinds·kind_guidance·event_map·reserve_kinds·user_execution_tool)만 L2에 둔다.
    ★`kind_guidance`는 등가 테스트 §③이 잡아냈다 — 산문에 kind 용어집이 박혀 있어
      도메인-불변이 아니었다. 게이트가 없었으면 그대로 승격했을 것이다.

    ★소비자 코드는 하나도 바꾸지 않는다 — 여기서 **구 형태와 바이트 동일한 dict**를 만든다.
    등가는 `test_claim_promotion.py`가 강제한다. 정본 =
    `CLAIM_AUDIT_ENGINE_PROMOTION_DESIGN_2026_07_31.md`.
    """
    ca, cb = merged.get("claim_audit"), merged.get("claim_bindings")
    if not (isinstance(ca, dict) and isinstance(cb, dict)):
        return                      # 미선언 도메인 = 레버 skip(U2′ 안전측·retail/airline 현행)
    merged.setdefault("claim_prov", {
        "question": (ca["question"].replace("{kinds}", cb["kinds"])
                     .replace("{kind_guidance}", cb.get("kind_guidance", ""))),
        "feedback_pending": ca["feedback_pending"],
        "event_map": cb["event_map"],
        "feedback": ca["feedback"],
        "reserve_kinds": cb["reserve_kinds"],
        "feedback_unavailable": ca["feedback_unavailable"],
    })
    merged.setdefault("completion_guard", {
        "user_execution_tool": cb["user_execution_tool"],
        "claim_question": ca["completion_question"],
        "feedback": ca["completion_feedback"],
    })


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

    @staticmethod
    def _gate_applies(g, tool_name, args):
        """applies_to 멤버십 + (선택) applies_when arg-조건 — 도메인-일반 결정론 멤버십 검사.
        applies_when: {"arg": <인자명>, "in": [...]} 또는 {"arg": ..., "not_in": [...]} (값 목록=A2 도메인 사실).
        용례: 디스패처형 도구(예: 이름-인자로 내부 도구를 고르는 wrapper)에서 일부 내부 대상만 게이트.
        인자 부재 시 조건 불성립으로 보아 게이트 적용(보수) — 단 not_in만 있으면 부재=적용."""
        if tool_name not in g.get("applies_to", []):
            return False
        aw = g.get("applies_when")
        if aw:
            v = str((args or {}).get(aw.get("arg")) or "")
            if "in" in aw and v not in set(aw["in"]):
                return False
            if "not_in" in aw and v in set(aw["not_in"]):
                return False
        return True

    def check(self, tool_name, args, last_user_msg=None, transfer_msg_sent=None):
        """returns (allowed, gate_id|None, reason|None).
        last_user_msg=None → confirm skip(replay). transfer_msg_sent=None → notice skip."""
        args = args or {}
        for g in self.gates:
            if not self._gate_applies(g, tool_name, args):
                continue
            kind = g.get("kind")

            if kind == "notice":
                # ★NOTICE-PERGATE(2026-07-11·NEXT_LEVER_GEN §1.1①): callable이면 per-gate 평가
                #   (그 게이트의 notice_text로 송신 여부 계산·다중 notice 공존 가능) /
                #   스칼라(bool·None)면 현행과 바이트-동일(None=skip·False=deny·True=allow).
                #   notice_text = A2 데이터 — 도메인 리터럴 0 불변식 유지.
                sent = (transfer_msg_sent(g.get("notice_text"))
                        if callable(transfer_msg_sent) else transfer_msg_sent)
                if sent is False:
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
    엔진=general op{count_where·count·sum·lookup·argmax_where·argmin_where·most_recent}만
    ·nested_field/cond_field/item_field/rank_field/id_field/date_field=A2 calc_specs(retail 필드 0).
    CALC-EXT(CENSUS_LEVERS_DESIGN_2026_07_11 §2a v1.1): argmax_where/argmin_where/most_recent 3종 추가
    (pairwise_diff_sum은 v1.1 리뷰서 confirm-시점 notice 채널로 이동 — 여기 구현 안 함).
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
        elif op in ("argmax_where", "argmin_where") and items is not None:
            # CALC-EXT: cond_field==cond_value 후보 중 rank_field 최대/최소 원소(동률=전부 나열).
            # 전 필드명=A2 spec 인자(도메인일반). cond_field 생략 시 무필터.
            cf, cv = sp.get("cond_field"), sp.get("cond_value")
            rk = sp.get("rank_field")
            idf = sp.get("id_field", "id")
            ranked = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                if cf is not None and it.get(cf) != cv:
                    continue
                try:
                    ranked.append((float(it.get(rk)), it))
                except (TypeError, ValueError):
                    continue
            if ranked:
                best = (max if op == "argmax_where" else min)(r[0] for r in ranked)
                val = "; ".join(f"{idf}={it.get(idf)} ({rk}={it.get(rk)})"
                                for r, it in ranked if r == best)
        elif op == "most_recent" and items is not None:
            # CALC-EXT: date_field(문자열 정렬가능 날짜) 최대 원소의 id_field(동률=전부 나열).
            # ⚠ retail A2엔 스펙 미부착 — comp gz 456 sim 전수서 어떤 tool 출력에도 날짜 필드 부재
            #   (calcext_offline_census.py로 확인) → 트리거 불가. op 자체는 도메인일반으로 유지.
            df = sp.get("date_field")
            idf = sp.get("id_field", "id")
            dated = [(str(it.get(df)), it) for it in items
                     if isinstance(it, dict) and it.get(df) is not None]
            if dated:
                best = max(d for d, _ in dated)
                val = "; ".join(f"{idf}={it.get(idf)} ({df}={d})"
                                for d, it in dated if d == best)
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
