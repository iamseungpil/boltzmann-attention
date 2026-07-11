#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""t2_eplan_patch.py — E-PLAN(plan/execute e2e) 결정론 로직 + 배선 스텁 (step c).

정본 설계 = `reports/facet_rft_2026/E_PLAN_LIVE_WIRING_DESIGN_2026_07_11.md` **v1.2**.
3컴포넌트: [CP0 plan-seed] → [discovery-enforce L1/L2] → [CP5 stop-time 재-plan coverage-walk].

★채널 절대규칙 (설계 §2·REPLAY_SAFE 교훈·`t2_gate_patch.py:997-1004`):
  합성 메시지를 committed 히스토리에 커밋하면 tau2 평가의 set_state replay
  (mutating tool 재실행·environment.py:389 assertion)가 깨져 infrastructure_error.
  E-PLAN의 모든 개입(L1/L2 deny-피드백·CP5 리마인더)은 **생성-레벨(작업버퍼)만** —
  히스토리 커밋 절대 금지. 커밋되는 것은 에이전트가 실제 수행한 호출·발화뿐.
  이 단계(step c)는 주입 코드 자체를 쓰지 않는다 — 배선은 전부 TODO 스텁.

★[[05]]: 이 파일 = 도메인일반 엔진(retail 리터럴 0). 도구명·엔티티 키는 전부
  `a2/<domain>.gate.json`의 "eplan" 키(ABox)서 로드. SCOPE_TOKEN 어휘·수량 소사전은
  도메인일반이라 엔진에 둔다(설계 §3).
★write 강제 금지(설계 §4 절대선): 강제되는 것은 read(discovery)뿐.
  CP5 = 리마인더(비강제·상한 1회 + step-budget 가드).

활성화: `T2_EPLAN=1` + `import t2_eplan_patch; t2_eplan_patch.apply()`.
tau2 임포트는 apply() 안에서만(지연 임포트) — 모듈 로드는 순수·오프라인 단위테스트 가능.
"""
import json
import os
import re
import sys

_A2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
_EPLAN_CACHE = {}


def _mark(msg):
    """stderr 마커 — t2_gate_patch 스타일. 스모크서 라이브 발화 검증용(설계 §5)."""
    print("[T2_EPLAN] %s" % msg, file=sys.stderr, flush=True)


def load_eplan_spec(domain):
    """a2/<domain>.gate.json의 "eplan" 키 로드. 없으면 None(=E-PLAN 비활성). 캐시.
    구조: {"list_enumerator": <목록 도구>, "detail_reader": <상세 도구>, "entity_key": <엔티티 인자명>}.
    도구명·키 = 전부 여기서(ABox) — 엔진 하드코딩 0([[05]])."""
    if domain in _EPLAN_CACHE:
        return _EPLAN_CACHE[domain]
    spec = None
    path = os.path.join(_A2_DIR, "%s.gate.json" % domain) if domain else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            spec = json.load(f).get("eplan")
    _EPLAN_CACHE[domain] = spec
    return spec


# ── SCOPE_TOKEN · 수량 — 도메인일반 어휘(엔진 보유 OK·설계 §3) ─────────────────
# SCOPE_TOKEN = 미해결 스코프 표지("전부"/"각각"류). 구체 id로 확장되기 전의 자리표.
_SCOPE_RE = re.compile(r"^(ALL|EACH|EVERY|BOTH|SCOPE_TOKEN)(_[A-Z]+)*$")

# 수량 소사전: 영어 수사 + both/couple/pair. 숫자는 1~10만 수량으로 인정
# (금액·연도·id 조각 오염 방지 — 그 이상은 수량 신호로 안 씀·보수적).
_QTY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "both": 2, "couple": 2, "pair": 2}
_QTY_DIGIT_RE = re.compile(r"\b([1-9]|10)\b")
_QTY_WORD_RE = re.compile(r"\b(%s)\b" % "|".join(_QTY_WORDS), re.I)


def is_scope_token(v):
    return isinstance(v, str) and bool(_SCOPE_RE.match(v.strip()))


def _norm(v):
    return str(v).strip() if v is not None else ""


def _idlike(s):
    """entity id형 필터(도메인일반): 숫자 포함·len>=2. 목록 출력의 문자열-리스트서 id만 수집."""
    return isinstance(s, str) and len(s.strip()) >= 2 and any(c.isdigit() for c in s)


def _extract_entity_ids(output_text, entity_key):
    """list-enumerator 출력(JSON)서 entity id 수집(도메인일반):
    (a) 어느 깊이든 key==entity_key 인 문자열 값
    (b) 문자열-리스트의 id형(_idlike) 원소  ← retail user record의 orders 형태
    파싱 실패(비JSON)면 빈 집합(관측 누락은 안전측 — deny가 아니라 미발화로 기움)."""
    ids = set()
    if not output_text:
        return ids
    try:
        rec = json.loads(output_text)
    except Exception:
        return ids

    def walk(v, key=None):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    if _idlike(x):
                        ids.add(_norm(x))
                else:
                    walk(x, key)
        elif key == entity_key and isinstance(v, str):
            ids.add(_norm(v))

    walk(rec)
    return ids


def _parse_qty(text):
    """발화 1건서 수량 후보 최대값(없으면 0). 단수 관사("a laptop")=수량 신호 없음."""
    if not text:
        return 0
    best = 0
    for m in _QTY_DIGIT_RE.finditer(text):
        best = max(best, int(m.group(1)))
    for m in _QTY_WORD_RE.finditer(text):
        best = max(best, _QTY_WORDS[m.group(1).lower()])
    return best


# ── PlanLedger — 턴간 완료-추적 (설계 §3 의사코드 구현) ───────────────────────
class PlanLedger:
    """planned/executed/listed/examined/replan 결정론 장부.
    planned·executed·replan 항목 = dict{intent_class, entity, items(frozenset), qty}.
    entity = 구체 id 또는 SCOPE_TOKEN 문자열. selector/verifier=결정론([[10]])."""

    def __init__(self, spec=None):
        self.spec = spec or {}          # A2 "eplan" 키 (도구명·entity_key = ABox)
        self.planned = []               # CP0 seed
        self.executed = []              # gated 관측(성공 write만)
        self.listed = set()             # list-enumerator 출력서 파생
        self.examined = set()           # detail-reader 호출 기록 (v1.2)
        self.replan = []                # CP5 stop-time 재-plan 산출 (v1.2)
        self.qty_mentioned = 1          # 사용자 발화 누적 수량(최대값 유지)

    # CP0: plan-spec 정규화 결과를 seed
    def seed(self, planned):
        self.planned = [self._entry(p) for p in planned]

    @staticmethod
    def _entry(p):
        return {"intent_class": p.get("intent_class"),
                "entity": _norm(p.get("entity")),
                "items": frozenset(p.get("items") or ()),
                "qty": int(p.get("qty") or 1)}

    def note_write(self, intent_class, entity, items=()):
        """실행 성공한 write 관측(gated 확장 지점서 호출)."""
        self.executed.append({"intent_class": intent_class,
                              "entity": _norm(entity),
                              "items": frozenset(items or ())})

    def note_read(self, tool_name, args=None, output_text=None):
        """read 관측: A2 도구명으로 list/detail 판별해 listed/examined 갱신.
        list-enumerator → 출력서 entity id 파싱해 listed.
        detail-reader   → 호출 인자의 entity_key 값을 examined."""
        if tool_name == self.spec.get("list_enumerator"):
            self.listed |= _extract_entity_ids(output_text, self.spec.get("entity_key"))
        elif tool_name == self.spec.get("detail_reader"):
            eid = (args or {}).get(self.spec.get("entity_key"))
            if eid:
                self.examined.add(_norm(eid))

    def accumulate_qty(self, user_text):
        """사용자 발화의 수량 언급 누적 — 최대값 유지("실은 두 대" 중반 계시 반영·v1.2).
        수량 신호 없으면 불변(단수 관사로 하향 안 함)."""
        q = _parse_qty(user_text)
        if q > self.qty_mentioned:
            self.qty_mentioned = q

    def set_replan(self, writes):
        """CP5 stop-time 재-plan 산출 기록(+마커)."""
        self.replan = [self._entry(w) for w in writes]
        _mark("replan: %d writes" % len(self.replan))

    def required_qty(self, intent_class=None):
        """요구 수량 N = planned qty(해당 intent-class)와 발화 누적 수량의 최대."""
        qs = [p["qty"] for p in self.planned
              if intent_class is None or p["intent_class"] == intent_class]
        return max(qs + [self.qty_mentioned, 1])


# ── 결정론 술어 (설계 §3 의사코드 그대로) ─────────────────────────────────────
def expand_scope(writes, listed):
    """SCOPE_TOKEN 확장(v1.1): (exchange, ALL_PENDING) × listed={W1,W2}
    → [(exchange,W1),(exchange,W2)]. discovery 전(listed 빈)엔 토큰 그대로 보존
    — "미확장 토큰 존재"가 L1 신호."""
    out = []
    for w in writes:
        if is_scope_token(w.get("entity")) and listed:
            for eid in sorted(listed):
                out.append(dict(w, entity=eid))
        else:
            out.append(dict(w))
    return out


def discovery_L1(ledger):
    """목록-수준(t81형): 멀티엔티티 의도(SCOPE_TOKEN 또는 수량>=2) ∧ 목록 미조회."""
    has_tok = any(is_scope_token(p.get("entity")) for p in ledger.planned)
    return (has_tok or ledger.required_qty() >= 2) and not ledger.listed


def discovery_L2(ledger, intent_class):
    """상세-수준(v1.2·t95 ⓐ tr1/tr3형): 요구수량 N > 매칭 distinct entity M
    ∧ 미검토 sibling(목록엔 있으나 detail 미조회) 존재 → 그 id들(정렬) 반환.
    N<2면 침묵(단일-엔티티 요청에 전수-읽기 강요 = over-read·설계 §4 Δtme 위반 방지).
    전부 examined면 [] — ⓑ(binding-gap·tr0/tr2형)은 CP5 재-plan walk 관할(술어 특이도 §5④)."""
    n = ledger.required_qty(intent_class)
    m = len({e["entity"] for e in ledger.executed
             if e["intent_class"] == intent_class})
    if n < 2 or n <= m:
        return []
    return sorted(ledger.listed - ledger.examined)


def _covers(e, p):
    """executed e가 planned/replan p를 커버하나: intent_class 일치 + entity 일치 +
    items 관대(어느 한쪽 비었거나 부분집합 관계면 커버 — 항목 과소지정 plan이
    실제 write를 gap으로 오판하는 false-positive 방지)."""
    if e["intent_class"] != p["intent_class"] or _norm(e["entity"]) != _norm(p["entity"]):
        return False
    pi, ei = p.get("items") or frozenset(), e.get("items") or frozenset()
    return (not pi) or (not ei) or pi <= ei or ei <= pi


def coverage_gap(ledger):
    """CP5 diff(v1.2): **replan** 기준(CP0 planned 아님) — expand_scope 후
    executed에 _covers 매칭 없는 항목. gap=∅면 리마인더 없이 즉시 종결(R1b)."""
    gaps = []
    for p in expand_scope(ledger.replan, ledger.listed):
        p = dict(p, items=frozenset(p.get("items") or ()))
        if not any(_covers(e, p) for e in ledger.executed):
            gaps.append(p)
    return gaps


# ── 히스토리 → ledger 결정론 재구성 (v1.3·CP0 LLM 불필요·④ 실증 로직) ────────
def _g(m, k, default=None):
    """dict/객체 겸용 접근자 — gz(dict)와 tau2 Message(attr) 양쪽서 동작."""
    if isinstance(m, dict):
        return m.get(k, default)
    return getattr(m, k, default)


def build_ledger_from_messages(messages, spec, write_tools):
    """committed 히스토리 1-pass → PlanLedger. 전부 결정론([[10]])·에이전트-기수행 관측만.
    qty=user 발화 누적 / listed·examined=read 관측 / executed=성공 write.
    ④(eplan_iso_probe)서 실궤적 검증된 로직의 객체-호환판."""
    led = PlanLedger(spec)
    res_by_id = {}
    for m in messages:
        if _g(m, "role") == "tool" and _g(m, "id") is not None:
            res_by_id[_g(m, "id")] = m
    for m in messages:
        role = _g(m, "role")
        c = _g(m, "content")
        if role == "user" and isinstance(c, str):
            led.accumulate_qty(c)
        if role == "assistant":
            for tc in (_g(m, "tool_calls") or []):
                nm = _g(tc, "name") or ""
                ar = _g(tc, "arguments")
                if isinstance(ar, str):
                    try:
                        ar = json.loads(ar)
                    except Exception:
                        ar = {}
                ar = ar if isinstance(ar, dict) else {}
                tm = res_by_id.get(_g(tc, "id"))
                ok = tm is not None and not _g(tm, "error")
                if nm in write_tools:
                    if ok:
                        led.note_write(nm, ar.get(spec.get("entity_key")),
                                       ar.get("item_ids") or ())
                else:
                    out = _g(tm, "content") if (tm is not None and not _g(tm, "error")) else None
                    led.note_read(nm, ar, out if isinstance(out, str) else None)
    return led


def cp5_gap_reminder(n, m, unexamined):
    """CP5 결정론 리마인더(v1.3·비강제·DB내용 0): 요청수량 N > 수행 M.
    미검토 sibling 있으면 read-지시·없으면 사용자-재확인 지시."""
    _mark("walk gap: qty=%d executed=%d unexamined=%d" % (n, m, len(unexamined)))
    if unexamined:
        return ("[E-PLAN] The user mentioned %d item(s)/record(s) for this request but you have "
                "completed %d. You have listed record(s) %s without reading their details — read "
                "them first, then decide." % (n, m, ", ".join(unexamined)))
    return ("[E-PLAN] The user mentioned %d item(s)/record(s) for this request but you have "
            "completed %d. Before ending, re-check with the user whether anything is left, "
            "then act or explain." % (n, m))


# ── 피드백 텍스트 (생성-레벨 전용·히스토리 커밋 금지) ─────────────────────────
def l1_feedback(ledger, spec):
    """L1 deny 피드백(t81형·"목록 먼저"). 도구명 = A2서."""
    _mark("L1 deny: multi-entity intent, list-enumerator not called")
    return ("[E-PLAN] This request may span MULTIPLE records. Before any write, first call "
            "%s to list the customer's records, then read the relevant ones."
            % spec.get("list_enumerator"))


def l2_feedback(ids, spec):
    """L2 deny 피드백(t95 ⓐ형·"미검토 주문 [ids]의 details 먼저").
    ids = 에이전트 자신이 가져온 목록 출력서 옴(규칙0 클린·DB 주입 아님)."""
    _mark("L2 deny: unexamined siblings %s" % ", ".join(ids))
    return ("[E-PLAN] The request quantity exceeds the records you have acted on. "
            "You listed record(s) %s but have not read their details yet — call %s "
            "for them first, then decide which records the request covers."
            % (", ".join(ids), spec.get("detail_reader")))


def cp5_reminder(gaps):
    """CP5 리마인더(비강제): 에이전트 자신의 재-plan 재진술 — gold 아님·DB 내용 0·
    read/write 강제 0. 생성-레벨로만 주입(TODO 배선)."""
    _mark("walk gap: %d item(s)" % len(gaps))
    lines = ["- %s on %s" % (p.get("intent_class"), p.get("entity")) for p in gaps]
    return ("[E-PLAN] Before ending: your own final plan for this conversation included "
            "the following item(s) you have not acted on. Re-check with the user whether "
            "they still want them, then act or explain:\n" + "\n".join(lines))


def discovery_precondition(ledger, spec, intent_class):
    """discovery-enforce 술어(순수) — write 시도 시 호출·deny 피드백 문자열 또는 None.
    기존 replay-safe deny+regen 게이트의 precondition으로 삽입(신규 후크 불요·설계 §1)."""
    if discovery_L1(ledger):
        return l1_feedback(ledger, spec)
    ids = discovery_L2(ledger, intent_class)
    if ids:
        return l2_feedback(ids, spec)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 배선 스텁 (step d서 구현 — 이번 단계는 시그니처+TODO만·주입 코드 없음)
#
# 설계 §2 배선점 표 (tau2/orchestrator/orchestrator.py):
# | 컴포넌트          | 후크                                        | 방식 |
# |------------------|---------------------------------------------|------|
# | CP0 plan-extract | 첫 agent step 직전(initialize() 후·first    | orchestrator 인스턴스에 _eplan_ledger 부착·1회 plan 생성 |
# |                  | user msg 확정 후)                            |      |
# | discovery-enforce| 기존 replay-safe deny+regen 게이트에         | plan-scope 미발견 ∧ intent-class 첫 write 시도 → 생성-레벨 deny + enumerator-선행 피드백 |
# |                  | precondition 추가(신규 후크 불필요)          |      |
# | ledger 관측      | gated(tool_calls) (기존 인터셉터 확장)       | 실행된 write를 executed에 기록·enumerator/detail 호출·결과서 listed/examined 갱신 |
# | CP5 coverage-walk| is_stop/_check_termination 직전              | 미완 replan 있으면 self.done 보류 + 생성-레벨 리마인더(히스토리 비커밋·상한 1회) |
# ═══════════════════════════════════════════════════════════════════════════

def _cp0_seed(orch, ledger, spec):
    """CP0 plan-seed 스텁 — 첫 agent step 직전 1회(역할 축소·v1.2: ledger seed + L1 조기 신호만).
    TODO(step d): 도메인일반 plan-추출 프롬프트(plan_execute_orch._plan_prompt 계열)로
      plan-spec 1회 생성 → plan_execute_orch.controller()로 정규화 → ledger.seed().
      orch에 `_eplan_ledger` 부착. 사용자 발화마다 ledger.accumulate_qty() 배선."""
    raise NotImplementedError("step d: live 배선 미구현")


def _observe_gated(ledger, tool_name, args, output_text, is_write,
                   intent_class=None, entity=None, items=()):
    """ledger 관측 — t2_gate_patch.gated() 확장 지점서 호출(성공 결과만·error 제외).
    이 함수 자체는 순수(단위테스트 가능). is_write/intent_class/entity/items 판별은
    호출측이 A2(ACTION_SPEC류)로 도출.
    TODO(step d): gated() 내 `results.extend(out)` 성공 경로에 배선."""
    if is_write:
        ledger.note_write(intent_class, entity, items)
    else:
        ledger.note_read(tool_name, args, output_text)


def _cp5_walk(orch, ledger, spec):
    """CP5 coverage-walk 스텁 — is_stop/_check_termination 직전.
    TODO(step d):
      1) stop-time 재-plan 1회(LLM·격리 plan-추출·전 대화 문맥 = C14 정보-맞춤)
         → ledger.set_replan(). gap=∅면 즉시 종결(추가 개입 0·R1b).
      2) gaps = coverage_gap(ledger).
      3) gap 있으면 self.done 보류 + cp5_reminder(gaps)를 **생성-레벨(작업버퍼)로만** 주입
         — LLMAgent._generate_next_message 패치(t2_gate_patch.py:993 패턴)로 다음 생성
         호출의 작업버퍼에만 추가. ★히스토리 커밋 절대 금지(채널 절대규칙).
      4) 상한 1회 하드코딩(R3) + 잔여 step-budget < walk 소요면 스킵 가드(§5②).
         그래도 미완이면 통과(강제 없음·write 강제 금지)."""
    raise NotImplementedError("step d: live 배선 미구현")


def load_write_tools(domain):
    """A2 파일서 confirm-게이트 applies_to = write 도구 집합(도메인일반 파생·gate 임포트 없이)."""
    path = os.path.join(_A2_DIR, "%s.gate.json" % domain) if domain else None
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        a2 = json.load(f)
    return {t for g in (a2.get("gates") or []) if g.get("kind") == "confirm"
            for t in (g.get("applies_to") or [])}


def apply():
    """E-PLAN 활성화(v1.3 step-d).
    · L1/L2 discovery deny + 리마인더 소비 = t2_gate_patch.unified()에 배선됨(T2_EPLAN=1).
    · CP5 walk(user_stop 보류 + 결정론 gap 리마인더) = 여기서 orchestrator wrap(T2_EPLAN_WALK=1).
    tau2 임포트는 여기서만(지연·모듈 로드 순수)."""
    if os.environ.get("T2_EPLAN") != "1":
        return None
    _mark("apply(): ledger+L1/L2=unified 배선·walk=%s"
          % ("ON" if os.environ.get("T2_EPLAN_WALK") == "1" else "off"))
    if os.environ.get("T2_EPLAN_WALK") != "1":
        return None

    import tau2.orchestrator.orchestrator as _om

    def _wrap(cls):
        orig = cls.__dict__.get("_check_termination")
        if orig is None or getattr(orig, "_t2_eplan_wrapped", False):
            return
        def wrapped(self, *a, _orig=orig, **kw):
            r = _orig(self, *a, **kw)
            try:
                if not getattr(self, "done", False):
                    return r
                if "user_stop" not in str(getattr(self, "termination_reason", "")).lower():
                    return r
                if getattr(self, "_t2_eplan_walked", False):
                    return r
                dom = getattr(getattr(self, "environment", None), "domain_name", None)
                spec = load_eplan_spec(dom)
                if not spec:
                    return r
                wt = load_write_tools(dom)
                msgs = self.get_messages() if hasattr(self, "get_messages") else []
                led = build_ledger_from_messages(msgs, spec, wt)
                n = led.required_qty()
                m = len({e["entity"] for e in led.executed})
                if n <= 1 or n <= m:
                    return r  # gap 없음 = 개입 0 (R1b)
                unexamined = sorted(led.listed - led.examined)
                self._t2_eplan_walked = True
                self.done = False
                self.termination_reason = None
                ag = getattr(self, "agent", None)
                if ag is not None:
                    ag._t2_eplan_reminder = cp5_gap_reminder(n, m, unexamined)
                _mark("walk: user_stop 보류(1회)·reminder 세팅 n=%d m=%d" % (n, m))
            except Exception as e:  # walk는 best-effort — 실패 시 종결 그대로
                _mark("walk skipped: %r" % (e,))
            return r
        wrapped._t2_eplan_wrapped = True
        cls._check_termination = wrapped

    for _cls in list(vars(_om).values()):
        if isinstance(_cls, type) and "_check_termination" in vars(_cls):
            _wrap(_cls)
    return True
