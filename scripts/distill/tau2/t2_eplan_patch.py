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

# ── qty 오염 가드 (t47 교정·T27_T103_PERSTEP §t27 부결함②·2026-07-11) ─────────
# 시간/기간 단위(닫힌 기능어 집합 — 도메인 어휘 아님)에 인접한 수사는 수량 신호가
# 아니다: "within 3 days"·"5 to 7 business days"의 3/7이 required_qty N으로 직행해
# 거짓 L1/L2를 유발(ap2 t47 = L1×1+L2×3 전부 이 오염·cap 소진). 파싱 전에 절제.
# ①bare 범위("5 to 7" — 단위 생략형 포함): 범위 표현 자체가 확정-수량이 아니라
#   추정/기간 언어 → 통째로 비신호(보수). ②수사+시간단위(선행 수식 business/working/
#   calendar·"couple of days"형 of 허용).
_TIME_UNIT = r"(?:second|minute|hour|day|week|month|year)s?"
_NUM_TOK = r"(?:[0-9]{1,3}|%s)" % "|".join(_QTY_WORDS)
_QTY_RANGE_RE = re.compile(
    r"\b%(n)s\s*(?:to|-|–|—)\s*%(n)s\b" % {"n": _NUM_TOK}, re.I)
_QTY_TIME_RE = re.compile(
    r"\b%(n)s(?:\s+of)?(?:\s+(?:business|working|calendar))?\s+%(u)s\b"
    % {"n": _NUM_TOK, "u": _TIME_UNIT}, re.I)

# ── 품목-나열 힌트 (A′ t81 교정·2026-07-11) — 도메인일반 *구문* 신호(어휘 0·[[05]]) ──
# 근거(t5c_aprime1 t81 정독): 사용자가 상품명 6개를 나열("the hiking boots, watch,
# keyboard, charger, jacket, and running shoes")했으나 수사/수량어 0 → qty=1 →
# L1/L2/walk 전부 침묵 → 두 주문 중 하나만 cancel(두번째 주문 미검토). 나열 자체가
# 멀티엔티티 신호인데 소사전이 못 봄.
# 술어(보수적·나열>=3만): 한 문장 안에서 쉼표/and 로 구분된 세그먼트 중
#   digit-free·1~4단어·순수 알파벳(상품명-류) 세그먼트의 *연속 run* 이 distinct >=3.
#   — 주소류("921 Park Avenue, Suite 892, Chicago, IL 60612")는 digit 세그먼트가
#     run 을 끊어 미발화 / 2-나열("a bookshelf and a jigsaw")= 미달 → 미발화.
_ENUM_MIN = 3
_ENUM_LEAD_RE = re.compile(
    r"^(?:the|a|an|my|your|his|her|our|their|some|this|that|these|those)\s+", re.I)
_ENUM_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z\-']*$")
# 담화 filler(닫힌 기능어류·도메인 어휘 아님): "Oh, um, yes, please" 같은 간투사 체인이
# 나열로 오인되는 것 차단(comp_retail_t4 4443발화 census서 확인된 유일 대량 FP류).
# filler-만 세그먼트 = 구분자 취급(skip·run 유지 — 실제 나열 사이 간투사는 무해).
_ENUM_FILLERS = {
    "oh", "um", "uh", "er", "ah", "hmm", "hm", "well", "okay", "ok", "alright",
    "yes", "yeah", "yep", "no", "nope", "please", "sure", "right", "thanks",
    "thank you", "hi", "hello", "hey", "wow", "so", "now", "also", "again",
    "actually", "honestly", "sorry", "great", "perfect", "fine", "correct",
    "exactly", "absolutely", "definitely", "of course", "got it", "i see",
    "you know", "let me see", "let's see", "wait", "anyway", "then", "too"}


def _enum_items(text):
    """발화 1건서 최대 나열 run 길이(distinct). >=_ENUM_MIN 일 때만 그 값, 아니면 0."""
    if not text:
        return 0
    best = 0
    for sent in re.split(r"[.!?;:\n]+", str(text)):
        if sent.count(",") < 2:          # 쉼표 2개 미만 = 3-나열 불가(보수)
            continue
        run, best_run = set(), 0
        for seg in re.split(r",|\band\b", sent, flags=re.I):
            s = _ENUM_LEAD_RE.sub("", seg.strip())
            if not s or s.lower() in _ENUM_FILLERS:
                continue                 # ", and" 아티팩트·간투사 = 구분자(run 유지)
            w = s.split()
            if 1 <= len(w) <= 4 and all(_ENUM_WORD_RE.match(x) for x in w):
                run.add(s.lower())
            else:                        # digit/장문 세그먼트 = run 절단(주소·문장 보호)
                best_run = max(best_run, len(run))
                run = set()
        best_run = max(best_run, len(run))
        best = max(best, best_run)
    return best if best >= _ENUM_MIN else 0


def is_scope_token(v):
    return isinstance(v, str) and bool(_SCOPE_RE.match(v.strip()))


def _norm(v):
    return str(v).strip() if v is not None else ""


def _idlike(s):
    """entity id형 필터(도메인일반): 숫자 포함·len>=2. 목록 출력의 문자열-리스트서 id만 수집."""
    return isinstance(s, str) and len(s.strip()) >= 2 and any(c.isdigit() for c in s)


_TEXT_KV_RE_CACHE = {}
def _text_kv_re(entity_key):
    """'entity_key: value' 텍스트 추출 정규식(도메인일반·entity_key=ABox). 캐시."""
    r = _TEXT_KV_RE_CACHE.get(entity_key)
    if r is None:
        r = re.compile(re.escape(entity_key) + r"\s*[:=]\s*['\"]?([A-Za-z0-9][\w\-]{1,})", re.I)
        _TEXT_KV_RE_CACHE[entity_key] = r
    return r


def _extract_entity_ids(output_text, entity_key):
    """list/detail reader 출력서 entity id 수집(도메인일반):
    (a) JSON: 어느 깊이든 key==entity_key 인 문자열 값 · 문자열-리스트의 id형(_idlike) 원소
    (b) 비-JSON 포맷문자열(banking 'transaction_id: txn_xxx'류·_note_eplan (a) 한계): entity_key 'key: value'
        정규식 추출(entity_key=ABox·리터럴0·[[05]]). 관측 누락은 안전측(미발화)."""
    ids = set()
    if not output_text:
        return ids
    try:
        rec = json.loads(output_text)
    except Exception:
        # (a) 비-JSON fallback: 'entity_key: value' 텍스트 추출(도메인일반)
        if entity_key:
            for m in _text_kv_re(entity_key).finditer(str(output_text)):
                v = _norm(m.group(1))
                if _idlike(v):
                    ids.add(v)
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
    """발화 1건서 수량 후보 최대값(없으면 0). 단수 관사("a laptop")=수량 신호 없음.
    시간/기간-인접 수사·bare 범위는 절제 후 파싱(t47 오염 가드·위 _QTY_*_RE)."""
    if not text:
        return 0
    text = _QTY_RANGE_RE.sub(" ", str(text))
    text = _QTY_TIME_RE.sub(" ", text)
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
        self.enum_items = 0             # 품목-나열 힌트: 최대 나열 길이(>= _ENUM_MIN만·t81)

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
        list-enumerator → 출력서 entity id 파싱해 listed (str 또는 list·C101 (b) banking 멀티-reader).
        list_from_reads=true → 어느 read 출력이든 entity_key id를 listed (banking bulk-reader·디스패처).
        detail-reader   → 호출 인자의 entity_key 값을 examined."""
        le = self.spec.get("list_enumerator")
        le_set = set(le) if isinstance(le, (list, tuple)) else ({le} if le else set())
        ek = self.spec.get("entity_key")
        if tool_name in le_set or self.spec.get("list_from_reads"):
            self.listed |= _extract_entity_ids(output_text, ek)
        if tool_name == self.spec.get("detail_reader"):
            eid = (args or {}).get(ek)
            if eid:
                self.examined.add(_norm(eid))
            # ★동일-도구 이중역할 (2026-07-21 §2bc·054 실측): detail_reader가 list_enumerator이기도
            #   한 도메인(banking: user_id-키 벌크 reader가 최심 reader·출력이 곧 전체 상세)은
            #   per-entity 인자 호출 자체가 불가능 — 인자-마킹만으론 examined가 구조적 공집합 =
            #   L2 deny가 **충족불가 술어**(054 t1: 피드백 지시대로 4회 재호출에도 탈출 불가→유저
            #   포기·전체 붕괴). A2가 선언한 도구 역할(양쪽 선언)에서 유도·엔진 리터럴 0:
            #   이 도구 출력에 전개된 entity id는 '검토됨'으로 마킹.
            if tool_name in le_set:
                self.examined |= _extract_entity_ids(output_text, ek)

    def accumulate_qty(self, user_text):
        """사용자 발화의 수량 언급 누적 — 최대값 유지("실은 두 대" 중반 계시 반영·v1.2).
        수량 신호 없으면 불변(단수 관사로 하향 안 함). 품목-나열 힌트도 여기서 누적(t81)."""
        q = _parse_qty(user_text)
        if q > self.qty_mentioned:
            self.qty_mentioned = q
        e = _enum_items(user_text)
        if e > self.enum_items:
            self.enum_items = e

    @property
    def multi_entity_hint(self):
        """품목-나열(>=3) 관측 여부 — L1/walk 의 멀티엔티티 조건에 OR (L2/required_qty 불변)."""
        return self.enum_items >= _ENUM_MIN

    def set_replan(self, writes):
        """CP5 stop-time 재-plan 산출 기록(+마커)."""
        self.replan = [self._entry(w) for w in writes]
        _mark("replan: %d writes" % len(self.replan))

    def required_qty(self, intent_class=None):
        """요구 수량 N = planned qty(해당 intent-class)와 발화 누적 수량의 최대."""
        qs = [p["qty"] for p in self.planned
              if intent_class is None or p["intent_class"] == intent_class]
        return max(qs + [self.qty_mentioned, 1])

    def item_coverage(self, extra=()):
        """실행 성공 write들 + (현재 시도 call의) 품목 id를 합친 distinct 개수.
        수량-conflation 가드(qty_item_covered)의 재료 — 중복 id(t95 ap2의
        ["x","x"] 이중지정)는 set이라 1로 계수(=미커버 신호 보존)."""
        if isinstance(extra, (str, bytes)):
            extra = (extra,)
        ids = {str(i) for e in self.executed for i in (e["items"] or ())}
        ids |= {str(x) for x in (extra or ()) if x is not None}
        ids.discard("")
        return len(ids)


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


def qty_item_covered(ledger, n, attempt_items=()):
    """★수량-conflation 보수화 가드 (T27_T103_PERSTEP §t27 부결함②·2026-07-11).
    발화 수량 N의 지시 대상이 entity(주문)인지 item(품목)인지는 *의미 판단*이라
    결정론으로 못 가른다("two items"=1주문 2품목 vs "two laptops"=실제 2주문 —
    표면 신호 동일·유저 둘 다 '한 주문' 주장). 대신 관측-가능 프록시로 보수화:
      N이 "이미 실행된 write들 + 지금 시도 중인 write의 distinct 품목 id 수"로
      전부 설명되면 → 수량은 품목-급으로 충족 가능 → qty만으로 deny하지 않는다.
    검증(오프라인 replay): t27(couple=2·return 2품목)=침묵 전환 / t47("both"=2·
    return 2품목)=침묵 / t95(two laptops·시도가 1품목 or 동일-id 중복)=발화 유지.
    ★정직 한계: (a) N이 여러 write에 분산되고 아직 아무것도 실행 전이면(ap1 t27:
    qty=3·첫 시도 exchange 1품목) 가드 불발 → deny 지속(상한=T2_EPLAN_DENY_CAP).
    (b) 진짜 멀티-entity 요청인데 에이전트가 한 entity에 N개 *distinct* 품목을 쓰면
    오-침묵(t95 ap2는 동일-id 중복이라 escape). (c) "size 9"류 속성-수사 오염은
    미해결(coverage 9 불충족 → 잔존·비표적 r=0 sim에서만 관측). 품목/entity 구분
    자체는 결정론화하지 않음 — semantic이라서."""
    return n >= 2 and ledger.item_coverage(attempt_items) >= n


def discovery_L1(ledger, attempt_items=()):
    """목록-수준(t81형): 멀티엔티티 의도(SCOPE_TOKEN 또는 수량>=2 또는 품목-나열>=3)
    ∧ 목록 미조회. 나열 힌트 OR = A′ t81 교정(수량어 없는 6-품목 나열).
    수량>=2 항만 qty_item_covered 가드 적용(토큰·나열 신호는 qty-conflation 무관)."""
    has_tok = any(is_scope_token(p.get("entity")) for p in ledger.planned)
    qty_sig = (ledger.required_qty() >= 2
               and not qty_item_covered(ledger, ledger.required_qty(), attempt_items))
    return ((has_tok or qty_sig or getattr(ledger, "multi_entity_hint", False))
            and not ledger.listed)


def walk_required_n(ledger, unexamined):
    """CP5 walk 의 요구수량 N. 기본 = required_qty(qty-소사전). 품목-나열 힌트는
    *미검토 sibling 존재 시에만* N에 기여(t81 시그니처 = 나열 + 안 읽은 주문 잔존).
    보수 근거(comp_retail_t4 456-sim census·2026-07-11): 무조건 기여 시 신규 walk 11건 중
    reward=1.0 sim 5건이 spurious — 그 5건 전부 unexamined=0, 실패 t81류는 전부 >=1
    → 이 조건이 passing-sim 과발화를 0으로 만들고 표적은 전부 보존.
    L2 는 불변(required_qty 그대로) — 나열=품목 수·L2 의 M=레코드 수라 conflation 방지."""
    n = ledger.required_qty()
    if unexamined:
        n = max(n, getattr(ledger, "enum_items", 0))
    return n


def discovery_L2(ledger, intent_class, attempt_items=()):
    """상세-수준(v1.2·t95 ⓐ tr1/tr3형): 요구수량 N > 매칭 distinct entity M
    ∧ 미검토 sibling(목록엔 있으나 detail 미조회) 존재 → 그 id들(정렬) 반환.
    N<2면 침묵(단일-엔티티 요청에 전수-읽기 강요 = over-read·설계 §4 Δtme 위반 방지).
    전부 examined면 [] — ⓑ(binding-gap·tr0/tr2형)은 CP5 재-plan walk 관할(술어 특이도 §5④).
    ★qty_item_covered 가드(t27 교정): N이 품목-급으로 충족되면 침묵 —
    attempt_items = 지금 deny 판정 대상인 write 호출의 품목 id(배선측이 전달)."""
    n = ledger.required_qty(intent_class)
    m = len({e["entity"] for e in ledger.executed
             if e["intent_class"] == intent_class})
    if n < 2 or n <= m:
        return []
    if qty_item_covered(ledger, n, attempt_items):
        return []
    return sorted(ledger.listed - ledger.examined)


def _intent_match(a, b):
    """intent 정합(도메인일반): 정확 일치 또는 소문자 포함관계 — 재-plan 산출의 축약형
    ('exchange')이 executed 도구명('exchange_delivered_order_items')을 커버(레버4·부록 Y).
    도메인 어휘 0 — 포함관계는 문자열 연산만."""
    a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def _covers(e, p):
    """executed e가 planned/replan p를 커버하나: intent 정합(_intent_match) + entity 일치 +
    items 관대(어느 한쪽 비었거나 부분집합 관계면 커버 — 항목 과소지정 plan이
    실제 write를 gap으로 오판하는 false-positive 방지)."""
    if not _intent_match(e["intent_class"], p["intent_class"]) \
            or _norm(e["entity"]) != _norm(p["entity"]):
        return False
    pi, ei = p.get("items") or frozenset(), e.get("items") or frozenset()
    return (not pi) or (not ei) or pi <= ei or ei <= pi


def chain_gap(messages, spec):
    """★intent-chain coverage (2026-07-24 피벗·C133 처방·write_tools 확장). qty 모델(다중-dispute)이
    못 잡는 **순차 사슬**(close/pay/log/apply·052 submit→deny 등)을 필수 read/write 집합으로 커버.
    A2 `intent_chains`=[{signals, required_reads, required_writes, phrase}]·엔진=신호매칭+집합차(결정론·
    도메인 리터럴 0·[[05]]/[[10]]). 반환: {phrase, missing_reads, missing_writes, executed_count} or None.
    성공 실행만 계수(에러 tool 제외)·디스패처 unwrap·suffix 무시(_fam)."""
    chains = spec.get("intent_chains") or []
    if not chains:
        return None
    _fam = lambda n: re.sub(r"_\d+$", "", str(n or ""))
    disp_tool = spec.get("dispatch_tool")
    disp_name_key = spec.get("dispatch_name_key", "agent_tool_name")
    utext = "\n".join(str(_g(m, "content") or "") for m in messages
                      if _g(m, "role") == "user").lower()
    res_by_id = {_g(m, "id"): m for m in messages
                 if _g(m, "role") == "tool" and _g(m, "id") is not None}
    executed = set()
    for m in messages:
        if _g(m, "role") != "assistant":
            continue
        for tc in (_g(m, "tool_calls") or []):
            nm = _g(tc, "name") or ""
            ar = _g(tc, "arguments")
            if isinstance(ar, str):
                try:
                    ar = json.loads(ar)
                except Exception:
                    ar = {}
            ar = ar if isinstance(ar, dict) else {}
            eff = _fam(ar.get(disp_name_key, "")) if (disp_tool and nm == disp_tool) else _fam(nm)
            tm = res_by_id.get(_g(tc, "id"))
            if tm is not None and not _g(tm, "error"):
                executed.add(eff)
    best = None
    for ch in chains:
        if not any(s.lower() in utext for s in (ch.get("signals") or [])):
            continue
        mr = [r for r in (ch.get("required_reads") or []) if _fam(r) not in executed]
        mw = [w for w in (ch.get("required_writes") or []) if _fam(w) not in executed]
        if not (mr or mw):
            continue
        req = set(_fam(x) for x in (ch.get("required_reads") or []) + (ch.get("required_writes") or []))
        ec = len(req & executed)
        cand = {"phrase": ch.get("phrase"), "missing_reads": mr, "missing_writes": mw,
                "executed_count": ec}
        if best is None or (len(mr) + len(mw)) < (len(best["missing_reads"]) + len(best["missing_writes"])):
            best = cand
    return best


def chain_reminder(chain):
    """intent-chain directive 리마인더(C116 처방·능동 완주·write 강제 0=에이전트 emit)."""
    _mark("chain gap: missing_reads=%d missing_writes=%d exec=%d"
          % (len(chain["missing_reads"]), len(chain["missing_writes"]), chain["executed_count"]))
    parts = []
    if chain["missing_reads"]:
        parts.append("read (unlock+call): " + ", ".join(chain["missing_reads"]))
    if chain["missing_writes"]:
        parts.append("then execute (with the customer's consent already given): "
                     + ", ".join(chain["missing_writes"]))
    tail = (" " + chain["phrase"]) if chain.get("phrase") else ""
    return ("[E-PLAN] This request is not finished — required steps remain. Do NOT end or defer to "
            "the user. Complete them now: " + "; ".join(parts) + "."
            + tail + " Find each discoverable tool's full suffixed name in the knowledge base if needed.")


# ── BRANCH-REGROUND: 조건단계 실제 정책문서 재부각 (C144·2026-07-24) ────────────
# C144 해리(실증): 분기점 재부각 실효는 단계 유형별로 다르다 —
#   · 단순 read(get_user_dispute_history): compact 이름만으로 회복(R_compact 4-5/5).
#   · 정책조건 write(apply_credit_card_account_flag): **실제 정책문서 재부각만** 회복
#     (R_rawdoc 4/4)·compact 요약(rationale 포함)으로도 불충분(R_compact_rat 0 —
#     압축 시 authority/인자명세 소실). ⇒ 남은 조건단계는 이미 retrieved된 실제 문서를 붙인다.
# 문서는 transcript(에이전트가 이미 긁어온 tool 출력)서 추출 — 새 도메인지식 0·엔진 순수([[05]]).
_DOC_HEADER_RE = re.compile(r"(?m)^[ \t]*\d+\.[ \t]+.*\n[ \t]*ID:[ \t]*(\S+)")


def resurface_doc(messages, tool_family, max_chars=5000):
    """이미 retrieved된 tool 출력에서 tool_family를 명시한 focused 정책문서 블록만 추출.
    KB 출력 형식 = 번호목록('N. <title>\\n   ID: <id>\\n   Score: ..\\n   Content: ..').
    문서-헤더(번호줄+다음줄 ID:) 경계로 분할해 대상 도구를 명시한 *가장 작은* 블록 반환
    (C143 희석 회피 — 35K 통짜는 실패·focused ~4-5K가 회복). 헤더 없으면 전체(작을 때만
    유효). 대상 명시 문서 없으면 None(→compact 폴백). 도메인 리터럴 0(순수 문자열 엔진)."""
    fam = re.sub(r"_\d+$", "", str(tool_family or ""))
    if not fam:
        return None
    best = None
    for m in messages:
        if _g(m, "role") != "tool":
            continue
        c = _g(m, "content")
        if not isinstance(c, str) or fam not in c:
            continue
        starts = [mo.start() for mo in _DOC_HEADER_RE.finditer(c)]
        if not starts:
            blocks = [c]
        else:
            bounds = starts + [len(c)]
            blocks = [c[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if starts[0] > 0:                     # 헤더 前 서두도 후보
                blocks.append(c[:starts[0]])
        for b in blocks:
            if fam in b and (best is None or len(b) < len(best)):
                best = b
    return best[:max_chars] if best is not None else None


def branch_reground_reminder(chain, messages, spec=None, for_finalize=False):
    """★BRANCH-REGROUND 재부각(C144 해리 구현). chain_reminder의 상위:
    남은 단순단계(missing_reads + 정책문서 없는 write)=compact 이름만·남은 정책조건 write=
    이미 retrieved된 실제 정책문서 재부각(요약 X). 정책문서 유무는 resurface_doc이 결정
    (transcript에 대상 도구 명시 문서 있으면 조건단계로 취급). write 강제 0(에이전트 emit).
    for_finalize=True(pre-close 트리거): finalize_writes(A2·close류)를 남은목록서 제외하고
    "종결 前 선행단계 먼저" 프레이밍 — apply_flag이 close 선행하도록(현 user_stop은 close 後)."""
    mr = list(chain.get("missing_reads") or [])
    mw = list(chain.get("missing_writes") or [])
    if for_finalize:
        _fin = {re.sub(r"_\d+$", "", f) for f in ((spec or {}).get("finalize_writes") or ())}
        mw = [w for w in mw if re.sub(r"_\d+$", "", w) not in _fin]
    _mark("branch-reground%s: missing_reads=%d missing_writes=%d exec=%d"
          % ("(pre-finalize)" if for_finalize else "", len(mr), len(mw),
             chain.get("executed_count", 0)))
    simple_w, doc_order, doc_names = [], [], {}
    for w in mw:
        name = re.sub(r"_\d+$", "", w)
        doc = resurface_doc(messages, w)
        if doc:
            # 같은 정책문서(예: close·apply_flag가 한 프로세스 문서에 공존)는 1회만 첨부·
            #   라벨 합침(C143 희석 회피 — 동일 4.6K 문서 중복 방지).
            if doc not in doc_names:
                doc_order.append(doc)
                doc_names[doc] = []
            doc_names[doc].append(name)
        else:
            simple_w.append(w)
    parts = []
    if mr:
        parts.append("read (unlock+call): " + ", ".join(mr))
    if simple_w:
        parts.append("then execute (consent already given): " + ", ".join(simple_w))
    tail = (" " + chain["phrase"]) if chain.get("phrase") else ""
    if for_finalize:
        intro = ("[E-PLAN] STOP — do NOT close/finalize yet. Required prerequisite steps remain "
                 "and MUST be completed FIRST, before you close: ")
    else:
        intro = ("[E-PLAN] This request is not finished — required steps remain before you "
                 "close/finalize. Do NOT end or defer to the user. Complete them now: ")
    body = (intro + "; ".join(parts) + "." + tail
            + " Find each discoverable tool's full suffixed name in the knowledge base if needed.")
    for doc in doc_order:
        body += ("\n\n[POLICY you already retrieved — apply it now for %s]\n%s"
                 % (", ".join(doc_names[doc]), doc))
    return body



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
    ④(eplan_iso_probe)서 실궤적 검증된 로직의 객체-호환판.
    items 인자 키 = A2 spec "items_key"(ABox·retail="item_ids") — 기본값은 기존
    엔진 리터럴과의 하위호환(spec 미지정 도메인 = 종전 동작 그대로)."""
    led = PlanLedger(spec)
    ikey = spec.get("items_key", "item_ids")
    ekey = spec.get("entity_key")
    # C101 (b): banking dispute write/read = 디스패처 nested. spec이 디스패처 배선 선언(도메인일반·ABox).
    disp_tool = spec.get("dispatch_tool")
    disp_name_key = spec.get("dispatch_name_key", "agent_tool_name")
    disp_args_key = spec.get("dispatch_args_key", "arguments")
    wt_all = set(write_tools) | set(spec.get("write_tools") or ())
    _fam = lambda n: re.sub(r"_\d+$", "", str(n or ""))
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
                out = _g(tm, "content") if ok else None
                out = out if isinstance(out, str) else None
                # C101 (b): 디스패처면 nested {name, args}로 풀어 실제 도구명/인자로 판별
                eff_nm, eff_ar = nm, ar
                if disp_tool and nm == disp_tool:
                    eff_nm = _fam(ar.get(disp_name_key, ""))
                    inner = ar.get(disp_args_key)
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except Exception:
                            inner = {}
                    eff_ar = inner if isinstance(inner, dict) else {}
                if eff_nm in wt_all or _fam(eff_nm) in wt_all:
                    if ok:
                        led.note_write(eff_nm, eff_ar.get(ekey), eff_ar.get(ikey) or ())
                else:
                    led.note_read(eff_nm, eff_ar, out)
    return led


# ── ★레버4 formalize-obligation (T2_EPLAN_REPLAN=1·부록 Y·t81 후계) ──────────
# CP5 walk의 서브콜 변형: 리마인더 전에 격리 서브콜 1회 — 프롬프트 = **결정론 ledger
# 요약(qty·listed·examined·executed 열거) + "사용자가 요청한 write-의무 목록 JSON"**
# (v1.3 ⑤ raw-실패의 교정 = 구조화 문맥). 산출 → 결정론 diff(coverage_gap replan 경로
# 재사용) → gap 있으면 기존 리마인더. 서브콜/파싱 실패 = 기존 결정론 신호 폴백(안전측).
# [[10]]: 의무-추출(NL→구조)=LLM / diff·판정=결정론. write 강제 0(리마인더 비강제 불변).

def ledger_summary(led):
    """결정론 ledger 요약(도메인일반 어휘) — 구조화 재-plan 프롬프트의 1부."""
    ex = ["  - %s on %s%s" % (e["intent_class"], e["entity"] or "?",
                              (" (items: %s)" % ", ".join(sorted(e["items"])))
                              if e["items"] else "")
          for e in led.executed] or ["  (none)"]
    return ("[LEDGER — deterministic summary of what actually happened]\n"
            "- max quantity the user mentioned: %d\n"
            "- records LISTED (returned by the list tool): %s\n"
            "- records EXAMINED (details read): %s\n"
            "- executed WRITE actions:\n%s"
            % (led.qty_mentioned,
               ", ".join(sorted(led.listed)) or "(none)",
               ", ".join(sorted(led.examined)) or "(none)",
               "\n".join(ex)))


OBLIGATION_PROMPT = (
    "You are auditing a finished customer-service conversation. Above is a DETERMINISTIC "
    "LEDGER of what actually happened; below is the transcript. Using BOTH, list every WRITE "
    "obligation the user requested in this conversation — including ones the agent failed to "
    "perform. Output ONLY a JSON array; each element: "
    '{"action": "<the exact write tool name>", "%s": "<the record id as it appears>", '
    '"items": ["..."]}. Use ids exactly as they appear in the ledger/transcript; '
    "do not invent ids."
)


def transcript_text(messages, max_chars=60000):
    """전사(텍스트 턴 + assistant 툴콜 라인) — dict/객체 겸용(_g). 절단=앞뒤 보존."""
    lines = []
    for m in messages:
        role, c = _g(m, "role"), _g(m, "content")
        if isinstance(c, str) and c.strip():
            lines.append("[%s] %s" % (role, c.strip()))
        if role == "assistant":
            for tc in (_g(m, "tool_calls") or []):
                ar = _g(tc, "arguments")
                if isinstance(ar, str):
                    try:
                        ar = json.loads(ar)
                    except Exception:
                        ar = {}
                lines.append("[assistant->tool] %s %s"
                             % (_g(tc, "name"),
                                json.dumps(ar if isinstance(ar, dict) else {},
                                           ensure_ascii=False, default=str)))
    txt = "\n".join(lines)
    if len(txt) > max_chars:
        keep = max_chars // 2
        txt = txt[:keep] + "\n...[middle truncated]...\n" + txt[-keep:]
    return txt


def structured_replan_prompt(led, transcript, entity_key):
    """구조화 재-plan 프롬프트 = ledger 요약 + 의무-JSON 지시 + 전사. entity_key=A2(ABox)."""
    return (ledger_summary(led) + "\n\n" + (OBLIGATION_PROMPT % entity_key)
            + "\n\n--- TRANSCRIPT ---\n" + transcript)


def parse_obligations(txt, entity_key):
    """서브콜 응답 → replan write 목록 | None(형식 실패 = 결정론 폴백 신호).
    빈 배열 = 유효(의무 0 → gap 진단은 coverage_gap 몫)."""
    if not isinstance(txt, str):
        return None
    i = txt.find("[")
    if i < 0:
        return None
    try:
        arr, _ = json.JSONDecoder().raw_decode(txt[i:])
    except Exception:
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for w in arr:
        if not isinstance(w, dict):
            continue
        act = w.get("action")
        if not isinstance(act, str) or not act.strip():
            continue
        ent = w.get(entity_key) or w.get("entity")
        items = w.get("items")
        out.append({"intent_class": act.strip(), "entity": _norm(ent),
                    "items": [str(x) for x in items] if isinstance(items, list) else (),
                    "qty": 1})
    return out


def _cp5_replan_subcall(orch, led, msgs, spec):
    """격리 서브콜 1회(히스토리 밖·비커밋) → 의무 목록 | None(실패=폴백). 예외는 호출측서."""
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage
    ag = orch.agent
    ekey = spec.get("entity_key") or "entity"
    prompt = structured_replan_prompt(led, transcript_text(msgs), ekey)
    try:
        um = UserMessage(role="user", content=prompt)
    except TypeError:
        um = UserMessage(content=prompt)
    kw = {k: v for k, v in dict(getattr(ag, "llm_args", None) or {}).items()
          if "tool" not in k}
    sub = la.generate(model=ag.llm, tools=None, messages=[um],
                      call_name="eplan_replan_subcall", **kw)
    _mark("replan subcall fired (structured ledger prompt)")
    return parse_obligations(getattr(sub, "content", None) or "", ekey)


def drive_decision(drives, K, exec_now, last_exec, has_gap):
    """지속 구동 결정(순수·2026-07-24 피벗). 반환: 'hold'(구동)·'release'(진전0 안전종료)·
    'terminate'(budget 소진 or plan 충족). progress-guard=§7.3 딜레마 회피(못 고치는 사슬 무한보류 금지)."""
    if drives >= K:
        return "terminate"                       # budget 소진 → 놓아줌
    if drives > 0 and exec_now <= last_exec:
        return "release"                          # 직전 구동 후 진전 0 → 안전 종료
    if not has_gap:
        return "terminate"                        # 필수 write 전부 실행 = 개입 0 (R1b)
    return "hold"                                 # gap 있음 + 진전/첫구동 → 지속 구동


def cp5_gap_reminder(n, m, unexamined, done_entities=(), action_phrase=None):
    """CP5 결정론 리마인더(v1.4 directive·2026-07-24 피벗·비강제·DB내용 0): 요청수량 N > 수행 M.
    ★C116 처방화: (a)이미 처리한 엔티티 명시(done_entities·에이전트 자기출력서) (b)남은 수 지목
    (c)수동 탈출구("손님께 재확인/설명") → **능동 완주 지시**(지속 구동의 효과 결정 변수). 미검토
    sibling 있으면 read-우선. action_phrase=spec(A2)·없으면 도메인일반 문구. write 강제 0(에이전트 emit)."""
    _mark("walk gap: qty=%d executed=%d unexamined=%d" % (n, m, len(unexamined)))
    _act = action_phrase or "take the required action for each remaining one"
    _done = (" You have already completed: %s." % ", ".join(sorted(done_entities))) if done_entities else ""
    if unexamined:
        return ("[E-PLAN] The user asked about %d record(s); you have completed %d.%s You listed "
                "record(s) %s but have not read their details — read them first, then %s. Do not "
                "end until every record the user asked about is handled."
                % (n, m, _done, ", ".join(unexamined), _act))
    return ("[E-PLAN] The user asked about %d item(s)/record(s); you have completed only %d.%s "
            "Do NOT end or defer to the user yet — %d remain. Identify the remaining one(s) from "
            "what the user described and %s now." % (n, m, _done, n - m, _act))


# ── 피드백 텍스트 (생성-레벨 전용·히스토리 커밋 금지) ─────────────────────────
def _tool_phrase(v):
    """도구명(str) 또는 도구명 목록(list) → 사람-읽기 문구('a or b'). 도메인일반."""
    if isinstance(v, (list, tuple)):
        return " or ".join(str(x) for x in v) or "the list tool"
    return str(v)


def l1_feedback(ledger, spec):
    """L1 deny 피드백(t81형·"목록 먼저"). 도구명 = A2서."""
    _mark("L1 deny: multi-entity intent, list-enumerator not called")
    return ("[E-PLAN] This request may span MULTIPLE records. Before any write, first call "
            "%s to list the customer's records, then read the relevant ones."
            % _tool_phrase(spec.get("list_enumerator")))


def l2_feedback(ids, spec):
    """L2 deny 피드백(t95 ⓐ형·"미검토 주문 [ids]의 details 먼저").
    ids = 에이전트 자신이 가져온 목록 출력서 옴(규칙0 클린·DB 주입 아님).
    ★READS-ONLY (T2_EPLAN_READS_ONLY=1·2026-07-13 REMEDIATION §3c·harm-mode III t32):
      eplan은 read만 강제 — surface된 record는 *관련성 확인용*이며 요청에 없으면 행동 금지.
      현 문구 "decide which records the request covers"가 과-행동 유발(t32 spurious return) →
      "확인만·요청 밖 record엔 write 금지" 명시로 교정."""
    _mark("L2 deny: unexamined siblings %s" % ", ".join(ids))
    if os.environ.get("T2_EPLAN_READS_ONLY") == "1":
        # ★#4 조건부화(2026-07-13 A1-v2 실패분석): 무조건 "행동금지"가 in-scope sibling까지 막아
        #   coverage 누락 8건 유발 → in-scope=행동 허용(coverage)·요청-무관만 금지(t32 over-action).
        return ("[E-PLAN] The request may span more records than you have read. "
                "You listed record(s) %s but have not read their details yet — call %s "
                "for them first. If a record IS part of what the user asked for, act on it too "
                "(complete every record the request covers). Only avoid acting on records the "
                "user did NOT ask about."
                % (", ".join(ids), _tool_phrase(spec.get("detail_reader"))))
    return ("[E-PLAN] The request quantity exceeds the records you have acted on. "
            "You listed record(s) %s but have not read their details yet — call %s "
            "for them first, then decide which records the request covers."
            % (", ".join(ids), _tool_phrase(spec.get("detail_reader"))))


def cp5_reminder(gaps):
    """CP5 리마인더(비강제): 에이전트 자신의 재-plan 재진술 — gold 아님·DB 내용 0·
    read/write 강제 0. 생성-레벨로만 주입(TODO 배선)."""
    _mark("walk gap: %d item(s)" % len(gaps))
    lines = ["- %s on %s" % (p.get("intent_class"), p.get("entity")) for p in gaps]
    return ("[E-PLAN] Before ending: your own final plan for this conversation included "
            "the following item(s) you have not acted on. Re-check with the user whether "
            "they still want them, then act or explain:\n" + "\n".join(lines))


def discovery_precondition(ledger, spec, intent_class, attempt_items=(), attempt_entity=None):
    """discovery-enforce 술어(순수) — write 시도 시 호출·deny 피드백 문자열 또는 None.
    기존 replay-safe deny+regen 게이트의 precondition으로 삽입(신규 후크 불요·설계 §1).
    attempt_items = 시도 중인 write 호출의 품목 id(A2 items_key로 배선측 추출) —
    qty_item_covered 가드 재료(미전달=기존 동작·qty-conflation 가드만 비활성).
    ★EXAMINED-SAFE (T2_EPLAN_EXAMINED_SAFE=1·2026-07-12 HANDOFF §6.2·A1 부작용 교정):
      attempt_entity(이 write의 대상 record id)가 *이미 examined*면 discovery-deny 생략.
      근거(A1 포렌식 t21/t32/t42): 에이전트가 대상 주문을 *읽고* write하는데 L2가
      미검토 sibling 이유로 그 정당 write를 deny → read-루프→transfer/partial(순손실 3).
      write-deny는 coverage의 잘못된 레버(LOCK §2d·§4d) — 커버리지는 CP5 walk 몫.
      "대상을 검토한 write = 정당" = 진짜 미검토-관련만 deny(§6.2 문구 그대로)."""
    if os.environ.get("T2_EPLAN_EXAMINED_SAFE") == "1" and attempt_entity is not None \
            and _norm(attempt_entity) in ledger.examined:
        _mark("examined-safe: write target %s examined — no discovery-deny" % _norm(attempt_entity))
        return None
    if discovery_L1(ledger, attempt_items):
        return l1_feedback(ledger, spec)
    ids = discovery_L2(ledger, intent_class, attempt_items)
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
                # ★지속 구동 (2026-07-24 피벗·EPLAN_PERSISTENT_DRIVER_DESIGN): 구판 1회 하드캡
                #   (_t2_eplan_walked)을 budget K + progress-guard로 교체 — 탐지만 하고 1회 포기하던
                #   것을 plan 충족까지 지속 구동. rall23b 실측 `walk gap qty=9 executed=1`=9필요 1완료를
                #   알면서 놓아줌. K=T2_EPLAN_DRIVE_K(기본 4). progress-guard=직전 구동 후 executed write가
                #   *커졌을 때만* 계속(§7.3 딜레마 회피=못 고치는 사슬 무한보류 금지·진전0→release).
                _K = int(os.environ.get("T2_EPLAN_DRIVE_K", "4"))
                _drives = getattr(self, "_t2_eplan_drives", 0)
                if _drives >= _K:                            # budget 소진 → 놓아줌
                    return r
                dom = getattr(getattr(self, "environment", None), "domain_name", None)
                spec = load_eplan_spec(dom)
                if not spec:
                    return r
                wt = load_write_tools(dom)
                msgs = self.get_messages() if hasattr(self, "get_messages") else []
                led = build_ledger_from_messages(msgs, spec, wt)
                _exec_now = len(getattr(led, "executed", []) or [])
                unexamined = sorted(led.listed - led.examined)
                reminder = None
                # ★intent-chain coverage (2026-07-24 피벗·C133): 순차 사슬(close/pay/log/apply·
                #   052형)을 qty 모델보다 우선. 사슬 gap 있으면 directive 리마인더·exec=사슬 실행 수
                #   (progress-guard가 사슬 write 완료를 진전으로 셈). 사슬 미적용/무gap → 아래 qty 폴백.
                _chain = chain_gap(msgs, spec)
                if _chain is not None:
                    # ★BRANCH-REGROUND(C144·T2_BRANCH_REGROUND=1): 남은 정책조건 write에
                    #   이미 retrieved된 실제 정책문서 재부각(compact 이름은 read엔 충분·조건
                    #   write엔 불충분). off면 기존 compact-only chain_reminder(A/B 대조).
                    if os.environ.get("T2_BRANCH_REGROUND") == "1":
                        reminder = branch_reground_reminder(_chain, msgs, spec)
                    else:
                        reminder = chain_reminder(_chain)
                    _exec_now = _chain["executed_count"]
                # ★레버4(T2_EPLAN_REPLAN=1): 격리 서브콜(구조화 ledger 프롬프트) →
                #   결정론 diff(coverage_gap replan 경로). 실패=기존 결정론 신호 폴백(안전측).
                #   chain이 발화했으면 skip(chain 우선·C133).
                if reminder is None and os.environ.get("T2_EPLAN_REPLAN") == "1":
                    try:
                        obligations = _cp5_replan_subcall(self, led, msgs, spec)
                    except Exception as _re:
                        obligations = None
                        _mark("replan subcall failed (%r) — deterministic fallback" % (_re,))
                    if obligations is not None:
                        led.set_replan(obligations)
                        gaps = coverage_gap(led)
                        if gaps:
                            reminder = cp5_reminder(gaps)
                            _mark("replan gap: %d item(s)" % len(gaps))
                        else:
                            _mark("replan: no gap — terminate")
                            return r  # 의무 전부 커버 = 개입 0 종결 (R1b)
                if reminder is None:  # flag off 또는 서브콜/파싱 실패 → 기존 결정론 신호
                    n = walk_required_n(led, unexamined)  # 나열 힌트 OR (t81·미검토 있을 때만)
                    m = len({e["entity"] for e in led.executed})
                    if n <= 1 or n <= m:
                        return r  # gap 없음 = 개입 0 (R1b)
                    if qty_item_covered(led, n):
                        # ★t27 교정: N이 실행된 write들의 distinct 품목 수로 설명되면
                        #   품목-급 수량 → 거짓 gap(ap2 t27 walk qty=2 executed=1) 억제.
                        #   t81(cancel=품목 인자 0→coverage 0)·t95(중복-id=1)는 불변 발화.
                        _mark("walk suppressed: qty item-covered (n=%d cov=%d)"
                              % (n, led.item_coverage()))
                        return r
                    _done_ent = {str(e.get("entity")) for e in led.executed if e.get("entity")}
                    reminder = cp5_gap_reminder(n, m, unexamined, _done_ent,
                                                spec.get("write_action_phrase"))
                # ★progress-guard (순수 결정): gap 있음 확정(reminder≠None)·budget 여유(drives<K) 상태서
                #   직전 구동 후 진전(executed↑) 없으면 release=안전종료(§7.3 무한보류/붕괴 회피).
                if drive_decision(_drives, _K, _exec_now,
                                  getattr(self, "_t2_eplan_last_exec", -1), True) != "hold":
                    _mark("drive released: no progress (exec=%d drives=%d)" % (_exec_now, _drives))
                    return r
                self._t2_eplan_drives = _drives + 1
                self._t2_eplan_last_exec = _exec_now
                self._t2_eplan_walked = True                 # 하위호환(다른 참조)
                self.done = False
                self.termination_reason = None
                ag = getattr(self, "agent", None)
                if ag is not None:
                    ag._t2_eplan_reminder = reminder
                _mark("drive %d/%d: user_stop 보류·reminder(exec=%d)"
                      % (self._t2_eplan_drives, _K, _exec_now))
            except Exception as e:  # walk는 best-effort — 실패 시 종결 그대로
                _mark("walk skipped: %r" % (e,))
            return r
        wrapped._t2_eplan_wrapped = True
        cls._check_termination = wrapped

    for _cls in list(vars(_om).values()):
        if isinstance(_cls, type) and "_check_termination" in vars(_cls):
            _wrap(_cls)
    return True
