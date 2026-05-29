"""tau2_airline_ontology.py  —  v1

τ²-bench airline 도메인 온톨로지 관계 정의 (42-relation, telecom 포맷 호환).

구성 방법 (LLM-AFOD, 2026-05-29):
  - 도구 인벤토리: src/tau2/domains/airline/tools.py (@is_tool, N=15)
  - [confirmed]:  airline tasks.json(43 tasks w/actions) 동시출현/순서 통계
  - [confirmed-policy]: data/tau2/domains/airline/policy.md 명시 규칙
  - [inferred]:   도구 docstring + 정책 의미론 추론 (Claude Opus 4.8 / Claude Code)
  - [hypothetical]: 이론적 추정

RELATION_GEOMETRY/PREDICTED_METHOD는 도메인-독립이라 telecom/retail에서 그대로.

airline 핵심 워크플로 (policy.md):
  - book:   user_id 확보 → search_direct/onestop → book_reservation
            (cabin 동일, ≤5 passengers, payment in profile, ≤1 cert/≤1 credit/≤3 gift)
  - modify: user_id + reservation_id(get_reservation_details) → update_flights/baggages/passengers
            (basic economy flight 변경 불가, origin/dest/triptype 불변, baggage add-only,
             insurance 추가 불가, passenger 수 변경 불가, cabin은 미운항 시만)
  - cancel: user_id + reservation_id → cancel_reservation
            (운항 segment 있으면 불가→transfer; 24h내/airline취소/business/insurance covered만 가능)
  - 모든 write 전 명시 confirm. 범위 밖/운항후취소 → transfer (첫 액션 금지).
  - send_certificate: compensation (명시 요청 + eligible 시에만).

Tools (N=15):
  READ:    get_user_details, get_reservation_details, get_flight_status,
           list_all_airports, search_direct_flight, search_onestop_flight
  WRITE:   book_reservation, cancel_reservation, update_reservation_flights,
           update_reservation_baggages, update_reservation_passengers, send_certificate
  GENERIC: calculate, transfer_to_human_agents
  THINK:   think
"""
from typing import Dict, FrozenSet, List, NamedTuple, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 타입 정의 (telecom/retail 포맷과 동일)
# ═══════════════════════════════════════════════════════════════════════════════

class ParamFeed(NamedTuple):
    source: str; target: str; param: str

class CausalLink(NamedTuple):
    achiever: str; beneficiary: str; predicate: str

class ConditionalOn(NamedTuple):
    tool: str; trigger: str; condition: str

class ExclusiveChoice(NamedTuple):
    condition: str; option_a: str; option_b: str

class AndJoin(NamedTuple):
    prerequisites: Tuple[str, ...]; target: str

class ErrorFallback(NamedTuple):
    primary: str; fallback: str

class RetryAfterFail(NamedTuple):
    failing_tool: str; fix_tool: str

class StateTransition(NamedTuple):
    tool: str; from_state: str; to_state: str

class DirectlyFollows(NamedTuple):
    source: str; target: str; freq: int

class BacktrackTo(NamedTuple):
    dead_end_tool: str; restore_point_tool: str

class FanOut(NamedTuple):
    source: str; successors: Tuple[str, ...]

class ScoredPreference(NamedTuple):
    preferred: str; alternative: str; context: str

class DecomposesInto(NamedTuple):
    goal: str; tools: Tuple[str, ...]

class Refines(NamedTuple):
    abstract_action: str; concrete_tool: str; context: str

class PlanRevisedTo(NamedTuple):
    trigger_observation: str; old_step: str; new_step: str


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 A: 순서/연쇄 관계 (Directional → A6) ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 1. PRECEDES — A는 B보다 먼저 호출
PRECEDES: List[Tuple[str, str]] = [
    ("get_user_details",         "get_reservation_details"),      # [confirmed] 42
    ("get_reservation_details",  "search_direct_flight"),         # [confirmed] 59
    ("get_reservation_details",  "cancel_reservation"),           # [confirmed] 40
    ("get_reservation_details",  "update_reservation_flights"),   # [confirmed] 22
    ("search_direct_flight",     "update_reservation_flights"),   # [confirmed] 37
    ("get_user_details",         "search_direct_flight"),         # [confirmed] 12
    ("search_direct_flight",     "update_reservation_baggages"),  # [confirmed] 4
    ("get_reservation_details",  "book_reservation"),             # [confirmed] 2
    ("cancel_reservation",       "book_reservation"),             # [confirmed] 5 (rebook)
    ("get_user_details",         "book_reservation"),             # [confirmed-policy] user_id first
    ("search_direct_flight",     "book_reservation"),             # [inferred] search before book
    ("get_reservation_details",  "update_reservation_passengers"),# [confirmed]
    ("get_reservation_details",  "update_reservation_baggages"),  # [confirmed] 2
]

# 2. DIRECTLY_FOLLOWS — A 직후 B (empirical adjacency, freq)
DIRECTLY_FOLLOWS: List[DirectlyFollows] = [
    DirectlyFollows("get_reservation_details", "search_direct_flight",        59),  # [confirmed]
    DirectlyFollows("get_user_details",        "get_reservation_details",     42),  # [confirmed]
    DirectlyFollows("get_reservation_details", "cancel_reservation",          40),  # [confirmed]
    DirectlyFollows("search_direct_flight",    "update_reservation_flights",  37),  # [confirmed]
    DirectlyFollows("get_reservation_details", "update_reservation_flights",  22),  # [confirmed]
    DirectlyFollows("get_user_details",        "search_direct_flight",        12),  # [confirmed]
    DirectlyFollows("cancel_reservation",      "book_reservation",             5),  # [confirmed]
    DirectlyFollows("search_direct_flight",    "update_reservation_baggages",  4),  # [confirmed]
    DirectlyFollows("update_reservation_flights","update_reservation_baggages",4),  # [confirmed]
]

# 3. CAUSAL_LINK — A가 만든 상태(predicate)를 B가 소비
CAUSAL_LINK: List[CausalLink] = [
    CausalLink("get_user_details",        "get_reservation_details",     "user_identified"),             # [confirmed-policy]
    CausalLink("get_user_details",        "book_reservation",            "user_profile_and_payment_known"),# [confirmed-policy]
    CausalLink("get_reservation_details", "cancel_reservation",          "reservation_located"),         # [confirmed-policy]
    CausalLink("get_reservation_details", "update_reservation_flights",  "reservation_located"),         # [confirmed-policy]
    CausalLink("search_direct_flight",    "update_reservation_flights",  "new_flight_options_known"),    # [confirmed]
    CausalLink("search_direct_flight",    "book_reservation",            "flight_availability_known"),   # [inferred]
    CausalLink("search_onestop_flight",   "book_reservation",            "connecting_options_known"),    # [inferred]
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 B: 의존/데이터 관계 ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 4. REQUIRES — B 호출에 A 선행 필요 (dependent, prerequisite)
REQUIRES: List[Tuple[str, str]] = [
    ("get_reservation_details",        "get_user_details"),          # [confirmed-policy] user_id first
    ("book_reservation",               "get_user_details"),          # [confirmed-policy] payment in profile
    ("book_reservation",               "search_direct_flight"),      # [inferred] flight options
    ("cancel_reservation",             "get_reservation_details"),   # [confirmed-policy]
    ("update_reservation_flights",     "get_reservation_details"),   # [confirmed-policy]
    ("update_reservation_flights",     "search_direct_flight"),      # [confirmed] new flights
    ("update_reservation_baggages",    "get_reservation_details"),   # [confirmed-policy]
    ("update_reservation_passengers",  "get_reservation_details"),   # [confirmed-policy]
    ("update_reservation_flights",     "get_user_details"),          # [confirmed-policy] payment method
]

# 5. ENABLES — A 호출 후 B 호출 가능해짐
ENABLES: List[Tuple[str, str]] = [
    ("get_user_details",        "get_reservation_details"),     # [confirmed-policy]
    ("get_user_details",        "book_reservation"),            # [confirmed-policy]
    ("get_reservation_details", "cancel_reservation"),          # [confirmed-policy]
    ("get_reservation_details", "update_reservation_flights"),  # [confirmed-policy]
    ("search_direct_flight",    "book_reservation"),            # [inferred]
    ("search_direct_flight",    "update_reservation_flights"),  # [confirmed]
]

# 6. PARAMETER_FEEDS — A 출력이 B 입력 인자로 사용
PARAMETER_FEEDS: List[ParamFeed] = [
    ParamFeed("get_user_details",        "get_reservation_details",    "reservation_id"),  # [confirmed]
    ParamFeed("search_direct_flight",    "book_reservation",           "flight_number"),   # [inferred]
    ParamFeed("search_direct_flight",    "update_reservation_flights", "flight_number"),   # [confirmed]
    ParamFeed("get_reservation_details", "cancel_reservation",         "reservation_id"),  # [confirmed]
    ParamFeed("get_reservation_details", "update_reservation_flights", "reservation_id"),  # [confirmed]
    ParamFeed("get_user_details",        "book_reservation",           "payment_id"),      # [confirmed-policy]
    ParamFeed("search_direct_flight",    "calculate",                  "flight_price"),    # [confirmed]
]

# 7. AND_JOIN — 여러 선행 도구 모두 완료돼야 target (BPMN AND-join)
AND_JOIN: List[AndJoin] = [
    AndJoin(("get_user_details", "search_direct_flight"),         "book_reservation"),           # [confirmed-policy]
    AndJoin(("get_reservation_details", "search_direct_flight"),  "update_reservation_flights"), # [confirmed]
    AndJoin(("get_user_details", "get_reservation_details"),      "cancel_reservation"),         # [confirmed-policy]
]

# 8. VALIDATES — A가 B 수행 전제 상태를 검증
VALIDATES: List[Tuple[str, str]] = [
    ("get_reservation_details", "cancel_reservation"),           # [confirmed-policy] 취소 자격(운항/24h/business/insurance)
    ("get_reservation_details", "update_reservation_flights"),   # [confirmed-policy] basic economy/운항 여부
    ("get_flight_status",       "update_reservation_flights"),   # [confirmed-policy] target flight available
    ("get_flight_status",       "cancel_reservation"),           # [confirmed-policy] no segment flown
    ("get_user_details",        "book_reservation"),             # [confirmed-policy] payment/membership
    ("get_user_details",        "send_certificate"),             # [confirmed-policy] compensation eligibility
]

# 9. RETRY_AFTER_FAIL — A 실패 후 B 수행 → A 재시도
RETRY_AFTER_FAIL: List[RetryAfterFail] = [
    RetryAfterFail("book_reservation",            "search_direct_flight"),  # [inferred] 좌석 사라짐→재검색
    RetryAfterFail("update_reservation_flights",  "search_direct_flight"),  # [inferred]
    RetryAfterFail("search_direct_flight",        "list_all_airports"),     # [hypothetical] 공항코드 확인
]

# 10. ERROR_FALLBACK — A 불가/실패 시 B 대안 (주로 escalation / 대체 검색)
ERROR_FALLBACK: List[ErrorFallback] = [
    ErrorFallback("search_direct_flight",       "search_onestop_flight"),     # [confirmed-policy] 직항 없음→경유
    ErrorFallback("cancel_reservation",         "transfer_to_human_agents"),  # [confirmed-policy] 운항후/불가
    ErrorFallback("update_reservation_flights", "transfer_to_human_agents"),  # [confirmed-policy] basic economy 등
    ErrorFallback("book_reservation",           "transfer_to_human_agents"),  # [inferred]
    ErrorFallback("update_reservation_passengers","transfer_to_human_agents"),# [confirmed-policy] 승객수 변경 불가
]

# 11. COMPENSATES — A가 B 효과를 역전
COMPENSATES: List[Tuple[str, str]] = [
    ("cancel_reservation", "book_reservation"),    # [inferred] 취소가 예약을 역전
    ("send_certificate",   "cancel_reservation"),  # [inferred] 취소 보상
    ("send_certificate",   "update_reservation_flights"),  # [hypothetical] 변경 불편 보상
]

# 12. TOOL_SUBSUMES — A가 B 기능을 포함/일반화
TOOL_SUBSUMES: List[Tuple[str, str]] = [
    ("get_user_details",        "get_reservation_details"),  # [inferred] user에 reservation 번호 포함
    ("get_reservation_details", "get_flight_status"),        # [inferred] reservation에 flight 정보 포함
    ("search_onestop_flight",   "search_direct_flight"),     # [inferred] 경유 검색이 직항 포함 탐색
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 C: 배타/병렬/조건 관계 (Symmetric/Conditional → T1) ────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 13. MUTEX — 같은 예약에 함께 적용 불가 (상태 배타)
MUTEX: List[Tuple[str, str]] = [
    ("cancel_reservation",         "update_reservation_flights"),     # [inferred] 취소된 예약 변경 불가
    ("cancel_reservation",         "update_reservation_baggages"),    # [inferred]
    ("cancel_reservation",         "update_reservation_passengers"),  # [inferred]
    ("cancel_reservation",         "book_reservation"),               # [inferred] 동일 예약 상태 배타
    ("update_reservation_flights", "book_reservation"),               # [hypothetical] 신규 vs 기존
]

# 14. EXCLUSIVE_CHOICE — 조건에 따라 A 또는 B 선택 (BPMN XOR)
EXCLUSIVE_CHOICE: List[ExclusiveChoice] = [
    ExclusiveChoice("direct_flight_available", "search_direct_flight",       "search_onestop_flight"),    # [confirmed-policy]
    ExclusiveChoice("user_wants_change_vs_cancel","update_reservation_flights","cancel_reservation"),     # [inferred]
    ExclusiveChoice("reservation_already_exists","update_reservation_flights","book_reservation"),        # [inferred]
]

# 15. PARALLEL_SAFE — 순서 무관 독립 조회
PARALLEL_SAFE: List[Tuple[str, str]] = [
    ("get_flight_status",   "get_user_details"),        # [inferred]
    ("list_all_airports",   "get_user_details"),        # [inferred]
    ("search_direct_flight","get_user_details"),        # [inferred]
    ("get_flight_status",   "list_all_airports"),       # [hypothetical]
]

# 16. CONDITIONAL_ON — 선행 도구 결과가 조건 충족 시만 호출
CONDITIONAL_ON: List[ConditionalOn] = [
    ConditionalOn("cancel_reservation",            "get_reservation_details", "no_segment_flown_and_eligible"),  # [confirmed-policy]
    ConditionalOn("update_reservation_flights",    "get_reservation_details", "not_basic_economy"),              # [confirmed-policy]
    ConditionalOn("update_reservation_flights",    "get_flight_status",       "target_flight_available"),        # [confirmed-policy]
    ConditionalOn("book_reservation",              "get_user_details",        "payment_method_in_profile"),      # [confirmed-policy]
    ConditionalOn("send_certificate",              "get_user_details",        "compensation_eligible"),          # [confirmed-policy]
    ConditionalOn("update_reservation_baggages",   "get_reservation_details", "only_adding_not_removing"),       # [confirmed-policy]
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 D: 상태/속성 관계 (Categorical → T1) ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 17. PRECONDITION_STATE — 도구 호출 전 충족돼야 할 상태 (PDDL :precondition)
PRECONDITION_STATE: Dict[str, str] = {
    "get_reservation_details":       "user_id_obtained",
    "book_reservation":              "user_id_flights_and_payment_ready",
    "cancel_reservation":            "reservation_located_and_no_segment_flown_and_eligible",
    "update_reservation_flights":    "reservation_located_and_not_basic_economy",
    "update_reservation_baggages":   "reservation_located",
    "update_reservation_passengers": "reservation_located",
    "send_certificate":              "compensation_requested_and_eligible",
    "transfer_to_human_agents":      "request_out_of_scope",
}

# 18. EFFECT_STATE — 도구 호출 후 상태 변화 (PDDL :effect add/del)
EFFECT_STATE: Dict[str, Dict[str, List[str]]] = {
    "get_user_details":              {"add": ["user_identified"], "del": []},
    "book_reservation":              {"add": ["reservation_created", "payment_charged"], "del": []},
    "cancel_reservation":            {"add": ["reservation_cancelled", "refund_initiated"], "del": ["reservation_active"]},
    "update_reservation_flights":    {"add": ["flights_updated", "price_difference_settled"], "del": []},
    "update_reservation_baggages":   {"add": ["baggage_added", "baggage_fee_charged"], "del": []},
    "update_reservation_passengers": {"add": ["passenger_info_updated"], "del": []},
    "send_certificate":              {"add": ["compensation_issued"], "del": []},
    "transfer_to_human_agents":      {"add": ["conversation_escalated"], "del": []},
}

# 19. STATE_TRANSITION — 도구가 유발하는 예약 상태 전이
STATE_TRANSITION: List[StateTransition] = [
    StateTransition("book_reservation",           "none",   "active"),                    # [confirmed-policy]
    StateTransition("cancel_reservation",         "active", "cancelled"),                 # [confirmed-policy]
    StateTransition("update_reservation_flights", "active", "active(flights_changed)"),   # [confirmed-policy]
    StateTransition("update_reservation_baggages","active", "active(baggage_added)"),     # [confirmed-policy]
]

# 20. WORKFLOW_ROLE — 워크플로 내 도구 역할
WORKFLOW_ROLE: Dict[str, str] = {
    "get_user_details":              "prerequisite",
    "get_reservation_details":       "prerequisite",
    "get_flight_status":             "prerequisite",
    "list_all_airports":             "prerequisite",
    "search_direct_flight":          "prerequisite",
    "search_onestop_flight":         "prerequisite",
    "calculate":                     "support",
    "think":                         "support",
    "book_reservation":              "main",
    "cancel_reservation":            "main",
    "update_reservation_flights":    "main",
    "update_reservation_baggages":   "main",
    "update_reservation_passengers": "main",
    "send_certificate":              "main",
    "transfer_to_human_agents":      "cleanup",
}

# 21. DOMAIN_CATEGORY — 도구 문제 도메인 분류
DOMAIN_CATEGORY: Dict[str, str] = {
    "get_user_details":              "user_info",
    "get_reservation_details":       "reservation_info",
    "get_flight_status":             "flight_search",
    "list_all_airports":             "flight_search",
    "search_direct_flight":          "flight_search",
    "search_onestop_flight":         "flight_search",
    "calculate":                     "computation",
    "think":                         "reasoning",
    "book_reservation":              "booking",
    "cancel_reservation":            "cancellation",
    "update_reservation_flights":    "flight_modification",
    "update_reservation_baggages":   "baggage",
    "update_reservation_passengers": "passenger_management",
    "send_certificate":              "compensation",
    "transfer_to_human_agents":      "escalation",
}

# 22. CHECKPOINT — 반드시 성공해야 다음 단계 진행
CHECKPOINT: List[str] = [
    "get_user_details",          # [confirmed-policy] user_id 게이트
    "get_reservation_details",   # [confirmed-policy] reservation locate 게이트 (modify/cancel)
    "search_direct_flight",      # [confirmed] flight 가용 게이트 (book/modify)
]

# 23. IDEMPOTENT — 같은 인자 반복 호출 시 결과 동일
IDEMPOTENT: Dict[str, bool] = {
    "get_user_details":              True,
    "get_reservation_details":       True,
    "get_flight_status":             True,
    "list_all_airports":             True,
    "search_direct_flight":          True,
    "search_onestop_flight":         True,
    "calculate":                     True,
    "think":                         True,
    "book_reservation":              False,  # 새 예약 생성
    "cancel_reservation":            False,
    "update_reservation_flights":    False,  # 가격 정산
    "update_reservation_baggages":   False,  # [confirmed-policy] add-only, 매번 과금
    "update_reservation_passengers": True,   # [inferred] 정보 재수정 가능
    "send_certificate":              False,
    "transfer_to_human_agents":      False,
}

# 24. REVERSIBLE — 호출 후 취소/되돌리기 가능 여부
REVERSIBLE: Dict[str, bool] = {
    "get_user_details":              True,
    "get_reservation_details":       True,
    "get_flight_status":             True,
    "list_all_airports":             True,
    "search_direct_flight":          True,
    "search_onestop_flight":         True,
    "calculate":                     True,
    "think":                         True,
    "book_reservation":              True,   # [inferred] cancel로 되돌림 가능
    "cancel_reservation":            False,  # [confirmed-policy] 환불은 되나 취소 자체 불가역
    "update_reservation_flights":    False,  # [confirmed-policy] 가격 정산됨
    "update_reservation_baggages":   False,  # [confirmed-policy] baggage 제거 불가 (add-only)
    "update_reservation_passengers": True,   # [inferred] 정보 재수정 가능
    "send_certificate":              False,
    "transfer_to_human_agents":      False,
}

# 25. MANDATORY_IN_FLOW — 대부분 플로우에 반드시 등장
MANDATORY_IN_FLOW: Dict[str, bool] = {
    "get_reservation_details":       True,   # [confirmed] 57, modify/cancel 필수
    "get_user_details":              True,   # [confirmed-policy] user_id 필수
    "search_direct_flight":          False,  # book/modify에만
    "search_onestop_flight":         False,
    "get_flight_status":             False,
    "list_all_airports":             False,
    "calculate":                     False,
    "think":                         False,
    "book_reservation":              False,
    "cancel_reservation":            False,
    "update_reservation_flights":    False,
    "update_reservation_baggages":   False,
    "update_reservation_passengers": False,
    "send_certificate":              False,
    "transfer_to_human_agents":      False,
}

# 26. OPTIONAL_IN_FLOW — MANDATORY의 반대
OPTIONAL_IN_FLOW: Dict[str, bool] = {k: not v for k, v in MANDATORY_IN_FLOW.items()}

# 27. LOOP_CAPABLE — 같은 세션에서 여러 번 의미 있게 호출 가능
LOOP_CAPABLE: Dict[str, bool] = {
    "get_user_details":              True,
    "get_reservation_details":       True,   # 여러 예약 조회 (57회 최다)
    "get_flight_status":             True,
    "list_all_airports":             False,
    "search_direct_flight":          True,    # 여러 노선 검색
    "search_onestop_flight":         True,
    "calculate":                     True,
    "think":                         True,
    "book_reservation":              False,
    "cancel_reservation":            False,
    "update_reservation_flights":    False,
    "update_reservation_baggages":   True,    # 여러 bag 추가 가능
    "update_reservation_passengers": True,
    "send_certificate":              False,
    "transfer_to_human_agents":      False,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 E/G: GoT/ToT/Harness 관계 ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 28. BACKTRACK_TO — 막다른 도구에서 복귀 지점으로 (ToT backtrack)
BACKTRACK_TO: List[BacktrackTo] = [
    BacktrackTo("book_reservation",           "search_direct_flight"),     # [inferred] 좌석 사라짐→재검색
    BacktrackTo("cancel_reservation",         "get_reservation_details"),  # [inferred] 취소 불가→재확인
    BacktrackTo("update_reservation_flights", "search_direct_flight"),     # [inferred]
]

# 29. PRUNED_BY — A 선택지가 B 수행으로 무효화됨 (ToT prune)
PRUNED_BY: List[Tuple[str, str]] = [
    ("update_reservation_flights",    "cancel_reservation"),  # [inferred] 취소되면 변경 차단
    ("update_reservation_baggages",   "cancel_reservation"),  # [inferred]
    ("update_reservation_passengers", "cancel_reservation"),  # [inferred]
]

# 30. FAN_OUT — 한 도구가 여러 후속으로 분기 (GoT branch)
FAN_OUT: List[FanOut] = [
    FanOut("get_reservation_details", ("cancel_reservation", "update_reservation_flights",
                                       "update_reservation_baggages", "update_reservation_passengers")),  # [confirmed]
    FanOut("get_user_details",        ("get_reservation_details", "book_reservation", "search_direct_flight")),  # [confirmed]
    FanOut("search_direct_flight",    ("book_reservation", "update_reservation_flights")),                # [confirmed]
]

# 31. OBSERVATION_TRIGGERS — 관찰된 상태가 특정 도구 호출을 유발
OBSERVATION_TRIGGERS: Dict[str, str] = {
    "user_unidentified":         "get_user_details",
    "reservation_id_unknown":    "get_user_details",
    "reservation_unknown":       "get_reservation_details",
    "flight_options_needed":     "search_direct_flight",
    "no_direct_flight":          "search_onestop_flight",
    "flight_status_unknown":     "get_flight_status",
    "request_out_of_scope":      "transfer_to_human_agents",
}

# 32. GUARDRAIL — 조건 활성화 시 도구 호출 차단 (Harness prohibition)
GUARDRAIL: Dict[str, str] = {
    "transfer_to_human_agents":       "is_first_action_or_request_in_scope",     # [confirmed-policy]
    "cancel_reservation":             "segment_already_flown_or_ineligible",     # [confirmed-policy]
    "update_reservation_flights":     "basic_economy_or_changes_origin_dest_triptype",  # [confirmed-policy]
    "update_reservation_baggages":    "attempt_to_remove_baggage",               # [confirmed-policy] add-only
    "update_reservation_passengers":  "attempt_to_change_passenger_count",       # [confirmed-policy]
    "book_reservation":               "payment_not_in_profile_or_over_five_passengers",  # [confirmed-policy]
    "send_certificate":               "regular_member_no_insurance_economy_or_not_requested",  # [confirmed-policy]
}

# 33. SCORED_PREFERENCE — 둘 다 유효하나 맥락에서 preferred 우위 (GoT score/keep)
SCORED_PREFERENCE: List[ScoredPreference] = [
    ScoredPreference("search_direct_flight",     "search_onestop_flight", "direct_route_exists"),    # [confirmed-policy]
    ScoredPreference("update_reservation_flights","cancel_reservation",   "user_wants_to_keep_trip"),# [inferred]
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 H: HTN 계층 관계 ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 34. DECOMPOSES_INTO — 추상 목표가 구체 도구 집합으로 분해 (HTN top-down)
DECOMPOSES_INTO: List[DecomposesInto] = [
    DecomposesInto("book_flight",
        ("get_user_details", "search_direct_flight", "book_reservation")),                                # [confirmed-policy]
    DecomposesInto("modify_flight",
        ("get_user_details", "get_reservation_details", "search_direct_flight", "update_reservation_flights")),  # [confirmed]
    DecomposesInto("cancel_flight",
        ("get_user_details", "get_reservation_details", "cancel_reservation")),                           # [confirmed]
    DecomposesInto("add_baggage",
        ("get_reservation_details", "update_reservation_baggages")),                                      # [confirmed]
    DecomposesInto("compensate_user",
        ("get_reservation_details", "send_certificate")),                                                 # [inferred]
]

# 35. SUBTASK_OF — 도구가 속하는 추상 단계 (DECOMPOSES_INTO 역관계)
SUBTASK_OF: Dict[str, str] = {
    "get_user_details":              "authentication_phase",
    "get_reservation_details":       "information_gathering_phase",
    "search_direct_flight":          "flight_search_phase",
    "search_onestop_flight":         "flight_search_phase",
    "list_all_airports":             "flight_search_phase",
    "get_flight_status":             "flight_search_phase",
    "calculate":                     "support_phase",
    "think":                         "support_phase",
    "book_reservation":              "booking_phase",
    "cancel_reservation":            "cancellation_phase",
    "update_reservation_flights":    "flight_modification_phase",
    "update_reservation_baggages":   "baggage_phase",
    "update_reservation_passengers": "passenger_phase",
    "send_certificate":              "compensation_phase",
    "transfer_to_human_agents":      "escalation_phase",
}

# 36. ACHIEVES_GOAL — 도구가 달성하는 고객 레벨 목표 (HTN top-level goal)
ACHIEVES_GOAL: Dict[str, str] = {
    "book_reservation":              "book_flight",
    "cancel_reservation":            "cancel_flight",
    "update_reservation_flights":    "change_flights",
    "update_reservation_baggages":   "add_baggage",
    "update_reservation_passengers": "update_passenger_info",
    "send_certificate":              "issue_compensation",
    "search_direct_flight":          "find_flight_options",
    "search_onestop_flight":         "find_connecting_options",
    "get_reservation_details":       "retrieve_reservation",
    "get_user_details":              "identify_customer",
    "transfer_to_human_agents":      "escalate_unresolved_issue",
}

# 37. REFINES — 추상 행동이 맥락에 따라 구체 도구로 구체화 (HTN method)
REFINES: List[Refines] = [
    Refines("search_flight",       "search_direct_flight",          "direct_route_preferred"),  # [confirmed-policy]
    Refines("search_flight",       "search_onestop_flight",         "no_direct_route"),         # [confirmed-policy]
    Refines("modify_reservation",  "update_reservation_flights",    "wants_different_flights"),  # [inferred]
    Refines("modify_reservation",  "update_reservation_baggages",   "wants_more_baggage"),       # [inferred]
    Refines("modify_reservation",  "update_reservation_passengers", "passenger_info_correction"),# [inferred]
    Refines("resolve_request",     "cancel_reservation",            "wants_cancel"),             # [inferred]
    Refines("resolve_request",     "book_reservation",              "wants_new_booking"),        # [inferred]
]


# ═══════════════════════════════════════════════════════════════════════════════
# 관계 기하 분류 (도메인-독립; telecom/retail과 동일) → 개입 방법 예측
# ═══════════════════════════════════════════════════════════════════════════════
RELATION_GEOMETRY: Dict[str, str] = {
    "precedes": "directional", "directly_follows": "directional", "causal_link": "directional",
    "requires": "directional", "enables": "directional", "parameter_feeds": "directional",
    "and_join": "directional", "validates": "directional", "retry_after_fail": "directional",
    "error_fallback": "directional", "compensates": "directional", "tool_subsumes": "directional",
    "state_transition": "directional", "backtrack_to": "directional", "pruned_by": "directional",
    "fan_out": "directional", "decomposes_into": "directional",
    "mutex": "symmetric", "exclusive_choice": "conditional", "parallel_safe": "symmetric",
    "conditional_on": "conditional", "scored_preference": "conditional", "refines": "conditional",
    "workflow_role": "categorical", "domain_category": "categorical", "checkpoint": "categorical",
    "precondition_state": "categorical", "effect_state": "categorical", "idempotent": "categorical",
    "reversible": "categorical", "mandatory_in_flow": "categorical", "optional_in_flow": "categorical",
    "loop_capable": "categorical", "observation_triggers": "categorical", "guardrail": "categorical",
    "subtask_of": "categorical", "achieves_goal": "categorical",
}

PREDICTED_METHOD: Dict[str, str] = {
    k: ("A6" if v == "directional" else "T1") for k, v in RELATION_GEOMETRY.items()
}


# ═══════════════════════════════════════════════════════════════════════════════
# 실측 빈도 (tasks.json 43 tasks w/actions, ordered-pair counts) — [confirmed]
# ═══════════════════════════════════════════════════════════════════════════════
PRECEDES_FREQ: Dict[Tuple[str, str], int] = {
    ("get_reservation_details", "search_direct_flight"): 59,
    ("get_user_details", "get_reservation_details"): 42,
    ("get_reservation_details", "cancel_reservation"): 40,
    ("search_direct_flight", "update_reservation_flights"): 37,
    ("get_reservation_details", "update_reservation_flights"): 22,
    ("get_user_details", "search_direct_flight"): 12,
    ("cancel_reservation", "book_reservation"): 5,
    ("get_user_details", "update_reservation_flights"): 5,
    ("get_user_details", "cancel_reservation"): 5,
    ("search_direct_flight", "update_reservation_baggages"): 4,
    ("update_reservation_flights", "update_reservation_baggages"): 4,
}

TOOL_FREQ: Dict[str, int] = {
    "get_reservation_details": 57, "update_reservation_flights": 20, "search_direct_flight": 20,
    "get_user_details": 14, "cancel_reservation": 11, "book_reservation": 10,
    "update_reservation_baggages": 5, "update_reservation_passengers": 3, "calculate": 1,
    "transfer_to_human_agents": 1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 도구 설명 (docstring 요약)
# ═══════════════════════════════════════════════════════════════════════════════
TOOL_DESC: Dict[str, str] = {
    "get_user_details":              "Get the user profile (reservations, payment methods, membership).",
    "get_reservation_details":       "Get the details of a reservation by id.",
    "get_flight_status":             "Get the status of a flight on a date (available/delayed/flying).",
    "list_all_airports":             "List all airports and their city codes.",
    "search_direct_flight":          "Search direct flights between origin and destination on a date.",
    "search_onestop_flight":         "Search one-stop (connecting) flights when no direct route fits.",
    "calculate":                     "Evaluate an arithmetic expression (e.g. price difference).",
    "think":                         "Internal reasoning scratchpad; no state change.",
    "book_reservation":              "Create a reservation (flights, passengers, cabin, payment, baggage, insurance).",
    "cancel_reservation":            "Cancel a reservation if eligibility rules are met; refund to original methods.",
    "update_reservation_flights":    "Change the flights of a reservation (not basic economy; same origin/dest/trip).",
    "update_reservation_baggages":   "Add checked bags to a reservation (add-only; 50 USD per extra bag).",
    "update_reservation_passengers": "Modify passenger info (cannot change the number of passengers).",
    "send_certificate":              "Issue a travel certificate as compensation (only if requested and eligible).",
    "transfer_to_human_agents":      "Escalate to a human agent when out of scope (not as first action).",
}

TOOL_TYPE: Dict[str, str] = {
    "get_user_details": "READ", "get_reservation_details": "READ", "get_flight_status": "READ",
    "list_all_airports": "READ", "search_direct_flight": "READ", "search_onestop_flight": "READ",
    "calculate": "GENERIC", "transfer_to_human_agents": "GENERIC", "think": "THINK",
    "book_reservation": "WRITE", "cancel_reservation": "WRITE", "update_reservation_flights": "WRITE",
    "update_reservation_baggages": "WRITE", "update_reservation_passengers": "WRITE", "send_certificate": "WRITE",
}
