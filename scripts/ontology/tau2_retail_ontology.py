"""tau2_retail_ontology.py  —  v1

τ²-bench retail 도메인 온톨로지 관계 정의 (42-relation, telecom 포맷 호환).

구성 방법 (LLM-AFOD, 2026-05-29):
  - 도구 인벤토리: src/tau2/domains/retail/tools.py (@is_tool, N=17)
  - [confirmed]:  retail tasks.json(112 tasks) 액션 동시출현/순서 통계 마이닝
  - [confirmed-policy]: data/tau2/domains/retail/policy.md 명시 규칙
  - [inferred]:   도구 docstring + 정책 의미론 추론 (Claude Opus 4.8 / Claude Code)
  - [hypothetical]: 이론적 추정, retail에서 희소

provenance 주석은 telecom 온톨로지(v4)와 동일 규약. RELATION_GEOMETRY/
PREDICTED_METHOD는 도메인-독립이라 telecom에서 그대로 가져옴.

retail 핵심 워크플로 (policy.md):
  1) 인증: find_user_id_by_email (기본) 또는 find_user_id_by_name_zip → user_id
  2) 조회: get_user_details / get_order_details / get_product_details (상태 확인)
  3) 액션 (pending: cancel/modify_*, delivered: return/exchange) — 항상 명시 확인 후
  4) 범위 밖이면 transfer_to_human_agents (첫 액션 금지)
  제약: 한 user/대화, write 전 명시 confirm, modify_items/return/exchange는 주문당 1회.

Tools (N=17):
  READ:    find_user_id_by_email, find_user_id_by_name_zip, get_user_details,
           get_order_details, get_product_details, get_item_details,
           list_all_product_types
  WRITE:   cancel_pending_order, modify_pending_order_address,
           modify_pending_order_items, modify_pending_order_payment,
           modify_user_address, return_delivered_order_items,
           exchange_delivered_order_items
  GENERIC: calculate, transfer_to_human_agents
  THINK:   think
"""
from typing import Dict, FrozenSet, List, NamedTuple, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 타입 정의 (telecom 포맷과 동일)
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

# 1. PRECEDES — A는 B보다 먼저 호출 (eventual ordering)
PRECEDES: List[Tuple[str, str]] = [
    ("find_user_id_by_name_zip",     "get_order_details"),            # [confirmed] 148
    ("find_user_id_by_email",        "get_order_details"),            # [confirmed] 37
    ("find_user_id_by_name_zip",     "get_user_details"),             # [confirmed] 52
    ("find_user_id_by_name_zip",     "get_product_details"),          # [confirmed] 53
    ("get_user_details",             "get_order_details"),            # [confirmed] 150
    ("get_order_details",            "return_delivered_order_items"), # [confirmed] 96
    ("get_order_details",            "cancel_pending_order"),         # [confirmed] 58
    ("get_order_details",            "modify_pending_order_items"),   # [confirmed] 45
    ("get_order_details",            "exchange_delivered_order_items"),# [confirmed] 30
    ("get_order_details",            "modify_pending_order_address"), # [confirmed] 19
    ("get_product_details",          "modify_pending_order_items"),   # [confirmed] 28
    ("get_product_details",          "exchange_delivered_order_items"),# [confirmed] 25
    ("get_order_details",            "modify_pending_order_payment"), # [inferred]
]

# 2. DIRECTLY_FOLLOWS — A 직후 B (empirical adjacency, freq)
DIRECTLY_FOLLOWS: List[DirectlyFollows] = [
    DirectlyFollows("get_user_details",             "get_order_details",             150),  # [confirmed]
    DirectlyFollows("find_user_id_by_name_zip",     "get_order_details",             148),  # [confirmed]
    DirectlyFollows("get_order_details",            "return_delivered_order_items",   96),  # [confirmed]
    DirectlyFollows("get_order_details",            "cancel_pending_order",           58),  # [confirmed]
    DirectlyFollows("get_order_details",            "modify_pending_order_items",     45),  # [confirmed]
    DirectlyFollows("get_product_details",          "modify_pending_order_items",     28),  # [confirmed]
    DirectlyFollows("get_product_details",          "exchange_delivered_order_items", 25),  # [confirmed]
    DirectlyFollows("modify_pending_order_address", "modify_pending_order_items",     18),  # [confirmed]
]

# 3. CAUSAL_LINK — A가 만든 상태(predicate)를 B가 소비 (PDDL causal chain)
CAUSAL_LINK: List[CausalLink] = [
    CausalLink("find_user_id_by_email",     "get_order_details",              "user_authenticated"),       # [confirmed-policy]
    CausalLink("find_user_id_by_name_zip",  "get_order_details",              "user_authenticated"),       # [confirmed-policy]
    CausalLink("get_user_details",          "get_order_details",              "order_id_resolved"),        # [inferred]
    CausalLink("get_order_details",         "cancel_pending_order",          "order_status_verified"),    # [confirmed-policy]
    CausalLink("get_order_details",         "return_delivered_order_items",  "order_status_verified"),    # [confirmed-policy]
    CausalLink("get_product_details",       "modify_pending_order_items",    "variant_availability_known"),# [confirmed-policy]
    CausalLink("get_product_details",       "exchange_delivered_order_items","variant_availability_known"),# [confirmed-policy]
    CausalLink("get_user_details",          "modify_pending_order_payment",  "payment_method_known"),     # [inferred]
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 B: 의존/데이터 관계 ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 4. REQUIRES — B 호출에 A 선행 필요 (dependent, prerequisite)
REQUIRES: List[Tuple[str, str]] = [
    ("get_order_details",               "find_user_id_by_email"),       # [confirmed-policy] auth-first (alt)
    ("get_order_details",               "find_user_id_by_name_zip"),    # [confirmed-policy] auth-first (alt)
    ("get_user_details",                "find_user_id_by_email"),       # [confirmed-policy]
    ("cancel_pending_order",            "get_order_details"),           # [confirmed-policy] status check
    ("modify_pending_order_items",      "get_order_details"),           # [confirmed-policy]
    ("modify_pending_order_address",    "get_order_details"),           # [confirmed-policy]
    ("modify_pending_order_payment",    "get_order_details"),           # [confirmed-policy]
    ("return_delivered_order_items",    "get_order_details"),           # [confirmed-policy]
    ("exchange_delivered_order_items",  "get_order_details"),           # [confirmed-policy]
    ("modify_pending_order_items",      "get_product_details"),         # [confirmed] variant validation
    ("exchange_delivered_order_items",  "get_product_details"),         # [confirmed]
]

# 5. ENABLES — A 호출 후 B 호출 가능해짐 (PDDL positive effect)
ENABLES: List[Tuple[str, str]] = [
    ("find_user_id_by_email",       "get_order_details"),            # [confirmed-policy]
    ("find_user_id_by_name_zip",    "get_order_details"),            # [confirmed-policy]
    ("find_user_id_by_email",       "get_user_details"),             # [confirmed-policy]
    ("get_order_details",           "cancel_pending_order"),         # [confirmed-policy]
    ("get_order_details",           "return_delivered_order_items"), # [confirmed-policy]
    ("get_order_details",           "modify_pending_order_items"),   # [confirmed-policy]
    ("get_product_details",         "exchange_delivered_order_items"),# [confirmed]
    ("get_product_details",         "modify_pending_order_items"),   # [confirmed]
]

# 6. PARAMETER_FEEDS — A 출력이 B 입력 인자로 사용 (Routine output binding)
PARAMETER_FEEDS: List[ParamFeed] = [
    ParamFeed("find_user_id_by_email",     "get_user_details",              "user_id"),    # [confirmed-policy]
    ParamFeed("find_user_id_by_name_zip",  "get_order_details",             "user_id"),    # [inferred]
    ParamFeed("get_user_details",          "get_order_details",             "order_id"),   # [confirmed] order list
    ParamFeed("get_order_details",         "cancel_pending_order",          "order_id"),   # [confirmed]
    ParamFeed("get_order_details",         "return_delivered_order_items",  "item_ids"),   # [confirmed]
    ParamFeed("get_product_details",       "modify_pending_order_items",    "new_item_id"),# [confirmed]
    ParamFeed("get_product_details",       "exchange_delivered_order_items","new_item_id"),# [confirmed]
    ParamFeed("get_order_details",         "calculate",                     "price_diff"), # [inferred]
]

# 7. AND_JOIN — 여러 선행 도구 모두 완료돼야 target 호출 (BPMN AND-join / GAP merge)
AND_JOIN: List[AndJoin] = [
    AndJoin(("get_order_details", "get_product_details"), "modify_pending_order_items"),    # [confirmed]
    AndJoin(("get_order_details", "get_product_details"), "exchange_delivered_order_items"),# [confirmed]
    AndJoin(("get_order_details", "get_user_details"),    "modify_pending_order_payment"),  # [inferred] 잔액 확인
]

# 8. VALIDATES — A가 B 수행 전제 상태를 검증 (KnowAgent verification)
VALIDATES: List[Tuple[str, str]] = [
    ("get_order_details",   "cancel_pending_order"),            # [confirmed-policy] status==pending 검증
    ("get_order_details",   "modify_pending_order_items"),      # [confirmed-policy] status==pending
    ("get_order_details",   "return_delivered_order_items"),    # [confirmed-policy] status==delivered
    ("get_order_details",   "exchange_delivered_order_items"),  # [confirmed-policy] status==delivered
    ("get_product_details", "modify_pending_order_items"),      # [confirmed] variant 존재/가용 검증
    ("get_item_details",    "exchange_delivered_order_items"),  # [inferred] item 가용 검증
    ("get_user_details",    "modify_pending_order_payment"),    # [inferred] 결제수단/잔액 검증
]

# 9. RETRY_AFTER_FAIL — A 실패 후 B 수행 → A 재시도 (Routine error recovery)
RETRY_AFTER_FAIL: List[RetryAfterFail] = [
    RetryAfterFail("find_user_id_by_email", "find_user_id_by_name_zip"),  # [confirmed-policy] email 실패→name+zip
    RetryAfterFail("get_order_details",     "get_user_details"),          # [inferred] order_id 모를 때 user에서 재탐색
    RetryAfterFail("modify_pending_order_items", "get_product_details"),  # [hypothetical] 잘못된 variant→재조회
]

# 10. ERROR_FALLBACK — A 불가/실패 시 B를 대안으로 (Routine alternative path)
#     retry와 차이: B는 재시도가 아닌 다른 접근(주로 escalation)
ERROR_FALLBACK: List[ErrorFallback] = [
    ErrorFallback("find_user_id_by_email",          "find_user_id_by_name_zip"),  # [confirmed-policy]
    ErrorFallback("cancel_pending_order",           "transfer_to_human_agents"),  # [confirmed-policy] 범위 밖
    ErrorFallback("modify_pending_order_items",     "transfer_to_human_agents"),  # [confirmed-policy]
    ErrorFallback("return_delivered_order_items",   "transfer_to_human_agents"),  # [confirmed-policy]
    ErrorFallback("exchange_delivered_order_items", "transfer_to_human_agents"),  # [confirmed-policy]
    ErrorFallback("modify_pending_order_payment",   "transfer_to_human_agents"),  # [inferred]
]

# 11. COMPENSATES — A가 B 효과를 역전 (PDDL negative effect / BPMN compensation)
COMPENSATES: List[Tuple[str, str]] = [
    ("cancel_pending_order",         "modify_pending_order_items"),   # [hypothetical] 취소가 변경을 무의미화
    ("modify_pending_order_payment", "modify_pending_order_payment"), # [hypothetical] 재변경으로 환불/재청구
    ("exchange_delivered_order_items","return_delivered_order_items"),# [hypothetical] 교환 vs 반품 상호 대체
]

# 12. TOOL_SUBSUMES — A가 B 기능을 포함/일반화 (KG entailment)
TOOL_SUBSUMES: List[Tuple[str, str]] = [
    ("get_user_details",       "get_order_details"),   # [inferred] user details에 주문 목록 포함
    ("list_all_product_types", "get_product_details"), # [inferred] 전체 목록이 단일 조회 포함
    ("get_product_details",    "get_item_details"),    # [inferred] product에 variant(item) 포함
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 C: 배타/병렬/조건 관계 (Symmetric/Conditional → T1) ────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 13. MUTEX — 동일 주문에 함께 적용 불가 (상태 배타 / once-only)
MUTEX: List[Tuple[str, str]] = [
    ("cancel_pending_order",          "return_delivered_order_items"),    # [confirmed-policy] pending vs delivered
    ("cancel_pending_order",          "exchange_delivered_order_items"),  # [confirmed-policy]
    ("modify_pending_order_items",    "return_delivered_order_items"),    # [confirmed-policy]
    ("modify_pending_order_items",    "exchange_delivered_order_items"),  # [confirmed-policy]
    ("cancel_pending_order",          "modify_pending_order_items"),      # [confirmed-policy] modify_items 후 cancel 불가
    ("return_delivered_order_items",  "exchange_delivered_order_items"),  # [confirmed-policy] 주문당 1회, 택1
]

# 14. EXCLUSIVE_CHOICE — 조건에 따라 A 또는 B 선택 (BPMN XOR gateway)
EXCLUSIVE_CHOICE: List[ExclusiveChoice] = [
    ExclusiveChoice("user_provided_email",      "find_user_id_by_email",        "find_user_id_by_name_zip"),    # [confirmed-policy]
    ExclusiveChoice("order_status_is_pending",  "modify_pending_order_items",   "cancel_pending_order"),        # [inferred]
    ExclusiveChoice("order_status_is_delivered","return_delivered_order_items", "exchange_delivered_order_items"),# [inferred]
    ExclusiveChoice("wants_same_product_variant","exchange_delivered_order_items","return_delivered_order_items"),# [inferred]
]

# 15. PARALLEL_SAFE — 순서 무관 독립 조회 (정책상 turn당 1 tool이나 논리적 독립)
PARALLEL_SAFE: List[Tuple[str, str]] = [
    ("get_product_details",    "get_user_details"),   # [inferred]
    ("get_item_details",       "get_user_details"),   # [inferred]
    ("list_all_product_types", "get_user_details"),   # [inferred]
    ("calculate",              "get_product_details"), # [hypothetical]
]

# 16. CONDITIONAL_ON — 선행 도구 결과가 조건 충족 시만 호출 (Routine conditional)
CONDITIONAL_ON: List[ConditionalOn] = [
    ConditionalOn("cancel_pending_order",           "get_order_details", "status_is_pending"),        # [confirmed-policy]
    ConditionalOn("modify_pending_order_items",     "get_order_details", "status_is_pending"),        # [confirmed-policy]
    ConditionalOn("modify_pending_order_address",   "get_order_details", "status_is_pending"),        # [confirmed-policy]
    ConditionalOn("modify_pending_order_payment",   "get_order_details", "status_is_pending"),        # [confirmed-policy]
    ConditionalOn("return_delivered_order_items",   "get_order_details", "status_is_delivered"),      # [confirmed-policy]
    ConditionalOn("exchange_delivered_order_items", "get_order_details", "status_is_delivered"),      # [confirmed-policy]
    ConditionalOn("modify_pending_order_payment",   "get_user_details",  "gift_card_balance_sufficient"),# [confirmed-policy]
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 D: 상태/속성 관계 (Categorical → T1) ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 17. PRECONDITION_STATE — 도구 호출 전 충족돼야 할 상태 (PDDL :precondition)
PRECONDITION_STATE: Dict[str, str] = {
    "get_order_details":              "user_is_authenticated",
    "get_user_details":               "user_is_authenticated",
    "cancel_pending_order":           "order_status_is_pending",
    "modify_pending_order_address":   "order_status_is_pending",
    "modify_pending_order_items":     "order_status_is_pending_and_not_yet_modified",
    "modify_pending_order_payment":   "order_status_is_pending",
    "return_delivered_order_items":   "order_status_is_delivered",
    "exchange_delivered_order_items": "order_status_is_delivered",
    "transfer_to_human_agents":       "request_is_out_of_scope",
}

# 18. EFFECT_STATE — 도구 호출 후 시스템 상태 변화 (PDDL :effect add/del)
EFFECT_STATE: Dict[str, Dict[str, List[str]]] = {
    "find_user_id_by_email":        {"add": ["user_authenticated"], "del": []},
    "find_user_id_by_name_zip":     {"add": ["user_authenticated"], "del": []},
    "cancel_pending_order":         {"add": ["order_cancelled", "refund_initiated"], "del": ["order_pending"]},
    "modify_pending_order_address": {"add": ["shipping_address_updated"], "del": []},
    "modify_pending_order_items":   {"add": ["order_items_modified", "order_locked"], "del": ["order_modifiable", "order_cancellable"]},
    "modify_pending_order_payment": {"add": ["payment_method_updated", "refund_or_charge_initiated"], "del": []},
    "modify_user_address":          {"add": ["profile_address_updated"], "del": []},
    "return_delivered_order_items": {"add": ["return_requested", "refund_pending"], "del": ["order_delivered_actionable"]},
    "exchange_delivered_order_items":{"add": ["exchange_completed"], "del": ["order_delivered_actionable"]},
    "transfer_to_human_agents":     {"add": ["conversation_escalated"], "del": []},
}

# 19. STATE_TRANSITION — 도구가 유발하는 주문 상태 전이 (KnowAgent action→state)
STATE_TRANSITION: List[StateTransition] = [
    StateTransition("cancel_pending_order",           "pending",   "cancelled"),               # [confirmed-policy]
    StateTransition("modify_pending_order_items",     "pending",   "pending(items_modified)"), # [confirmed-policy]
    StateTransition("return_delivered_order_items",   "delivered", "return_requested"),        # [confirmed-policy]
    StateTransition("exchange_delivered_order_items", "delivered", "exchanged"),               # [confirmed-policy]
]

# 20. WORKFLOW_ROLE — 워크플로 내 도구 역할
WORKFLOW_ROLE: Dict[str, str] = {
    "find_user_id_by_email":          "prerequisite",
    "find_user_id_by_name_zip":       "prerequisite",
    "get_user_details":               "prerequisite",
    "get_order_details":              "prerequisite",
    "get_product_details":            "prerequisite",
    "get_item_details":               "prerequisite",
    "list_all_product_types":         "prerequisite",
    "calculate":                      "support",
    "think":                          "support",
    "cancel_pending_order":           "main",
    "modify_pending_order_address":   "main",
    "modify_pending_order_items":     "main",
    "modify_pending_order_payment":   "main",
    "modify_user_address":            "main",
    "return_delivered_order_items":   "main",
    "exchange_delivered_order_items": "main",
    "transfer_to_human_agents":       "cleanup",
}

# 21. DOMAIN_CATEGORY — 도구 문제 도메인 분류 (KG type hierarchy)
DOMAIN_CATEGORY: Dict[str, str] = {
    "find_user_id_by_email":          "authentication",
    "find_user_id_by_name_zip":       "authentication",
    "get_user_details":               "user_info",
    "get_order_details":              "order_info",
    "get_product_details":            "catalog",
    "get_item_details":               "catalog",
    "list_all_product_types":         "catalog",
    "calculate":                      "computation",
    "think":                          "reasoning",
    "cancel_pending_order":           "order_cancellation",
    "modify_pending_order_address":   "order_modification",
    "modify_pending_order_items":     "order_modification",
    "modify_pending_order_payment":   "payment",
    "modify_user_address":            "profile_management",
    "return_delivered_order_items":   "returns",
    "exchange_delivered_order_items": "exchanges",
    "transfer_to_human_agents":       "escalation",
}

# 22. CHECKPOINT — 반드시 성공해야 다음 단계 진행 (Routine must-succeed gate)
CHECKPOINT: List[str] = [
    "find_user_id_by_email",      # [confirmed-policy] 인증 게이트 (실패 시 모든 액션 불가)
    "find_user_id_by_name_zip",   # [confirmed-policy] 인증 게이트 (대안)
    "get_order_details",          # [confirmed-policy] 상태 검증 게이트 (write 전 필수)
]

# 23. IDEMPOTENT — 같은 인자 반복 호출 시 결과 동일 (부작용 없음)
IDEMPOTENT: Dict[str, bool] = {
    "find_user_id_by_email":          True,
    "find_user_id_by_name_zip":       True,
    "get_user_details":               True,
    "get_order_details":              True,
    "get_product_details":            True,
    "get_item_details":               True,
    "list_all_product_types":         True,
    "calculate":                      True,
    "think":                          True,
    "cancel_pending_order":           False,  # [confirmed-policy] 1회
    "modify_pending_order_address":   False,  # [confirmed-policy] 주문당 1회
    "modify_pending_order_items":     False,  # [confirmed-policy] 1회 후 lock
    "modify_pending_order_payment":   False,  # [confirmed-policy]
    "modify_user_address":            True,   # [inferred] 프로필은 재변경 가능
    "return_delivered_order_items":   False,  # [confirmed-policy] 주문당 1회
    "exchange_delivered_order_items": False,  # [confirmed-policy] 주문당 1회
    "transfer_to_human_agents":       False,
}

# 24. REVERSIBLE — 호출 후 취소/되돌리기 가능 여부
REVERSIBLE: Dict[str, bool] = {
    "find_user_id_by_email":          True,   # read no-op
    "find_user_id_by_name_zip":       True,
    "get_user_details":               True,
    "get_order_details":              True,
    "get_product_details":            True,
    "get_item_details":               True,
    "list_all_product_types":         True,
    "calculate":                      True,
    "think":                          True,
    "cancel_pending_order":           False,  # [confirmed-policy] 취소 불가역
    "modify_pending_order_address":   True,   # [inferred] lock 전 재변경 가능
    "modify_pending_order_items":     False,  # [confirmed-policy] order lock
    "modify_pending_order_payment":   False,  # [inferred] 환불/재청구 발생
    "modify_user_address":            True,
    "return_delivered_order_items":   False,  # [confirmed-policy]
    "exchange_delivered_order_items": False,  # [confirmed-policy]
    "transfer_to_human_agents":       False,
}

# 25. MANDATORY_IN_FLOW — 대부분의 플로우에 반드시 등장 (process frequency)
MANDATORY_IN_FLOW: Dict[str, bool] = {
    "find_user_id_by_name_zip":       True,   # [confirmed-policy] 인증 필수 (또는 email)
    "find_user_id_by_email":          True,   # [confirmed-policy] 인증 필수 (또는 name+zip)
    "get_order_details":              True,   # [confirmed] 168회, write 전 필수
    "get_user_details":               False,  # [inferred]
    "get_product_details":            False,  # [inferred] 변경/교환 시에만
    "get_item_details":               False,
    "list_all_product_types":         False,
    "calculate":                      False,
    "think":                          False,
    "cancel_pending_order":           False,  # 취소 케이스에만
    "modify_pending_order_address":   False,
    "modify_pending_order_items":     False,
    "modify_pending_order_payment":   False,
    "modify_user_address":            False,
    "return_delivered_order_items":   False,
    "exchange_delivered_order_items": False,
    "transfer_to_human_agents":       False,
}

# 26. OPTIONAL_IN_FLOW — MANDATORY의 반대
OPTIONAL_IN_FLOW: Dict[str, bool] = {k: not v for k, v in MANDATORY_IN_FLOW.items()}

# 27. LOOP_CAPABLE — 같은 세션에서 여러 번 의미 있게 호출 가능
LOOP_CAPABLE: Dict[str, bool] = {
    "find_user_id_by_email":          False,  # 인증 1회
    "find_user_id_by_name_zip":       False,
    "get_user_details":               True,   # 여러 조회 가능
    "get_order_details":              True,    # 여러 주문 조회 (168회 최다)
    "get_product_details":            True,    # 여러 상품 조회
    "get_item_details":               True,
    "list_all_product_types":         False,   # 보통 1회
    "calculate":                      True,
    "think":                          True,
    "cancel_pending_order":           False,   # 주문당 1회
    "modify_pending_order_address":   False,
    "modify_pending_order_items":     False,
    "modify_pending_order_payment":   False,
    "modify_user_address":            True,    # 프로필 재변경 가능
    "return_delivered_order_items":   False,
    "exchange_delivered_order_items": False,
    "transfer_to_human_agents":       False,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 E/G: GoT/ToT/Harness 관계 ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 28. BACKTRACK_TO — 막다른 도구에서 복귀 지점으로 (ToT backtrack)
BACKTRACK_TO: List[BacktrackTo] = [
    BacktrackTo("modify_pending_order_items",     "get_product_details"),  # [inferred] variant 무효→상품 재조회
    BacktrackTo("cancel_pending_order",           "get_order_details"),    # [inferred] not pending→상태 재확인
    BacktrackTo("return_delivered_order_items",   "get_order_details"),    # [inferred] not delivered→상태 재확인
    BacktrackTo("exchange_delivered_order_items", "get_product_details"),  # [inferred]
]

# 29. PRUNED_BY — A 선택지가 B 수행으로 무효화됨 (ToT prune)
PRUNED_BY: List[Tuple[str, str]] = [
    ("cancel_pending_order",         "modify_pending_order_items"),  # [confirmed-policy] items modify 후 cancel 차단
    ("modify_pending_order_address", "modify_pending_order_items"),  # [confirmed-policy] order lock 후 차단
    ("modify_pending_order_items",   "cancel_pending_order"),        # [inferred] 취소되면 변경 무의미
]

# 30. FAN_OUT — 한 도구가 여러 후속으로 분기 (GoT branch)
FAN_OUT: List[FanOut] = [
    FanOut("get_order_details",  ("cancel_pending_order", "modify_pending_order_items",
                                  "return_delivered_order_items", "exchange_delivered_order_items")),  # [confirmed]
    FanOut("get_user_details",   ("get_order_details", "modify_user_address")),                         # [inferred]
    FanOut("get_product_details",("modify_pending_order_items", "exchange_delivered_order_items")),     # [confirmed]
]

# 31. OBSERVATION_TRIGGERS — 관찰된 상태가 특정 도구 호출을 유발 (Harness reactive rule)
OBSERVATION_TRIGGERS: Dict[str, str] = {
    "user_not_authenticated":       "find_user_id_by_email",
    "order_id_unknown":             "get_user_details",
    "order_status_unknown":         "get_order_details",
    "variant_options_unknown":      "get_product_details",
    "item_availability_unknown":    "get_item_details",
    "request_out_of_scope":         "transfer_to_human_agents",
    "price_difference_to_compute":  "calculate",
}

# 32. GUARDRAIL — 조건 활성화 시 도구 호출 차단 (Harness prohibition)
GUARDRAIL: Dict[str, str] = {
    "transfer_to_human_agents":       "is_first_action_or_request_in_scope",  # [confirmed-policy] 첫 액션/범위내 금지
    "cancel_pending_order":           "order_not_pending_or_user_unconfirmed", # [confirmed-policy]
    "modify_pending_order_items":     "order_not_pending_or_already_modified",  # [confirmed-policy]
    "modify_pending_order_address":   "order_not_pending_or_locked",            # [confirmed-policy]
    "modify_pending_order_payment":   "same_method_or_insufficient_balance",    # [confirmed-policy]
    "return_delivered_order_items":   "order_not_delivered_or_already_returned",# [confirmed-policy]
    "exchange_delivered_order_items": "order_not_delivered_or_already_exchanged",# [confirmed-policy]
    "get_order_details":              "request_for_another_user",               # [confirmed-policy] cross-user 금지
}

# 33. SCORED_PREFERENCE — 둘 다 유효하나 맥락에서 preferred 우위 (GoT score/keep)
SCORED_PREFERENCE: List[ScoredPreference] = [
    ScoredPreference("find_user_id_by_email",        "find_user_id_by_name_zip",       "email_available"),       # [confirmed-policy] email 기본
    ScoredPreference("get_order_details",            "get_user_details",                "order_id_known"),        # [inferred]
    ScoredPreference("exchange_delivered_order_items","return_delivered_order_items",   "wants_replacement_item"),# [inferred]
]

# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 H: HTN 계층 관계 ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 34. DECOMPOSES_INTO — 추상 목표가 구체 도구 집합으로 분해 (HTN top-down)
DECOMPOSES_INTO: List[DecomposesInto] = [
    DecomposesInto("authenticate_user",
        ("find_user_id_by_email", "find_user_id_by_name_zip")),                                  # [confirmed-policy]
    DecomposesInto("resolve_order_cancellation",
        ("get_order_details", "cancel_pending_order")),                                          # [confirmed]
    DecomposesInto("resolve_order_modification",
        ("get_order_details", "get_product_details", "modify_pending_order_items")),             # [confirmed]
    DecomposesInto("resolve_return",
        ("get_order_details", "return_delivered_order_items")),                                  # [confirmed]
    DecomposesInto("resolve_exchange",
        ("get_order_details", "get_product_details", "exchange_delivered_order_items")),         # [confirmed]
    DecomposesInto("update_profile",
        ("get_user_details", "modify_user_address")),                                            # [inferred]
]

# 35. SUBTASK_OF — 도구가 속하는 추상 단계 (DECOMPOSES_INTO 역관계)
SUBTASK_OF: Dict[str, str] = {
    "find_user_id_by_email":          "authentication_phase",
    "find_user_id_by_name_zip":       "authentication_phase",
    "get_user_details":               "information_gathering_phase",
    "get_order_details":              "information_gathering_phase",
    "get_product_details":            "information_gathering_phase",
    "get_item_details":               "information_gathering_phase",
    "list_all_product_types":         "information_gathering_phase",
    "calculate":                      "support_phase",
    "think":                          "support_phase",
    "cancel_pending_order":           "order_cancellation_phase",
    "modify_pending_order_address":   "order_modification_phase",
    "modify_pending_order_items":     "order_modification_phase",
    "modify_pending_order_payment":   "order_modification_phase",
    "modify_user_address":            "profile_management_phase",
    "return_delivered_order_items":   "return_phase",
    "exchange_delivered_order_items": "exchange_phase",
    "transfer_to_human_agents":       "escalation_phase",
}

# 36. ACHIEVES_GOAL — 도구가 달성하는 고객 레벨 목표 (HTN top-level goal)
ACHIEVES_GOAL: Dict[str, str] = {
    "cancel_pending_order":           "cancel_unwanted_order",
    "modify_pending_order_items":     "correct_order_items",
    "modify_pending_order_address":   "correct_shipping_address",
    "modify_pending_order_payment":   "change_payment_method",
    "modify_user_address":            "update_profile_address",
    "return_delivered_order_items":   "return_unwanted_items",
    "exchange_delivered_order_items": "exchange_for_correct_variant",
    "get_order_details":              "retrieve_order_information",
    "get_product_details":            "retrieve_product_information",
    "get_user_details":              "retrieve_profile_information",
    "find_user_id_by_email":          "authenticate_customer",
    "find_user_id_by_name_zip":       "authenticate_customer",
    "transfer_to_human_agents":       "escalate_unresolved_issue",
}

# 37. REFINES — 추상 행동이 맥락에 따라 구체 도구로 구체화 (HTN method)
REFINES: List[Refines] = [
    Refines("authenticate",         "find_user_id_by_email",          "email_provided"),               # [confirmed-policy]
    Refines("authenticate",         "find_user_id_by_name_zip",       "name_and_zip_provided"),         # [confirmed-policy]
    Refines("resolve_order_issue",  "cancel_pending_order",           "pending_and_unwanted"),          # [inferred]
    Refines("resolve_order_issue",  "modify_pending_order_items",     "pending_and_wrong_items"),        # [inferred]
    Refines("resolve_order_issue",  "return_delivered_order_items",   "delivered_and_unwanted"),         # [inferred]
    Refines("resolve_order_issue",  "exchange_delivered_order_items", "delivered_and_wrong_variant"),    # [inferred]
    Refines("update_address",       "modify_pending_order_address",   "for_a_pending_order"),            # [inferred]
    Refines("update_address",       "modify_user_address",            "for_default_profile"),            # [inferred]
]


# ═══════════════════════════════════════════════════════════════════════════════
# 관계 기하 분류 (도메인-독립; telecom과 동일) → 개입 방법 예측
# ═══════════════════════════════════════════════════════════════════════════════
RELATION_GEOMETRY: Dict[str, str] = {
    # Directional → A6 우위 예측
    "precedes": "directional", "directly_follows": "directional", "causal_link": "directional",
    "requires": "directional", "enables": "directional", "parameter_feeds": "directional",
    "and_join": "directional", "validates": "directional", "retry_after_fail": "directional",
    "error_fallback": "directional", "compensates": "directional", "tool_subsumes": "directional",
    "state_transition": "directional", "backtrack_to": "directional", "pruned_by": "directional",
    "fan_out": "directional", "decomposes_into": "directional",
    # Symmetric/Conditional → T1 우위 예측
    "mutex": "symmetric", "exclusive_choice": "conditional", "parallel_safe": "symmetric",
    "conditional_on": "conditional", "scored_preference": "conditional", "refines": "conditional",
    # Categorical → T1 우위 예측
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
# 실측 빈도 (tasks.json 112 tasks, ordered-pair counts) — [confirmed] 가중치용
# ═══════════════════════════════════════════════════════════════════════════════
PRECEDES_FREQ: Dict[Tuple[str, str], int] = {
    ("get_user_details", "get_order_details"): 150,
    ("find_user_id_by_name_zip", "get_order_details"): 148,
    ("get_order_details", "get_product_details"): 96,
    ("get_order_details", "return_delivered_order_items"): 96,
    ("get_order_details", "cancel_pending_order"): 58,
    ("find_user_id_by_name_zip", "get_product_details"): 53,
    ("find_user_id_by_name_zip", "get_user_details"): 52,
    ("get_order_details", "modify_pending_order_items"): 45,
    ("find_user_id_by_email", "get_order_details"): 37,
    ("get_order_details", "calculate"): 33,
    ("get_order_details", "exchange_delivered_order_items"): 30,
    ("get_product_details", "modify_pending_order_items"): 28,
    ("get_product_details", "exchange_delivered_order_items"): 25,
    ("get_order_details", "modify_pending_order_address"): 19,
    ("modify_pending_order_address", "modify_pending_order_items"): 18,
}

TOOL_FREQ: Dict[str, int] = {
    "get_order_details": 168, "find_user_id_by_name_zip": 61, "get_user_details": 57,
    "get_product_details": 54, "return_delivered_order_items": 41, "modify_pending_order_items": 39,
    "exchange_delivered_order_items": 35, "cancel_pending_order": 25, "modify_pending_order_address": 24,
    "find_user_id_by_email": 14, "calculate": 13, "modify_user_address": 11,
    "transfer_to_human_agents": 4, "get_item_details": 3, "modify_pending_order_payment": 1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 도구 설명 (docstring 요약)
# ═══════════════════════════════════════════════════════════════════════════════
TOOL_DESC: Dict[str, str] = {
    "find_user_id_by_email":          "Find user id by email (default authentication path).",
    "find_user_id_by_name_zip":       "Find user id by first name, last name, and zip code (fallback auth).",
    "get_user_details":               "Get details of a user including their orders.",
    "get_order_details":              "Get the status and details of an order.",
    "get_product_details":            "Get the inventory details of a product (its variants).",
    "get_item_details":               "Get the inventory details of a specific item (variant).",
    "list_all_product_types":         "List the name and product id of all product types.",
    "calculate":                      "Evaluate an arithmetic expression (e.g. price difference).",
    "think":                          "Internal reasoning scratchpad; no state change.",
    "cancel_pending_order":           "Cancel a pending order; refund to original method.",
    "modify_pending_order_address":   "Modify the shipping address of a pending order (needs confirmation).",
    "modify_pending_order_items":     "Modify items of a pending order to same-product variants (once only; locks order).",
    "modify_pending_order_payment":   "Change the payment method of a pending order (single new method).",
    "modify_user_address":            "Modify the user's default profile address.",
    "return_delivered_order_items":   "Request return of items in a delivered order (once per order).",
    "exchange_delivered_order_items": "Exchange items in a delivered order for same-product variants (once per order).",
    "transfer_to_human_agents":       "Escalate to a human agent when the request is out of scope (not as first action).",
}

# 도구 → ToolType (READ/WRITE/GENERIC/THINK)
TOOL_TYPE: Dict[str, str] = {
    "find_user_id_by_email": "READ", "find_user_id_by_name_zip": "READ",
    "get_user_details": "READ", "get_order_details": "READ", "get_product_details": "READ",
    "get_item_details": "READ", "list_all_product_types": "READ",
    "calculate": "GENERIC", "transfer_to_human_agents": "GENERIC", "think": "THINK",
    "cancel_pending_order": "WRITE", "modify_pending_order_address": "WRITE",
    "modify_pending_order_items": "WRITE", "modify_pending_order_payment": "WRITE",
    "modify_user_address": "WRITE", "return_delivered_order_items": "WRITE",
    "exchange_delivered_order_items": "WRITE",
}
