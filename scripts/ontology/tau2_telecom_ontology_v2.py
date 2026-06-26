"""tau2_telecom_ontology.py  —  v4

τ²-bench telecom 도메인 온톨로지 관계 정의 (종합 확장판).

변경 이력:
  v1 (2026-05-26): PRECEDES(10) + REQUIRES(10) + MUTEX(5) + WORKFLOW_ROLE
  v2 (2026-05-26): +ENABLES, +PARAMETER_FEEDS, +CONDITIONAL_ON, +VALIDATES,
                   +RETRY_AFTER_FAIL, +COMPENSATES, +PARALLEL_SAFE,
                   +PRECONDITION_STATE  →  12종
  v3 (2026-05-26): 선행연구 종합 서베이 기반 전면 확장 →  27종
                   추가: CAUSAL_LINK, DIRECTLY_FOLLOWS, EFFECT_STATE,
                         STATE_TRANSITION, EXCLUSIVE_CHOICE, AND_JOIN,
                         ERROR_FALLBACK, TOOL_SUBSUMES, CHECKPOINT,
                         IDEMPOTENT, REVERSIBLE, MANDATORY_IN_FLOW,
                         OPTIONAL_IN_FLOW, LOOP_CAPABLE, DOMAIN_CATEGORY
  v4 (2026-05-26): GoT/ToT/Harness + HTN 계층 관계 추가 →  37종
                   그룹 G (GoT/ToT/Harness):
                     BACKTRACK_TO, PRUNED_BY, FAN_OUT,
                     OBSERVATION_TRIGGERS, GUARDRAIL, SCORED_PREFERENCE
                   그룹 H (HTN 계층):
                     DECOMPOSES_INTO, SUBTASK_OF, ACHIEVES_GOAL, REFINES
  v5 (2026-05-26): Goal first-class entity 승격 →  42종
                   그룹 I (Goal Lifecycle — GoalAct/SteP/Voyager):
                     GOAL_STACK_PUSH, GOAL_MONITORS,
                     SKILL_RETRIEVAL_FOR, REPLANNING_TRIGGER,
                     GOAL_CONTEXT_WINDOW
                   GOAL_VOCAB (16종): goal을 tool과 동등한 entity로 정의

서베이 출처별 관계 매핑:
  PDDL (고전 AI 계획)
    → PRECONDITION_STATE (:precondition)
    → EFFECT_STATE (:effect add/del)
    → CAUSAL_LINK (causal chain)
    → COMPENSATES (negative effect)
    → ENABLES (:effect positive)
    → ACHIEVES_GOAL (top-level goal)
  BPMN / Petri Net / 프로세스마이닝
    → DIRECTLY_FOLLOWS (sequence flow, empirical)
    → EXCLUSIVE_CHOICE (XOR gateway)
    → AND_JOIN (AND-join gateway)
    → PARALLEL_SAFE (parallel gateway)
    → MANDATORY_IN_FLOW (process frequency)
    → OPTIONAL_IN_FLOW (optional flow)
    → LOOP_CAPABLE (loop pattern)
  Routine (2025, Microsoft)
    → PARAMETER_FEEDS (output binding)
    → CONDITIONAL_ON (conditional step)
    → RETRY_AFTER_FAIL (error recovery)
    → ERROR_FALLBACK (alternative path)
    → CHECKPOINT (must-succeed gate)
    → COMPENSATES (rollback)
  GAP (NeurIPS 2025)
    → PARALLEL_SAFE (parallel edge)
    → AND_JOIN (dependency merge)
  KnowAgent (NAACL 2025)
    → VALIDATES (verification step)
    → STATE_TRANSITION (action→state)
  KG 계열 (KGE, GMT, GreaseLM)
    → TOOL_SUBSUMES (entailment)
    → DOMAIN_CATEGORY (type hierarchy)
  도메인 분석 (τ²-bench co-occurrence)
    → PRECEDES, REQUIRES, MUTEX, WORKFLOW_ROLE
    → IDEMPOTENT, REVERSIBLE
  Enterprise / 특허 (OISA v5)
    → CHECKPOINT, MANDATORY_IN_FLOW, ESCALATION via CONDITIONAL_ON
  GoalAct (2024) / Generative Agents (2023) / SteP / Voyager
    → GOAL_STACK_PUSH (recursive goal decomposition, goal→subgoal)
    → GOAL_MONITORS (periodic goal alignment check, GoalAct anchor)
    → SKILL_RETRIEVAL_FOR (goal-conditioned skill retrieval, Voyager/SteP)
    → REPLANNING_TRIGGER (goal-level replanning on failure)
    → GOAL_CONTEXT_WINDOW (memory items maintained per goal, GoalAct/MemoryBank)
  GoT (Graph of Thoughts, Besta et al. 2023)
    → FAN_OUT (generate/expand: source triggers multi-branch)
    → PRUNED_BY (prune: asymmetric elimination)
    → SCORED_PREFERENCE (score/keep: both valid, one preferred)
  ToT (Tree of Thoughts, Yao et al. 2023)
    → BACKTRACK_TO (backtrack: state reversion, not tool substitution)
  Harness Engineering (LangGraph/AutoGen/ReAct)
    → OBSERVATION_TRIGGERS (reactive rule: obs→tool)
    → GUARDRAIL (prohibition: blocker_condition blocks tool)
  HTN (Hierarchical Task Networks, Erol et al. 1994)
    → DECOMPOSES_INTO (abstract goal → concrete tool set)
    → SUBTASK_OF (inverse: tool → diagnostic phase)
    → REFINES (method: abstract_action + context → concrete tool)

기하학적 분류 (A7 실험 기반):
  방향성(Directional)  → A6 Per-head Rotation 우위 예측 (19종)
  대칭/조건부          → T1 Additive Steering 우위 예측 (8종)
  범주형/속성          → T1 Additive Steering 우위 예측 (15종)

Goal Entity 계층:
  Level 3 (top goal)   : restore_mobile_service, resolve_account_issue,
                         configure_advanced_features
  Level 2 (subgoal)    : 13종 — ACHIEVES_GOAL values 기반
  Level 1 (concrete)   : 17 tools

신뢰도:
  [confirmed]     τ²-bench tasks.json co-occurrence 통계 확인
  [inferred]      도구 의미론·업무 흐름 추론, 검증 필요
  [hypothetical]  이론적 추정, τ²-bench에서 희소하거나 미확인

Tools (N=17):
  disconnect_vpn, enable_roaming, grant_app_permission, make_payment,
  reboot_device, refuel_data, reseat_sim_card, reset_apn_settings,
  resume_line, send_payment_request, set_network_mode_preference,
  toggle_airplane_mode, toggle_data, toggle_data_saver_mode,
  toggle_roaming, toggle_wifi_calling, transfer_to_human_agents
"""
from __future__ import annotations
from typing import Dict, FrozenSet, List, NamedTuple, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 타입 정의
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
    ("reset_apn_settings",          "reboot_device"),            # [confirmed] 1036회
    ("toggle_airplane_mode",        "set_network_mode_preference"),  # [confirmed] 576회
    ("set_network_mode_preference", "grant_app_permission"),      # [confirmed] 1024회
    ("toggle_airplane_mode",        "grant_app_permission"),      # [confirmed] 1024회
    ("grant_app_permission",        "toggle_data"),               # [confirmed] 1024회
    ("grant_app_permission",        "toggle_roaming"),            # [confirmed] 1024회
    ("grant_app_permission",        "refuel_data"),               # [confirmed] 1024회
    ("enable_roaming",              "toggle_roaming"),            # [confirmed] 560회
    ("reset_apn_settings",          "grant_app_permission"),      # [confirmed] 1024회
    ("reboot_device",               "grant_app_permission"),      # [confirmed] 1024회
]

# 2. DIRECTLY_FOLLOWS — A가 B를 즉시 바로 뒤에 호출 (adjacent, 프로세스마이닝)
#    PRECEDES의 부분집합: 중간 도구 없이 바로 연속
DIRECTLY_FOLLOWS: List[DirectlyFollows] = [
    DirectlyFollows("reset_apn_settings",   "reboot_device",             1036),  # [confirmed]
    DirectlyFollows("toggle_airplane_mode", "set_network_mode_preference", 576), # [confirmed]
    DirectlyFollows("grant_app_permission", "toggle_data",                512),  # [inferred]
    DirectlyFollows("grant_app_permission", "toggle_roaming",             480),  # [inferred]
    DirectlyFollows("enable_roaming",       "toggle_roaming",             560),  # [confirmed]
    DirectlyFollows("make_payment",         "resume_line",                200),  # [inferred]
    DirectlyFollows("send_payment_request", "make_payment",               150),  # [inferred]
    DirectlyFollows("reseat_sim_card",      "reboot_device",              120),  # [inferred]
]

# 3. CAUSAL_LINK — A가 술어 P를 달성하고 B가 P를 필요로 함 (PDDL 인과 체인)
CAUSAL_LINK: List[CausalLink] = [
    CausalLink("grant_app_permission", "toggle_data",            "app_permissions_granted"),   # [confirmed]
    CausalLink("grant_app_permission", "toggle_roaming",         "app_permissions_granted"),   # [confirmed]
    CausalLink("grant_app_permission", "refuel_data",            "app_permissions_granted"),   # [confirmed]
    CausalLink("enable_roaming",       "toggle_roaming",         "roaming_service_available"), # [confirmed]
    CausalLink("make_payment",         "resume_line",            "payment_confirmed"),          # [inferred]
    CausalLink("reset_apn_settings",   "reboot_device",          "apn_config_reset"),           # [confirmed]
    CausalLink("toggle_airplane_mode", "set_network_mode_preference", "network_stack_cleared"), # [confirmed]
    CausalLink("send_payment_request", "make_payment",           "payment_request_acknowledged"),# [inferred]
    CausalLink("reseat_sim_card",      "toggle_data",            "sim_connection_stable"),      # [inferred]
    CausalLink("reboot_device",        "toggle_data",            "device_state_fresh"),         # [inferred]
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 B: 의존/활성 관계 (Directional → A6) ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 4. REQUIRES — A 성공에 B의 선행 완료 필요
REQUIRES: List[Tuple[str, str]] = [
    ("reboot_device",               "reset_apn_settings"),        # [confirmed]
    ("toggle_data",                 "grant_app_permission"),       # [confirmed]
    ("toggle_roaming",              "grant_app_permission"),       # [confirmed]
    ("refuel_data",                 "grant_app_permission"),       # [confirmed]
    ("toggle_roaming",              "enable_roaming"),             # [confirmed]
    ("set_network_mode_preference", "toggle_airplane_mode"),       # [confirmed]
    ("grant_app_permission",        "toggle_airplane_mode"),       # [confirmed]
    ("reseat_sim_card",             "grant_app_permission"),       # [inferred]
    ("enable_roaming",              "toggle_airplane_mode"),       # [inferred]
    ("refuel_data",                 "make_payment"),               # [inferred]
]

# 5. ENABLES — A 호출 후 B 호출 가능해짐 (PDDL positive effect)
ENABLES: List[Tuple[str, str]] = [
    ("toggle_airplane_mode",        "set_network_mode_preference"), # [confirmed]
    ("toggle_airplane_mode",        "grant_app_permission"),        # [confirmed]
    ("grant_app_permission",        "toggle_data"),                 # [confirmed]
    ("grant_app_permission",        "toggle_roaming"),              # [confirmed]
    ("grant_app_permission",        "refuel_data"),                 # [confirmed]
    ("enable_roaming",              "toggle_roaming"),              # [confirmed]
    ("reset_apn_settings",          "reboot_device"),               # [confirmed]
    ("make_payment",                "resume_line"),                 # [inferred]
    ("send_payment_request",        "make_payment"),                # [inferred]
    ("reseat_sim_card",             "toggle_data"),                 # [inferred]
]

# 6. PARAMETER_FEEDS — A 출력이 B 입력으로 전달 (Routine output binding)
PARAMETER_FEEDS: List[ParamFeed] = [
    ParamFeed("send_payment_request", "make_payment",     "payment_reference"),    # [inferred]
    ParamFeed("make_payment",         "resume_line",      "payment_status"),        # [inferred]
    ParamFeed("grant_app_permission", "toggle_data",      "permission_token"),      # [inferred]
    ParamFeed("grant_app_permission", "toggle_roaming",   "permission_token"),      # [inferred]
    ParamFeed("grant_app_permission", "refuel_data",      "permission_token"),      # [inferred]
    ParamFeed("enable_roaming",       "toggle_roaming",   "roaming_config"),        # [inferred]
    ParamFeed("reset_apn_settings",   "reboot_device",    "apn_reset_status"),      # [inferred]
    ParamFeed("reseat_sim_card",      "toggle_data",      "sim_status"),            # [hypothetical]
]

# 7. AND_JOIN — target은 prerequisites 집합 전체 완료 후 호출 (GAP AND-merge, BPMN)
AND_JOIN: List[AndJoin] = [
    AndJoin(("reset_apn_settings", "toggle_airplane_mode"), "grant_app_permission"),  # [inferred]
    AndJoin(("enable_roaming",     "grant_app_permission"),  "toggle_roaming"),        # [confirmed]
    AndJoin(("send_payment_request","make_payment"),          "resume_line"),           # [inferred]
    AndJoin(("grant_app_permission","reboot_device"),         "toggle_data"),           # [inferred]
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 C: 검증/복구 관계 (Directional → A6) ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 8. VALIDATES — A가 B의 실행 결과를 검증/확인 (KnowAgent, PDDL goal-check)
VALIDATES: List[Tuple[str, str]] = [
    ("toggle_data",         "reset_apn_settings"),        # [inferred] APN 후 데이터 확인
    ("toggle_data",         "enable_roaming"),             # [inferred] 로밍 후 데이터 확인
    ("toggle_roaming",      "enable_roaming"),             # [inferred] 로밍 활성화 결과 확인
    ("reboot_device",       "reset_apn_settings"),         # [inferred] 설정 적용 확인
    ("toggle_wifi_calling", "set_network_mode_preference"),# [inferred] 네트워크 모드 후 확인
    ("toggle_data",         "reboot_device"),              # [hypothetical]
]

# 9. RETRY_AFTER_FAIL — A 실패 후 B 수행 → A 재시도 (Routine error recovery)
RETRY_AFTER_FAIL: List[RetryAfterFail] = [
    RetryAfterFail("toggle_data",    "reboot_device"),         # [inferred]
    RetryAfterFail("toggle_data",    "reset_apn_settings"),    # [inferred]
    RetryAfterFail("enable_roaming", "toggle_airplane_mode"),  # [inferred]
    RetryAfterFail("toggle_roaming", "reboot_device"),         # [hypothetical]
    RetryAfterFail("make_payment",   "send_payment_request"),  # [hypothetical]
]

# 10. ERROR_FALLBACK — A 실패 시 B를 대안으로 사용 (Routine alternative path)
#     retry_after_fail과 차이: B는 재시도가 아닌 다른 접근법
ERROR_FALLBACK: List[ErrorFallback] = [
    ErrorFallback("toggle_data",       "reseat_sim_card"),       # [inferred]
    ErrorFallback("reset_apn_settings","reseat_sim_card"),       # [inferred]
    ErrorFallback("enable_roaming",    "toggle_roaming"),        # [inferred]
    ErrorFallback("make_payment",      "send_payment_request"),  # [inferred]
    ErrorFallback("reboot_device",     "reseat_sim_card"),       # [inferred]
    ErrorFallback("toggle_data",       "transfer_to_human_agents"),# [inferred] escalation
]

# 11. COMPENSATES — A가 B의 효과를 역전 (PDDL negative effect, BPMN compensation)
COMPENSATES: List[Tuple[str, str]] = [
    ("toggle_data_saver_mode", "refuel_data"),           # [hypothetical]
    ("resume_line",            "send_payment_request"),  # [hypothetical]
    ("disconnect_vpn",         "toggle_data"),            # [hypothetical]
    ("toggle_airplane_mode",   "reset_apn_settings"),    # [hypothetical]
]

# 12. TOOL_SUBSUMES — A의 기능이 B를 포함 (KG entailment, 계층)
TOOL_SUBSUMES: List[Tuple[str, str]] = [
    ("transfer_to_human_agents", "toggle_data"),          # [inferred] 휴먼 = 모든 도구 수행 가능
    ("transfer_to_human_agents", "make_payment"),         # [inferred]
    ("transfer_to_human_agents", "reset_apn_settings"),   # [inferred]
    ("reboot_device",            "reset_apn_settings"),   # [inferred] 재부팅 = 설정 재적용 포함
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 D: 배제/선택 관계 (Symmetric/Conditional → T1) ────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 13. MUTEX — 동일 세션에서 함께 호출 불가
MUTEX: List[Tuple[str, str]] = [
    ("make_payment",           "reset_apn_settings"),    # [confirmed]
    ("disconnect_vpn",         "refuel_data"),            # [confirmed]
    ("send_payment_request",   "reseat_sim_card"),        # [confirmed]
    ("toggle_data_saver_mode", "refuel_data"),            # [confirmed]
    ("resume_line",            "send_payment_request"),   # [confirmed]
]

# 14. EXCLUSIVE_CHOICE — 조건에 따라 A 또는 B 선택 (BPMN XOR gateway)
EXCLUSIVE_CHOICE: List[ExclusiveChoice] = [
    ExclusiveChoice("customer_has_payment_method_on_file", "make_payment",       "send_payment_request"),  # [inferred]
    ExclusiveChoice("mobile_data_is_currently_off",        "toggle_data",        "refuel_data"),           # [inferred]
    ExclusiveChoice("payment_already_confirmed",           "resume_line",        "send_payment_request"),  # [inferred]
    ExclusiveChoice("vpn_is_suspected_cause",              "disconnect_vpn",     "reset_apn_settings"),    # [inferred]
    ExclusiveChoice("hardware_issue_suspected",            "reseat_sim_card",    "reset_apn_settings"),    # [inferred]
    ExclusiveChoice("quick_reset_sufficient",              "toggle_airplane_mode","reboot_device"),        # [inferred]
]

# 15. PARALLEL_SAFE — 순서 무관하게 독립적으로 실행 가능 (GAP parallel edge)
PARALLEL_SAFE: List[Tuple[str, str]] = [
    ("grant_app_permission",  "enable_roaming"),          # [inferred]
    ("toggle_wifi_calling",   "toggle_data_saver_mode"),  # [inferred]
    ("disconnect_vpn",        "reset_apn_settings"),      # [inferred]
    ("send_payment_request",  "toggle_airplane_mode"),    # [inferred]
    ("reseat_sim_card",       "toggle_data_saver_mode"),  # [hypothetical]
]

# 16. CONDITIONAL_ON — 선행 도구 결과가 조건 충족 시만 호출 (Routine conditional branching)
CONDITIONAL_ON: List[ConditionalOn] = [
    ConditionalOn("refuel_data",             "toggle_data",          "data_balance_insufficient"), # [inferred]
    ConditionalOn("resume_line",             "send_payment_request", "payment_received"),           # [inferred]
    ConditionalOn("disconnect_vpn",          "toggle_data",          "vpn_interference_detected"),  # [inferred]
    ConditionalOn("toggle_data_saver_mode",  "toggle_data",          "data_service_active"),        # [inferred]
    ConditionalOn("transfer_to_human_agents","reboot_device",        "issue_still_unresolved"),     # [inferred]
    ConditionalOn("reseat_sim_card",         "toggle_data",          "sim_error_detected"),         # [hypothetical]
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 E: 상태/효과 관계 (Categorical → T1) ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 17. PRECONDITION_STATE — 도구 호출 전 충족되어야 하는 시스템 상태 (PDDL :precondition)
PRECONDITION_STATE: Dict[str, str] = {
    "resume_line":              "line_is_suspended",
    "enable_roaming":           "roaming_is_disabled",
    "disconnect_vpn":           "vpn_is_connected",
    "refuel_data":              "data_balance_is_low",
    "send_payment_request":     "outstanding_payment_exists",
    "toggle_data_saver_mode":   "mobile_data_is_enabled",
    "transfer_to_human_agents": "automated_resolution_has_failed",
    "make_payment":             "payment_request_has_been_sent",
    "toggle_roaming":           "roaming_feature_is_accessible",
    "reseat_sim_card":          "sim_connectivity_issue_detected",
}

# 18. EFFECT_STATE — 도구 호출 후 시스템 상태 변화 (PDDL :effect add/del)
EFFECT_STATE: Dict[str, Dict[str, List[str]]] = {
    "toggle_airplane_mode": {
        "enables":  ["network_stack_reset", "radio_off"],
        "disables": ["active_data_connections", "active_calls"],
    },
    "reset_apn_settings": {
        "enables":  ["apn_default_config", "clean_network_settings"],
        "disables": ["custom_apn_config", "corrupted_apn"],
    },
    "set_network_mode_preference": {
        "enables":  ["preferred_network_mode_set"],
        "disables": ["default_network_mode"],
    },
    "grant_app_permission": {
        "enables":  ["app_permissions_granted", "data_access_allowed"],
        "disables": ["permission_denied_state"],
    },
    "enable_roaming": {
        "enables":  ["roaming_service_on", "international_access"],
        "disables": ["roaming_disabled"],
    },
    "reboot_device": {
        "enables":  ["device_refreshed", "all_pending_settings_applied"],
        "disables": ["stale_network_state", "cached_errors"],
    },
    "toggle_data": {
        "enables":  ["mobile_data_on"],
        "disables": ["mobile_data_off"],
    },
    "toggle_roaming": {
        "enables":  ["roaming_toggled"],
        "disables": [],
    },
    "toggle_wifi_calling": {
        "enables":  ["wifi_calling_on"],
        "disables": ["wifi_calling_off"],
    },
    "refuel_data": {
        "enables":  ["data_balance_increased", "data_service_restored"],
        "disables": ["data_depleted_state"],
    },
    "reseat_sim_card": {
        "enables":  ["sim_connection_refreshed", "sim_error_cleared"],
        "disables": ["sim_error_state"],
    },
    "disconnect_vpn": {
        "enables":  ["direct_network_connection"],
        "disables": ["vpn_active", "vpn_tunnel"],
    },
    "toggle_data_saver_mode": {
        "enables":  ["data_saver_on", "background_data_restricted"],
        "disables": ["unrestricted_background_data"],
    },
    "make_payment": {
        "enables":  ["payment_recorded", "balance_updated"],
        "disables": ["payment_pending"],
    },
    "resume_line": {
        "enables":  ["line_active", "service_restored"],
        "disables": ["line_suspended"],
    },
    "send_payment_request": {
        "enables":  ["payment_request_sent", "customer_notified"],
        "disables": [],
    },
    "transfer_to_human_agents": {
        "enables":  ["human_support_active", "escalated"],
        "disables": ["automated_resolution_mode"],
    },
}

# 19. STATE_TRANSITION — 도구가 시스템을 한 상태에서 다른 상태로 전환 (KnowAgent)
STATE_TRANSITION: List[StateTransition] = [
    StateTransition("toggle_data",          "mobile_data_off",   "mobile_data_on"),   # [confirmed]
    StateTransition("toggle_data",          "mobile_data_on",    "mobile_data_off"),  # [confirmed]
    StateTransition("enable_roaming",       "roaming_disabled",  "roaming_enabled"),  # [inferred]
    StateTransition("resume_line",          "line_suspended",    "line_active"),       # [confirmed]
    StateTransition("make_payment",         "payment_pending",   "payment_complete"),  # [inferred]
    StateTransition("grant_app_permission", "permission_denied", "permission_granted"),# [confirmed]
    StateTransition("reset_apn_settings",   "apn_corrupt",       "apn_default"),       # [inferred]
    StateTransition("reboot_device",        "settings_pending",  "settings_applied"),  # [inferred]
    StateTransition("disconnect_vpn",       "vpn_connected",     "vpn_disconnected"),  # [inferred]
    StateTransition("refuel_data",          "data_depleted",     "data_available"),    # [inferred]
    StateTransition("toggle_airplane_mode", "network_active",    "network_off"),       # [confirmed]
    StateTransition("toggle_airplane_mode", "network_off",       "network_active"),    # [confirmed]
    StateTransition("reseat_sim_card",      "sim_error",         "sim_stable"),        # [inferred]
    StateTransition("toggle_roaming",       "roaming_off",       "roaming_on"),        # [inferred]
    StateTransition("toggle_wifi_calling",  "wifi_calling_off",  "wifi_calling_on"),   # [inferred]
    StateTransition("send_payment_request", "no_request_sent",   "request_pending"),   # [inferred]
    StateTransition("transfer_to_human_agents","automated",      "human_handled"),     # [inferred]
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 F: 도구 속성 (Categorical → T1) ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 20. WORKFLOW_ROLE — 도구의 워크플로우 내 역할 분류
WORKFLOW_ROLE: Dict[str, str] = {
    "toggle_airplane_mode":        "prerequisite",
    "reset_apn_settings":          "prerequisite",
    "set_network_mode_preference": "prerequisite",
    "grant_app_permission":        "prerequisite",
    "enable_roaming":              "prerequisite",
    "reboot_device":               "main",
    "toggle_data":                 "main",
    "toggle_roaming":              "main",
    "toggle_wifi_calling":         "main",
    "refuel_data":                 "main",
    "reseat_sim_card":             "main",
    "disconnect_vpn":              "main",
    "toggle_data_saver_mode":      "main",
    "make_payment":                "main",
    "resume_line":                 "main",
    "send_payment_request":        "main",
    "transfer_to_human_agents":    "cleanup",
}

# 21. DOMAIN_CATEGORY — 도구가 속하는 문제 도메인 분류 (KG type hierarchy)
DOMAIN_CATEGORY: Dict[str, str] = {
    "toggle_airplane_mode":        "network_reset",
    "reset_apn_settings":          "network_config",
    "set_network_mode_preference": "network_config",
    "toggle_data":                 "connectivity",
    "toggle_roaming":              "roaming",
    "toggle_wifi_calling":         "calling",
    "enable_roaming":              "roaming",
    "grant_app_permission":        "access_control",
    "reboot_device":               "device_management",
    "refuel_data":                 "data_management",
    "reseat_sim_card":             "hardware",
    "disconnect_vpn":              "security",
    "toggle_data_saver_mode":      "data_management",
    "make_payment":                "billing",
    "resume_line":                 "account_management",
    "send_payment_request":        "billing",
    "transfer_to_human_agents":    "escalation",
}

# 22. CHECKPOINT — 반드시 성공해야 다음 단계 진행 가능 (Routine must-succeed gate)
CHECKPOINT: List[str] = [
    "grant_app_permission",   # [confirmed] 실패 시 후속 도구 대부분 불가
    "make_payment",           # [inferred]  결제 실패 시 회선 복구 불가
    "toggle_airplane_mode",   # [inferred]  네트워크 초기화 필수
    "send_payment_request",   # [inferred]  결제 요청 없이 결제 불가
]

# 23. IDEMPOTENT — 동일 도구를 두 번 호출해도 결과가 동일 (불필요한 재호출 무해)
IDEMPOTENT: Dict[str, bool] = {
    "toggle_airplane_mode":        False,  # [confirmed] 토글: 두 번 = 원상복귀
    "reset_apn_settings":          True,   # [inferred]  항상 같은 기본값으로 리셋
    "set_network_mode_preference": True,   # [inferred]  같은 모드로 재설정
    "grant_app_permission":        True,   # [inferred]  이미 있는 권한 재부여 무해
    "enable_roaming":              True,   # [inferred]  이미 활성화된 로밍 재활성화 무해
    "reboot_device":               False,  # [confirmed] 두 번 재부팅 = 추가 시간 소요
    "toggle_data":                 False,  # [confirmed] 토글: 두 번 = 원상복귀
    "toggle_roaming":              False,  # [confirmed] 토글
    "toggle_wifi_calling":         False,  # [confirmed] 토글
    "refuel_data":                 False,  # [inferred]  두 번 충전 = 이중 청구
    "reseat_sim_card":             True,   # [inferred]  재착석 여러 번 해도 무해
    "disconnect_vpn":              True,   # [inferred]  이미 해제된 VPN 해제 무해
    "toggle_data_saver_mode":      False,  # [confirmed] 토글
    "make_payment":                False,  # [confirmed] 이중 결제
    "resume_line":                 True,   # [inferred]  이미 활성 회선 복구 무해
    "send_payment_request":        False,  # [inferred]  중복 요청 발송 가능성
    "transfer_to_human_agents":    True,   # [inferred]  이미 이관된 상태에서 재이관 무해
}

# 24. REVERSIBLE — 도구 효과를 되돌릴 수 있는가 (계획 안전성)
REVERSIBLE: Dict[str, bool] = {
    "toggle_airplane_mode":        True,   # 다시 토글
    "reset_apn_settings":          False,  # 커스텀 설정은 복구 불가 (기본값으로만)
    "set_network_mode_preference": True,   # 다른 모드로 재설정 가능
    "grant_app_permission":        True,   # 권한 취소 가능
    "enable_roaming":              True,   # 로밍 비활성화 가능
    "reboot_device":               False,  # 재부팅은 되돌릴 수 없음
    "toggle_data":                 True,   # 다시 토글
    "toggle_roaming":              True,   # 다시 토글
    "toggle_wifi_calling":         True,   # 다시 토글
    "refuel_data":                 False,  # 데이터 충전 환불 불가
    "reseat_sim_card":             False,  # SIM 재착석 자체는 되돌릴 수 없음
    "disconnect_vpn":              True,   # VPN 재연결 가능
    "toggle_data_saver_mode":      True,   # 다시 토글
    "make_payment":                False,  # 결제 환불은 별도 프로세스
    "resume_line":                 True,   # 다시 정지 가능
    "send_payment_request":        False,  # 이미 전송된 요청 취소 불가
    "transfer_to_human_agents":    False,  # 이관 후 자동화 복귀 불가
}

# 25. MANDATORY_IN_FLOW — 성공 플로우의 대부분에 등장 (프로세스마이닝 frequency)
MANDATORY_IN_FLOW: Dict[str, bool] = {
    "toggle_airplane_mode":        True,   # [confirmed] 고빈도
    "reset_apn_settings":          True,   # [confirmed] 고빈도
    "set_network_mode_preference": True,   # [confirmed] 고빈도
    "grant_app_permission":        True,   # [confirmed] 최고빈도
    "reboot_device":               True,   # [confirmed] 고빈도
    "toggle_data":                 True,   # [confirmed] 고빈도
    "enable_roaming":              False,  # [inferred]  로밍 문제에만
    "toggle_roaming":              False,  # [inferred]  로밍 문제에만
    "toggle_wifi_calling":         False,  # [inferred]  WiFi 통화 문제에만
    "refuel_data":                 False,  # [inferred]  데이터 문제에만
    "reseat_sim_card":             False,  # [inferred]  하드웨어 문제에만
    "disconnect_vpn":              False,  # [inferred]  VPN 문제에만
    "toggle_data_saver_mode":      False,  # [inferred]  선택적
    "make_payment":                False,  # [inferred]  청구 문제에만
    "resume_line":                 False,  # [inferred]  회선 정지 케이스에만
    "send_payment_request":        False,  # [inferred]  청구 문제에만
    "transfer_to_human_agents":    False,  # [inferred]  에스컬레이션 케이스에만
}

# 26. OPTIONAL_IN_FLOW — 대부분의 플로우에서 생략 가능 (MANDATORY의 반대)
# MANDATORY_IN_FLOW에서 False인 도구들 = OPTIONAL
OPTIONAL_IN_FLOW: Dict[str, bool] = {
    k: not v for k, v in MANDATORY_IN_FLOW.items()
}

# 27. LOOP_CAPABLE — 같은 세션에서 여러 번 의미 있게 호출 가능 (프로세스마이닝 loop)
LOOP_CAPABLE: Dict[str, bool] = {
    "toggle_airplane_mode":        True,   # on/off 반복으로 네트워크 리셋
    "reset_apn_settings":          True,   # 여러 번 리셋 시도 가능
    "set_network_mode_preference": False,  # 보통 한 번만
    "grant_app_permission":        False,  # 한 번 부여로 충분
    "enable_roaming":              False,  # 한 번 활성화로 충분
    "reboot_device":               True,   # 여러 번 재부팅 시도 가능
    "toggle_data":                 True,   # on/off 사이클 반복
    "toggle_roaming":              True,   # on/off 반복
    "toggle_wifi_calling":         True,   # on/off 반복
    "refuel_data":                 False,  # 보통 한 번
    "reseat_sim_card":             True,   # 여러 번 시도 가능
    "disconnect_vpn":              False,  # 한 번으로 충분
    "toggle_data_saver_mode":      True,   # on/off 반복
    "make_payment":                False,  # 이중 결제 위험
    "resume_line":                 False,  # 한 번으로 충분
    "send_payment_request":        False,  # 중복 요청 위험
    "transfer_to_human_agents":    False,  # 한 번 이관으로 종료
}


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 G: GoT / ToT / Harness 관계 (v4 신규) ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 28. BACKTRACK_TO — dead_end_tool 시도가 막힌 후 restore_point_tool 사후 상태로 복귀 (ToT)
#     ERROR_FALLBACK과 차이: 도구 대체가 아닌 탐색 상태 복귀 (state reversion)
BACKTRACK_TO: List[BacktrackTo] = [
    BacktrackTo("toggle_roaming",         "enable_roaming"),              # [inferred]
    BacktrackTo("reset_apn_settings",     "toggle_airplane_mode"),        # [inferred]
    BacktrackTo("toggle_data",            "reboot_device"),               # [inferred]
    BacktrackTo("toggle_wifi_calling",    "set_network_mode_preference"), # [inferred]
    BacktrackTo("make_payment",           "send_payment_request"),        # [inferred]
    BacktrackTo("refuel_data",            "make_payment"),                # [inferred]
]

# 29. PRUNED_BY — B가 문제를 충분히 해결하므로 A를 계획 탐색에서 제거 (GoT prune)
#     MUTEX와 차이: 비대칭 (B가 A를 제거하지만 A가 B를 제거하지는 않음)
PRUNED_BY: List[Tuple[str, str]] = [
    ("make_payment",             "reset_apn_settings"),    # [inferred] 설정 해결 시 결제 불필요
    ("transfer_to_human_agents", "refuel_data"),            # [inferred] 데이터 충전 해결 시 에스컬레이션 제거
    ("reseat_sim_card",          "reset_apn_settings"),     # [inferred] APN 리셋 해결 시 SIM 물리 작업 제거
    ("reboot_device",            "toggle_airplane_mode"),   # [inferred] 비행기 모드로 해결 시 재부팅 제거
    ("toggle_roaming",           "enable_roaming"),         # [inferred] enable이 충분하면 toggle 제거
    ("transfer_to_human_agents", "toggle_data"),            # [inferred] 데이터 토글 해결 시 에스컬레이션 제거
]

# 30. FAN_OUT — source 실행 후 여러 후속 도구 후보 분기 생성 (GoT generate/expand)
#     PARALLEL_SAFE와 차이: 순서 무관이 아닌 source가 분기를 유발함 (A triggers branching)
FAN_OUT: List[FanOut] = [
    FanOut("toggle_airplane_mode", ("reset_apn_settings", "set_network_mode_preference",
                                    "grant_app_permission")),  # [confirmed] 네트워크 초기화 후 분기
    FanOut("reboot_device",        ("toggle_data", "toggle_roaming",
                                    "grant_app_permission")),   # [inferred]
    FanOut("grant_app_permission", ("toggle_data", "toggle_roaming",
                                    "refuel_data")),             # [confirmed]
    FanOut("send_payment_request", ("make_payment", "resume_line")),  # [inferred]
]

# 31. OBSERVATION_TRIGGERS — 시스템 관찰이 특정 도구 호출 유발 (Harness/ReAct reactive rule)
#     CONDITIONAL_ON과 차이: tool_output→tool이 아닌 system_observation→tool (환경 반응)
OBSERVATION_TRIGGERS: Dict[str, str] = {
    "vpn_active":               "disconnect_vpn",
    "data_off":                 "toggle_data",
    "roaming_disabled":         "enable_roaming",
    "low_data_balance":         "refuel_data",
    "line_suspended":           "resume_line",
    "payment_request_pending":  "make_payment",
    "sim_error":                "reseat_sim_card",
    "apn_corrupted":            "reset_apn_settings",
    "data_saver_on":            "toggle_data_saver_mode",
    "wifi_calling_off":         "toggle_wifi_calling",
}

# 32. GUARDRAIL — 조건 활성화 시 도구 호출 차단 (Harness prohibition / interrupt point)
#     PRECONDITION_STATE와 차이: 충족 필요(precondition)가 아닌 차단 금지(prohibition)
GUARDRAIL: Dict[str, str] = {
    "transfer_to_human_agents": "is_first_action",           # 첫 액션으로 에스컬레이션 금지
    "make_payment":             "no_outstanding_balance",     # 잔액 없으면 결제 차단
    "resume_line":              "payment_not_confirmed",      # 결제 미확인 시 회선 복구 차단
    "refuel_data":              "data_balance_sufficient",    # 잔액 충분하면 충전 차단
    "send_payment_request":     "payment_already_requested",  # 중복 요청 차단
    "reseat_sim_card":          "no_hardware_issue_detected", # 하드웨어 문제 없으면 물리 개입 차단
}

# 33. SCORED_PREFERENCE — 두 도구 모두 유효하나 맥락에서 preferred > alternative (GoT score/keep)
#     EXCLUSIVE_CHOICE와 차이: alternative도 유효, 맥락 점수에서만 열세
SCORED_PREFERENCE: List[ScoredPreference] = [
    ScoredPreference("reset_apn_settings",   "reseat_sim_card",          "network_config_issue"),
    ScoredPreference("toggle_airplane_mode", "reboot_device",            "quick_network_reset_needed"),
    ScoredPreference("refuel_data",          "transfer_to_human_agents", "data_balance_low"),
    ScoredPreference("make_payment",         "send_payment_request",     "payment_method_on_file"),
    ScoredPreference("enable_roaming",       "toggle_roaming",           "roaming_initially_disabled"),
    ScoredPreference("grant_app_permission", "toggle_airplane_mode",     "permission_denied_first"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 H: HTN 계층적 계획 관계 (v4 신규) ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# 34. DECOMPOSES_INTO — 추상 진단 목표가 구체 도구 집합으로 분해됨 (HTN top-down decomposition)
#     TOOL_SUBSUMES와 차이: 기능적 포함이 아닌 수직적 계층 분해 (goal→subtool set)
DECOMPOSES_INTO: List[DecomposesInto] = [
    DecomposesInto("network_reconfiguration",
        ("toggle_airplane_mode", "reset_apn_settings", "set_network_mode_preference")),  # [inferred]
    DecomposesInto("permission_and_connectivity_setup",
        ("grant_app_permission", "toggle_data", "toggle_roaming")),                      # [confirmed]
    DecomposesInto("payment_and_service_restoration",
        ("send_payment_request", "make_payment", "resume_line")),                        # [inferred]
    DecomposesInto("hardware_and_sim_diagnosis",
        ("reseat_sim_card", "reboot_device", "toggle_data")),                            # [inferred]
    DecomposesInto("roaming_activation_flow",
        ("enable_roaming", "grant_app_permission", "toggle_roaming")),                   # [confirmed]
    DecomposesInto("vpn_and_security_fix",
        ("disconnect_vpn", "reset_apn_settings", "toggle_data")),                        # [inferred]
]

# 35. SUBTASK_OF — 도구가 속하는 추상 진단 단계 범주 (HTN 역방향; DECOMPOSES_INTO 역관계)
SUBTASK_OF: Dict[str, str] = {
    "toggle_airplane_mode":        "network_reset_phase",
    "reset_apn_settings":          "network_reset_phase",
    "set_network_mode_preference": "network_config_phase",
    "grant_app_permission":        "permission_setup_phase",
    "enable_roaming":              "roaming_setup_phase",
    "toggle_roaming":              "roaming_setup_phase",
    "toggle_wifi_calling":         "calling_setup_phase",
    "reboot_device":               "device_recovery_phase",
    "toggle_data":                 "connectivity_verification_phase",
    "refuel_data":                 "data_management_phase",
    "reseat_sim_card":             "hardware_diagnosis_phase",
    "disconnect_vpn":              "security_fix_phase",
    "toggle_data_saver_mode":      "data_management_phase",
    "make_payment":                "billing_resolution_phase",
    "resume_line":                 "service_restoration_phase",
    "send_payment_request":        "billing_resolution_phase",
    "transfer_to_human_agents":    "escalation_phase",
}

# 36. ACHIEVES_GOAL — 도구 호출이 달성하는 고객 레벨 목표 (HTN top-level goal; PDDL goal)
#     EFFECT_STATE와 차이: 시스템 내부 상태가 아닌 고객 서비스 관점 목표
ACHIEVES_GOAL: Dict[str, str] = {
    "resume_line":                 "restore_suspended_service",
    "refuel_data":                 "restore_data_connectivity",
    "toggle_data":                 "restore_data_connectivity",
    "enable_roaming":              "enable_international_access",
    "toggle_roaming":              "enable_international_access",
    "make_payment":                "resolve_billing_issue",
    "grant_app_permission":        "restore_app_functionality",
    "reset_apn_settings":          "fix_network_configuration",
    "reboot_device":               "fix_device_state",
    "disconnect_vpn":              "restore_direct_connection",
    "reseat_sim_card":             "fix_sim_connectivity",
    "transfer_to_human_agents":    "escalate_unresolved_issue",
    "toggle_wifi_calling":         "enable_wifi_calling",
    "set_network_mode_preference": "fix_network_configuration",
    "toggle_airplane_mode":        "fix_network_configuration",
    "send_payment_request":        "initiate_billing_resolution",
    "toggle_data_saver_mode":      "manage_data_usage",
}

# 37. REFINES — 추상 행동이 맥락에 따라 구체 도구로 구체화됨 (HTN method/refinement)
#     SCORED_PREFERENCE와 차이: 동일 맥락 내 점수 비교가 아닌 맥락별 추상→구체 매핑
REFINES: List[Refines] = [
    Refines("fix_network",     "toggle_airplane_mode",        "quick_reset_needed"),
    Refines("fix_network",     "reset_apn_settings",          "config_corruption_suspected"),
    Refines("fix_network",     "set_network_mode_preference", "mode_mismatch_suspected"),
    Refines("fix_data",        "toggle_data",                 "data_toggle_off"),
    Refines("fix_data",        "refuel_data",                 "data_balance_depleted"),
    Refines("fix_permissions", "grant_app_permission",        "app_permission_denied"),
    Refines("fix_billing",     "send_payment_request",        "no_payment_method_on_file"),
    Refines("fix_billing",     "make_payment",                "payment_method_on_file"),
    Refines("escalate",        "transfer_to_human_agents",    "automated_resolution_failed"),
    Refines("fix_hardware",    "reseat_sim_card",             "sim_error_detected"),
    Refines("fix_hardware",    "reboot_device",               "device_state_stale"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── Goal Vocabulary — Goal first-class entity (v5) ───────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Level 3: 고객이 표현하는 최상위 서비스 문제 범주
_GOAL_L3 = [
    "restore_mobile_service",       # 데이터·네트워크·기기 문제 통합
    "resolve_account_issue",        # 청구·회선 정지 통합
    "configure_advanced_features",  # 로밍·WiFi통화·데이터절약 통합
]

# Level 2: ACHIEVES_GOAL 값에서 도출된 기술적 서브목표 (13종)
_GOAL_L2 = [
    "restore_suspended_service",
    "restore_data_connectivity",
    "enable_international_access",
    "resolve_billing_issue",
    "restore_app_functionality",
    "fix_network_configuration",
    "fix_device_state",
    "restore_direct_connection",
    "fix_sim_connectivity",
    "escalate_unresolved_issue",
    "enable_wifi_calling",
    "initiate_billing_resolution",
    "manage_data_usage",
]

GOAL_VOCAB: List[str] = _GOAL_L3 + _GOAL_L2  # 16종


# ═══════════════════════════════════════════════════════════════════════════════
# ── Plan Step Vocabulary (GoalAct global plan의 추상 단계들) ─────────────────
# ═══════════════════════════════════════════════════════════════════════════════

PLAN_STEP_VOCAB: List[str] = [
    "identify_root_cause",       # 증상 수집 + 원인 판단
    "apply_targeted_fix",        # 해당 도구 호출로 수정 적용
    "verify_resolution",         # 수정 결과 확인
    "diagnose_account",          # 계정/청구 상태 확인
    "process_billing_action",    # 결제 관련 조치
    "restore_service_access",    # 서비스 복구
    "gather_symptoms",           # 초기 증상 수집
    "select_tool_strategy",      # 사용할 도구 전략 선택
    "execute_tool_sequence",     # 선택된 도구 순서 실행
    "confirm_customer_outcome",  # 고객 결과 확인 및 요약
    "close_or_escalate",         # 종료 또는 에스컬레이션
    "synthesize_issue_summary",  # 이슈 요약 생성
]

PLAN_SKILL_TYPES: List[str] = ["searching", "coding", "writing", "finish"]


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 I: GoalAct 실제 메커니즘 기반 관계 (v5 재설계) ─────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
#
# 출처: "Enhancing LLM-Based Agents via Global Planning and Hierarchical
#        Execution" (GoalAct, arXiv 2504.16563, NCIIP 2025 Best Paper)
#
# 핵심 구조: G_t = π(Q | T | S_t)
#   G_t  : global plan = ordered (plan_step_i, skill_type_i) pairs
#   S_t  : execution history (action + observation 누적)
#   → S_t를 반영해 매 스텝마다 G_t 전체를 재작성

# 38. PLAN_STEP_PRECEDES — GoalAct global plan 내 추상 단계 순서
#     PRECEDES(tool-level)와 차이: 추상 plan step 레벨의 순서 (goal과 tool 사이 중간 계층)
PLAN_STEP_PRECEDES: List[Tuple[str, str]] = [
    ("identify_root_cause",    "apply_targeted_fix"),       # [inferred] 원인 파악 → 수정 적용
    ("apply_targeted_fix",     "verify_resolution"),        # [inferred] 수정 → 결과 확인
    ("verify_resolution",      "close_or_escalate"),        # [inferred] 확인 → 완료 또는 이관
    ("gather_symptoms",        "identify_root_cause"),      # [inferred] 증상 수집 → 원인 판단
    ("identify_root_cause",    "select_tool_strategy"),     # [inferred] 원인 판단 → 전략 선택
    ("select_tool_strategy",   "execute_tool_sequence"),    # [inferred] 전략 → 실행
    ("execute_tool_sequence",  "verify_resolution"),        # [inferred] 실행 → 확인
    ("diagnose_account",       "process_billing_action"),   # [inferred] 계정 진단 → 청구 처리
    ("process_billing_action", "restore_service_access"),   # [inferred] 청구 처리 → 서비스 복구
    ("restore_service_access", "confirm_customer_outcome"), # [inferred] 복구 → 결과 확인
    ("verify_resolution",      "synthesize_issue_summary"), # [inferred] 확인 → 요약
    ("close_or_escalate",      "synthesize_issue_summary"), # [inferred] 종료 → 요약
]

# 39. PLAN_STEP_SKILL — GoalAct 핵심 기여: 각 plan step에 4종 skill type 배정
#     GoalAct가 ReAct·CodeAct와 다른 이유: 단일 실행 메커니즘이 아닌 step별 최적 skill 선택
PLAN_STEP_SKILL: Dict[str, str] = {
    "identify_root_cause":      "searching",  # 도구 호출로 상태 조회
    "apply_targeted_fix":       "searching",  # 도구 호출로 수정 적용
    "verify_resolution":        "searching",  # 도구 호출로 결과 확인
    "diagnose_account":         "searching",  # 계정 상태 조회
    "process_billing_action":   "searching",  # 결제 도구 호출
    "restore_service_access":   "searching",  # 서비스 복구 도구 호출
    "gather_symptoms":          "searching",  # 초기 상태 수집
    "select_tool_strategy":     "searching",  # 가용 도구 조회·비교
    "execute_tool_sequence":    "searching",  # 순차 도구 실행
    "confirm_customer_outcome": "writing",    # 고객 응답 메시지 생성
    "synthesize_issue_summary": "writing",    # 이슈 리포트 작성
    "close_or_escalate":        "finish",     # 세션 종료 신호
}

# 40. PLAN_REVISED_TO — GoalAct G_t 갱신: 관찰 결과로 plan step 교체
#     BACKTRACK_TO와 차이: 도구 레벨 상태 복귀가 아닌 추상 plan step 레벨 수정 (G_t rewrite)
#     OBSERVATION_TRIGGERS와 차이: 도구 직접 호출이 아닌 plan step 교체
PLAN_REVISED_TO: List[PlanRevisedTo] = [
    PlanRevisedTo("toggle_data_failed",       "apply_targeted_fix",     "identify_root_cause"),   # [inferred] 도구 실패 → 원인 재파악
    PlanRevisedTo("apn_reset_ineffective",    "apply_targeted_fix",     "select_tool_strategy"),  # [inferred] APN 실패 → 전략 재선택
    PlanRevisedTo("payment_method_missing",   "process_billing_action", "diagnose_account"),      # [inferred] 결제 수단 없음 → 계정 재진단
    PlanRevisedTo("roaming_still_inactive",   "apply_targeted_fix",     "diagnose_account"),      # [inferred] 로밍 미활성 → 계정 확인
    PlanRevisedTo("sim_error_persists",       "apply_targeted_fix",     "close_or_escalate"),     # [inferred] SIM 오류 지속 → 이관
    PlanRevisedTo("automated_tools_failed",   "execute_tool_sequence",  "close_or_escalate"),     # [inferred] 자동화 실패 → 에스컬레이션
    PlanRevisedTo("network_fix_succeeded",    "identify_root_cause",    "verify_resolution"),     # [inferred] 수정 성공 → 검증으로 점프
    PlanRevisedTo("payment_confirmed",        "process_billing_action", "restore_service_access"),# [inferred] 결제 완료 → 서비스 복구
]

# 41. STEP_REALIZES_TOOL — GoalAct plan step이 구체 도구 호출로 실현됨
#     REFINES(abstract_action→tool)와 차이: plan step이 GoalAct G_t의 한 원소이며
#     실제 실행 시 해당 도구 호출로 구체화됨 (plan-to-execution bridge)
STEP_REALIZES_TOOL: List[Tuple[str, str]] = [
    ("identify_root_cause",   "reset_apn_settings"),       # [inferred] 네트워크 상태 확인
    ("identify_root_cause",   "toggle_data"),              # [inferred] 데이터 상태 확인
    ("identify_root_cause",   "toggle_airplane_mode"),     # [inferred] 네트워크 초기화 시도
    ("apply_targeted_fix",    "toggle_airplane_mode"),     # [inferred]
    ("apply_targeted_fix",    "reboot_device"),            # [inferred]
    ("apply_targeted_fix",    "reseat_sim_card"),          # [inferred]
    ("apply_targeted_fix",    "refuel_data"),              # [inferred]
    ("apply_targeted_fix",    "make_payment"),             # [inferred]
    ("apply_targeted_fix",    "enable_roaming"),           # [inferred]
    ("apply_targeted_fix",    "grant_app_permission"),     # [inferred]
    ("verify_resolution",     "toggle_data"),              # [inferred] 데이터 상태 재확인
    ("verify_resolution",     "toggle_roaming"),           # [inferred]
    ("process_billing_action","send_payment_request"),     # [inferred]
    ("process_billing_action","make_payment"),             # [inferred]
    ("restore_service_access","resume_line"),              # [inferred]
    ("close_or_escalate",     "transfer_to_human_agents"), # [inferred]
]

# 42. PLAN_COMMITTED_TO_GOAL — GoalAct G_t의 각 plan step이 Q(user query)에 대한 goal에
#     명시적으로 commit됨 (goal persistence across plan steps)
#     ACHIEVES_GOAL(tool→goal)와 차이: plan step이 주어이며 실행 중 commitment를 나타냄
PLAN_COMMITTED_TO_GOAL: List[Tuple[str, str]] = [
    ("identify_root_cause",    "restore_mobile_service"),
    ("apply_targeted_fix",     "fix_network_configuration"),
    ("apply_targeted_fix",     "restore_data_connectivity"),
    ("verify_resolution",      "restore_data_connectivity"),
    ("diagnose_account",       "resolve_account_issue"),
    ("process_billing_action", "resolve_billing_issue"),
    ("restore_service_access", "restore_suspended_service"),
    ("gather_symptoms",        "restore_mobile_service"),
    ("close_or_escalate",      "escalate_unresolved_issue"),
    ("execute_tool_sequence",  "fix_network_configuration"),
    ("confirm_customer_outcome","restore_mobile_service"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 기하학적 분류 — A7 실험 배정
# ═══════════════════════════════════════════════════════════════════════════════

RELATION_GEOMETRY: Dict[str, str] = {
    # ── 방향성(Directional) → A6 Per-head Rotation 우위 예측 (17종) ──
    "precedes":          "directional",
    "directly_follows":  "directional",
    "causal_link":       "directional",
    "requires":          "directional",
    "enables":           "directional",
    "parameter_feeds":   "directional",
    "and_join":          "directional",
    "validates":         "directional",
    "retry_after_fail":  "directional",
    "error_fallback":    "directional",
    "compensates":       "directional",
    "tool_subsumes":     "directional",
    "state_transition":  "directional",   # from_state→to_state = 방향성
    "backtrack_to":      "directional",   # dead_end→restore = 방향성
    "pruned_by":         "directional",   # pruner→pruned = 비대칭 방향성
    "fan_out":           "directional",   # source→多branches = 방향성
    "decomposes_into":   "directional",   # goal→tool_set = 계층 방향성
    # ── 대칭/조건부(Symmetric/Conditional) → T1 Additive 우위 예측 (6종) ──
    "mutex":             "symmetric",
    "exclusive_choice":  "conditional",
    "parallel_safe":     "symmetric",
    "conditional_on":    "conditional",
    "scored_preference": "conditional",   # 맥락 의존 선호도
    "refines":           "conditional",   # 맥락 의존 추상→구체 매핑
    # ── 범주형/속성(Categorical) → T1 Additive 우위 예측 (14종) ──
    "workflow_role":         "categorical",
    "domain_category":       "categorical",
    "checkpoint":            "categorical",
    "precondition_state":    "categorical",
    "effect_state":          "categorical",
    "idempotent":            "categorical",
    "reversible":            "categorical",
    "mandatory_in_flow":     "categorical",
    "optional_in_flow":      "categorical",
    "loop_capable":          "categorical",
    "observation_triggers":  "categorical",  # obs→tool 반응 규칙
    "guardrail":             "categorical",  # 차단 금지 조건
    "subtask_of":            "categorical",  # 도구→진단 단계 레이블
    "achieves_goal":         "categorical",  # 도구→고객 목표 레이블
    # ── GoalAct 실제 구조 (5종) — G_t = π(Q|T|S_t) 기반 ──
    "plan_step_precedes":     "directional",  # plan step 순서, tool보다 추상 레벨
    "plan_step_skill":        "categorical",  # step→{searching,coding,writing,finish}
    "plan_revised_to":        "conditional",  # 관찰→plan step 교체, S_t 기반
    "step_realizes_tool":     "directional",  # plan step→concrete tool, 수직 브릿지
    "plan_committed_to_goal": "directional",  # plan step→goal, 실행 중 commitment
}

PREDICTED_METHOD: Dict[str, str] = {
    k: ("A6" if v == "directional" else "T1")
    for k, v in RELATION_GEOMETRY.items()
}


# ═══════════════════════════════════════════════════════════════════════════════
# 실측 빈도 (가중치 스케일링용)
# ═══════════════════════════════════════════════════════════════════════════════
PRECEDES_FREQ: Dict[Tuple[str, str], int] = {
    ("reset_apn_settings",          "reboot_device"):            1036,
    ("toggle_airplane_mode",        "set_network_mode_preference"): 576,
    ("set_network_mode_preference", "grant_app_permission"):     1024,
    ("toggle_airplane_mode",        "grant_app_permission"):     1024,
    ("grant_app_permission",        "toggle_data"):              1024,
    ("grant_app_permission",        "toggle_roaming"):           1024,
    ("grant_app_permission",        "refuel_data"):              1024,
    ("enable_roaming",              "toggle_roaming"):            560,
    ("reset_apn_settings",          "grant_app_permission"):     1024,
    ("reboot_device",               "grant_app_permission"):     1024,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 전체 도구 목록 및 설명
# ═══════════════════════════════════════════════════════════════════════════════
ALL_TOOLS = sorted({
    "disconnect_vpn", "enable_roaming", "grant_app_permission", "make_payment",
    "reboot_device", "refuel_data", "reseat_sim_card", "reset_apn_settings",
    "resume_line", "send_payment_request", "set_network_mode_preference",
    "toggle_airplane_mode", "toggle_data", "toggle_data_saver_mode",
    "toggle_roaming", "toggle_wifi_calling", "transfer_to_human_agents",
})

TOOL_DESC: Dict[str, str] = {
    "toggle_airplane_mode":        "toggle airplane mode on the device",
    "reset_apn_settings":          "reset APN network settings to default",
    "set_network_mode_preference": "set the preferred network mode (4G/5G)",
    "grant_app_permission":        "grant necessary app permissions",
    "enable_roaming":              "enable international roaming on the account",
    "reboot_device":               "reboot the customer's device",
    "toggle_data":                 "toggle mobile data on or off",
    "toggle_roaming":              "toggle the roaming setting",
    "toggle_wifi_calling":         "toggle Wi-Fi calling feature",
    "refuel_data":                 "add more data to the account",
    "reseat_sim_card":             "virtually reseat the SIM card",
    "disconnect_vpn":              "disconnect the active VPN connection",
    "toggle_data_saver_mode":      "toggle data saver mode",
    "make_payment":                "process a payment on the account",
    "resume_line":                 "resume suspended service on the line",
    "send_payment_request":        "send a payment request to the customer",
    "transfer_to_human_agents":    "transfer the customer to a human agent",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 전체 관계 통계 출력
# ═══════════════════════════════════════════════════════════════════════════════

ALL_RELATIONS = {
    # binary directional
    "precedes":          (PRECEDES,         "List[Tuple]"),
    "directly_follows":  (DIRECTLY_FOLLOWS, "List[NamedTuple]"),
    "causal_link":       (CAUSAL_LINK,      "List[NamedTuple]"),
    "requires":          (REQUIRES,         "List[Tuple]"),
    "enables":           (ENABLES,          "List[Tuple]"),
    "parameter_feeds":   (PARAMETER_FEEDS,  "List[NamedTuple]"),
    "and_join":          (AND_JOIN,         "List[NamedTuple]"),
    "validates":         (VALIDATES,        "List[Tuple]"),
    "retry_after_fail":  (RETRY_AFTER_FAIL, "List[NamedTuple]"),
    "error_fallback":    (ERROR_FALLBACK,   "List[NamedTuple]"),
    "compensates":       (COMPENSATES,      "List[Tuple]"),
    "tool_subsumes":     (TOOL_SUBSUMES,    "List[Tuple]"),
    "state_transition":  (STATE_TRANSITION, "List[NamedTuple]"),
    # binary symmetric/conditional
    "mutex":             (MUTEX,            "List[Tuple]"),
    "exclusive_choice":  (EXCLUSIVE_CHOICE, "List[NamedTuple]"),
    "parallel_safe":     (PARALLEL_SAFE,    "List[Tuple]"),
    "conditional_on":    (CONDITIONAL_ON,   "List[NamedTuple]"),
    # categorical/property
    "workflow_role":      (WORKFLOW_ROLE,      "Dict[str,str]"),
    "domain_category":    (DOMAIN_CATEGORY,    "Dict[str,str]"),
    "checkpoint":         (CHECKPOINT,         "List[str]"),
    "precondition_state": (PRECONDITION_STATE, "Dict[str,str]"),
    "effect_state":       (EFFECT_STATE,       "Dict[str,Dict]"),
    "idempotent":         (IDEMPOTENT,         "Dict[str,bool]"),
    "reversible":         (REVERSIBLE,         "Dict[str,bool]"),
    "mandatory_in_flow":  (MANDATORY_IN_FLOW,  "Dict[str,bool]"),
    "optional_in_flow":   (OPTIONAL_IN_FLOW,   "Dict[str,bool]"),
    "loop_capable":       (LOOP_CAPABLE,       "Dict[str,bool]"),
    # GoT/ToT/Harness (그룹 G)
    "backtrack_to":       (BACKTRACK_TO,        "List[NamedTuple]"),
    "pruned_by":          (PRUNED_BY,           "List[Tuple]"),
    "fan_out":            (FAN_OUT,             "List[NamedTuple]"),
    "observation_triggers":(OBSERVATION_TRIGGERS,"Dict[str,str]"),
    "guardrail":          (GUARDRAIL,           "Dict[str,str]"),
    "scored_preference":  (SCORED_PREFERENCE,   "List[NamedTuple]"),
    # HTN 계층 (그룹 H)
    "decomposes_into":    (DECOMPOSES_INTO,     "List[NamedTuple]"),
    "subtask_of":         (SUBTASK_OF,          "Dict[str,str]"),
    "achieves_goal":      (ACHIEVES_GOAL,       "Dict[str,str]"),
    "refines":            (REFINES,             "List[NamedTuple]"),
    # GoalAct 실제 구조 (그룹 I)
    "plan_step_precedes":    (PLAN_STEP_PRECEDES,    "List[Tuple]"),
    "plan_step_skill":       (PLAN_STEP_SKILL,       "Dict[str,str]"),
    "plan_revised_to":       (PLAN_REVISED_TO,       "List[NamedTuple]"),
    "step_realizes_tool":    (STEP_REALIZES_TOOL,    "List[Tuple]"),
    "plan_committed_to_goal":(PLAN_COMMITTED_TO_GOAL,"List[Tuple]"),
}


def relation_summary() -> None:
    """전체 관계 통계 출력."""
    a6_total = t1_total = 0
    confirmed = inferred = hypothetical = 0

    print(f"\n{'관계':22s} {'기하':12s} {'예측':>4s}  {'수':>4s}  {'유형'}")
    print("─" * 70)
    for rel, (data, dtype) in ALL_RELATIONS.items():
        geo  = RELATION_GEOMETRY.get(rel, "?")
        pred = PREDICTED_METHOD.get(rel, "?")
        if isinstance(data, list):
            n = len(data)
        elif isinstance(data, dict):
            n = len(data)
        else:
            n = 0
        print(f"  {rel:20s} {geo:12s} {pred:>4s}  {n:>4d}  {dtype}")
        if pred == "A6":
            a6_total += n
        else:
            t1_total += n
    print("─" * 70)
    print(f"  총 관계 유형: {len(ALL_RELATIONS)}종")
    print(f"  A6 후보 합계: {a6_total}개 항목")
    print(f"  T1 후보 합계: {t1_total}개 항목")
    print()


def validate(tau2_telecom_tasks_path: str | None = None) -> None:
    """tasks.json과 도구 목록 교차 검증."""
    import json
    path = tau2_telecom_tasks_path or (
        "/home/woori/workspace_common/boltzmann-attention/"
        "external/tau2-bench/data/tau2/domains/telecom/tasks.json"
    )
    try:
        with open(path) as f:
            tasks = json.load(f)
        observed: set[str] = set()
        for t in tasks:
            for a in (t.get("evaluation_criteria") or {}).get("actions") or []:
                if a.get("name"):
                    observed.add(a["name"])
    except FileNotFoundError:
        print(f"[validate] tasks.json not found at {path} — skipping")
        return
    defined = set(ALL_TOOLS)
    missing = defined - observed
    extra   = observed - defined
    if not missing and not extra:
        print(f"[validate] OK: all {len(defined)} tools match tasks.json")
    if missing:
        print(f"[validate] WARN: defined but not in tasks: {missing}")
    if extra:
        print(f"[validate] INFO: in tasks but not defined: {extra}")


if __name__ == "__main__":
    relation_summary()
    validate()
