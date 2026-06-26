"""generate_contrast_pairs_v3.py

Phase 0 v3/v4: 37종 온톨로지 관계 대조쌍 생성.

관계 그룹:
  A. 순서/연쇄 (directional → A6): precedes, directly_follows, causal_link
  B. 의존/활성 (directional → A6): requires, enables, parameter_feeds, and_join
  C. 검증/복구 (directional → A6): validates, retry_after_fail, error_fallback,
                                    compensates, tool_subsumes, state_transition
  D. 배제/선택 (symmetric/cond → T1): mutex, exclusive_choice, parallel_safe,
                                       conditional_on
  E. 상태/효과 (categorical → T1): precondition_state, effect_state
  F. 도구 속성 (categorical → T1): workflow_role, domain_category, checkpoint,
                                    idempotent, reversible, mandatory_in_flow,
                                    optional_in_flow, loop_capable
  G. GoT/ToT/Harness (mixed): backtrack_to, pruned_by, fan_out [A6],
                               observation_triggers, guardrail [T1],
                               scored_preference [T1]
  H. HTN 계층 (mixed): decomposes_into [A6],
                        subtask_of, achieves_goal, refines [T1]
  I. Goal Lifecycle (mixed): goal_stack_push, goal_monitors [A6],
                              skill_retrieval_for, replanning_trigger [T1],
                              goal_context_window [T1]

Usage:
  python3 scripts/ontology/generate_contrast_pairs_v3.py \\
    --tau2 <tasks.json> \\
    --out  reports/facet_rft_2026/phase0_probing/contrast_pairs_v3.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── 온톨로지 임포트 ────────────────────────────────────────────────────────────
_ONTO_PATH = os.path.join(os.path.dirname(__file__), "tau2_telecom_ontology.py")
_spec = importlib.util.spec_from_file_location("onto", _ONTO_PATH)
_onto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_onto)

PRECEDES         = _onto.PRECEDES
DIRECTLY_FOLLOWS = _onto.DIRECTLY_FOLLOWS
CAUSAL_LINK      = _onto.CAUSAL_LINK
REQUIRES         = _onto.REQUIRES
ENABLES          = _onto.ENABLES
PARAMETER_FEEDS  = _onto.PARAMETER_FEEDS
AND_JOIN         = _onto.AND_JOIN
VALIDATES        = _onto.VALIDATES
RETRY_AFTER_FAIL = _onto.RETRY_AFTER_FAIL
ERROR_FALLBACK   = _onto.ERROR_FALLBACK
COMPENSATES      = _onto.COMPENSATES
TOOL_SUBSUMES    = _onto.TOOL_SUBSUMES
STATE_TRANSITION = _onto.STATE_TRANSITION
MUTEX            = _onto.MUTEX
EXCLUSIVE_CHOICE = _onto.EXCLUSIVE_CHOICE
PARALLEL_SAFE    = _onto.PARALLEL_SAFE
CONDITIONAL_ON   = _onto.CONDITIONAL_ON
WORKFLOW_ROLE    = _onto.WORKFLOW_ROLE
DOMAIN_CATEGORY  = _onto.DOMAIN_CATEGORY
CHECKPOINT       = _onto.CHECKPOINT
PRECONDITION_STATE = _onto.PRECONDITION_STATE
EFFECT_STATE     = _onto.EFFECT_STATE
IDEMPOTENT       = _onto.IDEMPOTENT
REVERSIBLE       = _onto.REVERSIBLE
MANDATORY_IN_FLOW= _onto.MANDATORY_IN_FLOW
OPTIONAL_IN_FLOW = _onto.OPTIONAL_IN_FLOW
LOOP_CAPABLE     = _onto.LOOP_CAPABLE
# 그룹 G
BACKTRACK_TO         = _onto.BACKTRACK_TO
PRUNED_BY            = _onto.PRUNED_BY
FAN_OUT              = _onto.FAN_OUT
OBSERVATION_TRIGGERS = _onto.OBSERVATION_TRIGGERS
GUARDRAIL            = _onto.GUARDRAIL
SCORED_PREFERENCE    = _onto.SCORED_PREFERENCE
# 그룹 H
DECOMPOSES_INTO  = _onto.DECOMPOSES_INTO
SUBTASK_OF       = _onto.SUBTASK_OF
ACHIEVES_GOAL    = _onto.ACHIEVES_GOAL
REFINES          = _onto.REFINES
# 그룹 I (GoalAct 실제 구조)
GOAL_VOCAB            = _onto.GOAL_VOCAB
PLAN_STEP_VOCAB       = _onto.PLAN_STEP_VOCAB
PLAN_SKILL_TYPES      = _onto.PLAN_SKILL_TYPES
PLAN_STEP_PRECEDES    = _onto.PLAN_STEP_PRECEDES
PLAN_STEP_SKILL       = _onto.PLAN_STEP_SKILL
PLAN_REVISED_TO       = _onto.PLAN_REVISED_TO
STEP_REALIZES_TOOL    = _onto.STEP_REALIZES_TOOL
PLAN_COMMITTED_TO_GOAL= _onto.PLAN_COMMITTED_TO_GOAL
ALL_TOOLS        = _onto.ALL_TOOLS
TOOL_DESC        = _onto.TOOL_DESC
PREDICTED_METHOD = _onto.PREDICTED_METHOD
RELATION_GEOMETRY= _onto.RELATION_GEOMETRY

TAU2_TELECOM = (
    "/home/woori/workspace_common/boltzmann-attention/"
    "external/tau2-bench/data/tau2/domains/telecom/tasks.json"
)


# ═══════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════

def get_descriptions(tasks: List[dict]) -> List[str]:
    descs = []
    for t in tasks:
        us = t.get("user_scenario") or {}
        instr = us.get("instructions") if isinstance(us, dict) else None
        if isinstance(instr, dict):
            text = f"{instr.get('reason_for_call','')} {instr.get('known_info','')}".strip()
        elif isinstance(instr, str):
            text = instr.strip()
        else:
            text = ""
        if text:
            descs.append(text)
    return descs

def wrong_tool(exclude: set, rng: random.Random) -> str:
    pool = [t for t in ALL_TOOLS if t not in exclude]
    return rng.choice(pool) if pool else rng.choice(list(ALL_TOOLS))

def wrong_goal(exclude: set, rng: random.Random) -> str:
    pool = [g for g in GOAL_VOCAB if g not in exclude]
    return rng.choice(pool) if pool else rng.choice(GOAL_VOCAB)

def wrong_plan_step(exclude: set, rng: random.Random) -> str:
    pool = [s for s in PLAN_STEP_VOCAB if s not in exclude]
    return rng.choice(pool) if pool else rng.choice(PLAN_STEP_VOCAB)

def record(pair_type, geometry, label, tool_a, tool_b, text, rng) -> dict:
    return {
        "pair_type": pair_type,
        "geometry":  geometry,
        "predicted_method": PREDICTED_METHOD.get(pair_type, "?"),
        "tool_a": tool_a,
        "tool_b": tool_b,
        "label": label,
        "text": text,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 A: 순서/연쇄 (Directional → A6) ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

TMPL_DIRECTIONAL = {
    "precedes": (
        "Troubleshooting: {desc} The agent first called [{A}], then later called [{B}].",
        "Troubleshooting: {desc} The agent first called [{B}], then later called [{A}].",
    ),
    "directly_follows": (
        "Step-by-step fix for: {desc} The agent called [{A}] and immediately next called [{B}].",
        "Step-by-step fix for: {desc} The agent called [{B}] and immediately next called [{A}].",
    ),
    "causal_link": (
        "Resolution of {desc}: [{A}] was completed first (achieving the required state), "
        "then [{B}] was called successfully.",
        "Resolution of {desc}: [{B}] was attempted first, before [{A}] had established "
        "the necessary precondition.",
    ),
    "requires": (
        "To resolve {desc}, the agent ensured [{B}] was completed before calling [{A}].",
        "The agent attempted [{A}] for {desc} without first completing [{B}].",
    ),
    "enables": (
        "After calling [{A}] for {desc}, the system state allowed [{B}] to be called.",
        "The agent tried to call [{B}] for {desc} before [{A}] had been executed.",
    ),
    "parameter_feeds": (
        "Workflow for {desc}: [{A}] was called and its output was passed as input to [{B}].",
        "Workflow for {desc}: [{B}] was called before [{A}] could provide its required output.",
    ),
    "validates": (
        "After fixing {desc} with [{B}], the agent called [{A}] to verify the result.",
        "The agent called [{A}] for {desc} before [{B}] had been executed.",
    ),
    "retry_after_fail": (
        "During {desc}: [{A}] failed, so the agent called [{B}] to recover, then retried [{A}].",
        "During {desc}: [{B}] was called before [{A}] had been attempted.",
    ),
    "error_fallback": (
        "After [{A}] repeatedly failed for {desc}, the agent switched to [{B}] as an alternative.",
        "The agent called [{B}] first for {desc} without attempting [{A}].",
    ),
    "compensates": (
        "After [{B}] was called for {desc}, [{A}] was used to reverse its effect.",
        "The agent called [{A}] for {desc} without the prior execution of [{B}].",
    ),
    "tool_subsumes": (
        "For {desc}, [{A}] was chosen because it covers all the functionality of [{B}] and more.",
        "For {desc}, [{B}] was chosen instead of [{A}], which could have handled the case more completely.",
    ),
    "state_transition": (
        "To resolve {desc}: [{A}] transitioned the system from its initial state to the target state.",
        "For {desc}: [{B}] was called expecting a transition, but [{A}] was the correct tool for this state change.",
    ),
    "backtrack_to": (
        "After [{A}] reached a dead end during {desc}, the agent backtracked to the state "
        "established by [{B}] and retried from there.",
        "After [{A}] reached a dead end during {desc}, the agent pressed forward and called "
        "[{B}] as the next step instead of backtracking.",
    ),
    "pruned_by": (
        "During {desc}: [{B}] resolved the issue, so [{A}] was eliminated from the plan.",
        "During {desc}: [{A}] resolved the issue, so [{B}] was eliminated from the plan.",
    ),
    "decomposes_into": (
        "The diagnostic goal for {desc} was broken down into sub-steps starting with [{A}], "
        "then [{B}], among others.",
        "The diagnostic goal for {desc} was addressed by [{B}] alone, without decomposing "
        "into sub-steps like [{A}].",
    ),
    "plan_step_precedes": (
        "In the GoalAct global plan for {desc}, step '{A}' was scheduled before step '{B}'.",
        "In the GoalAct global plan for {desc}, step '{B}' was scheduled before step '{A}'.",
    ),
    "step_realizes_tool": (
        "During {desc}, the plan step '{A}' was realized by calling [{B}].",
        "During {desc}, the plan step '{A}' was realized by calling [{B}], "
        "which is unrelated to that step.",
    ),
    "plan_committed_to_goal": (
        "Throughout {desc}, plan step '{A}' remained committed to goal [{B}].",
        "Throughout {desc}, plan step '{A}' was mistakenly committed to goal [{B}].",
    ),
}

def make_directional_pairs(
    rel_name: str, pairs: list, descs: List[str],
    rng: random.Random, n_per_pair: int,
) -> List[dict]:
    pos_t, neg_t = TMPL_DIRECTIONAL.get(rel_name, TMPL_DIRECTIONAL["precedes"])
    geo = RELATION_GEOMETRY.get(rel_name, "directional")
    records, seen = [], set()

    for entry in pairs:
        # NamedTuple 또는 Tuple 처리
        if hasattr(entry, "achiever"):        a, b = entry.achiever, entry.beneficiary
        elif hasattr(entry, "dead_end_tool"): a, b = entry.dead_end_tool, entry.restore_point_tool
        elif hasattr(entry, "source"):        a, b = entry.source, entry.target
        elif hasattr(entry, "failing_tool"):  a, b = entry.failing_tool, entry.fix_tool
        elif hasattr(entry, "primary"):       a, b = entry.primary, entry.fallback
        elif hasattr(entry, "tool"):          a, b = entry.tool, entry.to_state  # state_transition
        else:                                 a, b = entry[0], entry[1]

        for _ in range(n_per_pair):
            desc = rng.choice(descs)[:150]
            key = (rel_name, a, b, desc[:25])
            if key in seen: continue
            seen.add(key)
            records.append(record(rel_name, geo, 1, a, b, pos_t.format(desc=desc, A=a, B=b), rng))
            records.append(record(rel_name, geo, 0, a, b, neg_t.format(desc=desc, A=a, B=b), rng))

    rng.shuffle(records)
    return records


def make_and_join_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    records, seen = [], set()
    for entry in AND_JOIN:
        pre = entry.prerequisites
        tgt = entry.target
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key = ("and_join", str(pre), tgt, desc[:25])
            if key in seen: continue
            seen.add(key)
            pre_str = " and ".join(f"[{p}]" for p in pre)
            miss = rng.choice(pre)
            pos_txt = (f"For {desc}: after completing {pre_str}, the agent called [{tgt}].")
            neg_txt = (f"For {desc}: the agent called [{tgt}] after completing only [{miss}], "
                       f"without the other prerequisites.")
            records.append(record("and_join", "directional", 1, pre[0], tgt, pos_txt, rng))
            records.append(record("and_join", "directional", 0, pre[0], tgt, neg_txt, rng))
    rng.shuffle(records)
    return records


def make_state_transition_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    records, seen = [], set()
    for entry in STATE_TRANSITION:
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key = ("state_transition", entry.tool, entry.from_state, entry.to_state, desc[:25])
            if key in seen: continue
            seen.add(key)
            wrong = rng.choice([t for t in ALL_TOOLS if t != entry.tool])
            pos_txt = (f"For {desc}: the system was in [{entry.from_state}]. "
                       f"The agent called [{entry.tool}] to transition it to [{entry.to_state}].")
            neg_txt = (f"For {desc}: the system was in [{entry.from_state}]. "
                       f"The agent called [{wrong}] instead, which does not perform this transition.")
            records.append(record("state_transition", "directional", 1, entry.tool, None, pos_txt, rng))
            records.append(record("state_transition", "directional", 0, wrong, None, neg_txt, rng))
    rng.shuffle(records)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 D: 배제/선택 (Symmetric/Conditional → T1) ────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

TMPL_SYMMETRIC = {
    "mutex": (
        "During {desc}: the agent determined only [{A}] should be called — not [{B}].",
        "During {desc}: the agent called both [{A}] and [{B}] in the same session.",
    ),
    "parallel_safe": (
        "Fixing {desc}: [{A}] and [{B}] were called in either order — they are independent.",
        "Fixing {desc}: [{A}] had to be completed strictly before [{B}] could proceed.",
    ),
    "exclusive_choice": (
        "For {desc}: given the condition, the agent chose [{A}] over [{B}].",
        "For {desc}: the agent called both [{A}] and [{B}], ignoring the exclusive condition.",
    ),
    "conditional_on": (
        "For {desc}: [{B}] was called first to verify the condition; then [{A}] was called.",
        "For {desc}: [{A}] was called immediately without first checking via [{B}].",
    ),
}

def make_symmetric_pairs(
    rel_name: str, pairs: list, descs: List[str],
    rng: random.Random, n_per_pair: int,
) -> List[dict]:
    pos_t, neg_t = TMPL_SYMMETRIC.get(rel_name, TMPL_SYMMETRIC["mutex"])
    geo = RELATION_GEOMETRY.get(rel_name, "symmetric")
    records, seen = [], set()

    for entry in pairs:
        if hasattr(entry, "option_a"):      a, b = entry.option_a, entry.option_b  # ExclusiveChoice
        elif hasattr(entry, "tool"):        a, b = entry.tool, entry.trigger        # ConditionalOn
        else:                               a, b = entry[0], entry[1]

        w = wrong_tool({a, b}, rng)
        for _ in range(n_per_pair):
            desc = rng.choice(descs)[:150]
            key = (rel_name, a, b, desc[:25])
            if key in seen: continue
            seen.add(key)
            records.append(record(rel_name, geo, 1, a, b, pos_t.format(desc=desc, A=a, B=b), rng))
            # Negative: 잘못된 도구 조합 또는 순서 위반
            records.append(record(rel_name, geo, 0, a, w, neg_t.format(desc=desc, A=a, B=w), rng))

    rng.shuffle(records)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 E/F: 범주형/속성 (Categorical → T1) ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def make_dict_str_pairs(
    rel_name: str, role_dict: Dict[str, str],
    descs: List[str], rng: random.Random, n_per_tool: int,
) -> List[dict]:
    """Dict[tool, value] → 올바른 value vs 잘못된 value."""
    TMPL = {
        "workflow_role": (
            "In the workflow for {desc}, '{tool}' serves as a {val} step.",
            "In the workflow for {desc}, '{tool}' serves as a {wrong} step.",
        ),
        "domain_category": (
            "For issue type '{val}', the agent called '{tool}' in session: {desc}",
            "For issue type '{wrong}', the agent called '{tool}' in session: {desc}",
        ),
        "precondition_state": (
            "The agent verified [{val}] before calling '{tool}'. Context: {desc}",
            "The agent assumed [{wrong}] before calling '{tool}'. Context: {desc}",
        ),
        "observation_triggers": (
            "During {desc}, observing [{tool}] in system state correctly triggers '{val}'.",
            "During {desc}, observing [{tool}] in system state incorrectly triggers '{wrong}'.",
        ),
        "guardrail": (
            "For {desc}: '{tool}' is blocked because guardrail condition [{val}] is active.",
            "For {desc}: '{tool}' is blocked because guardrail condition [{wrong}] is active.",
        ),
        "subtask_of": (
            "In the troubleshooting workflow for {desc}, '{tool}' belongs to the {val} phase.",
            "In the troubleshooting workflow for {desc}, '{tool}' belongs to the {wrong} phase.",
        ),
        "achieves_goal": (
            "Calling '{tool}' for {desc} achieves the customer goal: [{val}].",
            "Calling '{tool}' for {desc} achieves the customer goal: [{wrong}].",
        ),
        "plan_step_skill": (
            "In the GoalAct plan for {desc}, step '{tool}' is assigned skill type '{val}'.",
            "In the GoalAct plan for {desc}, step '{tool}' is assigned skill type '{wrong}'.",
        ),
    }
    pos_t, neg_t = TMPL.get(rel_name, TMPL["workflow_role"])
    all_vals = list(set(role_dict.values()))
    geo = "categorical"
    records, seen = [], set()

    for tool, val in role_dict.items():
        wrong_vals = [v for v in all_vals if v != val]
        if not wrong_vals: continue
        for _ in range(n_per_tool):
            desc = rng.choice(descs)[:120]
            wr = rng.choice(wrong_vals)
            key = (rel_name, tool, desc[:20])
            if key in seen: continue
            seen.add(key)
            records.append(record(rel_name, geo, 1, tool, None,
                                  pos_t.format(desc=desc, tool=tool, val=val, wrong=wr), rng))
            records.append(record(rel_name, geo, 0, tool, None,
                                  neg_t.format(desc=desc, tool=tool, val=val, wrong=wr), rng))

    rng.shuffle(records)
    return records


def make_bool_property_pairs(
    rel_name: str, bool_dict: Dict[str, bool],
    descs: List[str], rng: random.Random, n_per_tool: int,
) -> List[dict]:
    """Dict[tool, bool] → True 도구 vs False 도구."""
    TMPL = {
        "idempotent": (
            "Calling '{tool}' multiple times in session for {desc} is safe — it is idempotent.",
            "Calling '{wrong}' multiple times in session for {desc} is safe — it is idempotent.",
        ),
        "reversible": (
            "The effect of '{tool}' can be undone if needed during {desc}.",
            "The effect of '{wrong}' can be undone if needed during {desc}.",
        ),
        "mandatory_in_flow": (
            "In standard troubleshooting flows for {desc}, '{tool}' is always called.",
            "In standard troubleshooting flows for {desc}, '{wrong}' is always called.",
        ),
        "optional_in_flow": (
            "In most troubleshooting flows for {desc}, '{tool}' may be skipped.",
            "In most troubleshooting flows for {desc}, '{wrong}' may be skipped.",
        ),
        "loop_capable": (
            "During {desc}, the agent called '{tool}' multiple times as part of the process.",
            "During {desc}, the agent called '{wrong}' multiple times as part of the process.",
        ),
    }
    pos_t, neg_t = TMPL.get(rel_name, TMPL["idempotent"])
    geo = "categorical"

    true_tools  = [t for t, v in bool_dict.items() if v]
    false_tools = [t for t, v in bool_dict.items() if not v]
    if not true_tools or not false_tools:
        return []

    records, seen = [], set()
    for tool in true_tools:
        for _ in range(n_per_tool):
            desc = rng.choice(descs)[:120]
            wrong = rng.choice(false_tools)
            key = (rel_name, tool, desc[:20])
            if key in seen: continue
            seen.add(key)
            # Positive: true_tool이 실제 해당 속성을 가짐
            records.append(record(rel_name, geo, 1, tool, None,
                                  pos_t.format(desc=desc, tool=tool, wrong=wrong), rng))
            # Negative: false_tool을 해당 속성을 가지는 것처럼 잘못 제시
            records.append(record(rel_name, geo, 0, wrong, None,
                                  neg_t.format(desc=desc, tool=tool, wrong=wrong), rng))
    rng.shuffle(records)
    return records


def make_checkpoint_pairs(
    descs: List[str], rng: random.Random, n_per_tool: int,
) -> List[dict]:
    non_checkpoints = [t for t in ALL_TOOLS if t not in CHECKPOINT]
    records, seen = [], set()
    for tool in CHECKPOINT:
        for _ in range(n_per_tool):
            desc = rng.choice(descs)[:120]
            wrong = rng.choice(non_checkpoints)
            key = ("checkpoint", tool, desc[:20])
            if key in seen: continue
            seen.add(key)
            pos_txt = (f"For {desc}: '{tool}' must succeed before any subsequent tool can proceed.")
            neg_txt = (f"For {desc}: '{wrong}' must succeed before any subsequent tool can proceed.")
            records.append(record("checkpoint", "categorical", 1, tool, None, pos_txt, rng))
            records.append(record("checkpoint", "categorical", 0, wrong, None, neg_txt, rng))
    rng.shuffle(records)
    return records


def make_effect_state_pairs(
    descs: List[str], rng: random.Random, n_per_tool: int,
) -> List[dict]:
    records, seen = [], set()
    for tool, effects in EFFECT_STATE.items():
        pos_states  = effects.get("enables", [])
        neg_states  = effects.get("disables", [])
        if not pos_states: continue
        # Collect states belonging to OTHER tools
        other_states = []
        for t2, eff2 in EFFECT_STATE.items():
            if t2 != tool:
                other_states.extend(eff2.get("enables", []))
        if not other_states: continue

        for _ in range(n_per_tool):
            desc  = rng.choice(descs)[:120]
            state = rng.choice(pos_states)
            wrong_state = rng.choice(other_states)
            key = ("effect_state", tool, desc[:20])
            if key in seen: continue
            seen.add(key)
            pos_txt = (f"Calling '{tool}' for {desc} results in the state: [{state}].")
            neg_txt = (f"Calling '{tool}' for {desc} results in the state: [{wrong_state}].")
            records.append(record("effect_state", "categorical", 1, tool, None, pos_txt, rng))
            records.append(record("effect_state", "categorical", 0, tool, None, neg_txt, rng))
    rng.shuffle(records)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 G: GoT/ToT/Harness 전용 생성 함수 ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def make_fan_out_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    """FAN_OUT: source → {successors} 분기. 단일 진행(positive) vs 단일 선택(negative)."""
    records, seen = [], set()
    for entry in FAN_OUT:
        src   = entry.source
        succs = entry.successors
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key = ("fan_out", src, desc[:25])
            if key in seen: continue
            seen.add(key)
            succ_str   = ", ".join(f"[{s}]" for s in succs)
            wrong_next = wrong_tool({src, *succs}, rng)
            pos_txt = (f"For {desc}: calling [{src}] opened multiple candidate next steps: "
                       f"{succ_str}.")
            neg_txt = (f"For {desc}: calling [{src}] led to a single next step [{wrong_next}], "
                       f"without generating further branches.")
            records.append(record("fan_out", "directional", 1, src, succs[0], pos_txt, rng))
            records.append(record("fan_out", "directional", 0, src, wrong_next, neg_txt, rng))
    rng.shuffle(records)
    return records


def make_scored_preference_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    """SCORED_PREFERENCE: 동일 맥락에서 두 유효 도구 중 preferred가 높은 점수."""
    records, seen = [], set()
    for entry in SCORED_PREFERENCE:
        pref, alt, ctx = entry.preferred, entry.alternative, entry.context
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key  = ("scored_preference", pref, alt, desc[:25])
            if key in seen: continue
            seen.add(key)
            pos_txt = (f"In context [{ctx}] for {desc}: both [{pref}] and [{alt}] were valid, "
                       f"but the agent scored [{pref}] higher and chose it.")
            neg_txt = (f"In context [{ctx}] for {desc}: both [{pref}] and [{alt}] were valid, "
                       f"but the agent scored [{alt}] higher and chose it instead.")
            records.append(record("scored_preference", "conditional", 1, pref, alt, pos_txt, rng))
            records.append(record("scored_preference", "conditional", 0, alt, pref, neg_txt, rng))
    rng.shuffle(records)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 H: HTN 계층 전용 생성 함수 ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def make_decomposes_into_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    """DECOMPOSES_INTO: goal → {tools}. 올바른 분해(positive) vs 잘못된 첫 단계(negative)."""
    records, seen = [], set()
    for entry in DECOMPOSES_INTO:
        goal  = entry.goal
        tools = entry.tools
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key  = ("decomposes_into", goal, desc[:25])
            if key in seen: continue
            seen.add(key)
            tools_str  = ", ".join(f"[{t}]" for t in tools)
            wrong_first = wrong_tool(set(tools), rng)
            pos_txt = (f"The diagnostic goal '{goal}' for {desc} was decomposed into: "
                       f"{tools_str}.")
            neg_txt = (f"The diagnostic goal '{goal}' for {desc} was decomposed into: "
                       f"[{wrong_first}]" +
                       (f", {', '.join(f'[{t}]' for t in tools[1:])}." if len(tools) > 1 else "."))
            records.append(record("decomposes_into", "directional", 1, goal, tools[0], pos_txt, rng))
            records.append(record("decomposes_into", "directional", 0, goal, wrong_first, neg_txt, rng))
    rng.shuffle(records)
    return records


def make_refines_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    """REFINES: abstract_action + context → concrete_tool. 올바른 구체화 vs 잘못된 구체화."""
    from collections import defaultdict
    by_abstract: Dict[str, List] = defaultdict(list)
    for entry in REFINES:
        by_abstract[entry.abstract_action].append(entry.concrete_tool)

    records, seen = [], set()
    for entry in REFINES:
        abstract, concrete, ctx = entry.abstract_action, entry.concrete_tool, entry.context
        # 잘못된 구체화: 같은 추상 행동의 다른 맥락 도구, 없으면 임의
        alternatives = [t for t in by_abstract[abstract] if t != concrete]
        wrong_concrete = rng.choice(alternatives) if alternatives else wrong_tool({concrete}, rng)
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key  = ("refines", abstract, concrete, desc[:25])
            if key in seen: continue
            seen.add(key)
            pos_txt = (f"For {desc} in context [{ctx}], the abstract action '{abstract}' "
                       f"was refined to the concrete step [{concrete}].")
            neg_txt = (f"For {desc} in context [{ctx}], the abstract action '{abstract}' "
                       f"was refined to the concrete step [{wrong_concrete}].")
            records.append(record("refines", "conditional", 1, abstract, concrete, pos_txt, rng))
            records.append(record("refines", "conditional", 0, abstract, wrong_concrete, neg_txt, rng))
    rng.shuffle(records)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# ── 그룹 I: GoalAct 실제 구조 전용 생성 함수 ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def make_plan_revised_to_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    """PLAN_REVISED_TO: GoalAct G_t = π(Q|T|S_t). 관찰로 인한 plan step 교체."""
    from collections import defaultdict
    step_replacements: Dict[str, set] = defaultdict(set)
    for e in PLAN_REVISED_TO:
        step_replacements[e.old_step].add(e.new_step)

    records, seen = [], set()
    for entry in PLAN_REVISED_TO:
        trigger, old_step, new_step = (
            entry.trigger_observation, entry.old_step, entry.new_step
        )
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key  = ("plan_revised_to", trigger, old_step, desc[:25])
            if key in seen: continue
            seen.add(key)
            wrong_new = wrong_plan_step(step_replacements[old_step] | {old_step}, rng)
            pos_txt = (f"After observing [{trigger}] during {desc}, the GoalAct global plan "
                       f"revised step '{old_step}' to '{new_step}'.")
            neg_txt = (f"After observing [{trigger}] during {desc}, the GoalAct global plan "
                       f"revised step '{old_step}' to '{wrong_new}'.")
            records.append(record("plan_revised_to", "conditional", 1,
                                  old_step, new_step, pos_txt, rng))
            records.append(record("plan_revised_to", "conditional", 0,
                                  old_step, wrong_new, neg_txt, rng))
    rng.shuffle(records)
    return records


def make_plan_committed_to_goal_pairs(
    descs: List[str], rng: random.Random, n_per_entry: int,
) -> List[dict]:
    """PLAN_COMMITTED_TO_GOAL: plan step이 goal에 commit. wrong = wrong_goal."""
    from collections import defaultdict
    step_goals: Dict[str, set] = defaultdict(set)
    for step, goal in PLAN_COMMITTED_TO_GOAL:
        step_goals[step].add(goal)

    records, seen = [], set()
    for step, goal in PLAN_COMMITTED_TO_GOAL:
        for _ in range(n_per_entry):
            desc = rng.choice(descs)[:150]
            key  = ("plan_committed_to_goal", step, goal, desc[:25])
            if key in seen: continue
            seen.add(key)
            wrong_g = wrong_goal(step_goals[step] | {step}, rng)
            pos_txt = (f"Throughout {desc}, plan step '{step}' remained committed to "
                       f"goal [{goal}].")
            neg_txt = (f"Throughout {desc}, plan step '{step}' was mistakenly committed to "
                       f"goal [{wrong_g}].")
            records.append(record("plan_committed_to_goal", "directional", 1,
                                  step, goal, pos_txt, rng))
            records.append(record("plan_committed_to_goal", "directional", 0,
                                  step, wrong_g, neg_txt, rng))
    rng.shuffle(records)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau2", default=TAU2_TELECOM)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-binary", type=int, default=12,
                   help="이진 관계 항목당 (pos, neg) 쌍 수")
    p.add_argument("--n-unary", type=int, default=20,
                   help="단항/속성 관계 도구당 쌍 수")
    p.add_argument("--out", default=(
        "/home/woori/workspace_common/boltzmann-attention-pi/"
        "reports/facet_rft_2026/phase0_probing/contrast_pairs_v3.json"
    ))
    args = p.parse_args()

    rng = random.Random(args.seed)

    with open(args.tau2) as f:
        tasks = json.load(f)
    descs = get_descriptions(tasks)
    print(f"Loaded {len(tasks)} tasks → {len(descs)} descriptions")

    all_pairs: List[dict] = []
    N = args.n_binary
    U = args.n_unary

    def add(label, pairs):
        all_pairs.extend(pairs)
        geo = RELATION_GEOMETRY.get(label, "?")
        pred = PREDICTED_METHOD.get(label, "?")
        print(f"  {label:22s}: {len(pairs):5d} pairs  [{geo:12s} → {pred}]")

    print("\n── 그룹 A/B/C: Directional (A6 후보) ──")
    for rel, data in [
        ("precedes",         PRECEDES),
        ("directly_follows", DIRECTLY_FOLLOWS),
        ("causal_link",      CAUSAL_LINK),
        ("requires",         REQUIRES),
        ("enables",          ENABLES),
        ("parameter_feeds",  PARAMETER_FEEDS),
        ("validates",        VALIDATES),
        ("retry_after_fail", RETRY_AFTER_FAIL),
        ("error_fallback",   ERROR_FALLBACK),
        ("compensates",      COMPENSATES),
        ("tool_subsumes",    TOOL_SUBSUMES),
    ]:
        add(rel, make_directional_pairs(rel, data, descs, rng, N))
    add("and_join",         make_and_join_pairs(descs, rng, N))
    add("state_transition", make_state_transition_pairs(descs, rng, N))

    print("\n── 그룹 D: Symmetric/Conditional (T1 후보) ──")
    for rel, data in [
        ("mutex",           MUTEX),
        ("exclusive_choice",EXCLUSIVE_CHOICE),
        ("parallel_safe",   PARALLEL_SAFE),
        ("conditional_on",  CONDITIONAL_ON),
    ]:
        add(rel, make_symmetric_pairs(rel, data, descs, rng, N))

    print("\n── 그룹 E: 상태/효과 (T1 후보) ──")
    add("precondition_state", make_dict_str_pairs("precondition_state", PRECONDITION_STATE, descs, rng, U))
    add("effect_state",       make_effect_state_pairs(descs, rng, U))

    print("\n── 그룹 F: 도구 속성 (T1 후보) ──")
    add("workflow_role",    make_dict_str_pairs("workflow_role", WORKFLOW_ROLE, descs, rng, U))
    add("domain_category",  make_dict_str_pairs("domain_category", DOMAIN_CATEGORY, descs, rng, U))
    add("checkpoint",       make_checkpoint_pairs(descs, rng, U))
    add("idempotent",       make_bool_property_pairs("idempotent", IDEMPOTENT, descs, rng, U))
    add("reversible",       make_bool_property_pairs("reversible", REVERSIBLE, descs, rng, U))
    add("mandatory_in_flow",make_bool_property_pairs("mandatory_in_flow", MANDATORY_IN_FLOW, descs, rng, U))
    add("optional_in_flow", make_bool_property_pairs("optional_in_flow", OPTIONAL_IN_FLOW, descs, rng, U))
    add("loop_capable",     make_bool_property_pairs("loop_capable", LOOP_CAPABLE, descs, rng, U))

    print("\n── 그룹 G: GoT/ToT/Harness (혼합) ──")
    for rel, data in [
        ("backtrack_to", BACKTRACK_TO),
        ("pruned_by",    PRUNED_BY),
    ]:
        add(rel, make_directional_pairs(rel, data, descs, rng, N))
    add("fan_out",           make_fan_out_pairs(descs, rng, N))
    add("observation_triggers", make_dict_str_pairs("observation_triggers", OBSERVATION_TRIGGERS, descs, rng, U))
    add("guardrail",         make_dict_str_pairs("guardrail", GUARDRAIL, descs, rng, U))
    add("scored_preference", make_scored_preference_pairs(descs, rng, N))

    print("\n── 그룹 H: HTN 계층 (혼합) ──")
    add("decomposes_into", make_decomposes_into_pairs(descs, rng, N))
    add("subtask_of",      make_dict_str_pairs("subtask_of", SUBTASK_OF, descs, rng, U))
    add("achieves_goal",   make_dict_str_pairs("achieves_goal", ACHIEVES_GOAL, descs, rng, U))
    add("refines",         make_refines_pairs(descs, rng, N))

    print("\n── 그룹 I: GoalAct 실제 구조 G_t=π(Q|T|S_t) (혼합) ──")
    for rel, data in [
        ("plan_step_precedes", PLAN_STEP_PRECEDES),
        ("step_realizes_tool", STEP_REALIZES_TOOL),
    ]:
        add(rel, make_directional_pairs(rel, data, descs, rng, N))
    add("plan_step_skill",        make_dict_str_pairs("plan_step_skill", PLAN_STEP_SKILL, descs, rng, U))
    add("plan_revised_to",        make_plan_revised_to_pairs(descs, rng, N))
    add("plan_committed_to_goal", make_plan_committed_to_goal_pairs(descs, rng, N))

    rng.shuffle(all_pairs)

    by_type = {}
    for pp in all_pairs:
        by_type[pp["pair_type"]] = by_type.get(pp["pair_type"], 0) + 1

    a6_total = sum(v for k, v in by_type.items() if PREDICTED_METHOD.get(k) == "A6")
    t1_total = sum(v for k, v in by_type.items() if PREDICTED_METHOD.get(k) == "T1")

    n_pos = sum(1 for x in all_pairs if x["label"] == 1)
    n_neg = sum(1 for x in all_pairs if x["label"] == 0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "metadata": {
            "version": "v5",
            "description": "42-relation ontology contrast pairs (27 base + 6 GoT/ToT/Harness + 4 HTN + 5 Goal-lifecycle)",
            "n_total": len(all_pairs),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_relations": len(by_type),
            "by_type": by_type,
            "geometry_split": {"A6_directional": a6_total, "T1_symmetric_categorical": t1_total},
            "seed": args.seed,
        },
        "pairs": all_pairs,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n총 {len(all_pairs)} pairs ({n_pos} pos / {n_neg} neg)")
    print(f"  A6 후보: {a6_total}  T1 후보: {t1_total}")
    print(f"Saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
