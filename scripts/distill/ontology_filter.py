"""ontology_filter.py — count tau2 workflow-ontology violations in a tool sequence.

Used by build_sft_dataset.py (facet variant) to keep only "ontology-clean"
trajectories.  The per-domain ontologies under scripts/ontology/ are pure data
modules (PRECEDES / REQUIRES / MUTEX / GUARDRAIL / ... , same names across
domains); this module is the checker on top of them.

GRADED STRICTNESS (--onto-level) — these are ALREADY env-validated successful
trajectories, so over-filtering on inferred relations discards correct behaviour.
The levels are a knob from conservative to aggressive so the distillation campaign
can sweep filter strictness as an experimental variable:

  L1  conservative — only violations unambiguous from the sequence:
        mutex_coexist     MUTEX(A,B): both A and B appear
        precedes_order    PRECEDES(A,B): both appear, B's 1st call before A's
        requires_order    REQUIRES(A,B): both appear, prereq B's 1st call AFTER A's
                          (omitting B entirely is NOT a violation at L1)
        guardrail_first   GUARDRAIL tool with "is_first_action" cond is the 1st action
  L2  moderate — L1 plus structural-completeness violations:
        requires_missing  REQUIRES(A,B): A present but prereq B entirely absent
        exclusive_both    EXCLUSIVE_CHOICE(cond,a,b): both a and b appear
  L3  aggressive — L2 plus efficiency/contradiction violations:
        compensates_both  COMPENSATES(A,B): both appear (A undoes B's effect)
        repeat_misuse     a non-LOOP_CAPABLE, non-idempotent tool called >1 time

GUARDRAILs whose condition is a DB/state predicate (e.g. "no_outstanding_balance")
are not sequence-checkable and are skipped at every level.  Relations a given domain
ontology doesn't define are simply skipped (getattr default), so the same checker
works for telecom / retail / airline.
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

_ONT_CACHE: dict = {}


def _load_ont(domain: str, ont_dir: str):
    key = (domain, ont_dir)
    if key in _ONT_CACHE:
        return _ONT_CACHE[key]
    path = Path(ont_dir) / f"tau2_{domain}_ontology.py"
    spec = importlib.util.spec_from_file_location(f"tau2_{domain}_ontology", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ONT_CACHE[key] = mod
    return mod


def count_ontology_violations(domain: str, tool_seq: list[str], ont_dir: str, level: int = 1):
    """Return (total:int, breakdown:dict, detail:list).

    tool_seq: ordered agent tool-call names for one trajectory.
    level: 1=conservative, 2=moderate, 3=aggressive (cumulative).
    """
    ont = _load_ont(domain, ont_dir)
    counts = Counter()
    detail = []

    present = set(tool_seq)
    first: dict[str, int] = {}
    for i, t in enumerate(tool_seq):
        first.setdefault(t, i)

    # ── L1 (conservative) ──────────────────────────────────────────────────
    # MUTEX — both present (symmetric co-existence violation)
    for a, b in getattr(ont, "MUTEX", []):
        if a in present and b in present:
            counts["mutex_coexist"] += 1
            detail.append(("mutex_coexist", a, b))

    # PRECEDES(A,B) — violation if B's first call precedes A's (both present)
    for a, b in getattr(ont, "PRECEDES", []):
        if a in present and b in present and first[b] < first[a]:
            counts["precedes_order"] += 1
            detail.append(("precedes_order", a, b))

    # REQUIRES(A,B) — both present but prerequisite B called after A (out-of-order)
    for a, b in getattr(ont, "REQUIRES", []):
        if a in present and b in present and first[b] > first[a]:
            counts["requires_order"] += 1
            detail.append(("requires_order", a, b))

    # GUARDRAIL — first-action prohibition
    guard = getattr(ont, "GUARDRAIL", {})
    if tool_seq and "is_first_action" in guard.get(tool_seq[0], ""):
        counts["guardrail_first"] += 1
        detail.append(("guardrail_first", tool_seq[0], guard.get(tool_seq[0])))

    # ── L2 (moderate) ──────────────────────────────────────────────────────
    if level >= 2:
        # REQUIRES(A,B) — A present but prerequisite B entirely absent
        for a, b in getattr(ont, "REQUIRES", []):
            if a in present and b not in present:
                counts["requires_missing"] += 1
                detail.append(("requires_missing", a, b))
        # EXCLUSIVE_CHOICE(cond, a, b) — both options taken
        for ec in getattr(ont, "EXCLUSIVE_CHOICE", []):
            a, b = ec.option_a, ec.option_b
            if a in present and b in present:
                counts["exclusive_both"] += 1
                detail.append(("exclusive_both", a, b))

    # ── L3 (aggressive) ────────────────────────────────────────────────────
    if level >= 3:
        # COMPENSATES(A,B) — both present (A reverses B's effect)
        for a, b in getattr(ont, "COMPENSATES", []):
            if a in present and b in present:
                counts["compensates_both"] += 1
                detail.append(("compensates_both", a, b))
        # repeat misuse — non-loop-capable, non-idempotent tool called >1 time
        loop_cap = getattr(ont, "LOOP_CAPABLE", {})
        idem = getattr(ont, "IDEMPOTENT", {})
        seq_counts = Counter(tool_seq)
        for t, c in seq_counts.items():
            if c > 1 and loop_cap.get(t) is False and idem.get(t) is False:
                counts["repeat_misuse"] += 1
                detail.append(("repeat_misuse", t, c))

    return sum(counts.values()), dict(counts), detail


def agent_tool_sequence(conv_messages: list[dict]) -> list[str]:
    """Extract the ordered agent tool-call names from converted chat messages."""
    seq = []
    for m in conv_messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name")
                if name:
                    seq.append(name)
    return seq
