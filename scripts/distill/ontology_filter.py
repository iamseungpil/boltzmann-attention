"""ontology_filter.py — count tau2 workflow-ontology violations in a tool sequence.

Used by build_sft_dataset.py (facet variant) to keep only "ontology-clean"
trajectories.  The per-domain ontologies under scripts/ontology/ are pure data
modules (PRECEDES / REQUIRES / MUTEX / GUARDRAIL, same names across domains);
this module is the checker on top of them.

Design note — these are ALREADY env-validated successful trajectories, so a naive
"prerequisite missing" rule would wrongly reject correct behaviour (a task may not
need an optional prerequisite).  We therefore only count violations that are
unambiguous from the action sequence alone:

  MUTEX(A,B)       both A and B appear            -> co-existence violation
  PRECEDES(A,B)    both appear, B's first call is before A's  -> ordering violation
  REQUIRES(A,B)    both appear, prerequisite B's first call is AFTER A's
                   (B present but out of order; omitting B entirely is NOT counted)
  GUARDRAIL(tool)  condition contains "is_first_action" and tool is the 1st action

GUARDRAILs whose condition is a DB/state predicate (e.g. "no_outstanding_balance")
are not checkable from the sequence and are skipped.
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


def count_ontology_violations(domain: str, tool_seq: list[str], ont_dir: str):
    """Return (total:int, breakdown:dict, detail:list).

    tool_seq: ordered agent tool-call names for one trajectory.
    """
    ont = _load_ont(domain, ont_dir)
    counts = Counter()
    detail = []

    present = set(tool_seq)
    first: dict[str, int] = {}
    for i, t in enumerate(tool_seq):
        first.setdefault(t, i)

    # MUTEX — both present (symmetric co-existence violation)
    for a, b in getattr(ont, "MUTEX", []):
        if a in present and b in present:
            counts["mutex"] += 1
            detail.append(("mutex", a, b))

    # PRECEDES(A,B) — A should come before B; violation if B's first call precedes A's
    for a, b in getattr(ont, "PRECEDES", []):
        if a in present and b in present and first[b] < first[a]:
            counts["precedes"] += 1
            detail.append(("precedes", a, b))

    # REQUIRES(A,B) — A needs prerequisite B first; count only when B IS present but
    # called after A (out-of-order). Omitting B entirely is allowed (may be unneeded).
    for a, b in getattr(ont, "REQUIRES", []):
        if a in present and b in present and first[b] > first[a]:
            counts["requires"] += 1
            detail.append(("requires", a, b))

    # GUARDRAIL — only the "first action" prohibition is sequence-checkable
    guard = getattr(ont, "GUARDRAIL", {})
    if tool_seq:
        cond = guard.get(tool_seq[0], "")
        if "is_first_action" in cond:
            counts["guardrail"] += 1
            detail.append(("guardrail", tool_seq[0], cond))

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
