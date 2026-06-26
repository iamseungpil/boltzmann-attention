#!/usr/bin/env python3
"""ontology_resolver.py — DETERMINISTIC executor for the layered agent (design §15.11).

The hierarchical planner emits an abstract PLAN_STEP
(gather_account_context / check_system_status / apply_targeted_fix /
 apply_policy_action / escalate_or_document). This module turns that step + the
OBSERVED state (accumulated read-tool outputs) into a CONCRETE (tool, args) with
NO LLM and NO training, using three induced, domain-swappable ontology maps:

  step_realization_<dom>.json : step_realizes_tool {tool: step}
        -> inverse gives the CANDIDATE concrete tools for an abstract step
  obs_triggers_<dom>.json     : observation_triggers {tool: {triggers:[{pred:"read|field=value",
                                                                         precision, support, recall}]}}
                                arg_source           {tool: {param: {source:"read.field", frac}}}

  resolve(step, state) = inverse(step) candidates
                          x observation_triggers   (pick the candidate whose state-predicate matches)
                          x arg_source             (fill the chosen tool's args from prior reads)

Returns a ResolveResult, or None on a miss (the caller then falls back to a
candidate-restricted LLM). The maps swap per domain (transfer == ABox swap); the
planner and the PLAN_STEP vocabulary stay fixed (TBox). This is the executor that,
per §15.11, works deterministically on telecom-style state-machine domains
(roaming_enabled=False -> enable_roaming, status=Paid -> resume_line, ...) and
degrades to LLM fallback on retail/airline catalogue-selection domains.

Key consistency requirement: the OBSERVED-state predicate strings must be built the
SAME way the inducer built them (induce_observation_triggers.py: flatten + _scalar),
or matches will silently fail. flatten()/_scalar() below mirror that script exactly.

Self-test:  python scripts/distill/ontology_resolver.py --domain telecom
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_INDUCED_DIR = "reports/facet_rft_2026/phase4_distill/induced"

# ---------------------------------------------------------------------------
# flatten / _scalar — MUST match induce_observation_triggers.py in behaviour so
# that runtime predicate keys line up with the induced pred strings.
# ---------------------------------------------------------------------------

def _scalar(v: Any) -> Optional[str]:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float, str)):
        s = str(v)
        return s if len(s) <= 40 else None
    return None


def flatten(obj: Any, prefix: str = "", out: Optional[dict] = None, depth: int = 0) -> dict:
    """Flatten dict/list to {dotted_key: scalar_str}, scalars only, depth<=3.

    Mirrors induce_observation_triggers.flatten (lists truncated to first 5,
    leaves stringified via _scalar). Used for pred matching.
    """
    if out is None:
        out = {}
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                flatten(v, f"{prefix}{k}.", out, depth + 1)
            else:
                out[f"{prefix}{k}"] = _scalar(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            if isinstance(v, (dict, list)):
                flatten(v, f"{prefix}{i}.", out, depth + 1)
            else:
                out[f"{prefix}{i}"] = _scalar(v)
    return out


def flatten_raw(obj: Any, prefix: str = "", out: Optional[dict] = None, depth: int = 0) -> dict:
    """Like flatten() but keeps RAW leaf values (preserve int/float/bool for arg filling).

    Same traversal/keys as flatten(); only the leaf value differs (raw vs stringified).
    """
    if out is None:
        out = {}
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                flatten_raw(v, f"{prefix}{k}.", out, depth + 1)
            else:
                out[f"{prefix}{k}"] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            if isinstance(v, (dict, list)):
                flatten_raw(v, f"{prefix}{i}.", out, depth + 1)
            else:
                out[f"{prefix}{i}"] = v
    return out


# ---------------------------------------------------------------------------
# Observed state: accumulated read-tool outputs over a rollout.
# ---------------------------------------------------------------------------

class ObservedState:
    """Accumulates flattened outputs of READ tool calls seen so far in a rollout.

    by_pred:   "<read_tool>|<field>" -> stringified value   (for trigger matching)
    by_source: "<read_tool>.<field>" -> raw value           (for arg filling)
    Latest write wins (a re-read overwrites).
    """

    def __init__(self) -> None:
        self.by_pred: dict[str, Optional[str]] = {}
        self.by_source: dict[str, Any] = {}
        self.reads_seen: list[str] = []

    def update(self, read_tool: str, content: Any) -> None:
        """Ingest one read tool result. `content` may be a JSON string or a dict/list."""
        d = content
        if isinstance(content, str):
            try:
                d = json.loads(content)
            except Exception:
                return
        if d is None:
            return
        self.reads_seen.append(read_tool)
        for k, v in flatten(d).items():
            if v is not None:
                self.by_pred[f"{read_tool}|{k}"] = v
        for k, v in flatten_raw(d).items():
            self.by_source[f"{read_tool}.{k}"] = v

    def matches_pred(self, pred: str) -> bool:
        """pred == '<read_tool>|<field>=<value>'. True iff observed state carries it."""
        key, _, val = pred.rpartition("=")
        if not key:
            return False
        return self.by_pred.get(key) == val

    def lookup_source(self, source: str) -> tuple[bool, Any]:
        """source == '<read_tool>.<field>'. (found, raw_value)."""
        if source in self.by_source:
            return True, self.by_source[source]
        return False, None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

@dataclass
class ResolveResult:
    tool: str
    args: dict[str, Any]
    score: float                      # sum of precision*support over matched triggers
    matched_preds: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    missing_args: list[str] = field(default_factory=list)   # params with no provenance found
    decisive: bool = True            # False -> single-candidate / weak match, prefer LLM check


class OntologyResolver:
    """Deterministic step->tool->args resolver over the induced per-domain ontology."""

    def __init__(self, domain: str, induced_dir: str = DEFAULT_INDUCED_DIR,
                 min_score: float = 1.0):
        self.domain = domain
        self.min_score = min_score
        with open(os.path.join(induced_dir, f"step_realization_{domain}.json")) as f:
            sr = json.load(f)
        with open(os.path.join(induced_dir, f"obs_triggers_{domain}.json")) as f:
            ob = json.load(f)
        self.step_realizes_tool: dict[str, str] = sr.get("step_realizes_tool", {})
        self.candidates_by_step: dict[str, list[str]] = defaultdict(list)
        for tool, step in self.step_realizes_tool.items():
            self.candidates_by_step[step].append(tool)
        self.observation_triggers: dict[str, dict] = ob.get("observation_triggers", {})
        self.arg_source: dict[str, dict] = ob.get("arg_source", {})

    def _score_tool(self, tool: str, state: ObservedState) -> tuple[float, list[str]]:
        info = self.observation_triggers.get(tool)
        if not info:
            return 0.0, []
        score, matched = 0.0, []
        for trig in info.get("triggers", []):
            pred = trig.get("pred", "")
            if state.matches_pred(pred):
                score += float(trig.get("precision", 0.0)) * float(trig.get("support", 0))
                matched.append(pred)
        return score, matched

    def _fill_args(self, tool: str, state: ObservedState) -> tuple[dict, list]:
        args: dict[str, Any] = {}
        missing: list[str] = []
        for param, spec in self.arg_source.get(tool, {}).items():
            found, val = state.lookup_source(spec.get("source", ""))
            if found:
                args[param] = val
            else:
                missing.append(param)
        return args, missing

    def resolve(self, plan_step: str, state: ObservedState) -> Optional[ResolveResult]:
        """Deterministic (tool, args) for `plan_step` given observed state, or None on a miss.

          1. candidates = tools realizing this abstract step.
          2. score each by observation-trigger matches against state; pick argmax.
          3. best score >= min_score -> decisive; fill args from arg_source.
          4. exactly one candidate -> return it (decisive=False, caller may LLM-verify).
          5. otherwise -> None (miss -> candidate-restricted LLM fallback).
        """
        candidates = self.candidates_by_step.get(plan_step, [])
        if not candidates:
            return None

        scored = [(self._score_tool(t, state), t) for t in candidates]
        scored.sort(key=lambda x: x[0][0], reverse=True)
        (best_score, best_matched), best_tool = scored[0]

        if best_score >= self.min_score:
            args, missing = self._fill_args(best_tool, state)
            return ResolveResult(best_tool, args, best_score, best_matched,
                                 candidates, missing, decisive=True)

        if len(candidates) == 1:
            tool = candidates[0]
            args, missing = self._fill_args(tool, state)
            return ResolveResult(tool, args, best_score, best_matched,
                                 candidates, missing, decisive=False)

        return None

    def candidates_for(self, plan_step: str) -> list[str]:
        return list(self.candidates_by_step.get(plan_step, []))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(domain: str, induced_dir: str) -> int:
    r = OntologyResolver(domain, induced_dir=induced_dir)
    print(f"[{domain}] steps -> candidates:")
    for step, cands in r.candidates_by_step.items():
        print(f"    {step:24} -> {cands}")
    print(f"[{domain}] tools with triggers: {list(r.observation_triggers)}")

    if domain == "telecom":
        st = ObservedState()
        st.update("get_details_by_id", {"roaming_enabled": False, "status": "Active"})
        st.update("get_customer_by_phone", {"customer_id": "C1001", "line_ids": ["L1", "L2"]})
        print("\n[roaming_enabled=False] ->", r.resolve("apply_targeted_fix", st))
        st2 = ObservedState()
        st2.update("get_details_by_id", {"status": "Paid", "total_due": 65.0})
        st2.update("get_customer_by_phone", {"customer_id": "C1001", "line_ids": ["L1", "L2"]})
        print("[status=Paid] ->", r.resolve("apply_targeted_fix", st2))
        print("[empty multi-candidate] ->", r.resolve("apply_targeted_fix", ObservedState()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom", choices=["telecom", "retail", "airline"])
    ap.add_argument("--induced-dir", default=DEFAULT_INDUCED_DIR)
    args = ap.parse_args()
    return _selftest(args.domain, args.induced_dir)


if __name__ == "__main__":
    raise SystemExit(main())
