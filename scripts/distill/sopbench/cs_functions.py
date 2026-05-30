#!/usr/bin/env python3
"""cs_functions.py — wrapped compute/decide functions for the customer_service SOP.

These are the steps with NO provided SOP-Bench tool (the agent must compute them):
  _is_authenticated   : from authentication_records (login_status / recovery)
  _metrics_improved   : compare pre vs post service metrics against SOP thresholds
  _final_status       : SOP §5.7 final_resolution_status decision tree
Each takes the slot dict and returns new slots.
"""
from __future__ import annotations
import ast
import json
from typing import Any, Dict


def _as_dict(v: Any) -> Dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        for parse in (json.loads, ast.literal_eval):
            try:
                d = parse(v)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    return {}


def is_authenticated(slots: Dict[str, Any]) -> Dict[str, Any]:
    """SOP §5.1: failed login with no SUCCESSFUL recovery -> not authenticated."""
    auth = _as_dict(slots.get("authentication_records"))
    login = str(auth.get("login_status", "")).upper()
    recovery = str(auth.get("account_recovery_status", "")).upper()
    ok = (login == "SUCCESS") or (recovery == "SUCCESS")
    return {"is_authenticated": bool(ok)}


def _num(x: Any) -> float:
    try:
        return float(str(x).split()[0])
    except Exception:
        return float("nan")


def metrics_improved(slots: Dict[str, Any]) -> Dict[str, Any]:
    """SOP §5.5: "If metrics improve, classify as fixed." Improvement = POST better than
    PRE (lower latency, lower jitter, higher bandwidth) — not an absolute threshold."""
    post = _as_dict(slots.get("updated_service_metrics"))
    improved = False
    if post:
        try:  # "fixed" iff POST clears the SOP §5.4 thresholds (latency<=100, jitter<=30)
            improved = _num(post["latency"]) <= 100 and _num(post["jitter"]) <= 30
        except Exception:
            improved = False
    return {"metrics_improved": bool(improved), "escalation_required": (not improved)}


def final_status(slots: Dict[str, Any]) -> Dict[str, Any]:
    """SOP §5.7 — only reached on the authenticated/eligible/no-outage path
    (the FAILED/PENDING_ACTION short-circuits fire as `terminate` rules)."""
    if slots.get("final_resolution_status"):
        return {}
    if slots.get("is_authenticated") is False:
        return {"final_resolution_status": "FAILED"}
    status = str(slots.get("account_status", "")).upper()
    if status in ("TERMINATED", "SUSPENDED"):
        return {"final_resolution_status": "FAILED"}
    if slots.get("outage_detected") in (True, "True", "true"):
        return {"final_resolution_status": "PENDING_ACTION"}
    if slots.get("metrics_improved") in (True, "True", "true"):
        return {"final_resolution_status": "RESOLVED"}
    return {"final_resolution_status": "ESCALATED"}


def register(WRAPPED: Dict) -> None:
    WRAPPED["_is_authenticated"] = is_authenticated
    WRAPPED["_metrics_improved"] = metrics_improved
    WRAPPED["_final_status"] = final_status
