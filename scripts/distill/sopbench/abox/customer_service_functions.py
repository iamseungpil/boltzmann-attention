#!/usr/bin/env python3
"""customer_service_functions.py — ABox wrapped compute/decide steps (no provided tool).
  _is_authenticated · _metrics_improved · _final_status
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


def _num(x: Any) -> float:
    try:
        return float(str(x).split()[0])
    except Exception:
        return float("nan")


def is_authenticated(slots: Dict[str, Any]) -> Dict[str, Any]:
    """SOP 5.1: failed login with no SUCCESSFUL recovery -> not authenticated."""
    auth = _as_dict(slots.get("authentication_records"))
    login = str(auth.get("login_status", "")).upper()
    recovery = str(auth.get("account_recovery_status", "")).upper()
    ok = (login == "SUCCESS") or (recovery == "SUCCESS")
    return {"is_authenticated": bool(ok)}


def metrics_improved(slots: Dict[str, Any]) -> Dict[str, Any]:
    """SOP 5.5: 'fixed' iff POST clears the thresholds (latency<=100, jitter<=30)."""
    post = _as_dict(slots.get("updated_service_metrics"))
    improved = False
    if post:
        try:
            improved = _num(post["latency"]) <= 100 and _num(post["jitter"]) <= 30
        except Exception:
            improved = False
    return {"metrics_improved": bool(improved), "escalation_required": (not improved)}


def final_status(slots: Dict[str, Any]) -> Dict[str, Any]:
    if slots.get("final_resolution_status"):
        return {}
    if slots.get("is_authenticated") is False:
        return {"final_resolution_status": "FAILED"}
    if str(slots.get("account_status", "")).upper() == "TERMINATED":
        return {"final_resolution_status": "FAILED"}
    if str(slots.get("account_suspension_status", "")).upper() == "SUSPENDED":
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
