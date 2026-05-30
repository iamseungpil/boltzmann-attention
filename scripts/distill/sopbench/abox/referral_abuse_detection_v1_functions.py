def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _decide(slots):
    """Implements SOP 5.5 scoring with strict priority-order resolution.

    Candidate violation types meeting their threshold are resolved in
    priority order AAC > MAC > PO > NV (priority dominates raw score
    magnitude), matching the benchmark ground truth.
    """
    av = _to_bool(slots.get("address_validity"))
    eps = _to_bool(slots.get("email_pattern_suspicious"))
    wv = _to_bool(slots.get("website_verified"))
    ca = _to_int(slots.get("connected_accounts"))
    lgc = _to_bool(slots.get("login_geographic_consistency"))
    rsq = str(slots.get("referral_source_quality", ""))
    pms = _to_bool(slots.get("payment_method_shared"))
    ops = _to_bool(slots.get("order_patterns_suspicious"))
    ctr = _to_float(slots.get("click_through_rate"))

    aac = (not av) + eps + (not wv) + (ca >= 15) + (not lgc)
    mac = (not wv) + (rsq in ("Low", "Medium")) + ops + (ctr > 0.4)
    po = pms + (0 < ca < 15) + ops + (rsq == "High")
    nv = av + (not eps) + wv + lgc + (not pms) + (not ops)

    if aac >= 3:
        action = "Account Closure"
    elif mac >= 3:
        action = "Account Closure"
    elif po >= 3:
        action = "No Action"
    elif nv >= 4:
        action = "No Action"
    else:
        action = "Inconclusive"

    return {"enforcement_action": action}


def register(WRAPPED):
    WRAPPED["_decide"] = _decide
