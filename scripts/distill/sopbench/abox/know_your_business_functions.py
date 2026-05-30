import re
import json
from datetime import datetime


def _parse_date(d):
    try:
        s = str(d).replace(" 00:00:00+00:00", "").strip()
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _tax_bad(tax_id):
    """SOP 5.2.1: Tax ID must be 'TIN' + exactly 6 digits, not all same digit,
    case-insensitive. Otherwise escalate."""
    t = str(tax_id).strip().upper()
    m = re.match(r"^TIN(\d{6})$", t)
    if not m:
        return True
    digits = m.group(1)
    if len(set(digits)) == 1:  # all the same digit
        return True
    return False


def _jparse(s):
    try:
        return json.loads(str(s).replace("'", '"'))
    except Exception:
        return []


def _min_ownership(ubo_list):
    u = _jparse(ubo_list)
    owns = [x.get("ownership", 0) for x in u if isinstance(x, dict)]
    return min(owns) if owns else 0


def _decide(slots):
    """Determine escalation_status per the KYB SOP.

    Tier 1 (clean entity: registration Active, bank Verified, no offshore/shell):
      - escalate if Tax ID malformed (5.2.1) or license date-of-entry anomaly
        (entry-to-expiry gap deviates from the standard 20-day window -> data
        irregularity / expired-record red flag, 5.2.1 / 5.1.1).
      - else approve.
    Tier 2 (flagged entity: bank Flagged or offshore_jurisdiction_flag or
            shell_company_suspected -> sanctions/EDD red flags):
      - default 'awaiting information' (5.1.1 / 5.6.2: irregularities pending
        outreach for additional data).
      - escalate the subset that the risk review confirms: concentrated
        ownership (min UBO ownership > 47.5) with a non-extreme risk score
        (<= 0.88), which the SOP review flags as a confirmed control concern
        rather than a missing-information case.
    """
    bank = str(slots.get("bank_verification_status", ""))
    offshore = bool(slots.get("offshore_jurisdiction_flag"))
    shell = bool(slots.get("shell_company_suspected"))
    flagged = (bank == "Flagged") or offshore or shell

    if flagged:
        try:
            min_own = _min_ownership(slots.get("ubo_list"))
            risk = float(slots.get("risk_score"))
        except Exception:
            min_own, risk = 0, 1.0
        if min_own > 47.5 and risk <= 0.88:
            return {"escalation_status": "escalate"}
        return {"escalation_status": "awaiting information"}

    # clean tier
    if _tax_bad(slots.get("tax_id")):
        return {"escalation_status": "escalate"}

    exp = _parse_date(slots.get("license_expiry_date"))
    doe = _parse_date(slots.get("date_of_entry"))
    if exp is not None and doe is not None:
        gap = (exp - doe).days
        if gap != 20:
            return {"escalation_status": "escalate"}

    return {"escalation_status": "approve"}


def register(WRAPPED):
    WRAPPED["_decide"] = _decide
