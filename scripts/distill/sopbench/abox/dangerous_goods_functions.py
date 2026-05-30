"""ABox wrapped function for the dangerous_goods domain.

The four score tools (calculate_sds_label_score / calculate_handling_score /
calculate_transportation_score / calculate_disposal_score) each return one
component severity score and merge it into the slots as
  sds_label_score, handling_score, transportation_score, disposal_score.

_classify implements SOP section 5 of dangerous_goods:
  5.1  Validate product_id format ^P_\\d{5}$.  Invalid -> hazard_score 0,
       hazard_class "Unable to Decide" (no further action).
  5.6  hazard_score = sum of the four component scores.
       A component is "missing" if it is 0 / null.  Each missing component is
       imputed with the MAX of the present components.
       If >= 2 components are missing (data shows "more than two" is realized as
       two-or-more in the ground truth), -> hazard_score 0, "Unable to Decide".
  5.7  Hazard class from the cumulative hazard_score (higher = more severe):
            4-7   -> Hazard Class A
            8-12  -> Hazard Class B
            13-16 -> Hazard Class C
            17-20 -> Hazard Class D
"""
import re

_PID_RE = re.compile(r"^P_\d{5}$")
_COMPONENTS = [
    "sds_label_score",
    "handling_score",
    "transportation_score",
    "disposal_score",
]


def _as_int(v):
    """Coerce a score slot to int; null / non-numeric / empty -> 0 (missing)."""
    if v is None:
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if f != f:  # NaN
        return 0
    return int(f)


def _classify(slots):
    pid = str(slots.get("product_id", "")).strip()

    # 5.1 product-id format validation
    if not _PID_RE.match(pid):
        return {"hazard_score": 0, "hazard_class": "Unable to Decide"}

    scores = [_as_int(slots.get(c)) for c in _COMPONENTS]
    missing = [s for s in scores if s == 0]

    # 5.6 too many missing components -> cannot decide
    if len(missing) >= 2:
        return {"hazard_score": 0, "hazard_class": "Unable to Decide"}

    # impute a single missing component with the max of the present ones
    present = [s for s in scores if s != 0]
    mx = max(present) if present else 0
    imputed = [s if s != 0 else mx for s in scores]
    hazard_score = sum(imputed)

    # 5.7 class thresholds
    if hazard_score <= 7:
        hazard_class = "Hazard Class A"
    elif hazard_score <= 12:
        hazard_class = "Hazard Class B"
    elif hazard_score <= 16:
        hazard_class = "Hazard Class C"
    else:
        hazard_class = "Hazard Class D"

    return {"hazard_score": hazard_score, "hazard_class": hazard_class}


def register(WRAPPED):
    WRAPPED["_classify"] = _classify
