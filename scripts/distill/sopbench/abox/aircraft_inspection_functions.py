import json
import math

OUTPUT_COLS = [
    "aircraft_ready",
    "mechanical_inspection_result",
    "electrical_inspection_result",
    "component_incident_response",
    "component_mismatch_response",
    "cross_check_response",
    "cross_check_reporting_response",
]


def _is_null(v):
    if v is None:
        return True
    if isinstance(v, float):
        try:
            return math.isnan(v)
        except Exception:
            return False
    s = str(v).strip().lower()
    return s in ("", "nan", "none", "null", "na", "n/a")


def _emit(slots):
    """Render a <final_output> JSON block into the reasoning trace so the
    fixed OutputParser (PRIORITY 1) can extract every required output column.
    Null-like values are emitted as the string "None" (parser treats none/nan
    as equivalent null values)."""
    payload = {}
    for col in OUTPUT_COLS:
        v = slots.get(col)
        if _is_null(v):
            payload[col] = "None"
        else:
            payload[col] = v
    block = "<final_output>" + json.dumps(payload) + "</final_output>"
    return {"final_output_block": block}


def register(WRAPPED):
    WRAPPED["_emit"] = _emit
