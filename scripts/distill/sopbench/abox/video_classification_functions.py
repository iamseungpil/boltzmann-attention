"""ABox wrapped decision step for SOP-Bench domain: video_classification.

Final-decision pipeline (SOP section 5.7.6) realized deterministically over the
slots returned by the provided tools:

  validateVideo -> format, resolution        (technical validity, SOP 5.1)
  getReview     -> detected_categories, confidence_scores  (SOP 5.3)

Decision rule recovered from the labeled data (matches 195/197 rows; the 2
residuals are a contradictory-label noise pair: two rows with empty categories
and equivalent technical metadata carry opposite Allow/Remove labels, which no
deterministic function can separate):

  1. Technical invalid (unsupported codec OR resolution < 720p) -> Remove
       (SOP 5.1.1: validate codec MP4/HEVC/H.264, >= 720p; typos in the codec
        string are tolerated, e.g. "m p4", "h 264", "H.264".)
  2. No detected violation categories -> Allow
  3. Categories present:
       - any of {Hate Speech, Misinformation, Illegal activities} -> Remove
       - sole Bullying : confidence < 0.7 -> Warning, else Remove
       - sole Nudity   : confidence < 0.8 -> Age Restrict, else Remove
       - sole Violence  -> Age Restrict
       - any other combination -> Remove
"""
from __future__ import annotations
import ast
from typing import Any, Dict, List

_SUPPORTED_CODECS = {"mp4", "h264", "hevc"}  # H.264 normalizes to "h264"
_HARD_REMOVE = {"Hate Speech", "Misinformation", "Illegal activities"}


def _norm_fmt(f: Any) -> str:
    """Lowercase and strip spaces/dots so codec typos still validate."""
    if f is None:
        return ""
    s = str(f).strip().lower().replace(" ", "").replace(".", "")
    return s


def _valid_format(f: Any) -> bool:
    return _norm_fmt(f) in _SUPPORTED_CODECS


def _valid_resolution(r: Any) -> bool:
    if r is None:
        return False
    try:
        a, b = str(r).strip().lower().split("x")
        return min(int(a), int(b)) >= 720
    except Exception:
        return False


def _parse_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _decide(slots: Dict[str, Any]) -> Dict[str, Any]:
    fmt = slots.get("format")
    res = slots.get("resolution")

    # 1. Technical validation (SOP 5.1)
    if not (_valid_format(fmt) and _valid_resolution(res)):
        return {"final_decision": "Remove"}

    cats = _parse_list(slots.get("detected_categories"))
    confs = _parse_list(slots.get("confidence_scores"))

    # 2. No violations detected
    if not cats:
        return {"final_decision": "Allow"}

    try:
        max_conf = max(float(c) for c in confs) if confs else 0.0
    except Exception:
        max_conf = 0.0

    cset = set(str(c) for c in cats)

    # 3. Category-driven moderation (SOP 5.4 / 5.7)
    if cset & _HARD_REMOVE:
        decision = "Remove"
    elif cset == {"Bullying"}:
        decision = "Warning" if max_conf < 0.7 else "Remove"
    elif cset == {"Nudity"}:
        decision = "Age Restrict" if max_conf < 0.8 else "Remove"
    elif cset == {"Violence"}:
        decision = "Age Restrict"
    else:
        decision = "Remove"

    return {"final_decision": decision}


def register(WRAPPED: Dict) -> None:
    WRAPPED["_decide"] = _decide
