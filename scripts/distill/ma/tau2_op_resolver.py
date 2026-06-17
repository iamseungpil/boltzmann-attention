#!/usr/bin/env python3
"""C8-2nd τ² op-IR resolver (C8_TAU2_SELECTION_TRANSFER_DESIGN §2). Executes a named OPERATION
(op-IR, same vocabulary as synth_depth) over a τ² retail variant catalog -> new_item_id.
Deterministic. The op-IR is what the (C8-trained) LLM names; this engine runs exp(op).

τ² catalog item = {"item_id", "options": {opt: val}, "available": bool}. Ordinal order for an
attr is decided per-case from the catalog values: known ordered words (ABox domain knowledge
below) first, then numeric extraction (4K=4000, 8GB=8, '6 inch'=6). Categorical attrs (color,
material, ...) have no order -> only usable in `among` (filter), not as a superlative `attr`.
"""
import re

# ABox: domain ordinal knowledge for categorical-but-ordered attrs (numeric attrs auto-parse).
ORD_WORDS = {
    "brightness": ["low", "medium", "high"],
    "ventilation": ["low", "medium", "high"],
    "room size": ["small", "medium", "large"],
    "processor": ["i5", "i7", "i9"],
    "backrest height": ["standard", "high-back"],
    "water resistance": ["not resistant", "IPX4", "IPX7"],
    "resolution": ["1080p", "4K", "5K"],  # video line; MP cameras fall through to numeric
}


def ordkey(attr, val):
    """Ordinal sort key for (attr, value), or None if not orderable."""
    if val is None:
        return None
    v = str(val).strip()
    if attr in ORD_WORDS:
        low = [x.lower() for x in ORD_WORDS[attr]]
        if v.lower() in low:
            return float(low.index(v.lower()))
    m = re.search(r'([\d.]+)\s*([KkMmTt]?)', v)
    if m:
        try:
            num = float(m.group(1))
        except ValueError:
            return None
        mult = {'K': 1e3, 'M': 1e6, 'T': 1e9}.get(m.group(2).upper(), 1)
        return num * mult
    return None


def is_ordinal(attr, catalog):
    """True if attr has a usable order across the catalog (>=2 distinct ordkeys)."""
    keys = {ordkey(attr, it["options"].get(attr)) for it in catalog if attr in it.get("options", {})}
    keys.discard(None)
    return len(keys) >= 2


def _among_match(it, among):
    return all(str(it["options"].get(k)) == str(v) for k, v in among.items())


def resolve_op_tau2(op_ir, catalog, anchor_id=None):
    """op_ir = {op, attr?, among?, dir?, k?, anchor_id?} -> item_id | None."""
    if not isinstance(op_ir, dict):
        return None
    among = op_ir.get("among") or {}
    cands = [it for it in catalog if _among_match(it, among)]
    avail = [it for it in cands if it.get("available")]
    pool = avail or cands
    op = op_ir.get("op")
    if op == "filter":
        if len(avail) == 1:
            return avail[0]["item_id"]
        if len(cands) == 1:
            return cands[0]["item_id"]
        return None
    attr = op_ir.get("attr")
    if attr is None or not pool:
        return None
    keyed = [(it, ordkey(attr, it["options"].get(attr))) for it in pool]
    keyed = [(it, k) for it, k in keyed if k is not None]
    if not keyed:
        return None
    if op == "argmax":
        return max(keyed, key=lambda x: x[1])[0]["item_id"]
    if op == "argmin":
        return min(keyed, key=lambda x: x[1])[0]["item_id"]
    if op == "rank":
        k = int(op_ir.get("k", 1))
        s = sorted(keyed, key=lambda x: -x[1])
        return s[k - 1][0]["item_id"] if len(s) >= k else None
    if op == "comparative":
        aid = op_ir.get("anchor_id") or anchor_id
        anchor = next((it for it in catalog if it["item_id"] == aid), None)
        if anchor is None:
            return None
        ak = ordkey(attr, anchor["options"].get(attr))
        if ak is None:
            return None
        dir_ = op_ir.get("dir", "greater")
        q = [(it, k) for it, k in keyed if (k > ak if dir_ == "greater" else k < ak)]
        if not q:
            return None
        return min(q, key=lambda x: abs(x[1] - ak))[0]["item_id"]
    return None
