# -*- coding: utf-8 -*-
"""Did a value copied into a payload survive the copy?

`task_018` passed and failed on one row. Both trials submitted the same six disputes;
the failing one added a seventh, and the seventh existed because the agent had typed
`rewards_earned: 1113` into the payload for a transaction whose record said
`rewards_earned: 487 points`. The engine computed honestly on what it was given and
produced a discrepancy that was not there. Two later calls in the same conversation
re-typed the same row correctly, so this is not a belief the model holds — it is a copy
that slipped.

That makes it a closed predicate ([[22]]): the record is in the conversation, the payload
claims to be that record, and the two disagree. Nothing here is a judgement about what
the customer wants, so the check is deterministic — but the correction is not ours to
make. The engine reports which field disagrees and what the record says, and the model
re-issues the call ([[10]]: the generator writes, the check verifies).

Only fields the payload and the record share are compared, and only rows the payload
identifies by the declared id. A field the record does not carry is not evidence of
anything and is left alone.
"""

import json
import re

__all__ = ["mismatches", "missing_fields", "unknown_ids", "field_sources", "note"]


def _txt(s):
    """A value as text, keeping falsy ones. `str(s or "")` turns 0 and False into "",
    which made a payload of `rewards_earned: 0` disagree with a record of `0` (x90 caught
    it before this ever denied anything)."""
    return "" if s is None else str(s)


def _num(s):
    """The leading number in a rendered value, or None — `$487.99` and `487 points` are numbers."""
    m = re.search(r"-?\d[\d,]*\.?\d*", _txt(s))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def same(a, b):
    """Do these two renderings of one value agree?

    The record prints what a human reads (`$487.99`, `487 points`) and the payload holds
    what a program takes (`487.99`, `487`). Comparing the strings would call every row a
    mismatch, so numbers are compared as numbers and everything else literally, trimmed.
    """
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-9
    return _txt(a).strip().lower() == _txt(b).strip().lower()


def _rows(value):
    """The payload's rows, whether it arrived as a list or as a JSON string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if isinstance(value, dict):
        value = [value]
    return [r for r in (value or []) if isinstance(r, dict)]


def mismatches(spec, args, records):
    """[(row_id, field, given, recorded)] — every copied value the record contradicts.

    `records` is {id: {field: rendered}} as the environment printed it; the caller parses
    that with the harness's own reader rather than a format rule of ours.
    """
    arg, idk = (spec or {}).get("arg"), (spec or {}).get("id_key")
    if not (arg and idk):
        return []
    out = []
    for row in _rows((args or {}).get(arg)):
        rid = _txt(row.get(idk)).strip()
        src = (records or {}).get(rid)
        if not rid or not src:
            continue                       # 출처가 문맥에 없으면 판정하지 않는다
        for k, v in row.items():
            if k == idk or k not in src:
                continue
            if not same(v, src[k]):
                out.append((rid, k, v, src[k]))
    return out


def missing_fields(spec, args, records, required=()):
    """[(row_id, field)] — the payload dropped a field the record has and the op needs.

    `task_022` typed seventy-seven rows by hand and left `account_open` off some of them.
    The engine could not judge those rows, reported `coverage 60 of 77`, and the customer
    ended up disputing five transactions instead of nine. The row was not wrong, it was
    incomplete — a different failure from `mismatches`, and the same cause: the rows were
    re-typed instead of referenced.

    The field does not have to live on the same record. `account_open` is an attribute of
    the account, not of the transaction, so it has to be joined in from the account read —
    which is exactly what `task_022` skipped for all seventy-seven rows. Requiring the
    field to be present in the same-id record made this check silent on its own target, so
    what is reported is simply: the declaration says this op needs the field, and the row
    does not have it.
    """
    arg, idk = (spec or {}).get("arg"), (spec or {}).get("id_key")
    need = [f for f in (required or ()) if f != idk]
    if not (arg and idk and need):
        return []
    out = []
    for row in _rows((args or {}).get(arg)):
        rid = _txt(row.get(idk)).strip()
        src = (records or {}).get(rid)
        if not rid or not src:
            continue
        for f in need:
            if f not in row:
                out.append((rid, f))
    return out


def unknown_ids(spec, args, records):
    """[row_id] — rows naming a record this conversation never read.

    `task_018` passed a row keyed `txn_890389b165_02`, which is the customer's own id with
    a suffix stuck on it: the shape of an invention, not of a read. The engine then computed
    on it. The check is the same closed one as `mismatches` — the record either was read or
    it was not — and it matters for a second reason: a settled row is only trustworthy if
    its inputs were, so this is what "the engine's own output is true" rests on.
    """
    arg, idk = (spec or {}).get("arg"), (spec or {}).get("id_key")
    if not (arg and idk):
        return []
    out = []
    for row in _rows((args or {}).get(arg)):
        rid = _txt(row.get(idk)).strip()
        if rid and rid not in (records or {}):
            out.append(rid)
    return out


def field_sources(spec, args, records, required=()):
    """[(field, tool, source_field)] — required fields the rows lack, and where the value lives.

    `task_022` typed seventy-seven rows and left `account_open` off every one, so the engine
    could judge sixty of them and the customer disputed five instead of nine. The obvious
    reading — the field was dropped in copying — is wrong: no record dump in any run carries
    a field by that name. It is called `date_of_account_open` where the card accounts are
    read and `date_opened` where the bank accounts are. The op asks for one name and the
    environment prints another, which is a mapping, and a mapping is something a declaration
    can hold ([[05]]: the function-schema config is ABox). So the engine says where to get
    it and the model goes and gets it — it does not fetch, and it does not rename anything.
    """
    arg, idk = (spec or {}).get("arg"), (spec or {}).get("id_key")
    src_map = (spec or {}).get("field_sources") or {}
    if not (arg and idk and src_map):
        return []
    rows = _rows((args or {}).get(arg))
    if not rows:
        return []
    out = []
    for f in (required or ()):
        if f == idk or f not in src_map:
            continue
        if any(f not in r for r in rows):
            s = src_map[f] or {}
            out.append((f, s.get("tool", ""), s.get("field", "")))
    return out


def note(tpl, bad, tool):
    """The declaration's sentence, filled with the disagreements. Empty when it says nothing."""
    if not (tpl and bad):
        return None
    detail = "; ".join("%s.%s = %s but the record says %s" % (r, k, g, s)
                       for r, k, g, s in bad[:6])
    return str(tpl).replace("{tool}", str(tool or "")).replace("{detail}", detail) \
                   .replace("{count}", str(len(bad)))
