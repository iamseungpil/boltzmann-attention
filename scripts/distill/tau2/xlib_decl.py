# -*- coding: utf-8 -*-
"""Reading a run's own output the way the declaration says it was written.

Offline instruments need what the live engine simply has: which rows a tool settled, and
which values a call carried. The engine now keeps its computed list (`_t2_sg_ids`), but a
census over persisted trajectories has only the text — so it has to re-read it, and *how*
it re-reads is exactly where instruments go wrong.

`x94` re-read it with its own `txn_` regex while the engine used a declared one, and the
declared one was dead (JSON `\\b` is a backspace, so it matched nothing). Two readers, two
answers, and the census could not see the engine's silence because it never used what the
engine used. This module exists so there is one reader: it takes the sentence forms we
declared and reads the list out of the position they define — no spelling rule anywhere.

  settled_ids(a2, tool, text)   the list a settle tool printed, per its declared template
  arg_values(args)              every scalar a call actually carried (membership, not shape)
"""

import json

__all__ = ["settled_ids", "arg_values", "forms"]


def forms(a2, name):
    """(sentence template, item template) — the base declaration and every variant."""
    out = []
    for d in ((a2 or {}).get("scaffold_get_tools") or []):
        if d.get("name") != name:
            continue
        out.append((d.get("return_template") or "{ids}", d.get("detail_item_template")))
        for v in (d.get("variants") or {}).values():
            if v.get("return_template"):
                out.append((v["return_template"],
                            v.get("detail_item_template") or d.get("detail_item_template")))
    return out


def settled_ids(a2, name, text, _cache={}):
    """The ids this output names, read out of the declared sentence.

    The fixed part of the template locates the list; the item template's first literal
    after `{id}` ends each item. Which template a run used is decided by the text itself,
    so base and variant forms are both tried.
    """
    key = (id(a2), name)
    fs = _cache.get(key)
    if fs is None:
        fs = _cache[key] = forms(a2, name)
    out = set()
    for tpl, item in fs:
        ph = "{details}" if "{details}" in tpl else "{ids}"
        head = tpl.partition(ph)[0].strip()
        i = text.find(head) if head else 0
        if i < 0:
            continue
        seg = text[i + len(head):].split("\n")[0]
        if ph == "{details}" and item:
            stop = item.partition("{id}")[2].partition("{")[0].strip()
            for it in seg.split(";"):
                it = it.strip()
                if stop:
                    it = it.partition(stop)[0].strip()
                if it and it != "(none)":
                    out.add(it)
        else:
            out |= {x.strip() for x in seg.split(",")
                    if x.strip() and x.strip() != "(none)"}
    return out


def arg_values(v, out=None):
    """Every scalar a call carried, nested JSON included — shape is never asked about."""
    if out is None:
        out = set()
    if isinstance(v, dict):
        for x in v.values():
            arg_values(x, out)
    elif isinstance(v, (list, tuple, set)):
        for x in v:
            arg_values(x, out)
    elif isinstance(v, str):
        out.add(v.strip())
        try:
            nested = json.loads(v)
        except Exception:
            nested = None
        if isinstance(nested, (dict, list)):
            arg_values(nested, out)
    elif v is not None and not isinstance(v, bool):
        out.add(str(v))
    return out
