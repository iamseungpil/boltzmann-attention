# -*- coding: utf-8 -*-
"""LB2 - decision-point isolation and deterministic execution (mechanism F2, symbolic operand).

One rule: arithmetic, comparisons and table lookups the policy fixes are done by the engine over
records; the model only supplies keys and copies values. Three doors, all declared in A2["LB2"]:

  computations  [{kind: ratio_cap | distinct | select, ...}]     checks on a call the model makes
  tools         [{name, description, params, op, ground, isolate, return_template, ...}]
                verifier tools injected into the agent's tool list; a call executes `op` here
  derived       [{out, inputs, op, shape, prompt, params, text}]  a DAG evaluated after a read tool
                ran (formalize = one sub-call turning a raw dump into rows / a date / a term)

The op vocabulary is data-driven (evaluate_op). A sub-call goes through `ask` only; the engine
never guesses a key and never writes a sentence of its own - every template is the declaration's.
"""

import datetime
import json

from lb_coordinator import Finding, DENY, GRADES, fam, fill, records_in, as_dict

LB = "LB2"
LEDGER, POLICY = GRADES["execution_ledger"], GRADES["policy_verbatim"]


# ---- shared: values, dates, paths ----------------------------------------------------------------
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def date(v, fmts=("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y")):
    s = str(v or "").split()
    for f in fmts:
        try:
            return datetime.datetime.strptime(s[0], f) if s else None
        except ValueError:
            pass
    return None


def add_months(d, m):
    from calendar import monthrange
    y, mo = d.year + (d.month - 1 + int(m)) // 12, (d.month - 1 + int(m)) % 12 + 1
    return d.replace(year=y, month=mo, day=min(d.day, monthrange(y, mo)[1]))


def month_index(anchor, target):
    da, dt = date(anchor), date(target)
    if da is None or dt is None or dt < da:
        return None
    k = (dt.year - da.year) * 12 + (dt.month - da.month)
    return k - 1 if add_months(da, k) > dt else k


def get(ctx, path):
    """Resolve 'a.b.c' against ctx; a plain number string is a literal; non-strings pass through."""
    if not isinstance(path, str):
        return path
    if not path[:1].isalpha() and path[:1] != "_":
        return num(path) if num(path) is not None else path
    cur = ctx
    for part in path.split("."):
        cur = cur.get(part.replace("[*]", "")) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def val(ctx, x):
    """An operand: nested op, path or literal."""
    return evaluate_op(x, ctx) if isinstance(x, dict) and x.get("op") else get(ctx, x)


def norm(s):
    return " ".join(str(s).split()).strip().lower()


# ---- the op vocabulary ---------------------------------------------------------------------------
def _cmp(a, b, c):
    return {">=": a >= b, ">": a > b, "<=": a <= b, "<": a < b, "==": a == b}.get(c)


def _lookup(spec, ctx):
    key = val(ctx, spec.get("key"))
    if key is None:
        return None
    for row in spec.get("table") or []:
        if row.get("cmp") is None:
            return val(ctx, row.get("result"))
        k = num(key)
        if k is not None and _cmp(k, row.get("thr"), row["cmp"]):
            return val(ctx, row.get("result"))
    return None


def _bool(spec, ctx):
    if "all" in spec or "any" in spec:
        vs = [_bool(c, ctx) for c in spec.get("all") or spec.get("any")]
        if "all" in spec:
            return False if False in vs else (None if None in vs else True)
        return True if True in vs else (None if None in vs else False)
    if "not" in spec:
        v = _bool(spec["not"], ctx)
        return None if v is None else not v
    v = val(ctx, spec["expr"]) if "expr" in spec else get(ctx, spec.get("ref"))
    if v is None:
        return None
    if "in" in spec:
        return v in spec["in"]
    for c in ("<=", ">=", "<", ">"):
        if c in spec:
            return None if num(v) is None else _cmp(num(v), spec[c], c)
    if "eq" in spec:
        yes, no = ("true", "yes"), ("false", "no")
        nb = lambda x: "true" if norm(x) in yes else ("false" if norm(x) in no else norm(x))
        return nb(v) == nb(spec["eq"])
    return bool(v)


def _match_verdict(spec, ctx):
    a, b = get(ctx, spec.get("a")), get(ctx, spec.get("b"))
    fields, thr = list(spec.get("fields") or []), int(spec.get("threshold", 2))
    grounded = spec.get("op") == "match_verdict_grounded"
    matched, missing = [], []
    if isinstance(a, dict) and (grounded or isinstance(b, dict)):
        for f in fields:
            av = a.get(f)
            if grounded:
                s = norm(av)
                outs = ctx.get("__tool_outputs") or {}
                hay = norm(" ".join(str(outs.get(t) or "") for t in spec.get("evidence_from") or []))
                ok = len(s) >= 4 and s in norm(ctx.get("__user_text") or "") and s in hay
            else:
                ok = av not in (None, "") and b.get(f) not in (None, "") and norm(av) == norm(b.get(f))
            (matched if ok else missing).append(f)
    else:
        missing = fields
    if grounded and not norm(" ".join(str((ctx.get("__tool_outputs") or {}).get(t) or "")
                                        for t in spec.get("evidence_from") or [])):
        tpl = spec.get("no_record_template") or spec.get("unmet_template") or "{count}"
    else:
        tpl = spec.get("met_template" if len(matched) >= thr else "unmet_template") or "{count}"
    return fill(tpl, count=len(matched), threshold=thr, matched=", ".join(matched) or "(none)",
                missing=", ".join(missing) or "(none)")


def _group_reduce(spec, ctx):
    items = val(ctx, spec.get("over"))
    if not isinstance(items, list):
        return None
    groups = {}
    for it in items:
        v = num((it or {}).get(spec.get("value_field"))) if isinstance(it, dict) else None
        if v is not None:
            groups.setdefault(str(it.get(spec.get("group_by"))), []).append(v)
    if any(str(g) not in groups for g in spec.get("required_groups") or []):
        return None
    exp = ctx.get("_expected_groups")
    if spec.get("require_complete_groups") and exp and exp.get("field") == spec.get("group_by"):
        if any(str(l) not in groups for l in exp.get("labels") or []):
            return None                                   # some window has no data: abstain
    reducers, red = spec.get("reducers") or {}, []
    for g, vs in groups.items():
        r = reducers.get(g, spec.get("default_reducer"))
        if r == "max1":
            red.append(max(vs))
        elif r == "sum":
            red.append(sum(vs))
        else:
            ctx.setdefault("_flags", []).append(g)
    across = spec.get("across", "sum")
    if not red:
        return None if across in ("min", "max") else 0.0
    return {"min": min, "max": max}.get(across, sum)(red)


def _bucket(spec, ctx):
    recs = get(ctx, spec.get("over"))
    if not isinstance(recs, list):
        return None
    anchor, df, of = get(ctx, spec.get("anchor")), spec.get("date_field", "date"), spec.get("out_field", "window")
    ctx["_expected_groups"] = {"field": of, "labels": list(range(12))}
    tagged = [(month_index(anchor, r.get(df)), r) for r in recs if isinstance(r, dict)]
    tagged = [(k, r) for k, r in tagged if k is not None]
    if spec.get("year_select") == "last_complete":
        k_asof = month_index(anchor, get(ctx, spec.get("as_of")))
        if k_asof is None:
            return None
        yy = k_asof // 12 - 1
        return [] if yy < 0 else [dict(r, **{of: k % 12}) for k, r in tagged if k // 12 == yy]
    return [dict(r, **{of: k}) for k, r in tagged if not spec.get("within_year", True) or 0 <= k <= 11]


def _select_discrepant(spec, ctx):
    recs = get(ctx, spec.get("over")) or []
    if not isinstance(recs, list):
        return []
    idf, af, tol = spec.get("id_field"), spec.get("actual_field"), num(spec.get("tolerance")) or 0
    steps = spec.get("steps") or {}
    order = list(range(len(recs)))
    if spec.get("order_field"):
        order.sort(key=lambda i: (date(recs[i].get(spec["order_field"])) or datetime.datetime.max, i))
    dups = _duplicates(recs, idf, spec.get("dup_field"))     # rows the caller marked as a second fee line
    ordinals = {}
    for name, st in steps.items():
        if isinstance(st, dict) and st.get("op") == "ordinal":
            count, vals = {}, [None] * len(recs)
            for i in order:
                if not isinstance(recs[i], dict) or str(recs[i].get(idf)) in dups:
                    continue                                    # a duplicate line consumes no free withdrawal
                key = str(get({"r": recs[i]}, st.get("partition"))) if st.get("partition") else "*"
                count[key] = count.get(key, 0) + 1
                vals[i] = count[key]
            ordinals[name] = vals
    # rebate axis (declared per level): a level whose document promises fee rebates up to a monthly cap
    # is judged net of rebates on both sides - actual_net = charged - rebated, expected_net = documented
    # rate - min(rate, cap left). The rebate owed is computed on the documented rate, not the charge,
    # so an overcharge is returned once. Undeclared cap (null) = this axis is off.
    rb = spec.get("rebate") or {}
    cap = rb.get("cap")
    cap_left = num(evaluate_op(cap, ctx) if isinstance(cap, dict) else cap)
    out, skipped, details = [], 0, ctx.setdefault("_details", [])
    for i in order:
        r = recs[i]
        if not isinstance(r, dict) or r.get(idf) in (None, ""):
            skipped += 1
            continue
        rctx = dict(ctx, r=r, steps={})
        for name, st in steps.items():
            rctx["steps"][name] = ordinals[name][i] if name in ordinals else evaluate_op(st, rctx)
        exp = get(rctx, spec["expected_ref"]) if spec.get("expected_ref") else evaluate_op(spec.get("expected"), rctx)
        if str(r.get(idf)) in dups:
            exp = 0                                             # the whole duplicate line is wrongly charged
        en, act = num(exp), num(r.get(af))
        if en is None or act is None:
            skipped += 1
            continue
        if rb.get("field") and cap_left is not None:
            if rb["field"] not in r:
                skipped += 1                                    # not judged: no rebate fact was formalized
                continue
            due = min(en, cap_left) if en > 0 else 0.0
            cap_left = round(cap_left - due, 2)
            en = round(en - due, 2)
            act = round(act - (num(r.get(rb["field"])) or 0.0), 2)
        if abs(en - act) > tol:
            out.append(r.get(idf))
            details.append({"id": r.get(idf), "actual": act, "expected": en, "delta": round(act - en, 2)})
    ctx["_stats"] = {"judged": len(recs) - skipped, "skipped": skipped, "total": len(recs)}
    return out


def _duplicates(recs, idf, dupf):
    """Ids the caller marked as a duplicate fee line (row[dupf] names the original).

    Every marked row is a duplicate, except that when a group points at each other and at least two
    are marked, the first by input position is the original and stays charged.
    """
    if not dupf:
        return set()
    rows = [r for r in recs if isinstance(r, dict) and r.get(idf) not in (None, "")]
    marked = {str(r[idf]) for r in rows if r.get(dupf)}
    pos = {}
    for i, r in enumerate(rows):
        pos.setdefault(str(r[idf]), i)
    parent = {}

    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for r in rows:
        if r.get(dupf) and str(r.get(dupf)) in pos:
            a, b = root(str(r[idf])), root(str(r[dupf]))
            if a != b:
                parent[a] = b
    groups = {}
    for k in pos:
        groups.setdefault(root(k), []).append(k)
    for members in groups.values():
        hit = sorted((m for m in members if m in marked), key=lambda k: pos[k])
        if len(hit) >= 2:
            marked.discard(hit[0])
    return marked


def _catalog_filter(spec, ctx):
    """Rows of a documented table filtered by the constraints the caller stated.

    Constraints are data: [{param, field, sense: le|ge|flag|min_score}]. A row whose fact is
    undocumented is 'unverified', never eligible - the engine does not claim what no document says.
    """
    elig, excl, unver = [], [], []
    seg = spec.get("segment") or {}
    for row in spec.get("table") or []:
        if seg and bool(row.get(seg["field"])) != bool(ctx.get(seg["param"])):
            continue
        why, missing = None, []
        for c in spec.get("constraints") or []:
            cv, rv = ctx.get(c["param"]), row.get(c["field"])
            if c["sense"] == "unless":
                cv = not cv
            if cv in (None, "", False):
                continue
            if rv is None:
                missing.append("%s (constraint %s=%s)" % (c["field"], c["param"], cv))
            elif c["sense"] == "flag" and not rv:
                why = "%s is documented as not available" % c["field"]
            elif c["sense"] == "unless" and rv:
                why = "%s applies and %s was not stated" % (c["field"], c["param"])
            elif c["sense"] in ("le", "ge") and num(cv) is not None and num(rv) is not None \
                    and ((c["sense"] == "le" and num(rv) > num(cv)) or (c["sense"] == "ge" and num(rv) < num(cv))):
                why = "%s=%s violates %s=%s" % (c["field"], rv, c["param"], cv)
            if why:
                break
        facts = {k: v for k, v in row.items() if k not in (spec.get("label_field", "card"), "source")}
        entry = {"item": row.get(spec.get("label_field", "card")), "facts": facts, "source": row.get("source")}
        (excl if why else unver if missing else elig).append(dict(entry, reason=why, undocumented=missing))
    return {"eligible": elig, "excluded": excl, "unverified": unver, "note": spec.get("note", "")}


def _catalog_compute(spec, ctx):
    out, unver = [], []
    lbl = spec.get("label_field") or "label"
    for row in spec.get("table") or []:
        rctx = dict(ctx, r=row, steps={})
        for nm, st in (spec.get("steps") or {}).items():
            rctx["steps"][nm] = evaluate_op(st, rctx)
        vals = {}
        for cn, ref in (spec.get("value_cols") or {}).items():
            v = num(val(rctx, ref)) if isinstance(ref, (str, dict)) else None
            vals[cn] = None if v is None else round(v, 2)
        (unver.append(row.get(lbl)) if any(v is None for v in vals.values())
         else out.append(dict({lbl: row.get(lbl)}, source=row.get("source"), **vals)))
    if spec.get("row_template"):
        lines = [fill(spec["row_template"], **e) for e in out]
        txt = "\n".join(lines) if lines else "(none computable)"
        if unver:
            txt += "\n" + fill(spec.get("not_computable_note", "not computable: {names}"), names=", ".join(map(str, unver)))
        return txt
    return {"rows": out, "not_computable": unver}


def _filter(spec, ctx):
    recs = get(ctx, spec.get("over")) or []

    def hit(r):
        for c in spec.get("match") or []:
            want = get(ctx, c.get("eq", c.get("contains")))
            if want in (None, ""):
                continue
            have = str((r or {}).get(c.get("field")) or "")
            if ("eq" in c and have != str(want)) or ("contains" in c and str(want).lower() not in have.lower()):
                return False
        return True

    hits = [r for r in recs if isinstance(r, dict) and hit(r)]
    pick = {1: hits[0] if hits else None, 0: None}.get(len(hits), None)
    if len(hits) > 1:
        pick = {"first": hits[0], "last": hits[-1]}.get(spec.get("on_ambiguous", "none"))
    return pick.get(spec["return"]) if pick and spec.get("return") else pick


OPS = {
    "const": lambda s, c: s.get("value"),
    "ref": lambda s, c: get(c, s.get("path")),
    "ref_op": lambda s, c: (evaluate_op(get(c, s["path"]), c) if isinstance(get(c, s.get("path")), dict)
                            else num(get(c, s.get("path")))),
    "min": lambda s, c: min([v for v in (num(val(c, x)) for x in s.get("of") or []) if v is not None] or [None]),
    "max": lambda s, c: max([v for v in (num(val(c, x)) for x in s.get("of") or []) if v is not None] or [None]),
    "sum": lambda s, c: sum(v for v in (num(val(c, x)) for x in s.get("of") or []) if v is not None),
    "multiply": lambda s, c: (None if None in (num(val(c, s.get("a"))), num(val(c, s.get("b"))))
                              else num(val(c, s["a"])) * num(val(c, s["b"]))),
    "diff": lambda s, c: (None if None in (num(val(c, s.get("a"))), num(val(c, s.get("b"))))
                          else num(val(c, s["a"])) - num(val(c, s["b"]))),
    "compare": lambda s, c: (None if None in (num(val(c, s.get("a"))), num(val(c, s.get("b"))))
                             else _cmp(num(val(c, s["a"])), num(val(c, s["b"])), s.get("cmp", ">="))),
    "clamp": lambda s, c: (None if num(get(c, s.get("value"))) is None else
                           min(max(num(get(c, s["value"])), num(get(c, s.get("min"))) if s.get("min") is not None else -1e300),
                               num(get(c, s.get("max"))) if s.get("max") is not None else 1e300)),
    "str_eq": lambda s, c: (None if get(c, s.get("a")) is None else norm(get(c, s["a"])) == norm(s.get("b"))),
    "days_between": lambda s, c: (None if None in (date(get(c, s.get("a"))), date(get(c, s.get("b"))))
                                  else abs((date(get(c, s["b"])) - date(get(c, s["a"]))).days)),
    "date_in_window": lambda s, c: (None if None in (date(get(c, s.get("anchor"))), date(get(c, s.get("target"))),
                                                     num(get(c, s.get("months"))))
                                    else date(get(c, s["anchor"])) <= date(get(c, s["target"]))
                                    <= add_months(date(get(c, s["anchor"])), num(get(c, s["months"])))),
    "date_between": lambda s, c: (None if None in (date(get(c, s.get("x"))), date(get(c, s.get("lo"))), date(get(c, s.get("hi"))))
                                  else date(get(c, s["lo"])) <= date(get(c, s["x"])) <= date(get(c, s["hi"]))),
    "count_where": lambda s, c: sum(1 for r in (get(c, s.get("over")) or [])
                                    if isinstance(r, dict) and r.get(s.get("cond_field")) == s.get("cond_value")),
    "case": lambda s, c: next((val(c, v) for k, v in (s.get("cases") or {}).items()
                               if val(c, s.get("key")) is not None and norm(val(c, s["key"])) == norm(k)),
                              val(c, s.get("default"))),
    "if_then": lambda s, c: (val(c, s.get("then")) if evaluate_op(s.get("cond"), c)
                             else val(c, s.get("else")) if evaluate_op(s.get("cond"), c) is not None
                             else (val(c, s.get("then")) if val(c, s.get("then")) == val(c, s.get("else")) else None)),
    "bool_expr": _bool, "lookup_table": _lookup, "match_verdict": _match_verdict,
    "match_verdict_grounded": _match_verdict, "group_reduce": _group_reduce, "bucket_month_window": _bucket,
    "select_discrepant": _select_discrepant, "catalog_filter": _catalog_filter,
    "catalog_compute": _catalog_compute, "filter": _filter,
}


def evaluate_op(spec, ctx):
    """Run an op tree over ctx. Unknown op or any failure -> None (abstain, never a guess)."""
    if not isinstance(spec, dict) or spec.get("op") not in OPS:
        return None
    try:
        return OPS[spec["op"]](spec, ctx)
    except Exception:
        return None


# ---- verifier tools ------------------------------------------------------------------------------
def over_params(op):
    """Array parameters an op tree reads (`over`), in declaration order."""
    out = []

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("over"), str) and o["over"] not in out:
                out.append(o["over"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(op)
    return out


def ground_operands(decl, ctx, corpora):
    """Drop operands whose value or cited source is not in the declared corpus. Returns the drops."""
    g, flags = decl.get("ground") or {}, []
    for af in g.get("array_fields") or []:
        arr = ctx.get(af.get("param"))
        if not isinstance(arr, list):
            continue
        hay = [norm(t) for c in af.get("corpus") or ["kb"] for t in corpora.get(c, [])]
        kept = []
        for el in arr:
            src = norm((el or {}).get(af.get("source_field", "source"))) if isinstance(el, dict) else ""
            src_ok = bool(src) and any(src in h for h in hay)
            v = num((el or {}).get(af.get("value_field", "value"))) if isinstance(el, dict) else None
            val_ok = not af.get("require_value_in_source", True) or v is None or \
                any(abs(v - n) < 1e-9 for n in numbers_in(src))
            (kept.append(el) if src_ok and val_ok else flags.append("%s=%s" % (el.get(af.get("label_field", "kind"), "?"), v)))
        ctx[af["param"]] = kept
    for sf in g.get("scalar_fields") or []:
        p = sf.get("param")
        if p not in ctx:
            continue
        hay = norm(" ".join(t for c in sf.get("corpus") or ["ledger"] for t in corpora.get(c, [])))
        if not grounded_scalar(ctx[p], hay, sf.get("kind")):
            flags.append("%s=%s" % (p, ctx[p]))
            ctx.pop(p, None)
    return flags


def numbers_in(text):
    out = []
    for tok in str(text or "").replace(",", "").replace("%", " ").replace("$", " ").split():
        v = num(tok.strip(".;:()"))
        if v is not None:
            out.append(v)
    return out


def grounded_scalar(value, hay, kind):
    s = norm(value)
    if kind == "number":
        return num(value) is not None and any(abs(num(value) - n) < 1e-9 for n in numbers_in(hay))
    if kind == "date":
        d = date(value)
        return d is not None and any(date(t) == d for t in hay.replace(",", " ").split())
    return bool(s) and s in hay


def render_result(decl, ctx, result):
    ids = result if isinstance(result, list) else []
    st = ctx.get("_stats") or {}
    if st and not st.get("judged"):
        # an empty result over zero judged rows is not "no discrepancy" - probe 017 read it as clean
        # and skipped the disputes the task is about
        return ("Error: [COVERAGE] none of the %d rows could be judged: the values the comparison needs were not "
                "established for any of them, so this result says nothing. Retrieve the document(s) that state "
                "those values, then call again." % st.get("total", 0))
    details = "; ".join("%s: recorded %s, expected %s (delta %s)" % (d["id"], d["actual"], d["expected"], d["delta"])
                        for d in ctx.get("_details") or [])
    slots = {k: v for k, v in ctx.items() if isinstance(v, (str, int, float))}
    if isinstance(result, list) and not result and st.get("skipped"):
        # a clean sweep of the rows that could be judged is not a clean sweep. On probe 017 the two
        # rows nobody could judge were the two the customer was disputing, and "no discrepancy" as
        # the opening sentence was read as the answer.
        return ("Error: [COVERAGE] %d of %d rows could not be judged (the values the comparison needs were not "
                "established for them), and none of the %d that were judged is discrepant. This is not a verdict: "
                "retrieve the document(s) covering the remaining rows, then call again."
                % (st["skipped"], st["total"], st.get("judged", 0)))
    if isinstance(result, list) and not result and decl.get("return_template_empty"):
        text = decl["return_template_empty"]
    else:
        text = fill(decl.get("return_template") or "{result}", result=json.dumps(result, ensure_ascii=False)
                    if isinstance(result, (dict, list)) else result, ids=", ".join(map(str, ids)) or "(none)",
                    details=details or "(none)", **slots)
    # The ids have to leave here as records, not only as prose. LB4's settled_rows reads them back
    # with records_in, which finds JSON objects by brace matching; the sentence above gave it nothing,
    # so `settled` was always empty and the rule never fired once in 39 simulations that ran this
    # tool - 14 of them ended short of the disputes it had just found (2026-09-09).
    idf = (decl.get("op") or {}).get("id_field")
    if idf and ids:
        text += "\n[ROWS] " + json.dumps([{idf: str(x)} for x in ids], ensure_ascii=False)
    if st.get("skipped"):
        text += " [coverage: %d of %d rows could not be judged - no policy rate was established for them]" % (st["skipped"], st["total"])
    return text


def run_tool(decl, args, corpora, evidence):
    """Execute one verifier tool: parse args, ground operands, run the op, render the declaration's text."""
    ctx = {k: (as_dict(v) or _list(v) if isinstance(v, str) and v[:1] in "[{" else v) for k, v in (args or {}).items()}
    ctx.update(evidence or {})
    bad = [p for p in over_params(decl.get("op")) if isinstance(ctx.get(p), str)]
    if bad:
        return ("Error: [ARGS-FORMAT] the '%s' argument could not be read as a JSON array. Re-issue this call with "
                "'%s' as a valid JSON array (double quotes, plain numbers)." % (bad[0], bad[0])), True
    flags = ground_operands(decl, ctx, corpora)
    result = evaluate_op(decl.get("op"), ctx)
    if result is None and flags:
        return "Abstained: these inputs are not supported by any record or document in this conversation: %s" % "; ".join(flags), True
    if result is None:
        return decl.get("missing_hint") or "Abstained: the inputs given do not determine a result.", True
    return render_result(decl, ctx, result), False


def _list(v):
    try:
        x = json.loads(v)
        return x if isinstance(x, list) else v
    except Exception:
        return v


# ---- derived DAG (facts after a read) ------------------------------------------------------------
def parse_rows(raw, keys):
    rows = []
    for r in records_in(raw):
        keep = {k: r[k] for k in keys if r.get(k) not in (None, "")}
        if len(keep) == len(keys):
            rows.append(keep)
    return rows


def derived_facts(a2, tool_outputs, ask, a3_rows=()):
    """Evaluate A2["LB2"]["derived"] over the tool outputs seen so far; return [(out, value, text)]."""
    nodes, vals, texts = ordered((a2.get("LB2") or {}).get("derived") or []), {}, []
    for n in nodes:
        ins = list(n.get("inputs") or [])
        src = [i for i in ins if i.startswith("tool:")]
        if any(fam(i[5:]) not in {fam(t) for t in tool_outputs} for i in src) or any(
                i in vals and vals[i] is None for i in ins):
            vals[n["out"]] = None
            continue
        p = dict(n.get("params") or {})
        try:
            if n["op"] == "formalize":
                text = "\n---\n".join(tool_outputs[t] for t in tool_outputs if src and fam(t) == fam(src[0][5:])) \
                    if src else "\n---\n".join(tool_outputs.values())
                raw = ask(fill(n.get("prompt") or "{text}", text=text[:60000], keys=", ".join(p.get("row_keys") or [])),
                          "lb2_formalize") if ask else None
                if n.get("shape") == "rows":
                    v = parse_rows(raw, p.get("row_keys") or []) or None
                elif n.get("shape") == "scalar":
                    tok = (str(raw or "").split() or [""])[0].strip('".,')
                    v = tok if date(tok, tuple(p.get("date_formats") or ())) else None
                else:
                    v = next((s for s in sorted({r.get("subject") for r in a3_rows if r.get("subject")})
                              if s and s in str(raw or "")), None)
                    v = {"subject": v} if v else None
            elif n["op"] == "a3_map":
                v = {r["subject"]: int(r["value"]) for r in a3_rows if r.get("axis") == p.get("axis")
                     and r.get("subject") is not None and r.get("value") is not None} or None
            else:
                v = DERIVED[n["op"]](vals.get(ins[0]), vals.get(ins[1]) if len(ins) > 1 else None, p)
        except Exception:
            v = None
        vals[n["out"]] = v
        if v is not None and n.get("text"):
            texts.append((n["out"], v, fill(n["text"], **_slots(v, p))))
    return texts


def ordered(nodes):
    """Nodes in dependency order (a node after every node it reads)."""
    out, done, pending = [], set(), list(nodes)
    while pending:
        ready = [n for n in pending if all(i in done or ":" in i or i in ("corpus", "a3") for i in n.get("inputs") or [])]
        if not ready:
            ready = pending[:1]                       # a cycle: evaluate in declaration order
        for n in ready:
            out.append(n)
            done.add(n["out"])
            pending.remove(n)
    return out


def _slots(v, p):
    s = dict(p)
    if isinstance(v, dict):
        s.update({k: x for k, x in v.items() if isinstance(x, (str, int, float))})
        groups = {k: x for k, x in v.items() if isinstance(x, (int, float, bool))}
        s["exhausted"] = ", ".join(k for k, x in groups.items() if x is not True and num(x) is not None and x <= 0) or "(none)"
        s["remaining_groups"] = ", ".join("%s (%s)" % (k, x) for k, x in groups.items() if x is True or (num(x) is not None and x > 0)) or "(none)"
        s["not_ok"] = ", ".join(k for k, x in groups.items() if x is False) or "(none)"
    else:
        s["value"] = v
    s.setdefault("max", p.get("window_max"))
    s.setdefault("days", p.get("window_days"))
    return s


def _tally(rows, _b, p):
    out = {}
    for r in rows or []:
        g = r.get(p["group_field"])
        if g:
            out[g] = out.get(g, 0) + 1
    return out


def _window_remaining(rows, now, p):
    ref = date(now, tuple(p["date_formats"]))
    if ref is None:
        return None
    used = sum(1 for r in rows or [] if date(r.get(p["date_field"]), tuple(p["date_formats"])) is not None
               and 0 <= (ref - date(r.get(p["date_field"]), tuple(p["date_formats"]))).days <= int(p["window_days"]))
    return {"used": used, "remaining": max(0, int(p["window_max"]) - used)}


def _days_since_earliest(rows, now, p):
    ref = date(now, tuple(p["date_formats"]))
    ds = [d for d in (date(r.get(p["age_field"]), tuple(p["date_formats"])) for r in rows or []) if d]
    return {"since": min(ds).strftime("%m/%d/%Y"), "days": (ref - min(ds)).days} if ref and ds else None


DERIVED = {
    "tally": _tally, "window_remaining": _window_remaining, "days_since_earliest": _days_since_earliest,
    "subtract_by_group": lambda usage, limits, _p: {g: int(l) - int((usage or {}).get(g, 0)) for g, l in (limits or {}).items()},
    "compare_ge": lambda days, mins, _p: None if days is None else {g: int(days["days"]) >= int(m) for g, m in (mins or {}).items()},
    "pick": lambda mapping, focus, _p: ({"subject": focus["subject"], "value": mapping[focus["subject"]]}
                                        if isinstance(focus, dict) and isinstance(mapping, dict)
                                        and focus.get("subject") in mapping else None),
}


# ---- computations on a call the model makes ------------------------------------------------------
def applies(spec, turn, call):
    name = str(getattr(call, "name", "") or "")
    if spec.get("applies_to") not in (None, name, fam(name)):
        return False
    w = spec.get("when") or {}
    return not w.get("arg") or str(as_dict(call.arguments).get(w["arg"]) or "").startswith(w.get("prefix", ""))


def ratio_cap(spec, turn, call):
    args = turn.args_of(call)
    value, rid = num(args.get(spec.get("param"))), args.get(spec.get("record_key"))
    recs = [r for o in turn.tool_outputs() for r in records_in(o, spec.get("record_key")) if str(r.get(spec["record_key"])) == str(rid)]
    if value is None or not recs:
        return None
    rec, pb = recs[-1], spec.get("pct_by") or {}
    limit, pct = num(rec.get(spec.get("limit_field"))), (pb.get("map") or {}).get(str(rec.get(pb.get("field"))))
    if limit is None or pct is None or value <= limit * pct:
        return None
    return fill(spec.get("feedback"), value=value, cap=limit * pct, pct=pct, limit=limit)


def distinct(spec, turn, call):
    args = turn.args_of(call)
    for a, b in spec.get("pairs") or []:
        if args.get(a) is not None and args.get(b) is not None and str(args[a]) == str(args[b]):
            return fill(spec.get("feedback"), a=a, b=b)
    return None


def select(spec, turn, call):
    """The record the customer described (criteria formalized by one sub-call) must be the one passed."""
    ask, given = turn.extras.get("ask"), turn.args_of(call).get(spec.get("param"))
    if ask is None or given in (None, ""):
        return None
    raw = ask(fill(spec.get("criteria_prompt") or DEFAULT_CRITERIA, fields=", ".join(spec.get("criteria_fields") or []),
                   text=turn.user_text[-6000:]), "lb2_criteria")
    crit = next(iter(records_in(raw)), None)
    if not crit or not any(crit.get(f) for f in spec.get("criteria_fields") or []):
        return None
    recs = [r for o in turn.tool_outputs() for r in records_in(o, spec.get("key_field"))
            if all(r.get(f) not in (None, "") for f in spec.get("require") or [])]
    pick = _filter({"over": "recs", "match": spec.get("match") or [], "return": spec.get("key_field"),
                    "on_ambiguous": spec.get("on_ambiguous", "none")}, {"recs": recs, "criteria": crit})
    if pick is None or str(pick) == str(given):
        return None
    return fill(spec.get("feedback") or DEFAULT_SELECT, given=given, pick=pick, criteria=json.dumps(crit, ensure_ascii=False))


DEFAULT_CRITERIA = ("From the customer's messages below, extract what they said about the record they mean. Reply with one "
                    "JSON object with these keys (empty string when not stated): {fields}.\n\n{text}")
DEFAULT_SELECT = ("Error: [REFERENCE] the customer described {criteria}; the only record matching that is {pick}, but the "
                  "call passed {given}. Use the record the customer described.")
KINDS = {"ratio_cap": ratio_cap, "distinct": distinct, "select": select}


def evaluate(turn):
    out = []
    for c in turn.calls:
        name = turn.name_of(c)
        for spec in (turn.a2.get("LB2") or {}).get("computations") or []:
            check = KINDS.get(spec.get("kind"))
            if check and spec.get("tool") in (None, name, fam(name)) and applies(spec, turn, c):
                text = check(spec, turn, c)
                if text:
                    out.append(Finding(LB, DENY, fam(name), c, text, grade=LEDGER, source=spec["kind"]))
    return out


if __name__ == "__main__":
    from lb_coordinator import Turn

    class C(object):
        def __init__(self, name, args=None):
            self.name, self.arguments, self.id = name, args or {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    # op vocabulary
    ctx = {"r": {"card": "Gold", "cat": "Travel", "amt": 100.0}, "steps": {}}
    rate = {"op": "case", "key": "r.card", "cases": {"Gold": 2.5, "Silver": {"op": "case", "key": "r.cat", "cases": {"Travel": 4}, "default": 1}}, "default": None}
    assert evaluate_op(rate, ctx) == 2.5 and evaluate_op({"op": "multiply", "a": "r.amt", "b": rate}, ctx) == 250.0
    sd = {"op": "select_discrepant", "over": "tx", "id_field": "id", "actual_field": "got", "tolerance": 1,
          "steps": {"rate": rate}, "expected": {"op": "multiply", "a": "r.amt", "b": "steps.rate"}}
    c2 = {"tx": [{"id": "t1", "card": "Gold", "amt": 100, "got": 250}, {"id": "t2", "card": "Gold", "amt": 100, "got": 100},
                 {"id": "t3", "card": "Silver", "cat": "Travel", "amt": 10, "got": 40}]}
    assert evaluate_op(sd, c2) == ["t2"] and c2["_details"][0]["delta"] == -150.0
    # rebate axis: task_072 Bluest facts - documented $2.00 out-of-network fee, rebated up to $50 a cycle.
    # 11/14 charged 2.00 with no rebate -> owed 2.00; 11/20 charged 2.50 rebated 2.00 -> owed 0.50;
    # 11/18 charged 2.00 rebated 2.00 -> nothing; a row with no rebate fact is not judged.
    rbs = {"op": "select_discrepant", "over": "tx", "id_field": "id", "actual_field": "fee", "order_field": "d",
           "expected": {"op": "const", "value": 2.0}, "rebate": {"field": "rb", "cap": 50.0}}
    c3 = {"tx": [{"id": "a", "d": "11/20/2025", "fee": 2.5, "rb": 2.0}, {"id": "b", "d": "11/18/2025", "fee": 2.0, "rb": 2.0},
                 {"id": "c", "d": "11/14/2025", "fee": 2.0, "rb": 0}, {"id": "n", "d": "11/12/2025", "fee": 2.0}]}
    assert evaluate_op(rbs, c3) == ["c", "a"], c3["_details"]
    assert {e["id"]: e["delta"] for e in c3["_details"]} == {"c": 2.0, "a": 0.5} and c3["_stats"]["skipped"] == 1
    # duplicate fee lines: the marked second line is expected 0 and consumes no free withdrawal
    dps = {"op": "select_discrepant", "over": "tx", "id_field": "id", "actual_field": "fee", "dup_field": "dup",
           "steps": {"n": {"op": "ordinal"}}, "expected": {"op": "lookup_table", "key": "steps.n", "table": [{"cmp": "<=", "thr": 1, "result": 0}, {"result": 1.5}]}}
    c4 = {"tx": [{"id": "x", "fee": 0}, {"id": "y", "fee": 1.5, "dup": "x"}, {"id": "z", "fee": 1.5}]}
    assert evaluate_op(dps, c4) == ["y"] and c4["_details"][0]["delta"] == 1.5, c4["_details"]
    assert _duplicates([{"id": "x", "dup": "y"}, {"id": "y", "dup": "x"}], "id", "dup") == {"y"}
    assert evaluate_op({"op": "if_then", "cond": {"op": "compare", "cmp": "<", "a": {"op": "days_between", "a": "x", "b": "y"}, "b": 90},
                        "then": {"op": "const", "value": "YOUNG"}, "else": {"op": "const", "value": "OLD"}},
                       {"x": "01/01/2026", "y": "03/01/2026"}) == "YOUNG"
    assert evaluate_op({"op": "lookup_table", "key": "n", "table": [{"cmp": "<=", "thr": 4, "result": 0}, {"result": 2.5}]}, {"n": 5}) == 2.5
    cf = {"op": "catalog_filter", "label_field": "card", "constraints": [{"param": "max_fee", "field": "fee", "sense": "le"}],
          "table": [{"card": "A", "fee": 0}, {"card": "B", "fee": 200}, {"card": "C"}]}
    r = evaluate_op(cf, {"max_fee": 50})
    assert [e["item"] for e in r["eligible"]] == ["A"] and [e["item"] for e in r["excluded"]] == ["B"] and r["unverified"][0]["item"] == "C"
    mv = {"op": "match_verdict", "a": "p", "b": "rec", "fields": ["dob", "email"], "threshold": 2, "met_template": "OK {count}", "unmet_template": "NO {count} {missing}"}
    assert evaluate_op(mv, {"p": {"dob": "1/1/90", "email": "A@x"}, "rec": {"dob": "1/1/90", "email": "a@x"}}) == "OK 2"
    # a verifier tool end to end
    decl = {"name": "chk", "op": sd, "return_template": "bad: {ids}", "return_template_empty": "none",
            "ground": {"scalar_fields": [{"param": "cap", "corpus": ["ledger"], "kind": "number"}]}}
    text, err = run_tool(decl, {"tx": json.dumps(c2["tx"]), "cap": "12"}, {"ledger": ["limit 12"]}, {})
    assert text.startswith("bad: t2") and not err, text
    assert run_tool(decl, {"tx": "not json"}, {}, {})[1]
    # derived DAG with a fake formalizer
    a2 = {"LB2": {"derived": [
        {"out": "rows", "inputs": ["tool:get_refs"], "op": "formalize", "shape": "rows", "prompt": "{text}", "params": {"row_keys": ["date", "kind"]}},
        {"out": "today", "inputs": ["corpus"], "op": "formalize", "shape": "scalar", "prompt": "{text}", "params": {"date_formats": ["%m/%d/%Y"]}},
        {"out": "usage", "inputs": ["rows"], "op": "tally", "params": {"group_field": "kind"}},
        {"out": "left", "inputs": ["rows", "today"], "op": "window_remaining", "params": {"date_field": "date", "date_formats": ["%m/%d/%Y"], "window_days": 9, "window_max": 2},
         "text": "in window {used}; {remaining} more allowed"},
        {"out": "annual", "inputs": ["usage", "limits"], "op": "subtract_by_group", "text": "no room: {exhausted}"},
        {"out": "limits", "inputs": ["a3"], "op": "a3_map", "params": {"axis": "limit"}}]}}
    answers = {"lb2_formalize": iter(['[{"date": "10/20/2025", "kind": "Blue"}, {"date": "10/25/2025", "kind": "Blue"}]', "10/26/2025"])}
    ask = lambda prompt, name: next(answers[name])
    facts = derived_facts(a2, {"get_refs_1": "raw"}, ask, a3_rows=[{"axis": "limit", "subject": "Blue", "value": 2}])
    assert [t for _o, _v, t in facts] == ["in window 2; 0 more allowed", "no room: Blue"], facts
    # computations
    A2 = {"dispatch": {"agent_call": "call", "name_args": {"call": "tool"}},
          "LB2": {"computations": [{"kind": "distinct", "tool": "w", "pairs": [["x", "y"]], "feedback": "{a} equals {b}"}]}}
    assert evaluate(Turn(A2, [], M(calls=[C("w", {"x": 5, "y": 5})])))[0].order == "x equals y"
    print("lb2_decision self-test OK")
