# -*- coding: utf-8 -*-
"""도메인-일반 compute op 라이브러리 (2026-07-13·사용자 아키텍처·[[05]]/[[11]] keystone).

★설계: 엔진 = 일반 op 라이브러리 + A2-스펙 인터프리터. A2가 도메인별로 '어느 도구의 어느
파라미터를 어느 op·어느 임계값으로 계산하는가'를 선언(도메인 데이터). scaffold/loop 소스 불변.
전이(retail↔banking) = A2 compute_ops만 교체.

op ∈ {const, ref, min, max, argmin, argmax, sum, count_where, diff, clamp, lookup_table,
      days_between, if_then}. 값은 ctx(수집 record·인자·타 파라미터)서 ref로 참조. 리터럴 0.

apply_op(spec, ctx) → 계산값 (실패=None·안전). ctx = {"args": 이번호출인자, "records": 조회record list,
  "params": 발견도구 nested 인자, "user": 사용자 발화값}. ref="params.disputed_amount" 형 경로."""
import re


def _get(ctx, path):
    """ref 경로 해소: 'params.disputed_amount' · 'args.x' · 'records[*].field' · 리터럴."""
    if not isinstance(path, str):
        return path                       # 리터럴(숫자 등)
    if not re.match(r"^[a-zA-Z_]", path):
        try:
            return float(path)
        except Exception:
            return path
    cur = ctx
    for part in path.split("."):
        m = re.match(r"([a-zA-Z_0-9]+)(\[\*\])?$", part)
        if not m:
            return None
        key = m.group(1)
        cur = cur.get(key) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def _num(v):
    try:
        return float(v)
    except Exception:
        return None


def _days_between(a, b):
    from datetime import datetime
    def p(x):
        for f in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
            try:
                return datetime.strptime(str(x), f)
            except Exception:
                pass
        return None
    da, db = p(a), p(b)
    if da is None or db is None:
        return None
    return abs((db - da).days)


def apply_op(spec, ctx):
    """A2 op-스펙을 ctx 위에서 실행. 반환 계산값 or None."""
    if not isinstance(spec, dict):
        return None
    op = spec.get("op")
    try:
        if op == "const":
            return spec.get("value")
        if op == "ref":
            return _get(ctx, spec.get("path"))
        if op in ("min", "max"):
            vals = [_num(_get(ctx, r)) for r in (spec.get("of") or [])]
            vals = [v for v in vals if v is not None]
            return (min if op == "min" else max)(vals) if vals else None
        if op == "sum":
            vals = [_num(_get(ctx, r)) for r in (spec.get("of") or [])]
            return sum(v for v in vals if v is not None)
        if op == "diff":
            a, b = _num(_get(ctx, spec.get("a"))), _num(_get(ctx, spec.get("b")))
            return None if (a is None or b is None) else a - b
        if op == "clamp":
            v = _num(_get(ctx, spec.get("value")))
            lo = _num(_get(ctx, spec.get("min"))) if spec.get("min") is not None else None
            hi = _num(_get(ctx, spec.get("max"))) if spec.get("max") is not None else None
            if v is None:
                return None
            if lo is not None:
                v = max(v, lo)
            if hi is not None:
                v = min(v, hi)
            return v
        if op == "days_between":
            return _days_between(_get(ctx, spec.get("a")), _get(ctx, spec.get("b")))
        if op in ("argmin", "argmax"):
            recs = _get(ctx, spec.get("over")) or []
            key = spec.get("key"); ret = spec.get("return")
            if not isinstance(recs, list) or not recs:
                return None
            best = (min if op == "argmin" else max)(
                recs, key=lambda r: _num((r or {}).get(key)) if _num((r or {}).get(key)) is not None else float("inf"))
            return best.get(ret) if ret else best
        if op == "count_where":
            recs = _get(ctx, spec.get("over")) or []
            cf, cv = spec.get("cond_field"), spec.get("cond_value")
            return sum(1 for r in recs if isinstance(r, dict) and r.get(cf) == cv)
        if op == "lookup_table":
            # 조건 리스트를 순서대로 평가 → 첫 매칭의 result. cond = {ref, op, value} 또는 days 비교.
            key = apply_op(spec.get("key"), ctx) if isinstance(spec.get("key"), dict) \
                else _get(ctx, spec.get("key"))
            def _res(row):
                r = row.get("result")
                return apply_op(r, ctx) if isinstance(r, dict) and r.get("op") else r
            for row in (spec.get("table") or []):
                cmp, thr = row.get("cmp"), row.get("thr")
                kn = _num(key)
                if cmp is None:
                    return _res(row)                        # default(마지막)
                if kn is None:
                    continue
                if (cmp == "<=" and kn <= thr) or (cmp == "<" and kn < thr) \
                        or (cmp == ">=" and kn >= thr) or (cmp == ">" and kn > thr) \
                        or (cmp == "==" and kn == thr):
                    return _res(row)
            return None
        if op == "if_then":
            cond = apply_op(spec.get("cond"), ctx)
            return apply_op(spec.get("then"), ctx) if cond else apply_op(spec.get("else"), ctx)
    except Exception:
        return None
    return None
