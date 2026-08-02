# -*- coding: utf-8 -*-
"""도메인-일반 compute op 라이브러리 (2026-07-13·사용자 아키텍처·[[05]]/[[11]] keystone).

★설계: 엔진 = 일반 op 라이브러리 + A2-스펙 인터프리터. A2가 도메인별로 '어느 도구의 어느
파라미터를 어느 op·어느 임계값으로 계산하는가'를 선언(도메인 데이터). scaffold/loop 소스 불변.
전이(retail↔banking) = A2 compute_ops만 교체.

op ∈ {const, ref, min, max, argmin, argmax, sum, count_where, diff, clamp, lookup_table,
      days_between, if_then, bool_expr, filter}. 값은 ctx(수집 record·인자·타 파라미터)서 ref로 참조. 리터럴 0.

apply_op(spec, ctx) → 계산값 (실패=None·안전). ctx = {"args": 이번호출인자, "records": 조회record list,
  "params": 발견도구 nested 인자, "user": 사용자 발화값}. ref="params.disputed_amount" 형 경로."""
import os
import re
import sys as _sys


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


def _parse_date(x):
    from datetime import datetime
    for f in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(str(x).split()[0] if x else x, f)
        except Exception:
            pass
    return None


def _days_between(a, b):
    da, db = _parse_date(a), _parse_date(b)
    if da is None or db is None:
        return None
    return abs((db - da).days)


def _add_months(d, m):
    """달력 산술(도메인일반): d + m개월. 월말 clamp(예: 1/31+1mo=2/28)."""
    from calendar import monthrange
    y, mo = d.year + (d.month - 1 + int(m)) // 12, (d.month - 1 + int(m)) % 12 + 1
    return d.replace(year=y, month=mo, day=min(d.day, monthrange(y, mo)[1]))


def _month_window_index(anchor, target):
    """★023 리베이트: target(posting date)이 anchor(개설일) 기준 몇 번째 월별-윈도우(0-based)인가.
    윈도우 = [anchor+k월, anchor+(k+1)월). 달력 산술만(도메인일반·KB의 '월별 anniversary 윈도우' 규칙).
    범위 밖(target<anchor)=None. cardmember year=0..11만 유효(호출측이 필터)."""
    da, dt = _parse_date(anchor), _parse_date(target)
    if da is None or dt is None or dt < da:
        return None
    k = (dt.year - da.year) * 12 + (dt.month - da.month)
    if _add_months(da, k) > dt:      # 일(day) 보정: anniversary일 이전이면 한 윈도우 앞
        k -= 1
    return k


def _date_in_window(anchor, target, months):
    """target ∈ [anchor, anchor+months] ? (결정론·`C113` rate 만료판정).
    ★C197: 입력 누락/파싱실패=None(미확정) — False로 강제하면 결여 leaf가 **침묵 오판정**이 된다
    (020 실측: account_open 누락→promo false 강제→8 FP+1 FN). None은 if_then/bool_expr의
    3-값 논리로 전파되어 해당 행이 '판정불가'로 계상된다(coverage에 표면화)."""
    da, dt = _parse_date(anchor), _parse_date(target)
    if da is None or dt is None or months is None:
        return None
    return da <= dt <= _add_months(da, months)


def _date_between(x, lo, hi):
    dx, dl, dh = _parse_date(x), _parse_date(lo), _parse_date(hi)
    if None in (dx, dl, dh):
        return None                       # ★C197: 미확정=None(위와 동일 원칙)
    return dl <= dx <= dh


def _business_days_between(a, b):
    """주말(토/일) 제외 영업일 수. Reg E '2 business days' 계산용(도메인일반·달력 사실)."""
    da, db = _parse_date(a), _parse_date(b)
    if da is None or db is None:
        return None
    from datetime import timedelta
    lo, hi = (da, db) if da <= db else (db, da)
    n = 0; cur = lo
    while cur < hi:
        cur += timedelta(days=1)
        if cur.weekday() < 5:                            # 0-4 = 월-금
            n += 1
    return n


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
            # of = ref 경로 or nested op-스펙(multiply/lookup_table 동형). 미확정(None)은 0 취급.
            vals = [_num(apply_op(r, ctx)) if isinstance(r, dict) and r.get("op")
                    else _num(_get(ctx, r)) for r in (spec.get("of") or [])]
            return sum(v for v in vals if v is not None)
        if op == "bucket_month_window":
            # ★023: 거래에 월별-윈도우 인덱스 태그(개설일 기준). group_reduce가 이 필드로 group_by.
            #   달력 산술=엔진(LLM 날짜계산 약점 회피·095). anchor=개설일·date_field=posting date·
            #   out_field=태그명·within_year=True면 0..11만(카드멤버 연도). 반환=태그된 record list.
            recs = _get(ctx, spec.get("over"))
            if not isinstance(recs, list):
                return None
            anchor = _get(ctx, spec.get("anchor"))
            df = spec.get("date_field", "date")
            of = spec.get("out_field", "window")
            within = spec.get("within_year", True)
            # ★year_select="last_complete" (2026-07-20·§2ak·as_of판 — [[05]] 회색지대 교정): "cardmember
            #   year"는 **연도별** 평가(KB doc_010)인데 구판은 개설 **첫해(0..11) 고정** — 개설 2022·거래
            #   2024-25(k=24..35)면 전량 드롭→빈 집계→오판정(smoke023b 실측·QUALIFIES를 DOES NOT로).
            #   평가 연도 = **as_of(현재시각·에이전트가 copy·grounding 가능) 기준 마지막 완결 기념년**
            #   Y=k(as_of)//12-1 — 데이터-의존 선택("거래 있는 최근 연도") 배제: 최근 연도 거래 0건이면
            #   빈 윈도우가 **보이는** 채로 평가된다(조용한 옛-연도 평가 방지). 달력 산술만(도메인일반)·
            #   as_of 미제공/파싱불가=None(abstain·3-값)·아직 완결 연도 없음(Y<0)=[]. 미선언=기존 거동보존.
            ysel = spec.get("year_select")
            # ★2026-08-03 §4-2(핸드오프 §4-2 "표본 충분성 미검"): 이 op은 **한 기념년의 완결 윈도
            #   집합**을 안다(달력 산술 — 12개월·도메인 리터럴 아님·아래 within 분기의 0..11과 동일
            #   근거). 그 기대 집합을 ctx 사이드채널로 공표해 group_reduce가 "빈 윈도=미측정"을
            #   판별할 수 있게 한다(구판은 1건만 받아도 그 1건의 min으로 **부정 판정을 발급**했다).
            #   공표는 사실 전달일 뿐 판정을 바꾸지 않는다(소비 측이 A2 선언으로 옵트인).
            if isinstance(ctx, dict) and (ysel == "last_complete" or within):
                ctx["_expected_groups"] = {"field": of, "labels": list(range(12))}
            tagged = []
            for r in recs:
                if not isinstance(r, dict):
                    continue
                k = _month_window_index(anchor, r.get(df))
                if k is None:
                    continue
                tagged.append((k, r))
            if ysel == "last_complete":
                k_asof = _month_window_index(anchor, _get(ctx, spec.get("as_of")))
                if k_asof is None:
                    return None                      # as_of 부재/불가 → abstain(추측 금지)
                yy = k_asof // 12 - 1
                if yy < 0:
                    return []                        # 첫 기념년 미완 → 평가할 완결 연도 없음
                return [dict(r, **{of: k % 12}) for k, r in tagged if k // 12 == yy]
            return [dict(r, **{of: k}) for k, r in tagged
                    if not (within and not (0 <= k <= 11))]
        if op == "group_reduce":
            # ★도메인-일반 프리미티브 (ACCOUNT_APY_OFFLOAD §2-0·리뷰① — apy_argmax/interest_delta를
            #   1개로 대체). 항목을 group_by로 묶고, 그룹별 A2-선언 reducer(max1|sum) 적용 후 총합.
            #   이름·상수 도메인 리터럴 0 — 엔진은 combinator 2개(max1/sum)만 안다. stack_rules 규칙표=A2.
            #   ★unknown-kind(리뷰④): reducers에 없는 group → **silent drop 금지**·미합성 + ctx["_gr_flags"]에
            #     플래그 push(scaffold `_sg_details` 동형 side-channel·[[03b]] 인접 회피).
            _ov = spec.get("over")
            items = apply_op(_ov, ctx) if isinstance(_ov, dict) and _ov.get("op") else _get(ctx, _ov)
            if not isinstance(items, list):
                return None
            gkey = spec.get("group_by"); vfield = spec.get("value_field")
            reducers = spec.get("reducers") or {}
            unknown_policy = spec.get("unknown_policy", "flag")
            # ★across (2026-07-20·023 리베이트): 그룹별 reduce값을 그룹-간 어떻게 합치나.
            #   "sum"(기본·APY 스택=base+Σ그룹) · "min"/"max"(023="모든 월≥임계"→월별합의 min).
            #   default_reducer = reducers에 없는 그룹의 기본 처리("sum"이면 자동합·미지정이면 flag).
            across = spec.get("across", "sum")
            default_reducer = spec.get("default_reducer")
            groups = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                g = str(it.get(gkey))
                v = _num(it.get(vfield))
                if v is None:
                    continue
                groups.setdefault(g, []).append(v)
            flags = ctx.setdefault("_gr_flags", []) if isinstance(ctx, dict) else []
            # ★required_groups (2026-07-20 §2as·095 0.0-포이즈닝): 핵심 그룹(예: APY의 base)이
            #   드롭/부재면 합계 "0.0%"는 **유해한 오판정**(에이전트 도구불신 유발·실측 msg25/31) —
            #   None(abstain·missing_hint 경로)으로. A2 선언 시만·미선언=거동보존. 도메인 리터럴 0.
            for _rg in (spec.get("required_groups") or []):
                if str(_rg) not in groups:
                    return None
            # ★2026-08-03 §4-2 (023 사슬의 마지막 고리·T2_SG_WINDOW_ABSTAIN=1 ∧ A2 선언 시만):
            #   **결여 그룹은 0이 아니라 미측정**이다. 023 라이브: byref 차단(§4-1) → 손-전사 1건 →
            #   12칸 중 11칸이 빈 채로 across=min이 그 1건을 최솟값으로 삼아 "DOES NOT QUALIFY"를
            #   **확정 발급**했다(정답 QUALIFIES). C197의 원칙("결여 leaf가 침묵 오판정")이 이
            #   경로에만 미적용이었다 — 여기서 동형화한다: 기대 윈도가 하나라도 비면 abstain +
            #   무엇이 비었는지 표면화(호출측이 `_gr_missing`을 읽어 문구를 붙인다).
            #   ⚠상쇄(등대 §1 모트): 진짜로 지출 0인 달도 abstain이 된다 — 그 경우의 결론은
            #   표면화 문구가 A2 문안으로 이행한다(정보 손실 0·판단은 모델 몫).
            _exp = (ctx or {}).get("_expected_groups") if isinstance(ctx, dict) else None
            if (spec.get("require_complete_groups")
                    and os.environ.get("T2_SG_WINDOW_ABSTAIN") == "1"
                    and isinstance(_exp, dict) and _exp.get("field") == gkey):
                _labels = list(_exp.get("labels") or [])
                _miss = [l for l in _labels if str(l) not in groups]
                if _miss:
                    ctx["_gr_missing"] = {"field": gkey, "missing": _miss,
                                          "expected": len(_labels), "present": len(groups)}
                    print("[T2_SG_WINDOW_ABSTAIN] group_reduce: %d/%d groups had no data -> abstain"
                          % (len(_miss), len(_labels)), file=_sys.stderr, flush=True)
                    return None
            reduced = []
            for g, vs in groups.items():
                red = reducers.get(g, default_reducer)
                if red == "max1":
                    reduced.append(max(vs))
                elif red == "sum":
                    reduced.append(sum(vs))
                else:                                   # unknown group — 미합성 + 플래그
                    if unknown_policy == "flag":
                        flags.append(g)
            if not reduced:
                return None if across in ("min", "max") else 0.0
            if across == "min":
                return min(reduced)
            if across == "max":
                return max(reduced)
            return sum(reduced)
        if op == "diff":
            _a, _b = spec.get("a"), spec.get("b")       # a/b = ref 경로 or nested op-스펙(multiply 동형)
            a = _num(apply_op(_a, ctx)) if isinstance(_a, dict) and _a.get("op") else _num(_get(ctx, _a))
            b = _num(apply_op(_b, ctx)) if isinstance(_b, dict) and _b.get("op") else _num(_get(ctx, _b))
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
            f = _business_days_between if spec.get("business") else _days_between
            return f(_get(ctx, spec.get("a")), _get(ctx, spec.get("b")))
        if op == "date_in_window":
            # ★C113: target ∈ [anchor, anchor+months] (rate 프로모 만료판정·결정론).
            #   anchor/target/months = ref(LLM formalize한 거래 필드) · 엔진=달력산술만([[05]]).
            return _date_in_window(_get(ctx, spec.get("anchor")), _get(ctx, spec.get("target")),
                                   _num(_get(ctx, spec.get("months"))))
        if op == "date_between":
            return _date_between(_get(ctx, spec.get("x")), _get(ctx, spec.get("lo")), _get(ctx, spec.get("hi")))
        if op in ("argmin", "argmax"):
            recs = _get(ctx, spec.get("over")) or []
            key = spec.get("key"); ret = spec.get("return")
            if not isinstance(recs, list) or not recs:
                return None
            # key = record 필드명(str) or per-record op-스펙(dict). op-스펙이면 각 record를 ctx["r"]로
            #   바인딩해 평가(base + group_reduce(boosts) 형 합성 지원·ACCOUNT_APY §2a). 미확정=±inf(제외).
            def _kv(r):
                if isinstance(key, dict) and key.get("op"):
                    v = _num(apply_op(key, {**ctx, "r": r}))
                else:
                    v = _num((r or {}).get(key))
                if v is None:
                    return float("inf") if op == "argmin" else float("-inf")
                return v
            best = (min if op == "argmin" else max)(recs, key=_kv)
            return best.get(ret) if (ret and isinstance(best, dict)) else best
        if op == "count_where":
            recs = _get(ctx, spec.get("over")) or []
            cf, cv = spec.get("cond_field"), spec.get("cond_value")
            return sum(1 for r in recs if isinstance(r, dict) and r.get(cf) == cv)
        if op == "catalog_filter":
            # ★C181(2026-07-25·C179 처방): A2-선언 카탈로그(spec.table=행 리스트)를 모델이
            #   formalize해 넘긴 제약(ctx)으로 결정론 필터([[10]]: formalize=LLM·실행=엔진).
            #   도메인-일반: 행 스키마·값은 전부 A2. 제약 키(있을 때만 적용):
            #   max_annual_fee/max_fx_fee/max_min_payment_pct(이하)·min_cashback(이상)·
            #   needs_virtual_card(참이면 행 virtual_card 참)·credit_score(행 min_score 이상)·
            #   business(행 business와 일치·미지정시 비즈니스 행 제외).
            #   invite_only 행은 ctx.invited 참 아니면 항상 제외(문서 사실).
            # ★C185(a)(2026-07-25·C184 실측 결함 교정·최우선): **결측 필드는 제약을 통과시키지
            #   않는다.** 구판은 (제약 있음 ∧ 행에 그 사실 없음)을 `continue`로 넘겨 카드를
            #   eligible에 실어, 결정론 도구가 "충족한다"고 **사실상 단언**했다 — W5 002 실측:
            #   `min_cashback=5` 호출에 cashback 필드 없는 Silver/EcoCard/Crypto가 eligible로
            #   통과 → 에이전트가 "Silver=5% cash back"(KB 정본 4%/1% = **거짓**)을 손님에게
            #   말하고 손님이 복창·gold(Platinum) 대신 Silver 신청. **우리 레버가 거짓 사실을
            #   생성**한 사례(등대 §1.3 게이트 자기-역효과). 교정 = 3버킷:
            #     eligible   문서화된 값이 제약을 충족
            #     excluded   문서화된 값이 제약을 위반(사유 명시)
            #     unverified **그 사실이 표에 없음** → 충족 주장 금지·원문 확인 지시
            #   (배제가 아니라 미검증으로 두는 이유: 하드-배제는 gold를 떨어뜨릴 위험·003의
            #    조건부 조항 문제와 동일 계열. 최종 선택은 여전히 모델 몫.)
            rows = spec.get("table") or []
            biz = bool(ctx.get("business"))
            elig, excl, unver = [], [], []
            # ★C204/D6 rev2 결함1: 토큰이 **어느 행의 category_rates 키에도 없으면** 행별 base 폴백
            #   주석이 정답 카드 요율을 깎아 보이게 한다(동의어 'flights'→Silver 1.0% 오표기) —
            #   그 경우 행별 주석을 전부 생략하고 정직한 안내만 낸다(§2-3b).
            _spc0 = str(ctx.get("spend_category") or "").strip().lower()
            _spc_known = any(_spc0 in {str(k).lower() for k in (r.get("category_rates") or {})}
                             for r in rows) if _spc0 else False
            for row in rows:
                name = row.get("card")
                why = None
                missing = []
                if bool(row.get("business")) != biz:
                    continue                                  # 세그먼트 불일치=조용히 스킵
                if row.get("invite_only") and not ctx.get("invited"):
                    why = "invitation-only"
                # ★C187(c): 조건부 값 — A2 `conditional_fields`가 (기본 필드 → 대체 필드, 조건
                #   파라미터)를 선언한다(예: fx_fee → fx_fee_with_premium, when=premium_subscriber).
                #   003 실측: Silver의 fx는 무구독 2.75%/**premium 구독 시 0%**인데 구판은 무조건부
                #   2.75만 보고 **gold를 하드 배제**했다. 조건이 참=대체값 / 거짓=기본값 /
                #   **미지=두 값의 판정이 갈리면 unverified**(조항 인용·배제 금지).
                cond = spec.get("conditional_fields") or {}
                for ck, rk, sense in (("max_annual_fee", "annual_fee", "le"),
                                      ("max_fx_fee", "fx_fee", "le"),
                                      ("max_min_payment_pct", "min_payment_pct", "le"),
                                      ("min_cashback", "cashback", "ge"),
                                      ("min_credit_limit", "limit_max", "ge")):
                    if why:
                        break
                    cv, rv = _num(ctx.get(ck)), _num(row.get(rk))
                    if cv is None:
                        continue                              # 모델이 제약 미지정=미적용
                    cspec = cond.get(rk) or {}
                    av = _num(row.get(cspec.get("alt"))) if cspec else None
                    wv = ctx.get(cspec.get("when")) if cspec else None
                    if av is not None and wv is not None:
                        rv = av if bool(wv) else rv            # 조건 확정 → 해당 값 사용
                        av = None
                    if rv is None and av is None:
                        missing.append("%s (constraint %s=%s)" % (rk, ck, cv))
                        continue                              # ★결측=통과 아님(unverified로)
                    def _bad(x):
                        return x is not None and ((sense == "le" and x > cv)
                                                  or (sense == "ge" and x < cv))
                    if av is not None and _bad(rv) != _bad(av):
                        missing.append("%s depends on %s (documented %s=%s, %s=%s) — confirm it"
                                       % (rk, cspec.get("when"), rk, rv, cspec.get("alt"), av))
                        continue                              # 조건 미지 & 판정 갈림 → unverified
                    if _bad(rv) if av is None else (_bad(rv) and _bad(av)):
                        why = "%s=%s violates %s=%s" % (rk, rv, ck, cv)
                for ck, rk in (("needs_virtual_card", "virtual_card"),
                               ("needs_purchase_protection", "purchase_protection")):
                    if why or not ctx.get(ck):
                        continue
                    rv = row.get(rk)
                    if rv is None:
                        missing.append("%s (constraint %s)" % (rk, ck))   # 결측=통과 아님
                    elif not rv:
                        why = "%s is documented as not available" % rk
                if not why:
                    cs, ms = _num(ctx.get("credit_score")), _num(row.get("min_score"))
                    if cs is not None and ms is not None and cs < ms:
                        why = "min_score %s > customer %s" % (ms, cs)
                facts = {k: v for k, v in row.items() if k not in ("card", "business")}
                # ★2026-08-03 (task_001 실측): 손님이 "일상 지출 최고 캐시백"을 물으면 모델은
                #   헤드라인 `cashback`으로 카드를 고르는데, 그 값이 **범주-한정**일 수 있다
                #   (Silver 4.0%는 travel/software 전용·전-구매는 1.0%). 001에서 모델이 Silver를
                #   추천했고 gold는 Gold(2.5% 전-구매)였다 — 사실은 전부 표에 있었지만 **비교 가능한
                #   숫자가 없었다**. ⇒ 표에서 파생되는 "전-구매 기준 요율"을 명시한다.
                #   순수 조회·판단 0(값 생성·순위 0)·도메인 리터럴 0(필드명은 A2 표의 키).
                _scope = str(row.get("cashback_scope") or "").strip().lower()
                _allrate = (row.get("base_cashback") if _scope and _scope != "all"
                            else (row.get("cashback") if row.get("cashback") is not None
                                  else row.get("base_cashback")))
                if _allrate is not None:
                    facts["all_purchases_rate"] = (
                        "%s%%%s" % (_allrate,
                                    "" if (not _scope or _scope == "all")
                                    else " (the headline rate applies only to: %s)" % _scope))
                # ★C204/D6(2026-07-27·`D6_CATEGORY_RATE_DESIGN` rev2): 손님이 말한 지출 카테고리
                #   (`spend_category`=모델 formalize·선택적)에 대한 **그 카드의 문서화 요율**을 주석.
                #   조회만 한다(값 생성·선택·순위 0): category_rates[token] → base_cashback →
                #   cashback(전-구매) → 'unverified'. 미제공=주석 자체가 없음(거동 변화 0).
                _spc = _spc0 if _spc_known else ""
                if _spc:
                    _cr = row.get("category_rates") or {}
                    _crl = {str(k).lower(): v for k, v in _cr.items()} if isinstance(_cr, dict) else {}
                    if _spc in _crl:
                        _rate = "%s%% (other categories: %s%%)" % (
                            _crl[_spc], row.get("base_cashback")) \
                            if row.get("base_cashback") is not None else "%s%%" % _crl[_spc]
                    elif _crl and row.get("base_cashback") is not None:
                        _rate = "%s%% (base rate — its documented bonus categories do not "\
                                "include this one)" % row.get("base_cashback")
                    elif row.get("base_cashback") is not None:
                        _rate = "%s%% (all purchases)" % row.get("base_cashback")
                    elif row.get("cashback") is not None and not _crl:
                        _rate = "%s%% (all purchases)" % row.get("cashback")
                    else:
                        _rate = "unverified — no documented rate for this category"
                    facts["rate_for('%s')" % _spc] = _rate
                if why:
                    excl.append({"card": name, "reason": why})
                elif missing:
                    unver.append({"card": name, "undocumented": missing, "facts": facts,
                                  "source": row.get("source")})
                else:
                    elig.append({"card": name, "facts": facts, "source": row.get("source")})
            # ★C189: eligible 공집합 안내 — (a)+(c)로 결측/조건부가 통과하지 않게 되면서 **과잉
            #   형식화 시 eligible이 빌 수 있다**(024 실측: 손님은 "$40k 구매에 최고 수익"만 말했는데
            #   모델이 min_cashback=2를 발명 → gold(Business Bronze 1.0%) 배제 → 신판에선 공집합).
            #   공집합을 "맞는 카드 없음"으로 오독해 사임하는 것이 다음 실패 모드이므로 대처를 명시.
            empty = ("" if elig else
                     " NOTE: no card is in 'eligible'. Do NOT conclude that nothing fits: re-check "
                     "which limits the CUSTOMER actually stated (do not invent thresholds they did "
                     "not give) and call again, and/or read the source docs of the 'unverified' "
                     "candidates.")
            # ★C204/D6 rev2: 미등재 카테고리 토큰 = 오표기 대신 정직 안내(문서화 어휘로 재호출 지시).
            if _spc0 and not _spc_known:
                empty += (" NOTE: no documented category-specific rate exists for spend_category="
                          "'%s' — it matches none of the documented category keywords; re-call "
                          "with one of the documented keywords (see the tool's parameter "
                          "description) or omit spend_category." % _spc0)
            return {"eligible": elig, "excluded": excl, "unverified": unver,
                    "note": (empty + " When the customer asks for the best rate on everyday or "
                             "general spending, compare 'all_purchases_rate' across the cards — a "
                             "headline rate that applies only to certain categories is NOT "
                             "comparable to an all-purchases rate. Deterministic filter over documented facts. 'eligible' = the "
                             "documented value satisfies your constraint. 'excluded' = it "
                             "violates it. 'unverified' = that fact is NOT in the catalog for "
                             "that card, so it is NOT known to satisfy your constraint — do not "
                             "claim that it does; read the cited source docs first if you want "
                             "to consider it. Some fees/rates also have CONDITIONAL clauses "
                             "(e.g. subscriber discounts) not in this table — before the final "
                             "choice, verify the cited source docs for each remaining candidate "
                             "and match the customer's stated preferences.")}
        if op == "lookup_table":
            # 조건 리스트를 순서대로 평가 → 첫 매칭의 result. cond = {ref, op, value} 또는 days 비교.
            key = apply_op(spec.get("key"), ctx) if isinstance(spec.get("key"), dict) \
                else _get(ctx, spec.get("key"))
            if key is None:
                return None                                 # ★입력 미확정 → abstain(default 행은 key 계산됐으나 미매칭일 때만)
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
            # ★3-값 논리(2026-07-20 관문3 자기감사): cond=None(미확정)은 False가 **아니다** — else로 새면
            #   드롭/미확정 입력에 **오판정**을 낸다(023 기전의 엔진-내 재현: grounding이 개설일 드롭→
            #   빈 버킷→min None→compare None→구판 else="DOES NOT QUALIFY"). bool_expr의 명시 철학
            #   (미확정=None=abstain·§3 안전)과 동형화. None이면 missing_hint 경로로 abstain.
            cond = apply_op(spec.get("cond"), ctx)
            if cond is None:
                # ★C197 정련: cond 미확정이어도 then/else가 **같은 값**이면 cond와 무관 → 그 값.
                #   (무-프로모 행: then=r.promo_mult=1 == else=1 → 1. 값이 갈리면 기존대로 abstain.)
                _tv = apply_op(spec.get("then"), ctx)
                _ev = apply_op(spec.get("else"), ctx)
                return _tv if _tv == _ev else None
            return apply_op(spec.get("then"), ctx) if cond else apply_op(spec.get("else"), ctx)
        if op == "bool_expr":
            # ★정책 불리언식(도메인일반·3-값 논리). all/any/not 트리 + leaf(ref|expr + eq|in|비교).
            #   미확정(값 None)=None 반환(abstain·§3 안전). value 있는 조건만 판정.
            def _leaf(cond):
                v = apply_op(cond["expr"], ctx) if "expr" in cond else \
                    (_get(ctx, cond["ref"]) if "ref" in cond else None)
                if v is None:
                    return None
                if "in" in cond:
                    return v in cond["in"]
                for c in ("<=", ">=", "<", ">"):
                    if c in cond:
                        nv = _num(v)
                        if nv is None:
                            return None
                        return {"<=": nv <= cond[c], ">=": nv >= cond[c],
                                "<": nv < cond[c], ">": nv > cond[c]}[c]
                if "eq" in cond:
                    def nb(x):
                        x = str(x).strip().lower()
                        return "true" if x in ("true", "yes") else ("false" if x in ("false", "no") else x)
                    return nb(v) == nb(cond["eq"])
                return bool(v)
            def _ev(cond):
                if "all" in cond:
                    vs = [_ev(c) for c in cond["all"]]
                    return False if False in vs else (None if None in vs else True)
                if "any" in cond:
                    vs = [_ev(c) for c in cond["any"]]
                    return True if True in vs else (None if None in vs else False)
                if "not" in cond:
                    v = _ev(cond["not"]); return None if v is None else (not v)
                return _leaf(cond)
            return _ev(spec)
        if op == "multiply":
            a = apply_op(spec["a"], ctx) if isinstance(spec.get("a"), dict) else _num(_get(ctx, spec.get("a")))
            b = apply_op(spec["b"], ctx) if isinstance(spec.get("b"), dict) else _num(_get(ctx, spec.get("b")))
            a, b = _num(a), _num(b)
            return None if (a is None or b is None) else a * b
        if op == "compare":
            # ★두 런타임 값 비교(도메인일반·023 자격판정 등): a cmp b → bool. a/b=ref or nested op.
            _a, _b = spec.get("a"), spec.get("b")
            a = _num(apply_op(_a, ctx)) if isinstance(_a, dict) and _a.get("op") else _num(_get(ctx, _a))
            b = _num(apply_op(_b, ctx)) if isinstance(_b, dict) and _b.get("op") else _num(_get(ctx, _b))
            if a is None or b is None:
                return None
            c = spec.get("cmp", ">=")
            return {">=": a >= b, ">": a > b, "<=": a <= b, "<": a < b, "==": a == b}.get(c)
        if op == "str_eq":
            a = _get(ctx, spec.get("a")); b = spec.get("b")
            return None if a is None else (str(a).strip().lower() == str(b).strip().lower())
        if op == "count_field_matches":
            # ★도메인일반: 두 dict(a,b)서 fields 각각 정규화 후 일치하는 개수. (verify=정책 2-of-N 대조·[[10]] leaf=LLM/read)
            a = _get(ctx, spec.get("a")); b = _get(ctx, spec.get("b"))
            if not isinstance(a, dict) or not isinstance(b, dict):
                return 0
            def _nm(x):
                return re.sub(r"\s+", " ", str(x)).strip().lower()
            n = 0
            for f in (spec.get("fields") or []):
                av, bv = a.get(f), b.get(f)
                if av not in (None, "") and bv not in (None, "") and _nm(av) == _nm(bv):
                    n += 1
            return n
        if op == "match_verdict":
            # ★도메인일반(§17.3·2026-07-17): count_field_matches의 확장 — 몇 개·어느 필드가 일치/미일치인지까지
            #   결정론으로 세서 A2 템플릿({count}/{threshold}/{matched}/{missing})에 포맷. 산술·집합연산을
            #   모델에게 남기지 않음(sim0형 카운트-자기모순 = closed-gap의 INFER 오분류·C92). 문구=A2·엔진 리터럴 0.
            a = _get(ctx, spec.get("a")); b = _get(ctx, spec.get("b"))
            fields = list(spec.get("fields") or [])
            thr = int(spec.get("threshold", 2))
            def _nm2(x):
                return re.sub(r"\s+", " ", str(x)).strip().lower()
            matched, missing = [], []
            if isinstance(a, dict) and isinstance(b, dict):
                for f in fields:
                    av, bv = a.get(f), b.get(f)
                    if av not in (None, "") and bv not in (None, "") and _nm2(av) == _nm2(bv):
                        matched.append(f)
                    else:
                        missing.append(f)
            else:
                missing = fields
            tpl = spec.get("met_template" if len(matched) >= thr else "unmet_template") or "{count}"
            try:
                return tpl.format(count=len(matched), threshold=thr,
                                  matched=", ".join(matched) or "(none)",
                                  missing=", ".join(missing) or "(none)")
            except Exception:
                return str(len(matched))
        if op == "match_verdict_grounded":
            # ★원장-결합 verdict (`VERIFY_IDENTITY_LEDGER_BINDING_DESIGN_2026_07_18`·2026-07-18).
            #   `match_verdict`의 대체: **`b`(=LLM이 준 record)를 안 받는다.** 그 슬롯이 곧 날조 표면이었다
            #   (record 날조 46%·grounded 0/24·A2 설명레버로도 0/24 — LLM이 주장과 증거를 둘 다 공급 = 비교가 공허).
            #   대신 필드값이 ①**user 발화**에 등장(고객이 실제로 말함) ∧ ②**선언된 producer 출력**에 등장
            #   (계정 기록과 일치)할 때만 매치로 센다 ⇒ **날조 슬롯 소멸 + 조회 강제**(출력 없으면 0 match).
            # ★[[03b]]: 엔진은 **추출/파싱을 하지 않는다** — operand는 LLM이 formalize해 인자로 주고,
            #   엔진은 "이 문자열이 저 영수증에 등장하나"만 묻는다(=PROV의 `_ctx_has` **동일 술어** 재사용).
            # ★[[05]]: 어느 도구가 record producer인가 = **A2 `evidence_from`**. 엔진에 도메인 리터럴 0.
            #   ctx의 `__user_text`/`__tool_outputs` = scaffold가 원장(실제 호출 이력)서 주입.
            from t2_gate_patch import _ctx_has     # 지연 import(순환 회피)·술어 단일화([[03b]])
            a = _get(ctx, spec.get("a"))
            fields = list(spec.get("fields") or [])
            thr = int(spec.get("threshold", 2))
            ev = [str(x) for x in (spec.get("evidence_from") or [])]
            utext = str((ctx or {}).get("__user_text") or "")
            outs = (ctx or {}).get("__tool_outputs") or {}
            rtext = " ".join(str(outs.get(t) or "") for t in ev).strip()
            matched, missing = [], []
            if isinstance(a, dict):
                for f in fields:
                    v = a.get(f)
                    s = "" if v is None else str(v).strip()
                    # len<4 = 우연 부분일치 방지(`_first_fab_call`과 동일 관례)
                    if len(s) >= 4 and _ctx_has(s, utext) and _ctx_has(s, rtext):
                        matched.append(f)
                    else:
                        missing.append(f)
            else:
                missing = list(fields)
            if not rtext:            # 아직 조회 안 함 → 날조가 아니라 **순서**의 문제 = 별도 문구(A2)
                # ★D1/D1b(INSTRUCTION_DEFECT §2b·2c · T2_NOREC_BRANCH=1): 현행 문구는 전제(조회 성공)를
                #   가정하고 "then call this tool again"으로 닫아 **종료 분기가 없다**(008: 조회가 영구
                #   `No records found` → 6-호출 사이클 ×30 → ctx 초과). v2는 ⑴같은 인자 반복 금지 +
                #   미사용 식별자 키 우선(x35 ③: 다른 인자 재조회 성공 73/138 = **즉시-ASK가 놓칠 회복
                #   상한 52.9%**) ⑵찾지 못하면 손님에게 말하고 다른 식별자 요청 ⑶줄 수 없으면 종결
                #   ⑷검증 통과 전 기록 금지(긍정형·[[42]]).
                tpl = ((spec.get("no_record_template_v2")
                        if os.environ.get("T2_NOREC_BRANCH") == "1" else None)
                       or spec.get("no_record_template") or spec.get("unmet_template") or "{count}")
            else:
                tpl = spec.get("met_template" if len(matched) >= thr else "unmet_template") or "{count}"
            try:
                return tpl.format(count=len(matched), threshold=thr,
                                  matched=", ".join(matched) or "(none)",
                                  missing=", ".join(missing) or "(none)")
            except Exception:
                return str(len(matched))
        if op == "case":
            # ★도메인일반 문자열-키 매핑(lookup_table의 문자열판). key(ref/op) → cases 매칭 → 값(op면 재귀).
            key = apply_op(spec.get("key"), ctx) if isinstance(spec.get("key"), dict) else _get(ctx, spec.get("key"))
            for k, v in (spec.get("cases") or {}).items():
                if key is not None and str(key).strip().lower() == str(k).strip().lower():
                    return apply_op(v, ctx) if isinstance(v, dict) and v.get("op") else v
            d = spec.get("default")
            return apply_op(d, ctx) if isinstance(d, dict) and d.get("op") else d
        if op == "select_discrepant":
            # ★도메인일반 op-DAG 실행기(역참조). 레코드마다: steps(이름있는 중간op)을 순서대로 계산해
            #   rctx["steps"][name]에 저장(뒤 step이 ref:steps.name으로 역참조) → expected_ref로 최종 기대값.
            #   actual_field와 tol 초과 차이면 id 수집. leaf(r.*)=LLM formalize·steps 공식=A2·엔진=실행만([[05]]/[[10]]).
            recs = _get(ctx, spec.get("over")) or []
            if not isinstance(recs, list):
                # ★C197: 리스트가 아니면(인자 파싱실패 잔류 str 등) 빈 결과로 위장하지 않는다 —
                #   stats를 계상해 C195 coverage("0 of 0")가 반드시 붙게(019 실측: str 잔류→stats 前
                #   조기반환→"(none)"이 깨끗한 빈 결과로 위장·완전 침묵). 판단 추가 0·표면화만.
                ctx["_sg_stats"] = {"judged": 0, "skipped": 0, "total": 0}
                return []
            idf, af = spec.get("id_field"), spec.get("actual_field")
            tol = _num(spec.get("tolerance")) or 0
            steps = spec.get("steps") or {}
            exp_ref = spec.get("expected_ref")
            espec = spec.get("expected")   # 하위호환(직접 expected op)
            out_ids = []
            skipped = 0
            # ★P4(C208④·DAY5_PRESCRIPTIONS §P4): op가 참조하는 행-필드(r.*) 집합 — 스킵 행의
            #   **어느 필드가 결핍인지**를 지목하기 위해. day5 020/026 [S]: account_open 누락으로
            #   14행 판정불가인데 메시지가 "14 could not be verified"뿐이라 모델이 자기-수복 불가
            #   (엔진은 null이 난 operand를 알면서 침묵). 수집=spec 문자열 워크(판단 0·도메인 무관).
            _refs = set()

            def _collect_refs(o):
                if isinstance(o, dict):
                    for v in o.values():
                        _collect_refs(v)
                elif isinstance(o, list):
                    for v in o:
                        _collect_refs(v)
                elif isinstance(o, str) and o.startswith("r."):
                    _refs.add(o[2:])
            _collect_refs(steps)
            _collect_refs(espec if isinstance(espec, dict) else {})
            _missing = {}
            for r in recs:
                if not isinstance(r, dict):
                    continue
                # ★F6a(C211·DAY7 §F6): id_field 결핍(P4b 강등 포함) 행은 판정 제외+계상 —
                #   구판(:638 out_ids.append(r.get(idf)))은 id를 무검사 append라 발명-행이
                #   discrepant로 에코돼 날조를 정당화(day6 020/028 [S]: 발명 txn id의 첫 등장=
                #   페이로드→에코→dispute 전파). 제외=지목 경로 합류(안전측).
                if idf and r.get(idf) in (None, ""):
                    skipped += 1
                    _missing[idf] = _missing.get(idf, 0) + 1
                    continue
                rctx = dict(ctx); rctx["r"] = r; rctx["steps"] = {}
                for nm, sp in steps.items():                 # 역참조 DAG: 순서대로·steps에 저장
                    rctx["steps"][nm] = apply_op(sp, rctx)
                exp = _get(rctx, exp_ref) if exp_ref else (apply_op(espec, rctx) if isinstance(espec, dict) else None)
                act = _num(r.get(af))                          # clean operand 전제(LLM formalize)
                en = _num(exp)
                if en is None or act is None:
                    # ★숫자로 안 읽히는 행은 판정 불가 → 건너뛴다(엔진이 '$'/단위를 파싱하면
                    #   엔진-formalize=[[03b]] 위반이므로 정규화는 LLM 몫이다). 단 **조용히** 건너뛰면
                    #   under-action이 "discrepant 0건"으로 위장된다 — 2026-07-18 실측 사고(계약문이
                    #   "레코드에 나타난 그대로"라 모델이 "$126.36"을 넘겨 17행 전부 탈락→0건). ⇒ 계측만 추가.
                    skipped += 1
                    for _f in _refs:                           # P4: 결핍 필드 계상(입력에 없거나 빈 값)
                        if r.get(_f) in (None, ""):
                            _missing[_f] = _missing.get(_f, 0) + 1
                    if act is None and r.get(af) not in (None, ""):
                        _missing["(unparsable %s)" % af] = _missing.get("(unparsable %s)" % af, 0) + 1
                    continue
                if abs(en - act) > tol:
                    out_ids.append(r.get(idf))
                    # ★discrepant 상세(2026-07-19·task_026 포렌식: 기대값 미반환→에이전트가 기록값
                    #   재기입 no-op). 엔진이 이미 계산한 expected를 노출만 한다 — 표시 여부/형식은
                    #   A2 return_template/detail_item_template이 정함([[05]]·계산=엔진 몫 [[10]]).
                    #   expected_disp = A2 선언 표기규칙(display_round_up_frac: 소수부≥임계면 올림·else 버림).
                    #   근거: gold 전수 census 10/10 적합(9 floor·1499.9→1500만 올림·§2i(6)).
                    thr = spec.get("display_round_up_frac")
                    disp = int(en) + (1 if thr is not None and (en - int(en)) >= float(thr) else 0)
                    ctx.setdefault("_sg_details", []).append(
                        {"id": r.get(idf), "actual": act, "expected": en,
                         "actual_int": int(act), "expected_floor": int(en), "expected_disp": disp})
            if skipped:
                print("[T2_COMPUTE] select_discrepant: %d/%d행 판정불가(operand가 숫자 아님) — "
                      "under-action 위험" % (skipped, len(recs)), file=_sys.stderr, flush=True)
            # ★C195(2026-07-26·야간 020/027/029 실측): "(none)" 단독 반환은 **판정 커버리지를 숨긴다**
            #   — 세 태스크서 도구가 "(none)"을 단언했고 gold는 4건+의 분쟁을 요구(2026-07-25 019는
            #   3/4만 검출). 원인 후보=격리 서브의 rate formalize 품질(카테고리-무시)·엔진은 그 값을
            #   신뢰할 수밖에 없으므로, 최소한 **몇 행을 무슨 근거로 판정했는지**를 결과에 병기해
            #   모델·후속 포렌식이 "(none)"을 무조건 신뢰하지 않게 한다(C185a unverified와 동일 원칙·
            #   엔진은 자기 집계의 표면화만·판단 추가 0).
            ctx["_sg_stats"] = {"judged": len(recs) - skipped, "skipped": skipped,
                                "total": len(recs), "missing_fields": _missing}
            return out_ids
        if op == "filter":
            # ★reference-filter(keystone): 수집 record를 criteria로 결정론 매칭 → return field.
            #   formalize(LLM)가 ctx["criteria"]={date,amount,merchant,type} 채움·엔진은 매칭만.
            #   match=[{field, eq|contains: ref}] — criteria 값 있는 조건만 적용(부분기준 허용).
            recs = _get(ctx, spec.get("over")) or []
            if not isinstance(recs, list):
                return None
            conds = spec.get("match") or []
            def hit(r):
                for c in conds:
                    fld = c.get("field")
                    if "eq" in c:
                        want = _get(ctx, c["eq"])
                        if want in (None, ""):
                            continue                       # 기준 미제공 → 이 조건 건너뜀
                        if str((r or {}).get(fld)) != str(want):
                            return False
                    elif "contains" in c:
                        want = _get(ctx, c["contains"])
                        if want in (None, ""):
                            continue
                        if str(want).lower() not in str((r or {}).get(fld) or "").lower():
                            return False
                return True
            matched = [r for r in recs if hit(r)]
            ret = spec.get("return")
            if len(matched) == 1:
                return matched[0].get(ret) if ret else matched[0]
            if len(matched) >= 2:
                oa = spec.get("on_ambiguous", "none")     # none|first|last|ask
                if oa == "first":
                    return matched[0].get(ret) if ret else matched[0]
                if oa == "last":
                    return matched[-1].get(ret) if ret else matched[-1]
                return None                                # ask/none → 미결정(호출측 ASK)
            return None                                    # 0 매칭
    except Exception:
        return None
    return None
