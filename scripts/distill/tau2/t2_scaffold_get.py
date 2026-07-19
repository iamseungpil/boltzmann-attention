# -*- coding: utf-8 -*-
"""t2_scaffold_get.py — scaffold-제공 GET 도구 (A2-선언·2026-07-16·BANK_IMPL_REDESIGN).
LLM은 구체 계산을 직접 안 하고 이 GET 도구를 *호출*→scaffold가 결정론 계산(t2_compute.apply_op)→결과 반환.
tau2 네이티브 아님·우리가 A2로 제공하는 일반 GET 함수. [[05]] 엔진=도메인일반·계산공식=A2 op-spec.
활성=T2_SCAFFOLD_GET=1. gate/unified 뒤에 apply(체이닝)."""
import os, json, sys as _sys

# ★삭제됨(2026-07-16·[[03b]]): `_parse_records`/`_gather` = 엔진이 tool 출력 텍스트를 정규식 파싱해
#   operand를 추출('$' strip 포함)하던 코드 = **엔진-formalize = 구현 속임**(이번 세션 위반 #2의 잔해).
#   호출부 0(전 repo grep 확인)이었으나, 남겨두면 재사용 유혹 = 표류 원천이므로 제거.
#   정본 경로: operand는 **LLM이 formalize**해 도구 인자로 넘기고(아래 exec2), 엔진은 op 실행만 한다([[10]]).


def _merge_json(text, keys):
    """텍스트 안 JSON을 **전부** 모아 병합(배열·별도객체·펜스·산문혼재 대응).
    ⚠️첫 객체만 집으면 나머지가 None이 되어 **모델 실패로 오독**된다(2026-07-18 실측·[[08]])."""
    text = text or ""
    out, i = {}, 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        for j in range(len(text), i, -1):
            try:
                val = json.loads(text[i:j])
            except Exception:
                continue
            for d in (val if isinstance(val, list) else [val]):
                if isinstance(d, dict):
                    out.update({k: v for k, v in d.items() if k in keys})
            i = j
            break
        else:
            i += 1
    return out


def _isolate_spec(d):
    """A2가 선언한 격리-formalize 스펙(미선언이면 None=거동 변화 0)."""
    return (d.get("isolate") or None) if isinstance(d, dict) else None


_DOC_CACHE = {}


def _load_domain_docs(domain):
    """도메인 KB 문서 전량 로드(`DATA_DIR/tau2/domains/<domain>/documents/*.json`). 캐시.
    ★도메인일반: 경로 규칙만·도메인 리터럴 0. 카드-스코프 선별은 호출부가 제목 접두로."""
    if domain in _DOC_CACHE:
        return _DOC_CACHE[domain]
    docs = []
    try:
        from tau2.utils.utils import DATA_DIR
        dd = os.path.join(str(DATA_DIR), "tau2", "domains", domain, "documents")
        for fn in sorted(os.listdir(dd)):
            if fn.endswith(".json"):
                o = json.load(open(os.path.join(dd, fn), encoding="utf-8"))
                docs.append({"title": o.get("title") or "", "content": o.get("content") or o.get("text") or ""})
    except Exception as e:
        print("[T2_SG_ISOLATE] 문서로드 실패(%s): %r" % (domain, e), file=_sys.stderr, flush=True)
    _DOC_CACHE[domain] = docs
    return docs


def _norm_ground(s):
    """grounding substring 매칭용 정규화(공백·대소문자·문장부호 흡수·엔진 결정론)."""
    return re.sub(r"[^a-z0-9%]+", " ", str(s).lower()).strip()


import re  # noqa: E402  (grounding 정규화용)


def _sub_formalize(orch, d, iso, ctx, run_env_calls):
    """★격리 서브 (2026-07-18 NIGHT+·`RATE_SUBAGENT_DESIGN §2b` LOCK — 사용자 원칙:
    *"operator operand 는 sub agent 로 부하 없이 격리로 결과 리턴 받아야 한다."*)

    메인 대화 문맥(20턴·25k·신원확인·도구저글링) 안에서 operand를 emit하면 부하로 열화한다
    (실측: 같은 32B가 격리서 base_rate 100% · 라이브서 오탐 10/26·task_020). ⇒ operand 산출을
    **자체 메시지 리스트만 가진 격리 서브요청**으로 옮긴다.

    - **메인 턴 소모 0**: 서브의 generate/도구호출은 `state.messages`에 안 들어간다. 메인이 보는 것은
      producer 호출 1건 + 그 결과 1건뿐(이 함수는 exec2=도구 실행 경로 안에서 돈다).
    - **GET는 진짜 도구로**: 서브가 A2 선언 getter(`iso.getter_tools`)를 호출하면 **env가 결정론 실행**
      (`run_env_calls`)하고 결과를 서브 문맥에 되먹인다 — 엔진이 문서를 골라주지 않는다([[03b]] spoon-feed 금지).
    - **1라운드 `tool_choice=required`**: 서브의 유일 임무 = getter 호출. required 경로는 vLLM Hermes 텍스트
      파서를 안 타므로(protocol.py:805 구조화디코딩), 봉투 오류로 호출이 **조용히 증발**하는 사고가 성립 불가
      (task_021 실패 기전). 개입레버 아님 — 채널만 강제([[16]] §7-1 무위반).
    - **엔진 리터럴 0**: 도구명·질의어·계약문 전부 A2(`iso`)서 온다. 엔진은 루프만 돈다.

    반환: {row_id: {operand: value}} · 실패 시 None(→ 호출부가 메인 인자로 폴백·거동 변화 0).
    """
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage
    ag = getattr(orch, "agent", None)
    rows = ctx.get(iso["over"])
    if ag is None or not isinstance(rows, list) or not rows:
        return None
    # ★★재설계(§2e): inject_docs면 카드당 격리+문서주입+grounding 경로로(검색 안 씀). 미선언=기존 검색 모드.
    if iso.get("inject_docs"):
        return _sub_inject(orch, d, iso, ctx, la, UserMessage)
    id_field = iso["id_field"]
    keep = set(iso.get("row_fields") or [])
    # 메인이 추측한 격리-operand는 **버린다**(누출 방지). 서브에 주는 행 = 원시 필드만.
    raw = [{k: v for k, v in r.items() if k in keep} for r in rows if isinstance(r, dict)]
    ids = [str(r.get(id_field)) for r in rows if isinstance(r, dict)]
    tools = [t for t in (getattr(ag, "tools", None) or [])
             if getattr(t, "name", None) in set(iso.get("getter_tools") or [])]
    if not tools:
        print("[T2_SG_ISOLATE] getter_tools 부재 → 격리 생략", file=_sys.stderr, flush=True)
        return None
    prompt = "%s\n\n=== ITEMS ===\n%s\n\n%s" % (
        iso["instructions"], json.dumps(raw, ensure_ascii=False, indent=1),
        iso["answer_format"].format(schema=json.dumps({i: iso.get("operand_schema", {}) for i in ids},
                                                      ensure_ascii=False)))
    try:
        um = UserMessage(role="user", content=prompt)
    except TypeError:
        um = UserMessage(content=prompt)
    msgs = [um]
    kw = {k: v for k, v in dict(getattr(ag, "llm_args", None) or {}).items() if "tool" not in k}
    # ★서브 온도 = A2 선언(`isolate.temperature`). ⚠️라이브 에이전트 llm_args는 이미 temp=0
    #   (`t2_run_gated.py:221`)이라 over-flag는 온도 아님(2026-07-18 정정·`RATE_SUBAGENT §2d`). 유지=명시성.
    if iso.get("temperature") is not None:
        kw["temperature"] = iso["temperature"]
    queries = []                                     # ★계측: 서브가 낸 KB 검색 질의(라이브 검색 가시화)
    for rnd in range(int(iso.get("max_rounds", 4))):
        try:
            resp = la.generate(model=ag.llm, tools=tools, messages=msgs,
                               call_name="sg_isolate",
                               **(dict(kw, tool_choice="required") if rnd == 0 else kw))
        except Exception as e:
            print("[T2_SG_ISOLATE] generate 실패(%d라운드): %r" % (rnd, e), file=_sys.stderr, flush=True)
            _isolate_trace(iso, d, {"error": str(e)[:200], "round": rnd, "queries": queries})
            return None
        tcs = list(getattr(resp, "tool_calls", None) or [])
        if tcs:
            for _tc in tcs:                          # 질의 기록
                _fn = getattr(_tc, "function", None) or _tc
                queries.append(getattr(_tc, "name", None) or getattr(_fn, "name", None))
            msgs.append(resp)
            msgs.extend(run_env_calls(tcs))          # ★GET = env 결정론 실행
            continue
        got = _merge_json(getattr(resp, "content", None) or "", set(ids))
        getter = sum(1 for m in msgs if getattr(m, "role", "") == "tool")
        print("[T2_SG_ISOLATE] %s: %d라운드·getter %d회·operand %d/%d행"
              % (d.get("name"), rnd + 1, getter, len(got), len(ids)), file=_sys.stderr, flush=True)
        # ★★계측: 서브 산출 operand 전수를 파일에 남긴다 — 라이브 서브는 메인 궤적 밖이라 여기 안 남기면
        #   over-flag가 서브 오독인지 검색부실인지 **영영 못 본다**(2026-07-18 디버깅공백·[[08]]).
        _isolate_trace(iso, d, {"round": rnd + 1, "getter": getter, "queries": queries,
                                "n_ids": len(ids), "n_operand": len(got), "operands": got})
        return got or None
    print("[T2_SG_ISOLATE] max_rounds 소진 → 격리 생략", file=_sys.stderr, flush=True)
    _isolate_trace(iso, d, {"error": "max_rounds", "queries": queries})
    return None


def _sub_inject(orch, d, iso, ctx, la, UserMessage):
    """★★재설계 격리(§2e·2026-07-18 실증 105/105): 카드당 격리 + 문서 주입(검색 0) + grounding.
    거래를 `group_by`(레코드 필드)로 그룹핑 → 그룹마다 그 그룹값의 문서를 제목접두로 주입 →
    서브가 `{base_rate, exclusion_quote}` formalize → 엔진 grounding(quote∈문서면 0 유지·아니면 default 백필).
    엔진 리터럴 0: 그룹키·필터규칙·계약문 전부 A2·값/인용 전부 LLM이 KB서·엔진은 substring+백필만."""
    ag = orch.agent
    domain = getattr(getattr(orch, "environment", None), "domain_name", None)
    all_docs = _load_domain_docs(domain) if domain else []
    if not all_docs:
        print("[T2_SG_ISOLATE] inject: 도메인 문서 0 → 격리 생략", file=_sys.stderr, flush=True)
        return None
    rows = ctx.get(iso["over"])
    id_field = iso["id_field"]
    # ★group_by = 단일 필드 or 복합키(list). 부하축소로 카드×카테고리 격리(§2h·2026-07-18 실증).
    gkeys = iso["group_by"] if isinstance(iso["group_by"], list) else [iso["group_by"]]
    doc_key = iso.get("doc_key", gkeys[0])   # 문서 필터는 카드 필드(카테고리별 문서 없음)
    keep = set(iso.get("row_fields") or [])
    kw = {k: v for k, v in dict(getattr(ag, "llm_args", None) or {}).items() if "tool" not in k}
    if iso.get("temperature") is not None:
        kw["temperature"] = iso["temperature"]

    groups = {}
    for r in rows:
        if isinstance(r, dict):
            gk = tuple(str(r.get(k)) for k in gkeys)
            groups.setdefault(gk, []).append(r)
    out = {}
    default_cache = {}    # 카드당 기본율 1회만(복합키로 카드가 여러 그룹에 나뉘어도 중복 호출 방지)
    for gk, grows in groups.items():
        gval = grows[0].get(doc_key)          # 문서 스코프 = 카드값(복합키의 doc_key 성분)
        docs = [x for x in all_docs if x["title"].startswith(str(gval) + ": ")]  # 결정론 제목접두(§2e)
        if not docs:
            print("[T2_SG_ISOLATE] inject: '%s' 문서 0 → 그룹 생략" % gval, file=_sys.stderr, flush=True)
            continue
        docnorm = _norm_ground(" ".join(x["content"] for x in docs))
        docstr = "\n\n".join("### %s\n%s" % (x["title"], x["content"]) for x in docs)
        raw = [{k: v for k, v in r.items() if k in keep} for r in grows]
        ids = [str(r.get(id_field)) for r in grows]
        schema = json.dumps({i: iso.get("operand_schema", {}) for i in ids}, ensure_ascii=False)
        prompt = iso["inject_instructions"].format(group=gval, docs=docstr, schema=schema,
                                                   items=json.dumps(raw, ensure_ascii=False, indent=1))
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        try:
            resp = la.generate(model=ag.llm, tools=None, messages=[um], call_name="sg_inject", **kw)
        except Exception as e:
            print("[T2_SG_ISOLATE] inject generate 실패(%s): %r" % (gval, e), file=_sys.stderr, flush=True)
            continue
        got = _merge_json(getattr(resp, "content", None) or "", set(ids))
        rate_f = iso.get("rate_field", "base_rate")
        # ★범위 가드+재질의 (A2 `rate_range` 선언 시만·2026-07-19 프로브 EcoCard-Green 0/6→6/6).
        #   근거: 028 셀 오류 — 서브가 "$5.00 points per dollar"를 ×100 스케일(500)로 formalize.
        #   범위=A2 선언·재질의 문구=A2·값은 여전히 서브가 산출(엔진 리터럴 0·[[07]] enforced).
        n_retry = 0
        rr = iso.get("rate_range")
        if rr and got:
            lo_r, hi_r = float(rr[0]), float(rr[1])

            def _rv(i):
                try:
                    return float((got.get(i) or {}).get(rate_f))
                except Exception:
                    return None
            bad_ids = [i for i in ids if _rv(i) is not None and not (lo_r <= _rv(i) <= hi_r)]
            if bad_ids and iso.get("range_retry_prompt"):
                extra = iso["range_retry_prompt"].format(ids=", ".join(bad_ids), lo=rr[0], hi=rr[1])
                try:
                    um2 = UserMessage(role="user", content=prompt + extra)
                except TypeError:
                    um2 = UserMessage(content=prompt + extra)
                try:
                    resp2 = la.generate(model=ag.llm, tools=None, messages=[um2],
                                        call_name="sg_inject_retry", **kw)
                    got2 = _merge_json(getattr(resp2, "content", None) or "", set(bad_ids))
                except Exception as e:
                    print("[T2_SG_ISOLATE] range-retry 실패(%s): %r" % (gval, e),
                          file=_sys.stderr, flush=True)
                    got2 = {}
                for i in bad_ids:
                    v2 = got2.get(i)
                    try:
                        r2 = float((v2 or {}).get(rate_f))
                    except Exception:
                        r2 = None
                    if r2 is not None and lo_r <= r2 <= hi_r:
                        got[i] = v2
                        n_retry += 1
                print("[T2_SG_ISOLATE] range-retry '%s': 위반 %d → 회복 %d"
                      % (gval, len(bad_ids), n_retry), file=_sys.stderr, flush=True)
            for i in ids:                     # 잔여 위반 = rate 제거(오탐 양산 대신 판정불가 abstain)
                rv = _rv(i)
                if rv is not None and not (lo_r <= rv <= hi_r):
                    (got.get(i) or {}).pop(rate_f, None)
        # ★셀-consensus 강등 가드 (A2 `consensus_demote_guard`=true·프로브 cons1=Patagonia 1→5).
        #   같은 (card×category) 셀은 같은 정책이 적용된다 — 소수 강등(0<rate<다수값)은 그 행의
        #   merchant/category가 담긴 인용이 문서에 실재할 때만 인정, 아니면 다수값으로 백필.
        #   다수값·인용·앵커 전부 데이터/서브 산출(엔진 리터럴 0). 0-rate는 기존 quote-grounding 경로 유지.
        n_cons = 0
        if iso.get("consensus_demote_guard") and got:
            from collections import Counter as _Counter
            q_f = iso.get("quote_field", "exclusion_quote")

            def _rv2(i):
                try:
                    return float((got.get(i) or {}).get(rate_f))
                except Exception:
                    return None
            _rates = [_rv2(i) for i in ids if _rv2(i) is not None]
            if len(_rates) >= 3:
                _modal, _cnt = _Counter(_rates).most_common(1)[0]
                if _cnt * 2 > len(_rates):
                    _byid = {str(r.get(id_field)): r for r in grows}
                    for i in ids:
                        rv = _rv2(i)
                        if rv is None or not (0 < rv < _modal):
                            continue
                        q = _norm_ground((got.get(i) or {}).get(q_f) or "")
                        anch = _norm_ground(str(_byid[i].get("merchant_name", ""))) in q or \
                            _norm_ground(str(_byid[i].get("category", ""))) in q
                        if not (len(q) >= int(iso.get("quote_min", 8)) and q in docnorm and anch):
                            got.setdefault(i, {})[rate_f] = _modal
                            n_cons += 1
            if n_cons:
                print("[T2_SG_ISOLATE] consensus '%s': 무근거 강등 %d행 → 다수값 백필"
                      % (gval, n_cons), file=_sys.stderr, flush=True)
        # 카드 기본율(default) — 근거없는 0 백필용. 카드당 1회 formalize·캐시(값=LLM·엔진 하드코딩0).
        if iso.get("base_default_prompt"):
            if gval not in default_cache:
                default_cache[gval] = _card_default(la, ag, iso, gval, docstr, UserMessage, kw)
            default = default_cache[gval]
        else:
            default = None
        kept = filled = 0
        for r in grows:
            tid = str(r.get(id_field))
            v = got.get(tid) or {}
            try:
                br = float(v.get(iso.get("rate_field", "base_rate")))
            except Exception:
                br = None
            # ★grounding: base_rate=0 이면 exclusion_quote가 문서에 실재하나
            if br == 0:
                q = _norm_ground(v.get(iso.get("quote_field", "exclusion_quote")) or "")
                grounded = len(q) >= int(iso.get("quote_min", 8)) and q in docnorm
                if grounded:
                    kept += 1                       # 진짜 예외 → 0 유지
                elif default is not None:
                    br = default                    # 근거없는 0 → 기본율 백필
                    filled += 1
            # ★서브 operand 전체를 rows에 병합(base_rate + promo 파라미터 등) — 엔진 op가 읽음.
            #   quote는 grounding용이라 제외. base_rate는 grounding/백필 반영된 최종값(br)으로 덮음.
            rate_f, quote_f = iso.get("rate_field", "base_rate"), iso.get("quote_field", "exclusion_quote")
            merged = {k: val for k, val in v.items() if k != quote_f}
            if br is not None:
                merged[rate_f] = br
            if merged:
                r.update(merged)                       # ★엔진 op가 읽을 operand로 병합(promo 포함)
                out[tid] = merged
        print("[T2_SG_ISOLATE] inject '%s': 문서 %d·거래 %d·operand %d·grounded유지 %d·백필 %d(default=%s)"
              % (gval, len(docs), len(grows), len([x for x in ids if x in out]), kept, filled, default),
              file=_sys.stderr, flush=True)
        _isolate_trace(iso, d, {"group": gval, "n_docs": len(docs), "n_rows": len(grows),
                                "kept": kept, "filled": filled, "default": default,
                                "range_retry": n_retry, "consensus": n_cons, "operands": got})
    return out or None


def _card_default(la, ag, iso, gval, docstr, UserMessage, kw):
    """그룹(카드) 기본율(all-other-purchases rate)을 KB서 formalize. 값=LLM·엔진 하드코딩 0([[05]])."""
    prompt = iso["base_default_prompt"].format(group=gval, docs=docstr)
    try:
        um = UserMessage(role="user", content=prompt)
    except TypeError:
        um = UserMessage(content=prompt)
    try:
        resp = la.generate(model=ag.llm, tools=None, messages=[um], call_name="sg_default", **kw)
        j = _merge_json(getattr(resp, "content", None) or "", {"base_default"})
        return float(j.get("base_default")) if j and j.get("base_default") is not None else None
    except Exception:
        return None


def _isolate_trace(iso, d, record):
    """서브 산출 operand를 JSONL로 남긴다(`T2_SG_ISOLATE_TRACE`=경로·미설정이면 no-op).
    ⚠️계측 전용(엔진 거동 무변화)·라이브 서브 가시화용([[08]] 포렌식)."""
    path = os.environ.get("T2_SG_ISOLATE_TRACE")
    if not path:
        return
    try:
        rec = dict(record)
        rec["tool"] = d.get("name")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[T2_SG_ISOLATE] trace 실패: %r" % (e,), file=_sys.stderr, flush=True)


def _build_tool(Tool, d):
    """A2 선언 → tau2 Tool. ★tau2 `Tool.__init__`은 **함수 객체에서** 스키마를 유도한다
    (`tool.py:61-73`: `name = func.__name__` · `parse_data(sig, doc, ...)`) — 우리가 넘기는
    `name=`/`long_desc=`/`params=`는 **무시된다**(predefined는 params 제외용).
    ⇒ 진짜 이름·시그니처·docstring을 가진 함수를 동적 생성해서 넘긴다.
    docstring은 `docstring_parser`가 읽는 형식(요약 + `:param x: 설명`)이어야 인자 설명이 스키마에 실린다."""
    name = d["name"]
    params = d.get("params") or {}
    doc = [str(d.get("description") or name).strip(), ""]
    for p, desc in params.items():
        doc.append(":param %s: %s" % (p, str(desc).replace("\n", " ").strip()))
    src = "def %s(%s):\n    pass\n" % (name, ", ".join("%s: str" % p for p in params))
    ns = {}
    exec(compile(src, "<a2_tool:%s>" % name, "exec"), ns)          # noqa: S102 — A2 선언서 생성
    fn = ns[name]
    fn.__doc__ = "\n".join(doc)
    return Tool(fn, examples=list(d.get("examples") or []))


def _variant(d, name=None):
    """★A2 변이 선택기 (도메인일반·2026-07-18·단일변수 arm용). `T2_A2_VARIANT=<name>`(또는 명시 `name`) →
    선언의 `variants[<name>]`을 얕은 병합. **엔진은 변이 *이름*만 알고 내용은 A2가 정한다**([[05]]).
    기본(미지정) = 원본 그대로 = **거동 변화 0**(진행 중 arm 보호).
    `name` 명시 = 프로브용(한 프로세스서 두 arm 비교) — **엔진과 병합 코드를 공유해야** 프로브가 라이브와
    같은 것을 잰다([[30]] 단위통과≠라이브발화·[[03b]] 별도 구현 금지)."""
    # ★다중 변이(2026-07-18·C113): `T2_A2_VARIANT=ledger,ratefix` → 각 도구는 **자기 variants에 있는 이름만**
    #   적용(도구마다 다른 변이). 단일값 하위호환. `name` 명시 시 그 하나만(프로브용).
    raw = name if name is not None else os.environ.get("T2_A2_VARIANT")
    if not raw:
        return d
    wanted = [x.strip() for x in str(raw).split(",") if x.strip()]
    have = d.get("variants") or {}
    hit = next((w for w in wanted if isinstance(have.get(w), dict)), None)
    if hit is None:
        return d
    d2 = {k: val for k, val in d.items() if k != "variants"}
    d2.update(have[hit])
    print("[T2_A2_VARIANT] %s ← '%s' (params=%s op=%s)"
          % (d2.get("name"), hit, list((d2.get("params") or {})), (d2.get("op") or {}).get("op")),
          file=_sys.stderr, flush=True)
    return d2


def _evidence_ctx(orch):
    """원장(=실제 호출 이력) → `{__user_text, __tool_outputs}`. `match_verdict_grounded`용.
    ★엔진은 **역할과 호출 이름만** 본다 — 내용 파싱/추출 0([[03b]]). 도메인 리터럴 0.
    ★NabaOS 대응: 그들 HMAC 영수증(`230-232` 런타임이 실행·`17` LLM 위조불가)의 자리를 **우리 원장**이
    대신한다. 엔진이 원장 소유자라 **서명 불요**(위조 경로 자체가 없음)."""
    users, outs, id2name = [], {}, {}
    try:
        for m in orch.get_messages():
            for tc in (getattr(m, "tool_calls", None) or []):
                id2name[getattr(tc, "id", None)] = getattr(tc, "name", None)
            r, c = getattr(m, "role", None), getattr(m, "content", None)
            if c is None:
                continue
            s = c if isinstance(c, str) else str(c)
            if r == "user":
                users.append(s)
            elif r == "tool" and getattr(m, "requestor", "assistant") == "assistant":
                nm = id2name.get(getattr(m, "id", None))
                if nm:
                    outs[nm] = (outs.get(nm, "") + " " + s)
    except Exception as e:
        print("[T2_SCAFFOLD_GET] evidence ctx fail: %r" % (e,), file=_sys.stderr, flush=True)
    return {"__user_text": " ".join(users).lower(),
            "__tool_outputs": {k: v.lower() for k, v in outs.items()}}


def _a2_named_in_args(tc, decls):
    """(a1) 호출 `tc`의 인자 값 중 **우리 A2 도구 이름과 정확히 일치**하는 게 있으면 그 이름을 반환.
    구조적 사실만 본다 — env 응답 텍스트를 읽지 않고([[03b]] 엔진-formalize 금지), 도메인 도구명을
    리터럴로 갖지 않는다([[05]]). 부분일치는 **안 본다**(자유 텍스트 오탐 방지: 산문 안에 도구명이
    스쳐도 가로채면 안 됨 — `FAB_PROBES §12` 오탐 사고와 동종)."""
    _args = getattr(tc, "arguments", None) or {}
    if not isinstance(_args, dict):
        return None
    for _v in _args.values():
        if isinstance(_v, str) and _v.strip() in decls:
            return _v.strip()
    return None


def apply():
    if os.environ.get("T2_SCAFFOLD_GET") != "1":
        return None
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    from tau2.data_model.message import ToolMessage
    from tau2.environment.tool import Tool
    from pydantic import create_model
    import t2_compute as _c
    import t2_gate_patch as _g

    # (1) 도구 스키마 주입 (per-sim·orchestrator init 후 agent.tools에 append)
    orig_init = BaseOrchestrator.__init__

    def init2(self, *a, **kw):
        orig_init(self, *a, **kw)
        env = getattr(self, "environment", None)
        ag = getattr(self, "agent", None)
        a2 = _g._domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if not a2 or ag is None:
            return
        decls = [_variant(d) for d in (a2.get("scaffold_get_tools") or [])]
        # ★키스톤 toggle(C103·2026-07-17): T2_SG_EXCLUDE=이름들(콤마) → 해당 A2 도구를 주입서 제외.
        #   단일 변수 대조(대안 도구 유/무)용 실험 스위치 — 엔진은 이름 필터만(도메인 리터럴 0).
        #   제외된 도구는 known-set에도 안 들어가므로 TOOLGATE가 진짜 부재처럼 취급(일관).
        _excl = {x.strip() for x in (os.environ.get("T2_SG_EXCLUDE") or "").split(",") if x.strip()}
        if _excl:
            decls = [d for d in decls if d.get("name") not in _excl]
            print("[T2_SCAFFOLD_GET] EXCLUDED by env: %s" % sorted(_excl), file=_sys.stderr, flush=True)
        tools = getattr(ag, "tools", None)
        if not decls or tools is None:
            return
        existing = {getattr(t, "name", None) for t in tools}
        for d in decls:
            if d["name"] in existing:
                continue
            try:
                tools.append(_build_tool(Tool, d))
            except Exception as e:
                print("[T2_SCAFFOLD_GET] inject fail %s: %r" % (d["name"], e), file=_sys.stderr, flush=True)
                continue
        # ★★라이브 검증 의무([[30]]: 단위통과≠라이브발화). 주입 결과 스키마를 실제로 찍는다 —
        #   2026-07-16 사고: Tool이 name/desc/params를 func에서 유도하는데 더미 `def _f(**k)`를 넘겨
        #   **이름 `_f`·설명 `_f`·인자 `k`** 로 들어갔고, 모델은 우리 도구를 *본 적이 없었다*.
        for t in tools:
            if getattr(t, "name", None) in {x["name"] for x in decls}:
                _s = t.openai_schema["function"]
                print("[T2_SCAFFOLD_GET] injected name=%s desc=%dch params=%s"
                      % (_s["name"], len(_s.get("description") or ""),
                         list((_s.get("parameters") or {}).get("properties") or {})),
                      file=_sys.stderr, flush=True)
        self._t2_sg_a2 = a2
        try:
            self._t2_known_tools = {getattr(t, "name", None) for t in (getattr(ag, "tools", None) or [])}
        except Exception:
            self._t2_known_tools = set()

    BaseOrchestrator.__init__ = init2

    # (2) 호출 intercept: 우리 도구면 결정론 계산·반환·env 우회 (gate/unified 뒤 체이닝)
    orig_exec = BaseOrchestrator._execute_tool_calls

    def exec2(self, tool_calls):
        a2 = getattr(self, "_t2_sg_a2", None)
        decls = {d["name"]: d for d in
                 (_variant(x) for x in ((a2 or {}).get("scaffold_get_tools") or []))}
        for _x in {x.strip() for x in (os.environ.get("T2_SG_EXCLUDE") or "").split(",") if x.strip()}:
            decls.pop(_x, None)          # 제외 도구는 실행 경로서도 부재(주입 필터와 일관)
        if not decls:
            return orig_exec(self, tool_calls)
        ours = {}
        rest = []
        for tc in tool_calls:
            # ★★requestor 격리 (2026-07-16 버그픽스·`ASSERTION_PROVENANCE_ARMS_DESIGN` §7):
            #   orchestrator._execute_tool_calls는 **AGENT와 USER 양쪽**의 호출을 처리한다
            #   (orchestrator.py:882 `from_role in [AGENT, USER] and to_role == ENV`).
            #   우리 도구·TOOLGATE는 *에이전트* 도구집합(agent.tools) 기준이므로 user 호출에 적용하면:
            #     (1) 사용자의 **gold 액션**을 차단 (task_019 gold 4/6 = requestor:user
            #         `call_discoverable_user_tool`) → 그 task는 **통과 불가**가 된다.
            #     (2) requestor 불일치 ToolMessage가 user-sim 히스토리에 들어가
            #         `user_simulator_base.py:102` ValueError → Retry → infrastructure_error.
            #   ⇒ 에이전트 호출만 다룬다. 나머지는 원본 실행 경로로.
            if getattr(tc, "requestor", "assistant") != "assistant":
                rest.append(tc)
                continue
            if getattr(tc, "name", None) in decls:
                d = decls[getattr(tc, "name")]
                # ★LLM이 formalize한 clean operand(각 인자)를 ctx로([[10]]). 엔진은 op 실행만·원시파싱 안함.
                _args = getattr(tc, "arguments", None) or {}
                _ctx = {}
                for _k, _v in (_args.items() if isinstance(_args, dict) else []):
                    if isinstance(_v, str):
                        try:
                            _v = json.loads(_v)
                        except Exception:
                            pass
                    _ctx[_k] = _v
                # ★원장-결합 op는 인자 밖 증거가 필요하다 — **op가 `evidence_from`을 선언할 때만** 주입
                #   (도메인일반 조건·미선언 op는 거동 변화 0).
                if (d.get("op") or {}).get("evidence_from"):
                    _ctx.update(_evidence_ctx(self))
                # ★격리 서브가 operand 산출 (T2_SG_ISOLATE=1·기본 OFF·A2 `isolate` 선언 시만)
                #   `RATE_SUBAGENT_DESIGN §2b` LOCK. 실패=None → 메인 인자로 폴백(거동 변화 0).
                _iso = _isolate_spec(d) if os.environ.get("T2_SG_ISOLATE") == "1" else None
                if _iso:
                    def _run(tcs, _self=self):
                        return orig_exec(_self, tcs)
                    _sub = _sub_formalize(self, d, _iso, _ctx, _run)
                    if _sub:
                        _rows = _ctx.get(_iso["over"]) or []
                        _hit = 0
                        for _r in _rows:
                            _v = _sub.get(str(_r.get(_iso["id_field"]))) if isinstance(_r, dict) else None
                            if isinstance(_v, dict):
                                _r.update(_v)          # 서브 operand가 메인 추측을 대체
                                _hit += 1
                        print("[T2_SG_ISOLATE] %s: %d/%d행 operand를 격리 서브가 산출"
                              % (getattr(tc, "name"), _hit, len(_rows)), file=_sys.stderr, flush=True)
                _res = _c.apply_op(d.get("op"), _ctx)
                if isinstance(_res, list):                    # 목록형(discrepancy ids)
                    _res = [str(i) for i in _res if i]
                    # ★{details}: op가 남긴 상세(_sg_details)를 A2 detail_item_template로 포맷.
                    #   A2 template이 {details}를 안 쓰면 거동 변화 0(여분 kwarg는 무해).
                    _dets = _ctx.get("_sg_details") or []
                    _item_t = d.get("detail_item_template", "{id}")
                    try:
                        _details = "; ".join(_item_t.format(**it) for it in _dets) if _dets else "(none)"
                    except Exception:
                        _details = ", ".join(_res) if _res else "(none)"
                    _txt = d.get("return_template", "{ids}").format(
                        ids=", ".join(_res) if _res else "(none)", details=_details)
                    _n = len(_res)
                else:                                         # 스칼라형(verdict 등)
                    _txt = d.get("return_template", "{result}").format(result=_res if _res is not None
                                                                       else d.get("missing_hint", "(could not compute — check your arguments)"))
                    _n = _res
                # requestor는 tau2 원본과 동형으로 **미러링**(environment.get_response: requestor=message.requestor).
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"), content=_txt)
                print("[T2_SCAFFOLD_GET] %s -> %s" % (getattr(tc, "name"), _n), file=_sys.stderr, flush=True)
            elif (os.environ.get("T2_SG_TRUTH") == "1"
                  and _a2_named_in_args(tc, decls)):
                # ★(a1) 인터페이스-사실 정정 (2026-07-18·FAB_PROBES §5.2). 우리가 A2 도구를 **도구 목록에만**
                #   주입하고 env의 도달 레지스트리엔 등록하지 않아서, env가 우리 도구에 대해 **거짓을 말한다**
                #   ("This tool is not available") → 모델이 자기 도구 목록보다 env를 믿고 → 눈대중 → 완료 날조.
                #   ⇒ env가 **우리 도구 이름을 인자로** 받은 호출은 우리가 가로채 **사실만** 답한다.
                #   [[05]]/[[16]] 경계: 판정 기준 = 우리 A2 도구명과의 **정확 일치**뿐(env 응답 텍스트 파싱 0·
                #   도메인 리터럴 0·엔진은 자기 도구명만 안다). 선택 유도 아님 — 모델은 **이미 그 도구를 골랐고**,
                #   우리는 호출 인터페이스 사실만 정정한다.
                #   ⚠️over-action 위험(모트 제1원리: 레버는 하나 사면 하나 판다) = 우리 도구명을 인자로 쓰는
                #   **정당한** 호출(예: KB 검색 query)까지 가로챌 수 있다 → Δspurious 계측 전엔 기본 OFF.
                _tn = _a2_named_in_args(tc, decls)
                _msg = ("`%s` is not managed by `%s`. `%s` is already one of the tools provided to you — "
                        "call `%s` directly with its arguments." % (_tn, getattr(tc, "name", "") or "", _tn, _tn))
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"), content=_msg)
                print("[T2_SG_TRUTH] '%s(%s)' -> interface fact (env would have denied our tool)"
                      % (getattr(tc, "name", "") or "", _tn), file=_sys.stderr, flush=True)
            elif (os.environ.get("T2_TOOLGATE") == "1"
                  and getattr(self, "_t2_known_tools", None)
                  and getattr(tc, "name", None) not in self._t2_known_tools):
                # ★invalid 선택 → ASK (GET/FIND/INFER/ASK의 ASK 분기·단순 fail/forcing/추천 아님).
                #   LLM은 유한 도구집합서 선택만 함(생성 아님). 매칭 실패 = 필요값을 사용자에게 물어라.
                _msg = ("'%s' is not one of your available tools, so nothing was called. Do not invent tools — "
                        "you may only call tools that are provided to you. If you are missing information needed "
                        "to use one of your available tools, ASK the customer to provide that information, then "
                        "call an available tool." % (getattr(tc, "name", "") or ""))
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"),
                                           error=True, content=_msg)
                print("[T2_TOOLGATE] invalid selection '%s' -> ASK prompt"
                      % (getattr(tc, "name", "") or ""), file=_sys.stderr, flush=True)
            else:
                rest.append(tc)
        rest_res = orig_exec(self, rest) if rest else []
        ri = iter(rest_res)
        out = []
        for tc in tool_calls:
            if id(tc) in ours:
                out.append(ours[id(tc)])
            else:
                try:
                    out.append(next(ri))
                except StopIteration:
                    pass
        return out

    BaseOrchestrator._execute_tool_calls = exec2
    print("[T2_SCAFFOLD_GET] ON", file=_sys.stderr, flush=True)
    return True
