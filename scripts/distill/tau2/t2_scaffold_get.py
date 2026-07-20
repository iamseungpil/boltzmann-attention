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


def _nums_in(text):
    """raw 텍스트의 숫자 토큰 → float 집합(소수점 보존). ★`_norm_ground`는 '.'을 공백으로 지워
    3.35→'3 35'로 부수므로 수치 매칭엔 못 쓴다 — grounding value 대조는 raw서 추출한다.
    ★천단위 콤마 흡수($2,000→2000): 안 하면 정답값도 드롭(false-drop·offline 검증서 실측)."""
    text = re.sub(r"(?<=\d),(?=\d)", "", str(text or ""))     # 2,000 → 2000
    out = set()
    for m in re.findall(r"\d+(?:\.\d+)?", text):
        try:
            out.add(float(m))
        except Exception:
            pass
    return out


def _dates_in(text):
    """raw 텍스트의 날짜 토큰 → date 집합(형식-불문 매칭·023 개설일 grounding용)."""
    import t2_compute as _c
    out = set()
    for m in re.findall(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", str(text or "")):
        dt = _c._parse_date(m)
        if dt is not None:
            out.add(dt.date() if hasattr(dt, "date") else dt)
    return out


def _corpus_texts(orch, which):
    """grounding 대조 코퍼스 (A2 선언 `corpus`: 'kb'|'ledger'). 도메인 리터럴 0·기존 소스 재사용.
    kb=도메인 KB 문서 전량 · ledger=지금까지 원장(에이전트 도구 출력)+사용자 발화. 엔진은 텍스트만 본다."""
    texts = []
    if "kb" in which:
        domain = getattr(getattr(orch, "environment", None), "domain_name", None)
        texts += [d0.get("content") or "" for d0 in (_load_domain_docs(domain) if domain else [])]
    if "ledger" in which:
        ev = _evidence_ctx(orch)
        texts += list((ev.get("__tool_outputs") or {}).values())
        texts.append(ev.get("__user_text") or "")
    return [t for t in texts if t]


def _as_float(v):
    """grounding 수치 파싱(%·$·, 흡수). 파싱 불가=None."""
    try:
        return float(str(v).replace("%", "").replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def _val_grounded(val, corpus_texts, kind=None):
    """값 하나가 코퍼스에 실재하나 = 에이전트가 지어내거나 오독하지 않았나(결정론 검증만·[[03b]]).
    ⚠️전-코퍼스 존재 검사라 **총체적 날조/오독**(레코드에 없는 값)은 잡지만, *다른 곳에 우연히
    있는* 틀린 값은 못 잡는다(source-필드 없는 스칼라의 원리적 한계). 날짜·숫자는 형식-불문 매칭."""
    if val is None or (isinstance(val, str) and not val.strip()):
        return True                       # 빈 값=grounding 대상 아님(op가 처리)
    if kind == "date":
        import t2_compute as _c
        dv = _c._parse_date(val)
        if dv is None:
            return True                   # 날짜 아님=통과(형식 게이트 몫)
        target = dv.date() if hasattr(dv, "date") else dv
        return any(target in _dates_in(t) for t in corpus_texts)
    fv = _as_float(val)
    if fv is not None:                    # 수치 값=숫자 토큰 매칭(형식-불문)
        return any(any(abs(fv - n) < 1e-9 for n in _nums_in(t)) for t in corpus_texts)
    nv = _norm_ground(val)                # 문자열 값=정규화 substring
    return bool(nv) and any(nv in _norm_ground(t) for t in corpus_texts)


class _SafeMap(dict):
    """format_map용: 미존재 키는 원문 유지(KeyError 회피·기존 {result}-only template 거동보존)."""
    def __missing__(self, k):
        return "{%s}" % k


def _render_scalar(d, ctx, res):
    """★스칼라 반환문 렌더 (2026-07-20 관문3·순수함수=단위테스트 공유·[[03b]]).
    template 키 확장: {result} 외에 **호출 인자**({계좌id} 등)를 에코할 수 있게 — WEV의 토큰+id
    공존 게이트가 우리 반환문을 증거로 쓰는 채널(user_id-키/id-공존 불성립 해소). 엔진=치환만."""
    sm = _SafeMap({k: v for k, v in ctx.items()
                   if isinstance(v, (str, int, float)) and not k.startswith("_")})
    sm["result"] = (res if res is not None
                    else d.get("missing_hint", "(could not compute — check your arguments)"))
    return str(d.get("return_template", "{result}")).format_map(sm)


def _ground_operands(orch, d, ctx):
    """★operand grounding (관문1·`ACCOUNT_APY_OFFLOAD §2a` 리뷰③·2026-07-20 배선). A2 `ground` 선언 시
    op 실행 前 각 grounded operand가 **KB/원장에 실재하는지 검증** — 미검증=드롭+플래그(→abstain).
    - [[03b]] **검증만**: 엔진이 KB서 정답값을 *추출*하지 않는다. LLM이 낸 (value, source)가 실재하는지·
      value가 자기 인용 안에 있는지만 본다. 엔진 리터럴 0(어느 필드·코퍼스=전부 A2 `ground`).
    - [[10]] 분담: 생성(값·인용)=LLM, 검증(존재 대조)=결정론 엔진.
    - 097(source 축자아님+base 추측)·095(값 오독)=array `require_value_in_source`가 드롭. 023(개설일
      오독)=scalar date-ledger 대조가 드롭. 셋 다 abstain으로 '가짜 정밀도'를 막는다(§2ab 역설).
    반환: 드롭 항목 설명 리스트(플래그). ctx는 in-place로 미검증 operand 제거."""
    gspec = d.get("ground")
    if not isinstance(gspec, dict):
        return []
    flags = []
    # (a) array-field: 원소별 {value, source} — source∈코퍼스(실재) + value∈source(오독 차단)
    for af in (gspec.get("array_fields") or []):
        arr = ctx.get(af.get("param"))
        if not isinstance(arr, list):
            continue
        vf, sf = af.get("value_field", "value"), af.get("source_field", "source")
        lf = af.get("label_field", "kind")
        norm_corpus = [_norm_ground(t) for t in _corpus_texts(orch, af.get("corpus") or ["kb"])]
        req_vis = af.get("require_value_in_source", True)
        kept = []
        for el in arr:
            if not isinstance(el, dict):
                kept.append(el)
                continue
            src = el.get(sf)
            ns = _norm_ground(src) if src else ""
            src_ok = bool(ns) and any(ns in nc for nc in norm_corpus if nc)
            val_ok = True
            if req_vis:
                fv = _as_float(el.get(vf))
                if fv is not None:
                    val_ok = any(abs(fv - n) < 1e-9 for n in _nums_in(src))
            if src_ok and val_ok:
                kept.append(el)
            else:
                why = ("source not found in the knowledge base" if not src_ok
                       else "the value is not present in the source you cited")
                flags.append("%s=%s (%s)" % (el.get(lf, "?"), el.get(vf), why))
        if len(kept) != len(arr):
            ctx[af.get("param")] = kept
    # (b) scalar-field: top-level operand가 원장/KB에 실재하는지(023 개설일·097 principal 등)
    for scf in (gspec.get("scalar_fields") or []):
        param = scf.get("param")
        if param not in ctx:
            continue
        # ★source_param(2026-07-20 관문3): 존재검사가 무력한 값(잔액 0 등 편재값)은 **축자 인용** 요구 —
        #   source∈코퍼스(실재) + value∈source(자기 인용 안 값·array-field와 동형). 날조-0 차단.
        sp = scf.get("source_param")
        if sp:
            src = ctx.get(sp)
            ns = _norm_ground(src) if src else ""
            corp = [_norm_ground(t) for t in _corpus_texts(orch, scf.get("corpus") or ["ledger"])]
            src_ok = bool(ns) and any(ns in c for c in corp if c)
            val_ok = _val_grounded(ctx.get(param), [str(src or "")], scf.get("kind"))
            if not (src_ok and val_ok):
                flags.append("%s=%s (%s)" % (param, ctx.get(param),
                             "source quote not found in the records" if not src_ok
                             else "the value is not present in the source you quoted"))
                if scf.get("on_fail", "drop") == "drop":
                    ctx[param] = None
            continue
        if not _val_grounded(ctx.get(param), _corpus_texts(orch, scf.get("corpus") or ["ledger"]),
                             scf.get("kind")):
            flags.append("%s=%s (not found in the records — re-read the exact value)"
                         % (param, ctx.get(param)))
            if scf.get("on_fail", "drop") == "drop":
                ctx[param] = None          # op가 missing_hint로 abstain(가짜 정밀도 차단)
    return flags


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
    # ★max_batch (A2 선언·2026-07-19 사용자 지시): 서브 호출당 최대 행수. 1이면 행마다 개별 호출 —
    #   리스트 serial-position 효과(내부 항목 저하·§2l 실측: 같은 행이 순서에 따라 5↔1) 원천 소멸
    #   (모든 항목이 유일 항목=가장자리). 미선언=그룹 통짜(기존 거동). 값 산출=여전히 서브·비용=로컬.
    mb = int(iso.get("max_batch") or 0)
    for gk, g_all in groups.items():
        gval = g_all[0].get(doc_key)          # 문서 스코프 = 카드값(복합키의 doc_key 성분)
        docs = [x for x in all_docs if x["title"].startswith(str(gval) + ": ")]  # 결정론 제목접두(§2e)
        if not docs:
            print("[T2_SG_ISOLATE] inject: '%s' 문서 0 → 그룹 생략" % gval, file=_sys.stderr, flush=True)
            continue
        docnorm = _norm_ground(" ".join(x["content"] for x in docs))
        docstr = "\n\n".join("### %s\n%s" % (x["title"], x["content"]) for x in docs)
        chunks = [g_all[i:i + mb] for i in range(0, len(g_all), mb)] if mb > 0 else [g_all]
        g_retry = 0
        g_got = {}
        for grows in chunks:
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
            # ★범위 가드+재질의 (A2 `rate_range` 선언 시만·§2i 프로브 EcoCard-Green 0/6→6/6).
            #   범위=A2 선언·재질의 문구=A2·값은 여전히 서브가 산출(엔진 리터럴 0·[[07]] enforced).
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
                            g_retry += 1
                for i in ids:                 # 잔여 위반 = rate 제거(오탐 양산 대신 판정불가 abstain)
                    rv = _rv(i)
                    if rv is not None and not (lo_r <= rv <= hi_r):
                        (got.get(i) or {}).pop(rate_f, None)
            # ★consensus·default백필 제거됨(2026-07-19 사용자 지시·[[10]]·§2k) — 엔진은 서브 operand를
            #   그대로 병합만(값 생성/override 0). 서브 오류는 원천(프롬프트·max_batch)서 수정.
            for r in grows:
                tid = str(r.get(id_field))
                v = got.get(tid) or {}
                quote_f = iso.get("quote_field", "exclusion_quote")
                merged = {k: val for k, val in v.items() if k != quote_f}  # quote=grounding용→op 제외
                if merged:
                    r.update(merged)                   # ★엔진 op가 읽을 operand로 병합(서브 산출 그대로)
                    out[tid] = merged
            g_got.update(got)
        all_ids = [str(r.get(id_field)) for r in g_all]
        print("[T2_SG_ISOLATE] inject '%s': 문서 %d·거래 %d·청크 %d(max_batch=%s)·operand %d·range_retry %d"
              % (gval, len(docs), len(g_all), len(chunks), mb or "∞",
                 len([x for x in all_ids if x in out]), g_retry),
              file=_sys.stderr, flush=True)
        _isolate_trace(iso, d, {"group": gval, "n_docs": len(docs), "n_rows": len(g_all),
                                "n_chunks": len(chunks), "range_retry": g_retry, "operands": g_got})
    return out or None


def _sub_wrap(orch, fa, tc, run_env_calls):
    """★기능 서브 (W)wrap 모드 (`FUNCTION_AGENT_ISOLATION_DESIGN_2026_07_19` — 사용자 리뷰 LOCK).
    메인이 부른 **자료-read**(KB 검색류)를 격리 서브가 자기 문맥에서 소비하고, 메인엔
    **답 + 근거 원문 인용(quotes)**만 반환한다 — 20K 덤프가 메인 대화에 안 쌓인다(§2g 일반형).

    - 게이트 증거 계약: wrap 대상은 A2 `wraps` 선언의 자료-read만(상태 read 금지는 A2 리뷰 규칙).
      quotes는 **무편집 원문** — WEV/observe가 스캔할 증거가 메인에 남는 채널.
    - quote_grounding: 반환 인용이 실제 KB 문서 substring인지 **엔진이 결정론 검증**(_norm_ground·
      §2e 재사용). found=true인데 근거 인용이 전부 탈락하면 **폴백**(서브 환각 차단·[[03b]]).
    - §2d 제약 내장: max_rounds/max_getter_calls/max_sub_chars cap·temp 0(A2)·실패=None→원 실행 폴백.
    - [[10]]: 서브 LLM=검색·해석(생성기)·라우팅/getter 실행/grounding=결정론. 엔진 도메인 리터럴 0
      (도구명·지시·계약·템플릿 전부 A2 `function_agents[]`).
    반환: 메인에 넣을 compact 텍스트 or None(폴백)."""
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage
    ag = getattr(orch, "agent", None)
    if ag is None:
        return None
    tools = [t for t in (getattr(ag, "tools", None) or [])
             if getattr(t, "name", None) in set(fa.get("getter_tools") or [])]
    if not tools:
        print("[T2_FN_ISOLATE] getter_tools 부재 → 폴백", file=_sys.stderr, flush=True)
        return None
    args = getattr(tc, "arguments", None) or {}
    prompt = "%s\n\n=== MAIN AGENT'S CALL ===\ntool: %s\narguments: %s\n\n%s" % (
        fa["instructions"], getattr(tc, "name", "") or "",
        json.dumps(args, ensure_ascii=False), fa["return_contract"])
    try:
        um = UserMessage(role="user", content=prompt)
    except TypeError:
        um = UserMessage(content=prompt)
    msgs = [um]
    kw = {k: v for k, v in dict(getattr(ag, "llm_args", None) or {}).items() if "tool" not in k}
    if fa.get("temperature") is not None:
        kw["temperature"] = fa["temperature"]
    getter_cap = int(fa.get("max_getter_calls", 4))
    char_cap = int(fa.get("max_sub_chars", 60000))
    getters = 0
    queries = []
    resp = None
    for rnd in range(int(fa.get("max_rounds", 4))):
        try:
            resp = la.generate(model=ag.llm, tools=tools, messages=msgs,
                               call_name="fn_isolate",
                               **(dict(kw, tool_choice="required") if rnd == 0 else kw))
        except Exception as e:
            print("[T2_FN_ISOLATE] generate 실패(%d라운드): %r" % (rnd, e), file=_sys.stderr, flush=True)
            _isolate_trace(fa, {"name": fa.get("name")}, {"error": str(e)[:200], "round": rnd})
            return None
        tcs = list(getattr(resp, "tool_calls", None) or [])
        if tcs:
            getters += len(tcs)
            if getters > getter_cap:
                print("[T2_FN_ISOLATE] getter cap %d 초과 → 폴백" % getter_cap, file=_sys.stderr, flush=True)
                _isolate_trace(fa, {"name": fa.get("name")}, {"error": "getter_cap", "queries": queries})
                return None
            for _tc in tcs:
                queries.append(json.dumps(getattr(_tc, "arguments", None) or {}, ensure_ascii=False)[:120])
            msgs.append(resp)
            msgs.extend(run_env_calls(tcs))              # GET = env 결정론 실행 (§2c 동형)
            _total = sum(len(str(getattr(m, "content", "") or "")) for m in msgs)
            if _total > char_cap:                        # §2d 결함1(부하 재생산) 가드
                print("[T2_FN_ISOLATE] sub chars %d > cap %d → 폴백" % (_total, char_cap),
                      file=_sys.stderr, flush=True)
                _isolate_trace(fa, {"name": fa.get("name")}, {"error": "char_cap", "queries": queries})
                return None
            continue
        break
    raw = str(getattr(resp, "content", None) or "") if resp is not None else ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        _isolate_trace(fa, {"name": fa.get("name")}, {"error": "no_json", "queries": queries})
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        _isolate_trace(fa, {"name": fa.get("name")}, {"error": "bad_json", "queries": queries})
        return None
    answer = str(obj.get("answer") or "").strip()
    quotes = [str(q) for q in (obj.get("quotes") or []) if str(q).strip()]
    found = bool(obj.get("found", True))
    if not answer:
        _isolate_trace(fa, {"name": fa.get("name")}, {"error": "no_answer", "queries": queries})
        return None
    dropped = 0
    if fa.get("quote_grounding"):
        docs = _load_domain_docs(getattr(getattr(orch, "environment", None), "domain_name", None))
        norm_docs = [_norm_ground(d0.get("content") or "") for d0 in docs]
        kept = [q for q in quotes if any(_norm_ground(q) in nd for nd in norm_docs if nd)]
        dropped = len(quotes) - len(kept)
        quotes = kept
        if found and not quotes:                          # 근거 전멸 = 서브 환각 의심 → 폴백
            print("[T2_FN_ISOLATE] grounded quote 0 → 폴백", file=_sys.stderr, flush=True)
            _isolate_trace(fa, {"name": fa.get("name")}, {"error": "quotes_ungrounded",
                                                          "dropped": dropped, "queries": queries})
            return None
    txt = fa.get("return_template", "{answer}\n{quotes}").format(
        answer=answer, quotes="\n".join("- " + q for q in quotes) if quotes else "(none)")
    print("[T2_FN_ISOLATE] %s: getter %d회·quotes %d(드롭 %d)·%dch 반환"
          % (fa.get("name"), getters, len(quotes), dropped, len(txt)), file=_sys.stderr, flush=True)
    _isolate_trace(fa, {"name": fa.get("name")}, {"getter": getters, "queries": queries,
                                                  "n_quotes": len(quotes), "dropped": dropped,
                                                  "found": found, "chars": len(txt)})
    return txt


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


def _reassemble(tool_calls, ours, rest_res, ToolMessage):
    """★exec 결과 재조립 (2026-07-20 크래시 픽스·순수함수=단위테스트 공유·[[03b]]).
    **불변식**: 반환은 `tool_calls`와 **정확히 1:1·같은 순서**. 안 지키면 full-duplex tick의
    agent_tool_calls↔agent_tool_results 쌍이 깨져 eval replay(`environment.get_actions_from_messages`)가
    "Tool call id mismatch"로 크래시(023/031 infrastructure_error의 근본·비결정론=orig_exec 순서 의존).
    - 우리가 답한 tc(`ours[id(tc)]`)는 그대로.
    - 나머지는 `orig_exec` 결과를 **tc.id로 매칭**(위치 의존 제거 — orig_exec가 순서/개수 바꿔도 안전).
      id 매칭 실패 시 남은 결과서 순서대로 소비(id 없는 백엔드 하위호환), 그래도 없으면 에러 ToolMessage로
      **채운다**(드롭 금지 — 드롭이 tick 쌍 붕괴의 직접 원인)."""
    by_id, leftover = {}, []
    for r in (rest_res or []):
        rid = getattr(r, "id", None)
        if rid is not None and rid not in by_id:
            by_id[rid] = r
        else:
            leftover.append(r)
    lo = iter(leftover)
    out = []
    for tc in tool_calls:
        if id(tc) in ours:
            out.append(ours[id(tc)])
            continue
        r = by_id.pop(getattr(tc, "id", None), None)
        if r is None:
            r = next(lo, None)
        if r is None:
            r = ToolMessage(id=getattr(tc, "id", None), role="tool",
                            requestor=getattr(tc, "requestor", "assistant"),
                            error=True, content="(no result returned for this tool call)")
        out.append(r)
    return out


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
        # ★T2_FN_ISOLATE=1: (W)wrap 기능서브 — A2 function_agents[mode=wrap]의 wraps 도구를 서브로 위임
        #   (FUNCTION_AGENT_ISOLATION_DESIGN·사용자 리뷰 LOCK). 실패=원 실행 폴백·기본 OFF.
        fa_map = {}
        if os.environ.get("T2_FN_ISOLATE") == "1":
            for _fa in ((a2 or {}).get("function_agents") or []):
                if _fa.get("mode") == "wrap":
                    for _w in (_fa.get("wraps") or []):
                        fa_map[_w] = _fa
        if not decls and not fa_map:
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
                # ★operand grounding (T2_SG_GROUND=1·기본 OFF·A2 `ground` 선언 시만·관문1·2026-07-20).
                #   op 실행 前 미검증(날조/오독) operand를 드롭+플래그 → abstain. 실패해도 폴백 없음
                #   (드롭=abstain이 목적). 미선언 or OFF = 거동 변화 0.
                _gflags = []
                if os.environ.get("T2_SG_GROUND") == "1" and d.get("ground"):
                    _gflags = _ground_operands(self, d, _ctx)
                    if _gflags:
                        print("[T2_SG_GROUND] %s: %d ungrounded operand 드롭 -> %s"
                              % (getattr(tc, "name"), len(_gflags), "; ".join(_gflags)),
                              file=_sys.stderr, flush=True)
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
                    _txt = _render_scalar(d, _ctx, _res)      # 순수함수(관문3·단위테스트 공유)
                    _n = _res
                # ★grounding 플래그를 반환문 맨 앞에 붙인다 — 드롭된 미검증 operand를 에이전트가 보고
                #   레코드를 다시 읽게(가짜 정밀도 신뢰 차단·§2ab). 플래그 없으면 거동 변화 0.
                if _gflags:
                    _txt = ("[GROUNDING WARNING] %d input value(s) could not be verified against the "
                            "account records / knowledge base and were dropped: %s. Re-read the exact "
                            "value(s) from the records before relying on this result.\n%s"
                            % (len(_gflags), "; ".join(_gflags), _txt))
                # requestor는 tau2 원본과 동형으로 **미러링**(environment.get_response: requestor=message.requestor).
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"), content=_txt)
                print("[T2_SCAFFOLD_GET] %s -> %s" % (getattr(tc, "name"), _n), file=_sys.stderr, flush=True)
            elif getattr(tc, "name", None) in fa_map:
                # ★(W)wrap 기능서브: 자료-read를 서브가 소비·메인엔 답+원문 인용만. 실패=원 실행 폴백.
                _fa = fa_map[getattr(tc, "name")]
                _fc = getattr(self, "_t2_fa_cache", None)
                if _fc is None:
                    _fc = self._t2_fa_cache = {}
                _fk = (getattr(tc, "name", None),
                       json.dumps(getattr(tc, "arguments", None) or {}, sort_keys=True, ensure_ascii=False))
                if _fk in _fc:                            # 동일 질의 재위임 방지(자료-read=정적)
                    _wtxt = _fc[_fk]
                else:
                    def _run_fa(tcs, _self=self):
                        return orig_exec(_self, tcs)
                    _wtxt = _sub_wrap(self, _fa, tc, _run_fa)
                    if _wtxt is not None:
                        _fc[_fk] = _wtxt
                if _wtxt is not None:
                    ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                               requestor=getattr(tc, "requestor", "assistant"),
                                               content=_wtxt)
                else:
                    rest.append(tc)                       # 폴백 = 원 도구 실행(거동 변화 0)
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
        return _reassemble(tool_calls, ours, rest_res, ToolMessage)

    BaseOrchestrator._execute_tool_calls = exec2
    print("[T2_SCAFFOLD_GET] ON", file=_sys.stderr, flush=True)
    return True
