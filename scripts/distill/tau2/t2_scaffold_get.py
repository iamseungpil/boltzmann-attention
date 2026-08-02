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


def _tok_in(needle_n, hay_n):
    """토큰-경계 포함(정규화된 두 문자열). raw substring의 부분-단어 매칭('target'⊂'targeting')
    차단 — C276★① 교훈·QUOTE_GROUND_PINKIND_REDESIGN §2b 구현 규칙(닫힌 정련·판단 0)."""
    return bool(needle_n) and (" %s " % needle_n) in (" %s " % hay_n)


def _quote_pin_check(qp, v, r, quote_f, quote_min, docnorm):
    """★C279 pin_kind 라우팅 + **식별표 멤버십**(정본 = QUOTE_GROUND_PINKIND_REDESIGN rev2 §8b).
    순수함수=단위테스트 공유. LLM이 선언한 핀(policy_field)·종류(kind_field)에 **닫힌 필요조건만**:
      ⑴ quote가 주입 문서의 축자인가(날조 차단)  ⑵ 핀이 그 quote에서 복사된 것인가(복사 검증)
      ⑶ **A2 `policy_group_rows`(유한 열거) 멤버십** — 지시 동일성 판단은 *저작 시점*에 1회 끝났고
         런타임은 집합 조회만 한다. 유사도·접두·포함 대조 **전부 없음**(C279 ⒜: 그 부류는 코퍼스-우연
         필요조건이라 4연속 기각됨).
    ⑶의 결과는 셋으로 갈린다(§8b ⒢-1): 행∈집합=pass / 키는 있는데 행∉(공집합 포함)=**판단된 무대응**
    =reject_member / 키 자체가 없음=**표 공백**=lookup_missing(재질의→abstain+갱신 신호).
    반환 (verdict, info): pass|category|reject|reject_member|lookup_missing|kind_missing."""
    q = str((v or {}).get(quote_f) or "").strip()
    if not q:
        return "pass", None                       # 강등형 아님(quote 없음) → 라우팅 대상 아님
    qn = _norm_ground(q)
    if len(q) < int(quote_min or 0) or not _tok_in(qn, docnorm):
        return "reject", {"why": "quote_unverbatim"}
    kind = str((v or {}).get(str(qp.get("kind_field") or "")) or "").strip().lower()
    pin = str((v or {}).get(str(qp.get("policy_field") or "")) or "").strip()
    if kind not in ("named_merchant", "category"):   # 결측·열거 밖(오타)=동일 취급(발견 6)
        return "kind_missing", {"kind": kind, "pin": pin}
    pn = _norm_ground(pin)
    if not pn or not _tok_in(pn, qn):                # 복사 검증(핀↔quote·둘 다 LLM 산출/원문)
        return "reject", {"why": "pin_not_in_quote", "pin": pin}
    tbl = qp.get("policy_group_rows")
    if not isinstance(tbl, dict):                    # 표 미선언 = rev1 거동(멤버십 검사 없음)
        return ("category" if kind == "category" else "pass"), {"pin": pin}
    rows = None
    for k, vs in tbl.items():                        # 키 대조 = norm 정확-동등(포함 아님·§8b ⒢-3)
        if _norm_ground(k) == pn:
            rows = vs or []
            break
    if rows is None:
        # 표 공백. category = 열거 없는 산문 범주 ⇒ R2 그대로(통과+마크·열린 잔여).
        return ("category", {"pin": pin}) if kind == "category" else ("lookup_missing", {"pin": pin})
    rown = _norm_ground(str((r or {}).get(str(qp.get("row_field") or "")) or ""))
    if rown and any(_norm_ground(x) == rown for x in rows):
        return "pass", {"pin": pin}
    return "reject_member", {"pin": pin}


def _qp_note(tpl, info, row, qp):
    """A2 문구 템플릿 치환({pin}/{merchant}) — 문구=A2·엔진 기본값=도메인 어휘 0(발견 4)."""
    pin = str((info or {}).get("pin") or "")
    merch = str((row or {}).get(str(qp.get("row_field") or "")) or "")
    tpl = tpl or "the pinned name '{pin}' does not match this row's value '{merchant}' — the mapping was rejected."
    try:
        return tpl.format(pin=pin, merchant=merch)
    except Exception:
        return str(tpl)


def _split_missing_fields(mf, iso):
    """★C278 §2c: 결핍 필드의 출처별 분리 — record-유래(row_fields="call again" 정당) vs
    sub-유래(operand_schema=이행-불가 지시 금지→unverified 정직 표기). row_fields 우선·어느 쪽도
    아니면 sub-측(안전측·발견 5). C275 ⑤정정(모순 지시)의 직접 수정·멤버십 대조만(판단 0)."""
    recf = set((iso or {}).get("row_fields") or [])
    rec = {k: n for k, n in (mf or {}).items() if k in recf}
    sub = {k: n for k, n in (mf or {}).items() if k not in recf}
    return rec, sub


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
    if "user" in which:
        # ★C203: **손님 발화만**(도구 출력 제외). 'ledger'는 도구 출력을 포함해 **자기-그라운딩**이
        #   생긴다 — 도구가 한 번 뱉은 값은 그 다음 호출부터 무조건 '실재'가 된다(003 실측: 2번째
        #   호출부터 경고 소멸). 손님이 실제로 말한 것만 볼 때는 이 코퍼스를 쓴다.
        texts.append((_evidence_ctx(orch).get("__user_text") or ""))
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


def _window_coverage_note(d, ctx, res):
    """★2026-08-03 §4-2: 미측정 윈도 abstain의 **표면화**(순수함수=단위테스트 공유·`_render_scalar` 선례).
    abstain 자체는 op이 이미 했다(`t2_compute.group_reduce` → None) — 여기서는 "무엇이 비었나"를
    붙여 에이전트가 자기-수복(전체 참조 or id 위임)하게 한다. 엔진은 자기 집계의 전사만 하고
    도메인 결론부는 A2 `incomplete_hint`([[05]]·엔진 리터럴 0). 해당 없으면 빈 문자열(거동보존)."""
    grm = (ctx or {}).get("_gr_missing")
    if res is not None or not isinstance(grm, dict):
        return ""
    miss = list(grm.get("missing") or [])
    try:                                   # 라벨이 0-based 정수면 1-based 표기(표기만·판단 0)
        lbl = ", ".join("#%d" % (int(x) + 1) for x in miss)
    except Exception:
        lbl = ", ".join(str(x) for x in miss)
    txt = ("\n[coverage] %d of %d windows had NO input records at all (%s). Those windows were "
           "NOT measured, so this is NOT a computed shortfall — the tool refuses to issue a "
           "verdict on them." % (len(miss), grm.get("expected", 0), lbl))
    if d.get("incomplete_hint"):
        txt += " " + str(d.get("incomplete_hint"))
    return txt


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
    # (c) ★intent-field (C203·D4′ 재설계·2026-07-26 D4 폐기 후속): **값의 실재가 아니라 제약 의도의
    #   실재**를 본다. 값-존재 검사(구 D4)는 두 방향으로 다 틀렸다 — (i)006/023: 손님이 말한 *다른*
    #   값(소득 95000)을 엔진이 실재로 보고 통과시켜 발명 제약을 못 막았다(엔진은 필드 의미를 모른다)
    #   (ii)003: 손님의 정성 표현("no foreign transaction fees")을 모델이 0으로 수치화하자 '0이 원장에
    #   없다'며 **정당한 제약을 드롭**했다. ⇒ 새 술어: 이 파라미터의 **주제어(A2 `cue_any`)가 손님
    #   발화에 실재하는가**. 값은 보지 않는다(수치화=모델 몫·[[10]]). 불성립=드롭(제약 소멸=후보 확대=
    #   안전 방향). 엔진=부분문자열 대조만·도메인 어휘는 전부 A2.
    for itf in (gspec.get("intent_fields") or []):
        param = itf.get("param")
        if param not in ctx or ctx.get(param) in (None, "", []):
            continue
        cues = [_norm_ground(c) for c in (itf.get("cue_any") or [])]
        if not any(cues):
            continue
        utext = _norm_ground(" ".join(_corpus_texts(orch, itf.get("corpus") or ["user"])))
        if not any(c and c in utext for c in cues):
            flags.append("%s=%s (the customer never mentioned this kind of requirement — "
                         "do not add limits they did not state)" % (param, ctx.get(param)))
            if itf.get("on_fail", "drop") == "drop":
                ctx[param] = None
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


def _sub_fetch_formalize(orch, d, iso, ctx, run_env_calls):
    """★fetch-first 격리 서브 (2026-07-20·023 컨텍스트 초과·isolate-승격·§2ah·사용자 지시).
    문제: 계산도구(check_rebate 등)가 `transactions`(전체 리스트) 등을 **인자로** 받으면 에이전트가
      레코드를 main 컨텍스트로 읽어 넘겨야 한다 — 그 read+인자에코가 main 컨텍스트를 부풀린다(023 overflow).
    이 모드: 에이전트는 **참조(`iso['ref_params']`·예 account id)만** 넘기고, 서브가 getter_tools로 레코드를
      **off-ledger fetch** + 전체 operand dict를 formalize → 메인은 참조+결과만 본다(레코드 read 0=진짜 turn-free).
    기존 `_sub_formalize`(row-기반·리스트가 이미 ctx에 있어야)와 대비 — 이 모드는 **fetch-first**(참조→서브가 읽음).
    - 메인 턴 소모 0(서브 generate/도구호출 state.messages 미기입·_sub_formalize와 동형).
    - GET=진짜 getter(env 결정론 실행)·엔진 리터럴 0(도구명·지시·형식·operand_keys 전부 A2·[[03b]]).
    반환: operand dict(top-level·ctx.update용) · 실패=None(폴백=에이전트 인자·거동보존)."""
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage
    ag = getattr(orch, "agent", None)
    if ag is None:
        return None
    ref = {k: ctx.get(k) for k in (iso.get("ref_params") or []) if ctx.get(k) is not None}
    if not ref:
        print("[T2_SG_ISOLATE] fetch: ref_params 부재 → 격리 생략", file=_sys.stderr, flush=True)
        return None
    tools = [t for t in (getattr(ag, "tools", None) or [])
             if getattr(t, "name", None) in set(iso.get("getter_tools") or [])]
    if not tools:
        print("[T2_SG_ISOLATE] fetch: getter_tools 부재 → 격리 생략", file=_sys.stderr, flush=True)
        return None
    keys = set(iso.get("operand_keys") or [])
    if not keys:
        print("[T2_SG_ISOLATE] fetch: operand_keys 미선언 → 격리 생략", file=_sys.stderr, flush=True)
        return None
    prompt = "%s\n\n=== REFERENCE ===\n%s\n\n%s" % (
        iso["instructions"], json.dumps(ref, ensure_ascii=False, indent=1), iso["answer_format"])
    try:
        um = UserMessage(role="user", content=prompt)
    except TypeError:
        um = UserMessage(content=prompt)
    msgs = [um]
    kw = {k: v for k, v in dict(getattr(ag, "llm_args", None) or {}).items() if "tool" not in k}
    if iso.get("temperature") is not None:
        kw["temperature"] = iso["temperature"]
    queries = []
    _maxr = int(iso.get("max_rounds", 4))
    _gfb = 0                      # ★서브-내 ground 피드백 발화 수(T2_SG_ISOFB·관측용)
    _ok_outs, _err_outs = [], []  # ★서브 getter 성공/실패 출력(§2be·0.0-주입 차단+비수렴 원인 관측)
    for rnd in range(_maxr):
        try:
            # ★마감 라운드(2026-07-21 §2ba·r095e/f 실측: 서브가 라운드 내내 getter만 돌고 답을 안 내
            #   소진→폴백): 마지막 라운드는 **도구 없이** 생성 — 구조적으로 tool-call 불가 → JSON 답 강제.
            _last = (rnd == _maxr - 1)
            resp = la.generate(model=ag.llm, tools=(None if _last else tools), messages=msgs,
                               call_name="sg_fetch_iso",
                               **(dict(kw, tool_choice="required") if rnd == 0 else kw))
        except Exception as e:
            print("[T2_SG_ISOLATE] fetch generate 실패(%d라운드): %r" % (rnd, e), file=_sys.stderr, flush=True)
            _isolate_trace(iso, d, {"error": str(e)[:200], "round": rnd, "queries": queries})
            return None
        tcs = list(getattr(resp, "tool_calls", None) or [])
        if tcs:
            for _tc in tcs:
                _fn = getattr(_tc, "function", None) or _tc
                queries.append(getattr(_tc, "name", None) or getattr(_fn, "name", None))
            msgs.append(resp)
            _res = run_env_calls(tcs)                # ★GET = env 결정론 실행(off-ledger)
            for _rm in _res:
                (_err_outs if getattr(_rm, "error", False) else _ok_outs).append(
                    str(getattr(_rm, "content", "") or ""))
            # ★서브 tool 출력 절단 (2026-07-22 §2bu·rall11 097 실측: 서브가 getter(KB APY규칙·
            #   전체계좌)를 라운드마다 누적→52854 tokens > 48640 max=ContextWindowExceededError→서브
            #   실패·폴백→메인 오추측(95000) 사용=097 부하의 인프라 형태). 서브는 값-추출이 목적이라
            #   전체 문서 불요·값은 출력 앞부분(레코드 필드)에 실재 → 서브 뷰만 절단(메인 무관·비커밋=
            #   replay-safe·_ok_outs 원문은 위에서 이미 확보=마감검증 무영향). 엔진 순수 절단(리터럴 0).
            # ★값-보존 절단 (2026-07-22 §2bv 강건화·rall12 097 실측: head-only 절단이 KB APY 규칙
            #   (문서 뒤/중간)을 잘라 actual_apy 산출 실패→GROUNDING 드롭→IC 14회 루프·apply 미도달).
            #   head+tail 절단(VIEW_COMPACT 동형): 레코드 필드(앞)+계산 규칙/예시(뒤) 둘 다 보존.
            _stc = int(os.environ.get("T2_SG_SUB_TOOLCAP", "4000"))

            def _capm(_m, _cap=_stc):
                _c = getattr(_m, "content", None)
                if isinstance(_c, str) and len(_c) > _cap:
                    _hd = int(_cap * 0.65)
                    _tl = _cap - _hd
                    _d = (_c[:_hd] + "\n...[middle truncated for sub-extraction; head+tail kept]...\n"
                          + _c[-_tl:])
                    try:
                        return _m.model_copy(update={"content": _d})
                    except Exception:
                        import copy as _cp
                        _m2 = _cp.copy(_m)
                        try:
                            _m2.content = _d
                            return _m2
                        except Exception:
                            return _m
                return _m
            msgs.extend(_capm(_rm) for _rm in _res)
            continue
        got = _merge_json(getattr(resp, "content", None) or "", keys)
        # ★무근거-답 차단 (2026-07-21 §2be·rall4 실측): interest 서브가 getter 전패(성공 출력 0)인
        #   채로 마감 라운드 강제-답에서 {principal:0.0, actual_apy:0.0} 리터럴을 주입 — §2as
        #   0.0-포이즈닝의 신형 재발(전 trial·t2 PASS는 에이전트가 판정 무시한 덕). 규칙(도메인일반):
        #   ①성공 도구출력 0건 → 답을 버리고 None(=에이전트-인자 폴백·거동보존 경로)
        #   ②마감 라운드(도구 없는 강제-답)의 스칼라 operand는 **서브 자신의 성공 출력**에 숫자-실재해야
        #     주입 — 아니면 None. (메인-원장 대조는 0.00 편재로 무력·§2be. 배열 operand는 ISOFB/관문1 관할.)
        _last_forced = (rnd == _maxr - 1)
        if got and not _ok_outs:
            print("[T2_SG_ISOLATE] fetch %s: 성공 getter 출력 0 → 답 폐기·폴백 (err=%s)"
                  % (d.get("name"), (_err_outs[0][:80] if _err_outs else "none")),
                  file=_sys.stderr, flush=True)
            _isolate_trace(iso, d, {"mode": "fetch", "round": rnd + 1, "queries": queries,
                                    "ground_fb": _gfb, "ok_outs": 0, "err_outs": len(_err_outs),
                                    "err0": (_err_outs[0][:120] if _err_outs else None),
                                    "discarded": got})
            return None
        if got and _last_forced:
            _bad = []
            for _k, _v in got.items():
                _fv = _as_float(_v)
                if _fv is not None and not any(abs(_fv - n) < 1e-9
                                               for t in _ok_outs for n in _nums_in(t)):
                    _bad.append("%s=%s" % (_k, _v))
            if _bad:
                print("[T2_SG_ISOLATE] fetch %s: 마감-답 값이 서브 출력에 부재(%s) → 폐기·폴백"
                      % (d.get("name"), "; ".join(_bad)), file=_sys.stderr, flush=True)
                _isolate_trace(iso, d, {"mode": "fetch", "round": rnd + 1, "queries": queries,
                                        "ground_fb": _gfb, "ok_outs": len(_ok_outs),
                                        "err_outs": len(_err_outs), "forced_bad": _bad,
                                        "discarded": got})
                return None
        # ★서브-내 ground 피드백 (T2_SG_ISOFB=1·2026-07-21 §2bb·r095g 실측): 서브 답을 메인 관문1과
        #   동일한 A2 `ground` 선언으로 즉석 검증 — 실패 플래그를 **검색 도구를 쥔 서브**에게 되먹여
        #   같은 루프서 재검색 기회를 준다(현행은 메인 쪽 드롭이라 서브가 실패를 모른 채 종료 —
        #   r095g: checking 값-없는 인용 4-trial 반복). 라운드 소진 임박(마감 라운드)이면 현행대로
        #   반환 = 거동보존·메인 관문1이 재검증(심층방어). 엔진=검증+반사만·값 생성=LLM([[03b]]/[[10]]).
        if (got and os.environ.get("T2_SG_ISOFB") == "1" and rnd < _maxr - 1
                and isinstance(d.get("ground"), dict)):
            try:
                _fl = _ground_operands(orch, d, dict(got))
            except Exception as _ge:
                _fl = []
                print("[T2_SG_ISOLATE] ground-피드백 검사 실패(no-op): %r" % (_ge,),
                      file=_sys.stderr, flush=True)
            if _fl:
                _gfb += 1
                msgs.append(resp)
                _fbt = ("GROUNDING CHECK FAILED - these items were rejected: %s. An item is only "
                        "accepted when its 'source' is a quote copied VERBATIM from a document or "
                        "record AND that quote itself contains the exact numeric value. Do not "
                        "guess or infer values. Search again for the line or table that states "
                        "the exact number, then re-send the complete JSON answer."
                        % "; ".join(_fl))
                try:
                    msgs.append(UserMessage(role="user", content=_fbt))
                except TypeError:
                    msgs.append(UserMessage(content=_fbt))
                print("[T2_SG_ISOLATE] fetch %s: ground-피드백 %d건 → 서브 재시도(%d라운드)"
                      % (d.get("name"), len(_fl), rnd + 1), file=_sys.stderr, flush=True)
                continue
        getter = sum(1 for m in msgs if getattr(m, "role", "") == "tool")
        print("[T2_SG_ISOLATE] fetch %s: %d라운드·getter %d회·operand keys=%s"
              % (d.get("name"), rnd + 1, getter, list(got or {})), file=_sys.stderr, flush=True)
        _isolate_trace(iso, d, {"mode": "fetch", "round": rnd + 1, "getter": getter,
                                "queries": queries, "ground_fb": _gfb,
                                "ok_outs": len(_ok_outs), "err_outs": len(_err_outs),
                                "err0": (_err_outs[0][:120] if _err_outs else None),
                                "operands": got})
        return got or None
    print("[T2_SG_ISOLATE] fetch %s: max_rounds 소진 → 격리 생략" % d.get("name"),
          file=_sys.stderr, flush=True)
    _isolate_trace(iso, d, {"mode": "fetch", "error": "max_rounds", "queries": queries,
                            "ground_fb": _gfb})
    return None


def _sub_inject(orch, d, iso, ctx, la, UserMessage):
    """★★재설계 격리(§2e·2026-07-18 실증 105/105): 카드당 격리 + 문서 주입(검색 0) + grounding.
    거래를 `group_by`(레코드 필드)로 그룹핑 → 그룹마다 그 그룹값의 문서를 제목접두로 주입 →
    서브가 `{base_rate, exclusion_quote}` formalize → 엔진 grounding(quote∈문서면 0 유지·아니면 default 백필).
    엔진 리터럴 0: 그룹키·필터규칙·계약문 전부 A2·값/인용 전부 LLM이 KB서·엔진은 substring+백필만."""
    ag = orch.agent
    orch._t2_qp_notes = []          # ★C278: quote-pin 사유/마크 수집(반환문 표면화·호출당 리셋)
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
            # ★C278 quote-pin 라우팅 (T2_QUOTE_PIN=1 ∧ A2 `quote_pin` 선언 시 — C197 검사 대체.
            #   QUOTE_GROUND_PINKIND_REDESIGN §2b: named=핀 축자 포함+선행 앵커·category=검사 미적용+
            #   표면화·결측/열거밖=재질의→abstain. 재질의=R4: guard-불성립 행만·sg_inject_retry 경로 1회.)
            quote_f = iso.get("quote_field", "exclusion_quote")
            qp = iso.get("quote_pin") if os.environ.get("T2_QUOTE_PIN") == "1" else None
            qp_verdicts = {}
            if qp and got:
                _qmin = iso.get("quote_min") or 0
                _rowof = {str(r.get(id_field)): r for r in grows}
                for tid0, r0 in _rowof.items():
                    qp_verdicts[tid0] = _quote_pin_check(qp, got.get(tid0) or {}, r0,
                                                         quote_f, _qmin, docnorm)
                # 재질의 대상(§8b ⒢-2·R4) = 조각-복사·오타·미선언처럼 **고쳐 쓸 수 있는** 것만.
                #   `reject_member`(판단된 무대응)·`quote_unverbatim`(날조)은 확정이라 재질의 안 한다.
                _RETRY_V = ("lookup_missing", "kind_missing")
                _bad = [t0 for t0, (vd0, i0) in qp_verdicts.items()
                        if (vd0 in _RETRY_V
                            or (vd0 == "reject" and (i0 or {}).get("why") == "pin_not_in_quote"))
                        and (got.get(t0) or {}).get(rate_f) is not None]
                _rp = qp.get("lookup_retry_prompt") or qp.get("retry_prompt")
                if _bad and _rp:
                    _fb = "\n".join("- %s: %s" % (t0, _qp_note(
                        qp.get("lookup_note") if qp_verdicts[t0][0] == "lookup_missing"
                        else qp.get("reject_note"), qp_verdicts[t0][1], _rowof[t0], qp))
                        for t0 in _bad)
                    extra2 = "\n\n\u2605FEEDBACK on item(s) %s:\n%s\n%s" % (
                        ", ".join(_bad), _fb, _rp)
                    try:
                        um3 = UserMessage(role="user", content=prompt + extra2)
                    except TypeError:
                        um3 = UserMessage(content=prompt + extra2)
                    try:
                        resp3 = la.generate(model=ag.llm, tools=None, messages=[um3],
                                            call_name="sg_inject_retry", **kw)
                        got3 = _merge_json(getattr(resp3, "content", None) or "", set(_bad))
                    except Exception as e:
                        print("[T2_SG_ISOLATE] quote-pin retry 실패(%s): %r" % (gval, e),
                              file=_sys.stderr, flush=True)
                        got3 = {}
                    for t0 in _bad:
                        if got3.get(t0):
                            got[t0] = got3[t0]
                            qp_verdicts[t0] = _quote_pin_check(qp, got3[t0], _rowof[t0],
                                                               quote_f, _qmin, docnorm)
                            g_retry += 1
            # ★consensus·default백필 제거됨(2026-07-19 사용자 지시·[[10]]·§2k) — 엔진은 서브 operand를
            #   그대로 병합만(값 생성/override 0). 서브 오류는 원천(프롬프트·max_batch)서 수정.
            for r in grows:
                tid = str(r.get(id_field))
                v = got.get(tid) or {}
                # quote·핀·종류 = grounding/라우팅 전용 → op operand에선 제외
                _meta_f = {quote_f}
                if qp:
                    _meta_f |= {str(qp.get("policy_field") or ""), str(qp.get("kind_field") or "")}
                merged = {k: val for k, val in v.items() if k not in _meta_f}
                # ★C197 quote-grounding(A2 `quote_must_contain_field` 선언 시만·미선언=거동 변화 0):
                #   서브가 exclusion으로 rate를 강등하며 붙인 quote를 엔진이 **결정론 대조**만 한다 —
                #   (a) quote가 실제 주입 문서의 축자인가(docnorm substring) (b) quote 안에 이 행의
                #   선언 필드값(예: merchant)이 실재하는가. 019 실측: ThredUp 제외문을 Thrive Market에
                #   오적용(이웃-상인 혼동)→false-negative. 불성립이면 rate 드롭=판정불가 abstain
                #   (엔진은 값 생성/승격 0·[[03b]]·문자열 포함 검사만).
                if qp:
                    # ★C278 판정 적용: reject/kind_missing=rate 드롭+사유 표면화(A2 reject_note) /
                    #   category=rate 유지+마크 표면화(A2 category_note·R2 "통과+마크") / pass=무개입.
                    _vd, _inf = qp_verdicts.get(tid, ("pass", None))
                    if _vd in ("reject", "reject_member", "lookup_missing", "kind_missing") \
                            and merged.get(rate_f) is not None:
                        merged.pop(rate_f, None)
                        _tpl = {"reject_member": qp.get("member_note"),
                                "lookup_missing": qp.get("lookup_note")}.get(_vd) or qp.get("reject_note")
                        getattr(orch, "_t2_qp_notes", []).append(_qp_note(_tpl, _inf, r, qp))
                        print("[T2_SG_ISOLATE] quote-pin %s: %s(%s) → rate 드롭(abstain)"
                              % (_vd, tid, str((_inf or {}).get("pin") or (_inf or {}).get("why") or "")[:40]),
                              file=_sys.stderr, flush=True)
                    elif _vd == "category" and merged.get(rate_f) is not None:
                        getattr(orch, "_t2_qp_notes", []).append(
                            _qp_note(qp.get("category_note"), _inf, r, qp))
                _mcf = None if qp else iso.get("quote_must_contain_field")
                if _mcf and merged.get(rate_f) is not None:
                    _q = str(v.get(quote_f) or "").strip()
                    if _q:
                        _qn = _norm_ground(_q)
                        _fv = _norm_ground(str(r.get(_mcf) or ""))
                        _qok = (len(_q) >= int(iso.get("quote_min") or 0)) and \
                               (_qn in docnorm) and bool(_fv) and (_fv in _qn)
                        if not _qok:
                            merged.pop(rate_f, None)
                            print("[T2_SG_ISOLATE] quote-ground 불성립: %s(%s) → rate 드롭(abstain)"
                                  % (tid, str(r.get(_mcf) or "")[:40]), file=_sys.stderr, flush=True)
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


class _ByrefError(Exception):
    """P6 참조-해석 실패(모델에게 그대로 통지되는 메시지)."""


def _dup_stub_content(n, prev=None, represent_on=False, shrunk=False):
    """★P8(C208①) 순수함수: DUPLICATE-COMPUTE 스텁 본문. represent_on(∧이전 결과 실재∧n≤2∧
    천장 비근접)이면 이전 결과를 재게시 — '위 출력 참조' 지시만으로는 재호출 유인을 못 끊는다
    (day5 020: 동일 인자 5회=창 29%). 상한·shrink-생략=W-d."""
    rep = ""
    if represent_on and prev and n <= 2 and not shrunk:
        rep = " Previous result (unchanged): %s" % prev
    extra = ("" if n < 3 else
             " You have now repeated this exact call %d times — STOP repeating it. Use the "
             "values already returned to take the next concrete step, or change the arguments "
             "if you meant a different computation." % (n + 1))
    return ("[DUPLICATE-COMPUTE] This exact call (same tool, same arguments) was already "
            "executed; this tool is deterministic, so the same arguments always return the "
            "SAME result — refer to the earlier output instead of re-computing.%s%s"
            % (rep, extra)), bool(rep)


def _parse_record_dump(text):
    """★P6(DAY5_PRESCRIPTIONS §P6): env 기계 포맷("Found N record(s) … Record ID: <id>
    <field>: <value> …") **전용** 결정론 파서. [[03b]] 경계: NL formalize가 아니라 env가 찍는
    고정 포맷의 전사이며, 다른 텍스트는 assert로 거부(경계-확장 선례 방지·리뷰 지시).
    값 정규화도 같은 포맷-층만: '$'/천단위 콤마/' points' 접미 — 표기 전사이지 판단 아님."""
    import re as _re2
    if "Record ID:" not in (text or ""):
        # ★P7 (2026-08-02·028 사슬): 거부만 하고 **다음 올바른 행동을 지목하지 않아** 에이전트가
        #   손-전사 폴백으로 이탈했다(→ [ARGS-FORMAT] 거부 → 가짜 음성 → 왜곡 id 제출). 참조 가능한
        #   출력의 조건을 명시하고 손-전사를 금지한다. 판정 불변(순수 문구·도메인 리터럴 0).
        raise _ByrefError(
            "the referenced output is not a record dump (no 'Record ID:' lines) — "
            "@last:/@call: may only reference the output of a record-read tool "
            "(one that printed \"Found N record(s)\" with 'Record ID:' lines). "
            "Re-read the records with that tool and reference ITS output. "
            "Do NOT hand-copy or re-type the rows into the argument — transcribed values "
            "are a common source of wrong ids.")
    parts = _re2.split(r"Record ID: ([A-Za-z0-9_\-]+)", text)
    rows = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1]
        row = {}
        for m in _re2.finditer(r"^\s{1,8}([A-Za-z_][A-Za-z0-9_]*):\s*(.+?)\s*$", body, _re2.M):
            k, v = m.group(1), m.group(2)
            mv = _re2.match(r"^\$?([\d,]+(?:\.\d+)?)(?:\s*points?)?$", v)
            row[k] = mv.group(1).replace(",", "") if mv else v
        if row:
            rows.append(row)
    if not rows:
        raise _ByrefError("the referenced record dump contained no parseable records")
    return rows


def _resolve_ref_output(orch, ref):
    """@last:<도구명> → 그 도구의 최신 비에러 커밋 출력 / @call:<tool_call_id> → 해당 결과."""
    kind, _, key = ref.partition(":")
    key = key.strip()
    if not key:
        raise _ByrefError("empty reference '%s'" % ref)
    id2name, best = {}, None
    for m in orch.get_messages():
        for tc in (getattr(m, "tool_calls", None) or []):
            id2name[getattr(tc, "id", None)] = getattr(tc, "name", None)
        if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        mid = getattr(m, "id", None)
        if kind == "@call" and mid == key:
            return c
        if kind == "@last" and id2name.get(mid) == key:
            best = c                                   # 마지막 것 유지
    if best is None:
        raise _ByrefError("no committed non-error output of '%s' found in this conversation "
                          "— call that tool first, then reference it" % key)
    return best


def _over_params(op):
    """★2026-08-03 §4-1 (핸드오프 HANDOFF_2026_08_02_NIGHT §4-1·byref 구조 결함): op **트리 전수**에서
    배열-파라미터로 참조되는 `over` 이름을 선언 순서대로 수집한다.
    구판은 `(d["op"]).get("over")` = **최상위 op의 over만** 읽었다. `check_rebate_qualification`처럼
    over가 중첩(`op.cond.a.over`)이면 항상 None ⇒ 모든 byref가 "이 도구는 by-ref 인자가 없다"로
    거부됐다(023 라이브: 에이전트의 **옳은** 전체-거래 참조 시도를 차단 → 손-전사 1건 폴백 →
    엔진이 그 1건으로 부정 판정. 028 손-전사 사슬과 공통 원인).
    워크는 순수 구조 순회다 — op 이름·도메인 어휘를 보지 않는다(엔진 리터럴 0·[[05]])."""
    out, seen = [], set()

    def _walk(o):
        if isinstance(o, dict):
            v = o.get("over")
            if isinstance(v, str) and v and v not in seen:
                seen.add(v)
                out.append(v)
            for x in o.values():
                _walk(x)
        elif isinstance(o, list):
            for x in o:
                _walk(x)
    _walk(op)
    return out


def _primary_over(d, ctx=None):
    """byref rows(=조인 대상)를 담은 파라미터. 트리 수집분 중 **ctx에 리스트로 실재하는 첫 번째**,
    없으면 선언 순서 첫 번째. 최상위 over만 있는 기존 스펙에서는 구판과 동일값(거동보존)."""
    ov = _over_params(d.get("op"))
    if isinstance(ctx, dict):
        for k in ov:
            if isinstance(ctx.get(k), list):
                return k
    return ov[0] if ov else None


def _byref_resolve(orch, d, ctx):
    """★P6 엔진: op의 `over` 배열 인자에 한해 @last:/@call: 참조를 해석해 rows로 치환.
    fetch는 모델이 이미 수행·커밋한 것만 재사용(autofetch 아님·E-PLAN C101 기계-파싱 선례).
    ★비-over(스칼라/행-필드) 인자의 참조는 이번 판에서 **미지원**(도메인 필드명 join을 엔진에
    박는 것은 [[05]] 위반 소지 — A2 join-spec 설계 후 별도). 시도 시 명확한 에러.
    ★2026-08-03: 허용 인자는 **중첩 op 트리 전수**에서 도출한다(§4-1·`_over_params`)."""
    overs = _over_params(d.get("op"))
    join_params = {(js or {}).get("from_ref_param") or t
                   for t, js in (d.get("byref_join") or {}).items()}
    for k in list(ctx.keys()):
        v = ctx.get(k)
        if not (isinstance(v, str) and (v.startswith("@last:") or v.startswith("@call:"))):
            continue
        if k in join_params:
            continue                                   # F7b: _byref_join이 처리
        if k not in overs:
            # ★P7⑥ (2026-08-02·023 실측): 구 문구는 `over`가 None인 스펙에서 "only the 'None'
            #   argument supports…"로 렌더돼 **깨진 안내**가 됐고(에이전트의 옳은 byref 시도를 차단),
            #   그 뒤 에이전트가 손-전사 폴백으로 이탈했다(028 사슬과 동형). ⇒ 허용 인자를 정확히
            #   선언하고, 허용 인자가 아예 없으면 그 사실을 말한다. 엔진 판정은 불변(순수 문구).
            _allow = (", ".join("'%s'" % o for o in overs) if overs
                      else "(none — this tool takes no by-reference argument)")
            raise _ByrefError(
                "the '%s' argument does not support @last:/@call: references. "
                "By-reference is accepted only for: %s. "
                "Provide '%s' as a literal value copied from the records."
                % (k, _allow, k))
        rows = _parse_record_dump(_resolve_ref_output(orch, v))
        _byref_map_fields(d, rows)                     # A2 선언 컬럼명 대응(§4-1 후속)
        _byref_require_fields(d, k, rows)              # 필요한 컬럼 부재 = 침묵 대신 지목
        ctx[k] = rows
        print("[T2_SG_BYREF] %s: '%s' resolved by reference -> %d row(s)"
              % (d.get("name"), v, len(rows)), file=_sys.stderr, flush=True)


_ROW_FIELD_KEYS = ("date_field", "value_field", "id_field", "actual_field", "cond_field")


def _row_fields(op):
    """op 트리가 **입력 행에서 직접 읽는** 필드명 집합. 엔진이 아는 op 키(`*_field`)만 본다 —
    도메인 어휘는 A2 값이고 엔진은 키 이름만 안다([[05]]). `out_field`(엔진 산출)·join의
    `source_field`/`row_field`(소스 덤프 몫)는 제외."""
    out = []

    def _walk(o):
        if isinstance(o, dict):
            for kk in _ROW_FIELD_KEYS:
                vv = o.get(kk)
                if isinstance(vv, str) and vv and vv not in out:
                    out.append(vv)
            for x in o.values():
                _walk(x)
        elif isinstance(o, list):
            for x in o:
                _walk(x)
    _walk(op)
    return out


def _byref_map_fields(d, rows):
    """★2026-08-03 §4-1 후속: A2 `byref_field_map`(op 필드명 ← 레코드 덤프 컬럼명) 결정론 복사.
    필요한 이유(023): 참조가 열려도 **덤프 컬럼명과 op 필드명이 다르면** 전 행이 무효가 되어
    빈 집계가 된다. 매핑값은 전부 A2(=env 레코드 스키마 기계-도출·[[23]] opex 0)·엔진=복사만."""
    fm = d.get("byref_field_map") or {}
    if not isinstance(fm, dict):
        return
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        for tgt, src in fm.items():
            if r.get(tgt) in (None, "") and r.get(src) not in (None, ""):
                r[tgt] = r[src]


def _byref_require_fields(d, param, rows):
    """참조된 덤프에 op이 요구하는 컬럼이 **하나도 없으면** 조용히 빈 판정으로 가지 않고 지목한다
    (023 사슬의 침묵 지점). 어느 컬럼이 실재하는지도 함께 알려 다음 행동을 결정 가능하게 한다."""
    need = _row_fields(d.get("op"))
    if not need or not rows:
        return
    have = set()
    for r in rows:
        if isinstance(r, dict):
            have |= {k for k, v in r.items() if v not in (None, "")}
    miss = [f for f in need if f not in have]
    if not miss:
        return
    raise _ByrefError(
        "the referenced records do not contain the field(s) %s that this computation reads from "
        "each row (the referenced records provide: %s). Reference the output of the tool whose "
        "records carry those values, or supply '%s' yourself with those fields filled in — and if "
        "this tool can read the records itself from an id argument, prefer that."
        % (", ".join("'%s'" % m for m in miss), ", ".join(sorted(have)) or "(no fields)", param))


def _byref_join(orch, d, ctx):
    """★F7b(C211·DAY7 §F7b): A2-선언 일반 equijoin — byref rows(거래 덤프 유래)에 없는 필드
    (account_open)를 참조된 소스 덤프에서 결정론 복사. 필드명 전부 A2 데이터(엔진=일반 실행기·
    [[05]])·**유일 매칭만 유효**(복수 매칭=불성립=abstain 안전측·리뷰 명세 보강1 — "첫 행" 채택은
    침묵-오값(D4형))·불성립 행=필드 미기입(P4 지목 경로 합류)."""
    spec = d.get("byref_join") or {}
    over = _primary_over(d, ctx)                    # ★2026-08-03 §4-1: 중첩 op 트리 대응
    rows = ctx.get(over)
    for tgt, js in spec.items():
        p = (js or {}).get("from_ref_param") or tgt
        v = ctx.get(p)
        if not (isinstance(v, str) and (v.startswith("@last:") or v.startswith("@call:"))):
            continue
        txt = _resolve_ref_output(orch, v)
        sel = str((js or {}).get("source_selector") or "").lower()
        if sel and sel not in txt.lower():
            raise _ByrefError("the output referenced for '%s' does not contain the expected "
                              "source records ('%s') — reference the read whose output holds "
                              "them" % (p, js.get("source_selector")))
        src = _parse_record_dump(txt)
        mf = js["match"]["row_field"]
        sf = js["match"]["source_field"]
        take = js["take"]
        groups = {}
        for s in src:
            groups.setdefault(str(s.get(sf, "")).strip().lower(), []).append(s)
        joined = amb = miss = 0
        for r in (rows or []):
            if not isinstance(r, dict):
                continue
            cand = groups.get(str(r.get(mf, "")).strip().lower()) or []
            if len(cand) == 1 and cand[0].get(take) not in (None, ""):
                r[tgt] = cand[0][take]
                joined += 1
            elif len(cand) > 1:
                amb += 1                                # 유일-매칭만 유효
            else:
                miss += 1
        if p != over:
            ctx.pop(p, None)                            # 소비된 참조 파라미터 제거
        print("[T2_SG_BYREF] %s: join '%s' -> joined=%d ambiguous=%d unmatched=%d"
              % (d.get("name"), tgt, joined, amb, miss), file=_sys.stderr, flush=True)


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


def _truth_text(outer, tn):
    """★SG_TRUTH 문구 단일 정본 (2026-07-20 replay-safety·§2aj): exec2 라이브 intercept와
    env-레벨 replay 패치가 **바이트 동일** 텍스트를 내야 eval replay 내용비교가 통과한다."""
    return ("`%s` is not managed by `%s`. `%s` is already one of the tools provided to you — "
            "call `%s` directly with its arguments." % (tn, outer, tn, tn))


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
        # ★C204/D9(2026-07-27·day3 019 실측): "get reward discrepancies"(공백형)로 KB를 6회+ 검색하며
        #   자기 도구를 끝내 못 찾음 — 정확-일치만 보던 게 사각. **정규화-동등**으로 확장(여전히 값
        #   전체의 equality — 산문 부분일치는 안 보므로 기존 오탐 방어 유지).
        if isinstance(_v, str):
            _nv = re.sub(r"[^a-z0-9]+", "_", _v.lower()).strip("_")
            for _dn in decls:
                if _nv == re.sub(r"[^a-z0-9]+", "_", str(_dn).lower()).strip("_"):
                    return _dn
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
            # ★P6(§P6): BYREF ON일 때만 over-인자 설명에 참조 문구 부가 — OFF면 문구도 없음
            #   (엔진이 해석 못 하는 지시를 모델에게 주지 않는다·문구=도메인-일반·A2 파일 불변).
            if os.environ.get("T2_SG_BYREF") == "1":
                # ★2026-08-03 §4-1: 안내도 **중첩 op 트리 전수**에서 도출한다. 구판(최상위 over만)은
                #   rebate처럼 over가 중첩된 도구에서 안내 자체가 없었다 = 어포던스 비가시.
                _ovks = [o for o in _over_params(d.get("op"))
                         if isinstance((d.get("params") or {}).get(o), str)]
                if _ovks:
                    d = dict(d)
                    d["params"] = dict(d["params"])
                    for _ovk in _ovks:
                        d["params"][_ovk] += (
                            " INSTEAD of retyping the rows, you MAY pass the string "
                            "\"@last:<name of the tool whose output contains these records>\" — "
                            "the deterministic system will reuse that exact earlier output.")
                    # ★F7b: join-선언 파라미터에도 참조 안내(예: account_open="@last:<accounts read>")
                    for _jt, _js in (d.get("byref_join") or {}).items():
                        _jp = (_js or {}).get("from_ref_param") or _jt
                        if isinstance(d["params"].get(_jp), str):
                            d["params"][_jp] += (
                                " If you pass the rows by \"@last:\" reference, you MAY also pass "
                                "this argument as \"@last:<name of the tool whose output contains "
                                "the records holding this value>\" — the deterministic system will "
                                "copy it into each row by exact record match.")
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
                # ★read-선행 게이트 (T2_SG_REQREADS=1·§2aw·r095 gather-순서 실측: 계산 前 저축 레코드·
                #   거래 read 미실행이 3trial 불변). A2 `requires_reads`(effective 도구명) 미실행이면 계산을
                #   거부하고 read 지시 — §1.5 허용축(read 강제=write 강제 아님)·엔진=집합 대조만·리터럴 0.
                #   우리 도구는 env 부재라 replay서 hallucinated-skip → replay-safe.
                _rr = d.get("requires_reads") or []
                if _rr and os.environ.get("T2_SG_REQREADS") == "1":
                    try:
                        _effc = {_g._eff_tool_name(_t2) for _m2 in self.get_messages()
                                 for _t2 in (getattr(_m2, "tool_calls", None) or [])}
                    except Exception:
                        _effc = None
                    _missing_r = [r for r in _rr if r not in _effc] if _effc is not None else []
                    if _missing_r:
                        # ★문구=A2 오버라이드 가능(`requires_reads_feedback`·{missing} 치환).
                        # ⚠C241 U6d 교정: 구 주석은 "기본 문구는 도메인 어휘 0"이라 **주장했으나
                        #   실제로는** `unlock_discoverable_agent_tool`·`call_discoverable_agent_tool`
                        #   두 도구명을 박고 있었다(자기모순 — prekb `:198`↔`:593`과 동일 계열).
                        #   이제 A2 `eplan`에서 읽고, 미선언 도메인이면 그 문장을 **생략**한다.
                        _ep_r = ((a2 or {}).get("eplan") or {})
                        _unl_r, _cal_r = _ep_r.get("unlock_tool"), _ep_r.get("dispatch_tool")
                        _fb_r = d.get("requires_reads_feedback")
                        if _fb_r:
                            _msg_r = _fb_r.replace("{missing}", ", ".join(_missing_r))
                        elif not (_unl_r and _cal_r):
                            # dispatcher 미선언 = 접미사/잠금해제 개념이 없는 도메인 → 구조 안내 생략
                            _msg_r = ("Error: [READ-FIRST] this calculation depends on records you have "
                                      "not read yet in this conversation. Missing required reads (BASE "
                                      "names): %s. Read them first, then retry this calculation."
                                      % ", ".join(_missing_r))
                        else:
                            _msg_r = ("Error: [READ-FIRST] this calculation depends on records you have not read "
                                      "yet in this conversation. Missing required reads (BASE names): %s. These are "
                                      "discoverable tools whose REAL names carry a numeric suffix - do NOT unlock "
                                      "the base name as-is; first find each tool's full suffixed name in the "
                                      "knowledge base (search for the base name), then %s "
                                      "with that full name and call it via %s. Read the "
                                      "ACTUAL values from the records, then call this tool again."
                                      % (", ".join(_missing_r), _unl_r, _cal_r))
                        ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                                   requestor=getattr(tc, "requestor", "assistant"),
                                                   error=True, content=_msg_r)
                        print("[T2_SG_REQREADS] %s denied: missing reads %s"
                              % (getattr(tc, "name"), _missing_r), file=_sys.stderr, flush=True)
                        continue
                # ★LLM이 formalize한 clean operand(각 인자)를 ctx로([[10]]). 엔진은 op 실행만·원시파싱 안함.
                _args = getattr(tc, "arguments", None) or {}
                # ★W-f(리뷰 필수3·DAY5_PRESCRIPTIONS §P4): abstain-지목 직후 같은 도구 재호출의
                #   비용 계측 — P4 지시가 P6 없이는 재타이핑 재호출(=C208① 재직렬화·CWE 경로)을
                #   유도하는지의 실측이 **P6 ON의 GO 판정 신호**. 계측만(거동 무변).
                if getattr(self, "_t2_abstain_last", None) == d.get("name"):
                    self._t2_abstain_last = None
                    try:
                        _wf_n = len(json.dumps(_args, ensure_ascii=False, default=str))
                    except Exception:
                        _wf_n = -1
                    print("[T2_ABSTAIN_FIELDS] refetch-recall %s args=%dch"
                          % (d.get("name"), _wf_n), file=_sys.stderr, flush=True)
                _ctx = {}
                for _k, _v in (_args.items() if isinstance(_args, dict) else []):
                    if isinstance(_v, str):
                        try:
                            _v = json.loads(_v)
                        except Exception:
                            pass
                    _ctx[_k] = _v
                # ★P6(C208①·DAY5_PRESCRIPTIONS §P6·T2_SG_BYREF=1·기본 OFF): 커밋-출력 참조 해석 —
                #   "@last:<도구명>" 값을 **모델이 이미 읽어 커밋한** 그 도구의 최신 비에러 출력으로
                #   해석해 재타이핑을 제거(fetch는 여전히 모델 수행=autofetch 아님·E-PLAN C101 선례).
                #   파서는 env 기계 포맷("Record ID:") **전용**(assert·[[03b]] 경계 확장 방지).
                if os.environ.get("T2_SG_BYREF") == "1":
                    try:
                        _byref_resolve(self, d, _ctx)
                        _byref_join(self, d, _ctx)      # F7b: A2-선언 equijoin(유일 매칭만)
                    except _ByrefError as _bre:
                        ours[id(tc)] = ToolMessage(
                            id=tc.id, role="tool",
                            requestor=getattr(tc, "requestor", "assistant"), error=True,
                            content="Error: [BYREF] %s" % _bre)
                        print("[T2_SG_BYREF] %s: %s" % (d.get("name"), _bre),
                              file=_sys.stderr, flush=True)
                        continue
                    except Exception as _bre2:
                        print("[T2_SG_BYREF] skipped (no-op): %r" % (_bre2,),
                              file=_sys.stderr, flush=True)
                # ★C197: 목록형 op(over 선언)의 인자가 json.loads 실패로 str 잔류 = **침묵 3중 통과**
                #   (019 실측: python-repr+leading-zero 인자 → isolate 무언 skip → select_discrepant
                #   stats 前 [] → C195 coverage 우회 → "(none)"이 빈 결과로 위장). 엔진이 대신 파싱하면
                #   엔진-formalize=[[03b]] 위반 — 재송신을 **요구**한다(formalize=LLM 몫 유지·리터럴 0).
                # ★2026-08-03 §4-1: 중첩 op 트리의 over 파라미터도 검사(구판=최상위만 → rebate류
                #   str 잔류가 무검출로 통과했다).
                _ov = next((o for o in _over_params(d.get("op"))
                            if isinstance(_ctx.get(o), str)), None)
                if _ov:
                    ours[id(tc)] = ToolMessage(
                        id=tc.id, role="tool",
                        requestor=getattr(tc, "requestor", "assistant"), error=True,
                        content=("Error: [ARGS-FORMAT] the '%s' argument could not be read as a "
                                 "JSON array — it arrived as a plain string that is not valid "
                                 "JSON. Re-issue this exact call with '%s' as a VALID JSON array: "
                                 "use double quotes for all keys and string values, plain numbers "
                                 "without leading zeros or unit words, and no Python-style "
                                 "quoting. Copy the raw field values exactly as they appear in "
                                 "the records." % (_ov, _ov)))
                    print("[T2_SG_ARGS] %s: '%s' 인자 str 잔류(JSON 파싱실패) → 재송신 요구"
                          % (getattr(tc, "name"), _ov), file=_sys.stderr, flush=True)
                    continue
                # ★C204/D7(2026-07-27·day3 022/003 실측): **동일-인자 계산도구 반복 차단**(T2_SG_DEDUP=1).
                #   022=같은 인자로 rate 도구 10회(매회 2,127자 인자 에코)→context_window_exceeded ·
                #   003=fit 도구 5회. 우리 스캐폴드 op는 결정론이라 같은 인자=같은 결과가 **보장**되므로
                #   재실행 대신 결정론 안내를 반환한다(READ_DEDUP의 계산도구 판·C194 에스컬 동형).
                #   제외(정합성): ①op가 `evidence_from` 선언(원장-상태 의존: verify_identity ledger형 —
                #   같은 인자여도 fetch 후 결과가 달라진다·005 실측) ②isolate mode=fetch_formalize(env
                #   DB를 서브가 읽음=가변). 기본 OFF=거동 변화 0.
                _pend_key = None
                if (os.environ.get("T2_SG_DEDUP") == "1"
                        and not (d.get("op") or {}).get("evidence_from")
                        and (_isolate_spec(d) or {}).get("mode") != "fetch_formalize"):
                    try:
                        _dk = (d.get("name"), json.dumps(_ctx, sort_keys=True, default=str))
                    except Exception:
                        _dk = None
                    if _dk is not None:
                        _seen = getattr(self, "_t2_sg_seen", None)
                        if _seen is None:
                            _seen = self._t2_sg_seen = {}
                        _n = _seen.get(_dk, 0)
                        if _n:
                            _seen[_dk] = _n + 1
                            # ★P8(C208①·DAY5_PRESCRIPTIONS §P8·T2_DUP_REPRESENT=1): 스텁이 이전
                            #   결과를 재제시하지 않으면 "earlier output 참조" 지시가 재호출 유인을
                            #   못 끊는다(020 실측: 동일 인자 5회=창 29%). 자기 출력 캐시 재게시만
                            #   (우리 주입 도구=env 밖=replay 무관)·상한 2회·**천장 근접(P1 shrink
                            #   발생) 시 생략**(W-d: 작은 창에서 재제시=역효과).
                            _prev = (getattr(self, "_t2_sg_out", None) or {}).get(_dk)
                            _shrunk = getattr(getattr(self, "agent", None),
                                              "_t2_dyn_shrunk", False)
                            _stub, _did_rep = _dup_stub_content(
                                _n, prev=_prev,
                                represent_on=(os.environ.get("T2_DUP_REPRESENT") == "1"),
                                shrunk=_shrunk)
                            ours[id(tc)] = ToolMessage(
                                id=tc.id, role="tool",
                                requestor=getattr(tc, "requestor", "assistant"), error=True,
                                content=_stub)
                            print("[T2_SG_DEDUP] %s repeat#%d — stub%s"
                                  % (d.get("name"), _n + 1,
                                     " (+prev result)" if _did_rep else ""),
                                  file=_sys.stderr, flush=True)
                            continue
                        _seen[_dk] = 1
                        _pend_key = _dk
                # ★원장-결합 op는 인자 밖 증거가 필요하다 — **op가 `evidence_from`을 선언할 때만** 주입
                #   (도메인일반 조건·미선언 op는 거동 변화 0).
                if (d.get("op") or {}).get("evidence_from"):
                    _ctx.update(_evidence_ctx(self))
                # ★격리 서브가 operand 산출 (T2_SG_ISOLATE=1·기본 OFF·A2 `isolate` 선언 시만)
                #   `RATE_SUBAGENT_DESIGN §2b` LOCK. 실패=None → 메인 인자로 폴백(거동 변화 0).
                _iso = _isolate_spec(d) if os.environ.get("T2_SG_ISOLATE") == "1" else None
                if os.environ.get("T2_SG_TRACE") == "1":
                    # ★계측(2026-07-21·r095e 침묵 진단): isolate 디스패치 진입을 무조건 가시화 —
                    #   침묵-스킵이 불가능하도록. 계측 전용(기본 OFF·거동 무변).
                    print("[T2_SG_TRACE] %s: iso=%s mode=%s ctx=%s" % (
                        getattr(tc, "name", ""), bool(_iso), (_iso or {}).get("mode"),
                        sorted(list(_ctx))[:8]), file=_sys.stderr, flush=True)
                if _iso:
                    def _run(tcs, _self=self):
                        # ★dedup 우회(2026-07-20·smoke023c 포렌식): main이 이미 읽은 (name,args)를 서브가
                        #   다시 부르면 READ_DEDUP이 "위 출력 참조" stub을 주는데 **서브 문맥엔 '위'가 없다**
                        #   → 서브가 빈손 날조(실측: 60건 대신 3건 날조→오판). 서브 env 호출은 신선 실행.
                        _self._t2_dedup_bypass = True
                        try:
                            return orig_exec(_self, tcs)
                        finally:
                            _self._t2_dedup_bypass = False
                    if _iso.get("mode") == "fetch_formalize":
                        # ★fetch-first(2026-07-20 isolate-승격): 서브가 참조로 레코드를 off-ledger fetch
                        #   → 전체 operand dict를 top-level 주입. 에이전트는 참조만 넘겨 레코드 read 0(turn-free).
                        _sub = _sub_fetch_formalize(self, d, _iso, _ctx, _run)
                        if isinstance(_sub, dict) and _sub:
                            _ctx.update(_sub)          # top-level operand(서브가 fetch+formalize)
                            print("[T2_SG_ISOLATE] %s: fetch-formalize operand 주입 keys=%s"
                                  % (getattr(tc, "name"), list(_sub)), file=_sys.stderr, flush=True)
                    else:
                        _sub = _sub_formalize(self, d, _iso, _ctx, _run)
                        if _sub:
                            _rows = _ctx.get(_iso["over"]) or []
                            _hit = 0
                            for _r in _rows:
                                _v = _sub.get(str(_r.get(_iso["id_field"]))) if isinstance(_r, dict) else None
                                if isinstance(_v, dict):
                                    _r.update(_v)      # 서브 operand가 메인 추측을 대체
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
                # ★P4b(C208④·DAY5_PRESCRIPTIONS §P4b·T2_PROD_BIND=1): 비-레코드-유래 operand의
                #   producer-binding — A2 `grounded_params` 선언 필드의 행 값이 **선언된 producer
                #   출력**(내용 selector로 식별) 안에 부분문자열로 실재하지 않으면 결핍(None)으로
                #   강등해 P4 지목 경로에 합류. day5 027 [S]: accounts 미조회인 채 account_open을
                #   전행 단일값(02/01/2025)으로 날조(실개설일 02/13)→우연히 gold 재현 — 날조와
                #   정직 생략을 같은 abstain+지시로 수렴시킨다. 검사=부분문자열 실재(C186 ledger
                #   동형)·selector=A2 데이터·엔진 판단 0.
                _gp = ((d.get("grounded_params") or {})
                       if os.environ.get("T2_PROD_BIND") == "1" else {})
                if _gp:
                    _ev2 = _evidence_ctx(self)
                    _outs2 = _ev2.get("__tool_outputs") or {}
                    _overk = _primary_over(d, _ctx)   # ★2026-08-03 §4-1: 중첩 op 트리 대응
                    _rows2 = _ctx.get(_overk) if _overk else None
                    _dem = {}
                    for _fld, _gs in _gp.items():
                        _sels = [str(s).lower()
                                 for s in ((_gs or {}).get("producer_contains") or [])]
                        _cands = [t for t in _outs2.values()
                                  if any(s in t for s in _sels)]
                        for _r2 in (_rows2 or []):
                            if not isinstance(_r2, dict):
                                continue
                            _v2 = _r2.get(_fld)
                            if _v2 in (None, ""):
                                continue
                            if not any(str(_v2).lower() in t for t in _cands):
                                _r2[_fld] = None
                                _dem[_fld] = _dem.get(_fld, 0) + 1
                    if _dem:
                        print("[T2_PROD_BIND] %s: ungrounded field value(s) demoted to "
                              "missing: %s" % (d.get("name"), _dem),
                              file=_sys.stderr, flush=True)
                _res = _c.apply_op(d.get("op"), _ctx)
                if isinstance(_res, list):                    # 목록형(discrepancy ids)
                    _res = [str(i) for i in _res if i]
                    # ★P8 (2026-08-03·AX32 설계서 §P8·T2_DISPATCH_LEDGER=1 ∧ A2 `dispatch_targets`):
                    #   다건-write 지시의 **대상 집합을 원장에 등재**한다. 020/027 실측: 대상-반환
                    #   도구가 낸 집합 중 일부만 제출하고 종료해도 아무도 못 본다([coverage]는
                    #   검증-감사지 제출-완결이 아님). 집합은 **엔진이 이미 계산한 ids**라 리터럴 0·
                    #   판단 0(등재만) — 대조/표면화는 터미널 훅(t2_eplan_patch)이 한다.
                    if os.environ.get("T2_DISPATCH_LEDGER") == "1" and d.get("dispatch_targets"):
                        _dl = getattr(self, "_t2_dispatch_ledger", None)
                        if _dl is None:
                            _dl = self._t2_dispatch_ledger = {}
                        _dl[d["name"]] = sorted(set(_dl.get(d["name"]) or []) | set(_res))
                        print("[T2_DISPATCH_LEDGER] %s: %d target(s) registered"
                              % (d["name"], len(_dl[d["name"]])), file=_sys.stderr, flush=True)
                    # ★{details}: op가 남긴 상세(_sg_details)를 A2 detail_item_template로 포맷.
                    #   A2 template이 {details}를 안 쓰면 거동 변화 0(여분 kwarg는 무해).
                    _dets = _ctx.get("_sg_details") or []
                    _item_t = d.get("detail_item_template", "{id}")
                    try:
                        _details = "; ".join(_item_t.format(**it) for it in _dets) if _dets else "(none)"
                    except Exception:
                        _details = ", ".join(_res) if _res else "(none)"
                    # ★D4 모순 제거(INSTRUCTION_DEFECT §2a·T2_RETURN_EMPTY=1): 판정 상세가
                    #   **공집합**인데 "표시된 정확한 값으로 갱신하라"를 함께 말하는 모순(실측 20회/14 sim,
                    #   그중 7회는 coverage 결손 동반). 술어 = `_sg_details` 공집합(닫힘·판단 0).
                    #   ★empty 문구는 **완결을 주장하지 않는다** — coverage 표면화에 위임(허위 신설 방지).
                    _tpl_key = "return_template"
                    if (os.environ.get("T2_RETURN_EMPTY") == "1" and not _dets
                            and d.get("return_template_empty")):
                        _tpl_key = "return_template_empty"
                    _txt = d.get(_tpl_key, "{ids}").format(
                        ids=", ".join(_res) if _res else "(none)", details=_details)
                    _n = len(_res)
                    # ★C195: 판정 커버리지 병기(op가 _sg_stats를 남긴 경우만·거동보존).
                    #   "(none)"의 침묵-신뢰 차단: 몇 행을 판정했고 몇 행이 판정불가였는지 +
                    #   빈 결과일 때 재확인 지시(도메인 어휘 0). 근거=야간 020/027/029 거짓 "(none)".
                    _st = _ctx.get("_sg_stats")
                    if isinstance(_st, dict):
                        _txt += ("\n[coverage] %d of %d rows were checked (%d could not be "
                                 "verified)." % (_st.get("judged", 0), _st.get("total", 0),
                                                 _st.get("skipped", 0)))
                        # ★P4(C208④·DAY5_PRESCRIPTIONS §P4·T2_ABSTAIN_FIELDS=1): abstain의
                        #   actionable화 — 어느 입력 필드가 결핍이라 판정불가였는지 지목+공급 지시.
                        #   day5 020/026 [S]: account_open 누락→14행 전멸인데 결핍 필드 미지목이라
                        #   자기-수복 불가(정직 생략이 날조(027)보다 낮은 점수). 필드명=A2 params
                        #   키(도메인 데이터)·문구 도메인-일반·판단 0(엔진 자기 집계 표면화).
                        _mf = (_st.get("missing_fields") or {}
                               if os.environ.get("T2_ABSTAIN_FIELDS") == "1" else {})
                        if _mf and _st.get("skipped", 0):
                            # ★C278 §2c(R3 버그픽스·플래그 독립): 결핍 필드를 출처별로 갈라 문구를
                            #   낸다 — record-유래만 "call again"(이행 가능)·sub-유래는 unverified 정직
                            #   표기(C275 ⑤정정: base_rate를 "레코드서 읽어 재호출하라"는 모순 지시였다).
                            #   isolate 미선언 도구는 분리 근거가 없으므로 기존 문구 유지(안전).
                            _iso3 = _isolate_spec(d)
                            if _iso3:
                                _rec, _subm = _split_missing_fields(_mf, _iso3)
                            else:
                                _rec, _subm = _mf, {}
                            if _rec:
                                _mtxt = ", ".join("'%s' (%d rows)" % (k, v)
                                                  for k, v in sorted(_rec.items(), key=lambda x: -x[1]))
                                _txt += (" The unverified rows are missing input field(s): %s. "
                                         "Read the missing value(s) from the records that contain "
                                         "them, then call again with the completed input for those "
                                         "rows." % _mtxt)
                                self._t2_abstain_last = d.get("name")   # W-f: 재호출 성장 계측 anchor
                            if _subm:
                                _stxt = ", ".join("'%s' (%d rows)" % (k, v)
                                                  for k, v in sorted(_subm.items(), key=lambda x: -x[1]))
                                _unv = (((_iso3 or {}).get("quote_pin") or {}).get("unverified_note")
                                        or "could not be determined from the source documents; "
                                           "those rows remain UNVERIFIED — do not supply these "
                                           "values yourself.")
                                _txt += " Field(s) %s %s" % (_stxt, _unv)
                        # ★C278: quote-pin 사유/마크 표면화(숨은 실패→감사 가능한 주장·§5-5)
                        _qpn = getattr(self, "_t2_qp_notes", None)
                        if _qpn:
                            for _l in _qpn:
                                _txt += "\n[quote-pin] %s" % _l
                            self._t2_qp_notes = []
                        if not _res and _st.get("judged", 0) > 0:
                            _txt += (" An empty result means the checked rows matched the rates "
                                     "that were looked up for them — it is only as reliable as "
                                     "those rates. If the customer insists specific items look "
                                     "wrong, re-read those items' rate lines in the policy "
                                     "documents (per-category rates are a common source of "
                                     "error) instead of repeating this call.")
                        elif not _res:
                            # ★C197: 0행 판정 = 빈 결과가 아니라 **무판정** — 신뢰 금지·입력 재점검 지시.
                            _txt += (" NOTHING was actually checked (0 rows were readable) — this "
                                     "is NOT a clean empty result. Do not rely on it: re-read the "
                                     "records and re-issue the call with every row's raw values "
                                     "copied as valid JSON.")
                else:                                         # 스칼라형(verdict 등)
                    _txt = _render_scalar(d, _ctx, _res)      # 순수함수(관문3·단위테스트 공유)
                    _n = _res
                    # ★2026-08-03 §4-2: 미측정 윈도 abstain의 **표면화**(t2_compute가 남긴 사실만).
                    #   abstain 자체는 op이 이미 했다(None) — 여기서는 "무엇이 비었나"를 붙여
                    #   에이전트가 자기-수복(전체 거래 재참조 or user_id 위임)하게 한다.
                    #   문구 중 도메인 결론부는 A2 `incomplete_hint`(엔진 리터럴 0·[[05]]).
                    _txt += _window_coverage_note(d, _ctx, _res)
                # ★grounding 플래그를 반환문 맨 앞에 붙인다 — 드롭된 미검증 operand를 에이전트가 보고
                #   레코드를 다시 읽게(가짜 정밀도 신뢰 차단·§2ab). 플래그 없으면 거동 변화 0.
                if _gflags:
                    # ★D3 헤더-상세 모순 제거(INSTRUCTION_DEFECT §2d′·T2_GROUND_HDR=1):
                    #   공통 헤더가 전 필드에 *"레코드에서 다시 읽어라"*를 말하는데, `intent_fields`
                    #   (corpus=user)의 개별 주석은 *"손님이 말한 적 없다"*로 **정반대 출처**를 가리킨다.
                    #   실측(x35 ②): ledger 파라미터 회복 38:20 vs user 파라미터 회복 7:43.
                    #   ⇒ 헤더에서 **지시문을 뺀다**(각 flag의 괄호 주석이 이미 클래스별로 정확하다).
                    #   엔진 분기 순증 0 · A2 순증 0.
                    _hdr_tail = ("" if os.environ.get("T2_GROUND_HDR") == "1"
                                 else " Re-read the exact value(s) from the records before "
                                      "relying on this result.")
                    _txt = ("[GROUNDING WARNING] %d input value(s) could not be verified against the "
                            "account records / knowledge base and were dropped: %s.%s\n%s"
                            % (len(_gflags), "; ".join(_gflags), _hdr_tail, _txt))
                # requestor는 tau2 원본과 동형으로 **미러링**(environment.get_response: requestor=message.requestor).
                # ★P8: 결정론 결과 캐시(같은 인자 재호출의 재제시용·우리 도구=replay 무관).
                if _pend_key is not None:
                    _oc = getattr(self, "_t2_sg_out", None)
                    if _oc is None:
                        _oc = self._t2_sg_out = {}
                    _oc[_pend_key] = _txt
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
                        _self._t2_dedup_bypass = True     # 서브 env 호출=dedup 우회(위 _run과 동형·§2al)
                        try:
                            return orig_exec(_self, tcs)
                        finally:
                            _self._t2_dedup_bypass = False
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
                _msg = _truth_text(getattr(tc, "name", "") or "", _tn)   # 정본 공유(replay와 동일 텍스트)
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"), content=_msg)
                print("[T2_SG_TRUTH] '%s(%s)' -> interface fact (env would have denied our tool)"
                      % (getattr(tc, "name", "") or "", _tn), file=_sys.stderr, flush=True)
            elif (os.environ.get("T2_SG_TRUTH") == "1"
                  and getattr(tc, "name", None) in ((a2 or {}).get("unavailable_tools") or {})):
                # ★unavailable-tool 사실 정정 (2026-07-20 §2ax·r095b t0 실측): 백엔드 미설정으로 **항상
                #   실패하는 도구**(KB_search_dense="Missing credentials")를 env가 목록에 노출 — 에이전트가
                #   연속 선택·낭비 후 조기 transfer(C108류 거짓 인터페이스). A2가 선언한 도구는 실행 없이
                #   대체-경로 안내를 답한다(SG_TRUTH 동류·A2-구동·리터럴 0). read 도구=replay 비교 제외로 안전.
                _msg = a2["unavailable_tools"][getattr(tc, "name")]
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"),
                                           error=True, content=_msg)
                print("[T2_SG_TRUTH] unavailable-tool fact: %s" % (getattr(tc, "name", "") or ""),
                      file=_sys.stderr, flush=True)
            elif (os.environ.get("T2_TOOLGATE") == "1"
                  and getattr(self, "_t2_known_tools", None)
                  and getattr(tc, "name", None) not in self._t2_known_tools
                  # ★e2e9 052 크래시(§2ao): env에 **실재**하는 이름(discoverable 접미사-직호출 등)은
                  #   가로채지 않는다 — env가 허용·실행하는 호출을 우리가 "not available"로 답하면
                  #   ①over-block ②replay(재실행=진짜 결과)와 내용 불일치→sim 무효. env-실재=통과
                  #   (기록=env 결과=replay 동일·정합). TOOLGATE는 **진짜 발명된 이름**만 ASK.
                  and not (hasattr(getattr(self, "environment", None), "_has_tool")
                           and self.environment._has_tool(getattr(tc, "name", "") or ""))):
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

    # (5) ★SG_TRUTH replay-safety (2026-07-20·§2aj·023 smoke023 크래시 근본): eval replay
    #   (`environment.set_state`)는 **mutating 도구를 재실행해 내용 비교**한다. exec2의 SG_TRUTH는
    #   라이브만 가로채므로 replay서 env가 원래 거짓("Unknown agent tool")을 내 기록과 불일치→
    #   ValueError→sim 무효. 코드베이스 불변식("응답 바꾸는 개입=히스토리서 strip이거나 env-동일")을
    #   충족시키는 정합 픽스 = **env 클래스 레벨**서도 같은 진실 텍스트를 응답(라이브·replay 공히).
    #   상태 무변경 분기(env는 그 호출에 원래 에러·state 변화 0)라 replay 상태분기 없음. 기본 OFF 동일.
    from tau2.environment.environment import Environment as _Env
    if not getattr(_Env, "_t2_sg_truth_wrapped", False):
        _orig_get_response = _Env.get_response

        def _get_response2(self, message):
            if os.environ.get("T2_SG_TRUTH") == "1":
                _a2e = _g._domain_a2(getattr(self, "domain_name", None))
                if _a2e:
                    _dn = {d.get("name") for d in (_a2e.get("scaffold_get_tools") or [])}
                    for _x in {x.strip() for x in (os.environ.get("T2_SG_EXCLUDE") or "").split(",")
                               if x.strip()}:
                        _dn.discard(_x)
                    _tn = _a2_named_in_args(message, _dn)
                    if _tn:
                        from tau2.data_model.message import ToolMessage as _TM2
                        return _TM2(id=getattr(message, "id", None), role="tool",
                                    requestor=getattr(message, "requestor", "assistant"),
                                    content=_truth_text(getattr(message, "name", "") or "", _tn))
            return _orig_get_response(self, message)

        _Env.get_response = _get_response2
        _Env._t2_sg_truth_wrapped = True
    print("[T2_SCAFFOLD_GET] ON", file=_sys.stderr, flush=True)
    return True
