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


def _json_array(text):
    """텍스트 안에서 **문자열 JSON 배열** 하나를 집는다(펜스·산문 혼재 대응·`_merge_json` 동형).
    클래스-선택 서브(`_docs_delivery`)의 답 파싱 전용 — 값 해석은 없다(소속 검산은 호출부·[[59]])."""
    text = text or ""
    i = 0
    while i < len(text):
        if text[i] != "[":
            i += 1
            continue
        for j in range(len(text), i, -1):
            try:
                v = json.loads(text[i:j])
            except Exception:
                continue
            if isinstance(v, list):
                return [str(x) for x in v if isinstance(x, str)]
            i = j
            break
        else:
            i += 1
    return []


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


_OWN_WARN_TAG = "[grounding warning]"      # 우리 층이 찍는 태그(빌드부 `:2382`)·닫힌 술어


def _strip_own_feedback(text):
    """★C203 기지 결함 수리 (2026-08-21·t7335 094 실측·P2). `[GROUNDING WARNING] …` 경고문이
    도구 출력으로 원장에 에코되어, 1차에 **드롭된 바로 그 값**(094: actual_apy=5.25·period_start)이
    경고 문자열 안에 '실재'하게 되고 같은 값의 2차 재전송이 존재 검사를 통과했다(에코-그라운딩 —
    1차 저지의 자기-무력화). ⇒ grounding 대조 코퍼스에 한해 **우리 층이 찍은 경고 문면**을 지운다.
    - 식별 = 닫힌 술어: 우리가 찍는 태그부터 그 줄 끝까지(드롭 플래그 목록 `param=value; …`·
      `_hdr_tail`·quote_hint 전부 이 한 줄에 실린다 — 개행 없음·빌드부 축자). 도메인 텍스트에서
      아무것도 **뽑지 않는다** — 우리 포맷의 제자리 삭제뿐([[59]] 무관·`Record ID:` membership 동형).
    - 두 안(경고문에 값 비축자 vs 코퍼스서 문면 제외) 중 **후자**를 고른 근거: 경고 문면은 라이브
      회복 채널의 실측(x35② ledger 파라미터 회복 38:20)이 걸린 모델-거동 채널이라 문면 변경은
      재측정 없이 회귀 위험이 있고, 코퍼스 제외는 모델이 보는 문면 불변 + 검사 우주만 정화 =
      1차 발화 시점과 동일 거동의 복원이다. 경고 뒷줄의 도구 산출 본문은 그대로 남는다(회귀 0).
    - 재현 단위검정: `test_ground_warning_echo.py`."""
    s = str(text or "")
    low = s.lower()
    out, i = [], 0
    while True:
        j = low.find(_OWN_WARN_TAG, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = low.find("\n", j)                # 태그~줄끝 제거(개행 자체는 보존)
        if k < 0:
            break
        i = k
    return "".join(out)


def _corpus_texts(orch, which):
    """grounding 대조 코퍼스 (A2 선언 `corpus`: 'kb'|'ledger'). 도메인 리터럴 0·기존 소스 재사용.
    kb=도메인 KB 문서 전량 · ledger=지금까지 원장(에이전트 도구 출력)+사용자 발화. 엔진은 텍스트만 본다."""
    texts = []
    if "kb" in which:
        domain = getattr(getattr(orch, "environment", None), "domain_name", None)
        texts += [d0.get("content") or "" for d0 in (_load_domain_docs(domain) if domain else [])]
    if "ledger" in which:
        ev = _evidence_ctx(orch)
        # ★P2(2026-08-21): 우리 경고 문면을 지운 뒤 대조(`_strip_own_feedback` 주석·C203 수리).
        texts += [_strip_own_feedback(t) for t in (ev.get("__tool_outputs") or {}).values()]
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


def _tool_backend_live(orch, name):
    """★2026-08-03 (alltools 전환 실측 교정): A2 `unavailable_tools`는 **그 백엔드가 없던 검색 설정**
    (bm25 전용)에서 저작됐다 — alltools에서는 `KB_search_dense`가 **실제로 작동한다**(2026-08-03 env
    프로브: 임베딩 698문서 계산·`Score: 0.2119` 반환). 그 상태에서 우리가 "사용 불가"로 가로채면
    ⑴손님에게 갈 사실이 **거짓**이 되고 ⑵작동하는 회수 채널을 우리가 막는다(레버의 자기-역효과·
    등대 §1.3). ⇒ **env가 실제로 그 도구를 살려두었는지**를 기계적으로 확인하고, 살아 있으면
    A2 선언을 적용하지 않는다. 판정 = **env 도구 등록 여부**(기계 사실·닫힘) — 이 검색 도구를
    등록하는 변이는 그 백엔드(임베더)를 함께 선언하므로 등록 = 구성됨이다(변이 정의 직독).
    판정 불가(env 부재/예외) = False = **기존 거동 유지**(안전측)."""
    try:
        env = getattr(orch, "environment", None)
        if env is None:
            return False
        return name in {getattr(t, "name", None) for t in (env.get_tools() or [])}
    except Exception:
        return False


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


def _omitted_rows_note(sr):
    """★FIX-14/15 (2026-08-14 야간·074 실물): 격리 서브가 **원천에서 읽은 행 수**와 **넘긴 행 수**의
    차이를 반환문에 표면화한다(순수함수=단위테스트 공유·`_window_coverage_note` 선례).

    074 는 원장 33행 중 8행만 형식화됐는데 반환문은 `8 of 8 rows were checked (0 could not be
    verified)` 였다 — 분모가 **넘어온 것**이라 자기 자신을 재고, 25행의 손실은 stderr 로그
    (`⚠MISMATCH sub=8 · source=33`)에만 있었다. 여기서 하는 일은 뺄셈 하나와 그 사실의 전달뿐이다.

    ⚠하드 비율 가드는 두지 않는다: 073 은 원천의 **진짜 부분집합**을 넘기는 것이 정상이라
    비율로 막으면 통과를 죽인다(FIX-11 이 `source=0` 만 잡는 것과 같은 계보). 누락이 정당한지는
    모델이 판단한다 — 엔진은 어떤 행도 고르지 않고 두 수의 차만 말한다. 문구는 도메인-일반
    (리터럴 0·[[05]])이고 거부가 아니므로 **무엇을 하면 되는지**까지 담는다([[64]]).
    해당 없으면 빈 문자열(거동보존)."""
    if not isinstance(sr, dict):
        return ""
    # ⛔**무효화 (2026-08-14 야간·같은 날 출시분의 자기 반증)**: 분모가 틀렸다.
    #   `source` 는 서브 getter 출력의 `Record ID:` **전수**인데, 서브는 A2 지시에 따라
    #   **종류로 걸러** 산출한다(fee 도구: *"ONE element per atm_fee line"*). 072 실측 —
    #   계좌 A 레코드 32 중 atm_fee **8** → 서브 8행(정확) · 계좌 B 26 중 6 → 6행(정확).
    #   그런데 이 문구는 *"24 further row(s) … NOT supplied"* 를 매 호출마다 내보냈다.
    #   **누락이 없는데 누락을 주장**한 것이고, 우리 도구는 100% 정답 의무다([[25]]).
    #   ⚠아침에 이 함수를 쓰며 *"073 은 진짜 부분집합이라 하드 가드는 위험"* 이라고 적어 놓고도
    #     같은 함정에 빠졌다 — **부분집합이 정상인 도구에 전수 분모를 붙였다**.
    #   되살리려면 **서브가 산출해야 할 모집단**과 비교 가능한 분모가 필요하다(예: 서브가 자기
    #   후보 수를 함께 선언). 그 선언이 생기기 전까지는 **말하지 않는다** — 틀린 경보는
    #   침묵보다 나쁘다(모델이 있지도 않은 결손을 좇는다).
    return ""
    try:                                                   # noqa: 아래는 재개용 원형 보존
        miss = int(sr.get("source") or 0) - int(sr.get("sub") or 0)
    except (TypeError, ValueError):
        return ""
    if miss <= 0:
        return ""
    return (" %d further row(s) were present in the source records but were NOT supplied to this "
            "call, so they are not part of the count above. If those rows belong in this request, "
            "re-read them and call again with every row's values included." % miss)


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
                # ★P5(N97 §5): 인용이 반려됐을 때 **A2가 `quote_hint`를 켠 필드에 한해**, 그리고
                #   **모델이 제시한 값이 원장에 실재할 때만** 원장 표기를 지목한다. 값이 없으면
                #   지목 없이 종전대로 드롭 = 날조 차단 불변(C226 spoonfeed 회피). 지목 여부는
                #   A2 선언이 정하고 엔진은 기전만 갖는다([[05]]).
                _qh = ""
                if not src_ok and scf.get("quote_hint"):
                    try:
                        import t2_quote_hint as _QH
                        _qh = _QH.hint(ctx.get(param),
                                       _corpus_texts(orch, scf.get("corpus") or ["ledger"]))
                    except Exception:
                        _qh = ""
                flags.append("%s=%s (%s)%s" % (param, ctx.get(param),
                             "source quote not found in the records" if not src_ok
                             else "the value is not present in the source you quoted", _qh))
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
        # ★A-7⑹ (2026-08-23·094): 개수만 남기면 어느 operand 가 왔는지 알 수 없어 다음
        #   포렌식이 값을 **산술로 역산**하게 된다. 종류(키) 목록을 병기한다(값은 사이드카에).
        _kinds = sorted({str(_k) for _v in (got or {}).values()
                         if isinstance(_v, dict) for _k in _v})
        print("[T2_SG_ISOLATE] %s: %d라운드·getter %d회·operand %d/%d행 (kind=%s)"
              % (d.get("name"), rnd + 1, getter, len(got), len(ids),
                 ",".join(_kinds) or "-"), file=_sys.stderr, flush=True)
        # ★★계측: 서브 산출 operand 전수를 파일에 남긴다 — 라이브 서브는 메인 궤적 밖이라 여기 안 남기면
        #   over-flag가 서브 오독인지 검색부실인지 **영영 못 본다**(2026-07-18 디버깅공백·[[08]]).
        _isolate_trace(iso, d, {"round": rnd + 1, "getter": getter, "queries": queries,
                                "n_ids": len(ids), "n_operand": len(got), "operands": got})
        return got or None
    print("[T2_SG_ISOLATE] max_rounds 소진 → 격리 생략", file=_sys.stderr, flush=True)
    _isolate_trace(iso, d, {"error": "max_rounds", "queries": queries})
    return None


def _docs_delivery(orch, d, iso, ref, ag, la, UserMessage):
    """★A3 읽기-명세 전달 (T2_SG_DOCS=1 · `isolate.docs` 선언 시만 · 2026-08-21 C582 처방).

    서브에게 검색을 시키지 않는다: A3 `isolate.docs` 가 선언한 (문서 id, content-기준 범위,
    앵커 40자)를 엔진이 env 코퍼스에서 **잘라** 재료로 싣는다([[71]] — bm25 는 baseline).

    [[71]] 계약 4문 답:
      ①기능 하나 — 여기서 도는 서브는 **클래스 선택 하나**뿐(닫힌 목록에서 고르기). 형식화는
        호출부 서브가 한다 — 결정 하나당 서브 하나([[65]]).
      ②재료는 선언에서 — id·범위·앵커 전부 A3(감사 `x453` 검산본). 이 코드에 도메인 리터럴 0([[05]]).
      ③전달 = 선언된 id 정확 집기 — 정책 문서 읽기는 우리 층 몫(C405ⓔ·`t2_search` §경계).
      ④엔진 해석 0 — 소속 검산(집합)·자르기(산수)·앵커 일치(문자열 비교)만([[59]]·[[62]]).

    [[62]] 답: 결손은 격리로 쟀다 — `x456`(C582: 반환 6/6 ↔ 관문1 생존 4/17·실패 문면
      `base=0.0 source:""` = 남은 결손은 **재료 도달**) · `x448`(C578: 재료 도달 시 26/26).
      ⇒ 이 레버는 **전달(부하 축소)뿐**이다. 값·kind·적용성 판단은 전부 LLM 에 남는다.

    앵커 불일치 = 그 문서 **전량 폴백 + 로그** — 밀린 바이트를 배달하고 모델 탓을 하는 사고
    (핸드오프 §3⑵)의 재발 방지·침묵 금지([[55]]). 전체 실패 = None → 종전 검색 경로(거동보존).
    """
    import t2_search as TS
    try:
        orch._t2_docs_mat = None          # ★계측: 이 호출에서 docs 전달이 실제 발화했나([[55]] 죽은 배선 방지)
    except Exception:
        pass
    dd = iso.get("docs") or {}
    bc = dd.get("by_class") or {}
    if not bc:
        print("[T2_SG_DOCS] docs.by_class 미선언 → 검색 폴백", file=_sys.stderr, flush=True)
        return None
    corpus = TS.corpus_from_env(getattr(orch, "environment", None))
    if not corpus:
        print("[T2_SG_DOCS] env 코퍼스 0편 → 검색 폴백", file=_sys.stderr, flush=True)
        return None
    kw = {k: v for k, v in dict(getattr(ag, "llm_args", None) or {}).items() if "tool" not in k}
    if iso.get("temperature") is not None:
        kw["temperature"] = iso["temperature"]
    # ① 클래스 선택 — **별도 서브 하나**·닫힌 목록([[22]] 열린 술어=LLM·엔진은 소속만 검산).
    #   지시가 재료보다 앞이다(C578: 위치 하나가 26/26 ↔ 0/26 을 갈랐다).
    classes = sorted(bc)
    # ★v2 (2026-08-21 x456 C팔 1차 실측·`x456_kb_sub_liveness_cdocs.json`): v1 문구는 픽커가
    #   REFERENCE 의 **첫 항목(계좌 자신)의 클래스를 빠뜨리는** 누락을 낳았다(관문1 문면
    #   `base=0.0 source not found` 로 확인·gold 무관). 이름이 모호한 항목도 한쪽만 골랐다.
    #   항목 전수 + 모호=전부 포함으로 고침 — 과포함 비용은 바이트뿐(전달량 로그로 가시)이고
    #   누락 비용은 필수 성분 전멸이라 비대칭이다.
    pick = ("You are a closed-list selection sub-task. REFERENCE names an account and the "
            "customer's products. From CLASSES below, select EVERY class that corresponds to "
            "ANY item in REFERENCE - the account itself AND each product. If more than one "
            "class name plausibly matches an item, include ALL plausible classes rather than "
            "choosing one. Copy the names VERBATIM from CLASSES; do not invent or edit names. "
            "Reply with exactly one JSON array of strings and nothing else.\n\n"
            "=== REFERENCE ===\n%s\n\n=== CLASSES ===\n%s"
            % (json.dumps(ref, ensure_ascii=False, indent=1),
               json.dumps(classes, ensure_ascii=False, indent=1)))
    try:
        um = UserMessage(role="user", content=pick)
    except TypeError:
        um = UserMessage(content=pick)
    # ★이 서브는 정본 `t2_subcall.sub_generate` 를 **우회한다**(래칫 `test_subcall_canonical` 이
    #   그 부채를 이미 세고 있다). 배선을 바꾸면 거동이 움직일 수 있어 지금은 **기록만** 붙인다 —
    #   클래스 선택은 ②범주 축의 결정점이라 *무엇을 받았나* 가 남지 않으면 [[76]] 진단 ①이 막힌다.
    try:
        import t2_subcall as _sc0
    except Exception:
        _sc0 = None
    try:
        resp = la.generate(model=ag.llm, tools=None, messages=[um],
                           call_name="sg_docs_class", **kw)
    except Exception as e:
        print("[T2_SG_DOCS] 클래스-선택 서브 실패: %r → 검색 폴백" % (e,), file=_sys.stderr, flush=True)
        if _sc0 is not None:
            _sc0._record_subcall("sg_docs_class", pick, "", err=e)
        return None
    if _sc0 is not None:
        _sc0._record_subcall("sg_docs_class", pick,
                             getattr(resp, "content", None) or "")
    raw = _json_array(getattr(resp, "content", None) or "")
    picked = [c for c in raw if c in bc]
    alien = [c for c in raw if c not in bc]
    if alien:
        print("[T2_SG_DOCS] 목록 밖 이름 %s → 버림(소속 검산)" % (alien,), file=_sys.stderr, flush=True)
    if not picked:
        print("[T2_SG_DOCS] 선택 클래스 0 → 검색 폴백", file=_sys.stderr, flush=True)
        _isolate_trace(iso, d, {"mode": "docs", "picked": [], "alien": alien})
        return None
    # ② always(전량) + by_class[선택](선언 범위) 를 코퍼스에서 잘라 붙인다. 같은 문서 중복 전달 금지.
    parts, texts, missing, anchor_fb, seen = [], [], [], 0, set()

    def _add(did, body):
        seen.add(did)
        parts.append("### %s\n%s" % (did, body))
        texts.append(body)

    for did in (dd.get("always") or []):
        t = corpus.get(did)
        if t is None:
            missing.append(did)
            continue
        _add(did, t)
    for c in picked:
        for e in (bc.get(c) or []):
            did = e.get("doc")
            if did in seen:
                continue
            t = corpus.get(did)
            if t is None:
                missing.append(did)
                continue
            segs, ok = [], True
            for rg in (e.get("ranges") or []):
                o, ln = int(rg[0]), int(rg[1])
                if len(rg) > 2 and " ".join(t[o:o + 40].split()) != rg[2]:
                    ok = False                     # 선언 앵커 ≠ 자른 자리 → 밀린 조각을 배달하지 않는다
                    break
                segs.append(t[o:o + ln])
            if ok and segs:
                _add(did, "\n[...]\n".join(segs))
            else:
                anchor_fb += 1
                print("[T2_SG_DOCS] %s: 앵커 불일치/범위 0 → 문서 전량 폴백" % did,
                      file=_sys.stderr, flush=True)
                _add(did, t)
    if missing:
        print("[T2_SG_DOCS] 선언 id 가 코퍼스에 없음 %d건: %s" % (len(missing), missing[:5]),
              file=_sys.stderr, flush=True)
    if not texts:
        print("[T2_SG_DOCS] 전달 재료 0 → 검색 폴백", file=_sys.stderr, flush=True)
        return None
    mat = "\n\n".join(parts)
    print("[T2_SG_DOCS] %s: 클래스 %s · 문서 %d편 · %d자 전달(검색 0)"
          % (d.get("name"), picked, len(parts), len(mat)), file=_sys.stderr, flush=True)
    out = {"text": mat, "texts": texts, "picked": picked, "alien": alien,
           "n_docs": len(parts), "chars": len(mat), "anchor_fallback": anchor_fb,
           "missing": missing}
    try:
        orch._t2_docs_mat = {k: out[k] for k in ("picked", "alien", "n_docs", "chars",
                                                 "anchor_fallback", "missing")}
    except Exception:
        pass
    _isolate_trace(iso, d, {"mode": "docs", "picked": picked, "alien": alien,
                            "n_docs": len(parts), "chars": len(mat),
                            "anchor_fallback": anchor_fb, "missing": missing})
    return out


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
    # ★R3 (2026-08-22·x481 확정): 에이전트가 **전사한 요약** 대신 **도구 출력 원문**을 싣는다.
    #   왜: 이 함수의 설계 의도는 바로 위 docstring 에 축자로 있다 — *"에이전트는 참조(예
    #   account id)만 넘기고 서브가 레코드를 읽는다"*. 그런데 `customer_products` 같은 인자는
    #   참조가 아니라 **에이전트가 자기 말로 요약한 문자열**이고, 그 전사에서 이름이 바뀐다:
    #   레코드 `level: "Green Account"` → 에이전트 `"Green Checking Account"` → KB 에 없는 이름 →
    #   서브가 페어링을 못 찾아 boost 를 통째로 놓친다(093 실측 apy 4.025 ↔ 정답 4.275).
    #   x481 격리 실측(각 4회·라이브 표기 고정):
    #       에이전트 요약        checking 0/4 · 합 4.025
    #       요약 + 대응 지시     checking 4/4 · 합 4.275
    #       **레코드 원문**      checking 4/4 · 합 4.275  ← 지시 문장 없이 해결
    #   ⇒ 설득하는 문장을 붙이는 대신 **재료를 제대로 준다**([[62]] 결정론기는 최소한·
    #     [[71]] 전달은 엔진이·[[65]] 재료가 메인을 거치면 손상된다).
    #   ⚠엔진은 **이름 대조 + 그대로 싣기**만 한다 — 파싱·선택·요약 0([[59]]). 어느 계좌인지,
    #     무엇이 boost 인지는 여전히 서브가 문서를 읽고 판단한다.
    #   ⚠미선언이면 거동 변화 0 · 출력을 못 찾으면 에이전트 인자를 그대로 쓴다(fail-open).
    _rfo = iso.get("ref_from_outputs") or {}
    if _rfo:
        _raw = (_evidence_ctx(orch).get("__tool_outputs_raw") or {})
        for _k, _sel in _rfo.items():
            _needles = [str(x).lower() for x in ((_sel or {}).get("producer_contains") or [])]
            _hit = [v for n, v in _raw.items()
                    if any(_nd in str(n).lower() for _nd in _needles)]
            if _hit:
                ref[_k] = (chr(10) + chr(10)).join(_hit)
                print("[T2_SG_REFRAW] %s.%s ← 도구 출력 원문 %d편(%d자·에이전트 전사 대체)"
                      % (d.get("name"), _k, len(_hit), len(ref[_k])),
                      file=_sys.stderr, flush=True)
    if not ref:
        print("[T2_SG_ISOLATE] fetch: ref_params 부재 → 격리 생략", file=_sys.stderr, flush=True)
        return None
    keys = set(iso.get("operand_keys") or [])
    if not keys:
        print("[T2_SG_ISOLATE] fetch: operand_keys 미선언 → 격리 생략", file=_sys.stderr, flush=True)
        return None
    # ★A3 읽기-명세 전달 (T2_SG_DOCS=1·C582 처방·[[71]]): 엔진이 선언된 범위를 잘라 재료로
    #   싣고 getter 는 **노출하지 않는다**(서브=형식화만·검색 0). 실패 = 종전 검색 경로(로그 위에서).
    _mat = None
    if os.environ.get("T2_SG_DOCS") == "1" and isinstance(iso.get("docs"), dict):
        _mat = _docs_delivery(orch, d, iso, ref, ag, la, UserMessage)
    tools = []
    if _mat is None:
        tools = [t for t in (getattr(ag, "tools", None) or [])
                 if getattr(t, "name", None) in set(iso.get("getter_tools") or [])]
        if not tools:
            print("[T2_SG_ISOLATE] fetch: getter_tools 부재 → 격리 생략", file=_sys.stderr, flush=True)
            return None
    if _mat is not None:
        # ★지시(형식 포함)가 재료보다 **앞**이다 — C578: 위치 하나가 26/26 ↔ 0/26 을 갈랐다.
        #   지시문 = A3 `docs.instructions`(검색 문구 없는 판·미선언이면 기존 instructions).
        prompt = "%s\n\n%s\n\n=== REFERENCE ===\n%s\n\n=== DOCUMENTS ===\n%s" % (
            (iso["docs"].get("instructions") or iso["instructions"]), iso["answer_format"],
            json.dumps(ref, ensure_ascii=False, indent=1), _mat["text"])
    elif os.environ.get("T2_SG_PROMPT_V2") == "1":
        # ★V2 조립 (2026-08-25·x525 격리 실측·기본 OFF)
        #   074 전사 결손을 격리로 이등분해 **두 변수**를 찾았다(chk_2 · 계약 기대 16행 · 각 n=6):
        #     ⒜ `=== REFERENCE ===` 를 **JSON 블록**으로 주면 행이 **떨어진다**
        #        (JSON 13~15 · 키 이름 중립화 14 · 블록을 뒤로 15 · **문장으로 주면 16/16**)
        #        ⇒ 키도 위치도 아니고 **블록의 형식**이다. 빠지는 행은 매번 *수수료 줄 없는 인출*.
        #     ⒝ `answer_format` 이 **재료보다 앞**이면 유령 `duplicate_of` 가 **+3** 붙는다
        #        (앞 19행 · 뒤 **16행** · 둘 다 cover 16/16)
        #   ⇒ 이기는 순서 = `instructions + params + 재료 + answer_format`(J_both·K_paramslast 6/6).
        #   ⚠엔진이 새로 **쓰는 문장은 없다** — 선언(`instructions`·`params`·`answer_format`)을
        #     그대로 쓰고 **조립 순서와 REFERENCE 렌더링**만 바꾼다([[05]] 도메인 리터럴 0·[[78]]②).
        #   ⚠`answer_format` 은 마감 라운드에 **재료 뒤로** 붙인다(아래 `_v2_close`).
        _ref_lines = "\n".join("%s: %s" % (k, v) for k, v in ref.items())
        _pblock = ""
        for _k in sorted(keys):
            _pd = (d.get("params") or {}).get(_k)
            if isinstance(_pd, str) and _pd.strip():
                _pblock += "\n%s: %s" % (_k, _pd)
        prompt = "%s\n\n=== REFERENCE ===\n%s%s" % (
            iso["instructions"], _ref_lines,
            ("\n\n=== FIELD CONTRACT ===" + _pblock) if _pblock else "")
    else:
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
    # ★서브 getter 성공/실패 출력(§2be·0.0-주입 차단+비수렴 원인 관측). docs 모드에선 **엔진이
    #   배달한 재료**가 그 자리에 선다 — 안 그러면 "성공 출력 0" 폐기가 채널을 또 죽인다(C581 동형).
    _ok_outs = list(_mat["texts"]) if _mat else []
    _err_outs = []
    _v2_close_sent = False        # ★V2: `answer_format` 재배치는 서브 호출당 한 번만
    for rnd in range(_maxr):
        try:
            # ★마감 라운드(2026-07-21 §2ba·r095e/f 실측: 서브가 라운드 내내 getter만 돌고 답을 안 내
            #   소진→폴백): 마지막 라운드는 **도구 없이** 생성 — 구조적으로 tool-call 불가 → JSON 답 강제.
            _last = (rnd == _maxr - 1)
            _tl = None if (_last or not tools) else tools     # docs 모드=도구 0(형식화만·[[71]])
            # ★V2: `answer_format` 을 **재료 뒤**에 놓는다 (x525 실측: 앞이면 유령 중복행 +3).
            #   도구 결과가 이미 msgs 에 들어온 뒤에만 붙이고, sim·서브당 한 번만 붙인다.
            #   ⚠엔진이 쓰는 문장 0 — A2 `isolate.answer_format` 축자 그대로다.
            # ★★수리 (2026-08-25·t7352 라이브 실측). 구판 조건은 `_tl is None` = **마지막
            #   라운드**뿐이었다. 그런데 이 서브는 라운드 1(도구가 아직 있는 라운드)에 답한다
            #   — 라이브 축자: `fetch get_atm_fee_discrepancies: **2라운드**·getter 1회·
            #   operand keys=**[]**`. V2 는 `answer_format` 을 **머리에서 이미 뺐으므로**,
            #   그 시점의 서브는 형식 지시를 **한 번도 못 본 채** 답한다 ⇒ 파싱 실패 ⇒
            #   `got` 공집합 ⇒ 메인 인자 폴백 ⇒ `[T2_COMPUTE] 9/17행 판정불가(operand가
            #   숫자 아님)`. 같은 태스크 t7348(V2 off)은 `operand keys=['transactions']` 였다.
            #   ⇒ 조건을 **도구 결과가 들어온 뒤**로 옮긴다(`_ok_outs` 비어 있지 않음).
            #     격리가 이긴 순서(`재료 → answer_format`)는 그대로 지켜진다 — 레코드는 이미
            #     tool 메시지로 들어왔고 그 **뒤**에 형식이 붙는다. 서브가 어느 라운드에 답하든
            #     형식을 본다. 엔진이 쓰는 문장 0(선언 축자)·라운드 소비 0.
            #   ⚠[[24]] 이 자리는 **조립부에 마커가 없어서** 死배선이 로그로 안 보였다.
            #     아래 인쇄를 조립 조건 쪽으로 옮기지 마라 — 붙였다는 사실이 유일한 증거다.
            if (os.environ.get("T2_SG_PROMPT_V2") == "1" and not _v2_close_sent
                    and (_tl is None or _ok_outs)):
                # ★2026-08-25 정정 (x525 실측): 마감 user 메시지에 **필드 계약 + 회수된 원장 +
                #   형식**을 함께 싣는다. 형식만 붙인 판은 chk_2 에서 cover 15/16 이었고, 16/16 을
                #   내는 팔(J_both·K_paramslast)과의 유일한 차이가 *원장이 user 메시지 안이냐*였다.
                #   엔진이 쓰는 문장 0 — 선언(`params`·`answer_format`)과 **도구가 낸 원문**뿐이다.
                _pb2 = ""
                for _k2 in sorted(keys):
                    _pd2 = (d.get("params") or {}).get(_k2)
                    if isinstance(_pd2, str) and _pd2.strip():
                        _pb2 += "\n%s: %s" % (_k2, _pd2)
                _recs2 = "\n\n".join(str(x) for x in (_ok_outs or []) if x)
                # ★T2_SG_RECORD_ORDER (2026-08-25·기본 OFF·격리 x536/x539 후 배선·[[78]]).
                #   덤프의 **순서만** 바꾼다(내용·값·판단 0). 근거와 부정통제는 `_reorder_records`.
                #   ⚠OFF 에서도 관측 한 줄을 남긴다 — 그것이 반증 경로다([[25]] 死배선 조기 발견).
                if _recs2 and "Record ID:" in _recs2:
                    _ro = _reorder_records(_recs2)
                    _chg = (_ro != _recs2)
                    if os.environ.get("T2_SG_RECORD_ORDER") == "1":
                        _recs2 = _ro
                        print("[T2_SG_RECORD_ORDER] %s: 덤프 재배열 %s (%d자)"
                              % (d.get("name"), "적용" if _chg else "무변",
                                 len(_recs2)), file=_sys.stderr, flush=True)
                    else:
                        print("[T2_SG_RECORD_ORDER] 관측(OFF) %s: 재배열하면 %s"
                              % (d.get("name"), "달라진다" if _chg else "같다"),
                              file=_sys.stderr, flush=True)
                _close = ((("=== FIELD CONTRACT ===" + _pb2 + "\n\n") if _pb2 else "")
                          + (("=== RECORDS ===\n" + _recs2 + "\n\n") if _recs2 else "")
                          + iso["answer_format"])
                try:
                    _um2 = UserMessage(role="user", content=_close)
                except TypeError:
                    _um2 = UserMessage(content=_close)
                msgs = list(msgs) + [_um2]
                _v2_close_sent = True
                print("[T2_SG_PROMPT_V2] %s: answer_format 을 재료 뒤로(마감 라운드)"
                      % d.get("name"), file=_sys.stderr, flush=True)
            # ★T2_SG_SCHEMA (2026-08-22·기본 OFF): **도구가 없는 라운드에만** 문법을 건다.
            #   왜: 마감 라운드는 산문 지시(`answer_format`)로 JSON 을 부탁할 뿐이라 서브가 형식
            #   예시의 **값을 그대로 베껴** 왔다 — `{principal: 0.0, actual_apy: 0.0}`(t7337·t7338
            #   두 런 재현). 이 자리는 `:2be` 주석이 *"§2as 0.0-포이즈닝의 신형 재발"* 로 이미
            #   이름 붙인 곳이고, 그때 처방은 **답 폐기**(증상 억제)였다. 폐기 → 폴백 → 메인 추측 →
            #   grounding 드롭 → 도구 None → 모델이 값을 자기 계산해 write → WEV deny 의 livelock 이
            #   거기서 나온다. ⇒ 부탁 대신 **문법으로 형식을 보장**해 베낄 예시 자체를 없앤다.
            #   ⚠**도구가 있는 라운드엔 절대 걸지 않는다**: `tools`+`guided_json` = tool_calls 0
            #   (t2_declfirst §배선 실측·C248) — 걸면 서브가 레코드를 못 읽는다. 마감 라운드는
            #   구조적으로 `_tl is None` 이므로 그 제약이 성립한다(declfirst 2패스와 동형: 프롬프트만
            #   32% ↔ 도구미제공+문법 96%·C250).
            #   ⚠엔진은 형식만 강제한다 — 값은 여전히 서브가 낸다([[62]]·[[10]]). 스키마 출처는
            #   A2 `isolate.operand_schema` 하나뿐이고 엔진 리터럴 0([[05]]).
            #   ⚠마감-답 검증(`_ok_outs` 숫자-실재)은 그대로 남는다 — 문법은 형식만 보장하지
            #   값의 진실성은 보장하지 않는다(날조 차단 불변).
            _kw = kw
            if (_tl is None and os.environ.get("T2_SG_SCHEMA") == "1"
                    and iso.get("operand_schema")):
                _kw = dict(kw)
                _kw.pop("tools", None)
                _eb = dict(_kw.get("extra_body") or {})
                _eb["guided_json"] = iso["operand_schema"]
                _eb["guided_decoding_backend"] = "xgrammar"
                _kw["extra_body"] = _eb
                print("[T2_SG_SCHEMA] %s: 마감 라운드에 문법 적용(도구 0)"
                      % d.get("name"), file=_sys.stderr, flush=True)
            resp = la.generate(model=ag.llm, tools=_tl, messages=msgs,
                               call_name="sg_fetch_iso",
                               **(dict(_kw, tool_choice="required") if (rnd == 0 and _tl) else _kw))
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
                # ★레코드 덤프는 절단하지 않는다 (2026-08-06·023 실측).
                #   이 절단은 097에서 **KB 문서(산문)** 가 라운드마다 누적돼 서브가 컨텍스트를 넘긴
                #   사고 때문에 들어왔다. 그런데 fetch_formalize의 operand 소스는 **레코드 덤프**이고,
                #   서브는 그것을 전량 봐야 전사가 성립한다. 023 실측: 덤프 18,557자·60행(행당 ≈309자)이
                #   4,000자(head 2,600+tail 1,400)로 잘려 서브가 본 것은 앞뒤 십여 행뿐 —
                #   그 전사본이 operand로 주입돼 12개 창 중 9개가 "입력 레코드 0"이 됐고
                #   `check_rebate_qualification`이 기권했다(부검 §G).
                #   술어는 파서가 쓰는 바로 그 문자열이다(`Record ID:`) — 판단 0·도메인 리터럴 0.
                if isinstance(_c, str) and "Record ID:" in _c:
                    print("[T2_SG_ISOLATE] sub-view: record dump kept whole (%d chars)" % len(_c),
                          file=_sys.stderr, flush=True)
                    return _m
                if isinstance(_c, str) and len(_c) > _cap:
                    print("[T2_SG_ISOLATE] sub-view: truncated %d -> %d chars" % (len(_c), _cap),
                          file=_sys.stderr, flush=True)
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
                        "guess or infer values. %s"
                        % ("; ".join(_fl),
                           ("Re-read the DOCUMENTS section for the line or table that states "
                            "the exact number, then re-send the complete JSON answer." if _mat
                            else "Search again for the line or table that states the exact "
                            "number, then re-send the complete JSON answer.")))
                try:
                    msgs.append(UserMessage(role="user", content=_fbt))
                except TypeError:
                    msgs.append(UserMessage(content=_fbt))
                print("[T2_SG_ISOLATE] fetch %s: ground-피드백 %d건 → 서브 재시도(%d라운드)"
                      % (d.get("name"), len(_fl), rnd + 1), file=_sys.stderr, flush=True)
                continue
        getter = sum(1 for m in msgs if getattr(m, "role", "") == "tool")
        # ★계기 (2026-08-06·023): 배열 operand는 서브가 **손으로 옮겨 적은 것**이다. 그 전사가
        #   원본을 다 담았는지는 지금까지 어디에도 기록되지 않았고, 그래서 이 결함의 규모를
        #   말할 수 없었다(부검 §G-3). 서브 자신의 getter 출력에서 레코드 수를 세어 나란히 찍는다.
        #   판정하지 않는다 — **두 수를 남길 뿐**이다(도메인 리터럴 0·거동 변화 0).
        _src_rows = 0
        for _t in _ok_outs:
            _src_rows += _t.count("Record ID:")
        _sub_rows = 0
        for _k, _v in (got or {}).items():
            if isinstance(_v, list):
                _sub_rows = max(_sub_rows, len(_v))
                print("[T2_SG_ISOLATE] operand-size %s.%s: sub=%d rows · source=%d rows%s"
                      % (d.get("name"), _k, len(_v), _src_rows,
                         "  ⚠MISMATCH" if _src_rows and len(_v) != _src_rows else ""),
                      file=_sys.stderr, flush=True)
        # ★FIX-14/15 (2026-08-14 야간 074 실물·핸드오프 §5): 위 두 수는 **stderr 에만** 있었다.
        #   074 는 원장 33행 중 서브가 8행만 형식화했는데 도구 반환문은 `8 of 8 (0 unverified)` 라
        #   손실이 어디에도 보이지 않았다 — 분모가 **서브가 넘긴 것**이라 자기 자신을 잰다.
        #   여기서는 판정하지 않고 두 수를 반환 경로가 읽을 수 있는 자리에 남기기만 한다
        #   (표면화는 렌더가·거동 변화는 문구뿐·도메인 리터럴 0).
        try:
            _sr = getattr(orch, "_t2_sub_srcrows", None)
            if _sr is None:
                _sr = orch._t2_sub_srcrows = {}
            _sr[d.get("name")] = {"source": _src_rows, "sub": _sub_rows}
        except Exception:
            pass
        # ★날조 안전판 (2026-08-14·t7283 072 실물·[[25]] 우리 도구는 100% 정답 의무):
        #   서브의 getter 가 **레코드를 한 건도 못 읽었는데**(source=0) 배열 operand 가 나오면
        #   그 행들은 읽은 것이 아니라 **지어낸 것**이다 — 072 실측: `txn_123456`·`txn_789012`
        #   같은 자리표시자 id 3행이 산출돼 엔진이 그 위에서 금액을 계산하고 손님에게 갔다.
        #   판정은 닫혀 있다(읽은 레코드 수 = 0 ∧ 배열 비어있지 않음·내용 판단 0). 폐기하면
        #   메인 인자 폴백 = 종전 거동(안전측). 격리 측정 불요 — 데이터 무결성 수리다.
        #
        # ★★축 정정 (2026-08-21·전수 실측): 위 계수기의 술어 `Record ID:` 는 **DB 레코드 덤프
        #   포맷 전용**이다. getter 가 KB 검색인 선언은 출력에 그 문자열이 **원리상 0건**이라
        #   정직하게 읽어도 항상 `source=0` → **항상 폐기**였다. 영속 로그 전수:
        #     get_correct_savings_apy   (getter=KB_search_bm25)  52/52 = 100% source=0 · 폐기 47
        #     get_atm_fee_discrepancies (getter=DB 디스패처)      67/348 = 19%       · 폐기 62
        #   앞의 것은 자기 `ground.array_fields`(인용이 선언 corpus 에 축자 실재 ∧ 값이 그
        #   인용 안에 실재)로 **축이 맞는 날조 차단을 이미** 걸고 있고 실제로 작동한다(같은
        #   로그: 서브내 재검색 172회 · 관문1 드롭 64회 `base=0.0 (source not found in the
        #   knowledge base)` 등). 그런데 서브 답은 그 관문1 심사를 **받아 보기 전에** 이 줄에서
        #   버려졌다(t7326 halfB task_063: 3679 sub=5·source=0 → 3680 폐기 → 3681 폴백이
        #   ungrounded → 3682 `-> None`). 즉 KB 축 서브콜 채널이 **구조적으로 죽어 있었다**
        #   ([[55]] 우리 배관·[[67]] t7290 동형).
        #   ⇒ 선언이 그 param 에 배열 근거 계약을 걸어 두었고 **그 계약이 실제로 집행될
        #     때만**(T2_SG_GROUND=1 — 병합 직후 관문1 `_ground_operands` 가 돈다) 이 계수기는
        #     서지 않는다. 계약이 없거나 집행이 꺼져 있으면 종전 그대로 = fail-closed.
        #   판정은 여전히 닫혀 있고 **선언을 읽어** 정한다(도메인 리터럴 0·[[59]]).
        _contracted = set()
        if os.environ.get("T2_SG_GROUND") == "1":
            for _af in ((d.get("ground") or {}).get("array_fields") or []):
                if isinstance(_af, dict) and _af.get("param"):
                    _contracted.add(_af["param"])
        _unguarded = [_k for _k, _v in (got or {}).items()
                      if isinstance(_v, list) and _v and _k not in _contracted]
        _deferred = [_k for _k, _v in (got or {}).items()
                     if isinstance(_v, list) and _v and _k in _contracted]
        # ★★축 정정 2 (2026-08-21 단위검정이 잡음·C581 동형): docs 전달 모드(_mat)는 **getter 가
        #   없다** — 근거 원문은 엔진이 앵커 검산으로 배달한 KB 절편이고(비어 있으면 이 지점에
        #   못 온다), "레코드 0건 읽고 배열 산출 = 날조"라는 이 계수기의 전제가 성립하지 않는다.
        #   `Record ID:` 는 DB 덤프 전용 축이라 여기서도 원리상 0건 → 세우면 채널이 또 죽는다.
        #   판정은 닫혀 있다(_mat is not None·내용 판단 0). 날조 방어는 관문1(계약 선언 시)이 맡는다.
        if _mat is not None and _src_rows == 0 and (_deferred or _unguarded):
            print("[T2_SG_ISOLATE] %s: docs 전달 모드 — Record ID: 계수기 미적용(근거=엔진 배달분 "
                  "%d편·날조 방어=관문1)" % (d.get("name"), len(_ok_outs)),
                  file=_sys.stderr, flush=True)
        elif _src_rows == 0 and _deferred and not _unguarded:
            # 침묵-스킵 금지([[55]]): 서지 않았다는 사실을 남긴다.
            print("[T2_SG_ISOLATE] %s: source=0 이지만 %s 는 배열 근거 계약이 있어 관문1 이 심사한다"
                  " — 계수기 미적용(축: Record ID: 는 DB 덤프 전용)"
                  % (d.get("name"), ",".join(sorted(_deferred))),
                  file=_sys.stderr, flush=True)
        elif _src_rows == 0 and _unguarded:
            print("[T2_SG_ISOLATE] %s: source=0 rows 인데 배열 operand 산출 — **폐기**(날조 방지·"
                  "메인 인자 폴백)" % d.get("name"), file=_sys.stderr, flush=True)
            _isolate_trace(iso, d, {"mode": "fetch", "round": rnd + 1, "getter": getter,
                                    "discarded": "src0_fabrication_guard", "operands": got})
            return None
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
            import t2_subcall as _SC
            _raw = _SC.sub_generate(ag, la, UserMessage, prompt, "sg_inject",
                                    temperature=iso.get("temperature"))
            if not _raw:
                print("[T2_SG_ISOLATE] inject generate 실패(%s)" % gval, file=_sys.stderr, flush=True)
                continue
            got = _merge_json(_raw, set(ids))
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
                    import t2_subcall as _SC2
                    got2 = _merge_json(_SC2.sub_generate(ag, la, UserMessage, prompt + extra,
                                                         "sg_inject_retry",
                                                         temperature=iso.get("temperature")),
                                       set(bad_ids))
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
                    import t2_subcall as _SC3
                    got3 = _merge_json(_SC3.sub_generate(ag, la, UserMessage, prompt + extra2,
                                                         "sg_inject_retry",
                                                         temperature=iso.get("temperature")),
                                       set(_bad))
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


def _augment_byref_params(d):
    """★P6 BYREF 스키마 보강 **정본** (2026-08-22 x466 리뷰로 apply() 인라인에서 추출·[[67]] 사본 금지).

    apply()의 주입 경로와 재생 프로브(x466 등)가 **같은 함수**를 거쳐야 라이브/재생 스키마가 같다
    (074 라이브 호출이 `@last:` 참조를 낸 것은 이 안내가 스키마에 있었다는 뜻 — 프로브가 이것을
    빠뜨리면 A_asis 재현성이 조용히 깨진다). T2_SG_BYREF=1 일 때만 over/join 파라미터 설명에
    참조 문구를 부가한 **사본**을 돌려준다(원본 선언 불변) — OFF면 원본 그대로
    (엔진이 해석 못 하는 지시를 모델에게 주지 않는다·문구=도메인-일반·A2 파일 불변).
    """
    if os.environ.get("T2_SG_BYREF") != "1":
        return d
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
    return d


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


def _reorder_records(text):
    """env 기계 포맷 레코드 덤프를 **타입별로 묶고 묶음 안은 날짜 오름차순**으로 재배열한다.

    ★왜 (2026-08-25·x536·x539 격리): 074 의 전사 결손은 재료가 아니라 **덤프의 순서**였다.
      같은 6,752자 원문을 순서만 바꿔 주면 네 계좌가 갈린다(n=3 결정론·[[57]] 부정통제 포함):

        계좌(기대)   N_wire   D_old_group   N_scramble(무의미 순서)
        msg37(18)      18         18            18      ← 변별 없음
        msg38(16)    **17**       16          **17**    ← 날조 id 한 줄
        msg39(16)    **17**       16          **15**    ← 행 빠짐
        msg40(16)    **17**       16            16

      무의미한 재배열은 두 계좌를 오히려 부순다 ⇒ 산 것은 *다시 렌더링한 것*이 아니라
      **순서의 내용**이다. 승자 팔은 `atm_withdrawal`·`atm_fee` 라는 도메인 낱말로 정의돼
      있었으므로 그대로는 엔진에 못 들어온다([[05]]) — 여기 있는 규칙은 그 낱말이 **하나도
      없는** 판이고, x539 가 두 판이 같은 수를 내는지 잰 뒤에만 켜진다.

    ⚠엔진이 쓰는 문장 0 · 값 변경 0 · 판단 0 · 순위 0: 블록을 **축자 그대로** 옮기고 번호만
      다시 매긴다. 술어는 env 기계 포맷의 필드 이름 둘(`type`·`date`)뿐이고 **값은 안 본다**.
    ⚠형식이 아니면 그대로 돌려준다(fail-open·`Record ID:` 부재 시 무동작).
    ⚠불변식: 산출은 입력의 **순열**이다 — 래칫 `test_sg_record_order.py` 가 id 집합과 개수를
      맞대어 지킨다(내용 손실이 이 자리에서 나면 [[25]] 위반이다).
    """
    import re as _re3
    if "Record ID:" not in (text or ""):
        return text
    # ⚠**덤프가 둘 이상이면 손대지 않는다.** 호출부(`_recs2`)는 여러 도구 출력을 이어 붙이므로,
    #   그 위에서 재배열하면 서로 다른 원장의 행이 **한 묶음으로 섞인다** — 순서를 고치려다
    #   전사 결손을 제조하는 일이다([[25]]). 술어는 env 머리말 계수 하나(닫힘).
    if len(_re3.findall(r"Found \d+ record\(s\)", text)) > 1:
        return text
    parts = _re3.split(r"\n(?=\s*\d+\.\s+Record ID:)", text)
    head = ""
    if parts and "Record ID:" not in parts[0]:
        head, parts = parts[0], parts[1:]
    blocks = []
    for b in parts:
        if "Record ID:" not in b:
            if b.strip():
                return text                 # 레코드가 아닌 본문이 중간에 있다 = 모르는 형식
            continue                        # 빈 조각(분리자 잔재)은 버린다
        i = _re3.search(r"Record ID:\s*(\S+)", b)
        t = _re3.search(r"^\s*type:\s*(\S+)\s*$", b, _re3.M)
        d = _re3.search(r"^\s*date:\s*(\S+)\s*$", b, _re3.M)
        if not (i and t):
            return text                     # 한 블록이라도 축이 없으면 손대지 않는다
        blocks.append((i.group(1), t.group(1), d.group(1) if d else "", b))
    if not blocks:
        return text
    seen = []
    for b in blocks:
        if b[1] not in seen:
            seen.append(b[1])               # 타입 순서 = 원본의 첫 등장(이름을 고르지 않는다)
    out = []
    for ty in seen:
        out += sorted([b for b in blocks if b[1] == ty], key=lambda x: x[2])
    body = "\n".join("%d. %s" % (k + 1, b[3].strip().split(". ", 1)[-1])
                     for k, b in enumerate(out))
    return (head + "\n" + body) if head else body


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
    # ★수리(2026-08-18·C526⒜→C531): 색인이 **래퍼 이름만** 담았다. 이 환경은 discoverable
    #   도구를 `call_discoverable_agent_tool(agent_tool_name=…)` 로 디스패치하므로, 그렇게 부른
    #   도구는 `@last:<실제이름>` 이 **영영 안 맞고** "no committed non-error output" 으로
    #   거짓 deny 된다(전수 census 49건/11 sim·072:33·073:11·074:5).
    #   언랩 규칙은 엔진 정본 `t2_gate_patch._exact_tool_name` 을 **그대로 쓴다**(사본 금지·[[67]]).
    #   ⚠**넓히기만 한다** — 래퍼 이름과 안쪽 이름을 **둘 다** 색인하므로 종전에 맞던 참조는
    #     그대로 맞고, 못 맞던 참조만 맞게 된다(새 실패 모드 0·안전측).
    try:
        import t2_gate_patch as _gp_ref
        _exact = _gp_ref._exact_tool_name
    except Exception:
        _exact = None
    id2names, best = {}, None
    for m in orch.get_messages():
        for tc in (getattr(m, "tool_calls", None) or []):
            _nm = getattr(tc, "name", None)
            _names = {_nm}
            if _exact is not None:
                try:
                    _names.add(_exact(tc))
                except Exception:
                    pass
            id2names[getattr(tc, "id", None)] = {n for n in _names if n}
        if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        mid = getattr(m, "id", None)
        if kind == "@call" and mid == key:
            return c
        if kind == "@last" and key in (id2names.get(mid) or ()):
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
        # ★순서 수리(2026-08-18·C526⒠→C531): 이 블록이 실패하면 호출부가 `continue` 로
        #   조기 반환해 **아래 `isolate: fetch_formalize` 를 선점**한다. 그런데 그 서브는
        #   레코드를 **모델에게서 받지 않고 스스로 fetch·formalize** 해 `ctx.update` 로 이 키를
        #   덮어쓴다 — 즉 참조가 안 풀려도 **그 호출은 성공했을 호출**이다.
        #   ⇒ 서브가 덮어쓸 키(A2 가 `isolate.operand_keys` 로 **이미 선언**)면 여기서 죽이지
        #     않고 넘긴다. **A2 어휘 순증 0 · 새 플래그 0 · 실패 경로에서만 동작**(성공 경로 불변).
        #   ⚠폴백은 살아 있다 — 서브까지 실패하면 `@last:` 문자열이 남고, 아래 over-str 검사가
        #     재송신을 요구한다(침묵 통과 아님).
        _iso_ref = _isolate_spec(d) or {}
        _iso_owns = (_iso_ref.get("mode") == "fetch_formalize"
                     and k in set(_iso_ref.get("operand_keys") or []))
        # ★A10 / OL-48 (t7336 마스터 §6.1·2026-08-22): 필드-요구 검사도 **이 우회 안**에 둔다.
        #   C526⒠→C531 이 우회를 만든 원 의도가 이 자리인데, 구판은 `try` 가 `_parse_record_dump`
        #   하나만 감싸서 `_byref_require_fields` 의 `_ByrefError` 가 우회를 **그대로 통과**했다.
        #   074#0/#1 실측: `[T2_SG_BYREF] … "supply 'transactions' yourself with those fields
        #   filled in"` ×4 → 모델이 손-전사 5회(6.7~7.8KB) → 그 행을 **격리 서브가 전부
        #   덮어쓴다**(`fetch-formalize operand 주입`) ⇒ 순수 낭비 33KB → `context_window_exceeded`.
        #   `isolate.fetch_formalize` 를 선언한 키에서 `@last:` 참조는 **원리상 성립 못 한다**
        #   (서브가 모델에게서 받지 않고 스스로 fetch 한다) — 그러니 그 키의 컬럼 부재로
        #   모델에게 전사를 요구하는 것은 어느 조건에서도 틀렸다.
        # ⚠폴백 유지: `continue` 라 `ctx[k]` 는 `@last:` 문자열로 남고, 서브까지 실패하면
        #   아래 over-str 검사가 여전히 재송신을 요구한다(침묵 통과 아님).
        # ⚠[[70]] 계측 의무 — 파는 것: `_iso_owns` 키에서 **컬럼 부재 지목이 사라진다**.
        #   서브가 잘못된 컬럼을 산출하면 종전에는 이 지목이 먼저 잡았다(이제는 op 의 abstain·
        #   `[coverage]`/`missing_fields` 가 잡는다). 다음 런 포렌식이 셀 것 =
        #   ⑴`[T2_SG_BYREF] … 미해석` 건수 ⑵손-전사 본문 크기(assistant 턴 바이트)
        #   ⑶`context_window_exceeded` 재발 ⑷서브 실패 시 over-str 재송신 요구가 살아 있는가.
        try:
            rows = _parse_record_dump(_resolve_ref_output(orch, v))
            _byref_map_fields(d, rows)                 # A2 선언 컬럼명 대응(§4-1 후속)
            _byref_require_fields(d, k, rows)          # 필요한 컬럼 부재 = 침묵 대신 지목
        except _ByrefError:
            if not _iso_owns:
                raise
            print("[T2_SG_BYREF] %s: '%s' 미해석 — isolate(fetch_formalize)가 '%s' 를 "
                  "산출하므로 deny 하지 않고 넘긴다" % (d.get("name"), v, k),
                  file=_sys.stderr, flush=True)
            continue
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
    # ★`__tool_outputs_raw` (2026-08-22·R3): 기존 두 키는 **소문자화**된다 — grounding 의
    #   substring 대조용이라 그렇게 굳었다. 그런데 서브에 **재료로 실을 때**는 원문이어야
    #   한다(KB 는 "Silver Account" 로 적혀 있고 소문자본을 주면 우리가 다시 표기를 흐린다).
    #   기존 키는 손대지 않는다 — 소비자가 여럿이다([[67]]).
    return {"__user_text": " ".join(users).lower(),
            "__tool_outputs": {k: v.lower() for k, v in outs.items()},
            "__tool_outputs_raw": dict(outs)}


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
            # ★P6(§P6): BYREF 스키마 보강 — 정본 `_augment_byref_params` 하나를 거친다
            #   (2026-08-22 x466 리뷰 추출·[[67]]: 재생 프로브가 라이브와 **같은** 스키마를 봐야 한다).
            d = _augment_byref_params(d)
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
                        # ★P3(N97 §3): 접미사 포함 호출형은 **어느 문구를 쓰든** 붙는다.
                        #   2026-08-05 스모크 실측: banking은 이 도구에 `requires_reads_feedback`를
                        #   선언해 두었고, 아래 오버라이드 분기가 먼저 반환하는 바람에 호출형이
                        #   **15회 중 0회** 동봉됐다(052가 같은 거부를 12회 반복하다 죽은 그 문구다).
                        #   해소 실패 시 빈 문자열이므로 어느 분기에서도 문구 변화는 없다.
                        try:
                            import t2_callable_hint as _CH
                            _hint_r = _CH.hint(self, _missing_r, _unl_r, _cal_r)
                        except Exception:
                            _hint_r = ""
                        _fb_r = d.get("requires_reads_feedback")
                        if _fb_r:
                            _msg_r = _fb_r.replace("{missing}", ", ".join(_missing_r)) + _hint_r
                        elif not (_unl_r and _cal_r):
                            # dispatcher 미선언 = 접미사/잠금해제 개념이 없는 도메인 → 구조 안내 생략
                            _msg_r = ("Error: [READ-FIRST] this calculation depends on records you have "
                                      "not read yet in this conversation. Missing required reads (BASE "
                                      "names): %s. Read them first, then retry this calculation."
                                      % ", ".join(_missing_r))
                        else:
                            # ★P3(N97 §3): 이 문구는 "접미사를 KB에서 찾아라"까지만 말한다. 같은 통지에서
                            #   050은 KB 검색으로 회복하고 052는 shell grep 6회 끝에 문맥 초과로 죽었다.
                            #   찾으라고 하는 대신 **접미사를 env 레지스트리에서 뽑아 준다**(A2 저작 0).
                            #   유일 해소만 지목하고, 못 풀면 빈 문자열이라 문구가 그대로 남는다.
                            try:
                                import t2_callable_hint as _CH
                                _hint_r = _CH.hint(self, _missing_r, _unl_r, _cal_r)
                            except Exception:
                                _hint_r = ""
                            _msg_r = ("Error: [READ-FIRST] this calculation depends on records you have not read "
                                      "yet in this conversation. Missing required reads (BASE names): %s. These are "
                                      "discoverable tools whose REAL names carry a numeric suffix - do NOT unlock "
                                      "the base name as-is; first find each tool's full suffixed name in the "
                                      "knowledge base (search for the base name), then %s "
                                      "with that full name and call it via %s. Read the "
                                      "ACTUAL values from the records, then call this tool again."
                                      % (", ".join(_missing_r), _unl_r, _cal_r)) + _hint_r
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
                # ★C564 후속(2026-08-20 밤·x443 스모크가 잡은 자리): 값 주석의 **금액 피연산자**가
                #   손님 발화에 실재하는지 확인하고, 없으면 그 인자를 **떨군다**(주석이 안 붙는다).
                #   스모크 실물 — 063 은 손님이 말한 적 없는 `spend_amount=6000` 을 냈다(발화에는
                #   $8,000 저축과 $10–15 차이만 있다). 그대로 두면 **날조된 수 위에서 우리가 곱셈을
                #   해 주고**, 그 값이 근거처럼 보인다. 검사는 C45 동형 — 모델이 낸 값의 **원문 실재
                #   확인**만 하고 엔진은 값을 만들지도 고르지도 않는다([[59]] ⓐ 허용 범주).
                _vf0 = ((d.get("op") or {}).get("value_formula") or {})
                _ap0 = _vf0.get("amount_param")
                if _ap0 and _ctx.get(_ap0) not in (None, ""):
                    _ut = (_evidence_ctx(self).get("__user_text") or "")
                    _raw = str(_ctx.get(_ap0))
                    _digits = "".join(ch for ch in _raw if ch.isdigit())
                    _forms = {_raw.lower(), _digits}
                    if _digits:
                        try:
                            _forms.add("{:,}".format(int(_digits)))
                        except Exception:
                            pass
                    if not any(f and f in _ut for f in _forms):
                        print("[T2_VALUE_FORMULA] %s: %s=%r is not in what the customer said — "
                              "dropped (no value annotation)" % (d.get("name"), _ap0, _raw),
                              file=_sys.stderr, flush=True)
                        _ctx = dict(_ctx)
                        _ctx.pop(_ap0, None)
                # ★배달 배선 — A2 선언 문서를 **격리 서브 하나**에 넘긴다
                #   (`T2_ARG_DOC_SUB=1`·기본 OFF·2026-08-21·사용자 정의 [[71]] 축자:
                #   *"모든 격리 서브에이전트는 필요한 내용만 받고 격리된 상태에서 필요한 기능만
                #   한다"* · *"A2 A3 에서 하는 것은 어떤 서브에이전트에서 어떤 결정을 위해 필요한
                #   문서가 뭔지 정확하게 기술하는 것만"*).
                #
                #   분담: A2 가 **무엇을 읽을지** 선언 · 엔진이 **읽어 넘기기만** · 서브(=같은 모델,
                #   격리)가 **값을 낸다** · 엔진은 인용이 **넘긴 재료 안에 실재하는지**만 검산한다
                #   (닫힌 술어 둘: 값의 선언-집합 소속 · 인용의 실재·[[59]]·[[22]]).
                #   ⚠엔진은 고르지 않는다 — 순위·최댓값 집기·"정답은 X" 가 없다([[62]]).
                #
                #   왜 전달인가([[62]] 사다리·측정 선행): 격리 n=71 에서 **문서를 안 주면 44/71**,
                #   **선언 문서를 주면 71/71**(C576·`x448_index_vs_all_iso.py` wide1) ⇒ 격리에서
                #   되므로 레버는 **전달뿐**이다. 왜 검색이 아닌가: bm25·dense 는 선언 12편 중
                #   **11편을 한 번도 안 돌려준다**(C577·`x449_dense_vs_declared_overlap.py`·71 전수)
                #   — 이 자리의 판정 문서에 **검색으로는 닿지 않는다**.
                #   정책 문서 읽기가 우리 층 몫인 것은 확정된 경계다(`t2_search` §경계·C405ⓔ);
                #   규칙 0 은 **DB 도구 출력**에 대한 것이다(`SCAFFOLD_AUDIT_RULE0_2026_07_08`).
                #   ★게이팅 정정 (2026-08-21·스모크 2판 실측): 처음엔 *"모델이 이미 그 인자를 낸
                #     호출에서만"* 으로 묶었는데, 라이브에서 모델은 fit 을 **인자 없이** 부른다
                #     (스모크 2/2 · 전수 326 sim 중 그 인자를 낸 것은 131 = 40%가 상한) ⇒ 레버가
                #     **거의 죽어 있었다**. 그리고 격리가 잰 것은 *"손님 발화 + 선언 문서로 범주를
                #     정하는 일"*(x448)이지 *"이미 낸 값을 고치는 일"* 이 아니다 — 즉 **없는 경우가
                #     오히려 측정된 조건**이다. ⇒ 그 도구가 불리면 **결정이 살아 있다**고 보고 서브가
                #     판단한다. 서브가 null 을 내면 인자는 그대로 없다(=기본 요율·⒡ ⊃ ⒟ 유지).
                #   ⚠같은 (인자·손님 발화)에는 **한 번만** 묻는다(호출마다 서브를 새로 띄우지 않는다).
                if (os.environ.get("T2_ARG_DOC_SUB") == "1"
                        and os.environ.get("T2_CATEGORY_CITE") != "1"):
                    _cad = (d.get("catalog_arg_docs")
                            or ((getattr(self, "_t2_sg_a2", None) or {}).get("catalog_arg_docs"))
                            or {})
                    for _ag2, _dcl in _cad.items():        # 선언 순서 그대로(우리가 정렬하지 않는다)
                        if _ag2[:1] == "_" or not isinstance(_dcl, dict):
                            continue
                        _vals = [k for k in _dcl if k[:1] != "_"]   # A2 가 적은 순서 그대로
                        _dids = []
                        for _k2 in _vals:
                            for _x2 in (_dcl.get(_k2) or []):
                                if isinstance(_x2, str) and _x2 not in _dids:
                                    _dids.append(_x2)
                        _mat, _docs = "", {}
                        try:
                            import t2_search as _ts4
                            import t2_subcall as _SC4
                            import tau2.agent.llm_agent as _la4
                            from tau2.data_model.message import UserMessage as _UM4
                            _cps = _ts4.corpus_from_env(getattr(self, "environment", None))
                            _docs, _miss = _ts4.read_docs(_dids, corpus=_cps)
                            if _miss:
                                print("[T2_ARG_DOC_SUB] 선언됐는데 코퍼스에 없는 문서: %r"
                                      % (_miss,), file=_sys.stderr, flush=True)
                            _mat = "\n\n".join("### %s\n%s" % (i, _docs[i]) for i in _docs)
                        except Exception as _e4:
                            print("[T2_ARG_DOC_SUB] skip=import/corpus %r" % (_e4,),
                                  file=_sys.stderr, flush=True)
                            _mat = ""
                        if not _mat:
                            print("[T2_ARG_DOC_SUB] skip=no-material arg=%s" % (_ag2,),
                                  file=_sys.stderr, flush=True)
                            continue
                        _utx = (_evidence_ctx(self).get("__user_text") or "")[:5000]
                        # ★같은 (인자·손님 발화)면 다시 묻지 않는다 — fit 은 한 sim 에서 여러 번
                        #   불리고, 같은 입력에 같은 답을 받으려고 서브를 또 띄우는 것은 비용일 뿐이다.
                        #   ⚠발화 대신 **메모 적중**을 세지 않도록 로그는 두 경우 모두 남긴다(死배선 탐지).
                        _memo = getattr(self, "_t2_argdoc_memo", None)
                        if _memo is None:
                            _memo = self._t2_argdoc_memo = {}
                        _mk = (_ag2, hash(_utx))
                        if _mk in _memo:
                            _hit = _memo[_mk]
                            _ctx = dict(_ctx)
                            if _hit:
                                _ctx[_ag2] = _hit
                            else:
                                _ctx.pop(_ag2, None)
                            print("[T2_ARG_DOC_SUB] %s=%r (메모 재사용)" % (_ag2, _hit),
                                  file=_sys.stderr, flush=True)
                            continue
                        # ★지시는 **재료보다 앞**에 둔다 (2026-08-21·`x450` 로 원인 확정).
                        #   첫 판은 `# Documents … # What the customer said … Decide ONE thing:` 순서라
                        #   지시가 **문서 15,000자 뒤**에 묻혔고, 그것만으로 판정이 뒤집혔다:
                        #   task_024 전수 26 사례에서 **격리 26/26 ↔ 라이브 형태 0/26**(C578).
                        #   제목 손실도 소문자화도 무죄였고(둘 다 0/26 그대로), **위치만** 앞으로 옮긴
                        #   팔(`H_front`)이 격리로 되돌렸다 ⇒ 고칠 것은 순서 하나다.
                        #   ⚠성적을 위해 문구를 튜닝한 것이 아니다 — 격리가 잰 조건(지시가 맨 앞)을
                        #     라이브가 그대로 받게 맞춘 것이다.
                        _pr4 = ("Decide ONE thing: the value of `%s`. Reply with ONE JSON object "
                                "only: {\"%s\": <one of: %s> or null, \"quote\": \"<one sentence "
                                "copied word for word from the '# Documents' section that shows "
                                "this>\"}. The quote MUST come from the documents, never from the "
                                "customer. If no document sentence supports a value, set it to null."
                                "\n\n# Documents\n%s\n\n# What the customer said\n%s\n"
                                % (_ag2, _ag2, ", ".join(_vals), _mat, _utx))
                        _raw4 = _SC4.sub_generate(getattr(self, "agent", None), _la4, _UM4,
                                                  _pr4, "sg_arg_docs")
                        _ans4 = _SC4.parse_contract(_raw4) or {}
                        _v4 = str(_ans4.get(_ag2) or "").strip().lower()
                        _q4 = str(_ans4.get("quote") or "")
                        _real4 = bool(_q4) and _ts4.quote_in(_q4, _mat)
                        _ctx = dict(_ctx)
                        if bool(_v4) and _v4 in _vals and _real4:
                            print("[T2_ARG_DOC_SUB] %s: %r -> %r (격리 서브·선언 %d편·인용 실재)"
                                  % (_ag2, _ctx.get(_ag2), _v4, len(_docs)),
                                  file=_sys.stderr, flush=True)
                            _ctx[_ag2] = _v4
                            _memo[_mk] = _v4
                        else:
                            print("[T2_ARG_DOC_SUB] %s=%r 없음 — 넘긴 문서 %d편에 근거가 없다"
                                  " (서브값=%r 인용실재=%s) ⇒ 기본 요율"
                                  % (_ag2, _ctx.get(_ag2), len(_docs), _v4, _real4),
                                  file=_sys.stderr, flush=True)
                            _ctx.pop(_ag2, None)
                            _memo[_mk] = None
                            _ctx["__cat_cite_note"] = (
                                "[T2_ARG_DOC_SUB] %s was not used: the base rate was applied "
                                "instead, because the documents that define it (%s) contain no "
                                "sentence supporting it." % (_ag2, ", ".join(_dids[:2])))

                # ★⒡ 범주 인용 게이트(2026-08-20 밤·`T2_CATEGORY_CITE=1`·기본 OFF).
                #   측정이 말한 것: 범주는 **문서 없이** 정해진다(C570: fit 호출의 77%가 KB 검색 0회)
                #   그리고 근거 없는 범주 위에서 곱하면 **~55%가 부풀린 값을 따라간다**(C566).
                #   A2 색인을 문서로 **읽어 주면** 격리에서 4/4 로 갈렸지만(C571), 그렇게 하면
                #   **에이전트가 안 가져온 DB 내용**을 엔진이 대신 쓰는 것이라 **규칙 0** 이다
                #   (`SCAFFOLD_AUDIT_RULE0_2026_07_08` 판정 기준 축자). ⇒ 여기서는 규칙 0 을 지킨다:
                #     · 엔진은 **문서 id 만 가리킨다**(내용 주입 0) — 무엇을 가져올지 말할 뿐이다
                #     · 인용은 **에이전트 자신이 가져온 도구 출력**에서만 검산한다(C45 동형·[[59]]ⓐ)
                #     · 못 대면 범주를 **떨구고**(=기본 요율만) 거절문이 **고칠 것을 이름 댄다**([[64]])
                if (os.environ.get("T2_CATEGORY_CITE") == "1"
                        and str(_ctx.get("spend_category") or "").strip()):
                    _cidx = ((d.get("catalog_arg_docs") or {}).get("spend_category") or {})
                    if not _cidx:
                        try:
                            _cidx = (((getattr(self, "_t2_sg_a2", None) or {}).get("catalog_arg_docs")
                                      or {}).get("spend_category") or {})
                        except Exception:
                            _cidx = {}
                    _cat = str(_ctx.get("spend_category")).strip().lower()
                    _own = " ".join((_evidence_ctx(self).get("__tool_outputs") or {}).values())
                    _q = str(_ctx.get("spend_category_quote") or "")
                    _ok = False
                    if _q:
                        try:
                            import t2_search as _ts
                            _ok = bool(_ts.quote_in(_q, _own))
                        except Exception:
                            _ok = _q.lower() in _own.lower()
                    if not _ok:
                        # ★지시는 **실행 가능한 꼴**이어야 한다(2026-08-20 밤·사용자 지적
                        #   *"문서를 읽게 인덱스하고 선택시 문서를 실제로 읽어서 판단하게 해야 한다"*).
                        # ⛔**정정 2026-08-21 (C572)**: 앞 판은 `KB_search` 로 **제목을 검색**하라고
                        #   했는데 그 도구는 **우리 설정에 없다** — 우리 런은 `alltools` 이고
                        #   `KB_search_bm25`·`KB_search_dense`·`shell` 뿐이다(go_stack.sh:205·궤적 실측).
                        #   샌드박스는 문서를 `<doc_id>.md` 로 내보내고 라이브 프롬프트에 축자로
                        #   *"File names are based on document IDs"* 가 있다 ⇒ **선언한 id 가 곧 파일명**
                        #   이라 `shell: cat <id>.md` 로 **정확히** 집힌다. 검색으로 우회할 필요가 없다.
                        # ★그리고 검색은 이 자리에서 **되지도 않는다**(C577): 선언 12편 중 **11편이**
                        #   bm25·dense 71 사례에서 **0회**이고, 이 범주를 정하는 문서(gold_003)도 0회다.
                        _ids = [x for x in (_cidx.get(_cat) or []) if isinstance(x, str)]
                        _titles = [x for x in (_cidx.get("_titles_%s" % _cat) or []) if isinstance(x, str)]
                        _ctx = dict(_ctx)
                        _ctx.pop("spend_category", None)
                        _hint = ("[T2_CATEGORY_CITE] spend_category=%r was not used: the base rate was "
                                 "applied instead, because no sentence from a document you retrieved "
                                 "supports that category." % _cat)
                        if _ids:
                            _hint += (" The document that defines it is %s.md — read it with the shell "
                                      "tool (`cat %s.md`)%s, then call this tool again with "
                                      "spend_category_quote set to a sentence from what you get back."
                                      % (_ids[0], _ids[0],
                                         (" — its title is \"%s\"" % _titles[0]) if _titles else ""))
                        print(_hint, file=_sys.stderr, flush=True)
                        _ctx["__cat_cite_note"] = _hint
                _res = _c.apply_op(d.get("op"), _ctx)
                # ★A2 `result_round` (2026-08-22·093 실시간 포렌식): 부동소수점 잔차를 선언된
                #   자릿수로 접는다. 실물 — 이 도구가 `32.999999999999986` 을 냈고 모델은 그것을
                #   통화로 **옳게** `33.0` 으로 반올림해 write 했는데, `T2_WRITE_EVIDENCE` 가
                #   *"the amount_difference (33.0) does not appear in any get_interest_correction
                #   tool output"* 로 **10회 반려**했다. 즉 **우리 도구의 표현 오차가 우리 게이트를
                #   스스로 막았다** — 모델은 아무것도 틀리지 않았다([[25]] 우리 도구는 100% 정답
                #   의무: 출력 결함이 유일한 근거원을 오염시킨다).
                #   원인은 op 의 `const 0.08333333333333333`(=1/12 근사)이고, 곱셈 순서상 잔차가
                #   마지막 자리에 남는다. 산수를 고치는 대신 **표현을 접는다** — 반올림은 통화의
                #   정의이고, 접은 값이 곧 우리가 증거로 쓰는 값이어야 한다.
                #   ⚠**범위 게이트(`result_range`)보다 앞**에 둔다: 두 검사와 반환문이 모두 같은
                #     수를 봐야 한다(접기 전후가 갈리면 A8 이 접기 전 값으로 판정한다).
                #   ⚠자릿수는 A2 선언뿐이고 엔진 리터럴 0([[05]]). 미선언 도구는 거동 변화 0.
                #   ⚠gold 미참조([[23]]) — 근거는 크레딧 도구의 인자 계약이 **달러 금액**을 요구한다는
                #     것(정책 축자·`policy_facts` 인용 행)이지 정답 대조가 아니다.
                #   ⚠[[70]] 무엇을 파는가: 접는 만큼 **정밀도**를 잃는다. 접기가 값을 바꾸는 자리
                #     (APY 2.775 → 2.78 처럼)에서는 손해가 실질이므로 **금액을 내는 스칼라 도구에만**
                #     선언한다 — 근거 없는 확대 금지([[62]]). 다음 런 포렌식이 셀 것 =
                #     `[T2_SG_ROUND]` 발화 수 ↔ 그 뒤 WEV deny 수(줄어야 한다).
                _rr = d.get("result_round")
                if (_rr is not None and isinstance(_res, (int, float))
                        and not isinstance(_res, bool)):
                    _r0 = _res
                    _res = round(float(_res), int(_rr))
                    if _r0 != _res:
                        print("[T2_SG_ROUND] %s: %r -> %r (자릿수 %s)"
                              % (d.get("name"), _r0, _res, _rr),
                              file=_sys.stderr, flush=True)
                if isinstance(_res, dict) and _ctx.get("__cat_cite_note"):
                    _res["note"] = (str(_res.get("note") or "") + " " + _ctx["__cat_cite_note"]).strip()
                # ★A8 / OL-11 (t7336 마스터 §6.1·2026-08-22): **결과 범위 게이트**(abstain).
                #   093#1 실측 — `expected<actual` 이면 이 계산은 음수를 내는데 반환문이
                #   *"Use this as the credit amount"* 로 지시한다. 그 크레딧 도구의 인자 계약은
                #   정책 축자로 *"amount (number): The positive dollar amount to credit (must be
                #   greater than 0)"* 다 ⇒ 현행 반환문은 **집행 불가능한 지시**다([[25]] 우리 도구는
                #   100% 정답 의무·[[64]] 거부는 무엇을 하면 풀리는지 담아야 한다).
                #   술어는 닫혀 있다: 산수 하나(`result <= min_exclusive`)뿐이고 도메인 판단 0.
                #   경계값·문면은 **전부 A2 선언**(`result_range`·`result_range_feedback`)이라
                #   엔진 리터럴 0 이고, 미선언 도구는 거동 변화 0 이다([[05]]).
                # ⚠엔진은 **고르지 않는다**: 무엇이 옳은 APY 인지·환수인지 보고인지는 말하지 않고,
                #   "이 수는 크레딧 인자로 쓸 수 없다 + 다시 확인할 read/계산의 이름" 만 준다.
                # ⚠[[70]] 계측 의무 — 파는 것: **정당한 음수 케이스**(과지급 환수·보고서의
                #   `amount_difference` 음수)에서도 이 도구가 값을 안 돌려준다. 다음 런 포렌식이
                #   셀 것 = ⑴`[T2_SG_RESULT_RANGE]` 발화 수와 그 sim ⑵그 뒤 크레딧 write 의
                #   부호 ⑶abstain 이 막은 turn 수(비용) ⑷093#1 재현 여부.
                #   끄기는 `T2_SG_RESULT_RANGE=0` 한 칸(=A/B 용·기본 ON·[[60]]).
                _rrg = d.get("result_range") or {}
                if (_rrg and os.environ.get("T2_SG_RESULT_RANGE", "1") != "0"
                        and isinstance(_res, (int, float)) and not isinstance(_res, bool)):
                    _lo = _rrg.get("min_exclusive")
                    if _lo is not None and float(_res) <= float(_lo):
                        _sm8 = _SafeMap({kk: vv for kk, vv in _ctx.items()
                                         if isinstance(vv, (str, int, float))
                                         and not str(kk).startswith("_")})
                        _sm8["result"] = _res
                        _sm8["min_exclusive"] = _lo
                        _fb8 = str(d.get("result_range_feedback") or "").format_map(_sm8)
                        if _fb8:
                            print("[T2_SG_RESULT_RANGE] %s abstain: result=%s <= min_exclusive=%s"
                                  % (d.get("name"), _res, _lo), file=_sys.stderr, flush=True)
                            self._t2_abstain_last = d.get("name")
                            ours[id(tc)] = ToolMessage(
                                id=tc.id, role="tool",
                                requestor=getattr(tc, "requestor", "assistant"),
                                error=True, content=_fb8)
                            continue
                        # 문면 미선언 = 말할 것이 없다 → 침묵(종전 거동·[[25]] 모르면 말하지 않는다)
                        print("[T2_SG_RESULT_RANGE] %s: 범위 밖(result=%s)이나 "
                              "`result_range_feedback` 미선언 — 종전 반환문 유지"
                              % (d.get("name"), _res), file=_sys.stderr, flush=True)
                if isinstance(_res, list):                    # 목록형(discrepancy ids)
                    _res = [str(i) for i in _res if i]
                    # ★이 호출이 무엇을 냈는지 호출 id로 등재한다 (2026-08-05·패턴 제거).
                    #   G3(미제출 확정 행)은 이 집합을 **엔진 출력 텍스트에서 다시 찾아** 썼고, 찾는
                    #   수단이 A2의 철자 규칙이었다. 그 규칙은 JSON `\b`가 백스페이스로 들어가 있어
                    #   **한 번도 매치된 적이 없다** — 레버는 켜진 채로 죽어 있었다. 엔진이 방금 계산한
                    #   목록을 그대로 두면 찾을 일도, 철자도 없다(관측 전용·플래그 무관·거동 0).
                    _sgi = getattr(self, "_t2_sg_ids", None)
                    if _sgi is None:
                        _sgi = self._t2_sg_ids = {}
                    _sgi[getattr(tc, "id", None)] = list(_res)
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
                    # ★{delta_total}(2026-08-13 t7274w 073: id만으론 모델이 차액 아닌 값을 크레딧 —
                    #   x288 A_DOCS 0/8 이 잰 산술 결손 범위 내). 엔진이 이미 남긴 delta 들의 합만
                    #   노출한다 — 표시 여부/문구는 A2 템플릿 몫(안 쓰면 거동 0·여분 kwarg 무해).
                    _dtot = round(sum((it.get("delta") or 0) for it in _dets), 2) if _dets else 0.0
                    _txt = d.get(_tpl_key, "{ids}").format(
                        ids=", ".join(_res) if _res else "(none)", details=_details,
                        delta_total=_dtot)
                    _n = len(_res)
                    # ★C195: 판정 커버리지 병기(op가 _sg_stats를 남긴 경우만·거동보존).
                    #   "(none)"의 침묵-신뢰 차단: 몇 행을 판정했고 몇 행이 판정불가였는지 +
                    #   빈 결과일 때 재확인 지시(도메인 어휘 0). 근거=야간 020/027/029 거짓 "(none)".
                    _st = _ctx.get("_sg_stats")
                    if isinstance(_st, dict):
                        _txt += ("\n[coverage] %d of %d rows were checked (%d could not be "
                                 "verified)." % (_st.get("judged", 0), _st.get("total", 0),
                                                 _st.get("skipped", 0)))
                        # ★FIX-14/15 (2026-08-14 야간·074): 위 분모는 **이 호출에 넘어온 행 수**라
                        #   자기 자신을 잰다. 격리 서브가 원천에서 읽은 행 수를 알고 있으므로,
                        #   넘어오지 않은 행이 있으면 그 수를 병기하고(FIX-14) 재공급 경로를
                        #   이름으로 댄다(FIX-15·[[64]] 거부는 고칠 방법까지). ⚠하드 비율 가드는
                        #   두지 않는다 — 073 은 원천의 **진짜 부분집합**을 넘기는 것이 정상이고
                        #   비율로 막으면 통과를 죽인다. 판정은 모델 몫(엔진=두 수의 뺄셈뿐).
                        _txt += _omitted_rows_note(
                            (getattr(self, "_t2_sub_srcrows", None) or {}).get(d.get("name")))
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
                                # ★2026-08-05(019): 어느 행인지 + 그 행만 다시 물으라는 지시.
                                #   문구는 A2, 엔진은 id 나열만 한다.
                                _uids = (_st.get("unverified_ids") or [])[:8]
                                _rq = (((_iso3 or {}).get("quote_pin") or {})
                                       .get("unverified_requery_note"))
                                if _uids and _rq:
                                    _txt += " " + str(_rq).replace("{ids}", ", ".join(_uids))
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
                  and getattr(tc, "name", None) in ((a2 or {}).get("unavailable_tools") or {})
                  and not _tool_backend_live(self, getattr(tc, "name", None))):
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
