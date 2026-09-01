#!/usr/bin/env python
"""tau2 게이트 hook: BaseOrchestrator._execute_tool_calls 몽키패치 — A2-구동(키스톤 후).

★엔진 = `gate_interpreter.GateInterpreter`(벤치-일반·도메인 분기 0). 이 패치는 wiring만:
  에이전트 툴콜을 실행 *전* GateInterpreter로 검사, deny면 실행 없이 게이트 메시지를
  ToolMessage(error)로 반환, allow면 원본 실행 후 결과로 게이트 상태 갱신.
도메인 활성화·도구셋·autofetch producer·식별arg-types·placeholder = 전부 `a2/<domain>.gate.json`서
로드(env.domain_name로 선택). 코드 하드코딩(retail 도구명·GATE_DOMAINS) 폐기(2026-06-21 키스톤).

활성화: `import t2_gate_patch; t2_gate_patch.apply()`. 게이트는 orchestrator 인스턴스당 1개.
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import (  # noqa: E402
    observe_tools,
    GateInterpreter, auth_satisfier_tools, load_domain_a2, resolvers_from_env,
    candidate_summary, nested_candidate_summary, compute_facts)

# ── ★P7(C208⑥·DAY5_PRESCRIPTIONS §P7): 레버 무음실패 금지 — try/except로 삼켜진 레버 예외가
#    "정상"으로 위장하는 것을 런 종료 요약이 자동 고발한다(day5 unavail 0/223 전량 NameError 실측).
#    카운터만(판단 0·도메인 무관). 종료 시 [T2_LEVER_HEALTH] 1줄.
_LEVER_HEALTH = {}


def _lever_health(lever, kind):
    _LEVER_HEALTH.setdefault(lever, {}).setdefault(kind, 0)
    _LEVER_HEALTH[lever][kind] += 1


def _lever_health_report():
    for lv, c in sorted(_LEVER_HEALTH.items()):
        line = " ".join("%s=%d" % (k, v) for k, v in sorted(c.items()))
        flag = "  ⚠ALL-SKIPPED" if (c.get("skipped") and not c.get("ok")) else ""
        print("[T2_LEVER_HEALTH] %s: %s%s" % (lv, line, flag), file=sys.stderr, flush=True)


import atexit  # noqa: E402

# ★층-1 등록점 (2026-08-07·정본 §5.5 배선). 레버가 **자기 이름을 대는 자리**에서 스택에
#   등록한다 — 바깥에서 `*_fb` 변수명으로 플래그를 되짚는 것은 추측이고, 실제로 틀렸다.
#   `orch`를 안 주면 종전 stderr 한 줄뿐이라 기존 호출부 거동은 불변이다.
from t2_lever_beat import beat as _lbeat  # noqa: E402
atexit.register(_lever_health_report)


def _dyn_mt_target(err_str, margin=64, floor=256):
    """★P1(C208①) 순수함수: vLLM CWE 에러 원문에서 (model_max, input_tokens)를 파싱해
    새 max_tokens를 계산. 파싱 실패 또는 플로어 미만(진짜 창 소진) = None → graceful-stop.
    추정 0(에러가 정확한 수를 준다)·도메인 무관."""
    m = re.search(r"maximum context length is (\d+) tokens and your "
                  r"request has (\d+) input tokens", str(err_str or ""))
    if not m:
        return None
    new_mt = int(m.group(1)) - int(m.group(2)) - int(margin)
    return new_mt if new_mt >= int(floor) else None

# ── 도메인-일반 기본값 (A2가 override·enrich; retail/도메인 하드코딩 아님) ──
# ★[[05]] 감사(2026-07-13): 도메인-일반 식별 토큰만 엔진에. 도메인-특화 어휘
#   (order_id·item=retail·reservation=airline)는 A2 identifying_arg_types로 이관 —
#   엔진 DEFAULT가 도메인-union이면 새 도메인(banking)에 retail/airline 어휘가 누수됨.
#   'id'가 *_id 계열(order_id·item_ids·reservation_id) 전부 포괄하므로 이관은 behavior-preserving.
#   payment·address = 커머스-교차 일반(retail∩airline)이라 엔진 잔류. 도메인 어휘는 A2가 union.
DEFAULT_ARG_HINTS = ("email", "name", "zip", "user_id", "username", "id",
                     "payment", "address", "phone")
DEFAULT_PLACEHOLDERS = {
    # 도메인-일반 placeholder만(엔진). 도메인-특화 포맷(#W0000000=retail 주문-id)은
    # A2 placeholders로 이관(2026-07-13 [[05]] 감사). 아래는 전 도메인 공통 test-값.
    "something@example.com", "jane_doe@example.com",
    "john.doe@example.com", "johndoe@example.com", "john@example.com",
    "jane@example.com", "user@example.com", "test@example.com",
    "example@example.com", "123 Main St", "123 Main Street",
    "ABC123", "XYZ789",  # 도메인-일반 영숫자 placeholder 패턴 (구 airline A2서 이관·minimize-A2)
}
PROV_ARG_HINT = DEFAULT_ARG_HINTS          # 호환 alias
COMMON_PLACEHOLDERS = DEFAULT_PLACEHOLDERS  # 호환 alias

# ★tool_choice='required' 강제 생성의 max_tokens 하한 (2026-07-23·vLLM #19051/#36794 근본원인):
#   강제 tool-call JSON이 절단되면 hermes 파서 EOF→오도성 400. 라이브(미설정) 무영향·소형설정 방어.
_FORCE_MIN_TOKENS = int(os.environ.get("T2_FORCE_MIN_TOKENS", "1024"))

_A2_CACHE = {}


def obs_tools_g(gate):
    """live GateInterpreter -> observe 대상 도구 집합 (A2-구동)."""
    try:
        return observe_tools(getattr(gate, 'gates', []) or [])
    except Exception:
        return set()


def _domain_a2(domain):
    """env.domain_name → augmented A2 dict (없으면 None=게이트 비활성). 캐시."""
    if domain in _A2_CACHE:
        return _A2_CACHE[domain]
    a2 = load_domain_a2(domain) if domain else None
    if a2 is not None:
        a2 = dict(a2)
        a2["_auth_tools"] = auth_satisfier_tools(a2["gates"])
        a2["_observe_tools"] = observe_tools(a2["gates"])
        a2["_hints"] = tuple(set(DEFAULT_ARG_HINTS) | set(a2.get("identifying_arg_types") or ()))
        a2["_placeholders"] = set(a2.get("placeholders") or ()) | DEFAULT_PLACEHOLDERS
        a2["_producer"] = (a2.get("producers") or {}).get("authenticated_user_record")
        # ⚠ deprecated(NOTICE-PERGATE 2026-07-11): first-notice 스칼라 — 진단 스크립트
        #   호환용으로만 보존. ★신규 소비 금지 — 게이트 판정은 per-gate 커링(check callable)로.
        a2["_notice_text"] = next(
            (g.get("notice_text") for g in a2["gates"] if g.get("kind") == "notice"), "")
        # ★축-레버 문구는 **L1 base/shared.json**(도메인-일반·새 도메인 비용 0)에서 읽는다.
        #   도메인 A2에 같은 키가 있으면 그것이 우선(도메인 정련 여지·기본은 L1 그대로).
        if "axis_notes" not in a2:
            try:
                _bp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "a2", "base", "shared.json")
                with open(_bp, encoding="utf-8") as _bf:
                    a2["axis_notes"] = (json.load(_bf) or {}).get("axis_notes") or {}
            except Exception:
                a2["axis_notes"] = {}
    _A2_CACHE[domain] = a2
    return a2


def _tok_overlap(name, registry, stem=False):
    """이름 토큰이 겹치는 레지스트리 항목 (FIX-7·x298_ownership_deny_probe.py 판정 B_OWN 6/8).

    순수 문자열 연산이다 — 도메인 어휘 0·판단 0([[59]]). 접미 숫자를 뗀 뒤 언더스코어 토큰
    집합으로 비교하고, **겹침 수가 최대인 항목들만** 남긴다(동률은 전부 남긴다 — 엔진은
    고르지 않는다·[[62]] ③④). 겹침이 없으면 빈 목록 = 이 레버는 침묵한다(fail-open).

    `stem=True`(FIX-8b·claim 회수 경로 전용)면 굴절을 흡수한다(`_tok_match`) — 자유 문장인
    claim `what` 에는 'opening/accounts' 같은 굴절형이 실제로 나온다(t7279 075 turn14 실물).
    **기본 False = FIX-7 경로는 x298 로 측정한 그 판정 그대로**(미측정 변경 금지·[[03b]])."""
    import re as _re
    toks = [t for t in _re.split(r"[_\W]+", str(name).lower()) if t]
    if not toks:
        return []
    scored = []
    for n in (registry or ()):
        base = _re.sub(r"_\d+$", "", str(n)).lower().split("_")
        c = sum(1 for t in toks if (_tok_match(t, set(base)) if stem else t in base))
        if c:
            scored.append((c, str(n)))
    if not scored:
        return []
    mx = max(c for c, _ in scored)
    return sorted(n for c, n in scored if c == mx)


def _tok_match(tok, toks):
    """토큰 일치 — 정확 일치 ∨ **어간 접두**(길이 4+·굴절 관용·2026-08-13 FIX-8b).

    왜: t7279 075 turn14(첫 접힘) 실물에서 약속 문구가 *"...through **opening** Green
    Fee-Free Account"* 였고 정확-토큰 매치가 빗나가 소유권 회수가 실패했다 — 그 자리에 나간
    문구는 x300 에서 **0/8** 인 일반 촉구(D_GEN)였고, 두 턴 뒤 turn16 에 매치가 붙었을 땐
    이미 접힘이 누적돼(x299: 누적 후 0/8) 늦었다. 굴절만 흡수한다(판단·의미 0)."""
    if tok in toks:
        return True
    if len(tok) < 4:
        return False
    return any(t.startswith(tok) or tok.startswith(t) for t in toks if len(t) >= 4)


def _tok_hits(text, name):
    """이름 토큰 중 text 토큰집합에 든 개수 (FIX-8 문턱용·순수 문자열 연산·판단 0)."""
    import re as _re
    ts = {t for t in _re.split(r"[_\W]+", str(text or "").lower()) if t}
    base = _re.sub(r"_\d+$", "", str(name or "")).lower().split("_")
    return sum(1 for t in base if t and _tok_match(t, ts))


VERDICT_GATE_FB = (
    "Error: [VERDICT] '{val}' conflicts with what the customer asked for.\n"
    "{line}\n"
    "Options on file that do NOT conflict: {ok}.\n"
    "Pick one of those verbatim - or tell the customer that none of them fits and why - "
    "then call the tool again.")


def _verdict_gate_fb(agent, messages, a2, group, val, subs, spec):
    """★VC **호출-트리거** (`T2_VERDICT_GATE`·기본 OFF·C543ⓓ).

    설계 = `reports/facet_rft_2026/VERDICT_CALL_TRIGGER_DESIGN_2026_08_18.md`.

    ## 왜 이 자리인가
      push 형(`T2_VERDICT_CARRY`)은 **결정점에 닿기만 하면** 발화한다 — 고를 것이 없는 073 에서
      `후보 10·OK 10·VIOLATES 0` 의 무정보 판정을 내고 쓰기를 밀어냈다(ctl 1.0 ↔ vconly 0.0·C543ⓐ).
      범위를 조건으로 자르려던 세 갈래(LLM 라벨·스키마 술어·pending)는 전부 막혔고(C543ⓒ), 그
      이유는 구조적이다: A3 의 `applies_to`/`applies_when` 관용구는 **호출-트리거**인데 push 는
      호출 이전에 밀어넣는 레버라 그 관용구로 조건을 만들 수 없다. ⇒ 레버를 관용구가 사는
      자리로 옮긴다. **비-선택 태스크에는 트리거 자체가 없다**(073 은 이 호출을 내지 않는다).

    ## 계약 (엔진이 하는 일 전부)
      ⑴군은 호출부가 준다(`group_arg`→`group_map`·닫힌 사상) ⑵요구 인용 = LLM, 엔진은
      `quote_in` 존재확인만(C45 동형) ⑶후보별 판정 = LLM(`verdict_lines`) ⑷엔진은 **제출값의
      판정을 조회**할 뿐이다(슬러그 키·문자열 파싱 0·[[59]]) ⑸거부 문면 = **LLM 이 쓴 줄 축자**
      + 충돌하지 않는 후보 명단([[64]] 무엇이 틀렸나 + 무엇을 하면 풀리나).
      ⚠엔진은 고르지 않고 후보를 제거하지도 않는다([[62]] ③④).

    ## fail-safe (모르면 막지 않는다·[[25]])
      템플릿 미선언 · 코퍼스 부재 · 요구 인용 0 · 판정 없음 · `UNCLEAR` · **근거 미검산** ·
      대안 0 → 전부 **침묵**(None) = OFF 와 바이트 동일. skip 사유는 stderr 로 남긴다(死배선 탐지).

    반환: 거부 문면 | None
    """
    try:
        import t2_search as _ts
        import tau2.agent.llm_agent as _la_v
        from tau2.data_model.message import UserMessage as _UM_v
    except Exception as _ie:
        print("[T2_VERDICT_GATE] skip=import %r" % (_ie,), file=sys.stderr, flush=True)
        return None
    _po = (a2 or {}).get("policy_ontology") or {}
    if not (_po.get("verdict_prompt") and group and val and subs):
        print("[T2_VERDICT_GATE] skip=undeclared group=%s" % (group,),
              file=sys.stderr, flush=True)
        return None
    _env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
    _corpus = _ts.corpus_from_env(_env)
    if not _corpus:
        print("[T2_VERDICT_GATE] skip=no-corpus", file=sys.stderr, flush=True)
        return None
    _utxt = "\n\n".join(_content_str(m) for m in (messages or [])
                        if getattr(m, "role", None) == "user")
    _reqs = getattr(agent, "_t2_vg_reqs", None)
    if _reqs is None:
        _reqs = []
        for _q in (_ts.sub_requirements(agent, _la_v, _UM_v, _po, _utxt) or []):
            _qs = str(_q).strip()
            if _qs and _ts.quote_in(_qs, _utxt):      # ★존재확인만 (추출 0·[[59]])
                _reqs.append(_qs)
        try:
            agent._t2_vg_reqs = _reqs
        except Exception:
            pass
    if not _reqs:
        print("[T2_VERDICT_GATE] skip=no-requirement", file=sys.stderr, flush=True)
        return None
    _cache = dict(getattr(agent, "_t2_vg_by", None) or {})
    if group in _cache:
        _by = _cache[group]
    else:
        _blk = "Customer's stated request:\n" + "\n".join("- " + q for q in _reqs)
        _lines, _st = _ts.verdict_lines(agent, _la_v, _UM_v, _po, _blk, group, corpus=_corpus)
        _by = (_st or {}).get("by_name") or {}
        _cache[group] = _by
        try:
            agent._t2_vg_by = _cache
        except Exception:
            pass
        if _lines:                     # ★감사(C508⒥ 규약): 판정 줄을 축자로 남긴다
            try:
                import t2_fbsidecar as _fbv
                _fbv.record("verdict-gate", "\n".join(_lines), messages,
                            channel="verdict", group=group, stats=_st)
            except Exception:
                pass
    if not _by:
        print("[T2_VERDICT_GATE] skip=no-verdict group=%s" % (group,),
              file=sys.stderr, flush=True)
        return None
    _slug = next((k for k in subs if _slug_disp(k) == str(val).strip()), None)
    _rec = _by.get(_slug) if _slug else None
    if not _rec:
        print("[T2_VERDICT_GATE] skip=unjudged val=%r group=%s" % (val, group),
              file=sys.stderr, flush=True)
        return None
    if _rec.get("verdict") != "VIOLATES" or not _rec.get("cited"):
        print("[T2_VERDICT_GATE] pass val=%r verdict=%s cited=%s"
              % (val, _rec.get("verdict"), _rec.get("cited")), file=sys.stderr, flush=True)
        return None
    _ok = sorted(_slug_disp(k) for k, r in _by.items() if r.get("verdict") == "OK")
    if not _ok:
        # 충돌하지 않는 후보가 하나도 없으면 **무엇을 하면 풀리는지 말할 수 없다** ⇒ 침묵.
        # 이름 없는 거부가 창 순환을 만든다는 것은 C536ⓑ 에서 이미 샀다([[64]]).
        print("[T2_VERDICT_GATE] skip=no-alternative val=%r" % (val,),
              file=sys.stderr, flush=True)
        return None
    # ★줄은 **정본 표기로 재조립**한다 (2026-08-18 검정에서 잡힘). `verdict_lines` 가 만든
    #   줄은 `t2_search._disp_name`(하이픈 뒤 소문자) 표기라 'Green Fee-free Account' 인데,
    #   같은 문면의 제출값·대안 명단은 `_slug_disp` 표기('Green Fee-Free Account') 다 —
    #   한 메시지 안에 같은 상품의 두 철자가 섞이면 우리 도구가 오표기를 가르치는 셈이고,
    #   그것이 FIX-6 에서 실제로 채점 칸을 죽인 그 결함이다([[25]] 우리 출력은 100% 정확).
    #   판정·인용은 **LLM 이 쓴 그대로**이고 바뀌는 것은 이름 표기뿐이다(push 경로 불변).
    _line = "- %s: %s%s" % (_slug_disp(_slug), _rec.get("verdict"),
                            (" - " + str(_rec.get("why"))) if _rec.get("why") else "")
    print("[T2_VERDICT_GATE] deny val=%r group=%s (대안 %d)" % (val, group, len(_ok)),
          file=sys.stderr, flush=True)
    return str((spec or {}).get("verdict_gate_feedback") or VERDICT_GATE_FB).format(
        val=val, line=_line, ok=", ".join(_ok))


def _slug_disp(k):
    """슬러그 → 표시명 기계 전개 (FIX-6·t7276 075 실측·[[55]] 우리층 수리).

    구판 두 사이트의 `w.capitalize()` 는 하이픈 구간을 대문자화하지 않아
    'green_fee-free_account' → **'Green Fee-free Account'(오표기)** 를 만들었다 —
    그 결과 WRITE_ARG_ENUM 소속 검사가 모델의 오표기 제출("Fee-free")과 **일치해
    조용히 통과**(deny 미발화·gold "Fee-Free" 와 불일치·reward 0). env 문서 title
    ('Green Fee-Free Account: …') 이 하이픈 대문자화가 정본임을 기계 검증한다.
    판단 0·출처 = env 파일명뿐(종전과 동일)."""
    return " ".join("-".join(s[:1].upper() + s[1:] for s in str(w).split("-"))
                    for w in str(k).split("_"))


def _subject_keys(subs):
    """A3 `doc_index[군]` 에서 **대상 계열 키**만 — `_general_` 류 색인-규약 키 제외 (닫힌 술어).

    ★2026-08-22 통합([[67]] 사본 금지): *"`_general_` 은 대상 계열이 아니다"* 라는 **한 사실**이
      네 자리에 흩어져 있었다 — `_degenerate_axes`(퇴화 군) · `_served_subjects`(배달 계열) ·
      `_rearm_subjects`(표시명 색인) · 그리고 `T2_WRITE_ARG_ENUM` 의 후보 명단(누수 수리에서
      새로 필요해진 자리). 앞의 셋은 `"_general_"` **문자열 리터럴**을 각자 들고 있었고, 네
      번째를 또 만들면 넷이 조용히 갈린다. ⇒ 술어를 **하나**로 두고 넷이 공유한다.

    술어 = **전개했을 때 빈 토막이 생기는 슬러그는 대상 계열이 아니다**(`_general_`.split('_')
    = ['', 'general', ''] → 탈락 · `green_fee-free_account` → 통과). 형상 판정이므로 이름
    리터럴이 **0** 이고, 같은 규약의 다른 키(`_faq_` 등)도 함께 걸린다.
    ⚠실물 A2 전수에서 구판 리터럴 술어와 **동치**임을 검정이 못박는다
      (`test_t7337_residual_debt.py` — banking gate/specific 9군 전수).
    """
    return {k for k in (subs or {}) if str(k) and all(w for w in str(k).split("_"))}


def _display_slugs(subs):
    """doc_index 그룹의 키 → **표시명 후보 명단** (닫힌 술어·형상 판정·도메인 리터럴 0).

    ★누수 수리 (2026-08-22 · t7336 마스터 잔여): doc_index 키는 문서 파일명 유도 슬러그다.
      대부분 제품 공식명으로 전개되지만 `_general_` 처럼 **앞뒤가 언더스코어로 감싸인**
      그룹-일반 문서 키가 섞여 있다(banking gate/specific 실측 3개 그룹: `checking_accounts`
      · `credit_cards` · `bank_accounts_bank_accounts`). `_slug_disp('_general_')` 은
      **`' General '`**(앞뒤 공백)을 내고, 그 문자열이 `T2_WRITE_ARG_ENUM` 의 deny 피드백에
      **공식 명단의 한 항목으로 실려 나갔다** — 우리 도구 출력이 유일 근거원을 오염시키는
      자리다([[25]]). 존재하지 않는 이름을 우리가 후보로 제시하면 [[64]] 의 *"무엇을 하면
      풀리나"* 가 거짓이 된다.

    술어 = **전개했을 때 빈 토막이 생기는 슬러그는 표시명이 아니다**. 이름 리터럴 0 —
    `_general_` 을 엔진에 박지 않는다(같은 형상의 다른 키도 함께 걸린다).
    """
    return sorted(_slug_disp(k) for k in _subject_keys(subs))


def _enum_seen_key(group, val):
    """★거절 원장 키 (2026-08-24·R4 수리) — **우리 게이트가 이미 집합 밖이라고 판정한 값**을
    기억하기 위한 닫힌 문자열 술어. 판단 0 · 도메인 리터럴 0 · 선택 0.

    왜 `(그룹, 값)` 쌍인가: *"집합 밖"* 이라는 판정은 **그 그룹의 명단에 대해서만** 참이다.
    같은 문자열이 다른 그룹에서는 집합 內일 수 있으므로 그룹을 벗겨서 기억하면 안 된다.

    정규화 = 공백 접기 + `casefold` 뿐이다(둘 다 형상 판정·[[22]] 닫힘). 대소문자·여분
    공백만 바꾼 재제출은 **같은 값**이다 — 집합 소속 검사(`_val in _names`)는 축자 비교라
    그 변형들은 어차피 집합 밖이고, 그래서 원장이 넓게 잡아도 집합 內 값을 막지 않는다.
    """
    return (str(group or ""), " ".join(str(val or "").split()).casefold())


def _flatten(v):
    """인자값의 leaf 스칼라들. ★JSON-문자열도 풀어서 leaf까지 간다(2026-07-16 버그픽스):
    구조화 인자(예: {"date_of_birth": ..., "phone_number": ...})가 **문자열**로 오면 예전엔
    JSON 덩어리 전체를 문맥서 찾아 **항상 실패 → 전부 '날조' 오판**했다(라이브 실측: 우리
    verify_identity의 정당한 호출이 매번 반려됨). leaf(11/03/1990·312-555-0481)는 문맥에 실재한다."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{" and s[-1:] in "]}":
            try:
                yield from _flatten(json.loads(s))
                return
            except Exception:
                pass
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _flatten(x)
    else:
        yield v


def _hint_hit(k, hints):
    """인자명이 식별자류인가. ★부분문자열 금지(2026-07-16 버그픽스): `"id" in "provided"` = True라
    `provided`가 식별자로 오판됐다. 토큰 분해 후 **토큰이 힌트로 시작**할 때만(→ `address1`·`item_ids`
    같은 접미변형은 유지, `provided`·`valid_until`류 오탐은 제거)."""
    toks = [t for t in re.split(r"[^a-z0-9]+", str(k).lower()) if t]
    return any(t.startswith(h) for t in toks for h in hints)


_UNKNOWN_RE = re.compile(r"Unknown (agent|discoverable) tool '([^']+)'")


def unknown_bl_collect(messages):
    """env가 거부한 이름을 **채널과 함께** 수집한다 (2026-07-31 교정).

    ★고친 버그(Y2-B 실측): 구판은 이름만 모아서 채널을 무시했다. task_017에서 실제로 일어난 일 —
      ①모델이 user 도구를 agent 채널로 unlock 시도 → env `Unknown **agent** tool 'X'`
      ②블랙리스트에 이름 X만 등록
      ③모델이 **올바른 채널**로 gold 액션 `give_discoverable_user_tool(X)` → **차단**(18회)
      ⇒ 우리 스캐폴드가 정답을 막았다. Y1에서도 46회 발화한 선재 버그.
    같은 접미사 이름이 채널마다 유효/무효가 갈리므로, 거부는 **거부가 난 그 채널에서만** 유효하다.

    반환 (blocked, kind_by_tool):
      blocked      = {(kind, name)}          — kind = env가 말한 'agent' | 'discoverable'
      kind_by_tool = {호출도구명: kind}       — 그 도구가 어느 채널인지 **관측으로** 확정(리터럴 0)
    """
    call_tool = {}
    for m in messages:
        if getattr(m, "role", None) == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                call_tool[getattr(tc, "id", None)] = getattr(tc, "name", None)
    blocked, kind_by_tool = set(), {}
    for m in messages:
        if getattr(m, "role", None) != "tool":
            continue
        for mt in _UNKNOWN_RE.finditer(str(getattr(m, "content", "") or "")):
            kind, name = mt.group(1), mt.group(2)
            blocked.add((kind, name))
            src = call_tool.get(getattr(m, "id", None))
            if src:
                kind_by_tool.setdefault(src, kind)
    return blocked, kind_by_tool


def unknown_bl_hit(blocked, kind_by_tool, tool_name, value):
    """이 호출이 **같은 채널에서** 이미 거부된 이름인가.

    채널을 모르면(그 도구로 거부가 난 적 없음) **막지 않는다** — 넓게 막다 gold를 막는 것보다
    좁게 막고 env가 한 번 더 거부하게 두는 편이 낫다(그때 채널이 확정된다)."""
    if not value:
        return False
    kind = kind_by_tool.get(tool_name)
    return bool(kind) and (kind, value) in blocked


def _args_dict(tc):
    """ToolCall.arguments 를 dict로 (string JSON도 robust 파싱)."""
    a = getattr(tc, "arguments", None)
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return {}


# ─── ★L1 bad_words 블랙리스트 (스키마-example + placeholder; 동적=세션-flagged) ───
_SUCH_AS_RE = re.compile(r"such as ['\"]([^'\"]+)['\"]|e\.g\.,?\s*['\"]?([^'\".)]+)", re.I)


def _blacklist_worthy(v):
    """generic 단어(John·CA·12345) 차단 회피: len>=6 & (이메일/숫자 포함) = ID/placeholder형만."""
    v = (v or "").strip()
    return len(v) >= 6 and (("@" in v) or any(c.isdigit() for c in v))


def _static_blacklist(tools, placeholders=None):
    bl = set(placeholders or DEFAULT_PLACEHOLDERS)
    for t in (tools or []):
        try:
            txt = json.dumps(t.openai_schema)
        except Exception:
            txt = str(getattr(t, "description", "") or "")
        for m in _SUCH_AS_RE.finditer(txt):
            v = next((g for g in m.groups() if g), "").strip()
            if _blacklist_worthy(v):
                bl.add(v)
    return bl


def _context_text(orch):
    """provenance 출처 = 모든 user 발화 + 도구 출력(assistant 제외)."""
    parts = []
    try:
        for m in orch.get_messages():
            r = getattr(m, "role", None)
            c = getattr(m, "content", None)
            if r in ("user", "tool") and c is not None:
                parts.append(c if isinstance(c, str) else str(c))
    except Exception:
        pass
    return " ".join(parts).lower()


def _ctx_with_toolnames(agent, ctx):
    """provenance ctx에 **에이전트에게 제시된 도구 이름**을 추가 (§15 오탐 수정·2026-07-16).
    도구 이름의 출처는 스키마(모델에게 제시됨)이지 대화가 아니다 — 라이브 실측: 에이전트가
    `unlock_discoverable_agent_tool(agent_tool_name="get_reward_discrepancies")`로 우리 도구를
    호출하려 하자 PROV가 "문맥에 없음=invented"로 11회 반려(ctl_20260716_2230). ★이름만 추가한다 —
    스키마 전체(설명·예시값)를 넣으면 C47(예시값=복사 원천 47%)의 날조 재료까지 정당화된다.

    ★2026-08-03 확장(라이브 실측·task_002): 같은 오탐이 **discoverable 도구 이름**에서 재발했다 —
    `give_discoverable_user_tool(discoverable_tool_name="apply_for_credit_card")`가 5회 연속 반려돼
    9분을 태웠다(에이전트가 `list_discoverable_*`를 부르지 않아 그 이름이 대화 어디에도 없었다).
    discoverable 이름의 출처도 **스키마가 아니라 env 레지스트리**이고 그 조회는 **닫혀 있다**
    (`t2_axis_levers.registry_from_env` = DISCOVERABLE_ATTR 기계 도출·opex 0·[[05]]). ⇒ 그 이름들도
    ctx에 넣는다. **실재하지 않는 이름은 여전히 반려된다**(날조 차단력 불변·오탐만 제거)."""
    try:
        names = " ".join(sorted({str(getattr(t, "name", "") or "").lower()
                                 for t in (getattr(agent, "tools", None) or [])}))
        try:
            # ★2026-08-03 rev2 (라이브 재확인): 첫 판은 `registry_from_env`(DISCOVERABLE_ATTR)만
            #   넣었는데 **그 집합이 아니었다** — `apply_for_credit_card`는 env의 **user 도구**
            #   (`get_user_tools()` 6종)이고 discoverable 레지스트리(deposit_check_3847 등)와 별개다.
            #   ⇒ env가 아는 도구 이름 **전부**(agent-side + user-side)를 넣는다. 전부 env 기계
            #   도출이라 opex 0·도메인 리터럴 0([[05]])이고, 없는 이름은 여전히 반려된다.
            _env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
            _envn = set()
            for _m in ("get_tools", "get_user_tools"):
                _f = getattr(_env, _m, None)
                try:
                    for _t in (_f() or []) if callable(_f) else []:
                        _n = str(getattr(_t, "name", "") or "").lower()
                        if _n:
                            _envn.add(_n)
                except Exception:
                    pass
            try:                               # discoverable 레지스트리도 함께(별도 집합)
                import t2_axis_levers as _AXn
                _o = getattr(agent, "_t2_orch", None)
                if _o is not None:
                    _a2s, _u2s = _AXn.registry_from_env(_o)
                    _envn |= {str(x).lower() for x in (_a2s | _u2s) if x}
            except Exception:
                pass
            if _envn:
                _j = " ".join(sorted(_envn))
                names = (names + " " + _j) if names else _j
        except Exception:
            pass
        return ctx + " " + names if names else ctx
    except Exception:
        return ctx


def _quote_tokens(s):
    """인용 실재성 검사용 토큰화 — **닫힌 연산만**(소문자·영숫자 외 제거·공백 분할).
    `gate_interpreter.notice_norm`과 동일 철학(유사도·의미 매칭 금지)."""
    import re as _re
    return _re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).split()


def _shared_span(text, source, min_tokens=4):
    """★P1 (2026-08-03·AX32 설계서 §P1): `text` 안에 `source`에서 **연속으로 복사된** 토큰 열이
    min_tokens 이상 존재하는가. 판정은 토큰-연속 부분열의 **실재**뿐(C289 인용-핀과 동형·[[22]]
    닫힌 술어) — 유사도·의미 대조 0, 도메인 리터럴 0. 발화 실재만 보고 '요청했는가'의 의미는
    모델 몫으로 남긴다."""
    n = max(1, int(min_tokens or 1))
    a, b = _quote_tokens(text), _quote_tokens(source)
    if len(a) < n or len(b) < n:
        return False
    grams = {" ".join(b[i:i + n]) for i in range(len(b) - n + 1)}
    return any(" ".join(a[i:i + n]) in grams for i in range(len(a) - n + 1))


_KB_SCORE_RE = re.compile(r"^\s*Score:\s*([0-9]*\.?[0-9]+)\s*$", re.M)


def _kb_zero_hit(text):
    """★P2/P10 신호 (2026-08-03·alltools 실측): 검색 출력의 **env 기계 포맷** `Score: <float>` 행을
    읽어 "반환 문서가 전부 0점"인지 본다 = 그 질의가 어휘적으로 아무것도 맞히지 못했다.
    [[03b]] 경계: `_parse_record_dump`와 동일 — env가 찍는 **고정 포맷의 전사**이지 NL formalize가
    아니다. 점수 행이 하나도 없으면 판정 불가(None) — 다른 채널(dense·shell) 출력을 오판하지 않는다.
    ★실측 근거(2026-08-03 env 프로브): alltools에서도 `KB_search_bm25`는 그대로 노출되고 무의미
    질의에 `Score: 0.0000`을 찍는다(검색은 **공집합을 반환하지 않는다** — 그래서 '결과 없음'이 아니라
    '전부 0점'이 신호다). dense는 항상 양수 유사도를 주므로 문턱이 필요해 **신호로 쓰지 않는다**
    (임계값=도메인 튜닝=[[05]] 회색지대)."""
    if not isinstance(text, str):
        return None
    vals = [float(m.group(1)) for m in _KB_SCORE_RE.finditer(text)]
    if not vals:
        return None
    return all(v == 0.0 for v in vals)


def _ctx_has(s, ctx):
    """값 s의 ctx-매칭 (PROV-RESCUE-PERARG ②: id '#'-접두 정규화).
    '#W8665881' vs ctx의 'w8665881'(사용자 발화) = 접두 불일치 거짓양성 fab(t17 1차 방아쇠) →
    '#' 접두만 벗겨 재매칭. 정규화는 '#' 하나에 한정(과잉 정규화 금지)·strip 후에도 4자 이상일 때만."""
    if s.lower() in ctx:
        return True
    t = s.lstrip("#")
    return t != s and len(t) >= 4 and t.lower() in ctx


# ★거절 문면 — **도메인 낱말 0**. 종전에는 A2 에 도메인별로 적혀 있었는데(`type_feedback`·
#   `feedback`), 문장 자체는 어느 도메인에서도 같으므로 고정층이 옳은 집이다([[05]]).
#   내용은 오늘 A2 에서 쓰던 것 축자 그대로다(문면 변경 0 — 바뀐 것은 **사는 곳**뿐).
_SPEC_TYPE_FB = ("Error: `%s` — this tool declares these arguments as booleans. A quoted word is "
                 "not a boolean value. Send the same answers as `true` or `false` and leave every "
                 "other argument unchanged.")
_SPEC_ENUM_FB = ("Error: `%s` is not one of the values `%s` accepts. Use exactly one of these, "
                 "copied verbatim: %s.")

_DECL_PARAM_RE = re.compile(
    r"^\s*-\s*(\w+):\s*(\w+)\s*\((required|optional)\)\s*-\s*(.*)$", re.M)
_DECL_TOOL_RE = re.compile(r"^Tool:\s*(\S+)\s*$", re.M)
# env 는 unlock 응답에서 `Tool unlocked: X` 를 먼저 찍고 그 아래 `Tool: X` 블록을 붙인다.
# 둘째 줄이 없는 변형(도구를 손님에게 넘길 때)도 있으므로 첫 줄을 폴백으로 받는다.
_DECL_TOOL_ALT_RE = re.compile(r"^Tool unlocked:\s*(\S+)\s*$", re.M)
_DECL_ONEOF_RE = re.compile(r"Must be one of:\s*(.+)$")
_DECL_QUOTED_RE = re.compile(r"'([^']+)'")


def _declared_params_by_tool(messages):
    """env 명세를 **도구별로** 읽는다 — {도구: {인자: (타입, [열거값…])}}.

    ★왜 도구별인가 (2026-08-25·필수): 같은 인자 이름이 도구마다 **다른 값 집합**을 갖는다.
      실물 — `card_action` 은 신용 분쟁에서 {keep_active, cancel_and_reissue} 이고
      직불 분쟁에서 {keep_active, freeze_pending_investigation, close_and_reissue} 다.
      이름만으로 합치면 **틀린 명단**으로 정당한 값을 거절한다([[25]] 계기는 100% 정답 의무).
    ⚠열거 **값**은 설명문 안에 있다(`Must be one of: '…', '…'`). 타입 세 토막은 env 가 기계
      생성하지만 이 줄은 독스트링 본문이므로, 값 추출은 **작은따옴표 안 토큰**이라는 한 가지
      규칙만 쓰고 그 밖의 해석을 하지 않는다. 등가성은 `x540_spec_derivation.py` 가 코퍼스
      실물로 쟀다: 손 선언 9건 ↔ 도출 9건 **전부 일치 · 다르다 0 · 대조 불가 0**.
    ⚠형식이 아니면 아무것도 돌려주지 않는다(fail-open).
    """
    out = {}
    for m in (messages or []):
        c = str(getattr(m, "content", "") or "")
        if "Parameters:" not in c:
            continue
        tm = _DECL_TOOL_RE.search(c) or _DECL_TOOL_ALT_RE.search(c)
        if not tm:
            continue
        d = out.setdefault(tm.group(1), {})
        for name, typ, _req, desc in _DECL_PARAM_RE.findall(c):
            hit = _DECL_ONEOF_RE.search(desc)
            vals = _DECL_QUOTED_RE.findall(hit.group(1)) if hit else []
            prev = d.get(name)
            if prev and prev[1] and not vals:
                continue                    # 값을 이미 본 칸을 빈 것으로 덮지 않는다
            d[name] = (typ, vals)
    return out


def _declared_params_for(messages, tc):
    """이 호출이 **실행하는 그 도구**의 명세만. 이름이 안 맞으면 빈 dict(fail-open)."""
    want = str(_exact_tool_name(tc) or "")
    return _declared_params_by_tool(messages).get(want) or {}


def _declared_params(messages):
    """env 가 unlock 시점에 찍는 **고정 포맷 명세**에서 (이름 → (타입, 열거값 리스트)) 를 읽는다.

    ⚠**도구를 가로질러 합친 판**이다 — 값 집합이 도구마다 다른 인자에는 쓰지 마라.
      값을 보는 자리는 `_declared_params_for` 를 쓴다. 이 함수는 *타입* 처럼 도구가 달라도
      같은 축에만 안전하다(2026-08-25 `card_action` 실물로 배운 것).

    ★2026-08-25 신설. 왜 (사용자 지적): 우리는 *"이 인자가 식별자처럼 생겼나"* 를 **이름 패턴**으로
      추측해 왔다(`identifying_arg_types`·`_hint_hit`). 추측할 이유가 없다 — env 가 같은 것을
      **선언해서 건네준다**. 라이브 축자(t7354 grpB1 task_040 msg5, env tool 메시지):

        Parameters:
          - transaction_id: string (required) - The unique identifier for the transaction …
          - contacted_merchant: boolean (required) - Whether the user attempted …
          - dispute_reason: string (required) - … Must be one of: 'unauthorized_fraudulent_charge', …

    ⚠[[59]] 경계: 이것은 NL formalize 가 아니라 **env 가 찍는 고정 포맷의 전사**다 —
      `_parse_record_dump`(`Record ID:` 덤프 전용) 와 같은 층이고 같은 규율을 진다:
      형식이 아니면 **아무것도 돌려주지 않는다**(fail-open). 술어는 세 토막
      (`이름`·`타입`·`(required|optional)`) 뿐이고, 열거 값은 설명문의
      `Must be one of: '…'` **작은따옴표 안 토큰** 한 규칙으로만 읽는다.
    ⚠엔진은 여전히 고르지 않는다: 여기서 나오는 것은 **타입·명단 사실**이고 어느 값을 쓸지는
      모델이 정한다([[62]]③④).
    """
    out = {}
    for tool, d in _declared_params_by_tool(messages).items():
        for name, fact in d.items():
            prev = out.get(name)
            if prev and prev[1] and not fact[1]:
                continue
            out[name] = fact
    return out


_A3_CACHE = {}


def _policy_facts(a2):
    """A3 정본(`policy_facts_file`)의 행. 파일 하나가 데이터 정본이고 A2 두 층엔 포인터만 산다.

    ★2026-08-25 신설(엔진의 첫 A3 소비자). 형식 = {subject, axis, value, sources:[{doc, quote}]}.
      행 2,226 · 축 1,092. 저작 근거는 그 파일 `_note_` 축자에 있다(x453 전수 감사 → x462 접기-안전
      군 → x457 v2 · 값·인용 전부 문서 축자 · gold·tasks 미참조·[[23]]).
    ⚠읽기만 한다 — 고르지도 순위 매기지도 않는다. 캐시는 파일 경로 기준.
    """
    fn = str((a2 or {}).get("policy_facts_file") or "")
    if not fn:
        return []
    if fn in _A3_CACHE:
        return _A3_CACHE[fn]
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", fn)
    try:
        with io.open(p, encoding="utf-8") as f:
            rows = (json.load(f) or {}).get("rows") or []
    except Exception:
        rows = []
    _A3_CACHE[fn] = rows
    return rows


def _policy_rows_for(a2, arg_names):
    """이 write 가 **선언한 인자 이름**과 A3 행의 `axis` 가 **같은** 행만 — 검색 0·순위 0·유사도 0.

    ★왜 이 모양인가 (2026-08-25·사용자 물음에 대한 답의 일반형): `write_rules` 는 *어느 정책
      문장을 결정점에 놓을지*를 실패를 보고 손으로 골랐다 — 그것이 오늘 남은 유일한 케이스별
      저작이었다. 그런데 A3 의 `axis` 는 **인자 이름 그대로**다(실측: `contacted_merchant` ·
      `police_report_filed` · `eligible_for_provisional_credit` …). 그래서 조인은 유사도가 아니라
      **동일성**이면 된다 — 어제 폐기한 토큰 검색기와 다른 종류다([[71]]③ bm25·embedding 금지).
      커버리지 실측(t7354 명세): 신용 분쟁 인자 15 중 **13**, 직불 17 중 9 에 행이 붙는다.
    ⚠순서는 결정론(축 이름 → 문서 id) — 점수가 아니다. 상한을 넘으면 **아무것도 주지 않는다**
      (자르면 무엇을 뺐는지 우리가 고른 것이 된다·[[62]]④).
    ⚠인용은 **축자**만 싣는다(`sources[].quote`). 요약·재서술 0.
    """
    want = set(arg_names or ())
    if not want:
        return None
    seen, out = set(), []
    for r in _policy_facts(a2):
        ax = str(r.get("axis") or "")
        if ax not in want:
            continue
        for s in (r.get("sources") or []):
            q = str(s.get("quote") or "").strip()
            if not q or (ax, q) in seen:
                continue
            seen.add((ax, q))
            out.append((ax, str(s.get("doc") or ""), q))
    if not out:
        return None
    out.sort(key=lambda x: (x[0], x[1]))
    txt = chr(10).join("- %s: %s" % (a, q) for a, _d, q in out)
    cap = int(os.environ.get("T2_ARG_POLICY_CAP", "4000"))
    return None if len(txt) > cap else txt


_A3_SUBJDOC_CACHE = {}


def _a3_subject_docs(a2):
    """주어 → **선언된 근거 문서 집합**. `policy_facts` + `policy_ontology.rows` 합본. 읽기만.

    ★A3 의 **두 번째 소비자**를 위한 재료 (2026-08-30 · 사용자 지시 *"부족한데 멈추거나,
      만족했는데 계속 찾는 경우를 없애는 최적의 방식을 지금 alltools 에 접목하라"*).
      첫 소비자(`_arg_policy_join`)는 **write 인자 이름 == 축 이름** 으로만 걸려서, 063 처럼
      결정이 **산문**인 자리(어느 저축+카드 조합이 이자를 최대화하나)에는 원리상 안 닿는다 —
      `apy` 27행이 선언돼 있어도 `apy` 라는 이름의 write 인자가 없기 때문이다.

    실측 규모(2026-08-30): 주어 **112** · 주어당 축 평균 17 · **주어당 문서 평균 5.9 · 중앙 7**.
      축은 많아도 문서는 적으므로 검산 단위는 **문서**로 잡는다(발화가 1~3줄로 끝난다).

    ⚠읽기만 한다 — 고르지도 순위 매기지도 않는다([[59]]·[[62]]). 값·인용은 전부 문서 축자이고
      저작 근거는 `policy_facts` 의 `_note_` 에 있다(gold·tasks 미참조·[[23]]).
    """
    key = id(a2)
    if key in _A3_SUBJDOC_CACHE:
        return _A3_SUBJDOC_CACHE[key]
    out = collections.defaultdict(set)
    try:
        for r in _policy_facts(a2):
            s = r.get("subject")
            for src in (r.get("sources") or []):
                d = src.get("doc")
                if s and d:
                    out[s].add(d)
        # ★주어 키 정규화 (2026-08-30 · 배선 차단 결함 수리). 두 저장소의 표기가 다르다 —
        #   `policy_ontology` 는 **표시명**(`Business Silver Rewards Card`),
        #   `policy_facts` 는 **군 접두 슬러그**(`business_credit_cards_business_silver_…`).
        #   병합하지 않으면 같은 상품에 대해 *확보 완료* 와 *11개 미달* 을 **동시에** 말한다
        #   ([[25]] 우리 출력 100% 정답 의무 위반).
        #
        #   ⚠규칙은 **이름 문자열로 추측하지 않는다**([[23]]) — *ontology 주어의 선언 문서가
        #   어느 facts 주어의 문서 집합에 포함되는가* 로 정한다. 관측된 포함 관계다.
        #   ⚠**포함하는 facts 주어가 유일할 때만** 병합한다. 둘 이상이면 어느 쪽인지 우리가
        #   고르는 것이 되므로 그대로 둔다([[62]]④).
        #   실측(`x641`/`x641b`): ontology 주어 41 중 **유일 35 · 모호 0 · 대응 없음 6**.
        #   대응 없는 6 은 전부 상태값(APPLIED·COMPLETE·ERROR·IN_PROGRESS·NO_PROGRESS·REJECTED)
        #   이라 제품 주어가 아니다 — 대응이 없는 것이 맞다.
        #   하드코딩된 사상표를 두지 않으므로 선언이 바뀌면 규칙이 따라온다(도메인 리터럴 0).
        _fnorm = {k: {_a3_norm_doc(d) for d in v} for k, v in out.items()}
        for r in (((a2 or {}).get("policy_ontology") or {}).get("rows") or []):
            s = r.get("subject")
            d = (r.get("source") or {}).get("doc")
            if not (s and d):
                continue
            if s not in out:
                holders = [k for k, fd in _fnorm.items() if fd and _a3_norm_doc(d) in fd]
                if len(holders) == 1:
                    s = holders[0]
            out[s].add(d)
    except Exception:
        pass
    out = dict(out)
    _A3_SUBJDOC_CACHE[key] = out
    return out


_A3_DOCPAT = re.compile(r"doc_[a-z0-9_()\-]{6,90}", re.I)


def _a3_norm_doc(s):
    """샌드박스 파일명 규약(`(general)` ↔ `__general__`)을 흡수해 맞댄다. 형상 하나·해석 0."""
    s = str(s or "").lower().replace(".md", "").replace("(", "_").replace(")", "_")
    return re.sub(r"_+", "_", s).strip("_")


def _closure_note(agent, a2, content):
    """★A3 **완결 검산** — 이 결과가 건드린 주어에 대해 *선언된 근거 문서 중 아직 없는 것*.

    사용자 설계 축자: *"특정 기능이나 판단에 필요한 모든 문서가 **완결되었다는 걸 검산**하는 거다"*.
    양 끝을 하나가 닫는다 — **부족한데 멈추는 것**(미달 목록이 뜬다) ·
    **만족했는데 계속 찾는 것**(확보 완료가 뜬다).

    술어는 전부 닫혔다([[22]]): 문서 id **원소 검사** + 집합 차. 엔진은 무엇이 정답인지 모르고
    어느 문서가 더 중요한지도 모른다. 순서는 결정론(주어명 → 문서 id)이지 점수가 아니다.

    ⚠**도구 무관** — bm25·dense·shell 어느 결과든 doc id 가 실리면 걸린다. 사용자 지적
      *"수동적으로 shell 이 불릴 때만 하면 **빠져 나갈 구멍**이 생긴다. 100% 를 원한다."*
    ⚠**reads 에만 붙는다** ⇒ replay-safe — DB 해시가 안 바뀌므로 reward 기전 불변([[69]]).
    ⚠상한을 넘으면 **아무것도 주지 않는다**(자르면 무엇을 뺐는지 우리가 고른 것이 된다·[[62]]④).
      같은 규약이 `_arg_policy_join` 에 이미 있다([[67]] 사본 금지).
    ⚠INDEX.md 전량 덤프(문서 60개 이상)는 *본 것* 으로 세지 않는다 — 목록을 본 것이지 내용이 아니다.
    """
    if os.environ.get("T2_SEARCH_CLOSURE") != "1" or a2 is None:
        return None
    subj_docs = _a3_subject_docs(a2)
    if not subj_docs:
        return None
    hits = {_a3_norm_doc(h) for h in _A3_DOCPAT.findall(str(content or ""))}
    if not hits or len(hits) >= 60:
        return None
    seen = getattr(agent, "_t2_a3_seen_docs", None)
    if seen is None:
        seen = set()
    seen |= hits
    try:
        agent._t2_a3_seen_docs = seen
    except Exception:
        pass
    lines = []
    for s in sorted(subj_docs):
        decl = {_a3_norm_doc(d) for d in subj_docs[s]}
        if not (decl & hits):                 # 이번 결과가 건드린 주어만
            continue
        miss = sorted(decl - seen)
        # ★id 목록을 싣지 않는다 (2026-08-30 계측이 잡은 내 위반). 구판은 `miss[:4]` 로 **넷을
        #   골라** 적었는데, 그건 이 함수가 지키려던 규약 자체를 어긴 것이다 - *자르면 무엇을
        #   뺐는지 우리가 고른 것이 된다*([[62]]④). 그리고 긴 id 가 줄당 200 B 를 먹어
        #   **도구 메시지 19개 중 18개가 상한(1200 B)에 걸려 침묵**했다(실측 2,500~3,700 B).
        #   개수만 말하면 줄이 ~60 B 라 8주어도 500 B 안에 들어가고, 모델은 이름만 알면
        #   `ls | grep` 으로 파일을 스스로 찾는다(x617 task_003·055 실측).
        if miss:
            lines.append("- %s: %d of %d declared source documents not yet seen"
                         % (s, len(miss), len(decl)))
        else:
            lines.append("- %s: all %d declared source documents are in hand"
                         % (s, len(decl)))
    if not lines:
        return None
    txt = (chr(10) + "[Declared sources for the subjects in this result. "
           "This accounting is complete.]" + chr(10) + chr(10).join(lines))
    cap = int(os.environ.get("T2_SEARCH_CLOSURE_CAP", "1200"))
    return None if len(txt) > cap else txt


def _looks_placeholder(s):
    """값이 **자리표시자 모양**인가 — 연속(0123…) 또는 동일(1111) 자릿수 4개 이상을 담았나.

    ★도메인 낱말 0 · 이름 안 봄 · 의미 추출 0. `DEFAULT_PLACEHOLDERS`(‘ABC123’·‘XYZ789’ 같은
      전 도메인 공통 test 값 집합)의 **닫힘**이다 — 그 집합이 열거로 못 담는 같은 종류를 담는다.
    실측(t7354 6배치 전수·2026-08-25): 이 술어 + *env 가 string 이라 선언* + *문맥 부재* 셋을
      함께 걸면 **20건**이 걸리고 전부 진짜 날조다 — `card_last_4_digits='1234'` 12건 ·
      `transaction_id='TRXN123456789x'` 8건. 오차단 **0**. 반면 셋 중 하나라도 빼면 갈린다:
      타입만 쓰면 `issue_noticed_date='11/14/2025'`(gold 값) 10건을 오차단하고, 모양만 쓰면
      `min_credit_limit='10000'`·`disputed_amount` 를 오차단한다.
    """
    # ⚠자릿수는 **덩어리 안에서만** 본다 — 구분자를 건너뛰고 이어 붙이면 전화번호
    #   `215-555-0267` 이 `2155550267` → `5555` 로 잡힌다(2026-08-25 래칫이 잡은 오차단).
    for d in re.findall(r"\d+", str(s)):
        for i in range(len(d) - 3):
            w = d[i:i + 4]
            if len(set(w)) == 1:
                return True
            if all(int(w[j + 1]) - int(w[j]) == 1 for j in range(3)):
                return True
    return False


# ─── ★도구-선택자 파라미터 (env 스키마 기계 도출·2026-08-24 R3) ───
# 무엇이 틀렸었나: `SELECTOR` 예외는 `_prov_scan_args`(= `T2_PROVENANCE` 전용 死경로)에만 있었고,
#   **실제로 치환하는** `_first_fab_call` 에는 없었다. 그래서 `T2_GROUND` 가 래퍼의 도구-선택자
#   슬롯(`agent_tool_name`)을 데이터 인자로 오인해 고객 이름으로 덮어썼다 —
#   뱅킹 코퍼스 **371/371 이 선택자 키 · 정답 도구명 산출 0/371**(31 sim·5 태스크·x499 verdict ①).
#   반면 리테일에서는 같은 기구가 `address1`/`zip`/`address2`/`item_ids` 에 78건 **정상** 발화한다.
#   ⇒ 끄는 것이 아니라 **선택자 슬롯만** 빼야 하고, 그 술어는 도메인-일반이어야 한다.
#
# 술어(닫힘·스키마만 읽는다·값 해석 0·[[59]]/[[22]]):
#   파라미터 p 가 **선택자**  ⟺  어떤 도구 T 의 스키마가
#     ⑴ 페이로드 파라미터(`arguments`)를 갖고(= 디스패처 형상·`_dispatch_tools` 와 같은 구조 신호),
#     ⑵ T 의 페이로드-아닌 **문자열** 파라미터가 정확히 하나이며 그것이 p 다.
#   유일하지 않으면 **기권**한다(모르면 안 뺀다 = 종전 거동).
#   도출된 이름 집합은 도구를 가리지 않고 적용된다 — `unlock_discoverable_agent_tool(agent_tool_name)`
#   처럼 페이로드 형제가 없는 잠금 도구의 같은 슬롯도 선택자이기 때문이다.
#
# ⛔0 [[62]] 자기점검: ①쟀나 — 코퍼스 전수 재계수(치환 449 중 뱅킹 371·정답 0)가 결손의 실측이다.
#   ②격리 — 이것은 새 레버가 아니라 **기존 기구의 커버리지 구멍**을 닫는 수리다(신설 레버 0).
#   ③사라지는 판단 — 없다. 엔진은 *도구를 고르지 않는다*; 선택자 슬롯을 **건드리지 않을 뿐**이다.
#   ④순위·최댓값·지목 — 없다.
# [[05]] 3문: ⑴도메인-특화 순증 0(도출은 스키마 구조·도메인 이름 리터럴 0) ⑵유동 판단 동결 없음
#   ⑶도메인 행동 수행 안 함. 리테일/에어라인은 페이로드 파라미터를 가진 도구가 없어 집합 = ∅
#   ⇒ **바이트 동일**(정상 발화 78건 보존).
# ⚠[[70]] 무엇을 파는가: 도구명 슬롯의 근거-검사를 판다. 실측 산출이 0/371 이라 순매수지만,
#   지어낸 이름이 env 로 나가 왕복 1턴을 쓰므로 **턴 비용은 A/B 에 계상**해야 한다.
#   (도구명 근거는 원래 별도 레버 `T2_UNLOCK_PROV`/`unknown_bl_*` 의 관할이다.)

_DISPATCH_PAYLOAD_KEY = "arguments"   # tau2 도구-프로토콜 어휘(`ToolCall.arguments` 동형)·도메인 어휘 아님

# 스키마를 못 얻는 오프라인 호출자용 **폴백**일 뿐 권위가 아니다(권위 = 위 술어).
_SELECTOR_FALLBACK = ("agent_tool_name", "user_tool_name", "tool_name", "discoverable_tool_name")


def _schema_props(tool):
    """Tool → {파라미터명: 스펙dict}. 실패/무인자 = {} (예외 0)."""
    try:
        sc = tool.openai_schema
    except Exception:
        return {}
    if not isinstance(sc, dict):
        return {}
    fn = sc.get("function") if isinstance(sc.get("function"), dict) else sc
    pr = ((fn.get("parameters") or {}) if isinstance(fn, dict) else {}).get("properties")
    return pr if isinstance(pr, dict) else {}


def _prop_is_string(spec):
    """파라미터 선언 타입이 문자열인가 (Optional[str] 의 anyOf/oneOf 도 인정)."""
    if not isinstance(spec, dict):
        return False
    if spec.get("type") == "string":
        return True
    for br in (spec.get("anyOf") or spec.get("oneOf") or []):
        if isinstance(br, dict) and br.get("type") == "string":
            return True
    return False


def _tool_objects(holder):
    """agent(`.tools`) 또는 env(`get_tools`/`get_user_tools`) 에서 Tool 객체 수집(중복 무해)."""
    out = []
    ts = getattr(holder, "tools", None)
    if ts:
        try:
            out.extend(list(ts))
        except Exception:
            pass
    for m in ("get_tools", "get_user_tools"):
        f = getattr(holder, m, None)
        if callable(f):
            try:
                out.extend(list(f() or ()))
            except Exception:
                pass
    return out


def _selector_arg_names(tools, payload_key=_DISPATCH_PAYLOAD_KEY):
    """위 술어의 구현 — 도구 스키마 집합 → 선택자 파라미터 이름 frozenset.

    열거 0·도메인 리터럴 0·값 미접근. 페이로드-아닌 문자열 파라미터가 2개 이상인
    디스패처는 라우팅 슬롯을 확정할 수 없으므로 **기권**한다(종전 거동 유지)."""
    names = set()
    for t in (tools or []):
        props = _schema_props(t)
        if payload_key not in props:
            continue
        cand = [p for p, spec in props.items()
                if p != payload_key and _prop_is_string(spec)]
        if len(cand) == 1:
            names.add(cand[0])
    return frozenset(names)


def selector_args_of(holder, payload_key=_DISPATCH_PAYLOAD_KEY):
    """agent 또는 env 를 받아 선택자 이름 집합을 낸다(캐싱은 호출측)."""
    return _selector_arg_names(_tool_objects(holder), payload_key=payload_key)


def _selector_args_cached(holder):
    """도구 수가 바뀌면(discoverable 잠금해제) 재도출하는 얕은 캐시."""
    n = len(_tool_objects(holder))
    c = getattr(holder, "_t2_selector_cache", None)
    if c is None or c[0] != n:
        c = (n, _selector_arg_names(_tool_objects(holder)))
        try:
            holder._t2_selector_cache = c
        except Exception:
            pass
    return c[1]


def _prov_scan_args(tc, selectors=None):
    """검사 대상 인자 (키, 값) — **discoverable 래퍼 안쪽까지 편다**.

    ★왜 (2026-08-15·085 실측): banking 의 write 는 거의 전부
    `call_discoverable_agent_tool({"agent_tool_name": …, "arguments": "<JSON 문자열>"})` 형태다.
    최상위 키만 훑으면 `arguments` 는 힌트에 안 걸리고 `agent_tool_name` 만 걸리는데 그 값은
    unlock 출력에 늘 있으므로 **항상 통과**한다 ⇒ 안쪽 `transaction_id`·`card_id` 가
    **한 번도 검사되지 않았다**. 085 가 그 자리다 — `transaction_id='tx111111'` ·
    `card_id='card123456'` · `account_id` 자리에 `user_id` 를 넣고도 통과했고, write **이전**
    도구 출력에는 실제 `btxn_` id 가 **하나도 없었다**.
    오늘 아침 고친 *"중첩 `arguments` 계약↔검산기 불일치"* 와 **같은 사각**이다.

    ⚠엔진은 값을 해석하지 않는다 — **키 이름이 식별자류인가**(도메인-일반 힌트)와
      **그 값이 이전 문맥에 있는가**(부분문자열)만 본다([[59]]).

    ⛔0 [[62]] 자기점검:
      ①쟀나 — C45(32B 날조 **67→0%** · over-block 0 · Δspurious 0)가 이 검사의 실측이고,
        오늘 t7295 포렌식이 라이브 자리를 짚었다(085 날조 인자가 write **이전** 문맥에 부재 ·
        반경 **7 sim/4 태스크**/중첩-인자 호출 363 중).
      ②격리에서 성공하나 — 이 레버는 **검증**이지 대체가 아니다. 값을 만드는 것도 고르는 것도
        여전히 모델이고, 엔진은 *어디에도 없는 값*만 거절한다.
      ③사라지는 판단 — **없다**. 문맥에 있는 값은 전부 통과한다(닫힌 술어·부분문자열 존재).
      ④순위·최댓값·지목 — **없다**. 거절문은 무엇이 틀렸고 무엇을 하면 되는지만 말한다([[64]]).
    [[05]] 3문: ⑴도메인-특화 순증 **0**(힌트는 도메인-일반·새 A2 키 0) · ⑵유동 판단 동결 **없음** ·
      ⑶도메인 행동 수행 **안 함**(거절만·getter 자동호출은 별도 플래그 `T2_AUTOFETCH`).

    ⚠거동 변화 범위: 이 함수는 `T2_PROVENANCE=1` 일 때만 불린다. 그 변수는 `go_stack.sh` 에도
      런처에도 **없어서** 현재 런에서는 **한 번도 실행되지 않는다**(t7295·t7296 `PROVENANCE_R1B` 0건).
      ⇒ 이 수리 자체의 즉시 거동 변화는 **0**이다. 켜는 것은 별건이고 **Δspurious 재측정 의무**가
      붙는다 — C45 의 over-block 0 은 *중첩을 안 보던 시절* 수치다.
    """
    # ★래퍼의 **도구 선택자**는 operand 가 아니다 (2026-08-15·x334 오프라인 재생).
    #   `agent_tool_name` 은 `_hint_hit` 에서 토큰 `name` 때문에 식별자로 잡히는데, 그것은
    #   *데이터 값*이 아니라 *어느 도구를 부를지*다. 재생 실측: over-block 28건이 **거의 전부**
    #   이 키였다(예: `agent_tool_name='apply_checking_account_credit_5829'` 를 "날조"로 차단).
    #   도구 이름의 근거 검사는 **별도 레버**(`T2_UNLOCK_PROV`)의 일이다 — 여기서 겹쳐 막으면
    #   정상 호출을 죽인다([[57]] 상쇄: 이 레버가 파는 것이 정확히 그것이었다).
    #   ★2026-08-24 R3: 이 집합은 더 이상 여기서 열거하지 않는다 — **env 스키마에서 도출**해
    #   (`_selector_arg_names`) 호출측이 넘긴다. `selectors is None` 은 스키마를 못 얻는
    #   오프라인 호출자(프로브·검정)용 폴백일 뿐이고 권위가 아니다.
    SELECTOR = _SELECTOR_FALLBACK if selectors is None else selectors
    out = []
    args = _args_dict(tc)
    for k, v in (args or {}).items():
        if k in SELECTOR:
            continue
        if k == "arguments" and isinstance(v, str):
            try:
                inner = json.loads(v)
            except Exception:
                inner = None
            if isinstance(inner, dict):
                out.extend(inner.items())
                continue
        out.append((k, v))
    return out


def _provenance_deny(tc, ctx, hints=DEFAULT_ARG_HINTS, selectors=None):
    """identifying 인자값이 컨텍스트에 없으면 fabricated → (gate, reason) 반환, 아니면 None.
    selectors = 스키마-도출 선택자 이름 집합(None = 오프라인 폴백·`_prov_scan_args` 참조)."""
    args = _prov_scan_args(tc, selectors=selectors)
    if not args:
        return None
    for k, v in args:
        if not _hint_hit(k, hints):
            continue
        for val in _flatten(v):
            s = str(val).strip()
            if len(s) < 4:
                continue
            if not _ctx_has(s, ctx):
                return ("PROVENANCE_R1B",
                        f"argument '{k}'='{s}' was not provided by the user nor returned by any tool — it looks invented "
                        "(possibly copied from a schema example value). Do NOT call any tool with a guessed/placeholder value. "
                        "Instead OBTAIN the real value first: if a lookup/getter tool can produce it "
                        "(e.g. call a getter to retrieve the user's records, payment methods, or addresses), call that and read the value from its output; "
                        "otherwise ASK the user for it.")
    return None


_DUMP_MARK = "Record ID:"          # env 레코드 덤프의 머리 — `_parse_record_dump` 와 같은 표지
_LABEL_RE = re.compile(r"(\w+):\s*([^\s,;]+)")
_DUMP_HEAD = "ID"                  # `Record ID:` 의 꼬리 — 필드 이름이 아니라 덤프의 머리다


def _record_labels(orch):
    """도구 출력의 **레코드 덤프**에서 `필드: 값` 을 축자로. 값의 뜻은 안 읽는다([[59]]).

    ⚠덤프에서만 읽는다. 도구 스키마 줄도 같은 모양이라(`account_id: string (required) - …`)
      걸러내지 않으면 `string` 이 그 인자의 '옳은 값' 으로 들어온다(x565 배선 확인이 잡았다).
    """
    out = {}
    try:
        for m in orch.get_messages():
            if getattr(m, "role", None) != "tool":
                continue
            c = " ".join(str(getattr(m, "content", "") or "").split())
            k0 = c.find(_DUMP_MARK)
            if k0 < 0:
                continue
            for f, v in _LABEL_RE.findall(c[k0:]):
                out.setdefault(f, set()).add(v)
    except Exception:
        pass
    return out


def _same_axis(asr, a, b):
    """같은 축의 동의어인가 — `arg_source_reads` 의 생산자 목록이 **완전히 같으면** 같은 축.
    (`phone`/`phone_number` 가 그것이고, 이 판정 없이는 040 에서 17건이 거짓 경보다.)"""
    return bool(asr.get(a)) and asr.get(a) == asr.get(b)


def _label_mismatch_deny(tc, a2, labels, selectors=None):
    """선언된 식별자 인자에 **env 가 다른 필드로 낸 값**이 들어갔으면 (gate, reason).

    ## 결함 (2026-08-27)

    `_provenance_deny` 의 술어는 `_ctx_has` — *"이 문자열이 문맥 어딘가에 있나"* 다. 그래서
    **출처는 맞고 종류가 틀린** 값이 전부 통과한다. env 는 레코드를 `필드: 값` 으로 찍으므로
    종류의 답은 이미 문맥에 있는데 우리가 안 보고 있었다.

    실측(`x564` · 채점 37 sim · 식별자 인자 720): 제 이름표로 나온 값 72% · **다른 이름표
    8%** · 이름표 없음 18% · 부재 2%. 다른 이름표 60건에서 잡음 둘(덤프 머리 `ID`,
    같은 축 동의어)을 걷으면 **040·057·074·079·085 다섯 태스크 12 sim** 이 남고 **reward 1.0
    인 것 0** 이다(손실 불가).

    ## [[62]] 4문
      ①결손 = 위 코퍼스 실측. ②격리(`x565`·8140·3팔): A_asis **4/16**(라이브 재현 — 옛 값
      그대로) · B_say **16/16**(전부 생산자 read 호출) · N_len **4/16**(부정통제 깨끗) ⇒
      전달로 산다. ③사라지는 모델 판단 **0** — 우리는 **어느 값을 쓰라고 말하지 않는다**.
      ④최댓값·argmax·*"정답은 X"* 0.

    ## [[05]] 3문
      ⑴도메인-특화 순증 0 — 인자 이름은 `arg_source_reads`(선언), 필드 이름은 env 출력.
      ⑵유동 판단 동결 아니오. ⑶도메인 행동 수행 0.
    """
    asr = {k: v for k, v in ((a2 or {}).get("arg_source_reads") or {}).items()
           if not k.startswith("_") and isinstance(v, list)}
    if not (asr and labels):
        return None
    for k, v in _prov_scan_args(tc, selectors=selectors):
        if k not in asr:
            continue
        for val in _flatten(v):
            sv = str(val).strip()
            if len(sv) < 4 or sv in labels.get(k, ()):
                continue
            src = sorted(f for f, vals in labels.items()
                         if f != k and f != _DUMP_HEAD and not _same_axis(asr, f, k)
                         and sv in vals)
            if not src:
                continue
            prod = (asr.get(k) or [""])[0]
            return ("ARG_LABEL",
                    "argument '%s'='%s' is what the records above give as `%s`, not as `%s`. "
                    "The values for `%s` are the ones the records list under that name; %s is "
                    "what produces them. Re-issue the call with a value the records give under "
                    "`%s`, and if none is there yet, run %s first."
                    % (k, sv, src[0], k, k, prod, k, prod))
    return None


def _autofetch_text(self, orig, gate, producer):
    """T2_AUTOFETCH: provenance-deny 시, 인증됐으면 A2 producer(getter)를 결정론 호출해
    그 출력을 텍스트로 반환(모델에 *실값* 제공). = '날조-FIRST' default를 엔진이 결정론으로 메움.
    producer = A2 producers.authenticated_user_record {tool, args_from:{argname:@auth_user}}.
    decidable·도메인-일반(도구·인자 = A2서 도출·airline swap = 동일 메커니즘)·side-effect 0(getter)."""
    if producer is None or getattr(gate, "auth_user", None) is None:
        return ""
    try:
        from tau2.data_model.message import ToolCall
        pargs = {k: (gate.auth_user if v == "@auth_user" else v)
                 for k, v in (producer.get("args_from") or {}).items()}
        ptc = ToolCall(id="autofetch", name=producer["tool"],
                       arguments=pargs, requestor="assistant")
        out = orig(self, [ptc])
        if out and not getattr(out[0], "error", False):
            return ("\nI have fetched the authenticated user's actual record for you — copy a REAL value "
                    "from it, never a placeholder: " + _content_str(out[0])[:1800])
    except Exception:
        pass
    return ""


def _call_key(tc):
    """(도구명, 정규화 인자) = 동일-호출 식별 (retry-loop 탐지·decidable)."""
    return (getattr(tc, "name", "") or "") + "::" + json.dumps(_args_dict(tc), sort_keys=True, ensure_ascii=False)


def apply():
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    orig = BaseOrchestrator._execute_tool_calls

    def gated(self, tool_calls):
        env = self.environment
        a2 = _domain_a2(getattr(env, "domain_name", None))
        if a2 is None:
            return orig(self, tool_calls)
        # T2_GATE_KINDS = 측정-격리 플래그(콤마구분 kind 화이트리스트). 미지정=전체.
        # 예 "preconditions"=G5만(flow-discipline 격리·floor 대비 confound 배제). 도메인-일반.
        _kinds = os.environ.get("T2_GATE_KINDS")
        _gate_list = a2["gates"]
        if _kinds:
            _allow = {k.strip() for k in _kinds.split(",") if k.strip()}
            _gate_list = [g for g in a2["gates"] if g.get("kind") in _allow]
        gate = getattr(self, "_t2_gate", None)
        if gate is None:
            gate = self._t2_gate = GateInterpreter(_gate_list, resolvers=resolvers_from_env(env))
        auth_tools = a2["_auth_tools"]
        producer = a2["_producer"]
        hints = a2["_hints"]
        last_user = _last_user_text(self)
        # ★NOTICE-PERGATE: 문구-매개 클로저(커링) — check()가 게이트별 notice_text로 평가.
        tms = lambda text: _transfer_msg_sent(self, text)  # noqa: E731

        # T2_PRESENT_READS=1 = REPLAY-SAFE present: 후보-producer 읽기응답에 clean 요약 덧붙임(deny 아님·측정 arm).
        present_on = os.environ.get("T2_PRESENT_READS") == "1"
        g6 = next((g for g in a2["gates"] if g.get("kind") == "select_confirm"), None) if present_on else None
        # T2_PRESENT_NESTED=1 = operand-grounding present 확장(L2 item/L3 variant): read record의
        # nested list/dict를 명시 choice-set으로 (priority-2·replay-safe·A2 present_specs 구동).
        nested_specs = (a2.get("present_specs") or []) if os.environ.get("T2_PRESENT_NESTED") == "1" else []
        # T2_CALC=1 = calc_NL offload(측정 arm): read record서 결정론 집계(available count·order total) 계산·주입.
        # 보고는 모델 → report-conversion 측정([COMPUTED FACTS] 블록=census 마커). A2 calc_specs 구동·엔진 general op.
        calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC") == "1" else []
        # T2_PROVENANCE=1 = orchestrator-레벨 게이트(날조 호출을 *실행 전* deny→error로 surface).
        prov_on = os.environ.get("T2_PROVENANCE") == "1"
        ctx = _context_text(self) if prov_on else None
        # ★T2_ARG_LABEL (2026-08-27·기본 OFF·`_label_mismatch_deny` 주석에 근거).
        #   이름표 판은 호출마다 다시 짓지 않는다 — 이 턴 안에서 한 번만.
        label_on = os.environ.get("T2_ARG_LABEL") == "1"
        _lab_cache = []

        def _rec_labels():
            if not _lab_cache:
                _lab_cache.append(_record_labels(self))
            return _lab_cache[0]
        # T2_RETRY_CONTROLLER=1 = C8 recovery: 반복-동일-실패호출 차단+다양화 지시 (decidable·offload·무학습).
        retry_on = os.environ.get("T2_RETRY_CONTROLLER") == "1"
        retry_k = int(os.environ.get("T2_RETRY_K", "3"))  # rule②: 연속 실패 K회 가드
        failed = getattr(self, "_t2_failed", None)
        if failed is None:
            failed = self._t2_failed = {}
        DIVERSIFY = ("Take a DIFFERENT action: (a) FETCH a missing value from the tool that produces it, "
                     "(b) ASK the user for the correct value, or (c) if you genuinely cannot proceed, "
                     "transfer to a human agent.")

        def _mark_fail(k, why):  # rule①·② 공통: 실패 기록 + 연속카운트++
            if retry_on:
                if k is not None:
                    failed[k] = why
                self._t2_consec = getattr(self, "_t2_consec", 0) + 1

        # ★T2_WRITE_CAP (S1a-1·F5): 성공-write 반복 루프 차단. 기존 retry-controller(rule①)는
        #   *실패* 호출만 잡는다(failed dict) → t102형 19× *성공*-write 재발emit은 통과. 도메인-일반
        #   (write 도구=_confirm_write_tools·A2 confirm 게이트서 도출·리터럴 0)·3중키=_call_key(name+args).
        #   성공 K회(기본2) 초과 동일-write → deny("이미 완료"). 다른 args=다른 키=미차단. Δ계측: 정당 재시도 오차단 0.
        wcap_on = os.environ.get("T2_WRITE_CAP") == "1"
        wcap_k = int(os.environ.get("T2_WRITE_CAP_K", "2"))
        wcap_tools = _confirm_write_tools(a2) if wcap_on else set()
        wdone = self._t2_wdone = getattr(self, "_t2_wdone", {})
        # ★T2_WRITE_EVIDENCE=1: A2 write_evidence_specs 구동(기본 OFF·§_write_evidence_deny).
        wev_specs = (a2.get("write_evidence_specs") or []) \
            if os.environ.get("T2_WRITE_EVIDENCE") == "1" else []

        results = []
        for tc in tool_calls:
            if getattr(tc, "requestor", "assistant") != "assistant":
                results.extend(orig(self, [tc]))  # user-side 툴콜(타 도메인)은 비대상
                continue
            key = _call_key(tc) if retry_on else None
            # rule① 정확-반복 차단 (순환 loop·census ~70%)
            if retry_on and key in failed:
                self.num_errors += 1
                _mark_fail(key, failed[key])
                results.append(_deny_msg(tc, "RETRY_LOOP",
                    f"You ALREADY called {tc.name} with these exact arguments and it FAILED: {failed[key][:160]}. "
                    "Do NOT repeat the identical call. " + DIVERSIFY))
                continue
            # rule② 연속-실패 K 가드 (다양-실패 loop·census ~17%)
            if retry_on and getattr(self, "_t2_consec", 0) >= retry_k:
                self.num_errors += 1
                self._t2_consec = 0  # 가드 발동 후 리셋(매콜 발동 방지)
                results.append(_deny_msg(tc, "RETRY_ESCALATE",
                    f"You have failed {retry_k}+ tool calls in a row. STOP this approach. " + DIVERSIFY))
                continue
            if prov_on:  # L2 provenance: 날조 인자값 차단 (R1B)
                # ★R3: 선택자 이름은 하드코딩 튜플이 아니라 **env 스키마에서 도출**한다.
                pd = _provenance_deny(tc, ctx, hints, selectors=_selector_args_cached(env))
                if pd:
                    self.num_errors += 1
                    extra = _autofetch_text(self, orig, gate, producer) if os.environ.get("T2_AUTOFETCH") == "1" else ""
                    _mark_fail(key, pd[1])
                    results.append(_deny_msg(tc, pd[0], pd[1] + extra))
                    continue
            if label_on:  # ★T2_ARG_LABEL: env 가 다른 필드로 낸 값을 이 인자에 넣은 것을 반려
                _lm = _label_mismatch_deny(tc, a2, _rec_labels(),
                                           selectors=_selector_args_cached(env))
                if _lm:
                    self.num_errors += 1
                    _mark_fail(key, _lm[1])
                    results.append(_deny_msg(tc, _lm[0], _lm[1]))
                    continue
            ok, g, why = gate.check(tc.name, tc.arguments or {}, last_user_msg=last_user,
                                    transfer_msg_sent=tms)
            if not ok:
                self.num_errors += 1
                _mark_fail(key, why)
                results.append(_deny_msg(tc, g, why))
                continue
            # ★T2_WRITE_EVIDENCE: 원장(도구 출력) 증거 없는 선언-write 반려 (§_write_evidence_deny)
            if wev_specs:
                wd = _write_evidence_deny(self, tc, wev_specs)
                if wd:
                    self.num_errors += 1
                    _mark_fail(key, wd)
                    print("[T2_WRITE_EVIDENCE] deny tool=%s" % tc.name, file=sys.stderr, flush=True)
                    results.append(_deny_msg(tc, "WRITE_EVIDENCE", wd))
                    continue
            # ★V7 T2_TOOL_SIGNATURE: A2가 선언한 호출 서명 밖 키 → deny+재발행(엔진은 인자를
            #   떼지 않는다·C151 compliance 패턴). 근거=도메인 **정책**(§12)·기본 OFF.
            try:
                import t2_signature as _sg
                _sv = _sg.signature_violation(tc.name, _args_dict(tc), a2)
            except Exception:
                _sv = None
            if _sv:
                self.num_errors += 1
                _mark_fail(key, _sv)
                print("[T2_TOOL_SIGNATURE] deny tool=%s" % tc.name, file=sys.stderr, flush=True)
                results.append(_deny_msg(tc, "TOOL_SIGNATURE", _sv))
                continue
            # ★T2_WRITE_CAP (S1a-1·F5): 이미 K회 성공한 동일-write → 재실행 안 함·deny("완료")
            if wcap_on and tc.name in wcap_tools:
                _wk = _call_key(tc)
                if wdone.get(_wk, 0) >= wcap_k:
                    self.num_errors += 1
                    print("[T2_WRITE_CAP] capped tool=%s (already succeeded %dx)"
                          % (tc.name, wdone[_wk]), file=sys.stderr, flush=True)
                    results.append(_deny_msg(tc, "WRITE_CAP",
                        "You ALREADY successfully performed this exact action %dx; it is DONE. "
                        "Do NOT repeat the identical call. Move to the next task or confirm completion to the user."
                        % wdone[_wk]))
                    continue
            out = orig(self, [tc])
            results.extend(out)
            if out and getattr(out[0], "error", False):
                _mark_fail(key, _content_str(out[0]))
            elif retry_on:
                self._t2_consec = 0  # 성공 → 연속카운트 리셋
            # ★T2_WRITE_CAP 기록: 성공한 write만 카운트(err=False)
            if wcap_on and tc.name in wcap_tools and out and not getattr(out[0], "error", False):
                _wk2 = _call_key(tc)
                wdone[_wk2] = wdone.get(_wk2, 0) + 1
            if tc.name in obs_tools_g(gate) and out and not out[0].error:
                gate.observe(tc.name, tc.arguments, _content_str(out[0]))
            # ★REPLAY-SAFE present (T2_PRESENT_READS=1): 후보-producer 읽기 응답에 clean 요약 덧붙임.
            # 읽기는 evaluation replay서 skip → content 증강 안전(write-deny=replay 깨짐과 대조).
            if (present_on and out and not getattr(out[0], "error", False)
                    and g6 is not None and tc.name == g6.get("user_producer")):
                uid = (tc.arguments or {}).get(g6.get("user_id_arg", "user_id"))
                summ = candidate_summary(gate.resolvers, g6, uid)
                if summ:
                    try:
                        out[0].content = _content_str(out[0]) + summ
                    except Exception:
                        pass
            # ★read-augment(nested present + calc): RAW 응답을 augment *전* 1회 파싱·공유.
            # (버그수정 2026-06-26: nested가 out.content에 텍스트 append하면 calc의 _parse_json이
            #  오염 content서 실패→calc 미발화 31/342. raw 1회 파싱으로 두 증강 모두 정상.)
            _rec = None
            if (nested_specs or calc_specs) and out and not getattr(out[0], "error", False):
                _rec = _parse_json(_content_str(out[0]))
            # operand-grounding present (T2_PRESENT_NESTED=1): nested operand 명시 choice-set(L2/L3).
            if nested_specs and _rec is not None:
                spec = next((s for s in nested_specs if s.get("trigger_tool") == tc.name), None)
                if spec is not None:
                    summ = nested_candidate_summary(_rec, spec)
                    if summ:
                        try:
                            out[0].content = _content_str(out[0]) + summ
                        except Exception:
                            pass
            # calc_NL offload (T2_CALC=1): 결정론 집계 계산·주입(보고는 모델). [COMPUTED FACTS]=census 마커.
            if calc_specs and _rec is not None:
                cs = [s for s in calc_specs if s.get("trigger_tool") == tc.name]
                if cs:
                    facts = compute_facts(_rec, cs)
                    if facts:
                        try:
                            out[0].content = _content_str(out[0]) + facts
                        except Exception:
                            pass
        return results

    BaseOrchestrator._execute_tool_calls = gated
    return orig


def _last_user_text(orch):
    try:
        for m in reversed(orch.get_messages()):
            if getattr(m, "role", None) == "user" and getattr(m, "content", None):
                return m.content if isinstance(m.content, str) else str(m.content)
    except Exception:
        pass
    return None


def _transfer_msg_sent(orch, notice_text):
    """notice: transfer 문구가 어시스턴트 발화로 이미 송신됐는가 (불가 판단 시 None).
    ★C213/G1: 전문-일치 → gate_interpreter.notice_sent_in **공용 정규화 술어**(032 [S])."""
    if not notice_text:
        return None
    try:
        from gate_interpreter import notice_sent_in
        texts = [getattr(m, "content", None) for m in orch.get_messages()
                 if getattr(m, "role", None) == "assistant"]
        return notice_sent_in(texts, notice_text)
    except Exception:
        return None


def _parse_json(s):
    """tool content 문자열 → dict/list (실패 시 None). nested present용."""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _arg_values(v, out=None):
    """호출 인자가 실제로 담은 스칼라 값들(중첩 JSON 포함).

    "이 id를 제출했는가"를 묻는 자리에서 쓴다. 예전 방식은 인자 문자열에 철자 규칙을 돌려
    id처럼 생긴 것을 뽑았는데, 그러면 **id의 생김새**를 엔진이 알아야 한다. 값을 그대로 모아
    엔진이 낸 집합과 대조하면 생김새를 알 필요가 없다 — 멤버십이 판정하고 철자는 관여하지 않는다.
    """
    if out is None:
        out = set()
    if isinstance(v, dict):
        for x in v.values():
            _arg_values(x, out)
    elif isinstance(v, (list, tuple, set)):
        for x in v:
            _arg_values(x, out)
    elif isinstance(v, str):
        out.add(v.strip())
        nested = _parse_json(v)
        if isinstance(nested, (dict, list)):
            _arg_values(nested, out)
    elif v is not None and not isinstance(v, bool):
        out.add(str(v))
    return out


def _content_str(tool_msg):
    c = tool_msg.content
    if isinstance(c, str):
        try:
            v = json.loads(c)
            return v if isinstance(v, str) else c
        except (ValueError, TypeError):
            return c
    return str(c)


def _axis_surface(orch, tool_calls, results):
    """★축-레버 표면화 진입점 (FAILURE_AXES / RUNAWAY 설계서·2026-08-02).

    [[05]] 3질문: ⑴도메인 리터럴 **0** — 도구 이름은 A2 `tool_registry`/`scaffold_get_tools`,
    문구는 A2 `axis_notes`. ⑵닫는 술어 = 집합 멤버십·unlock 상태·딕셔너리 diff·단수명 배열·
    토큰 축자 실재뿐(표면-불변). ⑶**표면화만** — 거부·재작성·값 생성 0.
    각 레버는 플래그 기본 OFF. A2에 선언이 없으면 무발화(거동 변화 0)."""
    import t2_axis_levers as AX
    env = getattr(orch, "environment", None)
    a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
    if not isinstance(a2, dict):
        return
    notes_cfg = a2.get("axis_notes") or {}
    if not notes_cfg:
        return
    scaffold, _ad, _ud = AX.registry_from_a2(a2)          # 스캐폴드 이름 = A2 선언
    agent_d, user_d = AX.registry_from_env(orch)          # ★discoverable = env 기계 도출(opex 0)
    agent_d |= _ad
    user_d |= _ud
    if not (agent_d or user_d):
        return
    unlocked = getattr(orch, "_t2_axis_unlocked", None)
    if unlocked is None:
        unlocked = orch._t2_axis_unlocked = set()

    by_id = {getattr(r, "id", None): r for r in (results or [])}
    called = set()
    for tc in (tool_calls or []):
        called.add(str(getattr(tc, "name", "") or ""))
        a = getattr(tc, "arguments", None)
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        a = a if isinstance(a, dict) else {}
        for k in ("agent_tool_name", "discoverable_tool_name"):
            if a.get(k):
                called.add(str(a[k]))
        r = by_id.get(getattr(tc, "id", None))
        if r is None:
            continue
        txt = _content_str(r) or ""
        if "Tool unlocked:" in txt:
            m = re.search(r"Tool unlocked:\s*([A-Za-z0-9_]+)", txt)
            if m:
                unlocked.add(m.group(1))
        add = []
        # ★P13 (2026-08-02·승인): CHANNEL은 여기서 **부착하지 않는다**. 발화 조건인 unlock/give/call이
        #   전부 env mutating(실측)이라 출력-부착은 replay를 깬다(041 사고) — 가드에 걸려 어차피 100%
        #   드롭된다. ⇒ 예방형 **생성-레벨**(호출 실행 前 교정)로 이설: `unified()` 말미 P13 블록.
        #   규약: **출력-부착 = 읽기 도구 전용** / mutating 피드백 = 생성-레벨.
        if os.environ.get("T2_SCALAR_ARRAY") == "1":
            n = AX.scalar_array_note(a, notes_cfg)
            if n:
                add.append(n)
        if os.environ.get("T2_FIT_DIFF") == "1" and notes_cfg.get("diff"):
            n = AX.fit_diff_note(txt, notes_cfg)
            if n:
                add.append(n)
        # ─── ★④ 회수 경계 표면화 (2026-08-03·TRANSFER_INSTRUCTION_FIDELITY_DESIGN §4.2) ───
        #   KB_search는 top-k만 조용히 반환하므로 "195개 걸림 중 8개"와 "4개 걸림 중 4개"가
        #   모델 눈에 같다. 앞은 좁히라는 신호, 뒤는 전수를 봤다는 근거인데 구분이 안 된다.
        #   엔진이 이미 셀 수 있는 수(내용어 AND·임계 없음·분모=전 코퍼스)를 표면화한다.
        #   ⚠P13 규약("출력-부착 = 읽기 도구 전용")을 **지킨다**: env 술어 실측(2026-08-03)
        #   `_is_mutating_tool("KB_search_bm25"/"_dense") = False` — 6393 주석의 "KB_search는
        #   env mutating"은 사실이 아니며, replay는 비-mutating 도구를 재실행하지 않는다
        #   (environment.py: "Non-mutating tools ... skip them"). 아래 P12 가드도 그대로 통과한다.
        if (os.environ.get("T2_MATCH_COUNT") == "1"
                and str(getattr(tc, "name", "") or "").startswith("KB_search")):
            try:
                import t2_match_count as _mc
                _n4 = _mc.note(a.get("query"), txt, orch)
                if _n4:
                    add.append(_n4)
            except Exception as _e4:
                print("[T2_MATCH_COUNT] skipped: %r" % (_e4,), file=sys.stderr, flush=True)
        if add:
            # ★P12 (2026-08-02·041 R0 사고): replay 가드를 **버스 여부와 무관하게 상시** 적용.
            #   사고: 041에서 mutating 도구(`call_discoverable_agent_tool`) 출력에 [axis] 노트가 붙어
            #   tau2 평가 replay(environment.py:378~390 — mutating 도구 재실행 후 content 비교)가
            #   불일치 → ValueError → sim 전체 재시도(R0 6,579s 폐기). 기존 코드는 이 가드를
            #   `T2_SURFACE_BUS=1`일 때만 태웠고, 라이브(버스 OFF)는 **무가드 직접 부착**이었다.
            #   ⇒ C208② 계열(우리 스캐폴드의 replay 위반) 3번째 재발. 가드는 이제 무조건.
            _replay_ok = _dedup_cache_safe(orch, str(getattr(tc, "name", "") or ""))
            if not _replay_ok:
                print("[T2_AXIS] skip(mutating·replay-safe) %s <- %d note(s) dropped"
                      % (getattr(tc, "name", ""), len(add)), file=sys.stderr, flush=True)
                add = []
            # ★T2_SURFACE_BUS=1 (CONSOLIDATION §2b v0·첫 이관 채널): 부착을 버스가 집행 —
            #   ①replay(위 가드와 동일 술어·이중 적용은 무해) ③예산 ④순서. OFF=직접 부착.
            if add and os.environ.get("T2_SURFACE_BUS") == "1":
                import t2_surface_bus as _sb
                _bus = _sb.get_bus(orch)
                for _x in add:
                    _bus.register("guidance", _x)
                add = _bus.flush(_replay_ok)
            if add:
                try:
                    r.content = (txt + "\n" + "\n".join("[axis] " + x for x in add))
                except Exception:
                    pass
                print("[T2_AXIS] %s <- %d note(s)" % (getattr(tc, "name", ""), len(add)),
                      file=sys.stderr, flush=True)

    # 본문-언급 / 터미널-턴 = 대화 이력이 필요한 레버(없으면 무발화)
    msgs = getattr(orch, "messages", None) or getattr(orch, "_t2_msgs", None) or []
    said, utext = "", ""
    for m in list(msgs)[-6:]:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        c = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if not isinstance(c, str):
            continue
        if role == "assistant":
            said += " " + c
        elif role == "user":
            utext += " " + c
    tail = results[-1] if results else None
    extra = []
    # ★발화 상한 (오프라인 재생 x42가 잡은 자기 결함·2026-08-02): 서사가 매 사이클 반복되면
    #   안내도 매번 붙어 **026에서 55회**가 된다 — 문맥 팽창을 고치려는 레버가 팽창을 만든다.
    #   ⇒ (sim, 대상) 당 상한 2회. 같은 말을 반복 제시해도 같은 선택이 재생산될 뿐이다(C194).
    _cap_n = int(os.environ.get("T2_AXIS_NOTE_CAP", "2") or 2)
    _fired = getattr(orch, "_t2_axis_fired", None)
    if _fired is None:
        _fired = orch._t2_axis_fired = {}

    def _allow(key):
        _fired[key] = _fired.get(key, 0) + 1
        return _fired[key] <= _cap_n

    if tail is not None and os.environ.get("T2_TOOL_CHANNEL") == "1" and said:
        for _m in AX.mention_note(said, called, agent_d, user_d, unlocked, notes_cfg):
            if _allow(("mention", _m[:60])):
                extra.append(_m)
    if tail is not None and os.environ.get("T2_TERMINAL_TURN") == "1" and utext:
        n = AX.terminal_turn_note(utext, notes_cfg.get("transfer_tokens") or [],
                                  any("transfer" in c for c in called), notes_cfg)
        if n and _allow(("terminal", "")):
            extra.append(n)
    if extra and tail is not None:
        try:
            tail.content = (_content_str(tail) or "") + "\n" + \
                "\n".join("[axis] " + x for x in extra)
            print("[T2_AXIS] tail ← %d note(s)" % len(extra), file=sys.stderr, flush=True)
        except Exception:
            pass


def _deny_msg(tc, gate_name, reason):
    from tau2.data_model.message import ToolMessage
    return ToolMessage(
        id=tc.id, role="tool", requestor="assistant", error=True,
        content=f"Error: [POLICY GATE {gate_name}] {reason}",
    )


# ─── ★권장 설계: agent 생성-레벨 내부 재생성 (T2_PROV_REGEN=1) ───
# 검증기가 날조 인자 감지 → state.messages(공식 대화) 오염 없이 *작업본*에 거부 피드백 추가
# → generate() 재호출(최대 K) → 유효 호출만 반환. 측정 무변경(가드된 시스템을 정직 측정).
REGEN_FEEDBACK = (
    "Error: [PROVENANCE] argument '{k}'='{s}' was not provided by the user nor returned by any tool "
    "— it looks invented (e.g. a schema example value). Do NOT use placeholder/example values and do NOT "
    "ask the user. Instead call a lookup/getter tool that produces this value (e.g. a getter to "
    "retrieve the user's records, payment methods, or addresses) and read the real value from its output. "
    "Now emit a corrected tool call."
)

# T5-C 스펙#2 (prov_mode=rescue 전용): 예시 나열 제거 — 나열 자체가 프라이밍([[42]] 동형·t61형 오도 [P])
REGEN_FEEDBACK_NEUTRAL = (
    # ★V2.5 t17 교정(2026-07-11): 중립화가 priming(필드 예시)뿐 아니라 *getter-호출 지시*까지
    #   약화 → 에이전트가 조회 대신 사용자에게 되물어 no-write(회귀). 필드 예시만 제거하고
    #   "getter 도구를 *호출*해 그 출력에서 읽어라"는 행동 지시는 강하게 유지(사용자-되묻기 탈출구 삭제).
    "Error: [PROVENANCE] argument '{k}'='{s}' was not provided by the user nor returned by any tool "
    "— it looks invented. Do NOT use placeholder/example values. Call the lookup/getter tool that "
    "produces this value and read the real value from its output, then emit a corrected tool call."
)

# ★B-max① t17 지시형(directive) 피드백: fab 인자에 A2 resolver_path가 있으면 generic("getter를 불러라")
#   대신 *어느* producer를 *어떤 알려진 입력*으로 호출해 *어느 필드*를 읽는지 지목 (사용자-되묻기 탈출구 차단).
#   근거 NEWSTACK_GAIN_SIDEEFFECT §G t17 + OVERNIGHT §결과2 (지시형 4지선다가 base ASK 21→loop 0).
REGEN_FEEDBACK_DIRECTIVE = (
    "Error: [PROVENANCE] argument '{k}'='{s}' was not provided by the user nor returned by any tool "
    "— it looks invented. Do NOT ask the user; instead call `{producer}`({in_arg}={in_val}) and read "
    "'{field}' from its output, then emit a corrected tool call with the real value from that output."
)


def _write_evidence_deny(orch, tc, specs):
    """구 apply() 경로 어댑터 — 코어는 _wev_deny_msgs (unified와 공유)."""
    try:
        msgs = orch.get_messages()
    except Exception:
        return None
    return _wev_deny_msgs(msgs, tc, specs)


def _param_cap_deny(agent, la, UserMessage, messages, tc, specs):
    """★정책-캡 게이트 (T2_PARAM_CAP=1·2026-07-22 §2br·rall9 054 실측). A2 `param_cap_check`:
    선언된 write의 수치 param이 [레코드 필드 × 티어별 비율(A2 맵)] 캡을 초과하면 deny+안내
    (consent 보존 — silent-repair 금지·정책 원문이 '최대치 안내 후 진행 여부 문의').
    엔진=산술·비교만(정책 수치·필드·맵 전부 A2·[[03b]]). 맵 미기재 카드=무개입.

    ★2026-08-17 이설: 레코드 입력을 **정규식 파서(`parse_records`) → 격리 서브 formalize**
      로 바꿨다(사용자 지시 *"절대 결정론기에서 … 어떠한 정규식도 쓰면 안된다"*).
      전제는 x345 가 쟀다(서브가 레코드 값을 축자 복원·det n=1). 서브가 못 만들면 **빈 목록**
      이고 그러면 종전처럼 아무 판정도 하지 않는다(fail-open — 조용한 오답보다 무판정).
      ⚠판정(상한 비교)은 그대로 엔진이다 — x347 에서 격리가 **등급 축에서 부분 성공**에
      그쳤기 때문이다(대상 등급 정답·최저 등급 오답)."""
    import t2_search as _ts2
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in (specs or []):
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        v = str(args.get(aw.get("arg")) or "")
        if aw.get("prefix") and not v.startswith(aw["prefix"]):
            continue
        pn = sp.get("param")
        nested = args.get("arguments")
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except Exception:
                nested = {}
        d = nested if isinstance(nested, dict) else args
        try:
            val = float(str(d.get(pn)).replace(",", "").replace("$", ""))
        except Exception:
            continue
        _keys = [sp.get("record_key_field", "account_id")] + list(sp.get("record_require") or ())
        recs = []
        for m in messages:
            if getattr(m, "role", None) == "tool" and not getattr(m, "error", False):
                c = getattr(m, "content", None)
                if isinstance(c, str) and all(k in c for k in _keys):
                    recs += _ts2.sub_records(agent, la, UserMessage, c, _keys)
        if not recs:
            continue
        rec = recs[-1]
        pb = sp.get("pct_by") or {}
        pct = (pb.get("map") or {}).get(str(rec.get(pb.get("field", "card_type"), "")).strip())
        if pct is None:
            continue
        try:
            # ★정규식 제거(2026-08-17): 숫자·소수점만 남기는 것은 문자 필터로 충분하다.
            #   `re.sub(r"[^0-9.]", …)` 이 마지막 잔재였다([[59]] 강화판 — 엔진에 정규식 0).
            _raw = str(rec.get(sp.get("limit_field", "credit_limit")))
            lim = float("".join(ch for ch in _raw if ch.isdigit() or ch == "."))
        except Exception:
            continue
        cap = pct * lim
        if val > cap + 1e-6:
            fb = (sp.get("feedback") or "Error: [POLICY-CAP] {value} exceeds {cap}.")
            for k, vv in (("{value}", val), ("{cap}", cap), ("{pct}", "%d%%" % round(pct * 100)),
                          ("{limit}", lim)):
                fb = fb.replace(k, ("%g" % vv) if not isinstance(vv, str) else vv)
            return fb
    return None


def _wev_expand(tokens, flat, idv):
    """`require_tokens_any` 의 자리표시자를 채운다: `{id}` + **`{arg:NAME}`**.

    왜 `{arg:...}` 가 필요한가 (2026-08-28 · t7378 `task_074#s626729`):
    우리 비교기가 msg[55~57] 에서 `14.5` · `4.75` · `3.7` 을 내고 반환문이 축자로
    *"That signed total is the net correction for THIS account - use it as the credit amount"*
    라고 말했는데, 모델은 손님이 msg[59] 에서 말한 `9.00` · `1.50` · `1.50` 을 제출했다
    (손님은 `_err` 접미사 붙은 행만 세어 부분합을 냈다). **우리 값도 지시도 있었는데** 층위가
    뒤집혔다([[25]] env·손님은 외부 주장 · 우리 도구가 정본).
    => 필요한 술어: *"이 금액이 이 대화의 어떤 도구 출력에 우리가 계산한 값으로 실재하는가"*.
    `{id}` 만으로는 표현할 수 없어 **호출 인자 값**을 토큰에 넣을 수 있게 한다.

    ⚠숫자는 표기가 갈린다(`14.5` ↔ `14.50` ↔ `14`). 오차단을 피하려고 **여러 표기를 모두**
      만들어 하나라도 맞으면 통과시킨다. 엔진은 값을 만들지도 고르지도 않는다 - 모델이 보낸
      그 값을 **찾을 뿐**이다([[03b]] provenance 계열).
    ⚠비교기 반환문에는 계좌 id 가 **없다**(축자 확인). 그래서 id 공존을 요구하는
      `require_tokens` 가 아니라 이쪽(`require_tokens_any`)에 얹는다.
    """
    out = []
    for t in (tokens or []):
        cand = [str(t).replace("{id}", str(idv))]
        while True:
            grew = False
            for c in list(cand):
                i = c.find("{arg:")
                if i < 0:
                    continue
                j = c.find("}", i)
                if j < 0:
                    continue
                nm = c[i + 5:j]
                v = flat.get(nm)
                if v is None:
                    cand.remove(c)
                    grew = True
                    continue
                forms = {str(v)}
                try:
                    f = float(v)
                    forms.update(("%g" % f, "%.1f" % f, "%.2f" % f))
                    if f == int(f):
                        forms.add(str(int(f)))
                except Exception:
                    pass
                cand.remove(c)
                for fm in forms:
                    cand.append(c[:i] + fm + c[j + 1:])
                grew = True
            if not grew:
                break
        out.extend(cand)
    return out


def _wev_deny_msgs(messages, tc, specs):
    """★T2_WRITE_EVIDENCE: A2 `write_evidence_specs` — 선언된 write 전, 요구 토큰이 대상 id와
    **같은 도구 출력**(role=tool·env 생성물·user *발화*는 제외)에 공존해야 실행.
    ★출처(2026-07-31 [[23]] 교정): 근거는 태스크 포렌식이 아니라 **KB**다 —
    `doc_credit_cards_(general)_004`가 "look up the user's **resolved** disputes … to find the
    transaction_id values that need rewards adjustments"라고 자격을 정하고,
    `doc_bank_accounts_(general)_035`가 상태 taxonomy를 열거한다(BANK_FAVOR=크레딧 없음).
    즉 gold 없이 사전에 쓸 수 있는 규칙이었고, 실제로 gold 경유로 쓴 탓에 `RESOLVED` substring이
    은행-승소까지 통과시키는 결함이 있었다(→ `forbid_tokens`). 도메인-일반: 도구명/조건/토큰/문구
    전부 A2·엔진은 substring 공존 실재확인만([[03b]] provenance 계열·값 추출/생성 0). id는 호출 인자
    (중첩 JSON-문자열 포함=_args_dict 계열·모델 자신의 출력 파싱). id 못 읽으면 skip(false-block 회피)."""
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in specs:
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        if aw.get("arg"):
            v = str(args.get(aw["arg"]) or "")
            pref = aw.get("prefix")
            if pref and not v.startswith(pref):
                continue
        idk = sp.get("id_key")
        present = idk in args
        idv = args.get(idk)
        if idv is None and idk:
            for vv in args.values():                   # 중첩 JSON-문자열 인자(디스패처형 도구)
                if isinstance(vv, str) and idk in vv:
                    present = True
                    try:
                        idv = (json.loads(vv) or {}).get(idk)
                    except Exception:
                        pass
                if idv:
                    break
        # ★키 부재 vs 빈 값 구분 (2026-07-21 §2bc·054 t0 실측): 구판 `if not idv: skip`은
        #   `card_last_4_digits: ""` **빈-값 write를 무검사 통과**시켰다(false-block 회피 분기의
        #   구멍·gold와 유일한 diff가 빈 last4). 키 자체가 없는 변형 = skip 유지(변형 오차단 회피)·
        #   키가 실재하는데 값이 비면 = 불완전 write → deny(증거 요구 문구 그대로).
        if not (str(idv).strip() if idv is not None else ""):
            if not present:
                continue
            fb = sp.get("feedback") or "Error: [WRITE-EVIDENCE] required evidence not found for {id}."
            return _wev_fill(fb, "(missing — the argument was left empty)", messages,
                             sp.get("require_tokens_any") or [])
        # ★require_tokens_any (2026-07-23 C121·rall19 031.1 실측): 구판 AND-substring은 KB doc의
        #   무관 예시 숫자("(e.g., 1234, 4321)")와 도구명 substring이 한 출력에 우연 공존하면 날조값도
        #   통과(evidence-collision). any-토큰은 "{id}"를 값으로 치환한 라벨-값 인접 문자열 중 하나가
        #   도구 출력에 실재해야 통과 — 라벨·문구는 전부 A2·엔진은 치환+substring 대조만([[03b]]).
        # ★forbid_tokens (2026-07-31·[[23]] 감사): 증거 판정에 **부정 조건**이 필요하다.
        #   require_tokens는 substring이라 `"RESOLVED"` 하나로 `RESOLVED_CUSTOMER_FAVOR`뿐 아니라
        #   **`RESOLVED_BANK_FAVOR`(거래 정당·크레딧 없음)**까지 증거로 인정했다 — 막으려던 오염을
        #   한 갈래 열어둔 셈이다. 그렇다고 토큰을 `RESOLVED_CUSTOMER_FAVOR`로 좁히면 도구 출력
        #   형식(`Status: RESOLVED - approved`·026/028 실재)이 막혀 회귀한다. ⇒ 긍정 토큰은 넓게
        #   두고 **자격 없는 상태명을 A2가 선언**해 배제한다. 엔진은 집합 비포함만 본다(리터럴 0).
        tokens = sp.get("require_tokens") or []
        forbid = sp.get("forbid_tokens") or []
        # 2026-08-28 - `{arg:NAME}` 치환(위 `_wev_expand`). 중첩 JSON 인자도 편다.
        _flat = dict(args)
        for _v9 in list(args.values()):
            if isinstance(_v9, str) and _v9.strip().startswith("{"):
                try:
                    _in9 = json.loads(_v9)
                except Exception:
                    continue
                if isinstance(_in9, dict):
                    for _k9, _vv9 in _in9.items():
                        _flat.setdefault(_k9, _vv9)
        any_tokens = _wev_expand(sp.get("require_tokens_any") or [], _flat, idv)
        found = False
        found_any = not any_tokens
        blocked_by = []                 # ★어느 forbid 토큰이 막았나(리뷰 D: over-block 사후 귀속)
        for m in messages:
            if getattr(m, "role", None) != "tool":
                continue
            c = getattr(m, "content", None)
            c = c if isinstance(c, str) else str(c or "")
            if not found and str(idv) in c and all(t in c for t in tokens):
                hit = [t for t in forbid if t in c]
                if hit:
                    blocked_by.extend(h for h in hit if h not in blocked_by)
                else:
                    found = True
            if not found_any and any(t in c for t in any_tokens):
                found_any = True
            if found and found_any:
                break
        if blocked_by and not found:
            # ★리뷰 D: over-block 사후 귀속 — 이 deny가 **어느 자격 없는 상태 때문**인지 남긴다.
            #   RESOLVED_PARTIAL(보수적 차단)이 몇 건인지 가려야 판정이 뒤집힐 때 근거가 된다.
            print("[T2_WRITE_EVIDENCE] deny forbid=%s id=%s"
                  % (",".join(blocked_by), str(idv)[:40]), file=sys.stderr, flush=True)
        if not (found and found_any):
            fb = sp.get("feedback") or "Error: [WRITE-EVIDENCE] required evidence not found for {id}."
            return _wev_fill(fb, str(idv), messages, any_tokens)
    return None


def _wev_fill(fb, idtxt, messages, any_tokens):
    """WEV 피드백 치환: {id} + {evidence}. {evidence}=A2 any-토큰의 라벨 프리픽스({id} 제거분)가
    실재하는 도구-출력 **라인의 축자 인용**(2026-07-23 C122·rall19 031.0 실측: 정답이 도구출력에
    실재하는데 deny 피드백이 교정으로 안 이어져 6회 공전 — C116 "처방적 구체성만 유효 변수" 적용).
    엔진=라인 인용만(값 추출·생성·기입 0·문구는 A2 feedback 소관·[[03b]] present 계열)."""
    if "{evidence}" in fb:
        labels = [t.replace("{id}", "").strip() for t in (any_tokens or [])]
        labels = [lb for lb in labels if lb]
        ev = []
        if labels:
            for m in messages:
                if getattr(m, "role", None) != "tool":
                    continue
                c = getattr(m, "content", None)
                c = c if isinstance(c, str) else str(c or "")
                for ln in c.splitlines():
                    if any(lb in ln for lb in labels):
                        ln = ln.strip()
                        if ln and ln not in ev:
                            ev.append(ln)
        fb = fb.replace("{evidence}", " | ".join(ev)[:300])
    return fb.replace("{id}", idtxt)


def _ref_verify_deny(agent, la, UserMessage, messages, tc, specs):
    """★T2_REF_VERIFY (2026-07-24 C128/C129·결정론 참조-검증기): 선언된 write가 가리키는 레코드의
    **판별 속성**(예: merchant_name)이 손님 발화에 없으면 deny+처방(손님이 실제 말한 속성값 나열).
    근거: rall19-22 wrong-pick 8/8이 전부 손님 미언급 상점(cross-merchant 인접-행 전사 슬립)·검증기
    8/8 검출·false-block 0(C128). LLM 재선택(REF_ISO)은 gold→wrong 해로운 switch(C129)라 이 결정론
    검증기가 robust. 엔진=substring 대조만(LLM 0·값 추출/생성 0·[[03b]]·[[10]] 검증기=결정론). 도구명·
    필드·문구 전부 A2. id의 레코드 못 찾으면 skip(false-block 회피). 속성값이 손님 발화에 있으면 통과.
    ★한계(A2 note): merchant-absence는 cross-merchant만 검출·동일상점 내 오선택은 amount 필요(별 스펙)."""
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in (specs or []):
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        if aw.get("arg"):
            v = str(args.get(aw["arg"]) or "")
            if aw.get("prefix") and not v.startswith(aw["prefix"]):
                continue
        idk = sp.get("id_key")
        idv = args.get(idk)
        if idv is None and idk:                             # 중첩 JSON-문자열(디스패처형)
            for vv in args.values():
                if isinstance(vv, str) and idk in vv:
                    try:
                        idv = (json.loads(vv) or {}).get(idk)
                    except Exception:
                        pass
                if idv:
                    break
        idv = str(idv or "").strip()
        if not idv:
            continue
        field = sp.get("record_field", "merchant_name")
        # 1) 도구 출력(producer)서 idv 레코드 블록의 field 값 추출
        # ★2026-08-17 이설: 값 추출을 **정규식 → 격리 서브 formalize** 로 바꿨다(사용자 지시
        #   *"절대 결정론기에서 … 어떠한 정규식도 쓰면 안된다"*). 전제는 x345 가 쟀다(서브가
        #   레코드 값을 축자 복원·det n=1). 판정(손님 발화 대조)은 그대로 엔진이다 — C129 가
        #   *LLM 재선택*은 해롭다고 이미 닫았고, *검증*은 격리 시험용 궤적이 영속본에 없어
        #   미측정이므로 함부로 옮기지 않는다. 서브가 못 만들면 rec_val=None → **skip**(fail-open).
        import t2_search as _ts3
        rec_val, all_vals = None, set()
        for m in messages:
            if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
                continue
            c = getattr(m, "content", None)
            c = c if isinstance(c, str) else str(c or "")
            if field not in c:
                continue
            for row in _ts3.sub_records(agent, la, UserMessage, c, [idk, field]):
                _v = str(row.get(field) or "").strip()
                if not _v:
                    continue
                all_vals.add(_v)
                if rec_val is None and str(row.get(idk) or "").strip() == idv:
                    rec_val = _v
        if not rec_val:                                     # id의 레코드/필드 못 읽음 → skip
            continue
        # 2) 손님 발화(user)에 그 field 값이 있나. ★정확-문자열은 취약(레코드 "Marriott Hotels" vs
        #    손님 "Marriott hotel"=단복수/접미어 차이로 gold 오차단·rall22 031 실측). 유의미 토큰
        #    (길이≥min_tok·언어일반·도메인 리터럴 0) 하나라도 일치하면 언급으로 간주(false-block 회피
        #    우선=miss가 false-block보다 안전·다른 층이 슬립 재검). min_tok=A2(기본 5·"home"류 generic 제외).
        min_tok = int(sp.get("match_min_token", 5))
        utext = "\n".join(str(getattr(m, "content", "") or "") for m in messages
                          if getattr(m, "role", None) == "user").lower()

        def _mentioned(val):
            if not val:
                return False
            if val.lower() in utext:
                return True
            for tok in re.findall(r"[A-Za-z0-9]+", val):
                if len(tok) >= min_tok and tok.lower() in utext:
                    return True
            return False

        if _mentioned(rec_val):                             # 손님이 언급 → 통과
            continue
        # 3) 손님이 실제 언급한 (목록 내) 값들 = 처방 피드백용
        mentioned = sorted({v for v in all_vals if _mentioned(v)})
        fb = sp.get("feedback") or (
            "Error: [REF-VERIFY] the record you are filing ({id}) is a '{value}' entry, but the "
            "customer never mentioned '{value}'. The customer referred to: {mentioned}. Re-check "
            "which record they meant — read the listing and match on what the customer actually "
            "described — and file that one instead.")
        return (fb.replace("{id}", idv).replace("{value}", rec_val)
                .replace("{mentioned}", ", ".join(mentioned) if mentioned else "(none found)"))
    return None


def _write_arg_ground_deny(messages, tc, specs):
    """★T2_WRITE_ARG_GROUND (2026-07-22 §2bs·rall10 031 실측): A2 `write_arg_grounding` —
    선언된 write의 선언된 인자 **값**이 대화의 실측 근거(role=tool 출력 ∪ user 발화)에
    부분문자열로 실재해야 실행. 031: WRITE_EVIDENCE(선행-read 강제)는 통과했으나 뷰에 실재한
    5320 대신 '1234'를 기입해 dispute 제출 — read-강제와 값-전사는 별개 구멍. 도메인-일반:
    도구/인자/문구 전부 A2·엔진=값 실재확인만(substring·값 추출/생성 0·[[03b]] provenance 계열
    — SG_GROUND의 discoverable-write 확장). user 발화 포함=고객이 직접 준 값은 정당 근거.
    값 없음/키 없음=skip(false-block 회피·빈-값은 WEV 담당)."""
    name = getattr(tc, "name", None)
    args = _args_dict(tc)
    for sp in specs:
        if name != sp.get("applies_to"):
            continue
        aw = sp.get("applies_when") or {}
        if aw.get("arg"):
            v = str(args.get(aw["arg"]) or "")
            pref = aw.get("prefix")
            if pref and not v.startswith(pref):
                continue
        inner = {}
        for vv in args.values():                 # 중첩 JSON-문자열 인자(디스패처형 도구)
            if isinstance(vv, str) and vv.strip().startswith("{"):
                try:
                    j = json.loads(vv)
                    if isinstance(j, dict):
                        inner.update(j)
                except Exception:
                    pass
        for k2, v2 in args.items():
            if not isinstance(v2, (dict, list)):
                inner.setdefault(k2, v2)
        _markers = sp.get("arg_corpus_marker") or {}
        for ga in (sp.get("grounded_args") or []):
            gv = inner.get(ga)
            gs = str(gv).strip() if gv is not None else ""
            if not gs:
                continue
            # ★arg_corpus_marker (2026-07-22 §2bv·054 time_verified): 특정 arg는 marker를 포함하는
            #   출력에서만 grounding 검색(오매칭 방지). 054: time_verified는 "current time is"(get_current_time
            #   출력)에만 매칭 — 다른 레코드의 2023 날짜에 우연-통과 차단·병렬호출(결과 전)이면 미실재→deny.
            _mk = _markers.get(ga)
            found = False
            for m in messages:
                # ★corpus_roles (2026-08-03·015 실측·사용자 지시 "손님 이야기도 근거를 확인하게 하라"):
                #   구판 코퍼스 = 도구 출력 ∪ **손님 발화** ⇒ 손님이 말했다는 사실만으로 값이 통과한다.
                #   015: 손님이 "Crypto-Cash Back Card의 리퍼럴"이라 주장 → 그 이름이 손님 발화에
                #   실재하므로 통과 → 문서(gold=Platinum)와 대조 없이 그대로 인자에 실려 0점.
                #   ⇒ **인자별 권위 코퍼스를 A2가 선언**한다([[52]]: 권위자는 저작 시점에 정해진다).
                #   자기-사실(email·전화)=ledger/user 정당 · **정책 주장(카드 자격·요율)=문서만**.
                #   미선언 = 기존 ("tool","user") = 거동보존.
                _roles = tuple((sp.get("corpus_roles") or {}).get(ga) or ("tool", "user"))
                if getattr(m, "role", None) not in _roles:
                    continue
                c = getattr(m, "content", None)
                c = c if isinstance(c, str) else str(c or "")
                if _mk and _mk not in c:
                    continue
                if gs in c:
                    found = True
                    break
            if not found:
                fb = sp.get("feedback") or ("Error: [WRITE-GROUNDING] value '{val}' for {arg} "
                                            "does not appear anywhere in this conversation.")
                return fb.replace("{arg}", str(ga)).replace("{val}", gs)
    return None


# ─── ★T2_ARG_EMPTY (2026-08-11 C419·010 t2·[[64]] 의 짝 문제) ───
# 010 t2 는 `log_verification(..., date_of_birth="", ...)` 한 칸으로 죽었다. 값은 대화에 있다
#   (turn 9 레코드 축자 `date_of_birth: 04/17/1979`) — 직전에 손님이 *"생일은 채팅으로 주기
#   싫다"* 고 말했고 에이전트가 그것을 **"우리도 그 값이 없다"** 로 옮겼다.
# **아무도 안 막았다**: `_write_arg_ground_deny` 는 *"값 없음 = skip"*(:1149) 이라 빈 값이 구조적으로
#   통과하고, `grounded_args` 는 `time_verified` 하나뿐이다. 즉 *근거 없는 값*을 보는 규칙은 있는데
#   **필수 인자가 비었다**를 보는 규칙이 없었다(C416⒜ 가 남긴 질문).
# 격리 인과(x250·n=8·010 t2 궤적): `A_LIVE` EMPTY **8/8**(결손 완전 재현) · **`B_NAME`(빈 인자를
#   이름으로 짚는 거부) HIT 8/8** · `C_GENERIC`(이름 없는 거부) 0/8 · `D_FREE`(우리 문장 0) 0/8.
#   ⇒ 뺄셈도 문맥 축소도 아니고 **거부가 이름을 대는 것** 하나가 산다([[64]]·[[57]] 부정통제 포함).
# [[05]] 3질문: (1)도메인-특화 순증? No — 필수 인자 목록은 **env 도구 스키마**(`parameters.required`)
#   에서 기계 도출한다(A2 리터럴 0·ABox-swap 불변). 문구는 A2 `arg_empty.feedback`(슬롯만).
#   (2)유동판단 동결? No — 엔진은 *비었다*만 말하고 **값은 주지 않는다**; 무엇을 채울지(또는 아예
#   기록하지 않을지)는 모델이 고른다. (3)스캐폴드가 write 수행? No — 거부뿐이고 재발화는 모델.
# ⚠보수적으로 좁힌다: **키가 있고 값이 빈 문자열**일 때만. 키 부재·0·False 는 건드리지 않는다
#   (false-block 회피). 중첩 디스패처 인자도 WAG 와 같은 방식으로 unwrap 한다.
#
# ─── ★R5 수리 (2026-08-24 · refute_4 claim 1 CONFIRMED · 死배선 커버리지 공백) ───
# 결함(재관측): 이 게이트는 **디스패처 경유 write 에 구조적으로 발화할 수 없었다.** 이름이 두 번
#   어긋난다 — ⑴`_schema_required` 가 `agent.tools`(= 에이전트에게 **노출된** 21개)만 캐시하는데
#   발견형 도구는 그 목록에 없고, ⑵`_eff_tool_name` 이 `_\d+$` 를 지워 `..._4829` 를 레지스트리에
#   없는 철자로 만든다. 그래서 `req=[]` → `return None`. 세 번째 다리도 있었다 — 배치 페이로드
#   (`{"disputes":[{…},{…}]}`)는 구판 unwrap 이 `inner={'disputes':[…]}` 로 접어 필수 키를 아예
#   못 봤다. 실측(전 코퍼스 `.results.json.gz` 전량·13,534 sim): `[T2_ARG_EMPTY] deny` **79 발화가
#   전부 등록 도구** · **dispatched 0**. 빈 슬롯 자리는 문자열만 세도 **132 중 93 이 디스패처 경유**다.
# 수리(같은 레버의 사각지대만 닫는다·새 레버 0·[[62]]): ⒜필수 목록의 출처를 **env 레지스트리 전체**
#   (`env.tools`·`env.user_tools` = 발견형 포함)로 넓히고, ⒝조회 이름을 `_exact_tool_name`(= 호출이
#   싣고 온 **환경 자신의 철자**)로 바꾸고, ⒞페이로드를 프레임(최상위 + 중첩 사전 + 목록 원소)으로
#   분해해 배치 발행도 본다. 술어 자체는 그대로다 — **키가 있고 값이 빈 문자열**.
#   실측 효과(같은 코퍼스·같은 술어): 보이는 자리 **39 → 95**(문자열 슬롯 기준).
# ⚠비-문자열 빈 값은 **일부러 안 넣었다**(측정 후 기각): 코퍼스 전량에서 그런 자리는 37건이고
#   **전부 `None`** 이며(`partial_refund_amount` 15 · `card_action` 22 · APY 보고 6) 기본값이 있는
#   선택 인자로 보인다 — `None` 을 "비었다"로 세면 게이트가 정당한 생략을 막는다(false-block).
#   *"필수 슬롯이 비어 도착했다"* 의 닫힌 핵은 여전히 빈 문자열 하나다.
# ⚠[[70]] **무엇을 파는가**: 발화 자리가 늘어난 만큼 WEV 블록의 **공유 cap**(`T2_WEV_CAP`=8)을 더
#   먹는다. 같은 sim 에서 WEV·WAG·REF_VERIFY 와 예산을 나눠 쓰므로, 빈 슬롯이 잦은 궤적에서는
#   선행-read 거부가 그만큼 덜 나간다. 끄지 않고 계측 대상으로 남긴다([[60]]).
# [[05]] 3질문(수리분): (1)도메인-특화 순증? No — 넓힌 출처도 전부 프레임워크 API
#   (`Toolkit.get_tools`/`get_discoverable_tools`/`.tools`)이고 이름은 **호출이 싣고 온 값**이다
#   (철자 규칙 0·C279 가 경고한 접미사 추정은 유일할 때의 폴백으로만 남는다).
#   (2)유동판단 동결? No — 여전히 *비었다*만 말한다. (3)스캐폴드가 write 수행? No — 거부뿐.
ARG_EMPTY_FEEDBACK = ("Error: [ARG-EMPTY] the call to {tool} left the required argument(s) {args} "
                      "as an empty string. An empty string is not a value. Re-issue the call with "
                      "{args} filled in, or do not file the record at all.")


def _decl_tool_collections(holder):
    """도구 선언이 담긴 컬렉션들 — 목록/사전이면 그대로, 툴킷이면 프레임워크 API 로 연다.

    형태가 여럿인 것은 우리 사정이 아니라 프레임워크 사정이다: `agent.tools` 는 Tool 목록,
    `env.tools` 는 툴킷이고 `get_tools()` 는 `{이름: Tool}`, `.tools` 는 `{이름: 함수}` 다
    (`_arg_consumers` 가 이미 뒤쪽 형태를 쓴다). 어느 하나만 알면 조용히 반쪽만 본다.
    """
    if holder is None:
        return []
    if isinstance(holder, (list, tuple, set, dict)):
        return [holder]
    out = []
    for getter in ("get_tools", "get_discoverable_tools"):
        try:
            g = getattr(holder, getter, None)
            c = g() if callable(g) else None
        except Exception:
            c = None
        if isinstance(c, (list, tuple, set, dict)) and c:
            out.append(c)
    try:
        c = getattr(holder, "tools", None)
        if isinstance(c, (list, tuple, set, dict)) and c:
            out.append(c)
    except Exception:
        pass
    return out


def _decl_required(name, obj):
    """이 선언이 말하는 **필수 인자 이름들**. 모르면 None(= 선언 없음·무발화).

    1순위는 스키마(`parameters.required`)다. 스키마가 없는 형태(툴킷 `.tools` 는 함수를 담는다)
    에서는 **기본값 없는 파라미터**가 곧 required 라는 프레임워크 자신의 규약을 읽는다.
    """
    try:
        sc = getattr(obj, "openai_schema", None)
    except Exception:
        sc = None
    if isinstance(sc, dict):
        fn = sc.get("function") if isinstance(sc.get("function"), dict) else sc
        nm = (fn or {}).get("name") or name
        rq = (((fn or {}).get("parameters") or {}).get("required"))
        if rq is not None:
            return str(nm), [str(x) for x in (rq or [])]
        return str(nm), []
    if callable(obj):
        try:
            import inspect
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return None
        req = []
        for pn, p in sig.parameters.items():
            if pn == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is p.empty:
                req.append(str(pn))
        return str(name), req
    return None


def _schema_required_index(agent):
    """(정확한 이름 → 필수 인자) 와 (접미사 제거 이름 → 필수 인자·유일할 때만) 두 색인.

    출처는 셋이고 전부 환경/프레임워크다(도메인 리터럴 0):
      ⑴ `agent.tools`      — 에이전트에게 노출된 목록(구판의 **유일** 출처)
      ⑵ `env.tools`        — agent-side 레지스트리(발견형 포함)
      ⑶ `env.user_tools`   — user-side 레지스트리(발견형 포함)
    캐시는 **환경 객체 신원**으로 무효화한다(sim 이 바뀌면 레지스트리도 바뀐다).
    """
    env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
    key = id(env)
    cached = getattr(agent, "_t2_schema_req", None)
    if isinstance(cached, tuple) and len(cached) == 3 and cached[0] == key:
        return cached[1], cached[2]
    exact, strip_hits = {}, {}
    for holder in (getattr(agent, "tools", None),
                   getattr(env, "tools", None), getattr(env, "user_tools", None)):
        for coll in _decl_tool_collections(holder):
            items = (coll.items() if isinstance(coll, dict)
                     else [(getattr(o, "name", None), o) for o in coll])
            for nm, obj in items:
                got = _decl_required(nm, obj)
                if not got:
                    continue
                dn, rq = got
                if dn and dn not in exact:
                    exact[dn] = rq
    for dn, rq in exact.items():
        strip_hits.setdefault(_SUFFIX_RE.sub("", dn), []).append(dn)
    strip = dict((s, exact[v[0]]) for s, v in strip_hits.items() if len(v) == 1)
    try:
        agent._t2_schema_req = (key, exact, strip)
    except Exception:
        pass
    return exact, strip


def _schema_required(agent, name):
    """이 도구가 **필수**라고 선언한 인자 이름들 — 근거는 env 스키마뿐(도메인 리터럴 0).

    정확한 이름이 먼저다. 접미사를 뗀 철자는 **그 철자로 접히는 도구가 유일할 때만** 폴백으로
    쓴다 — 구판은 이 폴백 하나뿐이었고, 발견형 이름이 레지스트리에 접미사째 있어서 통째로 빗나갔다.
    """
    exact, strip = _schema_required_index(agent)
    nm = str(name or "")
    if nm in exact:
        return exact[nm] or []
    return strip.get(_SUFFIX_RE.sub("", nm)) or []


def _arg_frames(args, max_frames=48, max_depth=5):
    """검사할 **인자 프레임** 목록 — 최상위 인자 + 중첩 페이로드(사전·목록 원소).

    구판은 최상위 JSON-문자열 사전 하나만 평평하게 합쳤다. 그래서 디스패처가 **배치**로 실어
    보내면(`{"disputes":[{…},{…}]}`) 필수 키가 한 겹 더 안에 있어 통째로 안 보였다(t7335 실물).
    깊이·개수를 묶어 두므로(기본 ≤48 프레임·≤5 겹) 비용은 상수고, 해석은 하지 않는다 —
    자료구조를 펼치기만 한다.
    """
    out, queue = [], [(args, 0)]
    while queue and len(out) < max_frames:
        node, d = queue.pop(0)
        if isinstance(node, str) and node.strip()[:1] in ("{", "["):
            try:
                node = json.loads(node)
            except Exception:
                continue
        if isinstance(node, dict):
            out.append(node)
            if d < max_depth:
                queue.extend((v, d + 1) for v in node.values())
        elif isinstance(node, (list, tuple)) and d < max_depth:
            queue.extend((v, d + 1) for v in node)
    return out


def _arg_empty_deny(agent, tc, a2=None, applies_to=None):
    """선언된 write 의 **필수 인자가 빈 문자열**이면 그 이름을 대고 거부한다(값 0·지시 0)."""
    exact, eff = _exact_tool_name(tc), _eff_tool_name(tc)
    if applies_to and not ({exact, eff} & set(applies_to)):
        return None
    name = exact if _schema_required(agent, exact) else eff
    req = _schema_required(agent, name)
    if not req:
        return None
    frames = _arg_frames(_args_dict(tc))
    bad = [k for k in req
           if any(isinstance(f.get(k), str) and not f[k].strip() for f in frames)]
    if not bad:
        return None
    tpl = str((((a2 or {}).get("arg_empty") or {}).get("feedback")) or ARG_EMPTY_FEEDBACK)
    return tpl.replace("{tool}", str(name)).replace("{args}", ", ".join("'%s'" % b for b in bad))


# ─── ★T2_HAVE_VALUE (2026-07-23 C115·"have-value → act" 일반레버) ───
# 통합 통찰(시각054·CLI052·last-4 039·flail050 = 한 가족): 에이전트가 write W의 필수 인자 A를
#   *이미 대화에 갖고 있는데도* 재요청·재확인을 반복하고 W를 재시도 안 함(temp0 고착). 개별 fix
#   (met_template·verdict-gate·dedup) 대신 provenance 채널서 단일 엔진으로 수렴.
# 정의: ①A가 이전엔 미충족(에이전트 재요청 이력) ②지금 대화에 실재(producer 성공 출력) ③에이전트가
#   A를 또 재요청(W 미시도) → None-anchor 리마인더("A는 이미 있다·다시 묻지 말고 W를 그 값으로 지금
#   호출"). W는 에이전트가 emit(강제 0).
# [[05]] 3질문: (1)도메인-특화 순증? No — 도구/인자/신호/문구 전부 A2 `have_value_reask`(리터럴0=WAG의
#   grounded_args 재사용 계열). 엔진=이력/실재/재요청 판정만. (2)유동판단 동결? No — 어느 값을 쓸지는
#   에이전트 판단·엔진은 "이미 있으니 재요청 말고 써라"만(값 선택 안 함·유일할 때만 인용=자기 fetch 회상).
#   (3)스캐폴드가 write 수행? No — 넛지만·W는 에이전트가 emit(autofetch/ToolCall 아님·§1.5 준수).

def _producer_outputs(messages, marker):
    """producer 성공-실행을 표시하는 marker(A2 문자열)를 담은 role=tool 출력 content 목록(시간순).
    ★user-측 실행(call_discoverable_user_tool)도 결과는 role=tool 메시지 → assistant-툴콜 페어링에
    의존 않고 marker 매칭으로 포착(039 실측: last-4는 손님이 실행). marker='Executed: <tool>'류가
    성공 출력에만 있어 에러출력(§'has not been given')·재요청과 구분(도메인 리터럴은 A2·엔진=substring)."""
    if not marker:
        return []
    ml = str(marker).lower()
    outs = []
    for m in messages:
        if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
            continue
        c = _content_str(m)
        if ml in str(c).lower():
            outs.append(c)
    return outs


def _pattern_values(outs, pattern):
    """producer 출력서 value_pattern(A2 regex·그룹1) 매칭값(중복제거·순서보존). 없으면 []."""
    if not pattern:
        return []
    try:
        rx = re.compile(pattern, re.I)
    except Exception:
        return []
    seen, res = set(), []
    for c in outs:
        for mm in rx.finditer(str(c)):
            v = (mm.group(1) if mm.groups() else mm.group(0)).strip()
            if v and v not in seen:
                seen.add(v)
                res.append(v)
    return res


HAVE_VALUE_FEEDBACK_DEFAULT = (
    "[HAVE-VALUE] You ALREADY have '{arg}' in this conversation ({producer} has been run and its "
    "result is in the tool output above, and the customer has stated it){valclause}. Do NOT ask "
    "for it again and do NOT re-run any lookup for it. Call {write} NOW using that value, then "
    "continue to the next item.")


def _have_value_reask_fb(am, messages, specs):
    """★have-value→act (C115): (spec 순회) W 미시도 ∧ producer 값 실재(marker) ∧ (이전+지금) 재요청
    → None-anchor 리마인더 문구(또는 None). 값 선택 안 함(value_pattern이 유일값 낼 때만 자기-fetch 인용)."""
    if not specs:
        return None
    cur_calls = {_eff_tool_name(tc) for tc in (getattr(am, "tool_calls", None) or [])}
    cur_text = str(getattr(am, "content", "") or "").lower()
    for sp in specs:
        W = sp.get("write")
        producer = sp.get("producer")
        A = sp.get("arg")
        marker = sp.get("producer_marker") or producer
        signals = [str(s).lower() for s in (sp.get("reask_signals") or [])]
        if not (W and A and signals):
            continue
        if W in cur_calls:                                   # 이미 W 호출 중 → 넛지 불요
            continue
        outs = _producer_outputs(messages, marker)           # ② 값 실재(producer 성공 출력·user측 포함)
        if not outs:
            continue
        # ① 이전 이력: committed 이전 assistant 발화에 재요청 신호(정당한 첫-질문서 오발 방지)
        prior = any(getattr(m, "role", None) == "assistant"
                    and any(s in str(getattr(m, "content", "") or "").lower() for s in signals)
                    for m in messages)
        if not prior:
            continue
        # ③ 지금 턴: A 재요청(prose 신호) 또는 producer 재호출 (그리고 위에서 W 미시도 확인됨)
        if not (any(s in cur_text for s in signals) or (producer in cur_calls)):
            continue
        vals = _pattern_values(outs, sp.get("value_pattern"))  # 인용은 옵션(기본=value-free)
        valclause = (" — its value is %s" % vals[0]) if len(vals) == 1 else ""
        fb = sp.get("feedback") or HAVE_VALUE_FEEDBACK_DEFAULT
        # ★값 미유일인데 문구가 {value} 인라인 참조 → 빈 치환(깨진 문구·''날조) 방지: 일반 폴백
        if "{value}" in fb and len(vals) != 1:
            fb = HAVE_VALUE_FEEDBACK_DEFAULT
        return (fb.replace("{arg}", str(A)).replace("{producer}", str(producer or marker))
                  .replace("{write}", str(W)).replace("{valclause}", valclause)
                  .replace("{value}", vals[0] if len(vals) == 1 else ""))
    return None


# ─── ★T2_VALUE_ACQUIRE (C119·2026-07-23·8-task per-step 포렌식): "값 획득 경로 표면화" ───
# have-value의 *앞단계*: agent가 write W의 인자 A를 재요청하는데 ①값이 아직 대화에 없고(producer
#   성공출력0) ②손님이 직접 제공 못 함(카드 없음) ③획득 도구 P(get_card_last_4_digits)를 손님에게
#   *give 안 함*(give_discoverable_user_tool(P) 미호출) → 손님이 값을 못 얻어 재요청 무한(031hv·039 실측).
# 넛지: "이 값은 손님이 P로 직접 조회해야 함 — give_tool로 P를 손님에게 주고 실행을 안내하라."
# [[05]]: 도구/문구=A2(value_acquisition_specs)·엔진=재요청∧값미실재∧give미실행 판정만·리터럴0·넛지만.

def _tool_given(messages, give_tool, acquire_tool):
    """committed서 give_tool(예: give_discoverable_user_tool)로 acquire_tool을 손님에게 준 적 있나."""
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            if getattr(tc, "name", "") == give_tool or _eff_tool_name(tc) == give_tool:
                a = _args_dict(tc)
                vals = " ".join(str(v) for v in a.values())
                if acquire_tool in vals or a.get("discoverable_tool_name") == acquire_tool \
                        or a.get("agent_tool_name") == acquire_tool:
                    return True
    return False


GIVE_REQUIRED_FEEDBACK = (
    "Error: [GIVE-REQUIRED] the customer tried to run `{tool}` and the environment refused it "
    "because you have not handed that tool to them. It is a customer-side tool: it does nothing "
    "until you give it. Do this now, before anything else, exactly as written:\n"
    "    give_discoverable_user_tool(discoverable_tool_name=\"{tool}\")\n"
    "Then tell the customer to run it again. Repeating your previous message will not change "
    "anything - the environment refused the same call {n} time(s) already.")


def _give_required_fb(messages, orch):
    """손님이 실행해야 하는 도구를 **아직 넘겨주지 않았다** → 정확한 호출을 지목한다.

    ★2026-08-26 신설. 왜(t7356 실측·사용자 지적): 017 은 도구 선택도 인자도 **gold 와 같고**
      마지막 호출 형태 하나에서 갈린다. 궤적 축자 —
        t7348(통과·91msg)  msg36~50 손님 호출 6회 전부 거절 → **msg55/57 give** → msg60 성공
        t7356(실패·61msg)  같은 거절 4회 → **give 를 한 번도 안 부른 채** 손님이 대화를 끝냈다
      즉 통과한 런도 19 메시지를 헤매다 우연히 도달한 것이다. 표적 수(t7356 전수):
      **017 미달 1 · 055 미달 1 · 057 미달 2(give 호출 0회)** = 3 태스크.

    ⚠env 는 이미 축자로 시킨다 — *"The agent must first use `give_discoverable_user_tool` to give
      this tool to you."* 그런데 모델은 017 에서 그것을 **4번 받고도** 안 했고 057 은 한 번도 안 했다.
      다른 것은 **채널**이다: env 의 그 문장은 도구 결과로 수동적으로 읽히고, 이 거절은 재생성
      채널로 나가 그 턴을 교체하고 다시 내게 한다(원장 C413·C414·[[64]]).
    ⚠목록은 **env 레지스트리의 user-side 에서 도출**한다(`registry_from_env` 의 둘째 반환) —
      A2 저작 0 · 도메인 리터럴 0([[05]]). 레지스트리를 못 얻으면 무발화(fail-open).
    ⚠엔진이 **실행하지 않는다**: `give` 는 변이 도구이고 표적 셋이 전부 `basis=['DB']` 라,
      자동 호출은 우리 층이 gold 가 요구하는 상태 변경을 대신 수행하는 것이 된다([[05]]③·[[03b]]).
      지목만 하고 부르는 것은 모델이다([[62]]③④).
    """
    try:
        import t2_axis_levers as _AX
    except Exception:
        return None
    user_reg = set()
    for cand in (orch, getattr(orch, "_t2_orch", None)):
        if cand is None:
            continue
        try:
            _a, _u = _AX.registry_from_env(cand)
        except Exception:
            continue
        if _u:
            user_reg = set(_u)
            break
    if not user_reg:
        return None
    tried = collections.Counter()
    for m in (messages or []):
        if str(getattr(m, "role", "")) not in ("user", "assistant"):
            continue
        for tc in (getattr(m, "tool_calls", None) or []):
            if str(getattr(tc, "name", "")) != "call_discoverable_user_tool":
                continue
            x = str((_args_dict(tc) or {}).get("discoverable_tool_name") or "")
            if x in user_reg:
                tried[x] += 1
    for x, n in tried.most_common():
        if not _tool_given(messages, "give_discoverable_user_tool", x):
            return GIVE_REQUIRED_FEEDBACK.format(tool=x, n=n)
    return None


def _call_form_repair(messages, orch):
    """3단계 ③ — **래퍼만** 바꿔 다시 부를 재료. 내용(도구·인자)은 모델 것 축자다.

    ★사용자 확정 2026-08-26(축자): *"dispatcher 를 안부르는건 **호출 형식** 문제이다. 형식과
      내용이 충돌할 때, **형식을 엔진으로 바꾸는 것은 문제가 안된다.** … 그래도 안되면, 엔진이
      호출 형식을 바꿔서 직접 부르는 것이다."*
      경계가 이것으로 확정된다 — 내용(어느 도구·어떤 인자)은 LLM, **형식(어느 래퍼로 부르나)은
      엔진**이다([[52]] 엔진=이론·LLM=해석 · 저장소 선례 `t2_callable_hint` = *"부를 수 있는
      형태로 말하게 한다"*). ⛔이 함수가 X 나 arguments 를 **바꾸면 그 순간 [[03b]] 위반**이다.
      래칫이 그 축자 보존을 검정한다.

    반환 = (X, args) — 손님-측 도구 X 를 넘겨줄 때 실을 인자. 인자는 **손님이 시도한 그 호출에서
    복사**하고, 없으면 빈 dict(그 형태를 env 가 받는다·t7348 통과본 축자). 없으면 None.
    """
    try:
        import t2_axis_levers as _AX
    except Exception:
        return None
    user_reg = set()
    for cand in (orch, getattr(orch, "_t2_orch", None)):
        if cand is None:
            continue
        try:
            _a, _u = _AX.registry_from_env(cand)
        except Exception:
            continue
        if _u:
            user_reg = set(_u)
            break
    if not user_reg:
        return None
    want = None
    for m in (messages or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            if str(getattr(tc, "name", "")) != "call_discoverable_user_tool":
                continue
            a = _args_dict(tc) or {}
            x = str(a.get("discoverable_tool_name") or "")
            if x in user_reg:
                want = (x, a.get("arguments"))     # ★모델이 실은 것 그대로
    if not want:
        return None
    x, inner = want
    if _tool_given(messages, "give_discoverable_user_tool", x):
        return None
    out = {"discoverable_tool_name": x}
    if isinstance(inner, str) and inner.strip() and inner.strip() != "{}":
        out["arguments"] = inner               # 축자 전달 — 파싱도 재작성도 하지 않는다
    return (x, out)


VALUE_ACQUIRE_FEEDBACK_DEFAULT = (
    "[VALUE-ACQUIRE] The customer cannot provide '{arg}' directly, and it is NOT in the account "
    "records. It must be retrieved by the CUSTOMER running {acquire_tool}. Stop re-asking — use "
    "{give_tool} to give {acquire_tool} to the customer now, then have them run it and read the "
    "value from its output, then proceed with {write}.")


def _value_acquire_fb(am, messages, specs, a2=None, executed=None):
    """★give 표면화: (spec) W 재요청 ∧ 값 미실재(producer 출력 0) ∧ acquire_tool을 give 안 함 → 넛지.
    반환=문구 or None. have-value(값 실재)와 상보(값 미실재+획득경로 미실행)·중복 회피(값 있으면 skip).

    ★E3-②(2026-08-06·022 실측): 이 레버는 표적(`acquire_tool`)을 **우리가** 고르는 push 레버다.
    그 표적을 **지금 돌고 있는 절차가 금지**하고 있으면 말하지 않는다 — 022는 이 조건이 없어서
    "지금 넘겨라"와 "정책이 금지한다"를 같은 턴에 세 번 주고받았다. 판정은 t2_speak(선언만 읽는
    도메인-일반 규칙·`T2_SPEAK_PROHIBIT=1`일 때만). 표적을 문자열에서 파싱하지 않고 여기서
    **스코프에 있는 변수**로 넘기는 것이 요점이다(계기가 표적을 반대로 지목한 전례·설계서 §4.2)."""
    if not specs:
        return None
    cur_text = str(getattr(am, "content", "") or "").lower()
    cur_calls = {_eff_tool_name(tc) for tc in (getattr(am, "tool_calls", None) or [])}
    for sp in specs:
        W = sp.get("write")
        arg = sp.get("arg")
        acq = sp.get("acquire_tool")
        give = sp.get("give_tool")
        signals = [str(s).lower() for s in (sp.get("reask_signals") or [])]
        if not (W and arg and acq and give and signals):
            continue
        # ★C4 철회(2026-08-05): 048의 last-4 우회를 막으려고 "선언된 write가 시도된 적이 있어야"를
        #   걸었더니 **원래 표적까지 침묵**했다 — 053에서 이 넛지는 dispute 도구를 해제하기 20 스텝 전에
        #   발화해야 옳다(손님이 값을 얻어와야 그 다음이 있다). 048과 053을 가르는 것은 "이 태스크가
        #   그 write로 가는가"인데 그건 열린 술어라 우리 몫이 아니다([[22]]). 표적을 좁히는 닫힌 술어를
        #   찾기 전까지 조건을 걸지 않는다 — 대신 x102로 우회 비용(048=10 메시지)을 계속 잰다.
        # ★그 "닫힌 술어"가 이것이다(2026-08-06): **선언이 그 도구를 금지했는가**. 열린 술어("이
        #   태스크가 그 write로 가는가")를 찾다 실패한 자리를, 절차 선언이 이미 답하고 있었다.
        #   오프라인 전수(x104 §C): 침묵 = 022의 3발뿐·048/051/035/053의 12발은 유지(over-block 0).
        # ★위치(2026-08-06 rev2·라이브 `20260806c` 실측 교정): 초판은 이 검사를 스펙 루프 **맨 앞**에
        #   뒀다. 거동은 같지만(어차피 침묵) **계기가 거짓말을 했다** — 레버가 애초에 말할 생각이 없던
        #   턴까지 "침묵"으로 세어 98건이 찍혔고, 실제 억제는 022뿐이었다. 조건 ①②③ 뒤로 옮겨
        #   **말하기로 결정된 것만** 침묵시키고 그것만 기록한다([[08]] 계기가 신호를 부풀리면 안 된다).
        # ① 값이 이미 있으면(producer 성공출력) → have-value 관할·skip
        if _producer_outputs(messages, sp.get("producer_marker") or ("Executed: " + acq)):
            continue
        # ② acquire_tool을 이미 give했으면(031 base) → skip(경로 이미 열림)
        if _tool_given(messages, give, acq):
            continue
        # ③ 재요청: 이전 or 지금 assistant가 arg 재요청(prose 신호), 그리고 W 미완(값 없으니 당연)
        prior = any(getattr(m, "role", None) == "assistant"
                    and any(s in str(getattr(m, "content", "") or "").lower() for s in signals)
                    for m in messages)
        cur_reask = any(s in cur_text for s in signals)
        if not (prior or cur_reask):
            continue
        # ④ 말하기로 결정됐다 — 그런데 지금 돌고 있는 절차가 이 표적을 금지하는가([[22]] 닫힌 술어).
        try:
            import t2_speak as _spk
            if _spk.prohibits_target(a2, executed, acq, lever="VALUE-ACQUIRE", messages=messages):
                continue
        except Exception:
            pass
        fb = sp.get("feedback") or VALUE_ACQUIRE_FEEDBACK_DEFAULT
        return (fb.replace("{arg}", str(arg)).replace("{acquire_tool}", str(acq))
                  .replace("{give_tool}", str(give)).replace("{write}", str(W)))
    return None


def _stale_call_ids(am, messages, wtools):
    """★T2_STALE_STRIP (over-action 억제): am.tool_calls 중 strip 대상 id 집합.
    ①같은 am 내 완전중복(동일 eff+args 2회+·read/write 공통) ②committed서 이미 성공한 *write* 재호출.
    read의 committed-재조회는 미포함(상태변화 존중·over-fire 방지). 도메인 리터럴 0(eff+args 대조).

    ★A1/OL-17 (t7336 §6.1·2026-08-22): `ok_ids` 가 **성공을 `m.error` 로만** 판정했다. tau2 env 는
      에러를 플래그 없이 content 로만 준다 — 해당 sim 의 `Error` 접두 tool 메시지 **15건 중 14건이
      `error=False`** 였다(실측). 그래서 **실패한 write 가 `done_w` 에 편입**되고 같은 인자의 정당한
      재시도가 085#1 에서 **8회**·079#0 1회·073#0 3회 제거됐다. 술어는 F8(`t2_prekb_patch._argprod_hits`)
      가 이미 쓰는 그것이고 정본 함수는 `t2_resolve._result_ok` 다 — **사본 0**([[67]])·여기서 재사용.
    ⚠[[70]] 무엇을 파는가: `done_w` 가 좁아지므로 **중복 write 통과가 늘 수 있다**. 다음 런 포렌식은
      `[T2_STALE_STRIP] dropped` 수와 DUP(동일-인자 write 재실행) 수를 **짝으로** 센다.
    ⚠import 실패는 fail-open: 규칙①(같은 턴 완전중복)만 남기고 규칙②를 끈다(틀린 `done_w` 로
      정당한 재시도를 지우는 것이 결손의 본체이므로 **모름은 안 지운다**)."""
    try:
        import t2_resolve as _rz_ok          # 정본 술어 재사용([[67]] 사본 0)
        _ok = _rz_ok._result_ok
    except Exception as _se:                 # noqa: BLE001 — fail-open(규칙② 비활성)
        print("[T2_STALE_STRIP] result-ok 술어 미가용 — 규칙②(완료 write) 비활성: %r" % (_se,),
              file=sys.stderr, flush=True)
        _ok = None
    ok_ids = ({getattr(m, "id", None) for m in messages
               if getattr(m, "role", None) == "tool" and _ok(m)}
              if _ok is not None else set())
    done_w = set()
    for m in messages:
        if getattr(m, "role", None) == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                if getattr(tc, "id", None) in ok_ids:
                    eff = _eff_tool_name(tc)
                    if eff in wtools or getattr(tc, "name", "") in wtools:
                        done_w.add((eff, _call_key(tc)))
    seen, stale = set(), set()
    for tc in (getattr(am, "tool_calls", None) or []):
        eff = _eff_tool_name(tc)
        key = (eff, _call_key(tc))
        is_w = eff in wtools or getattr(tc, "name", "") in wtools
        if key in seen:
            stale.add(id(tc))
        elif is_w and key in done_w:
            stale.add(id(tc))
        seen.add(key)
    return stale


def _resolver_directive(a2, tc, k, s):
    """★B-max① (2026-07-11): fab 인자 k에 A2 resolver_path가 있으면 지시형 피드백 문구, 없으면 None.
    resolver_path=[in_arg, producer, field] — 호출 인자에 in_arg 값이 실재(알려진 입력)할 때만 그 producer/
    입력/필드를 지목. 없으면 None → 호출측이 기존 중립 문구로 폴백. 엔진=도메인-일반(A2 소비만·값은 안 읽음=
    autofetch 아님·문구만). default_specs(원리-디폴트 silent 치환)와 regen_resolver_specs(directive-only) 모두 조회."""
    if not a2:
        return None
    specs = list(a2.get("default_specs") or []) + list(a2.get("regen_resolver_specs") or [])
    if not specs:
        return None
    nm = getattr(tc, "name", None)
    d = _args_dict(tc)
    for sp in specs:
        if sp.get("arg") != k:
            continue
        applies = sp.get("applies_to") or []
        if applies and nm not in applies:
            continue
        path = sp.get("resolver_path")
        if not path or len(path) < 3:
            continue
        in_arg, producer, field = path[0], path[1], path[2]
        in_val = d.get(in_arg)
        if not in_val:            # 알려진 입력이 없으면 producer/입력 지목 불가 → 중립 문구로
            return None
        return REGEN_FEEDBACK_DIRECTIVE.format(
            k=k, s=s, producer=producer, in_arg=in_arg, in_val=in_val, field=field)
    return None


def _ctx_from_messages(msgs):
    parts = []
    for m in msgs:
        r = getattr(m, "role", None)
        c = getattr(m, "content", None)
        if r in ("user", "tool") and c is not None:
            parts.append(c if isinstance(c, str) else str(c))
    return " ".join(parts).lower()


def _first_fab_call(am, ctx, hints=DEFAULT_ARG_HINTS, exclude=frozenset(),
                    selectors=frozenset()):
    """am.tool_calls 중 첫 날조 (tc, k, s) 또는 None.
    exclude (PROV-RESCUE-PERARG ①): rescue-스킵된 (id(tc), k, s) 집합 — 해당 인자만 건너뛰고
    같은 호출의 다음 fab 인자·다음 호출을 계속 스캔 (구현: per-call 첫 인자 반환+break의 입도 구멍 봉합).

    selectors (★2026-08-24 R3): **도구-선택자 파라미터 이름 집합**(`_selector_arg_names` 로 env
    스키마에서 도출). 그 슬롯의 값은 *데이터 operand* 가 아니라 *어느 도구를 부를지*이므로
    날조-스캔 대상이 아니다 — 같은 예외가 `_prov_scan_args` 에는 2026-08-15 부터 있었으나
    **치환하는 경로인 이 함수에는 없어서** T2_GROUND 가 뱅킹에서 371/371 오작동했다
    (정답 도구명 산출 0/371). 기본값 ∅ = 종전 거동 바이트 동일(집합을 안 넘긴 호출자 보호)."""
    for tc in (getattr(am, "tool_calls", None) or []):
        for k, v in _args_dict(tc).items():
            if k in selectors:
                continue
            if not _hint_hit(k, hints):
                continue
            for val in _flatten(v):
                s = str(val).strip()
                if len(s) < 4 or _ctx_has(s, ctx):
                    continue
                if (id(tc), k, s) in exclude:
                    continue
                return (tc, k, s)
    return None


# ─── ★L3 origin-prov (T2_PROV_ORIGIN=1·v3.2·t97/t96 first-mention 세탁 차단) ───
# 원리(보편·도메인 리터럴 0): 에이전트가 *스스로 제안*한 식별값을 user가 yes로 복창하면
# 값∈ctx가 되어 기존 prov를 통과(확인-세탁·A1_V3_PROBE_FORENSIC §2-1). 차단 조건 =
#   값의 최초 등장 role == assistant  ∧  어떤 tool 출력에도 부재(tool-never).
# tool-never가 리뷰 caveat (a)(getter가 늦게 확인한 값=정당)를 처리하고,
# first-role이 caveat (b)(user-first 명시값 t43=정당)를 처리한다.

def _origin_role(s, msgs):
    """값 s의 최초 등장 (role, tool_ever). role∈{user,assistant,tool,None}."""
    first = None
    tool_ever = False
    sl = str(s).strip()
    for m in msgs:
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        if _ctx_has(sl, c.lower()):
            r = getattr(m, "role", None)
            if r == "tool" and not getattr(m, "error", False):
                tool_ever = True
            if first is None and r in ("user", "assistant", "tool"):
                first = r
    return first, tool_ever


def _first_origin_fab(am, msgs, hints=DEFAULT_ARG_HINTS, exclude=frozenset()):
    """write 인자 중 origin-fab (tc, k, s) 또는 None — 값∈ctx인데 assistant-first ∧ tool-never.
    스코프: 주소류 free-text 인자만(_is_addr_arg·id류는 기존 rescue/fab 관할·Δspurious 보수)."""
    for tc in (getattr(am, "tool_calls", None) or []):
        for k, v in _args_dict(tc).items():
            if not _hint_hit(k, hints):
                continue
            if not _is_addr_arg(k):
                continue
            for val in _flatten(v):
                s = str(val).strip()
                if len(s) < 4 or (id(tc), k, s) in exclude:
                    continue
                first, tool_ever = _origin_role(s, msgs)
                if first == "assistant" and not tool_ever:
                    return (tc, k, s)
    return None


ORIGIN_FEEDBACK = (
    "Error: [PROVENANCE-ORIGIN] argument '{k}'='{s}' was first introduced by YOU (the assistant), "
    "not by the user or any tool output — a user's yes to your proposal does not make it real. "
    "Fetch the actual value from the relevant records (call the getter that produces it), or ask "
    "the user an OPEN question to provide the value themselves. Then re-emit the call."
)


# ─── ★FAB_STRIP 해소-지목 (2026-08-21 P4·[[64]]·t7335 halfB 079) ───
# 구판 차단 노트("items whose supporting records could not be verified were not processed")는
# **무엇이 미근거인지·무엇을 읽으면 풀리는지 무지목**이라 대화가 접혔다(079 [26]/[32] — user가
# "Did the freeze actually go through?"로 혼란·[[64]] 동형: 이름 없는 거부가 자기 원인을 재생산).
# 여기서 [[64]]의 두 칸을 채운다: ⑴어느 호출의 어느 인자·값이 미근거였나 ⑵어느 read 가 그 값을
# 내나. 도구명 출처는 **선언뿐** — ① a2["arg_source_reads"](인자명→원천-read 목록·env 레지스트리
# desc 축자에서 1회 저작·[[72]]·목록 순서=해소 순서) ② 폴백 a2["relations"]["by_tool"][eff]
# ["requires"](C586 requires_reads 선언). 엔진은 선언 조회·나열만(도메인 리터럴 0·선택 0·[[10]]).
# 판정(strip 집합)은 불변 — 문면만 보강이라 플래그 없이 상시([[64]] 의무·본문 억제 경로 아님).

def _fab_fix_note(stripped, a2):
    """stripped = [(eff명, [(인자, 값), ...]), ...] → 노트 꼬리 문장(영어=C125). 빈 입력 = ""."""
    amap = (a2 or {}).get("arg_source_reads") or {}
    rel = ((a2 or {}).get("relations") or {}).get("by_tool") or {}
    parts = []
    for eff, kvs in (stripped or []):
        wrongs, reads = [], []
        for k, v in (kvs or []):
            wrongs.append("%s='%s'" % (k, str(v)[:40]))
            for r in (amap.get(str(k)) or []):
                if not str(r).startswith("_") and r not in reads:
                    reads.append(r)
        if not wrongs:
            continue
        if not reads:   # 인자 선언이 없으면 write-도구 선언(requires_reads)으로 폴백
            reads = list((rel.get(eff) or {}).get("requires") or [])
        msg = ("%s was NOT executed: %s does not appear in any record read in this conversation"
               % (eff, " / ".join(wrongs)))
        if reads:
            msg += ("; to fix this, first read the real value with %s, then re-issue the call "
                    "with a value copied from that tool's output" % ", then ".join(reads))
        parts.append(msg)
    return (" " + "; ".join(parts) + ".") if parts else ""


# ─── ★v3.2 CONSISTENCY 가드 (T2_CONSISTENCY=1·L10 멤버십 t35형 + G-noop t71형) ───
# 검사 재료 = 에이전트 자신이 fetch한 tool 출력(grounded·규칙0)·키/도구명 = 전부 A2(eplan spec+
# confirm gates)·엔진 도메인 리터럴 0. 성격=예방(정직·PROBE_FORENSIC: t35/t71 실궤적은 문제
# write 미발행 — 실측 회복은 부분). Δspurious 보수: 컨테이너 상세 미조회면 침묵(read 강제=L2 몫).

def _record_for(msgs, id_key, id_val, lenient=True):
    """out[id_key]==id_val인 최신 tool 출력 record dict (없으면 None)."""
    if not id_key or not id_val:
        return None
    tgt = str(id_val).strip().lower()
    for out in _parse_tool_outputs(msgs, lenient=lenient):  # 최근 우선
        if isinstance(out, dict) and str(out.get(id_key, "")).strip().lower() == tgt:
            return out
    return None


def _ids_at_path(record, path):
    """record의 path=[container_key, id_field] 위치서 멤버 id 집합(lower). list/dict 컨테이너 지원."""
    if not (isinstance(record, dict) and path and len(path) >= 2):
        return set()
    seq = record.get(path[0])
    if isinstance(seq, dict):
        seq = list(seq.values())
    ids = set()
    for it in (seq or []):
        if isinstance(it, dict) and it.get(path[1]) is not None:
            ids.add(str(it.get(path[1])).strip().lower())
    return ids


def membership_violation(d, spec, msgs):
    """L10: d[items_key] 각 id ∈ d[entity_key] record 멤버 집합인지.
    위반 시 (bad_ids, oid, hint_oid|None) — hint = bad를 실제 담은 다른 grounded record."""
    ent_key = (spec or {}).get("entity_key")
    mem_key = (spec or {}).get("items_key")
    path = (spec or {}).get("items_id_path")
    if not (ent_key and mem_key and path):
        return None
    oid = d.get(ent_key)
    mems = d.get(mem_key)
    if not oid or not isinstance(mems, list) or not mems:
        return None
    rec = _record_for(msgs, ent_key, oid)
    if rec is None:
        return None
    ids = _ids_at_path(rec, path)
    if not ids:
        return None
    bad = [str(m) for m in mems if str(m).strip().lower() not in ids]
    if not bad:
        return None
    hint = None
    for out in _parse_tool_outputs(msgs):
        if (isinstance(out, dict) and out.get(ent_key)
                and str(out.get(ent_key)).lower() != str(oid).lower()
                and any(str(b).strip().lower() in _ids_at_path(out, path) for b in bad)):
            hint = str(out.get(ent_key))
            break
    return (bad, str(oid), hint)


def _leaf_scalar_map(obj, out=None):
    """중첩 record의 leaf 스칼라 {말단키(lower): str값} — first-win."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _leaf_scalar_map(v, out)
            elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
                out.setdefault(str(k).lower(), str(v).strip())
    elif isinstance(obj, list):
        for v in obj:
            _leaf_scalar_map(v, out)
    return out


def noop_write(d, spec, msgs, min_match=3):
    """G-noop(t71형): write의 스칼라 인자 전부가 대상 record 현재값과 동일(매칭 ≥min_match)
    = 아무것도 안 바꾸는 write → 참조 오바인딩 신호. 하나라도 다르면 정상(False)."""
    ent_key = (spec or {}).get("entity_key")
    if not ent_key:
        return False
    oid = d.get(ent_key)
    if not oid:
        return False
    rec = _record_for(msgs, ent_key, oid)
    if rec is None:
        return False
    leaves = _leaf_scalar_map(rec)
    matched = 0
    for k, v in d.items():
        if k == ent_key or v is None or isinstance(v, (list, dict)):
            continue
        cur = leaves.get(str(k).lower())
        if cur is None:
            continue
        if str(v).strip().lower() == cur.lower():
            matched += 1
        else:
            return False
    return matched >= min_match


CONS_MEMBER_FEEDBACK = (
    "[CONSISTENCY] item(s) {bad} do not belong to {ent}='{oid}' according to its latest fetched "
    "details.{hint} Re-check which record actually contains the item(s) the user means, then "
    "re-emit a corrected call."
)

CONS_NOOP_FEEDBACK = (
    "[CONSISTENCY] this call would change nothing — every requested value equals the record's "
    "current value. This usually means the WRONG record was selected. Re-check by CONTENT which "
    "record the user means (compare the records you have listed/read; do not rely on list order), "
    "or ask the user to identify it by content. Then re-emit."
)


# ─── ★v5 L2R read-all 강제 (T2_READALL=1·A1_V2_NT2_FORENSIC §4-1) ───
# 근거 [M]: nt2 flip 분기 분석서 F→P의 승리 패턴이 "write 전 후보 record 전수 열거"로 단일
# 수렴(coverage+BIND+PROV 동시 폐쇄·t2/74/81/107/22)·P→F 구제 최대 레버(t5/82/83/91/111).
# 기전: 첫 write 시도 시 listed−examined ≠ ∅ 이면 read-지시 deny(read-only·[[05]] write 강제 0).
# Δspurious 보수: 후보 과다(>max) 침묵·sim-cap·재료=ledger(에이전트 자신의 열람 기록·규칙0).

def readall_unread(listed, examined, max_candidates=8):
    """read-all 순수 술어: 미열람 후보 목록(정렬) — 없거나 후보 과다면 []."""
    listed = {str(x).strip() for x in (listed or ()) if str(x).strip()}
    examined = {str(x).strip() for x in (examined or ())}
    if not listed or len(listed) > max_candidates:
        return []
    return sorted(listed - examined)


READALL_FEEDBACK = (
    "[READ-ALL] Before modifying anything, read the details of every record you have listed — "
    "call {reader} for: {ids}. The user's request may cover records you have not read yet. "
    "After reading, re-emit your call(s), acting on every record the request covers and only those."
)


# ─── ★COV FIND-subset 백스톱 (T2_COV=1·COVERAGE_LOOP_DESIGN §3·in-flight) ───
# READALL이 못 닫는 잔여 = "다 읽고도 일부에만 행동"(v3-probe t81: 4주문 read 후 1개만 취소).
# 기전: ≥1 write 실행 후, M(요청이 커버하는 record 집합·LLM formalize 1회 캐시) ∖ acted ≠ ∅ 이면
# 생성-레벨 리마인더 1회(기존 eplan 버퍼 채널·비커밋·in-flight — walk의 stop-time 사인 해소).
# 분담([[10]]): M 산출=LLM(formalize·방식1 content-match v1·방식2 predicate는 후속),
# diff/캐시/cap=결정론. 재료=에이전트 가시 대화+열람 record만(규칙0·DB 주입 0). write 강제 0.

def _cov_parse_ids(text, known_ids):
    """LLM 응답서 record id 목록 추출(순수) — grounded 교집합만(발명 id 차단)."""
    import re as _re
    if not text:
        return []
    m = _re.search(r'\{[^{}]*"ids"[^{}]*\}', text, _re.S)
    raw = []
    if m:
        try:
            raw = [str(x).strip() for x in (json.loads(m.group(0)).get("ids") or [])]
        except Exception:
            raw = []
    if not raw:  # 폴백: known id의 문자 그대로 등장
        raw = [k for k in known_ids if k in text]
    known = {str(k).strip() for k in known_ids}
    seen, out = set(), []
    for r in raw:
        if r in known and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _cov_formalize_M(agent, la, UserMessage, msgs, ep_spec, a2):
    """M 산출 서브콜(격리·1회): 유저 발화 + 열람 record 요약 → 요청이 커버하는 record ids.
    실패/미확신 = [] (침묵·안전측)."""
    ent_key = (ep_spec or {}).get("entity_key")
    if not ent_key:
        return []
    recs = []
    seen = set()
    path = (ep_spec or {}).get("items_id_path") or ()   # [container_key, id_field] = A2([[05]])
    for out in _parse_tool_outputs(msgs):
        if isinstance(out, dict) and out.get(ent_key):
            oid = str(out.get(ent_key)).strip()
            if oid in seen:
                continue
            seen.add(oid)
            names = []
            if len(path) >= 2:
                seq = out.get(path[0])
                if isinstance(seq, dict):
                    seq = list(seq.values())
                for it in (seq or []):
                    if isinstance(it, dict):
                        names.append(str(it.get("name") or it.get(path[1]) or ""))
            recs.append("%s status=%s items=%s"
                        % (oid, out.get("status"), ", ".join(n for n in names[:8] if n)))
    if len(recs) < 2:
        return []
    users = [str(getattr(m, "content", "") or "") for m in msgs
             if getattr(m, "role", None) == "user"][:6]
    prompt = (
        "You are auditing a customer-service conversation. Based ONLY on what the user asked, "
        "decide which of these records the user's request requires MODIFYING (write actions).\n"
        "User said:\n- " + "\n- ".join(u[:300] for u in users) +
        "\nRecords:\n- " + "\n- ".join(recs[:8]) +
        '\nReply with JSON only: {"ids": ["..."]} — include a record ONLY if the request clearly '
        "covers it; if the request targets a single record, list just that one."
    )
    import t2_subcall as _SC
    return _cov_parse_ids(_SC.sub_generate(agent, la, UserMessage, prompt,
                                           "cov_formalize"), seen)


COV_REMINDER = (
    "[COVERAGE] Based on the conversation so far, record(s) {ids} may ALSO be covered by what "
    "the user asked, but you have not acted on them. If they are covered, handle them too now; "
    "if they are not, briefly confirm that with the user before finishing."
)

# ★T2_COV_MIDDRIVE (C118·EPLAN_MIDDRIVE_DESIGN §2.1): "종료시 1회" → "갭 열린 동안 매 드리프트 견인".
COV_REMINDER_DRIVE = (
    "[COVERAGE-DRIVE] The customer's request covers {n} item(s); you have completed {done} but "
    "{ids} are NOT yet done. Do the NEXT one NOW — call the write/action tool for it directly. "
    "Do not end, transfer, or move to other topics until every covered item is handled."
)


def _last_assistant_did_write(messages, write_tools):
    """가장 최근 assistant 메시지의 tool_calls에 write(eff∈write_tools)가 있나 (드리프트 판정 반대).
    write가 있으면 진행 중(드리프트 아님)·없으면(read/prose/타도구) 드리프트. write_tools=A2 도출."""
    for m in reversed(messages):
        if getattr(m, "role", None) != "assistant":
            continue
        for tc in (getattr(m, "tool_calls", None) or []):
            eff = _eff_tool_name(tc)
            if eff in write_tools or getattr(tc, "name", "") in write_tools:
                return True
        return False   # 최근 assistant 턴에 write 없음 = 드리프트
    return False


# ─── ★TOOLERR 도구-에러 라우팅 (T2_TOOLERR=1·사용자 지시 2026-07-13) ───
# 일반 로직(엔진): 방금 committed된 tool-error를 A2 `tool_error_specs`로 분류·라우팅.
#   class=recover → 인자 고쳐 재시도 강제(같은-실패-인자 재발행/조기 transfer deny).
#   class=abstain → 날조 금지·ASK/transfer 지시(비강제 directive).
# 도메인 정보(에러패턴·class·hint)는 전부 A2([[05]]). 재료=committed tool-error(규칙0).

def _trailing_tool_errors(msgs):
    """대화 말미의 tool-result 블록(에이전트가 지금 응답하려는) 중 error 메시지들(최근순).
    앞선 assistant-text/user를 만나면 중단 = '방금 난 에러'만."""
    errs, callmap = [], {}
    for m in msgs:
        if getattr(m, "role", None) == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                callmap[getattr(tc, "id", None)] = tc
    for m in reversed(msgs):
        r = getattr(m, "role", None)
        if r == "tool":
            if getattr(m, "error", False):
                errs.append(m)
        elif r == "assistant":
            break   # tool-call을 낸 assistant 턴 = 블록 경계
        else:
            break   # user 발화가 더 최근이면 이미 넘어감
    return errs, callmap


def classify_tool_error(msgs, a2):
    """방금 난 tool-error를 A2 tool_error_specs로 분류.
    반환 (spec, tool_name, failed_args) | None. spec={match,class,hint,applies_to?}."""
    specs = (a2 or {}).get("tool_error_specs") or []
    if not specs:
        return None
    errs, callmap = _trailing_tool_errors(msgs)
    if not errs:
        return None
    import re as _re
    for em in errs:                       # 최근 에러 우선
        content = str(getattr(em, "content", None) or "")
        tc = callmap.get(getattr(em, "id", None))
        tool = getattr(tc, "name", None) if tc else None
        for sp in specs:
            ap = sp.get("applies_to")
            if ap and tool not in ap:
                continue
            pat = sp.get("match")
            if pat and not _re.search(pat, content, _re.I):
                continue
            return (sp, tool, (_args_dict(tc) if tc else {}))
    return None


def _delivered_unused_agent_tools(orch, messages, a2):
    """이 대화가 **이미 받은 문서가 이름을 댔고 아직 안 부른** 에이전트-측 도구 (없으면 []).

    레지스트리 ∩ 이미 배달된 텍스트 − 이미 호출/해제. 이 집합이 **비었다** =
    *에이전트가 직접 할 수 있는 일이 남아 있지 않다* 이고, 그때는 손님을 가리키는 것이
    유일한 진행 경로다. P-A 의 침묵 자격이 여기에 걸린다.

    ⚠`T2_SEARCH_EXHAUST` 가 같은 계산을 **인라인으로** 갖고 있다([[67]] 위반). 그 경로를
      건드리는 것은 검정이 없어 미뤘다 — 다음에 이 함수 호출로 접어라.
    """
    try:
        reg = _agent_discoverable(getattr(getattr(orch, "_t2_orch", None), "environment", None))
        if not reg:
            return []
        txt = chr(10).join(str(getattr(m, "content", "") or "") for m in (messages or [])
                           if getattr(m, "role", None) == "tool")
        used = {_exact_tool_name(t) for m in (messages or [])
                for t in (getattr(m, "tool_calls", None) or [])}
        used |= _unlocked_names(messages, a2)
        return sorted(n for n in reg if n in txt and n not in used)
    except Exception:
        return []


def _transfer_tools(a2):
    """포기/이관 도구 집합 = A2서 도출(엔진 리터럴 0·[[05]]). 우선순위:
    (1) a2["transfer_tools"] 명시 (2) notice-kind 게이트의 applies_to(이관 전 고지=이관도구)."""
    if not a2:
        return set()
    tt = set(a2.get("transfer_tools") or [])
    if tt:
        return tt
    for g in (a2.get("gates") or []):
        if g.get("kind") == "notice":
            tt |= set(g.get("applies_to") or [])
    return tt


TOOLERR_RECOVER = (
    "[TOOL-ERROR:RECOVER] Your last call to {tool} FAILED: {hint} This is a fixable error — "
    "do NOT give up, transfer, or invent a substitute. Re-derive the correct argument value from "
    "prior tool outputs and re-emit the call with the CORRECTED argument."
)
TOOLERR_ABSTAIN = (
    "[TOOL-ERROR:ABSTAIN] Your last call to {tool} returned no usable result: {hint} Do NOT make "
    "up an answer to fill this gap. Tell the user you could not find the information, or transfer."
)


# ─── ★GROUND (T2_PROV_GROUND=1): config-도출 candidate-surfacing resolver ───
# 모델은 *의도/op* 명명만, 구체값은 결정론이 직전 tool 출력서 grounding (추출 = 도메인-일반).

def _sig(s):
    s = str(s).strip()
    if "@" in s and "." in s:
        return "email"
    if s.startswith("#"):
        return "hashid"
    if s.replace("-", "").isdigit() and len(s) >= 5:
        return "numid"
    return "other"


# 주소류 free-text 인자 토큰(도메인일반 식별-어휘·PROV-ADDR-FULL §6.3). id-형(numid/hashid) 아님.
_ADDR_ARG_TOKENS = ("address", "street", "city", "state", "zip", "postal", "country")


def _is_addr_arg(arg_key):
    """arg_key가 주소류 free-text 인자인가(도메인일반 어휘 매칭)."""
    kl = str(arg_key).lower()
    return any(t in kl for t in _ADDR_ARG_TOKENS)


def _key_tokens(arg_key):
    """arg_key → 의미 토큰 (id/ids 제거·단수화). 'order_id'→{'order'}, 'item_ids'→{'item'}."""
    toks = set()
    for t in str(arg_key).lower().split("_"):
        if t in ("id", "ids", "no", "num", "number", "code"):
            continue
        toks.add(t[:-1] if t.endswith("s") and len(t) > 3 else t)
    return toks or {str(arg_key).lower()}


def _iter_scalars(obj, key=None):
    """JSON에서 (enclosing_key, scalar) 재귀 수집 + dict-key 자체(ID형)."""
    if isinstance(obj, dict):
        for kk, vv in obj.items():
            if isinstance(vv, (dict, list)):
                if isinstance(kk, str) and any(c.isdigit() for c in kk) and len(kk) >= 5:
                    yield (key or kk, kk)
                yield from _iter_scalars(vv, kk)
            else:
                yield (kk, vv)
    elif isinstance(obj, list):
        for vv in obj:
            yield from _iter_scalars(vv, key)
    else:
        yield (key, obj)


def _parse_tool_outputs(msgs, lenient=False):
    """role==tool·비-error 메시지 content를 JSON 파싱(최근→과거 순).
    lenient=True(T5-C 신규 경로 전용): JSON 뒤에 augment 텍스트가 append된 content도
    leading-JSON으로 구제(N4). 기본 False = v1 동작 바이트-동일."""
    outs = []
    for m in reversed(msgs):
        if getattr(m, "role", None) != "tool":
            continue
        if getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        try:
            outs.append(json.loads(c))
        except Exception:
            if lenient:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(c.strip())
                    outs.append(obj)
                    continue
                except Exception:
                    pass
            outs.append(c)
    return outs


def _grounded_candidates(arg_key, fab_value, msgs, limit=8, lenient=False):
    """arg_key 타입에 맞는 grounded 후보값을 tool 출력서 추출(최근 우선·dedup·순서보존)."""
    toks = _key_tokens(arg_key)
    want_sig = _sig(fab_value)
    seen = set()
    key_cands, sig_cands = [], []
    for out in _parse_tool_outputs(msgs, lenient=lenient):
        if not isinstance(out, (dict, list)):
            continue
        for kk, vv in _iter_scalars(out):
            s = str(vv).strip()
            if len(s) < 4 or s in seen:
                continue
            kl = str(kk or "").lower()
            if any(t in kl or kl in t for t in toks):
                seen.add(s)
                key_cands.append(s)
            elif want_sig != "other" and _sig(s) == want_sig:
                seen.add(s)
                sig_cands.append(s)
    return (key_cands or sig_cands)[:limit]


GROUND_FEEDBACK = (
    "Error: [PROVENANCE] argument '{k}'='{s}' is invented — never use placeholder/guessed IDs. "
    "The real, grounded {k} value(s) already available from prior tool results are: {cands}. "
    "Use ONLY one of these. If you must determine WHICH one matches what the user described "
    "(e.g. which order contains that item), call the appropriate getter on these candidates "
    "and read the answer from its output. Now emit a corrected tool call."
)

# T5-C 스펙#2 (prov_mode=rescue 전용): 예시 절 제거(프라이밍 원천 중립화)
GROUND_FEEDBACK_NEUTRAL = (
    "Error: [PROVENANCE] argument '{k}'='{s}' is invented — never use placeholder/guessed IDs. "
    "The real, grounded {k} value(s) already available from prior tool results are: {cands}. "
    "Use ONLY one of these, or determine the right one by reading prior tool outputs. "
    "Now emit a corrected tool call."
)

# ─── ★T2_DISAMB=1: |C|>=2 write-인자 재확인 (E-AMB T5 라우터·규칙0 준수) ───
# 값이 문맥에 실재하되 같은-형식 후보가 2+개면(⋈ 지점) 한 번만 재확인 피드백 → 재생성.
# 열거하는 후보 = 이미 조회된 도구출력에서만(_grounded_candidates) = DB 주입 0.
DISAMB_FEEDBACK = (
    "Error: [DISAMBIGUATE] argument '{k}'='{s}' is one of {n} same-type values already seen in prior "
    "tool outputs: {cands}. Multiple candidates exist, so re-check against what the user explicitly "
    "asked for (the exact attributes, which order, the required payment rule) before writing. "
    "If '{s}' is exactly right, re-emit the SAME tool call unchanged. Otherwise emit the corrected "
    "call, or — only if the conversation truly does not determine the answer — ask the user to pick. "
    "Never invent values."
)

# ★enumerate 모드(T2_DISAMB_MODE=enumerate): 재추측 금지·후보 전부 사용자에 제시하고 선택 강제.
# frontier robust 방식(후보 나열+사용자 내용선택) 재현. list-order 관례 불필요.
DISAMB_ENUM_FEEDBACK = (
    "Error: [DISAMBIGUATE-FILTER] argument '{k}'='{s}' is one of {n} candidates already seen: {cands}. "
    "Before writing, FILTER these candidates by EVERY attribute the user explicitly asked about — the FULL "
    "request (all the items to change, all colors/sizes/details), not just one attribute. "
    "If EXACTLY ONE candidate matches the full request, use that one (correct '{s}' to it if different). "
    "If MORE THAN ONE matches, send the USER a message listing those matching candidates with their "
    "identifying details (what each contains) and ask which one they mean; wait for the user's choice. "
    "If NONE matches, ask the user to clarify. Do NOT guess by recency or position. Never invent values."
)


def _confirm_write_tools(a2):
    """A2-도출 write 도구 집합 = kind=='confirm' 게이트의 applies_to (도메인 리터럴 0)."""
    tools = set()
    for g in (a2 or {}).get("gates", []) or []:
        if g.get("kind") == "confirm":
            tools |= set(g.get("applies_to") or [])
    return tools


def _write_choice_arg(a2, tc):
    """이 write 호출이 담은 **선택의 인자 이름**(과 그 축) — A2 선언에서만 읽는다.

    왜 필요한가 (2026-08-23 R7 · `refute_5.json` §surviving_our_layer⑵): `DECIDE-FIRST`
    캐리는 *"It answers: X. … make the call again with that value."* 라고만 말하고 **X 가
    어느 인자의 답인지 말하지 않았다**. t7336_halfB `task_085#s373753` 에서 결정 서브가 고른
    것은 **문서 계열 라벨**(`General` — 형제 값이 'Blue Account'·'Gold Account'·'Sky Blue')
    인데, 축 이름이 없으니 그 값이 `dispute_category` 자리로 흘러들어 그 sim 의 11회 시도가
    전부 열거 밖 값으로 실패했다. 모델은 축자로 *"Based on the error message…"* 라며 **우리
    배달을 지목**한다 ⇒ 우리 층 결함이다([[55]]·[[64]]).

    술어는 전부 닫혀 있다 — **이름 동등성 + 선언된 접두 + dict 조회**뿐이고 의미 판단·유사도·
    도메인 리터럴 0([[22]]·[[59]]·[[66]]). 출처는 **이미 있는 A2 키 셋**뿐이다(새 키 0·[[62]]):

      · `write_arg_enum[]`      → `arg`      (`applies_to` + `applies_when` 접두로 지목)
      · `choice_grounding[]`    → `arg`      (`tool` 로 지목)
      · `recommendation_verify` → `operand`  (`action_tool` 로 지목)

    이 셋은 *"이 write 는 선택을 담는다"* 를 선언하는 바로 그 자리다 — 그래서 **인자를 댈 수
    있는 write** 와 **선택을 담은 write** 가 정확히 같은 집합이 되고, 어느 벤치·어느 도메인에서도
    같은 모양으로 전이된다(태스크·상품 이름을 술어에 넣지 않는다·[[05]]·[[70]]).
    셋 중 무엇도 이 호출을 지목하지 않으면 `(None, None)` = **축 미상**이고, 그 때 캐리는
    나가지 않는다(항목 지시: *name the argument, or do not deliver*).

    반환: `(arg_name | None, axis_group | None)`. 축(`axis_group`)은 종전 `_dax` 와 같은
    값이다 — `write_arg_enum.group_map` 의 dict 조회 하나뿐(2026-08-13 071 t1 부검에서
    저장 슬롯을 축별로 나눈 그 키). 예외는 전부 삼키고 `(None, None)`(fail-safe = 무발화).
    """
    a2 = a2 or {}
    try:
        nm = str(getattr(tc, "name", "") or "")
        outer = _args_dict(tc)
        # 이 호출이 실제로 실행하는 이름들 — 겉이름·디스패처 unwrap·env 축자 이름.
        # (선언은 `open_bank_account_4821` 처럼 env 축자 이름을 쓰기도 하고
        #  `apply_for_credit_card` 처럼 겉이름을 쓰기도 한다. 둘 다 **동등성**으로 본다.)
        names = set()
        for _n in (nm, _eff_tool_name(tc), _exact_tool_name(tc)):
            if _n:
                names.add(str(_n))
        inner = outer.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        if not isinstance(inner, dict):
            inner = {}
        # ⑴ write_arg_enum — 선언된 디스패처 이름 + 선언된 접두가 맞으면 `arg` 가 선택의 자리다.
        for sp in (a2.get("write_arg_enum") or []):
            if nm != str(sp.get("applies_to") or ""):
                continue
            aw = sp.get("applies_when") or {}
            if aw.get("arg"):
                _v = str(outer.get(aw.get("arg")) or "")
                _pref = str(aw.get("prefix") or "")
                if _pref and not _v.startswith(_pref):
                    continue
            _arg = str(sp.get("arg") or "") or None
            _gv = str(inner.get(sp.get("group_arg")) or "")
            _ax = (sp.get("group_map") or {}).get(_gv) or None
            if _arg:
                return _arg, _ax
        # ⑵ choice_grounding — `tool` 이 이 호출을 지목하면 `arg` 가 선택의 자리다.
        for cg in (a2.get("choice_grounding") or []):
            _t = str(cg.get("tool") or "")
            if _t and _t in names:
                _arg = str(cg.get("arg") or "") or None
                if _arg:
                    return _arg, None
        # ⑶ recommendation_verify — `action_tool` → `operand`.
        _rv = a2.get("recommendation_verify") or {}
        _t = str(_rv.get("action_tool") or "")
        if _t and _t in names:
            _arg = str(_rv.get("operand") or "") or None
            if _arg:
                return _arg, None
    except Exception:
        return None, None
    return None, None


# ★DECIDE-FIRST 캐리 문면 (2026-08-23 R7). 구판은 *"It answers: X … make the call again with
#   that value."* 였다 — **어느 인자의 답인지 한 번도 말하지 않는다**. 그래서 문서 계열 라벨이
#   `dispute_category` 자리로 흘러들었다(refute_5 §surviving⑵·11/11 실패).
#   ⚠도메인 낱말 0 — `{arg}` 는 런타임에 A2 선언에서 채운다([[05]] 전이).
#   ⚠[[64]] 두 조각을 모두 담는다: **무엇이 틀렸나**(이 write 가 담은 선택이 아직 안 내려졌다)
#     + **무엇을 하면 풀리나**(그 인자에 넣어 다시 부르거나, 그 인자가 가질 수 없는 값이면
#     이 답은 이 호출의 답이 아니니 원래대로 다시 불러라).
#   ⚠엔진은 고르지 않는다 — 문장은 **조건문**이고 값의 채택은 끝까지 모델 몫이다([[62]] ③④).
_SPEC_AT_WRITE_FB = (
    "Error: [SPEC-AT-WRITE] this write was held for one turn because the argument names it "
    "used are not the ones this tool declares. Nothing new is being told to you - the block "
    "below is the environment's own reply from earlier in this conversation, when this tool "
    "was made available. It is repeated here because the write is happening now and that "
    "reply is {dist} messages back:\n\n{spec}\n\n"
    "Make the call again using exactly the argument names listed above - do not rename, add "
    "or drop any of them - and keep the values you already decided on."
)


def _env_spec_for(wc, msgs):
    """이 write 의 표적 도구에 대해 **env 자신이 앞서 보낸 응답** — (본문, 인덱스, 거리).

    ★2026-08-25 신설. 왜(t7348 085 두 sim 궤적 직독): 파라미터 17개와 enum 4종을 전부 담은
      2,975자 블록이 **msg22** 에 도착하는데 첫 오답 write 는 **msg68 / msg80** 이다(거리 46·58).
      그 사이 모델은 `debit_card_id`·`category`·`date_first_noticed`·`type_of_transaction` …
      **10개를 지어내며 13턴**을 태운다. 재료 부재가 아니라 **거리**다 — x509 큐
      `plan_2026_08_24_pm.common_diagnosis` 축자 *"재료는 상류에 있고 결정점에 없다"* 와 같은 모양.

    술어는 전부 **구조**다([[22]]·[[59]] 텍스트 파싱 0·도메인 낱말 0):
      · 표적 = 이 write 가 실행하는 도구의 **레지스트리 이름**(`_exact_tool_name`).
      · 되돌아보며 (assistant tool_call, 그 다음 tool 결과) 짝을 찾는다.
      · 그 호출의 **인자**가 표적 이름을 담고 있고 호출의 **겉이름이 이 write 와 다르면**,
        그 결과가 *"이 도구에 대해 env 가 한 말"* 이다. 그대로 돌려준다 — 자르지도 고르지도
        요약하지도 않는다([[62]]③④).
      · 이 write 자신의 재시도(같은 겉이름)는 제외된다 ⇒ 자기 오류문을 되먹이지 않는다.
    ⚠못 찾으면 (None, -1, -1) 이고 그러면 아무 말도 하지 않는다([[25]]).
    """
    want = str(_exact_tool_name(wc) or "")
    mine = str(getattr(wc, "name", "") or "")
    if not want:
        return None, -1, -1
    ms = list(msgs or [])
    for i in range(len(ms) - 1, -1, -1):
        m = ms[i]
        tcs = getattr(m, "tool_calls", None) or []
        if not tcs:
            continue
        for tc in tcs:
            if str(getattr(tc, "name", "") or "") == mine:
                continue
            try:
                blob = json.dumps(_args_dict(tc), ensure_ascii=False)
            except Exception:
                blob = ""
            if want not in blob:
                continue
            for j in range(i + 1, min(i + 4, len(ms))):
                if str(getattr(ms[j], "role", "")) == "tool":
                    body = str(getattr(ms[j], "content", "") or "")
                    if body:
                        return body, j, len(ms) - j
            break
    return None, -1, -1


_RULE_AT_WRITE_FB = (
    "Error: [RULE-AT-WRITE] this write was held for one turn. Nothing new is being told to "
    "you - the line(s) below are the environment's own words from earlier in this "
    "conversation, and they mention the value you are about to write. They are repeated "
    "here because that text is {dist} messages back:\n\n{rules}\n\n"
    "Make the call again. If those lines change which record or which value this call should "
    "carry, change it; if they do not, send the same call unchanged."
)


def _decl_join(dw_fb, tc, txt):
    """선언 배달을 write 유예 자리에 **덧붙인다**(뺏지 않는다).

    ★2026-08-26 (x543 재생). 명세·규칙·인자-정책은 **서로 다른 재료**다 — 명세는 도구가
      선언한 인자/열거이고, 규칙은 정책이 정한 절차 문장이다. 한 자리를 두고 `elif` 로
      경쟁시키면 거리가 먼 명세가 **항상** 이기고 규칙은 영영 안 나간다: t7356 에서 셋이
      나갈 3 자리 전부 명세(2975·2975·2137자)와 규칙(303·303·74자)이 둘 다 있는데
      선점 3/3 이었고, 그 규칙에 실린 것이 x538 책임 한도 표였다.
    ★합성은 격리에서 이미 쟀다 — 큐 `findings_2026_08_25_night.N2` 의
      `x538b B_both(합성) 20/20`(단독 B_rule 20/20 ↔ A_asis 12/20 ↔ N_len 12/20).
    ⚠엔진은 고르지 않는다 — 선언된 것을 **순서대로 잇기만** 한다([[62]]④·[[59]]).
    """
    if dw_fb is None:
        return (tc, txt)
    return (dw_fb[0], str(dw_fb[1]) + "\n\n" + str(txt))


def _declared_rules_for(wc, a2):
    """이 write 가 실행하는 도구에 대해 **A2 가 선언한 절차 문장** — 검색 0·랭킹 0.

    ★2026-08-25 신설. 격리 `x537`(085·창 3·n4): 결정점 창 그대로면 **0/12**, 그 절차 문장 한 줄을
      결정점에 놓으면 **12/12**, 같은 길이의 무관한 문장이면 **0/12**([[57]] 통과).
      실물: 085 가 중복 청구에서 늦은 거래를 골랐다 — 규칙은 대화 초반 문서 본문에 축자로 있었고,
      write 는 한참 뒤였다. `_env_spec_for`(도구 명세)와 **같은 가족**이고 같은 진단이다:
      *재료는 상류에 있고 결정점에 없다*.

    ⚠**검색기를 짓지 않는다.** 초판은 궤적 문장을 토큰으로 긁었는데 검산에서 **다른 도구의
      unlock 문면**을 집었다(2026-08-25 실측). 순위를 매기면 엔진이 고르기 시작한다([[62]]④).
      그래서 후보를 만들지 않고 **선언된 것만** 읽는다 — 술어는 도구 이름 동등성 하나다([[22]]).
      출처 의무는 A2 쪽에 있다: 문장마다 정책 축자와 `_note_` 를 단다([[23]]).
    """
    want = str(_exact_tool_name(wc) or "")
    if not want:
        return None
    out = []
    for sp in ((a2 or {}).get("write_rules") or []):
        if not isinstance(sp, dict):
            continue
        tgt = str(sp.get("applies_to") or "")
        if not tgt or not (want == tgt or want.startswith(tgt)):
            continue
        t = str(sp.get("text") or "").strip()
        if t:
            out.append(t)
    if not out:
        return None
    return chr(10).join("- " + x for x in out)


_DECIDE_FIRST_FB = (
    "Error: [DECIDE-FIRST] this write was held for one turn because the decision it "
    "encodes had not been made in this conversation yet. It has now been made, and it "
    "answers exactly one thing - the '{arg}' argument of this call, and no other "
    "argument:\n{material}\n"
    "If that is the value '{arg}' should carry, make the call again with '{arg}' set to "
    "it and every other argument unchanged. If it is not a value '{arg}' can take, then "
    "it is not an answer for this call at all - ignore it and make the call again "
    "exactly as it was."
)


# ─── ★NL-NUM-PROV (T2_NLNUM_PROV=1) — t47형 NL-산술 환각 검사 (2026-07-11) ───
# 어시스턴트 *텍스트 발화*의 통화-금액(통화기호+숫자·도메인 어휘 0)이 ①이전 문맥(user/tool
# 텍스트)에 원문-부재 ∧ ②calculate류 도구 출력에도 부재(②⊂①: calc 출력=tool 텍스트)면
# 생성-레벨 regen 1회: "암산 금지·calculate 도구로 검증 후 재진술". 상한 1/턴·무과금·
# 채택 전 게이트 재검사(안전측). calculate 도구명은 A2 키 `calc_tool`(ABox·[[05]]).
_MONEY_RE = re.compile(r"[$€£¥]\s?(\d[\d,]*\.\d{1,2})\b")


def _num_variants(num):
    """금액 숫자부의 대조 변형 집합: 콤마 제거·float 정규형('875.50'→'875.5')."""
    s = num.replace(",", "")
    v = {s}
    try:
        v.add(repr(float(s)))
    except ValueError:
        pass
    t = s.rstrip("0").rstrip(".") if "." in s else s
    if t:
        v.add(t)
    return v


def _unverified_amounts(text, ctx_numeric):
    """text(어시스턴트 발화)의 통화-금액 중 ctx(user/tool 텍스트·콤마 정규화)에 부재분."""
    out = []
    for m in _MONEY_RE.finditer(text or ""):
        if not any(v in ctx_numeric for v in _num_variants(m.group(1))):
            out.append(m.group(0).strip())
    return out


NLNUM_FEEDBACK = (
    "Error: [NL-NUM] your reply states the amount '{amt}', which does not appear anywhere in the "
    "conversation or in any tool output — it looks like mental arithmetic. Do NOT do mental "
    "arithmetic. Verify the amount by calling the {tool} tool with the exact expression, then "
    "restate your reply using the verified value."
)


# ─── ★assertion-provenance (2026-07-16·`ASSERTION_PROVENANCE_ARMS_DESIGN_2026_07_16`) ───
# 병목: 에이전트가 producer 도구를 *선택하지 않고* 사용자에게 판단을 주장한다(t019g 3/3·호출 0).
# C45 출처선언은 **write 인자**에만 걸려 있어 이 지점을 통째로 놓친다. 여기서 assertion으로 확장.
# ★불변량: **엔진은 어시스턴트 답변 텍스트를 파싱하지 않는다.**
#   - discovery-required: 엔진이 보는 것 = {호출된 도구 이름} 집합뿐 (구조 이벤트).
#   - self-declaration : 엔진이 보는 것 = LLM이 내놓은 **선언 JSON 필드**뿐 ([[10]] LLM=formalize·엔진=검증).
#   텍스트 정규식으로 '판단'을 탐지하면 = 엔진-formalize + 도메인 리터럴 = [[03b]]/[[05]] 위반 = 실험무효.
# 도메인 사실(어느 데이터원/operand에 어느 producer가 붙는가)은 **전부 A2**(엔진 리터럴 0).

# ─── ★실효 write 술어 (도메인일반·2026-07-18·`A2_DOMAIN_GENERALIZATION_DESIGN §2.2`) ───
# 정본 = `BANK_EPLAN_ALLACTION_IMPL_DESIGN §3.1`의 `_is_write`. **한 정의만 둔다**([[03b]] 술어 이중화 금지) —
# 여기 hoist하고 `T2_FAB_STRIP`(2398)이 쓰던 인라인 사본을 이걸로 대체한다.
# ⚠️`mutates_state`(env 속성)를 쓰면 **안 된다**: 실측 결과 `log_verification`·`give_discoverable_user_tool`·
#   `unlock_…`이 전부 True라 "상태변경 0"이 거의 모든 sim서 거짓 → 게이트가 영영 안 뜬다(2026-07-18 설계 교정).
# `_NNNN` 접미사 제거 = env discoverable 명명 관행(도메인 리터럴 아님·패턴). 파일 내 인라인
# `re.sub(r"_\d+$", ...)` 관용구의 정본 상수(C241 U1'에서 신설).
_SUFFIX_RE = re.compile(r"_\d+$")
_READ_PREFIX_RE = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
_PROCEDURAL_RE = re.compile(
    r"(^log_|^verify_|_verification$|^kb_|^shell$|transfer_to_human)", re.I)
# ★C241 U1'(리뷰 B2): 도메인 어휘 가지(`discoverable`·`^give_`·`^unlock_`)를 **A2 파생**으로 분리.
#   ⚠전역 frozenset은 금지 — ①순서 의존(도메인 A2 로드 전 호출 시 `_is_effective_write("give_…")`가
#   True로 뒤집혀 `:4543` 회귀조건이 무너짐) ②교차-도메인 누출(전역 last-wins → airline A2-swap
#   스모크가 통과하면서 누출은 실재). ⇒ **a2를 명시 전달**한다(시그니처 불변 포기).
#   `transfer_to_human`은 5/5 도메인 공통이라 엔진에 남긴다.


def _a2_of(obj):
    """orchestrator **또는 에이전트** 에서 현 도메인 A2 도출 (C241 U1' 배선용·`_domain_a2` 캐시 재사용).

    ⚠2026-08-26 배선 수리 (x549 · t7359 스모크가 잡음). 구판은 `obj.environment` **하나만** 봤다.
      그런데 `unified()` 는 `LLMAgent._generate_next_message = unified`(:13993) 로 **에이전트**에
      설치되고 에이전트에는 `.environment` 가 없다(파일 전체에서 심는 곳 0). ⇒ `unified` 안의
      `_a2_of(self)` **여섯 자리가 전부 None** 을 받아 `_a2_procedural(None)` 이 공집합이 되고
      `unlock_…`·`give_…`·`call_…` 넷이 **실효 write 로 뒤집혔다**.

      이것은 `ENGINE_LITERAL_REMEDIATION_DESIGN_2026_07_30.md` §8-B 가 축자로 **예고한** 회귀다:
        *"①순서 의존 — `_is_effective_write` 가 … `_domain_a2()` 보다 먼저 불리면 집합이 비어
          `give_…`/`unlock_…` 이 write 로 판정되고, 이는 `:4531` 이 회귀 조건으로 못박은
          `_is_effective_write("give_…")=False` 가 정확히 무너지는 시나리오다."*
      그 문서의 처방은 *"호출부 6곳 수정 — **6곳 모두 a2 가 근처에 있는 오케스트레이터 래퍼 안**"*
      이었는데 **그 전제가 `unified` 에서 거짓**이었다. 전역은 없앴는데 전달을 안 했다.

      실측 폭발 반경(x549 · 최근 런 12개 23 sim · 태그별): `_any_effective_write` 가 참인 sim
      **100% ↔ 34.8%**(전-sim 판정 뒤집힘 15/23). `T2_WRITE_PROV`·`T2_CLAIM_PROV` 는 **상시 ON**
      (go_stack:110·162)이고 둘 다 이 술어로 갈리므로, 구판에선 :13240 의 `break` 가 사실상 항상
      걸려 완료-주장 대조가 죽어 있었다 — `LEVER_ROSTER_CANONICAL_2026_08_19.md:248` 의 미해결
      항목 *"WRITE_PROV 마크 12,181 : 실발화 3(524:1) … 격차 원인"* 과 부합한다.
      ⚠뒤집히는 15 중 **098×2·100 은 지금 reward=1.0** 이다 ⇒ 이 수리는 양날이고, 재스모크가
        회귀 대조를 함께 봐야 한다([[70]] — ± 를 공개한다).

      수리 = `init_inject` 가 에이전트에 **이미 심어 두는** `_t2_a2`(:7109·:7385)를 먼저 본다.
      새 선언 0 · 도메인 리터럴 0 · 오케스트레이터 경로(:6720)는 `_t2_a2` 가 없어 거동 불변.
      래칫 = `test_c241_u1_predicate.py` §배선 (구판 래칫은 **순수 함수만** 검정해 이걸 놓쳤다).
    """
    try:
        a2 = getattr(obj, "_t2_a2", None)        # 에이전트 경로 — init_inject 가 심은 정본
        if a2:
            return a2
        env = getattr(obj, "environment", None)  # 오케스트레이터 경로 (구판·거동 불변)
        if env is None:                          # 에이전트인데 _t2_a2 가 아직 없을 때의 보루
            env = getattr(getattr(obj, "_t2_orch", None), "environment", None)
        return _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
    except Exception:
        return None


def _a2_procedural(a2):
    """A2가 선언한 **절차적**(=실효 write 아님) 디스패처 이름 집합. 미선언이면 공집합.

    구 정규식의 `discoverable`/`^give_`/`^unlock_`가 잡던 것 = banking discoverable-dispatcher
    4종. 그 이름들은 A2 `eplan.{dispatch,unlock,list}_tool`·`scaffold_get_tools[].follow_up.tool`에
    **이미 선언**돼 있다(새 키 0)."""
    if not a2:
        return frozenset()
    ep = (a2.get("eplan") or {})
    out = {ep.get("dispatch_tool"), ep.get("unlock_tool"), ep.get("list_tool")}
    # ★user-측 디스패처도 절차적이다. banking은 agent-측(call_…agent_tool)과 **별개로**
    #   user-측(call_…user_tool)을 갖는데, U1' 초판이 agent 것만 읽어 회귀 테스트가
    #   `call_discoverable_user_tool`의 판정 뒤집힘(procedural→write) 1건을 잡았다.
    #   A2가 **이미 선언**한 곳에서 읽는다(순증 0): `completion_guard.user_execution_tool` ·
    #   `scaffold_get_tools[].follow_up.{tool, completion_guard.user_execution_tool}` ·
    #   `value_acquisition[].give_tool`.
    out.add(((a2.get("completion_guard") or {}) or {}).get("user_execution_tool"))
    for t in (a2.get("scaffold_get_tools") or []):
        if not isinstance(t, dict):
            continue
        fu = (t.get("follow_up") or {})
        out.add(fu.get("tool"))
        out.add(((fu.get("completion_guard") or {}) or {}).get("user_execution_tool"))
    for v in (a2.get("value_acquisition") or []):
        if isinstance(v, dict):
            out.add(v.get("give_tool"))
    return frozenset(_SUFFIX_RE.sub("", str(x)) for x in out if x)
# ★^verify_ 추가(2026-07-20): verify_identity(scaffold 판정 도구·read-성)가 실효-write로 오분류되던 실버그 —
#   CLAIM_PROV write축 거짓통과 + WRITEPROV 조기 break(완료-주장 게이트 약화) 교정. _verification$의 대칭·도메인 일반.
# ★C238 U0(2026-07-30): `|get_current_time` 가지 **삭제** — `_READ_PREFIX_RE`의 `^get_`가 이미 잡으므로
#   **죽은 중복**이었다(`_is_effective_write = not READ and not PROC` 이라 OR 관계). 실측: tau2 5도메인
#   public 도구명 122개 전수에서 이 가지 제거로 `is_write` 판정이 바뀌는 이름 **0개**. 두 사용처
#   (`_is_effective_write` 1818·`T2_FAB_STRIP` 4534)가 동일 논리라 양쪽 무해. 정본=
#   `ENGINE_LITERAL_REMEDIATION_DESIGN_2026_07_30.md` §4 U0 · 감사 도구 `x6h_engine_literal_audit.py`.


def _eff_tool_name(tc):
    """디스패처 unwrap: write-ness는 **내부 도구**에 종속 — 단 **`call_` 디스패처만** unwrap한다.
    ⚠️`unlock_discoverable_agent_tool(agent_tool_name=X)`는 X를 *실행*하지 않고 *잠금해제*만 하므로
      inner로 풀면 안 된다(unlock=procedural=무해). 겉이름이 `call_`일 때만 inner를 본다.
    `_NNNN` 접미사 제거 = env의 discoverable 명명 관행(도메인 리터럴 아님·패턴)."""
    nm = str(getattr(tc, "name", "") or "")
    if nm.startswith("call_"):
        ar = _args_dict(tc)
        inner = ar.get("agent_tool_name") or ar.get("user_tool_name") or ar.get("discoverable_tool_name") or ""
        if inner:
            return re.sub(r"_\d+$", "", str(inner))
    return re.sub(r"_\d+$", "", nm)


def sibling_paren_arg(tc):
    """★§T-8 — 인자가 **같은 호출의 다른 인자 값**을 괄호로 되풀이하는가.

    반환 `(도구, 인자, 값, 뺄 부분문자열)` 또는 None. **판단 0 · 도메인 리터럴 0**: 도메인 텍스트를
    해석하지 않고 한 호출 안 두 인자의 **문자열 관계**만 본다([[59]] 통과).

    근거(065 실물): `account_class="Green Account (savings)"` 인데 같은 호출에
    `account_type="savings"` 가 이미 있다 — 같은 정보를 이름에 한 번 더 넣었고, env 는 이 인자를
    검증하지 않고 그대로 저장해 DB 행이 gold 와 달라졌다.
    ⊖ 부호표: banking gold 문자열 인자 1,976 · 다도메인 영속 gold 3,577 **둘 다 형제-괄호 0건**
    ⇒ 정답을 막지 않는다.

    ⚠**디스패처 언랩이 하중이다**(재리뷰 W-2): 실물은 `call_discoverable_agent_tool` 이고 바깥
    인자키가 `['agent_tool_name','arguments']` 뿐이라, inner JSON 을 안 풀면 두 값이 서로 형제로
    보이지 않아 **영원히 거짓**이다. 같은 함정에 세 번 걸렸다(`_json` NameError · OL-A · `_fam_name`).
    """
    ar = _args_dict(tc) or {}
    bag = ar
    sub = ar.get("arguments")
    if isinstance(sub, str):
        try:
            _p = json.loads(sub)
            if isinstance(_p, dict):
                bag = _p
        except Exception:
            pass
    elif isinstance(sub, dict):
        bag = sub
    vals = {k: v for k, v in bag.items() if isinstance(v, str) and v.strip()}
    for k, v in vals.items():
        m = re.search(r"\(([^)]*)\)", v)
        if not m:
            continue
        inner = m.group(1).strip().lower()
        if not inner:
            continue
        for k2, v2 in vals.items():
            if k2 != k and str(v2).strip().lower() == inner:
                return (_exact_tool_name(tc) or getattr(tc, "name", "?"), k, v, m.group(0))
    return None


def free_text_drop(tool_calls, corpus_text, a2, log=None):
    """★자유서술 기본값 인자를 **근거 없으면 뺀다** (호출부: `unified()` · 단일 구현 [[67]]).

    · 표적은 A2 선언 `free_text_defaults = {도구: [인자]}` 뿐이다 — 엔진 리터럴 0([[05]]).
    · 거동: **호출은 그대로 두고 그 인자만 뺀다**(env 기본값이 정본). 값을 고르지 않는다([[10]]).
    · 코퍼스는 호출자가 만들어 넘긴다 — 자기-그라운딩 금지(우리가 방금 보낸 값이 도구 응답에
      메아리쳐 돌아오면 그 다음 호출부터 무조건 "실재"가 된다·003 실측).
    · 반환: 제거한 (도구, 인자) 목록. 부작용은 tool_call 인자 수정뿐이다.

    ⚠2026-09-01: 이 자리는 `_json` 미바인딩(NameError)을 안쪽 except 가 삼켜 **8/8 무발화**였다.
      소스-문자열 검정은 그것을 통과시켰다 ⇒ 실행 검정(`test_free_text_arg.py`)이 정본이다.
    """
    dropped = []
    ftd = (a2 or {}).get("free_text_defaults") or {}
    if not ftd:
        return dropped
    corp = (corpus_text or "").lower()
    for tc in (tool_calls or []):
        ar = _args_dict(tc) or {}
        inner = _exact_tool_name(tc) or ""
        sub = ar.get("arguments")
        if isinstance(sub, str):
            try:
                sub = json.loads(sub)
            except Exception:
                sub = None
        targets = ftd.get(str(inner)) or ftd.get(str(getattr(tc, "name", "")))
        if not targets:
            continue
        bag = sub if isinstance(sub, dict) else ar
        for k in targets:
            v = bag.get(k)
            if not isinstance(v, str) or not v.strip():
                continue
            if v.strip().lower() in corp:
                continue
            bag.pop(k, None)
            if isinstance(sub, dict) and bag is sub:
                ar["arguments"] = json.dumps(sub, ensure_ascii=False)
            try:
                tc.arguments = ar
            except Exception:
                pass
            dropped.append((inner or getattr(tc, "name", "?"), k, v))
            if log:
                log("[T2_FREE_TEXT_ARG] %s.%s 제거 — 발화·문서·직전 원장 어디에도 없다: %r"
                    % (inner or getattr(tc, "name", "?"), k, v[:60]))
    return dropped


def view_thresholds(cap_tokens=None, scale=None, mintotal=None, msgcap=None):
    """★뷰-압축 문턱을 **모델 컨텍스트에서 유도**한다 (§T-6a·2026-09-01).

    왜: 기존 상수 `min_total=60,000자 · msg_cap=8,000자` 는 컨텍스트 **44,672** 이던
    Qwen2.5-32B 시절 값이다. Q3.8(131,072)에서 그것은 **컨텍스트의 11%** 에서 지우기 시작한다는
    뜻이고, 실측 프롬프트는 p50 20,422 · **max 60,083** 이라 절반도 안 썼다. 남는 여유를 두고
    지우면 모델이 **다시 읽고**, 그 재열람이 스텝 예산을 태운다(base 51~81 메시지 ↔ ours
    209~293 · shell 0~13 ↔ 88~163 · `max_steps` 6/30). [[84]] 와 같은 종류의 사고다.

    · `scale != "auto"` 면 **종전 상수 그대로**(대조군 보존·[[54]]).
    · 명시 env(`T2_VIEW_COMPACT_MINTOTAL`·`T2_VIEW_MSG_CAP`)가 있으면 **그 값이 이긴다**(팔 고정용).
    · 압축을 없애지 않는다 — 컨텍스트에 실제로 근접할 때의 안전망은 남긴다.
    """
    scale = (os.environ.get("T2_VIEW_SCALE", "off") if scale is None else scale)
    cap = int(cap_tokens if cap_tokens is not None
              else (os.environ.get("T2_MAX_MODEL_LEN") or 0) or 0)
    mt, mc = 60000, 8000
    if str(scale).lower() == "auto" and cap > 0:
        mt = max(mt, int(cap * 0.5 * 3.5))   # 컨텍스트의 절반을 문자수로 (≈3.5 char/token)
        mc = max(mc, int(cap * 0.25))
    env_mt = os.environ.get("T2_VIEW_COMPACT_MINTOTAL") if mintotal is None else mintotal
    env_mc = os.environ.get("T2_VIEW_MSG_CAP") if msgcap is None else msgcap
    if env_mt:
        mt = int(env_mt)
    if env_mc:
        mc = int(env_mc)
    return mt, mc


def _t2_msg_empty(_m):
    """tau2 자신의 유효성 법(`data_model/message.py:311-318` · `utils/llm_utils.py:234`) —
    본문도 도구호출도 없으면 그 메시지는 **존재할 수 없다**(§S-2)."""
    return not (str(getattr(_m, "content", None) or "").strip()
                or (getattr(_m, "tool_calls", None) or []))


def _exact_tool_name(tc):
    """The environment's own name for what this call executes — no spelling rules.

    A dispatched call carries the exact registry name in its argument, so it is read
    rather than derived. `_eff_tool_name` strips a numeric suffix, which is the sort of
    pattern rule this project retired after it produced quiet mismatches (C279); a
    procedure declaration quotes the policy's spelling and must match by identity.
    Wrappers that do not execute the inner tool (unlock, give) keep their own name.
    """
    nm = str(getattr(tc, "name", "") or "")
    if nm.startswith("call_"):
        ar = _args_dict(tc)
        inner = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                 or ar.get("discoverable_tool_name") or "")
        if inner:
            return str(inner)
    return nm


def _executed_tool_names(messages, a2=None):
    """Tools whose call actually ran, by effective name.

    A call whose result came back as an error did not perform its step, so it must not
    count as a prerequisite satisfied — otherwise a procedure check would read a failed
    submission as a completed one. Both sides of the conversation count: discoverable
    steps are often executed by the customer.

    ★2026-08-07 (102 부검): 이 판정이 **플래그와 'Error:' 접두사만** 봤다. 그런데 이 환경은
    실패를 `error=False`로 돌려주고 실패 사실을 **본문에** 쓴다 — 실측 축자:
    `NOT_VERIFIED — only 1 of the required 2 values ... match` · `Failed to log verification:
    Record may already exist.` 그래서 **실패한 호출이 '실행됨'으로 잡히고**, 게이트가 거짓 충족돼
    의존 그래프가 조기 전진한다. 표지는 env 관측에서 오므로 A2가 선언하고(`failure_markers`),
    미선언이면 종전 거동을 유지한다(거동 변화 0).
    """
    marks = tuple((a2 or {}).get("failure_markers") or ())
    ok, pending = set(), {}
    for m in messages or []:
        for tc in (getattr(m, "tool_calls", None) or []):
            pending[getattr(tc, "id", None)] = _exact_tool_name(tc)
        if getattr(m, "role", None) == "tool":
            nm = pending.get(getattr(m, "id", None) or getattr(m, "tool_call_id", None))
            txt = str(getattr(m, "content", "") or "").lstrip()
            failed = (getattr(m, "error", False) or txt.startswith("Error:")
                      or any(txt.startswith(k) for k in marks))
            if nm and not failed:
                ok.add(nm)
    return ok


def _executed_tool_counts(messages):
    """Same as `_executed_tool_names`, keeping how many times — a set threw the count away.

    The purchase-decline document does not order steps, it counts them: the internal
    transfer tool serves "the first, second, and third transfer requests" and the regular
    one the fourth. A `Counter` still answers membership the way a set did, so every
    existing caller reads it unchanged.
    """
    out, pending = collections.Counter(), {}
    for m in messages or []:
        for tc in (getattr(m, "tool_calls", None) or []):
            pending[getattr(tc, "id", None)] = _exact_tool_name(tc)
        if getattr(m, "role", None) == "tool":
            nm = pending.pop(getattr(m, "id", None) or getattr(m, "tool_call_id", None), None)
            txt = str(getattr(m, "content", "") or "").lstrip()
            if nm and not (getattr(m, "error", False) or txt.startswith("Error:")):
                out[nm] += 1
    return out


def _agent_discoverable(env):
    """env의 **agent-side discoverable** 집합 (user 측과 동형·리터럴 0).

    잠금 여부와 무관하다 — 클래스 속성이라 unlock 전에도 이름이 들어 있다(2026-08-05 확인).
    그래서 "아직 안 풀린 도구의 정확한 이름"을 말해 줄 수 있다.
    """
    try:
        tk = getattr(env, "tools", None)
        return set(tk.get_discoverable_tools()) if tk is not None else set()
    except Exception:
        return set()


# ─── ★T2_CALL_FORM (2026-08-11 C418·099·[[64]]) ───
# 우리 `[ORDER]` 는 선행 요건을 **부를 수 없는 이름**으로 말한다: `(do it with:
#   get_all_user_accounts_by_user_id)` · `... has not been called` · 프런티어 목록. 이 env 의
#   발견형 도구는 도구 목록에 서지 않고 **디스패처로만** 불린다. C300 이 프런티어 이름을
#   접미사형으로 고쳤지만 **호출 형식**은 아직 아무 데서도 말하지 않는다 ⇒ 모델이 할 수 있는
#   최선이 `unlock_...` 이고, 그러고 나면 같은 요구가 글자 그대로 또 온다(C414 의 자기재생산).
# 격리 인과(x249·n=8·099 실패 궤적 2개): `A_LIVE` 5/8·3/8 → **`B_CALLFORM` 8/8·8/8** ·
#   `C_DROP`(그 문장을 뺌) 5/8·3/8(=A) · `D_FREE` 2/8·8/8. **오답의 모양이 진단 그대로다** —
#   실패는 전부 접미사 이름을 **직접 호출**하려는 시도다(도구 목록에 없어 죽는다).
#   ⇒ 이번 처방은 뺄셈([[63]])이 아니라 **이름 대기**([[64]])다.
# [[05]] 3질문: (1)도메인-특화 순증? No — 디스패처·잠금 도구 이름도, 발견형 집합도, 접미사 실명도
#   전부 **env 스키마/레지스트리에서 기계 도출**한다(A2 리터럴 0·ABox-swap 불변). 문구만 A2.
#   (2)유동판단 동결? No — *무엇을* 부를지·인자는 그대로 모델 몫이고, 바뀌는 것은 *어느 도구를
#   거쳐* 부르는가 하나(env 규약). (3)스캐폴드가 수행? No — 문자열 치환뿐.


def _dispatch_tools(agent):
    """(잠금 도구, 호출 도구) 이름 — **스키마 구조로만** 찾는다(도메인 리터럴 0).

    발견형 디스패처는 `agent_tool_name` 파라미터를 갖고, 그중 실행하는 쪽만 `arguments` 를
    함께 갖는다. 그 차이가 둘을 가른다.
    """
    unlock = call = None
    for t in (getattr(agent, "tools", None) or []):
        try:
            sc = t.openai_schema
            fn = sc.get("function") if isinstance(sc.get("function"), dict) else sc
            props = set(((fn.get("parameters") or {}).get("properties")) or {})
            if "agent_tool_name" not in props:
                continue
            if "arguments" in props:
                call = fn.get("name")
            else:
                unlock = fn.get("name")
        except Exception:
            pass
    return unlock, call


def _call_form_map(agent, env, names, a2=None):
    """이름 → **부를 수 있는 형식** 사전. 발견형이 아닌 이름은 넣지 않는다(치환 0)."""
    reg = _agent_discoverable(env)
    if not reg or not names:
        return {}
    unlock, call = _dispatch_tools(agent)
    if not call:
        return {}
    # ⚠기본 문구는 **프로브가 이긴 것과 같은 말**이어야 한다([[03b]]: 두 벌이 되면 갈린다).
    #   치환은 세 자리(`do it with:` · `has not been called` · 프런티어 목록)에 한 번에 들어가므로
    #   어느 위치에 놓여도 읽히는 **동격 삽입** 형태로 쓴다. x249 B_ENGINE 이 이 문자열을 쟀다.
    tpl = str((((a2 or {}).get("call_form") or {}).get("agent_discoverable"))
              or '{tool} (not in your tool list - it is a discoverable tool; the way to run it is '
                 '{call}(agent_tool_name="{tool}"))')
    import t2_precedence as _PCm
    out = {}
    for n in names:
        if n in reg:
            real = n
        else:
            hit = sorted(x for x in reg if _PCm._fam(x) == _PCm._fam(n))
            if len(hit) != 1:
                continue
            real = hit[0]
        out[n] = tpl.replace("{call}", call).replace("{unlock}", unlock or "").replace(
            "{tool}", real)
    return out


def _is_read_tool(env, name):
    """Does the environment itself class this tool as a read?

    Forcing a call is only permitted on the read side ([[05]] §1.5: forcing a read is not
    forcing a write), so the question has to be answered by the environment's own
    declaration rather than by anything we write down. `__tool_type__` is what tau2's own
    metrics use to split writes from reads. Unknown means **not** a read: the conservative
    answer is the one that declines to force.
    """
    for tk in (getattr(env, "tools", None), getattr(env, "user_tools", None)):
        fn = (getattr(tk, "tools", None) or {}).get(name)
        if fn is None:
            continue
        tt = getattr(fn, "__tool_type__", None)
        if tt is None:
            return False
        return "WRITE" not in str(getattr(tt, "name", tt)).upper()
    return False


def _read_routine_pin(agent, a2, messages):
    """남은 **조회**가 전부 read 면 그 집합으로 채널을 좁힌다 (읽기 루틴·2026-08-18).

    사용자 지시 축자: *"남은게 3개면 3개 도구를 루틴으로 연속으로 하면 되지 않나?"*

    ## 왜 이 모양인가
      t7317(050) 실측: 절차 안내는 남은 셋을 **다 이름으로 댔는데** 핀은 하나만 걸렸고, 제출이
      끝나 두 조회가 ready 가 된 뒤에는 트리거가 오지 않았다(체크리스트는 *침묵 3턴*을 기다리고
      손님이 먼저 닫았다). 그래서 트리거를 기다리지 않고, **절차가 살아 있고 남은 것이 전부
      read 인 동안** 유지한다. 하나 부를 때마다 집합에서 빠지고 비면 저절로 풀린다.

    ## 엔진이 하는 일 전부
      ⑴ 활성 절차·ready 노드 = A3 선언 + 관측(`t2_procedure`) ⑵ **환경이 read 로 선언한 것만**
      (하나라도 write 면 None·§1.5 Q5) ⑶ 잠긴 이름이면 `unlock_…`, 풀렸으면 `call_…`
      (048 은 지목된 도구를 8회 부르고 매번 잠금 오류로 죽었다) ⑷ 인자·순서는 모델이 정한다.

    반환: `(도구, 인자명, [이름…])` | None(=루틴 없음·종전 거동)
    """
    try:
        import t2_procedure as _PROC
    except Exception:
        return None
    procs = (a2 or {}).get("procedures") or []
    if not procs:
        return None
    env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
    if env is None:
        return None
    # ⑵ 손님이 방금 말한 턴은 **면제**한다 (2026-08-18). 핀은 `tool_choice=required` 를 함께
    #   걸기 때문에, 답해야 하는 자리에서 걸면 손님 질문을 통째로 무시하게 된다. 073 은 바로
    #   그 자리에서 손님이 대화를 닫았다 — 부작용을 만들 자리를 아예 비운다.
    for _m in reversed(messages or []):
        if getattr(_m, "role", None) in ("user", "assistant", "tool"):
            if getattr(_m, "role", None) == "user":
                return None
            break
    done = _executed_tool_counts(messages)
    unlocked = _unlocked_names(messages, a2)
    for p in _PROC.active_procedures(procs, done):
        st = _PROC.render_state(p, done, unlocked, None)
        cand = [t.strip() for t in str(st.get("ready_tools") or "").split(",") if t.strip()]
        cand = [t for t in cand if t not in done]
        if not cand:
            continue
        if not all(_is_read_tool(env, t) for t in cand):
            continue                      # write 가 섞이면 루틴 없음 (쓰기 강제 금지)
        # ⚠도구 이름은 **A3(L3 `<domain>.specific.json`) 선언에서 읽는다** — 엔진에 도메인
        #   리터럴을 박지 않는다([[05]] 1). 층 정의(`load_domain_a2` 축자): L1 공통 · L2 구조
        #   공통·값만 도메인 · **L3 = 그 도메인에만 있는 도구·규칙**. `dispatcher_role_check`
        #   도 `procedures` 도 L3 이다.
        spec = ((a2 or {}).get("dispatcher_role_check") or {})
        names = spec.get("name_args") or {}
        locked = [t for t in cand if t not in unlocked]
        if locked:                        # 잠긴 것이 있으면 **잠금해제**부터 (048 livelock)
            tool = spec.get("unlock_tool")
            picked = sorted(locked)
        else:
            tool = spec.get("agent_call")
            picked = sorted(cand)
        arg = names.get(tool) if tool else None
        if not (tool and arg):
            return None                   # 선언이 없으면 루틴 없음(미선언 도메인 = 침묵)
        # ⑶ **한 번 시도하고 놓아준다**: 같은 (도구, 집합)으로는 다시 걸지 않는다. 불응하면
        #   그대로 두고, 하나라도 부르면 집합이 줄어 **새 열쇠**가 되므로 이어서 걸린다
        #   ⇒ 진척이 있을 때만 계속된다. 강제를 매 턴 되풀이하지 않는다([[64]]·창 순환 방지).
        # ★지목한 이름을 **한 곳에 적어 둔다** (2026-08-19·t7324 실측). 핀은 스키마를 좁힐 뿐
        #   말하지 않으므로, `stated_names`(메시지에서 찾는다)로는 우리 핀이 안 잡힌다 — 그래서
        #   같은 층의 출처 가드가 우리가 방금 지목한 이름을 `operator-fab` 으로 막았다.
        #   출처는 A3 선언 + env read 분류이고 모델이 지어낼 수 없다([[25]]).
        try:
            _own = set(getattr(agent, "_t2_our_names", None) or set())
            _own |= {str(x) for x in picked}
            agent._t2_our_names = _own
        except Exception:
            pass
        _key = (tool, tuple(picked))
        _tried = getattr(agent, "_t2_routine_tried", None)
        if _tried is None:
            _tried = agent._t2_routine_tried = set()
        if _key in _tried:
            return None
        _tried.add(_key)
        return (tool, arg, picked)
    return None


def _arg_consumers(env, arg):
    """Which tools in this environment actually take an argument by this name.

    `task_048` spent ten messages getting a card's last four digits for a closure, and the
    closure tool does not take them — one tool in the whole domain does. The map from a value
    to the calls that consume it is not something we should write down (a table goes stale and
    a spelling rule guesses); it is readable from the environment's own signatures, so it is
    read. Both sides count: some values are only ever passed by the customer.
    """
    import inspect
    out = set()
    for tk in (getattr(env, "tools", None), getattr(env, "user_tools", None)):
        for name, fn in (getattr(tk, "tools", None) or {}).items():
            try:
                if arg in inspect.signature(fn).parameters:
                    out.add(name)
            except (TypeError, ValueError):
                continue
    return out


def _in_registry(name, registry):
    """이 이름이 env 레지스트리에 **그대로** 있는가. 집합 대조뿐 — 철자 규칙은 쓰지 않는다.

    구판(`_resolve_exact`)은 접미사를 떼어 선언명과 레지스트리명의 **대응을 추정**했다. 그것은
    사실을 패턴으로 판정하는 것이고(사용자 지시 2026-08-05: *"패턴 매칭은 치팅"*), C279가 이미
    *"패턴 규칙은 조용한 오탐을 낳는다"* 로 기록한 자리다. 실측도 그랬다 — 접미사 규칙으로 만든
    `T2_UNLOCK_NAME`은 x99에서 **7발 7오발화**였다(`verify_identity`·`check_cli_eligibility`).

    대응이 필요하면 **선언이 정확한 이름을 말하면 된다**(A2 `follow_up_chains`를 그렇게 고쳤다).
    매핑 표조차 필요 없었다 — 표는 도메인-일반으로 풀리지 않는 경우에만 쓴다.
    """
    nm = str(name or "")
    return nm if nm and nm in (registry or ()) else None


def _docs_naming_fallback(out, tool, corpus):
    """`_docs_naming` 의 **코퍼스 폴백** (2026-08-22·T2_REQUIRE_DOC_DELIVER).

    json 디렉터리(`T2_KB_DOCS_DIR`)가 없거나 비어 도출이 0편이면 **환경이 든 코퍼스**
    (`t2_search.corpus_from_env` 의 `{id: 본문}`)에서 **같은 술어**(도구명 축자 포함)로 도출한다.
    술어 사본이 아니라 재료원 하나를 더 받는 것이고([[67]]), 코퍼스 경유 결과는 캐시하지 않는다
    (객체가 호출마다 같다는 보장이 없다). 도출 0 이면 빈 집합 그대로(호출부가 침묵·로그).
    """
    if out or not corpus or not tool:
        return out
    try:
        return {str(i) for i, t in dict(corpus).items() if tool in str(t or "")}
    except Exception:
        return set()


def _docs_naming(tool, docs_dir, _cache={}, corpus=None):
    """이 도구 이름을 담은 문서 id 집합 — 코퍼스 사실이라 A2에 적지 않는다.

    `transfer_to_human_agents`의 docstring이 *"The proper transfer reason enum can be found in
    the knowledge base: search it before calling this tool"* 라고 스스로 말한다. 그 문서를 읽었는지는
    회수 이력이라 닫혀 있다 — 어느 프로토콜이 맞는지(열린 술어)는 여전히 모델 몫이고, 여기서는
    **읽지 않았다는 사실**만 말한다.
    """
    key = (tool, docs_dir)
    if key in _cache:
        return _docs_naming_fallback(_cache[key], tool, corpus)
    out = set()
    try:
        for f in os.listdir(docs_dir or ""):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(docs_dir, f), encoding="utf-8") as fh:
                d = json.load(fh)
            if tool in (d.get("content") or ""):
                out.add(d.get("id"))
    except Exception:
        out = set()
    _cache[key] = out
    return _docs_naming_fallback(out, tool, corpus)


def _docs_seen(messages):
    """지금까지 도구 출력에 등장한 문서 id — 회수 이력(구조 사실)."""
    txt = []
    for m in messages or []:
        if getattr(m, "role", None) == "tool":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                txt.append(c)
    return chr(10).join(txt)


def _ctx_fits(work, text, min_len=5000):
    """배달물이 이 턴의 생성 창에 들어가는가 → (들어감, 히스토리 자수). **산식 하나**([[67]] 사본 금지).

    원래 `_t2_cp2_pending` 소비 지점에 인라인이던 가드를 함수로 올렸다(거동 동일·2026-08-22) —
    T2_REQUIRE_DOC_DELIVER 가 같은 가드를 쓴다. 대용량(≥`min_len`)만 검사·초과면 호출부가
    **건너뛰고 기록**한다(축약·선별 0 — 엔진이 줄이면 [[62]]③).
    ★제수·오버헤드는 **실측 보정**이다(2026-08-16·t7303 커밋 생성호출 472건의
      `usage.prompt_tokens` 회귀). 자수/토큰 k: p10 3.554·p50 3.986. 그런데 프롬프트에는
      히스토리 밖 **상수 오버헤드**(system+도구 스키마+chat template) O 가 있다:
      p50 10,157·p90 11,069 tok. 초판(`/3`·오버헤드 0)은 실측 콜 463/463 에서
      토큰을 **과소추정**했다(발화점이 5,518자 늦다). 그래서 k=3.5 로 두고 캡에서
      O=11,000 을 뺀다 — 보수 가정에서도 캡을 안 넘는 발화점(C≈85.6k자).
    """
    if not text or len(text) < min_len:
        return True, 0
    try:
        hist = sum(len(_content_str(m) or "") for m in (work or []))
    except Exception:
        hist = 0
    # ★모델 의존 상수를 변수로 (2026-08-31): `44672` 는 **Qwen2.5-32B 의 max_model_len** 이고
    #   `8192` 는 그때의 생성 상한이었다. Q3.8 은 131,072 라 같은 산식이 배달을 필요보다 훨씬
    #   일찍 막는다(모델을 바꿔도 안 따라오는 상수 = [[84]] 사고와 같은 계열).
    #   출처 순서: 런처/프로필 선언 → `t2_run_gated` 의 서버 탐지 → 종전 상수(되돌리기 경로).
    #   산식·보정(k=3.5·O=11,000)은 실측이라 그대로 둔다.
    _cap = int(os.environ.get("T2_MAX_MODEL_LEN") or 44672)
    _gen = int(os.environ.get("T2_AGENT_MAX_TOKENS") or 8192)
    return (hist + len(text)) / 3.5 <= (_cap - _gen - 1024 - 11000), hist


# ★T2_REQUIRE_DOC_DELIVER 배달 헤더 — `x465_transfer_doc_iso.DELIVER_HEAD` **축자**(격리 조건 동형·
#   C578 지시-앞). 어느 문서·어느 프로토콜이 맞는지는 말하지 않는다(열린 술어=모델 몫·[[22]]).
_RDD_HEAD = ("[KB DELIVERY] Read the following before choosing your next action. These are, "
             "in full and verbatim, ALL knowledge-base documents that mention the tool %s.")
# [[64]] 두 칸 — 무엇이 틀렸나(미열람 문서 id 열거) + 무엇을 하면 풀리나(아래를 읽고 고르기). 차단 0.
_RDD_WHY = ("Why you are seeing this: you are about to call %s, and none of the documents that "
            "define it (%s) has been retrieved in this conversation. Nothing is blocked - read "
            "them below, then decide what to call.")


def _require_doc_deliver(agent, a2, messages, tool_calls, corpus=None, docs_dir=None):
    """★033형 — 정의 문서 **미열람**인 채 선언 도구를 시도하는 **그 턴**에 문서 전문을 재생성 버퍼에 싣는다.

    (2026-08-22·정본 `T7336_FORENSIC_033_2026_08_22.md`·격리 C592 `x465_transfer_doc_iso.py`·기본 OFF)
    결손: 모델이 KB 를 grep 한 줄만 보고 문서 본문을 끝내 열지 않아 정책이 요구하는 사슬을 발견조차
      못 한 채 일반 도구를 조기 실행했다(t7328·t7336 동일 = 안정 실패 모드). 우리 층 세 겹은 전부 비켜
      갔다: 절차 enforce 는 사슬 도구 터치가 진입이라 死·표면화는 문서 id 를 안 대고 1회 소진·검색 배달은
      상품 축이라 protocol 문서가 배달물에 없었다.

    [[62]] 4문: ①격리로 쟀다 — x465(n=7/팔): A_asis 일반 7/7 ↔ **B_docfull 사슬 6/7** ↔ N_neg(무내용
      재촉) 일반 7/7 ⇒ 원인은 **미전달**·재촉만으론 0([[57]]). ②따라서 레버는 **전달뿐** — 결정론기 0.
      ③사라지는 모델 판단 0(문서를 읽고 무엇을 부를지는 끝까지 모델). ④엔진은 고르지 않는다 —
      도출 집합 **전부**·코퍼스 축자·헤더 두 줄뿐(지목 문장 0·순위 0).
    [[71]] 4문: ①기능 하나 — 서브 없음. 재생성되는 **메인 턴 하나**가 다음 행동을 고른다(x465 B 팔과
      같은 인터페이스). ②재료는 선언에서 — 도구 집합 = A3 `require_doc_before.tools`, 문서 집합 =
      정본 `_docs_naming`(코퍼스에서 도구명 등장 문서 도출·x465 와 **같은 함수**). 이 코드에 도구명·
      문서 id 리터럴 0. ③전달 = 선언된 id → 코퍼스 정확 집기(검색 0·bm25 0). ④엔진 해석 0.
    ★x465 B 팔과의 동형성(C578 교훈: 지시가 재료 앞·조립 순서 대조):
      · 도출 함수 동일(`_docs_naming`)·집합 전부·순위 0·헤더 첫 줄 축자 동일·지시가 재료 **앞**·
        상한 90,000자 동일(절단 표시)·자리 = 결정점 직전 **문맥 맨 끝**.
      · 차이(명시): 격리는 마지막 **tool** 출력 꼬리에 붙였고, 라이브는 같은 자리의 **user** 메시지
        (비커밋 재생성 버퍼·C298 replay 불변식이 tool 출력 변조를 금한다). 둘째 줄(`_RDD_WHY`)은
        [[64]] 이행으로 추가 — 미열람 id 를 이름으로 대고 차단이 없음을 말한다. 문서마다 `### id`
        헤더 한 줄(C585 `_docs_delivery` 규약·모델이 뒤에 `cat` 할 수 있게).
    ★deny 0 (x93: gold 가 요구한 이관인데 미열람 6건 — 막으면 정답을 막는다). 배달만 하고 강제하지
      않는다. 표면화(`T2_REQUIRE_DOC`)와 같은 턴에 둘 다 나가면 *"검색하라"* ↔ *"여기 있다"* 가
      모순이라([[55]] 문구 모순) 이 배달이 나간 턴엔 표면화를 비운다(플래그 OFF 면 종전 그대로).
    ★반복 규율([[57]] 인자 변화): 같은 턴 안에서는 한 번(버퍼에 이미 실렸다) · 같은 sim 안에서는
      `T2_REQUIRE_DOC_DELIVER_CAP`(기본 3)회 — 재료는 한 턴만 살아 있으므로(비커밋) 미열람 상태의
      **시도마다** 다시 싣되, 모델이 문서를 실제로 `cat` 하면 술어가 닫혀 저절로 침묵한다.
    [[70]] 판 것 = 문맥 +N자/회(x465 실측 +16k)·지연. 성적은 본런 reward A/B 가 확정([[69]]).
    반환: {"text", "tool", "ids", "chars", "truncated", "missing"} · 무발화 = None(로그 위에서).
    """
    if os.environ.get("T2_REQUIRE_DOC_DELIVER") != "1":
        return None
    pdc = (a2 or {}).get("require_doc_before") or {}
    tools = list(pdc.get("tools") or [])
    if not tools:
        return None
    turn = len(messages or [])
    fired = int(getattr(agent, "_t2_rdd_fired", 0) or 0)
    cap = int(os.environ.get("T2_REQUIRE_DOC_DELIVER_CAP", "3"))
    maxc = int(os.environ.get("T2_REQUIRE_DOC_DELIVER_MAX", "90000"))
    if docs_dir is None:
        docs_dir = os.environ.get("T2_KB_DOCS_DIR")
    if corpus is None:
        try:
            import t2_search as _ts
            corpus = _ts.corpus_from_env(
                getattr(getattr(agent, "_t2_orch", None), "environment", None))
        except Exception as _ce:
            print("[T2_REQUIRE_DOC_DELIVER] 코퍼스 조회 실패: %r" % (_ce,), file=sys.stderr, flush=True)
            corpus = {}
    corpus = corpus or {}

    def _trace(nm, rec):
        try:
            from t2_scaffold_get import _isolate_trace
            _isolate_trace({}, {"name": nm}, dict(rec, mode="require_doc_deliver", turn=turn))
        except Exception:
            pass

    seen_txt = _docs_seen(messages)
    for c in (tool_calls or []):
        nm = _exact_tool_name(c)
        if nm not in tools:
            continue
        want = sorted(x for x in (_docs_naming(nm, docs_dir, corpus=corpus) or ()) if x)
        if not want:
            # [[64]] 를 우리 로그에도: 무엇이 없어서 못 했는지 — 코퍼스 경로부터 본다([[55]]).
            print("[T2_REQUIRE_DOC_DELIVER] %s: 정의 문서 도출 0편 — 침묵 (T2_KB_DOCS_DIR=%r · 코퍼스 %d편)"
                  % (nm, docs_dir, len(corpus)), file=sys.stderr, flush=True)
            _trace(nm, {"error": "no_docs", "docs_dir": docs_dir, "corpus_n": len(corpus)})
            continue
        if any(x in seen_txt for x in want):
            continue                    # 이미 읽었다 — 술어 불성립·무발화(로그도 없음: 정상 경로)
        if getattr(agent, "_t2_rdd_turn", None) == turn:
            print("[T2_REQUIRE_DOC_DELIVER] 같은 턴 재배달 생략(버퍼에 이미 실림) tool=%s turn=%d"
                  % (nm, turn), file=sys.stderr, flush=True)
            return None
        if fired >= cap:
            print("[T2_REQUIRE_DOC_DELIVER] cap %d reached — 침묵 tool=%s docs=%d unread turn=%d"
                  % (cap, nm, len(want), turn), file=sys.stderr, flush=True)
            return None
        parts, missing = [], []
        for did in want:
            body = corpus.get(did)
            if body is None:
                missing.append(did)
                continue
            parts.append("### %s\n%s" % (did, body))
        if missing:
            print("[T2_REQUIRE_DOC_DELIVER] 도출 id 가 코퍼스에 없음 %d건(조용히 넘기지 않는다): %s"
                  % (len(missing), missing[:6]), file=sys.stderr, flush=True)
        if not parts:
            _trace(nm, {"error": "no_body", "ids": want, "missing": missing})
            continue
        blob = "\n\n".join(parts)
        cut = len(blob) > maxc
        if cut:
            blob = blob[:maxc] + "\n[... truncated at %d chars by the delivery cap ...]" % maxc
        text = (_RDD_HEAD % nm) + "\n" + (_RDD_WHY % (nm, ", ".join(want))) + "\n\n" + blob
        # 창 추정은 커밋 히스토리(`messages`) 기준 — 재생성 버퍼는 이보다 작거나(뷰 압축) 조금 크다(fb).
        ok, hist = _ctx_fits(messages, text)
        if not ok:
            print("[T2_REQUIRE_DOC_DELIVER] skipped: est %d+%d chars > cap tool=%s turn=%d"
                  % (hist, len(text), nm, turn), file=sys.stderr, flush=True)
            _trace(nm, {"error": "ctx_cap", "ids": want, "chars": len(text), "hist": hist})
            return None
        agent._t2_rdd_turn = turn
        agent._t2_rdd_fired = fired + 1
        try:
            agent._t2_rdd_delivered = set(getattr(agent, "_t2_rdd_delivered", None) or set()) | set(want)
        except Exception:
            pass
        print("[T2_REQUIRE_DOC_DELIVER] deliver tool=%s docs=%d chars=%d turn=%d fired=%d/%d%s unread=%s"
              % (nm, len(parts), len(text), turn, fired + 1, cap, " ⚠절단" if cut else "",
                 ",".join(want)), file=sys.stderr, flush=True)
        _trace(nm, {"ids": want, "n_docs": len(parts), "chars": len(text), "truncated": cut,
                    "missing": missing, "fired": fired + 1})
        return {"text": text, "tool": nm, "ids": want, "chars": len(text),
                "truncated": cut, "missing": missing}
    return None


def _sibling_wait(tag, flagged, what):
    """한 턴의 **다른 호출**에게 보내는 대기 문구 — [[64]]: 무엇이 틀렸나 + 무엇을 하면 풀리나.

    ★왜 고쳤나 (2026-08-11·C416·x247 감사·`T2_KEEP_DENY_BODY` 와 같은 레버로 묶는다):
      구판은 네 자리에서 *"resolve the flagged call first; do not call this yet"* 류를 보냈는데,
      **어느 호출이 문제인지 말하지 않는다**. C413/C414 가 잰 그 병이다 — 같은 계열 문구가
      한 sim 에 3회 이상 나온 6건은 6/6 실패했고, 격리에서 그 문구는 정체 3/8 ↔ 이름을 댄
      본문 0/8 이었다. 여기서는 **문제된 호출의 이름**을 대고 다음 한 수를 준다.
    ⚠거동은 fail-closed 그대로다 — 이 호출은 여전히 실행되지 않는다. 문구만 정보를 얻는다.
    ⚠플래그 OFF 면 종전 문구 그대로(되돌리기 경로 유지).
    """
    name = getattr(flagged, "name", None) if flagged is not None else None
    if os.environ.get("T2_KEEP_DENY_BODY") != "1" or not name:
        return ("Error: [%s] resolve the flagged call first; do not call this one yet." % tag)
    # ★래퍼는 **대상 도구까지** 대야 형제가 갈린다 (2026-08-26·074 상류 정독).
    #   C416 이 이 문구를 만든 이유가 *"어느 호출이 문제인지 말하지 않는다"* 였는데, 디스패처
    #   경유 호출은 **형제가 전부 같은 겉이름**이라 그 수리가 닿지 않았다. 074 turn35·54 실물:
    #   모델이 한 턴에 `unlock_discoverable_agent_tool` 을 여섯 번(서로 다른 inner) 부르고
    #   하나가 막히자 나머지 다섯에 *"'unlock_discoverable_agent_tool' 을 먼저 고쳐라"* 가
    #   갔다 — **여섯 중 어느 것인지 말하지 않는 지시**다. 그 턴에서 정작 필요한
    #   `get_all_user_accounts_by_user_id` 까지 함께 죽었고, 모델은 그 뒤 이관으로 나갔다.
    #   실측(최근 12런 사이드카·태그별): 연쇄 문면 **89건 중 65건(73%)** 의 머리가 래퍼 겉이름
    #   이고 그중 `unlock_…` 이 53건이다.
    #   이 저장소의 정본 판단과 같다 — `t2_forensic.label` 독스트링: *"unlock/give/call 은 대상
    #   도구까지 붙여야 의미가 있다(래퍼 이름만으론 무정보·C470 계기수리)"*.
    #   ⚠선행: C416 은 네 자리에 이 헬퍼를 붙였고, C536 은 fb 조립의 `else` 가 빠진 것을 지적했다.
    #     이 수리는 그 둘과 다른 자리다 — **헬퍼가 쓰이는 자리에서도** 이름이 모호했다.
    #   ⚠도메인 낱말 0 — 디스패처 인자 키는 `_eff_tool_name` 이 이미 읽는 그 셋뿐이고, 값은
    #     **모델이 자기 호출에 쓴 문자열**이다(우리가 고르는 것이 아니다).
    #   ⚠못 대면 종전대로 겉이름만 쓴다(지어내지 않는다·C416 규율 유지).
    try:
        _ar = _args_dict(flagged) or {}
        _inner = (_ar.get("agent_tool_name") or _ar.get("user_tool_name")
                  or _ar.get("discoverable_tool_name"))
        if _inner:
            name = "%s(%s)" % (name, str(_inner))
    except Exception:
        pass
    return ("Error: [%s] this call was not run because another call in the same turn was blocked: "
            "'%s' (see its own error for %s). Fix that one first, then re-issue this call."
            % (tag, name, what))


def _decl_howto(eff, a2, cap=None):
    """거부 문면이 **A2 선언을 가리키게** 한다 — 새 저작 0·엔진은 조회와 나열만.

    ★결손 (2026-08-31·x692 `task_094` 실물): 이름 없는 거부(`_FB_GENERIC`)와 형제-대기 문면은
      *"무엇이 막혔나"* 까지만 말하고 **무엇을 하면 풀리는지**는 안 말한다. 그런데 그 답은 이미
      A2 에 네 겹으로 적혀 있다 — `relations.by_tool[t].requires`(선행 read) ·
      `scaffold_get_tools[].requires_reads` · 그 도구의 `params` 선언 · `arg_source_reads`
      (인자→원천 read). [[64]] 의 두 칸 중 뒷칸을 **선언 인용**으로 채운다.
    ★실패 305 액션의 **275(90%)가 WRONGARG**(handoff 2026-08-31 §2) — 부르기는 부르는데 인자가
      다르다. 그 자리에서 필요한 것은 "다시 하라" 가 아니라 **인자를 어디서 뜨는지**다.
    ⚠[[23]] 출처: A2 선언뿐(gold·tasks 무참조). ⚠[[59]]/[[10]]: 패턴매칭·선택 0 — 선언 조회와
      나열만 한다. ⚠[[24]]: `relations` 를 먼저 보고 없을 때만 `scaffold_get_tools` 로 폴백한다.
    ⚠거동 불변 — 이 호출은 여전히 실행되지 않는다(fail-closed). 문면만 정보를 얻는다.
    ⚠못 대면 **빈 문자열**(지어내지 않는다·C416 규율).
    """
    try:
        cap = int(cap if cap is not None else os.environ.get("T2_DENY_HOWTO_CAP", "900"))
    except Exception:
        cap = 900
    name = str(eff or "")
    if not name or not a2:
        return ""
    try:
        import t2_precedence as _PC
        fam = _PC._fam(name)
    except Exception:
        fam = name
    reads = []
    try:
        rel = ((a2 or {}).get("relations") or {}).get("by_tool") or {}
        for k in (name, fam):
            for r in ((rel.get(k) or {}).get("requires") or []):
                if r not in reads:
                    reads.append(r)
    except Exception:
        pass
    decl = None
    try:
        for d in ((a2 or {}).get("scaffold_get_tools") or []):
            if not isinstance(d, dict):
                continue
            if (d.get("name") or d.get("tool")) in (name, fam):
                decl = d
                break
    except Exception:
        decl = None
    if decl is not None:
        for r in (decl.get("requires_reads") or []):
            if r not in reads:
                reads.append(r)
    params = []
    if decl is not None and isinstance(decl.get("params"), dict):
        for k, v in decl["params"].items():
            if str(k).startswith("_"):
                continue
            params.append("%s: %s" % (k, str(v)))
    if not reads and not params:
        return ""
    out = ""
    if reads:
        out += (" Its declaration lists the reads this call depends on: %s — run those first and "
                "copy the values from their output." % ", ".join(reads))
    # ★[[70]] 무엇을 파나 — `params` 는 **주입 스키마에 이미 실려 있다**(`t2_scaffold_get`
    #   `injected name=… desc=…ch params=[…]`). 여기 다시 실으면 같은 텍스트가 문맥에 두 번
    #   앉는다([[65]] 부하). 그래서 기본은 **선행 read 만**이고, 인자 계약은 옵트인이다.
    #   선행 read 는 스키마 어디에도 없으므로 이쪽이 순증분이다.
    if params and os.environ.get("T2_DENY_HOWTO_PARAMS") == "1":
        body = " ".join("[%s]" % p for p in params)
        out += " Its declared argument contract is: " + body
    out = out if len(out) <= cap else (out[:cap] + " [...]")
    return " [HOW-TO from the declaration of '%s']%s" % (name, out)


def _degenerate_axes(po):
    """**고를 것이 없는 축** = A3 `doc_index[군]` 의 계열 집합이 `_general_` 뿐인 군 (닫힌 술어).

    ★A14 / OL-15 (2026-08-22 · t7336 마스터 §6.1 A14 · §5.2): `bank_accounts_bank_accounts` 는
      색인에 계열이 `_general_` 하나뿐이다. 그런데 결정 서브(`decide_from_docs`)는 *반드시 하나를
      고르게* 돼 있어 그 자리에서 표시명 `General` 이 나왔고, 그것이 `decided_by_docs_text`
      (*"It answers: X."*) 에 실려 `T2_DECISION_CARRY` → `T2_DECIDE_BEFORE_WRITE` 로 write
      결정점까지 갔다. 085#1 은 그 값을 `dispute_category` 로 **11/11 제출**해 전부 env 거부됐고
      KB 031 의 enum 9종에 `General` 은 없다(242자 바이트 검산 일치·[S]).

    ⇒ *"고를 것이 없다"* 는 **선언에서 기계 도출되는 사실**이다 — 집합 하나의 원소 검사이고
      도메인 판단·유사도·의도 해석 0([[22]]·[[59]]·[[66]]). *"It answers: X."* 는 고를 것이
      **실재할 때만** 참이므로, 퇴화 축에서는 결정문을 만들지 않는다([[25]] 100% 정답 의무).
    ⚠이 함수는 **선언만** 본다(대화·gold·env 무관). 색인이 없으면 빈 집합(fail-open).
    """
    idx = (po or {}).get("doc_index") or {}
    return {g for g, subs in idx.items() if not _subject_keys(subs)}


def _served_subjects(po, group, delivered=None, decided=None):
    """이번 배달분이 **덮은 대상 계열**(닫힌 집합 = A3 `doc_index[군]` 키·`_general_` 제외).

    ★왜 (T7336 016 포렌식·2026-08-21·정본 `T7336_FORENSIC_016_2026_08_21.md`): 축-소진 키가
      **군 하나**면 "무엇이 배달됐나"가 기록에 없어, 배달 이후 **대상(계열)이 바뀐 재수요**를
      원리상 볼 수 없다. 소진 키를 (군, 배달된 계열 집합) 으로 좁히는 첫 재료가 이 함수다.
    ⚠집합 대조뿐이다([[59]]·[[22]]): 문서-본문 배달이면 실린 **문서 id 헤더**(`[id]`) 로,
      결정문 배달이면 닫힌 계열 표시명(`_slug_disp` 규약 하나)이 결정문에 **축자로**
      들어 있는가로 본다(대소문자만 접음·정규식 0·뜻 해석 0·계열명은 전부 A3 에서 읽는다).
    """
    idx = ((po or {}).get("doc_index") or {}).get(group) or {}
    dec = " ".join(str(decided or "").split()).lower()
    mat = str(delivered or "")
    out = set()
    _keys = _subject_keys(idx)
    for s, ids in idx.items():
        if s not in _keys:
            continue
        if dec and _slug_disp(s).lower() in dec:
            out.add(s)
        elif mat and any(("[%s]" % d) in mat for d in (ids or ())):
            out.add(s)
    return out


def _record_served(agent, po, group, messages, delivered=None, decided=None):
    """배달 이력(어느 계열이·언제) 기록 — 속성 부기만, 출력·거동 0 (플래그 OFF 도 동일)."""
    try:
        sv = dict(getattr(agent, "_t2_search_served", None) or {})
        sv[group] = set(sv.get(group) or set()) | _served_subjects(po, group, delivered, decided)
        agent._t2_search_served = sv
        sa = dict(getattr(agent, "_t2_search_served_at", None) or {})
        sa[group] = len(messages or [])
        agent._t2_search_served_at = sa
    except Exception:
        pass


def _rearm_subjects(agent, po, gs, done, messages):
    """★T2_SEARCH_REARM 술어부 — 어느 소진 군에 **미배달 계열의 재수요**가 있는가.

    (T7336 016 포렌식 처방 1·2026-08-21·정본 `T7336_FORENSIC_016_2026_08_21.md` §레버 대조)
    결손: `_t2_search_done` 소진이 군 단위 **영구 잠금**이라, 계열 X 의 결정문만 배달된 채
    축이 닫힌 뒤 대화가 다른 계열 Y 를 축자로 확정해도 재요청이 전부 *"모두 처리됨 — 침묵"*
    이 되어 Y 의 요건 문서 전달 경로가 구조적으로 소멸했다(그 sim 의 유일한 KB 채널).

    술어는 전부 닫혔다([[22]]·유사도·의도 해석 0·[[59]]·[[66]]):
      ⑴ 계열 = A3 `doc_index[군]` 키(닫힌 집합) · 표시명은 `_slug_disp` 규약 하나.
      ⑵ 재수요 = 그 군의 **배달 시점 이후** user/assistant 발화에 계열 표시명이 **축자 등장**.
         ⚠`T2_REARM_USER_ONLY=1`(A-3′·아래 주석) 이면 이 칸이 **손님 발화 · 전 접두**로 바뀐다.
         등장 판정은 `t2_search.groups_in` 정본 파싱부 재사용([[67]] 사본 금지) — 긴 이름
         안의 포함 등장은 세지 않는다. 우주는 **전체 색인**의 표시명이라 그 억제가 군 경계
         밖에서도 작동한다.
         ⚠도구 출력은 안 본다 — 레코드 덤프는 전 계열명을 담을 수 있어 수요가 아니다.
           역할 필터는 위치·역할 술어이지 내용 판단이 아니다.
      ⑶ 신규 = 그 계열이 `_t2_search_served[군]`(배달된 계열 집합)에 **없다**.
    반환: (군, [신규 계열 슬러그…]) — 없으면 (None, None).
    ⚠같은 대상 재수요는 ⑶이 걸러 **여전히 침묵**(재요청 루프 방지 보존). 배달 이력이 없는
      군(`served_at` 밖)은 건드리지 않는다(모르면 안 연다·[[25]]).
    """
    served = getattr(agent, "_t2_search_served", None) or {}
    served_at = getattr(agent, "_t2_search_served_at", None) or {}
    idx_all = (po or {}).get("doc_index") or {}
    disp = {}
    for g2 in idx_all:
        for s in _subject_keys(idx_all.get(g2)):
            disp.setdefault(_slug_disp(s), set()).add((g2, s))
    if not disp:
        return None, None
    import t2_search as _ts
    # ★T2_REARM_USER_ONLY = 문서의 **A-3′** (2026-08-26·측정 `x553_rearm_role_split.py`·기본 OFF)
    #   문서 처방 A-3(`TASK_055.md:234`)는 *"재수요를 user 발화로 한정"* 이다. 그것만 하면
    #   **발화 63/78(81%)** 이 죽고 그중 통과 sim 의 발화가 10건(반증 견딘 것 8)이다 — 순손실이다.
    #   창을 **전 접두**로 함께 되돌리면(= 배달 이전에 손님이 부른 것도 센다) 죽는 발화가
    #   **27/78**, 통과 sim 발화는 3건이고 **반증을 견디는 것은 1건**뿐이다. 그러면서 표적
    #   세 태스크(016 6/7 · 055 4/4 · 057 6/7)는 **두 판이 완전히 같다** ⇒ [[70]] 절충은 A-3′ 다.
    #   왜 창까지 되돌리나: user-only 로 좁히면 *손님이 배달 **이전에** 부른 계열*이 통째로
    #   빠지는데, 그것이 바로 016 의 원래 결손(Bronze 만 배달된 채 Silver 가 닫힌 자리)이다.
    #   두 조각은 하나의 처방이라 한 플래그로 묶는다. OFF 면 바이트 동일.
    _uonly = os.environ.get("T2_REARM_USER_ONLY") == "1"
    _roles = ("user",) if _uonly else ("user", "assistant")
    for g in gs:
        if g not in done or g not in served_at:
            continue
        _from = 0 if _uonly else int(served_at.get(g) or 0)
        post = "\n".join(
            _content_str(m) for m in (messages or [])[_from:]
            if getattr(m, "role", None) in _roles
            and getattr(m, "content", None))
        if not post.strip():
            continue
        hits = _ts.groups_in(post, sorted(disp))
        new = sorted({s for d in hits for (g3, s) in disp[d]
                      if g3 == g and s not in (served.get(g) or set())})
        if new:
            return g, new
    return None, None


def _search_material(agent, a2, messages, decide=True):
    """검색 에이전트의 **한 줄 진입점** — 재료가 원장이 아니라 **문서** 쪽에 있는 계열용.

    ① LLM 이 문서군을 고른다(닫힌 집합 = A3 `doc_index` 키) → ② 엔진이 색인대로 읽고 **효력
    없는 문서를 뺀다** → ③ LLM 이 남은 것 중 고른다 → 그 답을 A2 문구에 실어 돌려준다.

    측정 (`x248`·`x250`·071 실물·n=8·프로덕션 경로): 두 축 **8/8**. 부정 통제 — 문서만 주면
    checking **0/8**, 만료를 안 빼면 savings **0/8**. ⇒ 엔진의 유일한 일이 값을 산다.

    ⚠엔진은 고르지 않는다([[10]]·⛔0 ③). ⚠재료를 못 만들면 **빈 문자열**(침묵).
    ⚠코퍼스는 환경이 든 것을 읽는다 — 경로 하드코딩 0([[05]]).
    """
    if not a2:
        return ""
    # ★T2_PROCEED_DOCBODY 중앙 스위치 (2026-08-16·t7304·심사 3인 일치 지적): 플래그가 켜지면
    #   **모든 호출 자리**에서 문서 본문을 돌려준다. 한 자리만 decide=False 로 플립하면 다른
    #   자리(MATERIAL_BYPASS·VIEW_FB·DECIDE-FIRST)가 그 축을 **먼저 소비**해(`_t2_search_done`
    #   전역·영구 잠금) 문서가 그 축에 영영 못 오는 누수가 생긴다 — 유료 런으로 배관 사실을
    #   사게 되는 형태(t7303 동형). 스위치를 함수 안에 두면 호출 자리 5곳이 자동으로 덮인다.
    #   플래그 OFF 면 인자 그대로(바이트 불변).
    if (os.environ.get("T2_PROCEED_DOCBODY") == "1"
            or os.environ.get("T2_DOCS_AT_WRITE") == "1"):
        decide = False
    import t2_search as _ts
    _po = (a2.get("policy_ontology") or {})
    _groups = list(_po.get("doc_index") or {})
    if not (_groups and _po.get("group_prompt") and _po.get("doc_decide_prompt")
            and _po.get("decided_by_docs_text")):
        return ""
    _env = getattr(getattr(agent, "_t2_orch", None), "environment", None)
    _corpus = _ts.corpus_from_env(_env)
    if not _corpus:
        print("[T2_SEARCH_AGENT] 환경에서 문서를 못 찾음 — 침묵", file=sys.stderr, flush=True)
        return ""
    _tx = [_content_str(_m) for _m in (messages or [])
           if getattr(_m, "role", None) in ("user", "tool")]
    # ★군 형식화에는 **손님의 말**을 준다 (2026-08-11 라이브 교정): 첫 판은 `user+tool` 의 마지막
    #   셋을 줬는데 그 셋이 대개 **도구 출력**이라 요청이 안 보였다 — 070 이 개인 체킹
    #   (`checking_accounts`)을 골랐다(gold 는 `business_checking_accounts`). x252 가 이미
    #   *요청이 머리에 와야 한다*를 쟀는데(C417⒠) 라이브 입력에서 그 규약을 어겼다.
    _users = [_content_str(_m) for _m in (messages or [])
              if getattr(_m, "role", None) == "user"]
    _ask = " --- ".join(_users[-4:] or _tx[-3:])[-6000:]
    import tau2.agent.llm_agent as _la
    from tau2.data_model.message import UserMessage as _UM
    # ★**축별 결정점** (2026-08-11·C419·사용자 지시 *"축별로 분리해서 결정점을 나누게 하라"*).
    #   071 은 요청이 둘이다(사업자 체킹 + 사업자 세이빙). 구판은 군을 하나만 고르고 **sim 당 1회**
    #   잠갔다 — 라이브에서 개인 체킹을 골라 손님이 **이미 가진** 계좌를 추천하고 그대로 끝났다.
    #   ⇒ LLM 이 **요청마다 하나씩** 군을 대고, 엔진은 아직 안 다룬 것을 **한 결정점에 하나씩**
    #     처리한다. 재료가 축마다 37K자라 한 턴에 다 싣는 것은 불가능하기도 하다.
    #   ⚠엔진은 고르지 않는다 — LLM 이 **답한 순서**대로 담고, 그 순서대로 꺼내 쓸 뿐이다.
    # ★T2_ELIG_LINE (2026-08-18·자격축 **상류 전용**·기본 OFF): 손님의 자격(개인/사업자) 한 줄을
    #   **요청 앞**에 싣는다. 판정은 LLM, 엔진은 닫힌 값 + 인용 실재만 본다(`t2_search.eligibility_line`).
    #   근거: `x364b`(n=27·짝지음) 요청 밖 군 11→6 · `business_*` 3→0 · 적중 27/27 불변 ·
    #         부정통제(자격 뒤집기) 오선택 3→**26**·적중 5 손실 ⇒ 내용이 원인.
    #         `x364c` 문면 수리로 라벨 22/29 → **29/29**.
    #   ⚠하류(클래스 결정)에는 **안 싣는다** — 31축 중 답이 바뀐 축 0(`x364`). 군 안 후보는 이미 그
    #     자격 것뿐이라 가를 수가 없다. 여기 한 자리에만 붙인다.
    #   ⚠못 만들면(값 미확정·인용 미검산) 빈 문자열 → 종전 경로와 **바이트 동일**(fail-safe·[[25]]).
    if os.environ.get("T2_ELIG_LINE") == "1":
        try:
            _el = _ts.eligibility_line(agent, _la, _UM, _po, _ask)
            if _el:
                _ask = _el + "\n\n" + _ask
        except Exception as _ele:
            print("[T2_ELIG] 실패(종전대로): %r" % (_ele,), file=sys.stderr, flush=True)
    _gs = _ts.formalize_groups(agent, _la, _UM, _po, [_ask], _groups)
    _done = getattr(agent, "_t2_search_done", None)
    if _done is None:
        _done = set()
    # ★A14 / OL-15 (2026-08-22 · t7336 마스터 §6.1 A14 · §5.2): **퇴화 축**(A3 색인의 계열이
    #   `_general_` 뿐)은 **결정 경로에서 제외**한다. 고를 것이 없는 자리에서 결정 서브에게
    #   *"하나 고르라"* 고 하면 나오는 것은 답이 아니라 표시명(`General`) 이고, 그것이
    #   `decided_by_docs_text` (*"It answers: X."*) 에 실려 write 결정점까지 갔다 —
    #   085#1 `dispute_category` **11/11 env 거부**([S]·§5.2 OL-15).
    #   ⚠`decide=False`(문서-본문 배달) 경로는 **답을 만들지 않으므로** 종전대로 둔다: 그 경로는
    #     문서를 나를 뿐이고 고르는 일은 끝까지 모델 몫이다([[62]] ③ 보존).
    #   ⚠[[70]] **무엇을 파는가** = 그 축의 `DOCDECIDE` 배달 수. 이 수리 뒤 퇴화 축에서
    #     `[T2_SEARCH_AGENT] 축 처리 완료` 와 `decided_by_docs_text` 가 0 이 되고, 그 대신
    #     `[T2_DEGENERATE_AXIS]` 줄이 선다 — 그 줄 수 = 판 배달 수. 그 축의 값은 이제 모델이
    #     KB 를 읽고 정한다(엔진 지목 0). 다른 축이 남아 있으면 결정점은 그쪽으로 넘어간다.
    _degen = _degenerate_axes(_po) if decide else set()
    _skip_degen = [g for g in _gs if g in _degen and g not in _done]
    if _skip_degen:
        print("[T2_DEGENERATE_AXIS] 결정 미배달 group=%s — 색인의 계열이 `_general_` 뿐이라 "
              "고를 것이 없다(퇴화 축·A3 선언에서 기계 도출)" % ",".join(_skip_degen),
              file=sys.stderr, flush=True)
    _g = next((g for g in _gs if g not in _done and g not in _degen), None)
    # ★관측 전용 계기 (2026-08-18·C517⒟) — **거동 불변**. 군→클래스 이득의 *순서·소모 채널*은
    #   라이브에서만 잰다(후보집합 채널은 격리에서 0 으로 나왔다: gold 군 적중 27/27).
    #   기록: ⑴모델이 답한 **첫 군** ⑵이번에 처리하는 군 ⑶지금까지 소모한 결정점 수 ⑷군 개수.
    #   gold 대조는 **사후 분석**에서 한다 — 엔진은 요청 군이 무엇인지 모르고, 알아서도 안 된다.
    #   ⚠이 마크가 S3 전에 있어야 그 런에서 이 채널을 잴 수 있다(끝나고 넣으면 못 잰다).
    try:
        print("[T2_GROUPORDER] first=%s this=%s consumed=%d n_groups=%d order=%s"
              % (_gs[0] if _gs else "-", _g or "-", len(_done), len(_gs), ",".join(_gs)),
              file=sys.stderr, flush=True)
    except Exception:
        pass
    # ★T2_SEARCH_REARM (2026-08-21·T7336 016 처방 1·기본 OFF·[[70]] A/B 는 x464): 소진 키를
    #   군 → **(군, 배달된 계열 집합)** 으로 좁힌다. 군 단위 영구 잠금은 배달 **이후** 대상이
    #   바뀐 재수요를 못 본다 — 그 sim 의 유일한 KB 채널이 구조적으로 닫혔다(포렌식 §레버 대조).
    #   재무장 시 배달은 신규 계열의 **문서 델타만**(선언 id → 정확 집기·[[71]]·아래 doc-only
    #   반환) — 서브 재결정 없음. 같은 대상 재수요는 served 집합이 걸러 **여전히 침묵**.
    #   ⛔0 [[62]] 4문: ①결손은 t7336 궤적+로그 전수([S]·정본 문서) ②재료가 닿으면 모델이
    #   쓰는 것은 기측정(x248 8/8·x335b 24/24) ⇒ 레버는 전달 재개뿐 ③사라지는 모델 판단 0
    #   (무엇을 읽고 어떻게 쓸지는 모델 몫 그대로) ④순위·최댓값·지목 문장 0.
    #   [[05]] 3문: ⑴도메인-특화 순증 0(계열·군·문서 id 전부 A3 선언에서 읽음·엔진 리터럴 0)
    #   ⑵유동 판단 동결 0 ⑶도메인 행동 수행 0(정책 문서 읽기 = 우리 층 몫·C405ⓔ 확정 경계).
    _rearm_new = None
    if not _g:
        if _gs and os.environ.get("T2_SEARCH_REARM") == "1":
            _rg, _rnew = _rearm_subjects(agent, _po, _gs, _done, messages)
            if _rg:
                _g, _rearm_new = _rg, _rnew
                print("[T2_SEARCH_REARM] group=%s 신규 대상 %s (기배달 %s) — 소진 해제·문서 델타"
                      % (_g, ",".join(_rnew),
                         ",".join(sorted((getattr(agent, "_t2_search_served", None) or {})
                                         .get(_g) or ())) or "-"),
                      file=sys.stderr, flush=True)
        if not _g:
            if _gs:
                print("[T2_SEARCH_AGENT] 요청 축 %s 모두 처리됨 — 침묵" % ",".join(_gs),
                      file=sys.stderr, flush=True)
            return ""
    import t2_ledger as _lg
    # ★`now_prompt` 는 `policy_ontology` 가 아니라 **`ledger_metrics`** 에 선언돼 있다.
    #   첫 판은 `_po` 를 넘겨 tpl 이 없어 **항상 None** 이었고, 그러면 `drop_expired` 가
    #   아무것도 안 뺀다 — 로그에는 `뺀 것 0` 으로만 남아 **성공처럼 보인다**(라이브 실패 3회째).
    #   그리고 그 값은 선언 형식(`%m/%d/%Y`)이라 ISO 비교 전에 **정규화**해야 한다.
    _nspec = next((s for s in (a2.get("ledger_metrics") or ()) if s.get("now_prompt")), None)
    # ★시계 출력을 **맨 앞에** 놓는다 (2026-08-12·C441). `formalize_now` 는 발췌를 **위치**로
    #   한다(head 3 + tail 8). 071 실측: `get_current_time` 을 msg 13 에 불렀는데 대화텍스트가
    #   24~28 이라 그 출력이 head 에도 tail 에도 못 들어가 **원값 None ×8** → 검색 서브가 통째로
    #   침묵했다(070 은 대화텍스트 10 이라 전체가 덮여 성공). 모델 능력이 아니라 **우리가 그
    #   문장을 안 보여 준 것**이다 — C437 과 같은 병(격리 계약 위반)이다.
    #   ⚠고르는 기준은 **도구 이름**(A2 `now_tool`·env 도출)이지 내용이 아니다([[59]] — 어느
    #     문장이 날짜인지 판정하는 것은 여전히 모델 몫이고, 엔진은 자리만 바꾼다).
    #   ⚠못 찾으면 종전대로(빈 목록) — 거동 회귀 0.
    _ntool = (_nspec or {}).get("now_tool")
    _nowtx = []
    if _ntool:
        try:
            _byid = {}
            for _m0 in (messages or []):
                for _tc0 in (getattr(_m0, "tool_calls", None) or []):
                    _byid[getattr(_tc0, "id", None)] = getattr(_tc0, "name", None)
            for _m0 in (messages or []):
                if getattr(_m0, "role", None) == "tool" \
                        and _byid.get(getattr(_m0, "id", None)) == _ntool:
                    _nowtx.append(_content_str(_m0))
        except Exception as _nte:
            _nowtx = []
            print("[T2_SEARCH_AGENT] now_tool 선별 실패(종전대로): %r" % (_nte,),
                  file=sys.stderr, flush=True)
    # ★시계가 **아직 안 불린** 자리에서는 엔진이 직접 부른다 (2026-08-15·`T2_NOW_SELFCALL`).
    #   왜 필요한가(t7295 실측·071): 검색 에이전트의 창은 A2 `action_tools` 푸시 **결정점에서만**
    #   열리는데, 071 세 sim 통틀어 그 줄은 **단 1개**였고 그 한 번이 시계보다 **앞**이었다
    #   (로그 2282 침묵 ↔ 시계 2424). 나머지 두 sim 은 창이 **아예 안 열렸다**. 코드가 약속한
    #   *"다음 결정점에서 재시도"* 는 두 번째 결정점이 없어 **공약**이 됐고, 그래서 만료 제거
    #   기계가 한 번도 재료를 못 냈다 — `now 미확정` 은 arm b 침묵 80회 중 **1위 사유**다.
    #   ⚠분담: 엔진은 A2 가 **이름으로 선언한** 도구를 부를 뿐이다 — 어느 문장이 날짜인지는
    #     여전히 `formalize_now`(LLM) 가 정한다([[59]] 유지·엔진은 문자열을 해석하지 않는다).
    #   ⚠부작용: `get_current_time` 은 상수를 돌려주는 **순수 읽기**(tools.py:387·DB 무접촉)라
    #     궤적·DB 해시를 건드리지 않는다. 그래도 실패하면 **종전대로**(빈 목록) 간다.
    #   ⚠기본 OFF — 켜지 않으면 바이트 동일(098·100 불변 의무·[[57]]).
    #
    #   ⛔0 [[62]] 자기점검 (4문):
    #     ①격리로 쟀나 — 쟀다. `x248`·`x250`(n=8·프로덕션 경로) 두 축 **8/8**. 부정통제:
    #       고지 없이 checking **0/8** · 만료 미제거 savings **0/8**. 오늘 추가: t7295 3축
    #       미도달(궤적 0·사이드카 0·질의 0) · `t2_liveness` 34전달/80침묵 · `x325` 영향반경.
    #     ②격리에서 성공하나 — **성공한다(8/8)**. ⇒ 살 것은 **전달뿐**이고 이 편집이 정확히
    #       그것이다. 계산도 판단도 대신하지 않는다.
    #     ③모델이 하던 판단 중 사라지는 것 — **없다**. 모델은 `get_current_time` 을 여전히
    #       스스로 부른다(071 에서 3/3). 이 호출은 **검색 서브에게 줄 재료**를 만드는 내부
    #       경로일 뿐 모델의 선택지를 줄이지 않는다.
    #     ④엔진이 순위·최댓값·지목 문장을 내나 — **안 낸다**. 이름으로 도구 하나를 부르고
    #       돌아온 문자열을 그대로 넘길 뿐이다.
    #
    #   [[05]] 3문: ⑴도메인-특화 순증 **0**(`now_tool` 은 C441 이 이미 넣은 키·새 키 0·코드에
    #     도메인 어휘 0) · ⑵유동 판단 동결 **없음**(날짜 형식화·문서군·최종 선택 전부 LLM) ·
    #     ⑶scaffold 가 도메인 행동 수행? — 가장 날카로운 질문이다. 이 도구는 **상수 반환·DB
    #     무접촉·READ** 이고 *대화와 무관한 고정 상수*라 `t2_search` §경계가 이미 선언한
    #     "정책 문서 읽기 = 우리 층 가능" 과 같은 부류다. **고객 DB 읽기는 손대지 않는다**
    #     (대화마다 달라지는 도메인 행동 = 모델 몫·경계 불변).
    if (not _nowtx and _ntool and os.environ.get("T2_NOW_SELFCALL") == "1"):
        try:
            _envc = getattr(getattr(agent, "_t2_orch", None), "environment", None)
            _res = _envc.make_tool_call(tool_name=_ntool, requestor="assistant")
            if _res:
                _nowtx.append(str(_res))
                print("[T2_NOW_SELFCALL] %s 직접 호출 — 시계 확보" % (_ntool,),
                      file=sys.stderr, flush=True)
        except Exception as _sce:
            print("[T2_NOW_SELFCALL] 실패(종전대로): %r" % (_sce,),
                  file=sys.stderr, flush=True)
    _now_raw = _lg.formalize_now(agent, _la, _UM, _nowtx + _tx, _nspec) if _nspec else None
    _now = _ts.to_iso(_now_raw, tuple((_nspec or {}).get("date_formats")
                                      or ("%m/%d/%Y", "%Y-%m-%d")))
    # ★`now` 를 모르면 **내보내지 않는다** (2026-08-11·라이브 4회째 교정).
    #   이유는 규율이 아니라 **측정**이다: 만료를 안 뺀 재료는 x248 의 `W_EXPIRED` 팔과 같은
    #   구성이고 그 팔은 savings 축에서 **0/8** 이었다(checking 은 8/8 — 한 축만 봤으면 못 봤다).
    #   즉 *엔진이 제 일을 못 하는 상태의 재료는 이득이 아니라 해악*이다.
    #   ⇒ 침묵하고 **잠그지도 않는다**: 호출부는 `_m3` 가 비면 `_t2_searchagent_fired` 를 안 세우므로
    #     에이전트가 `get_current_time` 을 부른 **다음 결정점에서 다시 시도**한다. 첫 판은 이 구분이
    #     없어 `now` 미확정인 첫 자리에서 내보내고 그대로 잠겼다(라이브 `뺀 것 0`).
    #   ⚠[[25]] 와도 같은 방향이다 — 모르면 빼지 않고, 뺄 수 없으면 말하지 않는다.
    if not _now:
        print("[T2_SEARCH_AGENT] now 미확정 — 침묵(잠그지 않음·다음 결정점에서 재시도) "
              "(스펙 %s · 원값 %r · 대화텍스트 %d)"
              % ("있음" if _nspec else "**없음**", _now_raw, len(_tx)),
              file=sys.stderr, flush=True)
        return ""
    # ★재무장 자리는 **신규 계열의 문서 델타만** 집는다(T2_SEARCH_REARM·선언 id → 정확 집기).
    #   전체 군을 다시 실으면 첫 배달과 같은 재료가 반복돼 이득이 아니라 부피다([[57]] 인자 변화).
    #   `per_doc` 는 정본 상수 `VERDICT_PER_DOC` 재사용([[67]]) — 기본 400자는 요건 문장이
    #   잘릴 수 있다(첫 배달은 결정문이었지 본문이 아니었다).
    if _rearm_new:
        _mat, _info = _ts.material_for(a2, _g, now=_now, corpus=_corpus,
                                       subjects=_rearm_new, general=False,
                                       windowed="none", per_doc=_ts.VERDICT_PER_DOC)
    else:
        _mat, _info = _ts.material_for(a2, _g, now=_now, corpus=_corpus)
    # ★`now` 가 없으면 **엔진의 유일한 일이 죽는다** — 그런데 `formalize_now` 는 실패를 인쇄하지
    #   않아서 `뺀 것 0` 한 글자가 유일한 단서였다(2026-08-11 라이브에서 두 판을 이걸로 태웠다).
    #   [[64]] 를 우리 로그에도 적용한다: **무엇이 없어서 못 했는지**를 말한다.
    # ★`turn=` 추가(2026-08-16·인쇄 전용·거동 0): P1 의 1차 종점이 배달 **횟수**가 아니라
    #   *"**첫 지목 이전**에 도달했는가"* 로 바뀌었기 때문이다(055·024 공통 기전 — 지목이 박히면
    #   그 뒤 재료는 안 먹는다). 턴을 안 찍으면 그 지표를 기계가 셀 수 없어서 **순서로 추정**하게
    #   되는데, 그것이 오늘 두 번 오진을 낳은 형태다([[08]]·[[55]]). `MATERIAL_GATE` 는 이미 찍는다.
    print("[T2_SEARCH_AGENT] group=%s · 문서 %d(뺀 것 %d: %s) · now=%s turn=%d "
          "(스펙 %s · 원값 %r · 대화텍스트 %d)"
          % (_g, _info["kept"], len(_info["dropped"]), ",".join(_info["dropped"])[:80], _now,
             len(messages or []),
             "있음" if _nspec else "**없음**", _now_raw, len(_tx)),
          file=sys.stderr, flush=True)
    if not _mat:
        return ""
    # ★재무장 배달 = **doc-only**(서브 재결정 없음·[[71]] 읽어 전달만). 그 (군, 신규 계열)은
    #   여기서 배달-완료로 적어 **1회로 묶는다** — 만료로 전부 빠져 제외-사유만 나가도 그것이
    #   그 계열에 대한 답이므로 적는다(재시도 루프 방지·[[57]]). 축 잠금(`_done`)은 그대로다 —
    #   열리는 것은 (군, 신규 계열) 델타뿐이고 다음 신규 계열은 다음 재수요가 연다.
    if _rearm_new:
        try:
            _sv = dict(getattr(agent, "_t2_search_served", None) or {})
            _sv[_g] = set(_sv.get(_g) or set()) | set(_rearm_new)
            agent._t2_search_served = _sv
            _sa = dict(getattr(agent, "_t2_search_served_at", None) or {})
            _sa[_g] = len(messages or [])
            agent._t2_search_served_at = _sa
        except Exception:
            pass
        print("[T2_SEARCH_REARM] group=%s 델타 배달 %d자 (문서 %d·뺀 것 %d) turn=%d"
              % (_g, len(_mat), _info["kept"], len(_info["dropped"]), len(messages or [])),
              file=sys.stderr, flush=True)
        return _mat
    # ★결정 ask = **후보 줄만** (2026-08-12·x269·사용자 지시: *"격리 서브에이전트에 A3 를
    #   통해서 정책을 리마인드하게 하라"*). 두 가지를 동시에 고친다:
    #   ⑴ **격리 계약 복원** — `decide_from_docs` 독스트링 축자 *"대화 잔여물은 한 글자도
    #     없다"* 인데 여기서 손님 발화 4개를 통째로 넣고 있었다. x269 실측: 대화를 실으면
    #     checking 0/8(손님이 수락한 오답 `True Blue` 를 복창)·빼면 8/8 — 기여 0·해악만.
    #   ⑵ **명명 정책 리마인드** — 정책 (general)_003 축자 *"account_class options include
    #     Navy Blue, Cobalt Blue, True Blue, etc."* = 맨이름. 서브는 37K자 속 그 한 줄을
    #     놓치고 상품 문서 제목형(`Sky Blue Account`)으로 답했고 그 형태는 인자까지 살아남아
    #     채점 칸을 죽인다(x268 `S_SUFF` 0/8 ↔ `S_BARE` 7/8). 후보 = A3 `doc_index[군]` 키의
    #     기계 전개(출처 = env 파일명뿐·x244·gold 무관) — **엔진은 선별하지 않고 전부 싣는다**
    #     ([[59]]·비-계좌 주어 포함). 문구 = A2 `decide_candidates_text`(측정한 그 문자열).
    #   실측(x269·n=8): checking 8/8 · savings 8/8 (후보만) / 후보 없이 대화만 = 두 축 0/8.
    #   ⚠축 선택(`_g`)은 종전대로 손님 발화에서 LLM 이 한다 — 바뀌는 것은 결정 ask 뿐이다.
    # ★T2_SUB_REQUIREMENT (2026-08-17·기본 OFF·x343 실측) — 서브에게 **손님 요구**를 준다.
    #
    #   x343(n=24=8×3·블록 편차 0): 이 서브가 문서+후보줄만 받으면 `Gold Account` **24/24 오답**,
    #   손님 요구 메시지를 축자로 받으면 `Silver Plus` **24/24 정답**, 무관한 요구를 주면 **0/24**
    #   (부정통제 통과). ⇒ 라이브 0/8 의 원인은 재료가 아니라 **요구가 서브에 없다**는 것이다.
    #   요구가 사라진 경위: 아래 `_dask` 치환이 x269(*"대화 잔여물이 해롭다"*)에서 왔는데
    #   **잔여물과 요구를 함께** 버렸다.
    #
    #   ⛔[[59]] 준수: 엔진은 손님 발화에서 아무것도 **뽑지 않는다**. LLM 이 인용을 내고
    #     (A2 `requirement_prompt`·도메인 어휘 0), 엔진은 그 인용이 손님 발화에 **실재하는지만**
    #     확인한다(`in` 연산·C45 동형·정규식 0). 검증을 통과한 인용만 싣는다.
    #   ⚠[[62]]: 이 자리는 격리로 **된다**(x343) ⇒ 새 결정론 0 — 엔진은 운반과 존재확인뿐이고
    #     고르는 일은 끝까지 서브(LLM)다.
    _reqs = []
    if (os.environ.get("T2_SUB_REQUIREMENT") == "1"
            or os.environ.get("T2_VERDICT_CARRY") == "1") and _po.get("requirement_prompt"):
        try:
            _utxt = "\n\n".join(_users)
            _rraw = _ts.sub_requirements(agent, _la, _UM, _po, _utxt)
            for _q in (_rraw or []):
                _qs = str(_q).strip()
                if _qs and _ts.quote_in(_qs, _utxt):  # ★존재확인만 (추출 0·강조 무시·C510)
                    _reqs.append(_qs)
            # ★관측(2026-08-18·C532⒢): 개수만 찍으면 **기각이 옳은 거부인지 과한 검산인지**
            #   가릴 수 없다. t7310 에서 098 이 1/1 기각·024 가 3/3 기각이었는데 로그에 인용문이
            #   없어 원인을 못 봤다 — S3(15시간)를 그 사각지대로 태울 수 없다.
            #   ⚠거동 불변(인쇄뿐)·기각분만·각 80자·최대 3개(로그 부피 통제).
            _rej = [str(_q).strip() for _q in (_rraw or [])
                    if str(_q).strip() and str(_q).strip() not in _reqs]
            print("[T2_SUB_REQUIREMENT] 인용 %d개 중 원문 검증 통과 %d개%s"
                  % (len(_rraw or []), len(_reqs),
                     ("" if not _rej else
                      " · 기각 %d: %s" % (len(_rej),
                                         " | ".join(q[:80] for q in _rej[:3])))),
                  file=sys.stderr, flush=True)
        except Exception as _re2:
            print("[T2_SUB_REQUIREMENT] 건너뜀(무발화): %r" % (_re2,),
                  file=sys.stderr, flush=True)
    _ctpl = _po.get("decide_candidates_text")
    _dask = _ask
    if _ctpl:
        try:
            _cands = ", ".join(_slug_disp(k)
                               for k in sorted((_po.get("doc_index") or {}).get(_g) or ()))
            if _cands:
                _dask = _ctpl.format(candidates=_cands)
        except Exception as _ce:
            print("[T2_SEARCH_AGENT] 후보 줄 실패(종전 ask 로): %r" % (_ce,),
                  file=sys.stderr, flush=True)
    # ★검증 통과한 요구 인용을 **후보줄 앞에** 붙인다(x343 구성: 요구가 머리·후보가 꼬리).
    #   인용이 하나도 검증되지 않으면 아무것도 안 붙이고 종전 거동으로 남는다(fail-safe).
    if _reqs:
        _dask = "Customer's stated request:\n" + "\n".join("- " + q for q in _reqs) \
                + ("\n\n" + _dask if _dask else "")
    # ★`decide=False` = **문서 자체를 돌려준다**(2026-08-16 발견·핸드오프 §0).
    #   기본 경로는 서브에이전트의 **결정**(243~263자)을 나른다. 그런데 격리에서 24/24 를 만든
    #   객체는 **문서 본문 51k자**였다 — 둘은 다른 것이고, 나는 하루 종일 같은 것으로 말했다.
    #   ⇒ *커밋-이전 전달* 실험은 **격리와 같은 객체**(문서)를 날라야 비교가 성립한다.
    #   ⚠[[62]] ③: 결정을 나르면 *"엔진이 답을 준" 것*에 가까워져 측정 대상이 흐려진다.
    #     문서만 나르면 고르는 일은 끝까지 모델 몫으로 남는다.
    if not decide:
        try:
            _done.add(_g)
            # ★영속 필수(2026-08-16·t7304 사전 점검에서 발견): `_done` 이 이 호출에서 갓
            #   만들어진 지역 set 이면 add 만으로는 **소비가 유실**된다 — tag h 실측: DOCONLY
            #   가 checking 을 배달한 뒤 ONPROCEED 가 **같은 축을 재처리**했다. 유실되면
            #   2축 태스크(055)에서 같은 문서를 예산 3 이 다할 때까지 재배달하고 둘째 축은
            #   영영 안 온다. decide=True 경로와 동일한 한 줄이다.
            agent._t2_search_done = _done
        except Exception:
            pass
        # 배달 이력 부기(T2_SEARCH_REARM 술어의 재료·출력 0): 실린 문서 id 헤더 → 계열 집합.
        _record_served(agent, _po, _g, messages, delivered=_mat)
        print("[T2_SEARCH_AGENT] 문서-only 반환 group=%s · %d자" % (_g, len(_mat)),
              file=sys.stderr, flush=True)
        return _mat
    # ★L-V **판정 이월** (`T2_VERDICT_CARRY`·기본 OFF·x356b/x357 v2 확증: 표적 25축 8→15·
    #   D_NEG 2·McNemar p=.092). 문서 전문 대신 **후보별 판정 줄**을 싣는다 — 고르는 일은 여전히
    #   모델이고 엔진은 운반·검산뿐이다([[65]] "결정점엔 답만"·[[66]] 판단 0).
    #   ⚠후보를 **제거하지 않는다**(리뷰 ⑤). 줄이 0개면 종전 재료로 떨어진다(fail-safe).
    _vmat = _mat
    if os.environ.get("T2_VERDICT_CARRY") == "1" and _reqs:
        try:
            _vblock = "Customer's stated request:" + chr(10) + chr(10).join(
                "- " + q for q in _reqs)
            _vlines, _vstats = _ts.verdict_lines(agent, _la, _UM, _po, _vblock, _g,
                                                 corpus=_corpus)
            if _vlines:
                _vmat = chr(10).join(_vlines)
                try:                      # ★감사(C508⒥): 실린 줄을 **축자로** 남긴다
                    import t2_fbsidecar as _fbv
                    _fbv.record("verdict-lines", _vmat, messages, channel="verdict",
                                group=_g, stats=_vstats)
                except Exception:
                    pass
        except Exception as _ve:
            print("[T2_VERDICT] 실패(종전 재료로): %r" % (_ve,), file=sys.stderr, flush=True)
    _choice = _ts.decide_from_docs(agent, _la, _UM, _po, _vmat, _dask)
    if not _choice:
        return ""
    # 이 축은 처리했다 — 다음 결정점은 **남은 축**을 본다(sim 1회 잠금이 아니라 축별 1회).
    try:
        _done.add(_g)
        agent._t2_search_done = _done
    except Exception:
        pass
    # 배달 이력 부기(T2_SEARCH_REARM 술어의 재료·출력 0): 결정문이 덮은 계열 집합.
    _record_served(agent, _po, _g, messages, decided=_choice)
    print("[T2_SEARCH_AGENT] 축 처리 완료: %s (남은 축 %s)"
          % (_g, ",".join(g for g in _gs if g not in _done) or "없음"),
          file=sys.stderr, flush=True)
    _out = _po["decided_by_docs_text"].format(choice=_choice)
    # ★서브가 낸 답을 보관한다 — 축이 처리된 뒤(침묵) write 자리에서 **그대로 다시** 내기
    #   위해서다(C439⒝·새 판단 0·C301 `_t2_deferred` 와 같은 형태의 재제시).
    # ★★축별 보관 (2026-08-13·재판정런 071 t1 부검): 구판은 **단일 슬롯**이라 두 축을 쓰는
    #   태스크에서 나중 축(savings)의 결정이 앞선 축(checking)의 결정을 **덮어썼다**. 실측:
    #   turn 30 `DOCDECIDE → 'Sky Blue'`(checking) → turn 32 `→ 'Gold Saver Account'`(savings)
    #   → 같은 turn 32 의 checking write 인자는 `True Blue`(오답)로 나갔고 `Sky Blue` 는
    #   재제시되지 않았다(`_t2_search_done` 잠금은 전역·영구라 서브도 다시 안 돈다).
    #   축은 A3 `doc_index` 키(닫힌 집합)이므로 dict 키로 쓰는 것에 해석이 없다([[22]]).
    try:
        agent._t2_last_decision = _out
        _ad = dict(getattr(agent, "_t2_axis_decision", None) or {})
        # ★계기 (2026-08-25·거동 변경 0·사용자 지시 *"수리할 방법이 없으면 다음 런을 위해
        #   원인파악을 위한 장치라도 달아두라"*). ②범주(057·063)의 남은 갈래는 큐 P1 이
        #   **전달**로 지목했고, 그 근거는 055 반증이 `DOCDECIDE` 결정문 둘을
        #   `outcome="clobbered"` 로 잡은 것이다. 그런데 라이브에는 덮어쓰기를 **세는 자리가
        #   없어서** 057 에서도 같은 일이 벌어지는지 판정 불가였다. 여기가 그 유일한 자리다.
        #   ⚠판단 0 — 같은 축 키에 다른 값이 들어오는지 문자열 비교 하나뿐이고 거동은 종전 그대로.
        if _g in _ad and _ad[_g] != _out:
            print("[T2_AXIS_CLOBBER] axis=%s old=%r new=%r"
                  % (_g, str(_ad[_g])[:120], str(_out)[:120]),
                  file=sys.stderr, flush=True)
        _ad[_g] = _out
        agent._t2_axis_decision = _ad
    except Exception:
        pass
    return _out


def _unlocked_names(messages, a2=None):
    """Discoverable names this conversation has asked to unlock (순수함수·리터럴 0).

    Whether the unlock succeeded is deliberately not checked: `task_048` unlocked
    successfully eight times and unlocked the *wrong* tool each time, so the question the
    checklist needs answered is "has this name ever been unlocked here", not "did some
    unlock work". The dispatcher and its argument name come from A2, never from spelling.
    """
    spec = ((a2 or {}).get("dispatcher_role_check") or {})
    tool = spec.get("unlock_tool")
    arg = (spec.get("name_args") or {}).get(tool) if tool else None
    if not (tool and arg):
        return set()
    out = set()
    for m in (messages or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            if getattr(tc, "name", None) == tool:
                v = _args_dict(tc).get(arg)
                if v:
                    out.add(str(v))
    return out


def _quiet_turns(messages, tools):
    """Assistant turns since one of `tools` was last called — how long the walk has been idle.

    Counted from the history rather than a counter on the agent, so a regeneration loop
    that runs three times inside one turn cannot inflate it. A knowledge-base search or a
    shell command is not a step of the procedure and does not reset this: `task_048` spent
    thirteen turns searching and never touched the declaration again.
    """
    n = 0
    for m in reversed(messages or []):
        if getattr(m, "role", None) != "assistant":
            continue
        if {_exact_tool_name(tc) for tc in (getattr(m, "tool_calls", None) or [])} & set(tools):
            return n
        n += 1
    return n


def _claim_verify_false(agent, spec, claims, evs):
    """구제된 완료-주장 중 **격리 서브가 거짓이라고 답한 것** (2026-08-18·사용자 지시).

    사용자 축자: *"LLM 격리로 env 정책과 실행한 도구, 현재 실행했다고 주장하는 도구를 참 거짓으로
    판단하게 별도의 검증 에이전트 돌리면 되는거 아닌가? 이건 LLM 이 잘 할 수 있다."*

    ## 왜 (t7318 task_073)
      `record_update`(환급) 주장이 **조회 도구**(`get_atm_fee_discrepancies`)를 지목했는데 그 이름이
      원장에 있다는 이유로 구제됐고, 환급은 끝내 실행되지 않은 채 *"처리했다"* 는 보고가 나갔다.
      이름 대조만으로는 *"그 도구가 그 일을 할 수 있는가"* 를 물을 수 없다 — 그것은 **해석**이고
      해석은 LLM 몫이다([[52]]).

    ## 계약
      서브에 들어가는 것은 **원장 이름 목록과 주장 한 줄뿐**이다([[65]] 대화 잔여물 0). 서브는
      `{"true": bool, "did": "<원장에 있는 도구 이름>"}` 을 낸다. 엔진은 ⑴`true` 를 읽고
      ⑵`did` 가 **원장에 실재**하는지만 본다(C45 동형·의미 판단 0).
      `true` 인데 `did` 가 원장 밖이면 **판정을 버린다**(모르면 막지 않는다·[[25]]).

    ⚠기본 OFF(`T2_CLAIM_VERIFY`) · 템플릿 미선언·호출 실패·파싱 실패 → 빈 목록(종전 거동).
    """
    if os.environ.get("T2_CLAIM_VERIFY") != "1":
        return []
    tpl = (spec or {}).get("verify_question")
    if not (tpl and claims and agent is not None):
        return []
    try:
        import t2_subcall as _SC
        import tau2.agent.llm_agent as _la_v
        from tau2.data_model.message import UserMessage as _UM_v
    except Exception as _ie:
        print("[T2_CLAIM_VERIFY] skip=import %r" % (_ie,), file=sys.stderr, flush=True)
        return []
    ledger = sorted({str(e) for e in (evs or ()) if str(e or "").strip()})
    out = []
    for c in (claims or []):
        body = tpl.format(ledger="\n".join("- " + n for n in ledger),
                          claim=str((c or {}).get("what") or "")[:400],
                          tool=str((c or {}).get("tool") or "")[:120])
        raw = _SC.sub_generate(agent, _la_v, _UM_v, body, "claim_verify")
        obj = _SC.parse_contract(raw, "true")
        if not isinstance(obj, dict):
            print("[T2_CLAIM_VERIFY] 판정 없음(그대로 둔다): tool=%r"
                  % ((c or {}).get("tool"),), file=sys.stderr, flush=True)
            continue
        verdict = bool(obj.get("true"))
        did = str(obj.get("did") or "").strip()
        if verdict and did and did not in set(ledger):
            print("[T2_CLAIM_VERIFY] 참이라는데 지목이 원장 밖(%r) — 판정 버린다" % (did,),
                  file=sys.stderr, flush=True)
            continue
        if not verdict:
            out.append(c)
            print("[T2_CLAIM_VERIFY] 거짓 판정: claim=%r tool=%r"
                  % (str((c or {}).get("what"))[:60], (c or {}).get("tool")),
                  file=sys.stderr, flush=True)
    return out


def _claim_unbacked(claims, emap, evs, messages, a2=None, kind_fallback_on_miss=False):
    """★claim_prov 원장대조 코어 (2026-07-20 관문5 추출·순수함수=단위테스트 공유·[[03b]]).
    LLM이 formalize한 주장 목록({kind, what, tool})을 실행 원장과 대조. 반환=미입증 목록.

    ★★tool-지목이 **원장 밖**이면 kind-색인으로 **강등**한다 (2026-08-21·t7335 050 DUP 실측·
      `kind_fallback_on_miss=True`인 호출만 = 과거형 claims 축):
      050 부검(정본 `T7335_NT1_FORENSIC_HALFB_2026_08_21.md`): approve·submit이 원장에 실재하고
      record_update 패턴에 `__effective_write__` 센티널까지 있었는데 unbacked=2가 났다 — 이 함수
      에서 그 판정에 도달하는 경로는 **지목 branch뿐**이다(지목이 원장 이름 집합 대조를 실패하면
      kind·센티널을 보지 않고 즉시 미입증). 지목은 kind-색인의 **개선**으로 넣은 것인데(위 ★★),
      지목이 빗나갔다고 kind-색인보다 **나쁜** 판정(맞는 행동을 틀렸다고)으로 떨어지는 것은 그
      도입 논리 자체와 모순이다. 그 거짓 피드백("ledger shows NO such event ... do it now")을
      모델이 문자대로 따라 **같은 승인을 재호출 = DUP 변이**를 우리가 제조했다([[25]]·[[64]]).
      ⇒ 지목 미스 = 미지목과 동급으로 강등해 구판(kind 색인) 경로로 떨어진다. 엔진은 여전히
      집합 대조만 한다(의미 판단 0·[[22]]).
    ⚠**pending(미래형)은 강등하지 않는다**(기본 False·거동 보존): 미이행 약속은 그 도구가
      원장에 없는 것이 정상이라, 강등하면 record_update 센티널이 **무관한 과거 write**로 약속을
      입증해 버려 038형 transfer-escape 방어(관문5)가 무너진다.

    ★★**주장이 도구를 지목하면 그것으로 판정한다** (2026-08-08·handoff §5-3·실측 확증):
      구판은 `kind` → A2 `event_map` 패턴 → 실행 이름 순으로 색인했다. 그런데 `kind` 는
      **모델이 붙인 라벨**이고 우리 패턴과 어긋난다 — run f 사이드카 실측: `log_verification`
      이 양쪽 sim 에서 **실제로 실행됐는데** 모델이 kind 를 `record_update` 로 선언해
      (A2 는 그 tool 을 `verify` 아래 둔다) 결정 턴마다 *"the conversation ledger shows NO
      such event: record_update: logged verification record"* 라는 **거짓**을 내보냈다.
      계좌 조회(`call_discoverable_agent_tool` 경유)도 같은 식으로 "없다"고 했다.
      우리 출력은 이 대화에서 **유일한 근거원**이라 거짓은 그 자체로 오염이다([[25]]).
    ⇒ 권위는 **실행 원장**이고 `kind` 는 해석이다([[52]]). 그래서 모델에게 *어느 호출이
      그것을 했는지* 를 함께 내게 하고(A2 `claim_audit` 질문·`done_report.tool` 과 같은 규약),
      엔진은 **이름 집합 대조**만 한다 — 의미 판단 0([[22]]).
      지목이 없으면 구판(kind 색인)으로 떨어진다 = 거동 보존.
    ⚠약함의 대가: 모델이 엉뚱한 도구를 대면 통과한다. 그래도 구판보다 낫다 — 구판은
      **맞는 행동을 했는데 틀렸다고** 말했고, 그건 모델이 고칠 수 없는 오류다.
    ⚠라이브 효과 미측정([[57]]) — 다음 런에서 `kind-index rescued` 발화 수로 잰다.
    """
    def _n(x):
        return re.sub(r"_\d+$", "", str(x or "").strip())

    named = {_n(e) for e in (evs or ()) if str(e or "").strip()}
    out = []
    for c in (claims or []):
        t = _n((c or {}).get("tool"))
        if t:
            if t in named:
                print("[T2_CLAIMPROV] kind-index rescued: kind=%r tool=%r 원장에 있다"
                      % ((c or {}).get("kind"), t), file=sys.stderr, flush=True)
                continue
            if not kind_fallback_on_miss:
                out.append(c)
                continue
            # ★050 수리(docstring ★★): 지목 미스 = 미지목과 동급 — kind 색인으로 강등.
            print("[T2_CLAIMPROV] tool-miss fallback: kind=%r tool=%r 원장 밖 — kind 색인으로 강등"
                  % ((c or {}).get("kind"), t), file=sys.stderr, flush=True)
        k = str((c or {}).get("kind", "")).strip().lower()
        spec = emap.get(k)
        if spec is None:
            continue
        if spec == "__effective_write__":
            if not _any_effective_write(messages, a2):
                out.append(c)
            continue
        pats = spec if isinstance(spec, list) else [spec]
        # ★센티널을 **목록 안에서도** 받는다 (2026-08-13·재판정런 070 t3 실측).
        #   모델이 계좌 개설을 `kind='record_update'` 로 라벨했는데 A2 의 그 kind 는
        #   `update_`·`apply_statement_credit` 계열만 가리켜 **실제로 실행한 개설**을
        #   *"the ledger shows NO such event"* 라고 단정했고, 모델은 그 말을 따라 같은 개설을
        #   **다시 호출했다**(turn 34 중복 write·`may already exist`). 우리 출력은 이 대화의
        #   유일한 근거원이라 거짓은 그 자체로 오염이다([[25]]).
        #   ⇒ 구제는 **A2 쪽**에서 한다(그 kind 가 실효 write 로도 충족되게). 엔진은 목록
        #     원소로 온 센티널을 해석할 수 있어야 하고, 그것이 이 두 줄이다. 무지목 날조
        #     탐지(kind 계열 실행이 아예 0인 경우)는 그대로 살아 있다 — 검정이 그걸 지킨다.
        if "__effective_write__" in pats and _any_effective_write(messages, a2):
            continue
        pats = [p for p in pats if p != "__effective_write__"]
        if not any(any(str(e).startswith(p) for e in evs) for p in pats):
            out.append(c)
    return out


def _split_claims_by_owner(claims, agent_names, user_names, registry=None, min_tok=2):
    """★주장이 지목한 도구의 **소유자**로 가른다 (순수함수·단위검정 공유·[[03b]]).

    왜 (2026-08-09·C348⒢ 실측): 결정 턴의 미이행-약속 문구가 `give: guide customer to use
    <tool>` 를 지목했는데, 그 `<tool>` 은 **에이전트 자신의 도구**였다. 두 방향으로 어긋난다 —
      · 도구가 **손님 소유**면 *안내하는 것이 곧 이행*이다. 그런데 우리는 *"실행되지 않았다"*
        고 말한다 ⇒ **한 일을 안 했다고 말하는 것**이고, 모델이 고칠 수 없는 오류다(C341 동형).
      · 도구가 **에이전트 소유**면 진짜 결함은 약속 위반이 아니라 **자기 도구를 손님에게
        떠넘긴 것**이다. 옳은 지적은 *"약속을 안 지켰다"* 가 아니라 **소유권 사실**이다.

    소유권은 레지스트리에서 기계적으로 나온다 — 의미 판단 0·도메인 리터럴 0([[22]] 닫힌 술어:
    도구 소유는 발화 변이에 불변이다). 양쪽에 다 있으면 **에이전트 우선**(부를 수 있으면 자기 것).
    모르면 `unknown` 으로 두고 **구판 거동을 보존**한다.

    반환: (own, theirs, unknown) — `theirs` 는 호출부가 **침묵**시킬 몫이다.
    """
    def _n(x):
        return re.sub(r"_\d+$", "", str(x or "").strip())

    a = {_n(x) for x in (agent_names or ()) if str(x or "").strip()}
    u = {_n(x) for x in (user_names or ()) if str(x or "").strip()}
    own, theirs, unknown = [], [], []
    for c in (claims or []):
        t = _n((c or {}).get("tool"))
        if t and t in a:
            own.append(c)
        elif t and t in u:
            theirs.append(c)
        else:
            # ★FIX-8 (2026-08-13·격리 `x300_early_note_probe.py` 3셀 n=8: **B_NOTE 8/8** ·
            #   A_NONE 0/8 · **D_GEN 0/8**). 라이브(t7278 075 turn30)의 미이행 약속은 도구가
            #   안 붙어 unknown 으로 떨어졌고, 그래서 나간 문구가 도구-이름 없는 일반 촉구
            #   (=D_GEN 동형)였다 — 그 문면은 격리에서 **0/8**이고, 소유권 사실(도구명)을 담은
            #   문면은 **8/8**이다. 즉 인자는 촉구가 아니라 **사실**이다([[64]]).
            #   여기서 하는 일: 도구 미지 주장의 what 토큰이 **에이전트 레지스트리** 항목과
            #   `min_tok` 개 이상 겹치고 최댓값이 **유일**하면 그 사실을 회수한다(기계 문자열
            #   연산·의미 판단 0·[[59]]). 동률·부족이면 unknown 유지 = 구판 문구(fail-open).
            #   엔진은 무엇을 부를지 고르지 않는다 — 소유자 사실만 말한다([[62]] ③④).
            _m = (_tok_overlap((c or {}).get("what"), registry or (), stem=True)
                  if registry else [])
            if len(_m) == 1 and _tok_hits((c or {}).get("what"), _m[0]) >= min_tok:
                c = dict(c or {})
                c["tool"] = _m[0]
                own.append(c)
            else:
                unknown.append(c)
    return own, theirs, unknown


def _known_tool_names(self_tools, env, msgs):
    """★C207/C2-a 대조 집합 (순수함수·리뷰 필수3): 에이전트가 **실제로 쓸 수 있는** 도구 이름 전체.
    `self.tools`만 보면 discoverable 도구(잠금 상태·목록 밖·`_NNNN` 접미사)를 **미보유로 오탐**한다
    — 022/019처럼 유저-측 dispute 도구를 정당히 안내하는 경로가 정확히 그 길을 밟는다.
    집합 = 도구목록 ∪ env user-side discoverable ∪ 이 대화서 unlock/give된 이름 ∪ 실제 호출된 이름.
    전부 접미사 strip 정규화(도메인 리터럴 0·구조 사실만)."""
    def _n(x):
        return re.sub(r"_\d+$", "", str(x or "").strip())
    out = {_n(getattr(t, "name", None)) for t in (self_tools or []) if getattr(t, "name", None)}
    out |= {_n(x) for x in _user_discoverable(env)}
    for m in (msgs or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            out.add(_n(getattr(tc, "name", None)))
            out.add(_n(_eff_tool_name(tc)))
            ar = _args_dict(tc)
            for k in ("agent_tool_name", "user_tool_name", "discoverable_tool_name"):
                if isinstance(ar, dict) and ar.get(k):
                    out.add(_n(ar[k]))
    return {x for x in out if x}


def _unavailable_promises(pending, known, discoverable=None, ledger_text=None):
    """약속(pending)에 실린 도구명이 `known`에 없으면 = **모델이 없는 기능을 약속**. 집합 대조만.
    `tool` 미선언 항목은 판정하지 않는다(구판 A2 하위호환·거동 보존).

    ★A3 (t7336 §6.1·OL-19·2026-08-22): ⑴구절 분할(`"A with B"`·`"A(B)"`) ⑵**원장-실재 전제** —
      `ledger_text`(궤적 축자·`_ledger_text()`)를 주면 그 문자열이 **궤적에 실재하는 이름만**
      판정한다. 상세·판 것은 본문 주석 참조.

    ★분할 반환 (2026-08-12·j런 070t0 t72): 그 도구가 agent-측 discoverable 레지스트리에
      **실재(잠금 상태)** 하면 "존재하지 않는다"는 거짓이다 — 우리 STEP2 가 5회 말한
      transfer_7291 을 이 문구가 "없다"고 단정해 개설-성공 공지 삭제·계획 반전의 방아쇠가
      됐다([[25]]). (없음, 잠김) 두 목록을 돌려주고 문면은 호출부가 A2 키로 가른다.
    """
    def _n(x):
        return re.sub(r"_\d+$", "", str(x or "").strip())
    # ★센티널·다중값 처리 (2026-08-13·재판정런 071 t0·010 t3 실측 — 둘 다 [[25]] 위반 거짓 발화).
    #   ⒜ A2 질문이 *"...or **omit** if none"* 이라 모델이 문자열 `"omit"` 을 답하는데, 구판은
    #      그것을 **실재 도구명**으로 받아 "그 도구는 존재하지 않는다"를 쐈다. 071 t0 은 그
    #      문장 직후 turn 30 부터 **26턴을 존재하지 않는 포털 절차**로 날조했다(로그 축자:
    #      `[T2_UNAVAIL] ... ['omit','omit','omit'] · locked: []`).
    #   ⒝ 모델이 도구를 **쉼표로 여러 개** 답하면(`'KB_search_dense, KB_search_bm25'`) 통째
    #      한 이름으로 대조돼 **보유한 도구**를 "없다"고 단정했다(010 t3 실측).
    #   ⇒ 센티널은 판정 제외(침묵·모르면 말하지 않는다), 다중값은 쪼개서 **하나라도 보유하면**
    #      약속은 이행 가능하므로 침묵한다. 판정은 전부 집합 대조뿐이다([[22]]).
    # ⚠2026-08-13 사고: 블록을 다시 쓰면서 이 `disc` 정의를 지우고 사용만 남겨 **NameError 로
    #   레버가 통째 죽었다**(밤샘 런 `[T2_UNAVAIL] skipped (no-op): NameError` ×7 — try/except 가
    #   삼켜 조용했다). 죽은-레버 5호. `test_no_undefined_names` 는 `X.attr` 꼴만 봐서 못 잡았고,
    #   그래서 그 검정을 **모든 미정의 지역명**까지 보도록 확장했다(같은 커밋).
    disc = {_n(x) for x in (discoverable or set())}
    _SENT = {"", "omit", "none", "null", "n/a", "na", "-", "unknown"}
    _led = str(ledger_text).lower() if ledger_text is not None else None
    out, locked = [], []
    for p in (pending or []):
        raw = str((p or {}).get("tool") or "")
        # ★A3-⑴ 구절 분할 (2026-08-22·t7336 OL-19·074×2 재현): 모델은 `tool` 칸에 **구절**을
        #   적는다 — `"A with B"`(도구 + 인자) · `"A(B)"`. 구판은 `[,;/]| or ` 만 갈라 통째로
        #   대조했고, 그래서 **이미 unlock 된 도구**가 "존재하지 않는다"는 통보를 받았다.
        #   추가 구분자는 구조 문자뿐이다(도메인 어휘 0·[[59]]).
        parts = [x.strip() for x in re.split(r"[,;/()\[\]]| or | with ", raw) if x.strip()]
        parts = [x for x in parts if x.lower().strip("'\"` ") not in _SENT]
        if not parts:
            continue                      # 지목 없음 = 판정 대상 아님(구판 하위호환 취지 동일)
        norm = [_n(x) for x in parts]
        if any(x in known for x in norm):
            continue                      # 하나라도 보유 = 약속은 이행 가능
        # ★A3-⑵ 원장-실재 전제 (2026-08-22·t7336 OL-19·C45 동형 substring 검산).
        #   074 가 통보받은 `apply_credits_to_account_1234` 는 **궤적에 0회 등장**한다 — 모델이
        #   꺼낸 이름이 아니라 **우리 서브가 만든 문자열**이었다. 우리가 낸 문자열을 모델의
        #   약속으로 되돌려 주고 "그런 도구는 없다"고 통보하는 것은 [[25]] 위반이다.
        #   ⇒ 궤적 어디에도 그 문자열이 없으면 **침묵**한다(모르면 말하지 않는다).
        #   `ledger_text=None`(미전달)이면 구판 거동 — 기존 검정·구 호출부 보존.
        # ⚠순서: **보유 판정(`known`) 뒤**에 온다. 앞에 두면 `"A with B"` 에서 실재 도구 A 가
        #   원장 문자열에 안 뜬다는 이유로 걸러지고 잔여 `B` 만 대조돼 **거짓 발화가 새로 생긴다**
        #   (첫 판이 그랬다). 이 전제는 오직 **억제**로만 작동해야 한다 — 구판이 말하던 것의
        #   부분집합만 말한다(단조).
        # ⚠[[70]] 무엇을 파는가: C207/C2-a 의 **원 표적**(궤적에 한 번도 안 뜬 순수 발명 기능,
        #   예: 없는 OTP 발송 약속)은 이 전제에 걸려 침묵한다. 판 것을 세는 계기 = 다음 런의
        #   `[T2_UNAVAIL]` 발화 수 · `ledger-absent` 침묵 수(호출부가 인쇄한다).
        if _led is not None and not any(x.lower().strip("'\"` ") in _led for x in parts):
            continue                      # 궤적에 없는 이름 = 우리(서브) 산출 → 침묵
        (locked if any(x in disc for x in norm) else out).append(p)
    return out, locked


def _ledger_text(messages):
    """궤적 축자 텍스트 (A3/OL-19 substring 검산용·순수 문자열·판단 0).

    포함 = 모든 메시지의 content + 모든 tool_call 의 이름·인자값. 즉 **대화에 실제로 뜬 문자열**
    전부다. 우리 서브가 만들었을 뿐 궤적에 없는 이름은 여기에 없다 — 그것이 판정의 요점이다.
    """
    buf = []
    for m in (messages or []):
        c = getattr(m, "content", None)
        if c:
            buf.append(str(c))
        for tc in (getattr(m, "tool_calls", None) or []):
            buf.append(str(getattr(tc, "name", "") or ""))
            try:
                buf.append(json.dumps(_args_dict(tc), ensure_ascii=False))
            except Exception:
                buf.append(str(_args_dict(tc)))
    return " ".join(buf).lower()


def _fu_window(cap_used, cap, reserve_declared, reserve_used, genuine_resign):
    """★C207/B1 chain 예산 판정 (순수함수·`_cpv_window`와 **동형 반환형**: None|'normal'|'reserve').
    035 day4b 실측: chain 3회 정확 발화(전부 빈손)→cap 소진→**정작 종국 notice 턴에 레버 부재**.
    ⇒ A2 `reserve: true` 선언 체인에 한해 sim당 1회 예비. 단 예비는 **진성 사임-턴**(텍스트-턴)에서만
    소비한다 — readloop 변환 턴(도구는 부르되 requires와 무관)서 태우면 종국에 또 비게 된다(리뷰 필수2)."""
    if cap_used < cap:
        return "normal"
    if reserve_declared and not reserve_used and genuine_resign:
        return "reserve"
    return None


def _claim_has_kind(claims, kinds):
    """★C201/D3 보조(순수함수·단위테스트 공유): 주장 목록에 A2 선언 `reserve_kinds` 중 하나가 있나.
    엔진은 kind 문자열 대조만 — 어떤 kind가 '중요'한지는 A2가 정한다(도메인 리터럴 0)."""
    ks = {str(k).strip().lower() for k in (kinds or [])}
    return any(str((c or {}).get("kind", "")).strip().lower() in ks for c in (claims or []))


def _limit_reduce_text(agent, a2, messages):
    """★상한·문턱 대조 문장을 만든다. 반환 = 붙일 문장(만들 수 없으면 "").

    ★왜 함수로 뺐나 (2026-08-08 부검·C324): 이 산수는 원래 `if _reqs or _bad:`
      (= 아직 밀어낼 요건이 남아 있는가) **안에** 살고 있었다. 그런데 발화 자리를 정하는
      `[T2_RESOLVE] user-action instruct` 는 그 조건 **밖**에서 찍힌다 — 즉 표적은 살아 있는데
      산수만 못 나가는 턴이 구조적으로 존재한다. 한 sim이 정확히 그 턴들이었다: 원장이 채워진
      뒤의 표적 턴이 전부 다른 분기(ORDER·dispatch_role·signature deny)로 갔고, 그래서
      *"이 그룹은 올해 자리가 없다"* 가 **한 번도 나가지 못했다**. 손님은 바로 그 그룹을
      골라 실행했다. 오프라인 재현이 그 문장이 실제로 만들어짐을 확증했다 — 원장 28행과
      A3 상한의 대조에서 소진 그룹 3개가 나왔고 손님이 고른 것이 그중 하나였다.
      ⇒ **발화 여부는 분기가 아니라 피연산자 가용성에 달려야 한다.**

    피연산자(누계·경과일)는 엔진이 이미 전사해 둔 것이고(`_t2_ledger_ops`), 상한·문턱은
    **A3 온톨로지 조회**다(사용자 지시 2026-08-08: *"정책 상수는 온톨로지에서, 사실은 db에서"*).
    값은 **fact DAG를 통해서만** 받는다 — 여기서 따로 조회 함수를 두면 같은 술어가 두 벌이 된다.
    `ask` 미전달이라 LLM 노드는 침묵하고, 런타임에 문서를 뒤지는 호출은 이 경로에 없다.
    """
    ops = getattr(agent, "_t2_ledger_ops", None) or {}
    if not ops:
        return ""
    import t2_ledger as _LG2
    import t2_factdag as _FD2
    _tx = [_content_str(_m) for _m in (messages or [])
           if getattr(_m, "role", None) in ("tool", "user")]
    _a3v = {}
    try:
        _a3v, _ = _FD2.evaluate(_FD2.load(a2),
                                _FD2.Inputs(corpus=_tx,
                                            a3=((a2 or {}).get("policy_ontology")
                                                or {}).get("rows") or ()))
    except Exception as _fe2:
        print("[T2_LIMIT_REDUCE] A3 조회 실패: %r" % (_fe2,), file=sys.stderr, flush=True)
    _lims3 = _a3v.get("doc_limits") or {}
    _mins3 = _a3v.get("doc_minimums") or {}
    # ★축 이름 → 조회 결과. 선언(`derived`)이 이미 축을 말하고 있으므로 여기서 이름을 짓지
    #   않는다 — 엔진에 도메인 어휘 0. 소비자(`eligible_text`)는 A2가 지목한 축만 꺼내 쓴다.
    _axm3 = {}
    for _n3 in ((a2 or {}).get("derived") or ()):
        if _n3.get("op") != "a3_map":
            continue
        _ax3 = (_n3.get("params") or {}).get("axis")
        _v3 = _a3v.get(_n3.get("out"))
        if _ax3 and _v3:
            _axm3[_ax3] = _v3
    _add = ""
    # ★[[65]] 메인은 **답만** 싣는다 (2026-08-11·C420·`T2_MAIN_ANSWERS_ONLY`·기본 OFF·사용자 지시
    #   *"메인 컨텍스트는 서브에이전트 호출과 결과만"*). 지금 메인으로 나가는 8조각 중 **답은
    #   하나**(`diagnosed_text`)뿐이고 나머지 일곱은 과정·재료다(표·상태 분해·창 산수·소진/미달/
    #   미판정 이름 목록). 그리고 그 중 하나는 **해롭다고 이미 측정됐다** — `ineligible_text` 가
    #   만드는 이름 목록은 x231 에서 *실제 문맥 위에 한 줄만 얹어도 task_100 8/8 → 0/8* 였다.
    #   같은 방향의 실측이 넷 더 있다: x187(대화 없는 격리가 전 셀 파레토 지배) · x190(표를 실으면
    #   어느 정렬로도 두 태스크를 못 잡고 근거 숫자를 5/5 틀리게 댄다) · C397(궤적 4% ↔ 격리 100%)
    #   · x248 `W_ALL` 4/8(다른 축 재료까지 얹으면 답이 섞인다).
    #   ⇒ 재료는 **서브의 격리 문맥**으로 보내고 메인에는 결정문만 남긴다. 새 결정론 0 —
    #     우리가 **덜 올릴 뿐**이고, 고르는 일은 그대로 모델이다.
    #   ⚠끄면 종전 그대로(되돌리기 경로 유지). ⚠서브가 안 도는 자리에서 재료를 통째로 잃지
    #     않도록, 옮긴 조각은 진단 서브 문맥(`onto_context`)에 **이어 붙인다**.
    _answers_only = os.environ.get("T2_MAIN_ANSWERS_ONLY") == "1"
    _subonly = []

    def _emit(text, is_answer=False):
        """답이면 메인으로, 과정이면 (플래그가 켜졌을 때) 서브 문맥으로."""
        if not text:
            return ""
        if is_answer or not _answers_only:
            return text
        _subonly.append(text)
        return ""
    # ★R8 (2026-08-09·C373 부검): **결정 블록이 나가는 메시지에는 다른 행동 지시를 섞지 않는다.**
    #   스모크 실측 — 통과한 100 은 블록이 혼자 나갔고, 실패한 099 는 같은 메시지에
    #   `[SOURCE]`(*"Search the knowledge base…"*) 2회 + `unmatched`(*"determine which before
    #   advising"*) 3회가 함께 나갔다(100 은 각 0회). 블록 위치는 둘 다 끝에서 ~450자로 같았고,
    #   에이전트는 KB 를 다시 뒤진 뒤 블록의 답을 버렸다. ⇒ 레버는 최근성이 아니라 **지시 충돌**.
    #   억제가 정당한 이유: 블록은 이미 **인용 있는 정책 상수**를 근거와 함께 싣는다 —
    #   *"문서를 찾아라"* 요구는 그 턴에 이미 충족돼 있다. 블록이 없는 턴에는 종전대로 나간다.
    _unm_parts, _decided = [], False
    # ★R8b — 결정 블록이 나가는 메시지에는 **다른 상품 이름의 목록**도 싣지 않는다
    #   (2026-08-10·`T2_DECISION_ISOLATE`·설계서 `DECISION_ACTION_SPLIT_DESIGN_2026_08_10`).
    #   근거(x231 leave-one-in·n=8): 깨끗한 바닥에서는 **어떤 문장을 얹어도 8/8** 인데, 실제
    #   문맥 위에서 `ineligible_text` 가 만든 두 문장(*"not reachable yet - Beige …"* ·
    #   *"Reachable on this criterion - Blue 30; Bluest 60; …"* = 이름 15개)을 **각각 하나만**
    #   얹으면 task_100 이 **0/8** 로 무너진다(정답 `Hunter Green` → `Hunter Green Business
    #   Checking`). x230 표식 실험도 같은 자리를 가리킨다 — 앞문구가 있으면 대화에 없는 표식
    #   조차 4/8 로 밀린다. ⇒ 그 목록은 **결정 서브가 이미 소비한 재료**이고, 블록이 나간 뒤
    #   메인에 다시 실으면 경쟁 표기만 늘린다.
    #   ⚠빼는 것은 **우리가 방금 만든 그 문자열**이다 — 도메인 텍스트 파싱이 아니다([[59]]).
    #   ⚠블록이 없는 턴에는 종전대로 나간다(그 턴엔 이 목록이 유일한 근거일 수 있다).
    _name_lists = []
    # 선언마다 **그 선언이 말하는 축만** 계산한다 — 상한은 상한을 선언한 쪽, 문턱은 문턱 쪽.
    for _e2 in ops.values():
        _sp2 = _e2.get("spec") or {}
        if _e2.get("tally") and _lims3:
            # ★C376 주어 정합 — 원장은 `Navy Blue Account`, A3 주어는 `Navy Blue` 다. 두 소비자가
            #   정확 일치로 맞대므로 그 접미사 하나가 둘 다 무력화했다(전수 실측: `unmatched` 발화
            #   77회 중 A3 주어와 일치 **0**·`exhausted` 발화 **0회**). 정합은 **LLM 이 하고**
            #   엔진은 A3 주어 집합의 원소인지만 본다([[22]]·[[59]] — 엔진이 접미사를 떼면 그것이
            #   도메인 패턴매칭이다). 못 고른 그룹은 정렬되지 않은 채 남아 종전대로 이름이 불린다.
            _al8 = {}
            try:
                import tau2.agent.llm_agent as _la8
                from tau2.data_model.message import UserMessage as _UM8
                _al8 = _LG2.formalize_subject_align(agent, _la8, _UM8, _sp2,
                                                    list(_e2["tally"]), list(_lims3))
            except Exception as _se8:
                print("[T2_SUBJ_ALIGN] 건너뜀: %r" % (_se8,), file=sys.stderr, flush=True)
            _tal8, _left8 = _LG2.align_tally(_e2["tally"], _al8)
            _add += _emit(_LG2.exhausted_text(_tal8, _lims3, _sp2))
            # ★판정하지 못한 그룹은 **이름을 말한다**(C327). 조용히 빼면 모델 쪽에서 침묵이
            #   *검사 통과*와 구별되지 않는다. 엔진은 집합 뺄셈만 하고, 이름이 같은 것을
            #   가리키는지는 여전히 모델 몫이다([[22]]). 이제 그 몫은 위에서 **실제로 물어본다**.
            _u8 = _LG2.unmatched_text(_left8, _lims3, _sp2)
            if _u8:
                _unm_parts.append(_u8)
            _add += _emit(_u8)
        # ★C378 상태별 세기 — 누계는 그룹 축으로 뭉개서 *어느 행이 완료되지 않았는지* 를 잃는다.
        #   010 이 그 자리다(손님: *"넷을 소개했는데 둘만 보너스를 받았다"*). 엔진은 **세기만**
        #   하고 상태 값이 무엇을 뜻하는지·왜 그 상태인지는 모델과 문서 몫이다([[22]]·[[25]]).
        #   ⚠상한 조회와 무관하므로 `_lims3` 밖에 둔다 — A3 가 비어도 이 사실은 말할 수 있다.
        if _e2.get("rows"):
            _add += _emit(_LG2.status_breakdown(_e2["rows"], _sp2))
            # ★C395 (2026-08-10·사용자 지시) — 상태값의 **뜻**을 A3 에서 싣는다. 검색 0.
            #   x211: 답을 든 문서를 에이전트 질의 24개 중 **12개만** 냈다 — 같은 정보가 대화마다
            #   있다가 없다가 한다. 뜻은 대화마다 달라지는 값이 아니라 **고정된 정책 상수**이므로
            #   불확정 채널(BM25·임베딩·grep)로 가져올 이유가 없다. 엔진은 그 값이 무엇을
            #   뜻하는지 모른 채 **A3 주어 집합의 원소인 것만** 꺼낸다([[22]]).
            _a3r = ((a2 or {}).get("policy_ontology") or {}).get("rows") or ()
            _add += _emit(_LG2.status_meanings_text(_e2["rows"], _sp2, _a3r))
            # ★C397 (2026-08-10·사용자 지시) — **결정점을 온톨로지로 지은 격리 문맥에서** 짓는다.
            #   격리 측정(`x213` `G_ONTO`·24셀): 실제 궤적 4% · 궤적 청소 29% · 이 문맥 **100%**
            #   (부정 통제 0/24). 099/100 의 재도출과 같은 2단 형태이고, 고르는 것은 서브다.
            #   문맥에는 대화가 한 글자도 안 들어간다 — 그것이 혼잡의 출처다.
            if _sp2.get("diagnose_prompt"):
                _blk = _LG2.onto_context(_e2["rows"], _sp2, _a3r)
                # ★메인에서 뺀 재료를 **서브의 격리 문맥**에 이어 붙인다([[65]]).
                #   빼기만 하고 안 옮기면 그 사실이 어디에도 없게 된다 — 그건 억제가 아니라
                #   손실이다(C403 이 본 자해와 같은 형태). 서브 문맥은 대화가 한 글자도 없는
                #   자리라 여기 얹는 것은 x231 이 잰 *메인 위에 얹기*와 다른 조작이다.
                if _blk and _subonly:
                    _blk = "\n".join([_blk] + _subonly)
                if _blk:
                    try:
                        import tau2.agent.llm_agent as _la9
                        from tau2.data_model.message import UserMessage as _UM9
                        _dg = _LG2.diagnose_choice(agent, _la9, _UM9, _sp2, _blk, _e2["rows"])
                    except Exception as _de9:
                        _dg = None
                        print("[T2_DIAG] 건너뜀: %r" % (_de9,), file=sys.stderr, flush=True)
                    if _dg and _sp2.get("diagnosed_text"):
                        _add += _emit(_sp2["diagnosed_text"].format(answer=_dg[1]), is_answer=True)
                    # ★T2_CARD_DOCS (2026-08-27·사용자 지시·기본 OFF) — 진단이 **주어를 정하면**
                    #   A3 `doc_index` 가 그 주어에 대해 선언한 문서만 격리 서브에게 주고 그
                    #   **답만** 메인에 올린다([[71]]·[[65]]). 근거는 `requirement_choice` 주석.
                    #   엔진은 색인을 **읽기만** 한다 — 검색도, 유사도도, 선별도 없다([[59]]).
                    if _dg and os.environ.get("T2_CARD_DOCS") == "1":
                        try:
                            import t2_search as _ts9
                            _po9 = (a2 or {}).get("policy_ontology") or {}
                            _idx9 = _po9.get("doc_index") or {}
                            # 이름 → (군, 주어): **닫힌 집합 소속**만 본다(표시명 규약 하나).
                            _pick9 = None
                            for _g9, _subs9 in _idx9.items():
                                for _s9 in _subject_keys(_subs9):
                                    if _slug_disp(_s9).strip().lower() == str(_dg[0]).strip().lower():
                                        _pick9 = (_g9, _s9)
                                        break
                                if _pick9:
                                    break
                            if not _pick9:
                                print("[T2_CARD_DOCS] 색인 밖 이름 = 침묵: %r" % (_dg[0],),
                                      file=sys.stderr, flush=True)
                            else:
                                _ids9 = list((_idx9.get(_pick9[0]) or {}).get(_pick9[1]) or ())
                                # ⚠이 자리엔 `self` 가 없다 — 감싸는 함수는
                                #   `_limit_reduce_text(agent, a2, messages)` 다. 1차 배선이
                                #   `self` 를 써서 `NameError` 로 18회 조용히 건너뛰었고
                                #   스모크 게이트가 그것을 잡았다(런 산출물 0).
                                _cps9 = _ts9.corpus_from_env(
                                    getattr(getattr(agent, "_t2_orch", None), "environment", None))
                                _docs9, _miss9 = _ts9.read_docs(_ids9, corpus=_cps9)
                                if _miss9:
                                    print("[T2_CARD_DOCS] 코퍼스에 없는 문서 %d: %r"
                                          % (len(_miss9), _miss9[:3]), file=sys.stderr, flush=True)
                                _body9 = chr(10).join("ID: " + _k9 + chr(10) + _docs9[_k9]
                                                   for _k9 in sorted(_docs9))
                                _rq9 = _LG2.requirement_choice(agent, _la9, _UM9, _sp2, _body9,
                                                               _dg[0], sorted(_docs9))
                                if _rq9 and _sp2.get("requirement_text"):
                                    _add += _emit(_sp2["requirement_text"].format(answer=_rq9),
                                                  is_answer=True)
                        except Exception as _ce9:
                            print("[T2_CARD_DOCS] 건너뜀: %r" % (_ce9,),
                                  file=sys.stderr, flush=True)
            # ★C379 — 상태만 말하면 손님의 *"왜"* 에 답이 안 된다(v010 실측: 상태는 알았는데
            #   이유를 못 찾아 이관으로 끝났다). 이유는 이미 선언된 창 상수와 날짜의 산수로
            #   나온다. 엔진은 **산수까지만** 말하고 인과는 모델·문서 몫이다([[25]]).
            _add += _emit(_LG2.window_history(_e2["rows"], _sp2))
        if _e2.get("days") is not None and _mins3:
            _il2 = _LG2.ineligible_text(_e2["days"], _mins3, _sp2)
            if _il2:
                _name_lists.append(_il2)          # R8b: 블록이 나가면 이 목록은 뺀다
            _add += _emit(_il2)
        # ★통과 집합 (2026-08-08·C337). 못 되는 것을 말하는 것만으로는 안 닫혔다 —
        #   x150 절제: 같은 표 0/5 vs **미달 행을 뺀 표 5/5**. 그래서 거르는 일 자체를
        #   엔진이 하고 남은 것만 준다. 누계는 **A2가 지목한 선언**의 것을 쓴다(계좌 선언의
        #   tally 는 손님 보유 level 이라 연간 상한과 무관하다 — 섞으면 조용히 틀린다).
        if _sp2.get("eligible_text"):
            _cfg2 = _sp2.get("eligible") or {}
            _tf2 = _cfg2.get("tally_from")
            # ⚠원장 선언이 아직 안 돌았으면 **None** 을 넘긴다 — 빈 dict 는 *"0 회 썼다"* 는
            #   주장이 되고, 그건 우리가 확인한 사실이 아니다([[25]]).
            _tal2 = (ops.get(str(_tf2)) or {}).get("tally") if _tf2 else None
            # ★대화에서 오는 피연산자 (2026-08-09). 자격 기준 중 둘은 DB 밖에 있다 —
            #   피추천자 예치액·회사 연령. 없다고 안 거르면 통과 집합의 최고액이 오답이 된다
            #   (099 실측: `Beige` 500 은 예치 100000 을 요구하고 손님은 30000).
            #   묻는 항목은 **A3 축 설명 그대로** 싣는다 — 엔진에 도메인 어휘 0.
            _want, _stated = [], {}
            _po2 = ((a2 or {}).get("policy_ontology") or {})
            _axd = (_po2.get("axes") or {})
            # ★R0 (2026-08-09·x188 실측): 묻는 문장은 **용도별로** 다르다. `axes` 는 문서에서
            #   문턱을 읽을 때의 정의라("최소 요구액"), 그대로 발화 추출 질문으로 쓰면 모델이
            #   옳게 기권한다 — 손님은 요구액이 아니라 **낼 금액**을 말한다. 6조건 전부 `{}`
            #   였다. 그래서 `axes_stated` 가 있으면 그것을 쓰고, 없으면 종전대로 떨어진다.
            _axs = (_po2.get("axes_stated") or {})
            for _c2 in (_cfg2.get("criteria") or ()):
                _ax2 = _c2.get("axis")
                _desc2 = _axs.get(_ax2) or _axd.get(_ax2)
                if _c2.get("operand") == "stated" and _desc2:
                    _want.append((_ax2, _desc2))
            if _want:
                try:
                    import tau2.agent.llm_agent as _la2
                    from tau2.data_model.message import UserMessage as _UM2
                    _stated = _LG2.formalize_case_facts(agent, _la2, _UM2, _tx, _sp2, _want)
                except Exception as _fe3:
                    print("[T2_LIMIT_REDUCE] 대화-피연산자 형식화 건너뜀: %r" % (_fe3,),
                          file=sys.stderr, flush=True)
            # ★종류 필터 (2026-08-10·C389·x201). 통과 표에는 개인 체킹·사업자 카드·카드가
            #   함께 실린다. 손님은 친구가 **계좌를 여는** 이야기를 하는데 모델은 카드의 단일
            #   최대 수를 집었다(`A_iso` 0/8 = `Business Platinum Rewards Card`). **전달 팔을
            #   먼저 쟀고**(`E_hint` = 한 줄로 무엇을 묻는지 말해 주기) 그것도 **0/8** 이라
            #   필터가 정당해졌다(⛔0 ②). 거른 표 8/8 · LLM 이 종류를 고르는 2단 구성도 8/8.
            #   ⚠종류를 고르는 것은 **LLM** 이고 엔진은 그 답이 A3 종류 집합의 원소인지만 본다.
            #     못 고르면 아무것도 안 거른다(종전 거동). 종류를 모르는 주어도 남는다([[25]]).
            _axm4, _kf2 = _axm3, _cfg2.get("kind_field")
            if _kf2:
                _kbs2 = _LG2.subject_kinds(_po2.get("rows") or (), _kf2)
                # 후보는 **표에 실릴 주어들의 종류**로 한정한다 — 원장 전체에서 뽑으면 이 자리에
                # 없는 종류까지 보기가 되어 x201 이 잰 것과 다른 구성이 된다(계기 정합).
                _subs2 = set(s for _m4 in _axm3.values() for s in (_m4 or {}))
                _cand2 = sorted(set(_kbs2[s] for s in _subs2 if s in _kbs2))
                try:
                    import tau2.agent.llm_agent as _la4
                    from tau2.data_model.message import UserMessage as _UM4
                    _kind2 = _LG2.formalize_kind(agent, _la4, _UM4, _cfg2, _tx, _cand2)
                except Exception as _ke2:
                    _kind2 = None
                    print("[T2_KIND] 건너뜀: %r" % (_ke2,), file=sys.stderr, flush=True)
                _axm4, _drop2 = _LG2.restrict_to_kind(_axm3, _kbs2, _kind2)
                if _drop2:
                    print("[T2_KIND] %s 아닌 주어 %d 제외: %s"
                          % (_kind2, len(_drop2), ", ".join(_drop2)[:120]),
                          file=sys.stderr, flush=True)
            _elig = _LG2.eligible_text(_e2.get("days"), _tal2, _axm4, _sp2, _stated)
            _erows = _LG2.eligible_text(_e2.get("days"), _tal2, _axm4, _sp2, _stated,
                                        as_rows=True) or []
            # ★표를 메인에 싣지 않는다 (2026-08-09·사용자 결정·C367·C370). x187 전 셀 대조에서
            #   `L3`(대화 없음)가 `L0`(대화 포함)를 **2모델×20셀 전부 파레토 지배**했고, x190 에서
            #   `with_table` 은 어느 정렬로도 두 태스크를 함께 잡지 못했다(asc 0/8·0/8 · desc 8/8·0/8).
            #   표가 있어도 근거 숫자를 5/5 틀리게 댄다. ⇒ 메인이 표에서 얻는 것이 없다.
            #   `decided_text` 가 선언돼 있으면 **결정 블록만** 내보내고, 없으면 종전대로 표를 싣는다.
            if not _sp2.get("decided_text"):
                _add += _emit(_elig)
            # ★2단 재도출 (2026-08-09·C344·x154/x155). 표를 궤적에 실어도 안 움직인다(0/5·라이브
            #   099 0/12). 움직인 유일한 것은 **판단 자리를 깨끗한 문맥으로 옮기고 그 답을
            #   되돌려 넣는 것**(0/5 → 5/5·두 태스크). 고르는 것은 두 번 다 모델이고 엔진은
            #   문맥 조립 + 집합 검사만 한다([[05]] Q2).
            #   ⚠구성은 **측정된 것 그대로**여야 한다 — 손님 발화를 그대로 실은 구성은 0/5 였다
            #     (x156 `only user`). 그래서 사실은 우리가 정제하고 목적 한 구절만 형식화한다.
            if _elig and _sp2.get("rederive_prompt"):
                try:
                    import tau2.agent.llm_agent as _la3
                    from tau2.data_model.message import UserMessage as _UM3
                    # ★목적 구절은 **싣지 않는다** (2026-08-09·x158 n=10 실측). 범위를 담게
                    #   프롬프트를 고쳐도 해로웠다 — 099 는 목적을 넣으면 **0/10**(전부 카드),
                    #   빼면 **10/10**. 100 도 5/10 → 7/10. 문장을 하나 더 얹는 순간 그 방향으로
                    #   에너지가 쏠린다(초안 §6.0: 추가는 포화할 때만 듣는다).
                    #   ⇒ 형식화 자체는 남겨 두되(다른 태스크·계열엔 필요할 수 있다) 재도출
                    #     문맥에는 **빈 문자열**을 넘긴다 = 측정된 조건과 축자 동일.
                    _obj = ""
                    if True:
                        _fl = ["%s = %s" % (k, _LG2._num(v)) for k, v in sorted((_stated or {}).items())]
                        if _e2.get("days") is not None:
                            _fl.insert(0, "days since the earliest account was opened = %d"
                                       % int(_e2["days"]))
                        _rows5 = [_s5 for _s5, _b5 in _erows]
                        _ops5 = "\n".join(_fl)
                        # ★[[65]] 메인에서 뺀 재료 중 **진단 서브가 안 가져간 것**을 여기 붙인다.
                        #   빼기만 하고 안 옮기면 그 사실이 어디에도 없게 된다 = 억제가 아니라
                        #   손실이다(C403 이 본 자해와 같은 형태). 재도출 문맥은 대화가 한 글자도
                        #   없는 자리이므로, 여기 얹는 것은 x231 이 잰 *메인 위에 얹기*와 다르다.
                        #   ⚠구성이 바뀌므로 플래그가 켜졌을 때만 그렇게 한다(측정된 조건 보존).
                        _tbl5 = _elig.strip()
                        if _answers_only and _subonly:
                            _tbl5 = "\n".join([_tbl5] + _subonly)
                            del _subonly[:]      # 재대입이 아니라 **비우기** — `_emit` 가 같은
                                                 # 리스트 객체를 계속 쓴다(클로저 공유).
                        _pick = _LG2.rederive_choice(agent, _la3, _UM3, _sp2, _tbl5,
                                                     _ops5, _obj, _rows5)
                        # ★D1c — 엔진이 재계산해 **불일치만** 잡고, 답이 아니라 **값**을 되돌려
                        #   다시 묻는다 (2026-08-09·x192·규격서 §5). 부정 통제 통과: 무내용
                        #   재시도는 실패 3셀을 하나도 못 고쳤고(0/8), 값만 되돌리면 8/8 이며
                        #   이름을 말한 상한과 같다. 이름을 돌려주면 그것이 우리 지목이 되어
                        #   [[05]] Q2 를 넘는다 — 그래서 `mismatch_value` 는 이름을 반환하지 않는다.
                        #   축은 **손님 말에서 형식화**한다(정적 선언 = [[23]] 위반). 못 구하면 안 한다.
                        _axall = (((a2 or {}).get("policy_ontology") or {}).get("axes") or {})
                        # ★C377 합-목적 (2026-08-09·사용자 지시 "새 함수로") — 서브가 **하나로
                        #   못 좁힌** 자리에서만 돈다. 098 축자: *"the best **combined** referral
                        #   bonus - the total of what I get plus what she gets"* ⇒ 목적이 두 축의
                        #   합이라 축 하나로는 표현할 수단이 없고, 실측상 그 sim 들은 지목 단계에서
                        #   `raw='NONE'` 로 끊겨 결정 블록이 아예 안 만들어졌다.
                        #   ⚠**지목을 두 번 해서 더하는 것이 아니다**: argmax(A)·argmax(B) 를 알아도
                        #     argmax(A+B) 는 알 수 없다. 덧셈이 정당한 곳은 **값 층위**뿐이라
                        #     엔진은 A3 값을 합해 순위만 만들고, 고르는 것은 끝까지 모델이다
                        #     ([[05]] Q2·[[52]]). 되돌리는 것도 **값**이지 이름이 아니다.
                        #   ⚠기존 단일-축 경로는 **건드리지 않는다** — 그 구성이 099/100 을 3/3 으로
                        #     세우고 있다(런 t·`raw='referrer_bonus_usd'` 6/6). 여기는 순증이다.
                        # ★C384 (사용자 교정): 합-목적 사슬을 **걷어냈다**. 엔진이 축을 합해
                        #   argmax 를 내면 결정을 우리가 하는 것이고, 그러면 측정 대상(모델이
                        #   무엇을 못하는가)이 사라진다 — *"결정론기 짜고 LLM 은 형식적으로 쓰는"*
                        #   것이라 gold 프로그램과 구별되지 않는다. 099/100 은 격리 프로브로
                        #   **모델의 결손을 먼저 재고** 그 자리에 레버를 놨다([[18]]). 098/010 은
                        #   그 측정을 안 한 채 기구부터 지었다. ⇒ 측정 뒤에 다시 세운다.
                        _oax, _olab, _omap = None, None, {}
                        try:
                            if _pick and _sp2.get("reask_prompt") and _axall and not _omap:
                                _oax = _LG2.formalize_objective_axis(agent, _la3, _UM3, _sp2,
                                                                     _tx, _axall)
                                _olab, _omap = _oax, ((_axm3 or {}).get(_oax) or {})
                            if _oax:
                                _mm = _LG2.mismatch_value(_erows, _omap, _pick)
                                if _mm:
                                    _rq = _sp2["reask_prompt"].format(
                                        axis=_oax, chosen=_LG2._num(_mm[0]),
                                        best=_LG2._num(_mm[1]))
                                    _p2 = _LG2.rederive_choice(
                                        agent, _la3, _UM3, _sp2, _elig.strip(),
                                        _ops5 + "\n" + _rq, _obj, _rows5)
                                    print("[T2_D1C] mismatch %s=%s<%s → 재질의 %s→%s"
                                          % (_oax, _mm[0], _mm[1], _pick, _p2 or "무응답"),
                                          file=sys.stderr, flush=True)
                                    _pick = _p2 or _pick
                        except Exception as _d1e:
                            print("[T2_D1C] 건너뜀(무발화): %r" % (_d1e,),
                                  file=sys.stderr, flush=True)
                        if _pick:
                            if _sp2.get("decided_text"):
                                # 순위 라벨·지도는 **어느 경로로 얻었든 같은 자리**를 쓴다 —
                                # 단일 축이면 그 축, 합-목적이면 합산 축(`A + B`).
                                _add += _LG2.decided_text(_sp2, _pick, _erows, _ops5,
                                                          _olab, _omap)
                                _decided = True
                            else:
                                _add += _sp2.get("rederived_text", "").format(choice=_pick)
                except Exception as _re5:
                    print("[T2_REDERIVE] 건너뜀: %r" % (_re5,), file=sys.stderr, flush=True)
    # ★R8 집행 — 블록이 나가면 같은 메시지의 **조사 지시**를 뺀다(위 주석의 근거).
    #   지우는 것은 **우리가 방금 만든 그 문자열**이라 도메인 텍스트 파싱이 아니다([[59]]).
    if _decided and _unm_parts:
        for _u8 in _unm_parts:
            _add = _add.replace(_u8, "")
        print("[T2_R8] 결정 블록과 함께 나갈 조사 지시 %d건 억제(unmatched)" % len(_unm_parts),
              file=sys.stderr, flush=True)
    # ★R8b 집행 (기본 OFF — 켠 런과 안 켠 런을 같은 코드로 비교할 수 있게 둔다)
    if _decided and _name_lists and os.environ.get("T2_DECISION_ISOLATE") == "1":
        for _n8 in _name_lists:
            _add = _add.replace(_n8, "")
        print("[T2_R8B] 결정 블록과 함께 나갈 이름 목록 %d건 억제(%d자)"
              % (len(_name_lists), sum(len(x) for x in _name_lists)),
              file=sys.stderr, flush=True)
    try:                       # 호출부가 `[SOURCE]` 도 같은 규칙으로 뺄 수 있게 알린다
        agent._t2_decided = bool(_decided)
    except Exception:
        pass
    if _answers_only and _subonly:
        # 어느 서브에도 못 실린 재료 — **조용히 사라지면 안 된다**([[64]] 의 정신).
        print("[T2_MAIN_ANSWERS_ONLY] 미소비 재료 %d조각 (%d자) — 서브 문맥이 없는 자리다"
              % (len(_subonly), sum(len(x) for x in _subonly)),
              file=sys.stderr, flush=True)
    return _add.strip()


def _cpv_window(resign, transfer, cur, cap, tr_spent, rsv_spent, has_reserve):
    """★C201/D3 발화창 판정 (순수함수·2026-07-26·§7-0 실측: `unbacked>0인데 regen 무발생` A11·B5 =
    cap 소진이 실재 → **행동-kind 주장 전용 예비 1회**를 sim당 보장). 반환: None | 'resign' | 'transfer' | 'reserve'.
    - 'reserve'는 cap 소진 후에만 열리고, 실제 regen은 unbacked에 reserve_kind가 있을 때만 집행(호출부).
    - 예비는 sim당 1회·전역 T2_REGEN_BUDGET과 함께 상한(컨텍스트 팽창=게이트 자신의 비용·등대 §1)."""
    if transfer and not tr_spent:
        return "transfer"
    if resign and cur < cap:
        return "resign"
    if resign and has_reserve and not rsv_spent:
        return "reserve"
    return None


def _is_transfer_call(am, emap):
    """★관문5(038 transfer-escape): 이번 응답에 transfer-류 호출이 있나 — 패턴=A2 event_map['transfer']
    재사용(새 A2 필드 0·엔진 리터럴 0). raw명+effective명 둘 다 대조."""
    pats = (emap or {}).get("transfer")
    pats = pats if isinstance(pats, list) else ([pats] if pats else [])
    if not pats:
        return False
    for tc in (getattr(am, "tool_calls", None) or []):
        for n in (str(getattr(tc, "name", "") or ""), _eff_tool_name(tc)):
            if any(n.startswith(p) for p in pats):
                return True
    return False


_T2_LEDGER_PROBED = False   # 계측 1회 표식(프로세스 전역·self 오염 금지)


def _regen_budget_ok(self):
    """★전역 per-sim regen 예산 (2026-07-20·023 컨텍스트 초과 진단·§2ah).
    개별 게이트 cap(FORCE=T2_ACTION_DENY_CAP·RESOLVE 3·writeprov/claimprov/discreq 1)은 있으나
    **전역 예산이 없어** struggling 태스크(023)서 게이트 스택 regen 누적+에이전트 조사가 vLLM
    max_model_len(44672)을 초과→ContextWindowExceededError→sim 무효. 등대 §1.3: 게이트 자신도 비용
    (over-action)을 낸다 — 그 비용(여기선 컨텍스트)을 **측정·상한**한다. 도메인-일반·리터럴 0.
    T2_REGEN_BUDGET=정수(총 regen 상한)·미설정=무제한(기존거동 불변). 소진 후 모든 regen skip→
    에이전트가 종단행동으로 수렴하거나 max_steps로 종료(FORCE→RESOLVE 무한루프 차단)."""
    _b = os.environ.get("T2_REGEN_BUDGET")
    if not _b:
        return True
    return getattr(self, "_t2_regen_total", 0) < int(_b)


def _regen_budget_spend(self):
    self._t2_regen_total = getattr(self, "_t2_regen_total", 0) + 1


def _resolve_cap_ok(self, messages=None, a2=None):
    """계약 경로의 진입 상한 — **정체에만 과금한다**.

    이력(2026-08-07·전부 실측):
      · 이 자리는 `_t2_resolve_deny < 3`으로 **하드코딩**돼 있었다. 환경변수도 끄는 방법도 없었다.
        그래서 `T2_ACTION_DENY_CAP`을 없앴을 때 **아무 변화가 없었다** — 바깥 cap이 먼저 물린다.
        계약 발화가 모든 arm에서 정확히 3회/sim이던 이유다. 그 3회는 turn 4·6·8에 소진되는데
        첫 요건이 충족되는 것은 turn 11 전후여서, 두 번째 요건(원장 조회)은 101 20 trial 내내
        **한 번도 "지금 하라"가 되지 못했다**(실제 조회 2/20).
      · 그래서 무제한으로 열어 봤다. 결과는 **진전이 아니라 반복**이었다: 발화 6 → **100**,
        그런데 요건 집합이 100회 **전부 동일**(`GB1,GB3,reads:…`)이고 큐는 한 번도 전진하지 않았다.
        [[57]] — 반복 억제는 '횟수'가 아니라 '인자 변화'로. 무제한은 그 반대편 실패다.

    ⇒ 세는 대상을 바꾼다. **말한 횟수가 아니라 제자리걸음 횟수**를 센다:
        지난 발화 이후 실제로 실행된 도구가 **하나라도 늘었으면** = 진행 → 카운터를 되돌린다(무과금).
        하나도 안 늘었으면 = 같은 걸음을 다시 요구하는 것 → 과금, 상한에서 침묵.
    순응하는 동안은 사실상 무제한이고, 불응하면 유한하다. 그리고 "발화해야 환급된다"는
    되돌아올 수 없는 상태(구 `T2_ACTION_PROGRESS_REFUND`의 결함)가 원리적으로 생기지 않는다.
    """
    _c = os.environ.get("T2_RESOLVE_CAP")
    cap = int(_c) if (_c or "").strip().isdigit() else 3        # 기본 = 종전 하드코딩 값
    if messages is not None:
        try:
            done = _executed_tool_names(messages, a2)
            prev = getattr(self, "_t2_resolve_done", None)
            if prev is not None and (done - prev):
                # ★관측 의무(C442)가 아래 ⓑ 경로에만 달려 있었다 — 이 경로는 **조용히** 리셋해서
                #   t7308 전수(24 sim)에서 ⓑ 마커가 0 인데 deny 가 sim 당 11~23 인 이유를
                #   **소스를 읽고 프로브를 짜야만** 알 수 있었다(x372). 같은 의무를 여기에도 단다.
                #   ⚠거동 불변 — 인쇄뿐이다. 그리고 **실효 리셋일 때만** 찍는다(이 함수는 한 턴에
                #   여러 번 불리고 스냅샷은 발화 시점에만 갱신되므로, 이미 0 인 카운터를 0 으로
                #   되돌리는 것은 사건이 아니다). ⓑ 도 같은 조건으로 맞춰 두 경로를 비교 가능하게 한다.
                # ⛔**대입이 먼저다**(2026-08-18·C538 복원). 앞서 이 자리는 `print` 가 대입보다
                #   위에 있었고 그 `print` 가 **`_sys` 미정의로 NameError** 를 던져 바깥
                #   `except: pass` 가 **리셋 대입까지 삼켰다** — *"마커만 추가·거동 변화 0"* 이라던
                #   커밋(`a627a18b`)이 상한을 **영구 래치**로 바꿨다(x381 줄-추적으로 확정).
                #   `_sys` 는 이 모듈의 **함수 안(:5377)** 에서만 정의된다 ⇒ 모듈-레벨 함수는
                #   `sys` 를 써야 한다. 관측이 기능보다 뒤에 오게 순서를 고정한다.
                _was = getattr(self, "_t2_resolve_deny", 0)
                self._t2_resolve_deny = 0                        # 진행 있음 → 정체 카운터 리셋
                if _was:
                    print("[T2_RESOLVE_CAP] 리셋(실행): 새 실행 %s (정체 %d회 → 0)"
                          % (sorted(done - prev)[:3], _was),
                          file=sys.stderr, flush=True)
        except Exception:
            pass
        # ★2026-08-14 (x303/x304/x305·087 실측): **새 이름 회수도 진행이다**.
        #   087 은 이름 노출 전 구간에서 캡 3회를 다 쓰고(그때 formalize 는 정직하게 `none`
        #   8/8 — x305 PRE), 옳은 이름이 KB 로 도착한 turn30~32 에는 `resolve_cap` 으로
        #   **침묵**했다. 그런데 그 이름이 후보에 들면 formalize 는 **8/8 로 그것을 고르고**
        #   (x305 POST), 그 이름을 담은 출시 문면은 그 컷을 **6/8 로 연다**(x304 B_STEP2).
        #   즉 남은 잔여는 모델이 아니라 우리 침묵이었다. [[57]] 대로 **횟수가 아니라 인자
        #   변화**로 되돌린다 — 회수 후보 집합이 **커졌을 때만** 리셋이라 반복 푸시의 상한은
        #   그대로다(캡의 목적 보존·반대편 계측).
        try:
            _u = ((a2 or {}).get("eplan") or {}).get("unlock_tool")
            if _u:
                import t2_resolve as _rz_cap
                _reg = _rz_cap.agent_discoverable_names(self)
                if _reg:
                    cur = set(_rz_cap._retrieved_unlockables(messages, _reg, _u))
                    pvn = getattr(self, "_t2_resolve_names", None)
                    if pvn is not None and (cur - pvn):
                        # 관측 의무(C442) — 어떤 이름이 리셋을 유발했는지까지 남긴다.
                        # ⚠**실효 리셋일 때만** 찍는다(위 ⓐ 와 같은 조건) — 두 경로의 마커 수를
                        #   그대로 비교할 수 있어야 한다. 옛 판은 무조건 찍어 ⓐ 와 셈이 어긋났다.
                        # ⛔대입이 먼저다(C538 복원·위 ⓐ 와 같은 이유). 이 경로는 옛 판이
                        #   *대입 → print* 순서라 살아 있었는데, `a627a18b` 가 순서를 뒤집어
                        #   같이 죽였다.
                        _was2 = getattr(self, "_t2_resolve_deny", 0)
                        self._t2_resolve_deny = 0
                        if _was2:
                            print("[T2_RESOLVE_CAP] 리셋(회수): 새 이름 %s (정체 %d회 → 0)"
                                  % (sorted(cur - pvn)[:3], _was2),
                                  file=sys.stderr, flush=True)
        except Exception:
            pass
        # ⚠스냅샷은 여기서 갱신하지 않는다. 이 함수는 한 턴에 여러 번 불리므로 검사마다 갱신하면
        #   `prev`가 항상 직전 검사 시점이 되어 **발화 사이의 진행을 못 본다**. 갱신은 발화 시점에서.
    return getattr(self, "_t2_resolve_deny", 0) < cap


def _chain_dispatch(fc, eff):
    """★관문2(2026-07-20·§2aa): follow_up_chain 1건의 발화 판정 (순수 함수·단위테스트 공유 —
    [[03b]] 별도구현 금지·라이브와 같은 코드를 잰다).
    - requires = 문자열 or **리스트(full required-set)** — 누락 있으면 feedback(`{missing}`=누락 전량 나열·
      050 follow-through+054 query-gap 동시 커버).
    - requires 전부 충족 + `decision_tools` 전부 미호출이면 decision_feedback(종단결정 nudge —
      approve 강제 아님·문구가 양방향(approve|decline) 명시·Δspurious 계측 대상).
    반환: (feedback_text, tag) or None. 엔진=집합 대조·치환만(도메인 리터럴 0).
    ★after = 문자열 or 리스트(2026-07-22 §2bv 강건화·rall12 052 실측): scaffold 판정도구(check_cli)가
    절차 anchor(submit)를 우회하는 경로서도 chain이 발화하도록 anchor 다중화 — anchor 중 하나라도
    호출되면 requires(submit 포함) 대조 → submit 미실행이 {missing}에 뜸(절차 대체 방지)."""
    _after = fc.get("after")
    _anchors = _after if isinstance(_after, list) else [_after]
    if not any(a in eff for a in _anchors):
        return None
    req = fc.get("requires") or []
    req = [req] if isinstance(req, str) else list(req)
    missing = [r for r in req if r not in eff]
    if missing and fc.get("feedback"):
        return fc["feedback"].replace("{missing}", ", ".join(missing)), "followup_chain"
    if (not missing and fc.get("decision_tools") and fc.get("decision_feedback")
            and not any(t in eff for t in fc["decision_tools"])):
        return fc["decision_feedback"], "followup_decision"
    return None


def _mut_key_of(tc):
    """변이 하나의 동일성 = 실행 이름 + 인자(문자열 접기). `t2_forensic.mut_key` 와 같은 모양."""
    try:
        a = _args_dict(tc) or {}
        return "%s|%s" % (_exact_tool_name(tc) or _eff_tool_name(tc),
                          json.dumps({k: str(v) for k, v in sorted(a.items())},
                                     ensure_ascii=False))
    except Exception:
        return ""


def _once_key_of(tc, a2):
    """A2 `write_once_keys` 가 선언한 **정책의 유일성 키**로 변이 키를 좁힌다 (없으면 None).

    왜 (2026-08-28 · t7378 `task_074#s361454` · `T2_DUP_WRITE` 는 그 런에서 **켜져 있었다**):
    `apply_checking_account_credit_5829` 의 도구 설명은 축자로 "may only be called ONCE per
    checking account per customer interaction" 이라고 **계좌당 1회**를 말한다. 그런데
    `_mut_key_of` 는 이름 + **인자 전체**를 키로 쓰므로 같은 계좌에 `amount=14.5` 를 적용한 뒤
    `amount=30.0` 을 다시 적용하는 것은 **다른 키**라 통과했다. `_DUP_WRITE_FB` 문면도
    *"same tool, same arguments"* 를 전제한다.
    => **가드의 키가 정책의 키와 달랐다.** 정책이 무엇으로 유일한지는 A2 가 선언하고 엔진은
    그 이름들의 값을 **읽어 이어 붙일 뿐**이다(도메인 낱말 0 · [[05]]).

    선언이 없으면 None 을 돌려 종전 거동(인자 전체 키)을 그대로 둔다 = fail-open.
    """
    try:
        specs = (a2 or {}).get("write_once_keys") or []
    except Exception:
        return None
    if not specs:
        return None
    # `applies_to` 는 **원 도구 이름**을 쓴다(`_wev_deny_msgs` 와 같은 규약).
    #   `_eff_tool_name` 은 디스패처를 이미 해석해 `apply_checking_account_credit` 를 돌려주므로
    #   그것만 보면 `call_discoverable_agent_tool` 선언과 안 맞는다 - 첫 판에서 실제로 안 맞았다.
    raw = str(getattr(tc, "name", "") or "")
    name = _eff_tool_name(tc) or ""
    exact = _exact_tool_name(tc) or ""
    args = _args_dict(tc) or {}
    flat = dict(args)
    for _v in list(args.values()):
        if isinstance(_v, str) and _v.strip().startswith("{"):
            try:
                _inner = json.loads(_v)
            except Exception:
                continue
            if isinstance(_inner, dict):
                for _k2, _v2 in _inner.items():
                    flat.setdefault(_k2, _v2)
    for sp in specs:
        if sp.get("applies_to") not in (raw, name, exact):
            continue
        aw = sp.get("applies_when") or {}
        if aw.get("arg"):
            v = str(flat.get(aw["arg"]) or "")
            pref = aw.get("prefix")
            if pref and not v.startswith(pref):
                continue
        keys = [k for k in (sp.get("keys") or []) if k in flat]
        if not keys:
            continue
        return "once|%s|%s" % (exact or name,
                               "|".join("%s=%s" % (k, flat[k]) for k in sorted(keys)))
    return None


def _succeeded_mut_keys(msgs, a2w):
    """이 대화에서 **성공한 변이**의 키 집합. 결과 메시지가 오류가 아니어야 한다."""
    out = {}
    ms = list(msgs or [])
    for i, m in enumerate(ms):
        for tc in (getattr(m, "tool_calls", None) or []):
            if not _is_effective_write(_eff_tool_name(tc), a2w):
                continue
            tid = getattr(tc, "id", None)
            for j in range(i + 1, len(ms)):
                mj = ms[j]
                if str(getattr(mj, "role", "")) != "tool" or getattr(mj, "id", None) != tid:
                    continue
                body = str(getattr(mj, "content", "") or "")
                if not getattr(mj, "error", False) and not body.lstrip().startswith("Error:"):
                    for k in (_mut_key_of(tc), _once_key_of(tc, a2w)):
                        if k and k not in out:
                            out[k] = (i, body)
                break
    return out


_DUP_WRITE_ONCE_FB = (
    "Error: [DUPLICATE-WRITE] This tool was already run successfully for this same "
    "target earlier in this conversation, and it may only be applied ONCE per target, so "
    "this call was REMOVED and not run. It ran at message {at} and returned:\n\n"
    "{result}\n\nThat change is already done. If the amount you were about to send "
    "differs from the one that went through, do NOT apply a second one - say what was "
    "already applied and, if it is wrong, follow the policy for correcting an applied "
    "credit.")

_DUP_WRITE_FB = (
    "Error: [DUPLICATE-WRITE] This exact call (same tool, same arguments) already succeeded "
    "earlier in this conversation, so this call was REMOVED and not run - running it twice "
    "would apply the same change twice. It ran at message {at} and returned:\n\n{result}\n\n"
    "That change is already done. Do NOT attempt this change again and do not do anything "
    "further about it. Use the result above and proceed to the next step.")


def _is_effective_write(name, a2=None):
    """실효 write 술어. ★C241 U1': 도메인 어휘는 a2에서 온다(전역 상태 없음)."""
    if not name:
        return False
    if _READ_PREFIX_RE.match(name) or _PROCEDURAL_RE.search(name):
        return False
    return _SUFFIX_RE.sub("", str(name)) not in _a2_procedural(a2)


def _any_effective_write(msgs, a2=None):
    """원장에 **실효 write 실행**이 하나라도 있나 (requestor 무관 — 사용자 실행도 세상을 바꾼다).
    ★`_called_tools`와 달리 user 호출을 **포함**한다: 완료-주장의 근거는 *누가 했든* 실행 이벤트다."""
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            if _is_effective_write(_eff_tool_name(tc), a2):
                return True
    return False


def _user_all_tools(env):
    """env의 **손님-측 전체 도구** 집합 — discoverable은 그 부분집합이다.

    `_user_discoverable`만 있으면 "discoverable이 아니다"와 "손님 것이 아니다"를 구별할 수 없다.
    구판이 그 둘을 합쳐 두어, 손님이 **이미 가진** 도구를 모델에게 "네 자신의 도구"라고 말했다.
    프레임워크 API 두 개의 차집합이라 도메인 리터럴 0이다(`toolkit.get_discoverable_tools`는
    `tools` 중 discoverable 표시가 붙은 것만 돌려준다 = 구조적 부분집합).
    """
    try:
        ut = getattr(env, "user_tools", None)
        return set(getattr(ut, "tools", {}) or {}) if ut is not None else set()
    except Exception:
        return set()


def _user_discoverable(env):
    """env의 **user-side discoverable** 집합 (도메인일반·리터럴 0).
    banking 실측: `{deposit_check_3847, get_card_last_4_digits, get_referral_link, submit_cash_back_dispute_0589}`.
    구조적 근거(축자 주석): *"These tools represent actions users take in the real world. The agent gives them
    to the user via `give_discoverable_user_tool` … NOT included in the default tool list."*"""
    try:
        ut = getattr(env, "user_tools", None)
        return set(ut.get_discoverable_tools()) if ut is not None else set()
    except Exception:
        return set()


def _called_tools(msgs):
    """지금까지 **에이전트가** 실제로 호출한 도구 이름 집합 (구조 이벤트만·텍스트 무관).
    ★requestor 격리: 사용자 실행 도구(gold `call_discoverable_user_tool` 등)는 세지 않는다 —
      이 arm의 술어는 *에이전트가* producer를 불렀는가이고, user 호출을 섞으면 §7 버그와 동종의
      범주 오류가 된다."""
    out = set()
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            n = getattr(tc, "name", None)
            if n and getattr(tc, "requestor", "assistant") == "assistant":
                out.add(n)
    return out


def _fu_target_called(msgs, tool, tool_args):
    """★C212/A1 (day7 022/027 [S]): follow_up 이행 판정 — A2가 `tool_args`를 선언하면
    그 인자 부분집합이 일치하는 assistant 호출이 실재해야 '이행'. 도구명 단위 판정은
    무관-대상 동명 호출(give(get_card_last_4_digits))로 영구 침묵했다(022/027 실측).
    엔진=dict 부분집합 문자열 대조만·대상 값=A2([[05]])."""
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            if (getattr(tc, "name", None) == tool
                    and getattr(tc, "requestor", "assistant") == "assistant"):
                if not tool_args:
                    return True
                ar = _args_dict(tc)
                if all(str(ar.get(k, "")) == str(v) for k, v in tool_args.items()):
                    return True
    return False


_COVERAGE_RE = re.compile(
    r"\[coverage\] (\d+) of (\d+) rows were checked \((\d+) could not be verified\)")


# 소비 지점 컨텍스트 가드의 **검사 문턱** — 이 값 미만의 배달물은 창 검사를 아예 안 받는다.
# 큐 이어붙임이 이 선을 넘기면, 각각은 무사통과하던 둘이 합쳐져 **통째로 skip** 될 수 있다
# (감사 실측: 4536+2000=6538 → 슬롯 None · OFF 는 2000 을 받았다). 그래서 두 곳이 같은
# 상수를 봐야 한다 — 리터럴을 두 벌 두면 조용히 갈라진다.
_CP2_GUARD_MIN = 5000


# ─── CP2 배달 생애 원장 (2026-08-23 · R4 · 원장 C502) ──────────────────────────
# ⛔**순환 종점 재발 금지.** t7303 A/B 가 무효가 된 이유는 결손이 아니라 *계기*였다 — 1차 종점이
#   `[T2_CP2_APPEND] … (queue)` 였는데 그 줄은 **플래그가 꺼진 팔에서는 존재할 수 없다**. 그래서
#   "0/8 → 8/8" 은 측정이 아니라 **처치 배정의 재인쇄**였다(C502 축자). ⇒ 아래 이벤트·outcome
#   어디에서도 `T2_CP2_QUEUE` 를 읽지 않는다.
# ⛔**계기가 이미 한 번 순환이었다**(2026-08-23 실측): 보관 사이드카 14파일 전수에서
#   `agent=decision_carry` 의 `arrived` 가 **100% True**(303행·False 0)인데 그 행 수는 도달 수가
#   아니라 **`VIEW_FB` 대입 수와 1:1** 이다. 등재가 다섯 배달 자리 중 하나에만 있어서, 결국
#   *한 자리가 몇 번 발화했나*를 도달률이라 불러 온 것이다([[25]]). ⇒ 등재를 **다섯 자리 공통
#   입구**(`_cp2_assign`)로 옮긴다.
# ⚠**`arrived` 를 쓰지 않는다** — 부착 지점이 생성 직전이라 "부착"과 "도달"이 같은 말이 된다.
#   대신 배달물 하나의 생애를 **닫힌 분할**로 적는다:
#       assign → close(attached) | close(clobbered) | close(ctx_skip) | (미종결 = sim 종료 시 잔존)
#   분할이 닫혀야 `대입 = 도달 + 손실` 검산식이 서고, 그래야 **양 팔 같은 규칙**으로 잴 수 있다.
# ⚠거동 불변 계약: `_t2_cp2_track`·`_t2_cp2_seq` 두 속성과 사이드카 파일 **밖으로 나가지 않는다**.
#   `work`·`fb`·`state.messages`·`_t2_cp2_pending`·`_t2_cp2_said` 에 대입하는 문장이 하나도 없다.
# ⚠[[62]]: 고르는 것이 0 — 순위·최댓값·지목 없이 *무엇이 어디까지 갔나*만 센다.
def _cp2_open(self, text, tag, disp):
    """배달물 1건을 **미결(open)** 로 열고 `assign` 행을 남긴다 — 그 행이 **분모**다.

    분모가 사이드카에 없으면 끝내 미소비로 죽은 배달물은 흔적 0 이 되고, 도달률의 분모를
    stderr grep 으로 세게 된다(그 grep 은 다른 채널이 섞여 125를 116으로 부풀린다·실측).
    도달 판정은 여기서 **하지 않는다** — 여기서 채우면 대입을 배달로 위조한다.
    """
    try:
        _n = getattr(self, "_t2_cp2_seq", 0) + 1
        self._t2_cp2_seq = _n
        try:
            import t2_lever_beat as _lb0
            _sim, _turn = (_lb0.current_sim() or "nosim"), _lb0.current_turn()
        except Exception:
            _sim, _turn = "nosim", None
        _rec = {"agent": "cp2", "cp2_id": "%s#%d" % (_sim, _n), "cp2_tag": str(tag),
                "cp2_n": len(text or ""), "cp2_disp": str(disp), "turn": _turn}
        _tr = list(getattr(self, "_t2_cp2_track", None) or [])
        _tr.append(dict(_rec, _text=text or ""))
        self._t2_cp2_track = _tr
        import t2_fbsidecar as _fbo
        _fbo.record("cp2", text or "", None, ev="assign", **_rec)
    except Exception as _eo:
        print("[T2_CP2_TRACK] open 실패(무시): %r" % (_eo,), file=sys.stderr, flush=True)


def _cp2_close(self, outcome, slot_n=None, via=None):
    """슬롯에 열려 있던 배달물 **전부**를 `outcome` 으로 종결한다.

    슬롯이 병합본이면 조각이 여럿이므로 전부 닫는다 — 하나만 닫으면 나머지가 영원히 미결로
    남아 검산식이 깨진다. 그리고 **배달물 단위**로 닫는다(부착 단위가 아니라): 병합본 1회
    부착에 조각이 둘이면 행도 둘이다. 부착 단위로 세면 큐 ON 이 2건을 1건으로 접어 도달률이
    구조적으로 낮게 나오고, 그 순간 두 팔의 분모 정의가 달라져 A/B 가 또 무효가 된다.

    ★`via` (2026-08-23 추가·계획서 밖): ASUB 우회 회차에는 `_gen` 이 안 불리고 `work` 는
      비커밋 감사 서브콜(claimprov·selfdecl)로만 간다. 계획서는 그 회차도 `attached` 로 닫는데,
      **그 구분이야말로 지금 남은 결함**이라 라벨을 지우지 않고 `via="asub"` 로 **적어 둔다**
      (판정은 안 한다 — 나중 감사 스크립트가 가른다). 내 재현으로는 우회 11건 중 뒤이어
      claimprov/selfdecl 이 보이는 것이 **5건**이라 계획서의 11/11 은 확인되지 않았다.
    """
    try:
        _tr = getattr(self, "_t2_cp2_track", None) or []
        self._t2_cp2_track = []
        if not _tr:
            return
        try:
            import t2_lever_beat as _lb1
            _turn = _lb1.current_turn()
        except Exception:
            _turn = None
        import t2_fbsidecar as _fbc
        for _r in _tr:
            _t = _r.pop("_text", "") or ""
            _r["ev"] = "close"
            _r["outcome"] = str(outcome)
            _r["cp2_close_turn"] = _turn
            if slot_n is not None:
                _r["cp2_slot_n"] = int(slot_n)
            if via is not None:
                _r["cp2_via"] = str(via)
            _fbc.record("cp2", _t, None, **_r)
    except Exception as _ec:
        print("[T2_CP2_TRACK] close(%s) 실패(무시): %r" % (outcome, _ec),
              file=sys.stderr, flush=True)


def _cp2_assign(self, text, tag):
    """`_t2_cp2_pending` **단일 슬롯**에 배달물을 넣는다 — 단 *조용한 덮어쓰기를 금지*한다.

    ★2026-08-16 t7303 tag h 실측(우리 층 결함·검증 워크플로가 잡음): 슬롯이 하나뿐이라
      같은 턴 안에서 `T2_SEARCH_ON_PROCEED`(247자 결정문)가 `T2_DELIVER_PRECOMMIT`(문서 본문
      50,421자)를 **덮어썼고**, 소비 지점(`[T2_DECISION_CARRY] … 부착`)이 하나뿐이라 그 문서는
      영영 사라졌다. 로그에는 *"선-배달 turn=2 · 재료 50421자"* 가 찍혀 있어서 **배달된 것처럼
      보였다** — task_055 4/4 sim 이 그렇게 위장됐고, 그 위에서 "전달했는데 선택이 안 바뀐다"는
      결론이 날 뻔했다([[55]] 우리 배관 먼저 · [[25]] 우리 계기는 100% 정답 의무).

    거동은 **바꾸지 않는다**(여전히 덮어쓴다) — 이 함수는 계기다. 무엇을 버렸는지 로그에 남겨야
    다음 설계(큐로 바꿀지·부착 시점을 옮길지)가 측정 위에서 결정된다. 부피를 그냥 얹으면
    프롬프트가 44,672 한도를 넘는다(같은 런에서 `ContextWindowExceededError` 5건·전부 treat).
    """
    _prev = getattr(self, "_t2_cp2_pending", None)
    # ★R4: 병합 분기가 `text` 를 `_prev + … + text` 로 **덮어쓰므로**, 계기가 기록할 *이번
    #   배달물* 원본을 여기서 잡아 둔다. 병합 후 값을 기록하면 조각 하나가 두 번 세어진다.
    _incoming = text
    # ★대용량 anti-clobber (2026-08-16·t7304·심사 3인 일치): 미소비 배달물이 **대용량**(≥10k자·
    #   문서 본문 급)이고 새 배달물이 다른 값이면 버리지 않고 **뒤에 이어붙인다**. t7303 의
    #   055 0/4 소멸이 정확히 이 자리였다. 소형↔소형(ctl 의 247자 결정문끼리)은 종전대로
    #   덮어써서 ctl 경로 바이트 불변. 이어붙임도 로그로 남긴다(계기 [[55]]).
    # ⛔`sys` 다(2026-08-18·C538). 이 함수도 **모듈 레벨**이라 `_sys`(:5377 함수 안 정의)를 쓰면
    #   NameError 다. 여기는 `try` 밖이라 **크래시**로 터진다 — 아직 안 터진 것은 두 분기 조건
    #   (미소비 배달물이 남아 있고 값이 다름)이 라이브에서 0회였기 때문이다(t7310·t7312 전수 0).
    #   즉 잠복이었다. 같은 회귀 가족이므로 같이 고친다.
    # ★2026-08-23 (t7346 098 실측·`T2_CP2_QUEUE`·기본 OFF): 위 anti-clobber 는 **≥10k자만**
    #   구제한다. 그런데 이번에 사라진 것은 **243자**였다 —
    #     `[T2_CP2_CLOBBER] SEARCH_ON_PROCEED 가 미소비 배달물 243자를 버리고 247자로 덮어씀`
    #   그 sim 은 098#s626729 로, 우리 검색 서브의 답이 모델의 이름 확정보다 늦게 도착해 실패한
    #   두 sim 중 하나다(t7336 의 같은 태스크는 CLOBBER **0건**·2/2 통과). 같은 런에서 057 ×2 ·
    #   063 ×2 도 맞았고 셋 다 0/2 다. 크기는 이 결함의 본질이 아니다 — **버린다는 것**이 본질이다.
    #   ⇒ 크기와 무관하게 **이어붙인다**. 소형↔소형까지 바뀌므로 ctl 바이트가 달라진다 ⇒ 플래그로
    #     감싸고 기본 OFF 다([[70]] 켜기 전에 손해도 재라 · 어제 A1~A16 을 안 재고 켠 대가를 치렀다).
    #   ⚠부피 상한을 넘으면 이어붙이지 않고 **종전대로 덮어쓰되 그 사실을 남긴다**(가시성 유지).
    #   소비 지점의 `_ctx_fits` 가드는 그대로 뒤를 받친다(≥5k자만 검사).
    #   ⛔**초판이 OFF 를 깼다**(2026-08-23·`test_cp2_queue_behavior` 가 잡음·감사 워크플로).
    #     상한 조건을 `_queue` **밖에** 걸어서, 구판이 무조건 이어붙이던 `len(_prev)>=10000` 영역이
    #     `len(_prev)+len(text)+2 > cap` 일 때 **덮어쓰기로 바뀌었다** — 그리고 `go_stack.sh` 가
    #     그 상한을 항상 export 하므로 라이브에서 유효했다. 즉 커밋 메시지의 *"default off, control
    #     bytes unchanged"* 가 거짓이었다. ⇒ 구판 구제(`_big`)는 **상한을 받지 않는다**(바이트 불변).
    #     상한은 **큐 분기에만** 건다.
    try:
        _cap = int(os.environ.get("T2_CP2_APPEND_MAX", "90000"))
    except (TypeError, ValueError):
        # ⛔`int()` 가 `try` 밖이라 비정수 env 하나로 5 배달 자리 전부가 크래시했다(같은 감사).
        #   이 함수의 `_sys` NameError 주석이 부른 '잠복'과 같은 종류라 같이 닫는다.
        _cap = 90000
        print("[T2_CP2_APPEND] T2_CP2_APPEND_MAX=%r 가 정수가 아니다 — 기본 %d 사용"
              % (os.environ.get("T2_CP2_APPEND_MAX"), _cap), file=sys.stderr, flush=True)
    _queue = os.environ.get("T2_CP2_QUEUE") == "1"
    # ★큐 ON 에서 **빈 배달물은 배달이 아니다** — 쌓인 것을 지우지 않는다(초판은 `and text` 조건
    #   때문에 빈 문자열이 clobber 분기로 떨어져 pending 을 지웠다). OFF 는 종전 그대로 둔다.
    if _queue and _prev and not text:
        print("[T2_CP2_APPEND] %s: 빈 배달물 — 미소비 %d자를 유지한다" % (tag, len(_prev)),
              file=sys.stderr, flush=True)
        return
    _big = bool(_prev and _prev != text and text and len(_prev) >= 10000)   # 구판 구제(상한 없음)
    #   ★그리고 **가드 문턱을 넘기면 이어붙이지 않는다**: 새 배달물이 혼자서는 검사조차 안 받는
    #     크기인데(<_CP2_GUARD_MIN) 합치면 검사 대상이 되어 **통째로 skip** 될 수 있다. 그러면 큐가
    #     OFF 보다 **덜** 배달한다(감사 실측 4536+2000 → 0자). 그 국면에서는 종전대로 덮어쓴다 ⇒
    #     큐 ON 은 어떤 국면에서도 OFF 보다 적게 전달하지 않는다.
    _qcross = bool(text and len(text) < _CP2_GUARD_MIN
                   and _prev and len(_prev) + len(text) + 2 >= _CP2_GUARD_MIN)
    _qok = bool(_queue and _prev and _prev != text and text and not _big and not _qcross
                and len(_prev) + len(text) + 2 <= _cap)                     # 큐 구제(상한 있음)
    # ★`_qcross` 의 **거울** (2026-08-24 · P1-B 실측): 기존 가드는 *작은 것이 합쳐서 문턱을 넘는*
    #   경우만 막는다. 반대 방향이 안 막혀 있다 — **확실히 배달될 소형**이 **검사를 받게 될 대형**에
    #   밀려 죽고, 그 대형마저 창 초과로 버려진다. 실물 057#s373753(t7348):
    #     turn1 247자 → turn9 clobbered · turn9 247자 → turn11 clobbered
    #     turn11 **87,407자** → turn19 `ctx_skip`      ⇒ 세 배달물 **전부 소실 · 모델은 못 받았다**
    #   `_prev < _CP2_GUARD_MIN` 은 소비 지점 가드가 **검사조차 안 하는** 크기라 반드시 배달된다.
    #   그것을 `>= _CP2_GUARD_MIN` 인 것과 바꾸는 것은 **확실한 배달을 불확실한 배달과 맞바꾸는 것**
    #   이고, 그 국면에서는 들어온 쪽을 버리는 편이 언제나 배달량이 많다.
    #   ⚠[[70]]: 파는 것 = 그 대형 배달물. 단 그것은 창 초과면 어차피 버려진다(그 자리가 `ctx_skip`).
    #     ctl 바이트가 달라지므로 **기본 OFF**·측정 후 승격한다(큐 플래그의 선례 그대로).
    #   ⛔★②범주 축에서는 켜지 마라 (2026-08-24 P3 실측): 이 자리에 배달되는 서브 결정문은
    #     **태스크와 무관한 상수**다 — 055·057·063 여섯 sim 전부 `Blue Account → Gold Account`
    #     순서이고 057 이 맞는 것은 gold 가 마침 Blue Account 라서다. 배달을 확실하게 만들면
    #     055·063 에 **오답을 확실히 배달**하게 된다. 그 축의 옳은 수리는 배달 **객체**를 바꾸는
    #     `T2_PROCEED_DOCBODY`(x335b 격리 24/24)이지 배달 **확실성**이 아니다.
    _keep_sure = os.environ.get("T2_CP2_KEEP_SURE") == "1"
    if (_keep_sure and _prev and text and _prev != text and not _big and not _qok
            and len(_prev) < _CP2_GUARD_MIN <= len(text)):
        print("[T2_CP2_KEEP] %s: 확실한 미소비 %d자를 지키고 들어온 %d자(가드 검사 대상)를 버린다"
              % (tag, len(_prev), len(text)), file=sys.stderr, flush=True)
        return
    if _big:
        # 문구도 구판 축자 그대로 — 과거 런 로그를 grep 하는 포렌식이 둘을 다 받게 하지 않는다.
        print("[T2_CP2_APPEND] %s: 미소비 대용량 %d자 뒤에 %d자 이어붙임"
              % (tag, len(_prev), len(text)), file=sys.stderr, flush=True)
        text = _prev + "\n\n" + text
    elif _qok:
        print("[T2_CP2_APPEND] %s: 미소비 %d자 뒤에 %d자 이어붙임 (queue)"
              % (tag, len(_prev), len(text)), file=sys.stderr, flush=True)
        text = _prev + "\n\n" + text
    elif _prev and _prev != text:
        _why = ""
        if _queue and text and len(_prev) + len(text) + 2 > _cap:
            _why = " ⚠상한 %d 초과라 이어붙이지 못함" % _cap
        elif _queue and _qcross:
            _why = (" ⚠합치면 %d자로 가드 문턱(%d) 을 넘어 통째로 skip 될 수 있어 이어붙이지 않음"
                    % (len(_prev) + len(text) + 2, _CP2_GUARD_MIN))
        print("[T2_CP2_CLOBBER] %s 가 미소비 배달물 %d자를 버리고 %d자로 덮어씀%s"
              % (tag, len(_prev), len(text or ""), _why), file=sys.stderr, flush=True)
    # ★생애 등록 (R4). 배달물 하나는 `attached · clobbered · ctx_skip` 중 정확히 하나로 끝나거나
    #   sim 종료까지 미결로 남는다(=잔존). 세 라벨 어느 것도 `_queue` 를 보지 않는다 — 그것이
    #   팔-대칭의 전부다(C502 가 무너진 자리).
    # ⚠빈 배달물은 배달이 아니다(열지 않는다). `_prev == _incoming` 재대입도 새 배달이 아니다 —
    #   슬롯 내용이 그대로라 이미 열린 건이 계속 유효하다.
    if _prev and _prev != _incoming and not (_big or _qok):
        _cp2_close(self, "clobbered")          # 앞 건은 여기서 죽는다(양 팔 같은 규칙)
    if _incoming and _incoming != _prev:
        _cp2_open(self, _incoming, tag,
                  "append" if (_big or _qok) else ("clobber" if _prev else "fresh"))
    self._t2_cp2_pending = text


def _coverage_pending(msgs):
    """★C212/B1 (day7 019/022/027 [S]): 엔진 자기-생성 `[coverage]` 라인의 미판정(skipped>0)
    잔존 검출 — 이후 같은 도구의 skipped==0 결과가 나오면 해소. 엔진↔엔진 프로토콜 파싱만
    (자기 템플릿 regex·NL 판단 0). 반환: (tool_name, coverage_line) or None."""
    id2name, pend = {}, None
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            id2name[getattr(tc, "id", None)] = getattr(tc, "name", None)
        if getattr(m, "role", None) != "tool":
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        mt = _COVERAGE_RE.search(c)
        if not mt:
            continue
        nm = id2name.get(getattr(m, "id", None))
        line = c[mt.start():].split("\n", 1)[0]
        if int(mt.group(3)) > 0:
            pend = (nm, line)
        elif pend and pend[0] == nm:
            pend = None
    return pend


_UNVERIFIED_RE = re.compile(r"'unverified':\s*\[\s*\{")


def _unverified_pending(msgs):
    """★C214/E1 (day8 003 [S]): scaffold 판정도구가 'unverified'(미문서화·조건부 사실) 행을
    돌려줬는데 **조건을 확정하는 재호출이 없는** 상태 검출. 엔진 자기-출력 구조 파싱만
    (coverage-FU와 동형·NL 판단 0). 반환 (tool_name, 요약) or None."""
    id2name, pend = {}, None
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            nm = getattr(tc, "name", None)
            id2name[getattr(tc, "id", None)] = nm
            if pend and nm == pend[0]:
                pend = None                      # 같은 도구 재호출 = 조건 확정 시도 → 해소
        if getattr(m, "role", None) != "tool":
            continue
        c = getattr(m, "content", None)
        if isinstance(c, str) and _UNVERIFIED_RE.search(c):
            nm = id2name.get(getattr(m, "id", None))
            if nm:
                seg = c[c.index("'unverified'"):][:220]
                pend = (nm, seg)
    return pend


_UNKNOWN_TOOL_RE = re.compile(r"Unknown discoverable tool '([^']+)'")
_UNEXPECTED_PARAM_RE = re.compile(r"Unexpected parameter: ([A-Za-z_][A-Za-z0-9_]*)")


def _rejected_params(msgs):
    """★C212/A3 (day7 018 [S]): env가 'Unexpected parameter'로 반려한 인자명 집합(축자·발명 0).
    018은 같은 오인자(correct_rewards) give를 6회 반복 전멸했다."""
    out = set()
    for m in msgs:
        if getattr(m, "role", None) != "tool" or not getattr(m, "error", True):
            continue
        c = getattr(m, "content", None)
        if isinstance(c, str):
            out.update(_UNEXPECTED_PARAM_RE.findall(c))
    return out


def _unknown_tool_names(msgs):
    """★C212/B3 (day7 010/014/015/016 [S]): env가 'Unknown discoverable tool'로 반려한
    이름 집합(에이전트 give 시도·user 호출 에러 양 채널). 이름=env 에러 축자(엔진 발명 0)."""
    out = set()
    for m in msgs:
        if getattr(m, "role", None) != "tool":
            continue
        c = getattr(m, "content", None)
        if isinstance(c, str):
            out.update(_UNKNOWN_TOOL_RE.findall(c))
    return out


def _parse_declaration(text):
    """LLM 선언의 JSON 라인만 파싱. 엔진은 이 **구조체**만 읽는다(답변 본문 아님).
    파싱 실패 = 선언 없음 = 무개입 폴백(T5-C 불변량 iv: 실패 시 레버 ≥ floor)."""
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip().rstrip(",")
        if not (ln.startswith("{") and ln.endswith("}")):
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if isinstance(d, dict) and "operand" in d:
            out.append(d)
    return out


# A2 `analysis_producers`: [{data_source, producer, subject}] — 읽은 데이터원에 붙는 분석 producer.
DISCREQ_FEEDBACK = (
    "Error: [DISCOVERY] you have read records from '{data_source}', but you have not called "
    "'{producer}' — the tool that determines {subject}. Do not judge {subject} yourself by reading "
    "the raw records; that is what '{producer}' is for. Call '{producer}', passing the values you "
    "read from the records, and base your reply on what it returns. If you are missing an argument "
    "it needs, ask the customer for it."
)

# A2 `assertion_operands`: {operand: producer} — 그 operand를 산출하는 도구.
SELFDECL_PROMPT = (
    "Before your reply is sent, declare its factual basis. For each item listed below, output exactly "
    "one JSON line:\n"
    '{{"operand": "<item>", "claimed": true|false, "source": "GET"|"FIND"|"INFER"|"ASK"}}\n'
    "claimed = true only if your reply states or implies a conclusion about that item.\n"
    "source: GET = a tool you called returned it · FIND = it appears verbatim in this conversation · "
    "INFER = you worked it out yourself · ASK = you still need the customer to tell you.\n"
    "Output only the JSON lines, nothing else.\nItems: {items}"
)
SELFDECL_FEEDBACK = (
    "Error: [DECLARATION] you declared that your reply's conclusion about '{operand}' is INFER — you "
    "worked it out yourself. But '{producer}' is one of your available tools and it returns exactly "
    "that. Do not INFER what a tool can return: call '{producer}', then restate your reply from its "
    "result. If you are missing an argument it needs, ask the customer for it."
)


# ═══════════════════════════════════════════════════════════════════════════
# ★T5-C silent repair (2026-07-11) — 정본 `T5C_SILENT_REPAIR_DESIGN_2026_07_11.md` §6
#   불변량: (i) 커밋될 턴 불파기 (ii) 대화에 새 텍스트 0 (iii) 실행=기록(replay-clean)
#           (iv) 실패 시 폴백=무개입(레버 ≥ floor pointwise)
# ═══════════════════════════════════════════════════════════════════════════

def _subst_arg_value(tc, k, old, new):
    """인자 k 안의 old 값만 제자리 치환(위치 보존). 성공 True·그 외 no-op False. (B1)
    스칼라=전체 일치 시 교체 / 리스트=일치 원소가 정확히 1개일 때 그 원소만(new 중복 시 no-op)
    / nested dict=no-op. str-JSON arguments도 재할당으로 보존(N2)."""
    try:
        d = _args_dict(tc)
        if k not in d:
            return False
        v = d[k]
        new_s = str(new).strip()
        if isinstance(v, dict):
            return False
        if isinstance(v, list):
            hits = [i for i, x in enumerate(v) if str(x).strip() == old]
            if len(hits) != 1 or any(str(x).strip() == new_s for x in v):
                return False
            v2 = list(v)
            v2[hits[0]] = new
            d[k] = v2
        else:
            if str(v).strip() != old:
                return False
            d[k] = new
        tc.arguments = d
        return True
    except Exception:
        return False


def _min_enclosing_record(obj, target):
    """target 스칼라를 담는 최소 dict 반환(깊은 쪽 우선·재귀). 없으면 None."""
    if isinstance(obj, dict):
        for vv in obj.values():
            if isinstance(vv, (dict, list)):
                r = _min_enclosing_record(vv, target)
                if r is not None:
                    return r
        for kk, vv in obj.items():
            if isinstance(vv, (dict, list)):
                if isinstance(vv, list) and any(
                        not isinstance(x, (dict, list)) and str(x).strip() == target for x in vv):
                    return obj
                continue
            if str(vv).strip() == target or str(kk).strip() == target:
                return obj
        return None
    if isinstance(obj, list):
        for vv in obj:
            if isinstance(vv, (dict, list)):
                r = _min_enclosing_record(vv, target)
                if r is not None:
                    return r
        return None
    return None


def _record_snippet(rec, limit=500):
    try:
        s = json.dumps(rec, ensure_ascii=False, default=str)
    except Exception:
        s = str(rec)
    if len(s) > limit and isinstance(rec, dict):
        flat = {kk: vv for kk, vv in rec.items() if not isinstance(vv, (dict, list))}
        try:
            s = json.dumps(flat, ensure_ascii=False, default=str)
        except Exception:
            s = str(flat)
    return s[:limit]


def _candidate_records(arg_key, orig_value, msgs, limit=6):
    """후보값 + 그 후보가 등장한 최소 enclosing 레코드 snippet. (E-ISO ③: id-only 열거 금지)
    원천 = 에이전트 자신이 조회한 tool 출력만(규칙0·리뷰 N5 검증)."""
    cands = _grounded_candidates(arg_key, orig_value, msgs, limit=limit, lenient=True)
    outs = _parse_tool_outputs(msgs, lenient=True)
    recs = []
    for c in cands:
        snip = ""
        for out in outs:
            if isinstance(out, (dict, list)):
                r = _min_enclosing_record(out, str(c).strip())
                if r is not None:
                    snip = _record_snippet(r)
                    break
        recs.append((c, snip))
    return recs


SUBCALL_SYS = (
    "You are resolving ONE ambiguous tool-call argument for a customer-service agent. "
    "Read the conversation transcript and the candidate values (each shown with the data record "
    "it came from), then decide which single candidate the user actually intends."
)


def _text_transcript(msgs, limit_chars=6000):
    """user/assistant 텍스트 턴만 전사(tool 원문 제외·_BLOCK_NOTE 등 개입 메타텍스트 절단=N5)."""
    parts = []
    for m in msgs:
        role = getattr(m, "role", None)
        c = getattr(m, "content", None)
        if role not in ("user", "assistant") or not isinstance(c, str) or not c.strip():
            continue
        t = c
        i = t.find(_BLOCK_NOTE)
        if i >= 0:
            t = t[:i]
        if t.strip():
            parts.append(("User: " if role == "user" else "Agent: ") + t.strip())
    out = "\n".join(parts)
    return out[-limit_chars:]


def _parse_subcall_answer(txt, cands):
    """서브콜 응답 파싱(N3): ① 응답 전체가 정확히 후보 1개(따옴표/공백/구두점 strip) → 수락
    ② 경계-인식 부분검색 유일 매치 → 수락 ③ 그 외 None(UNSURE)."""
    raw = (txt or "").strip()
    t = raw.strip().strip('"\'`.,;: \n\t')
    for c in cands:
        if t == str(c).strip():
            return c
    found = []
    for c in cands:
        cs = str(c).strip()
        idx = raw.find(cs)
        if idx >= 0:
            before = raw[idx - 1] if idx > 0 else " "
            after = raw[idx + len(cs)] if idx + len(cs) < len(raw) else " "
            if not (before.isalnum() or after.isalnum()):
                found.append(c)
    return found[0] if len(found) == 1 else None


def _apply_principle_default(am, a2, gate, ctx):
    """★T5-C P2 원리-디폴트(silent): write tool의 특정 인자가 A2 default_specs에 있으면
    그 인자의 *기본값*을 resolver_path로 조회(read-only)하고, 현재값이 기본값과 다르며
    사용자가 명시 override하지 않았으면(현재값이 user 발화 ctx에 미등장) 기본값으로 제자리 치환.
    C58 원리디폴트 .940(payment=주문 원결제·환불규칙). 턴 불파기·대화 불변·엔진 general.
    a2=augmented dict·gate=GateInterpreter(resolvers)·ctx=user-발화 lower 텍스트. 카운터 반환."""
    import sys as _s
    specs = (a2 or {}).get("default_specs") or []
    if not specs or gate is None:
        return 0
    rf = (getattr(gate, "resolvers", None) or {}).get("resolve_field")
    if not rf:
        return 0
    n = 0
    for tc in (getattr(am, "tool_calls", None) or []):
        nm = getattr(tc, "name", None)
        d = _args_dict(tc)
        for sp in specs:
            arg = sp.get("arg")
            if nm not in (sp.get("applies_to") or []) or arg not in d:
                continue
            path = sp.get("resolver_path")
            if not path or not d.get(path[0]):
                continue
            try:
                default_val = rf(path, d)          # 주문 원결제 (read-only)
            except Exception:
                default_val = None
            if not default_val:
                continue
            cur = str(d.get(arg)).strip()
            if cur == str(default_val).strip():
                continue                            # 이미 원결제 = 정답
            # 사용자가 현재값을 명시했나 (override) → 유지·유동성 보존
            if cur.lower() in (ctx or ""):
                continue
            # 원리-디폴트 위반 = 원결제로 제자리 치환 (str-JSON 재할당 N2 = _subst 사용)
            if _subst_arg_value(tc, arg, cur, str(default_val)):
                n += 1
                print("[T2_PRINCIPLE_DEFAULT] %s.%s %s -> %s" % (nm, arg, cur, default_val),
                      file=_s.stderr, flush=True)
    return n


def _env_verified_args(a2):
    """env가 lookup으로 검증하는 id-형 인자의 key-token 집합 (B3): A2 preconditions/ownership
    게이트의 resolver_path[0] 파생 — 기존 A2 사실만 사용·신규 도메인 리터럴 0."""
    toks = set()
    for g in (a2 or {}).get("gates", []) or []:
        if g.get("kind") not in ("preconditions", "ownership"):
            continue
        paths = [g.get("resolver_path")] + [ch.get("resolver_path")
                                            for ch in (g.get("checks") or [])]
        for rp in paths:
            if rp:
                toks |= _key_tokens(rp[0])
    return toks


def _in_error_loop(msgs, tool_name, npairs=6):
    """최근 npairs개 tool-result 중 같은 tool의 error=True 존재 여부 (N6: tool_calls.id↔ToolMessage.id join).
    unified/prov 경로선 deny가 히스토리에 안 남으므로 error=True ≡ env-오류."""
    id2name = {}
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            id2name[getattr(tc, "id", None)] = getattr(tc, "name", None)
    pairs = 0
    for m in reversed(msgs):
        if getattr(m, "role", None) != "tool":
            continue
        pairs += 1
        if pairs > npairs:
            break
        if getattr(m, "error", False) and id2name.get(getattr(m, "id", None)) == tool_name:
            return True
    return False


def _ref_iso_repair(self, la, UserMessage, msgs, am, specs):
    """★T2_REF_ISO (2026-07-24 C124/C125): 선언-write의 참조-인자를 write 직전 **최소-문맥 격리
    재선택**(유저 발화 축자 + producer 목록만) 후 불일치 시 제자리 치환. 근거=C124 실측: 동일 정보라도
    전체-궤적 문맥은 인접-행/자기-정박 슬립을 9/9 재현·최소-문맥은 9/9 정답 — DISAMB subcall(전체
    transcript)과 달리 **자기-생성 매핑을 프롬프트에서 제거**하는 것이 본질. 재선택 주체=모델(격리
    서브콜·§1.4 F2 처방 "격리 sub-call+결정론 실행")·엔진=문맥 재구성+정규식 대조+치환만(P-B
    disamb-subcall 선례·대화/턴 불변·모든 예외 no-op). A2 `ref_iso`(applies_to/applies_when/param/
    producer_tools/id_pattern) — 엔진 도메인 리터럴 0. 답이 목록에 실재해야만 채택(날조 차단)."""
    import re as _re
    import sys as _s
    for tc in (getattr(am, "tool_calls", None) or []):
        args = getattr(tc, "arguments", None)
        if not isinstance(args, dict):
            continue
        for sp in (specs or []):
            if getattr(tc, "name", None) != sp.get("applies_to"):
                continue
            aw = sp.get("applies_when") or {}
            if aw.get("arg") and not str(args.get(aw["arg"]) or "").startswith(aw.get("prefix", "")):
                continue
            pn = sp.get("param")
            nested = args.get("arguments")
            nd = None
            if isinstance(nested, str):
                try:
                    nd = json.loads(nested)
                except Exception:
                    nd = None
            elif isinstance(nested, dict):
                nd = nested
            cur = str((nd or {}).get(pn) or args.get(pn) or "")
            if not cur:
                continue
            # ★C126 라이브 교정(rall21): 같은 (param,값) 재검이 cap을 소진(031서 keep×8=동일 값)
            #   → verdict 메모이즈. switch 결과도 기억(같은 오값 재등장 시 무비용 치환).
            _memo = self._t2_refiso_memo = getattr(self, "_t2_refiso_memo", {})
            _mk = (pn, cur)
            if _mk in _memo:
                _mv = _memo[_mk]
                if _mv not in (None, cur) and nd is not None:
                    nd[pn] = _mv
                    if isinstance(nested, str):
                        args["arguments"] = json.dumps(nd)
                    print("[T2_REF_ISO] memo-switch param=%s %s->%s" % (pn, cur, _mv),
                          file=_s.stderr, flush=True)
                continue
            _prod = set(sp.get("producer_tools") or [])
            _pids = {getattr(c2, "id", None)
                     for m in msgs for c2 in (getattr(m, "tool_calls", None) or [])
                     if getattr(c2, "name", None) in _prod}
            listing = ""
            for m in msgs:
                if (getattr(m, "role", None) == "tool" and getattr(m, "id", None) in _pids
                        and not getattr(m, "error", False)):
                    c3 = getattr(m, "content", None)
                    if isinstance(c3, str) and len(c3) > len(listing):
                        listing = c3
            if not listing:
                continue
            utext = "\n".join(str(getattr(m, "content", "") or "") for m in msgs
                              if getattr(m, "role", None) == "user")[:6000]
            others = {k2: v2 for k2, v2 in (nd or {}).items() if k2 != pn}
            # ★C241 U3': 도메인 명사 제거. 이 서브콜의 과제는 "제시된 listing에서 어느 항목을
            #   가리키는가"의 참조 해소이고, 그 판정은 listing과 손님 발화에서 나온다 —
            #   업종 명사는 정보를 더하지 않는다. A2 키를 신설하지 않는 쪽을 택했다(순증 0).
            #   ⚠프롬프트 변경이므로 **행동 불변을 주장하지 않는다** — 측정 게이트 §5 참조.
            prompt = ("You are a precise assistant.\n\n"
                      "=== CUSTOMER MESSAGES (verbatim) ===\n" + utext
                      + "\n\n=== RECORD LISTING (tool output) ===\n" + listing[:20000]
                      + "\n\n=== ACTION BEING FILED ===\n"
                      + json.dumps(others, default=str)[:800]
                      + "\n\nWhich single '" + str(pn) + "' value from the RECORD LISTING does this "
                        "action refer to, based on the customer's messages? If the customer listed "
                        "several items, first match EVERY listed item to its record"
                      + ((" (" + str(sp.get("match_hint")) + ")") if sp.get("match_hint") else "")
                      + ", then answer for THIS action only. Answer with EXACTLY "
                        "one value copied from the listing, or UNSURE.")
            import t2_subcall as _SC
            stxt = _SC.sub_generate(self, la, UserMessage, prompt, "ref_iso_subcall")
            pat = sp.get("id_pattern") or r"[A-Za-z0-9_]{6,}"
            hits = [h for h in _re.findall(pat, stxt) if h in listing]
            self._t2_refiso = getattr(self, "_t2_refiso", 0) + 1
            # ★C126: 서로 다른 listing-멤버가 2개+ 언급되면(추론 산문 오염) 첫-hit 오채택 위험 →
            #   보수적으로 unsure. 단일 고유 hit만 채택.
            if "UNSURE" in stxt or not hits or len(set(hits)) > 1:
                _memo[_mk] = None
                print("[T2_REF_ISO] unsure param=%s cur=%s nhits=%d"
                      % (pn, cur, len(set(hits))), file=_s.stderr, flush=True)
                continue
            ans = hits[0]
            if ans == cur:
                _memo[_mk] = cur
                print("[T2_REF_ISO] keep param=%s val=%s" % (pn, cur), file=_s.stderr, flush=True)
                continue
            try:
                if nd is not None:
                    nd[pn] = ans
                    if isinstance(nested, str):
                        args["arguments"] = json.dumps(nd)
                else:
                    args[pn] = ans
            except Exception:
                continue
            _memo[_mk] = ans
            print("[T2_REF_ISO] switched param=%s %s->%s" % (pn, cur, ans),
                  file=_s.stderr, flush=True)


def _t5c_disamb_subcall(self, la, UserMessage, state_msgs, tc, k, s, sub_args):
    """P-B: 격리 서브콜로 |C|>=2 선택 판정 → 서브콜 답이 원값과 다르고 화이트리스트(A2
    disamb_sub_args) 타입일 때만 인자 제자리 치환. 원턴·대화 완전 불변·모든 예외 = no-op.
    반환: 'switch'|'keep'|'unsure'|'error' (계측용)."""
    import sys
    try:
        records = _candidate_records(k, s, state_msgs)
        if len(records) < 2:
            return "unsure"
        # ★FORMALIZE-EXEC(레버3·T2_FEXEC=1·NEXT_LEVER_GEN §2): 기준-형식화 서브콜 →
        #   결정론 실행 → 결과를 이 DISAMB 서브콜 *프롬프트*에 비커밋 후보-주석으로 첨부
        #   (P-B 좌석 공유·별도 후크 0·대화/턴 불변). 실패/none = 주석 없음(기존 폴백·§2.4).
        fx_note = ""
        if os.environ.get("T2_FEXEC") == "1":
            try:
                import t2_formalize_exec as _fx
                fx_note = _fx.fexec_for_disamb(self, la, UserMessage, state_msgs, k, s) or ""
            except Exception as _fe:
                print("[T2_FEXEC] error (no-op): %r" % (_fe,), file=sys.stderr, flush=True)
                fx_note = ""
        prompt = (SUBCALL_SYS + "\n\n=== Conversation ===\n" + _text_transcript(state_msgs)
                  + "\n\n=== Candidates for '" + str(k) + "' ===\n"
                  + "\n".join("- %s%s" % (c, ("   | record: " + sn) if sn else "")
                              for c, sn in records)
                  + (("\n\n" + fx_note) if fx_note else "")
                  + "\n\nThe agent currently chose '" + s + "'. Which single candidate does "
                    "the user intend? Answer with EXACTLY one candidate value, or UNSURE.")
        import t2_subcall as _SC
        _stxt = _SC.sub_generate(self, la, UserMessage, prompt, "disamb_subcall")
        self._t2_subcall_fired = getattr(self, "_t2_subcall_fired", 0) + 1
        ans = _parse_subcall_answer(_stxt, [c for c, _ in records])
        if ans is None:
            self._t2_subcall_unsure = getattr(self, "_t2_subcall_unsure", 0) + 1
            return "unsure"
        if str(ans).strip().lower() == s.lower():
            self._t2_subcall_keep = getattr(self, "_t2_subcall_keep", 0) + 1
            return "keep"
        if (_key_tokens(k) & set(sub_args or ())) and _subst_arg_value(tc, k, s, ans):
            self._t2_subcall_switch = getattr(self, "_t2_subcall_switch", 0) + 1
            print("[T2_SUBCALL] switched arg=%s from=%s to=%s" % (k, s, ans),
                  file=sys.stderr, flush=True)
            return "switch"
        self._t2_subcall_confirmonly = getattr(self, "_t2_subcall_confirmonly", 0) + 1
        return "keep"
    except Exception as e:
        try:
            print("[T2_SUBCALL] error (no-op): %r" % (e,), file=sys.stderr, flush=True)
        except Exception:
            pass
        return "error"


def apply_provenance_regen(max_retries=4, use_badwords=True, ground=False, domain=None, disamb=False,
                           disamb_mode="dialog", prov_mode="full"):
    """LLMAgent._generate_next_message 패치 — R1b 통합 (A2-구동 hints/placeholders).
      L1 = bad_words 디코드-마스크(정적 블랙리스트=A2 placeholders ∪ 스키마-example + 세션-flagged − context).
      L2 = provenance 검증기 + 내부 재생성.
      GROUND = config-도출 candidate-surfacing.
      DISAMB = |C|>=2 write-인자 1회 재확인. disamb_mode='subcall'(T5-C P-B)이면 재확인을
        in-dialogue 재생성 대신 격리 서브콜+제자리 치환으로 수행(원턴·대화 불변).
      prov_mode='rescue'(T5-C P-C): env-검증형 id 날조는 개입 생략(env가 거부)·free-text/에러-루프만 개입.
    domain 주면 A2서 hints/placeholders 도출(없으면 도메인-일반 기본)."""
    import sys
    from tau2.agent.llm_agent import LLMAgent
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import ToolMessage, UserMessage, MultiToolMessage

    a2 = _domain_a2(domain) if domain else None
    hints = a2["_hints"] if a2 else DEFAULT_ARG_HINTS
    placeholders = a2["_placeholders"] if a2 else DEFAULT_PLACEHOLDERS
    disamb_tools = _confirm_write_tools(a2) if disamb else set()
    env_args = _env_verified_args(a2) if prov_mode == "rescue" else set()
    sub_args = set((a2 or {}).get("disamb_sub_args") or [])  # B2: 치환 화이트리스트는 A2 필드
    if os.environ.get("T2_DISAMB_ORDER") == "1":            # ★order operand도 disamb 대상(filter-then-ask·env opt-in)
        sub_args |= {"order", "order_id"}

    def _append(state, message):
        if isinstance(message, UserMessage) and getattr(message, "is_audio", False):
            raise ValueError("audio not supported")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    def _gen(self, work, bad_words, call_name, tool_choice=None):
        kw = dict(self.llm_args)
        if use_badwords and bad_words:
            eb = dict(kw.get("extra_body") or {})
            eb["bad_words"] = sorted(bad_words)
            kw["extra_body"] = eb
        if tool_choice:                          # ★레버 A(2026-07-18): tau2 `generate`의 일급 파라미터로 통과
            kw["tool_choice"] = tool_choice
        _msgs = self._system_messages + work
        # ★T2_PROMPT_DUMP (2026-08-27·기본 OFF) — **모델이 실제로 본 것**을 남긴다.
        #   왜: 오늘 두 번, 라이브 실패가 같은 접두 위 격리에서 **재현되지 않았다**
        #   (x562 B_live 4/4 · x571 A_asis 가 옳게 답함). 이유를 코드로 추정할 수 없다 —
        #   라이브 프롬프트가 **어디에도 기록되지 않기 때문**이다. 영속 궤적은 커밋된 것만
        #   담고, 우리 층 주입은 비커밋 `work` 버퍼로 들어가며, 뷰 압축까지 거친다.
        #   그 셋을 합친 결과가 여기 `_msgs` 이고, 이것이 격리의 **유일한 참조점**이다([[78]]
        #   *"iso↔live 차이는 코드로 추정하지 말고 두 프롬프트를 찍어 diff"*).
        #   ⚠크다(턴당 30~40k자). 태스크 필터와 상한을 둔다 — 무제한으로 켜지 마라.
        if os.environ.get("T2_PROMPT_DUMP") == "1":
            try:
                import t2_fbsidecar as _fbp
                import t2_lever_beat as _lbp
                _cur = str(_lbp.current_sim() or "")
                _want = os.environ.get("T2_PROMPT_DUMP_TASKS", "")
                # ★필터는 **fail-open** 이다. `current_sim()` 은 스레드-로컬이라 이 자리에서
                #   비어 있을 수 있고(t7366 실측: 레코드 0·예외 0 — 조용히 전부 걸렸다),
                #   조용한 필터는 오늘만 세 번째다. 모르면 **기록한다** — 이 플래그는 어차피
                #   기본 OFF 이고 한 태스크 런에서만 켠다.
                if (not _want) or (not _cur) or any(t.strip() and t.strip() in _cur
                                                    for t in _want.split(",")):
                    _cap = int(os.environ.get("T2_PROMPT_DUMP_MAX", "60000"))
                    _parts = []
                    for _m in _msgs:
                        try:
                            _c = _content_str(_m) or ""
                        except Exception:
                            _c = str(getattr(_m, "content", "") or "")
                        _tc = " ".join(str(getattr(t, "name", "")) 
                                       for t in (getattr(_m, "tool_calls", None) or []))
                        _parts.append("[%s]%s %s" % (getattr(_m, "role", "?"),
                                                     (" CALLS " + _tc) if _tc else "", _c))
                    _fbp.record("prompt", (chr(10).join(_parts))[:_cap], work,
                                channel="gen", call=str(call_name))
            except Exception as _pe:
                print("[T2_PROMPT_DUMP] skipped: %r" % (_pe,), file=sys.stderr, flush=True)
        _r = la.generate(model=self.llm, tools=self.tools,
                         messages=_msgs, call_name=call_name, **kw)
        try:                                    # ★P3 살리기(C248·기본 OFF) — 위 경로와 동일
            import t2_salvage as _sv
            _sv.salvage_message(_r)
        except Exception:
            pass
        return _r

    def patched(self, message, state):
        if not hasattr(self, "_t2_static_bl"):
            self._t2_static_bl = _static_blacklist(self.tools, placeholders)
            self._t2_session_bl = set()
        self._system_messages = state.system_messages
        # ★W-A(arm③): A2 base-layer의 선언 가이드를 시스템 프롬프트 말미에 보간(미선언=skip).
        try:
            import t2_declfirst as _df
            # ★관찰자 ↔ 개입 분리(2026-07-31): 이 가이드 주입은 **행동을 바꾼다**(X13 A_PROMPT arm
            #   실측: 가이드만으로 턴의 31.8%에서 봉투 산출). 반면 2패스 형식화는 이미 확정된 턴을
            #   비커밋으로 재구성하므로 **에이전트에게 아무것도 말하지 않는다**. 둘을 한 플래그로
            #   묶어두면 "pass 레버 다 켜고 계측만 얹기"가 불가능하고, Y1(가이드 없음) 대비
            #   비교성도 깨진다. ⇒ `T2_DECLFIRST_GUIDE=0`이면 **관찰자만** 돈다.
            #   기본은 1(=종전 동작 보존·arm③ 아키텍처).
            _guide_on = os.environ.get("T2_DECLFIRST_GUIDE", "1") == "1"
            _g = _df.guide_text(a2) if (os.environ.get("T2_DECLFIRST") == "1" and _guide_on) else ""
            if _g and self._system_messages and not getattr(self, "_t2_df_guided", False):
                _sm = self._system_messages[-1]
                _sm.content = (getattr(_sm, "content", "") or "") + _g
                self._t2_df_guided = True
        except Exception:
            pass
        _append(state, message)
        ctx = _ctx_with_toolnames(self, _ctx_from_messages(state.messages))

        def bw():  # 동적: 정적∪세션 − context (진짜 값은 안 막음)
            return [v for v in (self._t2_static_bl | self._t2_session_bl) if v.lower() not in ctx]

        work = list(state.messages)
        am = _gen(self, work, bw(), "agent_response")
        n = 0
        subs = 0
        # ★R3: 도구-선택자 슬롯은 날조-스캔/치환 대상이 아니다(env 스키마 도출·리테일에선 ∅).
        sel_args = _selector_args_cached(self)
        rescue_skipped = set()  # PROV-RESCUE-PERARG ①: (id(tc), k, s) — rescue 개별 pass-through
        while n < max_retries:
            fab = _first_fab_call(am, ctx, hints, exclude=rescue_skipped, selectors=sel_args)
            if fab is None:
                break  # 잔여 fab 없음 or 전부 rescue-스킵일 때만 탈출 (구 break 의미 보존)
            tc, k, s = fab

            if ground and subs < 8:
                cands = _grounded_candidates(k, s, state.messages, lenient=True)
                # B1: 원소-치환 헬퍼(리스트 인자 통짜-덮어쓰기 버그 수정) — 실패 시 regen 폴백
                if len(cands) == 1 and cands[0] != s and _subst_arg_value(tc, k, s, cands[0]):
                    self._t2_ground_sub = getattr(self, "_t2_ground_sub", 0) + 1
                    subs += 1
                    print("[T2_GROUND] substituted arg=%s val=%s -> %s" % (k, s, cands[0]),
                          file=sys.stderr, flush=True)
                    _lbeat("T2_GROUND", orch=self, target=k,
                           fact="argument was replaced with the value that exists in context")
                    continue

            if prov_mode == "rescue" and (_key_tokens(k) & env_args) and \
                    _sig(s) in ("hashid", "numid") and \
                    not _in_error_loop(state.messages, getattr(tc, "name", None)):
                # P-C: env-검증형 id 날조는 개입 생략(환경이 거부=C61 H-D 100% 중복) — 에러-루프 시만 개입
                # ★PROV-RESCUE-PERARG ①(2026-07-11 t17): 구 break=regen 루프 전체 탈출 → 같은 호출의
                #   자유텍스트 fab(address1류)이 미검사. per-arg 스킵 후 continue로 다음 fab 계속 스캔.
                #   exclude가 재방문 차단 → 카운터·마커는 스킵당 1회. n 미증가(rescue 무과금·현행 보존).
                rescue_skipped.add((id(tc), k, s))
                self._t2_prov_skipped_envdup = getattr(self, "_t2_prov_skipped_envdup", 0) + 1
                print("[T2_PROV] rescue pass-through tool=%s arg=%s val=%s"
                      % (getattr(tc, "name", "?"), k, s), file=sys.stderr, flush=True)
                continue

            n += 1
            self._t2_session_bl.add(s)
            self._t2_regen = getattr(self, "_t2_regen", 0) + 1
            work = work + [am]
            cands = _grounded_candidates(k, s, state.messages,
                                         lenient=(prov_mode == "rescue")) if ground else []
            if ground and cands:
                tmpl = GROUND_FEEDBACK_NEUTRAL if prov_mode == "rescue" else GROUND_FEEDBACK
                main_reason = tmpl.format(k=k, s=s, cands=", ".join(repr(c) for c in cands))
            else:
                directive = _resolver_directive(a2, tc, k, s)  # ★B-max① t17: resolver_path 지목
                if directive is not None:
                    main_reason = directive
                else:
                    tmpl = REGEN_FEEDBACK_NEUTRAL if prov_mode == "rescue" else REGEN_FEEDBACK
                    main_reason = tmpl.format(k=k, s=s)
            for c in (am.tool_calls or []):
                reason = main_reason if c is tc else \
                    _sibling_wait("PROVENANCE", tc, "the invented value")
                work.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                        error=True, content=reason))
            am = _gen(self, work, bw(), "agent_response_regen")

        # ─── ARG-SCHEMA 위생: 스키마 밖 인자 키 → regen (T2_ARG_SCHEMA=1·기본 OFF) ───
        # 2026-07-19 포렌식: give_discoverable_user_tool에 'arguments' 키를 얹어 026/027/028 gold give
        # 3건 전부 evaluator exact-match 실패(예측 키집합으로 dict 비교).
        # ★2026-08-02 정정: `arguments`는 env 스키마상 **합법 파라미터**
        # (`give_discoverable_user_tool(self, discoverable_tool_name: str, arguments: str = "{}")`).
        # 따라서 당시 사고의 기전은 "스키마 밖 키"가 아니라 **gold 키집합 불일치**(PRED_EXTRA_KEY)다.
        # 이 검사(ARG_SCHEMA)는 최상위 키만 보므로 arguments **문자열 내부**의 오필드·placeholder는
        # 잡지 못한다(040 실측) — 그 표적은 write_arg_grounding(내포 unwrap·§2bs)의 몫.
        # ⚠설치 경로: 이 블록은 patched() 안이고 라이브 러너는 unified()를 설치한다 = 現 死코드(P11 이설 대상).
        # 도메인일반: 근거=자기 도구 스키마(properties)뿐·값 판단 0·재발화는 모델([[05]]/[[07]] enforced).
        if os.environ.get("T2_ARG_SCHEMA") == "1" and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_schema_props"):
                _props = {}
                for _t in (self.tools or []):
                    try:
                        _sc = _t.openai_schema
                        _fn = _sc.get("function") if isinstance(_sc.get("function"), dict) else _sc
                        _nm = _fn.get("name")
                        _pr = ((_fn.get("parameters") or {}).get("properties")) or {}
                        if _nm and _pr:
                            _props[_nm] = set(_pr.keys())
                    except Exception:
                        pass
                self._t2_schema_props = _props
            _tries = 0
            while _tries < 2 and getattr(am, "tool_calls", None):
                _bad = None
                for _tc in (am.tool_calls or []):
                    _allowed = self._t2_schema_props.get(getattr(_tc, "name", None))
                    if not _allowed:
                        continue
                    _extra = [k for k in _args_dict(_tc).keys() if k not in _allowed]
                    if _extra:
                        _bad = (_tc, _extra, _allowed)
                        break
                if _bad is None:
                    break
                _tc, _extra, _allowed = _bad
                _tries += 1
                self._t2_schema_regen = getattr(self, "_t2_schema_regen", 0) + 1
                print("[T2_ARGSCHEMA] regen tool=%s extra=%s" % (_tc.name, _extra),
                      file=sys.stderr, flush=True)
                work = work + [am]
                for _c in (am.tool_calls or []):
                    _reason = ("Error: [ARG-SCHEMA] '%s' does not accept argument(s): %s. Its schema "
                               "declares ONLY these argument(s): %s. Re-issue the call with ONLY declared "
                               "arguments — remove everything else."
                               % (_tc.name, ", ".join(repr(x) for x in _extra),
                                  ", ".join(sorted(_allowed)))) if _c is _tc else \
                        _sibling_wait("ARG-SCHEMA", _tc, "its argument schema")
                    work.append(ToolMessage(id=_c.id, role="tool", requestor="assistant",
                                            error=True, content=_reason))
                am = _gen(self, work, bw(), "agent_response_schema_regen")

        # ─── DISAMB: 문맥-실재값인데 같은-형식 후보 2+개 → 1회 재확인 (선택은 모델) ───
        if disamb_tools and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_disamb_seen"):
                self._t2_disamb_seen = set()
            hit = None
            for tc in am.tool_calls:
                if getattr(tc, "name", None) not in disamb_tools:
                    continue
                for k, v in _args_dict(tc).items():
                    if not _hint_hit(k, hints):
                        continue
                    for val in _flatten(v):
                        s = str(val).strip()
                        if len(s) < 4 or s.lower() not in ctx:
                            continue
                        memo = (tc.name, k, s.lower())
                        if memo in self._t2_disamb_seen:
                            continue
                        cands = _grounded_candidates(k, s, state.messages)
                        if len(cands) >= 2 and any(s.lower() == str(c).lower() for c in cands):
                            hit = (tc, k, s, cands, memo)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit and disamb_mode == "subcall":
                # ★T5-C P-B: in-dialogue 재확인 폐지 — 격리 서브콜 판정 + 제자리 치환(원턴·대화 불변)
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d mode=subcall"
                      % (tc.name, k, s, len(cands)), file=sys.stderr, flush=True)
                _t5c_disamb_subcall(self, la, UserMessage, state.messages, tc, k, s, sub_args)
                hit = None
            if hit:
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d" % (tc.name, k, s, len(cands)),
                      file=sys.stderr, flush=True)
                dwork = list(work) + [am]
                fb = DISAMB_FEEDBACK.format(k=k, s=s, n=len(cands),
                                            cands=", ".join(repr(c) for c in cands[:8]))
                for c in (am.tool_calls or []):
                    reason = fb if c is tc else \
                        _sibling_wait("DISAMBIGUATE", tc, "the ambiguous value")
                    dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                             error=True, content=reason))
                am2 = _gen(self, dwork, bw(), "agent_response_disamb")
                # 재확인 응답이 날조를 새로 들이면 prov 루프로 정화(2회 한도)·실패 시 원 응답 유지
                n2 = 0
                while n2 < 2:
                    fab2 = _first_fab_call(am2, ctx, hints, selectors=sel_args)
                    if fab2 is None:
                        break
                    tc2, k2, s2 = fab2
                    n2 += 1
                    self._t2_session_bl.add(s2)
                    dwork = dwork + [am2]
                    for c in (am2.tool_calls or []):
                        reason = REGEN_FEEDBACK.format(k=k2, s=s2) if c is tc2 else \
                            "Error: [PROVENANCE] resolve the invented value first; do not call this yet."
                        dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                                 error=True, content=reason))
                    am2 = _gen(self, dwork, bw(), "agent_response_regen")
                if _first_fab_call(am2, ctx, hints, selectors=sel_args) is None:
                    # ★T5-C fix(C61 H-C): 재확인 응답이 tool_calls 없는 텍스트-only면 원 호출 유지.
                    #   무조건 수락이 write 유실(39건·−37시행)의 코드-확정 기전 — 원값은 문맥-실재라 유지 무해.
                    if getattr(am2, "tool_calls", None):
                        sw = any(str(vv).strip().lower() != s.lower()
                                 for c2 in am2.tool_calls if c2.name == tc.name
                                 for kk, vv0 in _args_dict(c2).items() if kk == k
                                 for vv in _flatten(vv0))
                        if sw:
                            print("[T2_DISAMB] switched arg=%s from=%s" % (k, s),
                                  file=sys.stderr, flush=True)
                        am = am2
                    else:
                        self._t2_disamb_nowrite_keep = getattr(self, "_t2_disamb_nowrite_keep", 0) + 1
                        print("[T2_DISAMB] rejected: re-check dropped tool_calls; keeping original",
                              file=sys.stderr, flush=True)
        return am

    LLMAgent._generate_next_message = patched
    return patched


# ═══════════════════════════════════════════════════════════════════════════
# ★REPLAY-SAFE 게이트 (generation-level regen) — REPLAY_SAFE_GATE_DESIGN_2026_07_06
#   문제: apply()는 deny 시 합성 ToolMessage를 히스토리에 커밋 → tau2 평가의 set_state
#   replay(mutating tool 재실행·environment.py:389 assertion)가 깨짐 → infrastructure_error.
#   해결: 게이트를 생성-레벨로 이동 — deny면 작업버퍼서 피드백+재생성(K=MAX_REGEN), 유효 호출만
#   커밋. R8 종단: K 소진 후에도 deny면 차단 mutating 호출을 커밋 前 제거 → 히스토리 replay-clean.
#   R1 예산: deny turn마다 orchestrator.num_errors++ (best-of-K 아님·too_many_errors 동일 압박).
# ═══════════════════════════════════════════════════════════════════════════

_BLOCK_NOTE = ("\n\n[Note: the tool call(s) above were blocked by a policy gate and were NOT "
               "executed. Satisfy the gate requirement (authenticate / get explicit user confirmation / "
               "check the record's status / fix the operation) before attempting the action again.]")

# ★A15/OL-55 (t7336 §6.1·2026-08-22) — 기계 노트가 **손님 발화가 되는** 자리.
#   실물: 016#1 [52]·074#1 [57] — 모델 생성분(`am.content`)이 빈 문자열인 턴에 위 노트를 붙이자
#   노트가 **본문 전체**가 되어 손님에게 나갔고, 016#1 [53] 에서 user-sim 이 역할을 혼동했다.
#   같은 자리에서 사유 문자열이 `[:70]` 슬라이스로 `has been c` 처럼 **단어 중간에서 잘렸다**.
#   ⇒ ⑴사유는 단어 경계에서 자른다 ⑵빈 본문이면 **모델에게 본문을 다시 받는다**(재생성).
#     재생성이 실패하거나 또 비면 노트를 **본문으로 커밋하지 않는다** — 기계 노트가 손님 발화가
#     되는 것은 어느 조건에서도 틀렸다(마스터 §6.1 A15).
# ⚠[[70]] 무엇을 파는가: 재생성이 실패한 턴은 **빈 본문**으로 나간다(구판은 최소한 무언가를
#   말했다). 다음 런 포렌식이 세는 것 = `[T2_BLOCK_NOTE] empty-body` 턴 수 · `regen ok` 수.
_BLOCK_NOTE_ASK = (
    "Your tool call(s) in the draft above were blocked by a policy gate and were NOT executed, so "
    "this turn has no message for the customer yet. Write that message yourself in plain prose: say "
    "what has NOT been done and what is needed next. Do not claim anything was completed, and do not "
    "emit tool calls in this reply. The gate gave these reasons: ")

# ★OL-55 형제 (2026-08-22): `T2_STALE_STRIP` 도 `am.tool_calls` 를 **전부** 지울 수 있고
#   (`_kept or None`), 그 턴에 본문이 비어 있으면 아래 노트가 **손님 발화 전체**가 된다.
#   A15 와 같은 형상이므로 같은 정본(`_commit_machine_note`)을 쓴다 — 노트는 한 일(안 보냄)만
#   말하고 결과에 대해서는 아무것도 주장하지 않는다.
_STALE_NOTE = (" [Note: %d repeated tool call(s) in this turn were not sent again. This says"
               " nothing about whether the earlier attempt succeeded - re-read the tool results"
               " above before telling the customer anything is done.]")
_STALE_NOTE_ASK = (
    "The repeated tool call(s) in the draft above were not sent again, so this turn has no message "
    "for the customer yet. Write that message yourself in plain prose, based only on the tool "
    "results already above: say what they actually show and what is still needed. Do not claim "
    "anything was completed that those results do not show, and do not emit tool calls in this "
    "reply.")


def _um(text):
    """UserMessage 생성 (구/신 시그니처 양쪽·호출부마다 try/except 를 복제하지 않기 위한 정본)."""
    from tau2.data_model.message import UserMessage as _UMb
    try:
        return _UMb(role="user", content=text)
    except TypeError:
        return _UMb(content=text)


def _trunc_reason(s, n=70):
    """사유 문자열을 **단어 경계**에서 자른다 (OL-55: `has been c` 중간 절단). 순수 문자열."""
    t = " ".join(str(s or "").split())
    if len(t) <= n:
        return t
    cut = t[:n]
    sp = cut.rfind(" ")
    if sp >= n // 2:
        cut = cut[:sp]
    return cut.rstrip(" ,;:.-") + "..."


def _commit_machine_note(am, note, ask, regen=None, tag="T2_BLOCK_NOTE"):
    """★기계 노트를 **본문 전체로 커밋하지 않는다** — 노트 문자열에 **독립인** 정본.

    ★왜 일반화했나 (2026-08-22 · OL-55 형제 · t7336 마스터 §6.1 A15 잔여):
      A15 는 `_BLOCK_NOTE` **한 자리**만 고쳤는데, 같은 형상이 `T2_STALE_STRIP` 에도 있다 —
      거기서도 `am.tool_calls` 가 전부 제거되면(`_kept or None`) 그 턴은 **손님 발화**가 되고,
      본문이 비어 있으면 *"[Note: N repeated tool call(s) …]"* 라는 기계 노트가 손님에게
      **통째로** 나간다(016#1 [52]·074#1 [57] 과 같은 형상 — user-sim 이 역할을 혼동했다).
      없다고 새로 짜면 같은 처방이 두 벌이 되어 조용히 갈린다([[67]] 실물 2건) — 그래서
      **정본을 노트-문자열에 독립으로 만들고 두 자리가 함께 쓴다**.

    · 본문이 이미 있으면 종전대로 뒤에 붙인다(거동 보존).
    · 본문이 비었으면 `regen(ask) -> str` 로 **모델에게 본문을 받는다**. 받으면 그 본문 + 노트.
      못 받으면 노트도 붙이지 않는다(빈 본문 유지).
    `regen=None`(구 호출부·단위검정)이면 재생성 없이 빈 본문 유지 — 어느 쪽이든 노트가 본문
    전체가 되는 일은 없다. 반환 = 무엇을 했나(계기 문자열: appended / regen / empty).
    `tag` = 계기 인쇄 태그(포렌식이 자리별로 센다 — 기존 `[T2_BLOCK_NOTE]` 집계 보존).
    """
    body = str(getattr(am, "content", "") or "")
    if body.strip():
        am.content = body + note
        return "appended"
    new = ""
    if regen is not None:
        try:
            new = str(regen(ask) or "")
        except Exception as _bne:                    # noqa: BLE001 — 재생성 실패는 흡수
            print("[%s] regen failed (no-op): %r" % (tag, _bne),
                  file=sys.stderr, flush=True)
            new = ""
    if new.strip():
        am.content = new + note
        print("[%s] regen ok (%d chars) — note appended to model prose" % (tag, len(new)),
              file=sys.stderr, flush=True)
        return "regen"
    am.content = body
    print("[%s] empty-body: machine note NOT committed as the whole message" % tag,
          file=sys.stderr, flush=True)
    return "empty"


def _commit_block_note(am, note, regen=None):
    """★A15/OL-55: `_BLOCK_NOTE` 를 **본문 전체로 커밋하지 않는다**. 반환 = 무엇을 했나.

    본문 조립만 하고 나머지는 정본 `_commit_machine_note` 에 넘긴다(거동 100% 보존).
    """
    return _commit_machine_note(am, _BLOCK_NOTE + " (" + note + ")",
                                _BLOCK_NOTE_ASK + note, regen=regen, tag="T2_BLOCK_NOTE")


def _iter_tc_result_pairs(messages):
    """committed 히스토리서 (assistant ToolCall, matching tool ToolMessage) 쌍 yield."""
    by_id = {}
    for m in messages:
        if getattr(m, "role", None) == "tool" and getattr(m, "id", None) is not None:
            by_id[m.id] = m
    for m in messages:
        if getattr(m, "role", None) == "assistant":
            for tc in (getattr(m, "tool_calls", None) or []):
                yield tc, by_id.get(getattr(tc, "id", None))


def _ledger_event_names(messages):
    """★A2/OL-21 (t7336 §6.1·2026-08-22): 원장 이벤트 이름 집합 — **성공한 호출만**.

    구판(`_evs`)은 `tool_calls` 의 **이름만** 모으고 결과를 보지 않았다. 그래서 **env 가 거부한
    호출**이 *"원장에 있다"* 의 근거가 되어 `_claim_unbacked` 의 `kind-index rescued` 로 통과했고,
    094#0 은 `unbacked=0` 인 채 날조 완결 주장을 3턴 무저지로 냈다. [[69]] 축자와 정면 충돌한다 —
    *"reward = 궤적 전체 재실행 후 DB 해시 비교 … 상태를 안 바꿔서 해시에 안 남는 것"*.

    술어는 F8/`t2_resolve._result_ok` 정본 그대로다(**사본 0**·[[67]]): `error=True` 도, content
    `Error` 접두도 아닌 것만 성공. 짝을 **id 로** 잇고(역할 무관 — 손님이 실행한 도구도 세야 한다·
    `give_exec_idle` docstring 의 그 오독을 반복하지 않는다), **짝을 못 찾은 호출은 남긴다**
    (fail-open: 실패를 *증명한* 것만 뺀다 — 안 그러면 반대 방향의 거짓 고발이 된다·[[25]]).

    반환 = (성공 이름 집합, 결과로 실패가 확인돼 제외된 (이름, 사유) 목록[계기용]).
    ⚠[[70]] 무엇을 파는가: 원장이 좁아지므로 **CLAIMPROV 오탐(정당한 주장을 unbacked 로 고발)이
      늘 수 있다**. 다음 런 포렌식은 `unbacked` 계수와 `kind-index rescued` 수를 짝으로 센다.
    """
    try:
        import t2_resolve as _rz_ev              # 정본 술어 재사용([[67]] 사본 0)
        _ok = _rz_ev._result_ok
    except Exception as _ee:                     # noqa: BLE001 — fail-open(구판 거동)
        print("[T2_CLAIMPROV] result-ok 술어 미가용 — 원장 좁힘 비활성: %r" % (_ee,),
              file=sys.stderr, flush=True)
        _ok = None
    by_id = {}
    for m in (messages or []):
        if getattr(m, "role", None) == "tool" and getattr(m, "id", None) is not None:
            by_id[getattr(m, "id")] = m
    out, dropped = set(), []
    for m in (messages or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            nm = str(getattr(tc, "name", "") or "")
            rm = by_id.get(getattr(tc, "id", None))
            if _ok is not None and rm is not None and not _ok(rm):
                dropped.append((nm or _eff_tool_name(tc),
                                str(getattr(rm, "content", "") or "")[:60]))
                continue
            out.add(nm)
            out.add(_eff_tool_name(tc))
    out.discard("")
    out.discard(None)
    return out, dropped


def _rebuild_gate_state(gate, a2, messages):
    """committed clean 히스토리서 auth 상태 재구성(denied 호출 부재 = 정확)."""
    gate.state.auth_user = None
    auth_tools = a2["_auth_tools"]
    for tc, tm in _iter_tc_result_pairs(messages):
        name = getattr(tc, "name", None)
        if name in obs_tools_g(gate) and tm is not None and not getattr(tm, "error", False):
            gate.observe(name, _args_dict(tc), _content_str(tm))


def unused_grants(messages, a2):
    """**주거나 잠금해제해 놓고 끝내 쓰지 않은 도구** — 정책이 명시적으로 금지한 그것 (계기 전용).

    ★정책 축자 (`prompts/components/additional_instructions.md` · 이 문장은 에이전트 시스템
      프롬프트에 실린다 — `all_tools.md` 가 `{{component:additional_instructions}}` 로 부른다):
        *"IMPORTANT: Do not unlock tools that you do not plan on giving to the user and actually
          using: this causes issues in database logging."*
        *"Only give a tool when the user would like to perform an action, and the knowledge base
          explicitly has a tool that allows the user to perform this action"*
        *"…and do not unlock tools you do not plan to use."*
    ★왜 세나 (2026-08-31·base x644 `task_010` 실측): 그 sim 은 gold 액션 **2/2 를 정확한 인자로**
      실행했는데(`action_reward` 1.0 · 1.0) `db_match=false` 로 **reward 0.0** 이었다. 차이는
      `give_discoverable_user_tool(get_referral_link)` 한 번이다 — 손님은 그것을 쓰지 않고
      `submit_referral` 을 실행했다. env 소스가 그 한 줄을 DB 변이로 만든다:
        `discoverable_tool_record = {…, "status": "GIVEN"}` → `add_to_db("user_discoverable_tools", …)`
      즉 **정책 위반 한 건이 만점 궤적을 0점으로 만든다**.
    ⚠계기뿐이다 — 아무것도 막지 않는다. 규모를 먼저 재고 나서 집행을 논한다([[62]] 결손을 재라).
    ⚠도구 이름은 전부 **A2 선언**에서 온다(`dispatcher_role_check` → 없으면 `eplan`).
      도메인 리터럴 0 · 판단 0: 집합 차집합과 이름 대조뿐이다([[59]]).
    ⚠짝짓기는 **접미사를 뗀 base 이름**으로 한다(디스패처 경유 이름은 `_1234` 가 붙는다).

    반환: {"given": [...], "unlocked": [...], "unused_given": [...], "unused_unlocked": [...]}
    """
    drs = ((a2 or {}).get("dispatcher_role_check") or {})
    epl = ((a2 or {}).get("eplan") or {})
    give_tool = drs.get("give_tool")
    user_call = drs.get("user_call")
    unlock_tool = drs.get("unlock_tool") or epl.get("unlock_tool")
    agent_call = drs.get("agent_call") or epl.get("dispatch_tool")
    keys = ("agent_tool_name", "user_tool_name", "discoverable_tool_name",
            epl.get("dispatch_name_key") or "agent_tool_name")

    def _inner(tc):
        ar = _args_dict(tc) or {}
        for k in keys:
            v = ar.get(k)
            if v:
                return re.sub(r"_\d+$", "", str(v))
        return ""

    given, unlocked, ran_user, ran_agent = set(), set(), set(), set()
    for m in (messages or []):
        role = getattr(m, "role", None)
        for tc in (getattr(m, "tool_calls", None) or []):
            nm = getattr(tc, "name", None)
            iv = _inner(tc)
            if not iv:
                continue
            if nm == give_tool:
                given.add(iv)
            elif nm == unlock_tool:
                unlocked.add(iv)
            elif nm == user_call:
                ran_user.add(iv)
            elif nm == agent_call:
                ran_agent.add(iv)
        # 손님이 자기 도구를 직접 부른 형태(디스패처 미경유)도 실행으로 센다
        if role == "user":
            for tc in (getattr(m, "tool_calls", None) or []):
                nm = getattr(tc, "name", None)
                if nm:
                    ran_user.add(re.sub(r"_\d+$", "", str(nm)))
    return {"given": sorted(given), "unlocked": sorted(unlocked),
            "unused_given": sorted(given - ran_user),
            "unused_unlocked": sorted(unlocked - ran_agent)}


def give_exec_idle(messages, give_tool, user_call):
    """건네졌으나 손님이 **아직 실행하지 않은** discoverable user 도구 (닫힌 술어·순수함수).

    ★2026-08-05 결함 수정([S]·`x64_give_exec_predicate.py`): 구판은 손님의 실행을
    `call_discoverable_user_tool` 경유만 셌다. 그런데 실측 궤적에서 손님은 **도구 이름 그대로**
    호출한다(`role=user` 메시지의 tool_calls). 그래서 이미 실행한 손님에게도 *"아직 실행하지
    않았으니 지금 실행하라고 말하라"* 는 거짓 피드백이 나갔다 — A/B4 전수에서 **발화 17건 중
    5건이 오발화, 그중 3건은 통과 궤적**(002/t0·002/t1·003/t1)이다. 핸드오프 §5-6이 분석기에서
    잡은 것과 같은 오독이 엔진에도 있었다.

    실행 판정은 **두 형태를 모두** 센다: 디스패처 경유(`user_call` + `discoverable_tool_name`)와
    손님의 직접 호출(requestor=user 또는 role=user). 인계 성사는 종전대로 **오류 아닌 결과**가
    돌아온 give만 인정한다.
    ★2026-08-22 (A9): 본문을 `give_exec_state` 로 옮기고 이 함수는 그 차집합만 낸다 —
    같은 궤적 해석이 **두 벌**이 되지 않게 하기 위해서다([[67]] 사본 금지).
    """
    given, ran = give_exec_state(messages, give_tool, user_call)
    return sorted(given - ran)


def give_exec_state(messages, give_tool, user_call):
    """`(건네진 도구, 손님이 실행한 도구)` 두 집합 — `give_exec_idle` 의 파싱 정본 (순수함수).

    ★왜 분리했나 (2026-08-22 · t7336 마스터 §6.1 A9 · §5.7 OL-46): F8(`T2_ARG_PRODUCERS`)의
      억제 술어가 필요한 것은 *"건넸다"* 가 아니라 *"**값을 얻었다**"* 인데, 차집합만 내는
      `give_exec_idle` 로는 그 절반(`ran`)을 꺼낼 수 없었다. 없다고 새로 짜면 궤적 해석이
      두 벌이 되고 조용히 갈린다([[67]] 실물 2건) — 그래서 **정본에 추가**한다.

    닫힌 술어뿐이다: 이름 동치·역할·`requestor`·에러 플래그(전부 프레임워크 형상·도메인 리터럴 0).
      - `given` = give 호출의 결과가 **오류가 아닌** 것(인계 성사)
      - `ran`   = 디스패처 경유(`user_call` + `discoverable_tool_name`) **또는** 손님의 직접
                  호출(`role == "user"` 또는 `requestor == "user"`) — 실측 궤적의 두 형태 모두.
    """
    given, id2name, ran = set(), {}, set()
    for m in messages or []:
        role = getattr(m, "role", None)
        for tc in (getattr(m, "tool_calls", None) or []):
            name = getattr(tc, "name", None)
            inner = str(_args_dict(tc).get("discoverable_tool_name") or "")
            if name == give_tool:
                id2name[getattr(tc, "id", None)] = inner
            elif name == user_call and inner:
                ran.add(inner)
            elif name and (role == "user"
                           or getattr(tc, "requestor", "assistant") == "user"):
                ran.add(name)
        if role == "tool" and not getattr(m, "error", False):
            n = id2name.get(getattr(m, "id", None))
            if n:
                given.add(n)
    return given, ran


def user_tool_value_ready(messages, give_tool, user_call):
    """**값을 이미 얻은** 손님-측 도구 집합 = `give_exec_state(...)` 의 `ran` (닫힌 술어·순수함수).

    ★A9 / OL-46 (2026-08-22 · t7336 마스터 §6.1 A9 · §5.7): F8(`T2_ARG_PRODUCERS`)의 억제
      술어 정본. 구판(`t2_prekb_patch` 의 `_seen_tools`)은 **이름이 등장했는가**를 봤다 —
      메시지의 모든 tool_call 이름 + 인자 JSON 문자열에서 뽑은 `[a-z0-9_]+` 토막 + 문자열
      인자값 전부가 그 집합에 들어간다. 그래서 생산자 도구를 *건네기만* 해도(give 인자에
      이름이 실린다) 넛지가 영구 침묵했다. 주석이 적어 둔 의도는 *"이미 값을 얻음"* 인데
      구현은 *"이름이 등장함"* 이었고, 실측은 t7328 **7**·t7335 **5** 발화 → t7336 **0**
      (040#1 [84]/[86] 침묵·[S]).

    ⚠**건넸다 ≠ 값을 얻었다.** 건네고 손님이 실행하지 않았으면 값은 아직 없고, 그때 F8 이
      말하는 *"건네서 실행하게 하고 같은 도구를 재시도하라"* 는 여전히 참이다.
    ⚠이 함수는 실행을 **시도**했는가만 본다(성공 여부는 안 본다) — `ran` 정의 그대로다.
      더 좁히면(성공만 인정) F8 이 더 자주 울리므로, 억제 쪽으로 안전한 현행 정의를 쓴다.
    """
    return give_exec_state(messages, give_tool, user_call)[1]


def _regen_last_user(messages):
    for m in reversed(messages):
        if getattr(m, "role", None) == "user" and getattr(m, "content", None):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return None


def _regen_transfer_sent(messages, notice_text):
    # ★C213/G1: 공용 정규화 술어로 일원화(032 [S]).
    if not notice_text:
        return None
    from gate_interpreter import notice_sent_in
    texts = [getattr(m, "content", None) for m in messages
             if getattr(m, "role", None) == "assistant"]
    return notice_sent_in(texts, notice_text)


def _denied_calls(am, gate, last_user, transfer_sent):
    """am의 assistant tool_calls 중 gate-deny 되는 것 = [(tc, gid, why)]."""
    out = []
    for tc in (getattr(am, "tool_calls", None) or []):
        if getattr(tc, "requestor", "assistant") != "assistant":
            continue
        ok, gid, why = gate.check(getattr(tc, "name", "") or "", _args_dict(tc),
                                  last_user_msg=last_user, transfer_msg_sent=transfer_sent)
        if not ok:
            out.append((tc, gid, why))
    return out


def _dedup_cache_safe(orch, name):
    """★READ_DEDUP 캐시 가부 (2026-07-20 e2e10 038 크래시·§2at·순수함수=테스트 공유).
    근본: 우리 write-술어(_is_effective_write)와 **tau2 replay의 mutating-술어**가 불일치 —
    unlock은 우리에겐 procedural(non-write)이라 캐시됐는데 replay는 mutating으로 **재실행** →
    "Tool unlocked..." ≠ "[DUPLICATE-READ]" stub → sim 무효(실측 038 t0). 캐시 가부의 정본을
    tau2 술어로: env가 mutating으로 보는 도구는 **캐시 금지**(stub이 히스토리에 남으면 replay 불일치).
    판정 불가(env 부재/예외)=False(캐시 안 함·안전측). 도메인 리터럴 0."""
    env = getattr(orch, "environment", None)
    try:
        return env is not None and not env._is_mutating_tool(name)
    except Exception:
        return False


def _budget_tick(agent):
    """R1: 차단 turn마다 orchestrator.num_errors++ → too_many_errors 예산 동일(best-of-K 방지)."""
    orch = getattr(agent, "_t2_orch", None)
    if orch is not None:
        try:
            orch.num_errors = getattr(orch, "num_errors", 0) + 1
        except Exception:
            pass


def _install_overflow_guard():
    """★컨텍스트 초과 우아한 종료 (2026-07-20·023 진단·§2ah). 하네스-일반(도메인 무관·[[05]] 3질문 NO).
    문제: full_duplex `step()`이 `ContextWindowExceededError`를 안 잡아 예외가 러너까지 전파→sim 전체가
      `infrastructure_error`(무효·unscored·0 msg)로 **소실**. 023이 게이트스택 regen+에이전트 루프 누적으로
      46089>vLLM 44672 초과서 실측(§2ah). tau2엔 `CONTEXT_WINDOW_EXCEEDED` 종료사유가 **정의만 되고 미배선** —
      그 의도된 처리를 구현.
    처방: `step()`을 래핑해 overflow를 잡고 done=True + reason=CONTEXT_WINDOW_EXCEEDED로 정상 종료 →
      run 루프가 done 감지 → finalize → **이미 기록된 부분 tick으로 reward 계산(scored)**. 비수렴 궤적은
      태스크 미완이므로 reward≈0 — **소실(제외) 대신 정직한 실패 계상**(평균 인플레 방지). crash 픽스(_reassemble)로
      부분 tick의 call↔result 쌍이 이미 유효 → replay 안전."""
    try:
        from tau2.orchestrator.orchestrator import BaseOrchestrator as _BO
        from tau2.orchestrator.orchestrator import Orchestrator as _TO
        from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
        from tau2.data_model.simulation import TerminationReason
        from litellm import ContextWindowExceededError
    except Exception as _e:
        print("[T2_OVERFLOW_GUARD] not installed (import): %r" % (_e,), file=sys.stderr, flush=True)
        return

    def _wrap_step(cls):
        """cls.step을 CWE-가드로 래핑. ★e2e9 097 정정(§2ao): 구판은 FullDuplex만 래핑 — text-모드
        (banking 런 실사용)는 BaseOrchestrator.step이라 CWE가 그대로 새어 sim 무효. **양쪽** 래핑
        (서브클래스 override별 개별·이중래핑 방지 마커)."""
        if getattr(cls, "_t2_overflow_wrapped", False) or "step" not in cls.__dict__:
            return
        _orig_step = cls.step

        def _guarded_step(self, *a, **kw):
            try:
                return _orig_step(self, *a, **kw)
            except ContextWindowExceededError as _ce:
                # 예외 전 기록 히스토리는 유효(부분 궤적). done+reason → run 루프 종료 → finalize 채점.
                self.done = True
                self.termination_reason = TerminationReason.CONTEXT_WINDOW_EXCEEDED
                print("[T2_OVERFLOW_GUARD] context window exceeded -> terminate sim as scored failure. %s"
                      % (str(_ce)[:140],), file=sys.stderr, flush=True)
                return None

        cls.step = _guarded_step
        cls._t2_overflow_wrapped = True

    _wrap_step(_BO)                       # BaseOrchestrator.step (기본 구현)
    _wrap_step(_TO)                       # ★Orchestrator.step **override**(text-모드 실사용·e2e10 038t2/097t1
    #                                       CWE 크래시 실측: base만 래핑해선 서브클래스 override가 우회 — 3번째 우회)
    _wrap_step(FullDuplexOrchestrator)    # speech 모드(자체 step override)
    print("[T2_OVERFLOW_GUARD] ON (base+text+full_duplex)", file=sys.stderr, flush=True)

    # ★(§2bd) 평가-시점 id-mismatch post-mortem 덤프 (로그 전용·예외는 그대로 재전파):
    #   rall4 050t2 재현 2회인데 T2_PAIRCHECK(에이전트-턴 검사)는 무발화 = 부패가 마지막
    #   에이전트 턴 이후(유저-측 종반 or 평가 입력 조립)에서 발생. set_state 실패 시
    #   message_history의 (idx·role·requestor·tool·id) 압축 시퀀스를 덤프해 지점 특정.
    try:
        from tau2.environment.environment import Environment as _PmEnv
        if not getattr(_PmEnv, "_t2_pairdump_wrapped", False):
            _orig_ss = _PmEnv.set_state

            # ★2026-08-03 긴급 수리(스모크 실측): v1.0.1이 `set_state(..., strict: bool = True)`로
            #   **인자를 추가**했는데 이 래퍼가 고정 시그니처라 평가 replay가 전부
            #   `TypeError: _ss2() got an unexpected keyword argument 'strict'`로 죽었다
            #   (task_010/023 실측 → Retry 3회 → sim 소실). 우리 래퍼는 **시그니처를 흉내내지 말고
            #   그대로 통과**시킨다 — 상류 시그니처 변경에 다시 물리지 않도록 *args/**kwargs.
            def _ss2(self, initialization_data, initialization_actions, message_history,
                     *_a, **_kw):
                # ★T2_PAIRFIX @평가-입력 (§2bi 정정): rall6 실측 — 라이브 PAIRCHECK 침묵 + 평가서만
                #   mismatch = 스왑은 tick→message 변환/직렬화 층에서 발생. 여기(원 검사 직전)서
                #   같은 id 집합·순서-스왑 블록을 호출 순서로 교정(내용 불변·의미론 no-op).
                if os.environ.get("T2_PAIRFIX") == "1" and message_history:
                    try:
                        _nfx = _pairfix(message_history)
                        if _nfx:
                            print("[T2_PAIRFIX] eval-input: reordered %d swapped block(s)" % _nfx,
                                  file=sys.stderr, flush=True)
                    except Exception:
                        pass
                try:
                    return _orig_ss(self, initialization_data, initialization_actions,
                                    message_history, *_a, **_kw)
                except ValueError as _ve:
                    if "id mismatch" in str(_ve) or "Tool message" in str(_ve):
                        print("[T2_PAIRDUMP] set_state failed: %s" % str(_ve)[:160],
                              file=sys.stderr, flush=True)
                        for _i, _m in enumerate(message_history or []):
                            _r = getattr(_m, "role", "?")
                            _rq = getattr(_m, "requestor", "")
                            _tcs = getattr(_m, "tool_calls", None) or []
                            if _tcs:
                                _d = ",".join("%s#%s" % (getattr(t, "name", "?"),
                                                         str(getattr(t, "id", ""))[-8:])
                                              for t in _tcs)
                                print("[T2_PAIRDUMP] %3d %s(%s) CALLS %s" % (_i, _r, _rq, _d),
                                      file=sys.stderr, flush=True)
                            elif _r == "tool":
                                print("[T2_PAIRDUMP] %3d tool(%s) id..%s err=%s"
                                      % (_i, _rq, str(getattr(_m, "id", ""))[-8:],
                                         getattr(_m, "error", False)),
                                      file=sys.stderr, flush=True)
                    raise

            _PmEnv.set_state = _ss2
            _PmEnv._t2_pairdump_wrapped = True
    except Exception as _e2:
        print("[T2_PAIRDUMP] not installed: %r" % (_e2,), file=sys.stderr, flush=True)


def _install_regen_exec():
    """slim _execute_tool_calls: 실행 + auth observe + read-augment(present/nested/calc). deny 없음
    (denied 호출은 생성-레벨서 이미 strip). augment=reads라 replay-safe(기존 apply와 동일 로직)."""
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    orig_exec = getattr(BaseOrchestrator, "_t2_orig_exec", None) or BaseOrchestrator._execute_tool_calls
    BaseOrchestrator._t2_orig_exec = orig_exec

    def exec_augment(self, tool_calls):
        # ★T2_READ_DEDUP (2026-07-19·022 CWE 포렌식): 동일 (name,args) read 재호출 = 실행 생략·
        #   스텁 반환(전체 덤프 재적재 방지). 029 실측: byte-identical KB 20K×2·shell 4.8K×2 = 컨텍스트 22% 낭비.
        #   신선도 보장: **실효 write**(_eff_tool_name+_is_effective_write·정본 술어·user 호출 포함) 실행 시
        #   캐시 전체 무효화 → 상태-의존 read(dispute status 등)는 write 후 항상 재실행. 정보손실 0(원 출력이
        #   위에 실재)·대형 출력만 캐시(T2_READ_DEDUP_MIN 기본 2000자)라 시각/소형 read는 대상 밖.
        stub_ids = set()
        dedup_on = (os.environ.get("T2_READ_DEDUP") == "1"
                    and not getattr(self, "_t2_dedup_bypass", False))
        # ★_t2_dedup_bypass (2026-07-20·smoke023c 포렌식·§2al): 격리 서브의 env 호출은 dedup 제외 —
        #   main이 이미 읽은 (name,args)에 stub("위 출력 참조")을 주면 **서브 문맥엔 '위'가 없어**
        #   빈손 날조를 유발(실측: fetch-iso가 60건 대신 3건 날조→오판정). 서브는 신선 실행·캐시 불변.
        if dedup_on:
            from tau2.data_model.message import ToolMessage as _TM
            _a2w = _a2_of(self)                  # C241 U1': 실효-write 술어용 도메인 A2
            cache = self._t2_read_cache = getattr(self, "_t2_read_cache", {})
            # ★loop-break (2026-07-23 C123·rall19 043 실측): 캐시는 대형 출력(min_len)만 저장하므로
            #   한 줄짜리 결과의 동일-read 반복은 스텁이 영원히 안 걸림 — 043.1서 shell grep
            #   'checking_account_id' **15회+ 동일 반복**(값은 DB getter에만 실재) 방치. 동일
            #   (name,args) read가 K회(기본 3) *실행*되면 크기 무관 스텁+redirect. write는 대상 밖·
            #   실효 write 시 카운터도 리셋(신선도 동일 규율).
            seen = self._t2_read_seen = getattr(self, "_t2_read_seen", {})
            loop_k = int(os.environ.get("T2_READ_DEDUP_LOOP_K", "3"))
            to_run, stubs = [], {}
            _dgset = getattr(getattr(self, "agent", None), "_t2_view_digested", None) or set()
            for tc in tool_calls:
                k = _call_key(tc)
                # ★§2bi: 뷰-압축으로 다이제스트된 출력의 재열람은 stub 금지(재실행 허용) —
                #   안 그러면 "위 출력 참조" stub이 다이제스트를 가리켜 재열람 탈출구가 막힘.
                # ★★T2_NO_DIGEST_REEXEC=1 (RUNAWAY_AXIS_REDESIGN §2a·2026-08-02): 이 규칙이
                #   022에서 **동일 23,291자 테이블을 3부 재유입**시켰다(문맥 25.9%). 전수 계수
                #   446회/208 sim/7,578,692자. 재열람 탈출구를 **재실행**으로 준 것이 원인이므로
                #   플래그 ON이면 면제를 끄고 stub을 유지한다(원문은 위에 이미 있다).
                if k in cache and cache.get(k) in _dgset \
                        and os.environ.get("T2_NO_DIGEST_REEXEC") != "1":
                    cache.pop(k, None)
                # ★replay 위생 확장(2026-08-02·qp32p2 026 R1 [S]): §2at 가드는 **캐시 삽입만** 막고
                #   loop-break(seen>=loop_k) 스텁은 안 거쳤다 → env가 mutating으로 보는 디스패처
                #   호출(call_discoverable_agent_tool)이 반복되자 스텁이 히스토리에 남았고, eval
                #   set_state가 재실행 실물과 비교해 sim 무효(실측: failed_setstate_1785632213670).
                #   ⇒ 스텁 발행 자체에 같은 가드를 건다 — replay가 재실행할 도구는 **절대 스텁 금지**
                #   (반복 비용은 실행으로 지불·측정 정합성이 우선=C208 ⓐ).
                if ((k in cache or seen.get(k, 0) >= loop_k)
                        and not _is_effective_write(_eff_tool_name(tc), _a2w)
                        and _dedup_cache_safe(self, getattr(tc, "name", "") or "")):
                    # ★2026-07-23 (050 flail 근본원인 확정·KB probe): 반복된 read가 KB/검색 도구면
                    #   redirect 힌트 추가 — 도메인-일반. 원인=에이전트가 discoverable 도구를 *함수명*으로
                    #   BM25 검색(점수 0.0·문서 산문엔 함수명 없음)→무한반복. plain-words로 돌림(closure 피드백이
                    #   이미 가진 사실을 flail 지점에 표면화). 배제=[[10]] 생성 무해·행동 게이트 아님.
                    #   ★C123: 키워드 검사를 이름+인자로 확장(043 실측: 도구명 'shell'·grep은 인자에).
                    _dn = ((getattr(tc, "name", "") or "")
                           + " " + str(getattr(tc, "arguments", "") or "")).lower()
                    _redir = ""
                    if "search" in _dn or "bm25" in _dn or "kb_" in _dn or "grep" in _dn:
                        _redir = (" Do NOT repeat this exact search. If you are looking up a discoverable "
                                  "tool, note that a bare function-name query matches no document text — "
                                  "search PLAIN WORDS describing the action/step (the everyday words a policy "
                                  "document would use), not the tool's function name. If you already have the "
                                  "information you need, proceed to the next step instead of searching again.")
                    # ★C194(2026-07-26·16건 정독 실측): 동일-문구 stub은 루프를 못 끊는다 — 041은
                    #   동일 KB 쿼리 15회+(stub 매번 발화·행동 불변)·020/027 5~6회·035/012 동형.
                    #   동일 입력→동일 출력 어트랙터라 같은 텍스트 반복 제시는 같은 선택을 재생산.
                    #   교정: ①반복 횟수를 문구에 넣어 매번 다른 텍스트 ②3회+ 시 행동-전환 지시
                    #   +error 채널 승격(피드백 채널 자체 변경). read 한정·도메인 리터럴 0.
                    _rep = self._t2_dup_rep = getattr(self, "_t2_dup_rep", {})
                    _rep[k] = _rep.get(k, 0) + 1
                    _n_rep = _rep[k]
                    # ★T2_REPEAT_GOV=1 (CONSOLIDATION §2a·L1 준수·기본 OFF): 반복 채널의 문구 조립·
                    #   순서를 거버너 한 곳으로. **판정 술어는 아래 레거시와 동일하게 여기서 계산**해
                    #   넘긴다(행동 불변·동등성은 test_repeat_gov 바이트 검정 + x45). OFF=레거시 그대로.
                    if os.environ.get("T2_REPEAT_GOV") == "1":
                        import t2_repeat_gov as _rg
                        _dig = (k in cache and cache.get(k) in _dgset
                                and os.environ.get("T2_NO_DIGEST_REEXEC") == "1")
                        _isrch = ("search" in _dn or "bm25" in _dn or "kb_" in _dn
                                  or "grep" in _dn)
                        try:
                            _capk = int(os.environ.get("T2_REPEAT_CAP", "0") or 0)
                        except Exception:
                            _capk = 0
                        _gc, _gerr, _gcap = _rg.ladder(getattr(tc, "name", "") or "",
                                                       _n_rep, _isrch, _dig, _capk)
                        if _gcap:
                            _glog = getattr(self, "_t2_repeat_log", None)
                            if _glog is None:
                                _glog = self._t2_repeat_log = []
                            _glog.append((getattr(tc, "name", ""), _n_rep))
                        from t2_lever_beat import beat as _gbeat
                        _gbeat("T2_REPEAT_GOV")
                        stubs[getattr(tc, "id", None)] = _TM(
                            id=tc.id, role="tool",
                            requestor=getattr(tc, "requestor", "assistant"),
                            error=_gerr, content=_gc)
                        stub_ids.add(getattr(tc, "id", None))
                        self._t2_read_dedup = getattr(self, "_t2_read_dedup", 0) + 1
                        print("[T2_READ_DEDUP] stub tool=%s" % getattr(tc, "name", None),
                              file=sys.stderr, flush=True)
                        continue
                    _esc = ""
                    # ★T2_REPEAT_CAP=K (RUNAWAY §2c·x35 ① 사전계측: K=8에서 과차단 하한 0·표적 39 sim).
                    #   C194 esc(_n_rep>=3) **위에** 얹히는 3번째 강도라 K>3이어야 귀속이 깨지지 않는다.
                    #   실행 억제는 이미 stub이 하고 있으므로 여기서 더하는 것은 **3층 보고**다.
                    try:
                        _cap = int(os.environ.get("T2_REPEAT_CAP", "0") or 0)
                    except Exception:
                        _cap = 0
                    if _cap > 3 and _n_rep >= _cap:
                        _log = getattr(self, "_t2_repeat_log", None)
                        if _log is None:
                            _log = self._t2_repeat_log = []
                        _log.append((getattr(tc, "name", ""), _n_rep))
                        try:
                            from t2_lever_beat import beat as _cbeat
                            _cbeat("T2_REPEAT_CAP", "%s x%d" % (getattr(tc, "name", ""), _n_rep))
                        except Exception:
                            pass
                        _esc = (" [REPEAT-CAP] This identical call has now been issued %d times and is "
                                "no longer being executed. Stop this line of action: state to the "
                                "customer what you could not resolve, or take a DIFFERENT action. "
                                "This has been recorded as an unresolved blocker." % _n_rep)
                    elif _n_rep >= 3:
                        _esc = (" You have now issued this IDENTICAL call %d times and the result "
                                "has not changed once — repeating it again cannot produce new "
                                "information. Change what you do: use DIFFERENT search words, or "
                                "act on the information you already have, or ask the customer. Do "
                                "not issue this same call again." % _n_rep)
                    stubs[getattr(tc, "id", None)] = _TM(
                        id=tc.id, role="tool",
                        requestor=getattr(tc, "requestor", "assistant"), error=(_n_rep >= 3),
                        content=(
                            # ★상쇄 조정(2026-08-02·5축 동시-ON 감사): NO_DIGEST_REEXEC가 다이제스트된
                            #   재열람도 스텁으로 막는데, 기존 문구는 "full output is shown above"라고
                            #   말한다 — 뷰에는 다이제스트만 남아 **허위 문구**(D-계열 재생산)가 된다.
                            #   다이제스트 케이스는 정직한 대체 문구 + byref 탈출구(@last:)를 준다.
                            ("[DUPLICATE-READ] This exact call was already executed earlier; its "
                             "output was COMPACTED from view to save space and has not changed. Do "
                             "NOT re-run it. If a tool needs that data, pass it BY REFERENCE as "
                             "@last:%s instead of re-reading." % (getattr(tc, "name", "") or "")
                             if (k in cache and cache.get(k) in _dgset
                                 and os.environ.get("T2_NO_DIGEST_REEXEC") == "1") else
                             "[DUPLICATE-READ] This exact call (same tool, same arguments) was "
                             "already executed earlier in this conversation; its full output is "
                             "shown above and has not changed. Refer to that output instead of "
                             "re-reading.") + _redir + _esc))
                    stub_ids.add(getattr(tc, "id", None))
                    self._t2_read_dedup = getattr(self, "_t2_read_dedup", 0) + 1
                    print("[T2_READ_DEDUP] stub tool=%s" % getattr(tc, "name", None),
                          file=sys.stderr, flush=True)
                    _lbeat("T2_READ_DEDUP", orch=self, target=getattr(tc, "name", None),
                           fact="this exact read was already executed in this conversation")
                else:
                    # ★P5-3(C208①·DAY5_PRESCRIPTIONS §P5-3·T2_READ_NEARDUP=1·기본 OFF): 근사-중복
                    #   질의 안내 — 018/028/029 [S]: "get credit card transactions"↔"tool to get …"↔
                    #   "…by user" 재표현 5연발(각 15~23k자)이 exact-dedup을 전부 통과해 창 60% 소진.
                    #   판정=정규화 토큰집합 Jaccard(도메인 무관·검색류 read만)·안내 스텁(원 출력은
                    #   위에 실재). 오탐(정당 질의-정련) 리스크 → 기본 OFF·격리 arm 계측 후 승격.
                    _nd_hit = None
                    if (os.environ.get("T2_READ_NEARDUP") == "1"
                            and not _is_effective_write(_eff_tool_name(tc), _a2w)):
                        _dn2 = ((getattr(tc, "name", "") or "")
                                + " " + str(getattr(tc, "arguments", "") or "")).lower()
                        if "search" in _dn2 or "bm25" in _dn2 or "kb_" in _dn2:
                            _stop = {"the", "a", "an", "for", "to", "of", "in", "on", "how",
                                     "tool", "get", "and", "or", "with", "by", "is", "do"}
                            _tk = {w for w in re.findall(r"[a-z0-9_]+", _dn2) if w not in _stop}
                            _hist = self._t2_nd_hist = getattr(self, "_t2_nd_hist", [])
                            for _pk, _ptk in _hist:
                                if _pk != getattr(tc, "name", None) or not (_tk | _ptk):
                                    continue
                                _j = len(_tk & _ptk) / float(len(_tk | _ptk))
                                if _j >= float(os.environ.get("T2_READ_NEARDUP_J", "0.8")):
                                    _nd_hit = sorted((_tk ^ _ptk))[:6]
                                    break
                            if _nd_hit is None:
                                _hist.append((getattr(tc, "name", None), _tk))
                    if _nd_hit is not None:
                        stubs[getattr(tc, "id", None)] = _TM(
                            id=tc.id, role="tool",
                            requestor=getattr(tc, "requestor", "assistant"), error=False,
                            content=("[NEAR-DUPLICATE-READ] This query is nearly identical to an "
                                     "earlier one in this conversation (it differs only in: %s); "
                                     "the earlier output is shown above and this rephrasing will "
                                     "return largely the same documents. Refine with genuinely NEW "
                                     "terms, or proceed with the information you already have."
                                     % (", ".join(_nd_hit) or "(word order)")))
                        stub_ids.add(getattr(tc, "id", None))
                        print("[T2_READ_NEARDUP] stub tool=%s diff=%s"
                              % (getattr(tc, "name", None), _nd_hit),
                              file=sys.stderr, flush=True)
                        continue
                    if not _is_effective_write(_eff_tool_name(tc), _a2w):
                        seen[k] = seen.get(k, 0) + 1     # C123: 실행되는 read만 계수
                    to_run.append(tc)
            ran = orig_exec(self, to_run) if to_run else []
            _rby = {getattr(r, "id", None): r for r in (ran or [])}
            # ★크래시 픽스(2026-07-20·023/031 infrastructure_error): 결과는 tool_calls와 **1:1·순서**
            #   보장. 안 그러면 full-duplex tick의 agent_tool_calls↔agent_tool_results 쌍이 깨져
            #   eval replay(`environment.get_actions_from_messages`)가 "Tool call id mismatch"로 크래시.
            #   구판은 id 매칭 실패 시 None을 **드롭**해 results가 짧아졌다(비결정론=하위 레이어 id/순서 의존).
            #   id 매칭 우선·id없는 결과만 위치폴백·그래도 없으면 에러 ToolMessage로 채운다(드롭 금지).
            _idless = iter([r for r in (ran or []) if getattr(r, "id", None) is None])
            results = []
            for tc in tool_calls:
                _tid = getattr(tc, "id", None)
                r = stubs.get(_tid) if _tid in stub_ids else _rby.get(_tid)
                if r is None:
                    r = next(_idless, None)
                if r is None:
                    r = _TM(id=_tid, role="tool", requestor=getattr(tc, "requestor", "assistant"),
                            error=True, content="(no result returned for this tool call)")
                results.append(r)
            # ★★축-레버 표면화 (FAILURE_AXES_REDESIGN / RUNAWAY_AXIS_REDESIGN·2026-08-02).
            #   전부 플래그 기본 OFF·표면화만(거부/값생성 0)·도메인 리터럴 0(이름=A2 레지스트리·문구=A2).
            try:
                _axis_surface(self, tool_calls, results)
            except Exception as _e:                      # 레버 실패가 런을 죽이지 않는다
                print("[T2_AXIS] 표면화 실패(무시): %r" % (_e,), file=sys.stderr, flush=True)
            min_len = int(os.environ.get("T2_READ_DEDUP_MIN", "2000"))
            for tc in to_run:
                out = _rby.get(getattr(tc, "id", None))
                if out is None:
                    continue
                if _is_effective_write(_eff_tool_name(tc), _a2w):
                    if not getattr(out, "error", False):
                        cache.clear()  # 세상이 바뀜 → 이전 read 신선도 보장 불가
                        seen.clear()   # C123: loop-break 계수도 동일 규율로 리셋
                elif (not getattr(out, "error", False) and len(_content_str(out) or "") >= min_len
                      and _dedup_cache_safe(self, getattr(tc, "name", "") or "")):
                    # ★§2at: env-mutating(unlock 등)은 캐시 금지 — stub이 히스토리에 남으면 replay 불일치
                    # ★§2bi: 값=결과 메시지 id(뷰-압축 다이제스트 면제 판정용·구판 True와 truthy 동일)
                    cache[_call_key(tc)] = getattr(out, "id", True)
        else:
            results = orig_exec(self, tool_calls)
        env = getattr(self, "environment", None)
        a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if a2 is None:
            return results
        gate = getattr(getattr(self, "agent", None), "_t2_gate", None)
        by_id = {getattr(r, "id", None): r for r in (results or [])
                 if getattr(r, "id", None) not in stub_ids}
        auth_tools = a2["_auth_tools"]
        present_on = os.environ.get("T2_PRESENT_READS") == "1"
        g6 = next((g for g in a2["gates"] if g.get("kind") == "select_confirm"), None) if present_on else None
        nested_specs = (a2.get("present_specs") or []) if os.environ.get("T2_PRESENT_NESTED") == "1" else []
        calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC") == "1" else []
        # ★C5 이관 — 원장 산수 (T2_LEDGER=1·2026-08-07). **여기가 무조건 도는 자리다**:
        #   앞의 read-augment 구간은 통째로 `if dedup_on:` 안이라 `T2_READ_DEDUP`가 꺼진 arm에서는
        #   실행되지 않는다. 그것을 모르고 그 안에 배선한 탓에 계측 probe까지 여섯 번 무음이었다
        #   — "레버가 안 도는 것"이 아니라 "레버를 담은 블록이 꺼진 것"이었다.
        #   분담([[59]]): 전사는 모델(A2 formalize_prompt), 엔진은 그 위의 산수만.
        global _T2_LEDGER_PROBED
        if not _T2_LEDGER_PROBED:
            _T2_LEDGER_PROBED = True
            print("[T2_LEDGER] probe: flag=%r a2=%s metrics=%d tools=%s"
                  % (os.environ.get("T2_LEDGER"), "None" if a2 is None else "dict",
                     len((a2 or {}).get("ledger_metrics") or []),
                     ",".join(sorted({_eff_tool_name(_t) for _t in tool_calls}))),
                  file=sys.stderr, flush=True)
        if os.environ.get("T2_LEDGER") == "1":
            try:
                import t2_offload as _OFF
                import t2_ledger as _LG
                # ★잠복 버그(2026-08-07·`T2_LEDGER`를 처음 켜자 드러남): `la`(LLM 어댑터)는
                #   `apply_*_regen` 세 곳에서만 import되고 **이 훅 스코프엔 없다** → `NameError`.
                #   死배선이라 라이브에서 한 번도 실행된 적이 없어 아무도 못 봤다. 지역 import로 닫는다.
                #   ★2차(win_20260807i 라이브): `la`만 닫고 `UserMessage`를 빠뜨려 같은 자리에서
                #   `NameError("name 'UserMessage' is not defined")` 4회 — 한 훅에서 두 이름이
                #   빠져 있었는데 첫 예외가 둘째를 가렸다. 둘 다 지역 import로 닫는다.
                import tau2.agent.llm_agent as la
                from tau2.data_model.message import UserMessage
                # ★3차(같은 자리): 이 훅의 `self`는 **오케스트레이터**다(위 `getattr(self,"agent")`가
                #   그 증거). `formalize_*`는 `agent.llm`·`agent.llm_args`를 쓰므로 오케스트레이터를
                #   넘기면 AttributeError → 내부 `except`가 삼켜 **[]**를 돌려주고, 로그엔
                #   "transcription returned 0 rows"만 남아 *모델이 전사에 실패한 것처럼 보인다*.
                #   침묵이 오진을 만드는 형태라 이름을 명시적으로 가른다.
                _lgagent = getattr(self, "agent", None)
                if _lgagent is None:
                    raise RuntimeError("orchestrator has no .agent (ledger formalize needs the LLM agent)")
                for _tc in tool_calls:
                    _o = by_id.get(getattr(_tc, "id", None))
                    if _o is None or getattr(_o, "error", False):
                        continue
                    for _ls in _LG.specs_for(a2, _eff_tool_name(_tc)):
                        _rows = _LG.formalize_rows(_lgagent, la, UserMessage, _content_str(_o), _ls)
                        if not _rows:
                            print("[T2_LEDGER] %s: spec matched, transcription returned 0 rows"
                                  % _eff_tool_name(_tc), file=sys.stderr, flush=True)
                            continue
                        _tx = [_content_str(_m) for _m in self.get_messages()
                               if getattr(_m, "role", None) in ("tool", "user")]
                        _blk = _OFF.ledger_facts(_rows, _ls,
                                                 now=_LG.formalize_now(_lgagent, la, UserMessage, _tx, _ls))
                        if _blk:
                            # ★비커밋 채널로 (2026-08-07·led_j 라이브 set_state 실패 2회).
                            #   구판은 `_o.content`에 직접 이어 붙였고, 그것이 **궤적에 커밋된다**.
                            #   replay(`environment.set_state`)는 non-mutating 도구를 건너뛰므로
                            #   직접 read 증강은 무사했지만, 이 도메인의 발견형 통로
                            #   `call_discoverable_agent_tool`은 `@is_tool(ToolType.WRITE)`로
                            #   **선언**돼 있어(=`mutates_state=True`) 안쪽이 read여도 재실행·바이트
                            #   대조를 받는다. 그리고 replay는 **환경**의 `get_response`를 부르지
                            #   우리 훅을 타지 않는다 — 무엇을 붙이든 갈린다(블록을 결정론으로
                            #   만들어도 소용없다). 실측: 실패 2건의 도구 이름이 둘 다 그 통로였고,
                            #   같은 런에서 직접 read(`get_referrals_by_user`) 증강 sim은 통과했다.
                            #   ⇒ `_t2_view_fb`로 큐잉한다. 그 기구의 주석이 이미 같은 말을 한다 —
                            #   *"replay-비교 대상 도구의 피드백 뷰-채널 소비 · 작업버퍼에만 주입"*.
                            #   대가: 도구 출력 옆에 영구히 남지 않고 생성-뷰에 N회 노출된다
                            #   (1회는 무시된다는 실측이 있어 기본 3회·`T2_LEDGER_VIEW_KEEP`).
                            # ★대조에 쓸 **피연산자만** 여기 남긴다 (2026-08-08·lim_n 라이브).
                            #   상한/문턱 비교를 이 자리에서 하던 1차판은 **구조적으로 이른 시점**이라
                            #   한 번도 발화하지 못했다: 원장 read는 턴 10~12에 일어나는데 상품 문서는
                            #   그보다 뒤에 회수된다 — 상한이 문맥에 도착하기 전에 물어본 것이다.
                            #   그래서 비교는 **결정점**(제출 요구가 나가는 자리)으로 옮겼고, 여기서는
                            #   엔진이 전사한 수(누계·경과일)를 넘겨줄 뿐이다. 그 수는 지금 확정된다.
                            # ★선언**마다** 따로 보관한다 (2026-08-08·dp_p 계측이 잡음).
                            #   두 프롬프트는 서로 다른 선언에 산다 — `limit_prompt`는 추천 원장,
                            #   `threshold_prompt`는 계좌 선언. 구판은 `_t2_ledger_spec` 하나만
                            #   두어 마지막 호출의 선언으로 덮었고, 그래서 문턱 추출이 **0회**였다
                            #   (`min_days: model gave` 로그가 한 번도 안 찍혔다).
                            try:
                                _now0 = getattr(_lgagent, "_t2_ledger_now", None)
                                _ops = dict(getattr(_lgagent, "_t2_ledger_ops", None) or {})
                                _ops[str(_ls.get("trigger_tool"))] = {
                                    "spec": _ls,
                                    "tally": _LG.window_and_tally(_rows, _ls, now=_now0)[2],
                                    # ★행 자체도 둔다 (C378): 누계는 그룹 축으로 뭉갠 수라
                                    #   **행마다 다른 상태**를 되살릴 수 없다. 결정점에서
                                    #   상태별로 세려면 전사된 행이 그대로 있어야 한다.
                                    "rows": list(_rows or ()),
                                    "days": _LG.earliest_age(_rows, _ls, now=_now0)[1]}
                                _lgagent._t2_ledger_ops = _ops
                            except Exception as _se:
                                print("[T2_LEDGER] operand stash skipped: %r" % (_se,),
                                      file=sys.stderr, flush=True)
                            _q = list(getattr(_lgagent, "_t2_view_fb", None) or [])
                            _q.append([_blk.strip(),
                                       int(os.environ.get("T2_LEDGER_VIEW_KEEP", "3"))])
                            _lgagent._t2_view_fb = _q
                            print("[T2_LEDGER] %s rows=%d queued to view (non-committed)"
                                  % (_eff_tool_name(_tc), len(_rows)),
                                  file=sys.stderr, flush=True)
            except Exception as _e12:
                print("[T2_LEDGER] error (no-op): %r" % (_e12,), file=sys.stderr, flush=True)
        for tc in tool_calls:
            out = by_id.get(getattr(tc, "id", None))
            if out is None or getattr(out, "error", False):
                continue
            name = getattr(tc, "name", None)
            if gate is not None and name in obs_tools_g(gate):
                gate.observe(name, _args_dict(tc), _content_str(out))
            if present_on and g6 is not None and gate is not None and name == g6.get("user_producer"):
                uid = _args_dict(tc).get(g6.get("user_id_arg", "user_id"))
                summ = candidate_summary(gate.resolvers, g6, uid)
                if summ:
                    try:
                        out.content = _content_str(out) + summ
                    except Exception:
                        pass
            _rec = _parse_json(_content_str(out)) if (nested_specs or calc_specs) else None
            if nested_specs and _rec is not None:
                spec = next((s for s in nested_specs if s.get("trigger_tool") == name), None)
                if spec is not None:
                    summ = nested_candidate_summary(_rec, spec)
                    if summ:
                        try:
                            out.content = _content_str(out) + summ
                        except Exception:
                            pass
            if calc_specs and _rec is not None:
                cs = [s for s in calc_specs if s.get("trigger_tool") == name]
                if cs:
                    facts = compute_facts(_rec, cs)
                    if facts:
                        try:
                            out.content = _content_str(out) + facts
                        except Exception:
                            pass
        return results

    BaseOrchestrator._execute_tool_calls = exec_augment


def apply_gate_regen(max_regen=1):
    """replay-safe 게이트 (apply() 대체·A/B 위해 apply()는 보존). T2_GATE_KINDS 필터 동일 지원."""
    import tau2.agent.llm_agent as la
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    from tau2.data_model.message import ToolMessage, MultiToolMessage

    # (1) orchestrator.__init__ → env-bound gate를 agent에 주입 (per-sim)
    orig_init = BaseOrchestrator.__init__

    def init_inject(self, *a, **kw):
        orig_init(self, *a, **kw)
        env = getattr(self, "environment", None)
        ag = getattr(self, "agent", None)
        a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if a2 is not None and ag is not None and hasattr(ag, "_generate_next_message"):
            _kinds = os.environ.get("T2_GATE_KINDS")
            gl = a2["gates"]
            if _kinds:
                allow = {k.strip() for k in _kinds.split(",") if k.strip()}
                gl = [g for g in a2["gates"] if g.get("kind") in allow]
            ag._t2_gate = GateInterpreter(gl, resolvers=resolvers_from_env(env))
            ag._t2_a2 = a2
            ag._t2_orch = self
            ag._t2_max_regen = max_regen

    BaseOrchestrator.__init__ = init_inject

    # (2) LLMAgent._generate_next_message → gate deny + regen + R8 종단
    orig_gen = LLMAgent._generate_next_message

    def gen_gated(self, message, state):
        try:
            from t2_lever_beat import set_sim_from as _ssf
            _ssf(self)
        except Exception:
            pass
        gate = getattr(self, "_t2_gate", None)
        if gate is None:
            return orig_gen(self, message, state)
        a2 = self._t2_a2
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        _rebuild_gate_state(gate, a2, state.messages)
        last_user = _regen_last_user(state.messages)
        # ★NOTICE-PERGATE 커링 (apply()와 동형)
        transfer_sent = lambda text: _regen_transfer_sent(state.messages, text)  # noqa: E731
        base = state.system_messages + state.messages
        am = la.generate(model=self.llm, tools=self.tools, messages=base,
                         call_name="agent_response", **self.llm_args)
        n, max_regen_ = 0, getattr(self, "_t2_max_regen", 1)
        while n < max_regen_:
            denied = _denied_calls(am, gate, last_user, transfer_sent)
            if not denied:
                break
            _budget_tick(self)
            dids = {id(tc): (gid, why) for tc, gid, why in denied}
            # ★§S-2 3층 (2026-09-01): **빈 am 은 재생성 프롬프트에 싣지 않는다.**
            #   실측(095·모드 B): 빈 어시스턴트 메시지가 이 버퍼에 실려 나가
            #   `llm_utils.py:234 assert has_content_or_tool_calls` 로 죽고 태스크가 재시작됐다.
            fb = [] if _t2_msg_empty(am) else [am]
            for c in (am.tool_calls or []):
                if id(c) in dids:
                    gid, why = dids[id(c)]
                    content = f"Error: [POLICY GATE {gid}] {why}"
                else:
                    _flagged = next((t for t in (am.tool_calls or []) if id(t) in dids), None)
                    content = _sibling_wait("POLICY GATE", _flagged, "the policy gate")
                fb.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                      error=True, content=content))
            am = la.generate(model=self.llm, tools=self.tools, messages=base + fb,
                             call_name="agent_response_gateregen", **self.llm_args)
            n += 1
        # R8 종단: K 소진 후에도 deny → 차단 mutating 호출 제거(히스토리 replay-clean 보장)
        # ★예산(R1·리뷰⚠️2): exhaustion turn은 위 루프서 이미 1 tick 소비 → 여기서 재-tick 안 함
        #   = "차단 turn당 1 error"로 현 게이트와 동일 예산압박(best-of-K도 double-charge도 아님).
        denied = _denied_calls(am, gate, last_user, transfer_sent)
        if denied:
            denied_ids = {tc.id for tc, _, _ in denied}
            kept = [tc for tc in (am.tool_calls or []) if tc.id not in denied_ids]
            am.tool_calls = kept or None
            # ★A15/OL-55 (2026-08-22): 사유는 **단어 경계**서 자르고, 빈 본문이면 노트를 본문
            #   전체로 커밋하지 않는다 — 모델에게 본문을 다시 받는다(`_commit_block_note`).
            note = "; ".join(f"[{gid}] {_trunc_reason(why)}" for _, gid, why in denied)

            def _bn_regen(_ask, _base=base, _am=am):
                _kw = dict(self.llm_args)
                _kw.pop("tools", None)
                _r = la.generate(model=self.llm, tools=None,
                                 messages=_base + [_um(_ask)],
                                 call_name="agent_blocknote_body", **_kw)
                return getattr(_r, "content", "") or ""
            _commit_block_note(am, note, regen=_bn_regen)
        return am

    LLMAgent._generate_next_message = gen_gated

    # (3) slim exec: 실행 + auth observe + read-augment (deny 없음)
    _install_regen_exec()
    return orig_gen


# ═══════════════════════════════════════════════════════════════════════════
# ★E-COMP 통합 생성-레벨 검증 체인 (RETAIL_PASS_COMPOSITION_DESIGN_2026_07_10 §2·리뷰 반영)
#   = 게이트(replay-safe regen) + provenance 검증기 + (선택) DISAMB 를 한 패치로.
#   예산 semantics는 두 GO arm을 *그대로 승계* (리뷰 블로킹1 — 귀속 오염 방지):
#     게이트: deny 피드백 라운드 최대 1회·그 라운드만 num_errors++ (apply_gate_regen K=1 동일)
#             잔존 deny = R8 strip(재과금 없음·replay-safe)
#     prov  : 무과금·재발화 최대 max_prov_retries=4 (apply_provenance_regen=C53 동일)
#     DISAMB: 1회 재확인·★채택 전 게이트 재검사 — deny면 원 am 유지 (리뷰 블로킹2)
#   present/autofetch/GROUND 미지원(C34 폐기·scope 밖). 도메인-일반: 게이트·힌트 전부 A2-구동.
# ═══════════════════════════════════════════════════════════════════════════


def _compact_view(messages, keep_recent=6, min_len=800, min_total=60000, msg_cap=0):
    """★뷰-압축 (T2_VIEW_COMPACT=1·2026-07-21 §2bi·097 컨텍스트 레버·사용자 승인 기본안).
    원리: **커밋 히스토리는 불변**(replay-불변식 자동 충족·게이트/관문은 원문 대조 유지) — LLM
    생성-시점 프롬프트 뷰에서만 오래된 벌크 tool 출력을 기계적 다이제스트(head+tail 절단)로 대체.
    read 액션의 주체는 모델로 유지(서브-이관 변형은 [[05]]③ autofetch-류로 기각·§2bi 문답).
    - 대상: role=tool·비에러·min_len 초과·최근 keep_recent개 제외. 전체 뷰가 min_total 미만이면 무개입.
    - 다이제스트=순수 절단(head 300+tail 150)+안내문 — 엔진의 내용 추출/합성 0([[03b]]).
    - 반환: (뷰 리스트, 다이제스트된 ToolMessage id 집합) — id는 READ_DEDUP 면제(재열람 탈출구)용.
    ★P5(C208①⑤·DAY5_PRESCRIPTIONS §P5·2026-07-28):
    - min_total 기본 120,000→60,000자 — 구 문턱은 사망선(≈40k tok) 위라 32sim 중 6회만 발동.
    - msg_cap(신설·>0일 때): **최신 배치**(마지막 assistant 이후의 전 tool 출력 — 리뷰 필수1:
      멀티-콜 턴은 출력들이 함께 커밋되고 다음 생성이 첫 노출이라 "최신 1개"면 미열람 절단)를
      제외한 비에러 tool 출력이 cap 초과면 **총량과 무관하게** 다이제스트. 모델은 도착 턴에
      전문을 봤고 이후 턴부터 다이제스트(read 주체=모델 유지)."""
    msgs = list(messages)
    last_a = max([i for i, m in enumerate(msgs)
                  if getattr(m, "role", None) == "assistant"] or [-1])
    batch = {i for i in range(last_a + 1, len(msgs))
             if getattr(msgs[i], "role", None) == "tool"}
    total = sum(len(str(getattr(m, "content", "") or "")) for m in msgs)
    if total < int(min_total) and not msg_cap:
        return msgs, set()
    tool_idx = [i for i, m in enumerate(msgs) if getattr(m, "role", None) == "tool"]
    keep = (set(tool_idx[-int(keep_recent):]) if keep_recent else set()) | batch
    out, digested = [], set()
    for i, m in enumerate(msgs):
        c = getattr(m, "content", None)
        _is_tool = (getattr(m, "role", None) == "tool" and isinstance(c, str)
                    and not getattr(m, "error", False))
        _hit = False
        if _is_tool and i not in batch:
            if msg_cap and len(c) > int(msg_cap):
                _hit = True                                   # P5-2 per-메시지 캡
            elif total >= int(min_total) and i not in keep and len(c) > int(min_len):
                _hit = True                                   # 기존 총량-문턱 경로
        if _hit:
            d = (c[:300] + "\n...[view digest: %d chars total. The FULL output was recorded "
                 "earlier in this conversation; re-call the same tool if you need the "
                 "details again.]...\n" % len(c) + c[-150:])
            try:
                m2 = m.model_copy(update={"content": d})
            except Exception:
                import copy as _cp
                m2 = _cp.copy(m)
                try:
                    m2.content = d
                except Exception:
                    m2 = m
            if getattr(m2, "content", None) == d:
                digested.add(getattr(m, "id", None))
                out.append(m2)
            else:
                out.append(m)
        else:
            out.append(m)
    return out, digested


def _annotate_view(messages, specs):
    """★뷰 필드-주석 (T2_VIEW_ANNOTATE=1·2026-07-22 §2bs·rall10 054 실측).
    원리: 커밋 히스토리는 불변(_compact_view와 동일·replay-safe) — 생성-시점 뷰의 role=tool
    비에러 출력에 A2 선언 부분문자열 집합(contains 전부 공존)이 보이면 A2 주석(note)을 append.
    054: 교체주문이 기록한 status: CLOSED(舊카드)·account_status: ACTIVE 병존 뷰를 "계좌 폐쇄"로
    오독→CLI 전면 거부. KB(replacements_003)=카드≠계좌 명시 — 주석 텍스트 전부 A2(KB 인용).
    엔진=부분문자열 공존 검사+append만(내용 추출/합성 0·[[03b]]·도메인 리터럴 0).
    반환: (뷰 리스트, 주석된 메시지 수)."""
    out, n = [], 0
    for m in messages:
        c = getattr(m, "content", None)
        if (getattr(m, "role", None) == "tool" and isinstance(c, str)
                and not getattr(m, "error", False)):
            adds = [str(sp.get("note")) for sp in specs
                    if sp.get("note") and (sp.get("contains") or [])
                    and all(s in c for s in (sp.get("contains") or []))
                    and str(sp.get("note")) not in c]
            if adds:
                d = c + "\n" + "\n".join(adds)
                try:
                    m2 = m.model_copy(update={"content": d})
                except Exception:
                    import copy as _cp
                    m2 = _cp.copy(m)
                    try:
                        m2.content = d
                    except Exception:
                        m2 = m
                if getattr(m2, "content", None) == d:
                    n += 1
                    out.append(m2)
                    continue
        out.append(m)
    return out, n


def _pairfix(messages):
    """★커밋-시점 pairing 교정 (T2_PAIRFIX=1·2026-07-21 §2bi·rall6 054t2 PAIRDUMP 실측).
    실측 부패 시그니처: 다중 동일-도구 read 턴에서 결과 **집합은 정확·순서만 스왑**(호출 [a,b,c] vs
    결과 [a,c,b]) → 평가 replay "Tool call id mismatch"로 sim 무효. 부패 주입층은 미상(우리 두 래퍼는
    id-재조립 정합·§2ah/_reassemble)이나, **같은 id 집합·순서 불일치** 블록을 호출 순서로 재정렬하면
    내용 불변의 의미론 no-op이고 pairing이 복원된다(read는 replay 재실행-비교 제외·env-일치).
    반환: 교정 블록 수. messages는 in-place 재배열."""
    fixed, i, n = 0, 0, len(messages)
    while i < n:
        m = messages[i]
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) in ("assistant", "user") and tcs:
            j, k = i + 1, len(tcs)
            block = messages[j:j + k]
            if (len(block) == k and all(getattr(b, "role", None) == "tool" for b in block)):
                want = [getattr(t, "id", None) for t in tcs]
                have = [getattr(b, "id", None) for b in block]
                if want != have and set(want) == set(have) and len(set(have)) == k:
                    by = {getattr(b, "id", None): b for b in block}
                    messages[j:j + k] = [by[w] for w in want]
                    fixed += 1
                i = j + k
            else:
                i += 1
        else:
            i += 1
    return fixed


def _paircheck(messages):
    """커밋 히스토리의 call↔result 쌍 불변식 검사(로그 전용·2026-07-21 §2bd).
    evaluator `get_actions_from_messages`(environment.py:334)와 동일 보행 — rall4 054t1/050t2가
    평가-시점 "Tool call id mismatch"로 sim 무효(신규 계열·전 런 0회)인데 메시지 덤프가 없어
    부패 지점 특정 불가 → 라이브서 매 턴 검사·첫 위반의 도구명/id를 로그. 행동 무변경."""
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) in ("assistant", "user") and tcs:
            j = i + 1
            for tc in tcs:
                if j >= n:
                    return None          # 말미 미실행 호출 = 아직 결과 대기 중(정상·판정 불가)
                tm = messages[j]
                if getattr(tm, "role", None) != "tool":
                    return ("idx %d: %s(id=%s) 다음이 tool이 아님 role=%s"
                            % (j, getattr(tc, "name", None), getattr(tc, "id", None),
                               getattr(tm, "role", None)))
                if getattr(tm, "id", None) != getattr(tc, "id", None):
                    return ("idx %d: call %s id=%s vs result id=%s"
                            % (j, getattr(tc, "name", None), getattr(tc, "id", None),
                               getattr(tm, "id", None)))
                j += 1
            i = j
        else:
            i += 1
    return None


def apply_unified_regen(max_prov_retries=4, domain=None, disamb=False, use_badwords=False,
                        ground=False, disamb_mode="dialog", prov_mode="full"):
    import sys as _sys
    from tau2.agent.llm_agent import LLMAgent
    import tau2.agent.llm_agent as la
    from tau2.data_model.message import ToolMessage, UserMessage, MultiToolMessage
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    a2d = _domain_a2(domain) if domain else None
    hints = a2d["_hints"] if a2d else DEFAULT_ARG_HINTS
    placeholders = a2d["_placeholders"] if a2d else DEFAULT_PLACEHOLDERS
    disamb_tools = _confirm_write_tools(a2d) if disamb else set()
    env_args = _env_verified_args(a2d) if prov_mode == "rescue" else set()   # T5-C P-C (B3)
    sub_args = set((a2d or {}).get("disamb_sub_args") or [])                 # T5-C P-B (B2)
    if os.environ.get("T2_DISAMB_ORDER") == "1":            # ★order operand도 disamb 대상(filter-then-ask·env opt-in)
        sub_args |= {"order", "order_id"}

    # (1) per-sim 게이트 주입 (apply_gate_regen과 동일 패턴·T2_GATE_KINDS 지원)
    orig_init = BaseOrchestrator.__init__

    def init_inject(self, *a, **kw):
        orig_init(self, *a, **kw)
        env = getattr(self, "environment", None)
        ag = getattr(self, "agent", None)
        a2 = _domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if a2 is not None and ag is not None and hasattr(ag, "_generate_next_message"):
            _kinds = os.environ.get("T2_GATE_KINDS")
            gl = a2["gates"]
            if _kinds:
                allow = {k.strip() for k in _kinds.split(",") if k.strip()}
                gl = [g for g in a2["gates"] if g.get("kind") in allow]
            ag._t2_gate = GateInterpreter(gl, resolvers=resolvers_from_env(env))
            ag._t2_a2 = a2
            ag._t2_orch = self

    BaseOrchestrator.__init__ = init_inject

    def _append(state, message):
        if isinstance(message, UserMessage) and getattr(message, "is_audio", False):
            raise ValueError("audio not supported")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    def _gen(self, work, bad_words, call_name, tool_choice=None, pin=None):
        # ★`arrived` 실측 (설계 v1.5 §5 리뷰 N3 — **호출부에서 세지 않는다**).
        #   `_emit` 이 불렸는가가 아니라 *그 문자열이 이 턴 모델 입력에 실제로 있었는가*로 센다.
        #   `proc_fb` 死배선(:7404)이 정확히 이 실수였고 deny 11회를 인쇄로 만들었다([[55]]).
        #   여기가 모델 입력의 유일한 깔때기이므로 배달 판정은 구조적으로 위조될 수 없다.
        #   ⚠거동 불변: 읽기만 하고 `work` 도 `kw` 도 건드리지 않는다.
        try:
            _pend = getattr(self, "_t2_route_pending", None)
            if _pend:
                self._t2_route_pending = []
                _hay = "\n".join(str(getattr(_m, "content", "") or "")
                                 for _m in (self._system_messages + work))
                import t2_fbsidecar as _fbr0
                for _rec in _pend:
                    _txt = _rec.pop("_text", "") or ""
                    _rec["arrived"] = bool(_txt and _txt in _hay)
                    _rec["call_name"] = call_name
                    # ★본문을 record 에 그대로 넘긴다 (2026-08-13 p런 포렌식). 구판은 `_text` 를
                    #   pop 해서 `arrived` 판정에만 쓰고 `text=None` 으로 기록했다 — route 행이
                    #   전부 len=0 이 되어 071 결정-전달(CP2) 검증이 **원천 불가능**했다(handoff
                    #   §6.4 실물). TEXT=1 일 때만 본문 저장되는 규약은 record() 가 이미 지킨다.
                    _fbr0.record("route", _txt, work, **_rec)
        except Exception as _ea:
            print("[T2_ROUTE] arrived 계측 실패(무시): %r" % (_ea,),
                  file=_sys.stderr, flush=True)
        kw = dict(self.llm_args)
        _tools = self.tools
        if use_badwords and bad_words:
            eb = dict(kw.get("extra_body") or {})
            eb["bad_words"] = sorted(bad_words)
            kw["extra_body"] = eb
        # ★P1(N97 §1·x72 3/3): 이름만 지정하면 모델이 **내부 도구명을 날조**한다(x72 T2 실측
        #   `AccountLookupTool`). 디스패처 인자를 **단일값 enum**으로 함께 고정해야 표적에 닿는다.
        #   스키마를 안전하게 못 만들면 고정 자체를 포기한다(이름만 지정 = 날조 유도라 더 나쁨).
        if pin:
            try:
                import t2_pin_read as _PR
                _pt = _PR.tools_with_pin(self.tools, pin[0], pin[1], pin[2])
                if _pt is not None:
                    _tools, tool_choice = _pt, _PR.choice(pin[0])
                    # ★1회/sim 캡을 걷어냈다(C331). 멈춤 조건은 예산이 아니라 **그 호출이 실제로
                    #   나왔는가**다 — 캡이 있으면 권고 턴처럼 정작 필요한 자리에서 못 건다.
                    print("[T2_PIN_READ] pinned %s(%s=%s)" % (pin[0], pin[1], pin[2]),
                          file=_sys.stderr, flush=True)
                    try:
                        from t2_lever_beat import beat as _pbeat
                        _pbeat("T2_PIN_READ", pin[2])
                    except Exception:
                        pass
            except Exception as _pe:
                print("[T2_PIN_READ] skipped: %r" % (_pe,), file=_sys.stderr, flush=True)
        if tool_choice:                          # ★레버 A(2026-07-18): tau2 `generate`의 일급 파라미터로 통과
            kw["tool_choice"] = tool_choice
            # ★max_tokens 하한 (2026-07-23 근본원인 규명·vLLM #19051/#36794): tool_choice='required'는
            #   강제 tool-call JSON을 *완성*해야 하는데 max_tokens가 작으면 중간 절단 → hermes 파서 EOF →
            #   vLLM이 오도성 `__log_extra_fields__` 400 보고(실측: mt=20 실패·mt≥100 성공·전 도구수).
            #   ⚠2026-08-05 정정: 이 주석은 "라이브는 max_tokens 미설정"이라 적고 있었으나 **낡았다** —
            #   `go_stack.sh`가 C271 이후 `T2_AGENT_MAX_TOKENS=8192`를 설정하고 러너가 그대로 싣는다.
            #   8192는 실패 구간(20~100)보다 두 자릿수 위이고 하한(_FORCE_MIN_TOKENS)에도 안 걸린다.
            _mt = kw.get("max_tokens")
            if _mt is not None and _mt < _FORCE_MIN_TOKENS:
                kw["max_tokens"] = _FORCE_MIN_TOKENS
        # ★T2_PROMPT_DUMP — **이 자리**가 모델 입력의 유일한 깔때기다(위 주석). 첫 판은
        #   `apply_provenance_regen` 쪽 `_gen` 에 달았는데 라이브는 `unified` 를 쓴다:
        #   레코드 0 · 예외 0 으로 두 런을 태웠다(t7366 두 번). 자국 없는 계기는 계기가 아니다.
        if os.environ.get("T2_PROMPT_DUMP") == "1":
            try:
                import t2_fbsidecar as _fbp
                _cap = int(os.environ.get("T2_PROMPT_DUMP_MAX", "60000"))
                _parts = []
                for _m in (self._system_messages + work):
                    try:
                        _c = _content_str(_m) or ""
                    except Exception:
                        _c = str(getattr(_m, "content", "") or "")
                    _tc = " ".join(str(getattr(t, "name", ""))
                                   for t in (getattr(_m, "tool_calls", None) or []))
                    _parts.append("[%s]%s %s" % (getattr(_m, "role", "?"),
                                                 (" CALLS " + _tc) if _tc else "", _c))
                _fbp.record("prompt", (chr(10).join(_parts))[:_cap], work,
                            channel="gen", call=str(call_name))
            except Exception as _pe:
                print("[T2_PROMPT_DUMP] skipped: %r" % (_pe,), file=sys.stderr, flush=True)
        try:
            _r = la.generate(model=self.llm, tools=_tools,
                             messages=self._system_messages + work, call_name=call_name, **kw)
            # ★P3 살리기(C248·기본 OFF): hermes 파서가 텍스트로 강등한 호출을 회수한다.
            #   모델이 이미 낸 첫 블록만 복구하고 복제분은 버린다([[10]] 정합).
            try:
                import t2_salvage as _sv
                _sv.salvage_message(_r)
            except Exception:
                pass
            return _r
        except Exception as _ce:
            # ★force_required 안전판 (2026-07-23): 하한 보장 뒤에도 남는 400 = 병리적 케이스(퇴행 루프서
            #   강제 시 닫히지 않는 runaway JSON·mt를 아무리 키워도 미완=039 실측)·기타 transient. → 강제 없이
            #   1회 재시도(넛지·work 유지=효과 보존·프로즈 봉쇄만 포기). 크래시(sim 무효) 대신 우아한 강등.
            #   un_fb 등 모든 force 경로 공용.
            if tool_choice and ("BadRequest" in type(_ce).__name__ or " 400" in (" " + str(_ce))):
                print("[T2_FORCE] tool_choice=%s rejected (400) -> retry without force" % tool_choice,
                      file=_sys.stderr, flush=True)
                kw.pop("tool_choice", None)
                try:
                    return la.generate(model=self.llm, tools=_tools,
                                       messages=self._system_messages + work, call_name=call_name, **kw)
                except Exception as _ce2:
                    _ce = _ce2
            # ★CWE graceful-stop @_gen (§2bf·rall5 실측): step-래핑 가드가 4번째 경로로 우회 —
            #   LLM_DIAG가 특정한 두 누출(call_name=agent_response·followup_decision) 모두 _gen 경유.
            #   여기서 잡아 orch.done+CONTEXT_WINDOW_EXCEEDED(§2ah 의도된 종료사유)로 우아한 종료 →
            #   sim 무효(infra) 대신 부분 궤적 채점(정직한 실패 계상). step-가드는 백스톱 존치.
            if "ContextWindow" not in type(_ce).__name__:
                raise
            # ★P1(C208①·DAY5_PRESCRIPTIONS §P1): 동적 max_tokens — day5 ctxover 7건의 직접 사인은
            #   고정 예약(8192)이 깎은 천장(48,640−8,192=40,448·7건 전부 36.5~40.2k서 사망·모델 창
            #   초과 0건). vLLM 에러 원문이 정확한 수를 주므로 **추정 없이** 파싱→축소→1회 재시도.
            #   플로어 미만 = 진짜 창 소진 → 기존 graceful-stop 그대로. 도메인 무관·판단 0.
            if os.environ.get("T2_DYN_MT") == "1":
                _newmt = _dyn_mt_target(str(_ce),
                                        margin=int(os.environ.get("T2_DYN_MT_MARGIN", "64")),
                                        floor=int(os.environ.get("T2_MT_FLOOR", "256")))
                if _newmt is not None:
                    print("[T2_DYN_MT] shrink %s->%d (at %s)"
                          % (kw.get("max_tokens"), _newmt, call_name),
                          file=_sys.stderr, flush=True)
                    self._t2_dyn_shrunk = True         # P8이 참조(천장 근접 시 재제시 생략)
                    kw2 = dict(kw)
                    kw2["max_tokens"] = _newmt
                    try:
                        return la.generate(model=self.llm, tools=_tools,
                                           messages=self._system_messages + work,
                                           call_name=call_name, **kw2)
                    except Exception as _ce3:
                        if "ContextWindow" not in type(_ce3).__name__:
                            raise
                        _ce = _ce3                   # 재시도도 CWE → graceful-stop으로
            _orch = getattr(self, "_t2_orch", None)
            try:
                from tau2.data_model.simulation import TerminationReason as _TR2
                if _orch is not None:
                    _orch.done = True
                    _orch.termination_reason = _TR2.CONTEXT_WINDOW_EXCEEDED
            except Exception:
                pass
            print("[T2_OVERFLOW_GUARD] CWE at %s -> graceful stop (scored partial)" % call_name,
                  file=_sys.stderr, flush=True)
            _txt = "(context limit reached - conversation ending)"
            try:
                from tau2.data_model.message import AssistantMessage as _AM2
                return _AM2(role="assistant", content=_txt)
            except Exception:
                import types as _types
                return _types.SimpleNamespace(role="assistant", content=_txt, tool_calls=None)

    def _gen_action_sub(self, state, asub, call_name="agent_response_action_sub"):
        """★액션 서브 (2026-08-10·`T2_ACTION_SUB`·설계서 §2·근거 x228).

        손님에게 넘길 **그 턴의 발화만** 깨끗한 문맥에서 짓는다. 문맥은 셋뿐이다 —
          ⒜ 손님 발화 **축자 전부**(요약하지 않는다: 의미 참조가 거기 있다)
          ⒝ 결정 값(격리 서브가 이미 낸 블록·우리가 새로 짓지 않는다)
          ⒞ 도구 **소유자 표기**(지시문이 아니라 사실 — `{tool}({args})`, 도메인 어휘 0)

        x228 실측(3태스크×n=6): 같은 지시를 **메인**에 두면 소유권 발화 098 0/6·100 1/6,
        **격리**에서 지으면 6/6·5/6 이고 `external` 위반이 6/6 → **0/6**. 지시 없는 격리는
        0/6 이라 공로는 표기 쪽이다(부정 통제 통과).

        ⚠도구를 주지 않는다 — 이 턴은 **발화**이지 호출이 아니다. 호출이 필요한 턴에는
          호출부가 이 경로를 타지 않는다.
        ⚠실패하면 `None` 을 돌려 **종전 경로**로 떨어진다(조용한 거동 변경 금지).
        """
        try:
            from tau2.data_model.message import UserMessage as _UMx
            _us = [_content_str(_m) for _m in (state.messages or [])
                   if getattr(_m, "role", None) == "user" and _content_str(_m).strip()]
            if not _us or not (asub or {}).get("value"):
                return None
            # ★소유자 표기는 **x228 이 잰 그 형태**여야 한다(2026-08-10 스모크 교정).
            #   1차 구현은 한 줄로 줄였는데(`the CUSTOMER runs this one in this chat: …`),
            #   라이브 098 에서 *"1. Go to the Rho-Bank customer portal or app…"* 가 나왔다.
            #   x228 의 `F_SUB_TOOLTAB`(external 0/6)은 **두 칸 대조표**였다 — 에이전트가
            #   부르는 도구와 손님이 이 대화에서 실행하는 도구를 **나란히** 놓아 채널 자체를
            #   사실로 세운다. 지시가 아니라 사실이고, 이름은 **실제 도구 목록**에서 온다
            #   (엔진이 짓지 않는다·도메인 어휘 0).
            _own = ""
            if asub.get("tool"):
                _mine = []
                for _t in (self.tools or []):
                    _n = (getattr(_t, "name", None)
                          or (isinstance(_t, dict) and ((_t.get("function") or {}).get("name")
                                                        or _t.get("name"))))
                    if _n and _n != asub["tool"]:
                        _mine.append(str(_n))
                _own = ("Tool ownership on record:\n"
                        "  Tools you call: %s\n"
                        "  Tools the CUSTOMER runs in this chat: %s(%s)"
                        % (" · ".join(_mine) or "(none on record)",
                           asub["tool"], ", ".join(asub.get("args") or [])))
            _work = [_UMx(role="user", content=t) for t in _us]
            _work.append(_UMx(role="user", content=str(asub["value"]).strip()))
            if _own:
                _work.append(_UMx(role="user", content=_own))
            kw = dict(self.llm_args)
            kw.pop("tool_choice", None)
            _r = la.generate(model=self.llm, tools=None, messages=_work,
                             call_name=call_name, **kw)
            print("[T2_ACTION_SUB] 발화를 격리에서 지음 (손님 발화 %d건 · 값 %d자 · 표기 %s)"
                  % (len(_us), len(str(asub["value"])), "O" if _own else "X"),
                  file=_sys.stderr, flush=True)
            return _r
        except Exception as _ae:
            print("[T2_ACTION_SUB] 건너뜀(종전 경로): %r" % (_ae,),
                  file=_sys.stderr, flush=True)
            return None

    def unified(self, message, state):
        # 발화 로그에 sim을 붙인다 — 귀속(C294)이 판정의 1차 지표인데 로그가 무기명이면 못 한다.
        # 턴도 함께 심는다(C407) — 사이드카는 턴을 지니는데 stderr 마크는 순서뿐이라 두 채널을
        # 맞댈 수 없었다. 관측 전용·거동 불변.
        try:
            from t2_lever_beat import set_sim_from as _ssf, set_turn as _stn
            _ssf(self)
            _stn(state)
        except Exception:
            pass
        if not hasattr(self, "_t2_static_bl"):
            self._t2_static_bl = _static_blacklist(self.tools, placeholders)
            self._t2_session_bl = set()
        self._system_messages = state.system_messages
        # ★W-A(arm③): A2 base-layer의 선언 가이드를 시스템 프롬프트 말미에 보간(미선언=skip).
        try:
            import t2_declfirst as _df
            # ★관찰자 ↔ 개입 분리(2026-07-31): 이 가이드 주입은 **행동을 바꾼다**(X13 A_PROMPT arm
            #   실측: 가이드만으로 턴의 31.8%에서 봉투 산출). 반면 2패스 형식화는 이미 확정된 턴을
            #   비커밋으로 재구성하므로 **에이전트에게 아무것도 말하지 않는다**. 둘을 한 플래그로
            #   묶어두면 "pass 레버 다 켜고 계측만 얹기"가 불가능하고, Y1(가이드 없음) 대비
            #   비교성도 깨진다. ⇒ `T2_DECLFIRST_GUIDE=0`이면 **관찰자만** 돈다.
            #   기본은 1(=종전 동작 보존·arm③ 아키텍처).
            _guide_on = os.environ.get("T2_DECLFIRST_GUIDE", "1") == "1"
            # ⛔**이 자리는 죽어 있었다**(2026-08-16 발견·`test_no_unbound_a2` 로 봉인).
            #   `a2` 는 이 함수 **아래쪽**(`a2 = getattr(self, "_t2_a2", None)`)에서 바인딩되므로
            #   여기서 읽으면 매 턴 `UnboundLocalError` 이고 바로 아래 `except Exception: pass` 가
            #   그것을 **조용히 삼켰다** ⇒ `unified` 경로에서 declfirst 가이드는 **한 번도 주입된 적이
            #   없다**(같은 코드의 `patched` 경로 사본은 살아 있다 — 그래서 더 안 보였다).
            #   ⚠조용히 살리지 않는다: 살리면 **모든 과거 런과 베이스라인이 달라진다**.
            #   `T2_DECLFIRST_GUIDE_FIX=1` 일 때만 살리고 효과는 **별도 A/B** 로 잰다([[57]]).
            _a2g = (getattr(self, "_t2_a2", None)
                    if os.environ.get("T2_DECLFIRST_GUIDE_FIX") == "1" else None)
            _g = _df.guide_text(_a2g) if (os.environ.get("T2_DECLFIRST") == "1"
                                          and _guide_on and _a2g is not None) else ""
            if _g and self._system_messages and not getattr(self, "_t2_df_guided", False):
                _sm = self._system_messages[-1]
                _sm.content = (getattr(_sm, "content", "") or "") + _g
                self._t2_df_guided = True
        except Exception:
            pass
        _append(state, message)
        # ★T2_PAIRFIX (§2bi): 같은 id 집합·순서-스왑 블록을 호출 순서로 교정(내용 불변·크래시 계열 종결).
        if os.environ.get("T2_PAIRFIX") == "1":
            try:
                _nfx = _pairfix(state.messages)
            except Exception:
                _nfx = 0
            if _nfx:
                print("[T2_PAIRFIX] reordered %d swapped result block(s)" % _nfx,
                      file=_sys.stderr, flush=True)
        # ★T2_PAIRCHECK (§2bd·로그 전용): 커밋 히스토리 call↔result 쌍 불변식 라이브 검사 —
        #   rall4 id-mismatch(평가-시점 크래시·덤프 부재)의 부패 지점을 다음 발생 시 특정.
        if os.environ.get("T2_PAIRCHECK") == "1" and not getattr(self, "_t2_paircheck_hit", False):
            try:
                _pc = _paircheck(state.messages)
            except Exception:
                _pc = None
            if _pc:
                self._t2_paircheck_hit = True
                print("[T2_PAIRCHECK] pairing broken: %s" % _pc, file=_sys.stderr, flush=True)
        gate = getattr(self, "_t2_gate", None)
        a2 = getattr(self, "_t2_a2", None)
        # ★[T2_UNUSED_GRANT] 계기 (2026-08-31·기록 전용·거동 불변). 정책이 금지한 *쓰지 않을
        #   도구를 주기/잠금해제* 가 이 대화에서 몇 건인지 남긴다 — base `task_010` 은 그 한 건으로
        #   액션 만점(2/2)에서 reward 0.0 이 됐다(`unused_grants` 독스트링에 축자·기전).
        #   집합이 **바뀔 때만** 찍는다.
        if os.environ.get("T2_UNUSED_GRANT", "1") == "1":
            try:
                _ug = unused_grants(state.messages, a2)
                _sig = (tuple(_ug["unused_given"]), tuple(_ug["unused_unlocked"]))
                if _sig != getattr(self, "_t2_ug_sig", None):
                    self._t2_ug_sig = _sig
                    if _sig[0] or _sig[1]:
                        print("[T2_UNUSED_GRANT] 준 뒤 안 쓰임=%s · 잠금해제 뒤 안 쓰임=%s "
                              "(given=%d unlocked=%d)"
                              % (_ug["unused_given"] or "-", _ug["unused_unlocked"] or "-",
                                 len(_ug["given"]), len(_ug["unlocked"])),
                              file=_sys.stderr, flush=True)
            except Exception as _uge:
                print("[T2_UNUSED_GRANT] 계기 실패(무시): %r" % (_uge,),
                      file=_sys.stderr, flush=True)
        # ★T2_DELIVER_PRECOMMIT (2026-08-16·기본 OFF) — **배달 시점만** 앞으로 옮긴다.
        #
        # 왜. 지금 배달은 **결정 자리**에서만 난다(t7299 실측: 결정자리 22 · 일반자리 0). 그런데
        # 모델은 그보다 **먼저 이름을 확정**한다 — 055 첫 지목 msg 7~15 · 024 msg **4**(gold 문서가
        # msg 7 에 검색 1위로 왔는데도 안 바꿈) · 그리고 재료는 **한 턴만 산다**(재생성 버퍼·C498).
        # ⇒ 확정 뒤에 도착한 재료는 안 먹는다. 선행도 같은 방향을 본다(2606.22936 premature
        #   commitment · 2605.28721 증거 사용률 1/3 미만) — 다만 **주입 시점을 제어**하지는 않는다.
        #
        # 무엇을 하나. sim 당 **한 번**, 문서군이 형식화되는 **가장 이른 턴**에 재료를 배달한다.
        # 새 판단 기구 0 — 문서군을 고르는 것도(LLM), 무엇을 고를지도(모델) 그대로다. 옮긴 것은
        # **시점 하나**뿐이고 예산(총 3)도 그대로다.
        #
        # ⚠[[62]]: 이 결손은 격리로 쟀다(x335b 0/24→24/24 · x338 24/24 ↔ 라이브 0). 격리에서
        #   되므로 레버는 **전달뿐**이고, 계산·선택을 대신하지 않는다.
        # ⚠[[57]]: 이르게 주면 **손님이 요구를 다 말하기 전**일 수 있다(=군 오선택 위험). 그것이
        #   이 실험이 재려는 상쇄다. 1차 종점은 성적이 아니라 **첫 지목 이전 도달 sim 비율**.
        if (os.environ.get("T2_DELIVER_PRECOMMIT") == "1"
                and os.environ.get("T2_SEARCH_AGENT") == "1"
                and a2 is not None
                and not getattr(self, "_t2_precommit_done", False)):
            try:
                _pc = _search_material(self, a2, state.messages, decide=False)
            except Exception as _pce:
                _pc = ""
                print("[T2_DELIVER_PRECOMMIT] 건너뜀(무발화): %r" % (_pce,),
                      file=_sys.stderr, flush=True)
            if _pc:
                self._t2_precommit_done = True
                self._t2_searchagent_fired = getattr(self, "_t2_searchagent_fired", 0) + 1
                _cp2_assign(self, _pc, "PRECOMMIT")
                self._t2_cp2_said = _pc
                print("[T2_DELIVER_PRECOMMIT] 선-배달 turn=%d · 재료 %d자"
                      % (len(state.messages), len(_pc)), file=_sys.stderr, flush=True)
        last_user = transfer_sent = None
        if gate is not None:
            _rebuild_gate_state(gate, a2, state.messages)
            last_user = _regen_last_user(state.messages)
            # ★NOTICE-PERGATE 커링 (apply()와 동형)
            transfer_sent = lambda text: _regen_transfer_sent(state.messages, text)  # noqa: E731
        ctx = _ctx_with_toolnames(self, _ctx_from_messages(state.messages))

        # ★E-PLAN v1.3 (T2_EPLAN=1): committed 히스토리서 결정론 ledger 재구성(관측만·[[10]])
        #   discovery L1/L2 = read-강제 deny(§1.5 허용축)·CP5 리마인더 소비 = 생성-레벨(비커밋)
        ep_led = ep_spec = _epmod = None
        ep_writes = set()
        if os.environ.get("T2_EPLAN") == "1" and a2 is not None and a2.get("eplan"):
            try:
                import t2_eplan_patch as _epmod
                ep_spec = a2.get("eplan")
                # ep_writes = confirm-gate write 도구 ∪ eplan spec write_tools(C101 (c)·디스패처 nested용)
                ep_writes = _confirm_write_tools(a2) | set(ep_spec.get("write_tools") or ())
                ep_led = _epmod.build_ledger_from_messages(state.messages, ep_spec, ep_writes)
                # ★A12 / OL-20 (2026-08-22 · t7336 마스터 §6.1 A12 · §5.3): *"목록에는 있는데
                #   상세를 안 읽었다"* 는 `list_from_reads:true` 를 선언한 도메인에서 **항상 거짓**이다.
                #   그 선언의 뜻이 곧 *"이 도메인의 read 출력은 레코드를 통째로 뱉는다"* 이고
                #   (`note_read` 독스트링 축자: *"어느 read 출력이든 entity_key id를 listed"*),
                #   그래서 id 가 `listed` 에 들어왔다는 것 자체가 **그 행이 이미 배달됐다**는 뜻이다.
                #   `examined` 는 `tool_name == detail_reader` 일 때만 차므로, 선언된 detail_reader 가
                #   아닌 형제 enumerator 로 읽은 행은 영원히 `listed - examined` 에 남는다.
                #   실측(085#1·[S]): 바로 앞 출력에 **전량 실린** 5 레코드를 L2 가 *"have not read
                #   their details yet"* 로 판정하고 4턴을 태웠으며(deny cap 도달), 그 문면이 지목한
                #   `detail_reader` 는 credit 도구인데 대상은 체킹 `btxn_*` 이었다([[25]] 거짓 지목).
                #   ⇒ 선언에서 기계 도출되는 집합 보정 하나로 닫는다(합집합·닫힌 술어·리터럴 0).
                #   ⚠소비자 **둘 다** 이 한 자리에서 고쳐진다 — `discovery_L2`(=`listed - examined`)와
                #     `T2_READALL`(=`readall_unread(listed, examined)`). 사본 0([[67]]).
                #   ⚠`list_from_reads` 미선언 도메인(retail·airline)은 **바이트 불변**(no-op).
                #   ⚠[[70]] **무엇을 파는가** = 이 도메인에서 **EPLAN L2 · READALL 의 read-강제 전부**.
                #     계수 = `[T2_EPLAN_LISTED_IS_READ]` 줄(보정된 id 수) ↔ 종전 `L2 deny` 발화 수.
                #     L1(목록 도구 미호출)은 그대로 살아 있고, coverage 는 CP5 walk 관할이다.
                if ep_led is not None and ep_spec.get("list_from_reads"):
                    _lr = set(getattr(ep_led, "listed", ()) or ()) - set(
                        getattr(ep_led, "examined", ()) or ())
                    if _lr:
                        ep_led.examined |= _lr
                        print("[T2_EPLAN_LISTED_IS_READ] list_from_reads 선언 — 배달된 %d건을 "
                              "검토됨으로 보정(%s)" % (len(_lr), ",".join(sorted(_lr))[:120]),
                              file=_sys.stderr, flush=True)
            except Exception as _e:
                print("[T2_EPLAN] ledger build failed: %r" % (_e,), file=_sys.stderr, flush=True)
                ep_led = None

        # ★T2_WRITE_EVIDENCE unified 배선(2026-07-19 028 포렌식): 구 apply()에만 있던 WEV가
        #   unified 런(T2_GATE_REGEN∧T2_GROUND)서 死코드 → deny 0회/증거없는 update 6건 통과.
        #   생성-레벨 deny(ep/cons/ra/te와 동렬·무과금·sim당 cap)로 이설. 검사 코어=_wev_deny_msgs 공유.
        wev_specs = (a2.get("write_evidence_specs") or []) \
            if (a2 is not None and os.environ.get("T2_WRITE_EVIDENCE") == "1") else []
        # ★T2_WRITE_ARG_GROUND (§2bs): WEV 블록에 합류(동일 라운드·cap·적용 배관 공유 — 문서화됨)
        wag_specs = (a2.get("write_arg_grounding") or []) \
            if (a2 is not None and os.environ.get("T2_WRITE_ARG_GROUND") == "1") else []
        # ★T2_ARG_EMPTY (C419·x250): 필수 인자 빈 문자열 → 이름을 대고 거부. 필수 목록은 env
        #   스키마에서 도출하므로 A2 는 **문구와(선택적) 적용 범위**만 갖는다. 범위 미선언 =
        #   스키마가 required 를 선언한 모든 도구(빈 필수 인자는 어느 도구에서도 값이 아니다).
        ae_on = (a2 is not None and os.environ.get("T2_ARG_EMPTY") == "1")
        ae_tools = set(((a2 or {}).get("arg_empty") or {}).get("applies_to") or ()) or None
        # ★T2_REF_VERIFY (C128/C129): 결정론 참조-검증기 spec(도구/필드/문구=A2)·WEV 블록 합류
        rv_specs = (a2.get("ref_verify") or []) \
            if (a2 is not None and os.environ.get("T2_REF_VERIFY") == "1") else []
        _wev_cap = int(os.environ.get("T2_WEV_CAP", "8"))
        # ★T2_HAVE_VALUE (C115·2026-07-23): have-value→act 일반레버 spec(도구/인자/신호/문구=A2)
        hv_specs = (a2.get("have_value_reask") or []) \
            if (a2 is not None and os.environ.get("T2_HAVE_VALUE") == "1") else []
        # ★T2_VALUE_ACQUIRE (C119): give 표면화 spec(도구/문구=A2)
        va_specs = (a2.get("value_acquisition") or []) \
            if (a2 is not None and os.environ.get("T2_VALUE_ACQUIRE") == "1") else []

        def bw():
            return [v for v in (self._t2_static_bl | self._t2_session_bl) if v.lower() not in ctx]

        work = list(state.messages)
        # ★T2_VIEW_COMPACT (§2bi): 생성-뷰만 압축(커밋 히스토리·게이트 ctx는 원문 유지=replay-safe).
        if os.environ.get("T2_VIEW_COMPACT") == "1":
            work, _dg = _compact_view(
                state.messages,
                keep_recent=int(os.environ.get("T2_VIEW_COMPACT_KEEP", "6")),
                min_len=int(os.environ.get("T2_VIEW_COMPACT_MINLEN", "800")),
                # ★P5: 기본 60,000자(구 120,000=사망선 위·day5 6/32만 발동)·per-메시지 캡 8,000자.
                min_total=view_thresholds()[0],
                msg_cap=view_thresholds()[1])
            self._t2_view_digested = _dg
            # ★A-7⑶ (2026-08-23·016): 구판은 sim 당 1회라 실제 5개/4개가 로그 한 줄이 됐다.
            #   턴마다 다시 재되 **같은 집합이면 침묵**한다(부피는 안 늘고 변화는 남는다).
            if _dg:
                _vcsig = tuple(sorted(_dg)) if not isinstance(_dg, dict)                     else tuple(sorted(_dg.keys()))
                if _vcsig != getattr(self, "_t2_vc_logged_sig", None):
                    self._t2_vc_logged_sig = _vcsig
                    self._t2_vc_logged = True
                    print("[T2_VIEW_COMPACT] active: %d tool output(s) digested in view"
                          % len(_dg), file=_sys.stderr, flush=True)
        # ★T2_VIEW_ANNOTATE (2026-07-22 §2bs): A2-선언 필드-주석을 생성-뷰에만 부가(비커밋).
        #   COMPACT 뒤에 적용 — 실제로 보이는 뷰 기준으로만 주석. 기본 OFF(거동보존).
        if os.environ.get("T2_VIEW_ANNOTATE") == "1" and a2 is not None:
            _vas = a2.get("view_field_annotations") or []
            if _vas:
                work, _nva = _annotate_view(work, _vas)
                if _nva and not getattr(self, "_t2_va_logged", False):
                    self._t2_va_logged = True
                    print("[T2_VIEW_ANNOTATE] active: %d tool output(s) annotated in view"
                          % _nva, file=_sys.stderr, flush=True)
        # ★연기된 표적 재-제시 (2026-08-08·C300·층 3의 pin 절반). 거절한 표적의 선행이 **풀렸는지**
        #   매 턴 본다 — 풀렸으면 그 행동을 한 번 다시 내밀고 명부에서 지운다. 판정은 이미 선언된
        #   요건 그래프의 결정론이고(새 A2 키 0), 전달은 **비커밋 뷰 채널**이다(C298: 궤적에 쓰면
        #   replay가 깨진다). 우리가 실행하지 않는다 — 호출은 모델이 한다([[05]] Q3).
        _dfr0 = getattr(self, "_t2_deferred", None)
        if _dfr0 and a2 is not None:
            try:
                import t2_dominance as _DOMc
                _cl0 = _DOMc.cleared(a2, state.messages, _dfr0,
                                     executed=_executed_tool_names(state.messages, a2),
                                     unwrap=_exact_tool_name)
                if _cl0:
                    _q0 = list(getattr(self, "_t2_view_fb", None) or [])
                    for _tgt0, _txt0 in _cl0:
                        _q0.append([_txt0, int(os.environ.get("T2_DEFERRED_VIEW_KEEP", "2"))])
                        _dfr0.pop(_tgt0, None)
                        print("[T2_DEFERRED] re-offered target=%s (its precondition now holds)"
                              % _tgt0, file=_sys.stderr, flush=True)
                    self._t2_view_fb = _q0
                    self._t2_deferred = _dfr0
                    _lbeat("T2_DEFERRED", orch=self, target=_cl0[0][0],
                           fact="a step held back earlier is now permitted")
            except Exception as _de0:
                print("[T2_DEFERRED] skipped (no-op): %r" % (_de0,),
                      file=_sys.stderr, flush=True)
        _rem = getattr(self, "_t2_eplan_reminder", None)
        if _rem:  # CP5 walk 리마인더(작업버퍼만·히스토리 비커밋 = 채널 절대규칙)
            self._t2_eplan_reminder = None
            try:
                work = work + [UserMessage(role="user", content=_rem)]
            except TypeError:
                work = work + [UserMessage(content=_rem)]
        # ★P2(C208②·DAY5_PRESCRIPTIONS §P2): replay-비교 대상 도구의 피드백 뷰-채널 소비 —
        #   prekb가 큐잉(`_t2_view_fb`)·여기서 작업버퍼에만 주입(히스토리 비커밋=위 채널 절대규칙 동일).
        _vfb = getattr(self, "_t2_view_fb", None)
        if _vfb:
            # ★F5(C210): 항목=[텍스트, 잔여횟수] — 이번 뷰에 주입 후 잔여>0이면 다음 생성에도
            #   재노출(day6 033 [S]: 1회 노출은 무시됨). 구형 str 항목=1회(하위호환).
            _texts, _keep = [], []
            for _it in _vfb:
                if isinstance(_it, (list, tuple)) and len(_it) == 2:
                    _texts.append(str(_it[0]))
                    if int(_it[1]) - 1 > 0:
                        _keep.append([str(_it[0]), int(_it[1]) - 1])
                else:
                    _texts.append(str(_it))
            self._t2_view_fb = _keep or None
            _vtxt = "\n".join(_texts)
            print("[T2_FB_VIEW] %d queued feedback item(s) injected in view" % len(_texts),
                  file=_sys.stderr, flush=True)
            try:
                work = work + [UserMessage(role="user", content=_vtxt)]
            except TypeError:
                work = work + [UserMessage(content=_vtxt)]
        # ★COV FIND-subset 백스톱 (T2_COV=1): ≥1 write 후 M∖acted ≠ ∅ → in-flight 리마인더 1회
        if (os.environ.get("T2_COV") == "1" and ep_led is not None
                and not getattr(self, "_t2_cov_reminded", False)):
            try:
                execd = {str(e.get("entity") or "").strip()
                         for e in getattr(ep_led, "executed", [])}
                execd.discard("")
                if execd:
                    M = getattr(self, "_t2_cov_M", None)
                    if M is None:
                        M = _cov_formalize_M(self, la, UserMessage, state.messages,
                                             a2.get("eplan") if a2 else None, a2)
                        self._t2_cov_M = M  # []도 캐시(서브콜 1회)
                    remaining = [m for m in (M or []) if m not in execd]
                    if remaining and len(M) >= 2:
                        self._t2_cov_reminded = True
                        print("[T2_COV] reminder M=%s acted=%s remaining=%s"
                              % (",".join(M), ",".join(sorted(execd)), ",".join(remaining)),
                              file=_sys.stderr, flush=True)
                        _cr = COV_REMINDER.format(ids=", ".join(remaining))
                        try:
                            work = work + [UserMessage(role="user", content=_cr)]
                        except TypeError:
                            work = work + [UserMessage(content=_cr)]
            except Exception as _cve:
                print("[T2_COV] error (no-op): %r" % (_cve,), file=_sys.stderr, flush=True)
        # ★T2_COV_MIDDRIVE (C118·EPLAN_MIDDRIVE_DESIGN §2.1): "종료시 1회"(T2_COV) → "갭 열린 동안
        #   매 드리프트 견인". 트리거 3AND: ①M(coverage 대상·formalize)≥2 ②remaining=M∖executed≠∅
        #   ③직전 assistant 턴이 write 아님(드리프트). → 리마인더(남은 것 지금 처리) 매턴. 진행(executed↑)
        #   시 stall 리셋·K턴 연속 미진행 중단(무한넛지 방지·§2.1). M=LLM formalize(1회캐시·[[10]])·나머지
        #   결정론(drift/diff/K). write 강제 0(soft·§1.5). 052/043(CLI/close 사슬)엔 미적용(dispute-coverage).
        if (os.environ.get("T2_COV_MIDDRIVE") == "1" and ep_led is not None):
            try:
                execd = {str(e.get("entity") or "").strip()
                         for e in getattr(ep_led, "executed", [])}
                execd.discard("")
                M = getattr(self, "_t2_cov_M", None)
                if M is None:
                    M = _cov_formalize_M(self, la, UserMessage, state.messages,
                                         a2.get("eplan") if a2 else None, a2)
                    self._t2_cov_M = M
                remaining = [m for m in (M or []) if m not in execd]
                if len(execd) > getattr(self, "_t2_cov_execn", 0):   # 진행 감지 → stall 리셋
                    self._t2_cov_stall = 0
                self._t2_cov_execn = len(execd)
                _drift = not _last_assistant_did_write(state.messages, ep_writes)
                _K = int(os.environ.get("T2_COV_MIDDRIVE_K", "4"))
                if (remaining and len(M) >= 2 and _drift
                        and getattr(self, "_t2_cov_stall", 0) < _K):
                    self._t2_cov_stall = getattr(self, "_t2_cov_stall", 0) + 1
                    print("[T2_COV] mid-drive M=%s acted=%s remaining=%s stall=%d/%d"
                          % (",".join(M), ",".join(sorted(execd)), ",".join(remaining),
                             self._t2_cov_stall, _K), file=_sys.stderr, flush=True)
                    _cd = COV_REMINDER_DRIVE.format(n=len(M), done=(", ".join(sorted(execd)) or "none"),
                                                    ids=", ".join(remaining))
                    try:
                        work = work + [UserMessage(role="user", content=_cd)]
                    except TypeError:
                        work = work + [UserMessage(content=_cd)]
            except Exception as _cde:
                print("[T2_COV] mid-drive error (no-op): %r" % (_cde,), file=_sys.stderr, flush=True)
        # ★TOOLERR (T2_TOOLERR=1): 방금 난 tool-error를 A2로 분류·directive 주입(in-flight)
        terr = None
        if os.environ.get("T2_TOOLERR") == "1" and a2 is not None:
            try:
                terr = classify_tool_error(state.messages, a2)
                if terr is not None:
                    _sp, _tool, _fargs = terr
                    _cls = _sp.get("class")
                    _tmpl = TOOLERR_RECOVER if _cls == "recover" else TOOLERR_ABSTAIN
                    _td = _tmpl.format(tool=_tool, hint=_sp.get("hint", ""))
                    print("[T2_TOOLERR] %s tool=%s inject" % (_cls, _tool),
                          file=_sys.stderr, flush=True)
                    try:
                        work = work + [UserMessage(role="user", content=_td)]
                    except TypeError:
                        work = work + [UserMessage(content=_td)]
            except Exception as _te:
                terr = None
                print("[T2_TOOLERR] error (no-op): %r" % (_te,), file=_sys.stderr, flush=True)
        # ★write-착수 **사전(pre-draft) sync 서브** (2026-08-14·`T2_WRITE_SUB=2`·기본 OFF).
        #   여기는 **도구 결과가 막 들어왔고 메인 초안은 아직 없는** 자리다. 종전 배선은 초안을
        #   본 뒤에만 말할 수 있어(=`_ap_regen` 재생성) 옳은 시점을 못 잡았다:
        #     · 옳은 시점(근거 도착 ∧ 행동 미실행) 중 회피 턴은 **27%뿐**(감사 21 sim·t7288 32%)
        #     · 나머지 73%에 끼어들면 그 턴의 호출을 버리는데, 그중 **23%가 write**
        #       (대부분 `log_verification`)·10%가 새 read = **버리면 안 되는 것 33%**
        #   ⇒ 사전 자리에서 **동기로** 서브를 돌리고 답만 얹으면 버릴 초안이 없다(사용자 지시).
        #   ⚠폐기된 autofetch(C34)와 다르다: 새 정보를 **조회해 주입하지 않는다**. 근거는 이미
        #     에이전트가 받은 도구 결과이고, 판단은 격리 서브(LLM)가 한다 — 바뀌는 것은 **자리**뿐
        #     ([[62]] ②·x307 knows 7/8 ↔ acts 0/8 · x308 배치 7/8 · x309 전달 8/8 · x310 근거동봉 안전).
        if os.environ.get("T2_WRITE_SUB") in ("2", "3") and a2 is not None:
            try:
                import t2_subcall as _SCw
                import t2_resolve as _RZw
                _basis = _SCw.recent_tool_text(state.messages,
                                               ((a2.get("write_initiation") or {})
                                                .get("basis_max_chars") or 4000))
                _sig = hash(_basis)
                # ★③ 결정점 좁히기 (2026-08-14 사용자 지시 *"결정점 근처에서 부르면 되지 않나"*).
                #   `=2`(근거 변화마다)는 t7289 에서 **서브 77회** — 사전등록 문턱(≤15) 위반.
                #   결정점 = **그 행동의 선행 조건이 이제 충족됐는데(게이트 통과) 아직 미실행**.
                #   게이트는 매 턴 재구성돼 있으므로(`_rebuild_gate_state`) 새 판단 0·리터럴 0.
                _open = True
                if os.environ.get("T2_WRITE_SUB") == "3":
                    try:
                        _act = set((a2.get("action_tools") or []))
                        _dispatch = ((a2.get("eplan") or {}).get("dispatch_tool") or "")
                        _cands3 = [t for t in _act if t and t != _dispatch]
                        _open = any(gate.check(t, {}, last_user, transfer_sent)[0]
                                    for t in (_cands3 or [_dispatch]) if t)
                    except Exception:
                        _open = True          # 게이트 조회 실패 = 종전 거동(과침묵 금지)
                if _open and _basis and _sig != getattr(self, "_t2_write_basis", None):
                    self._t2_write_basis = _sig          # 같은 근거로 두 번 말하지 않는다([[57]])
                    _done = _RZw._executed_dispatch_names(state.messages, a2)
                    _fbw = _RZw.sub_write_proposal(self, la, UserMessage, state.messages, a2,
                                                   _RZw.registry_names(self) - set(_done))
                    if _fbw:
                        try:
                            work = work + [UserMessage(role="user", content=_fbw)]
                        except TypeError:
                            work = work + [UserMessage(content=_fbw)]
                        import t2_fbsidecar as _fbw0
                        _fbw0.record("reminder-user", _fbw, work, channel="writesub")
                        # ★A-7⑷ (2026-08-23·073): 이 숫자는 **트리거 코퍼스**(recent 창)이고
                        #   서브가 실제로 본 창은 A2 `basis_scope`/`basis_max_chars` 가 정한다.
                        #   073 은 이 두 값이 다른 줄 모르고 679↔2407 을 서브의 창으로 읽었다.
                        print("[T2_WRITE_SUB] pre-draft 전달(트리거 %d자·미실행 필터 %d종)"
                              % (len(_basis), len(_done)), file=_sys.stderr, flush=True)
            except Exception as _we2:
                print("[T2_WRITE_SUB] pre-draft 생략(종전 경로): %r" % (_we2,),
                      file=_sys.stderr, flush=True)
        # ★P3(C208③·DAY5_PRESCRIPTIONS §P3): 터미널-턴 유예의 그 1턴만 tool_choice=required —
        #   재-notice 산문 봉쇄(기존 FORCE_ACTION 기제 재사용·도구/인자 미지정=write 강제 아님).
        # ★읽기 루틴은 **최초 생성**에도 물린다 (2026-08-18·t7319 실측: 재생성 턴에서만
        #   해석돼 0회 발화했다 — 조회가 열린 그 턴들은 아무도 거부하지 않은 평범한 턴이었다).
        #   면제 둘이 부작용을 막는다: 손님이 방금 말한 턴은 걸지 않고(⑵), 같은 집합은 한 번만
        #   건다(⑶). 강제 대상은 **읽기뿐**이다(§1.5 Q5).
        _rt_pin = None
        if os.environ.get("T2_PIN_READ_STEPS") == "1" and a2 is not None:
            try:
                _rt_pin = _read_routine_pin(self, a2, state.messages)
                if _rt_pin:
                    print("[T2_READ_ROUTINE] %s(%s in %s)"
                          % (_rt_pin[0], _rt_pin[1], _rt_pin[2]),
                          file=_sys.stderr, flush=True)
            except Exception as _rte:
                _rt_pin = None
                print("[T2_READ_ROUTINE] 건너뜀(무발화): %r" % (_rte,),
                      file=_sys.stderr, flush=True)
        if getattr(self, "_t2_term_force", False):
            self._t2_term_force = False
            am = _gen(self, work, bw(), "agent_response", tool_choice="required")
        elif _rt_pin:
            am = _gen(self, work, bw(), "agent_response", pin=_rt_pin)
        else:
            am = _gen(self, work, bw(), "agent_response")
        gate_rounds = prov_rounds = eplan_rounds = cons_rounds = ra_rounds = te_rounds = wev_rounds = 0
        tl_rounds = 0
        subs = 0
        # ★R3: 도구-선택자 슬롯은 날조-스캔/치환 대상이 아니다(env 스키마 도출·리테일에선 ∅).
        sel_args = _selector_args_cached(self)
        rescue_skipped = set()
        rescue_excl = set()   # ★PERARG(C65): (id(tc),k,s) — rescue-스킵된 fab 제외하고 재스캔
        absent_fired = False  # ★D1′: 부재 표면화는 **턴당 1회**(재생성 루프가 같은 문구를 도배하지 않게)
        while True:
            force_required = False   # ★T2_FORCE_ACTION: say-don't-do → 다음 재생성서 tool_choice=required 강제
            fab = _first_fab_call(am, ctx, hints, exclude=rescue_excl, selectors=sel_args)
            # ★T5-C P-A (N1: _denied_calls 前 — 게이트 check는 상태-변이라 버려질 반복서 소진 금지)
            if fab is not None and ground and subs < 8:
                gtc, gk, gs = fab
                gcands = _grounded_candidates(gk, gs, state.messages, lenient=True)
                if len(gcands) == 1 and gcands[0] != gs and _subst_arg_value(gtc, gk, gs, gcands[0]):
                    self._t2_ground_sub = getattr(self, "_t2_ground_sub", 0) + 1
                    subs += 1
                    print("[T2_GROUND] substituted arg=%s val=%s -> %s" % (gk, gs, gcands[0]),
                          file=_sys.stderr, flush=True)
                    _lbeat("T2_GROUND", orch=self, target=gk,
                           fact="argument was replaced with the value that exists in context")
                    continue  # 치환값은 문맥-실재 → 다음 반복서 fab 해소·게이트가 최종 인자를 검사
            # ★T5-C P-C + PERARG(C65): env-검증형 id 날조 = *개별* 스킵 후 다음 fab 재스캔
            #   (구: fab=None → 같은 호출의 둘째 자유텍스트 fab[t17 address1]이 영영 미검사)
            while fab is not None and prov_mode == "rescue":
                rtc, rk, rs = fab
                if (_key_tokens(rk) & env_args) and _sig(rs) in ("hashid", "numid") \
                        and not _in_error_loop(state.messages, getattr(rtc, "name", None)):
                    rkey = (getattr(rtc, "name", None), rk, rs)
                    if rkey not in rescue_skipped:
                        rescue_skipped.add(rkey)
                        self._t2_prov_skipped_envdup = getattr(self, "_t2_prov_skipped_envdup", 0) + 1
                        print("[T2_PROV] rescue pass-through tool=%s arg=%s val=%s" % rkey,
                              file=_sys.stderr, flush=True)
                    rescue_excl.add((id(rtc), rk, rs))
                    fab = _first_fab_call(am, ctx, hints, exclude=rescue_excl, selectors=sel_args)
                else:
                    break
            # ★L3 origin-prov (T2_PROV_ORIGIN=1·v3.2): fab 무해(값∈ctx) 통과분 중 확인-세탁 검사
            ofab = None
            if fab is None and os.environ.get("T2_PROV_ORIGIN") == "1":
                ofab = _first_origin_fab(am, state.messages, hints, exclude=rescue_excl)
            denied = _denied_calls(am, gate, last_user, transfer_sent) if gate is not None else []
            denied_by_objid = {id(tc): (gid, why) for tc, gid, why in denied}
            do_gate = bool(denied) and gate_rounds < 1
            _pcall = fab if fab is not None else ofab
            fab_covered = _pcall is not None and do_gate and id(_pcall[0]) in denied_by_objid
            do_prov = (_pcall is not None) and prov_rounds < max_prov_retries and not fab_covered
            # ★E-PLAN discovery deny (L1/L2·read-강제만·무과금·상한 = 턴당 2 + ★sim당 T2_EPLAN_DENY_CAP)
            # ★sim-cap 근거(A′ t5c_aprime1 t103/t27 포렌식·2026-07-11): eplan_rounds는 턴-로컬이라
            #   모델이 deny 피드백에 불응(텍스트-사과 커밋)하면 매 턴 ledger가 동일 재구성 → 동일 L2
            #   deny 무한 반복(무과금=too_many_errors 탈출로도 없음) → t103 max_steps(200)·t27 유저 포기.
            #   read-강제는 sim당 유한 예산으로 — 소진 후엔 write 통과(env/gold 판정에 맡김·개입 최소).
            ep_fb = None
            _ep_cap = int(os.environ.get("T2_EPLAN_DENY_CAP", "4"))
            if (ep_led is not None and eplan_rounds < 2 and not do_gate and not do_prov
                    and getattr(self, "_t2_eplan_deny", 0) < _ep_cap):
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", None)
                    _cargs = _args_dict(c)
                    # ★디스패처 unwrap(C101 (c)·retail 무영향=dispatch_tool 없으면 skip):
                    #   banking dispute write = call_discoverable_agent_tool(nested {name,args}).
                    _dt = ep_spec.get("dispatch_tool")
                    if _dt and nm == _dt:
                        nm = re.sub(r"_\d+$", "", str(_cargs.get(
                            ep_spec.get("dispatch_name_key", "agent_tool_name"), "")))
                        _inner = _cargs.get(ep_spec.get("dispatch_args_key", "arguments"))
                        if isinstance(_inner, str):
                            try:
                                _inner = json.loads(_inner)
                            except Exception:
                                _inner = {}
                        _cargs = _inner if isinstance(_inner, dict) else {}
                    if nm in ep_writes and id(c) not in denied_by_objid:
                        try:
                            # ★qty-conflation 가드(t27): 시도 call의 품목 id를 술어에 전달
                            #   (키=A2 "items_key"·ABox) — N이 품목-급으로 충족되면 deny 안 함.
                            _ep_items = _cargs.get(
                                ep_spec.get("items_key", "item_ids")) or ()
                            _ep_ent = _cargs.get(ep_spec.get("entity_key"))
                            fb = _epmod.discovery_precondition(
                                ep_led, ep_spec, nm, attempt_items=_ep_items,
                                attempt_entity=_ep_ent)
                        except Exception:
                            fb = None
                        if fb:
                            ep_fb = (c, fb)
                            break
            # ★BRANCH-REGROUND pre-close 트리거 (C144/C146·T2_BRANCH_REGROUND=1): finalize
            #   write(close류·A2 finalize_writes) 시도 시 정책조건 선행단계(apply_flag 등) 미완이면
            #   deny + 실제 정책문서 재부각 → apply_flag이 close 선행(현 user_stop 경로는 close 後라 늦음).
            #   ep_fb 채널·cap 재사용(무과금·regen 공유·discovery 미발화 시만). finalize 제외=for_finalize.
            # ★C173-corr(2026-07-25): pre-close 예산을 공유 _ep_cap에서 **분리** — 044 실측:
            #   초반 E-PLAN/discovery deny들이 공유 4회를 소진해 msg66의 2번째 close 시도가
            #   무방비 통과(선행 read/write 미완인 채 CLOSED·상태오염 3). claimprov transfer-창
            #   별도예산(§2ao) 선례와 동일 패턴: finalize 방어는 독립 예비(기본 2/sim)로 보장.
            #   (구 C173의 T2_GATE_REGEN_K=2는 unified 경로서 미사용=no-op이라 철회.)
            if (ep_fb is None and ep_led is not None and eplan_rounds < 2
                    and os.environ.get("T2_BRANCH_REGROUND") == "1"
                    and not do_gate and not do_prov
                    and getattr(self, "_t2_preclose_deny", 0)
                        < int(os.environ.get("T2_PRECLOSE_CAP", "2"))):
                _finals = {re.sub(r"_\d+$", "", f)
                           for f in (ep_spec.get("finalize_writes") or ())}
                if _finals:
                    for c in (am.tool_calls or []):
                        nm = getattr(c, "name", None)
                        _cargs = _args_dict(c)
                        _dt = ep_spec.get("dispatch_tool")
                        if _dt and nm == _dt:
                            nm = re.sub(r"_\d+$", "", str(_cargs.get(
                                ep_spec.get("dispatch_name_key", "agent_tool_name"), "")))
                        else:
                            # ★C148 버그수정: close는 직접호출(name=close_..._7834·디스패처 아님)이
                            #   가능 → suffix fam-strip 안 하면 _finals 미매칭·게이트 우회(043 실측).
                            nm = re.sub(r"_\d+$", "", str(nm or ""))
                        if nm in _finals and id(c) not in denied_by_objid:
                            try:
                                _bchain = _epmod.chain_gap(state.messages, ep_spec)
                            except Exception:
                                _bchain = None
                            if _bchain is not None:
                                _prereq_w = [w for w in _bchain["missing_writes"]
                                             if re.sub(r"_\d+$", "", w) not in _finals]
                                if _bchain["missing_reads"] or _prereq_w:
                                    try:
                                        _brem = _epmod.branch_reground_reminder(
                                            _bchain, state.messages, ep_spec, for_finalize=True)
                                    except Exception:
                                        _brem = None
                                    if _brem:
                                        ep_fb = (c, _brem)
                                        # C173-corr: 전용 예산 소모(독립 카운터·기본 2/sim)
                                        # C174: kind 플래그 — 공유 증가 지점서 discovery
                                        #   카운터를 건드리지 않게 표시(예산 전면 독립).
                                        self._t2_ep_fb_preclose = True
                                        self._t2_preclose_deny = getattr(
                                            self, "_t2_preclose_deny", 0) + 1
                                        print("[T2_BRANCH_REGROUND] pre-close deny: finalize=%s "
                                              "prereq_reads=%d prereq_writes=%d (pc_deny=%d)"
                                              % (nm, len(_bchain["missing_reads"]), len(_prereq_w),
                                                 self._t2_preclose_deny),
                                              file=_sys.stderr, flush=True)
                                        break
            # ★DISCOVERY-DISPATCH deny (C151·T2_DISCOVERY_DISPATCH=1): discoverable 도구를 *직접
            #   호출*(suffixed name·dispatcher 미경유)하면 평가 리플레이가 등록 안 함(registration은
            #   call_discoverable_agent_tool 내부에서만). compliance 게이트로 deny+프로토콜 지시 →
            #   에이전트가 unlock→call_discoverable로 재발행(reroute 아님=스캐폴드 조작 회피·[[05]]/[[10]]).
            #   도메인일반: dispatch_tool 선언 도메인만·이름 suffix 휴리스틱(discoverable=랜덤 suffix).
            dd_fb = None
            # ★C241 U2'(리뷰 B3): banking 기본값을 지우는 것만으로는 **레버가 꺼지지 않는다** —
            #   `_unlock`/`_disp`는 아래 피드백 산문에 보간되므로 미선언 시 `"1) None(...)"`이
            #   모델에게 전달된다. 진입 조건이 `dispatch_tool` 하나뿐이라 나머지 두 키는 보호되지
            #   않았다. ⇒ **세 키(dispatch/unlock/list)가 다 선언된 도메인에서만** 이 레버를 켠다.
            #   banking은 3개 다 있으므로 행동 동일.
            _dd_keys = (ep_spec or {}) if ep_spec is not None else {}
            _dd_ready = all(_dd_keys.get(k) for k in ("dispatch_tool", "unlock_tool", "list_tool"))
            if (os.environ.get("T2_DISCOVERY_DISPATCH") == "1" and ep_spec is not None
                    and _dd_ready and ep_fb is None
                    and not do_gate and not do_prov
                    and getattr(self, "_t2_dd_deny", 0) < int(os.environ.get("T2_DD_CAP", "8"))):
                _disp = ep_spec.get("dispatch_tool")
                _unlock = ep_spec.get("unlock_tool")
                _nk = ep_spec.get("dispatch_name_key") or "agent_tool_name"
                _safe = {_disp, _unlock, ep_spec.get("list_tool")}
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", "") or ""
                    if nm not in _safe and re.search(r"_\d{3,4}$", nm) \
                            and id(c) not in denied_by_objid:
                        # ★C153: 처방적 구체성(C116) — 에이전트가 방금 쓴 실제 인자를 그대로 echo한
                        #   복사-가능 2단계 예시. 막연한 '{...}'은 포기 유발(C152 abandon)·구체 예시=재발행 유도.
                        _uarg = json.dumps(_args_dict(c)).replace('"', '\\"')
                        dd_fb = (c, "You called '%s' directly, but it is a DISCOVERABLE tool — a direct "
                                    "call is NOT registered and does NOT complete the step. Do NOT abandon "
                                    "this step. Redo it in TWO steps with the SAME arguments you just used:\n"
                                    "  1) %s(%s=\"%s\")\n"
                                    "  2) %s(%s=\"%s\", arguments=\"%s\")"
                                 % (nm, _unlock, _nk, nm, _disp, _nk, nm, _uarg))
                        break
            # ★v3.2 CONSISTENCY (T2_CONSISTENCY=1): L10 멤버십(t35형)+G-noop(t71형)·cap 2/sim
            cons_fb = None
            if (os.environ.get("T2_CONSISTENCY") == "1" and a2 is not None
                    and not do_gate and not do_prov and ep_fb is None
                    and cons_rounds < 1 and getattr(self, "_t2_cons_deny", 0) < 2):
                try:
                    _cspec = a2.get("eplan") or {}
                    _cwrites = _confirm_write_tools(a2)
                    for c in (am.tool_calls or []):
                        if getattr(c, "name", None) not in _cwrites:
                            continue
                        dargs = _args_dict(c)
                        mv = membership_violation(dargs, _cspec, state.messages)
                        if mv:
                            _bad, _oid, _hint = mv
                            cons_fb = (c, CONS_MEMBER_FEEDBACK.format(
                                bad=", ".join(_bad), ent=_cspec.get("entity_key"), oid=_oid,
                                hint=(" They appear in %s='%s'." % (_cspec.get("entity_key"), _hint))
                                     if _hint else ""))
                            print("[T2_CONS] membership deny tool=%s bad=%s oid=%s hint=%s"
                                  % (c.name, ",".join(_bad), _oid, _hint),
                                  file=_sys.stderr, flush=True)
                            break
                        # ★v6 선별(T2_CONS_NOOP=0으로 분리 가능): G-noop은 v4 nt2서 효과 0(발화1·결과불변)
                        if os.environ.get("T2_CONS_NOOP", "1") != "0" \
                                and noop_write(dargs, _cspec, state.messages):
                            cons_fb = (c, CONS_NOOP_FEEDBACK)
                            print("[T2_CONS] noop deny tool=%s oid=%s"
                                  % (c.name, dargs.get(_cspec.get("entity_key"))),
                                  file=_sys.stderr, flush=True)
                            break
                except Exception as _ce:
                    cons_fb = None
                    print("[T2_CONS] error (no-op): %r" % (_ce,), file=_sys.stderr, flush=True)
            # ★v5 L2R read-all (T2_READALL=1·NT2 §4-1): 첫 write 전 미열람 후보 read 강제
            ra_fb = None
            if (os.environ.get("T2_READALL") == "1" and ep_led is not None
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_rounds < 1
                    and getattr(self, "_t2_readall_deny", 0) < int(os.environ.get("T2_READALL_CAP", "2"))):
                try:
                    unread = readall_unread(getattr(ep_led, "listed", ()),
                                            getattr(ep_led, "examined", ()))
                    if unread:
                        _raw = _confirm_write_tools(a2)
                        for c in (am.tool_calls or []):
                            if getattr(c, "name", None) in _raw and id(c) not in denied_by_objid:
                                # ★A12 (범위 표면화·마스터 §6.1 A12 후반부): `detail_reader` 를
                                #   **목록으로도** 선언할 수 있게 렌더를 정본 `_tool_phrase` 에
                                #   맡긴다(문자열이면 그대로 = 현행 바이트 불변·사본 0·[[67]]).
                                #   한 이름을 지목하면 형제 enumerator 로 읽는 레코드에 **틀린 도구**를
                                #   대게 된다(085#1: 체킹 `btxn_*` 에 credit 도구 지목·[[25]]).
                                #   ⚠선언을 실제로 목록으로 바꾸려면 `t2_eplan_patch.note_read` 의
                                #     `tool_name == spec["detail_reader"]` 동치 비교도 **집합 소속**으로
                                #     같이 고쳐야 한다(그 파일은 이 그룹 소유 밖 — 보고서 §미수리 참조).
                                ra_fb = (c, READALL_FEEDBACK.format(
                                    reader=_epmod._tool_phrase(
                                        (a2.get("eplan") or {}).get("detail_reader")),
                                    ids=", ".join(unread)))
                                print("[T2_READALL] deny tool=%s unread=%s"
                                      % (c.name, ",".join(unread)), file=_sys.stderr, flush=True)
                                break
                except Exception as _re2:
                    ra_fb = None
                    print("[T2_READALL] error (no-op): %r" % (_re2,), file=_sys.stderr, flush=True)
            # ★TOOLERR teeth: (a) 같은-실패-인자 재발행 차단(class 무관·항상 안전) (b) recover면 조기 transfer 차단
            te_fb = None
            if (terr is not None and not do_gate and not do_prov
                    and ep_fb is None and cons_fb is None and ra_fb is None
                    and te_rounds < 1 and getattr(self, "_t2_toolerr_deny", 0) < 3):
                _sp, _tool, _fargs = terr
                _cls = _sp.get("class")
                _xfer = _transfer_tools(a2)
                for c in (am.tool_calls or []):
                    if id(c) in denied_by_objid:
                        continue
                    nm = getattr(c, "name", None)
                    dargs = _args_dict(c)
                    # (a) 동일 도구·동일 실패-인자 재발행 = 결정론적으로 또 실패 → deny(항상)
                    if nm == _tool and _fargs and dargs == _fargs:
                        te_fb = (c, TOOLERR_RECOVER.format(tool=_tool, hint=_sp.get("hint", ""))
                                 + " (You re-sent the SAME failing argument — you must change it.)")
                        print("[T2_TOOLERR] deny same-failed-args tool=%s" % nm,
                              file=_sys.stderr, flush=True)
                        break
                    # (b) recover-class인데 조기 transfer = 포기 차단(A2-판단부·Δspurious 계측대상)
                    if _cls == "recover" and nm in _xfer:
                        te_fb = (c, TOOLERR_RECOVER.format(tool=_tool, hint=_sp.get("hint", ""))
                                 + " (Do not transfer for a recoverable error — retry with a corrected argument.)")
                        print("[T2_TOOLERR] deny early-transfer(recover) tool=%s" % nm,
                              file=_sys.stderr, flush=True)
                        break
            # ★reference-filter (keystone·참조축·C77) — do_gate/do_prov와 *독립*.
            #   ⚠2026-08-19 이후: **치환을 하지 않는다** — 닫힌 술어 검사 + 거부 문면만(아래 주석).
            #   게이트가 *교정된* nested id를 확인하도록 do_gate 판정을 소비하는 fb-빌드(아래) 前에 실행.
            #   (배선버그 교정 2026-07-14: 前엔 T2_RESOLVE 가드의 not do_gate 안에 있어 dispute류
            #    confirm-게이트가 do_gate=True면 영영 미발화 → 사용자가 *틀린* 거래를 확인.)
            if (os.environ.get("T2_RESOLVE") == "1" and a2 is not None
                    and (a2 or {}).get("reference_filter")
                    and getattr(self, "_t2_reffilter", 0) < 3):
                try:
                    import t2_resolve as _rz_rf
                    _rf = _rz_rf.resolve_reference_filter(am, state.messages, a2, self, la, UserMessage)
                    if _rf.get("status") == "deny":
                        # ★2026-08-19 (사용자 결정·A안): **제자리 치환 폐기**. 엔진이 옳은 엔티티를 골라
                        #   인자에 써 넣으면 그 순간 측정 대상(모델이 무엇을 못 고르는가)이 사라진다([[62]]).
                        #   남기는 것은 **닫힌 술어 하나**뿐이다 — *지목한 id 가 이 대화에서 읽은 레코드에
                        #   있는가*. 없으면 거부하되 **옳은 값을 말하지 않고** 무엇을 하면 풀리는지만 준다([[64]]).
                        self._t2_reffilter = getattr(self, "_t2_reffilter", 0) + 1
                        _rtc = _rf.get("call")
                        # ⚠피드백 대상은 **그 호출 객체**다(`c` 는 다른 블록의 루프 변수 — 여기선 미정의일
                        #   수 있고, try 안 NameError 는 조용히 no-op 로 삼켜진다. C538 과 같은 함정).
                        if _rtc is not None and te_fb is None:
                            te_fb = (_rtc,
                                     "[REFERENCE] the %s you named does not appear in any record "
                                     "returned by the tools in this conversation. Re-read the records you "
                                     "already fetched and name a %s that appears in one of them; if none "
                                     "does, fetch the records first."
                                     % (_rf.get("param"), _rf.get("param")))
                        print("[T2_RESOLVE] deny reference-unmatched param=%s (치환 폐기·표면화 %s)"
                              % (_rf.get("param"), "배달" if _rtc is not None else "미배달"),
                              file=_sys.stderr, flush=True)
                except Exception as _rfe:
                    print("[T2_RESOLVE] reffilter error (no-op): %r" % (_rfe,),
                          file=_sys.stderr, flush=True)
            # ★compute 키스톤(§8·C81) — do_gate/prov와 독립·결정론 in-place silent-repair(정책-계산 param).
            #   §8-3: liability만(순+348)·provisional 드롭(net−4). 에이전트 제공값만·미확정=미개입.
            # ⛔2026-08-19 (사용자 결정·A안) — **이 기구는 폐기됐다.**
            #   엔진이 정책-계산 param 을 계산해 인자에 써 넣던 자리다. 두 가지가 동시에 틀렸다:
            #   ⑴ 여기서 나오던 값들(`customer_max_liability_amount`·`amount_difference`)은 **채점되는
            #      gold 인자** 그 자체다 — 엔진이 채우면 그 태스크에서 모델 기여가 0 이 되고
            #      측정 대상이 사라진다([[62]]).
            #   ⑵ 그 규칙의 상수가 **gold 로 맞춰졌다** — `bank_rule_fit.py` 가 `reward_info.action_checks`
            #      를 코퍼스로 임계를 훑었고, A2 주석이 축자로 *"T1=2(정책literal) 73.6% / T1=30(proxy)
            #      89.4%"* · *"gold-fit 확증"* 이라고 적어 두었다([[23]] 위반).
            #   실물 발화 근거: `bank_t7326_*_20260819q` 로그에
            #      `[T2_RESOLVE] compute silent-repair customer_max_liability_amount -1->50` **8회**(task_085).
            #   ⇒ A2 의 `compute_ops` 는 비웠고, 여기서는 **선언이 남아 있어도 실행하지 않는다.**
            #      정책의 표(예: 책임한도 구간)는 여전히 **문면으로 배달**하면 된다 — 금지된 것은
            #      엔진이 값을 *써 넣는* 것이지 정책을 *보여 주는* 것이 아니다.
            if (a2 or {}).get("compute_ops"):
                print("[T2_RESOLVE] compute_ops 선언 %d건 무시 — 값 산출 기구 폐기(2026-08-19·[[62]]/[[23]])"
                      % len((a2 or {}).get("compute_ops") or {}), file=_sys.stderr, flush=True)
            # ★T2_REF_ISO (2026-07-24 C124/C125): 참조-슬립 최소-문맥 격리 재선택 → 제자리 치환.
            #   WEV *앞* 배치(치환된 id 기준으로 증거 검사). cap=T2_REF_ISO_CAP(기본 8)·예외 no-op.
            if (os.environ.get("T2_REF_ISO") == "1" and a2 is not None
                    and (a2 or {}).get("ref_iso")
                    and getattr(self, "_t2_refiso", 0) < int(os.environ.get("T2_REF_ISO_CAP", "8"))):
                try:
                    _ref_iso_repair(self, la, UserMessage, state.messages, am, (a2 or {}).get("ref_iso"))
                except Exception as _rie:
                    print("[T2_REF_ISO] error (no-op): %r" % (_rie,), file=_sys.stderr, flush=True)
            # ★T2_WRITE_EVIDENCE (unified 배선·2026-07-19 028 포렌식): 증거(도구출력 token+id 공존)
            #   없는 선언-write deny. silent-repair(reffilter/compute) *뒤* 배치 = 교정된 최종 인자를 검사.
            #   무과금·turn당 1회·sim당 T2_WEV_CAP(기본 8) — E-PLAN cap 선례(불응 무한루프 방지·소진 후 통과).
            #   ★T2_WEV_ROUNDS (2026-07-24 C125·rall20 031.0 실측): turn당 1회 규칙의 구멍 — deny 후
            #   regen된 호출은 같은 턴서 무검사 커밋(1234 날조가 deny #2 직후 regen으로 통과·오프라인
            #   재실행은 deny 확인=술어 정상·배관 우회). 재검사 횟수 env화(기본 1=현행 불변·2=regen 1회 재검).
            # ★T2_PROCEDURE (2026-08-05·`t2_procedure.py`·A2 L3 `procedures`): 정책이 **순서를
            #   명령한** 절차만 A2가 index로 선언하고, 그 index 안의 호출에서만 선행 단계를 검사한다.
            #   차단은 선언이 허가할 때만(`enforce` + 정책 MUST 문장) — 허가 없는 절차는 표면화만.
            #   엔진에 도구명·필드명·숫자 0(테스트가 AST로 강제). 표적 = 스모크 051(요청 제출·분쟁
            #   이력을 건너뛴 채 승인 호출).
            # ★D0-a 선점 관측 (2026-08-05·`ABSENCE_DRIVEN_PROCEDURE_DESIGN_2026_08_05` §1):
            #   스모크 f `task_048`은 절차에 진입(`check_card_closure_eligibility`)했는데 라이브
            #   `[T2_PROCEDURE]`가 **closure에 대해 0건**이었다(4건 전부 CLI). 같은 인자로 오프라인
            #   재생하면 정상 deny가 난다 ⇒ 술어가 아니라 **배관**이 조용했다. 원인 후보는 이 블록의
            #   진입 조건 자체 — 앞선 레버가 그 턴의 피드백을 잡으면 절차는 **평가조차 되지 않는다**.
            #   그래서 술어는 **항상** 평가하고(순수함수·비용 0·거동 0), 못 뜬 턴엔 누가 선점했는지
            #   남긴다. `T2_TOOL_SIGNATURE_OBSERVE`가 V7 死경로를 확정한 것과 같은 방법이다([[08]]).
            proc_fb = None
            abs_fb = None
            rdd_fb = None   # ★T2_REQUIRE_DOC_DELIVER (2026-08-22): 정의 문서 전문 배달(생성-측·비커밋)
            tr_fb = None
            wd_fb = None
            fs_fb = None
            _procs = ((a2 or {}).get("procedures")
                      if (a2 is not None and os.environ.get("T2_PROCEDURE") == "1") else None)
            if _procs:
                _pchain = [("gate", do_gate), ("prov", do_prov), ("eplan", ep_fb),
                           ("cons", cons_fb), ("resolve_action", ra_fb), ("te", te_fb)]
                _pblocker = next((n for n, v in _pchain if v), None)
                _pcapped = (getattr(self, "_t2_proc_deny", 0)
                            >= int(os.environ.get("T2_PROCEDURE_CAP", "6")))
                try:
                    import t2_procedure as _PROC
                    # 개수-보존: `min_count` 노드(정책이 "첫 3회"처럼 세는 규칙)를 위해서다.
                    # Counter는 집합처럼 멤버십도 답하므로 나머지 판정은 그대로다.
                    _done = _executed_tool_counts(state.messages)
                    for c in (am.tool_calls or []):
                        _ar = _args_dict(c)
                        _also = {str(_ar.get(k)) for k in
                                 ("agent_tool_name", "user_tool_name", "discoverable_tool_name")
                                 if _ar.get(k)}
                        _dc = _PROC.decide(
                            _procs, _exact_tool_name(c), _ar, _done, also_names=_also,
                            unlocked=_unlocked_names(state.messages, a2),
                            pattern=((a2 or {}).get("discoverable_name_check") or {}).get("pattern"))
                        if _dc.get("verdict") == "deny" and _dc.get("notes"):
                            # ★호출-레벨 선점도 관측한다(2026-08-05 2차): 구판은 `denied_by_objid`를
                            #   술어 **앞에서** continue해, 다른 레버가 이 호출을 이미 막은 턴은
                            #   deny도 로그도 남기지 않았다 — 스모크 f의 048 침묵이 그 후보다.
                            #   거동은 그대로(여전히 건너뛴다), 사유만 남긴다.
                            if id(c) in denied_by_objid or _pblocker or _pcapped:
                                _why = ("call_denied" if id(c) in denied_by_objid
                                        else (_pblocker or "cap"))
                                # 거동 불변: 선점·소진·이미-막힌 호출은 종전대로 건너뛴다. 로그만 남는다.
                                print("[T2_PROCEDURE] would-fire but suppressed by=%s tool=%s "
                                      "missing=%s prohibited=%s"
                                      % (_why, _exact_tool_name(c),
                                         ",".join(_dc.get("missing") or []),
                                         _dc.get("prohibited") or "-"),
                                      file=_sys.stderr, flush=True)
                                if _why == "call_denied":
                                    continue          # 다른 호출은 계속 본다(턴 전체를 포기하지 않는다)
                                break
                            proc_fb = (c, _dc["notes"][0])
                            # ★2026-08-05: deny 뒤 **다음 재생성에서 누락 단계를 고정**한다(P1 기계 재사용:
                            #   디스패처 인자를 단일값 enum으로). 값은 선언에서 오고 엔진은 고르지 않는다.
                            #   ⚠단서(2026-08-05 2차 포렌식): 051의 "차단했는데 이행 안 함"은 **무효 판정**이었다
                            #   — break 가드에 `proc_fb`가 빠져 있어 그 문구가 모델에게 간 적이 없다(§1.5).
                            #   pin의 필요성은 배관을 고친 뒤 다시 재야 한다.
                            try:
                                _mn = (_dc.get("missing") or [None])[0]
                                _pp = _PROC.find_procedure(_procs, _exact_tool_name(c), _done)
                                _nd = next((n for n in ((_pp or {}).get("nodes") or [])
                                            if n.get("id") == _mn), None)
                                _tl = (_PROC._tools_of(_nd) if _nd else []) or []
                                if len(_tl) == 1:
                                    self._t2_proc_pin = ("call_discoverable_agent_tool",
                                                         "agent_tool_name", _tl[0])
                            except Exception:
                                pass
                            # ★A-7⑴ (2026-08-23·017): 금지 분기는 `missing=[]` 로 돌아온다 —
                            #   사유 칸이 비어 나가면 다음 포렌식이 "안 걸렸다"로 읽는다([[25]]).
                            print("[T2_PROCEDURE] deny %s missing=%s prohibited=%s"
                                  % (_exact_tool_name(c), ",".join(_dc.get("missing") or []),
                                     _dc.get("prohibited") or "-"),
                                  file=_sys.stderr, flush=True)
                            break
                except Exception as _pce:
                    proc_fb = None
                    print("[T2_PROCEDURE] error (no-op): %r" % (_pce,), file=_sys.stderr, flush=True)

            # ★D1′ 부재-구동 (2026-08-05·설계 §2·게이트=x86): 절차에 **들어와 놓고** 그 절차 쪽으로
            #   K턴 동안 아무 호출도 하지 않으면, 선언이 가진 체크리스트를 표면화한다. 차단이 아니라
            #   비커밋 피드백 1건이고, `verdict`도 `denied_by_objid`도 건드리지 않는다.
            #   x86 전수(194 sim·K=3): 발화 54회/29 sim · ▶유일 98.1% · **gold-밖 지목의 write 0** ·
            #   지목 도구의 **100%가 미-unlock** — 048 livelock에서 모델에게 없던 유일한 정보가 그것이다.
            #   ▶는 `is_mandatory`(=정책이 순서를 명령)일 때만 붙고, 동렬이면 목록만 준다([[10]]).
            # ★2026-08-18 연결: 예산은 **같은 말 반복**만 막는다(사용자 지적: 단순하게).
            #   구판은 sim 전체에 총 2회였고, 그 2회가 *유일한 ready 가 write* 인 구간에서
            #   소진되면(t7315 050 실측) 정작 두 **조회**가 열린 뒤에는 침묵했다.
            #   막는 것은 **루핑**뿐이다(사용자 지시 2026-08-18 축자: *"반복은 말 그대로 루핑이다.
            #   다른 시점에서 다시 부르는 걸 반복이라고 하면 안된다"* · *"DAG 로 중요 절차의
            #   상태를 확인하고 같은 상태로 반복적으로 돌아 오는걸 체크"*).
            #   ⇒ 술어를 **우리 문면이 아니라 걸음 자체**에 건다([[22]] 닫힌 술어): 상태 =
            #   그 절차 DAG 에서 **완료된 노드 집합**(`t2_procedure.checklist` 가 관측으로 낸다).
            #   같은 상태로 다시 오면 그 사이 한 걸음도 안 나간 것이고 그것이 루핑이다.
            #   한 걸음이라도 나가면 상태가 달라져 다시 말한다. 예산도 총량도 없다.
            #   ⚠문면 비교가 아니라 상태 비교라, 템플릿을 손봐도 판정이 조용히 바뀌지 않는다.
            _abs_seen = getattr(self, "_t2_proc_state_seen", None)
            if _abs_seen is None:
                _abs_seen = self._t2_proc_state_seen = set()
            if (_procs and abs_fb is None and proc_fb is None and not absent_fired
                    and os.environ.get("T2_PROC_ABSENT") == "1"):
                try:
                    import t2_procedure as _PROC
                    _done2 = _executed_tool_counts(state.messages)
                    _unl = _unlocked_names(state.messages, a2)
                    _pat = ((a2 or {}).get("discoverable_name_check") or {}).get("pattern")
                    _K = int(os.environ.get("T2_PROC_ABSENT_K", "3"))
                    # ★C3(2026-08-05·050/051 실측): **진입한 그 턴에** 체크리스트를 한 번 준다.
                    #   시간축 계량: 아무도 부르지 않은 도구의 첫 지목 위치 중앙값이 **대화의 0.63**
                    #   지점(남은 메시지 23)이고, 호출된 것은 0.35(남은 42)였다. 같은 두 read를 048은
                    #   **일찍 한 번** 받고 불렀고 050·051은 63~68% 지점에서 4~5회 받고도 안 불렀다.
                    #   K턴 침묵을 기다리는 것은 그 지연의 설계상 원인이므로, 절차가 열린 첫 순간에는
                    #   침묵 조건을 면제한다(sim·절차당 1회·표면화만·차단 아님).
                    _ann = getattr(self, "_t2_proc_announced", None)
                    if _ann is None:
                        _ann = self._t2_proc_announced = set()
                    for _p in _PROC.active_procedures(_procs, _done2):
                        _nt = {t for n in (_p.get("nodes") or []) for t in (_PROC._tools_of(n) or [])}
                        _entry = _p.get("id") not in _ann
                        if not _nt or (not _entry and _quiet_turns(state.messages, _nt) < _K):
                            continue
                        _dagk = (_p.get("id"),
                                 frozenset(_nid for _nid, _tls, _dn
                                           in _PROC.checklist(_p, _done2) if _dn))
                        if _dagk in _abs_seen:
                            continue                  # 같은 DAG 상태로 돌아왔다 = 걸음 0 = 루핑
                        _ann.add(_p.get("id"))
                        _msg = _PROC.absent_note(_p, _done2, _unl, _pat)
                        if not _msg:
                            continue
                        # ⚠증가는 **전달 자리**에서 한다([[55]] 로그 마크 != 전달). 여기서 올리면
                        #   하류에서 접히거나 다른 레버에 밀린 표면화도 예산을 먹는다.
                        self._t2_proc_absent_last = _dagk
                        abs_fb = _msg
                        absent_fired = True
                        # ★C15 read 강제 (2026-08-05·사용자 지시 "read 강제로 바로 가라"): 지목한
                        #   잔여 단계가 **환경이 read로 선언한** 유일 도구면, 다음 재생성의 채널을
                        #   그 호출로 고정한다(P1 기계 재사용: 디스패처 인자를 단일값 enum으로).
                        #   근거: 050/051은 같은 목록을 3~6회 받고서야 이행했고 048은 6회에도 못 했다
                        #   — 반복은 대책이 아니라 증상이다. write는 이 경로가 거부한다([[05]] §1.5).
                        if os.environ.get("T2_PIN_READ_STEPS") == "1":
                            try:
                                _st15 = _PROC.render_state(_p, _done2, _unl, _pat)
                                _cand15 = [t.strip() for t in
                                           str(_st15.get("ready_tools") or "").split(",")
                                           if t.strip()]
                                _env15 = getattr(getattr(self, "_t2_orch", None),
                                                 "environment", None)
                                _rd15 = [t for t in _cand15
                                         if t not in _done2 and _is_read_tool(_env15, t)]
                                # ★2026-08-18 연결: 구판은 **정확히 하나**일 때만 고정했다. 그런데
                                #   `credit_limit_increase` 는 제출 뒤 `disputes`·`pending_replacement`
                                #   **둘이 동렬로** 열린다 — t7315 replay 로 확인했고, 그래서 이 절차에서
                                #   read 강제는 한 번도 걸린 적이 없다(050 gold 3~6 이 세 팔 전부 MISS).
                                #   후보가 **전부 read** 면 그 집합을 다값 enum 으로 고정한다. 하나라도
                                #   write 가 섞이면 종전대로 침묵한다(§1.5 Q5: 쓰기 강제 금지).
                                if _rd15 and len(_rd15) == len(_cand15):
                                    _tgt15 = _rd15[0] if len(_rd15) == 1 else sorted(_rd15)
                                    self._t2_proc_pin = ("call_discoverable_agent_tool",
                                                         "agent_tool_name", _tgt15)
                                    print("[T2_PIN_READ_STEPS] pin target=%s" % (_tgt15,),
                                          file=_sys.stderr, flush=True)
                                else:
                                    print("[T2_PIN_READ_STEPS] no read-only target "
                                          "(reads %d of %d ready)"
                                          % (len(_rd15), len(_cand15)),
                                          file=_sys.stderr, flush=True)
                            except Exception as _p15:
                                print("[T2_PIN_READ_STEPS] error (no-op): %r" % (_p15,),
                                      file=_sys.stderr, flush=True)
                        print("[T2_PROC_ABSENT] surface %s quiet>=%d done=%s"
                              % (_p.get("id"), _K, _msg.split("(")[1].split(")")[0]
                                 if "(" in _msg else "?"), file=_sys.stderr, flush=True)
                        break
                except Exception as _pae:
                    abs_fb = None
                    print("[T2_PROC_ABSENT] error (no-op): %r" % (_pae,),
                          file=_sys.stderr, flush=True)

            # ★F5 전사 대조 (2026-08-05·설계 §·게이트=x90): 행 배열 인자에 **손-전사된 값**이
            #   원장(그 대화가 읽은 record dump)과 어긋나면 deny한다. 018 t0은 `rewards_earned`를
            #   1113으로 적었고(원장 487) 엔진이 그 값으로 **없는 불일치**를 만들어 여분 분쟁이 나갔다.
            #   x90 전수(194 sim): 발화 3건/2 sim · **gold 자신이 걸린 횟수 0**(오차단 0).
            #   엔진은 값을 고치지 않는다 — 어긋난 사실만 말하고 재발행은 모델이 한다([[10]]).
            _trs = (a2 or {}).get("transcription_check") or {}
            if (_trs and tr_fb is None and not do_gate and not do_prov
                    and os.environ.get("T2_TRANSCRIBE") == "1"
                    and getattr(self, "_t2_transcribe_deny", 0)
                    < int(os.environ.get("T2_TRANSCRIBE_CAP", "4"))):
                try:
                    import t2_transcribe as _TR
                    from t2_scaffold_get import _parse_record_dump as _PRD
                    _byid = {}
                    _specs = {k: v for k, v in _trs.items()
                              if not k.startswith("_") and isinstance(v, dict)}
                    for _m in state.messages:
                        if getattr(_m, "role", None) != "tool":
                            continue
                        try:
                            _rows = _PRD(str(getattr(_m, "content", "") or ""))
                        except Exception:
                            continue
                        for _r in _rows:
                            for _sp in _specs.values():
                                _rid = _r.get(_sp.get("id_key"))
                                if _rid:
                                    _byid[str(_rid)] = _r
                    for c in (am.tool_calls or []):
                        if id(c) in denied_by_objid:
                            continue
                        _sp = _specs.get(getattr(c, "name", None))
                        if not _sp:
                            continue
                        _bad = _TR.mismatches(_sp, _args_dict(c), _byid)
                        # ★관찰(2026-08-05): 발화 0이 "표적 없음"인지 "死배선"인지 구분되지 않아
                        #   스모크 j 판정을 못 냈다. 검사가 **돌았다는 사실**을 sim당 1회 남긴다.
                        if not getattr(self, "_t2_tr_seen", False):
                            self._t2_tr_seen = True
                            print("[T2_TRANSCRIBE] live tool=%s rows=%d records=%d"
                                  % (getattr(c, "name", None),
                                     len(_TR._rows(_args_dict(c).get(_sp.get("arg")))), len(_byid)),
                                  file=_sys.stderr, flush=True)
                        _msg = _TR.note(_trs.get("_feedback"), _bad, getattr(c, "name", None))
                        if not _msg:
                            # 원장에 없는 id = 계산의 입력이 날조된 것. 같은 채널·다른 문구.
                            _unk = _TR.unknown_ids(_sp, _args_dict(c), _byid)
                            _msg = _TR.note(_trs.get("_feedback_unknown"),
                                            [(u, "", "", "") for u in _unk],
                                            getattr(c, "name", None))
                            if _msg:
                                _msg = _msg.replace("{ids}", ", ".join(_unk[:6]))
                        if not _msg and not fs_fb:
                            # ★F31 필드-출처 표면화(2026-08-05·022 실측): op가 요구하는 필드가 행에
                            #   없고, 그 값이 **다른 이름으로** 다른 도구의 레코드에 있다.
                            #   022는 `account_open`을 77행 전부에서 빠뜨렸는데, 그 이름의 필드는
                            #   어떤 레코드에도 없다 — `date_of_account_open`(카드계좌)·`date_opened`
                            #   (예금계좌)가 실물이다. 이름 매핑은 선언이 쥘 수 있는 사실이므로
                            #   **어디서 가져오는지만** 말한다(엔진이 대신 가져오지 않는다).
                            _fs = _TR.field_sources(_sp, _args_dict(c), _byid,
                                                    _sp.get("require_fields") or ())
                            if _fs and _trs.get("_feedback_source"):
                                fs_fb = str(_trs["_feedback_source"]).replace(
                                    "{detail}", "; ".join(
                                        "%s -> %s.%s" % (f, tl, sf) for f, tl, sf in _fs[:4]))
                                print("[T2_FIELD_SOURCE] surface %s"
                                      % ",".join(f for f, _, _ in _fs[:4]),
                                      file=_sys.stderr, flush=True)
                        if not _msg:
                            # ★G3 전제: 입력이 깨끗했던 호출만 "확정 행"의 출처로 인정한다.
                            #   오염된 입력으로 나온 행은 가짜일 수 있고, x94 1차의 gold 반례
                            #   2건이 전부 그것이었다.
                            self._t2_clean_call = getattr(c, "id", None)
                        if _msg:
                            tr_fb = (c, _msg)
                            print("[T2_TRANSCRIBE] deny %s bad=%d first=%s"
                                  % (getattr(c, "name", None), len(_bad), _bad[0][:2]),
                                  file=_sys.stderr, flush=True)
                            _lbeat("T2_TRANSCRIBE", orch=self,
                                   target=_eff_tool_name(c), fact=_msg, order=_msg)
                            break
                except Exception as _tre:
                    tr_fb = None
                    print("[T2_TRANSCRIBE] error (no-op): %r" % (_tre,),
                          file=_sys.stderr, flush=True)

            # ★G3 확정-행 미제출 표면화 (2026-08-05·설계 `OPEN_PREDICATE_DECOMPOSITION` §2·게이트=x94):
            #   019 t1은 엔진이 확정한 3행 중 2행만 제출하고 하나를 **손님 산문에 설득당해 철회**했다.
            #   [[21]]("user-sim이 어떻게 반응해도 agent가 옳게")의 **닫힌 절반**이다 — 손님 문장은
            #   읽지 않고, 엔진 출력과 호출 이력만 본다.
            #   ⚠**F5 위에서만 건전하다**(x94 재판정): 오염된 입력으로 나온 확정 행은 가짜일 수 있고,
            #   실제로 gold 반례 2건이 전부 그것이었다. 그래서 **입력이 깨끗했던 출력의 행만** 센다.
            _wds = (a2 or {}).get("withdrawn_row_check") or {}
            # 깨끗했던 호출의 **결과**에서 id를 수확한다(엔진 출력 = 확정 행).
            if _wds.get("settle_tool"):
                try:
                    _cc = getattr(self, "_t2_clean_call", None)
                    _cl = getattr(self, "_t2_settled_clean", None)
                    if _cl is None:
                        _cl = self._t2_settled_clean = set()
                    # ★2026-08-05 패턴 제거: 확정 행을 **출력 텍스트에서 다시 찾지 않는다**.
                    #   엔진이 그 호출에서 계산한 목록을 그대로 받는다(t2_scaffold_get `_t2_sg_ids`).
                    #   ⚠이 자리의 옛 철자 규칙은 A2에서 JSON `\b`(=백스페이스)로 실려 있어 **한 번도
                    #   매치되지 않았다** — `T2_WITHDRAWN_ROW=1`인 채 확정 행 0으로 죽어 있었고,
                    #   x94는 자기 정규식을 따로 써서 그 사실을 못 봤다([[55]] 계기 부정통제).
                    _cl |= set((getattr(self, "_t2_sg_ids", None) or {}).get(_cc) or [])
                except Exception:
                    pass
            if (_wds.get("feedback") and wd_fb is None and tr_fb is None
                    and os.environ.get("T2_WITHDRAWN_ROW") == "1"
                    and not getattr(self, "_t2_wd_fired", False)):
                try:
                    _clean = getattr(self, "_t2_settled_clean", None)
                    if _clean is None:
                        _clean = self._t2_settled_clean = set()
                    _sub = set()
                    # ★제출 여부도 철자가 아니라 **멤버십**으로 본다: 제출 호출의 인자에 실린 값을
                    #   모아 엔진이 확정한 집합과 대조한다 — id의 생김새를 알 필요가 없다.
                    for _m9 in state.messages:
                        for _t9 in (getattr(_m9, "tool_calls", None) or []):
                            if _exact_tool_name(_t9) == _wds.get("submit_tool"):
                                _sub |= _arg_values(_args_dict(_t9))
                    _drop = sorted(_clean - _sub)
                    if _drop and _sub:
                        wd_fb = str(_wds["feedback"]).replace("{ids}", ", ".join(_drop[:6]))
                        self._t2_wd_fired = True
                        print("[T2_WITHDRAWN_ROW] surface dropped=%d" % len(_drop),
                              file=_sys.stderr, flush=True)
                except Exception as _wde:
                    wd_fb = None
                    print("[T2_WITHDRAWN_ROW] error (no-op): %r" % (_wde,),
                          file=_sys.stderr, flush=True)

            # ★G2-a 프로토콜 문서 미열람 표면화 (2026-08-05·설계 `OPEN_PREDICATE_DECOMPOSITION` §1·
            #   게이트=x93): 035는 신용정보국 사건인데 구매-거절 프로토콜 도구를 썼고, 032는 프로토콜
            #   없이 표준 이관했다. **어느 상황인가**는 열린 술어라 건드리지 않는다 — 닫힌 것은
            #   "이 도구를 정의한 문서를 읽었는가"뿐이고 그것만 말한다.
            #   x93 전수: 미열람 사용 27건 / **gold이 요구한 이관인데 미열람 6건** ⇒ **deny 금지**,
            #   표면화만. 도구→문서 지도는 코퍼스에서 실행시 도출(A2 아님).
            # ★C16(2026-08-05·048 q 실측): **이관을 시도하는 그 순간**에 미완 절차를 말한다.
            #   048은 손님이 두 번 이관을 요구했고(029·042) step 47에서 이관하며 끝났다 — 그때 gold
            #   7종이 미호출이었고, 우리가 그 순간 한 말은 이관 사유 등급 조언뿐이었다. 절차 미완은
            #   사임 턴이 아니라 **이 결정 시점**에 말해야 한다([[21]]: 손님이 요구해도 에이전트가 옳게).
            #   차단하지 않는다 — 무엇이 남았는지만 이름으로 말한다([[10]]).
            _pdc = (a2 or {}).get("require_doc_before") or {}
            if (_pdc.get("tools") and abs_fb is None and proc_fb is None
                    and os.environ.get("T2_TRANSFER_LEAVES_STEPS") == "1"
                    and not getattr(self, "_t2_tls_fired", False)):
                try:
                    import t2_procedure as _PROC16
                    _done16 = _executed_tool_counts(state.messages)
                    _unl16 = _unlocked_names(state.messages, a2)
                    for c in (am.tool_calls or []):
                        _tn16 = _exact_tool_name(c)
                        if _tn16 not in (_pdc.get("tools") or []):
                            continue
                        # ★C16 좁힘(2026-08-05·통과 sim 노출 계측): 035는 **이관이 정답**인 태스크이고
                        #   그 이관은 선언된 절차의 한 단계다 — 거기서 "이관은 그 단계들을 수행하지
                        #   않는다"고 말하면 통과를 깬다(노출 6회). 이관 자체가 선언된 단계이면 침묵한다.
                        _isstep16 = any(
                            _tn16 in {t for n in (_p16b.get("nodes") or [])
                                      for t in (_PROC16._tools_of(n) or [])}
                            for _p16b in _PROC16.active_procedures(_procs or [], _done16))
                        if _isstep16:
                            print("[T2_TRANSFER_LEAVES_STEPS] silent — transfer is a declared step",
                                  file=_sys.stderr, flush=True)
                            break
                        _left = []
                        for _p16 in _PROC16.active_procedures(_procs or [], _done16):
                            _st16 = _PROC16.render_state(_p16, _done16, _unl16, None)
                            # `ready_tools`는 쉼표 문자열이다(render_state 계약).
                            _left += [t.strip() for t in
                                      str(_st16.get("ready_tools") or "").split(",")
                                      if t.strip() and t.strip() not in _done16]
                        _left = sorted(set(_left))
                        # ★선언된 절차가 할 말이 없으면 **원장**을 본다 (2026-08-19·t7318 073).
                        #   그 sim 의 우리 로그엔 `walk gap: qty=9 executed=0` 이 있었는데, 이 자리는
                        #   절차 선언만 보고 침묵했다 — 073 에는 절차가 선언돼 있지 않고, 정책에
                        #   순서 문장이 없어 저작할 근거도 없다([[23]]). 원장은 선언 없이도 안다.
                        #   술어·수치는 walk 가 쓰는 것을 **그대로** 쓴다(사본 0·[[67]]).
                        _gapn = _gapm = 0
                        if not _left and ep_led is not None:
                            try:
                                import t2_eplan_patch as _EPL16
                                _unex16 = sorted(getattr(ep_led, "listed", set())
                                                 - getattr(ep_led, "examined", set()))
                                _gapn = _EPL16.walk_required_n(ep_led, _unex16)
                                _gapm = len({e.get("entity") for e in
                                             getattr(ep_led, "executed", []) if e.get("entity")})
                                if (_gapn <= 1 or _gapn <= _gapm
                                        or _EPL16.qty_item_covered(ep_led, _gapn)):
                                    _gapn = _gapm = 0          # walk 와 같은 억제 조건
                            except Exception as _le16:
                                _gapn = _gapm = 0
                                print("[T2_TRANSFER_LEAVES_STEPS] 원장 조회 건너뜀: %r" % (_le16,),
                                      file=_sys.stderr, flush=True)
                        if not _left and _gapn > _gapm:
                            abs_fb = ("Error: [WORK-INCOMPLETE] you are about to hand this "
                                      "conversation off, but this conversation's own record shows "
                                      "%d item(s) the customer asked about and %d you have actually "
                                      "acted on. A transfer does not perform the rest. Either do "
                                      "them now, or tell the customer plainly what you are leaving "
                                      "undone and why." % (_gapn, _gapm))
                            self._t2_tls_fired = True
                            print("[T2_TRANSFER_LEAVES_STEPS] surface ledger gap qty=%d executed=%d"
                                  % (_gapn, _gapm), file=_sys.stderr, flush=True)
                            break
                        if _left:
                            abs_fb = ("Error: [PROCEDURE-INCOMPLETE] you are about to hand this "
                                      "conversation off, but the procedure you entered still has "
                                      "steps nobody has done: %s. A transfer does not perform them. "
                                      "Either do them now, or tell the customer plainly which ones "
                                      "you are leaving undone and why." % ", ".join(_left[:5]))
                            self._t2_tls_fired = True
                            print("[T2_TRANSFER_LEAVES_STEPS] surface %d left" % len(_left),
                                  file=_sys.stderr, flush=True)
                            break
                except Exception as _tls:
                    print("[T2_TRANSFER_LEAVES_STEPS] error (no-op): %r" % (_tls,),
                          file=_sys.stderr, flush=True)

            # ★T2_REQUIRE_DOC_DELIVER (2026-08-22·정본 `T7336_FORENSIC_033_2026_08_22.md`·C592 x465·
            #   기본 OFF=바이트 동일): **같은 닫힌 술어**(선언 도구 시도 ∧ 정의 문서 미열람·새 판단 0·
            #   [[66]])에서 표면화 대신 **정의 문서 전문을 이 턴의 재생성 버퍼에 싣는다**. 격리 x465:
            #   일반 7/7 → 사슬 6/7 · 부정통제 0/7 ⇒ 원인은 미전달·레버는 전달뿐([[62]]②). deny 0
            #   (x93 gold-이관 6건 보호). 배선·동형성·반복 규율은 `_require_doc_deliver` 독스트링.
            #   ⚠`proc_fb`/`tr_fb` 가 선 턴엔 침묵(그 호출은 어차피 막혀 재생성된다 — 재료는 다음
            #     시도에 싣는다). `abs_fb`(LEAVES_STEPS) 와는 공존 — 그 문구는 "남은 단계" 진술이고
            #     이 재료는 그 단계를 정의한 문서라 모순이 아니다.
            if (_pdc.get("tools") and proc_fb is None and tr_fb is None
                    and os.environ.get("T2_REQUIRE_DOC_DELIVER") == "1"):
                try:
                    _rdd = _require_doc_deliver(self, a2, state.messages, am.tool_calls or [])
                    if _rdd:
                        rdd_fb = _rdd["text"]
                except Exception as _rdde:
                    print("[T2_REQUIRE_DOC_DELIVER] error (no-op): %r" % (_rdde,),
                          file=_sys.stderr, flush=True)

            # ★`rdd_fb is None` (2026-08-22): 배달이 나간 턴엔 *"검색하라"* 표면화를 비운다 — 같은 턴에
            #   *"검색하라"* 와 *"여기 있다"* 가 함께 가면 문구 모순([[55]]). 플래그 OFF 면 항상 None.
            if (_pdc.get("tools") and abs_fb is None and tr_fb is None and proc_fb is None
                    and rdd_fb is None
                    and os.environ.get("T2_REQUIRE_DOC") == "1"
                    and not getattr(self, "_t2_reqdoc_fired", False)):
                try:
                    _dd = os.environ.get("T2_KB_DOCS_DIR")
                    _seen_txt = _docs_seen(state.messages)
                    for c in (am.tool_calls or []):
                        _nm2 = _exact_tool_name(c)
                        if _nm2 not in (_pdc.get("tools") or []):
                            continue
                        _want = _docs_naming(_nm2, _dd)
                        if _want and not any(x and x in _seen_txt for x in _want):
                            abs_fb = str(_pdc.get("feedback") or "").replace("{tool}", _nm2)
                            self._t2_reqdoc_fired = True
                            print("[T2_REQUIRE_DOC] surface %s docs=%d unread"
                                  % (_nm2, len(_want)), file=_sys.stderr, flush=True)
                            break
                except Exception as _rde:
                    print("[T2_REQUIRE_DOC] error (no-op): %r" % (_rde,),
                          file=_sys.stderr, flush=True)

            wev_fb = None
            # ★A7 / OL-34 (2026-08-22 · t7336 마스터 §6.1 A7 · §5.5): 이 블록의 진입 술어에 있던
            #   `not do_gate and not do_prov` 가 **날조-차단 계열까지** 껐다.
            #   ⓐ `do_gate` 는 그 턴의 *다른* 호출에 붙는 정책 게이트 문구다(조언·`[POLICY GATE …]`).
            #      `do_prov` 도 마찬가지로 **그 한 호출**의 출처 재질의다. 둘 다 **호출-국소**인데
            #      술어는 **턴 전역**이라, 무해한 게이트 하나가 같은 턴의 *다른* 호출에 대한
            #      WAG/REF_VERIFY(=실행 차단)를 통째로 껐다.
            #   ⓑ 실측(074#1·[S]): `time_verified='2023-11-14 15:30:00 EST'` 날조가 그대로 나갔고
            #      그 sim 로그 703줄에 `WEV`/`WRITE_ARG`/`WRITE-GROUND` 가 **0건**·같은 턴
            #      `stop=other_lever(gate)`.
            #   ⇒ 게이트·prov 라운드에서는 **날조-차단 계열만** 돈다(`wag_specs`·`rv_specs`).
            #     조언 계열(WEV=선행-read 요구 · ARG_EMPTY · ASK_UNKNOWN_BOOL · HANDOFF)은
            #     종전대로 배제한다 — 그 턴엔 이미 조언이 하나 나가 있고, 조언 둘은 문구 모순이다([[55]]).
            #   ⚠[[70]] **무엇을 파는가** = Δspurious(과차단). 게이트가 이미 붙은 턴에 WAG/REF_VERIFY
            #     deny 가 **하나 더** 붙을 수 있다. 계수 = `[T2_WAG_DECOUPLED] fired …` 줄
            #     (tag=어느 계열 · phase=gate|prov). 그 줄이 0 이면 이 수리는 아무것도 안 판 것이고,
            #     늘어난 만큼이 판 것이다. 0 이 아닌데 태스크별 부호가 갈리면 [[70]] 절충 대상.
            #   ⚠cap 은 종전 그대로다(`T2_WEV_ROUNDS` 턴당 1 · `T2_WEV_CAP` sim당 8) — 새 예산 0.
            _fab_only = bool(do_gate or do_prov)
            _wev_live = bool(wag_specs or rv_specs) if _fab_only \
                else bool(wev_specs or wag_specs or rv_specs or ae_on)
            if (_wev_live and ep_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None and proc_fb is None
                    and wev_rounds < int(os.environ.get("T2_WEV_ROUNDS", "1"))
                    and getattr(self, "_t2_wev_deny", 0) < _wev_cap):
                try:
                    for c in (am.tool_calls or []):
                        if id(c) in denied_by_objid:
                            continue
                        # ★A7: prov 라운드의 **그 호출**은 `main_prov` 문구가 아래 `fb` 배타 체인에서
                        #   이긴다(순위 1) — 여기서 검사해 봐야 문구가 버려지고 cap 만 닳는다.
                        #   게이트-denied 호출은 바로 위 `denied_by_objid` 가 이미 건너뛴다.
                        if _fab_only and _pcall is not None and c is _pcall[0]:
                            continue
                        # ★C7(2026-08-05·053 근본원인): 증거는 **양쪽 히스토리**에서 찾는다.
                        #   `state.messages`는 에이전트가 본 것만 담아, **손님이 실행한** 도구의
                        #   출력(discoverable user tool)이 빠진다. 053에서 마지막 4자리는 오직
                        #   손님 실행 결과(step 32)에만 있었고, 그래서 우리는 "어떤 도구 출력에도
                        #   없다"고 11번 말했다 — 오프라인에서 같은 검사를 병합 히스토리로 돌리면
                        #   전 구간 통과다. `_executed_tool_names`가 이미 같은 원칙을 쓴다
                        #   ("양쪽이 센다 — discoverable 단계는 손님이 실행하는 일이 잦다").
                        _wev_msgs = state.messages
                        try:
                            _o7 = getattr(self, "_t2_orch", None)
                            if _o7 is not None:
                                _all7 = _o7.get_messages()
                                if _all7 and len(_all7) >= len(state.messages):
                                    _wev_msgs = _all7
                        except Exception:
                            _wev_msgs = state.messages
                        # ★A7: 게이트·prov 라운드에서는 **조언 계열은 건너뛴다**(선행-read 요구는
                        #   그 턴 이미 나간 게이트 문구와 경쟁 지시가 된다·[[55]]).
                        wd = None if _fab_only else _wev_deny_msgs(_wev_msgs, c, wev_specs)
                        _wtag = "T2_WRITE_EVIDENCE"
                        if not wd and wag_specs:
                            # ★값-grounding(§2bs·031): WEV(선행-read)와 별개 구멍=값-전사.
                            #   ★A7: **날조-차단 계열** — 게이트·prov 축과 분리돼 항상 돈다.
                            wd = _write_arg_ground_deny(_wev_msgs, c, wag_specs)
                            _wtag = "T2_WRITE_ARG_GROUND"
                        if not wd and ae_on and not _fab_only:
                            # ★T2_ARG_EMPTY(C419·x250 8/8): 필수 인자가 빈 문자열이면 이름을 댄다.
                            #   WAG 가 구조적으로 못 보는 자리다(:1149 *"값 없음 = skip"*).
                            wd = _arg_empty_deny(self, c, a2, ae_tools)
                            if wd:
                                _wtag = "T2_ARG_EMPTY"
                        if not wd and rv_specs:
                            # ★결정론 참조-검증기(C128/C129): 레코드 판별속성(merchant)이 손님
                            #   발화에 없으면 deny — 전사-슬립 8/8 검출·false-block 0·LLM 0.
                            #   ★A7: **날조-차단 계열** — 게이트·prov 축과 분리돼 항상 돈다.
                            wd = _ref_verify_deny(self, la, UserMessage, state.messages, c, rv_specs)
                            _wtag = "T2_REF_VERIFY"
                        if not wd and not _fab_only:
                            # ★N2b `T2_ASK_UNKNOWN_BOOL`(설계서 §2·기본 OFF): 닫힌 자료형(불리언·enum)
                            #   인자를 **모르면서 채운** 경우. 권위 소재로 판정한다([[52]]) — 인자명이
                            #   회수된 레코드의 필드면 레코드가 답하고, 아니면 손님만 아는 값이다.
                            #   001: 손님이 답하지 않자 `rho_bank_subscription: false`를 스스로 채웠다(gold=true).
                            try:
                                import t2_unknown_bool as _ub
                                _eff_n, _eff_a = _eff_tool_name(c), _args_dict(c)
                                _inner_a = _eff_a.get("arguments")
                                if isinstance(_inner_a, str):
                                    try:
                                        _inner_a = json.loads(_inner_a)
                                    except Exception:
                                        _inner_a = None
                                _unk = _ub.unknown_args(self, _eff_n,
                                                        _inner_a if isinstance(_inner_a, dict) else _eff_a,
                                                        state.messages)
                                if _unk:
                                    wd = _ub.feedback(_eff_n, _unk)
                                    _wtag = "T2_ASK_UNKNOWN_BOOL"
                                    from t2_lever_beat import beat as _ubeat
                                    _ubeat("T2_ASK_UNKNOWN_BOOL",
                                           ",".join(k for k, _ in _unk))
                            except Exception:
                                pass
                        if not wd and wag_specs and not _fab_only:
                            # ★N1 `T2_HANDOFF_ARG_GROUND`(설계서 §1·기본 OFF): give는 80회 중 75회가
                            #   **도구명만** 실어 보내고 값은 본문에 실린다 — 손님이 실행한 인자값의 90%가
                            #   에이전트 산문에 축자로 존재한다. 그래서 A2 give-측 규칙(P9/P10)이 검사할
                            #   것이 없다. 후보는 본문에서 뽑고 **판정은 위와 같은 함수**가 한다(A2 저작 0).
                            try:
                                import t2_handoff_ground as _hg
                                _hd = _hg.check(_write_arg_ground_deny, state.messages,
                                                getattr(am, "content", None), wag_specs,
                                                getattr(c, "name", None))
                                if _hd:
                                    wd = _hd
                                    _wtag = "T2_HANDOFF_ARG_GROUND"
                                    from t2_lever_beat import beat as _hbeat
                                    _hbeat("T2_HANDOFF_ARG_GROUND", getattr(c, "name", ""))
                            except Exception:
                                pass
                        if wd:
                            wev_fb = (c, wd)
                            _lbeat("T2_WRITE_EVIDENCE", orch=self, target=_eff_tool_name(c),
                                   fact=str(wd), order=str(wd))
                            # ★내부 도구명 로깅(§2ba 오귀속 교훈: per-도구 로그에 이름 필수)
                            _inner = _args_dict(c).get("agent_tool_name") or ""
                            print("[%s] deny tool=%s inner=%s"
                                  % (_wtag, getattr(c, "name", None), _inner),
                                  file=_sys.stderr, flush=True)
                            # ★A7 [[70]] 계측: 이 deny 는 **구판이면 안 나갔을** 것이다(게이트·prov
                            #   축에 묶여 있었으므로). 이 줄의 수 = 이 수리가 산 차단 = 판 것의
                            #   상한(Δspurious 후보). 0 이면 이 수리는 아무것도 안 판 것이다.
                            if _fab_only:
                                print("[T2_WAG_DECOUPLED] fired tag=%s phase=%s tool=%s"
                                      % (_wtag, "gate" if do_gate else "prov",
                                         getattr(c, "name", None)),
                                      file=_sys.stderr, flush=True)
                            break
                except Exception as _wve:
                    wev_fb = None
                    print("[T2_WRITE_EVIDENCE] error (no-op): %r" % (_wve,),
                          file=_sys.stderr, flush=True)
            # ★T2_HAVE_VALUE (C115): 값-실재인데 재요청 반복 → None-anchor 리마인더(W를 지금 호출).
            #   WAG(fab 값 차단)의 *반대* 케이스이므로 그 뒤에 배치(상호배타)·무과금·turn당1·sim당 cap.
            hv_fb = None
            if (hv_specs and not do_gate and not do_prov and ep_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None and wev_fb is None
                    and getattr(self, "_t2_havevalue_deny", 0)
                    < int(os.environ.get("T2_HAVE_VALUE_CAP", "3"))):
                try:
                    hv_fb = _have_value_reask_fb(am, state.messages, hv_specs)
                    if hv_fb:
                        print("[T2_HAVE_VALUE] reask-with-value → nudge (regen)",
                              file=_sys.stderr, flush=True)
                except Exception as _hve:
                    hv_fb = None
                    print("[T2_HAVE_VALUE] error (no-op): %r" % (_hve,),
                          file=_sys.stderr, flush=True)
            elif hv_specs and getattr(self, "_t2_hv_suppress_log", 0) < 8:
                # ★관측 전용(2026-08-05·거동 변화 0): 침묵이 "표적 부재"인지 "선행 레버에 밀림"인지
                #   구분이 안 됐다 — `[T2_HAVE_VALUE]`는 194 sim 전체에서 0회인데, x78 재생은
                #   task_040/t0의 **같은 턴에서 술어가 8회 성립**함을 보였다(값이 실재하고 재요청 중).
                #   상호배타 체인은 앞선 레버가 피드백을 내면 hv를 건너뛰므로, 건너뛴 사실과 그 원인을
                #   남긴다. 술어는 순수 함수라 추가 비용은 문자열 검사뿐이고 반환값은 쓰지 않는다.
                try:
                    if _have_value_reask_fb(am, state.messages, hv_specs):
                        _why = ("gate" if do_gate else "prov" if do_prov
                                else "eplan" if ep_fb is not None
                                else "cons" if cons_fb is not None
                                else "reask" if ra_fb is not None
                                else "term" if te_fb is not None
                                else "wev" if wev_fb is not None else "cap")
                        self._t2_hv_suppress_log = getattr(self, "_t2_hv_suppress_log", 0) + 1
                        print("[T2_HAVE_VALUE] would-fire but suppressed by=%s" % _why,
                              file=_sys.stderr, flush=True)
                except Exception:
                    pass
            # ★T2_GIVE_REQUIRED (2026-08-26·기본 OFF): 손님이 실행해야 하는 도구를 **안 넘겨줬다**.
            #   VALUE_ACQUIRE 의 형제이되 방아쇠가 다르다 — 그쪽은 *값을 재요청할 때*, 이쪽은
            #   **손님의 호출이 env 에 거절당했을 때**다. 근거·경계는 `_give_required_fb` 독스트링.
            #   같은 `hv_fb` 채널을 쓰고 앞선 레버가 말했으면 침묵한다(상호배타 보존).
            if (hv_fb is None and not do_gate and not do_prov and ep_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None and wev_fb is None
                    and os.environ.get("T2_GIVE_REQUIRED") == "1"
                    and getattr(self, "_t2_givereq_deny", 0)
                    < int(os.environ.get("T2_GIVE_REQUIRED_CAP", "2"))):
                try:
                    _gr = _give_required_fb(state.messages, self)
                    if _gr:
                        hv_fb = _gr
                        self._t2_givereq_deny = getattr(self, "_t2_givereq_deny", 0) + 1
                        print("[T2_GIVE_REQUIRED] deny (%d자) — 손님-측 도구 미전달"
                              % (len(_gr),), file=_sys.stderr, flush=True)
                        _lbeat("T2_GIVE_REQUIRED", orch=self,
                               fact="the tool the customer was told to run has not been handed over")
                    else:
                        print("[T2_GIVE_REQUIRED] 관측: 미전달 도구 없음 — 무발화",
                              file=_sys.stderr, flush=True)
                except Exception as _gre:
                    print("[T2_GIVE_REQUIRED] 건너뜀(무발화): %r" % (_gre,),
                          file=_sys.stderr, flush=True)
            # ★3단계 ③ T2_CALL_FORM_FIX (2026-08-26·기본 OFF) — **지목해도 안 되면 엔진이 부른다**.
            #   사용자 확정(축자): *"JSON 으로 호출 형식을 LLM 이 정하게 하고, 엔진이 검산해서 틀릴
            #   경우, 틀린이유와 부르는 방식을 정확하게 해서 LLM 에 알리고 다시 호출하게 하는건
            #   어떤가? **그래도 안되면, 엔진이 호출 형식을 바꿔서 직접 부르는 것이다.**"*
            #   ⇒ 경계는 **내용/형식**이다: 어느 도구를 어떤 인자로(내용)는 LLM 이 정하고 이 자리는
            #     그것을 **축자로 옮겨** 래퍼만 바꾼다. X 나 arguments 를 만들면 [[03b]] 위반이고
            #     래칫이 그 보존을 검정한다.
            #   ⚠상한(`T2_GIVE_REQUIRED_CAP`)까지 ②단계로 지목한 **뒤에만** 움직인다 — 순서가
            #     에스컬레이션의 전부다. sim 당 1회.
            #   ⚠선례: 이 엔진은 이미 `am.tool_calls` 에서 호출을 **제거**한다(`T2_FAB_STRIP`·
            #     `T2_STALE_STRIP`). 추가는 그 대칭 연산이다.
            if (os.environ.get("T2_CALL_FORM_FIX") == "1"
                    and getattr(self, "_t2_givereq_deny", 0)
                    >= int(os.environ.get("T2_GIVE_REQUIRED_CAP", "2"))
                    and not getattr(self, "_t2_formfix_done", 0)):
                try:
                    # ⚠이름을 `_fx` 로 쓰면 안 된다 — 같은 `unified()` 안에서 `_fx` 는 이미
                    #   **모듈 별칭**(`import t2_formalize_exec as _fx`)이고, 지역 대입이 그것을
                    #   가려 다른 분기를 UnboundLocalError 로 죽인다(죽은-레버 4호 부류).
                    #   `test_no_undefined_names.py` 가 이것을 배터리에서 잡아 런을 세웠다.
                    _cff = _call_form_repair(state.messages, self)
                    if _cff:
                        _cffname, _cffargs = _cff
                        from tau2.data_model.message import ToolCall as _TCfix
                        am.tool_calls = [_TCfix(id="t2formfix",
                                                name="give_discoverable_user_tool",
                                                arguments=_cffargs,
                                                requestor="assistant")] +                             list(getattr(am, "tool_calls", None) or [])
                        self._t2_formfix_done = 1
                        print("[T2_CALL_FORM_FIX] 엔진이 래퍼를 바꿔 호출한다 tool=%s 인자키=%s "
                              "(내용은 모델 것 축자)"
                              % (_cffname, sorted(_cffargs)), file=_sys.stderr, flush=True)
                        _lbeat("T2_CALL_FORM_FIX", orch=self, target=_cffname,
                               fact="the same call, in the form the environment accepts")
                    else:
                        print("[T2_CALL_FORM_FIX] 관측: 고칠 형식 없음 — 무발화",
                              file=_sys.stderr, flush=True)
                except Exception as _cffe:
                    print("[T2_CALL_FORM_FIX] 건너뜀(무발화): %r" % (_cffe,),
                          file=_sys.stderr, flush=True)
            # ★T2_VALUE_ACQUIRE (C119): 값 미실재 + give 미실행 → give 표면화 넛지(have-value 앞단계).
            if (va_specs and hv_fb is None and not do_gate and not do_prov and ep_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None and wev_fb is None
                    and getattr(self, "_t2_valacq_deny", 0)
                    < int(os.environ.get("T2_VALUE_ACQUIRE_CAP", "3"))):
                try:
                    _va = _value_acquire_fb(am, state.messages, va_specs, a2=a2,
                                            executed=_executed_tool_names(state.messages))
                    if _va:
                        # ★C9(2026-08-05·048·사용자 제안 "매핑을 알려주면 안 되나"): 값을 얻는 길만
                        #   말하면 048처럼 **아무 데도 쓰지 않을 값**을 열 메시지 동안 쫓는다. 그 값을
                        #   실제로 받는 호출이 무엇인지는 환경 시그니처에서 읽히므로 함께 말한다 —
                        #   표를 쓰지 않고 매번 환경에서 도출한다(리터럴 0·판단 0·표면화만).
                        try:
                            _arg9 = (va_specs[0] or {}).get("arg")
                            _cons9 = sorted(_arg_consumers(
                                getattr(getattr(self, "_t2_orch", None), "environment", None),
                                _arg9)) if _arg9 else []
                            if _cons9:
                                _va += (" In this domain the only call(s) that take '%s' are: %s"
                                        " — if you are not calling one of those, this value is not"
                                        " needed for what you are doing." % (_arg9, ", ".join(_cons9[:4])))
                                print("[T2_VALUE_ACQUIRE] consumers %s=%d" % (_arg9, len(_cons9)),
                                      file=_sys.stderr, flush=True)
                        except Exception as _c9e:
                            print("[T2_VALUE_ACQUIRE] consumers error (no-op): %r" % (_c9e,),
                                  file=_sys.stderr, flush=True)
                        hv_fb = _va   # hv_fb 채널 재사용(None-anchor 리마인더·상호배타)
                        self._t2_valacq_fired = True
                        print("[T2_VALUE_ACQUIRE] give-surfacing → nudge (regen)",
                              file=_sys.stderr, flush=True)
                except Exception as _vae:
                    print("[T2_VALUE_ACQUIRE] error (no-op): %r" % (_vae,),
                          file=_sys.stderr, flush=True)
            # ★T2_PARAM_CAP (§2br): 정책-캡 deny (A2 param_cap_check·054 실측)
            pc_fb = None
            _pcs = (a2 or {}).get("param_cap_check") or []
            if (os.environ.get("T2_PARAM_CAP") == "1" and _pcs
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None
                    and getattr(self, "_t2_paramcap_deny", 0)
                    < int(os.environ.get("T2_PARAM_CAP_CAP", "4"))):
                try:
                    for c in (am.tool_calls or []):
                        _pd = _param_cap_deny(self, la, UserMessage, state.messages, c, _pcs)
                        if _pd:
                            pc_fb = (c, _pd)
                            self._t2_paramcap_deny = getattr(self, "_t2_paramcap_deny", 0) + 1
                            print("[T2_PARAM_CAP] deny param over policy cap", file=_sys.stderr,
                                  flush=True)
                            break
                except Exception as _pce:
                    pc_fb = None
                    print("[T2_PARAM_CAP] error (no-op): %r" % (_pce,), file=_sys.stderr, flush=True)
            # ★T2_RESOLVE (통일 인터프리터·UNIFIED_OPERAND_A2 §7-3): per-operand 해소 디스패처.
            #   deny-kind(operator/membership/provenance) 통합 = L10+L3+operator 한 경로.
            #   개별 플래그(T2_CONSISTENCY/T2_PROV_ORIGIN) 대체용(driver가 상호배타 설정).
            rw_fb = None
            # ★계약 진입점 분리 (2026-08-07·arm-5 실측). 이 바깥 조건이 `T2_RESOLVE`만 보던 탓에
            #   그 안에 중첩된 **C1(`T2_SOURCE`)·C3(`T2_ARBITRATE`)가 껍데기와 함께 죽었다**:
            #   T2_* 120개를 끄고 두 계약만 켠 arm에서 마커 발화가 각각 **0**이었다. 계약이 독립된
            #   코드 단위가 아니라 기존 레버 **안의 가지**였다는 뜻이고, 그래서 "레버 5개만"이
            #   플래그로는 도달 불가였다. 진입 자격을 계약 플래그로 넓힌다 — 안쪽의 per-operand
            #   해소 루프는 여전히 `T2_RESOLVE` 전용이므로 **전부 켠 스택의 거동은 불변**이다.
            _contract_on = any(os.environ.get(_k) == "1" for _k in
                               ("T2_RESOLVE", "T2_FORCE_ACTION", "T2_ARBITRATE", "T2_SOURCE"))
            # ★계기 (2026-08-10·C407·사용자 지시 *"포렌식 원인 규명해서 12 pass 만들라"*):
            #   결정 재료(`_limit_reduce_text`)는 이 사슬을 **다섯 칸** 통과해야 나간다. 실패 sim
            #   에서 재료가 0회였는데 **어느 칸이 닫혔는지 로그에 없었다** — x239 가 넷째 칸(의도
            #   형식화)을 후보로 재현했지만 실패 sim 에서도 열려 있었다(가설 기각). 짐작을 한 번 더
            #   하는 대신 **칸마다 멈춘 이유**를 남긴다. 인쇄 전용이고 모델에 안 보인다([[55]]).
            _mgate = None
            _mgate_kind = None          # 라벨 문자열을 되파싱하지 않는다(구조로 남긴다)
            if not _contract_on or a2 is None:
                _mgate = "contract_off"
                _mgate_kind = "contract_off"
            elif not (not do_gate and not do_prov and ep_fb is None and cons_fb is None
                      and ra_fb is None and te_fb is None and wev_fb is None):
                _mgate = "other_lever(%s)" % ",".join(
                    n for n, v in (("gate", do_gate), ("prov", do_prov), ("eplan", ep_fb),
                                   ("cons", cons_fb), ("ra", ra_fb), ("te", te_fb),
                                   ("wev", wev_fb)) if v)
                _mgate_kind = "other_lever"
            elif not _resolve_cap_ok(self, state.messages, a2):
                _mgate = "resolve_cap(정체 %s회)" % getattr(self, "_t2_resolve_deny", "?")
                _mgate_kind = "resolve_cap"
            if _mgate:
                # ★패자 기록 (2026-08-23 · 사용자 지시 *"패자 기록 넣고 6단계까지 가라"*).
                #   구판은 **이긴 쪽만** 적었다(`stop=other_lever(prov)`). 그래서 439회 정지 중
                #   어느 것이 결정점을 막았는지 **사건별로 볼 수가 없었다** — x492 의 6단계가
                #   정확히 여기서 막혔다. 막힌 것은 언제나 아래 `t2_resolve` 해소·재료 배달
                #   경로이고, 남길 것은 *그때 그 경로가 무엇을 다루려 했나* 다.
                #   넷 다 **이미 손에 있는 상태**를 읽을 뿐이다 — 새 계산·새 술어·도메인 리터럴 0.
                #     calls    이 초안이 부르려던 도구(해소 루프가 다뤘을 대상)
                #     pending  미소비 배달물 크기(0 = 대기 중인 재료 없음)
                #     axes     아직 처리 안 된 결정 축 수(재료가 아직 빚인가)
                #     prose    초안에 도구가 없다(= 결정·사임 턴에 떨어진 정지인가 ← 최대 혐의)
                #   ⚠거동 불변: 인쇄 문자열만 길어진다. 기존 파서는 `stop=`·`turn=` 를 그대로 읽는다.
                try:
                    _lcalls = ",".join(sorted({_eff_tool_name(_c9)
                                               for _c9 in (getattr(am, "tool_calls", None) or [])})) or "-"
                    _lpend = len(str(getattr(self, "_t2_cp2_pending", "") or ""))
                    _ldone = getattr(self, "_t2_search_done", None) or set()
                    _lax = len([_g9 for _g9 in (((a2 or {}).get("policy_ontology") or {})
                                                .get("doc_index") or {}) if _g9 not in _ldone])
                    _lprose = not bool(getattr(am, "tool_calls", None))
                except Exception:
                    _lcalls, _lpend, _lax, _lprose = "?", -1, -1, "?"
                print("[T2_MATERIAL_GATE] stop=%s turn=%d calls=%s pending=%d axes=%d prose=%s"
                      % (_mgate, len(state.messages), _lcalls, _lpend, _lax, _lprose),
                      file=_sys.stderr, flush=True)
            # ★T2_MATERIAL_BYPASS (2026-08-16·`x335b`/C494 후속·기본 OFF) — **배달을 요구와 분리**.
            #
            # 무엇을 고치나. `_resolve_cap_ok` 는 *제자리걸음* 을 억제하려고 만든 상한인데(정체 3회),
            # 검색 에이전트의 **재료 배달**이 같은 관문 안에 갇혀 있어 함께 멎는다(두 배달 지점이
            # 모두 아래 블록 안이다). 실측: t7295 의 055 세 sim 에서 **결정점 전에 재료가 도착한
            # sim 0/3** 이고, 막은 자는 `now 미확정` 침묵 **하나가 아니라** 이 관문이었다
            # (t7297 수리 후에도 `resolve_cap` **97회** · `other_lever` 59회로 생존).
            #
            # 왜 분리가 옳은가. 둘은 성질이 다르다 — *요구*("이걸 하라")를 되풀이하면 소음이지만
            # (무제한으로 열었더니 같은 요건 100회·전진 0), *재료*는 아직 안 읽은 것이면 새 정보다.
            # 그리고 재료가 값을 산다는 것은 **격리로 쟀다**([[62]] ①): 055 checking 0/24 → **24/24**,
            # 재료가 없으면 24/24 전부 카탈로그 밖 상품명 **날조**(C494).
            #
            # 반복은 무엇이 막나(무제한 아님·[[57]]). ⑴`_t2_searchagent_fired < 3` 상한 그대로 ·
            # ⑵`t2_search` 자신의 축 중복 억제(*"요청 축 모두 처리됨 — 침묵"*) · ⑶같은 문자열이면
            # 재배달 안 함. 즉 **내용 기준**으로 세는 기존 논리를 재사용할 뿐 새 결정론은 0이다.
            #
            # ⚠범위: `resolve_cap` 으로 멎은 자리만 연다. `other_lever`(gate/prov)는 **여기서 열지
            #   않는다** — 그쪽은 양보가 아니라 **거부 본문에 재료를 합류**시키는 것이 옳고([[64]]),
            #   그건 별도 변경이라 따로 잰다.
            # ⚠[[62]] ③: 배달일 뿐이다. 무엇을 고를지 말하지 않고 도구를 지목하지 않는다.
            # ⚠[[57]] 부작용: 재료는 회당 ~10k 토큰이라 **지연을 판다**(t7296 실측 1.8×). 판정은
            #   ⓐ배선 → ⓑ결정점-전 도달률 → ⓒ성적 → **ⓓ지연·과행동** 순서로 상쇄를 같이 잰다.
            if (_mgate_kind == "resolve_cap"
                    and os.environ.get("T2_MATERIAL_BYPASS") == "1"
                    and os.environ.get("T2_SEARCH_AGENT") == "1"
                    and a2 is not None
                    and getattr(self, "_t2_searchagent_fired", 0) < 3):
                try:
                    _bp = _search_material(self, a2, state.messages)
                except Exception as _bpe:
                    _bp = ""
                    print("[T2_MATERIAL_BYPASS] 건너뜀(무발화): %r" % (_bpe,),
                          file=_sys.stderr, flush=True)
                if _bp and _bp != getattr(self, "_t2_cp2_said", None):
                    self._t2_searchagent_fired = getattr(self, "_t2_searchagent_fired", 0) + 1
                    _cp2_assign(self, _bp, "MATERIAL_BYPASS")
                    self._t2_cp2_said = _bp
                    print("[T2_MATERIAL_BYPASS] resolve_cap 우회 · 재료 %d자 배달" % len(_bp),
                          file=_sys.stderr, flush=True)
                elif _bp:
                    print("[T2_MATERIAL_BYPASS] 같은 재료 — 재배달 안 함",
                          file=_sys.stderr, flush=True)
            if (_contract_on and a2 is not None
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None
                    and _resolve_cap_ok(self, state.messages, a2)):
                try:
                    import t2_resolve as _rz
                    for c in (am.tool_calls or []):
                        if id(c) in denied_by_objid:
                            continue
                        rr = _rz.resolve_write(getattr(c, "name", None), _args_dict(c),
                                               state.messages, a2, self, la, UserMessage)
                        if rr.get("status") == "deny":
                            rw_fb = (c, rr["feedback"])
                            print("[T2_RESOLVE] deny tool=%s arg=%s reason=%s"
                                  % (getattr(c, "name", None), rr.get("arg"), rr.get("reason")),
                                  file=_sys.stderr, flush=True)
                            break
                    # ★Lever 4 pre-recommendation 검증(오추천 방지·user-실행 operand): 에이전트가
                    #   offer_tool로 action 제안 시 operand를 요구→formalize 검증·틀리면 교정. cap 2/sim.
                    if (rw_fb is None and getattr(self, "_t2_recommend_deny", 0) < 2):
                        _rvr = _rz.resolve_recommendation(am, state.messages, a2, self, la, UserMessage,
                                                          transfer_tools=_transfer_tools(a2))
                        if _rvr.get("status") == "deny":
                            rw_fb = (_rvr.get("call") or (am.tool_calls or [None])[0], _rvr["feedback"])
                            self._t2_recommend_deny = getattr(self, "_t2_recommend_deny", 0) + 1
                            print("[T2_RESOLVE] %s deny" % _rvr.get("reason", "recommendation-verify"),
                                  file=_sys.stderr, flush=True)
                    # ★action-required (turn-level): am이 회피(action_tool 미호출)면 operator 해소
                    #   GET→FIND(intent→도구)→execute|ASK. 조언/포기로 종결 금지.
                    # ★예산 폐지 (2026-08-07·사용자 지시 "예산 없애라. 예산이 주는 이익이 없다").
                    #   근거: 이 cap의 **해악은 측정됐고 이익은 측정된 적이 없다**([[56]] 근거 우세).
                    #   101/102 전수 부검 — 발화 3회가 turn 4·6·8에 전부 소진되는데 첫 요건이 충족되는
                    #   것은 turn 11 전후다. 그래서 요건 큐의 머리가 한 번도 바뀌지 않았고(`queue
                    #   advanced` 로그 0회), 환급 판정이 **발화 안에** 있는 탓에 예산이 바닥나면 환급도
                    #   받을 수 없었다 — 되돌아올 수 없는 상태. 그 결과 두 번째 요건(원장 조회)은
                    #   20 trial 내내 **한 번도 "지금 하라"가 되지 못했고**, 실제 조회는 2/20이었다.
                    #   over-action의 실측 사고(023 컨텍스트 초과)는 게이트별 cap이 아니라 **전역**
                    #   `T2_REGEN_BUDGET`이 막는 층이다 — 상한이 필요하면 거기 두는 것이 맞다.
                    #   미설정=무제한. 정수를 주면 종전처럼 그 수에서 멈춘다(되돌리기 경로 유지).
                    _adc = os.environ.get("T2_ACTION_DENY_CAP")
                    _adc = int(_adc) if (_adc or "").strip().lstrip("-").isdigit() else None
                    if (rw_fb is None and (_adc is None
                                           or getattr(self, "_t2_action_deny", 0) < _adc)):
                        # ★Lever 0(BANK_ACTIONREQ_PROBE_FORENSIC §3): action-required는 agent-실행
                        #   도구만 대상 — user-실행(apply/submit 등)은 에이전트가 못 부르므로 스퓨리어스.
                        #   ★실행 주체는 **env에서 도출**한다(2026-07-31·[[23]] 감사): 구판은 A2
                        #   `action_tool_executor` 맵을 읽었는데, 그 키의 주석이 출처를 축자로
                        #   **"gold `action_checks[].requestor`"**라 밝히고 있었다 = [[23]] 위반.
                        #   그런데 **gold를 볼 필요조차 없었다** — 에이전트가 스스로 부를 수 있는
                        #   도구면 agent-실행, 아니면 user-실행이라는 **인터페이스 구조**로 7/7
                        #   재현된다. ⇒ A2 키 삭제(opex −1)·엔진은 집합 소속만 본다(리터럴 0).
                        _agent_names = {getattr(t, "name", None)
                                        for t in (getattr(self, "tools", None) or [])}

                        # ★C4 역할 배선(2026-08-07): 실행 주체 판정을 **한 함수**로 보낸다.
                        #   T1(사실 모순)은 중재 대상이 아니라 제거 대상이다 — 우리 층 두 문구가
                        #   같은 명제에 반대 진리값을 말한 계열 A(`[ACTION]` "손님 도구다" ↔
                        #   `unified_regen` "네 도구다")가 이 판정이 복제돼 있어서 생겼다.
                        #   > 불변식 I1: 같은 명제에 다른 진리값을 말하는 턴 수 = 0
                        #   ⚠지금은 **거동 보존**을 위해 UNKNOWN을 종전대로 "user"로 떨어뜨린다.
                        #     설계서가 요구하는 "판정 불가면 문장을 뺀다"로 조이는 것은 별도 측정 단계다.
                        _envr = getattr(getattr(self, "_t2_orch", None), "environment", None)

                        def _exec_side(_n):
                            try:
                                import t2_role as _role
                                _r = _role.executor_of(_n, agent=self, env=_envr)
                            except Exception:
                                _r = None
                            if _r:
                                return _r
                            return "assistant" if _n in _agent_names else "user"
                        _acts = {t for t in ((a2 or {}).get("action_tools") or [])
                                 if _exec_side(t) == "assistant"}
                        # ★user-실행 분기 (2026-07-22 §2bo·rall9 023 실측): apply류(user-실행)는
                        #   기존에 스퓨리어스 방지로 필터만 되고 **아무 nudge도 없어**, 모델이 "에이전트-측
                        #   절차가 KB에 있을 것"이라는 거짓 전제로 8턴 검색-루프→transfer(C108 변형).
                        #   executor 맵(A2·기왕 선언)을 모델에 *전달*: intent가 user-실행 도구로 formalize
                        #   되고 그 도구가 아직 미실행이면 "고객이 실행하는 도구다·검색 중단·안내하라"
                        #   피드백. tool_choice 강제 없음(정답 행동=안내 텍스트)·cap=action_deny 공유.
                        _uacts = {t for t in ((a2 or {}).get("action_tools") or [])
                                  if _exec_side(t) == "user"}
                        # ⛔`T2_PENDING_DISCOVERED` **제거**(2026-08-23 · 수리 항목 R8-pending-disc-dead).
                        #   여기에 있던 "런타임 discoverable 손님 도구를 대기집합에 더한다" 블록은
                        #   한 번도 켜진 적 없는 배선이었고, 켰더라도 이 자리에서는 해가 이익보다
                        #   크다는 것이 **닫힌 계수**로 확정됐다. 되살리려면 아래 넷을 전부 뒤집어야 한다
                        #   (로컬 `sim_results/*.log.gz` 455개 전수 · 재현 검정 `test_actionreq_waitset_evidence.py`):
                        #   ⑴ **발화 0** — `[T2_PENDING_DISC]` **0줄**(error no-op 조차 0) ↔ 같은 자리의 숙주
                        #      `[T2_ACTIONREQ]` **12,323줄**. 런 스크립트는 전부 `T2_PENDING_DISCOVERED=0` 이고
                        #      `go_stack.sh` 에도 없다 ⇒ "켜져 있다고 믿게 만드는" 죽은 레버였다([[62]] · [[67]]).
                        #   ⑵ **종료 술어가 없다** — 대기집합의 유일한 제거 경로는 아래 `_uacts - _effall`
                        #      인데 `_effall` 은 `state.messages` 의 호출만 본다. **손님이 실행한 도구는 거기
                        #      없다**: `apply_for_credit_card` · `submit_referral` · `submit_transaction` 은
                        #      12,323/12,323 줄에 **전부** 남아 있고, 에이전트-측 래퍼인
                        #      `call_discoverable_user_tool` 만 5,278 로 빠진다(=제거 술어 자체는 살아 있다).
                        #      ⇒ discoverable 을 더하면 손님이 이미 실행한 뒤에도 영원히 pending 이고
                        #      넛지가 끝나지 않는다 — "인자 변화 없는 반복 억제"([[57]])의 정확한 형태다.
                        #   ⑶ **문면이 도메인 정책과 모순된다** — 이 자리가 내보내는 `user_action_feedback`
                        #      은 *"tell the customer to run {tool} themselves"* 인데, 도메인 정책 축자는
                        #      *"Just explaining isn't enough, you must use the
                        #      `give_discoverable_user_tool(discoverable_tool_name)` function"* 이다. discoverable 은
                        #      **먼저 건네야** 손님이 부를 수 있으므로, 대기집합에만 넣고 문면을 그대로 두면
                        #      우리 층이 **무엇을 하면 풀리는지를 틀리게** 말하게 된다([[64]] · 위 I1 불변식).
                        #   ⑷ **병목 표적은 이미 닿아 있다** — 막혀 있는 행동은 손님-측 실행이 아니라
                        #      에이전트-측 `give_discoverable_user_tool` 이고 그 이름은 정적 `action_tools`
                        #      **안에 있다** — 같은 455 로그에서 `formalized_target=give_discoverable_user_tool`
                        #      이 **291회** 찍혔다. 게다가 이 축은 reward 헤드룸이 0 이다
                        #      (`reports/facet_rft_2026/refute_2026_08_23/refute_6.json` claim 4 · 133/133 sim reward 0.0
                        #      · 진짜 구속조건은 상류 `open_bank_account_4821` 의 class 선택이다 · [[69]]).
                        #   ⚠제거는 **거동 보존**이다: 플래그가 모든 런에서 OFF 였으므로 지운 경로는
                        #     어느 런에서도 실행된 적이 없다(로그 0줄이 그 증명). 아래 `_uacts` · `_effall` ·
                        #     `[T2_ACTIONREQ]` 배선은 한 글자도 건드리지 않았다.
                        #   → 고아 선언: `t2_search.sub_tool_names()` 와 A2 `policy_ontology.tool_names_prompt` 은
                        #     이 블록이 유일한 소비자였다(라이브 `[T2_TOOL_NAMES]` 455 로그 전수 0회).
                        #     둘 다 이 항목의 파일 범위 밖이라 손대지 않고 기록만 남긴다.
                        _called = {getattr(c, "name", None) for c in (am.tool_calls or [])}
                        _tgt_pre = None      # ★공유 formalize(합집합 1회) — 아래 원 블록이 재사용(이중 서브콜 방지)
                        # ★C6 창 배선(2026-08-07·사용자 지시). `_agent_ending`은 **사임 턴만** 연다
                        #   (도구 0 또는 transfer만). 그것이 T6의 정체다 — 실패 198 sim 중 109(55%)가
                        #   우리 개입 0건이고, 101/102 부검이 같은 자리를 짚었다: ORDER는 turn 6 이후
                        #   침묵인데 우리 층은 turn 58까지 살아 있었고, 예산을 없애도 발화가 안 늘었다
                        #   (=cap이 아니라 창이 구속조건). 그런데 실제 write는 **산문**을 지난다 —
                        #   제출 유형이 직전 답변에 101 87/87 · 102 61/63.
                        #   ⇒ 창 = 사임 ∪ 행동 ∪ **지시**. 술어는 표적 이름의 등장뿐이다(산문 해석 0).
                        #   ⚠플래그 뒤에 둔다(`T2_WINDOW`): 창을 넓히는 것은 발화 기회를 늘리는 일이라
                        #     004형 "마지막 턴 소각" 위험이 있다. 미설정 = 종전 거동.
                        def _win_open():
                            if os.environ.get("T2_WINDOW") != "1":
                                return _rz._agent_ending(am, _transfer_tools(a2))
                            try:
                                import t2_window as _w
                                _k = _w.opened(am, sorted(set(_uacts) | set(_acts)),
                                               name_of=_eff_tool_name)
                                if _k:
                                    print("[T2_WINDOW] open=%s" % _w.why(_k),
                                          file=_sys.stderr, flush=True)
                                    return True
                            except Exception as _e13:
                                print("[T2_WINDOW] error (fallback): %r" % (_e13,),
                                      file=_sys.stderr, flush=True)
                            return _rz._agent_ending(am, _transfer_tools(a2))
                        if ((_uacts or _acts) and _win_open()):
                            _effall = {_eff_tool_name(tc) for m2 in state.messages
                                       for tc in (getattr(m2, "tool_calls", None) or [])}
                            _upending = sorted(_uacts - _effall)
                            if _upending or (_acts and not (_called & _acts)):
                                _tgt_pre = _rz.formalize_intent_tool(self, la, UserMessage,
                                                                    state.messages,
                                                                    set(_upending) | _acts)
                                _utgt = _tgt_pre
                                # ★침묵-사유 계측 (2026-08-12·batch4 010 trial0 [24] 부검: 창은
                                #   열렸는데 [ACTION]이 침묵한 턴의 원인을 로그로 특정할 수 없었다
                                #   — §4 "계기의 사각이 음성 관측으로 보인다". 발화는 아래에서
                                #   따로 로그되므로, 여기선 판정 재료만 남긴다. 행동 불변·print 1줄.)
                                print("[T2_ACTIONREQ] window=open pending_user=%s "
                                      "pending_agent=%s formalized_target=%s"
                                      % (_upending, sorted(_acts - _called), _tgt_pre),
                                      file=_sys.stderr, flush=True)
                                # ★P-A (2026-08-26·`TASK_072.md` §7-2 처방 구현·기본 OFF).
                                #   결함: `formalize_intent_tool` 이 **이 대화에 한 번도 안 나온**
                                #   손님-측 도구를 지목하면, 아래 `[ACTION]` 문면이 *"'X' 는 손님이
                                #   실행한다"* 고 말한다 — 참이지만 **이 대화와 무관**하고, 072 t0 에서
                                #   그 한 줄이 강제-행동 경로를 통째로 죽였다. 같은 site 를
                                #   `x505_TASK_073_t7348_perstep.md` §2.1 이 독립으로 지목했다.
                                #   빈도 실측(최근 12런·태그별): `formalized_target` 발화 **383건 중
                                #   29건(8%)** 이 궤적 축자 0회이고, 그중 **23건이 `submit_transaction`**
                                #   — 문서가 지목한 그 도구다. 태스크는 040(8)·085(6)·074(5)·057(5)·
                                #   063(4)·055(1) 로 **hard-0 여섯**에 걸친다(문서 추정 둘보다 넓다).
                                #   ⚠술어는 **집합 소속 + 축자 대조**뿐이다([[22]]·C45 동형) — 우리가
                                #     고르는 것이 없다. 이름이 대화에 있으면 종전대로 발화한다.
                                #   ⚠판단 0 — *무엇을 하라*는 말은 안 한다. 근거 없는 지목을
                                #     **안 하는 것**뿐이다([[64]] 이름을 못 대면 말하지 않는다).
                                # 침묵의 **자격** (2026-08-29 · per-step 포렌식).
                                #   P-A 는 *"대화에 없는 이름은 지목하지 않는다"* 인데, 072 에서
                                #   옳은 그 침묵이 016 에서는 대화를 막다른 곳으로 보냈다:
                                #   두 런(t7376·t7384) 다 두 seed 가 **인간 상담원 이관**으로 끝났고
                                #   통과 프레임인 `750` 발화가 23·12 -> 0 으로 사라졌다.
                                #   갈리는 것은 **그 시점에 에이전트가 직접 할 수 있는 일이
                                #   남아 있느냐**다. 코퍼스 전수(533 런·침묵 116 건·침묵 시점을
                                #   `[T2_SUBWIN] msgs=N` 으로 고정):
                                #       016  발화로 38 · 유지 3      <- 되찾을 자리
                                #       063  발화로  9 · 유지 6
                                #       072  발화로  0 · 유지 38     <- 한 건도 안 바뀐다
                                #       074  발화로  0 · 유지 13
                                #       040·055·057·085  전부 유지
                                #   072 는 침묵할 때마다 `apply_checking_account_credit_5829` 가
                                #   배달됐고 미호출이었다 = *"손님 시키지 말고 네가 해라"* 가 맞다.
                                #   016 은 38/41 자리에서 열린 도구가 **하나도 없었다**.
                                #   ⚠술어는 닫혀 있다(집합 차·[[22]]) - 도메인 낱말 0 · gold 무참조.
                                #   ⚠이관 도구를 따로 빼도 **수가 같다**(V1=V2) - 구분이 필요 없다.
                                _pa_on = os.environ.get("T2_ACTIONREQ_GROUNDED") == "1"
                                _pa_open = _delivered_unused_agent_tools(
                                    self, state.messages, a2) if _pa_on else None
                                if _pa_on and not _pa_open and _utgt and _utgt in _upending:
                                    print("[T2_ACTIONREQ] 침묵 안 함: 에이전트가 직접 할 수 있는 "
                                          "일이 남아 있지 않다 - 손님을 가리키는 것이 유일한 "
                                          "진행 경로다 (target=%s)" % _utgt,
                                          file=_sys.stderr, flush=True)
                                if (_pa_on and _pa_open
                                        and _utgt and _utgt in _upending):
                                    try:
                                        _seen_txt = []
                                        for _m9 in state.messages:
                                            _c9 = getattr(_m9, "content", None)
                                            if isinstance(_c9, str):
                                                _seen_txt.append(_c9)
                                            for _t9 in (getattr(_m9, "tool_calls", None) or []):
                                                _seen_txt.append(str(getattr(_t9, "name", "") or ""))
                                                _seen_txt.append(json.dumps(_args_dict(_t9) or {},
                                                                            ensure_ascii=False))
                                        if str(_utgt) not in chr(10).join(_seen_txt):
                                            print("[T2_ACTIONREQ] 침묵: formalized_target=%s 가 이 "
                                                  "대화 축자에 0회 — 근거 없는 지목은 하지 않는다 "
                                                  "(TASK_072 §7-2)" % _utgt,
                                                  file=_sys.stderr, flush=True)
                                            _utgt = None
                                    except Exception as _ge9:
                                        print("[T2_ACTIONREQ] grounded 검사 건너뜀: %r" % (_ge9,),
                                              file=_sys.stderr, flush=True)
                                if _utgt in _upending:
                                    # ★문구 축소 (2026-08-08·C334·라이브 부검). 구판은 두 가지를
                                    #   한 문장에 묶었는데 하나가 **과잉 일반화**였다: *"실행 절차를
                                    #   KB에서 찾을 필요가 없다"* 는 참이지만 거기 붙은 *"STOP
                                    #   searching"* 은 **추천의 근거를 찾는 조회까지** 금지한다.
                                    #   실측: 그 문장이 나간 직후 에이전트가 비교를 접고 **손님이
                                    #   보유한** 계좌 중에서 골랐다 — 정작 그 과제가 시험하는 것이
                                    #   *보유하지 않은 상품도 후보다* 였다(gold 미달·C329 099).
                                    #   우리 문구가 참이 아닌 것을 말하면 그건 우리 결함이다([[25]]).
                                    #   ⇒ 주장을 **누가 실행하는가**로 좁히고, 추천 뒤에 남은 일은
                                    #     모델 판단에 되돌린다(무엇이 부족한지는 말하지 않는다).
                                    _ufb = str((a2 or {}).get("user_action_feedback")
                                               or ("Error: [ACTION] '{tool}' is run by the CUSTOMER, "
                                                   "not by you. There is no agent-side procedure to "
                                                   "look up for running it, so do not search for one "
                                                   "and do not transfer for this. Once you have "
                                                   "everything your recommendation rests on, tell the "
                                                   "customer in your reply to run {tool} themselves "
                                                   "with their details, then confirm the result. If "
                                                   "something you would base that recommendation on is "
                                                   "still missing, get it first - this message is about "
                                                   "who runs the tool, not about skipping the work "
                                                   "behind the recommendation.")
                                               ).replace("{tool}", _utgt)
                                    # ★2026-08-03 (task_001 실측): "with their details"는 **어느 인자를
                                    #   말해야 하는지** 알려주지 않는다. 001: 에이전트가 카드는 정확히
                                    #   골랐는데(Gold=gold) 안내가 `{"card_type": …}` 하나뿐이라 손님이
                                    #   나머지를 자기 기억으로 채웠고 `rho_bank_subscription`을 틀려
                                    #   **그 한 필드로 0점**. 003(신용점수 미조회)과 같은 가족이다.
                                    #   ⇒ 그 도구의 **전체 인자 목록을 env 스키마에서 기계 도출**해 붙인다
                                    #   (도메인 리터럴 0·값 판단은 여전히 모델 몫).
                                    try:
                                        _envu = getattr(getattr(self, "_t2_orch", None),
                                                        "environment", None)
                                        _pn = []
                                        for _mm in ("get_user_tools", "get_tools"):
                                            _ff = getattr(_envu, _mm, None)
                                            for _tt in (_ff() or []) if callable(_ff) else []:
                                                if str(getattr(_tt, "name", "")) != _utgt:
                                                    continue
                                                _sc = _tt.openai_schema
                                                _fn = (_sc.get("function")
                                                       if isinstance(_sc.get("function"), dict) else _sc)
                                                _pn = list(((_fn.get("parameters") or {})
                                                            .get("properties") or {}).keys())
                                        if _pn:
                                            # ★rev2(사용자 지적): 문구는 **A2**(L1 도메인-일반),
                                            #   엔진은 인자 목록만 채운다. 미선언이면 목록만 실토
                                            #   (지시 없음 = 판단은 모델 몫·[[05]] Q2).
                                            _tpl_a = (((a2 or {}).get("axis_notes") or {})
                                                      .get("user_action_arglist")
                                                      or " Arguments of {tool}: {args}.")
                                            _ufb += _tpl_a.format(tool=_utgt, args=", ".join(_pn))
                                    except Exception:
                                        pass
                                    # ★T2 근거-등급 중재 (2026-08-06·사용자 지시 "무조건 근거를 확보한
                                    #   쪽이 우세하다"·정본 `CONFLICT_ARBITRATION_THEORY_2026_08_06`).
                                    #   102 실측: `gates[GB1].applies_to`에 `submit_referral`이 **이미**
                                    #   있는데 게이트가 한 번도 말하지 못했다 — 게이트는 *에이전트의 호출*에
                                    #   붙고 그 도구는 손님이 실행하므로 붙을 자리가 없다. 그 사이 이 push는
                                    #   *발화*에 붙어 늘 떴고, 손님이 5건을 제출했다(gold 1건).
                                    #   ⇒ 선언이 덮는 표적이면 **명령권은 게이트에 있다**: 게이트 술어는
                                    #   실행 원장(E1), push 표적은 formalize 산문(E5)이다.
                                    #   ⚠침묵이 아니라 **치환**이다(표적 이름 유지) — 지우면 012 재현
                                    #   (우리 deny가 일하던 문구의 트리거를 없앴다).
                                    # ★버그픽스(2026-08-07·20260807b/c/d 실측 7·7·3건): `_reqs`가
                                    #   아래 `T2_ARBITRATE` 블록 **안에서만** 대입되는데 :5656부터
                                    #   **밖에서** 쓰인다. 플래그가 꺼져 있으면 `UnboundLocalError`가
                                    #   나고, 감싸는 try/except가 그것을 삼켜 **호스트 레버(RESOLVE)가
                                    #   통째로 no-op**이 된다 — 로그엔 `[T2_RESOLVE] error (no-op)` 한 줄뿐.
                                    #   :5440 주석이 경고한 *"중첩된 계약이 껍데기와 함께 죽는다"* 의
                                    #   세 번째 형태이고, 이번엔 **껍데기 쪽이 죽는다**(방향이 반대).
                                    #   초기화 한 줄로 닫는다 — ARBITRATE가 켜져 있으면 거동 불변.
                                    _reqs = []
                                    _srctext = ""      # R8 억제 대상 문자열(이 턴에 [SOURCE]가 나가면 채워진다)
                                    if os.environ.get("T2_ARBITRATE") == "1":
                                        # ★C3 합병(2026-08-07): 하나를 고르지 않고 **덮는 요건을 전부**
                                        #   모아 한 번에 말한다. 구판은 첫 미충족 게이트만 돌려줬고,
                                        #   라이브에서 치환 24회가 **전부 같은 게이트**·뒤에 선 요건은
                                        #   **0회**였다(순수 우선순위가 하위를 굶긴다). 합병하면 뒤에
                                        #   선 요건이 굶지 않고, 같은 행동 앞의 다른 선행 read도
                                        #   밀려나지 않는다 — 명령은 하나, 사실은 합집합.
                                        try:
                                            import t2_dominance as _DOMm
                                            _reqs = _DOMm.requirements_for(
                                                a2, state.messages, _utgt,
                                                executed=_executed_tool_names(state.messages, a2),
                                                unwrap=_exact_tool_name)
                                        except Exception:
                                            _reqs = []
                                        # ★C1 출처(2026-08-07): 정책이 정하는 수량을 **근거 없이**
                                        #   단정하면 그 자리를 짚는다. 102 실측 — 원장에서 건수는
                                        #   정확히 셌는데(7) 한도는 어디서도 안 가져오고 "도달"로
                                        #   건너뛰어 **정답을 스스로 제외**했다. 원장에는 한도 필드가
                                        #   없다: 원장은 몇 건 썼는지, 문서는 몇 건까지인지를 말한다.
                                        #   뽑는 것은 LLM, 검증(인용한 doc이 실제 회수됐는가)은 엔진.
                                        #   ⚠출력은 **C3 합병 경로로 합류**시킨다 — 따로 내보내면
                                        #   같은 턴에 두 명령이 되어 T4b(슬롯 경합)를 재생산한다.
                                        _bad = []
                                        if os.environ.get("T2_SOURCE") == "1":
                                            try:
                                                import t2_source as _SRC
                                                _cl = _SRC.formalize_claims(
                                                    self, la, UserMessage, state.messages,
                                                    text=getattr(am, "content", None))
                                                _cp = _SRC.build_corpus(
                                                    state.messages,
                                                    env=getattr(getattr(self, "_t2_orch", None),
                                                                "environment", None),
                                                    agent=self, a2=a2)
                                                _bad = _SRC.unsourced_claims(_cl, _cp)
                                                if _cl:
                                                    print("[T2_SOURCE] claims=%d unsourced=%d"
                                                          % (len(_cl), len(_bad)),
                                                          file=_sys.stderr, flush=True)
                                                if _bad:
                                                    _lbeat("T2_SOURCE", orch=self, target=_utgt,
                                                           fact="%d claim(s) have no source in the "
                                                                "ledger or the retrieved documents"
                                                                % len(_bad))
                                            except Exception:
                                                _bad = []
                                        if _reqs or _bad:
                                            # ★덮어쓰기 → 병합 (2026-08-13·재판정런 010 전수 부검).
                                            #   `_ufb` 에는 이 시점까지 만들어 둔 **[ACTION] 소유권
                                            #   문장**이 들어 있다(*"...do not transfer for this ...
                                            #   tell the customer to run {tool} themselves"*). 구판은
                                            #   그것을 조건 없이 덮었고, `_reqs` 가 비고 `_bad` 만
                                            #   남는 턴(=선행 요건이 **다 충족된** 바로 그 결정 순간)엔
                                            #   빈 문자열이 되어 지시가 통째로 사라졌다.
                                            #   실측: 010 결정 턴 3/3 에서 소멸(`[T2_ARBITRATE] push
                                            #   dominated ... reqs= unsourced=1`), 유일한 통과 sim 은
                                            #   그 턴에 우리 층이 **거의 침묵**한 시행이었다.
                                            #   소유권(누가 실행하는가)과 요건/근거는 직교하므로
                                            #   둘 다 남긴다 — [[64]]: 무엇이 틀렸나와 무엇을 하면
                                            #   풀리나가 함께 있어야 한다.
                                            _mrg = _DOMm.merged_text(a2, _reqs, _utgt) if _reqs else ""
                                            _ufb = ((_ufb + "\n") if _ufb else "") + _mrg if _mrg \
                                                else _ufb
                                            # ★"아무 말 없이 deny 하니까 안 하는 거다"(사용자 지시
                                            #   2026-08-07). 실측이 그 진단을 지지한다: 상한을 풀었더니
                                            #   같은 단계 이름을 **106회** 반복했는데 모델은 그 도구를
                                            #   **한 번도 시도하지 않았다**. 이름을 다시 부르는 것은
                                            #   인자 변화가 아니다([[57]]).
                                            #   ⇒ 두 가지를 붙인다. 둘 다 **사실**이고 판단이 아니다:
                                            #     ⓐ 왜 미충족인가 — 시도한 적이 없는가, 아니면 시도했고
                                            #        env가 무엇을 돌려줬는가(축자·A2 failure_markers 기준)
                                            #     ⓑ 지금 할 수 있는 것 **전부** — C2 프런티어(선행이 모두
                                            #        충족된 노드). 하나를 고르는 것은 여전히 모델 몫이다
                                            #        ([[05]] Q2 — 목록은 사실, 선택은 유동 판단).
                                            try:
                                                import t2_precedence as _PC
                                                _dn = _executed_tool_names(state.messages, a2)
                                                # ★주체별 요건과 **이웃 두 조각을 정합시킨다**
                                                #   (2026-08-27·t7364 실측). `T2_READ_PER_ENTITY` 가
                                                #   *"그 주체로는 아직 안 돌았다"* 를 요구로 세웠는데,
                                                #   `_dn`·`_front` 는 여전히 이름만 보므로 같은
                                                #   메시지가 **"Steps that are possible right now:
                                                #   (none available)"** 를 붙였다. 축자(s1567 turn 38):
                                                #     *"Do that now with the real tool calls."* 두 줄 뒤
                                                #     *"…was called but has not succeeded yet"* ·
                                                #     *"…(none available)"*
                                                #   요구와 부정이 한 메시지에 같이 나가면 그 메시지는
                                                #   자기 원인을 지운다([[64]]·[[55]] 우리-문구 모순).
                                                #   ⇒ 그 read 를 **이 계산에서만** 미완으로 되돌린다.
                                                _pe_fams = {_PC._fam(x)
                                                            for r in (_reqs or ())
                                                            if "@" in str(r.get("id") or "")
                                                            for x in (r.get("satisfiers") or ())}
                                                if _pe_fams:
                                                    _dn = {n for n in _dn
                                                           if _PC._fam(n) not in _pe_fams}
                                                _front = _PC.frontier(_utgt, _dn,
                                                                      _PC.graph_for(a2, _utgt))
                                                # ★실패 사유를 **집합 뺄셈**으로 낸다 — 엔진이 도구 출력
                                                #   텍스트를 스캔하지 않는다([[59]]·hook이 1차판을 차단).
                                                #   시도(tool_calls 이름) − 성공(`_executed_tool_names`)
                                                #   = 불렀는데 성사되지 않은 것. 구조뿐이고 문면 0이다.
                                                #   축자 사유는 **이미 대화에 있어 모델이 본다** — 우리가
                                                #   다시 뜯어 옮길 이유가 없고, 뜯는 순간 규칙 위반이다.
                                                _tried = {_eff_tool_name(_c2) for _m2 in state.messages
                                                          for _c2 in (getattr(_m2, "tool_calls", None) or [])}
                                                _failed = _tried - _dn
                                                _steps = [s for r in _reqs for s in (r.get("satisfiers") or [])]
                                                _why = []
                                                for _s2 in dict.fromkeys(_steps):
                                                    if _PC._fam(_s2) in _pe_fams:
                                                        # ⛔여기서 뺄셈으로 이유를 내면 **거짓**이 된다 —
                                                        #   `_tried` 는 접미사 제거 이름이고 `_dn` 은
                                                        #   정확한 이름이라, 성공한 discoverable read 도
                                                        #   뺄셈에서 살아남아 *"was called but has not
                                                        #   succeeded yet"* 로 나온다(t7364 s1567 축자).
                                                        #   주체별 요건의 참인 이유는 하나뿐이다.
                                                        _who = sorted(
                                                            str(r.get("id") or "").split("@", 1)[-1]
                                                            for r in (_reqs or ())
                                                            if "@" in str(r.get("id") or "")
                                                            and _s2 in (r.get("satisfiers") or ()))
                                                        _why.append(
                                                            "%s has not been called for %s in this "
                                                            "conversation" % (_s2, ", ".join(_who))
                                                            if _who else
                                                            "%s has not been called for that party in "
                                                            "this conversation" % _s2)
                                                        continue
                                                    if _s2 in _failed:
                                                        _why.append("%s was called but has not succeeded "
                                                                    "yet - its result above says why" % _s2)
                                                    else:
                                                        _why.append("%s has not been called in this "
                                                                    "conversation" % _s2)
                                                _tpl2 = (((a2 or {}).get("arbitration") or {})
                                                         .get("why_options")
                                                         or "\nWhy it is still outstanding: {why}\n"
                                                            "Steps that are possible right now (any of "
                                                            "them, your choice): {options}")
                                                # ★프런티어를 **실호출 이름**으로 (2026-08-08·C300).
                                                #   우리가 올리던 `get_all_user_accounts_by_user_id`는
                                                #   그대로는 호출 불가다 — 이 env의 발견형 도구는
                                                #   `..._3847`처럼 접미사가 붙는다. 이름을 알려 주는
                                                #   `T2_DISCOVERY_NAMES`는 **도구를 하나도 안 부른 턴**
                                                #   에만 발화하므로(`_agent_ending`) 필요한 자리에 닿지
                                                #   않는다 — 두 런 연속 0회, 그 두 런 모두 계좌조회 누락.
                                                #   레지스트리는 **잠금 전에도** 정확한 이름을 갖고 있고
                                                #   (`_agent_discoverable` 주석 축자), 가족명 대조는 이미
                                                #   쓰는 술어다. 기계 도출이라 도메인 리터럴 0.
                                                try:
                                                    _reg12 = _agent_discoverable(
                                                        getattr(getattr(self, "_t2_orch", None),
                                                                "environment", None))
                                                    if _reg12:
                                                        _fixed12 = []
                                                        for _n12 in _front:
                                                            if _n12 in _reg12:
                                                                _fixed12.append(_n12)
                                                                continue
                                                            _hit12 = sorted(
                                                                x for x in _reg12
                                                                if _PC._fam(x) == _PC._fam(_n12))
                                                            _fixed12.append(_hit12[0] if len(_hit12) == 1
                                                                            else _n12)
                                                        if _fixed12 != list(_front):
                                                            print("[T2_CALLABLE_FRONTIER] %s -> %s"
                                                                  % (list(_front), _fixed12),
                                                                  file=_sys.stderr, flush=True)
                                                            _lbeat("T2_CALLABLE_FRONTIER", orch=self,
                                                                   target=_utgt,
                                                                   fact="a step was named in a form "
                                                                        "that cannot be called")
                                                        _front = _fixed12
                                                except Exception as _e12b:
                                                    print("[T2_CALLABLE_FRONTIER] skipped: %r" % (_e12b,),
                                                          file=_sys.stderr, flush=True)
                                                if _why or _front:
                                                    _ufb += _tpl2.format(
                                                        why="; ".join(_why) or "(no record)",
                                                        options=", ".join(_front) or "(none available)")
                                                # ★T2_CALL_FORM(C418·x249 16/16): 여기까지 만든
                                                #   `_ufb` 안의 **발견형 이름**을 부를 수 있는 형식으로
                                                #   바꾼다. 한 자리에서 하면 `do it with:`·`why`·
                                                #   프런티어 세 곳이 한꺼번에 정합해진다.
                                                if os.environ.get("T2_CALL_FORM") == "1":
                                                    try:
                                                        _cf = _call_form_map(
                                                            self,
                                                            getattr(getattr(self, "_t2_orch", None),
                                                                    "environment", None),
                                                            list(dict.fromkeys(
                                                                list(_steps) + list(_front))), a2)
                                                        for _k12, _v12 in sorted(
                                                                _cf.items(), key=lambda kv: -len(kv[0])):
                                                            if _k12 in _ufb and _v12 not in _ufb:
                                                                _ufb = _ufb.replace(_k12, _v12)
                                                        if _cf:
                                                            print("[T2_CALL_FORM] named %d step(s) in "
                                                                  "callable form" % len(_cf),
                                                                  file=_sys.stderr, flush=True)
                                                    except Exception as _e12c:
                                                        print("[T2_CALL_FORM] skipped: %r" % (_e12c,),
                                                              file=_sys.stderr, flush=True)
                                            except Exception as _e14:
                                                print("[T2_ARBITRATE] why/options skipped: %r" % (_e14,),
                                                      file=_sys.stderr, flush=True)
                                            # (상한·문턱 대조는 이 분기 **밖**으로 올렸다 — C324.
                                            #  `_reqs or _bad`가 거짓인 턴에도 표적은 살아 있고,
                                            #  그 턴에 산수가 못 나가서 한 sim이 통째로 침묵했다.
                                            #  아래 `_limit_reduce_text` 호출부가 정본이다.)
                                            if _bad:
                                                # ★R8 대상으로 **문자열을 기억해 둔다** — 이 턴에
                                                #   결정 블록이 나가면 아래에서 그대로 뺀다.
                                                # (재임포트: `_bad`≠[] 이면 6602 가 성공했으므로
                                                #  동적으론 안전하지만, 그 불변식에 기대지 않는다 —
                                                #  `_rz` 사고와 같은 부류의 구조를 남기지 않는다.)
                                                import t2_source as _SRC
                                                _srctext = _SRC.unsourced_text(a2, _bad)
                                                _ufb = ((_ufb + "\n") if _ufb else "") + _srctext
                                            print("[T2_ARBITRATE] push dominated target=%s reqs=%s "
                                                  "unsourced=%d"
                                                  % (_utgt, ",".join(r["id"] for r in _reqs),
                                                     len(_bad)), file=_sys.stderr, flush=True)
                                    # ★같은 말을 두 번 하지 않는다 (2026-08-07·사용자 지시
                                    #   "숫자 말고 논리적으로 막을 수 없나"). 실측이 정지 규칙을 준다:
                                    #   상한을 풀었더니 발화 **106회**가 전부 같은 요구(`log_verification`)
                                    #   였고 **turn 6~10** 다섯 구간에 몰렸다 = 턴당 ~21회. 턴을 넘나든
                                    #   반복이 아니라 **한 턴 안의 거부→재생성 루프**였고, 그동안 모델은
                                    #   요구된 도구를 **한 번도 시도하지 않았다**(verify_identity 호출 0).
                                    #   같은 입력에 같은 말을 다시 붙이는 것은 **구성상 무의미**하다 —
                                    #   [[57]]: 반복 억제는 '횟수'가 아니라 '인자 변화'로.
                                    #   ⇒ 지문 = (표적, 요건집합, 명령단계, 실행원장 크기, 손님 발화 수).
                                    #     하나라도 바뀌면 다시 말할 수 있고, 아무것도 안 바뀌면 침묵한다.
                                    #     숫자가 없고, 상한은 이제 원리적으로 물리지 않는다.
                                    # ★연기된 표적을 **기억한다** (2026-08-08·원장 C300·사용자 지시
                                    #   *"선행이 필요하면 선행부터 하고 실행하게 하면 되지 않나"*).
                                    #   거절만 하고 잊으면 모델이 그 행동을 하려던 **단 한 번의 순간**이
                                    #   거기서 소멸한다 — task_100이 정확히 그렇게 계좌조회를 잃었다.
                                    #   `how`에는 모델이 그때 시도한 호출을 **그대로** 담는다: 우리가
                                    #   프런티어에 올리는 이름은 접미사 없는 형태라 그대로는 호출 불가고
                                    #   (C300), 모델 자신의 문자열이 유일하게 호출 가능한 형태다.
                                    #   우리가 대신 실행하지는 않는다([[05]] Q3) — 되살리는 건 의도뿐.
                                    if _reqs and _utgt:
                                        try:
                                            _dfr = dict(getattr(self, "_t2_deferred", None) or {})
                                            _how = ""
                                            for _c9 in (am.tool_calls or []):
                                                if _eff_tool_name(_c9) == _utgt or \
                                                        getattr(_c9, "name", None) == _utgt:
                                                    _how = "%s(%s)" % (getattr(_c9, "name", ""),
                                                                       json.dumps(_args_dict(_c9),
                                                                                  ensure_ascii=False)[:200])
                                                    break
                                            _dfr[str(_utgt)] = {
                                                "requirement": "; ".join(r["predicate"] for r in _reqs),
                                                "how": _how or (_dfr.get(str(_utgt)) or {}).get("how", "")}
                                            self._t2_deferred = _dfr
                                        except Exception:
                                            pass
                                    # ★C324 (2026-08-08): 상한·문턱 대조는 **분기가 아니라 피연산자**에
                                    #   달린다. 구판은 `_reqs or _bad` 안에 있어서, 요건이 다 풀린 뒤의
                                    #   표적 턴(= 정작 손님이 실행 직전인 자리)에서 침묵했다. 한 sim은
                                    #   원장이 채워진 뒤 그 조건이 한 번도 참이 되지 않아 산수가 **0회**
                                    #   나갔고, 손님은 소진된 그룹을 골라 실행했다. 여기서는 표적이
                                    #   살아 있으면 피연산자가 있는 한 같은 문장을 싣는다.
                                    #   ⚠대가 = 발화가 늘어난다 = **Δspurious**(등대 §1.3: 부작용 없는
                                    #     레버는 없다). 그래서 아래 지문에 이 문장을 포함시켜, 내용이
                                    #     바뀌지 않는 한 다시 말하지 않게 한다([[57]] 인자 변화 규칙).
                                    # ★C330 (2026-08-08): **우리가 실제로 요구한 read**를 sim-범위로
                                    #   남긴다. P1 핀(`t2_pin_read`)의 수요 신호가 프록시 셋이었는데
                                    #   이 계열에서 셋 다 원리적으로 못 뜬다는 것이 전수로 확인됐다 —
                                    #   ⒜의존 도구가 **손님 실행**이라 assistant 호출 집합에 영영 없고,
                                    #   ⒝가 찾는 태그는 현 어휘에 없는 문자열인 데다 우리 통지는
                                    #   비커밋이라 `messages`에 아예 나타나지 않으며, ⒞의 관용구는
                                    #   env 실제 문구와 어긋난다. 그 결과 핀은 표적을 정확히 해소할 수
                                    #   있는 상태로 **한 번도 시도되지 않았다**(라이브 발화 0).
                                    #   프록시를 늘리는 대신 **원천**을 준다: 요건 큐가 이 턴에 이름으로
                                    #   밀고 있는 read가 곧 수요다. 새 A2 0·새 문자열 0.
                                    #   ⚠**큐의 머리일 때만** 기록한다. 오프라인 재현에서 이 조건을
                                    #     빼면 첫 발화(신원확인이 아직 머리인 시점)에 계좌 read가
                                    #     고정돼 **우리 게이트(검증 우선)를 우리가 위반**시키고,
                                    #     1회뿐인 핀도 거기서 탄다. 머리로 좁히면 라이브 로그의
                                    #     `queue advanced … -> reads:…` 시점 = 그 read가 실제로
                                    #     "지금 할 일"이 된 턴에서만 무장한다.
                                    #   ★C331(사용자 지적): 요건 종류로 가르지 않는다. 그래프는
                                    #     게이트든 선행 read든 **satisfier 하나**로 특정해 준다
                                    #     (GB1→verify_identity · GB3→get_referrals_by_user ·
                                    #      reads:→get_all_user_accounts_by_user_id). 짐작이 0이므로
                                    #     "무엇을 강제할지"는 여기서 이미 끝났다. 남는 배제는 성질뿐
                                    #     (상태를 바꾸는가·손님이 실행하는가)이고 그 판정은 소비자가 한다.
                                    try:
                                        _head = (_reqs or [None])[0]
                                        _sats = [str(x) for x in ((_head or {}).get("satisfiers") or ())]
                                        if len(_sats) == 1:          # 유일하게 특정될 때만
                                            self._t2_demanded_step = _sats[0]
                                            print("[T2_DEMANDED_STEP] head=%s → %s"
                                                  % ((_head or {}).get("id"), _sats[0]),
                                                  file=_sys.stderr, flush=True)
                                        else:
                                            self._t2_demanded_step = None
                                    except Exception as _dre:
                                        print("[T2_DEMANDED_STEP] skipped (no-op): %r" % (_dre,),
                                              file=_sys.stderr, flush=True)
                                    try:
                                        _add = _limit_reduce_text(self, a2, state.messages)
                                    except Exception as _lre:
                                        _add = ""
                                        print("[T2_LIMIT_REDUCE] skipped (no-op): %r" % (_lre,),
                                              file=_sys.stderr, flush=True)
                                    if _add:
                                        # ★R8 — 결정 블록이 나가는 턴에는 `[SOURCE]` 재검색 명령을
                                        #   함께 보내지 않는다(C373: 실패한 099 에만 2회 동반·
                                        #   통과한 100 은 0회). 블록이 이미 인용 있는 정책 상수를
                                        #   싣고 있어 *"문서를 찾아라"* 는 그 턴에 충족돼 있다.
                                        if getattr(self, "_t2_decided", False) and _srctext:
                                            _n8 = _ufb.count(_srctext)
                                            if _n8:
                                                _ufb = _ufb.replace(_srctext, "").strip()
                                                print("[T2_R8] 결정 블록과 동반할 [SOURCE] "
                                                      "재검색 명령 %d건 억제" % _n8,
                                                      file=_sys.stderr, flush=True)
                                        # ★액션 서브 (2026-08-10·`T2_ACTION_SUB`·설계서 §2).
                                        #   여기서 지시(`_ufb`)와 값(`_add`)이 **한 메시지로 합쳐진다**
                                        #   — x224 가 잰 그 배치다. x228 실측: 같은 지시라도 메인
                                        #   문맥에 두면 소유권 발화가 098 0/6·100 1/6 인데, **격리
                                        #   문맥**(손님 발화 + 값 + 소유자 표기)에서 지으면 6/6·5/6
                                        #   이고 `external` 위반이 6/6 → 0/6 으로 사라진다.
                                        #   ⇒ 발화를 지을 자리만 옮긴다. 고르는 것은 여전히 LLM 이고
                                        #     엔진은 문맥 조립만 한다(⛔0 ②·새 결정론 0).
                                        #   재료는 호출부(재생성 루프)가 쓴다 — 여기서는 넘겨만 준다.
                                        try:
                                            self._t2_asub = {"value": _add, "tool": str(_utgt),
                                                             "args": list(_pn or [])}
                                        except Exception:
                                            self._t2_asub = None
                                        _ufb = ((_ufb + "\n") if _ufb else "") + _add
                                        print("[T2_LIMIT_REDUCE] emitted at decision point",
                                              file=_sys.stderr, flush=True)
                                        _lbeat("T2_LIMIT_REDUCE", orch=self, target=_utgt,
                                               fact="arithmetic against the allowances "
                                                    "and minimums you retrieved")
                                    try:
                                        _nuser = sum(1 for _m in state.messages
                                                     if getattr(_m, "role", None) == "user")
                                        _sig = (str(_utgt),
                                                tuple(r["id"] for r in (_reqs or [])),
                                                tuple(s for r in (_reqs or [])
                                                      for s in (r.get("satisfiers") or [])),
                                                len(_executed_tool_names(state.messages, a2)),
                                                _nuser,
                                                _add)
                                    except Exception:
                                        _sig = None
                                    if _sig is not None and _sig == getattr(self, "_t2_arb_sig", None):
                                        print("[T2_ARBITRATE] identical demand suppressed "
                                              "(nothing changed since it was last said)",
                                              file=_sys.stderr, flush=True)
                                        # ★삼분 계측 (설계 v1.5 §5.1 — 판정 조건). 억제되면
                                        #   `_ufb=""` → `rw_fb=None` 이고, 배타 체인 계측은
                                        #   `None` 을 **미생성과 구별하지 못한다**(:_cands 는
                                        #   `_v9[0] is c` 를 요구한다). 그래서 억제는 여기서만
                                        #   기록할 수 있다 — 여기서 안 세면 다음 런의 `lost_to`
                                        #   판정이 성립하지 않는다(억제↔체인↔미생성 삼분 불능).
                                        #   ⚠거동 불변: 목록에 담기만 한다.
                                        try:
                                            _sl = list(getattr(self, "_t2_silenced", None) or [])
                                            _sl.append({"agent": "resolve_write",
                                                        "target": str(_utgt),
                                                        "text": _add})
                                            self._t2_silenced = _sl
                                        except Exception:
                                            pass
                                        _ufb = ""
                                    elif _sig is not None:
                                        self._t2_arb_sig = _sig
                                    rw_fb = ((am.tool_calls or [None])[0], _ufb) if _ufb else None
                                    self._t2_action_deny = getattr(self, "_t2_action_deny", 0) + 1
                                    # ★큐 전진 환급(2026-08-07·task_101 부검). 요건이 셋인데 발화
                                    #   예산은 캡 하나를 공유한다 — 101은 신원 확인에서 헤매느라
                                    #   6회를 **전부 첫 요건에** 쓰고, 그것이 충족된 뒤에는 채널이
                                    #   닫혀 두 번째 요건(원장 조회)이 **명령조차 되지 않았다**.
                                    #   순응했는데 다음을 말할 자리가 없는 것은 잔소리 억제가 아니라
                                    #   설계 오류다. 그래서 **명령한 요건이 실제로 충족되면 환급**한다
                                    #   (선례 = 이미 켜져 있는 T2_ACTION_PROGRESS_REFUND).
                                    #   불응하면 충족이 없으므로 환급도 없다 = 무한 반복 불가.
                                    if _reqs:
                                        _head = _reqs[0]
                                        _prev = getattr(self, "_t2_arb_head", None)
                                        if _prev and _prev != _head.get("id"):
                                            self._t2_action_deny = max(
                                                0, getattr(self, "_t2_action_deny", 0) - 1)
                                            print("[T2_ARBITRATE] queue advanced %s -> %s (refund)"
                                                  % (_prev, _head.get("id")),
                                                  file=_sys.stderr, flush=True)
                                        self._t2_arb_head = _head.get("id")
                                    print("[T2_RESOLVE] user-action instruct target=%s" % _utgt,
                                          file=_sys.stderr, flush=True)
                        # ★C17 단계 소유권 (2026-08-05·050 실측·사용자 질문 "일반화된 문제인가"):
                        #   같은 구간에서 GB1은 *"먼저 신원을 검증하라"*, 이 레버는 *"지금 전용 도구
                        #   발견 체인을 돌려라"* 를 보냈다. 050은 그 사이 `verify_identity`(이미 가진
                        #   표준 도구)를 KB에서 세 번 찾다 24턴을 태웠고, 환경조차 *"already one of the
                        #   tools provided"* 라고 답했다. 두 문구 다 국소적으로 옳아서 문구를 고쳐도
                        #   조합이 바뀌면 재발한다 — 그래서 **단계를 소유한 쪽만 말한다**:
                        #   선언된 auth 게이트가 미충족이면 검증이 그 단계의 주인이고, 행동-유도는 쉰다.
                        #   술어는 A2 선언(satisfier 실행 이력)뿐이고 도메인 어휘도 판단도 없다.
                        #   경계는 측정으로 정했다: "검증을 **시도했는가**"까지 조건에 넣으면 손해
                        #   9건만 지워지고 **통과 sim 노출은 0**이다(시도 전에는 계속 말한다).
                        _phase17 = "open"
                        try:
                            import t2_phase as _PH17
                            _phase17 = _PH17.phase_of(
                                a2, state.messages, _exact_tool_name,
                                executed=_executed_tool_names(state.messages))[0]
                        except Exception:
                            _phase17 = "open"
                        # ★T5 억제 자격(2026-08-06): 단계 소유권도 **다른 레버를 침묵시키는** 레버다.
                        #   근거를 A2에 선언해야 하고, 못 대면 침묵시키지 않는다(C13이 그 자격 없이
                        #   050/051의 이행을 만들던 반복을 지웠다).
                        try:
                            import t2_authority as _AUTHm
                            _may17 = _AUTHm.may_suppress(a2, "phase_owner")
                        except Exception:
                            _may17 = True
                        _off_phase = (_phase17 == "verify" and _may17
                                      and os.environ.get("T2_PHASE_OWNER") == "1")
                        # ★★DAG-우선 (2026-08-07 사용자 지시: *"DAG로 정의된 선행행동 순서에 따라
                        #   엔진이 동작해야 한다"*). 여기가 **침묵이 치환으로 바뀌는 자리**다.
                        #
                        #   왜: `phase`는 독립 축이 아니라 **선언에서 파생된 상태**다
                        #   (`t2_phase.phase_of`가 `gates[kind=auth]`의 satisfier 실행 이력만 본다).
                        #   그래서 같은 auth 게이트 선언을 두 기제가 읽고 **정반대 행동**을 냈다 —
                        #   선행 강제는 *"미충족 조상이 먼저다"* 라고 명령하고, 단계 소유권은
                        #   *"행동-유도 침묵"* 을 시켰다. 침묵당한 것이 바로 그 명령이라
                        #   **자기-강화 교착**이 된다(조상 미충족 → 침묵 → 명령 없음 → 조상 계속 미충족).
                        #   20260807b 실측: 침묵 6회 · 우리 층 발화는 claimprov 4건뿐 ·
                        #   그 게이트의 `applies_to`에 선언된 read 도구는 **호출 0**.
                        #
                        #   [[56]] C3의 문구가 이미 답이었다: **"진 쪽은 침묵이 아니라 치환된다."**
                        #   그래서 행동-유도를 지우되 그 자리에 **DAG가 말하는 미충족 조상**을 놓는다.
                        #   요건 산출은 `t2_dominance.requirements_for`(출처 = `gates[]` ·
                        #   `require_tool_before` · `requires_reads` — 새 A2 키 0)이고, 도메인 어휘 0이다.
                        #
                        #   ⚠**회귀 불가 설계**: DAG가 낼 요건이 없으면 종전 거동(침묵) 그대로다.
                        #   즉 이 변경은 *침묵을 명령으로 바꿀 수 있을 때만* 바꾼다.
                        if _off_phase:
                            _sub17 = ""
                            _t17 = None
                            try:
                                import t2_dominance as _DOM17
                                # ★★표적을 하나로 좁히지 않는다 (2026-08-07·x126 격리가 판정).
                                #   전판은 formalize가 낸 표적 하나에만 물었고, 그것이 후보 집합에
                                #   없으면 **그래프에 묻지도 않았다**. x126 재생 결과: phase=verify 턴
                                #   **6개 전부**에서 후보 7개 중 4개가 미충족 조상을 갖고 있었다 —
                                #   즉 그래프는 조용하지 않았고 표적 선택이 침묵을 만들었다.
                                #   처방은 [[56]] C3와 같은 형태다: **하나를 고르지 않는다.**
                                #   후보 전체를 돌아 요건을 모으고, 가장 많이 덮는 표적으로 말한다
                                #   (명령 하나 · 사실 합집합). 새 A2 키 0 · 도메인 어휘 0.
                                _exec17 = _executed_tool_names(state.messages, a2)
                                _cands17 = []
                                for _a17 in sorted(_acts):
                                    _r = _DOM17.requirements_for(
                                        a2, state.messages, _a17,
                                        executed=_exec17, unwrap=_exact_tool_name)
                                    if _r:
                                        _cands17.append((_a17, _r))
                                if _tgt_pre and any(_a == _tgt_pre for _a, _ in _cands17):
                                    _t17, _rq17 = next((c for c in _cands17 if c[0] == _tgt_pre))
                                elif _cands17:
                                    # formalize가 표적을 못 냈으면 **요건이 가장 많은 후보**로 말한다.
                                    _t17, _rq17 = max(_cands17, key=lambda c: len(c[1]))
                                else:
                                    _t17, _rq17 = None, []
                                if not _cands17:
                                    print("[T2_PHASE_PRECEDE] silent-DAG acts=%d reqs=0 "
                                          "(어느 후보에도 미충족 조상이 없다)" % len(_acts),
                                          file=_sys.stderr, flush=True)
                                else:
                                    _sub17 = _DOM17.merged_text(a2, _rq17, _t17)
                                    print("[T2_PHASE_PRECEDE] cands=%d picked=%s reqs=%s"
                                          % (len(_cands17), _t17,
                                             [r.get("id") for r in _rq17][:3]),
                                          file=_sys.stderr, flush=True)
                                    if not _sub17:
                                        print("[T2_PHASE_PRECEDE] empty-text target=%s reqs=%d"
                                              % (_t17, len(_rq17)), file=_sys.stderr, flush=True)
                            except Exception as _e17:
                                print("[T2_PHASE_PRECEDE] DAG requirement failed (keep silent): %r"
                                      % (_e17,), file=_sys.stderr, flush=True)
                                _sub17 = ""
                            if _sub17:
                                # ★자기정정(2026-08-07·20260807f 실측 9건): 여기서 `_ap_regen`을
                                #   부르면 **UnboundLocalError**다 — 그 클로저는 이 함수의 **뒤쪽**
                                #   (`:6886`)에서 정의되므로 파이썬이 지역변수로 보고 미대입 상태다.
                                #   감싸는 try/except가 그것을 삼켜서 **`substitute` 로그는 찍히고
                                #   전달은 0**이었다 = [[55]]의 *"로그 마크 ≠ 전달"* 을 내가 그대로 재현.
                                #   ⇒ 이웃 코드가 쓰는 채널로 보낸다: `rw_fb=(None, text)`는
                                #   순수-조언 형태로 `:6497`에서 UserMessage 리마인더가 되어 재생성된다.
                                # ★지문 억제(2026-08-07·20260807g 폭주 실측): 요건은 조상이 충족될
                                #   때까지 계속 참이므로 **캡이 없으면 매 턴 같은 말을 한다**.
                                #   g 런 실측: 우리 층 발화 4건(b) → **80건**, `[ORDER]` 계열 15회,
                                #   그 뒤를 "resolve the flagged call(s) first" 16회가 따라붙어
                                #   두 문구가 서로를 먹였다. [[57]] *"인자가 바뀌어야 다시 말한다"* 위반.
                                #   ⇒ 횟수 캡이 아니라 **지문**으로 막는다: (표적, 요건 집합)이 같으면 침묵.
                                #   조상이 하나라도 충족되면 지문이 바뀌므로 자동으로 다시 말한다.
                                _fp17 = (_t17, tuple(sorted(r.get("id") or "" for r in _rq17)))
                                _seen17 = getattr(self, "_t2_pp_seen", None)
                                if _seen17 is None:
                                    _seen17 = self._t2_pp_seen = set()
                                if _fp17 in _seen17:
                                    print("[T2_PHASE_PRECEDE] suppressed (same fingerprint) %s"
                                          % (_fp17,), file=_sys.stderr, flush=True)
                                elif rw_fb is None:
                                    _seen17.add(_fp17)
                                    rw_fb = (None, _sub17)
                                    print("[T2_PHASE_PRECEDE] substitute (was: silent) target=%s "
                                          "→ rw_fb(pure-advice)" % _t17,
                                          file=_sys.stderr, flush=True)
                                else:
                                    print("[T2_PHASE_PRECEDE] skipped — rw_fb already set",
                                          file=_sys.stderr, flush=True)
                            else:
                                print("[T2_PHASE_OWNER] action-push silent — phase=%s (no DAG req)"
                                      % _phase17, file=_sys.stderr, flush=True)
                        if rw_fb is None and not _off_phase \
                                and _acts and not (_called & _acts) and _rz._agent_ending(am, _transfer_tools(a2)):
                            _tgt = _tgt_pre if _tgt_pre in _acts else None
                            # ★고정밀(Δspurious): formalize가 구체 agent-실행 target을 낼 때만 발화.
                            #   target=None(=action-ask)은 미발화 — discovery/user-실행 의도서 스퓨리어스
                            #   (banking 잔여=⋈/reach이지 deflect-vs-ask 아님·BANK_ACTIONREQ_PROBE_FORENSIC).
                            _ar = (_rz.resolve_action_operator(
                                {"action_tools": list(_acts)}, am, state.messages, a2,
                                target_tool=_tgt, transfer_tools=_transfer_tools(a2),
                                # ★C442 진행-감응: 이름 집합은 **프레임워크 레지스트리**에서만
                                #   온다(`registry_names` — 이미 있는 함수·도메인 리터럴 0).
                                #   우리가 이름을 짓지 않는다는 것이 이 인자의 요점이다.
                                known_names=_rz.registry_names(self),
                                # ★후보-정합 formalize 재료 (2026-08-12·j런 070t0 5라운드
                                #   줄다리기 수리): 어느 회수-이름이 요청을 성취하는지 LLM 이
                                #   고른다 — 미주입이면 STEP2 는 이름 단정 없이 일반문 강등.
                                agent=self, la=la, UserMessage=UserMessage)
                                if _tgt else {"status": "ok"})
                            # ★재료는 **모델이 먼저 틀려야** 나가고 있었다 (2026-08-15·
                            #   `T2_SEARCH_ON_PROCEED`·기본 OFF). 아래 `deny` 분기 **안에서만**
                            #   `_search_material` 이 불린다 — 매끄럽게 진행하는 sim 은 정책
                            #   문서를 영영 못 본다. 071 실측: deny 가 난 1 sim 만 창이 열렸고
                            #   (그나마 시계보다 앞서 침묵) 나머지 **2 sim 은 창이 안 열렸다**.
                            #   `T2_NOW_SELFCALL` 만으로는 그 둘이 안 열린다 — 두 결함은 독립이다.
                            #   ⚠배달 채널은 원래 deny 와 무관하다(`_t2_cp2_pending` = 재생성
                            #     버퍼·`T2_DECISION_CARRY`). 막던 것은 **코드 위치**뿐이었다.
                            #   ⚠나르는 것은 **재료**(만료 제외된 문서 발췌)이지 지시가 아니다 —
                            #     고르는 것은 끝까지 모델이다([[62]] ③④: 순위·최댓값·지목 0).
                            #   ⚠[[57]] 재발화는 횟수가 아니라 **인자 변화**로: 같은 문자열이면
                            #     안 넣는다(`_t2_cp2_said` 비교는 아래 CP2 와 같은 규약).
                            # ★행동 촉구 (2026-08-15·`T2_ACT_DEMAND`·기본 OFF).
                            #   격리 3런 재현(x330 11/24 · x331 13/24 · x332 16/24 ↔ 기준선
                            #   2/0/6 · 부정통제 `D_EARLY` 세 런 모두 **0/24**): 같은 문맥·같은
                            #   도구에서 **한 줄 요구**가 실행률을 올린다. 반대로 *"세고 체크하고
                            #   처리하라"* 는 **0/24 로 해로웠다**(x332 B_SELFLIST) — 묘사를
                            #   시키면 묘사가 는다. ⇒ **열거 없는 행동 명령만** 쓴다.
                            #   발화 자리 = 격리 컷과 **구조적으로 동일**하다: 이 블록의 조건이
                            #   곧 *"행동 도구를 안 부른 채 턴을 끝내려 하고, 구체 대상이 형식화됐다"*
                            #   이다 — 의도 판정이 아니라 **닫힌 구조 조건**이다([[66]] 위반 아님).
                            #   ⚠[[62]] ①격리 실측 위 3런 ②부분 성공(16/24)이라 라이브 이관이
                            #     다음 단계 ③**사라지는 판단 없음** — 무엇을 할지는 여전히 모델이
                            #     정하고 우리는 도구를 **지목하지 않는다**(x322: 지목은 24/24→0/24)
                            #     ④순위·최댓값·정답 문장 0.
                            #   ⚠[[05]] ⑴도메인 어휘 0(문장에 은행 용어 없음) ⑵유동 판단 동결 없음
                            #     ⑶엔진이 도메인 행동을 수행하지 않는다 — 요구만 한다.
                            #   ⚠[[57]] 부작용 계측 의무: over-action(gold 없는 write)이 늘면 손해다.
                            if (os.environ.get("T2_ACT_DEMAND") == "1"
                                    and os.environ.get("T2_DECISION_CARRY") == "1"):
                                _dm = "Carry out the next step of this request now."
                                if _dm != getattr(self, "_t2_cp2_said", None):
                                    _cp2_assign(self, _dm, "ACT_DEMAND")
                                    self._t2_cp2_said = _dm
                                    print("[T2_ACT_DEMAND] 행동 촉구 1줄 배달(도구 지목 0)",
                                          file=_sys.stderr, flush=True)
                                else:
                                    print("[T2_ACT_DEMAND] 같은 문자열 — 재배달 안 함",
                                          file=_sys.stderr, flush=True)
                            if (_ar.get("status") != "deny"
                                    and os.environ.get("T2_SEARCH_ON_PROCEED") == "1"
                                    and os.environ.get("T2_SEARCH_AGENT") == "1"
                                    and os.environ.get("T2_DECISION_CARRY") == "1"
                                    and getattr(self, "_t2_searchagent_fired", 0) < 3):
                                # ★T2_PROCEED_DOCBODY (2026-08-16·기본 OFF·t7304=S1 재설계) —
                                #   배달 **객체**를 서브 결정문(243~263자) → **유효 문서 본문**
                                #   (37k~50k자)으로 바꾼다. 왜. t7303 로그 직독: 이 자리에 배달되던
                                #   서브 결정 자체가 오답이었다(055 양팔 `DOCDECIDE → 'Blue
                                #   Account'`·gold Purple). 격리 24/24 를 만든 객체는 문서 본문이다
                                #   (x335b) — 자리·예산·슬롯·축 소비 전부 불변, 객체만 격리와 일치.
                                #   ⚠스위치는 `_search_material` **안**에 있다(심사 3인 일치: 한
                                #   자리만 플립하면 다른 자리가 축을 먼저 소비해 문서가 영영 못 온다).
                                #   컨텍스트 가드는 **소비 지점**(부착 직전) 하나에 있다.
                                #   [[62]]③: 오히려 엔진-측 결정문을 **제거**하는 방향이다.
                                # ★T2_DOCS_AT_WRITE 이면 이 자리(이른 proceed)는 **비운다** —
                                #   예산을 늘리는 게 아니라 **옮긴다**. 여기서 축을 소비하면
                                #   write 자리에서 서브가 침묵해(모두 처리됨) 재료가 안 온다.
                                #   t7303 전수: 이 자리의 배달은 요구 진술보다 먼저 끝난다.
                                try:
                                    _mp = ("" if os.environ.get("T2_DOCS_AT_WRITE") == "1"
                                           else _search_material(self, a2, state.messages))
                                except Exception as _mpe:
                                    _mp = ""
                                    print("[T2_SEARCH_ON_PROCEED] 건너뜀(무발화): %r" % (_mpe,),
                                          file=_sys.stderr, flush=True)
                                if _mp:
                                    self._t2_searchagent_fired = getattr(
                                        self, "_t2_searchagent_fired", 0) + 1
                                    if _mp != getattr(self, "_t2_cp2_said", None):
                                        _cp2_assign(self, _mp, "SEARCH_ON_PROCEED")
                                        self._t2_cp2_said = _mp
                                        print("[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 %d자 배달"
                                              % len(_mp), file=_sys.stderr, flush=True)
                                    else:
                                        print("[T2_SEARCH_ON_PROCEED] 같은 문자열 — 재배달 안 함",
                                              file=_sys.stderr, flush=True)
                            if _ar.get("status") == "deny":
                                _fb_ar = _ar["feedback"]
                                # ★C11b(2026-08-06·032 실측): 발견을 시키는 그 문장에, **이 대화가 이미
                                #   회수한 문서가 이름을 말한** 미호출 도구를 함께 짚는다. 소진 시점까지
                                #   기다리던 것을 앞당기는 것뿐이고 새 정보는 0이다(레지스트리 ∩ 이미 받은
                                #   텍스트). 근거: 아무도 부르지 않은 gold 도구 23건 중 12건이 그 집합이었다.
                                #   ⚠"검색이 도구명을 못 물어왔다"류의 일반 넛지는 만들지 않았다 —
                                #   전수 측정에서 실패 26% vs 통과 32%로 구분력이 없었다.
                                if os.environ.get("T2_DISCOVERY_NAMES") == "1":
                                    try:
                                        _reg11 = _agent_discoverable(
                                            getattr(getattr(self, "_t2_orch", None), "environment", None))
                                        _txt11 = "\n".join(
                                            str(getattr(_m11, "content", "") or "")
                                            for _m11 in state.messages
                                            if getattr(_m11, "role", None) == "tool")
                                        _used11 = {_exact_tool_name(_t11) for _m11 in state.messages
                                                   for _t11 in (getattr(_m11, "tool_calls", None) or [])}
                                        _used11 |= _unlocked_names(state.messages, a2)
                                        _cand11 = sorted(n for n in (_reg11 or set())
                                                         if n in _txt11 and n not in _used11)
                                        if _cand11:
                                            _fb_ar += (" The documents you have ALREADY retrieved name"
                                                       " these tools, and you have not called them: %s."
                                                       % ", ".join(_cand11[:5]))
                                            print("[T2_DISCOVERY_NAMES] surface %d" % len(_cand11),
                                                  file=_sys.stderr, flush=True)
                                    except Exception as _e11:
                                        print("[T2_DISCOVERY_NAMES] error (no-op): %r" % (_e11,),
                                              file=_sys.stderr, flush=True)
                                # ★M3 — 결정점을 **행위자 무관**으로 넓힌다 (`T2_DECIDE_ANY`·기본 OFF·
                                #   설계서 `TASK_070_071_DESIGN_2026_08_09` §3-M3).
                                #   지금 결정 블록은 *"손님이 실행할 도구"* 분기 안에서만 만들어진다.
                                #   070/071 의 gold(`open_bank_account_4821`)는 **에이전트가** 부르므로
                                #   그 분기가 영원히 거짓이고, 블록·재도출·D1c 가 **한 번도 발화하지
                                #   않는다**(설계서 §2⒞ [S]). ⇒ A2 가 `action_tools` 로 선언한 도구를
                                #   **누구든** 밀고 있으면 같은 재료를 만든다. 엔진이 보는 것은
                                #   **멤버십뿐**이고, 무엇이 결정 시점인지 의미로 판정하지 않는다([[22]]).
                                #   ⚠부작용 계측 의무: 발화 자리가 늘면 Δspurious 가 생긴다. 099/100 은
                                #     같은 조건에서 **거동 불변**이어야 한다(플래그 OFF = 바이트 동일).
                                #   ⚠C404 유보: 이 자리는 지시(`_fb_ar`)와 값이 **한 메시지로 합쳐지는**
                                #     배치다 — x231 이 해롭다고 잰 그 모양이다. 먼저 **닿게** 한 뒤
                                #     자리는 따로 잰다(전달 없이는 잴 것도 없다).
                                if os.environ.get("T2_DECIDE_ANY") == "1":
                                    try:
                                        _m3 = _limit_reduce_text(self, a2, state.messages)
                                    except Exception as _m3e:
                                        _m3 = ""
                                        print("[T2_DECIDE_ANY] 건너뜀(무발화): %r" % (_m3e,),
                                              file=_sys.stderr, flush=True)
                                    # ★검색 에이전트 — **두 번째 재료 소스**(2026-08-11·C418·
                                    #   `T2_SEARCH_AGENT`·기본 OFF·설계서 §3-3 그대로).
                                    #   `_limit_reduce_text` 는 **원장**이 있어야 말한다. 070/071 은
                                    #   그 원장이 없어 무발화였다(handoff §7 의 M3 무발화). 재료가
                                    #   문서 쪽에 있는 계열은 여기서 채운다.
                                    #   측정(`x248`·`x250`·071 실물·n=8·**프로덕션 경로**): 두 축 8/8 —
                                    #   checking `Sky Blue` · savings `Gold Saver Account`. 만료로
                                    #   빠지는 것이 정확히 두 태스크를 오답으로 끌던 고지 둘이다.
                                    #   부정 통제: 문서만(고지 없이) checking **0/8** · 만료를 안 빼면
                                    #   savings **0/8** ⇒ **엔진의 유일한 일(만료 제거)이 값을 산다.**
                                    #   ⚠분담: 군을 고르는 것도 답을 고르는 것도 **LLM** 이다.
                                    #     엔진은 **읽고·비교하고·자르기**만 한다(⛔0 ③) — 순위도
                                    #     최댓값도 내지 않고, 지목 문장도 만들지 않는다.
                                    #   ⚠코퍼스는 **환경이 든 것**을 읽는다 — 경로 하드코딩 0([[05]]).
                                    #   ⚠한 sim 에 **한 번**만(재료는 대화와 무관한 정책 상수라 반복
                                    #     발화가 이득이 아니다·[[57]]).
                                    # ★T2_MATERIAL_RESERVE (2026-08-16·C498·기본 OFF) — **예산을 결정점에
                                    #   남긴다**. 배달은 `state.messages` 가 아니라 **그 턴의 재생성 버퍼**
                                    #   에만 붙는다(위 `_t2_cp2_pending` · 비커밋 · C298 replay 불변식).
                                    #   즉 재료는 **한 턴만 살아 있다**. 그런데 t7298 의 055 네 sim 은
                                    #   sim 당 예산 3회를 **`대화텍스트 1`(손님이 요구를 말하기도 전)** 부터
                                    #   전부 써 버렸고(DELIVER 3·3·2·3), 정작 상품을 고르는 turn 14+ 에는
                                    #   재료가 문맥에 **없었다** — 궤적 전수 검색에서 재료 표지 0건(4 중 3).
                                    #   격리 24/24 ↔ 라이브 0/4 의 기전이 이것이다([[62]] 규칙4·C497).
                                    #   ⇒ 여기(일반 자리)의 배달을 **1회로 묶고** 나머지 예산을 결정 자리
                                    #     (`T2_SEARCH_ON_PROCEED`)에 남긴다. **총량은 그대로 3**이고 새 판단
                                    #     기구도 없다 — 같은 예산의 **사용처만** 옮긴다([[63]] 형태).
                                    #   ⚠1차 종점은 성적이 아니라 **결정 직전 생성에 재료가 있었는가**다.
                                    if (not _m3 and os.environ.get("T2_SEARCH_AGENT") == "1"
                                            and getattr(self, "_t2_searchagent_fired", 0) < 3
                                            and (os.environ.get("T2_MATERIAL_RESERVE") != "1"
                                                 or getattr(self, "_t2_sa_early", 0) < 1)):
                                        try:
                                            _m3 = _search_material(self, a2, state.messages)
                                            if _m3:
                                                self._t2_searchagent_fired = getattr(
                                                    self, "_t2_searchagent_fired", 0) + 1
                                                self._t2_sa_early = getattr(
                                                    self, "_t2_sa_early", 0) + 1
                                                print("[T2_SEARCH_AGENT] 일반 자리 배달 %d회째(예약 %s)"
                                                      % (self._t2_sa_early,
                                                         os.environ.get("T2_MATERIAL_RESERVE") or "off"),
                                                      file=_sys.stderr, flush=True)
                                        except Exception as _sae:
                                            print("[T2_SEARCH_AGENT] 건너뜀(무발화): %r" % (_sae,),
                                                  file=_sys.stderr, flush=True)
                                    # ★ACTION-INDEX 1회 표면화 (2026-08-14·`T2_ACTION_INDEX`·기본 OFF·
                                    #   사용자 지시 *"도구 설명 표면화하라 · 비용이 최소가 되게"*).
                                    #   A3 `action_index` = **행동을 기술하는 문서 43줄**(제목 + 그 문서가
                                    #   대는 도구명). 빌드 시점 기계 도출(`t2_index_build`)·저작 0·gold 무접촉.
                                    #   측정(x319·n=24·블록 8·8·8·잡음 바닥 ±4 밖):
                                    #     도움 없음 **10/24** → 이 43줄 **24/24** · 도구 설명 91종 23/24 ·
                                    #     이름만 91종 16/24 ⇒ 표면화가 열고, **의미를 담은 것이 이름보다 낫다**.
                                    #   왜 여기인가: 위 검색 재료가 **없을 때**의 폴백이다 — 재료가 오면
                                    #   그쪽이 더 구체적이라 굳이 목록을 얹지 않는다(더하기는 해롭다·C404).
                                    #   ⚠엔진은 고르지 않는다 — 43줄을 인쇄만 하고 선택은 LLM([[62]] ④).
                                    #   ⚠sim 당 **1회**(정책 상수라 반복이 이득이 아니다·[[57]]).
                                    if (not _m3 and os.environ.get("T2_ACTION_INDEX") == "1"
                                            and not getattr(self, "_t2_actionidx_fired", False)):
                                        try:
                                            import t2_search as _ts2
                                            _m3 = _ts2.action_index_note(a2)
                                            if _m3:
                                                self._t2_actionidx_fired = True
                                                print("[T2_ACTION_INDEX] 1회 표면화 %d자"
                                                      % len(_m3), file=_sys.stderr, flush=True)
                                        except Exception as _aie:
                                            _m3 = ""
                                            print("[T2_ACTION_INDEX] 건너뜀(무발화): %r" % (_aie,),
                                                  file=_sys.stderr, flush=True)
                                    # ★CP2 DECISION-CARRY (설계 v1.5 §4·`T2_DECISION_CARRY`·
                                    #   기본 OFF). 전문가의 결론이 지금 `_fb_ar` 를 타는데,
                                    #   그것은 `rw_fb` = **배타 체인 rank 11**이다(C429). 앞의
                                    #   `wev`(rank 8)가 같은 호출에 걸리면 통째로 버려지고,
                                    #   지문 억제는 그 전에 `_ufb` 를 비운다. 그것이 §3 표의
                                    #   *"검색/결정 8/8 → 경로 없음 → 라이브 0/6"* 의 정체다.
                                    #   ⇒ 값에 **체인 밖 채널**을 준다: 비커밋 생성-뷰
                                    #   (`_t2_view_fb`·C298 — 커밋하면 replay 가 깨진다).
                                    #   ⚠**엔진은 고르지 않는다.** 나르는 것은 서브가 이미 낸
                                    #     문자열 그대로다 — 순위·최댓값·지목 문장 0([[62]] ③④).
                                    #   ⚠[[57]] 재발화는 **횟수가 아니라 인자 변화**로: 같은
                                    #     문자열이면 안 넣는다. 값이 바뀌면 다시 넣는다.
                                    #   ⚠비교도 **문자열 동등성**이다 — 값을 뽑아내려고 도메인
                                    #     텍스트를 파싱하면 그것이 [[59]] 위반이다.
                                    #   근거 = C435: 서브가 스스로 낸 값이 gold 주입과 구분 불가
                                    #     (`B_SUB` 7/8 ↔ `B_VALUE` 6/8·p=1.000) · 부정통제
                                    #     `B_NULL` 0/8(p=0.0014) ⇒ 나를 값이 실재한다.
                                    if _m3 and os.environ.get("T2_DECISION_CARRY") == "1":
                                        try:
                                            if _m3 != getattr(self, "_t2_cp2_said", None):
                                                # ★C443: 뷰 큐(`_t2_view_fb`)는 **다음 턴**에
                                                #   소비된다 — 결정점도 write 도 이 턴이라
                                                #   한 턴 늦었다(`arrived=False` 실측). 이 턴의
                                                #   재생성 버퍼로 보낸다(§`work = work + fb` 뒤).
                                                _cp2_assign(self, _m3, "VIEW_FB")
                                                self._t2_cp2_said = _m3
                                                # ★배달을 **모델 입력에서** 잰다 (C441⒡).
                                                #   사이드카는 뷰 채널을 안 남긴다 — 그래서
                                                #   CP2 는 발화만 보이고 도달이 안 보였다.
                                                #   `_gen` 의 `arrived` 훅에 같은 형식으로
                                                #   등재하면 다른 경로와 한 표에서 비교된다.
                                                _qr = list(getattr(
                                                    self, "_t2_route_pending", None) or [])
                                                _qr.append(dict(
                                                    agent="decision_carry", rank=None,
                                                    target=_tgt, outcome="view",
                                                    lost_to=None, folded=False, _text=_m3))
                                                self._t2_route_pending = _qr
                                                print("[T2_DECISION_CARRY] 결정 값을 체인 밖 "
                                                      "뷰 채널로 (target=%s · %d자)"
                                                      % (_tgt, len(_m3)),
                                                      file=_sys.stderr, flush=True)
                                                _lbeat("T2_DECISION_CARRY", orch=self,
                                                       target=_tgt,
                                                       fact="the decision a sub-agent already "
                                                            "reached, carried to this turn")
                                            else:
                                                print("[T2_DECISION_CARRY] 같은 값 — 재발화 0 "
                                                      "([[57]] 인자 변화 기준)",
                                                      file=_sys.stderr, flush=True)
                                            # 체인에는 싣지 않는다 — 16번째 경쟁자가 되면
                                            # 오늘의 0/6 이 그대로 재생된다(설계 §3.1).
                                            _m3 = ""
                                        except Exception as _cp2e:
                                            print("[T2_DECISION_CARRY] 건너뜀(무발화): %r"
                                                  % (_cp2e,), file=_sys.stderr, flush=True)
                                    if _m3:
                                        _fb_ar = _fb_ar + "\n" + _m3
                                        print("[T2_DECIDE_ANY] 에이전트-실행 결정점에 재료 동반 "
                                              "(target=%s · %d자)" % (_tgt, len(_m3)),
                                              file=_sys.stderr, flush=True)
                                rw_fb = ((am.tool_calls or [None])[0], _fb_ar)
                                self._t2_action_deny = getattr(self, "_t2_action_deny", 0) + 1
                                # ★진행-감응 환급용 target 스냅샷 (2026-07-22 §2bt·rall10 097 실측:
                                #   초반 flail이 cap3 소진→종반 say-loop 8연속 무방비 = FOLLOWUP
                                #   cap-소진과 동형). 발화 시점에 미실행인 target만 기록 —
                                #   이후 시도-수준 착수가 보이면 cap 1회 환급(1b와 동일 원리).
                                _effa2 = {_eff_tool_name(tc) for m2 in state.messages
                                          for tc in (getattr(m2, "tool_calls", None) or [])}
                                self._t2_action_target = ({str(_tgt)}
                                                          if str(_tgt) not in _effa2 else None)
                                # ★T2_FORCE_ACTION (2026-07-20·사용자 통찰 "say한 tool do하면 됨"): say-don't-do
                                #   (모델이 실행 의도를 텍스트로만 말하고 호출 0) = 재생성서 tool_choice=required 강제.
                                #   의도는 이미 모델 안에 있으니 산문→호출로 뒤집힘. required=vLLM 구조화디코딩(봉투드롭 불가).
                                #   ★[[10]] 정합: 어느 도구·인자는 모델 몫(강제 안 함)·디코딩 제약만. cap=action_deny 승계·
                                #   provenance/게이트가 다음 라운드서 backstop(날조 인자 차단). 기본 OFF.
                                if os.environ.get("T2_FORCE_ACTION") == "1" and not (am.tool_calls or []):
                                    force_required = True
                                    print("[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성",
                                          file=_sys.stderr, flush=True)
                                print("[T2_RESOLVE] action-required reason=%s target=%s"
                                      % (_ar.get("reason"), _tgt), file=_sys.stderr, flush=True)
                    # ★Lever 3 verify-persistence (task_023형·신원수집+검증미완+포기): action-required가
                    #   안 걸렸을 때(예: apply=user-실행 intent) 검증 완결 강제. cap 1/sim.
                    if (rw_fb is None and getattr(self, "_t2_verify_deny", 0)
                            < int(os.environ.get("T2_VERIFY_DENY_CAP", "1"))):
                        _vr = _rz.resolve_verify_persistence(am, state.messages, a2,
                                                             transfer_tools=_transfer_tools(a2))
                        if _vr.get("status") == "deny":
                            rw_fb = ((am.tool_calls or [None])[0], _vr["feedback"])
                            self._t2_verify_deny = getattr(self, "_t2_verify_deny", 0) + 1
                            print("[T2_RESOLVE] verify-persistence deny", file=_sys.stderr, flush=True)
                except Exception as _rze:
                    rw_fb = None
                    print("[T2_RESOLVE] error (no-op): %r" % (_rze,), file=_sys.stderr, flush=True)
            # ★action-required (turn-level·회피=순수조언은 tool_call 0 → 앵커할 ToolMessage 없음).
            #   ✅라이브 배선(2026-07-13): rw_fb[0] is None 케이스는 아래 fb-빌드서 UserMessage 리마인더로
            #   재생성(regenerate-with-directive·eplan-reminder류·작업버퍼만·test_action_reminder 14/14).
            #   transfer-only 회피는 tool_call 앵커로 이미 처리·per-operand rw_fb(deny-kind)도 라이브.
            # ★T2_TOOLLIST (2026-07-21 §2bb·r095g g-t0 실측): 도구목록 **밖** 이름 호출(발명명+
            #   discoverable 접미사-직호출 공히) = 생성-레벨 deny+재생성. 근거: ①TOOLGATE env-실재
            #   통과(§2ao 픽스)가 접미사 직호출을 허용 → unlock+디스패처 쌍(gold 액션형식) 건너뜀
            #   (g-t0 액션 1/9) ②gold census(로컬 results 전수): gold는 전 태스크서 직호출 0 =
            #   over-block 0 ③실행-레벨 deny는 mutating replay 불일치(§2ao·052) — 생성-레벨은
            #   작업버퍼만(비커밋)=replay-clean. 술어=자기 도구목록 대조뿐(도메인 리터럴 0)·도메인
            #   안내문=A2 `nonlisted_tool_feedback`. cap 소진 후엔 통과(liveness·env-실재=replay 정합).
            tl_fb = None
            if (os.environ.get("T2_TOOLLIST") == "1"
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_rounds < 1
                    and getattr(self, "_t2_toollist_deny", 0)
                    < int(os.environ.get("T2_TOOLLIST_CAP", "6"))):
                _vis = {getattr(t, "name", None) for t in (self.tools or [])}
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", None)
                    if nm and _vis and nm not in _vis:
                        _extra = str((a2 or {}).get("nonlisted_tool_feedback") or "").strip()
                        tl_fb = (c, ("'%s' is not one of your provided tools, so it was not "
                                     "called. Only call tools that appear in your tool list.%s"
                                     % (nm, (" " + _extra) if _extra else "")))
                        print("[T2_TOOLLIST] deny nonlisted tool=%s" % nm,
                              file=_sys.stderr, flush=True)
                        break
            # ★T2_UNLOCK_NAME (2026-07-21 §2bh·rall5 실측): A2 `discoverable_name_check` — 선언된
            #   (도구→인자) 값이 접미사 패턴에 불일치하면 생성-레벨 deny+required regen(비커밋=replay-clean).
            #   근거: FOLLOWUP 강제는 행동을 움직였으나(체크 unlock 3회씩 시도) 전부 bare-name→env
            #   "Unknown tool"→포기 반복(6 sim×2도구×3회) = 마지막 고리가 이름-형식. 엔진=패턴 대조만
            #   (이름 리터럴 0·KB 검색·이름 선택=모델·[[05]](3) autocomplete-변형 기각). cap 소진=통과(현행).
            # ★T2_DISPATCH_ROLE (2026-07-22 §2bl·rall8 031 실측): 디스패처 역할 혼동 deny —
            #   dispute를 user-디스패처로 제출·user-도구를 에이전트가 직접 호출(4회). 술어=대화
            #   자기-이력(이 대화서 unlock된 이름=agent-도구·give된 이름=user-도구)·엔진 이름-리터럴 0.
            #   (구 +tool_arg_allowlist strip은 2026-07-31 V7로 대체·삭제 — 아래 §인자-strip 폐기.)
            # ★T2_PRESCRIPTION (2026-07-22 §2bu·rall11 038 실측·격리 L2 8/8=활성화 실패): 처방-오선택 deny.
            #   038: 사기 dispute 요청(unauthorized/fraudulent)인데 apply_statement_credit 오선택(dispute 미착수).
            #   L2 프로브=명시질의 시 file_dispute 8/8 앎 → 자유생성 미발현(활성화). 게이트가 상기:
            #   대화에 dispute-신호 ∧ file_dispute 미호출인데 statement_credit류 write면 deny+dispute 안내.
            #   KB 근거=doc_014(dispute)·doc_017(statement_credit=선의/프로모션/수수료만). 신호·문구=A2·엔진=
            #   문자열 대조+집합(리터럴 0). Δspurious 계측(dispute 후 정당 credit=absent_tool 조건으로 통과).
            pr_fb = None
            _prs = (a2 or {}).get("prescription_redirect") or []
            if (os.environ.get("T2_PRESCRIPTION") == "1" and _prs
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and pc_fb is None
                    and getattr(self, "_t2_prescription_deny", 0)
                    < int(os.environ.get("T2_PRESCRIPTION_CAP", "3"))):
                _conv = " ".join(str(getattr(m2, "content", "") or "") for m2 in state.messages
                                 if getattr(m2, "role", None) in ("user", "tool")).lower()
                _effp = {_eff_tool_name(tc) for m2 in state.messages
                         for tc in (getattr(m2, "tool_calls", None) or [])}
                for c in (am.tool_calls or []):
                    for sp in _prs:
                        _pn = str(_args_dict(c).get(sp.get("arg", "")) or getattr(c, "name", "") or "")
                        if not _pn.startswith(sp.get("prefix", "\0")):
                            continue
                        _abt = sp.get("requires_absent_tool")
                        if _abt and _abt in _effp:
                            continue                     # 이미 정답 도구 호출됨(예: dispute 접수) → 정당 통과
                        if any(sig in _conv for sig in (sp.get("signals") or [])):
                            pr_fb = (c, str(sp.get("feedback") or "Error: [PRESCRIPTION] wrong tool."))
                            self._t2_prescription_deny = getattr(self, "_t2_prescription_deny", 0) + 1
                            print("[T2_PRESCRIPTION] deny tool=%s prefix=%s"
                                  % (_pn, sp.get("prefix")), file=_sys.stderr, flush=True)
                            break
                    if pr_fb:
                        break
            dr_fb = None
            _drs = (a2 or {}).get("dispatcher_role_check") or {}
            if (pr_fb is None and os.environ.get("T2_DISPATCH_ROLE") == "1" and _drs
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and getattr(self, "_t2_dispatchrole_deny", 0)
                    < int(os.environ.get("T2_DISPATCH_ROLE_CAP", "6"))):
                _na = _drs.get("name_args") or {}
                def _iname(tc):
                    _ar = _args_dict(tc)
                    return str(_ar.get(_na.get(getattr(tc, "name", None), ""), "") or "")
                _unl = {_iname(tc) for m2 in state.messages
                        for tc in (getattr(m2, "tool_calls", None) or [])
                        if getattr(tc, "name", None) == _drs.get("unlock_tool")}
                _gvn = {_iname(tc) for m2 in state.messages
                        for tc in (getattr(m2, "tool_calls", None) or [])
                        if getattr(tc, "name", None) == _drs.get("give_tool")}
                for c in (am.tool_calls or []):
                    nm = getattr(c, "name", None); iv = _iname(c)
                    if not iv:
                        continue
                    fbt = None
                    if nm == _drs.get("user_call") and iv in _unl:
                        fbt = _drs.get("agent_via_user_feedback")
                    elif nm == _drs.get("agent_call") and iv in _gvn:
                        fbt = _drs.get("user_via_agent_feedback")
                    elif nm == _drs.get("user_call") and iv in _gvn:
                        fbt = _drs.get("agent_runs_user_feedback")
                    # ★give 대상=agent-도구 deny (2026-07-22 §2bs·rall10 031 실측): 에이전트가
                    #   자기 도구(get_credit_card_accounts_by_user)를 give → env "Unknown discoverable
                    #   tool" 2회에도 같은 오선택 고수. 술어=자기 도구 목록 소속(인터페이스-구조·
                    #   카탈로그 census 불요·이름 리터럴 0): 에이전트가 스스로 부를 수 있는 도구는
                    #   정의상 user-측 discoverable이 아님. 문구=A2(user-지명 이름 재사용 힌트 포함).
                    # ★판정 집합 교체(2026-07-31·C257·`T2_DISPATCH_ROLE_ENVSET=1`·기본 OFF):
                    #   구판은 give 대상이 **`self.tools` 소속**일 때만 deny한다. 그런데 **잠긴
                    #   agent-discoverable 도구는 `self.tools`에 없어서** 그 검사를 빠져나간다 —
                    #   Y1 전수에서 give 89회 중 **18회가 env user-discoverable 집합 밖**이었고,
                    #   그 우회가 `unlock`→`call` 미호출 **55건(전체 실패의 27%)**으로 이어졌다.
                    #   ⇒ 새 레버가 아니라 **판정 집합을 env가 실제로 넘길 수 있는 것**으로 바꾼다.
                    #   오차단 구조적 불가: 정당한 give 71건은 전부 집합 안이라 통과(038 자해는
                    #   **접미사 패턴**이었고 이건 **집합 소속**이다). 정책 근거 = "The unlock step is
                    #   required before calling" · "Do not invent or guess user discoverable tools".
                    elif (nm == _drs.get("give_tool")
                          and os.environ.get("T2_DISPATCH_ROLE_ENVSET") == "1"
                          and iv and iv not in _user_discoverable(
                              getattr(getattr(self, "_t2_orch", None), "environment", None))):
                        # ★T1 소속의 3갈래 판정 (2026-08-06 실측·조정설계 §1 처방 A).
                        #   구판은 discoverable이 아닌 것을 **전부** "네 자신의 에이전트 도구"라고 말했다.
                        #   실측된 실패: 손님이 **이미 가진** 도구를 그렇게 말했고, 같은 턴의 다른 문구는
                        #   "그건 손님이 실행한다"고 말했다 = 같은 명제에 두 진리값([[25]] 정본 오염).
                        #   우선순위로 풀 문제가 아니다 — 하나가 **틀렸다**.
                        #   판정은 레지스트리 세 집합의 소속뿐이라 도메인 리터럴 0이고 deny 여부도
                        #   바뀌지 않는다(문구만 사실에 맞춘다).
                        _envg = getattr(getattr(self, "_t2_orch", None), "environment", None)
                        if iv in _user_all_tools(_envg):
                            fbt = (_drs.get("give_user_held_feedback")
                                   or _drs.get("give_agent_tool_feedback"))
                        elif iv in {getattr(t, "name", None)
                                    for t in (getattr(self, "tools", None) or [])}:
                            fbt = _drs.get("give_agent_tool_feedback")
                        else:
                            fbt = (_drs.get("give_unknown_name_feedback")
                                   or _drs.get("give_agent_tool_feedback"))
                    elif (nm == _drs.get("give_tool")
                          and os.environ.get("T2_DISPATCH_ROLE_ENVSET") != "1"
                          and iv in {getattr(t, "name", None)
                                     for t in (getattr(self, "tools", None) or [])}):
                        fbt = _drs.get("give_agent_tool_feedback")
                    if fbt:
                        dr_fb = (c, str(fbt).replace("{name}", iv))
                        print("[T2_DISPATCH_ROLE] deny tool=%s name=%s" % (nm, iv),
                              file=_sys.stderr, flush=True)
                        break
            # ★인자-strip 폐기 (2026-07-31 사용자 결정: "V7으로 대체하고 A2도 V7으로 통일하라").
            #   여기 있던 `tool_arg_allowlist` strip은 **엔진이 모델의 호출을 대신 고쳤다** —
            #   C151이 gaming으로 기각한 바로 그 방식이고, 로그상 위반을 0으로 만들어 V7(실행-레벨
            #   deny+재발행)이 **볼 입력을 구조적으로 없앴다**(Z4 실측: strip 2회·V7 0회).
            #   이제 서명 위반은 V7 한 곳에서만 판정한다(§379) — 고치는 주체는 모델([[10]] 분담).
            #   A2도 `tool_signatures` 하나로 통일(`tool_arg_allowlist` 삭제 = 도메인 opex −1키).
            #   ⚠딸린 폐기: `T2_DISPATCH_ROLE_NOTE`(strip된 값의 결정론 재진술·C212/A2)는 strip이
            #     없으면 재진술할 값이 없다. 021형 좌초(유저가 실행 템플릿을 못 받음)를 막는 책임은
            #     V7 피드백 문구("인자 값이 필요하면 호출이 아니라 **답변 본문에** 적어라")로 넘어간다
            #     — 결정론 릴레이 → 모델 행동으로 바뀌므로 **재스모크에서 021형을 확인**해야 한다.
            # ★V7 이설(2026-07-31·포렌식으로 근본원인 확정): 구판은 V7을 `gated`
            #   (=`BaseOrchestrator._execute_tool_calls`)에 뒀는데, go_stack은 `T2_GATE_REGEN=1`이라
            #   런처가 `_unified` 분기를 타고 **`t2_gate_patch.apply()`를 아예 호출하지 않는다**
            #   (`t2_run_gated.py:196`). 실행 훅은 `exec_augment`("deny 없음")가 차지한다 ⇒ V7은
            #   **구조적으로 발화 불가한 죽은 경로**에 있었다. Z4·Z5·Y2에서 deny 0이었던 진짜 이유다
            #   (앞선 "strip 선점" 진단은 불완전했다 — strip이 없어도 못 떴다).
            #   ★실증: Y2 015·021이 `give_discoverable_user_tool(discoverable_tool_name, arguments)`로
            #   호출해 채점에서 `PRED_EXTRA_KEY`로 실패했는데 V7 deny는 0이었다. 술어 자체는 그 인자로
            #   VIOLATION을 정상 판정한다(오프라인 확인) ⇒ 배선 문제.
            #   ⇒ 다른 deny 레버와 같은 자리(생성 레벨)로 옮긴다. 엔진은 여전히 인자를 떼지 않고
            #   deny+재발행만 한다(C151 compliance 패턴·[[10]] 분담).
            sig_fb = None
            # ★관찰 전용 모드(2026-07-31): 레버가 OFF여도 **술어는 평가해 로그만** 남긴다.
            #   V7을 끈 런에서도 "몇 번 물었을 것인가"가 남아야 §7 상쇄-arm의 모집단이 실측된다.
            #   `T2_TOOL_SIGNATURE=1`일 때만 실제 deny(`sig_fb`)를 세운다 — 동작 변화 0.
            _sig_on = os.environ.get("T2_TOOL_SIGNATURE") == "1"
            if _sig_on or os.environ.get("T2_TOOL_SIGNATURE_OBSERVE") == "1":
                # ★계측 추가(2026-07-31·Y2-B 26 sim 포렌식): V7은 이 사슬의 **맨 끝**이라 앞 레버가
                #   피드백을 잡으면 그 턴엔 발화하지 못한다. 실측 = deny 28회인데 **위반 호출 24건이
                #   그대로 채점에 도달**했고 cap(6)은 sim당 1.08회라 닿지도 않았다 ⇒ 선점이 유력하나
                #   로그가 그걸 구분하지 못했다. 그래서 술어는 **항상** 평가하고(순수함수·비용 0),
                #   못 뜬 턴엔 **누가 선점했는지**를 남긴다. 다음 런은 이 줄로 원인을 확정한다([[08]]).
                _chain = [("gate", do_gate), ("prov", do_prov), ("eplan", ep_fb), ("cons", cons_fb),
                          ("resolve_action", ra_fb), ("te", te_fb), ("wev", wev_fb),
                          ("resolve_write", rw_fb), ("toollist", tl_fb), ("dispatch_role", dr_fb),
                          ("prekb", pr_fb)]
                _blocker = next((n for n, v in _chain if v), None)
                _capped = (getattr(self, "_t2_signature_deny", 0)
                           >= int(os.environ.get("T2_TOOL_SIGNATURE_CAP", "6")))
                try:
                    import t2_signature as _sg
                    for c in (am.tool_calls or []):
                        _sv = _sg.signature_violation(getattr(c, "name", None), _args_dict(c), a2,
                                                      force=True)
                        if not _sv:
                            continue
                        if not _sig_on:      # 관찰 전용 — 레버는 꺼져 있고 로그만 남긴다
                            print("[T2_TOOL_SIGNATURE] would-deny tool=%s but observe-only"
                                  % getattr(c, "name", None), file=_sys.stderr, flush=True)
                            break
                        if _blocker or _capped:
                            print("[T2_TOOL_SIGNATURE] would-deny tool=%s but %s"
                                  % (getattr(c, "name", None),
                                     "capped" if _capped else "preempted-by=%s" % _blocker),
                                  file=_sys.stderr, flush=True)
                            break
                        sig_fb = (c, _sv)
                        print("[T2_TOOL_SIGNATURE] deny tool=%s" % getattr(c, "name", None),
                              file=_sys.stderr, flush=True)
                        break
                except Exception as _sge:
                    print("[T2_TOOL_SIGNATURE] 배선 예외(무시): %r" % (_sge,),
                          file=_sys.stderr, flush=True)

            un_fb = None
            _unspec = (a2 or {}).get("discoverable_name_check") or {}
            if (os.environ.get("T2_UNLOCK_NAME") == "1" and _unspec
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None
                    and getattr(self, "_t2_unlockname_deny", 0)
                    < int(os.environ.get("T2_UNLOCK_NAME_CAP", "6"))):
                # ★트리거 교체(2026-08-05·사용자 지시): 구판은 *"이름에 `_숫자`가 없다"* 는 **철자
                #   규칙**으로 발화했고 x99에서 **7발 7오발화**였다 — `verify_identity`·
                #   `check_cli_eligibility` 처럼 애초에 discoverable이 아닌 도구를 붙잡았다.
                #   env가 discoverable 목록을 주므로 술어는 **집합 밖인가**로 충분하다(패턴 0).
                # ★★거짓 진술 교정 (2026-08-06·022 루프 부검): 레지스트리를 **agent 측만** 봤다.
                #   `submit_cash_back_dispute_0589`는 **user-측 discoverable**(env `user_tools`)이라
                #   이 집합 밖으로 떨어졌고, 우리는 이름이 완전한 실재 도구에 대고 *"접미사가 없다·
                #   KB를 지금 검색하라"* 고 말했다. 022는 그 문구 6회 + UNKNOWN_NAME_BL 6회("그 이름은
                #   존재하지 않는다")를 번갈아 받으며 **40턴을 돌다 중단**됐다. 이름이 어느 쪽이든
                #   레지스트리에 있으면 이 레버는 할 말이 없다 — 채널이 틀린 것이고, 그건 TOOL-CHANNEL이
                #   이미 정확히 말한다(같은 궤적 turn 48·74). 여기서는 **침묵**이 교정이다(A2 순증 0).
                _env2 = getattr(getattr(self, "_t2_orch", None), "environment", None)
                _reg2 = _agent_discoverable(_env2)
                _regu2 = _user_discoverable(_env2)
                for c in (am.tool_calls or []):
                    _uarg = (_unspec.get("tools") or {}).get(getattr(c, "name", None))
                    if not _uarg:
                        continue
                    _uval = str(_args_dict(c).get(_uarg) or "")
                    if _uval and _reg2 and not _in_registry(_uval, _reg2) \
                            and not _in_registry(_uval, _regu2):
                        # ★{name_words} (2026-07-22 §2bs·rall10 050/052 실측): bare-name을 자연어
                        #   질의로 파생(suffix 제거+언더스코어→공백) — 도구명 질의=BM25 0.0 마찰의 해소.
                        #   엔진=순수 문자열 연산(리터럴 0)·문구는 A2.
                        # ★★지시-모순 교정 (2026-08-05·x95·048 실측): 구판 문구는 *"You do NOT know
                        #   the suffix - search the knowledge base NOW"* 라고 **단정**했다. 그런데 절차
                        #   문구는 같은 대화에서 *"the name above is complete - do not search"* 를 준다.
                        #   048은 두 지시를 **모두 따라** 중복 검색 8회를 쓰고 40여 메시지를 잃었다.
                        #   이건 모델의 불이행이 아니라 **우리 결정론의 버그**다. 레지스트리에서
                        #   이름이 풀리면 검색을 시키지 않고 **그 이름을 준다**; 못 풀 때만 종전 문구.
                        # ★★오발화 교정 (2026-08-05·x99: 이 레버는 **7발 7오발화**였다 —
                        #   `verify_identity`·`check_cli_eligibility` 처럼 **discoverable이 아닌
                        #   일반 도구**에 대해 "접미사가 없다·검색하라"고 말했고, 019는 그 때문에 t2에서
                        #   신원 검증을 잃었다. env는 discoverable 44종을 **목록으로** 준다 —
                        #   철자 규칙으로 대신할 이유가 없다([[22]]: 닫힌 사실은 권위 출처에서 읽는다).
                        #   접미사 없는 이름이 그 목록의 어떤 것과도 대응하지 않으면 **discoverable이
                        #   아닌 것**이고, 그때는 검색이 아니라 "직접 부르라"가 옳다.
                        # 여기 도달 = 이름이 discoverable 레지스트리 **밖**이다(집합 대조).
                        # 그 이름의 도구가 실재하면 unlock 대상이 아니라는 뜻이고, 실재하지도
                        # 않으면 없는 이름이다. 두 경우 모두 검색어를 **파생하지 않는다** —
                        # 접미사를 떼어 질의를 만드는 것도 패턴이다.
                        # ★정규화 불일치 교정(2026-08-06·022): `_known_tool_names`는 접미사를 **떼어**
                        #   집합을 만드는데(`_n`) 탐침은 원형이라, 접미사 있는 이름은 **영원히 미상**으로
                        #   판정돼 "접미사가 없다" 갈래로 떨어졌다. 양쪽을 같은 정규화로 맞춘다.
                        _known = re.sub(r"_\d+$", "", _uval) in _known_tool_names(
                            getattr(self, "tools", None), _env2, state.messages)
                        # ★잘못-접미사 갈래 (2026-08-12·j런 071t1 이 실제로 밟은 소비처):
                        #   base 가 같은 discoverable 이 레지스트리에 실재하면 "there is none" 은
                        #   거짓 — 그 금지문이 유일 복구 경로(KB 검색→정본 이름)를 차단했다.
                        _base_u = re.sub(r"_\d+$", "", _uval)
                        _same_u = any(re.sub(r"_\d+$", "", str(r)) == _base_u for r in (_reg2 or ()))
                        _tpl = ((_unspec.get("feedback_wrong_suffix") if _same_u else None)
                                or (_unspec.get("feedback_not_discoverable") if _known else None)
                                or _unspec.get("feedback")
                                or "Error: '{name}' is not a discoverable tool in this domain.")
                        # ★레지스트리 목록 동봉 — 소비처 #2(name-arg 분기)와 같은 근거·같은
                        #   키(2026-08-13·[[64]]). base 실재(_same_u)면 wrong_suffix 가 이미
                        #   경로를 말하므로 미동봉. 키 없으면 침묵=종전 거동.
                        if not _same_u:
                            _lstu = _unspec.get("feedback_registry_listing")
                            if _lstu and _reg2:
                                _tpl = str(_tpl) + str(_lstu).replace(
                                    "{names}", ", ".join(sorted(_reg2)))
                        un_fb = (c, str(_tpl).replace("{name}", _uval))
                        force_required = True     # ★사용자 제안: 재생성은 반드시 도구 호출(KB 검색 유도)
                        print("[T2_UNLOCK_NAME] deny bare name tool=%s val=%s"
                              % (getattr(c, "name", None), _uval), file=_sys.stderr, flush=True)
                        break
            # ★T2_UNKNOWN_NAME_BL (2026-07-22 §2bt·rall11 050 실측): env가 "Unknown ... tool"로
            #   이미 거부한 이름의 재시도 차단. 050: 환각 접미사 _8374를 env 에러 후에도 3연발 —
            #   에러 에코로 이름이 ctx에 들어가 PROV의 ctx-실재 검사가 무력화되는 구멍.
            #   엔진=env 에러 문자열의 인용명 수집(인터페이스 사실)+집합 소속 deny(리터럴 0).
            if (un_fb is None and os.environ.get("T2_UNKNOWN_NAME_BL") == "1"
                    and not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None
                    and getattr(self, "_t2_unknownbl_deny", 0)
                    < int(os.environ.get("T2_UNKNOWN_NAME_BL_CAP", "6"))):
                # ★채널-인식으로 교정(2026-07-31·Y2-B 실측) — 구판은 이름만 모아 **gold를 막았다**
                #   (task_017: agent 채널 거부 → user 채널 gold give를 18회 차단). `unknown_bl_*`
                #   헬퍼 docstring에 사고 전말·회귀 테스트 = `test_unknown_name_channel.py`.
                #   창(-14)은 유지하되 도구 대응은 **전 대화**서 모은다(호출-결과 짝이 창 밖일 수 있다).
                _ubl = getattr(self, "_t2_unknown_bl", None)
                _ukind = getattr(self, "_t2_unknown_kind", None)
                if _ubl is None:
                    _ubl = self._t2_unknown_bl = set()
                    _ukind = self._t2_unknown_kind = {}
                _b2, _k2 = unknown_bl_collect(state.messages)
                _ubl |= _b2
                _ukind.update(_k2)
                if _ubl:
                    _na3 = dict(((a2 or {}).get("dispatcher_role_check") or {}).get("name_args") or {})
                    for _k3, _v3 in ((_unspec.get("tools") or {}) if _unspec else {}).items():
                        _na3.setdefault(_k3, _v3)
                    for c in (am.tool_calls or []):
                        _ua3 = _na3.get(getattr(c, "name", None))
                        _uv3 = str(_args_dict(c).get(_ua3) or "") if _ua3 else ""
                        if unknown_bl_hit(_ubl, _ukind, getattr(c, "name", None), _uv3):
                            # ★거짓 진술 차단(2026-08-06·022): 아래 문구는 *"that exact name does not
                            #   exist"* 라고 단정한다. env가 agent 채널에서 거부한 이름이 **user-측
                            #   레지스트리에 실재하면 그 단정은 거짓**이고(존재하지 않는 게 아니라
                            #   채널이 틀렸다), 022는 이 거짓을 6회 받으며 루프에 갇혔다. 그 경우엔
                            #   침묵하고 TOOL-CHANNEL(이미 옳게 말한다)에 맡긴다.
                            if _in_registry(_uv3, _user_discoverable(
                                    getattr(getattr(self, "_t2_orch", None), "environment", None))):
                                continue
                            _upat3 = (_unspec.get("pattern") if _unspec else None) or "_[0-9]+$"
                            un_fb = (c, ("Error: '{name}' was already rejected by the environment "
                                         "as an unknown tool earlier in this conversation - that "
                                         "exact name does not exist and retrying it will fail again. "
                                         "Do NOT guess or reuse it. Search the knowledge base with "
                                         "plain words describing the step (for '{name}' e.g. "
                                         "\"{name_words}\") to find the correct full suffixed name, "
                                         "then retry with that name.")
                                     .replace("{name}", _uv3)
                                     .replace("{name_words}",
                                              re.sub(_upat3, "", _uv3).replace("_", " ").strip()))
                            self._t2_unknownbl_deny = getattr(self, "_t2_unknownbl_deny", 0) + 1
                            force_required = True
                            print("[T2_UNKNOWN_NAME_BL] deny env-rejected name tool=%s val=%s"
                                  % (getattr(c, "name", None), _uv3), file=_sys.stderr, flush=True)
                            break
            # ★가드 **앞**으로 옮겼다 (2026-08-12·`test_regen_break_guard.py` 가 잡았다).
            #   아래 break 가드는 세우는 fb 가 전부 None 이면 루프를 끊는다. 두 레버를
            #   가드 뒤에 두었더니 **그 턴의 유일한 발화일 때 계산조차 되지 않았다** —
            #   2026-08-05 `proc_fb` 사고와 **같은 실수**이고, 그때 만든 그 검정이
            #   이번에도 잡았다(내가 안 돌렸을 뿐이다). 라이브 실측: `Sky Blue Business
            #   Checking`(집합 外)이 나갔는데 ENUM deny **0회**.
            # ★공식 명칭 소속 (T2_WRITE_ARG_ENUM·기본 OFF·2026-08-12·C439⒠④).
            #   실측: 070/071 이 제출한 `account_class` **24건 중 19건이 존재하지 않는 이름**이다
            #   (`Lime Green Account`×5 · `Gold Saver`×4[on file `Gold Saver Account`] ·
            #    `Light Blue Business Checking Account`×2 · `Business Bronze Saver`×3 …).
            #   모델은 `Account` 를 **붙이기도 떼기도** 한다 — 규칙이 아니라 그때그때 고쳐 쓴다.
            #   ⚠술어는 **닫혀 있다**([[22]]): 후보 집합 **소속 판정**뿐이다. 엔진은 고르지
            #     않는다 — 8~10개가 그대로 남고 어느 것인지는 모델이 정한다([[62]] ③④).
            #   ⚠후보 출처는 **env 뿐**: `doc_index` 주어 슬러그의 기계 전개(파일명 유도·x244).
            #     엔진이 도메인 텍스트를 뜯지 않는다([[59]]) — A3 에 적힌 것을 읽기만 한다.
            #   ⚠**fail-open**: 축을 못 정하거나 후보가 비면 아무 말도 하지 않는다(모르면 막지
            #     않는다·[[25]]). 상한 = sim 당 `T2_WRITE_ARG_ENUM_CAP`(기본 3) — 살아 있는
            #     이름을 계속 거절하는 livelock 을 만들지 않기 위해서다.
            #   ★2026-08-24·R4: 그 상한이 세는 단위가 **거절 횟수 → 처음 보는 값의 수**로
            #     바뀌었다. 구판은 상한이 소진되면 블록 전체가 침묵해서 *이미 집합 밖이라
            #     판정한 그 값*이 통과했다(= fail-closed 가 fail-open 이 된다). 이제 원장에
            #     있는 값은 계속 거절하고, **처음 보는 값**만 상한 뒤에 통과한다. 자세한
            #     근거·실물은 아래 deny 자리 주석.
            #   ⚠[[64]]: 무엇이 틀렸는지(집합 밖)와 **무엇을 하면 풀리는지**(후보 명단)를 함께.
            en_fb = None
            # ★T2_SCHEMA_ENUM (2026-08-25·기본 OFF·[[64]] 형식): 인자 값이 **그 도구 스키마가
            #   선언한 enum** 밖이면 무엇이 틀렸는지와 **무엇을 쓰면 되는지**를 함께 돌려준다.
            #   근거(실측·정본 `t2_forensic.action_diff` 귀속): t7348 에서 040 의 gold 호출 8건이
            #   env 에 거절됐고 사유가 *"Invalid dispute_reason. Must be one of: [...]"* 였다.
            #   085 도 같은 계열 3건. 즉 모델은 **gold 거래 id 까지 맞히고** 열거값에서 되튕긴다.
            #   ⚠출처는 스키마 하나다 — gold 도 env 오류문도 우리가 지은 목록도 아니다([[23]]).
            #   ⚠술어는 닫혀 있다: 소속 여부만 본다. enum 이 없으면 아무 말도 안 한다(fail-open).
            #   ⚠[[62]]: 엔진이 값을 고르지 않는다 — 후보를 되돌려주고 고르는 것은 모델이다.
            if os.environ.get("T2_SCHEMA_ENUM") == "1":
                try:
                    import t2_role as _role2
                    _envr2 = getattr(getattr(self, "_t2_orch", None), "environment", None)
                    for c in (am.tool_calls or []):
                        if en_fb is not None:
                            break
                        _eff2 = _eff_tool_name(c)
                        _ad2 = _args_dict(c)
                        _in2 = _ad2.get("arguments")
                        if isinstance(_in2, str):
                            try:
                                _in2 = json.loads(_in2)
                            except Exception:
                                _in2 = {}
                        _vals2 = dict(_in2 or {})
                        for _k2, _v2 in _ad2.items():
                            if _k2 not in ("arguments", "agent_tool_name", "user_tool_name",
                                           "discoverable_tool_name"):
                                _vals2.setdefault(_k2, _v2)
                        for _k2, _v2 in _vals2.items():
                            if not isinstance(_v2, str) or not _v2:
                                continue
                            _en2 = _role2.enum_of(_eff2, _k2, agent=self, env=_envr2)
                            if not _en2 or _v2 in _en2:
                                continue
                            en_fb = (c, "`%s` is not a value that `%s` accepts for `%s`. "
                                        "Use exactly one of these: %s."
                                     % (_v2[:80], _eff2, _k2, ", ".join(_en2)))
                            print("[T2_SCHEMA_ENUM] deny tool=%s arg=%s val=%r (enum %d)"
                                  % (_eff2, _k2, _v2[:40], len(_en2)),
                                  file=_sys.stderr, flush=True)
                            break
                except Exception as _se2:
                    en_fb = None
                    print("[T2_SCHEMA_ENUM] error (no-op): %r" % (_se2,),
                          file=_sys.stderr, flush=True)
            # ★T2_WRITE_ARG_FAB (2026-08-25·기본 OFF): **자리표시자로 채운 인자**를 되돌려준다.
            #   왜 (t7354 grpB1 task_040 t0 궤적 축자): msg98 에서 에이전트가
            #   `give_discoverable_user_tool(get_card_last_4_digits, cc_01f21c9970_gold)` 로 도구를
            #   손님에게 넘기고 env 가 *"Tool given to user … The user can now execute this"* 를
            #   돌려준 **바로 다음 메시지 msg100** 에서, 손님이 실행하기도 전에 분쟁 4건을
            #   `card_last_4_digits='1234'` 로 접수했다. 첫 카드에서는 기다렸다 — msg76 의
            #   *"Last 4 digits of card: 0581"* 을 받아 msg78 부터 정확히 썼다. 같은 sim 에
            #   `transaction_id='TRXN1234567890'` 계열 4건도 있다.
            #   ⇒ 결손은 *아직 받지 못한 값을 쓰는 write* 이고, 그 값은 대화 어디에도 없다(전수 0건).
            #   ★술어 셋은 **전부 선언이거나 값의 모양**이고 이름 패턴이 **하나도 없다**
            #     (2026-08-25 사용자 지적으로 교체: `identifying_arg_types.digit` 는 철회했다):
            #       ⓐ env 가 그 인자를 `string` 이라 **선언**했고 열거 선언이 **없다**(`_declared_params`)
            #       ⓑ 값이 자리표시자 모양이다(`_looks_placeholder` — 연속·동일 자릿수 4)
            #       ⓒ 값이 대화 어디에도 없다(`_ctx_has` — 손님이 말해 줬으면 그대로 통과)
            #   실측 폭발 반경(t7354 6배치 전수·[[66]] 공유 상류 노드라 반드시 잰다):
            #     셋 결합 **20건 전부 040 의 진짜 날조**(오차단 0) ↔ ⓐ만 쓰면 gold 날짜 10건 오차단
            #     (`issue_noticed_date='11/14/2025'`) ↔ ⓑ만 쓰면 `min_credit_limit='10000'` 오차단.
            #   ⚠엔진은 값을 만들지도 고르지도 않는다 — *어디에도 없는 자리표시자*만 되돌려주고
            #     무엇을 부를지는 모델이 정한다([[62]]③④). 문면은 기존 검증분을 그대로 쓴다(C45).
            #   ⚠sim 당 인자별 1회(livelock 금지) · 형식이 아니면 무발화(fail-open·[[25]]).
            if os.environ.get("T2_WRITE_ARG_FAB") == "1" and en_fb is None:
                try:
                    _dpt = _declared_params_by_tool(state.messages)
                    if _dpt:
                        _fctx = _ctx_from_messages(state.messages)
                        _fseen = getattr(self, "_t2_argfab_deny", None)
                        if _fseen is None:
                            _fseen = self._t2_argfab_deny = set()
                        for c in (am.tool_calls or []):
                            if en_fb is not None:
                                break
                            _dp = _dpt.get(str(_exact_tool_name(c) or "")) or {}
                            for _fk, _fv in _prov_scan_args(
                                    c, selectors=_selector_args_cached(
                                        getattr(getattr(self, "_t2_orch", None),
                                                "environment", None))):
                                _ft = _dp.get(_fk)
                                if not _ft or _ft[0] != "string" or _ft[1]:
                                    continue        # 미선언·비문자열·열거 = 이 검사 밖
                                _fs = str(_fv).strip()
                                if len(_fs) < 4 or not _looks_placeholder(_fs):
                                    continue
                                if _ctx_has(_fs, _fctx):
                                    continue        # 문맥에 있으면 통과(우리는 판정하지 않는다)
                                _fkey = (str(_exact_tool_name(c) or ""), _fk)
                                if _fkey in _fseen:
                                    continue
                                _fseen.add(_fkey)
                                en_fb = (c, REGEN_FEEDBACK.format(k=_fk, s=_fs[:80]))
                                print("[T2_WRITE_ARG_FAB] deny tool=%s arg=%s val=%r "
                                      "(env 선언 string·열거 아님·자리표시자 모양·문맥 부재)"
                                      % (_eff_tool_name(c), _fk, _fs[:40]),
                                      file=_sys.stderr, flush=True)
                                _lbeat("T2_WRITE_ARG_FAB", orch=self,
                                       target=_eff_tool_name(c),
                                       fact="a real value read from the tool that produces it")
                                break
                    else:
                        print("[T2_WRITE_ARG_FAB] 관측: env 명세 블록 없음 — 무발화",
                              file=_sys.stderr, flush=True)
                except Exception as _fe:
                    en_fb = None
                    print("[T2_WRITE_ARG_FAB] 건너뜀(무발화): %r" % (_fe,),
                          file=_sys.stderr, flush=True)
            # ★T2_SPEC_ARG_FACTS (2026-08-25·기본 OFF) — **손 선언을 대체하는 파생**.
            #   사용자 물음(*"일반화로는 지금 문제를 해결 못하는 건가?"*)에 대한 답의 구현부다.
            #   오늘 A2 에 손으로 적은 것(값 목록 6칸·불리언 2세트)은 전부 env 가 unlock 때
            #   **고정 포맷으로 건네주는 것**이었다. 손으로 베낀 이유는 *discoverable 도구는 agent
            #   스키마 목록에 없다* 하나였는데, unlock 메시지가 같은 명세를 준다 ⇒ 막힌 채널이
            #   아니라 **안 뚫은 채널**이었다([[05]] 도메인-특화 순증의 뿌리).
            #   측정(둘 다 코퍼스 실물·gold 미접촉):
            #     `x540_spec_derivation.py`  명세 블록 61 · 도구 16 · 대조 9건 **전부 일치**
            #                                (다르다 0 · 대조 불가 0) + 우리가 선언한 적 없는
            #                                열거 3칸(`apply_credit_card_account_flag_6147` 2 ·
            #                                `open_bank_account_4821` 1)까지 덮는다
            #     폭발 반경                  도출이 손 선언보다 **새로 막는 것 0건**(t7354 전 배치)
            #   ⚠명세는 **도구별로** 읽는다 — `card_action` 은 신용 2값·직불 3값이라 이름만으로
            #     합치면 정당한 값을 거절한다.
            #   ⚠엔진은 고르지 않는다: 타입 사실과 소속 판정 + 명단 반환뿐이고 어느 값이 옳은지는
            #     모델이 정한다([[62]]③④·[[22]] 닫힌 술어). 문면에 도메인 낱말 0.
            #   ⚠sim 당 (도구,축) 1회 · 명세가 없으면 무발화(fail-open).
            if os.environ.get("T2_SPEC_ARG_FACTS") == "1" and en_fb is None:
                try:
                    _dpt2 = _declared_params_by_tool(state.messages)
                    _s2 = getattr(self, "_t2_specfacts_deny", None)
                    if _s2 is None:
                        _s2 = self._t2_specfacts_deny = set()
                    _sel2 = _selector_args_cached(
                        getattr(getattr(self, "_t2_orch", None), "environment", None))
                    for c in (am.tool_calls or []):
                        if en_fb is not None:
                            break
                        _tn2 = str(_exact_tool_name(c) or "")
                        _d2 = _dpt2.get(_tn2) or {}
                        if not _d2:
                            continue
                        _a2v = dict(_prov_scan_args(c, selectors=_sel2))
                        _bad2 = [k for k, v in _a2v.items()
                                 if (_d2.get(k) or ("", []))[0] == "boolean"
                                 and not isinstance(v, bool)]
                        if _bad2 and (_tn2, "\0bool") not in _s2:
                            _s2.add((_tn2, "\0bool"))
                            en_fb = (c, _SPEC_TYPE_FB % ("`, `".join(sorted(_bad2)),))
                            print("[T2_SPEC_ARG_FACTS] type deny tool=%s 비불리언 %d: %s"
                                  % (_eff_tool_name(c), len(_bad2), sorted(_bad2)),
                                  file=_sys.stderr, flush=True)
                            _lbeat("T2_SPEC_ARG_FACTS", orch=self, target=_eff_tool_name(c),
                                   fact="the type this tool declared when it was unlocked")
                            break
                        for _ek, _ev in sorted(_a2v.items()):
                            _en3 = (_d2.get(_ek) or ("", []))[1]
                            _es = str(_ev).strip()
                            if not _en3 or not _es or _es in _en3:
                                continue
                            if (_tn2, _ek, _es) in _s2:
                                continue
                            _s2.add((_tn2, _ek, _es))
                            en_fb = (c, _SPEC_ENUM_FB % (_es[:80], _ek, ", ".join(_en3)))
                            print("[T2_SPEC_ARG_FACTS] enum deny tool=%s arg=%s val=%r (후보 %d)"
                                  % (_eff_tool_name(c), _ek, _es[:40], len(_en3)),
                                  file=_sys.stderr, flush=True)
                            _lbeat("T2_SPEC_ARG_FACTS", orch=self, target=_eff_tool_name(c),
                                   fact="the values this tool declared when it was unlocked")
                            break
                except Exception as _s2e:
                    en_fb = None
                    print("[T2_SPEC_ARG_FACTS] 건너뜀(무발화): %r" % (_s2e,),
                          file=_sys.stderr, flush=True)
            _ens = (a2 or {}).get("write_arg_enum") or []
            if os.environ.get("T2_WRITE_ARG_ENUM") == "1" and _ens:
                # ★R4 (2026-08-24): 상한은 **블록 전체를 잠그던 것**에서 *처음 보는 값*의
                #   수를 세는 것으로 좁아졌다(아래 deny 자리 주석). `_encap_open` 은 종전
                #   조건 그대로라 상한에 매달려 있던 두 이웃 레버(ARG_AXIS·VERDICT_GATE)의
                #   발화 범위는 **한 글자도 넓히지 않는다**(거동 보존).
                _encap_open = (getattr(self, "_t2_enum_deny", 0)
                               < int(os.environ.get("T2_WRITE_ARG_ENUM_CAP", "3")))
                try:
                    _di = ((a2 or {}).get("policy_ontology") or {}).get("doc_index") or {}
                    for c in (am.tool_calls or []):
                        if en_fb is not None:
                            break
                        _ad = _args_dict(c)
                        for _sp in _ens:
                            if str(getattr(c, "name", "")) != str(_sp.get("applies_to")):
                                continue
                            _aw = _sp.get("applies_when") or {}
                            if _aw and not str(_ad.get(_aw.get("arg")) or "").startswith(
                                    str(_aw.get("prefix") or "\0")):
                                continue
                            _ia = _ad.get("arguments")
                            try:
                                _ia = json.loads(_ia) if isinstance(_ia, str) else (_ia or {})
                            except Exception:
                                _ia = {}
                            if not isinstance(_ia, dict):
                                continue
                            _val = str(_ia.get(_sp.get("arg")) or "").strip()
                            _gval = str(_ia.get(_sp.get("group_arg")) or "")
                            # ★축 표면화 (T2_ARG_AXIS·기본 OFF·C444). 모델은 축을 **말할 수
                            #   있는데**(라이브 문맥 8/8·부정통제 0/8) 인자에 안 썼다 —
                            #   `account_type="checking"` 로 개인 계좌를 열었다(gold=business).
                            #   엔진은 **LLM 출력 둘을 맞대기만** 한다: 격리 형식화 집합에
                            #   실제 인자가 **속하면 통과**(다중 요청 오차단 방지·C10).
                            #   ⚠고르지 않는다 — 어느 축이 옳은지 판정하지 않고, 다르다는
                            #     사실만 알린다([[62]] ③④).
                            if (en_fb is None and os.environ.get("T2_ARG_AXIS") == "1"
                                    and _encap_open        # ★R4 거동 보존(종전 상한 조건)
                                    and _sp.get("axis_prompt") and _gval
                                    and not getattr(self, "_t2_axis_deny", 0)):
                                # ⚠지역 임포트 — 초판은 존재하지 않는 이름을 썼고, 이 블록이
                                #   try 안이라 NameError 가 삼켜져 **레버가 조용히 죽었을**
                                #   것이다(오늘 `main_prov`·break-가드와 같은 종류).
                                import tau2.agent.llm_agent as _la_ax
                                from tau2.data_model.message import UserMessage as _UM_ax
                                # ⚠`_rz` 재임포트 필수 (h런 실측 UnboundLocalError ×2): 6389 의
                                #   `import t2_resolve as _rz` 는 resolve-계약 분기 **안**이라
                                #   그 분기가 안 돈 턴엔 지역 `_rz` 가 미대입이다. 이 블록은
                                #   자기 발로 서야 한다 — 죽은-레버 4호를 만들지 마라.
                                import t2_resolve as _rz
                                _want = _rz.formalize_arg_axis(
                                    self, _la_ax, _UM_ax, state.messages,
                                    _sp.get("group_arg"), list(_sp.get("group_map") or {}),
                                    _sp.get("axis_prompt"))
                                if _want and _gval not in _want:
                                    self._t2_axis_deny = 1
                                    en_fb = (c, _rz.ARG_AXIS_FB.format(
                                        arg=_sp.get("group_arg"), got=_gval,
                                        want=" and ".join(sorted(_want))))
                                    print("[T2_ARG_AXIS] deny got=%s want=%s"
                                          % (_gval, sorted(_want)),
                                          file=_sys.stderr, flush=True)
                                    break
                            # ★선언된 **타입** 검사 (2026-08-25·`T2_WRITE_ARG_TYPE`·기본 OFF).
                            #   왜(t7354 실측·085 와 040 전 분쟁): 도구 명세가 `(boolean)` 이라
                            #   선언한 인자에 모델이 **문자열 `"Yes"`/`"No"`** 를 보낸다.
                            #     085  written_statement_provided='Yes' · police_report_filed='No' ·
                            #          card_in_possession='Yes' … 접수된 분쟁 **전건**
                            #     040  contacted_merchant gold=True ↔ got='Yes' ·
                            #          eligible_for_provisional_credit gold=False ↔ got='Yes' · **8/8**
                            #   env 는 그것을 **받아 저장**하므로 호출은 성공하고 `db_match` 만
                            #   조용히 실패한다(040 trial0: gold 8건을 **전부 축자 접수**하고 reward 0).
                            #   ⚠**의미는 모델이 이미 맞혔다** — Yes↔True. 우리는 값을 바꾸지 않고
                            #     *선언된 타입*을 알려 주고 다시 내게 한다([[62]]③④ 판단 제거 0).
                            #   ⚠술어는 닫혀 있다: 선언된 이름 목록 + `isinstance(v, bool)` 뿐이고
                            #     엔진이 변환하지도 고르지도 않는다. 도메인 낱말은 A2 에만 있다([[05]]).
                            if en_fb is None and _sp.get("booleans"):
                                _bad = [(_bk, _ia.get(_bk)) for _bk in _sp["booleans"]
                                        if _bk in _ia and not isinstance(_ia.get(_bk), bool)]
                                if _bad and os.environ.get("T2_WRITE_ARG_TYPE") == "1":
                                    _tseen = getattr(self, "_t2_argtype_deny", None)
                                    if _tseen is None:
                                        _tseen = self._t2_argtype_deny = set()
                                    # 2026-08-28 수리 - 캡의 키를 **도구 이름**에서 **변이 키**로.
                                    #   캡은 재생성 무한루프를 막으려고 있는데, 도구 이름으로 잠그면
                                    #   *같은 도구를 여러 건 부르는* 태스크에서 **첫 건만** 고쳐지고
                                    #   나머지가 그대로 나간다. 실측(t7376 `task_040`): 한 sim 이
                                    #   `file_credit_card_transaction_dispute_4829` 를 8번 부르며
                                    #   `contacted_merchant`/`eligible_for_provisional_credit` 를
                                    #   문자열 `'true'`/`'false'` 로 보냈는데 이 레버는 런 전체에서
                                    #   **2회**(sim 당 1회)만 발화했고 **7건이 그대로 접수**됐다.
                                    #   env 는 문자열을 받아 저장하므로 호출은 성공하고 `db_match` 만
                                    #   조용히 실패한다 - 그래서 로그로도 안 보였다.
                                    #   변이 키(이름+인자 접기·`_mut_key_of` 정본 재사용·[[67]])로
                                    #   잠그면 **같은 호출의 재발행만** 막히고 새 건은 제 몫의 한 번을
                                    #   받는다. 술어는 그대로 닫혀 있다(`isinstance(v, bool)` 뿐).
                                    _tk = _mut_key_of(c) or str(_exact_tool_name(c) or "")
                                    if _tk and _tk not in _tseen:
                                        _tseen.add(_tk)
                                        en_fb = (c, str(_sp.get("type_feedback") or "").format(
                                            names=", ".join(
                                                "%s (you sent %r)" % (k, v) for k, v in _bad)))
                                        print("[T2_WRITE_ARG_TYPE] deny tool=%s 비불리언 %d: %s"
                                              % (_eff_tool_name(c), len(_bad),
                                                 [k for k, _ in _bad]),
                                              file=_sys.stderr, flush=True)
                                        _lbeat("T2_WRITE_ARG_TYPE", orch=self,
                                               target=_eff_tool_name(c),
                                               fact="the type this tool declares for these arguments")
                                        break
                                elif _bad:
                                    print("[T2_WRITE_ARG_TYPE] 관측(OFF) tool=%s 비불리언 %d: %s"
                                          % (_eff_tool_name(c), len(_bad), [k for k, _ in _bad]),
                                          file=_sys.stderr, flush=True)
                            if _sp.get("values"):
                                # ★값-목록 갈래 (2026-08-25): 후보를 A3 색인 슬러그가 아니라
                                #   **선언된 목록**에서 받는다. 왜 필요한가(t7348·정본 `action_diff`
                                #   귀속): 040 의 gold 호출 **8건**이 env 에 거절됐고 사유가
                                #   *"Invalid <arg>. Must be one of: [...]"* 였다. 085 도 같은 계열 3건.
                                #   모델은 gold 거래 id 까지 맞히고 **열거값에서** 되튕긴다.
                                #   ⚠스키마 경로는 막혀 있다(실측): agent 도구 17개 중 enum 을 선언한
                                #     인자는 **하나뿐**이고 표적 도구들은 discoverable 이라 그 목록에
                                #     없다 ⇒ 선언 경로가 유일하다.
                                #   ⚠출처는 **도구 사용법 문서 축자**다 — gold 도 env 오류문도 아니다
                                #     ([[23]]). 값마다 `_note_` 에 인용을 남긴다.
                                #   ⚠엔진은 여전히 고르지 않는다: 소속 판정 + 명단 반환뿐([[62]]③④).
                                _grp = "(declared)"
                                _subs = None
                                _names = [str(x) for x in (_sp.get("values") or [])]
                            else:
                                _grp = (_sp.get("group_map") or {}).get(_gval)
                                _subs = _di.get(_grp) or {}
                                _names = _display_slugs(_subs)
                            # ★fail-open 술어는 **명단** 기준이어야 한다(2026-08-22 누수 수리):
                            #   `_subs` 는 있는데 표시명이 하나도 없는 그룹이 실재하고
                            #   (`bank_accounts_bank_accounts` = `_general_` 하나뿐),
                            #   그때 `_subs` 로 판정하면 **빈 후보 명단으로 deny** 한다 =
                            #   [[64]] 의 "무엇을 하면 풀리나" 가 비어 버린다.
                            if not (_val and _grp and _names):
                                continue          # fail-open: 모르면 막지 않는다
                            if _val in _names:
                                # ★VC 호출-트리거 (T2_VERDICT_GATE·기본 OFF·C543ⓓ). 이름은 있는데
                                #   **손님 요구와 충돌하는** 값이면 LLM 자신의 판정 줄로 되돌린다.
                                #   판정·인용 = LLM · 엔진 = 조회 하나 · 상한 = sim 당 CAP(기본 1).
                                if (os.environ.get("T2_VERDICT_GATE") == "1"
                                        and _encap_open    # ★R4 거동 보존(종전 상한 조건)
                                        and getattr(self, "_t2_vgate_deny", 0)
                                        < int(os.environ.get("T2_VERDICT_GATE_CAP", "1"))):
                                    try:
                                        _vfb = _verdict_gate_fb(self, state.messages, a2,
                                                                _grp, _val, _subs, _sp)
                                    except Exception as _vge:
                                        _vfb = None
                                        print("[T2_VERDICT_GATE] 건너뜀(무발화): %r" % (_vge,),
                                              file=_sys.stderr, flush=True)
                                    if _vfb:
                                        self._t2_vgate_deny = getattr(self, "_t2_vgate_deny", 0) + 1
                                        en_fb = (c, _vfb)
                                        _lbeat("T2_VERDICT_GATE", orch=self,
                                               target=_eff_tool_name(c),
                                               fact="the customer's stated requirement")
                                        break
                                continue          # 집합 內 — 선택이 옳은지는 우리가 판정하지 않는다
                            # ★R4 거절 원장 (2026-08-24 수리 · refute C1 CONFIRMED).
                            #   구판은 **블록 전체**를 sim 당 상한으로 잠갔다. 그래서 상한이
                            #   소진되면 게이트가 침묵하고 *우리가 이미 집합 밖이라고 판정한
                            #   바로 그 값*이 다음 시도에 통과해 DB 를 바꿨다 — fail-closed 가
                            #   재시도 한 번으로 **fail-open** 이 된다. 실물:
                            #   `bank_t7296_treat_20260815p|task_071#s554706` 은 turn 22·34 에
                            #   같은 값을 두 번 deny 한 뒤 msg41 에 그 값으로 계좌를 열었고
                            #   (gold 3행 MATCHED 뒤) reward 가 0 이 됐다. 전 코퍼스 로그 455개:
                            #   deny 164줄/92 sim 중 20 sim 이 상한 도달 · 그중 4 sim 에서 집합
                            #   밖 값이 이후 성공 · **2 sim 에서 같은 값**이 성공.
                            #   ⇒ 상한의 **단위**를 바꾼다: '거절 횟수' → '처음 보는 값의 수'.
                            #     ⓐ 원장에 있는 값 = 횟수와 무관하게 **계속 거절**(우리 판정을
                            #       우리가 되돌려주지 않는다).
                            #     ⓑ 상한 소진 + **처음 보는 값** = 종전 그대로 fail-open —
                            #       livelock 탈출구는 살아 있다(우리 명단이 불완전할 때 sim 을
                            #       인질로 잡지 않는다·[[25]]).
                            #   ⚠판단 0([[62]] ③④): 술어는 `(그룹, 정규화 값)` 집합 소속뿐이고
                            #     `_val in _names` 를 **매번 먼저** 보므로, 명단이 나중에 그 값을
                            #     담게 되면 차단이 저절로 풀린다. 엔진은 여전히 고르지 않는다.
                            #   ⚠도메인 조건 0([[05]]·[[70]]): 태스크·상품·군 이름이 술어에
                            #     들어가지 않는다 — *"이 게이트가 이미 거절한 값"* 하나뿐이라
                            #     코퍼스 전체에 같은 모양으로 전이된다.
                            _seen = getattr(self, "_t2_enum_rejected", None)
                            if _seen is None:
                                _seen = self._t2_enum_rejected = set()
                            _rkey = _enum_seen_key(_grp, _val)
                            _again = _rkey in _seen
                            if not (_again or _encap_open):
                                continue      # 상한 소진 + 처음 보는 값 = 종전대로 fail-open
                            _seen.add(_rkey)
                            self._t2_enum_deny = (getattr(self, "_t2_enum_deny", 0)
                                                  + (0 if _again else 1))
                            en_fb = (c, str(_sp.get("feedback") or "").format(
                                val=_val, arg=_sp.get("arg"), group=_grp,
                                candidates=", ".join(_names)))
                            # ★[[64]]: 재제출에는 *무엇이 틀렸나*(이미 거절된 값이라 또 거절된다)
                            #   와 *무엇을 하면 풀리나*(위 명단에서 고르거나 쓰기 전에 조회) 를
                            #   함께 얹는다. 도메인 낱말 0 — 어느 벤치·어느 축에서도 같은 문장.
                            #   ⚠**뒤에** 붙인다: 문면 머리의 채널 마크(A2 가 넣는다)를 밀어내면
                            #     그 마크로 세는 사이드카 집계가 갈린다([[25]] 계기 오염 금지).
                            if _again:
                                en_fb = (en_fb[0], en_fb[1] + "\nYou submitted this exact value "
                                         "earlier in this conversation and it was already rejected "
                                         "for the same reason; it is refused again rather than "
                                         "written. Retrying it will not make it valid - either use "
                                         "one of the names listed above verbatim, or look the "
                                         "correct name up with a read tool before writing.")
                            # ★이 축의 결정이 이미 있으면 **함께** 싣는다 (2026-08-13·071 t1).
                            #   구판은 후보 8~10개만 줬다 = 메뉴. 실측에서 그 메뉴는 오답을
                            #   매끄럽게 확정시켰다(`True Blue Business Checking` → `True Blue`).
                            #   서브가 이 축에서 이미 낸 답(`Sky Blue`)이 저장돼 있는데 그 자리에
                            #   없었던 것이 유일한 교정 기회의 낭비였다. 새 판단 0 — 저장된
                            #   **LLM 자신의 출력**을 그대로 재제시할 뿐이고, 무엇을 쓸지는
                            #   여전히 모델이 정한다([[62]] ③④·[[64]] 무엇을 하면 풀리나).
                            _dsav = (getattr(self, "_t2_axis_decision", None) or {}).get(_grp)
                            if _dsav:
                                en_fb = (en_fb[0], en_fb[1] + "\n" + str(_dsav))
                                print("[T2_WRITE_ARG_ENUM] 저장된 축 결정 동봉 group=%s (%d자)"
                                      % (_grp, len(str(_dsav))), file=_sys.stderr, flush=True)
                            # ★계기: 재제출 거절을 **다른 마크**로 찍는다. 상한이 세는 것은
                            #   이제 *처음 보는 값*이므로, 구판처럼 `deny val=` 줄 수를 세서
                            #   `_t2_enum_deny` 를 역산하면 틀린다(refute C1 정정 ⓑ 동형).
                            print("[T2_WRITE_ARG_ENUM] %s val=%r group=%s (후보 %d · 원장 %d)"
                                  % ("deny(재제출)" if _again else "deny",
                                     _val, _grp, len(_names), len(_seen)),
                                  file=_sys.stderr, flush=True)
                            _lbeat("T2_WRITE_ARG_ENUM", orch=self, target=_eff_tool_name(c),
                                   fact="the official names on file for this product group")
                            break
                except Exception as _ene:
                    en_fb = None
                    print("[T2_WRITE_ARG_ENUM] 건너뜀(무발화): %r" % (_ene,),
                          file=_sys.stderr, flush=True)
            # ★결정-선행 write (T2_DECIDE_BEFORE_WRITE·기본 OFF·2026-08-12·사용자 지시:
            #   *"순서를 지켜야 하는 부분들은 e-plan 이나 절차 엔진을 통해서 강제해도 된다"* ·
            #   *"read 에 의한 결정점이 나와야 발화하는게 맞다"*).
            #   당연한 순서 read → 결정 → write 가 코드에 없었다: 결정점은 조언 턴
            #   (`_agent_ending`)에만 열려 070 에서 서브의 정답('Sky Blue Account')이
            #   **쓰기 두 메시지 뒤에** 도착했다(bank_all6b msg36 write · 발화 대화텍스트 21).
            #   규칙 한 줄: **이 대화에 결정 재료가 없으면 write 를 1턴 미루고, 그 자리서
            #   같은 서브를 돌려 재료를 담아 돌려준다**([[64]] — 무엇이 없었고 무엇이 답인지).
            #   ⚠write 강제 아님 — 지연 1턴뿐. 서브가 침묵하면(재료·now 없음) **그냥 통과**
            #     (막다른 골목 금지). 값 선택은 전부 LLM(서브+메인) 몫([[62]] ③④).
            #   ⚠술어 전부 닫힘([[22]]): write 집합=A2 도출 · "재료 있었나"=`_t2_search_done`
            #     공집합 판정 · 도메인 낱말 0.
            # ★T2_DUP_WRITE (2026-08-26·기본 OFF) — **이미 성공한 변이의 재실행을 지운다**.
            #   측정 선행([[62]]①): x546/x547 재생 — 중복 실행을 전부 빼면 만점 sim **14/14 불변**
            #   (비용 0)이고 0점 sim 142 중 **8** 이 1.0 으로 뒤집힌다(074·073·050). 정제 술어
            #   (*순수 반복만*)는 8 중 1 만 살려 더 나쁘다. x548 격리 — 이 문면은 재발행을
            #   **4/4 → 0/4** 로 막고, 이름 없는 거절(4/4)·같은 길이 무관 문장(4/4)은 못 막는다
            #   ([[57]] 부정통제 통과).
            #   ⛔**stub 이 아니라 재생성 채널**이다. 읽기 dedup 이 write 를 제외하는 이유가
            #     그것이다(2026-08-02 `failed_setstate_1785632213670`: 스텁이 히스토리에 남아
            #     eval 재실행과 어긋나 sim 무효). 재생성은 호출을 통째로 지우므로 그 문제가 없다.
            #   ⚠**알려진 노출**: 051 은 gold 가 *거절·상환 뒤 같은 인자 재제출*을 요구한다
            #     ([2]↔[17] 동일). 이 가드는 그것도 막는다 — x548 에서 탈출 단서를 붙인 판도
            #     열지 못했다(0/4). 051 은 코퍼스 전 sim 이 0점이라 실제로 잃은 점수는 없다.
            #   ⚠술어는 전부 닫혀 있다([[22]]): 실행 이름 동등성 + 인자 접기 + 결과 오류 여부.
            #     도메인 낱말 0 · 도구 이름 열거 0 · gold 미접촉([[23]]).
            dup_fb = None
            if os.environ.get("T2_DUP_WRITE") == "1":
                try:
                    _dupmap = _succeeded_mut_keys(state.messages, _a2_of(self))
                    for _dc in (am.tool_calls or []):
                        if not _is_effective_write(_eff_tool_name(_dc), _a2_of(self)):
                            continue
                        # 2026-08-28 - 정책이 선언한 유일성 키(`write_once_keys`)를 먼저 본다.
                        #   `_mut_key_of` 는 인자 전체를 키로 쓰므로 *같은 계좌·다른 금액* 의
                        #   재적용을 통과시켰다(t7378 `task_074#s361454`: 14.5 뒤 30.0).
                        _dk = None
                        for _cand in (_once_key_of(_dc, _a2_of(self)), _mut_key_of(_dc)):
                            if _cand and _cand in _dupmap:
                                _dk = _cand
                                break
                        if not _dk:
                            continue
                        _dat, _dres = _dupmap[_dk]
                        _tpl = (_DUP_WRITE_ONCE_FB if str(_dk).startswith("once|")
                                else _DUP_WRITE_FB)
                        dup_fb = (_dc, _tpl.format(at=_dat, result=str(_dres)[:700]))
                        print("[T2_DUP_WRITE] deny tool=%s (앞선 성공 msg=%s)"
                              % (_eff_tool_name(_dc), _dat), file=_sys.stderr, flush=True)
                        _lbeat("T2_DUP_WRITE", orch=self, target=_eff_tool_name(_dc),
                               fact="this exact change already succeeded in this conversation")
                        break
                except Exception as _dupe:
                    dup_fb = None
                    print("[T2_DUP_WRITE] 건너뜀(무발화): %r" % (_dupe,),
                          file=_sys.stderr, flush=True)
            #   ⚠sim 당 1회(cap) — 두 번 미루면 지연이 손실이 된다(Δspurious 계측 동반·§8 P4).
            dw_fb = None
            if (os.environ.get("T2_DECIDE_BEFORE_WRITE") == "1" and not do_gate
                    and ep_fb is None and dd_fb is None
                    and cons_fb is None and ra_fb is None and te_fb is None
                    and wev_fb is None and tr_fb is None and proc_fb is None
                    and rw_fb is None and tl_fb is None and sig_fb is None
                    and un_fb is None and pc_fb is None and dr_fb is None and pr_fb is None
                    and not getattr(self, "_t2_dwrite_deny", 0)):
                # ★C439⒝ 교정: 초판 가드가 `not _t2_search_done`(=결정이 **아직 없으면** 유예)
                #   이었는데, 실측된 실패는 *결정이 **있는데 쓰이지 않는** 것*이라 **정반대
                #   조건**을 보고 있었다 — 그래서 8 sim 에서 0회 발화(P0 실패). 이제 조건을
                #   빼고, 이미 축이 처리돼 서브가 침묵하면 **그때 낸 답을 그대로 다시 낸다**
                #   (새 판단 0 · 저장해 둔 LLM 출력의 재제시 = C301 `_t2_deferred` 와 같은 형태).
                try:
                    _wrset = _confirm_write_tools(a2) | set(
                        ((a2 or {}).get("eplan") or {}).get("write_tools") or [])
                    # ★T2_DOCS_AT_WRITE (2026-08-16·기본 OFF·t7304 재설계) — **선택을 담은 write**
                    #   자리를 write 집합에 넣는다. 왜: `eplan.write_tools` 가 이 태스크군의 write
                    #   도구를 담고 있지 않아 이 자리가 **구조적으로 발화 0** 이었다(t7303 실측·
                    #   팔당 0회). 그런데 t7303 전수는 배달이 turn 2·6 에 끝나고 손님이 요구를
                    #   진술하는 것은 그 **뒤**임을 보였다(8/8·간격 중앙 29.5 메시지). 재료는 한 턴만
                    #   살므로 결정 순간엔 없다. 반면 이 자리는 모델이 **직접 값을 쓰겠다고 나선
                    #   순간**이라 요구가 이미 진술돼 있다.
                    #   집합 출처 = **A2 가 이미 선언한 선택-인코딩 write**(새 키 0):
                    #     `choice_grounding[].tool` · `recommendation_verify.action_tool`.
                    #   ⇒ 선언이 없는 write(인증·조회·그 밖)는 안 걸린다 — 부정통제 태스크가
                    #     그 부류라 통제가 보존된다. sim 당 1회(`_t2_dwrite_deny`).
                    if os.environ.get("T2_DOCS_AT_WRITE") == "1":
                        _wrset |= {c.get("tool") for c in ((a2 or {}).get("choice_grounding") or [])
                                   if c.get("tool")}
                        _rv = (a2 or {}).get("recommendation_verify") or {}
                        if _rv.get("action_tool"):
                            _wrset.add(_rv["action_tool"])
                    _wc = next((c for c in (am.tool_calls or [])
                                if _eff_tool_name(c) in _wrset
                                or getattr(c, "name", "") in _wrset), None)
                    if _wc is not None:
                        # ★축-정합 재제시 (2026-08-13·071 t1 부검): 저장이 단일 슬롯일 때는
                        #   savings 결정이 checking 결정을 덮어 **틀린 축의 답**을 되돌려
                        #   줄 수 있었다. 이 write 의 축은 A2 `write_arg_enum.group_map` 이
                        #   이미 선언한다(집합 대조뿐·엔진 해석 0).
                        # ★★R7 (2026-08-23·`refute_5.json` §surviving⑵): 축(group)만으로는
                        #   부족했다 — 캐리 문면이 **어느 인자의 답인지** 한 번도 말하지 않아
                        #   문서 계열 라벨(`General`)이 `dispute_category` 자리로 흘러들었고
                        #   그 sim 의 11회 시도가 전부 열거 밖 값으로 실패했다. 이제 **인자
                        #   이름**을 A2 선언에서 함께 읽는다(`_write_choice_arg` — 이름 동등성·
                        #   선언된 접두·dict 조회뿐·[[22]]). 못 대면 **나가지 않는다**.
                        _darg, _dax = _write_choice_arg(a2, _wc)
                        # ★2026-08-26 배치 수리 (x543 재생 · `x543_spec_at_write_reach_2026_08_26.json`):
                        #   계기와 아래 **선언 배달**(SPEC/RULE/ARG_POLICY)은 `_darg` 와 **무관**하다.
                        #   종전엔 셋 다 `if not _darg:` 안에 있었는데, t7356 33 sim·778 호출을 재생하니
                        #   이 가지 앞까지 온 write **29건이 29/29 전부 `_darg` 를 댔다**(`dispute_category`
                        #   18 · `dispute_reason` 11) ⇒ 셋은 조건을 **볼 기회조차** 없었고 도달 표지는
                        #   15 배치 중 14 에서 0 이었다. 격리는 멀쩡했으므로(x532 A_asis 1/6 ↔ B 6/6 ·
                        #   x537 0/12 ↔ 12/12 · x538 A_asis 12/20 ↔ B_rule 20/20 ↔ N_len 12/20 —
                        #   큐 `findings_2026_08_25_night.N2` 인용) 자격이 아니라 **위치**가
                        #   틀렸다([[76]]⒜). 셋이 산 결손은 전부
                        #   *이름을 댈 수 있을 때* 나는 것들이다(x532=이름이 틀림 · x537=키 17/17 정확한데
                        #   늦은 중복 · x538=책임 한도 티어) ⇒ `not _darg` 는 **캐리의 조건**이지 이들의
                        #   조건이 아니다.
                        _spec, _si, _sd = _env_spec_for(_wc, state.messages)
                        print("[T2_SPEC_DIST] tool=%s src_msg=%s dist=%s len=%s darg=%s"
                              % (_eff_tool_name(_wc), _si, _sd,
                                 len(_spec) if _spec else 0, _darg or "-"),
                              file=_sys.stderr, flush=True)
                        if not _darg:
                            # ★[[64]] 의 두 번째 가지: *이름을 못 대면 말하지 마라*. A2 가 이
                            #   write 의 선택 인자를 선언하지 않았다는 것은 우리가 **무엇에 대한
                            #   답인지 모른다**는 뜻이고, 그 상태의 캐리는 값을 아무 슬롯에나
                            #   놓으라는 초대가 된다(085 실물). 서브도 여기서 돌리지 않는다 —
                            #   `_search_material` 은 축 잠금(`_t2_search_done`)을 소모하므로,
                            #   배달하지 않을 결정을 위해 축을 태우면 그 축이 영영 안 온다.
                            #   ⚠도메인 조건 0: 태스크·상품 이름이 아니라 *선언의 유무*가 술어다.
                            # ★계기 (2026-08-25·거동 변경 0·사용자 지시 *"수리할 방법이 없으면
                            #   다음 런을 위해 원인파악을 위한 장치라도 달아두라"*): 이 자리가
                            #   침묵할 때 **재료가 얼마나 뒤에 있었는지**를 남긴다. 085 를 가른
                            #   것이 정확히 이 수(거리 46·58)였고, 그것이 없어서 핸드오프는
                            #   결손을 *"키 허용목록이 없다"* 로 잘못 적었다. 세 축의 공통
                            #   진단(*"재료는 상류에 있고 결정점에 없다"*)을 코퍼스 전체에서
                            #   grep 하나로 세게 하는 것이 목적이다.
                            print("[T2_DECIDE_BEFORE_WRITE] 축 미상 — 캐리 무발화 tool=%s "
                                  "(A2 가 이 write 의 선택 인자를 선언하지 않았다)"
                                  % (_eff_tool_name(_wc),),
                                  file=_sys.stderr, flush=True)
                        else:
                            _saved = (getattr(self, "_t2_axis_decision", None) or {}).get(_dax) \
                                if _dax else None
                            # ★A-7⑸ (2026-08-23·055): **어느 축의 결정문을 실었는지** 남긴다.
                            #   구판 로그는 `(재료 %d자)` 뿐이라, 두 축의 결정문이 둘 다 247자인
                            #   055 에서는 무엇이 실렸는지 로그만으로 가릴 수 없었다([[25]]).
                            _dsrc = "search"
                            _dmat = _search_material(self, a2, state.messages)
                            if not _dmat:
                                _dsrc = "saved" if _saved else "last"
                                _dmat = _saved or getattr(self, "_t2_last_decision", "")
                            if _dmat:
                                self._t2_dwrite_deny = 1
                                dw_fb = (_wc, _DECIDE_FIRST_FB.format(arg=_darg,
                                                                      material=_dmat))
                                # ★A-7⑸ (2026-08-23·055): 축 이름과 출처를 병기한다 — 두 축의
                                #   결정문이 같은 길이면 로그만으로는 무엇을 실었는지 알 수 없다.
                                #   ★R7: 인자 이름도 함께 — 배달 문면이 무엇을 지목했는지가
                                #     로그에서 검산돼야 한다([[25]] 계기는 100% 정답 의무).
                                print("[T2_DECIDE_BEFORE_WRITE] write 1턴 유예 tool=%s arg=%s "
                                      "axis=%s src=%s (재료 %d자)"
                                      % (_eff_tool_name(_wc), _darg, _dax or "-",
                                         _dsrc, len(_dmat)),
                                      file=_sys.stderr, flush=True)
                                _lbeat("T2_DECIDE_BEFORE_WRITE", orch=self,
                                       target=_eff_tool_name(_wc),
                                       fact="the decision this write encodes, made before it runs")
                        # ★자리를 **뺏지 않고 덧붙인다**: 캐리는 같은 두 도구·같은 인자에서 발화한다
                        #   (t7356 4/4 — 074#s361454 · 085#s373753 · 085#s361454 · 040#s626729) ⇒ 우선순위를
                        #   매기면 어느 한쪽이 영영 0 이 된다. sim 당 유예는 그대로 **1회**고, 캐리가 이미
                        #   미뤘으면 같은 메시지에 **실어** 보낸다(`en_fb` 가 위에서 쓰는 그 관용).
                        _carry_hold = dw_fb
                        dw_fb = None
                        # ★T2_SPEC_AT_WRITE (기본 OFF·격리 x532 통과 후 배선·[[78]]).
                        #   격리: A_asis 1/6 ↔ **B_spec 6/6** ↔ N_neg 2/5(같은 길이 무관 블록)
                        #   ⇒ 산 것은 길이가 아니라 **내용**이고([[57]]), A_asis 가 라이브
                        #   오답 키를 재현했으므로 격리가 공정하다([[62]] 2b).
                        #   하는 일은 **전달 하나**다 — env 가 앞서 보낸 그 응답을 자르지도
                        #   고르지도 않고 되붙인다. 값 선택은 전부 모델 몫이다([[62]]③④).
                        #   sim 당 도구별 1회 — 두 번 미루면 지연이 손실이 된다.
                        if (os.environ.get("T2_SPEC_AT_WRITE") == "1" and _spec
                                and _sd >= int(os.environ.get("T2_SPEC_AT_WRITE_MIN", "8"))):
                            _sseen = getattr(self, "_t2_spec_at_write", None)
                            if _sseen is None:
                                _sseen = self._t2_spec_at_write = set()
                            _skey = str(_exact_tool_name(_wc) or "")
                            if _skey and _skey not in _sseen:
                                _sseen.add(_skey)
                                self._t2_dwrite_deny = 1
                                dw_fb = _decl_join(dw_fb, _wc,
                                                   _SPEC_AT_WRITE_FB.format(dist=_sd,
                                                                            spec=_spec))
                                print("[T2_SPEC_AT_WRITE] write 1턴 유예 tool=%s "
                                      "src_msg=%s dist=%s (%d자 재제시)"
                                      % (_eff_tool_name(_wc), _si, _sd, len(_spec)),
                                      file=_sys.stderr, flush=True)
                                _lbeat("T2_SPEC_AT_WRITE", orch=self,
                                       target=_eff_tool_name(_wc),
                                       fact="what this tool itself declared when it "
                                            "was made available")
                        # ★T2_RULE_AT_WRITE (기본 OFF·격리 x537 통과 후 배선·[[78]]).
                        #   격리(085·창 3·n4): 창 그대로 **0/12** ↔ 선언 문장 한 줄을 결정점에
                        #   놓으면 **12/12** ↔ 같은 길이 무관 문장 **0/12**([[57]] 통과·
                        #   A_asis 가 라이브 오답을 재현하므로 공정 [[62]]2b).
                        #   ⚠SPEC_AT_WRITE 와 **다른 자리**를 산다: 그쪽은 인자 **이름**이 틀렸을
                        #     때고, 이쪽은 이름이 다 맞는데 **어느 기록을 고르나**가 틀릴 때다
                        #     (085 실물: 키 17/17 정확한 호출이 늦은 중복을 골랐다).
                        #   ⚠엔진은 검색도 순위도 하지 않는다 — **선언된 문장을 그대로** 싣는다.
                        #     출처 의무는 A2 `_note_` 에 있다([[23]] 정책 축자).
                        if (os.environ.get("T2_RULE_AT_WRITE") == "1"
                              and _declared_rules_for(_wc, a2)):
                            _rseen = getattr(self, "_t2_rule_at_write", None)
                            if _rseen is None:
                                _rseen = self._t2_rule_at_write = set()
                            _rkey2 = str(_exact_tool_name(_wc) or "")
                            if _rkey2 and _rkey2 not in _rseen:
                                _rseen.add(_rkey2)
                                _rtxt = _declared_rules_for(_wc, a2)
                                self._t2_dwrite_deny = 1
                                dw_fb = _decl_join(dw_fb, _wc, _RULE_AT_WRITE_FB.format(
                                    dist=(_sd if _sd > 0 else 0), rules=_rtxt))
                                print("[T2_RULE_AT_WRITE] write 1턴 유예 tool=%s (%d자 규칙)"
                                      % (_eff_tool_name(_wc), len(_rtxt)),
                                      file=_sys.stderr, flush=True)
                                _lbeat("T2_RULE_AT_WRITE", orch=self,
                                       target=_eff_tool_name(_wc),
                                       fact="the procedure the documents state for this write")
                        # ★T2_ARG_POLICY_AT_WRITE (2026-08-25·기본 OFF) — `write_rules` 의
                        #   **일반형**. 손으로 고른 문장 대신, 이 write 가 **선언한 인자
                        #   이름**과 A3 행의 `axis` 가 **같은** 행을 축자로 놓는다.
                        #   인자 이름은 env 명세에서 나오고(`_declared_params_for`) 문장은
                        #   A3 선언에서 나온다 — 우리가 고른 것은 **아무것도 없다**.
                        #   조인 커버리지 실측(t7354 명세): 신용 분쟁 인자 15 중 **13**,
                        #   직불 17 중 9 에 행이 붙는다. 실물 — 040 의 열린 축이 거기 있다:
                        #   `eligible_for_provisional_credit` → *"Agent must determine this
                        #   based on the Provisional Credit Eligibility Guidelines article in
                        #   this knowledge base. Pass true or false."*([[64]] 무엇을 하면 풀리나)
                        #   ⚠유사도 검색이 아니라 **동일성**이다 — 어제 폐기한 토큰 검색기와
                        #     다른 종류다([[71]]③). 순위 0 · 상한 넘으면 전부 안 준다.
                        #   ★2026-08-26 ON — 모델 반응을 쟀다(`x551`·040 그 축):
                        #     A_asis **2/4** ↔ B_rule **4/4** ↔ N_len **2/4**(부정통제가 A 와
                        #     행별 답까지 동일). 창은 **전 접두**여야 한다 — 12메시지×1500자로
                        #     자르면 A_asis 가 3/4 로 올라 결손 자체가 사라진다.
                        #   ★그리고 위 인용은 **포인터였다**(= A_asis 조건 그 자체). 같은 날 A3
                        #     그 행에 Guidelines 문서의 기준 5개를 축자로 더해 **선언을 완결**
                        #     시켰다(doc `..._015`·827자·[[72]]). 배달은 3,033자(cap 4000).
                        #   ⚠격리가 잰 것(기준 827자)과 이 조인이 보내는 것(3,033자)은 **다르다**.
                        #     추가 부하가 해로운지는 런이 판정한다([[70]] ± 를 공개한다).
                        if os.environ.get("T2_ARG_POLICY_AT_WRITE") == "1":
                            _pargs = list(_declared_params_for(state.messages, _wc) or {})
                            _ptxt = _policy_rows_for(a2, _pargs)
                            _pseen = getattr(self, "_t2_argpolicy_deny", None)
                            if _pseen is None:
                                _pseen = self._t2_argpolicy_deny = set()
                            _pkey = str(_exact_tool_name(_wc) or "")
                            if _ptxt and _pkey and _pkey not in _pseen:
                                _pseen.add(_pkey)
                                self._t2_dwrite_deny = 1
                                dw_fb = _decl_join(dw_fb, _wc, _RULE_AT_WRITE_FB.format(
                                    dist=(_sd if _sd > 0 else 0), rules=_ptxt))
                                print("[T2_ARG_POLICY_AT_WRITE] write 1턴 유예 tool=%s "
                                      "(선언 인자 %d · 정책 인용 %d자)"
                                      % (_eff_tool_name(_wc), len(_pargs), len(_ptxt)),
                                      file=_sys.stderr, flush=True)
                                _lbeat("T2_ARG_POLICY_AT_WRITE", orch=self,
                                       target=_eff_tool_name(_wc),
                                       fact="what the documents say about the arguments "
                                            "this call declares")
                            else:
                                print("[T2_ARG_POLICY_AT_WRITE] 무발화 tool=%s "
                                      "(선언 인자 %d · 조인 %s)"
                                      % (_eff_tool_name(_wc), len(_pargs),
                                         "0행" if not _ptxt else "이미 배달"),
                                      file=_sys.stderr, flush=True)
                        if dw_fb is None:
                            print("[T2_DECIDE_BEFORE_WRITE] 선언 배달 무발화 tool=%s "
                                  "(darg=%s · 명세 %d자 dist=%s)"
                                  % (_eff_tool_name(_wc), _darg or "-",
                                     len(_spec) if _spec else 0, _sd),
                                  file=_sys.stderr, flush=True)
                        if _carry_hold is not None:
                            dw_fb = (_carry_hold[0], str(_carry_hold[1])
                                     + ("\n\n" + str(dw_fb[1]) if dw_fb else ""))
                except Exception as _dwe:
                    dw_fb = None
                    print("[T2_DECIDE_BEFORE_WRITE] 건너뜀(무발화): %r" % (_dwe,),
                          file=_sys.stderr, flush=True)
            # ★`proc_fb` 누락 교정 (2026-08-05·스모크 g 포렌식·`ABSENCE_DRIVEN_PROCEDURE_DESIGN` §1.5):
            #   이 가드는 루프에서 세우는 fb 15종 중 **`proc_fb` 하나만 빠뜨리고 있었다**(AST 전수 확인).
            #   결과: 절차 레버가 그 턴의 **유일한** 발화면 여기서 루프가 끊겨 (a) 피드백 조립(§5336)에
            #   닿지 못해 **deny 문구가 모델에게 전달된 적이 없고**, (b) 카운터(§5279)도 오르지 않아
            #   sim당 cap이 한 번도 물리지 않았다(048에서 11회 deny·cap 6). 로그의 `[T2_PROCEDURE] deny`는
            #   **차단이 아니라 인쇄**였다 ⇒ "차단했는데 이행하지 않았다"는 판정 전부 무효(§1.5 철회표).
            #   재발 방지 = `test_regen_break_guard.py`(AST로 fb 전수 대조).
            if (not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None and un_fb is None and dr_fb is None and pc_fb is None
                    and pr_fb is None and hv_fb is None and dd_fb is None and sig_fb is None
                    and proc_fb is None and abs_fb is None and tr_fb is None and wd_fb is None
                    and fs_fb is None and rdd_fb is None
                    and dw_fb is None and en_fb is None and dup_fb is None):
                break
            main_prov = None
            if do_prov and fab is None:
                # ★L3 origin 케이스: 값∈ctx이나 assistant-first ∧ tool-never = 확인-세탁(t97)
                prov_rounds += 1
                ptc, k, s = ofab
                self._t2_session_bl.add(s)
                self._t2_prov_origin = getattr(self, "_t2_prov_origin", 0) + 1
                print("[T2_PROV] origin regen fired tool=%s arg=%s val=%s"
                      % (getattr(ptc, "name", "?"), k, s), file=_sys.stderr, flush=True)
                _directive = _resolver_directive(a2, ptc, k, s)
                main_prov = (ptc, _directive if _directive is not None
                             else ORIGIN_FEEDBACK.format(k=k, s=s))
            elif do_prov:
                prov_rounds += 1
                ptc, k, s = fab
                self._t2_session_bl.add(s)
                self._t2_regen = getattr(self, "_t2_regen", 0) + 1
                # 관측성(행동 무변경): p4-비용 귀속용 발화 로그 (C53 p4 −5.3pp·§3c)
                print("[T2_PROV] regen fired tool=%s arg=%s val=%s" % (getattr(ptc, "name", "?"), k, s),
                      file=_sys.stderr, flush=True)
                _directive = _resolver_directive(a2, ptc, k, s)  # ★B-max① t17: resolver_path 지목
                # ★PROV-ADDR-FULL (T2_PROV_ADDR_FULL=1·2026-07-12 HANDOFF §6.3·A1 부작용 교정):
                #   주소류 free-text 인자는 rescue 중립문(getter-일반)이 약해 날조가 통과(t43/96) →
                #   full 문구(REGEN_FEEDBACK=주소-getter 명시 예시)로 강제 조회 유도. rescue 스킵은
                #   env-검증 id(hashid/numid)만 — 주소=free-text는 애초 스킵 대상 아님(§6.3 "rescue=env-id만").
                _addr_full = (os.environ.get("T2_PROV_ADDR_FULL") == "1"
                              and prov_mode == "rescue" and _is_addr_arg(k))
                if _addr_full:
                    self._t2_prov_addr_full = getattr(self, "_t2_prov_addr_full", 0) + 1
                _tmpl = REGEN_FEEDBACK if (prov_mode != "rescue" or _addr_full) \
                    else REGEN_FEEDBACK_NEUTRAL
                # ★C8(2026-08-05·028/050/051/053 실측): 날조된 값이 **도구 이름**이면 위 문구는
                #   이행 불가다 — "getter를 불러 그 값을 읽어라"라고 말하지만 **도구 이름을 돌려주는
                #   getter는 없다**. 이름의 진실은 레지스트리에 있고 우리는 이미 레지스트리를 읽는다.
                #   그래서 이름 인자일 때는 A2의 이름-검사 문구(레지스트리 대조·검색 경로)로 보낸다.
                #   매핑 표는 만들지 않는다 — 표는 낡고 레지스트리는 낡지 않는다.
                _fb8 = None
                if k in ("agent_tool_name", "discoverable_tool_name", "user_tool_name"):
                    _dnc = (a2 or {}).get("discoverable_name_check") or {}
                    try:
                        _reg8 = _agent_discoverable(
                            getattr(getattr(self, "_t2_orch", None), "environment", None))
                    except Exception:
                        _reg8 = None
                    # ★잘못-접미사 분기 (2026-08-12·j런 071t1: close_..._4822 시도에
                    #   `not_discoverable` 이 "suffixed version ... there is none" 을 말했는데
                    #   레지스트리에 같은 base 의 close_..._7392 가 **실재** — 그 거짓 금지문이
                    #   유일 복구 경로(KB 검색)를 차단해 무단 개설 오염을 영구화했다([[64]]).
                    #   base 대조 = `_[0-9]+$` strip 문자열 판정뿐(닫힌 술어·선례 7538행).
                    _fb8 = None
                    if _reg8 and s not in _reg8:
                        _base8 = re.sub(r"_[0-9]+$", "", str(s))
                        _same8 = any(re.sub(r"_[0-9]+$", "", str(r)) == _base8 for r in _reg8)
                        _fb8 = ((_dnc.get("feedback_wrong_suffix") or
                                 _dnc.get("feedback_not_discoverable"))
                                if _same8 else _dnc.get("feedback_not_discoverable"))
                        # ★레지스트리 목록 동봉 (2026-08-13 t7273 073t1 [61] 실측: base 자체가
                        #   미등록인 날조 이름에 "there is none" 만 나가자 모델이 "수동으로
                        #   조정하겠다" 날조로 접힘 — [[64]] 거부는 해법을 담아야 한다. 목록은
                        #   레지스트리 기계 나열뿐·선택은 모델. 키 없으면 침묵=종전 거동.)
                        # ★FIX-7 소유권 정정 (2026-08-13·격리 `x298_ownership_deny_probe.py`
                        #   3셀 n=8: A_LIVE 1/8 · **B_OWN 6/8** · D_BARE 0/8·사전등록 문턱).
                        #   실물(t7277 075 엔진로그 118행): 모델의 착수 시도는 **손님에게 도구를
                        #   넘기는 채널**(give_discoverable_user_tool·val='open_account')이었는데
                        #   우리 문구는 *"unlock_ 은 적용 안 된다"* + 45개 목록 = 채널이 어긋났고
                        #   **그 이름이 사실 에이전트 자신의 도구**라는 말이 없었다 → 수동 안내로
                        #   접힘. 술어는 닫혀 있다([[22]]): ⓐ이 인자가 손님-측 채널 키인가(A2
                        #   `user_tool_channel_args` 선언) ⓑ이름 토큰이 **에이전트** 레지스트리와
                        #   겹치는가(`_tok_overlap`·기계·판단 0). 겹치면 그 사실만 말한다 — 무엇을
                        #   호출할지·인자는 여전히 모델이 정한다([[62]] ③④·[[64]]).
                        # ★A13/OL-05 (t7336 §6.1·2026-08-22): 구판은 **손님-측 레지스트리를 한
                        #   번도 조회하지 않은 채** *"there is no customer-side tool by that name
                        #   on file"* 을 단언했다 — `_tok_overlap` 이 토큰 하나(`deposit`)만 겹쳐도
                        #   최대-겹침 항목을 돌려주므로, 손님 도구 `deposit_check_3847` 이 **실재**
                        #   하는데도 그 거짓이 나갔다(055#0 지연·055#1 치명·give 재시도 0). [[25]]
                        #   *"우리 도구는 100% 정답 의무"* 정면 위반이다.
                        #   ⇒ ⑴문면에서 **부정 존재 단언 삭제**(A2 두 층) ⑵여기서 손님-측
                        #     레지스트리를 **선조회**하고, 후보가 하나라도 겹치면 소유권 주장을
                        #     **접는다**(fail-open — 확인 못 한 사실은 말하지 않는다) ⑶대신 손님-측
                        #     **레지스트리 목록을 병기**해 모델이 고르게 한다(선점 금지·[[62]]③④).
                        #   `_user_discoverable`/`_user_all_tools` 는 **같은 파일에 실재**하고
                        #   (`:4287`/`:4272`) 주석이 그 구분을 이미 적어 뒀다 — 사본 0([[67]]).
                        #   선례: `:9031` 이 같은 조회로 `T2_UNKNOWN_NAME_BL` 의 거짓을 이미 막는다.
                        # ⚠[[70]] 무엇을 파는가: x298 이 잰 B_OWN 6/8 중 **손님-측과 토큰이 겹치는
                        #   자리**의 소유권 발화를 판다(그 자리에선 목록 병기로 대체). 계기 =
                        #   `[T2_OWNERSHIP_FIX] fired` ↔ `suppressed(user-side)` 짝 · give 성사율 ·
                        #   오-give 증가(마스터 §6.1 A13 행).
                        try:
                            _uenv8 = getattr(getattr(self, "_t2_orch", None), "environment", None)
                            _ureg8 = ((_user_discoverable(_uenv8) or set())
                                      | (_user_all_tools(_uenv8) or set()))
                        except Exception:
                            _ureg8 = set()          # fail-open: 조회 실패 = 손님-측 판정 없음
                        _uchan8 = k in set(_dnc.get("user_tool_channel_args") or ())
                        _uown8 = _tok_overlap(s, _ureg8) if (_uchan8 and _ureg8) else []
                        _own8 = (_tok_overlap(s, _reg8 or ())
                                 if (_uchan8 and _dnc.get("feedback_user_tool_is_agents")) else None)
                        if _uown8:
                            # 손님-측에도 후보가 있다 = "손님 것이 아니다"·"네 도구다" 둘 다
                            # 우리가 확인하지 못한 주장이다. 두 문면 모두 접고 목록만 준다.
                            print("[T2_OWNERSHIP_FIX] suppressed(user-side): give-name=%s "
                                  "customer-side candidate(s) %s" % (s, _uown8),
                                  file=_sys.stderr, flush=True)
                            _own8 = None
                            _ulist8 = _dnc.get("feedback_user_registry_listing")
                            if _ulist8:
                                _fb8 = str(_ulist8).replace(
                                    "{names}", ", ".join(sorted(_ureg8)))
                        if _own8:
                            _fb8 = str(_dnc["feedback_user_tool_is_agents"]).replace(
                                "{matches}", ", ".join(_own8))
                            # 선점 금지([[70]] A13): 단일 `{matches}` 로 몰지 않도록 레지스트리 병기.
                            _lst8o = _dnc.get("feedback_registry_listing")
                            if _lst8o and _reg8:
                                _fb8 = str(_fb8) + str(_lst8o).replace(
                                    "{names}", ", ".join(sorted(_reg8)))
                            print("[T2_OWNERSHIP_FIX] fired give-name=%s → agent tool(s) %s"
                                  % (s, _own8), file=_sys.stderr, flush=True)
                        elif _fb8 and not _same8 and not _uown8:
                            _lst8 = _dnc.get("feedback_registry_listing")
                            if _lst8 and _reg8:
                                _fb8 = str(_fb8) + str(_lst8).replace(
                                    "{names}", ", ".join(sorted(_reg8)))
                    else:
                        _fb8 = _dnc.get("feedback")
                    if _fb8:
                        print("[T2_PROV] name-arg → registry message tool=%s val=%s"
                              % (getattr(ptc, "name", "?"), s), file=_sys.stderr, flush=True)
                main_prov = (ptc, str(_fb8).replace("{name}", str(s)) if _fb8
                             else (_directive if _directive is not None
                                   else _tmpl.format(k=k, s=s)))
            if do_gate:
                gate_rounds += 1
                self._t2_gate_rounds = getattr(self, "_t2_gate_rounds", 0) + 1
                _budget_tick(self)  # ★게이트 라운드만 과금 (prov=무과금=C53 semantics)
            if ep_fb is not None:
                eplan_rounds += 1
                # ★C174(사용자 지시 2026-07-25): deny 예산 **전면 독립** — 공용 예산 폐지.
                #   같은 종류의 deny는 자기 cap 안에서 여러 번 발화 가능하되, 무관한 deny끼리는
                #   서로의 예산을 건드리지 않는다. pre-close 발화는 발화 지점서 자기 카운터
                #   (_t2_preclose_deny)만 소모하므로 여기선 discovery 카운터를 증가시키지 않음.
                if getattr(self, "_t2_ep_fb_preclose", False):
                    self._t2_ep_fb_preclose = False
                else:
                    self._t2_eplan_deny = getattr(self, "_t2_eplan_deny", 0) + 1
                    if self._t2_eplan_deny == _ep_cap:  # 관측 마커: 이후 discovery deny 중단
                        print("[T2_EPLAN] deny cap %d reached — no further discovery denies this sim"
                              % _ep_cap, file=_sys.stderr, flush=True)
            if cons_fb is not None:
                cons_rounds += 1
                self._t2_cons_deny = getattr(self, "_t2_cons_deny", 0) + 1
            if dd_fb is not None:
                self._t2_dd_deny = getattr(self, "_t2_dd_deny", 0) + 1
                print("[T2_DISCOVERY_DISPATCH] deny direct call=%s → force dispatcher"
                      % getattr(dd_fb[0], "name", "?"), file=_sys.stderr, flush=True)
            if ra_fb is not None:
                ra_rounds += 1
                self._t2_readall_deny = getattr(self, "_t2_readall_deny", 0) + 1
            if te_fb is not None:
                te_rounds += 1
                self._t2_toolerr_deny = getattr(self, "_t2_toolerr_deny", 0) + 1
            if proc_fb is not None:
                self._t2_proc_deny = getattr(self, "_t2_proc_deny", 0) + 1
            if rdd_fb is not None:
                self._t2_rdd_attached = getattr(self, "_t2_rdd_attached", 0) + 1
            if abs_fb is not None:
                self._t2_proc_absent = getattr(self, "_t2_proc_absent", 0) + 1
                # ★말한 **DAG 상태**를 기억한다 — 전달된 것만([[55]] 로그 마크 != 전달).
                _last6 = getattr(self, "_t2_proc_absent_last", None)
                if _last6:
                    _seen6 = getattr(self, "_t2_proc_state_seen", None)
                    if _seen6 is None:
                        _seen6 = self._t2_proc_state_seen = set()
                    _seen6.add(_last6)
                    self._t2_proc_absent_last = None
                    print("[T2_PROC_ABSENT] 상태 %s (완료 %d) — 말한 상태 %d종"
                          % (_last6[0], len(_last6[1]), len(_seen6)),
                          file=_sys.stderr, flush=True)
            if tr_fb is not None:
                self._t2_transcribe_deny = getattr(self, "_t2_transcribe_deny", 0) + 1
            if wev_fb is not None:
                wev_rounds += 1
                self._t2_wev_deny = getattr(self, "_t2_wev_deny", 0) + 1
                if self._t2_wev_deny == _wev_cap:  # 관측 마커(sim당 1회): 이후 WEV deny 중단
                    print("[T2_WRITE_EVIDENCE] deny cap %d reached — no further WEV denies this sim"
                          % _wev_cap, file=_sys.stderr, flush=True)
            if rw_fb is not None:
                self._t2_resolve_deny = getattr(self, "_t2_resolve_deny", 0) + 1
                # ★정체 판정의 기준점 = **발화 시점의 실행 집합**(2026-08-07).
                #   다음 진입에서 이 집합보다 커졌으면 진행이 있었던 것이고, 그러면 위 카운터를
                #   되돌린다(`_resolve_cap_ok`). 검사 시점에 갱신하면 발화 사이의 진행을 못 본다.
                try:
                    self._t2_resolve_done = _executed_tool_names(state.messages, a2)
                except Exception:
                    pass
                # ★같은 기준점의 두 번째 축 = **회수된 unlock 후보 집합**(2026-08-14·x305).
                #   다음 진입에서 새 이름이 늘었으면 그것도 진행이다(`_resolve_cap_ok`).
                try:
                    _u2 = ((a2 or {}).get("eplan") or {}).get("unlock_tool")
                    if _u2:
                        import t2_resolve as _rz_cap2
                        _rg2 = _rz_cap2.agent_discoverable_names(self)
                        if _rg2:
                            self._t2_resolve_names = set(
                                _rz_cap2._retrieved_unlockables(state.messages, _rg2, _u2))
                except Exception:
                    pass
            if tl_fb is not None:
                tl_rounds += 1
                self._t2_toollist_deny = getattr(self, "_t2_toollist_deny", 0) + 1
                if self._t2_toollist_deny == int(os.environ.get("T2_TOOLLIST_CAP", "6")):
                    print("[T2_TOOLLIST] deny cap reached — nonlisted calls pass through hereafter",
                          file=_sys.stderr, flush=True)
            if sig_fb is not None:
                self._t2_signature_deny = getattr(self, "_t2_signature_deny", 0) + 1
            if un_fb is not None:
                self._t2_unlockname_deny = getattr(self, "_t2_unlockname_deny", 0) + 1
            if dr_fb is not None:
                self._t2_dispatchrole_deny = getattr(self, "_t2_dispatchrole_deny", 0) + 1
            if hv_fb is not None:
                if getattr(self, "_t2_valacq_fired", False):   # ★C119: va가 hv_fb 채널 재사용 → 별도 카운터
                    self._t2_valacq_deny = getattr(self, "_t2_valacq_deny", 0) + 1
                    self._t2_valacq_fired = False
                else:
                    self._t2_havevalue_deny = getattr(self, "_t2_havevalue_deny", 0) + 1
                # ★force_required (T2_UNLOCK_NAME 선례·[[10]]): 실패모드=프로즈 재요청(say-don't-do).
                #   넛지 후 재생성을 tool_choice=required로 봉쇄 → 산문 대신 반드시 도구 호출(어느 도구=모델).
                #   기본 ON. ★2026-07-23 근본원인 규명: required는 라이브 경로서 정상 동작(라이브 llm_args가
                #   max_tokens 미설정=vLLM 기본 대형 → 강제 tool-call 완성). 앞선 프로브 400은 스크립트가
                #   max_tokens=450/20을 하드코딩해 강제 JSON이 절단된 아티팩트(vLLM #19051/#36794)였음 —
                #   _gen의 max_tokens 하한이 교정. 병리적 runaway(039 퇴행루프)만 _gen 폴백으로 강등.
                if os.environ.get("T2_HAVE_VALUE_FORCE", "1") == "1":
                    force_required = True
            # ★출구 ② 배선 (2026-08-07·"단일 출구" 전제 교정). `admit()`은 `_ap_regen`(텍스트 발화)
            #   한 곳에만 걸려 있었는데, deny·치환 문구는 **이 `fb` 배치**라는 두 번째 출구로 나간다.
            #   그래서 창을 켠 런에서도 `[ORDER] 'submit_referral' …`이 **14회·전부 바이트 동일**로
            #   나갔다(sim 2개·사이드카 `channel=unified_regen` 실측). `T2_ARBITRATE`의 국소 지문은
            #   손님 발화 수·실행 원장 크기를 포함해 턴이 바뀌면 다시 말한다 — 문구가 안 바뀐 재발화를
            #   접는 것은 `admit()`뿐이다([[57]] 인자 변화 기준).
            #   ⚠**deny는 fail-closed**(버스 불변): 접힐 때도 오류는 그대로 나가고 **본문만** 이미
            #   아래에 있는 일반 문구로 내려간다. 문구를 새로 만들지 않는다. 지침(UserMessage)은
            #   반대로 무부착 — 막는 말이 아니므로 접으면 그냥 안 붙인다.
            _FB_GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."
            _fbtag = {}
            # ★단계 1 게이트⒝ 계기: **현행 체인이 실제로 말한 표적**을 모은다(설계서 §7e).
            #   `audit()`에 넘겨 `route()` 판정과 갈리는 자리를 세기 위한 것이고, 거동은 안 바꾼다.
            # ★출처 순서를 **한 벌로** 둔다 (2026-08-11·설계 §5). 아래 배타 체인의 순서와
            #   같아야 하고, 두 벌이 되면 갈린다([[03b]]). rank = 이 튜플의 색인 + 2
            #   (1 = do_gate·main_prov 자리).
            _SRC8 = (("eplan", ep_fb), ("discovery", dd_fb), ("cons", cons_fb),
                     ("resolve_action", ra_fb), ("toolerr", te_fb), ("wev", wev_fb),
                     ("transcribe", tr_fb), ("proc", proc_fb), ("resolve_write", rw_fb),
                     ("toollist", tl_fb), ("signature", sig_fb), ("unlockname", un_fb),
                     ("prekb", pc_fb), ("dispatch_role", dr_fb), ("prov", pr_fb),
                     ("decide_write", dw_fb), ("write_enum", en_fb),
                     ("dup_write", dup_fb))
            _chose8 = []
            for _n8, _v8 in _SRC8:
                if _v8 is not None and _v8[0] is not None:
                    _fbtag.setdefault(id(_v8[0]), _n8)
            # ★§S-2 3층 (2026-09-01): **빈 am 은 재생성 프롬프트에 싣지 않는다.**
            #   실측(095·모드 B): 빈 어시스턴트 메시지가 이 버퍼에 실려 나가
            #   `llm_utils.py:234 assert has_content_or_tool_calls` 로 죽고 태스크가 재시작됐다.
            fb = [] if _t2_msg_empty(am) else [am]
            for c in (am.tool_calls or []):
                if do_gate and id(c) in denied_by_objid:
                    gid, why = denied_by_objid[id(c)]
                    content = f"Error: [POLICY GATE {gid}] {why}"
                elif main_prov is not None and c is main_prov[0]:
                    content = main_prov[1]
                elif ep_fb is not None and c is ep_fb[0]:
                    content = "Error: " + ep_fb[1]
                elif dd_fb is not None and c is dd_fb[0]:
                    content = "Error: [DISCOVERY] " + dd_fb[1]
                elif cons_fb is not None and c is cons_fb[0]:
                    content = "Error: " + cons_fb[1]
                elif ra_fb is not None and c is ra_fb[0]:
                    content = "Error: " + ra_fb[1]
                elif te_fb is not None and c is te_fb[0]:
                    content = "Error: " + te_fb[1]
                elif wev_fb is not None and c is wev_fb[0]:
                    # A2 feedback이 "Error:"로 시작하면 그대로(이중 접두 방지)
                    content = wev_fb[1] if str(wev_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + wev_fb[1]
                elif tr_fb is not None and c is tr_fb[0]:
                    content = tr_fb[1] if str(tr_fb[1]).lstrip().startswith("Error:")                         else "Error: " + tr_fb[1]
                elif proc_fb is not None and c is proc_fb[0]:
                    content = proc_fb[1] if str(proc_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + proc_fb[1]
                elif rw_fb is not None and c is rw_fb[0]:
                    content = "Error: " + rw_fb[1]
                elif tl_fb is not None and c is tl_fb[0]:
                    content = "Error: " + tl_fb[1]
                elif sig_fb is not None and c is sig_fb[0]:
                    content = sig_fb[1] if str(sig_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + sig_fb[1]
                elif un_fb is not None and c is un_fb[0]:
                    content = un_fb[1] if str(un_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + un_fb[1]
                elif pc_fb is not None and c is pc_fb[0]:
                    content = pc_fb[1] if str(pc_fb[1]).lstrip().startswith("Error:") else "Error: " + pc_fb[1]
                elif dr_fb is not None and c is dr_fb[0]:
                    content = dr_fb[1] if str(dr_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + dr_fb[1]
                elif pr_fb is not None and c is pr_fb[0]:
                    content = pr_fb[1] if str(pr_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + pr_fb[1]
                elif dw_fb is not None and c is dw_fb[0]:
                    content = dw_fb[1] if str(dw_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + dw_fb[1]
                elif en_fb is not None and c is en_fb[0]:
                    content = en_fb[1] if str(en_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + en_fb[1]
                elif dup_fb is not None and c is dup_fb[0]:
                    content = dup_fb[1] if str(dup_fb[1]).lstrip().startswith("Error:") \
                        else "Error: " + dup_fb[1]
                else:
                    content = _FB_GENERIC
                if content != _FB_GENERIC:
                    try:
                        import t2_stack as _stk8
                        _ok8, _why8 = _stk8.admit(self, _fbtag.get(id(c), "fb"), content)
                        if not _ok8:
                            # ★R9 — **접힘이 본문을 지우지 않는다** (2026-08-11·C414·`T2_KEEP_DENY_BODY`
                            #   ·기본 OFF·근거 x246·n=8). 구판은 접힐 때 본문을 `_FB_GENERIC` 으로
                            #   갈아 끼웠는데, 그 문구는 **무엇을 고칠지를 말하지 않는다**. 에이전트는
                            #   해소할 대상을 모른 채 같은 호출을 반복하고, 같은 deny 가 또 접히고…
                            #   ⇒ **접힘이 자기 원인을 재생산한다**. 실측(3 런·30 sim): 그 문구가 한
                            #   sim 에 3회 이상 나온 6건은 **6/6 전부 실패**(uq 010 t2 는 11회·마지막
                            #   22턴을 되뇜으로 태웠다). 격리(x246·같은 문맥·다음 한 수): 일반 문구
                            #   3회 = 정체 **3/8** · 1회 = 2/8 · **원본 본문 = 0/8** · 아무것도 안 냄 =
                            #   **0/8**. ⇒ 문제는 반복 횟수가 아니라 **문구 자체**다.
                            #   ⚠deny 는 fail-closed 이므로 *안 내보내기*(동률 8/8)가 아니라 **원본
                            #     본문을 남기는 쪽**을 택한다 — 도구는 여전히 막히고 이유만 살아난다.
                            #   ⚠[[57]] 과 충돌하지 않는다: 인자-변화 기준은 **조언 채널**의 것이고,
                            #     도구-결과 채널의 deny 는 접어도 *무엇을 고칠지*가 남아야 한다.
                            #   ⚠새 결정론 0 — 우리가 **덜 지울 뿐**이다.
                            _keep9 = os.environ.get("T2_KEEP_DENY_BODY") == "1"
                            print("[T2_STACK] window folded fb tag=%s (%s) — deny stays, body %s"
                                  % (_fbtag.get(id(c), "fb"), _why8,
                                     "kept (R9)" if _keep9 else "generic"),
                                  file=_sys.stderr, flush=True)
                            if not _keep9:
                                content = _FB_GENERIC
                    except Exception:
                        pass
                    if content != _FB_GENERIC:      # 접히지 않고 실제로 나간 것만 센다
                        _chose8.append((_fbtag.get(id(c), "fb"), _eff_tool_name(c)))
                # ★[[64]] — 마지막 남은 **이름 없는 거부 본문**을 이름 있는 것으로 바꾼다
                #   (2026-08-18·C536ⓑ). C416 이 만든 `_sibling_wait`(막힌 호출의 **이름** + 다음 한
                #   수)는 네 자리에만 붙었고 이 fb 조립의 `else` 는 빠져 있었다. t7313 `task_040` 은
                #   그 문구를 **5회** 받고 turn 104 까지 같은 호출을 되뇌었다. 근거는 이미 격리로
                #   잼 — 일반 문구 3회 = 정체 **3/8** ↔ 원본 본문 **0/8**(x246·C414·n=8).
                #   ⚠**계기·회계는 한 글자도 안 건드린다**: 위 `_FB_GENERIC` 센티널 비교(접힘·
                #     `_chose8` 계수)는 그대로 두고, **내보내는 문자열만** 여기서 바꾼다.
                #   ⚠OFF(`T2_KEEP_DENY_BODY` != 1)면 **바이트 동일**로 종전 문구가 나간다.
                #   ⚠fail-closed 불변 — 이 호출은 여전히 실행되지 않는다. 새 결정론 0·도메인 어휘 0.
                _body8 = content
                if content == _FB_GENERIC and os.environ.get("T2_KEEP_DENY_BODY") == "1":
                    _flag8 = next((x for x in (am.tool_calls or [])
                                   if x is not c
                                   and ((do_gate and id(x) in denied_by_objid)
                                        or (main_prov is not None and x is main_prov[0])
                                        or any(v is not None and v[0] is x
                                               for _n8b, v in _SRC8))), None)
                    if _flag8 is not None:
                        _body8 = _sibling_wait("BLOCKED", _flag8, "what to fix")
                # ★[[64]] 뒷칸 — **무엇을 하면 풀리나**를 A2 선언 인용으로 붙인다
                #   (2026-08-31·`T2_DENY_HOWTO`·기본 OFF·격리 전까지 배선만).
                #   x692 `task_094` 실물: `[T2_TOOL_OBS] err=True -> Error: resolve the flagged
                #   call(s) first; do not call this tool yet.` — 이 문면은 *다음 한 수*가 없다.
                #   그런데 그 답(선행 read·인자 계약)은 A2 에 이미 있다(`_decl_howto` 독스트링).
                #   ⚠fail-closed 불변·새 결정론 0·도메인 어휘 0. 못 대면 빈 문자열이라 OFF 와
                #     **바이트 동일**이다. 계기·회계(`_FB_GENERIC` 센티널·`_chose8`)는 위에서
                #     이미 끝났으므로 한 글자도 안 건드린다.
                if os.environ.get("T2_DENY_HOWTO") == "1":
                    try:
                        _hw8 = _decl_howto(_eff_tool_name(c), a2)
                        if _hw8:
                            _body8 = _body8 + _hw8
                            print("[T2_DENY_HOWTO] appended tool=%s chars=%d"
                                  % (_eff_tool_name(c), len(_hw8)), file=_sys.stderr, flush=True)
                    except Exception as _he8:
                        print("[T2_DENY_HOWTO] skip: %r" % (_he8,), file=_sys.stderr, flush=True)
                fb.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                      error=True, content=_body8))
                # ★배달 계측 (2026-08-11·설계 §5·§7-1·원장 C427) — **거동 불변**.
                #   문자열도 순서도 안 바꾼다. 세는 것은 하나: *이 호출을 두고 몇 전문가가
                #   말하려 했고, 누가 이겼고, 누가 밀렸나.*
                #   왜: 위 `elif` 는 같은 tool_call 에 대해 **하나만** 내보낸다. 그래서 오프라인
                #   32/32 인 문장이 라이브에서 3/6 만 닿았고(C419⒠), 원인의 절반이 이 배타성인
                #   것을 오늘에야 코드 추적으로 알았다 — **계수가 없어서 몰랐다.**
                #   `_chose8`(이긴 쪽)은 이미 있었지만 **밀린 쪽은 아무도 세지 않았다.**
                if os.environ.get("T2_ROUTE_TRACE", "1") == "1":
                    try:
                        _win = _fbtag.get(id(c))
                        # rank = 체인 위치. `_SRC8` 앞에 **두 분기**가 더 있다
                        #   (1 `do_gate` · 2 `main_prov`) ⇒ 색인 + 3.
                        #   ⚠초판은 +2 였고 `test_route_trace.py` 가 잡았다 — 계측이 거짓 rank 를
                        #   보고하면 C427 을 늦게 발견한 것과 같은 종류의 사고가 된다.
                        _cands = [(_i9 + 3, _n9) for _i9, (_n9, _v9) in enumerate(_SRC8)
                                  if _v9 is not None and _v9[0] is c]
                        if _cands:
                            # ★삼분 (설계 v1.5 §5.1). `outcome` 이 **억제·체인·미생성**을 가른다:
                            #     won   이 호출의 출구를 이겼다
                            #     lost  말하려 했는데 배타 체인에서 밀렸다 (`lost_to`)
                            #     suppressed  지문 억제로 아예 생성되지 않았다 (아래 별도 루프)
                            #   `arrived` 는 여기서 **안 정한다** — `_gen` 이 모델 입력을 보고
                            #   채운다(리뷰 N3). 여기서 채우면 `_emit` 호출을 배달로 위조한다.
                            _q9 = list(getattr(self, "_t2_route_pending", None) or [])
                            for _rk9, _n9 in _cands:
                                _q9.append(dict(
                                    agent=_n9, rank=_rk9,
                                    target=_eff_tool_name(c),
                                    outcome=("won" if _win == _n9 else "lost"),
                                    lost_to=(None if _win == _n9 else _win),
                                    folded=bool(content == _FB_GENERIC),
                                    _text=(content if _win == _n9 else
                                           str((dict(_SRC8).get(_n9) or (None, ""))[1] or ""))))
                            self._t2_route_pending = _q9
                            if len(_cands) > 1:
                                print("[T2_ROUTE] %s 경합 %d → %s 승 · 밀림 %s"
                                      % (_eff_tool_name(c), len(_cands), _win,
                                         ",".join(n for _r, n in _cands if n != _win)),
                                      file=_sys.stderr, flush=True)
                    except Exception as _e9:
                        print("[T2_ROUTE] 계측 실패(무시): %r" % (_e9,),
                              file=_sys.stderr, flush=True)
            # ★억제분을 같은 스트림에 넣는다 (설계 v1.5 §5.1 삼분의 셋째 칸).
            #   호출 루프 **밖**이다 — 억제된 레버에는 짝지을 `tool_call` 이 없다.
            #   이것이 없으면 억제와 미생성이 사이드카에서 똑같이 '레코드 없음'으로 보인다.
            try:
                _sl9 = getattr(self, "_t2_silenced", None)
                if _sl9:
                    self._t2_silenced = []
                    _q8 = list(getattr(self, "_t2_route_pending", None) or [])
                    for _s9 in _sl9:
                        _q8.append(dict(agent=_s9.get("agent"), rank=None,
                                        target=_s9.get("target"),
                                        outcome="suppressed", lost_to=None, folded=False,
                                        _text=str(_s9.get("text") or "")))
                    self._t2_route_pending = _q8
            except Exception as _e8:
                print("[T2_ROUTE] 억제 계측 실패(무시): %r" % (_e8,),
                      file=_sys.stderr, flush=True)
            # ★순서 검사 (거동 변경 0). 층-1 레버들이 `beat(orch=…)`로 등록해 둔 후보를 비우고,
            #   `route()`가 골랐을 층·표적만 남긴다. `speak()`를 실제 출구로 쓰는 것은 순서를
            #   뒤집는 일이라 **먼저 두 판정이 어디서 갈리는지**가 있어야 한다 — 갈리는 자리가
            #   없으면 뒤집을 이유도 없고, 있으면 그 자리가 곧 실험 표적이다.
            #   ⚠비우지 않으면 등록분이 sim 내내 쌓인다(등록점만 만들고 드레인을 안 둔 것이
            #   이 코드베이스의 死배선 패턴이다).
            try:
                import t2_stack as _stkA
                _aud = _stkA.audit(self, chose_targets=[t for _g, t in _chose8])
                if _aud:
                    # ★한 줄에 **양쪽 판정과 갈림 여부**를 함께 찍는다. 구판은 `route()` 쪽만
                    #   찍어서, 그 줄만 보면 *"현행과 같은지 다른지"* 를 영영 알 수 없었다 —
                    #   단계 2b의 착수 조건이 정확히 그 비교인데 계기가 그것을 안 담았다.
                    print("[T2_STACK] audit route=%s chose=%s differs=%s suppressed=%s"
                          % (_aud["pick"], _chose8, _aud["target_differs"], _aud["suppressed"]),
                          file=_sys.stderr, flush=True)
            except Exception:
                pass
            # ★action-required 리마인더 채널 (순수-조언 회피=tool_call 0 → 앵커할 ToolMessage 없음).
            #   rw_fb[0] is None = 순수-조언 action-required(2085행). UserMessage 리마인더로 재생성.
            #   작업버퍼(work)만·state.messages 비커밋 = 채널 절대규칙(1849·replay-clean).
            # ★L4 `T2_CLAIM_BLOCK`(ACTION_HANDOFF_LEVERS rev2·기본 OFF): 실행이 하나도 없는데
            #   "처리됐습니다"라고 보고하는 발화를 되돌린다. 술어는 전수 재계량으로 가장 좁은 것을
            #   골랐다(주장 ∧ 이 메시지에 호출 없음 ∧ 지금까지 실효 write 0 = 28건/23 sim·과차단 후보 1).
            #   **막는 것은 주장이지 행동이 아니다** — 무엇을 할지는 모델이 정한다([[06]]).
            # ★L5-a `T2_TRANSFER_PREREQ`(기본 OFF): 검색 한 번 없이 사람에게 넘기는 턴만 되돌린다
            #   (전수 9건/9 sim). 이관을 막지 않고 전제만 요구한다 — 생성-측이라 replay 무관.
            try:
                import t2_transfer_prereq as _tp
                if _tp.missing_prereq(state.messages, am):
                    try:
                        fb.append(UserMessage(role="user", content=_tp.FEEDBACK))
                    except TypeError:
                        fb.append(UserMessage(content=_tp.FEEDBACK))
                    from t2_lever_beat import beat as _tbeat
                    _tbeat("T2_TRANSFER_PREREQ")
            except Exception:
                pass
            try:
                import t2_claim_block as _cb
                if _cb.blocks(state.messages, getattr(am, "content", None),
                              bool(am.tool_calls or []), _eff_tool_name):
                    try:
                        fb.append(UserMessage(role="user", content=_cb.FEEDBACK))
                    except TypeError:
                        fb.append(UserMessage(content=_cb.FEEDBACK))
                    from t2_lever_beat import beat as _cbeat
                    _cbeat("T2_CLAIM_BLOCK")
            except Exception:
                pass
            if rw_fb is not None and rw_fb[0] is None and not (am.tool_calls or []):
                # 지침 채널이라 접히면 **무부착**이다(위 deny와 반대 — 막는 말이 아니다).
                _ok9 = True
                try:
                    import t2_stack as _stk9
                    _ok9, _why9 = _stk9.admit(self, "resolve_write", rw_fb[1])
                    # 통과도 찍는다 — 접힘만 찍으면 "안 접혔다"와 "창을 안 거쳤다"가 같아 보인다.
                    print("[T2_STACK] guidance tag=resolve_write %s (%s)"
                          % ("passed" if _ok9 else "dropped", _why9),
                          file=_sys.stderr, flush=True)
                except Exception:
                    _ok9 = True
                if _ok9:
                    try:
                        fb.append(UserMessage(role="user", content=rw_fb[1]))
                    except TypeError:
                        fb.append(UserMessage(content=rw_fb[1]))
            # ★T2_HAVE_VALUE 리마인더 (None-anchor·산문 회피 또는 producer 재호출 커버·비커밋=replay-clean)
            # ★D1′ 부재 표면화 (비커밋·hv_fb와 같은 채널 규약): 차단이 아니라 상태 진술이라
            #   특정 호출에 붙지 않는다 — 호출이 없는 것이 바로 이 레버의 조건이다.
            if fs_fb is not None:
                try:
                    fb.append(UserMessage(role="user", content=fs_fb))
                except TypeError:
                    fb.append(UserMessage(content=fs_fb))
            if wd_fb is not None:
                try:
                    fb.append(UserMessage(role="user", content=wd_fb))
                except TypeError:
                    fb.append(UserMessage(content=wd_fb))
            if abs_fb is not None:
                try:
                    fb.append(UserMessage(role="user", content=abs_fb))
                except TypeError:
                    fb.append(UserMessage(content=abs_fb))
            # ★T2_REQUIRE_DOC_DELIVER 부착 — 같은 비커밋 채널(재생성 버퍼·replay 불변식). 부착 마크를
            #   여기서 찍는다([[55]] 로그 마크≠전달 — 위 `deliver` 줄은 조립이고 이 줄이 전달이다).
            if rdd_fb is not None:
                try:
                    fb.append(UserMessage(role="user", content=rdd_fb))
                except TypeError:
                    fb.append(UserMessage(content=rdd_fb))
                print("[T2_REQUIRE_DOC_DELIVER] 이 턴 재생성 버퍼에 부착 (%d자)" % len(rdd_fb),
                      file=_sys.stderr, flush=True)
            if hv_fb is not None:
                try:
                    fb.append(UserMessage(role="user", content=hv_fb))
                except TypeError:
                    fb.append(UserMessage(content=hv_fb))
            # ★사이드카(MT_PROBE_DESIGN §1-d·2026-07-30): 이 `fb`는 **비커밋**이라 궤적에 남지
            #   않는다(바로 위 "채널 절대규칙"). 그래서 메시지-수준 포렌식이 불가능했다([[08]]).
            #   궤적은 그대로 두고 **별도 파일에만** 발화 사실을 남긴다. T2_FB_SIDECAR 미설정=no-op.
            if fb:
                try:
                    import t2_fbsidecar as _fbsc
                    _fbsc.record_many(fb, state.messages, channel="unified_regen")
                except Exception:
                    pass
            work = work + fb
            # ★CP2 를 **이 턴의 재생성 버퍼**에 붙인다 (2026-08-12·C443 교정).
            #   초판은 비커밋 뷰 큐(`_t2_view_fb`)에 넣었는데 그 큐는 **다음 턴** `unified()`
            #   시작에서 소비된다 — 결정점도 write 도 **이 턴**이라 한 턴 늦었다. 계측이
            #   그것을 그대로 찍었다: `agent=decision_carry · arrived=False`(070·071 둘 다),
            #   그리고 행동도 같은 말을 했다 — 서브가 `Sky Blue` 를 냈는데 제출은 `Hunter
            #   Green`(값 없이 후보 명단만 받으면 메뉴가 된다·C440 동형).
            #   ⚠여전히 **비커밋**이다: `work` 는 생성-시점 버퍼이고 `state.messages` 가 아니다
            #     (C298 replay 불변식 유지).
            #   ⚠배타 체인 밖은 그대로다 — `fb` 뒤에 **따로** 붙지, 어느 tool_call 도 차지하지
            #     않는다(억제·경쟁 무관).
            _cp2 = getattr(self, "_t2_cp2_pending", None)
            # ★컨텍스트 가드 — **소비 지점 하나**에 둔다(2026-08-16·t7304·심사 권고: 대입 자리
            #   5곳을 한 가드로 덮으려면 여기여야 한다). 대용량(≥5k자)만 검사·보수 추정(자수/3)·
            #   초과면 **건너뛰고 기록**(축약·선별 0 — 엔진이 줄이면 [[62]]③). 소형 배달물은
            #   종전 그대로(ctl 바이트 불변). skip 수는 ⓔ 부작용 표에 계상된다.
            if _cp2 and len(_cp2) >= _CP2_GUARD_MIN:
                # 산식·보정 근거는 `_ctx_fits` 독스트링(2026-08-22 함수로 올림·거동 동일).
                _fit2, _hist = _ctx_fits(work, _cp2)
                if not _fit2:
                    print("[T2_DOC_DELIVERY] skipped: est %d+%d chars > cap"
                          % (_hist, len(_cp2)), file=_sys.stderr, flush=True)
                    self._t2_cp2_pending = None
                    # ★R4: 창 초과로 여기서 죽는다 — 이 사실이 사이드카에 안 남으면 그 배달물은
                    #   *대입은 됐는데 아무 데도 없는* 유령이 되고 검산식이 그 자리에서 깨진다.
                    _cp2_close(self, "ctx_skip")
                    _cp2 = None
            if _cp2:
                self._t2_cp2_pending = None
                try:
                    work = work + [UserMessage(role="user", content=_cp2)]
                except TypeError:
                    work = work + [UserMessage(content=_cp2)]
                print("[T2_DECISION_CARRY] 이 턴 재생성 버퍼에 부착 (%d자)" % len(_cp2),
                      file=_sys.stderr, flush=True)
            # ★P1: A2 `require_tool_before`가 선언한 선행 read가 미실행이면 이 재생성 1회를
            #   그 read로 고정한다. deny 스텁이 아니라 **생성-측 제약**이라 replay가 비교할
            #   tool 출력을 만들지 않는다 — require_tool_before를 권고로 강등시킨 C210 사유가
            #   여기엔 닿지 않는다. 조건 미충족·해소 실패면 None = 종전 거동.
            _pin_r = None
            # ★절차 고정이 예약돼 있으면 그것을 먼저 쓴다(1회 소모). 절차가 이름으로 지목한 단계는
            #   A2 선언에서 온 값이라 P1의 "이름만 지정하면 날조" 위험이 없다(단일값 enum 동일 경로).
            _pp_pin = getattr(self, "_t2_proc_pin", None)
            if _pp_pin:
                # ★D2 sticky (2026-08-05·설계 §3·**기본 OFF**): 구판은 핀을 **1회 소모**해서, 같은 턴의
                #   후속 재생성이 곧바로 덮었다. 재무장하면 표적이 실제로 나올 때까지 유지된다.
                #   ⚠근거 상태: 원래 근거("deny해도 이행 안 한다")는 F19로 **소멸**했다 — 그 deny는
                #   모델에게 간 적이 없었다(§1.5). 그래서 `T2_PROC_PIN_REARM` 기본 0 = **거동 불변**이고,
                #   켜는 것은 x87이 "전달된 deny로도 부족하다"를 보인 뒤에만이다([[10]]: 핀은 엔진이
                #   다음 행동을 고르는 데 가장 가까운 레버다).
                _rearm = int(os.environ.get("T2_PROC_PIN_REARM", "0"))
                _used = getattr(self, "_t2_proc_pin_used", 0)
                _pv = _pp_pin[2]
                _pv = list(_pv) if isinstance(_pv, (list, tuple, set)) else [_pv]
                _exec_now = _executed_tool_names(state.messages)
                if any(v in _exec_now for v in _pv) or _used >= _rearm:
                    self._t2_proc_pin = None          # 표적이 나왔거나 예산 소진 = 해제
                    self._t2_proc_pin_used = 0
                else:
                    self._t2_proc_pin_used = _used + 1
                _pin_r = _pp_pin
            else:
                try:
                    import t2_pin_read as _PRm
                    _pin_r = _PRm.pin_for(self, am, a2, state.messages)
                except Exception:
                    _pin_r = None
                # ★읽기 루틴 (2026-08-18·사용자 지시): 절차가 남긴 조회가 **전부 read** 면 그
                #   집합으로 채널을 좁히고, 하나씩 빠져 **비면 저절로 풀린다**. 트리거(침묵 3턴)를
                #   기다리지 않는다 — t7317 에서 그 트리거는 손님이 대화를 닫을 때까지 오지 않았다.
                #   기존 핀이 이미 있으면 건드리지 않는다(명시 지목 우선).
                if _pin_r is None and os.environ.get("T2_PIN_READ_STEPS") == "1":
                    try:
                        _pin_r = _read_routine_pin(self, a2, state.messages)
                        if _pin_r:
                            print("[T2_READ_ROUTINE] %s(%s in %s)"
                                  % (_pin_r[0], _pin_r[1], _pin_r[2]),
                                  file=_sys.stderr, flush=True)
                    except Exception as _rre:
                        print("[T2_READ_ROUTINE] 건너뜀(무발화): %r" % (_rre,),
                              file=_sys.stderr, flush=True)
            # ★액션 서브 (2026-08-10·`T2_ACTION_SUB`·기본 OFF). 이 재생성이 **손님에게
            #   도구를 넘기는 발화**를 짓는 자리이고(rw_fb = 그 ACTION 되먹임), x228 은 그
            #   발화를 격리에서 지으면 소유권이 0/6 → 6/6 이고 `external` 위반이 6/6 → 0/6
            #   임을 쟀다. 여기서는 **자리만 옮긴다** — 값도 선택도 모델이 낸다.
            #   ⚠도구 호출이 필요한 턴에는 타지 않는다(`_pin_r`·`force_required` 가 걸린 턴 제외).
            _am_sub = None
            if (os.environ.get("T2_ACTION_SUB") == "1" and rw_fb
                    and not force_required and not _pin_r
                    and getattr(self, "_t2_asub", None)):
                _am_sub = _gen_action_sub(self, state, self._t2_asub)
            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
                                 tool_choice="required" if force_required else None, pin=_pin_r)
            if _cp2:
                # ★R4: 종결은 **생성이 돌아온 뒤**에만 찍는다. 부착 인쇄(`부착 (N자)`)를 세면
                #   생성기가 예외로 죽은 회차까지 도달로 위조한다 — `proc_fb` 死배선이 deny 11회를
                #   인쇄로 만든 것과 같은 종류의 사고다([[55]] 로그 마크 ≠ 전달).
                # ⚠`via`: `_am_sub` 가 참인 회차엔 `_gen` 이 **안 불리고** `work` 는 비커밋 감사
                #   서브콜(claimprov·selfdecl)로만 간다. 계획서는 그 회차도 그냥 `attached` 로
                #   닫는데, **그 구분이 지금 남은 결함 그 자체**라 라벨을 지우지 않고 적어 둔다.
                #   판정은 여기서 안 한다 — 감사 스크립트가 나중에 가른다([[62]] 고르지 않는다).
                _cp2_close(self, "attached", slot_n=len(_cp2),
                           via=("asub" if _am_sub else None))

        # ★W-B~W-D(arm③·검출 전용): 턴의 **최종 응답**에 대해 2패스 형식화 + §1d 검증.
        #   · 도구 **미제공** + guided_json (문법과 도구를 같이 걸면 tool_calls가 0이 된다·rev3 §3-2b)
        #   · **비커밋**: state.messages에 넣지 않는다(C208 replay 위생) — 기록은 사이드카
        #   · ENFORCE=0이면 세기만 한다(집행은 Δspurious를 만들므로 별 마일스톤)
        if os.environ.get("T2_DECLFIRST") == "1":
            try:
                import t2_declfirst as _df

                def _df_gen(msgs, schema):
                    kw = dict(self.llm_args)
                    kw.pop("tools", None)
                    eb = dict(kw.get("extra_body") or {})
                    eb["guided_json"] = schema
                    eb["guided_decoding_backend"] = "xgrammar"
                    kw["extra_body"] = eb
                    r = la.generate(model=self.llm, tools=None, messages=msgs,
                                    call_name="declfirst_formalize", **kw)
                    return getattr(r, "content", "") or ""

                _writes = {_eff_tool_name(tc) for m in state.messages
                           for tc in (getattr(m, "tool_calls", None) or [])
                           if _is_effective_write(_eff_tool_name(tc), a2)}
                # ★alias_fn(2026-07-31 Z4 교정): 디스패처 도메인에서 모델은 **내부** 도구명을
                #   선언하는데 검증기는 **외피** 이름만 갖고 있어 R4가 오탐했다. 내부 이름 해석은
                #   엔진(여기)이 알고 t2_declfirst는 모른다 — 모듈 일반성 유지.
                _res = _df.run(a2, _df_gen, self._system_messages + work, am, _writes,
                               alias_fn=_eff_tool_name)
                if _res and _res.get("violations"):
                    self._t2_df_viol = getattr(self, "_t2_df_viol", 0) + len(_res["violations"])
            except Exception as _de:
                print("[T2_DECLFIRST] 배선 예외(무시): %r" % (_de,), file=_sys.stderr, flush=True)

        # R8 종단: 잔존 게이트-deny 호출 strip (재과금 없음·히스토리 replay-clean)
        if gate is not None:
            denied = _denied_calls(am, gate, last_user, transfer_sent)
            if denied:
                d_ids = {tc.id for tc, _, _ in denied}
                kept = [tc for tc in (am.tool_calls or []) if tc.id not in d_ids]
                am.tool_calls = kept or None
                # ★A15/OL-55 (2026-08-22): 단어-경계 절단 + 빈 본문이면 재생성(노트를 본문
                #   전체로 커밋하지 않는다). 재생성은 **도구 없이**(tools=None) 산문만 받는다 —
                #   여기서 새 호출이 나오면 게이트를 우회한다.
                note = "; ".join(f"[{gid}] {_trunc_reason(why)}" for _, gid, why in denied)

                def _bn_regen_u(_ask, _work=work, _am=am):
                    _kw = dict(self.llm_args)
                    _kw.pop("tools", None)
                    _r = la.generate(model=self.llm, tools=None,
                                     messages=self._system_messages + _work + [_um(_ask)],
                                     call_name="agent_blocknote_body", **_kw)
                    return getattr(_r, "content", "") or ""
                self._t2_blocknote = getattr(self, "_t2_blocknote", collections.Counter())
                self._t2_blocknote[_commit_block_note(am, note, regen=_bn_regen_u)] += 1
                self._t2_gate_strips = getattr(self, "_t2_gate_strips", 0) + 1
                print("[T2_UNIFIED] R8 strip: %s" % note[:140], file=_sys.stderr, flush=True)
        # ★[T2_FREE_TEXT_ARG] 자유서술 기본값 인자는 **근거 없으면 넘기지 않는다** (2026-08-31·R-A1).
        #   선언: A2 `free_text_defaults` = {도구: [인자]} — 출처는 **env 시그니처**다
        #     (`tools.py:2508` `close_bank_account_7392(..., reason: str = "Customer requested
        #      closure", ...)` · 독스트링 *"reason (string, optional)"*). gold 근거 0.
        #   정책 축자(`prompts/components/policy_header.md:8`):
        #     *"Do not make up policies, information or actions that you can take on behalf of the user."*
        #   결손(base 전수 실측): gold 는 이 인자를 **안 넘겨** 행이 기본값으로 남는데 모델은 매번
        #     자기 문장을 채운다 — 060 065 066 067 068 069, 전부 `gold=None ↔ act='Customer …'`.
        #   부호표(base 98 sim · **자기-그라운딩 제거**: 호출 직전 문맥만 코퍼스로):
        #     ⊕실패 sim 발화 6 · ⊖**통과 sim 발화 0** · 무발화 92.
        #   ⚠거동: **호출은 그대로 실행**하고 그 인자만 뺀다(엔진 기본값이 정본). 값을 고르지 않는다.
        #   ⚠자기-그라운딩 금지: 코퍼스는 `state.messages`(=커밋된 직전 문맥)뿐이다. 우리가 방금
        #     보낸 값이 도구 응답에 메아리쳐 돌아오면 그 다음 호출부터 무조건 "실재"가 된다(003 실측).
        #   ⚠OFF(`T2_FREE_TEXT_ARG` != 1)면 한 글자도 안 바뀐다.
        if (os.environ.get("T2_FREE_TEXT_ARG") == "1" and getattr(am, "tool_calls", None)
                and ((a2 or {}).get("free_text_defaults"))):
            try:
                _corp9 = []
                for _m9 in (state.messages or []):
                    if getattr(_m9, "role", None) in ("user", "tool"):
                        _corp9.append(str(getattr(_m9, "content", "") or "").lower())
                try:                   # KB 도 코퍼스에 넣는다(회수 여부와 무관하게 문서 축자면 정당)
                    import t2_scaffold_get as _sg9
                    _dom9 = getattr(getattr(self, "_t2_orch", None), "environment", None)
                    _dom9 = getattr(_dom9, "domain_name", None)
                    if _dom9:
                        _corp9 += [str(_d9.get("content") or "").lower()
                                   for _d9 in (_sg9._load_domain_docs(_dom9) or [])]
                except Exception:
                    pass
                free_text_drop(am.tool_calls, " ".join(_corp9), a2,
                               log=lambda m: print(m, file=_sys.stderr, flush=True))
            except Exception as _fe9:
                print("[T2_FREE_TEXT_ARG] skip: %r" % (_fe9,), file=_sys.stderr, flush=True)

        # ★§T-8 계기 (2026-09-01·`T2_SIBLING_PAREN`) — **거동 변화 0**. 인자가 같은 호출의 다른
        #   인자 값을 괄호로 되풀이하는 모양을 세기만 한다. 왜 계기부터인가: 전 코퍼스 101건은
        #   **과거 런 분포**이고 현 스택 예측치가 아니다(재리뷰 W-4). 반려(`deny`)는 이 수를 보고
        #   붙인다 — 그리고 그때는 **반려 상한 2회 후 경고 부착 통과**로 비용을 유계로 묶는다(W-5:
        #   과거 한 sim 최다 18회 반복 · 회복률은 반려가 없던 데이터라 오프라인 측정 불가).
        if os.environ.get("T2_SIBLING_PAREN") in ("log", "deny") and getattr(am, "tool_calls", None):
            try:
                for _tcp in (am.tool_calls or []):
                    _sp = sibling_paren_arg(_tcp)
                    if _sp:
                        print("[T2_SIBLING_PAREN] %s.%s 가 형제 인자를 괄호로 되풀이한다 — %r 에서 "
                              "%r 를 빼야 한다" % (_sp[0], _sp[1], _sp[2][:70], _sp[3]),
                              file=_sys.stderr, flush=True)
            except Exception as _spe:
                print("[T2_SIBLING_PAREN] skip: %r" % (_spe,), file=_sys.stderr, flush=True)

        # ★EXHAUSTION→FAIL (T2_FAB_STRIP=1·BANK_IMPL_REDESIGN §2·2026-07-16):
        #   regen 소진 후에도 근거 없는(id-operand ∉ctx) WRITE 호출 = pass-through 금지 → strip + abstain.
        #   (C12 "id 날조는 env가 거부" 가정이 banking 디스패처 dispute엔 불성립=날조 txn이 reward0로 통과.)
        #   read/procedural=무해(strip 안함)·over-block 방지=id-operand가 ctx에 없는 write만·디스패처 nested unwrap.
        if os.environ.get("T2_FAB_STRIP") == "1" and getattr(am, "tool_calls", None):
            _RDP, _PRC = _READ_PREFIX_RE, _PROCEDURAL_RE   # ★hoist 정본 재사용([[03b]] 술어 이중화 제거·동일 정규식)
            def _fab_write_ungrounded(tc):
                nm = getattr(tc, "name", "") or ""
                ar = _args_dict(tc)
                # ★F6b(C211·리뷰 가드레일: **국소 수정** — 공유 _PROCEDURAL_RE/_eff_tool_name 불변):
                #   ⓐ inner-key에 discoverable_tool_name 추가(give의 실효 이름은 이 키에 실림 —
                #   _eff_tool_name(1667)엔 있는데 여기 없던 unwrap 불일치가 day6 발명-id give 통과 구멍)
                #   ⓑ give/call 디스패처는 inner 이름으로만 면제 판정(외피 'give_/discoverable' 매칭 금지·
                #   unlock은 구판 외피-판정 유지=안전측). 회귀: _is_effective_write("give_…")=False 유지.
                inner = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                         or ar.get("discoverable_tool_name") or "")
                eff = re.sub(r"_\d+$", "", str(inner or nm))
                if inner and re.match(r"^(give|call)_", str(nm), re.I):
                    if _RDP.match(eff) or _PRC.search(eff):
                        return eff, []  # inner가 read/procedural일 때만 무해
                elif not eff or _RDP.match(eff) or _PRC.search(eff):
                    return eff, []  # read/procedural = 무해
                sub = ar.get("arguments")
                if isinstance(sub, str):
                    try:
                        sub = json.loads(sub)
                    except Exception:
                        sub = {}
                d = sub if isinstance(sub, dict) else ar
                bad = []   # ★P4(2026-08-21·[[64]]): 미근거 (인자,값) 수집 — strip 판정은 구판과
                for k, v in (d or {}).items():   # 동일(bad 비면 무해·인자당 첫 값 하나)
                    if not _hint_hit(k, hints):
                        continue  # id-like 인자만
                    for val in _flatten(v):
                        s = str(val).strip()
                        if len(s) >= 4 and s.lower() not in ctx:
                            bad.append((k, s))  # 근거없는 id-operand 있는 write
                            break
                return eff, bad
            _fab_ids, _fab_bad = set(), []
            for tc in (am.tool_calls or []):
                _feff, _fbad = _fab_write_ungrounded(tc)
                if _fbad:
                    _fab_ids.add(id(tc))
                    _fab_bad.append((_feff, _fbad))
            if _fab_ids:
                _kept = [tc for tc in (am.tool_calls or []) if id(tc) not in _fab_ids]
                am.tool_calls = _kept or None
                # C125: 유저-대면 문자열은 영어(한글 산문 노출이 rall20 043.0 [24] 유저 혼란 실측)
                # ★P4(2026-08-21·[[64]]·t7335 halfB 079): 무지목 노트가 접힘을 재생산 — 미근거
                #   인자·값과 해소-read(A2/relations 선언 기계 도출·_fab_fix_note)를 문면에 싣는다.
                am.content = ((am.content or "")
                              + " [Note: items whose supporting records could not be verified were"
                              + " not processed." + _fab_fix_note(_fab_bad, a2) + "]")
                self._t2_fab_strips = getattr(self, "_t2_fab_strips", 0) + len(_fab_ids)
                print("[T2_FAB_STRIP] dropped %d ungrounded write call(s) (exhaustion->abstain)"
                      % len(_fab_ids), file=_sys.stderr, flush=True)
                _lbeat("T2_FAB_STRIP", orch=self, target="write",
                       fact="%d ungrounded write call(s) were not sent" % len(_fab_ids))
        # prov-fab 잔존 = 통과 (기존 prov semantics·id 날조는 env가 거부=C12)

        # ★T2_STALE_STRIP (2026-07-23·8-task per-step 포렌식): over-action 억제 — 한 턴에 12개 도구
        #   대량 병렬 중복 재호출(043[24]·054[55]·038[36] 실측: 이미 성공한 조회/write를 재호출=낭비/DB오염).
        #   ①같은 am 내 완전중복(동일 eff+args 2회+·read/write 공통·명백 무의미) ②committed서 이미 성공한
        #   *write* 재호출(중복 write=DB오염·054 gold-diff). strip만(넛지 없음·R8/FAB_STRIP 동형). read의
        #   committed-재조회는 상태변화 가능성 존중해 미strip(over-fire 방지·같은-턴 반복만). 도메인 리터럴 0
        #   (eff+args 대조·write집합=A2 confirm/eplan 도출). [[05]]: 결정론 중복감지·모델판단 아님·strip만.
        if os.environ.get("T2_STALE_STRIP") == "1" and getattr(am, "tool_calls", None):
            _wtools = _confirm_write_tools(a2) | set(((a2 or {}).get("eplan") or {}).get("write_tools") or [])
            _stale = _stale_call_ids(am, state.messages, _wtools)
            if _stale:
                _kept = [tc for tc in (am.tool_calls or []) if id(tc) not in _stale]
                am.tool_calls = _kept or None
                # ★A1/OL-18 (t7336 §6.1·2026-08-22): 구판 노트는 **한국어 + 거짓**이었다 —
                #   *"[중복 호출 제거: **이미 완료한** 조회/작업은 반복하지 않았습니다.]"*.
                #   ⑴바로 위 `:9956` 에 *"C125: 유저-대면 문자열은 영어"* 규칙이 축자로 있는데
                #     이 자리만 한글이었고, ⑵`_stale_call_ids` 는 **완료를 판정하지 않는다**(같은
                #     턴 중복 ∨ 원장의 같은-인자 write) — 그런데 노트가 완료를 단언해 user-sim 이
                #     085#1 [110] *"we've already handled the first two"* 로 **미완료를 완료로
                #     닫았다**([[25]] 유일 근거원 오염). 여기서는 **한 일(안 보냄)만** 말하고
                #     결과에 대해서는 아무것도 주장하지 않으며, 다음 행동을 지목한다([[64]]).
                # ⚠[[70]] 무엇을 파는가: 노트가 길어져 본문 끝에 붙는 문자열이 늘고, "완료" 단언이
                #   사라져 모델이 **같은 조회를 한 번 더** 시도할 수 있다. 다음 런은 `[T2_STALE_STRIP]
                #   dropped` 수와 동일-인자 재호출 수를 짝으로 센다.
                # ★OL-55 형제 (2026-08-22): 남은 호출이 있으면 이 턴은 **도구 호출 턴**이라
                #   노트가 손님에게 가지 않는다 — 종전대로 붙인다. 전부 지워졌으면 이 턴이
                #   **손님 발화**가 되므로 정본(`_commit_machine_note`)에 넘겨 빈 본문이면
                #   모델에게 본문을 다시 받는다. 재생성은 **도구 없이** 산문만 받는다.
                _snote = _STALE_NOTE % len(_stale)
                if _kept:
                    am.content = (am.content or "") + _snote
                else:
                    def _sn_regen(_ask, _work=work):
                        _kw = dict(self.llm_args)
                        _kw.pop("tools", None)
                        _r = la.generate(model=self.llm, tools=None,
                                         messages=self._system_messages + _work + [_um(_ask)],
                                         call_name="agent_stalenote_body", **_kw)
                        return getattr(_r, "content", "") or ""
                    self._t2_stalenote = getattr(self, "_t2_stalenote", collections.Counter())
                    self._t2_stalenote[_commit_machine_note(
                        am, _snote, _STALE_NOTE_ASK, regen=_sn_regen,
                        tag="T2_STALE_NOTE")] += 1
                self._t2_stale_strips = getattr(self, "_t2_stale_strips", 0) + len(_stale)
                print("[T2_STALE_STRIP] dropped %d stale/dup call(s)" % len(_stale),
                      file=_sys.stderr, flush=True)
                _lbeat("T2_STALE_STRIP", orch=self, target="turn",
                       fact="%d repeated call(s) were not sent again" % len(_stale))

        # ── DISAMB: 문맥-실재값·같은-형식 후보 2+ → 1회 재확인 (기존 로직 이식) ──
        # ★#2 operand 스코프(2026-07-13 A1-v2 실패분석): variant operand(new_item_ids)는 L4 전담 —
        #   order-filter가 여기서 new_item에 오발화해 틀린 변형 치환(3234800602 반복)했음. 제외.
        _v_ops = set((a2 or {}).get("variant_operand") or [])
        if disamb_tools and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_disamb_seen"):
                self._t2_disamb_seen = set()
            hit = None
            for tc in am.tool_calls:
                if getattr(tc, "name", None) not in disamb_tools:
                    continue
                for k, v in _args_dict(tc).items():
                    if k in _v_ops:                     # ★L4 전담 operand는 disamb-filter 제외
                        continue
                    if not _hint_hit(k, hints):
                        continue
                    for val in _flatten(v):
                        s = str(val).strip()
                        if len(s) < 4 or s.lower() not in ctx:
                            continue
                        memo = (tc.name, k, s.lower())
                        if memo in self._t2_disamb_seen:
                            continue
                        cands = _grounded_candidates(k, s, state.messages)
                        if len(cands) >= 2 and any(s.lower() == str(c).lower() for c in cands):
                            hit = (tc, k, s, cands, memo)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit and disamb_mode == "subcall":
                # ★T5-C P-B: 격리 서브콜 판정 + 제자리 치환(원턴·대화 불변). switch가 게이트-deny를
                #   ★*새로* 유발할 때만 원값 복원(리뷰 블로킹2 정신). 스위치는 whitelist operand(item)만
                #   바꾸므로 통상 게이트-무관 — *이미* deny되던 호출(다른 사유)은 R8이 어차피 strip하니
                #   되돌리면 좋은 switch만 손실. 그래서 pre/post deny 비교로 switch-유발分만 복원.
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d mode=subcall"
                      % (tc.name, k, s, len(cands)), file=_sys.stderr, flush=True)
                import copy as _copy
                _snap = _copy.deepcopy(getattr(tc, "arguments", None))
                _pre_denied = ({d[0].id for d in _denied_calls(am, gate, last_user, transfer_sent)}
                               if gate is not None else set())
                res = _t5c_disamb_subcall(self, la, UserMessage, state.messages, tc, k, s, sub_args)
                if res == "switch" and gate is not None:
                    _post = {d[0].id for d in _denied_calls(am, gate, last_user, transfer_sent)}
                    if tc.id in _post and tc.id not in _pre_denied:  # switch가 *새로* 유발한 deny만
                        tc.arguments = _snap
                        self._t2_disamb_gate_reject = getattr(self, "_t2_disamb_gate_reject", 0) + 1
                        print("[T2_UNIFIED] SUBCALL switch reverted: switch-caused gate-deny",
                              file=_sys.stderr, flush=True)
                hit = None
            if hit and disamb_mode == "enumerate":
                # ★결정론 filter-substitute (2026-07-12 HANDOFF §6.1·LOCK §4d FIND=fexec):
                #   t71 실증 = filter-then-ask *지시*는 32B 미준수([[42]]) → 지시 대신 엔진이 직접
                #   LLM-formalize→결정론 필터 실행. 1 통과=제자리 치환(subcall switch 동형:
                #   whitelist(B2)+게이트 재검사·switch-유발 deny만 복원) / ≥2=통과분으로 축소해
                #   열거-ASK / 판정불가·empty소진=기존 열거 피드백 폴백(아래 if hit: 블록).
                tc, k, s, cands, memo = hit
                fsub = None
                if _key_tokens(k) & sub_args:
                    try:
                        import t2_formalize_exec as _fx
                        fsub = _fx.fexec_filter_decide(self, la, UserMessage, state.messages, k, s)
                    except Exception as _fe:
                        print("[T2_FSUB] error (fallback): %r" % (_fe,), file=_sys.stderr, flush=True)
                        fsub = None
                if fsub and fsub["status"] == "one":
                    self._t2_disamb_seen.add(memo)
                    self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                    print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d mode=fsub"
                          % (tc.name, k, s, len(cands)), file=_sys.stderr, flush=True)
                    win = str(fsub["ids"][0]).strip()
                    if win.lower() == s.lower():
                        self._t2_fsub_keep = getattr(self, "_t2_fsub_keep", 0) + 1
                        print("[T2_FSUB] confirmed arg=%s val=%s" % (k, s),
                              file=_sys.stderr, flush=True)
                    else:
                        import copy as _copy
                        _snap = _copy.deepcopy(getattr(tc, "arguments", None))
                        _pre = ({d[0].id for d in _denied_calls(am, gate, last_user, transfer_sent)}
                                if gate is not None else set())
                        if _subst_arg_value(tc, k, s, win):
                            self._t2_fsub_switch = getattr(self, "_t2_fsub_switch", 0) + 1
                            print("[T2_FSUB] substituted arg=%s from=%s to=%s" % (k, s, win),
                                  file=_sys.stderr, flush=True)
                            if gate is not None:
                                _post = {d[0].id for d in
                                         _denied_calls(am, gate, last_user, transfer_sent)}
                                if tc.id in _post and tc.id not in _pre:  # switch가 *새로* 유발한 deny만
                                    tc.arguments = _snap
                                    self._t2_disamb_gate_reject = \
                                        getattr(self, "_t2_disamb_gate_reject", 0) + 1
                                    print("[T2_UNIFIED] FSUB switch reverted: switch-caused gate-deny",
                                          file=_sys.stderr, flush=True)
                        else:
                            self._t2_fsub_nosub = getattr(self, "_t2_fsub_nosub", 0) + 1
                            print("[T2_FSUB] substitution no-op arg=%s (value shape)" % k,
                                  file=_sys.stderr, flush=True)
                    hit = None
                elif fsub and fsub["status"] == "many":
                    # 통과 후보로 축소 → 아래 기존 열거-ASK 피드백이 축소본을 소비
                    self._t2_fsub_narrowed = getattr(self, "_t2_fsub_narrowed", 0) + 1
                    print("[T2_FSUB] narrowed arg=%s ncand %d->%d" % (k, len(cands), len(fsub["ids"])),
                          file=_sys.stderr, flush=True)
                    hit = (tc, k, s, [str(i) for i in fsub["ids"]], memo)
            if hit:
                tc, k, s, cands, memo = hit
                self._t2_disamb_seen.add(memo)
                self._t2_disamb = getattr(self, "_t2_disamb", 0) + 1
                print("[T2_DISAMB] fired tool=%s arg=%s val=%s ncand=%d" % (tc.name, k, s, len(cands)),
                      file=_sys.stderr, flush=True)
                dwork = list(work) + [am]
                _fbtmpl = DISAMB_ENUM_FEEDBACK if disamb_mode == "enumerate" else DISAMB_FEEDBACK
                fbtxt = _fbtmpl.format(k=k, s=s, n=len(cands),
                                       cands=", ".join(repr(c) for c in cands[:8]))
                for c in (am.tool_calls or []):
                    reason = fbtxt if c is tc else \
                        "Error: [DISAMBIGUATE] re-check pending; re-emit this call after resolving."
                    dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                             error=True, content=reason))
                am2 = _gen(self, dwork, bw(), "agent_response_disamb")
                n2 = 0
                while n2 < 2:  # 재확인 응답의 신규 날조는 prov 루프로 정화(2회 한도·무과금)
                    fab2 = _first_fab_call(am2, ctx, hints, selectors=sel_args)
                    if fab2 is None:
                        break
                    tc2, k2, s2 = fab2
                    n2 += 1
                    self._t2_session_bl.add(s2)
                    dwork = dwork + [am2]
                    for c in (am2.tool_calls or []):
                        reason = REGEN_FEEDBACK.format(k=k2, s=s2) if c is tc2 else \
                            "Error: [PROVENANCE] resolve the invented value first; do not call this yet."
                        dwork.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                                 error=True, content=reason))
                    am2 = _gen(self, dwork, bw(), "agent_response_regen")
                if _first_fab_call(am2, ctx, hints, selectors=sel_args) is None:
                    # ★리뷰 블로킹2: 재확인 switch가 게이트-deny 호출을 들여오면 원 am(게이트-클린) 유지
                    if gate is not None and _denied_calls(am2, gate, last_user, transfer_sent):
                        self._t2_disamb_gate_reject = getattr(self, "_t2_disamb_gate_reject", 0) + 1
                        print("[T2_UNIFIED] DISAMB rejected: switch is gate-denied; keeping original",
                              file=_sys.stderr, flush=True)
                    elif getattr(am2, "tool_calls", None):
                        sw = any(str(vv).strip().lower() != s.lower()
                                 for c2 in am2.tool_calls if c2.name == tc.name
                                 for kk, vv0 in _args_dict(c2).items() if kk == k
                                 for vv in _flatten(vv0))
                        if sw:
                            print("[T2_DISAMB] switched arg=%s from=%s" % (k, s),
                                  file=_sys.stderr, flush=True)
                        am = am2
                    else:
                        # ★T5-C fix(C61 H-C): 텍스트-only 재확인은 원 호출 유지(write 유실 차단)
                        self._t2_disamb_nowrite_keep = getattr(self, "_t2_disamb_nowrite_keep", 0) + 1
                        print("[T2_DISAMB] rejected: re-check dropped tool_calls; keeping original",
                              file=_sys.stderr, flush=True)
        # ★L4 fexec-variants (opt-in T2_L4=1·A1_V3): write 도구의 variant operand(A2 variant_operand,
        #   예 new_item_ids)를 극값/속성 결정론 선택으로 치환. I7 floor-guard: one=치환·그 외 no-op.
        #   [[05]]: variant_operand/spec = A2 데이터(엔진 리터럴 0)·fexec_variant_decide = 도메인일반.
        #   ★v3.2 (A1_V3_PROBE_FORENSIC §3): 치환 성적 2/2 오답(t58 정답파괴=교차-품목 기준누출 F1·
        #   t20 제약절단 F2+승인-불일치 F4) → T2_L4_MODE 기본 "keep"(관측·audit only·치환 없음).
        #   "substitute"는 재설계 요건(F1 per-slot attested·F2 constrained/no-op·F4 G-approve) 충족 후.
        if os.environ.get("T2_L4") == "1" and a2 is not None and getattr(am, "tool_calls", None):
            v_ops = set((a2 or {}).get("variant_operand") or [])
            v_spec = (a2 or {}).get("variant_spec")
            if v_ops and v_spec:
                req_text = " ".join(str(getattr(m, "content", "") or "")
                                    for m in state.messages if getattr(m, "role", None) == "user")
                try:
                    import t2_formalize_exec as _fx
                    import copy as _copy
                    anchor_op = (v_spec or {}).get("anchor_operand")
                    for tc in (am.tool_calls or []):
                        d = _args_dict(tc)
                        for k in list(d.keys()):
                            if k not in v_ops:
                                continue
                            newv = d.get(k)
                            newl = newv if isinstance(newv, list) else [newv]
                            anchl = d.get(anchor_op) if anchor_op else None
                            anchl = anchl if isinstance(anchl, list) else ([anchl] if anchl else [])
                            for i, cur in enumerate(newl):    # ★인덱스 짝: new_item[i]↔item[i]=anchor
                                cur = str(cur).strip()
                                if len(cur) < 3:
                                    continue
                                anchor = anchl[i] if i < len(anchl) else (anchl[0] if len(anchl) == 1 else None)
                                vr = _fx.fexec_variant_decide(self, la, UserMessage, state.messages,
                                                              k, cur, v_spec, req_text, anchor_id=anchor)
                                if vr.get("status") == "one":
                                    win = str(vr["ids"][0]).strip()
                                    self._t2_l4 = getattr(self, "_t2_l4", 0) + 1
                                    if win.lower() != cur.lower():
                                        # ★v3.2 keep-모드(기본): 치환 없이 관측만(audit 라인 유지)
                                        if os.environ.get("T2_L4_MODE", "keep") != "substitute":
                                            print("[T2_L4] keep-mode: would-substitute arg=%s from=%s to=%s"
                                                  % (k, cur, win), file=_sys.stderr, flush=True)
                                            continue
                                        # ★G-approve (F4·t20): cur가 대화(비-tool 발화)에 verbatim 등장
                                        #   = 제시·승인된 값 → 몰래 치환 금지 (도메인 리터럴 0·문자열 대조)
                                        _dlg = " ".join(str(getattr(m, "content", "") or "")
                                                        for m in state.messages
                                                        if getattr(m, "role", None) in ("user", "assistant"))
                                        if cur and cur in _dlg:
                                            print("[T2_L4] G-approve: arg=%s val=%s surfaced in dialog — no substitute"
                                                  % (k, cur), file=_sys.stderr, flush=True)
                                            continue
                                        _snap = _copy.deepcopy(getattr(tc, "arguments", None))
                                        _pre = ({dd[0].id for dd in _denied_calls(am, gate, last_user, transfer_sent)}
                                                if gate is not None else set())
                                        if _subst_arg_value(tc, k, cur, win):
                                            self._t2_l4_sub = getattr(self, "_t2_l4_sub", 0) + 1
                                            print("[T2_L4] substituted arg=%s from=%s to=%s"
                                                  % (k, cur, win), file=_sys.stderr, flush=True)
                                            if gate is not None:
                                                _post = {dd[0].id for dd in
                                                         _denied_calls(am, gate, last_user, transfer_sent)}
                                                if tc.id in _post and tc.id not in _pre:  # I7: switch-유발 deny 복원
                                                    tc.arguments = _snap
                                                    print("[T2_L4] reverted: switch-caused gate-deny",
                                                          file=_sys.stderr, flush=True)
                                    else:
                                        print("[T2_L4] confirmed arg=%s val=%s" % (k, cur),
                                              file=_sys.stderr, flush=True)
                except Exception as _le:
                    print("[T2_L4] error (no-op): %r" % (_le,), file=_sys.stderr, flush=True)
        # ★T5-C P2 원리-디폴트(opt-in T2_PRINCIPLE_DEFAULT=1): write operand 기본값(원결제 등)
        #   위반 시 제자리 치환. user-발화만 override 근거(tool출력의 계정값 아님).
        if os.environ.get("T2_PRINCIPLE_DEFAULT") == "1" and gate is not None:
            uctx = " ".join(str(getattr(m, "content", "") or "").lower()
                            for m in state.messages if getattr(m, "role", None) == "user")
            nsub = _apply_principle_default(am, a2, gate, uctx)
            if nsub:
                self._t2_principle_default = getattr(self, "_t2_principle_default", 0) + nsub
        # ★NL-NUM-PROV (opt-in T2_NLNUM_PROV=1·t47형): 최종 반환 직전 텍스트 발화의
        #   통화-금액 provenance 검사 → 생성-레벨 regen 1회(무과금·비커밋). 상한 1/턴
        #   (이 블록은 턴당 1회 실행·루프 없음). 채택 전 게이트 재검사 — regen이
        #   게이트-deny 호출을 새로 들이면 원 am 유지(안전측·부작용 0).
        if (os.environ.get("T2_NLNUM_PROV") == "1" and (a2 or {}).get("calc_tool")
                and isinstance(getattr(am, "content", None), str)):
            ctx_num = _ctx_from_messages(state.messages).replace(",", "")
            bad = _unverified_amounts(am.content, ctx_num)
            if bad:
                self._t2_nlnum = getattr(self, "_t2_nlnum", 0) + 1
                print("[T2_NLNUM] fired amount=%s" % bad[0], file=_sys.stderr, flush=True)
                fbtxt = NLNUM_FEEDBACK.format(amt=bad[0], tool=a2["calc_tool"])
                try:
                    nfb = UserMessage(role="user", content=fbtxt)
                except TypeError:
                    nfb = UserMessage(content=fbtxt)
                am2 = _gen(self, work + [am, nfb], bw(), "agent_response_nlnum")
                if gate is not None and _denied_calls(am2, gate, last_user, transfer_sent):
                    self._t2_nlnum_gate_reject = getattr(self, "_t2_nlnum_gate_reject", 0) + 1
                    print("[T2_NLNUM] rejected: regen introduced gate-denied call; keeping original",
                          file=_sys.stderr, flush=True)
                else:
                    am = am2
        # ★assertion-provenance 2-arm (opt-in·기본 OFF·floor 불변).
        #   발화 조건 = **사임**(tool_calls 없는 텍스트 발화 = 턴을 사용자에게 넘김) — 구조 이벤트.
        #   상한 1/sim(에이전트 인스턴스=sim). regen 채택 전 게이트 재검사(NLNUM과 동형·안전측).
        _resign = (not getattr(am, "tool_calls", None)
                   and isinstance(getattr(am, "content", None), str) and am.content.strip())
        # ★T2_FOLLOWUP_READLOOP (2026-07-22 §2bk·rall7 050 실측): post-submit에 모델이 KB-검색류
        #   read만 계속 돌리면 사임(텍스트-턴) 이벤트가 영영 없어 chain nudge 0회로 종료(리서치-루프
        #   회피). 확장 술어(도메인일반·A2 chain 선언 대조): after-도구 실행됨 ∧ requires 누락 ∧ 이번
        #   턴 호출 전부 비-write ∧ 그 호출들이 누락 requires와 무관 → 사임-등가로 카운트.
        if (os.environ.get("T2_FOLLOWUP_READLOOP") == "1" and not _resign
                and getattr(am, "tool_calls", None) and a2 is not None):
            try:
                _effh = {_eff_tool_name(tc) for m2 in state.messages
                         for tc in (getattr(m2, "tool_calls", None) or [])}
                _hitc = next(((_fc, _chain_dispatch(_fc, _effh))
                              for _fc in (a2.get("follow_up_chains") or [])
                              if _chain_dispatch(_fc, _effh) is not None), None)
                if _hitc is not None:
                    _ame = [_eff_tool_name(tc) for tc in (am.tool_calls or [])]
                    _reqs = _hitc[0].get("requires")
                    _reqs = set(_reqs if isinstance(_reqs, list) else [_reqs])
                    if (all(not _is_effective_write(e, a2) for e in _ame)
                            and not (set(_ame) & _reqs)):
                        _resign = True
                        # ★C207/B1(리뷰 필수2): readloop 변환 턴을 **표시**한다 — chain 예비-예산은
                        #   진성 사임-턴에서만 소비해야 한다(035 day4b 실측: cap 소진 국면이 전부
                        #   KB-루프 턴이었고 발화 3회 전부 빈손 → 예비도 같은 곳에 버려질 위험).
                        self._t2_fu_readloop_turn = True
                        print("[T2_FOLLOWUP] readloop-turn counted as resignation",
                              file=_sys.stderr, flush=True)
            except Exception:
                pass

        def _ap_regen(fbtxt, tag, tool_choice=None, am_override=None):
            from t2_lever_beat import beat as _beat
            # ★빈 문구 차단(2026-08-07·20260807g 실측 **11건**): 내용 없는 UserMessage를 열한 번
            #   보냈다. 재생성 비용은 그대로 내면서 모델에겐 아무 정보도 안 준다 = 순손실이고,
            #   사이드카에도 빈 줄로 남아 포렌식을 흐린다. 버스의 ②정직 불변이 걸러야 할 것인데
            #   그 채널은 버스를 안 거치므로(버스는 axis notes 한 채널만) **여기서 막는다** —
            #   모든 생성면 발화가 지나는 유일한 자리다.
            if not str(fbtxt or "").strip():
                print("[T2_GATE_REGEN] refused empty feedback tag=%s" % tag,
                      file=_sys.stderr, flush=True)
                return None
            # ★R8c — **잠금만 하고 안 부른 도구가 있는 동안 우리 층은 조용히 한다**
            #   (2026-08-11·C408·`T2_UNLOCK_QUIET`·기본 OFF·근거 x241·n=8).
            #   측정: 같은 턴 문맥으로 다음 한 수를 재면 **궤적만 주면 8/8** 이 그 도구를 부른다
            #   (`A_FREE`). 그런데 **우리가 실제로 넣었던 문장들을 되돌리면 1/8**(`H_LIVE_TRUE`) 이고,
            #   우리 문장 **하나만** 얹어도 4/8 이다(`B_TELL`). 오답은 "안 부른다"가 아니라 **부르는
            #   형태가 틀어진다**(잠긴 이름을 디스패처 없이 직접 호출 7/8). 즉 이 상태에서 우리
            #   조언은 도움이 아니라 **경쟁 지시**다 — C403 이 본 자해와 같은 계열이고, C404 가
            #   예측한 *"말해 주는 것으로는 안 된다"* 의 여덟 번째 사례다.
            #   ⚠격리 서브로 옮기는 길은 **먼저 재고 접었다**: 격리 문맥 `E_ISO` **2/8**(오답이
            #     `verify_identity` 로 되돌아감), 사전 상태 사실을 되돌린 `G_ISO_STATE` 는 6/8 로
            #     회복하지만 `A_FREE` 8/8 을 못 넘는다. 이 자리에 필요한 것은 *주어진 사실로 고르기*
            #     가 아니라 *대화가 어디까지 왔는지 아는 것*이라 격리가 잘하는 종류가 아니다.
            #   ⚠새 결정론 0 — 우리가 **말을 얹지 않을 뿐**이고, 무엇을 부를지는 끝까지 모델이 고른다.
            #   ⚠자기-제한적 상태다: 그 도구를 부르는 순간 조건이 사라지고 전 레버가 되돌아온다.
            #     ⇒ [[60]] 의 "끄지 마라"에 걸리지 않는다(끄는 것이 아니라 **한 상태에서 미루는 것**).
            #   ⚠Δspurious 계측 의무(§1.3): 침묵이 게이트 거부까지 미루므로 위반이 늘 수 있다.
            #     런에서 거부 수·over-action 을 함께 센다.
            if os.environ.get("T2_UNLOCK_QUIET") == "1":
                try:
                    _unlq = _unlocked_names(state.messages, a2)
                    _calq = {_exact_tool_name(_t) for _m in state.messages
                             for _t in (getattr(_m, "tool_calls", None) or [])
                             if str(getattr(_t, "name", "") or "").startswith("call_")}
                    _idleq = sorted(_unlq - _calq)
                    if _idleq:
                        print("[T2_UNLOCK_QUIET] 억제 tag=%s (미호출 잠금 %s)"
                              % (tag, ",".join(_idleq[:3])), file=_sys.stderr, flush=True)
                        return None
                except Exception as _uq:
                    print("[T2_UNLOCK_QUIET] error (no-op): %r" % (_uq,),
                          file=_sys.stderr, flush=True)
            _beat("T2_GATE_REGEN", tag)
            # ★단일 진입점 배선 — 관찰자 단계(2026-08-07·`t2_stack.observe`).
            #   생성면의 발화는 **전부 이 한 자리를 지난다**(호출 26곳). 그래서 55개 호출부를
            #   하나씩 고칠 필요가 없다 — 여기서 tag를 레버로 되돌려 스택에 등록한다.
            #   ⚠**거동은 바꾸지 않는다.** 지금은 귀속과 "route()라면 어떻게 판정했을지"만 남긴다.
            #   순서를 뒤집기 전에 순서를 검사할 수 있어야 하고, 검사 없이 켠 배선이 이 코드베이스의
            #   반복 사고였다(MATCH_COUNT 의존물 누락·WRITE_EVIDENCE 死코드·LEDGER 무음 6회).
            #   ★2026-08-07 승격: 관찰자 → **출구 게이트**. `admit()`이 발화 창을 집행한다
            #   (같은 tag·같은 문구면 재발화 접기 = [[57]]). 문구가 조금이라도 바뀌면 통과하므로
            #   *인자 변화* 기준이지 *횟수* 기준이 아니다. 레버는 그대로 켜져 있다([[60]]).
            #   비상구 = `T2_STACK_WINDOW=0`(귀속 arm 전용·태그에 기록할 것).
            try:
                import t2_stack as _stk
                _stk.observe(self, tag, text=fbtxt)
                _ok_w, _why_w = _stk.admit(self, tag, fbtxt)
                if not _ok_w:
                    print("[T2_STACK] window suppressed tag=%s (%s)" % (tag, _why_w),
                          file=_sys.stderr, flush=True)
                    return None
            except Exception:
                pass
            """피드백 1회 → regen. 게이트-deny 유입 시 원본 유지(부작용 0). 성공 시 새 am.
            ★tool_choice(레버 A·2026-07-18·`HANDOFF_LEVER_DESIGN §2`): regen 응답의 **채널만** 강제
            (어느 도구를 부를지는 모델이 고름). 실측 근거 = forced 프로브: 강제 하 24/24 정답 선택 ·
            같은 지시를 **말로** 하면 56%로 악화(단일변수·`forced_probe_20260718`).
            ★전역 regen 예산(§2ah): 소진 시 발화 skip(원본 유지)=컨텍스트 누적 상한(023 overflow 차단)."""
            if not _regen_budget_ok(self):
                print("[%s] skipped: global regen budget exhausted" % tag.upper(),
                      file=_sys.stderr, flush=True)
                return None
            try:
                _fb = UserMessage(role="user", content=fbtxt)
            except TypeError:
                _fb = UserMessage(content=fbtxt)
            # ★사이드카 확장(2026-08-05·[[55]]): 구판은 `unified_regen` 한 채널만 기록해서
            #   이 경로(SEARCH_EXHAUST·FOLLOWUP 체인 등)로 나간 지시가 사후 감사에 잡히지 않았다.
            #   048 오진 3회의 구조적 원인이 그 사각이다 — 무엇을 보냈는지 모르면 "모델이 안 했다"와
            #   "우리가 모순되게 말했다"를 가릴 수 없다. 궤적은 그대로 두고 파일에만 남긴다.
            try:
                import t2_fbsidecar as _fbsc2
                _fbsc2.record_many([_fb], state.messages, channel=tag)
            except Exception:
                pass
            # ★C207(리뷰 필수1): regen 프롬프트에 실리는 직전 응답을 **호출부가 대체**할 수 있게 한다.
            #   근거: 폭주 응답(33k자·8k토큰)을 그대로 실으면 regen 호출 자체가 창 초과로 죽고, 그 예외는
            #   여기서 전파돼 **ctxover로 끝날 sim을 더 이른 크래시로** 바꾼다. blob은 어차피 비커밋이라
            #   절단본 대체는 역사 훼손이 아니다(프레임워크 층·도메인 리터럴 0).
            _am_for_prompt = am if am_override is None else am_override
            try:
                _am2 = _gen(self, work + [_am_for_prompt, _fb], bw(),
                            "agent_response_" + tag, tool_choice=tool_choice)
            except Exception as _ge:
                # ★C207: regen 실패(창 초과 등)는 **원본 유지로 흡수**(현행 거동) — 크래시 승격 금지.
                print("[%s] regen failed (keeping original): %r" % (tag.upper(), _ge),
                      file=_sys.stderr, flush=True)
                return None
            if gate is not None:
                _den = _denied_calls(_am2, gate, last_user, transfer_sent)
                if _den:
                    # ★C170 부분-수용(W6 조정·2026-07-25): all-or-nothing 기각이 좋은 호출까지 버렸다
                    #   (cand4 035 실측: regen=unlock+call(emergency)+transfer 묶음서 transfer만 GB2-deny
                    #   인데 전체 기각→emergency 호출 소실→유저 토큰 종료로 실행 기회 상실). 조정:
                    #   denied 호출만 제거·허용 호출 유지. 전부 denied면 원본 유지(구 거동). 게이트
                    #   판정 자체는 불변(denied는 어차피 실행 단계서도 deny)·부작용-0 원칙 유지·도메인 리터럴 0.
                    _denset = {id(x[0]) for x in _den}
                    _keep = [tc for tc in (getattr(_am2, "tool_calls", None) or [])
                             if id(tc) not in _denset]
                    if not _keep:
                        print("[%s] rejected: regen introduced gate-denied call; keeping original"
                              % tag.upper(), file=_sys.stderr, flush=True)
                        return None
                    _am2.tool_calls = _keep
                    print("[%s] partial-accept: dropped %d gate-denied, kept %d call(s)"
                          % (tag.upper(), len(_den), len(_keep)), file=_sys.stderr, flush=True)
            # ★A-1 절차 재평가 (2026-08-23·`reports/facet_rft_2026/tasks__20260822/TASK_050.md`
            #   §7-①/§9-1·축 E "게이트 우회 채널"·CONFIRMED).
            #   **결손**: 이 함수가 낸 호출은 `T2_PROCEDURE` 절차 게이트를 **평가조차 받지 않고**
            #   커밋된다. 재검사 목록은 `gate`(_denied_calls)·`T2_UNLOCK_NAME`·`T2_UNLOCK_PROV`
            #   뿐이었다. 이 파일의 `T2_WEV_ROUNDS` 주석이 *"deny 후 regen된 호출은 같은 턴서
            #   무검사 커밋"* 이라고 그 구멍을 이미 자백해 놓고 WEV 에만 이식했다.
            #   **인과(n=1 짝 대조)**: t7346 `task_050` trial 0 은 사임-경로 regen 이 낸 승인 호출이
            #   무검사 커밋돼 요청-제출 write 가 MISSING → DB 해시 갈림 → reward 0.0.
            #   같은 sha·같은 A2 의 trial 1 은 **동일 호출이 원본 am 에 있었기에** 아래와 축자
            #   동일한 deny 를 받고 선행을 먼저 밟아 reward 1.0 을 받았다. 즉 여기서 되살리는
            #   문자열은 짝 trial 이 **라이브로 받은 그것**이다(문면 저작 0).
            #   **새 결정론 0**([[62]]): 술어는 기존 `t2_procedure.decide` 를 그대로 재호출하고
            #   순서는 A2 `procedures` 선언에서만 온다. 엔진에 도구명·필드명·숫자 0([[59]]).
            #   ⚠[[70]] 무엇을 파는가: 사임-경로 regen 이 절차 deny 로 접히면 그 턴이 **빈손**으로
            #     끝날 수 있다(over-action↓ / no-action↑). 부정통제 4칸([[57]]) =
            #     `T2_PROC_REGEN=1↔0` × `T2_PROCEDURE=1↔0`, 계수 = `[T2_PROCEDURE] regen-*` 라인.
            #   ⚠cap 은 메인 경로와 **공유**한다(`_t2_proc_deny`/`T2_PROCEDURE_CAP`) — 이 자리가
            #     따로 예산을 갖고 불응 루프를 돌지 않게.
            def _proc_first_deny(_amX):
                """(호출, decide결과) 또는 None. 순수 조회 — 상태 변경 0."""
                for _cX in (getattr(_amX, "tool_calls", None) or []):
                    _arX = _args_dict(_cX)
                    _alsoX = {str(_arX.get(_k)) for _k in
                              ("agent_tool_name", "user_tool_name", "discoverable_tool_name")
                              if _arX.get(_k)}
                    _dcX = _PROCR.decide(
                        _procsR, _exact_tool_name(_cX), _arX, _doneR, also_names=_alsoX,
                        unlocked=_unlocked_names(state.messages, a2),
                        pattern=((a2 or {}).get("discoverable_name_check") or {}).get("pattern"))
                    if _dcX.get("verdict") == "deny" and _dcX.get("notes"):
                        return (_cX, _dcX)
                return None

            _procsR = ((a2 or {}).get("procedures")
                       if (a2 is not None and os.environ.get("T2_PROCEDURE") == "1") else None)
            if _procsR and os.environ.get("T2_PROC_REGEN", "1") == "1":
                try:
                    import t2_procedure as _PROCR
                    _doneR = _executed_tool_counts(state.messages)
                    _hitR = _proc_first_deny(_am2)
                    if _hitR is not None:
                        _cR, _dcR = _hitR
                        _missR = ",".join(_dcR.get("missing") or [])
                        if (getattr(self, "_t2_proc_deny", 0)
                                >= int(os.environ.get("T2_PROCEDURE_CAP", "6"))):
                            # 거동 불변: 메인 경로와 같은 cap 에 걸리면 종전대로 통과시킨다. 로그만.
                            print("[T2_PROCEDURE] regen-would-fire but suppressed by=cap tag=%s "
                                  "tool=%s missing=%s prohibited=%s"
                                  % (tag, _exact_tool_name(_cR), _missR,
                                     _dcR.get("prohibited") or "-"),
                                  file=_sys.stderr, flush=True)
                        else:
                            self._t2_proc_deny = getattr(self, "_t2_proc_deny", 0) + 1
                            print("[T2_PROCEDURE] regen-deny (tag=%s) %s missing=%s "
                                  "prohibited=%s"
                                  % (tag, _exact_tool_name(_cR), _missR,
                                     _dcR.get("prohibited") or "-"),
                                  file=_sys.stderr, flush=True)
                            # 문면은 메인 경로와 **동일 규칙**으로 만든다(이중 접두 방지 포함).
                            _pnote = _dcR["notes"][0]
                            _pcontent = (_pnote if str(_pnote).lstrip().startswith("Error:")
                                         else "Error: " + str(_pnote))
                            _pfbm = ToolMessage(id=_cR.id, role="tool", requestor="assistant",
                                                error=True, content=_pcontent)
                            _amR = _gen(self, work + [_am_for_prompt, _fb, _am2, _pfbm], bw(),
                                        "agent_response_" + tag + "_procfix",
                                        tool_choice="required")
                            _okR = not (gate is not None
                                        and _denied_calls(_amR, gate, last_user, transfer_sent))
                            if _okR and _proc_first_deny(_amR) is None:
                                _am2 = _amR
                                print("[T2_PROCEDURE] regen-fix accepted tag=%s" % tag,
                                      file=_sys.stderr, flush=True)
                            else:
                                # 불응: 절차-위반 호출만 제거한다. 전부면 원본 유지(부작용 0 원칙).
                                _keepR = [_t for _t in (getattr(_am2, "tool_calls", None) or [])
                                          if id(_t) != id(_cR)]
                                if not _keepR:
                                    print("[T2_PROCEDURE] regen-fix refused; dropping regen "
                                          "(keeping original) tag=%s" % tag,
                                          file=_sys.stderr, flush=True)
                                    return None
                                _am2.tool_calls = _keepR
                                print("[T2_PROCEDURE] regen-fix refused; dropped 1 out-of-order "
                                      "call, kept %d tag=%s" % (len(_keepR), tag),
                                      file=_sys.stderr, flush=True)
                except Exception as _pre:
                    print("[T2_PROCEDURE] regen recheck error (no-op): %r" % (_pre,),
                          file=_sys.stderr, flush=True)
            # ★§2bi (rall6 실측·UNLOCK_NAME 0발화 원인): bare-name unlock이 태어나는 곳이 바로 이
            #   resign-경로 regen인데, 반환 am은 while-루프의 un_fb 검사를 **우회**해 그대로 커밋됐다
            #   (chain 18발화·un_fb 0·bare 3회 커밋 = rall6 정합). 여기서 name-check 교정 1회 수행.
            _ns = (a2 or {}).get("discoverable_name_check") or {}
            if _ns and os.environ.get("T2_UNLOCK_NAME") == "1":
                for _c2 in (getattr(_am2, "tool_calls", None) or []):
                    _ua = (_ns.get("tools") or {}).get(getattr(_c2, "name", None))
                    _uv = str(_args_dict(_c2).get(_ua) or "") if _ua else ""
                    _upat2 = _ns.get("pattern") or "_[0-9]+$"
                    _fb2 = None
                    if _uv and not re.search(_upat2, _uv):
                        self._t2_unlockname_deny = getattr(self, "_t2_unlockname_deny", 0) + 1
                        print("[T2_UNLOCK_NAME] deny bare name (followup-regen) tool=%s val=%s"
                              % (getattr(_c2, "name", None), _uv), file=_sys.stderr, flush=True)
                        _fb2 = str(_ns.get("feedback") or "Error: '{name}' needs its suffix.")
                    # ★T2_UNLOCK_PROV (2026-07-22 §2bt·rall11 050 실측): regen-경로 접미사-환각 차단.
                    #   §2bi가 bare-name만 이식해, regen이 낸 fabricated 접미사(_8374)는 무검사 커밋
                    #   → env "Unknown tool" 3연발. 메인 PROV와 동형 술어: suffixed 값이 대화 실측
                    #   근거(role=tool∪user)에 부재 or env-거부 이력(_t2_unknown_bl)이면 deny+재생성.
                    #   KB서 발견한 진짜 이름은 검색결과에 실재→통과. 엔진=부분문자열 실재확인(리터럴 0).
                    elif (_uv and os.environ.get("T2_UNLOCK_PROV") == "1"):
                        _ctx2 = " ".join(
                            str(getattr(_m3, "content", "") or "") for _m3 in state.messages
                            if getattr(_m3, "role", None) in ("tool", "user")).lower()
                        # ★우리가 지목한 이름은 출처가 있다 (2026-08-19·`T2_PROV_OURS` 확장).
                        #   후속-체인의 힌트는 **재생성 버퍼**에 있어 `state.messages` 에 없고,
                        #   핀은 아예 말하지 않는다 — 그 둘을 여기서 인정하지 않으면 우리 지목이
                        #   우리 가드에 막힌다(t7324 050 실측: `unlock-hint` 직후 이 deny).
                        _ours2 = (getattr(self, "_t2_our_names", set())
                                  if os.environ.get("T2_PROV_OURS") == "1" else set())
                        # ★A5/OL-01 (t7336 §6.1·2026-08-22): **env 레지스트리도 출처다**.
                        #   050#0 실측 — regen 이 낸 `approve_credit_limit_increase_5847` 은
                        #   **레지스트리에 실재하는 gold 이름**인데 이 가드가 "unprovenanced" 로
                        #   막았고(halfB UNLOCK_PROV deny 4건 중 **3건이 실재 이름**·오차단율 3/4),
                        #   모델은 `shell` 로 후퇴해 산문을 날조했다. [[25]]: *"env 가 '없다'고 해도
                        #   레지스트리에 있으면 있는 것"* — 그 역도 참이다. 엔진은 이 레지스트리를
                        #   **이미 읽는다**(같은 파일 `_agent_discoverable` 소비처 다수) — 사본 0.
                        #   env-거부 이력(`_t2_unknown_bl`)은 **먼저** 검사하므로 그대로 남는다.
                        # ⚠[[70]] 무엇을 파는가: **레지스트리에 실재하나 이 태스크에는 엉뚱한
                        #   이름**의 unlock 이 통과한다(구판은 KB 에 안 뜬 이름을 전부 막았다).
                        #   계기 = `T2_PROV_OURS=1↔0` × `T2_UNLOCK_PROV=1↔0` 4칸 + over-action
                        #   (마스터 §6.1 A5 행). 다음 런 포렌식이 `registry-provenanced` 수를 센다.
                        try:
                            _reg2 = _agent_discoverable(
                                getattr(getattr(self, "_t2_orch", None), "environment", None)) or set()
                        except Exception:
                            _reg2 = set()          # fail-open: 조회 실패 = 출처 추가 없음(구판)
                        if _uv and _uv in _reg2 and _uv.lower() not in _ctx2 and _uv not in _ours2:
                            print("[T2_UNLOCK_PROV] registry-provenanced (allow) tool=%s val=%s"
                                  % (getattr(_c2, "name", None), _uv),
                                  file=_sys.stderr, flush=True)
                        if (_uv in getattr(self, "_t2_unknown_bl", set())
                                or (_uv.lower() not in _ctx2 and _uv not in _ours2
                                    and _uv not in _reg2)):
                            print("[T2_UNLOCK_PROV] deny unprovenanced name (followup-regen) "
                                  "tool=%s val=%s" % (getattr(_c2, "name", None), _uv),
                                  file=_sys.stderr, flush=True)
                            _fb2 = ("Error: '{name}' does not appear in any knowledge-base result "
                                    "or tool output in this conversation - you may be inventing "
                                    "the numeric suffix, and a guessed name will be rejected. Do "
                                    "NOT guess suffixes. Search the knowledge base with plain "
                                    "words describing the step (for '{name}' e.g. \"{name_words}\") "
                                    "to find the real full suffixed name, then retry with it.")
                    if _fb2:
                        _fb2 = _fb2.replace("{name}", _uv)\
                            .replace("{name_words}",
                                     re.sub(_upat2, "", _uv).replace("_", " ").strip())
                        _fb2m = ToolMessage(id=_c2.id, role="tool", requestor="assistant",
                                            error=True, content=_fb2)
                        _am3 = _gen(self, work + [am, _fb, _am2, _fb2m], bw(),
                                    "agent_response_" + tag + "_namefix", tool_choice="required")
                        if not (gate is not None and _denied_calls(_am3, gate, last_user,
                                                                   transfer_sent)):
                            _am2 = _am3
                        break
            _regen_budget_spend(self)
            # ★§S-2 3층 (2026-09-01): **빈 재생성은 원본을 덮지 않는다**. `_ap_regen` 은 이미
            #   8곳에서 `return None = 원본 유지` 계약을 쓰므로 새 계약이 아니고, 호출부 29곳은
            #   손대지 않는다. 근거: 밤샘런 전손 5건이 전부 태스크 **전체 재시작**으로 끝났고
            #   (`Retry` 5 · 폐기 16,746초), 그중 모드 B(095)는 빈 메시지가 **재생성 프롬프트에
            #   실려** `llm_utils.py:234 assert has_content_or_tool_calls` 로 죽었다.
            if _t2_msg_empty(_am2):
                print("[T2_GATE_REGEN] empty regen (keeping original) tag=%s" % (tag,),
                      file=_sys.stderr, flush=True)
                return None
            return _am2

        # (a0) follow-up required — **완료 날조(fabricated completion) 차단** (2026-07-16 §14.3).
        #   실측: 실패 4/10 sim 전부 = 에이전트가 "제출됐다" 주장(가짜 케이스번호 발급)하고 후속 도구를
        #   안 부름 → 사용자 제출 0/4. 그중 3/4은 follow_up 도구 호출 0 = **구조 이벤트만으로 탐지**.
        #   엔진이 보는 것: {호출된 도구 이름} 집합뿐(텍스트 파싱 0·[[03b]]). 어느 도구에 어느 후속이
        #   붙는지·피드백 문구 = A2(`scaffold_get_tools[].follow_up`). 상한 1/sim·regen 채택 전 게이트 재검사.
        #   ★임계=사임 2회째(기본·env 조정가능): 오프라인 replay 실측 — 1회째는 pass 6/6 전부에 발화
        #   (pass 궤적도 결과 제시→확인 사임이 한 번 있음)·2회째는 실패 4/4 커버 + pass 2/6만 접촉.
        # ★T2_FOLLOWUP_CAP (2026-07-20 §2au·e2e10 050/052 실측): 고정 1/sim은 chain 1회 발화 후 소진 —
        #   4체크 중 2개만 진행돼도 레버 끝(CLAIMPROV cap 전소와 동일 패턴). 기본 1=거동보존·재런 3.
        # ★진행-감응 cap 환급 T2_FOLLOWUP_PROGRESS_REFUND (2026-07-22 §2bs·rall10 043/050/052 실측):
        #   chain cap3 < 사슬 6단계 — 발화가 견인한 턴은 매 발화 ~1쌍 전진했는데 "일하는 발화"까지
        #   cap을 소모해 소진 후 잔여 사슬 무방비(043 TRANSFER-포기·050/052 wrap-up). 직전 chain
        #   발화의 {missing} 중 하나라도 이후 **시도-수준** 호출(§2bh: 성공 불문)이 생기면 그 발화의
        #   cap 소모를 1회 환급 — 무진행 발화만 예산 소모(over-fire 억제는 기존과 동일).
        #   엔진=집합 교집합만(리터럴 0·[[05]]). 기본 OFF(거동보존).
        if os.environ.get("T2_FOLLOWUP_PROGRESS_REFUND") == "1":
            _cmv = getattr(self, "_t2_chain_missing", None)
            if _cmv:
                _effn = {_eff_tool_name(tc) for m in state.messages
                         for tc in (getattr(m, "tool_calls", None) or [])}
                _effn |= {_eff_tool_name(tc)
                          for tc in (getattr(am, "tool_calls", None) or [])}
                _hitp = _cmv & _effn
                if _hitp:
                    self._t2_followup_chain = max(0, getattr(self, "_t2_followup_chain", 0) - 1)
                    print("[T2_FOLLOWUP] chain progress refund hit=%s"
                          % (sorted(_hitp),), file=_sys.stderr, flush=True)
                self._t2_chain_missing = None
        # ★T2_ACTION_PROGRESS_REFUND (2026-07-22 §2bt·rall10 097 실측): action-required deny의
        #   진행-감응 환급 — 1b(FOLLOWUP)와 동일 원리. 097: cap3이 초반 compute-flail에 소진돼
        #   종반 say-loop([122-134] 8연속 무-도구 산문·write 0)가 무방비. 발화 시 기록한 target에
        #   시도-수준 착수(§2bh)가 보이면 cap 1회 환급 — 무진행 발화만 예산 소모.
        #   엔진=집합 교집합만(리터럴 0·[[05]]). 기본 OFF(거동보존).
        if os.environ.get("T2_ACTION_PROGRESS_REFUND") == "1":
            _atv = getattr(self, "_t2_action_target", None)
            if _atv:
                _effn2 = {_eff_tool_name(tc) for m in state.messages
                          for tc in (getattr(m, "tool_calls", None) or [])}
                _effn2 |= {_eff_tool_name(tc)
                           for tc in (getattr(am, "tool_calls", None) or [])}
                _hita = set(_atv) & _effn2
                if _hita:
                    self._t2_action_deny = max(0, getattr(self, "_t2_action_deny", 0) - 1)
                    print("[T2_RESOLVE] action progress refund hit=%s"
                          % (sorted(_hita),), file=_sys.stderr, flush=True)
                self._t2_action_target = None
        # ★★C207/A2·A4 (2026-07-27·`RUNAWAY_CONVERSION_DESIGN` §8-1/§8-2·day4b 실측):
        #   **모든 의미-게이트보다 먼저** 돈다 — 폭주로 오염된 응답을 뒤 게이트가 판정하면 전부 오판이고
        #   (사임-창으로 분류돼 CLAIMPROV formalize 서브콜까지 낭비), 그 blob이 커밋되면 3~5턴 만에 창이 죽는다.
        #   근거(006 m4 재파싱): 닫힌 tool_call 블록 7/7 **JSON 유효**·깨진 곳은 미종결 8번째뿐 ⇒ 형식 위반이
        #   아니라 **정지 실패**. vLLM hermes 파서가 all-or-nothing이라 유효 7개가 통째로 폐기되고 33k가 content로.
        #   엔진은 봉투 태그(서빙 포맷 상수·env) 존재와 finish_reason만 본다 — 도메인 리터럴 0([[05]]).
        _envtag = os.environ.get("T2_ENVELOPE_TAG", "<tool_call>")

        def _trunc_for_prompt(_m):
            """폭주 blob을 regen 프롬프트에 싣지 않기 위한 절단본(비커밋·리뷰 필수1)."""
            _c = str(getattr(_m, "content", None) or "")
            _lim = int(os.environ.get("T2_ENVELOPE_TRUNC", "1200") or 1200)
            if len(_c) <= _lim:
                return _m
            try:
                _cp = _m.model_copy(deep=True)
            except Exception:
                try:
                    _cp = _m.copy(deep=True)
                except Exception:
                    return _m
            _cp.content = _c[:_lim] + "\n…[truncated: the reply degenerated into repeated output]"
            return _cp

        if (os.environ.get("T2_ENVELOPE_GUARD") == "1"
                and not getattr(am, "tool_calls", None)
                and _envtag in str(getattr(am, "content", None) or "")
                and getattr(self, "_t2_envguard", 0)
                < int(os.environ.get("T2_ENVELOPE_CAP", "2") or 2)):
            self._t2_envguard = getattr(self, "_t2_envguard", 0) + 1
            print("[T2_ENVGUARD] tool-call envelope unparsed (len=%d) — required-channel regen"
                  % len(str(getattr(am, "content", None) or "")), file=_sys.stderr, flush=True)
            _newE = _ap_regen(
                "Error: [TOOL-CALL ENVELOPE] your previous reply contained tool-call markup that "
                "could not be parsed, so NO tool was executed and the customer saw nothing useful. "
                "Do not repeat the same call over and over. Issue ONE tool call now, as a real tool "
                "call, and stop.", "envguard", tool_choice="required",
                am_override=_trunc_for_prompt(am))
            if _newE is not None:
                am = _newE
                print("[T2_ENVGUARD] regen tool_calls=%s"
                      % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                      file=_sys.stderr, flush=True)
        # A4: 길이 상한 절단 응답은 **커밋하지 않는다**(cap 1 — 재생성도 잘리면 통과·무한 regen 금지).
        if (os.environ.get("T2_TRUNC_GUARD") == "1"
                and not getattr(self, "_t2_truncguard", 0)):
            _fr = None
            try:
                _fr = ((getattr(am, "raw_data", None) or {}).get("choices") or [{}])[0].get("finish_reason")
            except Exception:
                _fr = None
            if _fr == "length":
                self._t2_truncguard = 1
                print("[T2_TRUNCGUARD] finish_reason=length — regen (cap 1)",
                      file=_sys.stderr, flush=True)
                _newT = _ap_regen(
                    "Error: [TRUNCATED] your previous reply hit the length limit and was cut off — "
                    "it was almost certainly repeating itself. Answer again BRIEFLY, without "
                    "repetition; if an action is needed, make the tool call instead of describing it.",
                    "truncguard", am_override=_trunc_for_prompt(am))
                if _newT is not None:
                    am = _newT
        # ★A2/A4가 am을 교체했을 수 있다 — 뒤 게이트의 사임-판정을 **재계산**(오판 방지).
        if os.environ.get("T2_ENVELOPE_GUARD") == "1" or os.environ.get("T2_TRUNC_GUARD") == "1":
            _resign = (not getattr(am, "tool_calls", None)
                       and isinstance(getattr(am, "content", None), str) and am.content.strip())
        # ★C213/N1 (day8 021 [S]: 성공한 비-gold give↔db_match=false 완전 상관 8/8·7/7, n 소):
        #   give 대상이 원장(도구 결과·KB 회수문)에 미등장 → **넛지 1회**(강제 금지 — 술어는
        #   닫혔으나 "무관하다" 처방은 열림·경계정본 §3-2). 정당한 정책-지식 선제-give가 있을 수
        #   있어 판단은 모델에 — 발화 건 정당/무관 분류 실측 후에만 강제 승격 논의.
        if (os.environ.get("T2_GIVE_RELEVANCE_NUDGE") == "1"
                and not getattr(self, "_t2_giverel", 0)
                and getattr(am, "tool_calls", None)):
            _gtn2 = ((a2 or {}).get("dispatcher_role_check") or {}).get("give_tool")
            if _gtn2:
                _ledger_txt = "\n".join(
                    str(getattr(m2, "content", "") or "") for m2 in state.messages
                    if getattr(m2, "role", None) == "tool")
                # ★C213/N1 rev1 (day9 스모크 [S]): 초판 술어("원장 미등장")의 첫 발화가
                #   task_001의 **gold give**(apply_for_credit_card)였다 = 리뷰가 예고한 오탐 경로
                #   실현(discoverable 도구명이 KB 회수문에 문자열로 안 나오는 태스크는 gold give도
                #   항상 미등장). 강화: **이 대화에서 다른 give가 이미 성사**된 뒤의 추가 give만
                #   넛지 → 단독/최초 give(=gold 단일-give 태스크)는 발화 0.
                _prior_gives, _gid2n = set(), {}
                for _m3 in state.messages:
                    for _tc4 in (getattr(_m3, "tool_calls", None) or []):
                        if (getattr(_tc4, "name", None) == _gtn2
                                and getattr(_tc4, "requestor", "assistant") == "assistant"):
                            _gid2n[getattr(_tc4, "id", None)] = str(
                                _args_dict(_tc4).get("discoverable_tool_name") or "")
                    if getattr(_m3, "role", None) == "tool" and not getattr(_m3, "error", False):
                        _n3 = _gid2n.get(getattr(_m3, "id", None))
                        if _n3:
                            _prior_gives.add(_n3)
                for _tc3 in (am.tool_calls or []):
                    if getattr(_tc3, "name", None) != _gtn2:
                        continue
                    _tgt = str(_args_dict(_tc3).get("discoverable_tool_name") or "")
                    if _tgt and _tgt not in _ledger_txt and (_prior_gives - {_tgt}):
                        self._t2_giverel = 1
                        print("[T2_GIVE_RELEVANCE] nudge target=%s (not in ledger)" % _tgt,
                              file=_sys.stderr, flush=True)
                        _newG = _ap_regen(
                            "Error: [GIVE-RELEVANCE] you are about to give the customer the "
                            "tool '%s', but nothing retrieved in this conversation (no policy "
                            "document or tool output) mentions it. Confirm it is actually "
                            "required for the customer's CURRENT request — if not, do NOT "
                            "give it and proceed with the request instead; if policy you have "
                            "read does require it, give it again." % _tgt, "giverel")
                        if _newG is not None:
                            am = _newG
                            _resign = (not getattr(am, "tool_calls", None)
                                       and isinstance(getattr(am, "content", None), str)
                                       and am.content.strip())
                        break
        # ★C212/B3 (day7 010/014/015/016 [S]): env가 이미 'Unknown discoverable tool'로 반려한
        #   이름을 응답 본문이 다시 지시 → 반복 차단 regen(cap 2). 이름=env 에러 축자(발명 0)·
        #   대안 선택은 모델. 010/014는 같은 지시를 에러 후 2~3회 반복했다.
        if (os.environ.get("T2_UNKNOWN_REPEAT_GUARD") == "1"
                and getattr(self, "_t2_unkrep", 0) < 2
                and isinstance(getattr(am, "content", None), str) and am.content.strip()):
            _unk = _unknown_tool_names(state.messages)
            _hitn = next((n for n in _unk if n and n in am.content), None)
            if _hitn:
                self._t2_unkrep = getattr(self, "_t2_unkrep", 0) + 1
                print("[T2_UNKNOWN_REPEAT] name=%s (cap %d/2)" % (_hitn, self._t2_unkrep),
                      file=_sys.stderr, flush=True)
                _newU = _ap_regen(
                    "Error: [UNKNOWN-TOOL REPEAT] '%s' does not exist — the environment already "
                    "returned \"Unknown discoverable tool\" for that exact name. Do NOT instruct "
                    "the customer to run it again and do NOT call it again. Search the knowledge "
                    "base with PLAIN WORDS describing the action to find the real tool or "
                    "procedure; if none exists, say so honestly instead." % _hitn, "unkrepeat")
                if _newU is not None:
                    am = _newU
                    _resign = (not getattr(am, "tool_calls", None)
                               and isinstance(getattr(am, "content", None), str)
                               and am.content.strip())
        # ★C212/A3 (day7 018 [S]): env가 이미 'Unexpected parameter'로 반려한 인자를 give-경로
        #   호출이 다시 실음 → 반복 차단 regen(cap 2). 인자명=env 에러 축자·give 도구명=A2
        #   dispatcher_role 선언(리터럴 0)·재작성은 모델.
        if (os.environ.get("T2_UNKNOWN_REPEAT_GUARD") == "1"
                and getattr(self, "_t2_argrep", 0) < 2
                and getattr(am, "tool_calls", None)):
            _gtn = ((a2 or {}).get("dispatcher_role_check") or {}).get("give_tool")
            _rej = _rejected_params(state.messages) if _gtn else set()
            _hitp = None
            if _rej:
                for _tcx in (am.tool_calls or []):
                    if getattr(_tcx, "name", None) != _gtn:
                        continue
                    _blob = json.dumps(_args_dict(_tcx), ensure_ascii=False, default=str)
                    _hitp = next((p for p in _rej if p in _blob), None)
                    if _hitp:
                        break
            if _hitp:
                self._t2_argrep = getattr(self, "_t2_argrep", 0) + 1
                print("[T2_ARG_REPEAT] param=%s (cap %d/2)" % (_hitp, self._t2_argrep),
                      file=_sys.stderr, flush=True)
                _newP = _ap_regen(
                    "Error: [ARG-REPEAT] the environment already rejected the parameter '%s' "
                    "(\"Unexpected parameter\") on a previous call. Re-issue the call WITHOUT "
                    "'%s' — pass ONLY the parameters that tool accepts; put any extra values "
                    "in your reply text for the customer instead." % (_hitp, _hitp),
                    "argrepeat", tool_choice=("required" if os.environ.get(
                        "T2_FOLLOWUP_FORCE") == "1" else None))
                if _newP is not None:
                    am = _newP
                    _resign = (not getattr(am, "tool_calls", None)
                               and isinstance(getattr(am, "content", None), str)
                               and am.content.strip())
        # ★C212/B1 (day7 019/022/027 [S]): 엔진 [coverage] 미판정-행 재호출 지시가 무시된 채
        #   사임 → 1회 regen으로 표면화(지시문=엔진 자기-생성 라인 재인용·판단 0). 019/022는
        #   그 미판정 행이 gold 디스퓨트를 직접 삼켰다.
        if (os.environ.get("T2_COVERAGE_FOLLOWUP") == "1" and _resign
                and not getattr(self, "_t2_covfu", 0)):
            _cvp = _coverage_pending(state.messages)
            if _cvp and _cvp[0]:
                self._t2_covfu = 1
                print("[T2_COVERAGE_FU] fired tool=%s" % _cvp[0],
                      file=_sys.stderr, flush=True)
                _newC = _ap_regen(
                    "Error: [COVERAGE-FOLLOWUP] an earlier '%s' result reported rows it could "
                    "not verify: \"%s\" — that instruction has not been carried out. Read the "
                    "missing value(s) from the records that contain them and call '%s' again "
                    "with the completed input for those rows BEFORE concluding or handing off."
                    % (_cvp[0], _cvp[1], _cvp[0]), "covfollowup",
                    tool_choice=("required"
                                 if os.environ.get("T2_FOLLOWUP_FORCE") == "1" else None))
                if _newC is not None:
                    am = _newC
                    _resign = (not getattr(am, "tool_calls", None)
                               and isinstance(getattr(am, "content", None), str)
                               and am.content.strip())
        # ★C214/E1 (day8 003 [S]): 판정도구가 'unverified'(조건부·미문서화) 행을 돌려줬고 그 조건을
        #   확정하는 재호출이 없는 채 사임 → 1회 넛지. 003=Silver의 fx_fee가 premium 조건부라
        #   unverified였는데 유저가 premium 보유를 밝힌 뒤에도 재실행 0 → gold 카드 오선택.
        #   **강제 없음**(지금 재호출이 맞는지는 상황 의존·경계정본 §3-2).
        if (os.environ.get("T2_UNVERIFIED_FOLLOWUP") == "1" and _resign
                and not getattr(self, "_t2_unvfu", 0)):
            _uvp = _unverified_pending(state.messages)
            if _uvp:
                self._t2_unvfu = 1
                print("[T2_UNVERIFIED_FU] fired tool=%s" % _uvp[0], file=_sys.stderr, flush=True)
                _newV = _ap_regen(
                    "Error: [UNVERIFIED-FOLLOWUP] an earlier '%s' result flagged fact(s) it could "
                    "NOT verify: %s — a conditional/undocumented value is NOT known to hold. If the "
                    "customer has since told you something that settles that condition, call '%s' "
                    "again with it (or read the cited source document) BEFORE recommending or "
                    "concluding; if it cannot be settled, say so plainly instead of assuming."
                    % (_uvp[0], _uvp[1], _uvp[0]), "unverifiedfu")
                if _newV is not None:
                    am = _newV
                    _resign = (not getattr(am, "tool_calls", None)
                               and isinstance(getattr(am, "content", None), str)
                               and am.content.strip())
        # ★C214/E2 (day7 019 [S]): give는 성사됐는데 유저가 그 도구를 한 번도 실행하지 않은 채
        #   사임 → 실행 안내 1회 넛지(019=포털 불가라던 유저에게 "이 대화에서 실행 가능"을 끝내
        #   안 알려 디스퓨트 0건). 술어=구조 사실(give 성공 ∧ user 호출 0)·처방=안내(넛지).
        if (os.environ.get("T2_GIVE_EXEC_NUDGE") == "1" and _resign
                and not getattr(self, "_t2_givexec", 0)):
            _gtn3 = ((a2 or {}).get("dispatcher_role_check") or {}).get("give_tool")
            _ucall = ((a2 or {}).get("dispatcher_role_check") or {}).get("user_call")
            if _gtn3 and _ucall:
                # 술어 = 순수함수(단위테스트 공유·`test_give_exec_user_direct.py`).
                _idle = give_exec_idle(state.messages, _gtn3, _ucall)
                if _idle:
                    self._t2_givexec = 1
                    print("[T2_GIVE_EXEC] nudge idle=%s" % _idle, file=_sys.stderr, flush=True)
                    _newX = _ap_regen(
                        "Error: [GIVE-EXEC] you gave the customer %s but they have not run it. "
                        "They do NOT need a portal, an app, or a link — the tool runs right here "
                        "in this conversation. Tell them plainly to run it now (name it and give "
                        "the exact arguments), instead of ending or handing off."
                        % ", ".join("'%s'" % t for t in _idle), "givexec")
                    if _newX is not None:
                        am = _newX
                        _resign = (not getattr(am, "tool_calls", None)
                                   and isinstance(getattr(am, "content", None), str)
                                   and am.content.strip())
        # ★C214/E3 (day7 012/033 · day8 032 [S]): 같은-검색 반복이 엔진 스텁으로 이미 여러 번
        #   반려됐는데 계속 검색·표류 → 1회 넛지(012=8회 전패 후 앱 절차 날조·033/032=1~수회 후
        #   전용 경로 포기). 술어=엔진 자기 스텁 계수(닫힘)·처방=전략 전환 권고(넛지).
        # ★C12(2026-08-05·053 실측): **열어 놓고 부르지 않은 도구**를 사임 턴에 한 번 짚는다.
        #   053은 처방 후 gold 액션 16개 중 15개를 맞췄고, 남은 하나가 `approve_credit_limit_increase_5847`
        #   — unlock은 했는데 인자를 실은 호출이 없었다. 술어는 완전히 닫혀 있다(해제 이력 ∧ 호출 이력).
        #   판단도, 도메인 어휘도 없다: 이름은 대화가 이미 말한 것이고 부를지는 모델이 정한다.
        if (os.environ.get("T2_UNCALLED_UNLOCK") == "1" and _resign
                and not getattr(self, "_t2_uncalled_fired", 0)):
            try:
                _unl12 = _unlocked_names(state.messages, a2)
                _called12 = {_exact_tool_name(_t) for _m in state.messages
                             for _t in (getattr(_m, "tool_calls", None) or [])
                             if str(getattr(_t, "name", "") or "").startswith("call_")}
                _idle12 = sorted(_unl12 - _called12)
                if _idle12:
                    self._t2_uncalled_fired = 1
                    print("[T2_UNCALLED_UNLOCK] surface %s" % ",".join(_idle12[:4]),
                          file=_sys.stderr, flush=True)
                    _newS = _ap_regen(
                        # ⚠`_ap_regen`은 (문구, tag)를 받는다 — tag 없이 부르면 TypeError로 조용히
                        #   no-op이 되고 표적을 8번 잡고도 **한 번도 전달되지 않는다**(20260805q 실측 4회).
                        "Error: [UNLOCKED-NOT-CALLED] you unlocked %s in this conversation and never "
                        "called it. Unlocking only makes a tool available — it performs nothing. If "
                        "that step is still required, call it now with its arguments; if it is not "
                        "required, say plainly why you are not calling it."
                        % ", ".join(_idle12[:4]), "uncalled_unlock")
                    if _newS is not None:
                        am = _newS
                        _resign = (not getattr(am, "tool_calls", None)
                                   and isinstance(getattr(am, "content", None), str)
                                   and am.content.strip())
            except Exception as _u12:
                print("[T2_UNCALLED_UNLOCK] error (no-op): %r" % (_u12,),
                      file=_sys.stderr, flush=True)

        # ★(2) 분기 판정 표면화 (2026-08-05·사용자 지시 "1,2,3만"): 우리 검사가 이미 낸 판정을
        #   결정 시점에 그대로 인용한다. 051은 approve와 deny 중 하나를 골라야 했고 둘 다 안 불렀다.
        #   종단-결정 문구는 **선행 단계가 전부 끝나야** 발화해서 r에서 1회뿐이었다 — 그래서 그 조건과
        #   분리한다: 판정이 실재하고 결정 도구가 미호출이면 말한다. 어느 쪽인지는 고르지 않는다([[10]]).
        if (os.environ.get("T2_VERDICT_SURFACE") == "1" and _resign
                and not getattr(self, "_t2_verdict_fired", 0)):
            try:
                _dec2 = [d for _fc2 in ((a2 or {}).get("follow_up_chains") or [])
                         for d in (_fc2.get("decision_tools") or [])]
                _eff2 = _executed_tool_names(state.messages)
                if _dec2 and not (set(_dec2) & _eff2):
                    _chk2 = sorted({t for _fc2 in ((a2 or {}).get("follow_up_chains") or [])
                                    for t in ((_fc2.get("after") if isinstance(_fc2.get("after"), list)
                                               else [_fc2.get("after")]) or []) if t})
                    _line2, _pend2 = None, {}
                    for _m2 in state.messages:
                        if getattr(_m2, "role", None) == "assistant":
                            for _t2 in (getattr(_m2, "tool_calls", None) or []):
                                _pend2[getattr(_t2, "id", None)] = _exact_tool_name(_t2)
                        elif getattr(_m2, "role", None) == "tool":
                            _nm2v = _pend2.get(getattr(_m2, "id", None)
                                               or getattr(_m2, "tool_call_id", None))
                            if _nm2v in _chk2:
                                _c2v = str(getattr(_m2, "content", "") or "").strip()
                                if _c2v and not _c2v.startswith("Error:"):
                                    _line2 = _c2v.split("\n")[0][:220]
                    if _line2:
                        self._t2_verdict_fired = 1
                        print("[T2_VERDICT_SURFACE] surface decision=%s" % ",".join(_dec2[:3]),
                              file=_sys.stderr, flush=True)
                        _newV = _ap_regen(
                            "Error: [VERDICT] the check you ran returned: \"%s\" — and the declared "
                            "terminal decision has not been made: %s. Choose the one that follows from "
                            "that result and call it with its arguments, or state plainly why neither "
                            "applies. Describing the outcome does not record it."
                            % (_line2, ", ".join(_dec2[:3])), "verdict_surface")
                        if _newV is not None:
                            am = _newV
                            _resign = (not getattr(am, "tool_calls", None)
                                       and isinstance(getattr(am, "content", None), str)
                                       and am.content.strip())
            except Exception as _v2e:
                print("[T2_VERDICT_SURFACE] error (no-op): %r" % (_v2e,),
                      file=_sys.stderr, flush=True)

        # ★2026-08-31 수리③ (`T2_SEARCH_EXHAUST_MID=1`·기본 OFF): 종전에는 `_resign`
        #   (모델이 **마무리하려는 순간**)에만 걸렸다. 실측 x686: 넛지가 **093(통과)에서만**
        #   발화하고 **094(실패)에서는 0회** — 094 는 28번 검색하는 **도중**이라 resign 이
        #   아니었고, 그 42 메시지 동안 아무 개입이 없었다.
        #   ⛔반복이 아니다: 고유 명령 27/28 · 완전동일 최대 2회 → `T2_REPEAT_CAP` 은 정상 무발화.
        #   결손은 **탐색 종료 판정**이다. dry(새 문서 id 0) 가 임계를 넘으면 resign 전에도 말한다.
        #   ⛔거동 변경 · [[70]] 부호표 대상. cap 은 아래 `_t2_srchex` 가 이미 1회로 묶는다.
        _srchex_mid = os.environ.get("T2_SEARCH_EXHAUST_MID") == "1"
        if (os.environ.get("T2_SEARCH_EXHAUST_NUDGE") == "1" and (_resign or _srchex_mid)
                and not getattr(self, "_t2_srchex", 0)):
            _stubs = sum(1 for _m6 in state.messages
                         if getattr(_m6, "role", None) == "tool"
                         and isinstance(getattr(_m6, "content", None), str)
                         and ("[DUPLICATE-READ]" in _m6.content
                              or "[NEAR-DUPLICATE-READ]" in _m6.content))
            # ★2026-08-05(012 실측): 술어가 **같은 질의의 반복**만 셌다. 012는 질의를 매번 바꿔
            #   가며 검색했고 중복 스텁이 문턱에 못 미쳐 침묵했으며, 그 다음 턴에 KB에 없는 앱
            #   경로를 지어냈다. 소진의 일반형은 "같은 말을 반복했나"가 아니라 **회수가 더 이상
            #   자라지 않는가**다 — 새 문서 id가 0인 검색을 연속으로 센다(마커는 A2 선언·판단 0).
            _idm = str((((a2 or {}).get("axis_notes") or {}).get("doc_id_marker") or "")).strip()
            if _idm:
                _seen_ids, _dry = set(), 0
                for _m7 in state.messages:
                    if getattr(_m7, "role", None) != "tool":
                        continue
                    _c7 = getattr(_m7, "content", None)
                    if not isinstance(_c7, str) or _idm not in _c7:
                        continue
                    _ids = set()
                    for _part in _c7.split(_idm)[1:]:
                        _ids.add(_part.split()[0] if _part.split() else "")
                    _new = _ids - _seen_ids
                    _seen_ids |= _ids
                    _dry = 0 if _new else _dry + 1
                _stubs = max(_stubs, _dry)
            if _stubs >= int(os.environ.get("T2_SEARCH_EXHAUST_TH", "2") or 2):
                self._t2_srchex = 1
                print("[T2_SEARCH_EXHAUST] nudge stubs=%d" % _stubs, file=_sys.stderr, flush=True)
                # ★C11(2026-08-05): 소진 시점에 **이미 회수한 문서가 이름을 말한 도구**를 다시
                #   짚는다. 스모크 실측: 아무도 부르지 않은 gold 도구 16건 중 **12건이 이미 검색
                #   결과 본문에 있었다**. 레지스트리를 통째로 열거하면 그건 발견을 대신 해 주는
                #   것이라 하지 않는다([[05]] Q3) — 교집합(레지스트리 ∩ 이 대화가 이미 받은 텍스트)
                #   에서 아직 해제·호출되지 않은 것만, 새 정보 없이 다시 말한다.
                _seen9 = ""
                try:
                    _reg9 = _agent_discoverable(
                        getattr(getattr(self, "_t2_orch", None), "environment", None))
                    if _reg9:
                        _txt9 = "\n".join(str(getattr(_m9, "content", "") or "")
                                          for _m9 in state.messages
                                          if getattr(_m9, "role", None) == "tool")
                        _used9 = {_exact_tool_name(_t9) for _m9 in state.messages
                                  for _t9 in (getattr(_m9, "tool_calls", None) or [])}
                        _used9 |= _unlocked_names(state.messages, a2)
                        _cand9 = sorted(n for n in _reg9 if n in _txt9 and n not in _used9)
                        if _cand9:
                            _seen9 = (" The documents you have ALREADY retrieved name these tools,"
                                      " which you have not called yet: %s. Their names are exact —"
                                      " unlock and call one of them rather than searching again."
                                      % ", ".join(_cand9[:6]))
                            print("[T2_SEARCH_EXHAUST] retrieved-but-unused %d" % len(_cand9),
                                  file=_sys.stderr, flush=True)
                except Exception as _s9e:
                    print("[T2_SEARCH_EXHAUST] retrieved-name error (no-op): %r" % (_s9e,),
                          file=_sys.stderr, flush=True)
                _newS = _ap_regen(
                    "Error: [SEARCH-EXHAUST] the knowledge base has already rejected %d repeated "
                    "search(es) as duplicates — repeating them will not return anything new. "
                    "Either search with DIFFERENT plain words describing the action a policy "
                    "document would use, or act on what you already retrieved. Do NOT invent a "
                    "procedure, menu path, or tool name that no document gave you; if nothing "
                    "covers this request, say so honestly and follow the escalation policy. %s"
                    % (_stubs, (a2 or {}).get("search_exhaust_escalation") or ""),
                    "searchexhaust")
                if _newS is not None:
                    am = _newS
                    _resign = (not getattr(am, "tool_calls", None)
                               and isinstance(getattr(am, "content", None), str)
                               and am.content.strip())
        # ★follow-up 예산 폐기 (2026-08-05·사용자 지시). 그 전에 예산이 실제로 한 일:
        #   기본 cap 1을 scaffold 넛지와 체인 디스패처가 **함께** 썼고, give 넛지가 먼저 쓰면
        #   종국 턴의 체인은 술어가 성립해도 침묵했다 — 028에서 `chain[3]`(분쟁→보상갱신)이 정확히
        #   그렇게 죽었다(술어 HIT·발화 마크 0·억제 마크 0). 실제 상한은 러너의 `--max_steps 200`이고,
        #   예산이 막아준 것은 확인되지 않았다. 카운터는 **관측 전용**으로 남긴다(몇 번 말했는지 재려고).
        if (os.environ.get("T2_FOLLOWUP_REQUIRED") == "1" and _resign):
            _called0 = _called_tools(state.messages)
            for _d0 in ((a2 or {}).get("scaffold_get_tools") or []):
                _fu = _d0.get("follow_up") or {}
                _ft = _fu.get("tool")
                # ★C212/A1: tool_args 선언 시 인자-대조 이행 판정 — 무관-대상 동명 give가
                #   조건을 영구 충족시키던 갭(day7 022/027 [S]) 차단. 미선언=종전 동작.
                if (_ft and _d0.get("name") in _called0
                        and not _fu_target_called(state.messages, _ft,
                                                  _fu.get("tool_args") or {})
                        and _fu.get("feedback")):
                    _th = int(os.environ.get("T2_FOLLOWUP_RESIGN_TH", "2") or 2)
                    self._t2_fu_resigns = getattr(self, "_t2_fu_resigns", 0) + 1
                    if self._t2_fu_resigns < _th:
                        break
                    self._t2_followup_sg = getattr(self, "_t2_followup_sg", 0) + 1
                    print("[T2_FOLLOWUP] fired tool=%s missing_follow_up=%s"
                          % (_d0.get("name"), _ft), file=_sys.stderr, flush=True)
                    # ★레버 A(T2_FOLLOWUP_FORCE=1·기본 OFF): FOLLOWUP regen의 **빈손 43~50%**(regen이 도구
                    #   대신 산문을 냄·nt=20 실측 dreq2 6/14·ctl2 7/14)를 채널 강제로 닫는다. 이 순간은
                    #   ASK가 정당한 출구가 아니다 — 데이터는 이미 손에 있고 남은 일이 인계뿐.
                    _new0 = _ap_regen(_fu["feedback"], "followup",
                                      tool_choice=("required"
                                                   if os.environ.get("T2_FOLLOWUP_FORCE") == "1" else None))
                    if _new0 is not None:
                        am = _new0
                        print("[T2_FOLLOWUP] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                    break
            # ★follow_up_chains (2026-07-20 Q1 coverage·050형 "submit 후 절차 미완 만족종료"):
            #   scaffold follow_up의 **디스패처 확장** — after/requires를 effective 도구명(_eff_tool_name·
            #   call_ unwrap·suffix strip)으로 대조. A2 선언(도구쌍·문구)·엔진=집합 대조만(리터럴 0).
            #   같은 사임-임계·1/sim cap 공유.
            # ★관문2 확장(2026-07-20·§2aa): requires = 문자열 or **리스트(full required-set)**.
            #   050 실증 = 단일 requires(history)는 이미 호출됐고 **pending을 건너뜀** → 단일 대조는 못 잡음.
            #   전량 대조 + 누락 도구 **전량 나열**(`{missing}` 치환·050 follow-through+054 query-gap 동시 커버).
            #   ＋종단결정 nudge: requires 전부 충족·사임·`decision_tools` 미호출이면 `decision_feedback` 1회
            #   (approve **강제 아님** — decline-정답 케이스(052)가 있어 문구가 양방향 명시·Δspurious 계측 대상).
            # ★C207/B1: cap 소진 후에도 A2 `reserve` 선언 체인은 **진성 사임-턴** 1회 보장(_fu_window).
            _fu_res_declared = any(_fc.get("reserve")
                                   for _fc in ((a2 or {}).get("follow_up_chains") or []))
            _fu_genuine = not getattr(self, "_t2_fu_readloop_turn", False)
            # 예산 폐기 — 술어가 성립하면 말한다. `_fu_window`/`reserve`는 예산 시절의 장치라
            # 더 이상 호출하지 않는다(순수함수와 그 검정은 이력 보존을 위해 남겨 둔다).
            _fu_mode = "normal"
            if _fu_mode is not None:
                # ★패턴 제거(2026-08-05·사용자 지시 "패턴 매칭은 치팅"): 구판은 접미사를 떼서
                #   (`_eff_tool_name`) 선언과 맞췄다 — 철자 규칙으로 **대응을 추정**한 것이다.
                #   선언이 레지스트리의 정확한 이름을 말하도록 고쳤으므로(A2 follow_up_chains)
                #   여기서는 **집합 대조만** 한다. 표도 만들지 않는다: 진실을 선언이 말하면 된다.
                _eff0 = {_exact_tool_name(tc) for m in state.messages
                         for tc in (getattr(m, "tool_calls", None) or [])}
                for _fc in ((a2 or {}).get("follow_up_chains") or []):
                    if _fu_mode == "reserve" and not _fc.get("reserve"):
                        continue              # 예비-창은 선언 체인에만
                    _hit1 = _chain_dispatch(_fc, _eff0)     # (feedback, tag) or None — 순수함수(단위테스트 공유)
                    if _hit1 is None:
                        continue
                    _fb1, _tag1 = _hit1
                    # ★2026-08-05(028 실측): 체인은 발화했는데(`chain fired` 8회) 모델은 **unlock 호출
                    #   0회**였다. 문구가 `{missing}`에 **접미사 없는 이름**만 넣어, 048과 똑같이
                    #   "부를 수 있는 이름"도 "잠겨 있다는 사실"도 주지 않았다. env 레지스트리로
                    #   정확한 이름을 풀고(모호하면 침묵) 잠금 절을 A2 문구로 덧붙인다.
                    try:
                        _reg = _agent_discoverable(
                            getattr(getattr(self, "_t2_orch", None), "environment", None))
                        _unl1 = _unlocked_names(state.messages, a2)
                        _rq0 = _fc.get("requires")
                        _rq0 = _rq0 if isinstance(_rq0, list) else [_rq0]
                        # 선언이 정확한 이름을 말하므로 집합 대조만 한다(패턴 0).
                        _need = [x for x in (_in_registry(r, _reg) for r in _rq0 if r)
                                 if x and x not in _unl1]
                        _uh = (a2 or {}).get("tool_unlock_hint")
                        if _need and _uh:
                            _fb1 = _fb1 + " " + str(_uh).replace("{tools}", ", ".join(_need[:3]))
                            print("[T2_FOLLOWUP] unlock-hint %s" % ",".join(_need[:3]),
                                  file=_sys.stderr, flush=True)
                    except Exception as _uhe:
                        print("[T2_FOLLOWUP] unlock-hint error (no-op): %r" % (_uhe,),
                              file=_sys.stderr, flush=True)
                    # ★C201/D2(2026-07-26·리뷰 결함2 실측): 전역 임계(기본 2)는 **1회 사임 뒤 종료되는
                    #   궤적**(035: 에스컬 호출→notice 1턴→유저 terminal)에서 구조적 미발화. 체인별
                    #   `resign_th`로 override(미선언=env 기본=거동 보존). 전역을 낮추면 기존 체인까지
                    #   조기 발화해 Δspurious가 스택 전체로 번지므로 per-chain으로 국소화.
                    _th = int(os.environ.get("T2_FOLLOWUP_RESIGN_TH", "2") or 2)
                    if _fc.get("resign_th") is not None:
                        _th = int(_fc["resign_th"])
                    self._t2_fu_resigns = getattr(self, "_t2_fu_resigns", 0) + 1
                    if self._t2_fu_resigns < _th:
                        # ★C214 진단(day7 028 [S]: dispute→update 체인이 A2에 선언돼 있는데
                        #   라이브 미발화 — 원인 미확정). 억제 사실을 마크로 남겨 다음 런에서
                        #   "임계 미달 억제"인지 "판정 자체 미도달"인지 구분한다(계측만·거동 0).
                        print("[T2_FOLLOWUP] chain suppressed(th=%d resigns=%d) after=%s"
                              % (_th, self._t2_fu_resigns, _fc.get("after")),
                              file=_sys.stderr, flush=True)
                        break
                    # ★C13(2026-08-05): 예산 대신 **상태 조건**. 예산을 없앤 뒤 같은 FOLLOW-UP이 한
                    #   sim에서 최대 6회 반복됐고(048은 중복읽기 1→10·메시지 63→98로 악화), 반복이
                    #   행동을 바꾼 적은 계량상 없다(048에서 4~5회 반복 후에도 미호출). 숫자 제한이
                    #   아니라 "새로 할 말이 있을 때만 말한다" — 미이행 집합이 직전과 같으면 침묵한다.
                    #   ⚠2026-08-05 교정(p↔q 실측): "미이행 집합 불변이면 침묵"은 **너무 강했다**.
                    #   050은 FOLLOW-UP 4회를 받고 12/13을 맞췄는데 억제 후 1회·6/13으로, 051은 6회·
                    #   11/20에서 1회·8/20으로 후퇴했다. 반복이 무효였던 것은 048 하나뿐인데 그 한
                    #   사례로 일반화했다([[08]]). 조건을 좁힌다 — **직전 발화 이후 도구 호출이 하나도
                    #   없었을 때만** 침묵한다(공전은 끊고, 뭔가 하고 있으면 계속 짚는다).
                    _said = getattr(self, "_t2_chain_said", None)
                    if _said is None:
                        _said = self._t2_chain_said = {}
                    _rq13 = _fc.get("requires")
                    _rq13 = _rq13 if isinstance(_rq13, list) else ([_rq13] if _rq13 else [])
                    _key13 = (str(_fc.get("after")), _tag1)
                    _ncalls13 = sum(1 for _m13 in state.messages
                                    for _t13 in (getattr(_m13, "tool_calls", None) or []))
                    _cur13 = (frozenset(r for r in _rq13 if r not in _eff0), _ncalls13)
                    if _said.get(_key13) == _cur13:
                        print("[T2_FOLLOWUP] chain unchanged — silent after=%s"
                              % (_fc.get("after"),), file=_sys.stderr, flush=True)
                        break
                    _said[_key13] = _cur13
                    self._t2_followup_chain = getattr(self, "_t2_followup_chain", 0) + 1
                    print("[T2_FOLLOWUP] chain fired(%s) after=%s"
                          % (_tag1, _fc.get("after")), file=_sys.stderr, flush=True)
                    # ★진행-감응 환급용 스냅샷(2026-07-22 §2bs): 발화 시점의 미이행 집합 기록.
                    #   다음 평가에서 이 중 하나라도 시도-수준 호출이 보이면 cap 소모 1회 환급.
                    if _tag1 == "followup_decision":
                        self._t2_chain_missing = {d for d in (_fc.get("decision_tools") or [])
                                                  if d not in _eff0}
                    else:
                        _rq1 = _fc.get("requires")
                        _rq1 = _rq1 if isinstance(_rq1, list) else [_rq1]
                        self._t2_chain_missing = {r for r in _rq1 if r not in _eff0}
                    # ★§2bh: 구판은 followup_chain만 required — decision regen(종단결정 nudge)은
                    #   강제 없이 빈손 가능(rall5 실측: followup_decision 3회 발화·미이행). 문구가
                    #   양방향(approve/deny)이라 방향-중립·도구-호출 강제는 안전.
                    _new1 = _ap_regen(_fb1, _tag1,
                                      tool_choice=("required"
                                                   if os.environ.get("T2_FOLLOWUP_FORCE") == "1"
                                                   and _tag1 in ("followup_chain", "followup_decision")
                                                   else None))
                    if _new1 is not None:
                        am = _new1
                    break
        # (a1) write-provenance — **완료-주장 게이트**(③형·2026-07-17 사용자 제안: "출력도 출처를 밝혀라").
        #   C45(입력 출처선언)의 출력측 쌍대: 완료를 주장하려면 근거 이벤트가 원장에 있어야 한다.
        #   ③형의 급소 = pass("당신이 실행하라")와 fail("내가 제출했다")이 **구조 동일·말만 다름** →
        #   LLM이 자기 답변의 완료-주장을 **이진 선언**(formalize·[[10]])하고, 엔진은 선언 구조체 +
        #   결정론 원장만 검증. 답변 텍스트 파싱 0([[03b]]).
        # ★2026-07-18 일반화(`A2_DOMAIN_GENERALIZATION_DESIGN §2.2`): 트리거를 **task-불변**으로.
        #   ~~구판: follow_up 도구가 호출됨~~ = task_019 리터럴(give-missing 5건은 give를 *안* 불러서 못 잡음).
        #   신판: **완료 주장 ∧ 원장에 실효 write 실행 0 ∧ 사임**. producer/follow_up 개념 불요.
        #   `_any_effective_write`가 user·agent 실행 양쪽을 본다 → agent-side 태스크(69/97)서 에이전트가
        #   WRITE 실행했으면 발화 안 함(오탐 방지·F2). give 태스크(21/97)서 사용자 실행 0이면 주장은 거짓.
        #   completion_guard 문구 = A2 도메인-수준(`a2['completion_guard']`) 우선, 없으면 follow_up 사본(하위호환).
        # ★T2_PROCEDURE_LEFT (2026-08-26·기본 OFF) — **끝내는 자리에서 남은 칸을 이름으로 말한다**.
        #   t7361 실물: 셋이 같은 모양으로 끝났고, 우리 층은 매번 **무엇이 남았는지 알고 있었다**.
        #     050  `[T2_PROCEDURE] checklist proc=credit_limit_increase nodes=7 done=5 left=['decision']`
        #          → 그 직후 `[T2_CLAIMPROV] window hit(resign) claims=3 unbacked=0` → `regen tool_calls=[]`
        #          ⇒ 남은 칸이 `decision`(=승인) 하나인 걸 알면서 완료 선언을 통과시켰다. reward 1.0→0.0.
        #     074  `[T2_WRITEPROV] window hit (no effective write in ledger) declared_completion=False`
        #          → 모델은 완료를 *주장*하지 않고 **인간 이관**으로 나갔다 ⇒ 기존 게이트가 침묵.
        #     085  `[E-PLAN]` 이 분쟁 write 를 4회 막은 뒤 이관 ⇒ 같은 자리.
        #   기존 둘이 왜 못 잡나: `WRITE_PROV` 는 *완료 주장* 을 전제하고, `CLAIM_PROV` 는
        #   *"어떤 write 가 원장에 있나"* 를 본다. 050 은 `submit_…_request` 가 원장에 있어
        #   `unbacked=0` 이 됐다 — **어떤 write 가 아니라 절차의 남은 칸**이 물음이었다.
        #   ⇒ 술어는 이미 있고 이미 계산된다(`t2_procedure.checklist` · [[67]] 정본 지도의
        #     *"무엇이 아직 안 됐나(체크리스트·빼기)"* 칸). 여기서는 **읽어서 이름만** 말한다.
        #   ⚠엔진은 **고르지 않는다** — 순위도 최댓값도 없다. 선언이 `done is False` 로 표시한
        #     노드를 **전부** 인쇄한다(부분집합을 우리가 집으면 그 순간 우리가 답을 고른 것이다).
        #   ⚠[[63]] 빼기 형태다 — *"남은 것은 이것뿐"* 이 모델이 닫을 수 있는 유일한 모양이다.
        #   ⚠gold 미접촉([[23]]) · 도메인 낱말 0 · 태스크 id 0 · 새 A2 키 0(`procedures` 직독).
        _left_fb = None
        if (os.environ.get("T2_PROCEDURE_LEFT") == "1" and _resign
                and not getattr(self, "_t2_procleft", 0)):
            try:
                import t2_procedure as _PL
                _procs = (a2 or {}).get("procedures") or []
                _done = _executed_tool_counts(state.messages)
                _rows, _pids = [], []
                for _p in _PL.active_procedures(_procs, _done):
                    for _nid, _tools, _ok in _PL.checklist(_p, _done):
                        if _ok is False:
                            _rows.append((_nid, list(_tools or [])))
                            if _p.get("id") not in _pids:
                                _pids.append(_p.get("id"))
                if _rows:
                    self._t2_procleft = 1
                    _lines = chr(10).join(
                        "- %s: %s" % (_n, (", ".join(_t) if _t else "(no tool named)"))
                        for _n, _t in _rows)
                    _left_fb = (
                        "Error: [PROCEDURE-LEFT] you are closing this conversation, but the "
                        "procedure(s) %s still have steps the policy asked for that have not "
                        "been done here:%s%s%s"
                        "These are all of them - nothing else is outstanding. Do them now, using "
                        "the tool named on each line, before you finish. If one of them truly "
                        "cannot be done, say which and why, instead of ending as if it were done."
                        % (", ".join(str(x) for x in _pids), chr(10), _lines, chr(10)))
                    print("[T2_PROCEDURE_LEFT] 종료 창에서 남은 칸 %d개 전달: %s"
                          % (len(_rows), [r[0] for r in _rows]),
                          file=_sys.stderr, flush=True)
                    _lbeat("T2_PROCEDURE_LEFT", orch=self,
                           target=",".join(str(x) for x in _pids),
                           fact="the declared procedure still has unmet steps at closing time",
                           order=_lines)
                    # 채널은 **재생성**이다 — 이 자리엔 도구 호출이 없을 수 있어(마지막 산문
                    #   턴) `(tc, msg)` 짝 채널을 못 쓴다. `WRITE_PROV` 가 같은 창에서 쓰는
                    #   경로를 그대로 쓴다(스텁이 히스토리에 안 남는다·[[30]] eval 재실행 정합).
                    _newL = _ap_regen(_left_fb, "procleft")
                    if _newL is not None:
                        am = _newL
                        print("[T2_PROCEDURE_LEFT] regen tool_calls=%s"
                              % ([getattr(t, "name", None)
                                  for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
            except Exception as _ple:
                _left_fb = None
                print("[T2_PROCEDURE_LEFT] 건너뜀(무발화): %r" % (_ple,),
                      file=_sys.stderr, flush=True)

        _cgd = (a2 or {}).get("completion_guard") or {}
        if not _cgd:
            for _d1 in ((a2 or {}).get("scaffold_get_tools") or []):
                _c0 = (_d1.get("follow_up") or {}).get("completion_guard") or {}
                if _c0:
                    _cgd = _c0
                    break
        if (os.environ.get("T2_WRITE_PROV") == "1" and _resign
                and not getattr(self, "_t2_writeprov", 0)
                and _cgd.get("claim_question") and _cgd.get("feedback")):
            for _once in (True,):                 # 구조 유지용 단일 루프(break 재사용)
                _cg = _cgd
                if _any_effective_write(state.messages, _a2_of(self)):
                    break                          # 세상을 바꾼 실행이 원장에 있음 → 완료 주장 정당(오탐 방지)
                _claims = None
                try:
                    try:
                        _dm1 = _gen(self, work + [am, UserMessage(role="user", content=_cg["claim_question"])],
                                    bw(), "agent_writeprov")
                    except TypeError:
                        _dm1 = _gen(self, work + [am, UserMessage(content=_cg["claim_question"])],
                                    bw(), "agent_writeprov")
                    for _ln in (getattr(_dm1, "content", None) or "").splitlines():
                        _ln = _ln.strip().rstrip(",")
                        if _ln.startswith("{") and _ln.endswith("}"):
                            try:
                                _j = json.loads(_ln)
                                if isinstance(_j, dict) and "claims_completion" in _j:
                                    _claims = bool(_j["claims_completion"])
                            except Exception:
                                pass
                except Exception as _we:
                    print("[T2_WRITEPROV] declaration failed (no-op): %r" % (_we,),
                          file=_sys.stderr, flush=True)
                print("[T2_WRITEPROV] window hit (no effective write in ledger) declared_completion=%s"
                      % (_claims,), file=_sys.stderr, flush=True)
                # ★2026-08-30 수리(§L-8 · x659 실측): **예산은 질의 자체가 소모한다.**
                #   종전에는 이 증가가 `if _claims:` 안에 있어, 답이 파싱되지 않으면(_claims=None)
                #   상한이 영원히 0 으로 남아 매 resign 턴마다 8192-토큰 서브콜을 다시 태웠다.
                #   실측: `WRITEPROV window hit` 가 **x659 31회 · x668 6회**(설계는 sim 당 1회).
                #   Q2.5 는 간결한 한 줄 JSON 을 내어 파싱되고 1회로 끝났고, 장황한 Q3.8 만 무한
                #   재발화했다 — **모델 의존 결함**이지 도메인 문제가 아니다([[05]] 안전).
                #   ⛔거동 변경: 파싱 실패 시 이 게이트는 이제 sim 당 1회만 시도한다([[70]] 부호표).
                self._t2_writeprov = getattr(self, "_t2_writeprov", 0) + 1
                if _claims:
                    _lbeat("T2_WRITE_PROV", orch=self, target="write",
                           fact="completion was declared but no effective write is in the ledger",
                           order=str(_cg["feedback"]))
                    _new1 = _ap_regen(_cg["feedback"], "writeprov")
                    if _new1 is not None:
                        am = _new1
                        print("[T2_WRITEPROV] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                break
        # ★C193 notice-재발화 억제 (2026-07-26·야간 97-런 회귀 3/3 실측: 005·032·035 — 어제 pass가
        #   전부 이 기전으로 fail. +004·016 동형). 도메인 정책(notice 게이트)은 [notice 송신 →
        #   손님 동의 → transfer 도구 호출] 순서를 요구하는데, 에이전트가 **동의 확보 후에도
        #   notice를 재발화**하며 턴을 넘기면 user-sim이 터미널 신호로 응답해 sim이 그 자리에서
        #   종료 → gold transfer 도구 영영 미호출(레이스). 실측: 005 msg19 손님 "Yes, please" 평문
        #   동의 → msg20 재발화 → 터미널 / 035는 도구 출력이 즉시-transfer를 지시했는데도 재발화.
        #   엔진이 보는 것 = A2 notice_text 문자열의 {현재 발화 포함 ∧ 과거 송신 실재 ∧ transfer
        #   호출 부재} 결정론 대조뿐. **도구명도 A2에서**: notice 게이트 tool 키 ∨
        #   claim_prov.event_map['transfer'] 패턴([[05]] 엔진 리터럴 0). 교정=재발화 대신 도구
        #   호출을 지시하는 regen 1회(cap 1/sim·호출·인자·계속여부=모델·[[10]]).
        if (os.environ.get("T2_NOTICE_REPEAT", "1") == "1"
                and not getattr(self, "_t2_noticerep", 0)):
            _ngate = next((g for g in ((a2 or {}).get("gates") or [])
                           if g.get("kind") == "notice" and g.get("notice_text")), None)
            _ntxt = (_ngate or {}).get("notice_text") or ""
            _emap4 = ((a2 or {}).get("claim_prov") or {}).get("event_map") or {}
            _trname = ((_ngate or {}).get("tool")
                       or (_emap4.get("transfer") if isinstance(_emap4.get("transfer"), str) else None)
                       or "the transfer tool named in your policy")
            _amc4 = str(getattr(am, "content", None) or "")
            _nkey = _ntxt[:40] if _ntxt else ""
            if _nkey and _nkey in _amc4:
                _sent_before = any(
                    _nkey in str(getattr(_m4, "content", None) or "")
                    for _m4 in state.messages
                    if getattr(_m4, "role", None) == "assistant")
                if _sent_before and not _is_transfer_call(am, _emap4):
                    self._t2_noticerep = 1
                    print("[T2_NOTICEREP] repeated notice without transfer call — regen",
                          file=_sys.stderr, flush=True)
                    _new4 = _ap_regen(
                        "You already sent that exact transfer notice earlier in this conversation, "
                        "and the customer has already responded to it. Do NOT send the notice again "
                        "— repeating the question only stalls the request. If the transfer should "
                        "proceed, CALL %s NOW (as a tool call, with an appropriate summary); "
                        "otherwise continue helping the customer without re-sending the notice."
                        % _trname, "noticerep")
                    if _new4 is not None:
                        am = _new4
                        print("[T2_NOTICEREP] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
        # (a1b) ★T2_CLAIM_PROV (2026-07-20·사용자: "모든 '했다' 주장을 원장대조") — WRITEPROV의 일반형.
        #   완료-주장(write축)만 묻던 이진 선언을 **모든 과거-행동 주장 목록 formalize**로 확장:
        #   LLM이 자기 답변의 "이미 했다" 주장들을 {kind, what}로 선언([[10]] formalize) → 엔진은
        #   A2 `claim_prov.event_map`(kind→도구 접두 패턴 or "__effective_write__")으로 원장 이벤트 실재만
        #   대조(집합 교차·텍스트 파싱 0·[[03b]]). 미등재 kind=skip(오탐 방지). 043 확인-날조(KB 0회·
        #   "checked" 주장) 직접 표적·완료-주장은 kind=write로 흡수(WRITEPROV 상위호환·병행 시 중복 주의).
        #   기본 OFF(T2_CLAIM_PROV=1)·사임-윈도우·1/sim.
        _cpv = (a2 or {}).get("claim_prov") or {}
        # ★관문5(2026-07-20·038 transfer-escape·§2ad): 발화창 = 사임 ∨ **transfer-류 호출**.
        #   038 실측: "I will file 3 disputes..."(SAY)→TRANSFER NOTICE로 탈출(정당 도구호출이라
        #   FORCE_ACTION 사각·미래형이라 구판 CLAIM 사각). transfer 패턴=A2 event_map['transfer'] 재사용.
        _cpv_transfer = _is_transfer_call(am, _cpv.get("event_map") or {})
        # ★cap env화(2026-07-20·smoke023d 포렌식·§2am): 1/sim 고정은 빈손 regen(tool_calls=[]·기지의
        #   43~50%) 1회에 레버 전소 → 이후 완료날조·transfer-escape 무방비(실측: 첫 발화 빈손→msg21
        #   "이미 logged" 날조·msg37 transfer 탈출 모두 무검사). 기본 1=거동보존·스모크 3.
        _cpv_cap = int(os.environ.get("T2_CLAIMPROV_CAP", "1") or 1)
        # ★transfer-창 별도 예산(2026-07-20·e2e9 038 포렌식·§2ao): resign-창 발화가 cap을 소진하면
        #   **탈출 직전**(transfer 호출) 최후 감사가 무산(038 실측: 16 hit 전부 resign·transfer-창 0).
        #   transfer-창은 cap과 독립적으로 sim당 1회 보장(사임-창과 상호배타: transfer=tool_call 있음).
        # ★C201/D3(2026-07-26·§7-0): cap 소진 뒤에도 **행동-kind 주장 전용 예비 1회**(A2 reserve_kinds).
        #   근거 실측: unbacked>0인데 regen 무발생 A11·B5 — 초반 저가치 발화가 예산을 태우고 종단
        #   완료-날조(032 transfer·026/028 record_update)가 무검사 통과. 판정=순수함수 _cpv_window.
        _cpv_rsv_kinds = _cpv.get("reserve_kinds") or []
        _cpv_mode = _cpv_window(bool(_resign), bool(_cpv_transfer),
                                getattr(self, "_t2_claimprov", 0), _cpv_cap,
                                getattr(self, "_t2_claimprov_tr", 0),
                                getattr(self, "_t2_claimprov_rsv", 0), bool(_cpv_rsv_kinds))
        _cpv_win_ok = _cpv_mode is not None
        if (os.environ.get("T2_CLAIM_PROV") == "1" and (_resign or _cpv_transfer)
                and _cpv_win_ok
                and _cpv.get("question") and _cpv.get("feedback") and _cpv.get("event_map")):
            for _once in (True,):
                _cl, _pd = None, None
                try:
                    try:
                        _dm2 = _gen(self, work + [am, UserMessage(role="user", content=_cpv["question"])],
                                    bw(), "agent_claimprov")
                    except TypeError:
                        _dm2 = _gen(self, work + [am, UserMessage(content=_cpv["question"])],
                                    bw(), "agent_claimprov")
                    _txt2 = getattr(_dm2, "content", None) or ""
                    _mj = re.search(r"\{.*\}", _txt2, re.S)
                    if _mj:
                        _j2 = json.loads(_mj.group(0))
                        if isinstance(_j2, dict) and isinstance(_j2.get("claims"), list):
                            _cl = _j2["claims"]
                        # ★관문5 미래형: pending = 대화 전체서 "하겠다" 약속·미이행 목록(A2 question v2가 요구).
                        if isinstance(_j2, dict) and isinstance(_j2.get("pending"), list):
                            _pd = _j2["pending"]
                except Exception as _ce2:
                    print("[T2_CLAIMPROV] declaration failed (no-op): %r" % (_ce2,),
                          file=_sys.stderr, flush=True)
                if not _cl and not _pd:
                    print("[T2_CLAIMPROV] window hit claims=%s pending=%s" % (_cl, _pd),
                          file=_sys.stderr, flush=True)
                    break
                # 원장 이벤트 집합: 원명 + effective명(디스패처 unwrap·suffix strip)
                # ★A2/OL-21 (t7336 §6.1·2026-08-22): **성공한 호출만** 원장이다 — 구판은 이름만
                #   모아 env 가 거부한 호출을 "했다"의 근거로 썼다(094#0 `unbacked=0` 날조 완결).
                #   판정·계기는 `_ledger_event_names`(정본 술어 재사용·사본 0) 참조.
                _evs, _evs_drop = _ledger_event_names(state.messages)
                if _evs_drop:
                    print("[T2_CLAIMPROV] ledger narrowed: %d failed call(s) excluded %s"
                          % (len(_evs_drop), [d[0] for d in _evs_drop][:4]),
                          file=_sys.stderr, flush=True)
                _emap = _cpv["event_map"]
                # ★과거형 claims만 tool-미스→kind 강등 (050 DUP 수리·docstring ★★).
                #   pending(:아래)은 기본 False 유지 — 038형 탈출-티켓 방어 보존.
                _unbacked = _claim_unbacked(_cl, _emap, _evs, state.messages, _a2_of(self),
                                            kind_fallback_on_miss=True)
                # ★격리 검증 (2026-08-18·`T2_CLAIM_VERIFY`·기본 OFF): 이름 대조로 **구제된**
                #   주장만 서브에게 다시 묻는다 — *"그 도구가 정말 그 일을 했는가"*. 이름이
                #   원장에 있다는 사실은 t7318 에서 조회 도구를 환급으로 통과시켰다.
                try:
                    _resc = [c for c in (_cl or [])
                             if str((c or {}).get("tool") or "").strip() and c not in _unbacked]
                    # ⚠문면은 **L1 선언에서 직접** 읽는다 — `claim_prov` 합성에 키를 흘리면
                    #   레거시 생성물 `<domain>.gate.json` 과 등가가 깨진다([[24]]·3층 검정이 잡았다).
                    _cvspec = {"verify_question": ((_a2_of(self) or {}).get("claim_audit")
                                                   or {}).get("verify_question")}
                    for _cf in _claim_verify_false(self, _cvspec, _resc, _evs):
                        if _cf not in _unbacked:
                            _unbacked.append(_cf)
                except Exception as _cve:
                    print("[T2_CLAIM_VERIFY] 건너뜀(무발화): %r" % (_cve,),
                          file=sys.stderr, flush=True)
                # 미래-약속: 같은 원장대조 — 이 창(사임/transfer)에서 미이행 약속 = 영영 미이행(탈출티켓).
                #   feedback_pending 미선언(구판 A2)이면 발화 0(거동보존).
                _unb_p = (_claim_unbacked(_pd, _emap, _evs, state.messages, _a2_of(self))
                          if _cpv.get("feedback_pending") else [])
                # ★C207/C2-a(2026-07-27): **미보유 기능 약속** — 약속에 실린 도구가 쓸 수 있는 도구
                #   집합에 아예 없으면(=OTP/SMS 발송처럼 존재하지 않는 기능) 원장 대조 이전에 불가능한
                #   약속이다(004·035 실측: 없는 OTP를 여러 턴 반복 약속하며 창 소진). 엔진=집합 대조만·
                #   discoverable(잠금·접미사)까지 포함해 오탐 0(리뷰 필수3).
                _unavail = []
                _unavail_locked = []      # try 실패 시 미대입 방지(오늘 `_rz` 사고 부류)
                if os.environ.get("T2_UNAVAIL_PROMISE") == "1" and _cpv.get("feedback_unavailable"):
                    try:
                        # ★P7(C208⑥·DAY5_PRESCRIPTIONS §P7): 구판 `getattr(orch, ...)`는 이 스코프에
                        #   `orch`가 없어 **전량 NameError**(day5 0/223 무음 스킵). env는 init_inject가
                        #   심어둔 `_t2_orch`(orchestrator)에서 해석한다.
                        _known = _known_tool_names(getattr(self, "tools", None),
                                                   getattr(getattr(self, "_t2_orch", None),
                                                           "environment", None),
                                                   state.messages)
                        # ★A3/OL-19 (2026-08-22): 원장-실재 전제 + 구절 분할. `_ledger_text` 는
                        #   궤적 축자(모든 content + tool_call 이름/인자)이고, 여기에 없는 이름은
                        #   **모델이 낸 것이 아니다** — 074 의 `apply_credits_to_account_1234` 가
                        #   그랬다(궤적 0회·우리 서브 산출). 실재 안 하면 침묵한다([[25]]).
                        # ⚠범위는 **서브가 본 그대로**여야 한다 = `work + [am]`. `_pd` 는 위
                        #   `_gen(self, work + [am], …)` 서브의 JSON 산출이므로, 이 턴의 약속은
                        #   아직 `state.messages` 에 없다 — 거기서만 찾으면 **정당한 발화까지
                        #   전멸**한다(첫 판이 그랬다). `work ⊇ state.messages` 이므로 상위집합이고,
                        #   fail-open(더 많이 인정 = 거짓 고발 축소·[[25]]) 방향이다.
                        _led3 = _ledger_text(list(work) + [am])
                        _unavail, _unavail_locked = _unavailable_promises(
                            _pd, _known,
                            discoverable=_agent_discoverable(
                                getattr(getattr(self, "_t2_orch", None), "environment", None)),
                            ledger_text=_led3)
                        # 계기([[70]]): 원장-실재 전제가 **몇 건을 침묵시켰나** = 판 것의 크기.
                        _u_old, _ul_old = _unavailable_promises(
                            _pd, _known,
                            discoverable=_agent_discoverable(
                                getattr(getattr(self, "_t2_orch", None), "environment", None)))
                        _n_sil = (len(_u_old) + len(_ul_old)) - (len(_unavail) + len(_unavail_locked))
                        if _n_sil > 0:
                            print("[T2_UNAVAIL] ledger-absent silenced=%d (names not in trajectory)"
                                  % _n_sil, file=_sys.stderr, flush=True)
                        _lever_health("unavail", "ok")
                        if _unavail or _unavail_locked:
                            _lever_health("unavail", "fired")
                            print("[T2_UNAVAIL] promised tools not available: %s · locked: %s"
                                  % ([p.get("tool") for p in _unavail][:3],
                                     [p.get("tool") for p in _unavail_locked][:3]),
                                  file=_sys.stderr, flush=True)
                    except Exception as _ue:
                        _lever_health("unavail", "skipped")
                        print("[T2_UNAVAIL] skipped (no-op): %r" % (_ue,),
                              file=_sys.stderr, flush=True)
                print("[T2_CLAIMPROV] window hit(%s) claims=%d unbacked=%d pending=%d unb_p=%d %s"
                      % ("transfer" if _cpv_transfer and not _resign else "resign",
                         len(_cl or []), len(_unbacked), len(_pd or []), len(_unb_p),
                         [c.get("kind") for c in (_unbacked + _unb_p)][:4]), file=_sys.stderr, flush=True)
                # ★C201/D3: 예비-창은 **행동-kind 주장**에만 쓴다(저가치 주장으로 예비 소진 방지).
                if _cpv_mode == "reserve" and not (_claim_has_kind(_unbacked, _cpv_rsv_kinds)
                                                   or _claim_has_kind(_unb_p, _cpv_rsv_kinds)):
                    print("[T2_CLAIMPROV] reserve window: no action-kind claim — skip",
                          file=_sys.stderr, flush=True)
                    break
                if _unbacked or _unb_p:
                    _lbeat("T2_CLAIM_PROV", orch=self, target="claim",
                           fact="%d claimed action(s) are not in the execution ledger"
                                % (len(_unbacked) + len(_unb_p)))
                    self._t2_claimprov = getattr(self, "_t2_claimprov", 0) + 1
                    if _cpv_mode == "reserve":
                        self._t2_claimprov_rsv = 1       # 예비-창(1/sim) 소진 마킹
                    if _cpv_transfer and not _resign:
                        self._t2_claimprov_tr = 1        # transfer-창 예산(1/sim) 소진 마킹

                    def _desc3(cc):
                        return "; ".join("%s: %s" % (c.get("kind"), str(c.get("what"))[:60])
                                         for c in cc[:3])
                    _parts = []
                    if _unbacked:
                        _parts.append(_cpv["feedback"].replace("{claims}", _desc3(_unbacked)))
                    # ★C348⒢(2026-08-09): 미이행-약속을 **도구 소유자**로 가른다.
                    #   · 손님 소유 → 안내가 곧 이행이므로 **침묵**(한 일을 안 했다고 말하지 않는다)
                    #   · 에이전트 소유 → 진짜 결함은 약속 위반이 아니라 **자기 도구를 떠넘긴 것**
                    #     ⇒ 소유권 **사실만** 표면화한다(C216 §2-3b: claim 축은 표면화만·8398행 주석의
                    #       쓰기-강제 철회 이력을 반복하지 않는다).
                    #   · 모름 → 구판 문구 그대로(거동 보존). A2 미선언이면 전체가 구판이다.
                    _pend_rest = _unb_p
                    if _unb_p and _cpv.get("feedback_ownership"):
                        try:
                            _own_p, _their_p, _unk_p = _split_claims_by_owner(
                                _unb_p,
                                [getattr(t, "name", None) for t in (getattr(self, "tools", None) or [])],
                                _user_discoverable(getattr(getattr(self, "_t2_orch", None),
                                                           "environment", None)),
                                # ★FIX-8: 도구 미지 주장의 소유권 회수용 후보 = **에이전트
                                #   discoverable 레지스트리**(env 기계 사실). 격리 x300.
                                registry=_agent_discoverable(
                                    getattr(getattr(self, "_t2_orch", None), "environment", None)))
                            print("[T2_CLAIMPROV] owner split: agent=%d user=%d unknown=%d"
                                  % (len(_own_p), len(_their_p), len(_unk_p)),
                                  file=_sys.stderr, flush=True)
                            if _own_p:
                                _parts.append(_cpv["feedback_ownership"].replace(
                                    "{claims}", "; ".join(
                                        "%s (tool: %s)" % (str(c.get("what"))[:50], c.get("tool"))
                                        for c in _own_p[:3])))
                            _pend_rest = _unk_p          # theirs = 침묵 · own = 위에서 다룸
                        except Exception as _oe:
                            print("[T2_CLAIMPROV] owner split skipped (no-op): %r" % (_oe,),
                                  file=_sys.stderr, flush=True)
                            _pend_rest = _unb_p
                    if _pend_rest:
                        _parts.append(_cpv["feedback_pending"].replace("{claims}", _desc3(_pend_rest)))
                    if _unavail:                       # C207/C2-a
                        _parts.append(_cpv["feedback_unavailable"].replace(
                            "{claims}", "; ".join("%s (tool: %s)" % (str(p.get("what"))[:50], p.get("tool"))
                                                  for p in _unavail[:3])))
                    # ★잠김-분기 (2026-08-12·070t0 t72): 레지스트리에 실재하는 도구를
                    #   "존재하지 않는다"고 말하던 자리 — 새 키 없으면 **침묵**한다(거짓 문구로
                    #   강등하지 않는다·[[25]] 유일 근거원 오염 방지).
                    if _unavail_locked and _cpv.get("feedback_unavailable_locked"):
                        _parts.append(_cpv["feedback_unavailable_locked"].replace(
                            "{claims}", "; ".join("%s (tool: %s)" % (str(p.get("what"))[:50], p.get("tool"))
                                                  for p in _unavail_locked[:3])))
                    # ☠2026-08-05: 이 자리에 `tool_choice="required"`(탐지→행동 전환 강제)를 붙였다가
                    #   **철회**했다. 등대가 이미 세 번 금지한 것이다 — ①C216 §2-3b: claim 축은
                    #   **표면화만**으로 강등(코어 6층 동결) ②C216 금지선: *"열린 술어 ∨ 열린 처방 ∨
                    #   사례-표적 위 강제"*, 그리고 CLAIMPROV는 개입 482건 = 최대 단일이라 3b 강등의
                    #   40% 제거 대상 ③C218 딥리서치 [S-lit]: **required 강제 = 유해 문서화**(jailbreak·
                    #   환각 유발) + Verifier Tax(94% 차단해도 safe-success<5%). 여기에 §1.5 Q5
                    #   (쓰기 강제 금지·p<0.5면 기대-유해)가 더 있다.
                    #   전환율 0.15(로그 실측)는 **레버로 메울 구멍이 아니라 prompt-ceiling의 표시**이고
                    #   등대는 그 잔여를 learn 축으로 이관해 뒀다([[13]] 순서·C216 결정).
                    _new2 = _ap_regen("\n".join(_parts), "claimprov")
                    if _new2 is not None:
                        am = _new2
                        print("[T2_CLAIMPROV] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                break
        # (a) discovery-required — 엔진이 보는 것: {호출된 도구 이름}뿐.
        if (os.environ.get("T2_DISCOVERY_REQUIRED") == "1" and (a2 or {}).get("analysis_producers")
                and _resign and not getattr(self, "_t2_discreq", 0)):
            _called = _called_tools(state.messages)
            for _sp in (a2.get("analysis_producers") or []):
                _ds, _pr = _sp.get("data_source"), _sp.get("producer")
                if _ds in _called and _pr and _pr not in _called:
                    self._t2_discreq = getattr(self, "_t2_discreq", 0) + 1
                    print("[T2_DISCREQ] fired data_source=%s producer=%s" % (_ds, _pr),
                          file=_sys.stderr, flush=True)
                    _new = _ap_regen(DISCREQ_FEEDBACK.format(
                        data_source=_ds, producer=_pr, subject=_sp.get("subject") or "this"), "discreq")
                    if _new is not None:
                        am = _new
                        print("[T2_DISCREQ] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                    break
        # (b) self-declaration — 엔진이 보는 것: LLM 선언 JSON 필드뿐([[10]]). 선언 sub-call은 비커밋.
        elif (os.environ.get("T2_SELF_DECLARATION") == "1" and (a2 or {}).get("assertion_operands")
                and _resign and not getattr(self, "_t2_selfdecl", 0)):
            _ao = a2.get("assertion_operands") or {}
            try:
                _dp = SELFDECL_PROMPT.format(items=", ".join(sorted(_ao)))
                try:
                    _dm = _gen(self, work + [am, UserMessage(role="user", content=_dp)], bw(),
                               "agent_selfdecl")
                except TypeError:
                    _dm = _gen(self, work + [am, UserMessage(content=_dp)], bw(), "agent_selfdecl")
                _decls = _parse_declaration(getattr(_dm, "content", None))
            except Exception as _de:
                print("[T2_SELFDECL] declaration failed (no-op): %r" % (_de,), file=_sys.stderr, flush=True)
                _decls = []
            print("[T2_SELFDECL] declared=%s" % (_decls or "(none — no-op)",), file=_sys.stderr, flush=True)
            _called = _called_tools(state.messages)
            for _d in _decls:
                _op = str(_d.get("operand", "")).strip()
                if not _d.get("claimed") or str(_d.get("source", "")).upper() != "INFER":
                    continue
                _pr = _ao.get(_op)
                if _pr and _pr not in _called:
                    self._t2_selfdecl = getattr(self, "_t2_selfdecl", 0) + 1
                    print("[T2_SELFDECL] fired operand=%s producer=%s" % (_op, _pr),
                          file=_sys.stderr, flush=True)
                    _new = _ap_regen(SELFDECL_FEEDBACK.format(operand=_op, producer=_pr), "selfdecl")
                    if _new is not None:
                        am = _new
                        print("[T2_SELFDECL] regen tool_calls=%s"
                              % ([getattr(t, "name", None) for t in (getattr(am, "tool_calls", None) or [])],),
                              file=_sys.stderr, flush=True)
                    break

        # ─── ★P13 (2026-08-02·승인): CHANNEL을 **예방형 생성-레벨**로 이설 ───
        #   구판은 unlock/give/call 출력에 [axis] 노트를 붙였는데 셋 다 env mutating(실측) ⇒ replay 파괴
        #   (041 R0 6,579s 폐기) → P12 가드에 걸려 100% 드롭. 여기서는 **호출이 실행되기 전에** 채널
        #   오분류를 잡아 피드백 1회 + regen(비커밋·replay 무영향·ARG_SCHEMA/WAG와 동형).
        #   판정 = 레지스트리 멤버십(env 기계-도출·opex 0·C208②)뿐 — 값 판단 0·도메인 리터럴 0.
        if os.environ.get("T2_TOOL_CHANNEL") == "1" and getattr(am, "tool_calls", None):
            try:
                import t2_axis_levers as _AX13
                _o13 = getattr(self, "_t2_orch", None)
                _nc13 = (a2 or {}).get("axis_notes") or {}
                if _o13 is not None and _nc13:
                    _sc13, _ad13, _ud13 = _AX13.registry_from_a2(a2)
                    _agd13, _usd13 = _AX13.registry_from_env(_o13)
                    _agd13 |= _ad13
                    _usd13 |= _ud13
                    _unl13 = getattr(_o13, "_t2_axis_unlocked", None) or set()
                    if _agd13 or _usd13:
                        for _tc13 in (am.tool_calls or []):
                            _n13 = _AX13.channel_note(
                                str(getattr(_tc13, "name", "") or ""), _args_dict(_tc13),
                                _sc13, _agd13, _usd13, _unl13, _nc13)
                            if not _n13:
                                continue
                            from t2_lever_beat import beat as _beat13
                            _beat13("T2_TOOL_CHANNEL", "channel_pre")
                            print("[T2_TOOL_CHANNEL] pre-call regen: %s"
                                  % getattr(_tc13, "name", ""), file=_sys.stderr, flush=True)
                            _new13 = _ap_regen("Error: [TOOL-CHANNEL] " + _n13, "channel")
                            if _new13 is not None:
                                am = _new13
                            break
            except Exception as _e13:
                print("[T2_TOOL_CHANNEL] pre-call check skipped: %r" % (_e13,),
                      file=_sys.stderr, flush=True)


        # ─── ★T2_CHOICE_GROUND (2026-08-05·계좌개설 계열 실측): 회수하지 않은 이름을 고른다 ───
        #   `open_bank_account`류는 상품 이름을 열린 문자열로 받는다(env가 열거하지 않는다). 실측:
        #   gold 요구 52건 중 47건 실패이고 그 절반이 **오선택**이며, 선택값이 회수 문서 어디에도
        #   없는 경우가 7건 있었다(x84 — 지어낸 이름). 같은 계량이 **gold도 3건 미접지**임을 보여
        #   deny는 오차단이 되므로 **넛지 1회**로만 둔다. 술어=선언된 (도구,인자)의 값이 이 대화가
        #   회수한 도구 출력에 축자로 있는가(포함 검사·추출 0).
        if (os.environ.get("T2_CHOICE_GROUND") == "1" and getattr(am, "tool_calls", None)
                and not getattr(self, "_t2_choiceground", False)):
            try:
                _cg = (a2 or {}).get("choice_grounding") or []
                if _cg:
                    _seen_cg = " ".join(_content_str(m) or "" for m in (state.messages or [])
                                        if getattr(m, "role", None) == "tool")
                    for _tc_cg in (am.tool_calls or []):
                        _nm_cg = _exact_tool_name(_tc_cg)
                        _ar_cg = _args_dict(_tc_cg)
                        for _spec_cg in _cg:
                            if _spec_cg.get("tool") != _nm_cg:
                                continue
                            _v_cg = str(_ar_cg.get(_spec_cg.get("arg")) or "").strip()
                            if not _v_cg or _v_cg in _seen_cg:
                                continue
                            self._t2_choiceground = True
                            print("[T2_CHOICE_GROUND] regen: %s=%r not in retrieved text"
                                  % (_spec_cg.get("arg"), _v_cg), file=_sys.stderr, flush=True)
                            _fb_cg = str(_spec_cg.get("feedback") or "").replace("{value}", _v_cg)
                            _new_cg = _ap_regen(_fb_cg, "choiceground")
                            if _new_cg is not None:
                                am = _new_cg
                            break
                        if getattr(self, "_t2_choiceground", False):
                            break
            except Exception as _ecg:
                print("[T2_CHOICE_GROUND] skipped: %r" % (_ecg,), file=_sys.stderr, flush=True)

        # ─── ★T2_UNINSTRUCTABLE (2026-08-05·012 실측): 실행할 수 없는 지시 ───
        #   손님에게 도구 실행을 안내했는데 **아직 아무 도구도 전달되지 않은** 상태 =
        #   손님은 그 지시를 실행할 수 없다. 012는 그 위에 존재하지 않는 도구 이름과 앱 경로를
        #   지어냈다(코퍼스 grep 0건). 술어 = A2 선언 토큰의 포함 ∧ 전달-이력 부재(정규식 추출 0).
        #   도구 호출이 없는 **산문 턴**에서 나므로 출력-부착이 아니라 생성-레벨이어야 한다.
        if (os.environ.get("T2_UNINSTRUCTABLE") == "1" and not getattr(am, "tool_calls", None)
                and not getattr(self, "_t2_uninst_done", False)):
            try:
                _ax_u = (a2 or {}).get("axis_notes") or {}
                _toks = _ax_u.get("user_exec_tokens") or []
                _mark = str(_ax_u.get("given_marker") or "")
                _fbu = _ax_u.get("uninstructable")
                _said = str(getattr(am, "content", "") or "")
                if _fbu and _toks and _mark and any(t in _said for t in _toks):
                    _given = any(_mark in (_content_str(m) or "")
                                 for m in (state.messages or [])
                                 if getattr(m, "role", None) == "tool")
                    if not _given:
                        self._t2_uninst_done = True
                        from t2_lever_beat import beat as _beatu
                        _beatu("T2_UNINSTRUCTABLE", "uninstructable")
                        print("[T2_UNINSTRUCTABLE] regen: instruction with nothing given",
                              file=_sys.stderr, flush=True)
                        _newu = _ap_regen(_fbu, "uninstructable")
                        if _newu is not None:
                            am = _newu
            except Exception as _eu:
                print("[T2_UNINSTRUCTABLE] skipped: %r" % (_eu,), file=_sys.stderr, flush=True)

        # ─── ★P5 (2026-08-02): user-tool 안내 표준문 — 생성-레벨(P13 규약) ───
        #   018/040 실측: 대화-내 user-tool 실행을 "portal/app 제출"로 오설명 → 손님 2회 거부 → 이관 →
        #   gold write 0. give는 env mutating이라 출력-부착 불가(P13) ⇒ **give 호출 직전** 1회 표면화.
        #   표면화만·설득 금지([[21]] 흡수 지점 = 오설명 제거). sim당 1회(예산·중복 억제).
        if (os.environ.get("T2_USER_TOOL_NOTE") == "1" and getattr(am, "tool_calls", None)
                and not getattr(self, "_t2_utn_done", False)):
            try:
                _tpl5 = ((a2 or {}).get("axis_notes") or {}).get("user_tool_channel")
                _giv5 = next((t for t in (am.tool_calls or [])
                              if str(getattr(t, "name", "")) == "give_discoverable_user_tool"), None)
                if _tpl5 and _giv5 is not None:
                    _want5 = str(_args_dict(_giv5).get("discoverable_tool_name") or "").strip()
                    if _want5:
                        self._t2_utn_done = True
                        from t2_lever_beat import beat as _beat5
                        _beat5("T2_USER_TOOL_NOTE", "usertool_note")
                        print("[T2_USER_TOOL_NOTE] pre-give note: %s" % _want5,
                              file=_sys.stderr, flush=True)
                        _new5 = _ap_regen("Note: " + _tpl5.format(tool=_want5), "usertoolnote")
                        if _new5 is not None:
                            am = _new5
            except Exception as _e5:
                print("[T2_USER_TOOL_NOTE] skipped: %r" % (_e5,), file=_sys.stderr, flush=True)

        # ─── ★G2 인계 술어 `T2_HANDOFF_PREDICATE` (2026-08-18·기본 OFF) ───
        #   술어: **손님-측 discoverable 도구 이름을 발화했는데 아직 `give` 를 안 했다**.
        #   출처는 gold 가 아니라 **환경의 도구 계약**이다 — env 가 `KnowledgeUserTools` 의
        #   `__discoverable__` 집합을 런타임에 선언하고("에이전트가 건네야 손님이 부른다"),
        #   도구 독스트링도 축자로 같은 말을 한다([[23]]·사용자 지적 2026-08-18).
        #   실측(`x368`·최근 439 sim): 손님-측 이름을 발화한 sim **82** 중 give 0 이 **51**(62%)이고
        #   그 sim 들의 pass 는 **2/51 = 4%**(전체 base ≈25%).
        #   ★왜 여기인가: 같은 일을 하는 기존 레버(`T2_GIVE_EXEC_NUDGE`·`T2_UNCALLED_UNLOCK`)는
        #     `_resign` 창에서만 발화한다 = **에이전트가 포기하려 할 때**. 위반은 훨씬 이른
        #     *이름을 말한 턴*에 생긴다 ⇒ 이 레버는 **타이밍만** 옮긴다(부하 축소·⛔0 허용 형태).
        #   ⚠엔진은 **이름 집합 대조**만 한다 — 어느 도구가 옳은지·언제 건네야 하는지는 판단하지
        #     않고, 도구를 **대신 부르지도 않는다**(등대 §1.5 write 강제 금지·[[06]] 게이트 금지 축).
        #   ⚠[[64]]: 무엇이 빠졌는지와 **무엇을 하면 풀리는지**를 같이 말한다.
        #   ⚠집합이 비면(다른 도메인) 침묵 · sim 당 상한으로 잔소리 루프를 막는다.
        if (os.environ.get("T2_HANDOFF_PREDICATE") == "1" and a2 is not None
                and getattr(self, "_t2_hop_n", 0) < int(os.environ.get("T2_HANDOFF_CAP") or 2)):
            try:
                _tplh = ((a2 or {}).get("axis_notes") or {}).get("handoff_missing")
                _envh = getattr(getattr(self, "_t2_orch", None), "environment", None)
                _nowgive = any(str(getattr(t, "name", "")) == "give_discoverable_user_tool"
                               for t in (getattr(am, "tool_calls", None) or []))
                import t2_search as _tsh
                _missh = ([] if (_nowgive or not _tplh) else
                          _tsh.handoff_missing(_envh, state.messages, _content_str(am),
                                               _content_str, _args_dict))
                if _missh:
                    self._t2_hop_n = getattr(self, "_t2_hop_n", 0) + 1
                    from t2_lever_beat import beat as _beath
                    _beath("T2_HANDOFF_PREDICATE", "handoff_missing")
                    print("[T2_HANDOFF] named-but-not-given: %s" % ",".join(_missh),
                          file=_sys.stderr, flush=True)
                    # ⚠엔진은 **고르지 않는다**: 모델이 방금 발화한 이름을 그대로 되읽어 주고,
                    #   여럿이면 **전부** 적는다(하나를 고르면 그 순간 엔진이 판단한 것이 된다).
                    _newh = _ap_regen(_tplh.format(tool=", ".join(_missh)), "handoffpred")
                    if _newh is not None:
                        am = _newh
            except Exception as _eh:
                print("[T2_HANDOFF] skipped: %r" % (_eh,), file=_sys.stderr, flush=True)

        # ─── ★P2/P10 (2026-08-03·AX32 설계서 §P2·§P10 — alltools 재설계판) ───
        #   r7의 전제("alltools 전환으로 bm25 신호 무효")는 **실측으로 기각**: alltools는
        #   KB_search_bm25 + KB_search_dense + shell을 함께 노출하고, bm25는 그대로 `Score:`를
        #   찍는다(무의미 질의 → 전부 0.0000·env 프로브 2026-08-03). ⇒ 원 신호가 살아 있다.
        #   · P2 = **k회 연속 전-0점** → "없는 절차를 지어내지 말라·이관 검토" 노트 1회(강제 아님).
        #   · P10 = 그 무득점 질의가 **손님 발화의 축자 부분열**을 담고 있으면(=손님 주장을 그대로
        #     찾아본 것인데 KB가 뒷받침하지 않음) 확인-전 수용 금지 노트 1회. 술어 = P1과 **동일**
        #     `_shared_span`(닫힘·재사용).
        #   채널 = **생성-레벨**(P13 규약: KB_search는 env mutating이라 출력-부착 금지·041 사고).
        #   dense는 문턱이 필요해 신호로 쓰지 않는다([[05]] 회색지대 회피) → **과소 발화** 방향.
        if (os.environ.get("T2_KB_NOHIT_SURFACE") == "1"
                and not getattr(self, "_t2_nohit_done", False)):
            try:
                _an2 = (a2 or {}).get("axis_notes") or {}
                _z, _zq = 0, []
                _id2n2 = {}
                for _m2 in state.messages:
                    for _t2 in (getattr(_m2, "tool_calls", None) or []):
                        _id2n2[getattr(_t2, "id", None)] = _args_dict(_t2)
                    if getattr(_m2, "role", None) != "tool":
                        continue
                    _hit2 = _kb_zero_hit(getattr(_m2, "content", None))
                    if _hit2 is None:
                        continue                       # 점수 없는 채널 = 판정 대상 아님
                    if _hit2:
                        _z += 1
                        _qa = _id2n2.get(getattr(_m2, "id", None)) or {}
                        _zq.append(str(_qa.get("query") or ""))
                    else:
                        _z = 0                          # 득점하면 연속 카운트 리셋
                        _zq = []
                _th2 = int(os.environ.get("T2_KB_NOHIT_K", "2") or 2)
                if _z >= _th2:
                    _ut2 = " ".join(str(getattr(m, "content", "") or "")
                                    for m in state.messages
                                    if getattr(m, "role", None) == "user")
                    _min2 = int(_an2.get("give_quote_min_tokens") or 4)
                    _claim = next((q for q in _zq if _shared_span(q, _ut2, _min2)), None)
                    _tpl2 = (_an2.get("kb_claim_nohit") if _claim else None) \
                        or _an2.get("kb_nohit")
                    if _tpl2:
                        self._t2_nohit_done = True
                        from t2_lever_beat import beat as _beat2
                        _beat2("T2_KB_NOHIT_SURFACE", "claim" if _claim else "nohit")
                        # ★관측 병기(2026-08-03·사용자 지적 "다른 채널로 피해가는 것 아닌가"):
                        #   dense는 항상 양수 유사도라 끼는 즉시 streak가 리셋된다(=과소 발화·안전).
                        #   그러나 **shell(grep)은 점수 행이 없어 리셋하지 않는다** = 알려진 구멍 —
                        #   무득점 구간에 shell이 있었는지를 함께 찍어 아침 포렌식이 이 발화를
                        #   "정당/오발화"로 귀속할 수 있게 한다(조정의 입력·[[19]]).
                        _sh = sum(1 for _m3 in state.messages
                                  for _t3 in (getattr(_m3, "tool_calls", None) or [])
                                  if "shell" in str(getattr(_t3, "name", "")).lower())
                        print("[T2_KB_NOHIT_SURFACE] zero-score streak=%d claim_span=%s shell_calls=%d"
                              % (_z, bool(_claim), _sh), file=_sys.stderr, flush=True)
                        _new2p = _ap_regen(_tpl2.format(n=_z, query=(_claim or _zq[-1])[:80]),
                                           "kbnohit")
                        if _new2p is not None:
                            am = _new2p
            except Exception as _e2p:
                print("[T2_KB_NOHIT_SURFACE] skipped: %r" % (_e2p,), file=_sys.stderr, flush=True)

        # ─── ★P1 (2026-08-03·AX32 설계서 §P1): give-인용 표면화 — 생성-레벨 ───
        #   010 실측(재현 2/2): 양 pass의 **유일 dbdiff**가 여분 give(`get_referral_link GIVEN`) —
        #   손님이 요청하지 않은 도구를 건네 태스크가 죽었다. 레버 = give 직전, 응답 본문에 손님의
        #   말이 **축자로** 실재하는지만 본다(닫힌 술어·[[22]]) → 불성립 시 재질의 1회(fail-open).
        #   ☠채널 제약(재제안 방지): 인용을 **give 인자에 얹는 것 금지**(여분 키 = evaluator
        #   exact-match 파괴 실측·T2_ARG_SCHEMA와 정면 충돌) ⇒ 생성-레벨 선언만.
        #   ★사전등록 지표: "인용-불성립 후 give 철회율" — regen 후 give가 사라졌는지 로그로 계수
        #   (≈0이면 접는다·C263 공집합 NO-GO 동형). sim당 1회(예산).
        if (os.environ.get("T2_GIVE_QUOTE") == "1" and getattr(am, "tool_calls", None)
                and not getattr(self, "_t2_gq_done", False)):
            try:
                _an1 = (a2 or {}).get("axis_notes") or {}
                _tpl1 = _an1.get("give_quote")
                _giv1 = next((t for t in (am.tool_calls or [])
                              if str(getattr(t, "name", "")) == "give_discoverable_user_tool"), None)
                if _tpl1 and _giv1 is not None:
                    _min1 = int(_an1.get("give_quote_min_tokens") or 4)
                    _utext1 = " ".join(str(getattr(m, "content", "") or "")
                                       for m in state.messages
                                       if getattr(m, "role", None) == "user")
                    if not _shared_span(getattr(am, "content", "") or "", _utext1, _min1):
                        self._t2_gq_done = True
                        _want1 = str(_args_dict(_giv1).get("discoverable_tool_name") or "").strip()
                        from t2_lever_beat import beat as _beat1
                        _beat1("T2_GIVE_QUOTE", "give_quote")
                        print("[T2_GIVE_QUOTE] no verbatim customer span in message before give=%s"
                              % _want1, file=_sys.stderr, flush=True)
                        _new1p = _ap_regen(_tpl1.format(tool=_want1 or "this tool", min=_min1),
                                           "givequote")
                        if _new1p is not None:
                            am = _new1p
                        # 사전등록 지표: 재질의 후 같은 give가 남았는가(철회=사라짐)
                        _still = any(str(getattr(t, "name", "")) == "give_discoverable_user_tool"
                                     for t in (getattr(am, "tool_calls", None) or []))
                        print("[T2_GIVE_QUOTE] retract=%d (give_present_after_reask=%d)"
                              % (0 if _still else 1, 1 if _still else 0),
                              file=_sys.stderr, flush=True)
            except Exception as _e1p:
                print("[T2_GIVE_QUOTE] skipped: %r" % (_e1p,), file=_sys.stderr, flush=True)

        # ─── ★P15 (2026-08-03·task_004 실측): 이관 사유 **티어** 표면화 ───
        #   정책 문서(banking doc_..._042) 축자: 사유 코드는 4단 티어이고
        #   *"always select from the highest tier that applies"*. 004 실측: 손님 식별자로 조회했는데
        #   레코드 0(=신원 확인 실패)인데도 TIER3 사유로 이관 → gold는 TIER1. db_check는 통과했고
        #   **이 사유 하나로 0점**.
        #   ★[[22]] 경계: "어떤 상황인가"는 열림(모델 몫). 엔진이 판정하는 것은 **닫힌 둘뿐** —
        #     ⑴원장에 A2-선언 마커가 실재하는가(부분문자열) ⑵티어 정수 비교.
        #   ★[[05]]: 도구명·사유코드·티어·마커·문구가 **전부 A2**(`transfer_reason_policy`).
        #     미선언 도메인 = 레버 skip(엔진 도메인 리터럴 0).
        #   강제 아님: 더 높은 티어 후보와 그 문서적 조건을 알리고 재발화 1회(sim당 1회).
        if (os.environ.get("T2_TRANSFER_TIER") == "1" and getattr(am, "tool_calls", None)
                and not getattr(self, "_t2_tier_done", False)):
            try:
                _tp = (a2 or {}).get("transfer_reason_policy") or {}
                _tiers = _tp.get("tiers") or {}
                _apply7 = [str(x) for x in (_tp.get("applies_to") or [])]
                _tc7 = next((t for t in (am.tool_calls or [])
                             if str(getattr(t, "name", "")) in _apply7), None) if _apply7 else None
                if _tiers and _tc7 is not None:
                    _chosen = str(_args_dict(_tc7).get(_tp.get("reason_arg") or "") or "")
                    _ct = int(_tiers.get(_chosen, 99))
                    _outs7 = "\n".join(str(getattr(m, "content", "") or "")
                                       for m in state.messages
                                       if getattr(m, "role", None) == "tool")
                    for _sg7 in (_tp.get("signals") or []):
                        _code = str(_sg7.get("code") or "")
                        _kt = int(_tiers.get(_code, 99))
                        if _kt >= _ct:
                            continue                     # 더 높은 티어일 때만 말한다
                        _ev = next((mk for mk in (_sg7.get("ledger_contains") or [])
                                    if str(mk) in _outs7), None)
                        if not _ev:
                            continue
                        self._t2_tier_done = True
                        from t2_lever_beat import beat as _beat7
                        _beat7("T2_TRANSFER_TIER", _code)
                        print("[T2_TRANSFER_TIER] chosen=%s(tier %s) -> higher applicable=%s(tier %s) "
                              "evidence=%r" % (_chosen, _ct, _code, _kt, _ev[:40]),
                              file=_sys.stderr, flush=True)
                        _new7 = _ap_regen((_tp.get("note") or "").format(
                            chosen=_chosen, chosen_tier=_ct, code=_code, code_tier=_kt,
                            why=_sg7.get("why", ""), evidence=_ev), "transfertier")
                        if _new7 is not None:
                            am = _new7
                        break
            except Exception as _e7:
                print("[T2_TRANSFER_TIER] skipped: %r" % (_e7,), file=_sys.stderr, flush=True)

        # ─── ★P11 (2026-08-02): ARG-SCHEMA 위생을 unified 경로로 이설 ───
        #   死코드 사고: 이 검사는 `patched()`(apply_provenance_regen) 안에만 있었는데 라이브 러너는
        #   `_unified`(T2_GATE_REGEN ∧ ground2) 조건에서 `apply_unified_regen`만 호출한다 ⇒
        #   `T2_ARG_SCHEMA=1`이 go_stack에 켜져 있어도 **실행되지 않았다**(x43 설치-경로 감사로 확정·
        #   WEV가 겪은 것과 동형). 표적은 0건(040의 give는 스키마-합법이라 원래 무발화) = **위생 수정**.
        #   검사 = 자기 도구 스키마(properties) 밖 최상위 키 → 피드백 1회 + regen. 값 판단 0·도메인 리터럴 0.
        if os.environ.get("T2_ARG_SCHEMA") == "1" and getattr(am, "tool_calls", None):
            if not hasattr(self, "_t2_schema_props"):
                _props = {}
                for _t in (self.tools or []):
                    try:
                        _sc = _t.openai_schema
                        _fn = _sc.get("function") if isinstance(_sc.get("function"), dict) else _sc
                        _nm2 = _fn.get("name")
                        _pr2 = ((_fn.get("parameters") or {}).get("properties")) or {}
                        if _nm2 and _pr2:
                            _props[_nm2] = set(_pr2.keys())
                    except Exception:
                        pass
                self._t2_schema_props = _props
            _tries = 0
            while _tries < 2 and getattr(am, "tool_calls", None):
                _bad = None
                for _tc2 in (am.tool_calls or []):
                    _allowed = self._t2_schema_props.get(getattr(_tc2, "name", None))
                    if not _allowed:
                        continue
                    _extra = [k for k in _args_dict(_tc2).keys() if k not in _allowed]
                    if _extra:
                        _bad = (_tc2, _extra, _allowed)
                        break
                if _bad is None:
                    break
                _tc2, _extra, _allowed = _bad
                _tries += 1
                from t2_lever_beat import beat as _beat_as
                _beat_as("T2_ARG_SCHEMA", "argschema")
                print("[T2_ARGSCHEMA] regen tool=%s extra=%s" % (_tc2.name, _extra),
                      file=_sys.stderr, flush=True)
                _new2 = _ap_regen(
                    "Error: [ARG-SCHEMA] '%s' does not accept argument(s): %s. Its schema declares ONLY "
                    "these argument(s): %s. Re-issue the call with ONLY declared arguments — remove "
                    "everything else." % (_tc2.name, ", ".join(repr(x) for x in _extra),
                                          ", ".join(sorted(_allowed))), "argschema")
                if _new2 is None:
                    break
                am = _new2

        # ─── ★V7 최종-발화 서명 재검 (2026-08-03·선점 실증 후 [[19]] 조정) ───
        #   위 블록(4763~)의 V7은 피드백 사슬의 **맨 끝**이라 gate/prov/wev가 그 턴을 잡으면
        #   발화하지 못한다. 2026-07-31 계측이 "선점이 유력"이라 적어뒀고, 2026-08-03 스모크가
        #   그것을 **확정**했다: 015에서 `[T2_TOOL_SIGNATURE] would-deny … preempted-by=prov`
        #   → 같은 호출이 `arguments`를 실은 채 커밋 → gold give 불일치(PRED_EXTRA_KEY)로 0점.
        #   ⇒ 여기(모든 regen이 끝난 **최종 메시지**)서 한 번 더 본다. 선점 불가·순수 인자-형태
        #   검사(도메인 사실은 A2 `tool_signatures`)·엔진은 여전히 인자를 떼지 않는다(C151).
        if os.environ.get("T2_TOOL_SIGNATURE") == "1" and getattr(am, "tool_calls", None):
            try:
                import t2_signature as _sgf
                _sf_tries = 0
                while _sf_tries < 2 and getattr(am, "tool_calls", None):
                    _hit = None
                    for _c9 in (am.tool_calls or []):
                        _v9 = _sgf.signature_violation(getattr(_c9, "name", None),
                                                       _args_dict(_c9), a2, force=True)
                        if _v9:
                            _hit = (_c9, _v9)
                            break
                    if _hit is None:
                        break
                    _sf_tries += 1
                    self._t2_signature_deny = getattr(self, "_t2_signature_deny", 0) + 1
                    from t2_lever_beat import beat as _beat9
                    _beat9("T2_TOOL_SIGNATURE", "final")
                    print("[T2_TOOL_SIGNATURE] final-word deny tool=%s (try %d)"
                          % (getattr(_hit[0], "name", None), _sf_tries),
                          file=_sys.stderr, flush=True)
                    _new9 = _ap_regen("Error: " + _hit[1], "signature")
                    if _new9 is None:
                        break
                    am = _new9
            except Exception as _e9:
                print("[T2_TOOL_SIGNATURE] final-word skipped: %r" % (_e9,),
                      file=_sys.stderr, flush=True)
        return am

    LLMAgent._generate_next_message = unified
    # (3) exec-side: auth observe + nested/calc 읽기증강 (게이트 경로와 동일·직교)
    _install_regen_exec()
    # (4) 컨텍스트 초과 우아한 종료 (023 진단·§2ah): overflow를 sim-무효 대신 scored 실패로.
    _install_overflow_guard()
    # (5) ★LLM CWE 발생원 로거 (§2bd·로그 전용): rall4 095t0 CWE가 overflow 가드(step-래핑)를
    #   우회해 러너까지 전파(4번째 우회 경로) — 어느 generate(call_name)서 새는지 특정.
    #   래핑=la.generate 모듈 함수·예외는 그대로 re-raise(행동 무변경).
    if not getattr(la, "_t2_llmdiag", False):
        _og_gen = la.generate

        def _gen_diag(*_a, **_kw):
            try:
                return _og_gen(*_a, **_kw)
            except Exception as _e:
                if "ContextWindow" in type(_e).__name__:
                    print("[T2_LLM_DIAG] CWE escaped at call_name=%s"
                          % _kw.get("call_name"), file=_sys.stderr, flush=True)
                raise

        la.generate = _gen_diag
        la._t2_llmdiag = True
    return unified


if __name__ == "__main__":
    apply()
    print("[t2_gate_patch] applied")
