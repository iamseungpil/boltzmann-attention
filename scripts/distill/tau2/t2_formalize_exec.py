#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""t2_formalize_exec.py — FORMALIZE-EXEC (레버3·NEXT_LEVER_GEN_DESIGN §2·2026-07-11).

E1′형 격리 직렬화(3단): ①LLM 격리 서브콜이 선택-기준을 도메인-일반 JSON으로 *형식화*
  {"op": argmax|argmin|filter, "field": ..., "constraints": [{"field":..,"value":..}, ...]}
  (+ "none"=기준 없음→P-B DISAMB 폴백 / "unresolvable"=기준은 있으나 실행 재료 부재→ASK-유도)
②결정론 실행기가 에이전트-*기조회* 후보 record 위에서 계산(compute_facts op 커널 동형·DB 접근 0)
③채널 = **비커밋 후보-주석**(1차): DISAMB 서브콜(P-B 좌석)이 변형-선택 인자서 fire할 때
  그 서브콜 *프롬프트*에 실행기 결과를 주석으로 첨부 — 대화/턴/히스토리 완전 불변.
  (silent 치환 채널은 V0 A/B 후 판단 — §2.1③.)

[[05]]/[[10]]: op 어휘·실행기·JSON 검증 = 엔진(도메인일반·필드 리터럴 0) /
  fire 지점 = 기존 A2 도출(confirm-write·disamb_sub_args) 재사용 / NL→formalize=LLM·
  concrete 계산=결정론([[10]] 분담 그대로). write 생성 0·조회 0(기조회 데이터만).
전 예외 = no-op(주석 없이 기존 DISAMB 그대로). toggle `T2_FEXEC=1`·마커 `[T2_FEXEC]`.

★V0 게이트(설계 §2.6·불통과=스택 미편입): 형식화 정확도 EM 선측정(fexec_iso_probe.py)
  통과 전 T2_FEXEC 기본 off — 이 모듈 존재 자체는 스택 불변.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 헬퍼 재사용(중복 구현 금지): 후보 도출·record 탐색·전사 필터(N5/N11)는 t2_gate_patch 정본.
from t2_gate_patch import (  # noqa: E402
    _grounded_candidates, _parse_tool_outputs, _min_enclosing_record, _text_transcript)

_MISSING = object()
OPS = ("argmax", "argmin", "filter", "none", "unresolvable")
CONS_OPS = ("eq", "ne", "le", "ge", "lt", "gt")   # 제약 비교자(도메인일반 상수·기본 eq)


def _mark(msg):
    print("[T2_FEXEC] %s" % msg, file=sys.stderr, flush=True)


# ═════════════════ ① 형식화 서브콜 프롬프트 (도메인-일반 어휘) ═════════════════
FORMALIZE_SYS = (
    "You are formalizing the SELECTION CRITERION for ONE ambiguous tool-call argument in a "
    "customer-service conversation. Read the transcript and the candidate records, then express "
    "the criterion the user stated as ONE JSON object with EXACTLY this schema:\n"
    '{"op": "argmax" | "argmin" | "filter" | "none" | "unresolvable",\n'
    ' "field": "<record field the criterion ranks (argmax/argmin) — null for filter/none>",\n'
    ' "constraints": [{"field": "<record field>", "op": "eq|ne|le|ge|lt|gt", "value": <required value>}, ...]}\n'
    "- argmax/argmin: the user asks for the candidate with the LARGEST/SMALLEST value of 'field' "
    "(e.g. most expensive, cheapest, biggest) among candidates satisfying the constraints.\n"
    "- filter: the criterion only restricts attributes (constraints), no ranking.\n"
    "- none: the user stated no formalizable criterion (a pure pick among candidates).\n"
    "- unresolvable: the user DID state a criterion, but the data it needs does not appear in "
    "the candidate records at all.\n"
    "Use ONLY field names that actually appear in the candidate records, and constraint values "
    "taken from the conversation ('op' defaults to 'eq' and may be omitted). "
    "When the user's words refer to a record attribute, write the constraint value EXACTLY as it "
    "is spelled in the candidate records (verbatim, not a paraphrase). "
    "If the user named SEVERAL things that must all be in the same record (e.g. several items in "
    "one order), emit one constraint per thing. "
    "Output the JSON object and NOTHING else."
)


def parse_formalize(txt):
    """서브콜 응답 → 검증된 spec dict | None(UNSURE). 형식 위반/파싱 실패 = None(no-op)."""
    if not isinstance(txt, str):
        return None
    i = txt.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(txt[i:])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    op = str(obj.get("op") or "").strip().lower()
    if op not in OPS:
        return None
    field = obj.get("field")
    if field is not None and not isinstance(field, str):
        return None
    cons_in = obj.get("constraints")
    cons = []
    if cons_in is not None:
        if not isinstance(cons_in, list):
            return None
        for c in cons_in:
            if not isinstance(c, dict) or not isinstance(c.get("field"), str) \
                    or "value" not in c:
                return None
            cop = str(c.get("op") or "eq").strip().lower()
            if cop not in CONS_OPS:
                return None
            cons.append({"field": c["field"], "op": cop, "value": c["value"]})
    if op in ("argmax", "argmin") and not field:
        return None
    return {"op": op, "field": field, "constraints": cons}


# ═════════════════ ② 결정론 실행기 (기조회 record 위·DB 접근 0) ═════════════════
def _field_lookup(rec, field):
    """단일 값 필드 탐색(argmax/argmin rank용): 점-경로/단일-세그먼트 겸용·첫 값. 없으면 _MISSING."""
    vals = _field_values(rec, field)
    return vals[0] if vals else _MISSING


def _leaf_scalars(node):
    """node 아래 스칼라 leaf 전수(dict/list 재귀·containment any-match 재료)."""
    if isinstance(node, list):
        out = []
        for x in node:
            out.extend(_leaf_scalars(x))
        return out
    if isinstance(node, dict):
        return []          # dict-값 자체는 스칼라 아님(하위는 _field_values_path가 세그먼트로 진입)
    return [node]


def _field_values_path(node, segs):
    """점-경로 세그먼트 순차 진입(각 레벨 dict=key 매칭·list=flatMap). 마지막 세그먼트 값들.
    ★32B가 제약 필드를 'items.name'류 점-경로로 형식화(라이브 프로브 확증·2026-07-12) → 지원."""
    if not segs:
        return _leaf_scalars(node)
    seg = segs[0].strip().lower()
    out = []
    if isinstance(node, list):
        for x in node:
            out.extend(_field_values_path(x, segs))
    elif isinstance(node, dict):
        for kk, vv in node.items():
            if str(kk).strip().lower() == seg:
                out.extend(_field_values_path(vv, segs[1:]))
    return out


def _field_values(rec, field):
    """필드 → record 내 일치하는 *모든* 스칼라 값. 없으면 [].
    ★order-필터(2026-07-12 HANDOFF §6.1): 후보=order record면 같은 필드(name 등)가 items
    하위에 반복 — 'record가 X를 담는가'(containment)=any-match라 전수 수집. 두 형식 지원:
      · 점-경로('items.name') = 세그먼트 순차 진입(라이브 프로브 확증·32B 기본 형식)
      · 단일-세그먼트('name'·'status') = 재귀 BFS any-match(중첩 깊이 무관·폴백 관대)."""
    if not isinstance(field, str) or not field.strip():
        return []
    field = field.strip()
    if "." in field:
        vals = _field_values_path(rec, field.split("."))
        if vals:
            return [v for v in vals if not isinstance(v, (dict, list))]
        # 점-경로가 root서 안 걸리면 마지막 세그먼트로 재귀 폴백(관대·구조 변형 흡수)
        field = field.split(".")[-1]
    fl = field.lower()
    vals, queue = [], [rec]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            for kk, vv in cur.items():
                if str(kk).strip().lower() == fl:
                    if isinstance(vv, list):
                        vals.extend(x for x in vv if not isinstance(x, (dict, list)))
                    elif not isinstance(vv, dict):
                        vals.append(vv)
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(x for x in cur if isinstance(x, (dict, list)))
    return vals


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cons_match(val, cons):
    """제약 1건 판정(도메인일반 값 비교: 숫자 우선·불리언·문자열 case-insensitive)."""
    want, cop = cons["value"], cons.get("op", "eq")
    fv, fw = _as_float(val), _as_float(want)
    if cop in ("le", "ge", "lt", "gt"):
        if fv is None or fw is None:
            return False
        return {"le": fv <= fw, "ge": fv >= fw, "lt": fv < fw, "gt": fv > fw}[cop]
    if isinstance(val, bool) or isinstance(want, bool):
        tv = {True: True, False: False, "true": True, "false": False}
        eq = tv.get(val if isinstance(val, bool) else str(val).strip().lower()) == \
            tv.get(want if isinstance(want, bool) else str(want).strip().lower())
    elif fv is not None and fw is not None:
        eq = fv == fw
    else:
        eq = str(val).strip().lower() == str(want).strip().lower()
    return eq if cop == "eq" else not eq


def execute_formalized(spec, records):
    """spec(검증됨) × records=[(cand_id, record dict|None)] → 판정 dict:
    {"status": "ok"|"unresolvable"|"empty", "ids": [...], "why": str}.
    - 제약 field가 record *전부*에 부재 = unresolvable(재료 부재 판정 — t71형=ASK 위계).
    - argmax/argmin rank field 전부 부재/비수치 = unresolvable.
    - 동률 = 전부 반환(단일 아님=치환류 금지·주석/DISAMB 폴백 몫).
    """
    op = spec["op"]
    if op in ("none", "unresolvable"):
        return {"status": op if op == "unresolvable" else "none", "ids": [], "why": "subcall verdict"}
    recs = [(c, r) for c, r in records if isinstance(r, dict)]
    if not recs:
        return {"status": "unresolvable", "ids": [], "why": "no candidate records"}
    # 제약 필터 (필드가 전 record 부재면 unresolvable)
    # ★any-match(2026-07-12 order-필터 확장): 필드가 record에 여러 번 등장(order.items[].name 등)하면
    #   "어느 한 값이라도 만족 = 통과"(containment 정의). 단일-등장 필드는 종전과 동일 판정.
    #   ne만 쌍대: *어느* 값도 want와 같지 않아야 통과 (any-match의 not-exists).
    for cons in spec["constraints"]:
        vals = [(c, _field_values(r, cons["field"])) for c, r in recs]
        if all(not v for _, v in vals):
            return {"status": "unresolvable", "ids": [],
                    "why": "constraint field '%s' absent from every candidate record" % cons["field"]}
        if cons.get("op") == "ne":
            eqc = {"field": cons["field"], "op": "eq", "value": cons["value"]}
            keep = {c for (c, v) in vals if v and not any(_cons_match(x, eqc) for x in v)}
        else:
            keep = {c for (c, v) in vals if any(_cons_match(x, cons) for x in v)}
        recs = [(c, r) for c, r in recs if c in keep]
        if not recs:
            return {"status": "empty", "ids": [],
                    "why": "no candidate satisfies constraint %s" % json.dumps(cons, default=str)}
    if op == "filter":
        return {"status": "ok", "ids": [c for c, _ in recs], "why": "constraints satisfied"}
    ranked = []
    for c, r in recs:
        f = _as_float(_field_lookup(r, spec["field"]))
        if f is not None:
            ranked.append((f, c))
    if not ranked:
        return {"status": "unresolvable", "ids": [],
                "why": "rank field '%s' absent/non-numeric in every remaining record" % spec["field"]}
    best = (max if op == "argmax" else min)(f for f, _ in ranked)
    return {"status": "ok", "ids": [c for f, c in ranked if f == best],
            "why": "%s(%s)=%s" % (op, spec["field"], best)}


# ═════════════════ ③ 채널 — 비커밋 후보-주석 (DISAMB 서브콜 프롬프트에만) ═════════════════
def fexec_annotation(spec, result, arg_key):
    """실행 결과 → DISAMB 서브콜 프롬프트 첨부용 주석(비커밋·대화 불변). None=주석 없음."""
    if result["status"] == "none":
        return None
    head = "[FORMALIZED CRITERION — deterministic check over the candidate records above]\n"
    crit = json.dumps({"op": spec["op"], "field": spec.get("field"),
                       "constraints": spec.get("constraints") or []}, default=str)
    if result["status"] == "ok":
        return (head + "The user's stated criterion formalizes to %s. Deterministic evaluation over "
                "the retrieved candidate records for '%s' selects: %s (%s)."
                % (crit, arg_key, ", ".join(str(i) for i in result["ids"]), result["why"]))
    if result["status"] == "empty":
        return (head + "The user's stated criterion formalizes to %s, but NO retrieved candidate "
                "satisfies it (%s). Do not force a choice; the right move may be to tell the user."
                % (crit, result["why"]))
    # unresolvable — 재료 부재 판정 = ASK-유도 (§2.2 t71·C48 위계)
    return (head + "The user's stated criterion formalizes to %s, but it CANNOT be evaluated from "
            "the records retrieved so far (%s). Do not guess: the safe answer is UNSURE unless the "
            "conversation itself pins the candidate; the agent should ask the user or read more data."
            % (crit, result["why"]))


def _candidate_record_dicts(arg_key, orig_value, msgs, limit=6):
    """후보값 + *전체* enclosing record(dict) — 실행기 입력(스니펫 절단본 아님).
    원천 = 에이전트 자신이 조회한 tool 출력만(규칙0·_parse_tool_outputs)."""
    cands = _grounded_candidates(arg_key, orig_value, msgs, limit=limit, lenient=True)
    outs = _parse_tool_outputs(msgs, lenient=True)
    recs = []
    for c in cands:
        rec = None
        for out in outs:
            if isinstance(out, (dict, list)):
                r = _min_enclosing_record(out, str(c).strip())
                if r is not None:
                    rec = r
                    break
        recs.append((c, rec))
    return recs


def build_formalize_prompt(state_msgs, arg_key, cur_value, records, limit_chars=6000):
    """격리 형식화 프롬프트(서브콜 전사 필터 = T5C N5/N11 승계 — _text_transcript 재사용)."""
    rec_lines = []
    for c, r in records:
        try:
            rs = json.dumps(r, ensure_ascii=False, default=str)[:600] if r is not None else "(no record)"
        except Exception:
            rs = str(r)[:600]
        rec_lines.append("- %s   | record: %s" % (c, rs))
    return (FORMALIZE_SYS + "\n\n=== Conversation ===\n" + _text_transcript(state_msgs, limit_chars)
            + "\n\n=== Candidate records for '" + str(arg_key) + "' ===\n" + "\n".join(rec_lines)
            + "\n\nThe agent currently chose '" + str(cur_value) + "'. Formalize the user's "
            "selection criterion for this argument as the JSON object.")


# ═════════════ ④ 결정론 filter-substitute 판정 (2026-07-12 HANDOFF §6.1·LOCK §4d) ═════════════
# t71 실증: DISAMB filter-then-ask *지시*(DISAMB_ENUM_FEEDBACK)는 32B가 미준수([[42]] prompt-limit)
# → advise가 아니라 formalize(LLM)→엔진 결정론 필터→치환/ASK 분기 자체를 엔진이 실행.
# 유일 semantic 잔여 = formalize 정확도(오형식화→empty→재형식화→fallback=열거-ASK 자기교정).
def fexec_filter_decide(agent, la, UserMessage, state_msgs, arg_key, cur_value, max_formalize=2):
    """LLM formalize(사용자 전체제약→predicate) → 엔진이 후보 record 위에서 결정론 평가.
    반환 {"status": "one"|"many"|"fallback", "ids": [...], "why": str}:
      one  = 정확히 1 후보 통과 → 호출측이 제자리 치환(whitelist·게이트 재검사는 호출측 관할)
      many = ≥2 통과 → 호출측이 통과분으로 축소해 열거-ASK
      fallback = 판정 불가 → 기존 열거 피드백 그대로 (record<2·후보-record 비귀속·
                 formalize none/unresolvable/파싱실패·empty 재형식화 소진)
    empty(0 통과) = 오형식화 가능성 → 재형식화 1회(§6.1 "0=re-formalize") 후 소진 시 fallback.
    ★결정론 치환의 안전 가드(Δspurious≤0 모트): *모든* 후보가 서로 다른 dict record로 귀속될
    때만 판정 — record 미조회 후보가 있으면 그 후보를 필터할 수 없어 false-unique 위험 = fallback."""
    records = _candidate_record_dicts(arg_key, cur_value, state_msgs)
    recs = [(c, r) for c, r in records if isinstance(r, dict)]
    if len(recs) < 2:
        return {"status": "fallback", "ids": [], "why": "records<2"}
    if len(recs) < len(records):
        _mark("filter fallback: %d/%d candidates lack a record (details not fetched)"
              % (len(records) - len(recs), len(records)))
        return {"status": "fallback", "ids": [], "why": "candidate without record"}
    by_rec = {}
    for c, r in recs:
        by_rec.setdefault(id(r), []).append(c)
    if any(len(v) > 1 for v in by_rec.values()):
        _mark("filter fallback: shared enclosing record — fields not attributable per candidate")
        return {"status": "fallback", "ids": [], "why": "shared record"}
    prompt = build_formalize_prompt(state_msgs, arg_key, cur_value, records)
    kw = {kk: vv for kk, vv in dict(getattr(agent, "llm_args", None) or {}).items()
          if "tool" not in kk}
    for attempt in range(max_formalize):
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="filter_formalize_subcall", **kw)
        agent._t2_fsub_formalized = getattr(agent, "_t2_fsub_formalized", 0) + 1
        spec = parse_formalize(getattr(sub, "content", None) or "")
        if spec is None or spec["op"] in ("none", "unresolvable"):
            _mark("filter fallback: formalize %s" % ("UNSURE" if spec is None else spec["op"]))
            return {"status": "fallback", "ids": [], "why": "formalize"}
        result = execute_formalized(spec, records)
        if result["status"] == "ok":
            st = "one" if len(result["ids"]) == 1 else "many"
            _mark("filter %s arg=%s op=%s ids=%s (%s)"
                  % (st, arg_key, spec["op"], ",".join(map(str, result["ids"])), result["why"]))
            return {"status": st, "ids": result["ids"], "why": result["why"]}
        if result["status"] == "empty" and attempt + 1 < max_formalize:
            agent._t2_fsub_reformalize = getattr(agent, "_t2_fsub_reformalize", 0) + 1
            _mark("filter empty (%s) — re-formalize" % result["why"])
            prompt = prompt + ("\n\n[Note] A previous formalization %s matched NO candidate (%s). "
                               "Re-read the records: use only field names and values that actually "
                               "appear in them, or output op \"none\" if no criterion was stated."
                               % (json.dumps(spec, default=str), result["why"]))
            continue
        _mark("filter fallback: %s (%s)" % (result["status"], result["why"]))
        return {"status": "fallback", "ids": [], "why": result["status"]}
    return {"status": "fallback", "ids": [], "why": "re-formalize exhausted"}


def fexec_for_disamb(agent, la, UserMessage, state_msgs, arg_key, cur_value):
    """P-B DISAMB fire 지점서 호출(t2_gate_patch._t5c_disamb_subcall) — 주석 텍스트 반환.
    ""/None = 주석 없음(기존 DISAMB 그대로 = 폴백·§2.4 관할). 전 예외 = 호출측 no-op."""
    records = _candidate_record_dicts(arg_key, cur_value, state_msgs)
    if sum(1 for _, r in records if isinstance(r, dict)) < 2:
        return None
    prompt = build_formalize_prompt(state_msgs, arg_key, cur_value, records)
    try:
        um = UserMessage(role="user", content=prompt)
    except TypeError:
        um = UserMessage(content=prompt)
    kw = {kk: vv for kk, vv in dict(getattr(agent, "llm_args", None) or {}).items()
          if "tool" not in kk}
    sub = la.generate(model=agent.llm, tools=None, messages=[um],
                      call_name="formalize_subcall", **kw)
    agent._t2_fexec_fired = getattr(agent, "_t2_fexec_fired", 0) + 1
    spec = parse_formalize(getattr(sub, "content", None) or "")
    if spec is None:
        agent._t2_fexec_unsure = getattr(agent, "_t2_fexec_unsure", 0) + 1
        _mark("formalize UNSURE (parse/format) — DISAMB fallback")
        return None
    if spec["op"] == "none":
        agent._t2_fexec_none = getattr(agent, "_t2_fexec_none", 0) + 1
        _mark("op=none — DISAMB fallback")
        return None
    result = execute_formalized(spec, records)
    note = fexec_annotation(spec, result, arg_key)
    if note:
        agent._t2_fexec_annotated = getattr(agent, "_t2_fexec_annotated", 0) + 1
        _mark("annotated arg=%s op=%s status=%s ids=%s"
              % (arg_key, spec["op"], result["status"], ",".join(map(str, result["ids"])) or "-"))
    return note


# ═══════════════════════════════════════════════════════════════════════════
# ★L4 fexec-variants (A1_V3_IMPLEMENTATION §2·2026-07-13) — 변형 극값/속성 선택.
#   fexec 엔진을 product-variant record에 적용. I1(field-class)·I2/R5(numeric-safe)·
#   I6(variant dotted-path)·I7(floor-guard) 반영. [[05]]: variant 구조=A2 present_spec
#   (nested_field=variants·id_field=item_id) 재사용·엔진 리터럴(price/variants/options) 0.
# ═══════════════════════════════════════════════════════════════════════════

# L4a 극값어 사전 (eval-blind 영어·op-only·R2 gold-미참조).
# ★I1: op축과 field-class를 분리. PRICE_WORDS=field 결정론(price)·BARE_MAGNITUDE=field formalize 잔여.
_PRICE_WORDS = {
    "most expensive": "argmax", "priciest": "argmax", "dearest": "argmax",
    "highest priced": "argmax", "most costly": "argmax",
    "cheapest": "argmin", "least expensive": "argmin", "lowest priced": "argmin",
    "most affordable": "argmin", "best price": "argmin",
}
_BARE_MAGNITUDE = {
    "largest": "argmax", "biggest": "argmax", "greatest": "argmax", "maximum": "argmax",
    "smallest": "argmin", "tiniest": "argmin", "minimum": "argmin",
}


def _iter_product_variants(msgs, spec):
    """get_product_details 출력 각각 -> variants dict (제품 단위·안 합침). yield vs(dict)."""
    nested = spec.get("nested_field") or "variants"
    for o in _parse_tool_outputs(msgs, lenient=True):
        if not isinstance(o, dict):
            continue
        vs = o.get(nested)
        if not isinstance(vs, dict):
            for v in o.values():
                if isinstance(v, dict) and isinstance(v.get(nested), dict):
                    vs = v[nested]
                    break
        if isinstance(vs, dict):
            yield vs


def _vs_to_records(vs, idf):
    out = []
    for iid, rec in vs.items():
        if not isinstance(rec, dict) or rec.get("available") is False:
            continue
        out.append((str(rec.get(idf) or iid).strip(), rec))
    return out


def _variant_records_for(msgs, spec, anchor_id):
    """★제품별 스코핑(2026-07-13 probe 버그수정): anchor_id(대체 대상 원품목 item_id)를
    variants에 담는 *그 제품* 하나로 스코프 — 전역 pool 금지(t20 argmax 전역-max·t0 오매칭 원인).
    anchor 못 찾으면 [] (스코프 불가 = 치환 안 함·보수)."""
    idf = spec.get("id_field") or "item_id"
    aid = str(anchor_id or "").strip()
    if not aid:
        return []
    for vs in _iter_product_variants(msgs, spec):
        keys = {str(k).strip() for k in vs.keys()} | {str(r.get(idf)).strip() for r in vs.values()
                                                       if isinstance(r, dict)}
        if aid in keys:
            return _vs_to_records(vs, idf)
    return []


def _variant_records(msgs, spec, limit=40):
    """(폴백·전역) get_product_details variants 전부 -> [(item_id, record)]. 스코핑 실패 시만.
    A2 present_spec 재사용(엔진 리터럴 0). available==False 제외."""
    idf = spec.get("id_field") or "item_id"
    out = []
    for vs in _iter_product_variants(msgs, spec):
        out.extend(_vs_to_records(vs, idf))
        if len(out) >= limit:
            break
    seen, uniq = set(), []
    for rid, rec in out:
        if rid in seen:
            continue
        seen.add(rid)
        uniq.append((rid, rec))
    return uniq


def _option_value_type(records):
    """variant options의 각 키 -> 값-타입('num'|'str') (R5 field-type 화이트리스트).
    options 내부만(price/item_id/available 제외 = FP 차단). 혼합=str(보수)."""
    keytype = {}
    for _, rec in records:
        opts = rec.get("options")
        if not isinstance(opts, dict):
            continue
        for k, v in opts.items():
            kl = k.lower()
            t = "num" if _as_float(v) is not None else "str"
            if kl not in keytype:
                keytype[kl] = t
            elif keytype[kl] != t:
                keytype[kl] = "str"
    return keytype


def _ground_variant_criterion(request_text, records):
    """결정론 기준 추출(I1 field-class + I2/R5 numeric-safe). 반환 spec | None(->formalize).
    L4a 극값어(price=field결정론·bare=None) / L4b 속성(요청토큰 ∩ options 값·타입안전)."""
    rt = " " + (request_text or "").lower() + " "
    op = field = None
    for phrase, o in _PRICE_WORDS.items():
        if phrase in rt:
            op, field = o, "price"
            break
    if op is None:
        for word, o in _BARE_MAGNITUDE.items():
            if (" " + word + " ") in rt:
                op = o
                break
    keytype = _option_value_type(records)
    opt_vals = {}
    for _, rec in records:
        opts = rec.get("options")
        if isinstance(opts, dict):
            for k, v in opts.items():
                opt_vals.setdefault(k.lower(), set()).add(str(v).strip().lower())
    toks = re.findall(r"[a-z0-9\-]+", rt)
    cons = []
    for i, tok in enumerate(toks):
        is_num = bool(re.match(r"^\d+(ml|gb|tb|l|inch|in|oz)?$", tok))
        for k, vals in opt_vals.items():
            if is_num:
                if keytype.get(k) != "num":
                    continue
                keyhit = (" " + k + " ") in rt or (k in toks[max(0, i - 3):i + 1])
                if not keyhit:
                    continue
                tnum = re.match(r"^(\d+)", tok)
                if tnum and any(re.match(r"^(\d+)", vv) and re.match(r"^(\d+)", vv).group(1) == tnum.group(1)
                                for vv in vals):
                    cons.append({"field": "options." + k, "op": "eq", "value": tnum.group(1)})
            else:
                if tok in vals and len(tok) >= 3:
                    cons.append({"field": "options." + k, "op": "eq", "value": tok})
    seen, uc = set(), []
    for c in cons:
        key = (c["field"], str(c["value"]))
        if key not in seen:
            seen.add(key)
            uc.append(c)
    if op is None and not uc:
        return None
    if op in ("argmax", "argmin") and field is None:
        return None
    return {"op": op or "filter", "field": field, "constraints": uc}


def _floor_ok(result, cur_value):
    """★보수 floor-guard(2026-07-13 probe 버그수정): 결과 status/ids -> 치환 판정.
    - cur가 기준-만족집합 ∈ → 'keep'(에이전트 이미 옳음·정답 파괴 금지·t0)
    - cur ∉ ∧ 단일 → 'one'(치환) / ≥2 → 'many' / 0 → fallback."""
    ids = [str(i).strip() for i in result.get("ids") or []]
    if not ids:
        return None
    if str(cur_value).strip() in ids:
        return {"status": "keep", "ids": [str(cur_value).strip()], "why": "agent-value in criterion-set"}
    if len(ids) == 1:
        return {"status": "one", "ids": ids, "why": result.get("why", "")}
    return {"status": "many", "ids": ids, "why": result.get("why", "")}


def fexec_variant_decide(agent, la, UserMessage, msgs, arg_key, cur_value, a2_spec,
                         request_text, anchor_id=None, max_formalize=2):
    """L4: new_item(variant) operand 해소. ★제품별 스코핑(anchor_id=대체 대상 원품목)·전역 pool 금지.
    결정론 기준(L4a/L4b) 先 -> 없으면 formalize 폴백. I7 보수 floor-guard(cur∈집합=keep=정답 미파괴)."""
    records = _variant_records_for(msgs, a2_spec, anchor_id) if anchor_id else []
    scope = "product" if records else "none"
    if len(records) < 2:
        # 스코핑 실패(anchor 못 찾음)=치환 안 함(보수). 전역 pool 폴백은 금지(t20/t0 파손 원인).
        return {"status": "fallback", "ids": [], "why": "no product scope (%s)" % scope}
    spec = _ground_variant_criterion(request_text, records)
    if spec is not None:
        agent._t2_l4_field_det = getattr(agent, "_t2_l4_field_det", 0) + 1
        result = execute_formalized(spec, records)
        if result["status"] == "ok":
            fg = _floor_ok(result, cur_value)
            if fg is None:
                return {"status": "fallback", "ids": [], "why": "det:empty"}
            # ★보수화(2026-07-13 probe: t0 L4b 오추출 파손): 극값(argmax/argmin·price=신뢰)만 치환.
            #   filter(속성 grounding)은 복합기준서 오추출 위험 → keep/no-op만(치환 금지).
            if fg["status"] == "one" and spec["op"] not in ("argmax", "argmin"):
                _mark("L4 no-sub(filter det·conservative) arg=%s ids=%s scope=%s"
                      % (arg_key, ",".join(fg["ids"]), scope))
                return {"status": "fallback", "ids": [], "why": "det:filter-no-substitute"}
            _mark("L4 %s(det) arg=%s op=%s ids=%s scope=%s (%s)"
                  % (fg["status"], arg_key, spec["op"], ",".join(fg["ids"]), scope, result["why"]))
            return {"status": fg["status"], "ids": fg["ids"], "why": "det:" + result["why"]}
    agent._t2_l4_field_form = getattr(agent, "_t2_l4_field_form", 0) + 1
    prompt = build_formalize_prompt(msgs, arg_key, cur_value, records)
    kw = {kk: vv for kk, vv in dict(getattr(agent, "llm_args", None) or {}).items() if "tool" not in kk}
    for attempt in range(max_formalize):
        try:
            um = UserMessage(role="user", content=prompt)
        except TypeError:
            um = UserMessage(content=prompt)
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="l4_variant_formalize", **kw)
        fspec = parse_formalize(getattr(sub, "content", None) or "")
        if fspec is None or fspec["op"] in ("none", "unresolvable"):
            return {"status": "fallback", "ids": [], "why": "form:" + (fspec["op"] if fspec else "unsure")}
        result = execute_formalized(fspec, records)
        if result["status"] == "ok":
            fg = _floor_ok(result, cur_value)
            if fg is not None:
                # ★보수화: formalize 폴백은 극값만 치환·filter/속성은 keep/no-op(오추출 harm 차단).
                if fg["status"] == "one" and fspec["op"] not in ("argmax", "argmin"):
                    return {"status": "fallback", "ids": [], "why": "form:filter-no-substitute"}
                _mark("L4 %s(form) arg=%s ids=%s" % (fg["status"], arg_key, ",".join(fg["ids"])))
                return {"status": fg["status"], "ids": fg["ids"], "why": "form:" + result["why"]}
        if result["status"] == "empty" and attempt + 1 < max_formalize:
            prompt = prompt + "\n\n[Note] previous formalization matched NO variant; re-read options."
            continue
        return {"status": "fallback", "ids": [], "why": "form:" + result["status"]}
    return {"status": "fallback", "ids": [], "why": "form:exhausted"}
