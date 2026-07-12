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
    """도메인일반 재귀 필드 탐색(BFS·key 일치 case-insensitive 첫 값). 없으면 _MISSING."""
    if not isinstance(field, str):
        return _MISSING
    fl = field.strip().lower()
    queue = [rec]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            for kk, vv in cur.items():
                if str(kk).strip().lower() == fl:
                    return vv
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(x for x in cur if isinstance(x, (dict, list)))
    return _MISSING


def _field_values(rec, field):
    """도메인일반 재귀 필드 탐색 — 일치하는 *모든* 스칼라 값(BFS 순). 없으면 [].
    ★order-필터 확장(2026-07-12 HANDOFF §6.1): 후보=order record면 같은 필드(name 등)가
    items 하위에 반복 등장 — 'record가 X를 담는가'(containment)는 any-match가 정의라 전수 수집.
    field 값이 스칼라-리스트면 그 원소들을 수집(리스트 containment 동형)."""
    if not isinstance(field, str):
        return []
    fl = field.strip().lower()
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
