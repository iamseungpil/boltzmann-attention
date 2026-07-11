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
    for cons in spec["constraints"]:
        vals = [(c, _field_lookup(r, cons["field"])) for c, r in recs]
        if all(v is _MISSING for _, v in vals):
            return {"status": "unresolvable", "ids": [],
                    "why": "constraint field '%s' absent from every candidate record" % cons["field"]}
        keep = {c for (c, v) in vals if v is not _MISSING and _cons_match(v, cons)}
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
