# -*- coding: utf-8 -*-
"""선언-우선 실 배선 (Y2 arm③·W-A~W-D) — 2026-07-31.

설계 정본 = `DECLFIRST_LIVE_WIRING_DESIGN_2026_07_31.md`.
상위 = `DECLARATION_FIRST_REDESIGN_2026_07_29.md` §0c LOCK · §1c 순서 · §1d R1~R13.

★배선 형태가 실측으로 좁혀졌다(어젯밤 3건):
  · `tools` + `guided_json` → **tool_calls 0**(선언만 하고 행동 0) ⇒ **행동 요청에 문법 금지**
  · hermes 파서는 `<tool_call>` **뒤 텍스트를 버린다**(C248) ⇒ 선언은 **호출 앞** 또는 **별도 호출**
  · X13 본런(C250): 프롬프트만 32% vs **two-pass 96%**(드롭 보정 구간 비중첩)
  ⇒ **기본형 = 2패스**(1패스=행동·문법 없음 / 2패스=도구 미제공+문법으로 형식화).

★1차 마일스톤 = **검출 전용**(`T2_DECLFIRST_ENFORCE=0`). 위반을 세고 기록만 하고 deny·regen은
  하지 않는다 — 집행은 Δspurious를 만들고([[19]]·C212), 그러면 Y2에서 "아키텍처 효과"와
  "개입 부작용"이 섞인다. **먼저 재고 나서 켠다.**

★비커밋: 2패스 요청·응답은 `state.messages`에 **넣지 않는다**(C208 replay 위생). 기록은 사이드카.

A2(base-layer·도메인-불변 내용):
    "declaration": {"schema": {...JSON Schema...}, "guide": "<시스템 프롬프트 말미 문구>"}
미선언 도메인이면 **레버 전체 skip**(U2′ 안전측).

플래그: `T2_DECLFIRST=1`(2패스+검증) · `T2_DECLFIRST_ENFORCE=1`(집행·2차 마일스톤·기본 0).
"""
import json
import os
import sys


# ── W-A: A2에서 선언 로드 ────────────────────────────────────────────────────
def declaration_of(a2):
    """A2 base-layer의 `declaration`. 미선언이면 None = 레버 skip."""
    dec = (a2 or {}).get("declaration")
    if not isinstance(dec, dict):
        return None
    if not isinstance(dec.get("schema"), dict) or not dec.get("guide"):
        return None
    return dec


def guide_text(a2):
    dec = declaration_of(a2)
    return dec["guide"] if dec else ""


# ── W-B: 2패스 형식화 프롬프트 ───────────────────────────────────────────────
def formalize_prompt(content, tool_names):
    """★프롬프트 누출 교정(C250 ⑥): 초판이 `tool_calls=[...]`를 그대로 넣어 모델이 **키 이름을
    복사**했다(`KB_search`를 `SEARCH_KB`로 지어내기도). 이름 목록을 주고 **축자 복사**를 요구한다."""
    return ("This is the turn you just produced.\n"
            "Message text: %r\n"
            "Tools you actually called (exact names): %s\n\n"
            "Emit the declaration envelope for that turn. For next_action, copy ONE name VERBATIM "
            "from the list above (do not invent or abbreviate); if the list is empty, use null."
            % ((content or "")[:800], ", ".join(tool_names) if tool_names else "(none)"))


def parse_envelope(text):
    """균형 중괄호 스캔 + `turn_type` 보유 객체만 인정(X13에서 확립한 파서)."""
    t = (text or "").strip()
    if not t or "{" not in t:
        return None
    blocks = []
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(t)):
            c = t[j]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(t[i:j + 1])
                    break
    for b in blocks:
        try:
            o = json.loads(b)
        except Exception:
            continue
        if isinstance(o, dict) and "turn_type" in o:
            return o
    return None


# ── W-C: 검증기 (§1d 중 **닫힌 것만**) ───────────────────────────────────────
def verify(envelope, tool_names, executed_writes):
    """선언 ↔ 실제 행동 정합. 전부 닫힌 술어(문자열 동등·집합 포함).

    executed_writes = 이 sim에서 **실제로 실행된** write의 실효 이름 집합(R5용).
    """
    v = []
    if envelope is None:
        return ["NO_ENVELOPE"]
    tt = envelope.get("turn_type")
    acted = bool(tool_names)
    if tt == "ACT" and not acted:
        v.append("R1_ACT_WITHOUT_CALL")
    if tt in ("ASK", "INFORM", "DONE") and acted:
        v.append("R2_CALL_WITHOUT_ACT")
    na = envelope.get("next_action")
    if acted and na and na not in tool_names:
        v.append("R4_NEXT_ACTION_MISMATCH")
    if tt == "ASK" and not envelope.get("ask"):
        v.append("R13_ASK_WITHOUT_SLOT")
    # ★R5 = 이 배선의 본체 가치: 완료 주장을 **실행 원장과 대조**.
    #   ⚠산문("filed the dispute")과 도구명(`submit_cash_back_dispute`)을 문자열 포함으로 잇는 것은
    #   **열린 술어**다(초판 시도가 정직한 주장을 위반으로 오판했다). 대신 **스키마가 도구 이름을
    #   담게** 해서 판정을 집합 소속으로 닫는다 — 모델이 형식화하고 엔진은 대조만 한다([[22]]).
    for item in (envelope.get("done_report") or []):
        if not isinstance(item, dict):
            continue
        if not str(item.get("what") or item.get("kind") or "").strip():
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool:
            v.append("R5_CLAIM_UNVERIFIABLE")      # 주장은 했는데 무엇으로 했는지 선언 안 함
            continue
        if tool not in executed_writes:
            v.append("R5_CLAIM_WITHOUT_WRITE")     # 선언한 도구가 실행 원장에 없다
    return v


# ── W-D: 검출 전용 실행 ──────────────────────────────────────────────────────
def run(a2, gen_fn, base_messages, am, executed_writes, seed=0):
    """2패스 형식화 + 검증 + 기록. **검출 전용**(기본)이면 위반을 돌려주기만 한다.

    gen_fn(messages, guided_schema) -> 텍스트  (호출측이 도구 **미제공**을 보장해야 한다)
    반환 {"envelope":…, "violations":[…], "enforced":False} 또는 None(레버 skip).
    """
    if os.environ.get("T2_DECLFIRST") != "1":
        return None
    dec = declaration_of(a2)
    if dec is None:
        return None                                   # 미선언 도메인 = skip(U2′)
    names = [getattr(tc, "name", None) or "" for tc in (getattr(am, "tool_calls", None) or [])]
    try:
        txt = gen_fn(base_messages + [{"role": "user",
                                       "content": formalize_prompt(getattr(am, "content", ""), names)}],
                     dec["schema"])
    except Exception as e:
        print("[T2_DECLFIRST] 2패스 실패(무시): %r" % (e,), file=sys.stderr, flush=True)
        return None
    env = parse_envelope(txt)
    viol = verify(env, names, executed_writes or set())
    enforce = os.environ.get("T2_DECLFIRST_ENFORCE") == "1"
    print("[T2_DECLFIRST] envelope=%s viol=%s%s"
          % ("yes" if env else "NO", ",".join(viol) or "-", " ENFORCE" if enforce else ""),
          file=sys.stderr, flush=True)
    try:
        import t2_fbsidecar as _sc
        _sc.record("declfirst", json.dumps(env, ensure_ascii=False) if env else "",
                   None, viol=",".join(viol), turn_type=(env or {}).get("turn_type"),
                   n_calls=len(names))
    except Exception:
        pass
    return {"envelope": env, "violations": viol, "enforced": enforce}


if __name__ == "__main__":
    SCHEMA = {"type": "object", "properties": {"turn_type": {"type": "string"}},
              "required": ["turn_type"]}
    A2 = {"declaration": {"schema": SCHEMA, "guide": "Emit an envelope."}}

    assert declaration_of({}) is None and declaration_of(A2) is not None
    assert declaration_of({"declaration": {"schema": SCHEMA}}) is None, "guide 없으면 skip"
    print("  W-A 로드·미선언 skip: OK")

    p = formalize_prompt("hello", ["KB_search"])
    assert "tool_calls" not in p and "KB_search" in p and "VERBATIM" in p
    print("  W-B 프롬프트 누출 없음(키 이름 미포함·축자 요구): OK")

    assert parse_envelope('prose {"a":1} then {"turn_type":"ACT","prose":"x"}')["turn_type"] == "ACT"
    assert parse_envelope("no json here") is None
    print("  파서(산문 뒤 봉투·중괄호 혼입): OK")

    cases = [
        ("ACT인데 호출 0", {"turn_type": "ACT"}, [], set(), "R1_ACT_WITHOUT_CALL"),
        ("호출했는데 DONE", {"turn_type": "DONE"}, ["x"], set(), "R2_CALL_WITHOUT_ACT"),
        ("next_action 불일치", {"turn_type": "ACT", "next_action": "y"}, ["x"], set(),
         "R4_NEXT_ACTION_MISMATCH"),
        ("ASK인데 slot 없음", {"turn_type": "ASK"}, [], set(), "R13_ASK_WITHOUT_SLOT"),
        ("★거짓 완료 주장", {"turn_type": "INFORM",
                        "done_report": [{"kind": "dispute", "what": "filed it",
                                         "tool": "submit_cash_back_dispute"}]},
         [], set(), "R5_CLAIM_WITHOUT_WRITE"),
        ("정직한 완료 주장", {"turn_type": "INFORM",
                        "done_report": [{"kind": "dispute", "what": "filed it",
                                         "tool": "submit_cash_back_dispute"}]},
         [], {"submit_cash_back_dispute"}, None),
        ("도구 미선언 주장", {"turn_type": "INFORM",
                        "done_report": [{"kind": "dispute", "what": "filed it"}]},
         [], {"submit_cash_back_dispute"}, "R5_CLAIM_UNVERIFIABLE"),
        ("봉투 없음", None, ["x"], set(), "NO_ENVELOPE"),
    ]
    ok = 0
    for name, env, names, writes, want in cases:
        got = verify(env, names, writes)
        hit = (want in got) if want else (got == [])
        ok += hit
        print("  %-18s -> %-28s %s" % (name, ",".join(got) or "(없음)", "OK" if hit else "FAIL"))
    os.environ.pop("T2_DECLFIRST", None)
    assert run(A2, None, [], None, set()) is None, "기본 OFF여야 한다"
    print("  W-D 기본 OFF no-op: OK")
    print("verify selftest %d/%d" % (ok, len(cases)))
    sys.exit(0 if ok == len(cases) else 1)
