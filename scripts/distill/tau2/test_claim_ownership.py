# -*- coding: utf-8 -*-
"""회귀 검정 (C348⒢): 미이행-약속을 **도구 소유자**로 가른다 — 그리고 남의 것에는 침묵한다.

무엇을 막는 검정인가 —
 ⒜ **한 일을 안 했다고 말하는 것**. 약속이 지목한 도구가 **손님 소유**면 *안내하는 것이 곧
    이행*이다. 그런데 구판은 실행 원장에 없다는 이유로 *"never actually executed"* 를 냈다.
    모델이 고칠 수 없는 오류이고, 우리 출력은 이 대화의 유일한 근거원이다([[25]]·C341 동형).
    ⇒ 손님 소유는 **침묵**해야 한다.
 ⒝ **엉뚱한 지적**. 도구가 **에이전트 소유**인데 손님에게 떠넘겼다면, 결함은 약속 위반이
    아니라 **소유권**이다(런 m 실측: `submit_referral` 은 `@is_tool(ToolType.WRITE)` = 에이전트
    도구인데 에이전트가 *"use it yourself"* 로 넘겼다). ⇒ 소유권 **사실**이 나가야 한다.
 ⒞ **명령으로 번지는 것**. claim 축은 **표면화만**이다(C216 §2-3b · 같은 자리에서
    `tool_choice="required"` 를 붙였다가 철회한 이력 = `t2_gate_patch` 주석). 문구에 쓰기
    강제가 들어가면 안 된다([[45]] §1.5 Q5: p<0.5면 기대-유해).
 ⒟ **거동 파괴**. 소유를 모르는 주장은 구판 문구 그대로여야 하고, A2 미선언 도메인은 거동 0.
 ⒠ **死코드**([[24]]): 정본(base)에 넣은 문구가 로더 병합과 `gate.json` 양쪽에 실려야 한다.

오프라인 전용: tau2·서버·LLM 불요. 실행: py -3 test_claim_ownership.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G                                      # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

DOMAIN = "banking_knowledge"
FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


# 도구 이름은 **레지스트리 대역**에서 온다 — 이 검정이 도메인 어휘를 짓지 않는다([[59]]).
AGENT = ["alpha_write_tool", "beta_read_tool"]
USER = ["gamma_user_tool"]


def claim(what, tool):
    return {"kind": "give", "what": what, "tool": tool}


def main():
    # ── ⒜⒝ 소유자로 갈린다 ──────────────────────────────────────────────────
    cs = [claim("do the write", "alpha_write_tool"),      # 에이전트 소유
          claim("guide customer to run it", "gamma_user_tool"),  # 손님 소유
          claim("something else", "delta_unknown_tool"),  # 모름
          claim("no tool named", None)]                   # 지목 없음
    own, theirs, unk = G._split_claims_by_owner(cs, AGENT, USER)
    chk([c["tool"] for c in own] == ["alpha_write_tool"], "에이전트 소유가 own 으로 간다")
    chk([c["tool"] for c in theirs] == ["gamma_user_tool"],
        "손님 소유가 theirs 로 간다  ← 이쪽은 호출부가 침묵시킨다")
    chk(len(unk) == 2, "모름·미지목은 unknown 으로 남는다(구판 거동 보존)")

    # 접미사 정규화 (`_NNNN` 붙은 discoverable 실물)
    own2, theirs2, _ = G._split_claims_by_owner(
        [claim("x", "alpha_write_tool_3847"), claim("y", "gamma_user_tool_0589")], AGENT, USER)
    chk(len(own2) == 1 and len(theirs2) == 1, "접미사가 붙어도 소유자를 찾는다")

    # 양쪽에 있으면 에이전트 우선(부를 수 있으면 자기 것)
    own3, theirs3, _ = G._split_claims_by_owner([claim("z", "dual")], ["dual"], ["dual"])
    chk(len(own3) == 1 and not theirs3, "양쪽에 있으면 에이전트 우선(부를 수 있다)")

    # 빈 입력·None 에서 죽지 않는다
    chk(G._split_claims_by_owner(None, None, None) == ([], [], []), "빈 입력에 안 죽는다")
    o4, t4, u4 = G._split_claims_by_owner([claim("w", "alpha_write_tool")], [], [])
    chk(len(u4) == 1 and not o4 and not t4,
        "레지스트리가 비면 전부 unknown  ← 모르면 구판대로(안전측)")

    # ── ⒠ 문구가 3층 전부에 실렸나 (死코드 방지·[[24]]) ─────────────────────
    a2 = load_domain_a2(DOMAIN) or {}
    cp = a2.get("claim_prov") or {}
    body = cp.get("feedback_ownership")
    chk(bool(body), "로더 병합에 feedback_ownership 이 있다")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "a2", "%s.gate.json" % DOMAIN), encoding="utf-8") as fh:
        mono = json.load(fh)
    chk((mono.get("claim_prov") or {}).get("feedback_ownership") == body,
        "gate.json 도 같은 값이다  ← 한쪽만 고치면 死코드/등가 FAIL")
    chk("{claims}" in (body or ""), "자리표시자 {claims} 가 있다(엔진이 채운다)")

    # ── ⒞ 표면화만: 쓰기 강제 어휘가 없어야 한다 ─────────────────────────────
    low = (body or "").lower()
    banned = [w for w in ("call them now", "call it now", "do it now", "you must call",
                          "immediately call", "right now") if w in low]
    chk(not banned, "명령형 쓰기-강제 문구가 없다 (발견: %s)  ← C216 §2-3b 표면화만" % (banned or "없음"))
    chk("your own tool list" in low or "not the customer" in low,
        "소유권 사실을 말한다(약속 위반이 아니라)")

    # ── 도메인 어휘 0 (base 층 자격) ─────────────────────────────────────────
    with open(os.path.join(here, "a2", "base", "shared.json"), encoding="utf-8") as fh:
        base = json.load(fh)
    bbody = ((base.get("claim_audit") or {}).get("feedback_ownership") or "")
    chk(bbody == body, "정본은 base(도메인-불변 층)에 있다  ← 새 도메인 opex 0")
    dirty = [w for w in ("referral", "account", "card", "bank", "deposit", "submit_")
             if w in bbody.lower()]
    chk(not dirty, "base 문구에 업종·도구 명사가 없다 (발견: %s)" % (dirty or "없음"))

    # ── ⒟ 미선언 도메인은 거동 0 ────────────────────────────────────────────
    for d in ("retail", "airline"):
        other = ((load_domain_a2(d) or {}).get("claim_prov") or {})
        chk("feedback_ownership" not in other,
            "%s 는 claim_audit 미선언 → 이 키가 안 생긴다(거동 보존)" % d)

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
