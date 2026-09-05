# -*- coding: utf-8 -*-
"""★D12 — `[ACTION]` 문면을 **초안의 아무 호출에나** 붙이지 않는다 (설계서 §D12 · 014).

무엇이 있었나. 구판은 `rw_fb = ((am.tool_calls or [None])[0], _ufb)` 였다 — 손님-실행 도구를
겨눈 문면이 **초안의 첫 호출**에 붙었고, 그 첫 호출이 이관 도구이면 문면이 **그 호출의 오류
관측**으로 나갔다. 014 실측에서 그 결과로 gold 이관 액션이 궤적에서 **0회**가 됐다
(채점축 ACTION · MISSING 1). 부수로 기본 문면의 `"and do not transfer for this."` 가
**무조건절**이라 이관 자체를 포기하게 만들었다.

이 검정이 지키는 계약 셋 (전부 닫힌 술어):
  ⓐ 기본 문면의 이관 금지절이 **{tool} 로 좁혀져 있다** — 무조건절이 아니다.
  ⓑ 부착 대상 선택이 `_transfer_tools(a2)` 를 **제외**하고 `_utgt` 를 **우선**한다.
  ⓒ 구판의 `(am.tool_calls or [None])[0]` 직접 부착이 **이 채널에** 남아 있지 않다.
    (파일 전체엔 같은 패턴이 셋 더 있으나 전부 다른 되먹임 채널이라 D12 의 사정권 밖이다.)

⚠소스-계약 검정이다. 라이브 거동이 아니라 **코드가 그 계약을 유지하는지**를 본다 —
그래서 되돌리면 실패한다(아래 `__main__` 의 부정통제가 그것을 실증한다).
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()

FAILS = []


def check(name, ok, detail=""):
    if not ok:
        FAILS.append(name)
    print("%-34s %s%s" % (name, "ok" if ok else "FAIL", (" | " + str(detail)[:90]) if detail else ""))
    return ok


def _action_default_text(src):
    """`user_action_feedback` 기본 문면 블록을 잘라 온다."""
    i = src.index('"Error: [ACTION] ')
    j = src.index('.replace("{tool}", _utgt)', i)
    return src[i:j]


def _attach_block(src):
    """★`_ufb`(ACTION) 채널의 `rw_fb` 대입 구간만 잘라 온다.

    ⚠파일에 `rw_fb` 대입은 여섯 곳이 있고 그중 셋은 **다른 되먹임 채널**이다
    (resolve-verify · verify · action-required). D12 는 `user_action_feedback` 채널
    하나에 대한 판정이므로, 근거 없이 다른 채널까지 넓히면 범위 확장이다 —
    `_ufb` 를 쓰는 대입만 본다."""
    m = re.search(r"^\s*rw_fb = .*_ufb.*$", src, re.M)
    assert m, "_ufb 채널의 rw_fb 대입을 못 찾았다 — 계약 위치가 바뀌었다"
    head = src.rfind("\n", 0, max(0, m.start() - 2600))
    return src[head:m.end()]


def t_a_transfer_clause_is_scoped(src=None):
    """ⓐ 이관 금지절이 무조건절이 아니라 {tool} 로 좁혀져 있다."""
    body = _action_default_text(src if src is not None else SRC)
    flat = re.sub(r"\s+", " ", body)
    check("A_no_unconditional_clause", "do not transfer for this" not in flat, flat[:0])
    m = re.search(r"do not transfer[^.]*?\{tool\}", flat)
    check("A_clause_scoped_to_tool", bool(m), (m.group(0)[:80] if m else flat[:80]))


def t_b_attach_excludes_transfer_and_prefers_target(src=None):
    """ⓑ 부착 대상이 이관 도구를 제외하고 _utgt 를 우선한다."""
    blk = _attach_block(src if src is not None else SRC)
    check("B_excludes_transfer_tools", "_transfer_tools(" in blk, "이관 집합을 A2 에서 도출해 제외")
    check("B_prefers_utgt", ("_utgt" in blk) and ("_eff_tool_name(" in blk),
          "_utgt 와 이름이 같은 호출을 우선")
    check("B_no_domain_literal",
          not re.search(r'"[a-z][a-z0-9_]{4,}_\d{4}"', blk), "엔진에 도구명 리터럴 0")


def t_c_old_first_call_attach_is_gone(src=None):
    """ⓒ 구판의 첫-호출 직접 부착이 남아 있지 않다."""
    blk = _attach_block(src if src is not None else SRC)
    m = re.search(r"rw_fb\s*=\s*\(\(am\.tool_calls or \[None\]\)\[0\]", blk)
    check("C_old_first_call_gone", m is None, "구판 패턴 잔존" if m else "이 채널에 한정")


if __name__ == "__main__":
    for fn in (t_a_transfer_clause_is_scoped,
               t_b_attach_excludes_transfer_and_prefers_target,
               t_c_old_first_call_attach_is_gone):
        fn()

    # ── ★부정통제 ([[57]]) — 되돌린 소스로 돌리면 **실패해야** 한다 ──────────────
    print("\n--- 부정통제: 수리를 되돌린 소스로 재검정 ---")
    reverted = SRC
    reverted = re.sub(r"and do not transfer the conversation in order to \"\s*\n\s*\"get \{tool\} run\.",
                      'and do not transfer for this.', reverted)
    reverted = re.sub(r"^\s*rw_fb = .*$",
                      "                                    rw_fb = ((am.tool_calls or [None])[0], _ufb) if _ufb else None",
                      reverted, count=1, flags=re.M)
    before = len(FAILS)
    for fn in (t_a_transfer_clause_is_scoped,
               t_b_attach_excludes_transfer_and_prefers_target,
               t_c_old_first_call_attach_is_gone):
        try:
            fn(reverted)
        except Exception as e:
            FAILS.append("neg:%s(%r)" % (fn.__name__, e))
    neg = len(FAILS) - before
    if neg > 0:
        print("부정통제 OK — 되돌리면 %d 항목이 실패한다 (이 검정은 무의미하지 않다)" % neg)
        del FAILS[before:]                       # 부정통제의 실패는 기대된 것이므로 되돌린다
    else:
        FAILS.append("NEGATIVE_CONTROL_VACUOUS")
        print("⛔부정통제 실패 — 수리를 되돌려도 검정이 통과한다. 이 검정은 아무것도 안 지킨다.")

    print("\n%d FAIL" % len(FAILS) if FAILS else "\nALL PASS (D12 계약 3종 + 부정통제)")
    sys.exit(1 if FAILS else 0)
