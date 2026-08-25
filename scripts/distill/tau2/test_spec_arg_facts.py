# -*- coding: utf-8 -*-
"""T2_SPEC_ARG_FACTS 래칫 (2026-08-25) — 손 선언을 **대체하는 파생**이 옳고, 좁아지지 않는가.

왜: 사용자 물음(*"태스크별로 특정한 방식을 만들지 않고 일반화로는 지금 문제를 해결 못하는
건가?"*)에 대한 답이 이 레버다. 오늘 A2 에 손으로 적은 값 목록 6칸·불리언 2세트는 전부 env 가
unlock 때 고정 포맷으로 건네주던 것이고, 등가성은 `x540_spec_derivation.py` 가 코퍼스 실물로
쟀다(명세 블록 61 · 도구 16 · 대조 9건 전부 일치 · 다르다 0 · 대조 불가 0).

이 검정이 지키는 것:
  ① **도구별 키잉** — 같은 인자 이름이 도구마다 다른 값 집합을 갖는다(`card_action` 실물).
     이름만으로 합치면 정당한 값을 거절한다. 합친 판(`_declared_params`)과 도구별 판이
     이 자리에서 **갈리는지**를 검정한다.
  ② **A2 손 선언과 등가** — 병합 A2 의 `write_arg_enum` 값·불리언이 도출과 집합으로 같다.
  ③ **거절 문면에 도메인 낱말 0** — 문장은 고정층에 살고 어느 도메인에서도 같다([[05]]).
  ④ **엔진이 고르지 않는다** — 게이트 블록에 순위·최댓값 어휘가 없다([[62]]④).
  ⑤ 기본 OFF · 명세가 없으면 무발화(fail-open).
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 같은 인자 이름(`card_action`)이 **두 도구에서 다른 값 집합**을 갖는 실물 (t7354 축자).
TWO_TOOLS = """Tool unlocked: file_credit_card_transaction_dispute_4829

Tool: file_credit_card_transaction_dispute_4829
Parameters:
  - card_action: string (required) - Flag indicating the card's status. Must be one of: 'keep_active' (card remains active, dispute only), 'cancel_and_reissue' (card is being cancelled and replaced).
  - contacted_merchant: boolean (required) - Whether the user attempted to resolve the issue first
"""

TWO_TOOLS_B = """Tool unlocked: file_debit_card_transaction_dispute_6281

Tool: file_debit_card_transaction_dispute_6281
Parameters:
  - card_action: string (required) - Action to take on the card. Must be one of: 'keep_active', 'freeze_pending_investigation', 'close_and_reissue'
  - pin_compromised: string (required) - Whether the PIN may have been compromised. Must be one of: 'yes_shared', 'yes_observed', 'no', 'unknown'
  - card_in_possession: boolean (required) - Whether the customer still has the card
"""


class _M(object):
    def __init__(self, c):
        self.content = c


def main():
    import t2_gate_patch as G
    from gate_interpreter import load_domain_a2

    msgs = [_M(TWO_TOOLS), _M(TWO_TOOLS_B)]

    # ① 도구별 키잉
    by = G._declared_params_by_tool(msgs)
    ccd = by.get("file_credit_card_transaction_dispute_4829") or {}
    dbd = by.get("file_debit_card_transaction_dispute_6281") or {}
    assert ccd.get("card_action", ("", []))[1] == ["keep_active", "cancel_and_reissue"], \
        ccd.get("card_action")
    assert dbd.get("card_action", ("", []))[1] == [
        "keep_active", "freeze_pending_investigation", "close_and_reissue"], dbd.get("card_action")
    assert ccd["card_action"][1] != dbd["card_action"][1], (
        "두 도구의 같은 인자가 같은 명단을 받았다 — 도구별 키잉이 죽었다")
    assert dbd.get("card_in_possession", ("", []))[0] == "boolean"
    assert G._declared_params_by_tool([_M("형식이 아닌 산문")]) == {}, "fail-open 이 깨졌다"

    # ② A2 손 선언과 등가 (도구 이름은 접두로 잇는다 — 선언이 접두로 지목하므로)
    a2 = load_domain_a2("banking_knowledge")
    specs = [s for s in (a2.get("write_arg_enum") or []) if s.get("values") or s.get("booleans")]
    assert specs, "대조할 손 선언이 없다"
    seen = 0
    for sp in specs:
        prefix = (sp.get("applies_when") or {}).get("prefix") or ""
        tools = [t for t in by if t.startswith(prefix)]
        if not tools:
            continue                    # 이 픽스처에 없는 도구는 x540 이 코퍼스로 대조한다
        d = by[tools[0]]
        if sp.get("values") and sp.get("arg") in d:
            got = d[sp["arg"]][1]
            assert sorted(got) == sorted(str(x) for x in sp["values"]), (
                "%s.%s 선언 %r ↔ 도출 %r" % (tools[0], sp["arg"], sp["values"], got))
            seen += 1
        if sp.get("booleans"):
            got = sorted(k for k, v in d.items() if v[0] == "boolean")
            miss = [b for b in sp["booleans"] if b in d and b not in got]
            assert not miss, "불리언인데 도출이 놓쳤다: %r" % miss
            seen += 1
    assert seen, "픽스처가 어떤 손 선언과도 안 겹친다 — 검정이 헛돈다"

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()

    # ③ 문면에 도메인 낱말 0
    for fb in (G._SPEC_TYPE_FB, G._SPEC_ENUM_FB):
        low = fb.lower()
        for bad in ("card", "dispute", "account", "atm", "transaction", "bank"):
            assert bad not in low, "고정층 문면에 도메인 낱말이 들어왔다: %r" % bad

    # ④ 게이트가 고르지 않는다
    i = src.index('os.environ.get("T2_SPEC_ARG_FACTS")')
    blk = src[i:src.index("_ens = (a2 or {}).get(", i)]
    for bad in ("max(", "argmax", "score", "rank", "sorted(_en3", "bm25"):
        assert bad not in blk, "게이트가 고르기 시작했다: %r" % bad
    assert "_declared_params_by_tool" in blk, "도구별 판을 쓰지 않는다 — 명단이 섞인다"
    assert "_hint_hit" not in blk, "이름 패턴이 돌아왔다"

    # ⑤ 기본 OFF
    assert "T2_SPEC_ARG_FACTS=0" in io.open(os.path.join(HERE, "go_stack.sh"),
                                            encoding="utf-8").read(), "기본 OFF 가 아니다"
    print("OK T2_SPEC_ARG_FACTS: 도구별 키잉(같은 인자 다른 명단) · 손 선언 대조 %d건 통과 · "
          "문면 도메인 낱말 0 · 순위 0 · 기본 OFF" % seen)


if __name__ == "__main__":
    main()
