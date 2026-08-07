# -*- coding: utf-8 -*-
"""단계 1 게이트⒝ **계기 검정** — `audit()`의 갈림 판정이 진짜인가 (유료 0).

정본 = `FACT_DAG_DESIGN_2026_08_08.md` §7e. 배선의 목적은 *"현행이 고른 것 ≠ `route()`가 골랐을 것"*
사례를 세는 것이고, **N=0이면 단계 2b를 착수하지 않는다.** 그러므로 이 계기가 틀리면 착수 판단이
통째로 틀린다.

특히 막아야 하는 것 — **거짓 갈림**: 현행 체인의 표지는 채널 이름(`eplan`·`proc`…)이고 등록
플래그는 `T2_*`다. 두 이름 공간을 그대로 맞대면 **언제나 "다름"** 이 나온다(구현 전 초안이 정확히
그랬다). 그래서 비교는 **표적**으로만 한다.

실행: `python test_audit_divergence.py`
"""

import io
import re  # noqa: F401
import sys
import types

sys.path.insert(0, ".")

import t2_stack as S   # noqa: E402


def _orch():
    """`_owner()`가 에이전트로 내려가지 않도록 `llm` 없는 단순 객체를 쓴다."""
    return types.SimpleNamespace()


def test_same_target_is_not_a_divergence():
    """현행이 말한 표적을 `route()`도 골랐으면 **갈린 것이 아니다**."""
    o = _orch()
    S.register(o, flag="T2_LIMIT_REDUCE", target="submit_referral", fact="사실")
    a = S.audit(o, chose_targets=["submit_referral"])
    assert a["target_differs"] is False, a


def test_different_target_is_a_divergence():
    """현행은 A를 말했는데 `route()`는 B를 골랐다 = **갈림 1건**."""
    o = _orch()
    S.register(o, flag="T2_LIMIT_REDUCE", target="log_verification", fact="사실")
    a = S.audit(o, chose_targets=["submit_referral"])
    assert a["target_differs"] is True, a


def test_channel_names_do_not_fake_a_divergence():
    """★채널 이름(`proc`)과 플래그(`T2_*`)를 맞대 **거짓 갈림**을 만들지 않는다."""
    o = _orch()
    S.register(o, flag="T2_PROCEDURE", target="submit_referral", fact="사실")
    a = S.audit(o, chose="proc", chose_targets=["submit_referral"])
    assert a["target_differs"] is False, a          # 표적 축 = 같다
    assert a["differs"] is True, a                  # 플래그 축은 이름 공간이 달라 언제나 다르다
    # ⇒ 그래서 **판정에 쓰는 것은 `target_differs`** 이고, 로그도 그것을 찍는다.


def test_no_choice_is_not_a_divergence():
    """현행이 아무 말도 안 했으면(접혔거나 deny 없음) 갈림으로 세지 않는다 — 분모 오염 금지."""
    o = _orch()
    S.register(o, flag="T2_LIMIT_REDUCE", target="submit_referral", fact="사실")
    a = S.audit(o, chose_targets=[])
    assert a["target_differs"] is False, a


def test_registrations_are_drained():
    """감사는 등록분을 **비운다** — 안 비우면 sim 내내 쌓여 다음 턴 판정이 오염된다."""
    o = _orch()
    S.register(o, flag="T2_LIMIT_REDUCE", target="t", fact="f")
    assert S.audit(o) is not None
    assert S.audit(o) is None, "드레인되지 않았다"


def test_every_registering_lever_has_a_layer():
    """★상설 감사 — `beat(orch=…)`로 **등록하는 레버는 전부 층이 있어야 한다**.

    층이 없으면 `route()`가 그 후보를 **조용히 버린다**(빈 `pick`). 이 검정을 처음 쓸 때 없는
    이름(`T2_ARBITRATE`)을 넣었다가 빈 `pick`을 받았고, 그것이 *"코드가 버그"* 로 보였다 —
    실제로는 실등록 12종이 전부 분류돼 있었다. 그 착시가 반대 방향(진짜 미분류)으로 일어나면
    레버 하나가 말없이 사라지므로([[48]] §5.6b "코드 없는 레버 0") 여기서 상시로 막는다.
    """
    import re
    src = io.open("t2_gate_patch.py", encoding="utf-8").read()
    flags = sorted(set(re.findall(r'_lbeat\(\s*"([A-Z0-9_]+)"', src)))
    assert flags, "등록 지점을 못 찾았다 — 이 감사가 무력하다"
    missing = [f for f in flags if S.layer_of(f) is None]
    assert not missing, "층 미분류 = route()가 조용히 버린다: %s" % missing


def test_unknown_flag_vanishes_without_a_word():
    """현행 거동을 **문서화**한다: 미지 플래그는 경고 없이 사라진다(위 감사가 필요한 이유)."""
    o = _orch()
    S.register(o, flag="T2_DOES_NOT_EXIST", target="t", fact="f")
    a = S.audit(o, chose_targets=["t"])
    assert a["pick"] == [], a
    assert a["target_differs"] is True, "사라진 것이 '갈림'으로 보인다 — 해석 시 주의"


def test_two_targets_partial_overlap_counts_as_divergence():
    """겹치되 같지 않으면 갈림이다 — 부분 일치를 '같다'로 세면 N이 과소 집계된다."""
    o = _orch()
    S.register(o, flag="T2_LIMIT_REDUCE", target="a", fact="f")
    S.register(o, flag="T2_PROCEDURE", target="b", fact="f")
    a = S.audit(o, chose_targets=["a"])
    assert a["target_differs"] is True, a


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            bad += 1
            print("  FAIL  %s\n        %s" % (name, e))
        except Exception as e:
            bad += 1
            print("  ERROR %s\n        %r" % (name, e))
    print("\n%s %d/%d" % ("[계기 검정 PASS]" if not bad else "[FAIL]",
                          len(tests) - bad, len(tests)))
    sys.exit(1 if bad else 0)
