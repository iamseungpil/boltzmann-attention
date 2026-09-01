# -*- coding: utf-8 -*-
"""★§T-6a 뷰-압축 문턱 — 모델 컨텍스트에서 유도되는가 · 대조군이 보존되는가.

근거: 상수 60,000자/8,000자 는 컨텍스트 44,672 이던 Q2.5 시절 값이다. Q3.8(131,072)에서는
컨텍스트의 **11%** 에서 지우기 시작한다는 뜻이고, 그 지움이 재열람을 낳아 스텝을 태웠다
(base 51~81 메시지 ↔ ours 209~293 · `max_steps` 6/30).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G


def t_off_keeps_legacy_constants():
    assert G.view_thresholds(cap_tokens=131072, scale="off") == (60000, 8000)


def t_auto_scales_with_context():
    mt, mc = G.view_thresholds(cap_tokens=131072, scale="auto")
    assert mt == int(131072 * 0.5 * 3.5) and mc == int(131072 * 0.25), (mt, mc)
    assert mt > 60000 * 3, "Q3.8 문턱은 구 상수보다 3배 이상이어야 한다"


def t_auto_never_below_legacy():
    """작은 컨텍스트에서 문턱이 **더 조여지면** 안 된다 — 구 팔의 조건을 밑돌지 않는다."""
    mt, mc = G.view_thresholds(cap_tokens=8192, scale="auto")
    assert mt >= 60000 and mc >= 8000, (mt, mc)


def t_explicit_env_wins():
    mt, mc = G.view_thresholds(cap_tokens=131072, scale="auto",
                               mintotal="12345", msgcap="678")
    assert (mt, mc) == (12345, 678), "팔이 값을 고정할 수 있어야 한다"


def t_no_context_falls_back():
    assert G.view_thresholds(cap_tokens=0, scale="auto") == (60000, 8000)


def t_q25_profile_is_close_to_legacy():
    """Q2.5(44,672)에서 파생값이 종전 규모와 같은 자릿수인가 — 급변이면 대조가 깨진다."""
    mt, _ = G.view_thresholds(cap_tokens=44672, scale="auto")
    assert 60000 <= mt <= 100000, mt



def t_msgcap_zero_disables_the_per_message_path():
    """★§T-6b — 압축 경로 ①(메시지당 캡)을 끄는 팔. `_compact_view` 는 `msg_cap` 이 0이면
    `total < min_total` 에서 **조기 반환**하므로 경로 ① 자체가 죽고 ②(총량 문턱)만 남는다."""
    mt, mc = G.view_thresholds(cap_tokens=131072, scale="auto", msgcap="0")
    assert mc == 0 and mt == int(131072 * 0.5 * 3.5), (mt, mc)


def t_arm_files_declare_the_two_paths_distinctly():
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arms")
    vs = open(os.path.join(d, "viewscale.env"), encoding="utf-8").read()
    vm = open(os.path.join(d, "viewscale_max.env"), encoding="utf-8").read()
    # viewscale 은 ①을 파생값으로 남기고, viewscale_max 는 ①을 끈다 — 한 칸만 다르다.
    assert "T2_VIEW_MSG_CAP=0" in vm and "unset T2_VIEW_MSG_CAP" in vs
    assert "T2_VIEW_SCALE=auto" in vs and "T2_VIEW_SCALE=auto" in vm

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
