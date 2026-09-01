# -*- coding: utf-8 -*-
"""회귀 — 모델 프로필 config 계약 (Qwen2.5 ↔ Qwen3.8 교대 실험).

★왜 (2026-08-31·[[84]]): 모델을 바꿀 때 **같이 바뀌어야 하는 값**이 코드와 임시 런 스크립트에
  흩어져 있어서 조용히 어긋났다 — 표면형(hermes ↔ qwen3_coder)과 문맥 상한(44,672 ↔ 131,072),
  생성 상한(8192 를 3072 로 덮음). 이제 그 값들은 `model_profiles/*.env` 한 곳에 있고,
  런처가 서빙 중인 모델 id 로 고르며, 없으면 발사를 거부한다.

⚠이 검정은 **값이 맞는지**를 판정하지 않는다(그건 서버 로그·격리의 몫). 계약만 본다:
  필수 키가 다 있는가 · 표면형이 아는 값인가 · 두 모델이 실제로 다른 표면형을 선언하는가.
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DIR = os.path.join(HERE, "model_profiles")
# ★2026-09-01 추가: `T2_PROBE_MAX_TOKENS` 는 **모델에 매인 값**이다(사고를 쓰는 모델에서 256 은
#   사고 예산과 같아 답 자리가 0 — 밤샘런 TRUNC 85건 전량이 그 호출이었다). 코드 기본값에 두면
#   Q2.5 와 Q3.8 을 **동시에** 돌릴 때 한쪽 값이 다른 쪽으로 샌다 ⇒ 프로필 필수 키로 올린다.
REQUIRED = ["T2_TOOL_SURFACE", "T2_MAX_MODEL_LEN", "T2_AGENT_MAX_TOKENS", "T2_STOP_FIRST_TOOLCALL",
            "T2_PROBE_MAX_TOKENS"]
KNOWN_SURFACES = {"hermes", "qwen3_xml"}


def _parse(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S*)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _profiles():
    return sorted(glob.glob(os.path.join(DIR, "*.env")))


def test_profiles_exist_for_both_families():
    names = [os.path.basename(p) for p in _profiles()]
    assert any("Qwen3.8" in n for n in names), names
    assert any("Qwen2.5" in n for n in names), names


def test_every_profile_declares_the_required_keys():
    for p in _profiles():
        got = _parse(p)
        missing = [k for k in REQUIRED if k not in got]
        assert not missing, "%s 에 %s 가 없다" % (os.path.basename(p), missing)


def test_surface_values_are_known():
    for p in _profiles():
        sf = _parse(p)["T2_TOOL_SURFACE"]
        assert sf in KNOWN_SURFACES, "%s: %r" % (p, sf)


def test_the_two_families_declare_different_surfaces():
    """이 차이가 이번 사고의 본체다 — 같아지면 둘 중 하나가 틀린 것이다."""
    q38 = [_parse(p) for p in _profiles() if "Qwen3.8" in p][0]
    q25 = [_parse(p) for p in _profiles() if "Qwen2.5" in p][0]
    assert q38["T2_TOOL_SURFACE"] == "qwen3_xml"
    assert q25["T2_TOOL_SURFACE"] == "hermes"
    assert q38["T2_MAX_MODEL_LEN"] != q25["T2_MAX_MODEL_LEN"]


def test_every_value_carries_a_source_comment():
    """값마다 출처 한 줄([[77]]) — 근거 없는 상수가 다시 스며들지 않게."""
    for p in _profiles():
        text = open(p, encoding="utf-8").read()
        assert re.search(r"tool_call_parser", text), "%s: 서버 기동 인자 인용이 없다" % p
        assert len([l for l in text.splitlines() if l.startswith("#")]) >= 5, p


def test_grammar_builder_supports_both_surfaces():
    import t2_guided_patch as G
    tools = [{"function": {"name": "t_one"}}]
    for sf, marker in (("hermes", "namefirst"), ("qwen3_xml", "<function=")):
        os.environ["T2_TOOL_SURFACE"] = sf
        G._CACHE.clear()
        g = G.grammar_for_tools(tools)
        assert g and marker in g, (sf, g[:80])


def test_ctx_cap_follows_the_declared_context_length():
    """`_ctx_fits` 의 캡이 모델을 따라가야 한다 — 44,672 는 Qwen2.5 의 값이었다."""
    import importlib
    import t2_gate_patch as GP
    # 소형 캡(44,672)에서는 넘고 대형 캡(131,072)에서는 들어가는 크기:
    #   (자수/3.5) 가 24,456 토큰 초과 ∧ 110,856 토큰 이하 → 85,596자 < N < 388,000자
    long_text = "x" * 150000
    os.environ["T2_MAX_MODEL_LEN"] = "44672"; os.environ["T2_AGENT_MAX_TOKENS"] = "8192"
    small_ok, _ = GP._ctx_fits([], long_text)
    os.environ["T2_MAX_MODEL_LEN"] = "131072"
    big_ok, _ = GP._ctx_fits([], long_text)
    assert big_ok and not small_ok, (small_ok, big_ok)
    importlib.reload  # noqa: B018  (참조만 — 부작용 없음)


def test_launcher_refuses_without_a_profile():
    src = open(os.path.join(HERE, "run_ours_task.sh"), encoding="utf-8").read()
    assert "REFUSING: 모델 프로필이 없다" in src
    assert "x704_surface_preflight.py" in src, "표면형 선발사 검산이 빠졌다"
    assert "--profile" in src



def test_thinking_profiles_leave_room_for_the_answer():
    """사고를 쓰는 프로필은 프로브 상한이 예산 하한(256)의 **2배 이상**이어야 한다.
    같으면 답이 들어갈 자리가 0 이고, 그것이 밤샘런 TRUNC 85건의 전부였다."""
    for p in _profiles():
        kv = _parse(p)
        if "T2_THINK_BUDGET" not in kv:
            continue
        assert int(kv["T2_PROBE_MAX_TOKENS"]) >= 512, os.path.basename(p)


def test_arms_are_single_axis_and_ctl_is_legacy():
    """팔은 **하나의 축만** 바꾸고 `ctl` 은 종전 거동을 명시적으로 고정한다(기본값에 기대지 않는다)."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arms")
    ctl = open(os.path.join(d, "ctl.env"), encoding="utf-8").read()
    vs = open(os.path.join(d, "viewscale.env"), encoding="utf-8").read()
    assert "T2_VIEW_SCALE=off" in ctl and "T2_VIEW_COMPACT_MINTOTAL=60000" in ctl
    assert "T2_VIEW_SCALE=auto" in vs
    # viewscale 은 문턱을 **고정하지 않는다** — 모델을 바꾸면 파생식이 따라와야 하기 때문이다.
    assert not any(l.strip().startswith("export T2_VIEW_COMPACT_MINTOTAL")
                   for l in vs.splitlines())
    # ⛔그리고 **go_stack 의 명시값을 지워야** 파생식이 산다. 안 지우면 팔이 무력화된다
    #   (실측: 첫 발사에서 arm=viewscale 인데 view_mintotal=60000 이 찍혔다).
    assert "unset T2_VIEW_COMPACT_MINTOTAL" in vs and "unset T2_VIEW_MSG_CAP" in vs
    go = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "go_stack.sh"),
              encoding="utf-8").read()
    assert "T2_VIEW_COMPACT_MINTOTAL=60000" in go, "이 전제가 깨지면 위 unset 의 이유가 사라진다"


def test_launcher_accepts_an_arm():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_ours_task.sh"), encoding="utf-8").read()
    assert "--arm" in src and "arms/$ARM.env" in src
    assert "유효 config:" in src, "발사 로그에 유효값을 박아야 두 모델 동시 실행 시 새는 것이 보인다"

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
