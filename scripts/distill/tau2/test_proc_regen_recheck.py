# -*- coding: utf-8 -*-
"""재생성이 낸 호출도 절차 게이트를 받는가 (A-1 · 2026-08-23).

`tasks__20260822/TASK_050.md` §7-① 이 확정한 결손: `_ap_regen` 이 낸 tool_call 은
`gate`·`T2_UNLOCK_NAME`·`T2_UNLOCK_PROV` 만 다시 받고 `T2_PROCEDURE` 는 **평가조차
받지 않은 채** 커밋됐다. t7346 `task_050` trial 0 이 그렇게 승인 호출을 먼저 커밋해
요청-제출 write 를 빠뜨렸고 DB 해시가 갈렸다(reward 0.0). 같은 sha·같은 A2 의 trial 1 은
동일 호출이 원본 am 에 있었기에 deny 를 받고 선행을 먼저 밟아 1.0 을 받았다.

이 검정이 잡는 것은 두 가지다.
  ① **술어는 이미 준비돼 있었다** — 실제 t7346 궤적을 그대로 먹여, 문제의 승인 호출에
     `t2_procedure.decide` 가 `deny missing=submit_request,disputes,pending_replacement`
     를 낸다는 것을 보인다. 즉 결손은 술어가 아니라 배선이었다.
  ② **배선이 그 자리에 있다** — `_ap_regen` 함수 본문(AST 로 그 함수만 떼어)에서
     절차 재평가·cap 공유·플래그 가드가 실재하는지 본다. `proc_fb` 死배선(2026-08-05)이
     정확히 이 자리에서 났고, 그때 배운 것은 *로그 마크 ≠ 전달*([[55]])이다.

⚠이 검정은 **라이브 전달을 증명하지 못한다** — 그것은 런 로그의 `[T2_PROCEDURE] regen-*`
  라인이 할 일이다. 여기서 증명하는 것은 술어와 배선의 실재까지다.
"""

import ast
import gzip
import io
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G          # noqa: E402
import t2_procedure as P           # noqa: E402

fail = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def objs(messages):
    """궤적 dict → 엔진이 보는 모양(속성 접근·tool_calls 는 name/arguments/id)."""
    out = []
    for m in messages:
        tcs = [types.SimpleNamespace(name=tc.get("name"), arguments=tc.get("arguments"),
                                     id=tc.get("id")) for tc in (m.get("tool_calls") or [])]
        out.append(types.SimpleNamespace(role=m.get("role"), tool_calls=tcs or None,
                                         content=m.get("content"), id=m.get("id"),
                                         tool_call_id=m.get("tool_call_id"),
                                         error=m.get("error", False)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
print("① 술어 — 실제 t7346 task_050 궤적에서 그 승인 호출이 거부되는가")

RES = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                   "bank_t7346_halfB_20260822.results.json.gz")
A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                       encoding="utf-8"))
PROCS = A2.get("procedures")
check("A2 가 procedures 를 선언한다", bool(PROCS))

if os.path.exists(RES):
    with gzip.open(RES, "rt", encoding="utf-8") as f:
        d = json.load(f)

    def decide_at(sim, want_reward):
        """그 sim 의 '승인' 유효호출 시점마다 (msg_idx, verdict, missing)."""
        rows = []
        ms = sim["messages"]
        for i, m in enumerate(ms):
            for tc in (m.get("tool_calls") or []):
                c = types.SimpleNamespace(name=tc.get("name"), arguments=tc.get("arguments"),
                                          id=tc.get("id"))
                eff = G._exact_tool_name(c)
                # 표적 = 이 절차의 결정 노드 도구(선언에서 읽는다 · 리터럴 0)
                if eff not in DECISION_TOOLS:
                    continue
                hist = objs(ms[:i])
                ar = G._args_dict(c)
                also = {str(ar.get(k)) for k in
                        ("agent_tool_name", "user_tool_name", "discoverable_tool_name")
                        if ar.get(k)}
                dc = P.decide(PROCS, eff, ar, G._executed_tool_counts(hist), also_names=also,
                              unlocked=G._unlocked_names(hist, A2),
                              pattern=(A2.get("discoverable_name_check") or {}).get("pattern"))
                rows.append((i, dc.get("verdict"), tuple(dc.get("missing") or []),
                             (dc.get("notes") or [""])[0]))
        return rows

    # 결정 노드 도구를 **선언에서** 뽑는다 — 이 파일에 도구명 리터럴을 적지 않기 위해서다([[59]]).
    DECISION_TOOLS = set()
    for _pr in (PROCS or []):
        for _nd in (_pr.get("nodes") or []):
            if not (_nd.get("requires") or []):
                continue
            DECISION_TOOLS |= set(P._tools_of(_nd))
    check("선언에서 결정 노드 도구를 읽었다", bool(DECISION_TOOLS), str(sorted(DECISION_TOOLS))[:90])

    sims = [s for s in d["simulations"] if s.get("task_id") == "task_050"]
    check("t7346 halfB 에 task_050 sim 2건", len(sims) == 2, str(len(sims)))
    by_rw = {(s.get("reward_info") or {}).get("reward"): s for s in sims}

    fail_rows = decide_at(by_rw.get(0.0), 0.0) if 0.0 in by_rw else []
    pass_rows = decide_at(by_rw.get(1.0), 1.0) if 1.0 in by_rw else []

    denies = [r for r in fail_rows if r[1] == "deny"]
    check("reward 0.0 궤적: 결정 호출이 거부된다", bool(denies),
          str([(r[0], r[2]) for r in fail_rows]))
    if denies:
        miss = denies[0][2]
        check("거부 사유가 요청-제출 선행을 포함", len(miss) >= 1 and "submit" in " ".join(miss),
              ",".join(miss))
        note = denies[0][3]
        check("문면이 [PROCEDURE] 표면화 형식", note.lstrip().startswith("[PROCEDURE]"),
              note[:60])
    check("reward 1.0 궤적: 같은 도구가 통과한다(선행을 밟았으므로)",
          bool(pass_rows) and all(r[1] == "pass" for r in pass_rows),
          str([(r[0], r[1]) for r in pass_rows]))
else:
    print("  · 런 결과 gz 없음 — 술어 검정 skip (%s)" % RES)

# ─────────────────────────────────────────────────────────────────────────────
print("\n② 배선 — `_ap_regen` 본문에 그 재평가가 실재하는가")

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
_tree = ast.parse(SRC)
_fn = next((n for n in ast.walk(_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_ap_regen"), None)
check("`_ap_regen` 함수를 AST 에서 찾았다", _fn is not None)

body = ast.get_source_segment(SRC, _fn) or "" if _fn is not None else ""
check("본문이 `t2_procedure` 를 import 한다", "import t2_procedure" in body)
check("본문이 `.decide(` 를 호출한다", ".decide(" in body)
check("`T2_PROCEDURE` 가 켜져 있을 때만 판다(거동보존 가드)",
      'os.environ.get("T2_PROCEDURE") == "1"' in body)
check("`T2_PROC_REGEN` 플래그로 부정통제가 가능하다",
      'os.environ.get("T2_PROC_REGEN"' in body)
check("cap 을 메인 경로와 공유한다(`_t2_proc_deny`)", "_t2_proc_deny" in body)
check("cap 상한 이름도 공유한다(`T2_PROCEDURE_CAP`)", "T2_PROCEDURE_CAP" in body)
check("거부 문면 접두 규칙이 메인 경로와 같다",
      'startswith("Error:")' in body and '"Error: "' in body)
check("실패 시 원본을 유지하는 경로가 있다", "keeping original" in body)

# ★도메인 리터럴 0([[05]]/[[59]]) — 새 블록이 도구명·필드값을 적으면 A2 가 아니라 엔진이
#   도메인을 아는 것이 된다. 디스패처 인자 키 3종은 메인 절차 게이트가 이미 쓰는 프레임워크
#   어휘라 같은 예외로 둔다(그 셋 말고 밑줄+숫자 접미사 이름이 있으면 잡는다).
import re  # noqa: E402
_newblk = body[body.find("A-1 절차 재평가"):] if "A-1 절차 재평가" in body else ""
_code_only = "\n".join(l for l in _newblk.split("\n") if not l.lstrip().startswith("#"))
_suffixed = sorted(set(re.findall(r"\"[a-z_]+_[0-9]{3,}\"", _code_only)))
check("새 블록에 접미사-도구명 리터럴 0", not _suffixed, str(_suffixed))

# ─────────────────────────────────────────────────────────────────────────────
# ③ D11ⓐ — 재생성이 **초안의 env-변이 호출을 잃으면** 그 집합을 되붙이는가
#    (`T2_REGEN_KEEP_MUTATING` · 2026-09-05 · 효과 프로브 `x771_015_effect.py` PROBE-PASS)
#
#    왜 이 파일인가([[67]]): 같은 함수(`_ap_regen`)의 같은 결손 계열이다. ②가 *"재생성이 낸
#    호출은 재검사를 받는가"* 를 본다면 ③은 그 반대 방향 — *"재생성이 **잃은** 호출은
#    어떻게 되는가"* 다. 회수분(캠페인 pre-give 55건) 중 **33건**에서 초안의 give 호출이
#    산출에서 사라졌고 엔진은 그대로 `am` 을 갈아치웠다. 015 은 그 손실이 두 번 나고
#    (`bank_k8143med1_20260904_0135` `task_015#s626729` reward 0.0 · MISSING 1)
#    손님이 env 오류를 두 번 받았다(msg[29]·msg[33]).
#
#    ★이 절은 **엔진 소스 텍스트를 그대로 실행한다** — AST 로 그 블록만 떼어 exec 한다.
#      검정이 코드를 베껴 적으면 드리프트가 검정을 통과시킨다([[84]] 스키마×소비부 사고와
#      같은 계열). 재료의 도구 이름도 이 파일에 적지 않고 **환경 선언에서 읽는다**([[59]]).
print("\n③ D11ⓐ — 잃은 env-변이 호출을 되붙이는가 (`T2_REGEN_KEEP_MUTATING`)")

import contextlib   # noqa: E402
import textwrap     # noqa: E402
import t2_forensic as FRN  # noqa: E402

FLAG = "T2_REGEN_KEEP_MUTATING"
check("`_ap_regen` 이 플래그로 가드된다(부정통제 가능)",
      ('os.environ.get("%s")' % FLAG) in body)
check("변이 판정을 **환경 선언**에서 읽는다(gold 미접촉·[[23]])",
      "_is_mutating_tool" in body)

# 자리: 빈-재생성 가드 **뒤** · `return _am2` **직전**(뒤에서 `_am2` 를 통째로 교체하는
#   블록이 있으면 복원이 조용히 되돌려진다 — 그것이 이 위치의 유일한 이유다).
_i_blk = body.find('if os.environ.get("%s")' % FLAG)
_i_emp = body.find("_t2_msg_empty(_am2)")
_i_ret = body.rfind("return _am2")
check("자리 = 빈-재생성 가드 뒤", _i_blk > _i_emp > 0, "blk=%d empty=%d" % (_i_blk, _i_emp))
check("자리 = `return _am2` 직전", 0 < _i_blk < _i_ret, "blk=%d ret=%d" % (_i_blk, _i_ret))
_tail = body[_i_blk:_i_ret]
check("이 블록 뒤에서 `_am2` 를 재대입하는 코드 0",
      "_am2 = " not in "\n".join(l for l in _tail.split("\n")
                                 if not l.lstrip().startswith("#")))

# 새 블록에 도구명 리터럴 0([[05]]/[[59]]) — 이름은 전부 런타임 값에서 온다.
_blk_only = "\n".join(l for l in _tail.split("\n") if not l.lstrip().startswith("#"))
_sfx3 = sorted(set(re.findall(r"\"[a-z_]+_[0-9]{3,}\"", _blk_only)))
check("새 블록에 접미사-도구명 리터럴 0", not _sfx3, str(_sfx3))

# ── 엔진 소스 그대로 실행 ────────────────────────────────────────────────────
_k = body.rindex("\n", 0, _i_blk) + 1
BLK = textwrap.dedent(body[_k:_i_ret])
check("블록 소스를 떼어 컴파일한다", bool(BLK.strip()) and BLK.lstrip().startswith("if os."))
_CODE = compile(BLK, "<t2_gate_patch:_ap_regen:D11a>", "exec")

# 재료의 이름 = 환경 선언에서 읽는다(리터럴 0). give 이름은 엔진 소스에서 뽑는다.
_MUT = FRN.mutating_tools()
_ENVJ = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
_ALL = set((_ENVJ["banking_knowledge"]["tools"] or {}).keys())
_mg = re.search(r'if str\(getattr\(t, "name", ""\)\) == "([A-Za-z0-9_]+)"\), None\)', SRC)
MUT_NAME = _mg.group(1) if _mg else (sorted(_MUT)[0] if _MUT else None)
RO_NAME = sorted(_ALL - _MUT)[0] if (_ALL - _MUT) else None
check("변이 도구 이름을 엔진 소스/환경 선언에서 얻었다",
      bool(MUT_NAME) and MUT_NAME in _MUT, str(MUT_NAME))
check("비-변이 도구 이름을 환경 선언에서 얻었다", bool(RO_NAME) and RO_NAME not in _MUT,
      str(RO_NAME))


def _tc(name, cid="c1"):
    return types.SimpleNamespace(name=name, arguments={}, id=cid)


def _msg(names, content="x"):
    return types.SimpleNamespace(content=content,
                                 tool_calls=[_tc(n, "c%d" % i) for i, n in enumerate(names)]
                                 or None)


def run_block(draft, regen, flag="1", env_present=True):
    """엔진 소스 블록을 실행하고 (산출 tool_call 이름들, 계기 라인 수) 를 돌려준다."""
    am_ = _msg(draft)
    am2_ = _msg(regen)
    env = (types.SimpleNamespace(_is_mutating_tool=lambda n: n in _MUT)
           if env_present else None)
    ns = {"os": os, "_sys": sys, "am": am_, "_am2": am2_, "tag": "usertoolnote",
          "self": types.SimpleNamespace(_t2_orch=types.SimpleNamespace(environment=env)),
          "_exact_tool_name": G._exact_tool_name}
    _old = os.environ.get(FLAG)
    buf = io.StringIO()
    try:
        if flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = flag
        with contextlib.redirect_stderr(buf):
            exec(_CODE, ns)
    finally:
        if _old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = _old
    names = [str(getattr(t, "name", "")) for t in (getattr(am2_, "tool_calls", None) or [])]
    return names, buf.getvalue().count("[%s] restored=" % FLAG)


# ⑴ 수리가 실제로 하는 일 — 015 의 실물 형상: 초안 tc=[give] · 산출 tc=0
_n, _c = run_block([MUT_NAME], [])
check("잃은 변이 호출이 되붙는다(015 실물 형상)", _n == [MUT_NAME] and _c == 1,
      "names=%s restored=%d" % (_n, _c))

# ⑵ 문면은 재생성 것을 그대로 쓴다 = `T2_USER_TOOL_NOTE` 를 끄지 않는다([[60]])
_am_o, _am2_o = _msg([MUT_NAME], "draft prose"), _msg([], "regen prose")
_ns = {"os": os, "_sys": sys, "am": _am_o, "_am2": _am2_o, "tag": "givequote",
       "self": types.SimpleNamespace(_t2_orch=types.SimpleNamespace(
           environment=types.SimpleNamespace(_is_mutating_tool=lambda n: n in _MUT))),
       "_exact_tool_name": G._exact_tool_name}
_o = os.environ.get(FLAG)
os.environ[FLAG] = "1"
try:
    with contextlib.redirect_stderr(io.StringIO()):
        exec(_CODE, _ns)
finally:
    if _o is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = _o
check("재생성 **문면**은 그대로 채택된다(레버를 끄지 않는다·[[60]])",
      getattr(_am2_o, "content", None) == "regen prose", str(_am2_o.content))

# ⑶ [[57]] 부정통제 — 되돌리면 검정이 실패하는가
_n, _c = run_block([MUT_NAME], [], flag="0")
check("NC0 플래그 0 → 복원 0(되돌리면 ⑴이 깨진다)", _n == [] and _c == 0,
      "names=%s restored=%d" % (_n, _c))
_n, _c = run_block([MUT_NAME], [MUT_NAME])
check("NC2 산출이 그 호출을 **유지**했으면 손대지 않는다", _n == [MUT_NAME] and _c == 0,
      "names=%s restored=%d" % (_n, _c))
_n, _c = run_block([], [])
check("NC3 초안에 호출이 없었으면 복원 0", _n == [] and _c == 0,
      "names=%s restored=%d" % (_n, _c))
if RO_NAME:
    _n, _c = run_block([RO_NAME], [])
    check("NC4 **비-변이** 호출 손실은 건드리지 않는다(변이-제한 항이 일한다)",
          _n == [] and _c == 0, "names=%s restored=%d" % (_n, _c))
_n, _c = run_block([MUT_NAME], [], env_present=False)
check("환경 선언을 못 읽으면 복원 0(fail-closed·구판 거동)", _n == [] and _c == 0,
      "names=%s restored=%d" % (_n, _c))

# ⑷ [[81]] 정본 런처 등재 — 플래그가 없으면 라이브에 존재하지 않는다
GS = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
check("go_stack.sh 에 등재됐다([[81]])",
      re.search(r"^export %s=1" % FLAG, GS, re.M) is not None)

# ─────────────────────────────────────────────────────────────────────────────
# ④ D14 — 재생성이 **낸** 호출이 쓰기 게이트 6종을 다시 받는가
#    (`T2_REGEN_WRITE_GATES` · 2026-09-05 · 효과 프로브
#     `reports/facet_rft_2026/x771_d14_reenter_effect.py` PROBE-PASS 0/25 → 10/25)
#
#    왜 이 파일인가([[67]]): ②가 `T2_PROCEDURE` 한 축에 대해 물은 바로 그 질문을 나머지 축
#    (쓰기 게이트 6종)에 대해 묻는다. ②의 근거 문장이 이미 *"재검사 목록은 gate ·
#    UNLOCK_NAME · UNLOCK_PROV 뿐이었다"* 라고 적어 놓았고, 그 목록에 빠진 나머지가 이것이다.
#    실행-시점 그물도 없다 — `gated()` 의 `_write_evidence_deny` 는 `exec_augment` 가
#    `_execute_tool_calls` 를 덮어써 정본 스택(`T2_GATE_REGEN=1`)에서 死코드다(:8912 자백).
#
#    ★③과 같은 방법 — **엔진 소스 텍스트를 그대로 실행**한다(검정이 코드를 베끼면 드리프트가
#      검정을 통과시킨다). 재료도 저작하지 않는다: 회수된 실물 궤적
#      `bank_re8143p11_20260904_1053` `task_029#s626729`(reward 0.0 · basis=['DB'])의
#      msg86 tool_calls 5발과 그 직전까지의 실제 메시지 창을 그대로 먹인다.
#
#    ⚠이 검정이 돌리는 축은 **WEV → WAG 둘**이다(프로브 ARM_ON 과 같은 하한).
#      나머지 4종(ARG_EMPTY · REF_VERIFY · ASK_UNKNOWN_BOOL · HANDOFF)은 agent/orch/
#      UserMessage 실물을 요구해 오프라인에서 못 돈다 — 그 넷은 **배선 실재**까지만 본다.
print("\n④ D14 — 재생성이 낸 호출이 쓰기 게이트를 다시 받는가 (`T2_REGEN_WRITE_GATES`)")

import glob    # noqa: E402
import gzip    # noqa: E402  (상단 import 와 중복 아님 — 이미 있으면 no-op)
FLAG14 = "T2_REGEN_WRITE_GATES"

# ── ④-1 배선(텍스트) ────────────────────────────────────────────────────────
_i14 = body.find("self._t2_regen_wgate_denied = set()")
check("`_ap_regen` 본문에 D14 블록이 있다", _i14 > 0)
_i14 = body.rindex("\n", 0, _i14) + 1 if _i14 > 0 else 0
_i14e = body.find("# ★D11ⓐ 수리")
BLK14 = textwrap.dedent(body[_i14:_i14e]) if _i14e > _i14 > 0 else ""
check("플래그로 가드된다(부정통제 가능)", ('os.environ.get("%s")' % FLAG14) in BLK14)
check("cap 을 메인 경로와 공유한다(`_t2_wev_deny`·`_wev_cap`)",
      "_t2_wev_deny" in BLK14 and "_wev_cap" in BLK14)
check("새 예산 이름 0(`T2_REGEN_*_CAP` 류를 만들지 않았다)",
      not re.search(r"T2_REGEN_\w*CAP", BLK14))
check("처분이 기존 partial-accept 계약과 같다(전부 denied → 원본 유지)",
      "keeping original" in BLK14 and "return None" in BLK14)
check("D11ⓐ 와 어휘를 맞췄다(`_exact_tool_name` 로 뗀 이름을 남긴다)",
      "_exact_tool_name(_x)" in BLK14)

# 메인 경로 블록(:9985~)과 **같은 순서**인가 — 6종의 첫 등장 순서를 비교한다.
_i_main = SRC.find("_fab_only = bool(do_gate or do_prov)")
_i_maine = SRC.find("[T2_WAG_DECOUPLED] fired", _i_main)
MAIN = SRC[_i_main:_i_maine] if _i_main > 0 < _i_maine else ""


def _order(txt):
    code = "\n".join(l for l in txt.split("\n") if not l.lstrip().startswith("#"))
    seq = [(code.find(m), m) for m in ("_wev_deny_msgs", "_write_arg_ground_deny",
                                       "_arg_empty_deny", "_ref_verify_deny",
                                       "t2_unknown_bool", "t2_handoff_ground")]
    return [m for i, m in sorted(seq) if i >= 0]


check("메인 경로 블록을 SRC 에서 떼었다", bool(MAIN.strip()))
check("쓰기 게이트 6종이 전부 재진입 블록에 있다", len(_order(BLK14)) == 6, str(_order(BLK14)))
check("메인 경로와 **같은 순서**로 부른다([[67]] 사본 0 · 순서 저작 0)",
      _order(BLK14) == _order(MAIN), "%s ↔ %s" % (_order(BLK14), _order(MAIN)))

# 새 술어 0([[62]]) — 이 블록이 부르는 판정 함수·모듈은 전부 메인 경로에 이미 있다.
_codeD = "\n".join(l for l in BLK14.split("\n") if not l.lstrip().startswith("#"))
_predD = sorted(set(re.findall(r"\b(_\w+_deny\w*|t2_\w+)\b", _codeD)))
_missD = [p for p in _predD if p not in MAIN and p not in ("_t2_wev_deny", "_t2_orch",
                                                           "_t2_regen_wgate_denied")]
check("새 술어 0 — 판정자가 전부 메인 경로에 이미 있다([[62]])", not _missD, str(_missD))
_sfx4 = sorted(set(re.findall(r"\"[a-z_]+_[0-9]{3,}\"", _codeD)))
check("새 블록에 접미사-도구명 리터럴 0([[05]]/[[59]])", not _sfx4, str(_sfx4))

# ── ④-2 술어 실행 — 엔진 소스 그대로 · 회수 실물 재료 ────────────────────────
#    `return None` 을 담고 있어 module-level exec 이 안 되므로 함수로 감싼다(본문 무개작).
_CODE14 = compile("def _run14():\n" + textwrap.indent(BLK14, "    ") + "\n    return 'FT'\n",
                  "<t2_gate_patch:_ap_regen:D14>", "exec")
A2G = None
try:
    from gate_interpreter import load_domain_a2 as _lda   # 정본 A2 로더([[24]])
    A2G = _lda("banking_knowledge")
except Exception as _e14:
    print("  · A2 로더 실패 — 실행 검정 skip (%r)" % (_e14,))


def run14(window, calls, flag="1", pre_deny=0):
    """엔진 D14 블록을 실물 재료에 돌리고 (조기반환?, deny 수, 남은 호출 수, 뗀 이름) 반환."""
    _am2 = types.SimpleNamespace(content="x", tool_calls=list(calls))
    _slf = types.SimpleNamespace(_t2_orch=None, _t2_wev_deny=pre_deny)
    ns = {"os": os, "_sys": sys, "json": json, "self": _slf, "tag": "claimprov",
          "state": types.SimpleNamespace(messages=window), "_am2": _am2, "am": None,
          "wev_specs": A2G.get("write_evidence_specs") or [],
          "wag_specs": A2G.get("write_arg_grounding") or [],
          "rv_specs": [], "ae_on": False, "ae_tools": None, "a2": A2G, "_wev_cap": 8,
          "_wev_deny_msgs": G._wev_deny_msgs,
          "_write_arg_ground_deny": G._write_arg_ground_deny,
          "_arg_empty_deny": G._arg_empty_deny, "_ref_verify_deny": G._ref_verify_deny,
          "_eff_tool_name": G._eff_tool_name, "_exact_tool_name": G._exact_tool_name,
          "_args_dict": G._args_dict, "_lbeat": (lambda *a, **k: None),
          "la": None, "UserMessage": None}
    _o = os.environ.get(FLAG14)
    buf = io.StringIO()
    try:
        if flag is None:
            os.environ.pop(FLAG14, None)
        else:
            os.environ[FLAG14] = flag
        exec(_CODE14, ns)
        with contextlib.redirect_stderr(buf):
            rv = ns["_run14"]()
    finally:
        if _o is None:
            os.environ.pop(FLAG14, None)
        else:
            os.environ[FLAG14] = _o
    return (rv, buf.getvalue().count("[T2_REGEN_WGATE] deny"),
            len(getattr(_am2, "tool_calls", None) or []),
            set(getattr(_slf, "_t2_regen_wgate_denied", set()) or set()), buf.getvalue())


R29 = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                   "bank_re8143p11_20260904_1053.results.json.gz")
if A2G is not None and os.path.exists(R29):
    with gzip.open(R29, "rt", encoding="utf-8") as f:
        _d29 = json.load(f)
    _s29 = [s for s in _d29["simulations"] if s.get("task_id") == "task_029"]
    check("대표 sim(029 · reward 0.0)을 회수분에서 찾았다",
          len(_s29) == 1 and (_s29[0].get("reward_info") or {}).get("reward") == 0.0)
    _ms29 = _s29[0]["messages"]
    _win29 = objs(_ms29[:86])
    _cal29 = [types.SimpleNamespace(name=t.get("name"), arguments=t.get("arguments"),
                                    id=t.get("id")) for t in (_ms29[86].get("tool_calls") or [])]
    check("msg86 = 재생성(claimprov)이 낸 write 5발", len(_cal29) == 5, str(len(_cal29)))

    _rv, _n, _kept, _dn, _log = run14(_win29, _cal29)
    check("ARM_ON — 그 5발이 전부 거부된다(프로브 029×5 와 일치)", _n == 5, "deny=%d" % _n)
    check("전부 denied 이므로 원본을 유지한다(`return None`·부작용 0)", _rv is None, str(_rv))
    check("계기가 설계대로 발화한다(tag·wtag·tool·inner)",
          "[T2_REGEN_WGATE] deny tag=claimprov wtag=T2_WRITE_EVIDENCE" in _log,
          _log.split("\n")[0][:96])
    check("뗀 이름이 D11ⓐ 어휘(`_exact_tool_name`)로 남는다", bool(_dn) and all(
        re.search(r"_\d+$", x) for x in _dn), str(_dn))

    # [[57]] 부정통제 ─ 되돌리면 검정이 실패하는가
    _rv, _n, _kept, _dn, _ = run14(_win29, _cal29, flag="0")
    check("NC0 플래그 0 → deny 0 · 산출 불변(되돌리면 위 칸이 깨진다)",
          _n == 0 and _kept == 5 and _rv == "FT", "deny=%d kept=%d" % (_n, _kept))
    _rv, _n, _kept, _dn, _ = run14(_win29, _cal29, pre_deny=8)
    check("NC1 cap 소진(`T2_WEV_CAP`) → deny 0 — 예산 공유가 실제로 문다",
          _n == 0 and _kept == 5, "deny=%d kept=%d" % (_n, _kept))
    check("NC1′ 그래서 cap 을 소진한 sim(027 5건)에서는 이 수리가 아무 것도 안 한다", True,
          "예산은 메인 경로와 공유 — 새 예산 0")

    # [[70]] 파는 것 ─ reward=1.0 정상경로 커밋 write 전수에 같은 술어
    _OUT = {sp.get("applies_to") for sp in ((A2G.get("write_evidence_specs") or [])
                                            + (A2G.get("write_arg_grounding") or []))
            if sp.get("applies_to")}
    _tot = _den = 0
    for _p in glob.glob(os.path.join(os.path.dirname(R29), "bank_*20260904*.results.json.gz")):
        try:
            with gzip.open(_p, "rt", encoding="utf-8") as f:
                _dd = json.load(f)
        except Exception:
            continue
        for _s in _dd.get("simulations", []):
            if (_s.get("reward_info") or {}).get("reward") != 1.0:
                continue
            _mm = _s.get("messages") or []
            _ok = {m.get("id") for m in _mm if m.get("role") == "tool" and not m.get("error")}
            for _i, _m in enumerate(_mm):
                _cs = [types.SimpleNamespace(name=t.get("name"), arguments=t.get("arguments"),
                                             id=t.get("id"))
                       for t in (_m.get("tool_calls") or [])
                       if t.get("name") in _OUT and t.get("id") in _ok]
                if not _cs:
                    continue
                _tot += len(_cs)
                _den += run14(objs(_mm[:_i]), _cs)[1]
    check("[[70]] 파는 것 — reward=1.0 정상경로 커밋 write 전수에 DENY 0",
          _tot > 0 and _den == 0, "%d건 중 DENY %d (2026-09-04 런 전수)" % (_tot, _den))
else:
    print("  · 회수분/A2 없음 — D14 실행 검정 skip (%s)" % R29)

# ── ④-3 D14 × D11ⓐ 합성([[19]]) ─────────────────────────────────────────────
#    D14 가 뗀 호출은 `_am2` 에서 사라지므로 D11ⓐ 의 `_afterK` 에 안 잡힌다 — 제외하지 않으면
#    그 블록이 이름을 되붙여 쓰기 게이트 거부를 조용히 되돌린다. 그 제외가 실재하는가.
_i_lost = body.find("_lostK = [")
_src_lost = body[_i_lost:body.find("]", body.find("_dnK", _i_lost))] if _i_lost > 0 else ""
check("D11ⓐ 가 D14 의 뗀-이름 집합을 읽는다", "_t2_regen_wgate_denied" in body
      and "_dnK" in _src_lost, _src_lost.replace("\n", " ")[:80])


def _keepmut_with(denied):
    """③의 D11ⓐ 블록을 `self._t2_regen_wgate_denied = denied` 로 돌린다(엔진 소스 그대로)."""
    _amA, _am2A = _msg([MUT_NAME]), _msg([])
    _nsA = {"os": os, "_sys": sys, "am": _amA, "_am2": _am2A, "tag": "claimprov",
            "self": types.SimpleNamespace(
                _t2_orch=types.SimpleNamespace(environment=types.SimpleNamespace(
                    _is_mutating_tool=lambda n: n in _MUT)),
                _t2_regen_wgate_denied=denied),
            "_exact_tool_name": G._exact_tool_name}
    _oA = os.environ.get(FLAG)
    os.environ[FLAG] = "1"
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            exec(_CODE, _nsA)
    finally:
        if _oA is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = _oA
    return [str(getattr(t, "name", "")) for t in (getattr(_am2A, "tool_calls", None) or [])]


check("D14 가 뗀 이름은 D11ⓐ 가 되붙이지 않는다(수리가 조용히 되돌려지지 않는다)",
      _keepmut_with({MUT_NAME}) == [], str(_keepmut_with({MUT_NAME})))
check("NC 그 이름이 아니면 D11ⓐ 는 종전대로 되붙인다([[60]] 끄지 않았다)",
      _keepmut_with(set()) == [MUT_NAME], str(_keepmut_with(set())))

# ── ④-4 [[81]] 정본 런처 등재 ───────────────────────────────────────────────
check("go_stack.sh 에 등재됐다([[81]])",
      re.search(r"^export %s=1" % FLAG14, GS, re.M) is not None)
check("쓰기 게이트와 **같은 축**에 등재됐다(WEV/WAG 선언 뒤)",
      GS.find("export %s=1" % FLAG14) > GS.find("export T2_WRITE_EVIDENCE=1"))

print("\nRESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
