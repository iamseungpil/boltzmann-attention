# -*- coding: utf-8 -*-
r"""R4 — `T2_WRITE_ARG_ENUM` 의 sim 당 상한이 fail-closed 를 fail-open 으로 뒤집던 자리.

## 결함 (refute C1 · CONFIRMED · falsifier 3종 전부 돌아감)

구판은 **블록 전체**를 상한으로 잠갔다 —

    if (os.environ.get("T2_WRITE_ARG_ENUM") == "1" and _ens
            and getattr(self, "_t2_enum_deny", 0)
            < int(os.environ.get("T2_WRITE_ARG_ENUM_CAP", "3"))):

상한이 소진되면 게이트가 침묵하고, **우리가 이미 집합 밖이라고 판정한 바로 그 값**이 다음
시도에 통과해 DB 를 바꾼다. 실물 `bank_t7296_treat_20260815p|task_071#s554706` — turn 22·34 에
같은 값을 두 번 deny 하고, gold 3행을 MATCHED 한 뒤, msg41 에 그 값으로 계좌가 열려 reward 0.
전 코퍼스 로그 455개: enum deny 164줄/92 sim 중 **20 sim 이 상한 도달**, 그중 4 sim 에서 집합
밖 값이 이후 성공, **2 sim 에서 같은 값**이 성공.

## 수리 = 상한이 세는 **단위**를 바꾼다 (새 레버 0 · 새 플래그 0 · [[62]])

  ⓐ 원장(`(그룹, 정규화 값)` 집합)에 있는 값 = 횟수와 무관하게 **계속 거절**.
  ⓑ 상한 소진 + **처음 보는 값** = 종전 그대로 **fail-open** — livelock 탈출구는 살아 있다.

## 이 검정이 재는 것 (모델 0 · env 0 · 오프라인)

`t2_gate_patch.py` 의 **실제 배송 소스**에서 게이트 블록을 통째로 떼어 함수로 감싸 돌린다
(사본 금지·[[67]]). 도메인은 **합성**이다 — 상품명·태스크 id 가 하나도 안 들어가는 것이
곧 일반화 시험이다(같은 술어가 어느 축에서나 같은 모양으로 돈다).

  ⓟ 양성대조 = 구판 형상(상한 소진 시 블록 미진입)에서 **이미 거절한 값이 통과**한다.
  ⓝ 수리 후   = 같은 상태에서 그 값이 **계속 거절**되고 [[64]] 두 요소가 문면에 있다.
  ⓒ 부정통제  = ①처음 보는 값은 상한 뒤에도 통과(fail-open 보존) ②집합 內은 무발화
                ③상한 전 거동 불변 ④그룹이 다르면 원장이 안 잡는다 ⑤명단이 나중에 그
                값을 담으면 차단이 저절로 풀린다 ⑥레버 OFF 면 무발화 ⑦명단이 비면 무발화
                ⑧상한이 세는 단위 = 처음 보는 값(같은 값 재제출은 상한을 안 먹는다).
"""
import io
import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP  # noqa: E402

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + str(extra)) if extra else ""))


# ══════════════════════════════════════════════════════════════════════════════
# 0. 배송 소스에서 블록을 떼어 **그대로** 돌린다 (사본 0)
# ══════════════════════════════════════════════════════════════════════════════
_m = re.search(r"( +en_fb = None\n +_ens = \(a2 or \{\}\)\.get\(\"write_arg_enum\".*?)"
               r"\n +# ★결정-선행 write", SRC, re.S)
chk("배송 소스에서 게이트 블록을 떼어냈다", _m is not None)
BLOCK = textwrap.dedent(_m.group(1)) if _m else "en_fb = None"
FN_SRC = ("def _drive(self, am, a2, state):\n"
          + textwrap.indent(BLOCK, "    ") + "\n    return en_fb\n")
NS = dict(vars(GP))
NS["_sys"] = sys      # `unified` 안의 지역 임포트(`import sys as _sys`) — 격리에도 그대로 준다
exec(compile(FN_SRC, "<t2_gate_patch:write_arg_enum>", "exec"), NS)  # noqa: S102
drive = NS["_drive"]


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Call(object):
    """엔진이 보는 최소 형상(도메인 무관) — name + arguments(JSON 문자열)."""

    def __init__(self, name, args):
        self.id = "tc0"
        self.name = name
        self.arguments = json.dumps(args)


# ── 합성 도메인 (상품명·태스크 id 0 — 이름이 아니라 **형상**만 쓴다) ──────────────
TOOL = "write_thing_0000"
SPEC = {"applies_to": TOOL, "arg": "klass", "group_arg": "kind",
        "group_map": {"kind_a": "grp_a", "kind_b": "grp_b"},
        "feedback": ("Error: {val} is not one of the {group} names on file for {arg}. "
                     "Names on file: {candidates}. Use one of those exactly.")}
A2 = {"write_arg_enum": [SPEC],
      "policy_ontology": {"doc_index": {
          "grp_a": {"alpha_one_thing": 1, "beta_two_thing": 1},
          "grp_b": {"alpha_one_thing": 1}}}}
IN_SET = "Alpha One Thing"        # 기계 전개된 표시명(엔진이 만든다·리터럴 아님)
OUT_SET = "Alpha One"             # 집합 밖(모델이 흔히 내는 접미사 탈락형)
STATE = _Obj(messages=[])


def call(klass, kind="kind_a", tool=TOOL):
    # 엔진은 `arg`·`group_arg` 를 **중첩 `arguments`** 에서 읽는다(디스패처 형상).
    return _Obj(tool_calls=[_Call(tool, {"arguments": {"kind": kind, "klass": klass}})])


def fresh(deny=0, seen=None):
    a = _Obj()
    if deny:
        a._t2_enum_deny = deny
    if seen is not None:
        a._t2_enum_rejected = set(seen)
    return a


ENV = {"T2_WRITE_ARG_ENUM": "1", "T2_WRITE_ARG_ENUM_CAP": "3",
       "T2_ARG_AXIS": "0", "T2_VERDICT_GATE": "0"}
for k, v in ENV.items():
    os.environ[k] = v
CAP = int(ENV["T2_WRITE_ARG_ENUM_CAP"])


def run(agent, klass, kind="kind_a", a2=A2):
    return drive(agent, call(klass, kind), a2, STATE)


# ══════════════════════════════════════════════════════════════════════════════
print("[ⓟ 양성대조] 구판 형상 — 상한이 블록 전체를 잠근다")
# ══════════════════════════════════════════════════════════════════════════════
# 구판을 **손으로 베끼지 않는다**: 배송 본문은 그대로 두고 바깥 가드 한 줄만 되돌린다
# (그 한 줄이 구판의 전부였다 — refute C1 앵커 축자).
_OLD_GUARD = ('if (os.environ.get("T2_WRITE_ARG_ENUM") == "1" and _ens\n'
              '        and getattr(self, "_t2_enum_deny", 0)\n'
              '        < int(os.environ.get("T2_WRITE_ARG_ENUM_CAP", "3"))):')
_NEW_GUARD = 'if os.environ.get("T2_WRITE_ARG_ENUM") == "1" and _ens:'
chk("ⓟ 배송 소스에 현재 가드가 정확히 한 번 있다", BLOCK.count(_NEW_GUARD) == 1)
OLD_BLOCK = BLOCK.replace(_NEW_GUARD, _OLD_GUARD, 1)
_NS_OLD = dict(vars(GP))
_NS_OLD["_sys"] = sys
exec(compile("def _drive_old(self, am, a2, state):\n"     # noqa: S102
             + textwrap.indent(OLD_BLOCK, "    ") + "\n    return en_fb\n",
             "<t2_gate_patch:write_arg_enum:OLD>", "exec"), _NS_OLD)
drive_old = _NS_OLD["_drive_old"]

# 배선 확인 — 구판도 상한 **전**에는 정상적으로 막는다(스텁이 죽어서 조용한 게 아니다)
chk("ⓟ 배선: 구판도 상한 전에는 집합 밖 값을 막는다",
    drive_old(fresh(), call(OUT_SET), A2, STATE) is not None)
# 실물 궤적(t7296|071)의 형상을 그대로 태운다 — 집합 밖 값 세 개를 거절해 상한을 소진하고,
# 그중 **첫 값을 재제출**한다. 손으로 상태를 심지 않는다(자연 재현).
SEQ = [OUT_SET, "Other Name A", "Other Name B"]
_old_agent = fresh()
_old_denied = [drive_old(_old_agent, call(v), A2, STATE) is not None for v in SEQ]
chk("ⓟ 구판: 집합 밖 값 3개를 거절해 상한을 소진한다",
    all(_old_denied) and getattr(_old_agent, "_t2_enum_deny", 0) == CAP)
chk("ⓟ 구판: 상한 소진 후 **이미 거절한 그 값**이 통과한다(= fail-open · DB 가 바뀐다)",
    drive_old(_old_agent, call(OUT_SET), A2, STATE) is None)

# ══════════════════════════════════════════════════════════════════════════════
print("\n[ⓝ 수리 후] 같은 상태에서 그 값은 계속 거절된다")
# ══════════════════════════════════════════════════════════════════════════════
_a = fresh()
_new_denied = [run(_a, v) is not None for v in SEQ]      # ⓟ 와 **같은 순서·같은 값**
chk("ⓝ 같은 순서로 상한을 소진한다(비교 가능성)",
    all(_new_denied) and getattr(_a, "_t2_enum_deny", 0) == CAP)
_fb = run(_a, OUT_SET)
chk("ⓝ 상한 소진 + 원장에 있는 값 = deny", _fb is not None)
_body = (_fb or (None, ""))[1]
chk("ⓝ [[64]] 무엇이 틀렸나: 집합 밖 + **이미 거절한 값**이라 또 거절된다",
    "not one of the" in _body and "already rejected" in _body
    and "refused again rather than written" in _body)
chk("ⓝ [[64]] 무엇을 하면 풀리나: 명단 + 조회 경로",
    IN_SET in _body and "look the" in _body and "up with a read tool" in _body)
chk("ⓝ 채널 마크가 문면 머리에 남는다(계기 오염 0·[[25]])",
    _body.startswith(SPEC["feedback"][:6]))
chk("ⓝ 재제출 거절은 상한을 **더 먹지 않는다**(단위 = 처음 보는 값)",
    getattr(_a, "_t2_enum_deny", 0) == CAP)

# 실제로 두 번, 세 번 재제출해도 계속 막힌다(livelock 이 아니라 **일관성**이다)
_again2 = run(_a, OUT_SET)
_again3 = run(_a, OUT_SET.upper() + "   ")     # 대소문자·공백만 바꾼 재제출
chk("ⓝ 반복 재제출도 계속 deny", _again2 is not None and _again3 is not None)
chk("ⓝ 정규화(공백 접기 + casefold)로 표기만 바꾼 재제출도 같은 값으로 본다",
    GP._enum_seen_key("grp_a", OUT_SET)
    == GP._enum_seen_key("grp_a", "  " + OUT_SET.upper() + " "))

# ══════════════════════════════════════════════════════════════════════════════
print("\n[ⓒ 부정통제] 넓히지 않았다는 증거")
# ══════════════════════════════════════════════════════════════════════════════
# ① livelock 탈출구 보존 — 상한 소진 + **처음 보는** 집합 밖 값은 종전대로 통과
_a1 = fresh(deny=CAP, seen=[GP._enum_seen_key("grp_a", OUT_SET)])
chk("ⓒ① 상한 소진 + 처음 보는 값 = 종전대로 fail-open(우리 명단이 인질을 잡지 않는다)",
    run(_a1, "Some Other Name") is None)
chk("ⓒ① 그 통과는 원장도 상한도 건드리지 않는다",
    getattr(_a1, "_t2_enum_deny", 0) == CAP
    and GP._enum_seen_key("grp_a", "Some Other Name")
    not in getattr(_a1, "_t2_enum_rejected", set()))

# ② 집합 內은 원장과 무관하게 무발화 (선택이 옳은지 우리가 판정하지 않는다·[[62]])
_a2 = fresh(deny=CAP, seen=[GP._enum_seen_key("grp_a", IN_SET)])
chk("ⓒ② 집합 內 값은 원장에 있어도 통과(엔진이 고르지 않는다)", run(_a2, IN_SET) is None)

# ③ 상한 **전** 거동 불변 — 처음 보는 집합 밖 값은 deny + 상한 1 증가 + 원장 등재
_a3 = fresh()
_f3 = run(_a3, OUT_SET)
chk("ⓒ③ 상한 전: 처음 보는 집합 밖 값 = 종전대로 deny",
    _f3 is not None and getattr(_a3, "_t2_enum_deny", 0) == 1
    and GP._enum_seen_key("grp_a", OUT_SET) in getattr(_a3, "_t2_enum_rejected", set()))
chk("ⓒ③ 그 문면에는 재제출 안내가 **없다**(첫 거절은 종전 문면 그대로)",
    "already rejected" not in _f3[1] and IN_SET in _f3[1])

# ④ 그룹 단위 — 같은 문자열이라도 다른 군에서는 원장이 안 잡는다
_a4 = fresh(deny=CAP, seen=[GP._enum_seen_key("grp_a", OUT_SET)])
chk("ⓒ④ 다른 군의 같은 문자열은 원장 밖(판정은 그 군의 명단에 대해서만 참이다)",
    run(_a4, OUT_SET, kind="kind_b") is None)

# ⑤ 명단이 나중에 그 값을 담으면 차단이 **저절로** 풀린다(원장이 명단을 이기지 않는다)
_a5 = fresh(deny=CAP, seen=[GP._enum_seen_key("grp_a", IN_SET)])
chk("ⓒ⑤ `_val in _names` 를 매번 먼저 본다 = 명단이 자라면 차단 해제",
    run(_a5, IN_SET) is None)

# ⑥ 레버 OFF = 무발화 (기본 OFF 원칙)
os.environ["T2_WRITE_ARG_ENUM"] = "0"
_a6 = fresh(deny=CAP, seen=[GP._enum_seen_key("grp_a", OUT_SET)])
chk("ⓒ⑥ 레버 OFF 면 원장이 있어도 무발화", run(_a6, OUT_SET) is None)
os.environ["T2_WRITE_ARG_ENUM"] = "1"

# ⑦ 명단이 비면 무발화 (fail-open: 모르면 막지 않는다·[[25]])
_A2_EMPTY = {"write_arg_enum": [SPEC],
             "policy_ontology": {"doc_index": {"grp_a": {"_general_": 1}}}}
_a7 = fresh(deny=CAP, seen=[GP._enum_seen_key("grp_a", OUT_SET)])
chk("ⓒ⑦ 표시명 후보가 비면 원장이 있어도 무발화(빈 명단으로 deny 하지 않는다·[[64]])",
    run(_a7, OUT_SET, a2=_A2_EMPTY) is None)

# ⑧ 상한이 세는 단위 = **처음 보는 값** — 같은 값을 세 번 내도 1 만 먹는다
_a8 = fresh()
for _ in range(3):
    run(_a8, OUT_SET)
chk("ⓒ⑧ 같은 값 3회 거절 = 상한 소비 1(단위가 '거절 횟수' 가 아니다)",
    getattr(_a8, "_t2_enum_deny", 0) == 1)
for _n in ("N1", "N2", "N3", "N4"):
    run(_a8, _n)
chk("ⓒ⑧ 처음 보는 값 3개까지 먹고 4번째는 통과(상한이 살아 있다)",
    getattr(_a8, "_t2_enum_deny", 0) == CAP and run(_a8, "N5") is None)
chk("ⓒ⑧ 상한을 다 쓴 뒤에도 원장의 값은 계속 deny", run(_a8, OUT_SET) is not None)

# ══════════════════════════════════════════════════════════════════════════════
print("\n[배선·일반화] 소스 술어 검사")
# ══════════════════════════════════════════════════════════════════════════════
_blk = BLOCK
_code = "\n".join(l for l in _blk.split("\n") if not l.lstrip().startswith("#"))
chk("바깥 가드에서 상한이 빠졌다(블록 전체를 잠그지 않는다)",
    re.search(r'if os\.environ\.get\("T2_WRITE_ARG_ENUM"\) == "1" and _ens:', _code) is not None)
chk("상한은 남아 있다(livelock 방지)", "T2_WRITE_ARG_ENUM_CAP" in _code and "_encap_open" in _code)
chk("원장은 정본 술어 하나만 쓴다(사본 0·[[67]])",
    _code.count("_enum_seen_key(") == 1 and SRC.count("def _enum_seen_key(") == 1)
chk("이웃 레버 거동 보존: ARG_AXIS·VERDICT_GATE 가 종전 상한 조건을 유지한다",
    _code.count("and _encap_open") >= 2)
chk("선택기 0 (argmax/max/min/sorted[...] 없음·[[62]] ④)",
    not re.search(r"\bargmax\b|\bmax\s*\(|\bmin\s*\(|sorted\s*\([^)]*\)\s*\[", _code))
_added = "\n".join(l for l in _code.split("\n")
                   if "_seen" in l or "_rkey" in l or "_again" in l or "_rep_pre" in l)
chk("일반화: 새 술어에 태스크 id·상품명·도메인 낱말 0([[05]]·[[70]] 조건부 금지)",
    not re.search(r"task_\d|account|card|bank|savings|checking|retail|airline|telecom",
                  _added, re.I), _added.count("\n") + 1)
chk("새 환경 플래그 0 (레지스트리 래칫 보존)",
    set(re.findall(r'environ\.get\(\s*["\'](T2_[A-Z_0-9]+)["\']', _code))
    == {"T2_WRITE_ARG_ENUM", "T2_WRITE_ARG_ENUM_CAP", "T2_ARG_AXIS", "T2_VERDICT_GATE",
        "T2_VERDICT_GATE_CAP"})
chk("계기: 재제출 거절이 다른 마크로 찍힌다(deny 줄 수로 상한 역산 금지)",
    "deny(재제출)" in _blk)

# 실물 A2 왕복 — 배송 선언으로도 같은 형상이 돈다(도메인 값은 단정하지 않는다·[[23]])
_live = 0
for _lay in ("gate", "specific"):
    _p = os.path.join(HERE, "a2", "banking_knowledge.%s.json" % _lay)
    if not os.path.exists(_p):
        continue
    _A = json.load(io.open(_p, encoding="utf-8"))
    _sp = (_A.get("write_arg_enum") or [None])[0]
    if not _sp:
        continue
    _gm = _sp.get("group_map") or {}
    _di = ((_A.get("policy_ontology") or {}).get("doc_index") or {})
    _gv = next((g for g, grp in _gm.items() if GP._display_slugs(_di.get(grp) or {})), None)
    if not _gv:
        continue
    _bogus = "Zz Nonexistent Zz"      # 어느 명단에도 없는 형상(제품명 아님)
    _aw = _sp.get("applies_when") or {}
    _outer = {"arguments": {_sp.get("group_arg"): _gv, _sp.get("arg"): _bogus}}
    if _aw.get("arg"):
        _outer[_aw["arg"]] = str(_aw.get("prefix") or "")
    _ag = _Obj()
    _c = _Obj(tool_calls=[_Call(str(_sp.get("applies_to")), _outer)])
    _f1 = drive(_ag, _c, _A, STATE)
    _ag._t2_enum_deny = CAP           # 상한 소진 상태로 밀어 놓고 같은 값을 재제출
    _f2 = drive(_ag, _c, _A, STATE)
    _live += 1
    chk("실물 A2(%s): 상한 소진 후에도 같은 값은 계속 deny" % _lay,
        _f1 is not None and _f2 is not None and "already rejected" in _f2[1])
chk("실물 A2 선언을 실제로 돌렸다", _live == 2, "%d층" % _live)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
