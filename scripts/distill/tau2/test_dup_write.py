# -*- coding: utf-8 -*-
r"""래칫 — `T2_DUP_WRITE` 는 **재생성 채널로만** 지운다 (2026-08-26).

## 왜 이 래칫인가

읽기 dedup 은 반복 호출을 **스텁**으로 막는데, 그 코드가 write 를 명시로 제외한다. 이유가
주석에 박혀 있다(2026-08-02 · `failed_setstate_1785632213670`):

    "env 가 mutating 으로 보는 디스패처 호출이 반복되자 스텁이 히스토리에 남았고,
     eval set_state 가 재실행 실물과 비교해 **sim 무효** ⇒ replay 가 재실행할 도구는
     **절대 스텁 금지**"

⇒ 쓰기 중복을 막는 유일하게 안전한 길은 **재생성 채널**이다(호출을 히스토리에서 통째로
   지우므로 재실행 대조가 어긋날 대상 자체가 없다). 이 래칫이 그 경계를 고정한다.

## 측정 선행 ([[62]]① — 이 레버가 왜 있는가)

    x546/x547 재생  중복을 전부 빼면 만점 sim **14/14 불변**(비용 0) · 0점 sim 142 중 **8** 이 1.0
                    정제 술어(*순수 반복만*)는 8 중 **1** 만 살려 더 나쁘다
    x548 격리       이 문면은 재발행을 **4/4 → 0/4** · 이름 없는 거절 4/4 · 같은 길이 무관 4/4
                    ([[57]] 부정통제 통과)

## 고정하는 것

  ① 플래그로만 켜진다(기본 OFF)
  ② **스텁을 만들지 않는다** — 이 블록에 `stubs[` 가 없다
  ③ 재생성 break 가드·조립 목록·배달 분기 **셋 다**에 채널이 등록돼 있다
     (하나라도 빠지면 배달이 모델에게 안 간다 — 2026-08-05 `proc_fb` 사고)
  ④ 술어가 닫혀 있다: 성공 판정은 결과의 오류 여부뿐 · 도메인 낱말 0
  ⑤ 실동작: 같은 키의 두 번째만 잡고, **실패한 앞선 호출은 근거가 되지 않는다**

실행: PYTHONIOENCODING=utf-8 py -3 test_dup_write.py
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
FAILED = []


def chk(cond, what, extra=""):
    print("  %-4s %s%s" % ("OK" if cond else "FAIL", what,
                           (" — %s" % (extra,)) if extra else ""))
    if not cond:
        FAILED.append(what)


print("① 플래그로만 켜진다")
chk('os.environ.get("T2_DUP_WRITE") == "1"' in SRC, "플래그 술어가 있다")
chk('os.environ.get("T2_DUP_WRITE", "1")' not in SRC, "기본값을 1 로 두지 않았다")
# ★2026-09-05 (x771 · D6) — 이 칸은 **"go_stack 이 0 인가"** 를 물었었다. 그 물음은
#   [[81]] 을 못 잡았다: 정본은 `go_stack.sh:701 =0` 인데 라이브 런처는 5곳 전부 `=1` 이라
#   **도는 스택 ≠ 정본 스택** 이었고, 그래서 이 레버의 라이브 거동은 아무도 래칫하지 않았다.
#   물음을 값에서 **일치**로 바꾼다 — 정본과 런처가 같은 값이면 무엇이든 통과, 다르면 FAIL.
#   런처 목록을 손으로 적지 않는다 — 새 런처가 생기면 그것도 자동으로 검사 대상이다([[58]]).
_VAL_RE = re.compile(r"T2_DUP_WRITE=(\d)")


def _dup_vals(path):
    return _VAL_RE.findall(io.open(path, encoding="utf-8", errors="replace").read())


_canon = _dup_vals(os.path.join(HERE, "go_stack.sh"))
chk(len(_canon) == 1, "정본(go_stack.sh)이 이 플래그를 **한 번** 정한다", str(_canon))
_bad = []
_seen_launchers = 0
for _p in sorted(glob.glob(os.path.join(HERE, "run_*.sh"))):
    _v = _dup_vals(_p)
    if not _v:
        continue
    _seen_launchers += 1
    if set(_v) != set(_canon):
        _bad.append("%s=%s" % (os.path.basename(_p), _v))
chk(_seen_launchers > 0, "이 플래그를 정하는 런처를 찾았다", "%d 개" % _seen_launchers)
chk(not _bad, "[[81]] 런처가 정본을 덮어쓰지 않는다",
    "정본=%s · 어긋난 런처=%s" % (_canon, _bad or "없음"))

print("\n② 스텁이 아니라 재생성 채널이다 (2026-08-02 sim 무효 사고)")
m = re.search(r'if os\.environ\.get\("T2_DUP_WRITE"\) == "1":(.*?)\n            dw_fb = None',
              SRC, re.S)
chk(m is not None, "블록을 찾았다")
blk = m.group(1) if m else ""
chk("stubs[" not in blk, "이 블록은 **스텁을 만들지 않는다**")
chk("_TM(" not in blk, "도구 결과 메시지를 직접 조립하지 않는다")
chk("dup_fb = (" in blk, "재생성 채널(`dup_fb`)로만 나간다")

print("\n③ 채널이 세 곳에 등록돼 있다")
chk("and dup_fb is None):" in SRC, "재생성 break 가드에 있다")
chk('("dup_write", dup_fb)' in SRC, "조립 목록(`_SRC8`)에 있다")
chk("elif dup_fb is not None and c is dup_fb[0]:" in SRC, "배달 분기에 있다")

print("\n④ 술어가 닫혀 있다")
chk("_mut_key_of" in SRC and "_succeeded_mut_keys" in SRC, "헬퍼 둘이 있다")
h = re.search(r"def _succeeded_mut_keys\(msgs, a2w\):(.*?)\n\n\n", SRC, re.S)
hb = h.group(1) if h else ""
chk('startswith("Error:")' in hb, "성공 판정은 결과의 오류 여부다")
chk("_is_effective_write" in hb, "변이 여부는 기존 술어를 쓴다(새 목록 0)")
chk(not re.search(r'"[a-z_]*(credit|dispute|account|card)[a-z_]*"', hb),
    "도메인 낱말이 없다([[05]]·[[59]])")

print("\n⑤ 실동작 — 같은 키의 두 번째만 잡고, 실패한 앞선 호출은 근거가 아니다")
try:
    import t2_gate_patch as G

    class TC(object):
        def __init__(self, i, name, args):
            self.id, self.name, self.arguments = i, name, args

    class M(object):
        def __init__(self, role, tool_calls=None, mid=None, content="", error=False):
            self.role, self.tool_calls, self.id = role, tool_calls, mid
            self.content, self.error = content, error

    A = {"agent_tool_name": "apply_checking_account_credit_5829",
         "arguments": '{"account_id": "chk_1", "amount": 27.0}'}
    a2 = G._domain_a2("banking_knowledge")
    ok = [M("assistant", [TC("t1", "call_discoverable_agent_tool", A)]),
          M("tool", mid="t1", content="Credit applied successfully!")]
    bad = [M("assistant", [TC("t2", "call_discoverable_agent_tool", A)]),
           M("tool", mid="t2", content="Error: Account not found.")]
    seen_ok = G._succeeded_mut_keys(ok, a2)
    seen_bad = G._succeeded_mut_keys(bad, a2)
    chk(len(seen_ok) == 1, "성공한 변이는 키로 기록된다", len(seen_ok))
    chk(len(seen_bad) == 0, "**실패한 호출은 기록되지 않는다**", len(seen_bad))
    # ★2026-09-05 (x771 · D6) — 이 두 칸은 종전에 **`_mut_key_of` 로** 잡히는지를 물었다.
    #   이제 원장에는 선언 키만 실리므로 물음을 그 축으로 옮긴다(같은 시나리오·다른 축).
    t3 = TC("t3", "call_discoverable_agent_tool", A)
    chk(G._once_key_of(t3, a2) in seen_ok, "같은 호출이 **선언 키**로 잡힌다")
    chk(G._mut_key_of(t3) not in seen_ok,
        "★D6 — 인자-전체 키(`_mut_key_of`)는 원장에 **없다**")
    A2 = dict(A, arguments='{"account_id": "chk_1", "amount": 14.5}')
    t4 = TC("t4", "call_discoverable_agent_tool", A2)
    chk(G._once_key_of(t4, a2) in seen_ok,
        "같은 계좌·다른 금액도 선언 키로 잡힌다(t7378 074)")
    chk(G._mut_key_of(t4) not in seen_ok, "인자-전체 키로는 여전히 안 잡힌다")

    # ─── ★절차적 래퍼는 잡지 않는다 (2026-08-26·t7359 스모크가 잡은 결함·x549) ───
    # 스모크 첫 판이 `[T2_DUP_WRITE] deny tool=unlock_discoverable_agent_tool` 을 냈다.
    # unlock 은 대상 도구를 **실행하지 않고 잠금해제만** 하므로 실효 write 가 아니다
    # (`_eff_tool_name` 독스트링 축자 · `DECLFIRST_LIVE_WIRING_DESIGN §5-c` 가 계측 정의를
    #  같은 이유로 고쳤다: *"unlock을 write로 계상 → … Z7-③이 2/3으로 과대"*).
    # 원인은 술어가 아니라 **재료 미전달**이었다 — `_a2_of(self)` 가 None 이라 A2 파생
    # 절차 집합이 비었다. 그래서 이 칸은 술어를 **배선이 주는 a2 로** 검정한다.
    W = {"agent_tool_name": "apply_checking_account_credit_5829"}
    for _wrap in ("unlock_discoverable_agent_tool", "give_discoverable_user_tool"):
        chk(G._is_effective_write(G._eff_tool_name(TC("w", _wrap, W)), a2) is False,
            "절차적 래퍼는 실효 write 가 아니다 — %s" % _wrap)

    class _AgentLike(object):        # init_inject 가 심는 것만 갖는다(`.environment` 없음)
        pass

    _ag = _AgentLike()
    _ag._t2_a2, _ag._t2_orch = a2, None
    _live = G._a2_of(_ag)
    chk(_live is not None, "배선: 에이전트에서 a2 가 온다(구판은 None 이었다)")
    for _wrap in ("unlock_discoverable_agent_tool", "give_discoverable_user_tool"):
        chk(G._is_effective_write(G._eff_tool_name(TC("w", _wrap, W)), _live) is False,
            "**라이브 경로로도** 래퍼를 안 잡는다 — %s" % _wrap)
    _wrap_msgs = [M("assistant", [TC("u1", "unlock_discoverable_agent_tool", W)]),
                  M("tool", mid="u1", content="Tool unlocked."),
                  M("assistant", [TC("u2", "unlock_discoverable_agent_tool", W)]),
                  M("tool", mid="u2", content="Tool unlocked.")]
    chk(len(G._succeeded_mut_keys(_wrap_msgs, _live)) == 0,
        "반복된 unlock 은 중복 원장에 **아예 안 실린다**(발화 0)")

    # ══════════════════════════════════════════════════════════════════════
    # ⑥ ★D6 (2026-09-05 · x771) — 억제는 **A2 가 선언한 write** 로만 한다
    #
    # 무엇이 틀렸었나: 원장이 `_mut_key_of`(이름+인자 전체)를 함께 실어서, 정책이 유일하다고
    # **말한 적 없는** write 의 재제출까지 지웠다. 독스트링은 그것을 *fail-open* 이라 불렀지만
    # 실제 거동은 fail-CLOSED 였다 — 선언이 없을수록 더 넓게 막았다.
    # 실물: `bank_k8143med1_20260904_0135 / task_051`(reward 0.0). gold 가 요구한 재제출
    # (`051_7` = `051_2` 바이트 동일)이 turn 61·63·65·67 에 4발 지워졌고, 회수분 fb 사이드카
    # 전수에서 이 문면은 **221 발 중 213** 이 미선언 write 였다(banking 52).
    # 무엇이 답인가([[64]]): 무엇이 유일한지는 A2 `write_once_keys` 가 정하고, 엔진은 그
    # 집합의 소속만 본다([[05]]·[[22]] 닫힌 술어 · 도메인 낱말 0).
    print("\n⑥ ★D6 — 억제는 선언된 write 로만 (x771)")

    _reg = re.search(r"def _succeeded_mut_keys\(msgs, a2w\):(.*?)\n    return out", SRC, re.S)
    _regb = _reg.group(1) if _reg else ""
    chk("_mut_key_of(tc)" not in _regb,
        "등록: 원장에 인자-전체 키를 싣지 않는다 (:6121)")
    chk("_once_key_of(tc, a2w)" in _regb, "등록: 선언 키는 싣는다")
    _look = re.search(r"_dk = None\n(.*?)\n\s+if not _dk:", SRC, re.S)
    _lookb = _look.group(1) if _look else ""
    chk(bool(_lookb) and "_mut_key_of(_dc)" not in _lookb,
        "조회: 인자-전체 키로 되묻지 않는다 (:12314)")
    chk("_once_key_of(_dc, _a2_of(self))" in _lookb, "조회: 선언 키는 본다")

    # ⛔`_mut_key_of` 함수 자체는 살아 있어야 한다 — `T2_WRITE_ARG_TYPE` 의 sim-당 cap 키가
    #   같은 함수를 쓴다([[67]] 정본 재사용). 지우면 2026-08-28 t7376 task_040 회귀가 돌아온다.
    chk(callable(getattr(G, "_mut_key_of", None)), "`_mut_key_of` 는 삭제되지 않았다")
    chk("_tk = _mut_key_of(c) or str(_exact_tool_name(c) or \"\")" in SRC,
        "다른 소비처(`T2_WRITE_ARG_TYPE` cap 키)가 그대로 있다")

    # ── 실동작. 재료는 **미선언 write** — 051 이 막힌 그 모양이다.
    U = {"agent_tool_name": "submit_credit_limit_increase_request_7392",
         "arguments": '{"card_id": "cc_1", "requested_credit_limit": 5000}'}
    _uc = TC("u", "call_discoverable_agent_tool", U)
    chk(G._is_effective_write(G._eff_tool_name(_uc), a2) is True,
        "재료가 실효 write 이긴 하다(억제 대상이었다)")
    chk(G._once_key_of(_uc, a2) is None, "그러나 A2 가 유일하다고 **선언한 적 없다**")
    _um = [M("assistant", [TC("u1", "call_discoverable_agent_tool", U)]),
           M("tool", mid="u1", content="Request submitted successfully.")]
    _uled = G._succeeded_mut_keys(_um, a2)
    chk(len(_uled) == 0, "★미선언 write 는 원장에 안 실린다 => 재제출이 살아난다", str(_uled))

    # ── [[57]] 부정통제 ① : **되돌린 팔**(등록에 인자-전체 키를 다시 넣는다)로는 잡힌다.
    #    = 이 검정은 «무조건 통과» 가 아니라, 바로 그 한 줄에만 반응한다.
    def _register_pre(msgs, a2w):
        """수리 **전** 등록부 — HEAD 축자 `(_mut_key_of(tc), _once_key_of(tc, a2w))`."""
        out, ms = {}, list(msgs or [])
        for i, m in enumerate(ms):
            for tc in (getattr(m, "tool_calls", None) or []):
                if not G._is_effective_write(G._eff_tool_name(tc), a2w):
                    continue
                for j in range(i + 1, len(ms)):
                    mj = ms[j]
                    if str(getattr(mj, "role", "")) != "tool" or getattr(mj, "id", None) != tc.id:
                        continue
                    body = str(getattr(mj, "content", "") or "")
                    if not getattr(mj, "error", False) and not body.lstrip().startswith("Error:"):
                        for k in (G._mut_key_of(tc), G._once_key_of(tc, a2w)):
                            if k and k not in out:
                                out[k] = (i, body)
                    break
        return out

    _upre = _register_pre(_um, a2)
    chk(G._mut_key_of(_uc) in _upre,
        "[부정통제] 되돌린 팔에서는 **같은 재료가 막힌다**(=수리가 실효했다)")
    chk(set(G._succeeded_mut_keys(_um, a2)) != set(_upre),
        "[부정통제] 라이브 엔진 원장 ≠ 되돌린 팔 원장")

    # ── [[57]] 부정통제 ② : **선언된** write 는 두 팔 모두 막아야 한다(보호 유지 · 오차단 0).
    _dled = G._succeeded_mut_keys(ok, a2)
    chk(G._once_key_of(t3, a2) in _dled, "선언 write 는 라이브 엔진에서도 막힌다")
    chk(G._once_key_of(t3, a2) in _register_pre(ok, a2), "되돌린 팔에서도 막힌다(불변)")
    _other = TC("t5", "call_discoverable_agent_tool",
                dict(A, arguments='{"account_id": "chk_2", "amount": 27.0}'))
    chk(G._once_key_of(_other, a2) not in _dled, "다른 대상은 두 팔 모두 통과(오차단 0)")
except Exception as e:                                                  # pragma: no cover
    chk(False, "실동작 검정이 돈다", repr(e))

print("\nRESULT: %s%s" % ("PASS" if not FAILED else "FAIL",
                          "" if not FAILED else " " + str(FAILED)))
sys.exit(0 if not FAILED else 1)
