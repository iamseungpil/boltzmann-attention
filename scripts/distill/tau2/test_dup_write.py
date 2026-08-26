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
chk("T2_DUP_WRITE=0" in io.open(os.path.join(HERE, "go_stack.sh"),
                                encoding="utf-8", errors="replace").read()
    or "T2_DUP_WRITE" not in io.open(os.path.join(HERE, "go_stack.sh"),
                                     encoding="utf-8", errors="replace").read(),
    "go_stack 이 기본으로 켜지 않는다")

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
    k = G._mut_key_of(TC("t3", "call_discoverable_agent_tool", A))
    chk(bool(k) and k in seen_ok, "같은 인자의 새 호출이 그 키로 잡힌다")
    A2 = dict(A, arguments='{"account_id": "chk_1", "amount": 14.5}')
    chk(G._mut_key_of(TC("t4", "call_discoverable_agent_tool", A2)) not in seen_ok,
        "인자가 다르면 잡히지 않는다(같은 도구라도)")

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
except Exception as e:                                                  # pragma: no cover
    chk(False, "실동작 검정이 돈다", repr(e))

print("\nRESULT: %s%s" % ("PASS" if not FAILED else "FAIL",
                          "" if not FAILED else " " + str(FAILED)))
sys.exit(0 if not FAILED else 1)
