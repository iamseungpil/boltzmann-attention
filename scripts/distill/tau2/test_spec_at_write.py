# -*- coding: utf-8 -*-
r"""`T2_SPEC_AT_WRITE` 래칫 — **전달만 하는가**를 코드에 고정한다 (2026-08-25).

## 무엇을 지키나

이 레버가 사는 근거는 격리 `x532` 하나다: A_asis 1/6 ↔ **B_spec 6/6** ↔ N_neg 2/5.
B_spec 이 준 것은 *env 가 앞서 보낸 응답 전문* 이고, 그것을 **자르거나 고르거나 요약하면**
격리에서 잰 것과 다른 물건이 된다([[62]] 측정 대상 소실 · [[76]] 격리=서브).

그래서 세 가지를 래칫으로 박는다:
  ① 되붙이는 본문은 **env 메시지 그대로**다 — 엔진이 키 이름을 뽑거나 목록을 짓지 않는다.
  ② 술어에 **도메인 낱말이 0** 이다([[05]] 전이 — 도메인을 갈아 끼워도 같아야 한다).
  ③ 자기 오류문을 되먹이지 않는다 — 이 write **자신의 겉이름**으로 나간 호출은 출처에서 제외.
  ④ 기본 OFF + sim 당 도구별 1회(무한 유예 금지).

⚠이 검정은 *레버가 옳다*를 증명하지 않는다. 격리가 잰 물건과 **같은 물건이 배선됐는지**만 본다.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
FAIL = []


def chk(cond, what, got=""):
    print(("  OK  " if cond else "  XX  ") + what + (("  -- %s" % (got,)) if got else ""))
    if not cond:
        FAIL.append(what)


m = re.search(r"\ndef _env_spec_for\(wc, msgs\):(.*?)\n_DECIDE_FIRST_FB", SRC, re.S)
chk(m is not None, "출처 헬퍼 `_env_spec_for` 가 있다")
BODY = m.group(1) if m else ""

print("[1] 되붙이는 것은 env 본문 그대로다")
chk("getattr(ms[j], \"content\", \"\")" in BODY, "tool 메시지의 content 를 그대로 읽는다")
chk(not re.search(r"\bre\.(findall|search|match|sub)\b", BODY),
    "본문을 정규식으로 뜯지 않는다([[59]])")
chk("[:2" not in BODY and "[:1" not in BODY and "[:5" not in BODY,
    "본문을 자르지 않는다(격리가 잰 것은 전문이다)")
chk("sorted(" not in BODY and "max(" not in BODY.replace("min(", ""),
    "고르지 않는다 — argmax·최댓값 0([[62]]③④)")

print("[2] 술어에 도메인 낱말 0 ([[05]] 전이)")
# ⚠**실행부만** 본다 — 독스트링에는 085 증거 축자(`debit_card_id` 등)가 있어야 하고,
#   그것을 지우면 왜 이 배선이 있는지가 사라진다([[77]] ②근거 축자 의무).
CODE = re.sub(r'"""(?:.|\n)*?"""', "", BODY, count=1)
CODE = "\n".join(ln.split("#", 1)[0] for ln in CODE.splitlines())
DOMAIN = ("dispute", "debit", "card", "account", "bank", "referral", "transaction",
          "credit", "task_", "_6281", "_4821", "_9173")
low = CODE.lower()
hits = [w for w in DOMAIN if w in low]
chk(not hits, "도메인·태스크 리터럴이 없다", ",".join(hits) or "없음")
chk("_exact_tool_name(wc)" in BODY, "표적은 env 레지스트리 이름에서 온다(철자 규칙 0)")

print("[3] 자기 오류문 되먹임 금지")
chk('mine = str(getattr(wc, "name", "") or "")' in BODY, "이 write 의 겉이름을 잡는다")
chk(re.search(r'if str\(getattr\(tc, "name", "" \) or ""\) == mine:\s*\n\s*continue', BODY)
    or re.search(r'== mine:\s*\n\s*continue', BODY),
    "같은 겉이름의 호출은 출처에서 제외한다")

print("[4] 기본 OFF · 상한 · 못 찾으면 침묵")
chk('os.environ.get("T2_SPEC_AT_WRITE") == "1"' in SRC, "플래그로만 켜진다")
chk("_t2_spec_at_write" in SRC, "sim 당 도구별 1회 원장이 있다")
chk("T2_SPEC_AT_WRITE_MIN" in SRC, "거리 하한이 있다(바로 앞이면 되붙일 이유가 없다)")
chk("return None, -1, -1" in BODY, "못 찾으면 None — 아무 말도 하지 않는다([[25]])")

print("[5] 계기는 플래그와 무관하게 찍힌다 (다음 런 원인파악 장치)")
chk("[T2_SPEC_DIST] tool=%s src_msg=%s dist=%s len=%s" in SRC,
    "거리 마커가 있다 — 켜든 끄든 남는다")
i_dist = SRC.find('"[T2_SPEC_DIST] tool=')
i_flag = SRC.find('os.environ.get("T2_SPEC_AT_WRITE") == "1"')
chk(0 < i_dist < i_flag, "마커가 플래그 검사보다 **앞**이다(OFF 런에서도 남는다)")

print("[6] 문면은 출처를 밝힌다 ([[64]] 무엇이 틀렸고 무엇을 하면 풀리나)")
fb = re.search(r"_SPEC_AT_WRITE_FB = \((.*?)\n\)\n", SRC, re.S)
chk(fb is not None, "문면 상수가 있다")
FB = fb.group(1) if fb else ""
chk("{spec}" in FB and "{dist}" in FB, "본문과 거리를 함께 싣는다")
chk("environment's own reply" in FB, "새 정보가 아니라 env 자신의 응답임을 밝힌다")

print("[7] **함수를 실제로 돌린다** — 死배선은 정규식이 아니라 실행이 잡는다([[24]])")
# 왜 여기서: 라이브 스모크로 이 갈래에 닿으려면 085 를 한 sim(실측 28~46분) 태워야 하고,
# 그 비용의 대부분은 우리가 이미 아는 것(부모 인쇄 자리가 살아 있다 — t7348 로그에서
# `축 미상` 이 040 54회·085 44회)을 다시 사는 데 쓰인다. 남은 위험은 **이름 해석**이고
# 그건 오프라인에서 전부 잡힌다([[09]] 무료 검증 우선).
try:
    import t2_gate_patch as G

    class _TC(object):
        def __init__(self, name, args):
            self.name, self.arguments = name, args

    class _M(object):
        def __init__(self, role, content="", tool_calls=None):
            self.role, self.content, self.tool_calls = role, content, tool_calls or []

    # ⚠이름은 엔진의 **기존 규약**을 따른다: `_exact_tool_name` 은 `call_` 접두 디스패처일
    #   때만 인자에서 내부 이름을 푼다. 도메인 낱말은 쓰지 않는다(가짜 이름).
    INNER = "widget_filer_9001"
    SPEC = "Tool unlocked: %s\nParameters:\n  - alpha: string\n  - beta: number" % INNER
    msgs = [_M("user", "hello"),
            _M("assistant", tool_calls=[_TC("unlock_thing", {"agent_tool_name": INNER})]),
            _M("tool", SPEC),
            _M("assistant", tool_calls=[_TC("call_thing", {"agent_tool_name": INNER})]),
            _M("tool", "Error: Invalid arguments: got an unexpected keyword argument 'alfa'")]
    wc = _TC("call_thing", {"agent_tool_name": INNER})
    body, idx, dist = G._env_spec_for(wc, msgs)
    chk(body == SPEC, "env 반환문을 **글자 그대로** 돌려준다", repr(str(body)[:40]))
    chk(idx == 2, "출처 인덱스를 정확히 짚는다", idx)
    chk(dist == 3, "거리를 센다", dist)
    # 자기 오류문 되먹임 금지: 같은 겉이름(runner_tool)의 결과(msg4)를 고르면 안 된다.
    chk(body != msgs[4].content, "이 write 자신의 오류문은 출처가 아니다")
    b2, i2, d2 = G._env_spec_for(_TC("call_thing", {"agent_tool_name": "nothing_here"}), msgs)
    chk(b2 is None and i2 == -1, "못 찾으면 None — 침묵한다([[25]])", repr(b2))
except Exception as _e:
    chk(False, "헬퍼가 실행된다(死배선 아님)", repr(_e))

print("\n" + ("test_spec_at_write PASS" if not FAIL
              else "test_spec_at_write FAIL: %s" % (FAIL,)))
sys.exit(1 if FAIL else 0)
