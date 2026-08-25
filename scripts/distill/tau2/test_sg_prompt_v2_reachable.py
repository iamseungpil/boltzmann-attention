# -*- coding: utf-8 -*-
r"""`T2_SG_PROMPT_V2` 도달성 래칫 — **형식 지시가 서브에게 닿는가** (2026-08-25).

## 무엇이 있었나 (t7352 라이브 실측)

V2 는 격리(`x525j` N_wire)에서 chk_1~4 커버리지를 닫고 실렸다. 그런데 라이브 축자:

    [T2_SG_ISOLATE] fetch get_atm_fee_discrepancies: **2라운드**·getter 1회·operand keys=**[]**
    [T2_SG_ARGS] get_atm_fee_discrepancies: 'transactions' 인자 str 잔류(JSON 파싱실패) → 재송신
    [T2_COMPUTE] select_discrepant: **9/17행 판정불가(operand가 숫자 아님)**

같은 태스크의 대조군 t7348(V2 off)은 `operand keys=['transactions']` 였다.

원인: V2 는 `answer_format` 을 **머리에서 빼고** 마감 메시지로 옮기는데, 그 마감 메시지가
`_tl is None`(= **마지막** 라운드)에만 붙었다. 이 서브는 `max_rounds=3` 인데 **라운드 1** 에
답하므로, 그 시점의 서브는 형식 지시를 **한 번도 못 본 채** 답한다.

⇒ 격리 통과가 라이브 도달을 보장하지 않는다([[24]] 死배선 · [[30]] *"단위통과 ≠ 라이브 발화"*).

## 이 검정이 지키는 것

  ① 마감 문면이 **마지막 라운드에만 매이지 않는다** — 도구 결과가 들어온 뒤면 붙는다.
  ② V2 가 켜졌을 때 `answer_format` 은 **어느 경로로든 반드시 한 번** 서브 앞에 선다.
  ③ 붙였다는 사실이 **로그로 남는다**(이 결함이 안 보였던 이유가 마커 부재다).
  ④ 격리가 이긴 순서는 보존 — 형식은 **재료 뒤**다.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
FAIL = []


def chk(cond, what, got=""):
    print(("  OK  " if cond else "  XX  ") + what + (("  -- %s" % (got,)) if got else ""))
    if not cond:
        FAIL.append(what)


print("[1] 마감 문면이 마지막 라운드에만 매이지 않는다")
m = re.search(r'if \(os\.environ\.get\("T2_SG_PROMPT_V2"\) == "1"[^:]*?\):', SRC, re.S)
chk(m is not None, "V2 마감 조건이 있다")
COND = m.group(0) if m else ""
chk("_ok_outs" in COND,
    "조건이 **도구 결과 도착**(`_ok_outs`)을 본다 — 마지막 라운드 전용이 아니다", COND[-70:])
chk("_v2_close_sent" in COND, "서브 호출당 한 번만 붙는다")

print("[2] V2 가 머리에서 뺀 것이 마감에 반드시 들어간다")
head = re.search(r'elif os\.environ\.get\("T2_SG_PROMPT_V2"\) == "1":(.*?)\n    else:', SRC, re.S)
chk(head is not None, "V2 조립 갈래가 있다")
H = head.group(1) if head else ""
chk('iso["answer_format"]' not in H,
    "머리 프롬프트에는 `answer_format` 이 **없다**(격리가 이긴 순서)")
close = re.search(r'_close = \(\((.*?)\+ iso\["answer_format"\]\)', SRC, re.S)
chk(close is not None, "마감 문면이 `answer_format` 을 **축자로** 싣는다")
C = close.group(1) if close else ""
chk("RECORDS" in C, "마감 문면이 **재료를 형식보다 앞**에 둔다(x525 J_both·K_paramslast)")

print("[3] 붙였다는 사실이 로그로 남는다 (이 결함이 안 보였던 이유)")
chk("[T2_SG_PROMPT_V2]" in SRC, "마커가 있다")
i_mark = SRC.find('print("[T2_SG_PROMPT_V2]')
i_cond = SRC.find('if (os.environ.get("T2_SG_PROMPT_V2") == "1" and not _v2_close_sent')
chk(0 < i_cond < i_mark, "마커가 그 조건 **안**에 있다(붙었을 때만 찍힌다)", "%d<%d" % (i_cond, i_mark))

print("[4] 라운드를 소비하지 않는다")
seg = SRC[i_cond:i_cond + 2200] if i_cond > 0 else ""
chk("continue" not in seg.split("_v2_close_sent = True")[0],
    "마감 문면을 붙이고 **곧바로 생성**한다 — 라운드를 버리지 않는다")

print("[5] 회귀 축자 — 무엇이 관측됐는지 문서에 남아 있다")
chk("operand keys=**[]**" in SRC or "operand keys=[]" in SRC,
    "수리 주석이 라이브 축자를 인용한다([[77]] ②)")

print("\n" + ("test_sg_prompt_v2_reachable PASS" if not FAIL
              else "test_sg_prompt_v2_reachable FAIL: %s" % (FAIL,)))
sys.exit(1 if FAIL else 0)
