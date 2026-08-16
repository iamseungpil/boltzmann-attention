# -*- coding: utf-8 -*-
"""⛔0번 규칙 hard-constraint — **결정론기에서 도메인 산문을 정규식으로 뜯지 않는다**.

사용자 지시(2026-08-16·축자·재론 금지):
  *"절대 결정론기에서 패턴 매칭이나 어떠한 정규식도 쓰면 안된다."*
  *"격리로 안되는 것만 엔진으로 남기되, LLM 이 formalize 하게 해야 한다."*

앞선 판들이 금지 범위를 *"도메인 텍스트에서 사실 추출"* 로 좁혀 읽을 여지를 뒀고, 나는 그
여지로 **세 번** 들어갔다(`t2_ledger` → x301 A2-상수 포장 → x342 프로브 재료). 그래서 이 검정은
**이설이 끝난 자리가 되돌아오지 않게** 못을 박는다(soft 지시로는 내가 못 지킨다·[[07]]).

잠그는 것(2026-08-17 이설 완료분):
  ① `parse_records`(정규식 줄 파서)를 엔진이 **부르지 않는다** — 입력은 `sub_records`(LLM formalize)
  ② `_ref_verify_deny` 가 `re.finditer(re.escape(field)…)` 로 값을 뽑지 않는다
  ③ 두 함수 모두 **서브를 받을 수 있게** agent/la/UserMessage 를 인자로 받는다
  ④ `sub_records` 는 **존재확인만** 한다(추출 0) — 정규식 없이 JSON 경계만 자른다
  ⑤ 숫자 정규화도 정규식 없이(`isdigit`) — `re.sub(r"[^0-9.]")` 잔재 0
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
SEA = io.open(os.path.join(HERE, "t2_search.py"), encoding="utf-8").read()
OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


def code_lines(text):
    """주석·독스트링 줄을 뺀 **코드 줄만**. 주석에 적힌 옛 정규식 인용이 검정을 오탐시켰다."""
    return "\n".join(l for l in text.split("\n") if not l.strip().startswith("#"))


def body_of(name, end):
    m = re.search(r"def %s\(.*?%s" % (re.escape(name), re.escape(end)), SRC, re.S)
    return code_lines(m.group(0)) if m else ""


print("[①] parse_records 를 엔진이 부르지 않는다")
calls = [l for l in SRC.split("\n")
         if "parse_records(" in l and not l.strip().startswith("#")]
chk(not calls, "엔진에 parse_records 호출 0 — 실제: %s" % (calls or "없음"))

print("[②③] 두 검증기가 서브 입력을 쓴다")
pc = body_of("_param_cap_deny", "cap = pct * lim")
rv = body_of("_ref_verify_deny", "if not rec_val:")
chk("def _param_cap_deny(agent, la, UserMessage," in SRC, "param_cap 이 서브 인자를 받는다")
chk("def _ref_verify_deny(agent, la, UserMessage," in SRC, "ref_verify 가 서브 인자를 받는다")
chk("sub_records(agent, la, UserMessage" in pc, "param_cap 입력 = sub_records")
chk("sub_records(agent, la, UserMessage" in rv, "ref_verify 입력 = sub_records")
for nm, b in (("param_cap", pc), ("ref_verify", rv)):
    chk(not re.search(r"re\.(finditer|search|findall|sub|compile|match)", b),
        "%s 본문에 정규식 0" % nm)

print("[④] sub_records 는 존재확인만")
sr = re.search(r"def sub_records\(.*?return out", SEA, re.S)
sb = sr.group(0) if sr else ""
chk(bool(sr), "정본 진입점 t2_search.sub_records 가 있다")
chk("if str(v) in text:" in sb, "각 값의 **원문 존재**만 확인한다(C45 동형)")
chk(not re.search(r"re\.(finditer|search|findall|sub|compile|match)", sb),
    "sub_records 에 정규식 0 — JSON 경계는 find/rfind")
chk('raw.find("[")' in sb and 'raw.rfind("]")' in sb, "문자열 연산으로만 JSON 을 자른다")

print("[⑤] 숫자 정규화도 정규식 없이")
_code = code_lines(SRC)
chk('re.sub(r"[^0-9.]"' not in _code, "숫자만 남기는 정규식 잔재 0(코드 줄 기준)")
chk("ch.isdigit() or ch ==" in SRC, "문자 필터로 대체돼 있다")

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
