# -*- coding: utf-8 -*-
"""회귀 — 진행-감응 발견 요구 (T2_DISCOVERY_STEP2 · 오프라인 · 모델 0 · 원장 C442).

이 레버가 사는 근거는 x273(n=8): 출시 문구는 이름을 이미 알아도 **UNLOCK 0/8**(전부 재검색),
(2)단계 문구로 바꾸면 **8/8**. 그래서 검정할 것은 그 경계다 —

  ⑴ 기본 OFF · ⑵ 이름은 **레지스트리 ∩ 회수 텍스트**에서만(짓지 않는다)
  ⑶ 이미 unlock 시도한 이름은 다시 말하지 않는다 · ⑷ 못 찾으면 종전 문구(회귀 0)
  ⑸ 출시 문구 = x273 이 이긴 문자열(축자·[[03b]]) · ⑹ 도메인 리터럴 0
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
sys.path.insert(0, HERE)

SRC = io.open(os.path.join(HERE, "t2_resolve.py"), encoding="utf-8").read()
GATE = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


chk("기본 OFF", 'os.environ.get("T2_DISCOVERY_STEP2") == "1"' in SRC)
chk("문구 상수가 있다", "DISCOVERY_STEP2_FB" in SRC)
m = re.search(r'DISCOVERY_STEP2_FB = \((.*?)\n\)', SRC, re.S)
body = m.group(1) if m else ""
chk("x273 이 이긴 문자열 축자", "Do not search again" in body and "must be unlocked" in body)
chk("이름·도구는 주입", "{name}" in body and "{unlock}" in body)
chk("도메인 리터럴 0",
    not re.search(r"open_bank_account|account_class|Sky Blue|checking", body))

# ★2026-08-12 j런 수리: 판정 로직은 복수형 `_retrieved_unlockables` 로 이사(후보 전부 반환·
#   선택은 formalize=LLM). 단수형은 하위호환 래퍼. 검정 대상 = 복수형 본문.
m2 = re.search(r"def _retrieved_unlockables\(.*?\n(?=\n\ndef |\nDISC|\ndef )", SRC, re.S)
fn = m2.group(0) if m2 else ""
chk("판정 함수를 찾았다", bool(fn))
chk("이름 집합은 호출부가 준다(짓지 않는다)", "known_names" in fn and "if not known_names" in fn)
chk("회수 텍스트 실재를 본다", "in hay" in fn)
chk("이미 시도한 이름은 제외", "tried" in fn and "not in tried" in fn)
chk("빈 후보는 빈 목록(종전 경로)", "return []" in fn)
for pat, why in ((r"re\.(search|findall|match)\s*\(", "정규식"),
                 (r"\bmax\s*\(", "max("), (r"\bargmax\b", "argmax")):
    chk("도메인 해석·선택기 없음: %s" % why, not re.search(pat, fn))

chk("호출부가 레지스트리를 넘긴다", "known_names=_rz.registry_names(self)" in GATE)
chk("registry_names 는 프레임워크 출처", "get_discoverable_tools" in SRC)

# ── j런 수리 회귀 3건 (2026-08-12·블로킹 [1]) ──
import importlib
R = importlib.import_module("t2_resolve")


class _TC:
    def __init__(self, name, args=None):
        self.name, self.arguments = name, (args or {})


class _M:
    def __init__(self, role, content="", calls=None, error=False):
        self.role, self.content, self.error = role, content, error
        self.tool_calls = calls or []


# (ⓑ) 후보-정합: 후보 2개 중 formalize 가 고른 것만 STEP2 이름이 된다 — 스텁 la 로 검증
class _Sub:
    content = '{"tool": "open_thing_1111"}'


class _LA:
    @staticmethod
    def generate(**kw):
        return _Sub()


class _UM:
    def __init__(self, **kw):
        pass


class _Agent:
    llm = "stub"
    llm_args = {}
    tools = []


os.environ["T2_DISCOVERY_STEP2"] = "1"
A2 = {"action_tools": ["call_x"],
      "eplan": {"unlock_tool": "unlock_x", "dispatch_tool": "call_x", "list_tool": "list_x"},
      "operands": {"call_x": {"agent_tool_name": {"operator_resolution": "discoverable",
                                                  "getter": "KB_search"}}}}
msgs = [_M("user", "please do the thing"),
        _M("tool", "docs say: open_thing_1111 and move_thing_2222 exist")]
am = _M("assistant", "I will explain how instead")   # 회피 턴(도구 0)
r = R.resolve_action_operator({"action_tools": ["call_x"]}, am, msgs, A2,
                              target_tool="call_x", transfer_tools={"transfer_h"},
                              known_names={"open_thing_1111", "move_thing_2222"},
                              agent=_Agent(), la=_LA, UserMessage=_UM)
chk("회귀ⓑ: formalize 정합 이름만 지목", r.get("reason") == "discovery-step2"
    and "open_thing_1111" in str(r.get("feedback")) and "move_thing_2222" not in str(r.get("feedback")))


class _SubNone:
    content = '{"tool": "none"}'


class _LANone:
    @staticmethod
    def generate(**kw):
        return _SubNone()


r2 = R.resolve_action_operator({"action_tools": ["call_x"]}, am, msgs, A2,
                               target_tool="call_x", transfer_tools={"transfer_h"},
                               known_names={"move_thing_2222"},
                               agent=_Agent(), la=_LANone, UserMessage=_UM)
chk("회귀ⓑ2: 정합 없음(none)=이름 단정 없이 일반문", r2.get("reason") == "discovery-required")

# (ⓒ) 이관-후 침묵: transfer 가 실행 성공한 궤적에선 ok
msgs_tr = [_M("user", "transfer me"),
           _M("assistant", "", calls=[_TC("transfer_h", {"summary": "s"})]),
           _M("tool", "Transfer successful. A human agent will assist.")]
r3 = R.resolve_action_operator({"action_tools": ["call_x"]}, am, msgs_tr, A2,
                               target_tool="call_x", transfer_tools={"transfer_h"},
                               known_names={"open_thing_1111"},
                               agent=_Agent(), la=_LA, UserMessage=_UM)
chk("회귀ⓒ: 이관 실행 후 침묵", r3.get("status") == "ok")

# (ⓒ-역) 이관이 **거부**됐으면(우리 deny 채널) 침묵하지 않는다 — 과침묵 방지
msgs_den = [_M("user", "transfer me"),
            _M("assistant", "", calls=[_TC("transfer_h", {"summary": "s"})]),
            _M("tool", "Error: notice not sent yet", error=True)]
r4 = R.resolve_action_operator({"action_tools": ["call_x"]}, am, msgs_den, A2,
                               target_tool="call_x", transfer_tools={"transfer_h"},
                               known_names={"open_thing_1111"},
                               agent=_Agent(), la=_LA, UserMessage=_UM)
chk("회귀ⓒ-역: 거부된 이관은 안 센다", r4.get("status") == "deny")

try:
    import ast
    ast.parse(SRC)
    ast.parse(GATE)
    chk("파싱", True)
except Exception as e:
    chk("파싱", False, e)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
