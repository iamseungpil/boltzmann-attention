# -*- coding: utf-8 -*-
"""BYREF 두 수리 회귀 — ⑴래퍼 언랩(거짓 deny 제거) ⑵isolate 선점 해소 (2026-08-18·C531).

## 무엇을 막는가

⑴ `_resolve_ref_output` 의 색인이 **래퍼 이름만** 담았다. 이 환경은 discoverable 도구를
   `call_discoverable_agent_tool(agent_tool_name=…)` 로 디스패치하므로, 그렇게 부른 도구는
   `@last:<실제이름>` 이 **영영 안 맞고** "no committed non-error output" 으로 거짓 deny 됐다.
   전수 census: 거짓 deny **49건/11 sim**(072:33 · 073:11 · 074:5).

⑵ 그 실패가 `continue` 로 조기 반환해 **`isolate: fetch_formalize` 를 선점**했다. 그런데 그 서브는
   레코드를 모델에게서 받지 않고 **스스로 fetch·formalize** 해 그 키를 `ctx.update` 로 덮어쓴다
   — 참조가 안 풀려도 **그 호출은 성공했을 호출**이다.

## 불변식

  ① 언랩은 정본(`t2_gate_patch._exact_tool_name`)을 쓴다 — 사본 금지([[67]]).
  ② **넓히기만 한다** — 래퍼 이름 참조도 여전히 맞는다(새 실패 모드 0).
  ③ isolate 가 `operand_keys` 로 **선언한** 키만 넘긴다 — A2 어휘 순증 0 · 새 플래그 0.
  ④ 그 밖의 키는 여전히 **fail-closed**(raise) — 침묵 통과를 만들지 않는다.
  ⑤ 성공 경로 불변.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_scaffold_get as SG   # noqa: E402

OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


class TC(object):
    def __init__(self, id, name, arguments=None):
        self.id, self.name, self.arguments = id, name, (arguments or {})


class Msg(object):
    def __init__(self, role, tool_calls=None, id=None, content=None, error=False):
        self.role, self.tool_calls, self.id = role, tool_calls, id
        self.content, self.error = content, error


class Orch(object):
    def __init__(self, msgs):
        self._m = msgs

    def get_messages(self):
        return self._m


DUMP = "1. Record ID: txn_1\n   amount: 10\n"

print("[①②] 래퍼로 부른 도구를 실제 이름으로 참조할 수 있다 — 그리고 래퍼 이름도 여전히 맞는다")
disp = TC("c1", "call_discoverable_agent_tool",
          {"agent_tool_name": "get_credit_card_transactions_by_user"})
orch = Orch([Msg("assistant", [disp]), Msg("tool", id="c1", content=DUMP)])
try:
    got = SG._resolve_ref_output(orch, "@last:get_credit_card_transactions_by_user")
    chk(got == DUMP, "안쪽 이름으로 해석된다 (수리 전에는 거짓 deny)")
except Exception as e:
    chk(False, "안쪽 이름 해석 실패: %r" % (e,))
try:
    got2 = SG._resolve_ref_output(orch, "@last:call_discoverable_agent_tool")
    chk(got2 == DUMP, "래퍼 이름 참조도 그대로 맞는다 (넓히기만·새 실패 모드 0)")
except Exception as e:
    chk(False, "래퍼 이름 해석이 깨졌다(회귀): %r" % (e,))

print("[②b] 없는 도구는 여전히 실패한다 (넓히기가 아무거나 맞추지 않는다)")
try:
    SG._resolve_ref_output(orch, "@last:nonexistent_tool")
    chk(False, "없는 이름이 통과했다")
except SG._ByrefError:
    chk(True, "없는 이름은 _ByrefError")

print("[③④] isolate 가 산출할 키만 deny 를 면한다")
D_ISO = {"name": "t", "op": {"over": "transactions"},
         "isolate": {"mode": "fetch_formalize", "operand_keys": ["transactions"]}}
D_NOISO = {"name": "t", "op": {"over": "transactions"}}
empty = Orch([])                                    # 참조 대상이 아무것도 없다 → 반드시 실패
ctx = {"transactions": "@last:get_credit_card_transactions_by_user"}
try:
    SG._byref_resolve(empty, D_ISO, dict(ctx))
    chk(True, "isolate(fetch_formalize)+operand_keys 선언 키 → deny 하지 않고 넘긴다")
except SG._ByrefError as e:
    chk(False, "선언 키인데 deny 됐다: %s" % e)
try:
    SG._byref_resolve(empty, D_NOISO, dict(ctx))
    chk(False, "isolate 선언이 없는데 통과했다 — fail-closed 가 깨졌다")
except SG._ByrefError:
    chk(True, "isolate 선언이 없으면 여전히 fail-closed(raise)")

D_OTHERKEY = {"name": "t", "op": {"over": "transactions"},
              "isolate": {"mode": "fetch_formalize", "operand_keys": ["other"]}}
try:
    SG._byref_resolve(empty, D_OTHERKEY, dict(ctx))
    chk(False, "선언되지 않은 키가 면제됐다")
except SG._ByrefError:
    chk(True, "operand_keys 에 없는 키는 면제되지 않는다")

D_WRONGMODE = {"name": "t", "op": {"over": "transactions"},
               "isolate": {"mode": "formalize", "operand_keys": ["transactions"]}}
try:
    SG._byref_resolve(empty, D_WRONGMODE, dict(ctx))
    chk(False, "fetch_formalize 가 아닌 모드가 면제됐다")
except SG._ByrefError:
    chk(True, "mode != fetch_formalize 면 면제되지 않는다(그 모드는 ctx 를 안 덮는다)")

print("[⑤] 성공 경로 불변")
orch2 = Orch([Msg("assistant", [disp]), Msg("tool", id="c1", content=DUMP)])
c2 = {"transactions": "@last:get_credit_card_transactions_by_user"}
try:
    SG._byref_resolve(orch2, D_ISO, c2)
    chk(isinstance(c2.get("transactions"), list) and len(c2["transactions"]) == 1,
        "해석되면 종전대로 rows 로 치환된다 — 실제 %r" % (type(c2.get("transactions")).__name__,))
except Exception as e:
    chk(False, "성공 경로가 깨졌다: %r" % (e,))

print("[①b] 정본 언랩을 쓴다 (사본 금지)")
SRC = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
chk("_gp_ref._exact_tool_name" in SRC, "t2_gate_patch._exact_tool_name 을 그대로 쓴다")
# ⚠사본 판정은 **코드 패턴**으로 한다 — 주석이 규칙을 *설명*하는 것은 사본이 아니다
#   (초판 단언이 주석의 낱말을 세어 거짓 실패를 냈다).
chk('.get("agent_tool_name")' not in SRC and ".get('agent_tool_name')" not in SRC,
    "안쪽 이름을 이 파일이 직접 꺼내는 코드가 없다 — 언랩은 정본 몫")

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
