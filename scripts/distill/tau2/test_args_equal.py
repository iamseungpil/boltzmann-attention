# -*- coding: utf-8 -*-
"""`t2_forensic.norm_args`/`args_equal` 계약 검정 (2026-08-17 계기 수리).

실화: 벤치 채점기(`tasks.py:195`)가 중첩 JSON 을 **문자열째** 비교해서, 공백·키순서만 다른
**같은 실행**이 `action_match=false` 로 찍혔다. 6런 gold_nested 1,222건 중 **121건**.
그 값을 읽던 우리 포렌식·센서스의 주장이 그만큼 오염됐다(reward 는 무영향).

못:
  ⑴ 중첩 JSON 문자열을 풀어 비교한다 (공백·키순서 무관)
  ⑵ **다른 값은 여전히 다르다** (수리가 참을 거짓으로도, 거짓을 참으로도 만들면 안 된다)
  ⑶ 숫자 표기(750 ↔ "750")를 접는다 — 양쪽에 같은 변환
  ⑷ 파싱 불가 문자열은 버리지 않고 공백만 정규화해 비교한다
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402

bad = []


def chk(cond, name):
    print("%s %s" % ("OK  " if cond else "FAIL", name))
    if not cond:
        bad.append(name)


A = {"discoverable_tool_name": "submit_cash_back_dispute_0589",
     "arguments": '{"user_id": "af0581dcbf", "transaction_id": "txn_f093f96e2001"}'}
B = {"discoverable_tool_name": "submit_cash_back_dispute_0589",
     "arguments": '{"transaction_id":"txn_f093f96e2001","user_id":"af0581dcbf"}'}
C = {"discoverable_tool_name": "submit_cash_back_dispute_0589",
     "arguments": '{"user_id": "af0581dcbf", "transaction_id": "txn_OTHER"}'}

chk(A != B, "전제: 문자열 비교로는 이 둘이 다르다(오염의 실물)")
chk(F.args_equal(A, B), "⑴ 공백·키순서만 다른 같은 실행 → 같다")
chk(not F.args_equal(A, C), "⑵ 거래 id 가 다르면 → 다르다")
chk(F.args_equal({"amount": 750}, {"amount": "750"}), "⑶ 숫자 표기를 접는다")
chk(not F.args_equal({"amount": 750}, {"amount": 751}), "⑶b 다른 수는 다르다")
chk(F.args_equal({"a": "x  y"}, {"a": "x y"}), "⑷ 파싱 불가 문자열은 공백만 정규화")
chk(not F.args_equal({"a": "x"}, {"a": "y"}), "⑷b 다른 문자열은 다르다")
chk(F.args_equal({"n": [1, {"k": '{"z": 1}'}]}, {"n": ["1", {"k": '{"z":1}'}]}),
    "중첩 리스트·이중 중첩도 푼다")
chk(not F.args_equal({}, {"a": 1}), "빈 인자와 있는 인자는 다르다")
chk(F.args_equal(None, None), "None 쌍은 같다")

# 정본 사용 강제 — 사본 금지([[67]])
import io  # noqa: E402
src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_forensic.py"),
              encoding="utf-8").read()
chk("def norm_args" in src and "def args_equal" in src, "정본에 함수가 있다")

print("\n%s" % ("test_args_equal PASS" if not bad else
                "test_args_equal FAIL %d건: %s" % (len(bad), ", ".join(bad))))
sys.exit(1 if bad else 0)
