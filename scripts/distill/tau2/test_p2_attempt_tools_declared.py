#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P2 — 「검증 시도」 판정 도구가 **선언에서** 오는가 (x737 §9b · [[05]]).

구판 `t2_phase.py:75` 는 `gather = {"verify_identity"}` 였다 — 엔진에 은행 도구명 리터럴.
[[05]] 고정층(Scaffold 엔진)에 도메인 특화가 들어간 자리이고 [[58]] 위반이다.

반증조건: ①선언했는데 verify 로 안 가면 FAIL ②미선언인데 verify 로 가면 FAIL
          ③엔진 파일에 은행 도구명이 남아 있으면 FAIL
"""
import io, json, os, re, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_phase as PH

HERE = os.path.dirname(os.path.abspath(__file__))


class _M(object):
    def __init__(self, names): self.tool_calls = [{"name": n} for n in names]


def _unwrap(tc): return tc.get("name")


def _a2(attempt=None):
    g = {"id": "G_AUTH", "kind": "auth", "satisfiers": {"log_verification": ["name"]}}
    if attempt is not None:
        g["attempt_tools"] = attempt
    return {"gates": [g]}


class P2Declared(unittest.TestCase):

    def test_declared_attempt_tool_reaches_verify(self):
        ph, why = PH.phase_of(_a2(["verify_identity"]), [_M(["verify_identity"])], _unwrap)
        self.assertEqual(ph, "verify", why)

    def test_undeclared_domain_is_unchanged(self):
        """airline/retail 처럼 선언이 없으면 이 가지는 **발화하지 않는다**(거동 변화 0)."""
        ph, _ = PH.phase_of(_a2(None), [_M(["verify_identity"])], _unwrap)
        self.assertNotEqual(ph, "verify")

    def test_satisfier_called_skips_the_gate(self):
        """게이트가 이미 충족되면 verify 단계가 아니다(구판 거동 보존)."""
        ph, _ = PH.phase_of(_a2(["verify_identity"]),
                            [_M(["log_verification", "verify_identity"])], _unwrap)
        self.assertNotEqual(ph, "verify")

    def test_other_declared_tool_name_works(self):
        """도메인이 다른 이름을 선언해도 그대로 동작해야 한다(엔진은 이름을 모른다)."""
        ph, _ = PH.phase_of(_a2(["confirm_passenger_id"]), [_M(["confirm_passenger_id"])], _unwrap)
        self.assertEqual(ph, "verify")

    def test_no_domain_literal_left_in_engine(self):
        src = io.open(os.path.join(HERE, "t2_phase.py"), encoding="utf-8").read()
        code = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
        for lit in ("verify_identity", "log_verification", "change_user_email",
                    "call_discoverable_agent_tool"):
            self.assertNotIn('"%s"' % lit, code, "엔진에 도메인 리터럴 %s 가 남아 있다" % lit)
            self.assertNotIn("'%s'" % lit, code, "엔진에 도메인 리터럴 %s 가 남아 있다" % lit)

    def test_a2_layers_declare_it(self):
        for f in ("a2/banking_knowledge.gate.json", "a2/split/banking_knowledge.core.json"):
            d = json.load(io.open(os.path.join(HERE, f), encoding="utf-8"))
            au = [g for g in (d.get("gates") or []) if g.get("kind") == "auth"]
            self.assertTrue(au, "%s 에 auth 게이트가 없다" % f)
            self.assertIn("verify_identity", set(au[0].get("attempt_tools") or ()),
                          "%s 가 attempt_tools 를 선언하지 않았다 ([[24]] 양방향)" % f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
