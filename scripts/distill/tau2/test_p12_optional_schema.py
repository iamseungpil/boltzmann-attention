#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P12 — A2 `optional` 선언이 **스키마 required 에서 빠지는가** (x737 §9b · x817 §4).

왜: `_build_tool` 이 모든 인자를 `p: str`(기본값 없음)로 냈고, 파이썬이 그것을 필수로 만들고,
tau2 `parse_data` 가 스키마 required 를 유도하고, `_schema_required` 가 읽어 `[ARG-EMPTY]` 로
반려했다. A2 는 `check_card_application_fit` 을 **13/13 optional** 로 선언했는데(하나는 축자
*"leave this out"*) 스키마는 13/13 required 였다 — 우리 지시를 따른 모델을 우리가 벌줬다.

반증조건: `optional` 을 선언했는데 시그니처에 기본값이 없으면 FAIL.
          `optional` 미선언 도구에 기본값이 생기면 FAIL(거동 변화 0 이어야 한다).
"""
import inspect, io, json, os, re, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_scaffold_get as SG

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = [os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
      os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
      os.path.join(HERE, "a2", "split", "banking_knowledge.core.json")]


class _StubTool(object):
    """tau2 `Tool` 대역 — 우리가 넘긴 **함수 객체**만 붙잡는다(스키마 유도는 tau2 몫)."""
    def __init__(self, fn, examples=None):
        self.fn = fn
        self.examples = examples


def _sig(decl):
    return inspect.signature(SG._build_tool(_StubTool, decl).fn).parameters


class P12Optional(unittest.TestCase):

    def test_declared_optional_gets_default(self):
        d = {"name": "t_a", "description": "d",
             "params": {"x": "required thing", "y": "number, optional"},
             "optional": ["y"]}
        p = _sig(d)
        self.assertIs(p["x"].default, inspect.Parameter.empty, "필수 인자에 기본값이 붙었다")
        self.assertEqual(p["y"].default, "", "optional 인자에 기본값이 없다 → 스키마 required")

    def test_no_optional_key_is_unchanged(self):
        """`optional` 미선언 = 구판과 **완전히 동일**해야 한다(진행 중 팔 보호·[[54]])."""
        d = {"name": "t_b", "description": "d", "params": {"x": "a", "y": "b", "z": "c"}}
        p = _sig(d)
        self.assertEqual(list(p), ["x", "y", "z"], "인자 순서가 바뀌었다")
        for k in p:
            self.assertIs(p[k].default, inspect.Parameter.empty, "%s 에 기본값이 생겼다" % k)

    def test_optional_params_come_last(self):
        """파이썬 문법: 기본값 있는 인자는 뒤. 앞에 오면 컴파일 자체가 실패한다."""
        d = {"name": "t_c", "description": "d",
             "params": {"a": "opt", "b": "req", "c": "opt"}, "optional": ["a", "c"]}
        p = _sig(d)
        self.assertEqual(list(p), ["b", "a", "c"])

    def test_docstring_keeps_declaration_order(self):
        """모델이 보는 설명 순서는 선언 순서 그대로여야 한다(시그니처 재정렬과 무관)."""
        d = {"name": "t_d", "description": "d",
             "params": {"a": "opt", "b": "req"}, "optional": ["a"]}
        doc = SG._build_tool(_StubTool, d).fn.__doc__
        self.assertLess(doc.index(":param a:"), doc.index(":param b:"))

    def test_a2_three_layers_declare_optional(self):
        """A2 3층 전부에 구조화 선언이 있고, **설명 축자와 일치**해야 한다([[24]] 양방향)."""
        for f in A2:
            self.assertTrue(os.path.exists(f), "A2 층 없음: %s" % f)
            d = json.load(io.open(f, encoding="utf-8"))
            for t in (d.get("scaffold_get_tools") or []):
                ps = t.get("params") or {}
                said = {p for p, desc in ps.items() if re.search(r"\boptional\b", str(desc), re.I)}
                decl = set(t.get("optional") or [])
                self.assertEqual(decl, said,
                    "%s / %s — 구조화 선언과 설명 축자가 어긋난다 (선언=%s 설명=%s)"
                    % (os.path.basename(f), t.get("name"), sorted(decl), sorted(said)))

    def test_the_real_tool_is_fully_optional(self):
        """실물 회귀 방지: `check_card_application_fit` 13/13 이 기본값을 받아야 한다."""
        d = json.load(io.open(A2[0], encoding="utf-8"))
        tool = next(t for t in d["scaffold_get_tools"] if t["name"] == "check_card_application_fit")
        p = _sig(tool)
        self.assertEqual(len(p), 13)
        for k in p:
            self.assertEqual(p[k].default, "", "%s 가 여전히 필수다" % k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
