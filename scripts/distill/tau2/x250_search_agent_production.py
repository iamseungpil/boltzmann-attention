# -*- coding: utf-8 -*-
r"""x250 — **프로덕션 경로 그대로** 071 을 돌린다 (배선 직전 마지막 관문 · 유료 0 · 8141).

## 왜

x248 은 프로브가 **자기 문구**로 돌렸다. 배선하면 도는 것은 그게 아니라 **A2 문구 +
`t2_search.formalize_group` / `decide_from_docs`** 다. 둘이 같은 답을 내는지 확인하지 않으면,
라이브에서 갈렸을 때 *프로브가 틀렸나 배선이 틀렸나*를 못 가른다(오늘만 그 자리에서 두 번 헤맸다).

  ① `t2_search.formalize_group(...)`  ← A2 `policy_ontology.group_prompt`
  ② `t2_search.material_for(...)`     ← A3 `doc_index` + `doc_windows` (엔진)
  ③ `t2_search.decide_from_docs(...)` ← A2 `policy_ontology.doc_decide_prompt`

기대: x248 재측정판과 같은 자리 — checking `Sky Blue` · savings `Gold Saver Account`.

⚠**8141 전용**([[30]]·사용자 지시 2026-08-11). 8140 은 유료 런 자리다.
⚠gold 는 채점에만.

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python x250_search_agent_production.py [N]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as S                                             # noqa: E402
from x216_read_and_offset import chat, URL                        # noqa: E402
from x248_search_agent_e2e import a2, requirements, DOCS, NOW, GOLD   # noqa: E402


class _Msg(object):
    def __init__(self, role, content):
        self.role, self.content = role, content


class _UM(object):
    def __init__(self, role=None, content=None):
        self.role, self.content = role or "user", content


class _LA(object):
    """`llm_agent` 자리 — 온도 0·이름만 받으면 되므로 짧게."""
    MAX = 32

    @staticmethod
    def generate(model=None, tools=None, messages=(), call_name=None, **kw):
        p = "".join(str(getattr(m, "content", "") or "") for m in messages)
        try:
            out = chat(p, None, 0.0, _LA.MAX).get("content", "")
        except Exception as e:
            out = "ERR %s" % type(e).__name__
        return _Msg("assistant", out)


class _Agent(object):
    llm = None
    llm_args = {}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    if ":8141" not in URL:
        print("중단: 8141 이 아니다 (%s) — 유료 런 자리를 쓰지 않는다." % URL)
        return 1
    A = a2()
    spec = (A.get("policy_ontology") or {})
    groups = list(spec.get("doc_index") or {})
    req = requirements()
    print("URL %s · 군 %d · group_prompt %s · doc_decide_prompt %s"
          % (URL, len(groups), bool(spec.get("group_prompt")), bool(spec.get("doc_decide_prompt"))))
    for axis, lead in (("business_checking", "The customer is asking which BUSINESS CHECKING "
                                             "account to open."),
                       ("business_savings", "The customer is asking which BUSINESS SAVINGS "
                                            "account to open.")):
        ask = lead + "\n\n" + req
        g = S.formalize_group(_Agent(), _LA, _UM, spec, [ask], groups)
        print("=" * 92)
        print("%s · gold=%r · ① %s" % (axis, GOLD[axis], g or "침묵"))
        if not g:
            continue
        mat, info = S.material_for(A, g, DOCS, NOW)
        print("   ② 재료 %d자 · 문서 %d(뺀 것 %d: %s)"
              % (len(mat), info["kept"], len(info["dropped"]), ", ".join(info["dropped"]) or "-"))
        c = collections.Counter()
        _LA.MAX = 24
        for i in range(n):
            out = S.decide_from_docs(_Agent(), _LA, _UM, spec, mat, ask)
            c[str(out)[:44]] += 1
        gold = GOLD[axis]
        hit = sum(v for k, v in c.items()
                  if re.fullmatch(r"%s( Account)?" % re.escape(gold), str(k).strip(), re.I))
        print("   ③ gold %d/%d   %s" % (hit, n, c.most_common(2)))
    print("\n※ x248 재측정판과 같으면 프로브와 프로덕션이 갈리지 않는다 — 그때 배선한다."
          "\n  갈리면 **배선하지 말고** 어느 쪽이 다른지부터 밝힌다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
