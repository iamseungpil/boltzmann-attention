# -*- coding: utf-8 -*-
r"""⒡ 범주 인용 게이트 회귀 검정 — **규칙 0 을 지키는 형태**인지까지 고정한다

계약:
    ⒜ 플래그가 없으면 아무 일도 안 한다(ctl = 종전 스택)
    ⒝ 인용이 없으면 `spend_category` 를 **떨군다** ⇒ 값 주석은 기본 요율로 계산된다
    ⒞ 거절문이 **무엇을 하면 되는지** 말한다 — A2 색인의 **문서 id**를 이름 대고 재호출을 지시([[64]])
    ⒟ ★**엔진은 문서 내용을 읽지 않는다** — id 만 가리킨다(규칙 0·`SCAFFOLD_AUDIT_RULE0_2026_07_08`)
    ⒠ 인용이 **에이전트 자신이 가져온 도구 출력**에 실재하면 범주가 살아남는다
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def a2():
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    ok = True
    d = a2()
    tool = [x for x in d["scaffold_get_tools"] if x["name"] == "check_card_application_fit"][0]

    # ⒞ 색인이 선언돼 있고 id 만 담는다(내용 0)
    idx = (d.get("catalog_arg_docs") or {}).get("spend_category") or {}
    cats = [k for k in idx if not k.startswith("_")]
    ids = [x for k in cats for x in idx[k]]
    print("  %s 색인 선언 — 범주 %d · 문서 id %d" % ("OK" if cats and ids else "X", len(cats), len(ids)))
    ok = ok and bool(cats) and bool(ids)

    # ⒟ 선언에 **문서 본문이 없다**(id·제목만) = 규칙 0 안전
    body_like = [x for x in ids if len(x) > 120 or " " in x]
    print("  %s 선언에 문서 **내용**이 없다(id 만·%d 건 위반)" % ("OK" if not body_like else "X", len(body_like)))
    ok = ok and not body_like

    # ⒝ 인용 파라미터가 선언돼 있다
    has_q = "spend_category_quote" in (tool.get("params") or {})
    print("  %s `spend_category_quote` 파라미터 선언" % ("OK" if has_q else "X"))
    ok = ok and has_q

    # ⒜ 엔진 코드가 플래그 뒤에 있고, 문서를 읽는 코드가 없다
    src = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
    # ★앵커 정정(2026-08-21): `T2_CATEGORY_CITE` 는 배달 배선의 **상호배제 조건**에도 나온다
    #   — 첫 등장을 잡으면 엉뚱한 블록을 검사한다. 게이트 자신의 주석 표지에 건다.
    i = src.find("★⒡ 범주 인용 게이트")
    seg = src[i:i + 2600] if i > 0 else ""
    flagged = i > 0
    reads_docs = ("DOCDIR" in seg) or ("open(" in seg) or ("glob" in seg)
    print("  %s 게이트가 플래그 뒤에 있다" % ("OK" if flagged else "X"))
    print("  %s 게이트가 **문서를 읽지 않는다**(DOCDIR/open/glob 부재) = 규칙 0" % ("OK" if not reads_docs else "X"))
    ok = ok and flagged and not reads_docs

    # ⒠ 검산은 에이전트 자신의 도구 출력에 건다
    own = "__tool_outputs" in seg
    print("  %s 검산 대상이 **에이전트 자신이 가져온 도구 출력**이다" % ("OK" if own else "X"))
    ok = ok and own

    print("\nRESULT: %s" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
