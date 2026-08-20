# -*- coding: utf-8 -*-
r"""배달 배선(`T2_ARG_DOC_SUB`) 회귀 검정 — **분담**이 코드에 고정돼 있는지까지 본다

계약([[71]] 격리 서브에이전트 계약):
    ⒜ 플래그가 없으면 아무 일도 안 한다 (ctl = 종전 스택·바이트 동일)
    ⒝ **문서 id 리터럴이 엔진에 없다** — 인자명·값 집합·문서 id 전부 A2 선언에서 나온다([[05]])
    ⒞ 엔진이 읽는 경로가 **정본**이다(`t2_search.corpus_from_env`/`read_docs`) — 새 I/O 0([[67]])
    ⒟ 서브 호출이 **정본**이다(`t2_subcall.sub_generate`) — 인라인 la.generate 0([[67]])
    ⒠ 검산이 **닫힌 술어 둘**뿐이다 — 값의 선언-집합 소속 · 인용의 실재(`t2_search.quote_in`)
    ⒡ 엔진이 **고르지 않는다** — 순위/최댓값 집기/"정답은 X" 가 그 블록에 없다([[62]])
    ⒢ 서브가 근거를 못 대면 인자는 **없는 채로** 둔다(⒡ ⊃ ⒟ — 최악이 기본 요율)
    ⒤ 같은 (인자·손님 발화)에는 **한 번만** 묻는다(중복 서브 호출 0)
    ⒣ A2 색인이 3부에 **동기화**돼 있고 선언된 문서가 실재한다([[24]])
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

BLOCK_START = "★배달 배선 — A2 선언 문서를"
BLOCK_END = "★⒡ 범주 인용 게이트"


def engine_block():
    with io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8") as f:
        t = f.read()
    i, j = t.find(BLOCK_START), t.rfind(BLOCK_END)
    assert i >= 0 and j > i, "배달 배선 블록을 못 찾음"
    return t[i:j]


def a2(layer):
    p = {"gate": "banking_knowledge.gate.json",
         "specific": "banking_knowledge.specific.json",
         "core": os.path.join("split", "banking_knowledge.core.json")}[layer]
    with io.open(os.path.join(HERE, "a2", p), encoding="utf-8") as f:
        return json.load(f)


def main():
    ok = True
    blk = engine_block()
    code = "\n".join(l for l in blk.split("\n") if not l.strip().startswith("#"))

    # ⒜ 플래그 게이트
    t = ("os.environ.get(\"T2_ARG_DOC_SUB\") == \"1\"" in code)
    print("  %s ⒜ 플래그 없으면 무동작" % ("OK" if t else "X"))
    ok = ok and t

    # ⒝ 문서 id·범주 리터럴 0
    lits = re.findall(r"[\"']doc_[a-z0-9_()]+[\"']", code) + re.findall(
        r"[\"'](?:travel|software|operations|media_advertising|green)[\"']", code)
    t = not lits
    print("  %s ⒝ 도메인 리터럴 0 %s" % ("OK" if t else "X", ("— 발견: %r" % lits[:3]) if lits else ""))
    ok = ok and t

    # ⒞⒟ 정본 경로
    for tag, needle in (("⒞ 읽기=t2_search 정본", "corpus_from_env"),
                        ("⒞ 읽기=t2_search 정본", "read_docs"),
                        ("⒟ 서브=t2_subcall 정본", "sub_generate")):
        t = needle in code
        print("  %s %s (%s)" % ("OK" if t else "X", tag, needle))
        ok = ok and t
    t = "la.generate" not in code
    print("  %s ⒟ 인라인 la.generate 0" % ("OK" if t else "X"))
    ok = ok and t

    # ⒠ 닫힌 술어 둘
    t = ("quote_in" in code) and ("in _vals" in code)
    print("  %s ⒠ 검산 = 인용 실재 + 선언-집합 소속" % ("OK" if t else "X"))
    ok = ok and t

    # ⒡ 엔진이 고르지 않는다
    bad = [s for s in ("argmax", "sorted(", "max(", "[0][0]", "rank") if s in code]
    t = not bad
    print("  %s ⒡ 엔진 선택 0 %s" % ("OK" if t else "X", ("— 발견: %r" % bad) if bad else ""))
    ok = ok and t

    # ⒢ 서브가 null 을 내면 인자는 **없는 채로** 둔다(=기본 요율·⒡ ⊃ ⒟)
    t = "_ctx.pop(_ag2, None)" in code
    print("  %s ⒢ 근거 없으면 인자 없음 = 기본 요율" % ("OK" if t else "X"))
    ok = ok and t

    # ⒤ 같은 (인자·손님 발화)에는 한 번만 묻는다(호출마다 서브를 새로 띄우지 않는다)
    t = ("_t2_argdoc_memo" in code) and ("메모 재사용" in code)
    print("  %s ⒤ 메모로 중복 서브 호출 차단" % ("OK" if t else "X"))
    ok = ok and t

    # ⒣ 3부 동기화 + 문서 실재
    decls = {}
    for layer in ("gate", "specific", "core"):
        decls[layer] = (a2(layer).get("catalog_arg_docs") or {}).get("spend_category") or {}
    t = decls["gate"] == decls["specific"] == decls["core"] and bool(decls["gate"])
    print("  %s ⒣ A2 3부 동기화" % ("OK" if t else "X"))
    ok = ok and t
    ids = sorted({x for k, v in decls["gate"].items() if k[:1] != "_" for x in v})
    print("     선언 문서 %d편 · 범주 %d"
          % (len(ids), len([k for k in decls["gate"] if k[:1] != "_"])))

    print("\n%s" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
