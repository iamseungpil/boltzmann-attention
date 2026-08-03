# -*- coding: utf-8 -*-
"""④ 회수 경계 표면화 회귀 — 세는 연산이 설계서 §4.2와 같은 것인가만 본다.

불변식:
  ① 세는 연산 = 내용어 AND(임계 없음·분모=전 코퍼스) — 구 매칭이 아니다
  ② 표시 수 >= 매칭 수이면 "all N shown"(= 전수를 봤다는 근거)
  ③ 매칭 0은 "이 단어들을 포함한 문서가 없다"로만 말한다(세계에 대한 진술 금지)
  ④ 코퍼스가 없으면 무발화(None) — 조용히 틀린 수를 내지 않는다
"""
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_match_count as MC  # noqa: E402

FAILED = []


def chk(cond, name):
    print(("  ✓ " if cond else "  ✗ ") + name)
    if not cond:
        FAILED.append(name)


def main():
    d = tempfile.mkdtemp()
    docs = {
        "a.md": "Transfer the customer to a human agent when the policy says so.",
        "b.md": "A human operator may be needed. Transfer requests are logged.",
        "c.md": "Credit card rewards are calculated per transaction.",
        "d.md": "The agent should transfer only after asking.",
    }
    for k, v in docs.items():
        io.open(os.path.join(d, k), "w", encoding="utf-8").write(v)
    os.environ["T2_KB_DOCS_DIR"] = d
    MC._CACHE.clear()
    corpus = MC.load_corpus(None)

    print("[①] 내용어 AND ≠ 구 매칭")
    chk(MC.count("human agent", corpus) == 1, "'human agent' = 1 (a만 두 단어 모두)")
    chk(MC.count("transfer", corpus) == 3, "'transfer' = 3 (a,b,d)")
    chk(MC.content_words("how to transfer the customer") == ["transfer", "customer"],
        "불용어 제거 후 내용어만")

    print("\n[②] 전수를 봤다는 근거")
    n = MC.note("human agent", "1. a.md\n", None)
    chk("all 1 shown" in n, "표시>=매칭 -> all N shown: %r" % n)
    n = MC.note("transfer", "1. a\n2. b\n", None)
    chk("2 shown (1 not shown)" in n, "표시<매칭 -> 미표시 수 명시: %r" % n)

    print("\n[③] 0은 단어에 대한 진술")
    n = MC.note("zebra quantum", "1. a\n", None)
    chk("no document contains all of these words" in n, "0 문구: %r" % n)
    chk("nothing" not in n.lower() and "no results" not in n.lower(),
        "세계에 대한 진술 금지")

    print("\n[④] 코퍼스 없으면 무발화")
    os.environ.pop("T2_KB_DOCS_DIR", None)
    MC._CACHE.clear()
    chk(MC.note("human agent", "1. a\n", None) is None, "코퍼스 미해결 -> None")

    print("\nRESULT: " + ("ALL PASS" if not FAILED else "FAIL " + str(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
