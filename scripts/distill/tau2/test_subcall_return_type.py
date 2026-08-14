# -*- coding: utf-8 -*-
"""`sub_generate` 반환형 오용 래칫 — **조용히 전부 죽는** 회귀를 막는다.

사건(2026-08-14 야간): `t2_subcall.sub_generate` 는 **문자열**을 반환한다(실패 시 `""`).
그런데 24곳 이관 뒤 **8곳**이 옛 계약(메시지 객체)을 그대로 두고 있었다:

    sub = SC.sub_generate(...)                       # → str
    raw = (getattr(sub, "content", None) or "")      # → 항상 ""  ← 무조건 실패

문자열에는 `.content` 가 없으므로 이 여덟 자리는 **입력이 무엇이든 실패**했다. 그중 하나가
`t2_ledger.formalize_now` 였고, 그래서 `T2_SEARCH_AGENT` 가 t7290 에서 **10회 발화·10회 침묵**
(`now 미확정·원값 None`)했다. 온톨로지 색인은 695/698 을 덮고 있었고 필요한 문서도 그 안에 있었다 —
**레버는 있었고 배선이 죽어 있었다**. 그 사실을 모른 채 072 의 결손을 프로브 3판으로 팠다.

같은 자리에 `t2_search.doc_decide`(검색 **결정** 단계)와 `t2_formalize_exec` 3곳도 있었다.

⚠이 부류가 위험한 이유: 예외도 로그도 남기지 않는다. 폴백이 조용히 정상처럼 보인다.
생존 감사(`t2_liveness`)가 잡아 준 것도 **침묵 로그가 있었던 한 곳**뿐이었다.
"""
import glob
import io
import os
import re
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CALL = re.compile(r"(\w+)\s*=\s*(?:_?SC\d?\.)?sub_generate\(")


def scan():
    bad = []
    for p in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        fn = os.path.basename(p)
        if fn.startswith("test_"):
            continue
        s = io.open(p, encoding="utf-8", errors="replace").read()
        for m in CALL.finditer(s):
            var = m.group(1)
            tail = s[m.end():m.end() + 400]
            if (re.search(r'getattr\(\s*%s\s*,\s*["\']content' % re.escape(var), tail)
                    or re.search(r"\b%s\.content\b" % re.escape(var), tail)):
                bad.append((fn, s[:m.start()].count("\n") + 1, var))
    return bad


def main():
    print("정본: t2_subcall.sub_generate → **str** (실패 시 '')\n")
    bad = scan()
    for fn, ln, var in bad:
        print("  FAIL %s:%d  `%s` 를 메시지 객체로 다룬다" % (fn, ln, var))
    if bad:
        print("\nFAIL — 반환형 오용 %d곳. 고치는 법: `getattr(sub, \"content\", None) or \"\"` 를"
              " **`sub` 자체**로 바꿔라(이미 문자열이다)." % len(bad))
        return 1
    print("  ok   반환형 오용 0곳")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
