# -*- coding: utf-8 -*-
"""A2 3층 분리 회귀 — 분리가 **내용을 바꾸지 않았는가**만 본다.

정본 = `A2_THREE_LAYER_SPLIT_DESIGN_2026_07_31.md`. 이 분리의 가치는 회계이지 성능이 아니므로,
**동작이 바뀌면 그게 버그**다. 그래서 여기서 강제하는 불변식은 셋뿐이다:
  ① 병합(L1+L2+L3) == 분리 전(base + 단일파일)  — 실키 기준(`_`는 주석)
  ② 한 키가 두 파일에 동시에 있지 않다          — duplication이 회계를 망친다
  ③ 분리 파일이 없어도 로더가 단일파일로 동작    — 구 체크아웃 호환
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.path.join(HERE, "a2")
sys.path.insert(0, HERE)
try:                                       # 콘솔 코드페이지가 cp949면 체크마크에서 죽는다
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import gate_interpreter as G  # noqa: E402

DOMAINS = ["banking_knowledge", "retail", "airline"]
OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def real(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def test_equivalence():
    print("[①] 병합 == 분리 전")
    base = real(load(os.path.join(A2, "base", "shared.json")))
    for dom in DOMAINS:
        mono = os.path.join(A2, dom + ".gate.json")
        if not os.path.exists(mono):
            continue
        want = dict(base)
        want.update(real(load(mono)))
        got = real(G.load_domain_a2(dom))
        chk(canon(got) == canon(want), "%s 실키 %d개 일치" % (dom, len(want)))


def test_no_duplicate():
    print("[②] 키가 한 층에만 존재")
    base = set(real(load(os.path.join(A2, "base", "shared.json"))))
    for dom in DOMAINS:
        st = os.path.join(A2, dom + ".settings.json")
        sp = os.path.join(A2, dom + ".specific.json")
        if not (os.path.exists(st) and os.path.exists(sp)):
            chk(False, "%s 분리 파일 부재(x18 --emit 필요)" % dom)
            continue
        s, p = set(real(load(st))), set(real(load(sp)))
        chk(not (s & p), "%s settings ∩ specific = ∅" % dom)
        chk(not (base & s) and not (base & p), "%s base와 겹치지 않음" % dom)


def test_fallback():
    print("[③] 분리 파일 없는 도메인 = 단일파일 폴백")
    chk(G.load_domain_a2("nosuch_domain_xyz") is None, "미존재 도메인 = None(게이트 비활성)")


if __name__ == "__main__":
    for fn in (test_equivalence, test_no_duplicate, test_fallback):
        fn()
        print()
    print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
