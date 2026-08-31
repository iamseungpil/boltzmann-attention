# -*- coding: utf-8 -*-
"""X608 — 레버 마커 ↔ A2 선언 키 매핑, 그리고 **도메인별 선언 충족률** (2026-08-29·무료).

왜 (사용자 지시 2026-08-29): 특허를 증명하려면 도메인-일반 규칙이 **엔진에** 있어야 한다.
그런데 2026-08-29 에 retail 이 자기 7월 기준 대비 **−11.8pp** 회귀했고, 그 두 달 동안 한 일은
banking 저작과 그에 맞춘 엔진 조정이다. ⇒ **엔진이 문법적으로 도메인-일반이어도(리터럴 0)
거동이 중립이 아닐 수 있다.**

이 프로브가 재는 것 — *"선언이 비어 있는데도 발화하는 레버"*:
    레버가 읽는 A2 키를 그 도메인이 **선언하지 않았는데도** 그 레버가 도는가.
    돈다면 그 레버는 그 도메인에서 **엔진 기본값**으로 도는 것이고, 그 기본값은
    banking 을 보고 정해졌다. **그것이 중립성 결함의 조작적 정의다.**

⛔이것은 **구조 검사**이지 인과가 아니다([[70]]). 인과는 같은-sha 절제 A/B(`x607`)가 준다.
   이 표는 그 A/B 의 **대상 목록**을 만든다.

방법: 엔진 폐포의 각 함수에서 ⑴발화 마커 `[T2_...]` 와 ⑵그 함수가 참조하는 **A2 최상위 키**를
      함께 모은다(AST·함수 단위). 함수 밖 코드는 ±60줄 창으로 보완한다.
      A2 키의 권위 목록은 `x595_a2_key_role.py` 가 낸 71키를 쓴다([[67]] 사본 금지).
"""
import ast
import collections
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

MARK = re.compile(r"\[(T2_[A-Z0-9_]+)\]")


def engine_files():
    from x6h_engine_literal_audit import discover_engine_files
    return discover_engine_files()


def a2_keys_and_domains():
    """71 최상위 키와 도메인별 선언 여부 — x595 를 재사용한다."""
    import x595_a2_key_role as X
    rows = X.build()
    keys = set(rows)
    decl = {}
    for k, r in rows.items():
        decl[k] = set(r["domains"]) | ({"_shared"} if r["file_layer"].get("_shared") else set())
    return keys, decl


def strings_in(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return out


def scan():
    keys, decl = a2_keys_and_domains()
    m2k = collections.defaultdict(set)      # marker -> A2 keys
    m2f = collections.defaultdict(set)      # marker -> files
    seen_marks = set()

    for f in engine_files():
        p = os.path.join(_HERE, f)
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        base = os.path.basename(f)
        for mk in MARK.findall(src):
            seen_marks.add(mk)
            m2f[mk].add(base)

        # ★스코프는 **가장 가까운 블록**이어야 한다 (2026-08-29 자기교정).
        #   처음엔 함수 단위로 잡았는데 `t2_gate_patch.py` 가 하나의 거대한 함수라
        #   213 마커 중 대부분이 **같은 40키**를 가리키는 퇴화가 났다 — 스코프 실패이지 매핑이 아니다.
        #   그래서 마커가 찍힌 줄에서 **위로 12줄·아래로 12줄**만 본다. 레버는 자기 키를
        #   출력 직전·직후에서 읽는다. 창이 커지면 이웃 레버의 키를 빨아들인다.
        WIN = 12
        lines = src.splitlines()
        for i, line in enumerate(lines):
            found = MARK.findall(line)
            if not found:
                continue
            lo, hi = max(0, i - WIN), min(len(lines), i + WIN + 1)
            blob = "\n".join(lines[lo:hi])
            near = set(k for k in keys
                       if ('"%s"' % k) in blob or ("'%s'" % k) in blob)
            for mk in found:
                m2k[mk] |= near

    return m2k, m2f, decl, seen_marks


def main():
    m2k, m2f, decl, marks = scan()
    doms = ["banking_knowledge", "retail", "airline"]
    fired = {}
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        fired = json.load(io.open(sys.argv[1], encoding="utf-8"))

    rows = []
    for mk in sorted(marks):
        ks = sorted(m2k.get(mk) or ())
        cov = {}
        for d in doms:
            if not ks:
                cov[d] = None
            else:
                cov[d] = sum(1 for k in ks if d in (decl.get(k) or set()))
        rows.append((mk, ks, cov))

    print("레버 마커 %d · A2 키를 찾은 것 %d" % (len(rows), sum(1 for r in rows if r[1])))
    print()
    print("%-26s %4s %9s %9s %9s  %s"
          % ("marker", "keys", "banking", "retail", "airline", "선언 키"))
    # 선언 격차가 큰 순서 = banking 은 선언하는데 retail/airline 은 안 하는 것
    def gap(r):
        _, ks, c = r
        if not ks:
            return -1
        b = c["banking_knowledge"] or 0
        return b - min(c["retail"] or 0, c["airline"] or 0)
    for mk, ks, c in sorted(rows, key=gap, reverse=True):
        if not ks:
            continue
        print("%-26s %4d %9s %9s %9s  %s"
              % (mk[:26], len(ks), c["banking_knowledge"], c["retail"], c["airline"],
                 ",".join(ks[:4]) + ("…" if len(ks) > 4 else "")))
    n0 = sum(1 for r in rows if not r[1])
    print()
    print("A2 키를 못 찾은 마커 %d개 = **선언 없이 도는 것**(엔진 기본값 전용 후보)" % n0)
    print("  " + " ".join(sorted(mk for mk, ks, _ in rows if not ks))[:600])


if __name__ == "__main__":
    main()
