# -*- coding: utf-8 -*-
r"""미임포트 모듈명 검정 (2026-08-12 신설 · 죽은-레버 3호 사고).

사고: `t2_resolve.py` 가 `sys.stderr` 인쇄 3곳을 쓰면서 `import sys` 가 없었다.
`formalize_arg_axis` 의 **성공 경로** print 가 NameError 를 던져 바깥
`T2_WRITE_ARG_ENUM` try 까지 통째로 죽였고(070/071 g런: 집합外
'Cobalt Blue Business Checking Account' 무검사 통과), 로그엔
"건너뜀(무발화): NameError" 로만 남아 **계기의 사각이 음성 관측으로 보였다**.
`test_regen_break_guard` 도 기존 레버 검정도 이 부류를 안 본다 — 그래서 상설.

검사: 엔진 경로 모듈 전수에서 `X.attr` 로 쓰인 이름 X 가 모듈 내 어디서도
정의(임포트·대입·def·인자·global)되지 않으면 FAIL. 휴리스틱이지만 이 사고
부류(모듈을 임포트 없이 attribute 접근)는 정확히 잡는다.

⚠제외: `t2_a2_scale_census.py` — HEAD 부터 구문 오류(별건·standalone·임포트 0곳).
"""
import ast
import builtins
import glob
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 라이브 엔진 경로 + 포렌식 도구. census 는 HEAD 부터 깨진 standalone(별건).
TARGETS = sorted(set(glob.glob(os.path.join(HERE, "t2_*.py"))
                     + glob.glob(os.path.join(HERE, "x27*.py"))))
EXCLUDE = {"t2_a2_scale_census.py"}


def undefined_module_names(src):
    """X.attr 의 X 중 모듈 내 미정의 이름 → {이름: 첫 행}."""
    tree = ast.parse(src)
    defined = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            defined |= {(a.asname or a.name.split(".")[0]) for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            defined |= {(a.asname or a.name) for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            defined.add(n.id)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, ast.Global):
            defined |= set(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                if isinstance(it.optional_vars, ast.Name):
                    defined.add(it.optional_vars.id)
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            nm = n.value.id
            if nm not in defined and nm not in out:
                out[nm] = n.lineno
    return out


def conditional_import_escapes(src):
    """죽은-레버 4호 부류: 함수 안에서 `import X as Y` 가 **If 분기 안에만** 있는데
    Y 를 그 분기 subtree **밖**에서 쓰면 UnboundLocalError (h런 실측: `_rz` —
    resolve-계약 분기 안 임포트를 AXIS 분기가 참조). → {(함수, 이름): 사용 행}."""
    tree = ast.parse(src)
    out = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # 이 함수의 **직속** 본문만 본다(중첩 함수는 자기 차례에 별도로 처리).
        nodes = []
        stack = list(fn.body)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nodes.append(n)
            stack.extend(ast.iter_child_nodes(n))
        # 이름별: 임포트 바인딩(및 그것을 감싸는 If subtree 행범위)·일반 대입·사용
        imports = {}      # name -> [(lineno, if_range or None)]
        stores = {}       # name -> [If 밖 무조건 대입 존재?]
        ifs = [n for n in nodes if isinstance(n, ast.If)]

        def if_range_of(node):
            best = None
            for i in ifs:
                s, e = i.lineno, getattr(i, "end_lineno", i.lineno)
                if s <= node.lineno <= e and (best is None or s > best[0]):
                    best = (s, e)
            return best

        for n in nodes:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    nm = a.asname or a.name.split(".")[0]
                    imports.setdefault(nm, []).append((n.lineno, if_range_of(n)))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                if if_range_of(n) is None:
                    stores[n.id] = True
        for n in nodes:
            if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
                continue
            nm = n.id
            if nm not in imports or stores.get(nm):
                continue
            covered = any(
                ln < n.lineno and (rng is None or rng[0] <= n.lineno <= rng[1])
                for ln, rng in imports[nm])
            if not covered:
                out.setdefault((fn.name, nm), n.lineno)
    return out


def main():
    ok = True

    # ── 양성 대조: 스캐너가 사고 그 자체(임포트 없는 sys.stderr)를 잡는가 ──
    planted = "import os\ndef f():\n    print('x', file=sys.stderr)\n"
    got = undefined_module_names(planted)
    if "sys" in got:
        print("  양성 대조: 심은 미임포트 sys 를 잡는다                PASS")
    else:
        print("  양성 대조: 심은 버그를 못 잡는다                     FAIL")
        ok = False

    # ── 음성 대조: 정의된 이름은 울지 않는가 ──
    clean = "import sys\nimport os as o\nx = object()\nprint(o.sep, x.__class__, file=sys.stderr)\n"
    if not undefined_module_names(clean):
        print("  음성 대조: 정상 코드에 무발화                        PASS")
    else:
        print("  음성 대조: 정상 코드에 오탐                          FAIL")
        ok = False

    # ── 양성 대조 2 (죽은-레버 4호 부류): If 분기 안 임포트를 딴 분기가 쓰는가 ──
    planted2 = (
        "def f(a, b):\n"
        "    if a:\n"
        "        import json as _rz\n"
        "        _rz.dumps({})\n"          # 같은 subtree — OK
        "    if b:\n"
        "        _rz.loads('{}')\n"        # 딴 분기 — UnboundLocalError 부류
        "    return 0\n")
    got2 = conditional_import_escapes(planted2)
    if ("f", "_rz") in got2:
        print("  양성 대조2: 분기-탈출 임포트 사용을 잡는다           PASS")
    else:
        print("  양성 대조2: 분기-탈출 임포트 사용을 못 잡는다        FAIL")
        ok = False
    # 음성 대조 2: 각 사용 분기가 자기 임포트를 가지면 무발화 + 선행 무조건 대입도 면제
    clean2 = (
        "def g(a, b):\n"
        "    _x = None\n"
        "    if a:\n"
        "        import json as _x\n"
        "    if _x:\n"
        "        _x.loads('{}')\n"
        "def h(a, b):\n"
        "    if a:\n"
        "        import json as _y\n"
        "        _y.dumps({})\n"
        "    if b:\n"
        "        import json as _y\n"
        "        _y.loads('{}')\n")
    if not conditional_import_escapes(clean2):
        print("  음성 대조2: 안전 패턴에 무발화                       PASS")
    else:
        print("  음성 대조2: 안전 패턴에 오탐 %s                    FAIL"
              % conditional_import_escapes(clean2))
        ok = False

    # ── 본검사: 전수 스캔 ──
    for path in TARGETS:
        base = os.path.basename(path)
        if base in EXCLUDE:
            continue
        src = io.open(path, encoding="utf-8").read()
        try:
            bad = undefined_module_names(src)
            bad2 = conditional_import_escapes(src)
        except SyntaxError as e:
            print("  %-38s 구문 오류: %s                FAIL" % (base, e))
            ok = False
            continue
        if bad:
            for nm, ln in sorted(bad.items(), key=lambda kv: kv[1]):
                print("  %-38s :%d 미정의 모듈명 %r        FAIL" % (base, ln, nm))
            ok = False
        if bad2:
            for (fname, nm), ln in sorted(bad2.items(), key=lambda kv: kv[1]):
                print("  %-38s :%d %s() 분기-탈출 임포트 %r  FAIL"
                      % (base, ln, fname, nm))
            ok = False
    if ok:
        print("  엔진 경로 %d개 모듈: 미정의 모듈명 0                 PASS"
              % len([p for p in TARGETS if os.path.basename(p) not in EXCLUDE]))
    print("\nRESULT: %s" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
