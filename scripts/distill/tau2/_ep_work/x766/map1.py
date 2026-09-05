# -*- coding: utf-8 -*-
"""x766: A2 key -> engine function -> emitted [TAG] literal 지도.

목적: 사이드카에 남은 발화 태그를 A2 키로 되돌리기 위한 조인 테이블.
읽기 전용(엔진/ a2 수정 0).
"""
import ast
import glob
import json
import os
import re
import sys

ENG = r"C:\workspace\ba-frft\scripts\distill\tau2"
KEYS = json.load(open(sys.argv[1], encoding="utf-8"))
TAG = re.compile(r"\[([A-Z][A-Z0-9 _\-]{2,40})\]")

files = sorted(glob.glob(os.path.join(ENG, "t2_*.py")))
files += [os.path.join(ENG, "gate_interpreter.py")]

tagloc = {}
keyloc = {}

for path in files:
    if not os.path.exists(path):
        continue
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception:
        continue
    base = os.path.basename(path)
    lines = src.split("\n")
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, getattr(node, "end_lineno", node.lineno), node.name))
    spans.sort()

    def func_at(ln):
        best = None
        for a, b, n in spans:
            if a <= ln <= b:
                if best is None or a >= best[0]:
                    best = (a, b, n)
        return best[2] if best else "<module>"

    for i, L in enumerate(lines, 1):
        s = L.lstrip()
        if s.startswith("#"):
            continue
        for m in TAG.finditer(L):
            tagloc.setdefault(m.group(1), []).append([base, func_at(i), i])
        for k in KEYS:
            if ('"%s"' % k) in L or ("'%s'" % k) in L:
                keyloc.setdefault(k, []).append([base, func_at(i), i])

json.dump({"tagloc": tagloc, "keyloc": keyloc},
          open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("tags", len(tagloc), "keys_found", len(keyloc))
print("keys_with_no_engine_read:", [k for k in KEYS if k not in keyloc])
