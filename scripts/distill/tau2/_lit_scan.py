# -*- coding: utf-8 -*-
"""_lit_scan — scaffold_guard.py 의 축을 엔진 파일 전체에 그대로 적용한다.

축의 정의는 이 파일이 아니라 C:/workspace/.claude/hooks/scaffold_guard.py +
scaffold_rules.json 이다. 여기서는 **같은 리터럴 목록**을 읽어 같은 판정을 하되,
훅이 못 하는 것 하나를 더 한다 = tokenize 로 **코드 / 문자열 / 주석 / docstring** 을 가른다.
(훅 자기고백 축자: "보지 않는 것: 문자열 밖 로직·주석·새 위반 유형")

출력 = JSON. 세는 법은 전부 여기 코드에 있다(재현 가능).
"""
import io
import json
import os
import re
import sys
import tokenize

RULES_PATH = r"C:/workspace/.claude/hooks/scaffold_rules.json"
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))

RULES = json.load(io.open(RULES_PATH, encoding="utf-8"))

# ── 카테고리 맵: 파일의 각 문자 위치가 code / string / docstring / comment 중 무엇인가 ──
CODE, STR, DOC, COM = "code", "string", "docstring", "comment"


def categorize(src):
    """길이 len(src) 의 카테고리 배열을 만든다. 기본 = code(공백·연산자 포함)."""
    cats = [CODE] * len(src)
    # 줄 시작 오프셋
    starts, off = [], 0
    for line in src.splitlines(True):
        starts.append(off)
        off += len(line)
    starts.append(off)

    def pos(r, c):
        r -= 1
        if r < 0:
            return 0
        if r >= len(starts):
            return len(src)
        return min(starts[r] + c, len(src))

    prev_sig = None  # 직전 '의미있는' 토큰 타입
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            t, s, a, b, _ = tok
            p0, p1 = pos(*a), pos(*b)
            if t == tokenize.COMMENT:
                cat = COM
            elif t == tokenize.STRING:
                # docstring 근사: 직전 의미토큰이 NEWLINE/INDENT/DEDENT/없음 = 문(statement)
                # 자리에 홀로 놓인 문자열 ⇒ docstring. 그 밖의 문자열 = 코드가 만드는 값.
                cat = DOC if prev_sig in (None, tokenize.NEWLINE, tokenize.INDENT,
                                          tokenize.DEDENT, tokenize.ENCODING) else STR
            else:
                cat = None
            if t not in (tokenize.NL, tokenize.COMMENT):
                prev_sig = t
            if cat:
                for i in range(p0, p1):
                    cats[i] = cat
    except Exception as e:                       # 토큰화 실패 = 줄 기반 폴백
        sys.stderr.write("tokenize fail: %s\n" % e)
        for i, line in enumerate(src.splitlines(True)):
            h = line.find("#")
            if h >= 0:
                for j in range(starts[i] + h, starts[i] + len(line)):
                    cats[j] = COM
    return cats


def line_of(src, off):
    return src.count("\n", 0, off) + 1


def selftest_lines(src):
    """오프라인 자기검정 구역의 줄 범위. = `if __name__ == "__main__":` 블록
    ∪ 이름에 selftest/_test/test_ 가 든 함수. 그 안의 도메인 리터럴은 **픽스처**다
    (런타임에 실행되지 않고 모델에게 가지 않는다)."""
    import ast
    rng = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return rng
    for n in ast.walk(tree):
        hit = False
        if isinstance(n, ast.If):
            d = ast.dump(n.test)
            hit = "__name__" in d and "__main__" in d
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nm = n.name.lower()
            hit = ("selftest" in nm or nm.startswith("test_") or nm.endswith("_test")
                   or "_selftest" in nm)
        if hit:
            rng.update(range(n.lineno, (getattr(n, "end_lineno", n.lineno) or n.lineno) + 1))
    return rng


def scan_file(path, axes):
    src = io.open(path, encoding="utf-8", errors="replace").read()
    cats = categorize(src)
    st = selftest_lines(src)
    low = src.lower()
    hits = []
    for axis, lits in axes.items():
        for lit in lits:
            l = lit.lower()
            if not l:
                continue
            i = low.find(l)
            while i >= 0:
                # 식별자형 리터럴은 단어 경계를 요구한다(부분일치로 부풀지 않게).
                ok = True
                if re.match(r"^[a-z_][a-z0-9_]*$", l):
                    before = low[i - 1] if i else " "
                    after = low[i + len(l)] if i + len(l) < len(low) else " "
                    ok = not (re.match(r"[a-z0-9_]", before) or re.match(r"[a-z0-9_]", after))
                if ok:
                    ln = line_of(src, i)
                    cat = cats[i]
                    if cat in (CODE, STR) and ln in st:
                        cat = "selftest"      # 픽스처 = 런타임 아님·모델에 안 감
                    hits.append({
                        "axis": axis, "lit": lit, "cat": cat, "line": ln,
                        "text": src.splitlines()[ln - 1].strip()[:200]
                        if ln - 1 < len(src.splitlines()) else "",
                    })
                i = low.find(l, i + 1)
    return src, hits


def main():
    guarded = set(RULES["guarded_engine"])

    # ── 축 1~4 = 훅이 실제로 보는 것 (scaffold_rules.json 축자) ──────────────────
    axes = {
        "A_ident": RULES["engine_denylist"],            # 식별자·도메인 분기
        "B_prose": RULES["engine_prose_denylist"],      # 도메인 어휘(훅은 "..."≥8 안만)
        "C_prescriptive": RULES["engine_prescriptive_flags"],
        "D_soft": RULES["engine_soft_flags"],
    }

    # ── 축 E = 훅이 **보지 않는** 것. env_surface.json 이 선언한 실제 도구 이름 전수.
    #    훅 denylist 는 retail 11 + airline 7 + 공용 1 뿐이고 banking 69 는 0개다.
    surf = json.load(io.open(os.path.join(ENGINE_DIR, "a2", "env_surface.json"),
                             encoding="utf-8"))
    known = set(x.lower() for x in RULES["engine_denylist"])
    e_bank, e_other = [], []
    for dom, v in surf.items():
        for t in v["tools"]:
            if t.lower() in known:
                continue
            (e_bank if dom == "banking_knowledge" else e_other).append(t)
    axes["E_bank_tools_unseen"] = sorted(set(e_bank))
    axes["E_other_tools_unseen"] = sorted(set(e_other))
    # 축 F = 도메인 이름 자체. 훅은 `'retail'`/`'airline'`/`== "retail"` 만 본다.
    axes["F_domain_name"] = ["banking_knowledge", "banking", "retail", "airline",
                             "telecom", "doordash"]

    files = sorted([f for f in os.listdir(ENGINE_DIR)
                    if (f.startswith("t2_") and f.endswith(".py"))
                    or f == "gate_interpreter.py"])

    report = {"axes_sizes": {k: len(v) for k, v in axes.items()},
              "guarded_engine_count": len(guarded), "files": {}}
    for f in files:
        p = os.path.join(ENGINE_DIR, f)
        src, hits = scan_file(p, axes)
        if not hits:
            continue
        report["files"][f] = {
            "guarded": f in guarded,
            "lines": src.count("\n") + 1,
            "hits": hits,
        }
    json.dump(report, io.open(os.path.join(ENGINE_DIR, "_lit_scan_out.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)
    print("files scanned:", len(files), " with hits:", len(report["files"]))


main()
