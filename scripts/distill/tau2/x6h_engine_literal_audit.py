#!/usr/bin/env python3
"""X6-(h) 엔진 도메인-리터럴 전수 감사 — [[05]] "엔진 리터럴 0" 주장의 기계적 검증.

배경: `X_FREE_TRACK_RESULTS_2026_07_30.md` §13d가 grep으로 "실제 코드 위반 2건"이라 판정했다.
grep은 (a)주석·docstring (b)selftest 픽스처 (c)실행 코드를 구분하지 못하고, 포맷 플레이스홀더
(`'{tool}'` = A2가 채우는 *정답* 패턴)와 하드코딩된 이름도 구분하지 못한다 → 주장이 과·소 양쪽으로
틀릴 수 있다. 이 도구는 그 셋을 AST로 분리해 **실행 코드의 문자열 리터럴만** 판정한다.

판정 어휘를 감사자가 고르면 순환(내가 고른 단어만 위반이 됨)이라, **tau2 도메인 정의에서 기계
수확한 권위 namespace**를 쓴다(`tau2_domain_toolnames.json`·`src/tau2/domains/<d>/tools.py`의
public 메서드 = 그 도메인의 도구 이름). 공통/특화 판별도 데이터가 한다:

    도구명이 **모든** 도메인에 존재  → framework-common (엔진 참조 허용·예: transfer_to_human_agents)
    도구명이 **일부** 도메인에만 존재 → domain-specific  (엔진 참조 = [[05]] 위반 후보)

추가로 도메인 명사(banking/retail/airline/telecom/doordash)를 직접 검사한다 — 도구명이 아니라
산문에 박힌 경우(예: 서브콜 프롬프트 "a precise banking assistant")를 잡기 위함.

**플레이스홀더 제외**: `"{tool} was denied"`처럼 `{...}`로 A2 값을 주입받는 자리는 파라미터화의
정답이므로 매칭 전에 제거한다. 남은 *맨* 이름만이 하드코딩이다.

출력: 위반 후보를 파일·행·컨텍스트와 함께 **건별로** 나열([[08]] per-case). 집계만 보고 결론
내지 말 것 — 각 건을 정독해 (위반 / 프레임워크 / 무해)를 사람이 확정한다.

용법: py -3 x6h_engine_literal_audit.py [--json out.json]
      (Windows 콘솔은 PYTHONIOENCODING=utf-8 권장 — cp949 함정 [[30]])
"""
import argparse
import ast
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLNAMES = os.path.join(_HERE, "tau2_domain_toolnames.json")

# 엔진 = 우리 스캐폴드 코드(도메인이 늘어도 무수정이어야 하는 층). tau2 domains/<d>/tools.py는
# 도메인 공급물이라 대상 아님(§13e 표의 "도메인 도구"층).
#
# ★★2026-07-30 리뷰 B1 교정: 초판은 이 목록을 **손으로 적었고** 그래서 `t2_resolve.py`를 놓쳤다 —
#   그 모듈은 `t2_gate_patch.py`가 런타임에 4곳(:547·:3885·:3911·:4035)에서 import하는 **라이브
#   엔진**이고 최소 5건의 도메인 리터럴을 갖고 있었다(그중 `DISCOVERY_REQUIRED_FB`는 실제 suffixed
#   도구명 `open_bank_account_4821`과 banking 액션 어휘를 **모델 입력 산문**에 먹인다).
#   ⇒ 손으로 적은 목록은 **proxy**다. [[05]] §메타 실패모드가 경고한 그 형태이며, 실패 위치가
#   방법(grep→AST)에서 **스코프**로 옮겨간 것에 불과했다. 이제 **라이브 드라이버에서 시작한
#   import 폐포**로 산출한다(목록을 손으로 늘리는 대신 폐포가 늘어나면 자동 편입).
ENTRY_POINTS = ["t2_run_gated.py"]      # 라이브 드라이버 (실제 실험이 실행하는 것)
# ⚠버그 이력: 초판 `[\w,\s]+` 의 `\s` 가 **개행을 먹어** 여러 import 를 한 덩어리로 삼켰다
#   (폐포가 3개로 과소 산출). import 목록은 **한 줄**로 제한한다.
_IMPORT_RE = re.compile(r"^[ \t]*(?:import[ \t]+([^\n]+)|from[ \t]+(\.?[\w.]+)[ \t]+import)", re.M)
# 폐포에서 제외 — 도메인 공급물·표준/서드파티·분석 스크립트가 아닌 것만 남긴다는 뜻이 아니라,
# **우리 디렉터리의 로컬 모듈만** 엔진으로 센다(외부 패키지는 우리 코드가 아니다).


def discover_engine_files(verbose=False):
    """라이브 드라이버에서 시작한 **로컬 import 폐포** = 엔진 파일 집합.

    손으로 적은 목록이 리뷰 B1에서 스코프 누락(t2_resolve.py)을 낳았으므로 기계 산출로 대체.
    같은 디렉터리의 `.py`만 로컬 모듈로 인정한다(외부 패키지는 우리 엔진이 아님).
    """
    seen, queue, order = set(), list(ENTRY_POINTS), []
    while queue:
        fn = queue.pop(0)
        if fn in seen:
            continue
        path = os.path.join(_HERE, fn)
        if not os.path.exists(path):
            continue
        seen.add(fn)
        order.append(fn)
        src = open(path, encoding="utf-8", errors="replace").read()
        for m in _IMPORT_RE.finditer(src):
            mods = []
            if m.group(1):
                mods = [x.strip().split(" as ")[0].strip() for x in m.group(1).split(",")]
            elif m.group(2):
                mods = [m.group(2).lstrip(".")]
            for mod in mods:
                cand = mod.split(".")[0] + ".py"
                if cand not in seen and os.path.exists(os.path.join(_HERE, cand)):
                    queue.append(cand)
    if verbose:
        print(f"[scope] import 폐포 {len(order)}개 (진입점 {ENTRY_POINTS}): {sorted(order)}")
    return sorted(order)

# tau2 5도메인 이름. 산문 리터럴에 박히면 도메인-특화.
DOMAIN_NOUNS = ("banking", "retail", "airline", "telecom", "doordash")

_FIELD_NS = None
_PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")     # f-string/format 주입점 = 파라미터화(정답)


def load_field_namespace():
    """★2차 감사(리뷰 §8-D·C241): **DB 필드·인자 이름** namespace.

    도구명 namespace로는 `parse_records(key_field="transaction_id")` 같은 **키워드-인자 기본값**을
    못 잡는다(필드명은 도구명이 아니므로). 필드 이름은 A2가 선언한 것에서 수확한다 —
    A2는 정의상 도메인 데이터이므로 순환이 아니고, 도구명 때와 같은 판별을 쓴다:
      필드명이 **일부 도메인 A2에만** 등장 → domain-specific (엔진 참조 = 위반)
    수확 위치 = 값이 필드 이름인 A2 키(`field_ops`의 카테고리 목록·`identifying_arg_types`·
    `operands[].param`·`ref_iso[].param`/`id_key`·`reference_filter[].{param,key_field}`).
    """
    a2dir = os.path.join(_HERE, "a2")
    f2d = {}
    if not os.path.isdir(a2dir):
        return {}, set()
    doms = set()
    for fn in sorted(os.listdir(a2dir)):
        if not fn.endswith(".json"):
            continue
        dom = fn.split(".")[0]
        doms.add(dom)
        try:
            data = json.load(open(os.path.join(a2dir, fn), encoding="utf-8"))
        except Exception:
            continue
        fields = set()
        fo = data.get("field_ops") or {}
        if isinstance(fo, dict):
            for k, v in fo.items():
                if k.startswith("_") or not isinstance(v, list):
                    continue
                fields |= {x for x in v if isinstance(x, str)}
        for x in (data.get("identifying_arg_types") or []):
            if isinstance(x, str):
                fields.add(x)
        for key in ("operands", "ref_iso", "reference_filter", "ref_verify", "assertion_operands"):
            v = data.get(key)
            it = v if isinstance(v, list) else ([v] if isinstance(v, dict) else [])
            for e in it:
                if not isinstance(e, dict):
                    continue
                for kk in ("param", "key_field", "id_key", "record_field", "operand"):
                    if isinstance(e.get(kk), str):
                        fields.add(e[kk])
        for f in fields:
            if re.match(r"^[a-z][a-z0-9_]{2,}$", f):
                f2d.setdefault(f, set()).add(dom)
    return f2d, doms


def load_namespaces():
    """도메인 -> public 도구명 집합. 반환 (tool2domains, all_domains)."""
    with open(_TOOLNAMES, encoding="utf-8") as f:
        raw = json.load(f)
    all_domains = set(raw)
    tool2domains = {}
    for dom, names in raw.items():
        for n in names:
            if n.startswith("_"):
                continue                      # private 헬퍼 = 도구 아님
            tool2domains.setdefault(n, set()).add(dom)
    return tool2domains, all_domains


def regex_literals(tree, doc_lines):
    """`re.*(pattern, ...)`의 첫 인자로 쓰인 문자열 리터럴 = 정규식. {lineno: pattern}.

    ★왜 별도 검사인가: 전체-단어 매칭은 `re.compile(r"discoverable|^give_|^unlock_")`처럼
    **도구명 조각으로 만든 패턴**을 놓친다(조각 자체는 도구명이 아니므로). 어휘 휴리스틱으로
    조각을 판정하려 했더니 평범한 영어(customer·item·available)가 우연히 한 도메인 도구명에만
    등장해 오탐 152건 — 신호 없음이라 폐기했다. 대신 **함수적으로** 판정한다: 패턴을 실제
    도구명 122개에 돌려 *어느 도메인의 이름을 매치하는지* 본다. 일부 도메인만 매치 = 그 패턴이
    도메인 명명 관행을 판별함 = 도메인 리터럴.
    """
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        mod = getattr(getattr(f, "value", None), "id", None)
        if mod != "re" or not node.args:
            continue
        a0 = node.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.lineno not in doc_lines:
            out[a0.lineno] = a0.value
    return out


def _split_alts(pattern):
    """최상위 `|`로 정규식을 가지 분해 (괄호·클래스 깊이 고려).

    ★왜: 선언형 패턴은 **범용 가지 하나가 전체를 도메인-일반으로 위장**시킨다. 실측 —
    `(^log_|...|discoverable|transfer_to_human|^give_|^unlock_|get_current_time)`은
    `transfer_to_human`(5/5 도메인) 때문에 통째로는 "일반"으로 통과하지만, 가지
    `discoverable`·`^give_`·`^unlock_`은 banking 전용이다. 가지 단위가 옳은 판정 입도.
    """
    # 겉을 감싼 그룹 1겹 벗김: (a|b|c) -> a|b|c
    p = pattern.strip()
    if p.startswith("(") and p.endswith(")"):
        depth = 0
        for i, ch in enumerate(p):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(p) - 1:
                    break
        else:
            p = p[1:-1]
    alts, buf, depth, cls, esc = [], "", 0, False, False
    for ch in p:
        if esc:
            buf += ch
            esc = False
            continue
        if ch == "\\":
            buf += ch
            esc = True
            continue
        if ch == "[":
            cls = True
        elif ch == "]":
            cls = False
        elif not cls and ch == "(":
            depth += 1
        elif not cls and ch == ")":
            depth -= 1
        if ch == "|" and depth == 0 and not cls:
            alts.append(buf)
            buf = ""
        else:
            buf += ch
    alts.append(buf)
    return [a for a in alts if a.strip()]


def regex_domain_skew(pattern, tool2domains, all_domains):
    """패턴(및 그 가지들)이 도메인을 판별하는지. 반환 [(가지, 도메인집합, 예시)]."""
    out = []
    branches = _split_alts(pattern)
    cands = branches if len(branches) > 1 else [pattern]
    for br in cands:
        try:
            rx = re.compile(br, re.I)
        except re.error:
            continue
        hit_doms, examples = set(), []
        for name, doms in tool2domains.items():
            if rx.search(name):
                hit_doms |= doms
                if len(examples) < 5:
                    examples.append(name)
        if hit_doms and not (hit_doms >= all_domains):
            out.append((br, hit_doms, examples))
    return out


def docstring_lines(tree):
    """모듈·함수·클래스 docstring이 점유하는 행 집합 (리터럴이지만 주석 성격)."""
    lines = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            lo = first.value.lineno
            lines.update(range(lo, (getattr(first.value, "end_lineno", lo) or lo) + 1))
    return lines


def selftest_range(tree):
    """`if __name__ == "__main__":` 블록 행 범위 (selftest 픽스처 = 엔진 동작 아님)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "__name__"):
            lo = node.lineno
            hi = max((getattr(n, "end_lineno", lo) or lo) for n in ast.walk(node))
            return (lo, hi)
    return None


def audit_file(path, tool2domains, all_domains):
    """실행 코드의 문자열 리터럴만 검사해 hit 리스트 반환."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    src_lines = src.splitlines()
    tree = ast.parse(src)
    doc_lines = docstring_lines(tree)
    st = selftest_range(tree)
    rx_lits = regex_literals(tree, doc_lines)

    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        ln = node.lineno
        if ln in doc_lines:
            continue                                  # docstring = 주석 성격
        zone = "selftest" if (st and st[0] <= ln <= st[1]) else "live"
        bare = _PLACEHOLDER_RE.sub(" ", node.value)   # 주입점 제거 후 남은 *맨* 텍스트
        low = bare.lower()
        words = set(re.findall(r"[a-z][a-z0-9_]*", low))

        found = []
        for noun in DOMAIN_NOUNS:
            if noun in words:                          # 전체-단어 일치
                found.append((noun, "domain-noun", [noun]))
        for w in words:
            doms = tool2domains.get(w)
            if not doms:
                continue
            kind = "framework-common" if doms >= all_domains else "domain-specific"
            found.append((w, kind, sorted(doms)))
        if _FIELD_NS:
            f2d, fdoms = _FIELD_NS
            for w in words:
                fd = f2d.get(w)
                if fd and not (fd >= fdoms):
                    found.append((w, "domain-field", sorted(fd)))
        if not found:
            continue
        hits.append({
            "line": ln,
            "zone": zone,
            "literal": node.value if len(node.value) <= 200 else node.value[:197] + "...",
            "code": (src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else ""),
            "tokens": [{"tok": t, "kind": k, "domains": d} for t, k, d in found],
        })

    for ln, pat in sorted(rx_lits.items()):
        if st and st[0] <= ln <= st[1]:
            continue
        skew = regex_domain_skew(pat, tool2domains, all_domains)
        if not skew:
            continue
        hits.append({
            "line": ln, "zone": "live",
            "literal": pat if len(pat) <= 200 else pat[:197] + "...",
            "code": (src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else ""),
            "tokens": [{"tok": br, "kind": "domain-discriminating-regex",
                        "domains": sorted(d), "matches": ex} for br, d, ex in skew],
        })
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="결과 JSON 저장 경로")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")      # cp949 함정 [[30]]
    except Exception:
        pass

    global _FIELD_NS
    _FIELD_NS = load_field_namespace()
    print(f"[ns2] A2 유래 필드명 {len(_FIELD_NS[0])}개 (2차 감사·리뷰 §8-D)")
    engine_files = discover_engine_files(verbose=True)
    print()
    tool2domains, all_domains = load_namespaces()
    common = sorted(t for t, d in tool2domains.items() if d >= all_domains)
    print(f"[ns] 도메인 {len(all_domains)}: {sorted(all_domains)}")
    print(f"[ns] public 도구명 {len(tool2domains)}개 · 전-도메인 공통 {len(common)}: {common}")
    print()

    report, tot = {}, {"noun": 0, "specific": 0, "field": 0, "regex": 0, "common": 0, "selftest": 0}
    for fn in engine_files:
        path = os.path.join(_HERE, fn)
        if not os.path.exists(path):
            print(f"[skip] {fn} 없음")
            continue
        hits = audit_file(path, tool2domains, all_domains)
        report[fn] = hits
        viol = [h for h in hits if h["zone"] == "live"
                and any(t["kind"] in ("domain-specific", "domain-noun", "domain-field", "domain-discriminating-regex")
                        for t in h["tokens"])]
        comm = [h for h in hits if h["zone"] == "live"
                and all(t["kind"] == "framework-common" for t in h["tokens"])]
        stf = [h for h in hits if h["zone"] == "selftest"]
        tot["common"] += len(comm)
        tot["selftest"] += len(stf)
        print(f"=== {fn}: live-위반후보 {len(viol)} · live-프레임워크 {len(comm)} · selftest {len(stf)} ===")
        for h in viol:
            bad = [t for t in h["tokens"]
                   if t["kind"] in ("domain-specific", "domain-noun", "domain-field", "domain-discriminating-regex")]
            ks = {t["kind"] for t in bad}
            if "domain-noun" in ks:
                tot["noun"] += 1
            elif "domain-specific" in ks:
                tot["specific"] += 1
            elif "domain-field" in ks:
                tot["field"] += 1
            else:
                tot["regex"] += 1
            for t in bad:
                print(f"  L{h['line']} {t['tok']} [{t['kind']}·{'/'.join(t['domains'])}]")
            print(f"     code: {h['code'][:130]}")
            print(f"     lit : {h['literal'][:130]}")
        print()

    print("=== 총계 ===")
    print(f"live 도메인-명사 리터럴  : {tot['noun']}")
    print(f"live 도메인-특화 도구명  : {tot['specific']}")
    print(f"live 도메인-필드명(2차)   : {tot['field']}")
    print(f"live 도메인-판별 정규식   : {tot['regex']}")
    print(f"live 프레임워크-공통(허용): {tot['common']}")
    print(f"selftest 픽스처(비위반)  : {tot['selftest']}")
    print("⚠ 집계로 결론 금지([[08]]) — 위 건별 목록을 정독해 확정할 것.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"domains": sorted(all_domains), "report": report}, f,
                      ensure_ascii=False, indent=1)
        print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
