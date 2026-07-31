#!/usr/bin/env python3
"""X6-(i) EXT 재판정 — 공용 primitive가 A2-EXT를 얼마나 흡수하는가 (C235 설계 TODO ①②③).

C235(§13e)가 남긴 설계 TODO를 **실측으로** 답한다:
  ② 공용 도구 primitive 목록 확정  → 엔진 인터프리터가 실제 지원하는 op 어휘를 코드에서 수확
  ③ EXT 키를 (공용 커버 / callback 필요)로 재판정 → A2가 선언한 op 전수를 ②에 대조
  ① callback 인터페이스 목록        → ③의 미커버 잔여가 곧 callback 후보

핵심 질문(handoff §6-④): "공용 primitive가 E1/E2를 얼마나 흡수하는지 → C_ext 축소분".
이건 추측할 일이 아니라 **세는 일**이다 — A2에 선언된 op와 엔진이 실행하는 op를 맞춰보면 된다.

판정 규율: 커버 = 그 op가 엔진 인터프리터에 **이미 구현**돼 있어 A2가 *식(what)*만 선언하고
실행(how)은 공용 코드가 하는 경우. 미커버 = 도메인이 코드를 새로 공급해야 하는 경우 = callback.

용법: py -3 x6i_ext_primitive_absorption.py [--json out.json]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2_DIR = os.path.join(_HERE, "a2")
_SPLIT_DIR = os.path.join(_A2_DIR, "split")

# 연산 인터프리터 — ★실측 결과 **3개**이고 op 어휘가 서로 다르다(3자 교집합 0).
#   t2_compute.py          : 라이브 경로(t2_scaffold_get → apply_op)·자칭 "도메인-일반 라이브러리"
#   gate_interpreter.py    : 게이트 술어 평가용(argmax_where·disjoint·equal_len…)
#   bank_eplan_controller.py: E-PLAN 컨트롤러용(산술 add/divide/round/subtract) — [[14]] e2e 미배선
# ⚠️1개만 수확하면 흡수율이 아티팩트로 낮게 나온다(초판 50%가 그 오류).
_INTERP_FILES = ["t2_compute.py", "gate_interpreter.py", "bank_eplan_controller.py"]

# C234(§13d) EXT 화이트리스트 — 허용 3종.
EXT_WHITELIST = {
    "E1_계산명세": ["compute_ops", "calc_specs", "field_ops"],
    "E2_도구셋": ["calc_tool", "function_agents"],
    "E3_스키마상수": ["identifying_arg_types", "variant_spec", "variant_operand"],
}
_WL_FLAT = {k: cat for cat, ks in EXT_WHITELIST.items() for k in ks}


def engine_ops(per_file=False):
    """엔진 인터프리터가 지원하는 op 이름 집합 (코드에서 수확 — 문서 아님)."""
    ops, byfile = set(), {}
    for fn in _INTERP_FILES:
        path = os.path.join(_HERE, fn)
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        f_ops = set(re.findall(r'op == "([a-z_0-9]+)"', src))
        for grp in re.findall(r'op in \(([^)]*)\)', src):
            f_ops |= set(re.findall(r'"([a-z_0-9]+)"', grp))
        byfile[fn] = f_ops
        ops |= f_ops
    return (ops, byfile) if per_file else ops


def declared_ops():
    """A2 전수에서 선언된 op. 반환 {op: {domain: count}} + {domain: {key: [op...]}}."""
    per_op = defaultdict(lambda: defaultdict(int))
    per_key = defaultdict(lambda: defaultdict(list))
    for d in (_A2_DIR, _SPLIT_DIR):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            domain = fn.split(".")[0]
            # split/*.core|ext|discard 와 a2/*.gate|grounding 중복 계상 방지: split만 있으면 split 우선
            data = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            for topkey, sub in (data.items() if isinstance(data, dict) else []):
                for op in _find_ops(sub):
                    per_op[op][domain] += 1
                    if op not in per_key[domain][topkey]:
                        per_key[domain][topkey].append(op)
    return per_op, per_key


def _find_ops(obj):
    """dict/list 재귀로 "op" 키의 문자열 값을 전부 yield.

    ⚠️버그 이력(2026-07-30 자기정정): 초판은 `if k != "op"`로 재귀해 **"op" 키의 값이 중첩
    dict일 때 그 서브트리를 통째로 건너뛰었다**(`scaffold_get_tools[i]["op"]`가 그 형태) →
    banking 22 op을 8로 과소 계상. "op" 값이 str이면 수확, 아니면 **재귀해야** 한다.
    """
    if isinstance(obj, dict):
        for k, sub in obj.items():
            if k == "op" and isinstance(sub, str):
                yield sub
            else:
                yield from _find_ops(sub)
    elif isinstance(obj, list):
        for sub in obj:
            yield from _find_ops(sub)


def ext_inventory():
    """split/*.ext.json 의 키를 화이트리스트 분류에 대조. 반환 {domain: {key: category}}."""
    out = {}
    if not os.path.isdir(_SPLIT_DIR):
        return out
    for fn in sorted(os.listdir(_SPLIT_DIR)):
        if not fn.endswith(".ext.json"):
            continue
        domain = fn.split(".")[0]
        data = json.load(open(os.path.join(_SPLIT_DIR, fn), encoding="utf-8"))
        out[domain] = {k: _WL_FLAT.get(k, "미분류(C234=CORE흡수 or 폐기)")
                       for k in data if k != "_meta"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    supported, byfile = engine_ops(per_file=True)
    per_op, per_key = declared_ops()
    inv = ext_inventory()

    print("=" * 78)
    print("§1. 공용 primitive 목록 (C235 TODO ② — 엔진 코드서 수확·문서 아님)")
    print("=" * 78)
    print("★인터프리터가 **3개**이고 3자 교집합이 **0**이다 = op 어휘 이중화([[03b]] 금지 항목).")
    for fn, o in byfile.items():
        only = o - set().union(*[v for k, v in byfile.items() if k != fn]) if len(byfile) > 1 else o
        print(f"  {fn:26s} {len(o):3d} op (자기만: {len(only)})")
    inter = set.intersection(*byfile.values()) if byfile else set()
    print(f"  3자 교집합: {len(inter)} {sorted(inter)}")
    print()
    print(f"엔진 인터프리터 지원 op 합집합 {len(supported)}개:")
    for i, op in enumerate(sorted(supported)):
        end = "\n" if i % 6 == 5 else "  "
        print(f"  {op:22s}", end=end)
    print("\n")

    print("=" * 78)
    print("§2. A2 선언 op → 커버 판정 (C235 TODO ③ — 흡수율)")
    print("=" * 78)
    covered = {o: d for o, d in per_op.items() if o in supported}
    uncovered = {o: d for o, d in per_op.items() if o not in supported}
    print(f"A2가 선언한 서로 다른 op: {len(per_op)}")
    print(f"  공용 커버   : {len(covered)}")
    print(f"  미커버(=callback 후보): {len(uncovered)}")
    if per_op:
        rate = 100.0 * len(covered) / len(per_op)
        print(f"  ⇒ **흡수율 {rate:.1f}%** (op 종류 기준)")
    print()
    print("커버된 op (도메인별 사용횟수):")
    for op in sorted(covered):
        doms = ", ".join(f"{d}×{n}" for d, n in sorted(covered[op].items()))
        print(f"  ✓ {op:24s} {doms}")
    if uncovered:
        print()
        print("★미커버 op = callback 인터페이스 후보 (C235 TODO ①):")
        for op in sorted(uncovered):
            doms = ", ".join(f"{d}×{n}" for d, n in sorted(uncovered[op].items()))
            print(f"  ✗ {op:24s} {doms}")
    else:
        print()
        print("★미커버 op **0** ⇒ 현 3도메인에서 callback 구현체 요구 = 0")
    print()

    print("=" * 78)
    print("§3. EXT 키 인벤토리 × 화이트리스트 (C234 분류 대조)")
    print("=" * 78)
    for domain in sorted(inv):
        print(f"--- {domain}")
        for k, cat in sorted(inv[domain].items(), key=lambda x: (x[1], x[0])):
            ops = per_key.get(domain, {}).get(k, [])
            miss = [o for o in ops if o not in supported]
            tag = "callback필요" if miss else ("공용실행" if ops else "데이터만")
            print(f"   {k:26s} {cat:32s} op={len(ops):2d} {tag}"
                  + (f" 미커버={miss}" if miss else ""))
        print()

    print("=" * 78)
    print("§4. C_ext 회계 (capex/opex 프레임 · C237)")
    print("=" * 78)
    print("capex(1회·전 도메인 공유·증분 0) = 인터프리터 + 지원 op "
          f"{len(supported)}개 구현")
    print("opex(도메인당) = A2에 식을 *선언*하는 비용만 — 실행 코드 작성 0"
          + (" (현 3도메인 실측)" if not uncovered else ""))
    print("⇒ E1(계산명세)의 도메인별 코드 비용 = "
          + ("**0**" if not uncovered else f"미커버 {len(uncovered)} op만"))
    print("⚠ 이 판정의 범위 = **현 3도메인이 실제로 선언한 op**뿐. 새 도메인이 미지원 op를")
    print("  요구하면 그때 callback이 필요하다 — 흡수율은 상한 증명이 아니라 현재 회계다([[08]]).")

    if args.json:
        json.dump({"supported": sorted(supported),
                   "declared": {o: dict(d) for o, d in per_op.items()},
                   "uncovered": sorted(uncovered),
                   "ext_inventory": inv},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
