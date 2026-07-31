# -*- coding: utf-8 -*-
"""X6-(e) EXT 화이트리스트 제한 — 도메인-특화 허용 범위를 '계산·도구셋·스키마-파생'으로 닫는다.

지시(2026-07-30 사용자): "도구 사용의 기본 스킬로 공유될 수 있는 기능들 + 그에 필요한 세팅을
A2에 두는 건 문제 없다. 문제는 계산이나 도메인 특화로 어쩔 수 없이 추가되는 새 도구인데,
그것은 **엄격하게 계산도구 등에 한정**하면 될 것 같다."

정식화: EXT를 **닫힌 3종**으로만 허용하고, 그 밖의 EXT 키는 CORE 흡수 또는 폐기로 처분한다.
  E1 계산 명세 — 도메인 정책이 정의한 계산식·연산 (엔진은 고정 실행기·식은 데이터)
  E2 도구셋 제공 — 도메인이 노출하는 결정론 도구·실행 경로 ([[16]] §8: 데이터 접근 제공=정당)
  E3 스키마-파생 상수 — 도구 스키마에서 기계적으로 얻는 타입·식별자 종류
그 밖(E-OUT)은 scaffold 편의물이므로 CORE(도메인-불변 기능군)로 흡수하거나 폐기한다.
⇒ 이렇게 닫으면 "도메인이 늘 때 새 *종류*가 늘어난다"는 유한성 위협이 E1~E3 안으로 유계된다.
"""
import json, io, os, glob, sys

E1_CALC = {"compute_ops", "calc_specs", "field_ops"}
E2_TOOLSET = {"calc_tool", "function_agents"}
E3_SCHEMA = {"identifying_arg_types", "variant_spec", "variant_operand"}

# E-OUT의 처분(근거를 코드에 명시)
ABSORB_TO_CORE = {
    "placeholders": "날조-방지 상수 = 접지(R9) 계열의 도메인-불변 기능군",
    "tool_error_specs": "오류 계약 포맷 = deny_g2 처방의 일부(도메인-불변)",
    "param_cap_check": "형식-층 캡 = 봉투/형식 검증기 소관",
    "ref_iso": "참조 격리 재선택 = controller(fexec) 계열",
    "reference_filter": "제약→predicate 필터 = fexec(CORE)",
    "analysis_producers": "생산자 선언의 한 종류 = producers(CORE)로 병합",
    "assertion_operands": "주장 operand = claim_prov(CORE)의 인자 선언",
}
DISCARD_EXT = {
    "regen_resolver_specs": "regen 문구·해소 트릭 = P2-b/anti-drift(개입 문구)",
    "view_field_annotations": "뷰 주석 = 표면화 문구(엔진 고정 포맷으로 대체)",
}


def cls_ext(k):
    if k in E1_CALC:
        return "E1-calc"
    if k in E2_TOOLSET:
        return "E2-toolset"
    if k in E3_SCHEMA:
        return "E3-schema"
    if k in ABSORB_TO_CORE:
        return "ABSORB->CORE"
    if k in DISCARD_EXT:
        return "DISCARD"
    return "UNCLASSIFIED"


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "split")
    rows = {}
    for p in sorted(glob.glob(os.path.join(base, "*.ext.json"))):
        dom = os.path.basename(p).split(".")[0]
        d = json.load(io.open(p, encoding="utf-8"))
        rows[dom] = sorted(k for k in d if not k.startswith("_"))

    print("=== EXT 화이트리스트 재분류 (지시: 계산·도구셋에 한정) ===")
    agg = {}
    for dom, keys in rows.items():
        print("\n[%s] EXT %d키" % (dom, len(keys)))
        for k in keys:
            c = cls_ext(k)
            agg[c] = agg.get(c, 0) + 1
            note = ABSORB_TO_CORE.get(k) or DISCARD_EXT.get(k) or ""
            print("  %-24s -> %-14s %s" % (k, c, note))
    print("\n=== 집계 ===")
    for c in ("E1-calc", "E2-toolset", "E3-schema", "ABSORB->CORE", "DISCARD", "UNCLASSIFIED"):
        if c in agg:
            print("  %-14s %d" % (c, agg[c]))
    keep = sum(agg.get(c, 0) for c in ("E1-calc", "E2-toolset", "E3-schema"))
    print("\n  허용 EXT(E1~E3) = %d키 / 처분 대상 = %d키"
          % (keep, sum(agg.values()) - keep))
    if agg.get("UNCLASSIFIED"):
        print("  ⚠UNCLASSIFIED 존재 — 화이트리스트 정의 보완 필요")


if __name__ == "__main__":
    main()
