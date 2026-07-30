# -*- coding: utf-8 -*-
"""X6-(b) A2 2층 분할 — A2-core(도메인-불변 기능군) vs A2-ext(도메인-특화 opex) + 폐기분.

지시(2026-07-30 사용자): "A2를 도메인 특화로 2개로 쪼개야 하면 쪼개라."
목적: 유한성 주장의 분모를 A2-core로 한정하고, 도메인-특화분을 별도 회계로 노출한다.

분류 기준(판단 근거를 코드에 명시 — 재현·반박 가능하게):
  CORE : 선언-우선 코어(§1d rev3)가 해석하는 도메인-불변 기능군.
         = 정책 게이트 / GET-생산자 / 레지스트리·membership / 완료-보고 대조 /
           근거-접지 / 선행조건 / 수요-원장·coverage
  DISCARD: P2-b 폐기 목록(§1b) 또는 [[16]] anti-drift 금지(개입레버·default·present)
  EXT  : 위 어디도 아닌 도메인-특화 스펙(도메인 계산식·변형·문구·뷰 등) = 정직한 opex
"""
import json, io, glob, os, collections

CORE = {
    # 정책 게이트·선행조건
    "gates", "require_tool_before",
    # GET·생산자(operand 해소)
    "producers", "arg_producers", "scaffold_get_tools", "operands",
    # 레지스트리·membership(instruct_user_run·도구명)
    "discoverable_name_check", "nonlisted_tool_feedback", "tool_arg_allowlist",
    "unavailable_tools", "dispatcher_role_check",
    # 완료-보고 대조(done_report ⊆ executed_events)
    "claim_prov", "completion_guard",
    # 근거-접지(evidence_quote ∈ ledger)
    "write_arg_grounding", "write_evidence_specs", "ref_verify",
    # 수요-원장·coverage(DONE-게이트)
    "eplan", "follow_up_chains",
    # 행동 도구 집합(turn_type=ACT 판정 대상)
    "action_tools",
}

DISCARD = {
    # 프록시 술어 기반 레버(X4 flip으로 열림 확정)
    "have_value_reask", "value_acquisition",
    # 개입레버·[[16]] anti-drift 금지(present·default·disamb-override)
    "present_specs", "default_specs", "disamb_sub_args",
    # 사례-표적 의심(Q3) — 재분류 전까지 폐기 측
    "prescription_redirect", "recommendation_verify",
}


def cls(k):
    if k in CORE:
        return "CORE"
    if k in DISCARD:
        return "DISCARD"
    return "EXT"


def size_of(v):
    return len(v) if isinstance(v, (dict, list)) else 1


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
    doms = {}
    for p in sorted(glob.glob(os.path.join(base, "*.gate.json"))):
        doms[os.path.basename(p).split(".")[0]] = json.load(io.open(p, encoding="utf-8"))
    names = sorted(doms)

    keysets = {n: {k for k in doms[n] if not k.startswith("_")} for n in names}
    layers = collections.defaultdict(lambda: collections.defaultdict(set))
    for n in names:
        for k in keysets[n]:
            layers[cls(k)][n].add(k)

    print("=== A2 2층 분할 (+폐기분) — 도메인별 기능군 수 ===")
    print("  %-20s %-8s %-8s %-8s" % ("domain", "CORE", "EXT", "DISCARD"))
    for n in names:
        print("  %-20s %-8d %-8d %-8d" % (n, len(layers["CORE"][n]),
                                          len(layers["EXT"][n]), len(layers["DISCARD"][n])))

    print("\n=== 유한성 판정 (분모 = CORE만) ===")
    core_sets = [layers["CORE"][n] for n in names]
    u, i = set().union(*core_sets), set.intersection(*core_sets)
    print("  CORE union=%d  intersection=%d" % (len(u), len(i)))
    print("  CORE 공통:", ", ".join(sorted(i)))
    for n in names:
        missing = u - layers["CORE"][n]
        print("  %-20s CORE 미보유 %2d: %s" % (n, len(missing),
                                              ", ".join(sorted(missing))[:100] or "-"))

    print("\n=== 폐기 후 예상 diff (CORE+EXT 기준·DISCARD 제거) ===")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = names[a], names[b]
            kx = layers["CORE"][x] | layers["EXT"][x]
            ky = layers["CORE"][y] | layers["EXT"][y]
            print("  %s -> %s: +%d / -%d (기존 +%d / -%d)"
                  % (x, y, len(ky - kx), len(kx - ky),
                     len(keysets[y] - keysets[x]), len(keysets[x] - keysets[y])))

    print("\n=== EXT (도메인-특화 opex·정직 회계) ===")
    for n in names:
        print("  %-20s %s" % (n, ", ".join(sorted(layers["EXT"][n])) or "-"))
    print("\n=== DISCARD (P2-b·anti-drift) ===")
    for n in names:
        print("  %-20s %s" % (n, ", ".join(sorted(layers["DISCARD"][n])) or "-"))


if __name__ == "__main__":
    main()
