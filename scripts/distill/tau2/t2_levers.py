# -*- coding: utf-8 -*-
"""레버 레지스트리 — 11셀·셀당 진입점 하나. 플래그는 **내부 detail**로 내린다.

사용자 지시 2026-08-07: *"코드에서도 통합하라. 80 몇 개나 레버가 되니 복잡해서 문제가 된다.
통합하고 일반화하고 단순화하라."* · *"레버 끄지 마라. 모두 통합하라."*

문제의 형태: `go_stack.sh`가 켜는 `T2_*`가 **128종**(레버 97 + 파라미터·설정 31)이다. 이 수 자체가
사고를 만들었다 — 하루에 세 번, 끈 것이 호스트·채널·보호였고 arm이 조용히 능력을 잃었다.
그리고 어느 플래그가 어느 실패를 고치는지 **코드 어디에도 적혀 있지 않았다**(주석에만 있었다).

이 모듈이 하는 일은 셋뿐이다:
  1. 11셀을 **이름·기전·철학**과 함께 한 곳에 둔다(정본 `UNIFIED_TAXONOMY_2026_07_09`).
  2. 라이브 플래그를 셀에 **귀속**시킨다 — 귀속 없는 플래그는 미분류로 드러난다.
  3. `enabled(cell)` 하나로 묻는다. **기본은 항상 켬**([[60]]) — 끄기는 이 API에 없다.

⚠이것은 *분류의 코드화*이지 동작 변경이 아니다. 각 셀의 구현 병합(중복 제거)은 이 위에서
한 셀씩 진행한다 — 한 번에 바꾸면 무엇이 회귀했는지 귀속할 수 없다.
"""

import os

__all__ = ["CELLS", "HARNESS", "META", "enabled", "cell_of", "audit"]

# ── 11셀: (기전, 철학, 이 셀에 속한 라이브 플래그) ─────────────────────────────
CELLS = {
    "조건 게이트": (
        "정책 미집행 — 확인·조건을 안 지키고 진행",
        "정책이 선언한 조건이 충족되기 전에는 그 행동을 열지 않는다. 조건은 정책 축자에서만 오고, "
        "게이트는 진위만 본다 — 행동의 좋고 나쁨은 판단하지 않는다.",
        ["T2_TOOLGATE", "T2_PROCEDURE", "T2_SPEAK_PROHIBIT", "T2_PHASE_OWNER",
         "T2_ENVELOPE_GUARD", "T2_UNAVAIL_PROMISE"]),
    "완결 게이트": (
        "미완결·완료 오판정 — 남았는데 끝났다고 함",
        "끝났다는 주장은 실행 원장으로만 검증한다. 모델의 자기보고는 근거가 아니다.",
        ["T2_COVERAGE_FOLLOWUP", "T2_FOLLOWUP_REQUIRED", "T2_FOLLOWUP_FORCE",
         "T2_FOLLOWUP_READLOOP", "T2_WITHDRAWN_ROW", "T2_DISPATCH_LEDGER"]),
    "선행 강제": (
        "선행 미해결 — 필요한 선행 행동·자원에 도달 못 함",
        "표적의 미충족 조상이 있으면 그 요건이 먼저 말한다. 명령은 지금 실행 가능한 걸음이어야 한다.",
        ["T2_FORCE_ACTION", "T2_EPLAN", "T2_EPLAN_WALK", "T2_SCAFFOLD_GET", "T2_SG_REQREADS",
         "T2_PREKB", "T2_PIN_READ", "T2_PIN_READ_STEPS", "T2_PROC_PIN_REARM",
         "T2_CALLABLE_HINT", "T2_COV_MIDDRIVE", "T2_UNVERIFIED_FOLLOWUP",
         # 하위: 미사용 표면화 — 출처 안에 있는데 아직 안 쓴 것
         "T2_DISCOVERY_NAMES", "T2_UNCALLED_UNLOCK", "T2_VERDICT_SURFACE",
         "T2_TRANSFER_LEAVES_STEPS"]),
    "계산 이관": (
        "집계 미수행·산수 오류 — 받아 놓고 수로 줄이지 않음",
        "구조화된 값 위의 산수는 엔진, 전사·해석은 모델. 엔진은 도메인 텍스트를 읽지 않는다.",
        ["T2_COMPUTE", "T2_RESOLVE", "T2_LEDGER", "T2_SG_ISOLATE", "T2_SG_ISOFB",
         "T2_SG_TRACE", "T2_SG_DEDUP", "T2_PRESCRIPTION"]),
    "직렬화·되묻기": (
        "대상 오지목 — 의도→대상 매칭 실패",
        "판별 기준이 형식화되면 결정론으로 좁히고, 안 되면 되묻는다. 추측으로 좁히지 않는다.",
        ["T2_REF_VERIFY", "T2_SG_BYREF", "T2_CHOICE_GROUND", "T2_MATCH_COUNT", "T2_ARG_SCHEMA"]),
    "정본 상태 대조": (
        "상태 발산 — 한 번 어긋난 상태가 하류를 오염",
        "손으로 옮긴 값은 원장과 대조한다. 엔진은 고치지 않고 어긋난 사실만 말한다.",
        ["T2_TRANSCRIBE", "T2_BRANCH_REGROUND", "T2_STALE_STRIP", "T2_READ_DEDUP"]),
    "종료 판정": (
        "조기 포기·과잉 지속",
        "종료는 남은 절차 단계의 유무로 판정한다. 피로·반복 횟수로 판정하지 않는다.",
        ["T2_TERM_GRANT", "T2_TERM_GRANT_USERDEMAND", "T2_TRANSFER_TIER", "T2_REQUIRE_DOC"]),
    "근거 확인": (
        "근거 없는 값 — 출처 없이 수·이름을 단정",
        "행동을 좌우하는 사실 주장은 출처를 대야 한다. 엔진은 출처만 검증하고 값을 만들지 않는다.",
        ["T2_SOURCE", "T2_PROV_REGEN", "T2_GROUND", "T2_SG_GROUND", "T2_WRITE_PROV",
         "T2_WRITE_EVIDENCE", "T2_WRITE_ARG_GROUND", "T2_CLAIM_PROV", "T2_GIVE_QUOTE",
         "T2_QUOTE_HINT", "T2_QUOTE_PIN", "T2_UNLOCK_PROV", "T2_UNKNOWN_NAME_BL",
         "T2_UNLOCK_NAME", "T2_FAB_STRIP", "T2_SG_TRUTH", "T2_PROD_BIND"]),
    "역할 확인·실행 강제": (
        "실행 회피 — 말만 하고 실행 안 함·비요청 실행",
        "누가 실행하는지는 레지스트리에서 도출한다. 판정 불가면 그 문장을 뺀다. "
        "말로 실행을 대체하지 않는다.",
        ["T2_DISPATCH_ROLE", "T2_DISPATCH_ROLE_ENVSET", "T2_TOOL_CHANNEL", "T2_TOOL_SIGNATURE",
         "T2_TOOLLIST", "T2_VALUE_ACQUIRE", "T2_HAVE_VALUE", "T2_HAVE_VALUE_FORCE",
         "T2_ARG_PRODUCERS", "T2_GUIDED"]),
    "부재 종결": (
        "부재 미종결 — 없는 것을 못 닫고 검색 루프",
        "없음은 멈출 근거다. 단 찾은 범위를 함께 말한다(없다와 못 찾았다를 가른다).",
        ["T2_SEARCH_EXHAUST_NUDGE", "T2_KB_NOHIT_SURFACE", "T2_PROC_ABSENT",
         "T2_SG_WINDOW_ABSTAIN", "T2_ABSTAIN_FIELDS"]),
    "지시 scaffold": (
        "손님 유도 실패",
        "손님이 실행할 것은 실행 가능한 형태로 준다(도구·인자·값). "
        "지시했다는 사실을 실행으로 세지 않는다.",
        ["T2_USER_TOOL_NOTE", "T2_UNINSTRUCTABLE", "T2_GIVE_EXEC_NUDGE",
         "T2_GIVE_RELEVANCE_NUDGE", "T2_DUP_REPRESENT"]),
}

# 판정하지 않는 것 — 끄면 능력이 조용히 사라진다(2026-08-07 실측 3회)
HARNESS = ["T2_GATE_REGEN", "T2_PROV_REGEN_K", "T2_PROV_MODE", "T2_OVERFLOW_GUARD",
           "T2_TRUNC_GUARD", "T2_DYN_MT", "T2_MAXPROMPT", "T2_VIEW_COMPACT", "T2_VIEW_ANNOTATE",
           "T2_PAIRCHECK", "T2_PAIRFIX", "T2_FAILED_PERSIST", "T2_A2_VARIANT",
           "T2_KB_DOCS_DIR", "T2_LLM_TIMEOUT", "T2_LLM_RETRIES", "T2_AGENT_MAX_TOKENS",
           "T2_REGEN_BUDGET"]

# 레버가 아니라 레버에 대한 규칙
META = ["T2_ARBITRATE", "T2_WINDOW"]

# [[57]] 위반(횟수로 억제) — 정체-과금·지문 억제가 대체함
RETIRED = ["T2_REPEAT_CAP", "T2_UNKNOWN_REPEAT_GUARD"]


def enabled(cell):
    """이 셀이 켜져 있는가 — **기본 항상 참**([[60]] 레버는 끄지 않는다).

    `T2_OFF_CELLS`에 셀 이름을 쉼표로 주면 그 셀만 꺼진다. 이 문은 **귀속 실험용 비상구**이고
    운영 구성이 아니다 — 쓰면 태그에 기록해야 한다.
    """
    off = {c.strip() for c in (os.environ.get("T2_OFF_CELLS") or "").split(",") if c.strip()}
    return cell not in off


def cell_of(flag):
    for name, (_mech, _phil, flags) in CELLS.items():
        if flag in flags:
            return name
    if flag in HARNESS:
        return "(하네스)"
    if flag in META:
        return "(메타)"
    if flag in RETIRED:
        return "(폐기)"
    return None


def audit(live_flags):
    """라이브 플래그 중 **어느 셀에도 없는 것**을 돌려준다. 비어야 정상이다.

    미분류가 남는다는 것은 그 레버가 무슨 실패를 고치는지 아무도 적지 않았다는 뜻이고,
    그 상태에서 통합하면 조용히 기능이 사라진다.
    """
    return sorted(f for f in live_flags if cell_of(f) is None)


if __name__ == "__main__":
    import io
    import re
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    here = os.path.dirname(os.path.abspath(__file__))
    src = io.open(os.path.join(here, "go_stack.sh"), encoding="utf-8", errors="replace").read()
    live = sorted(set(re.findall(r"\bT2_[A-Z_0-9]+(?==)", src)))
    print("셀 %d개 · 라이브 플래그 %d종" % (len(CELLS), len(live)))
    for name, (mech, _phil, flags) in CELLS.items():
        print("  %-22s ← %-38s %d개" % (name, mech[:36], len(flags)))
    miss = audit(live)
    print("\n미분류 %d종:" % len(miss))
    for m in miss:
        print("   ", m)
