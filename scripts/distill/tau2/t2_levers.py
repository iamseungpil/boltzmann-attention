# -*- coding: utf-8 -*-
"""레버 레지스트리 — **4축**(태스크·관측 실패·원인·해결책). 정본 = `UNIFIED_TAXONOMY_2026_07_09.md`.

사용자 지시 2026-08-07: *"실패 분류 · 실패 원인 분류 · 해결책 분류 · 태스크 분류 4축으로 만들라.
원인 분류는 **LLM이 뭘 못하는지** 정확하게 표현하고, 해결책 분류를 **LLM이 못하는 걸 우리가 어떤
방법으로 해결하는지**로 분류하라."* · *"코드에서도 통합하라. 80 몇 개나 레버가 되니 복잡해서
문제가 된다."* · *"레버 끄지 마라. 모두 통합하라."*

**왜 축을 갈랐나.** 이전판은 관찰·근본기능·레버를 11개씩 1:1:1로 묶었다. 그래서 "셀"이 원인처럼도
해결책처럼도 읽혔고(내가 C1~C7을 원인 축으로 착각한 이유), 세 자리에서 1:1이 실제로 깨져 있었다:
  · 옛 `중단판단`(조기 포기 **/** 과잉 지속)의 과잉 절반이 옛 `부재판정`과 **이중 계상**
  · 그런데 레버는 이미 `종료 판정`/`부재 종결`로 갈라져 있었다 — **원인이 해결책보다 거칠었다**
  · `권한 월권`은 실측(카드 12/14)됐는데 **레버가 없고**, `의미 소속(⋈)`은 레버가 **있을 수 없다**(경계)

이 모듈이 하는 일은 넷뿐이다:
  1. **결손**(LLM이 못하는 것)을 등급과 함께 둔다 — 등급이 방법을 고른다.
  2. **방법**(엔진에게 허용된 동작)을 **측정된 부작용**과 함께 둔다 — 부작용 없는 레버는 없다.
  3. 셀 = **(결손 × 방법)** 짝 중 우리가 구현한 진입점. 라이브 플래그를 여기에 귀속시킨다.
  4. `enabled(cell)` 하나로 묻는다. **기본은 항상 켬**([[60]]) — 끄기는 이 API에 없다.

⚠이것은 *분류의 코드화*이지 동작 변경이 아니다. 각 셀의 구현 병합(중복 제거)은 이 위에서
한 셀씩 진행한다 — 한 번에 바꾸면 무엇이 회귀했는지 귀속할 수 없다.
"""

import os

__all__ = ["CAUSES", "METHODS", "CELLS", "PARAMS", "HARNESS", "META", "ARM_ONLY", "RETIRED",
           "enabled", "cell_of", "audit", "audit_declared"]

# ── 축 C · 원인 = LLM이 정확히 무엇을 못하는가 ───────────────────────────────
# 등급이 처방을 고른다:
#   미집행 = p_iso ≈ p_traj 높은데 안 한다(scale-flat)  → 결정론이 대신한다 · pass비용 ≈ 0
#   부하   = p_iso > p_traj                            → 궤적에서 덜어낸다
#   능력   = p_iso도 낮고 scale이 올린다                → 산다(thinking·scale·learn) · 매도 있음
#   경계   = p_iso 천장 · scale/CoT/RL 전부 무효        → 권위 이전(되묻기)뿐
# ⚠경계 선언 전 정보-맞춘 격리 의무([[18]]) — 039는 경계가 아니라 전사-슬립이었다.
CAUSES = {
    "조건 재소환 실패": (
        "미집행",
        "정책을 읽고 이해하지만, write를 내는 그 순간에 그 조건을 술어로 세워 진위를 확인하지 않는다.",
        "위반율 scale-flat .103/.070/.075"),
    "집계 미발화": (
        "부하",
        "자료를 손에 쥐고도 수로 줄이지 않는다. 산수 능력의 문제가 아니다 — 격리하면 푼다.",
        "원장 29행 수신 후 창_잔여 미산출 19/22 · 오프라인 결정론 풀이는 100/101 gold 도달(x124)"),
    "사슬 역행 실패": (
        "부하",
        "목표에서 미충족 선행으로 내려가지 못한다. 요건을 인식해도 '나중' 칸에 두고 "
        "지금 실행 가능한 걸음으로 바꾸지 못한다.",
        "PLAN_PROBE t99 격리 2/2 정답 vs 런 누락+날조 · 101 원장조회 2/20"),
    "미검증 단정·정박 치환": (
        "미집행",
        "모르는 값을 모른다고 표시하지 않고 채운다. 근거가 없으면 문맥에서 가장 가까운 토큰으로 채운다.",
        "invited null 7 / false 18 / true 0 (모름을 말할 수단이 있었다) · 날조 70%가 인접 id edit≤2 변형"),
    "완료 무검증": (
        "미집행",
        "자기가 한 일을 실행 원장과 대조하지 않고 끝났다고 믿는다.",
        "telecom 'exhausted all steps'(실제 미해결) · 호출 없는 완료 주장 8건"),
    "전사 발산": (
        "부하",
        "손으로 옮긴 값 하나가 어긋나면 하류 전부를 오염시킨다. 산술은 무손상.",
        "inject 자기일관 0.969@32B — 틀린 상태 위에서 정확히 계산한다"),
    "의미 소속 판정 불가": (
        "경계",
        "'$40,000 트럭이 operations인가' · 후보 2+ 중 옳은 대상 선택. 변이 불변이 아닌 열린 술어.",
        "격리 천장 ~.44 (scale·budget·CoT·RL 전부) · frontier도 동일(14 vs 12)"),
    "부재 미종결": (
        "미집행",
        "없다는 사실을 결론으로 쓰지 못한다. 종결 대신 같은 검색을 반복한다.",
        "t014 검색 루프 · 무제한 열자 동일 요구 106회(전진 0)"),
    "발화-행동 등가 오인": (
        "미집행",
        "말을 행동으로 계상하고 종료한다. 자기 몫을 다른 주체(상담원·손님)에게 넘기고 끝낸다.",
        "실패 40건 중 ≈29건 · 041/t1은 도구 호출 없이 JSON을 본문에 출력하고 이관됐다고 말함"),
    "권한 월권": (
        "미집행",
        "자기 권한 밖 판정을 대신한다. '어느 카드를 원하는가'의 권위자는 손님인데 1장으로 좁혀 제시한다.",
        "12/14 = 결정 시점에 정답이 손님 앞에 없었다 · 반증 003/t0: 모른 채 열거만 하고 손님이 골라 PASS"),
    "유도 실패": (
        "미집행",
        "손님이 실행할 것을 실행 가능한 형태로 주지 못한다. 답을 인자 예시값 자리에 넣어 손님이 그것을 신청한다.",
        "오답 카드를 꺼낸 문장 10건 중 7건이 도구-사용법 지시형(`card_type: Business Gold Rewards Card`)"),
}

# 횡단 속성 — 셀이 아니라 위 결손들의 수식어(C60). 새 결손으로 세지 말 것.
CROSSCUTTING = {
    "표현-민감": "등가 표현 변주에 결정이 반전한다(사실-토큰 동일 5/17에서도). 결정이 φ(X)가 아니라 "
                 "표면형 X에 결합된 증상. 지배 레버=직렬화·열거 · 잔여=learn.",
    "체계핵": "전-trial 동일-오답. 표현-민감의 반대쪽.",
}

# ── 축 S · 해결책 = 엔진에게 허용된 동작 ─────────────────────────────────────
# 방법을 가르는 것은 *엔진이 무엇을 하느냐*이지 어느 실패를 겨냥하느냐가 아니다([[52]] 엔진=이론/LLM=해석).
METHODS = {
    "차단": ("조건 미충족이면 그 호출을 열지 않는다 (닫힘·진위만)",
             "탈출구 이동 — 통행료를 싸게 하자 이관률 37.5→42.2% · over-block"),
    "치환·고정": ("붙을 자리가 없으면 문장을 바꾸거나 다음 채널을 단일값 enum으로 고정한다",
                  "자율성 축소 · 기회비용 미측정"),
    "계산 이관": ("구조화된 값 위의 산수를 엔진이 한다 (닫힘·유한·전수열거)",
                  "잘못된 operand를 참되게 계산한다 (024: operations 전제가 틀렸는데 2.5%를 정확히 반환)"),
    "검증": ("LLM이 {주장, 출처참조}를 내면 엔진은 출처만 본다. 값을 만들지 않는다",
             "자기정당화 통로 — 인접 항목을 핀해 통과할 수 있다"),
    "표면화": ("출처집합 안에 있는데 아직 안 쓴 것을 그 자리에 보인다. 막지 않는다",
               "후보 밀도↑ = ADB(84.56→66.47) · 비강제 신호는 무시된다(Recuse Signal 0/40)"),
    "되묻기": ("판정을 권위자에게 넘긴다",
               "턴 비용 · 과잉 질의"),
}

# 레버가 아니라 능력 구매 — scaffold 축 밖. 매도(sell)를 반드시 함께 잰다.
PURCHASE = {
    "thinking": "F2 결정정확도를 사고 완결·persistence를 판다 (순 ≈0)",
    "scale": "horizon을 산다. compliance·coverage는 사지 못한다(scale-flat)",
    "learn": "무망각·최후순위([[13]])",
    "retry": "☠죽은 레버 — 소형 음성 확증(127 vs 84·p=.004) · 32B는 null(p=.66)",
}

# 등급 → 허용 방법. 등급이 방법을 고른다(§4.2).
ALLOWED = {
    "미집행": ["차단", "검증", "표면화"],
    "부하": ["계산 이관", "치환·고정"],
    "능력": [],          # 능력 구매만 — 결정론 강제 금지
    "경계": ["되묻기"],  # ★scaffold 선택기 금지([[05]] 금지선: 열린 술어 위 강제)
}

# ── 11셀 = (결손 × 방법) 짝 중 구현된 진입점: (결손, 방법들, 철학, 라이브 플래그) ──
CELLS = {
    "조건 게이트": (
        "조건 재소환 실패", ["차단"],
        "정책이 선언한 조건이 충족되기 전에는 그 행동을 열지 않는다. 조건은 정책 축자에서만 오고, "
        "게이트는 진위만 본다 — 행동의 좋고 나쁨은 판단하지 않는다.",
        ["T2_TOOLGATE", "T2_PROCEDURE", "T2_SPEAK_PROHIBIT", "T2_PHASE_OWNER",
         "T2_ENVELOPE_GUARD", "T2_UNAVAIL_PROMISE"]),
    "완결 게이트": (
        "완료 무검증", ["차단", "검증"],
        "끝났다는 주장은 실행 원장으로만 검증한다. 모델의 자기보고는 근거가 아니다.",
        ["T2_COVERAGE_FOLLOWUP", "T2_FOLLOWUP_REQUIRED", "T2_FOLLOWUP_FORCE",
         "T2_FOLLOWUP_READLOOP", "T2_WITHDRAWN_ROW", "T2_DISPATCH_LEDGER"]),
    "선행 강제": (
        "사슬 역행 실패", ["차단", "치환·고정"],
        "표적의 미충족 조상이 있으면 그 요건이 먼저 말한다. 명령은 지금 실행 가능한 걸음이어야 한다 "
        "(사슬의 끝을 명령하지 않는다).",
        ["T2_FORCE_ACTION", "T2_EPLAN", "T2_EPLAN_WALK", "T2_SCAFFOLD_GET", "T2_SG_REQREADS",
         "T2_PREKB", "T2_PIN_READ", "T2_PIN_READ_STEPS", "T2_PROC_PIN_REARM",
         "T2_CALLABLE_HINT", "T2_COV_MIDDRIVE", "T2_UNVERIFIED_FOLLOWUP"]),
    "미사용 표면화": (
        # 같은 결손인데 방법이 다르다(차단이 아니라 표면화) ⇒ 선행 강제의 하위가 아니라 자기 진입점.
        "사슬 역행 실패", ["표면화"],
        "출처집합 안에 있는데 아직 쓰이지 않은 이름을 그 자리에서 보여준다. 막지 않는다. "
        "경계는 `레지스트리 ∩ 실제 회수 텍스트` — 이 교집합 밖은 말하지 않는다.",
        ["T2_DISCOVERY_NAMES", "T2_UNCALLED_UNLOCK", "T2_VERDICT_SURFACE",
         "T2_TRANSFER_LEAVES_STEPS"]),
    "계산 이관": (
        "집계 미발화", ["계산 이관"],
        "구조화된 값 위의 산수는 엔진, 전사·해석은 모델. 엔진은 도메인 텍스트를 읽지 않는다([[59]]).",
        ["T2_COMPUTE", "T2_RESOLVE", "T2_LEDGER", "T2_SG_ISOLATE", "T2_SG_ISOFB",
         "T2_SG_TRACE", "T2_SG_DEDUP", "T2_PRESCRIPTION"]),
    "직렬화·되묻기": (
        "의미 소속 판정 불가", ["계산 이관", "되묻기"],
        "판별 기준이 형식화되면 결정론으로 좁히고, 안 되면 되묻는다. 추측으로 좁히지 않는다.",
        ["T2_REF_VERIFY", "T2_SG_BYREF", "T2_CHOICE_GROUND", "T2_MATCH_COUNT", "T2_ARG_SCHEMA"]),
    "정본 상태 대조": (
        "전사 발산", ["검증"],
        "손으로 옮긴 값은 원장과 대조한다. 엔진은 고치지 않고 어긋난 사실만 말한다.",
        ["T2_TRANSCRIBE", "T2_BRANCH_REGROUND", "T2_STALE_STRIP", "T2_READ_DEDUP"]),
    "종료 판정": (
        "발화-행동 등가 오인", ["차단"],
        "종료는 남은 절차 단계의 유무로 판정한다. 피로·반복 횟수로 판정하지 않는다.",
        ["T2_TERM_GRANT", "T2_TERM_GRANT_USERDEMAND", "T2_TRANSFER_TIER", "T2_REQUIRE_DOC"]),
    "근거 확인": (
        "미검증 단정·정박 치환", ["검증"],
        "행동을 좌우하는 사실 주장은 출처를 대야 한다. 엔진은 출처만 검증하고 값을 만들지 않는다.",
        ["T2_SOURCE", "T2_PROV_REGEN", "T2_GROUND", "T2_SG_GROUND", "T2_WRITE_PROV",
         "T2_WRITE_EVIDENCE", "T2_WRITE_ARG_GROUND", "T2_CLAIM_PROV", "T2_GIVE_QUOTE",
         "T2_QUOTE_HINT", "T2_QUOTE_PIN", "T2_UNLOCK_PROV", "T2_UNKNOWN_NAME_BL",
         "T2_UNLOCK_NAME", "T2_FAB_STRIP", "T2_SG_TRUTH", "T2_PROD_BIND"]),
    "역할 확인·실행 강제": (
        "발화-행동 등가 오인", ["치환·고정", "검증"],
        "누가 실행하는지는 레지스트리에서 도출한다. 판정 불가면 그 문장을 뺀다. "
        "말로 실행을 대체하지 않는다.",
        ["T2_DISPATCH_ROLE", "T2_DISPATCH_ROLE_ENVSET", "T2_TOOL_CHANNEL", "T2_TOOL_SIGNATURE",
         "T2_TOOLLIST", "T2_VALUE_ACQUIRE", "T2_HAVE_VALUE", "T2_HAVE_VALUE_FORCE",
         "T2_ARG_PRODUCERS", "T2_GUIDED"]),
    "부재 종결": (
        "부재 미종결", ["차단", "표면화"],
        "없음은 멈출 근거다. 단 찾은 범위를 함께 말한다(없다와 못 찾았다를 가른다).",
        ["T2_SEARCH_EXHAUST_NUDGE", "T2_KB_NOHIT_SURFACE", "T2_PROC_ABSENT",
         "T2_SG_WINDOW_ABSTAIN", "T2_ABSTAIN_FIELDS"]),
    "지시 scaffold": (
        "유도 실패", ["치환·고정", "표면화"],
        "손님이 실행할 것은 실행 가능한 형태로 준다(도구·인자·값). "
        "지시했다는 사실을 실행으로 세지 않는다.",
        ["T2_USER_TOOL_NOTE", "T2_UNINSTRUCTABLE", "T2_GIVE_EXEC_NUDGE",
         "T2_GIVE_RELEVANCE_NUDGE", "T2_DUP_REPRESENT"]),
}

# ★결손인데 셀이 없는 자리 — 다음 작업의 목록이다(§5.1).
GAPS = {
    "권한 월권": "레버 미구현. 표적 12/14로 실측된 최대 축. 방법=표면화(권위 가드: "
                 "'적격 집합을 축약 없이 인계했는가'는 닫힘 — 판정은 손님에게 남긴다). 최우선 구현 후보.",
    "의미 소속 판정 불가": "경계 — 레버가 있을 수 없다. `직렬화·되묻기`는 기준형(닫힘)만 닫는다. "
                           "⋈ 잔여가 모트다. scaffold 선택기 금지([[05]]).",
}

# ── 파라미터: 레버가 아니라 **소속 레버의 조정 손잡이** ────────────────────────
# ⚠파라미터는 레버가 아니지만 무해하지도 않다. 하드코딩 `_t2_resolve_deny < 3`이
# 원장 조회가 한 번도 명령된 적 없던 이유였다(3회가 turn 4·6·8에 소진 · 첫 요건 충족은 turn 11 전후).
# 횟수로 억제하는 파라미터는 [[57]] 위반이다 — 인자 변화로 억제할 것.
PARAMS = {
    "T2_PROCEDURE_CAP": "조건 게이트", "T2_ENVELOPE_CAP": "조건 게이트",
    "T2_FOLLOWUP_CAP": "완결 게이트", "T2_FOLLOWUP_PROGRESS_REFUND": "완결 게이트",
    "T2_COV_MIDDRIVE_K": "선행 강제", "T2_EPLAN_DRIVE_K": "선행 강제",
    "T2_ACTION_PROGRESS_REFUND": "선행 강제",
    "T2_CLAIMPROV_CAP": "근거 확인", "T2_WEV_ROUNDS": "근거 확인",
    "T2_TRANSCRIBE_CAP": "정본 상태 대조",
    "T2_SEARCH_EXHAUST_TH": "부재 종결", "T2_KB_NOHIT_K": "부재 종결",
    "T2_PROC_ABSENT_K": "부재 종결", "T2_PROC_ABSENT_CAP": "부재 종결",
}

# 레버가 아니라 레버에 대한 규칙(주체=우리 층)
META = ["T2_ARBITRATE", "T2_WINDOW",
        "T2_VERIFY_DENY_CAP", "T2_PARAM_CAP", "T2_PRECLOSE_CAP"]

# 판정하지 않는 것 — 끄면 능력이 조용히 사라진다(2026-08-07 실측 3회). arm 정의에서 상수.
HARNESS = ["T2_GATE_REGEN", "T2_GATE_REGEN_K", "T2_PROV_REGEN_K", "T2_PROV_MODE",
           "T2_OVERFLOW_GUARD", "T2_TRUNC_GUARD", "T2_DYN_MT", "T2_MT_FLOOR",
           "T2_DYN_MT_MARGIN", "T2_MAXPROMPT", "T2_VIEW_COMPACT", "T2_VIEW_ANNOTATE",
           "T2_VIEW_COMPACT_MINTOTAL", "T2_VIEW_MSG_CAP",
           "T2_PAIRCHECK", "T2_PAIRFIX", "T2_FAILED_PERSIST", "T2_A2_VARIANT",
           "T2_KB_DOCS_DIR", "T2_LLM_TIMEOUT", "T2_LLM_RETRIES", "T2_AGENT_MAX_TOKENS",
           "T2_REGEN_BUDGET"]

# 귀속 arm 전용 — 운영 스택에서는 켜지 않는다(기본 OFF가 정상)
ARM_ONLY = ["T2_EPLAN_WALK_HOLD"]

# [[57]] 위반(횟수로 억제) — 정체-과금·지문 억제가 대체함
RETIRED = ["T2_REPEAT_CAP", "T2_UNKNOWN_REPEAT_GUARD", "T2_DD_FB"]


def enabled(cell):
    """이 셀이 켜져 있는가 — **기본 항상 참**([[60]] 레버는 끄지 않는다).

    `T2_OFF_CELLS`에 셀 이름을 쉼표로 주면 그 셀만 꺼진다. 이 문은 **귀속 실험용 비상구**이고
    운영 구성이 아니다 — 쓰면 태그에 기록해야 한다.
    """
    off = {c.strip() for c in (os.environ.get("T2_OFF_CELLS") or "").split(",") if c.strip()}
    return cell not in off


def cell_of(flag):
    for name, (_cause, _methods, _phil, flags) in CELLS.items():
        if flag in flags:
            return name
    if flag in PARAMS:
        return "(파라미터→%s)" % PARAMS[flag]
    if flag in META:
        return "(메타)"
    if flag in HARNESS:
        return "(하네스)"
    if flag in ARM_ONLY:
        return "(arm전용)"
    if flag in RETIRED:
        return "(폐기)"
    return None


def audit(live_flags):
    """라이브 플래그 중 **어느 칸에도 없는 것**. 비어야 정상이다.

    미분류가 남는다는 것은 그 레버가 무슨 **결손**을 고치는지 아무도 적지 않았다는 뜻이고,
    그 상태에서 통합하면 조용히 기능이 사라진다.
    """
    return sorted(f for f in live_flags if cell_of(f) is None)


def audit_declared(live_flags):
    """반대 방향 — 셀에 **선언은 됐는데 라이브가 아닌** 플래그. 死배선·미구현의 탐지자다.

    `audit()`만 돌리면 이쪽이 안 보인다. 2026-08-07에 `T2_LEDGER`가 셀에 있으면서 실행 0이던
    사고가 정확히 이 방향이었다.
    """
    live = set(live_flags)
    out = []
    for name, (_c, _m, _p, flags) in CELLS.items():
        for f in flags:
            if f not in live:
                out.append((name, f))
    return sorted(out)


def _fmt_cell(name):
    cause, methods, _phil, flags = CELLS[name]
    grade = CAUSES[cause][0] if cause in CAUSES else "?"
    return "%-16s ← %-20s [%s] via %-18s %2d개" % (
        name, cause, grade, "+".join(methods), len(flags))


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

    print("축 C 결손 %d종 · 축 S 방법 %d종 · 셀 %d개 · 라이브 플래그 %d종\n"
          % (len(CAUSES), len(METHODS), len(CELLS), len(live)))

    print("[결손 → 등급 → 허용 방법]")
    for cause, (grade, _stmt, _ev) in CAUSES.items():
        cells = [n for n, v in CELLS.items() if v[0] == cause] or ["**셀 없음**"]
        print("  %-18s [%-4s] 허용=%-18s → %s"
              % (cause, grade, ",".join(ALLOWED.get(grade, [])) or "구매만", " · ".join(cells)))

    print("\n[셀 = (결손 × 방법)]")
    for name in CELLS:
        print("  " + _fmt_cell(name))

    print("\n[결손인데 셀이 없는 자리]")
    for k, v in GAPS.items():
        print("  ! %s — %s" % (k, v.split(".")[0]))

    miss = audit(live)
    print("\n미분류 %d종%s" % (len(miss), ":" if miss else " ✓"))
    for m in miss:
        print("   ", m)

    dead = audit_declared(live)
    print("\n선언됐으나 비-라이브 %d종%s" % (len(dead), ":" if dead else " ✓"))
    for cell, f in dead:
        print("    %-18s %s" % (cell, f))
