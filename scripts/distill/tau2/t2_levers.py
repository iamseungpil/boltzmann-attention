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
    "출처 근거 확보": ("LLM이 {주장, 출처참조}를 내면 엔진은 출처만 본다. 값을 만들지 않는다. "
                       "**주체를 가리지 않는다** — 모델·env 출력·손님·우리 층 넷 다 같은 검정",
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

# ══════════════════════════════════════════════════════════════════════════════
#  ★코드 체계 — 이름은 바뀌어도 **코드는 안 바뀐다**
# ══════════════════════════════════════════════════════════════════════════════
#
# 사용자 지시 2026-08-07: *"각 축에 코드도 부여하라. 원인 코드와 해결책 코드를 부여하고
# **이 코드를 관리함으로써 나중에 새로운 레버가 추가돼도 기전을 뒤집지 않게** 하라."*
#
# ⚠[[48]]의 *"새 코드 만들지 말 것"* 과 충돌하지 않는다 — **정반대의 처방이다.**
# 금지된 것은 *이름 족을 계속 새로 만드는 것*(F→G→BC→N→C·표류 5회)이고, 그 원인은
# **안정된 식별자가 없어서 매번 새 이름을 지은 것**이다. 코드는 그 백신이다:
#   · 이름은 더 정확해질 수 있다. **코드는 그대로 간다.** ⇒ 개명이 재분류가 되지 않는다.
#   · 새 레버는 **기존 코드 짝에 붙는다.** 붙을 데가 없으면 그건 코드 신청이고 설계 리뷰다.
#
# 접두사 충돌 확인(2026-08-07·reports 전수 grep): DF·MT·OB·LY **전부 0건**.
# 기존 사용 중이라 피한 것: F·G·BC·N·C·M·P·L·W·T·E·A·B.

FAMILIES = {                       # OB = OBservation (관측 족)
    "OB1": "날조",
    "OB2": "오선택",
    "OB3A": "over-action (행동)",
    "OB3B": "over-action (판정)",
    "OB4": "under-action",
    "OB5": "선행 미충족",
    "OB6": "상태 오염",
}

CAUSE_CODES = {                    # DF = DeFicit (결손 = LLM이 못하는 것)
    "DF1": ("미검증 단정·정박 치환", "OB1"),
    "DF2": ("의미 소속 판정 불가", "OB2"),
    "DF3": ("조건 재소환 실패", "OB3A"),
    "DF4": ("권한 월권", "OB3B"),
    "DF5": ("완료 무검증", "OB4"),
    "DF6": ("발화-행동 등가 오인", "OB4"),
    "DF7": ("부재 미종결", "OB4"),
    "DF8": ("유도 실패", "OB4"),
    "DF9": ("사슬 역행 실패", "OB5"),
    "DF10": ("집계 미발화", "OB6"),
    "DF11": ("전사 발산", "OB6"),
}

METHOD_CODES = {                   # MT = MeThod (해결책 = 엔진에게 허용된 동작)
    "MT1": "차단",
    "MT2": "치환·고정",
    "MT3": "계산 이관",
    "MT4": "출처 근거 확보",
    "MT5": "표면화",
    "MT6": "되묻기",
}

LAYER_CODES = {                    # LY = LaYer (실시 순서 — 방법과 1:1이 아니다)
    "LY0": ("하네스", None),
    "LY1": ("출처 근거 확보", "MT4"),
    "LY2": ("차단", "MT1"),
    "LY3": ("선행", "MT2"),        # ⚠LY2·LY3 둘 다 deny를 쓴다 — 층은 *순서*이지 방법이 아니다
    "LY4": ("계산 이관", "MT3"),
    "LY5": ("표면화", "MT5"),
    "LY6": ("되묻기", "MT6"),
}

# 발급했다가 접은 코드 — **번호 재사용 금지**(재사용하면 옛 로그가 조용히 오역된다)
RETIRED_CODES = {
    # 옛 "중단판단"은 코드를 받기 전에 해체됐다(조기=DF6 / 과잉=DF7·2026-08-07).
}

# ★코드 관리 규칙 — 새 레버가 기전을 뒤집지 못하게 하는 다섯 문장
CODE_RULES = [
    "1. 코드는 **불변**이다. 한 번 발급하면 뜻을 바꾸지 않는다. 이름은 더 정확해져도 된다.",
    "2. 폐기는 RETIRED_CODES에 사유와 함께 남기고 **번호는 재사용하지 않는다**.",
    "3. 새 레버는 **기존 (DFn, MTn) 짝에 붙는다.** 코드를 새로 만들며 들어오지 않는다.",
    "4. 붙을 짝이 없으면 그것은 코드 신청이고 **설계 리뷰 트리거**다 — "
    "실측 근거([S]/[M])와 *왜 기존 코드로 안 되는가*(비-포함)를 대야 발급한다.",
    "5. 감사는 코드로 한다: 미분류 0 · 死배선 0 · **코드 없는 레버 0** · 폐기코드 재사용 0.",
]

# ★다중 결손 선언 — 한 레버가 둘 이상을 겨냥할 수 있다. **단 선언해야 한다.**
#
# 코드 체계를 켜자마자 셀↔세부기전 불일치 **7건**이 나왔다. 뜯어보니 버그가 아니라
# **1:1 가정이 또 너무 좁았던 것**이다(옛 3축의 1:1:1과 같은 병). 그래서 다중을 허용하되
# **미선언 불일치는 여전히 위반**으로 둔다 — 그래야 새 레버가 조용히 기전을 뒤집지 못한다.
# 형식: flag → (주 결손, [부 결손…], 왜 둘인가)
MULTI_CAUSE = {
    "T2_UNAVAIL_PROMISE": ("DF1", ["DF3"],
        "미보유 기능을 약속하는 것은 **없는 것을 만드는 것**(DF1)이고, 차단은 조건 게이트가 한다(DF3)."),
    "T2_BRANCH_REGROUND": ("DF11", ["DF9"],
        "분기 후 재접지 = 어긋난 상태를 다시 세우는 것(DF11)이면서, 그 효과는 close 선행 차단(DF9)으로 실측됐다(C146/C149)."),
    "T2_DISPATCH_ROLE_ENVSET": ("DF1", ["DF6"],
        "존재하지 않는 도구 이름을 손님에게 건네는 것은 **이름 날조**(DF1)이자 실행 회피(DF6)다. x88: 집합 밖 252건·gold 오차단 0."),
    "T2_ARG_PRODUCERS": ("DF8", ["DF6"],
        "필수 인자의 생산자를 짚는 것은 손님 유도(DF8)이면서 오도구 전환(DF6) 표적이다(040/041)."),
    "T2_KB_NOHIT_SURFACE": ("DF1", ["DF7"],
        "전-0점 표면화는 **절차 날조 금지**(DF1·012)와 **없음의 종결 근거**(DF7·014/015) 둘 다에 쓰인다."),
    "T2_ABSTAIN_FIELDS": ("DF5", ["DF7"],
        "abstain하면서 결핍 필드를 지목하는 것은 완결 판정(DF5)이자 부재 표면화(DF7)다."),
    "T2_PHASE_OWNER": ("DF9", ["DF3"],
        "auth 조상 미충족을 보는 것은 조건 판정(DF3)이지만, 2026-08-07 DAG-우선 이후 **하는 일은 "
        "미충족 조상을 명령하는 것**(DF9)이다. 같은 선언에서 두 기전이 나오므로 둘 다 선언한다."),
    "T2_DUP_REPRESENT": ("DF10", ["DF8"],
        "DUP-COMPUTE 스텁의 이전 결과 재제시는 계산 결과 보존(DF10)이고, 그 재제시가 손님 안내에도 쓰인다(DF8)."),
}

_CAUSE_BY_NAME = {v[0]: k for k, v in CAUSE_CODES.items()}
_METHOD_BY_NAME = {v: k for k, v in METHOD_CODES.items()}


def code_of_cause(name):
    return _CAUSE_BY_NAME.get(name)


def code_of_method(name):
    return _METHOD_BY_NAME.get(name)


def lever_codes(flag):
    """이 레버의 (원인코드, [해결책코드…], 확정도). 확정도 = 'fixed' | 'inherited' | None.

    'fixed'     = `t2_stack.MECHANISMS`가 **세부 기전과 함께** 층(=방법)을 배정했다.
    'inherited' = 아직 세부 기전이 없어 **소속 셀의 방법을 물려받았다** — 방법이 둘이면 안 갈렸다.
                  규칙 3의 대상이고, 이 상태로는 `speak()`가 말할 수 없다.
    """
    hit = None
    for name, (cause, methods, _p, flags) in CELLS.items():
        if flag in flags:
            hit = (name, cause, methods)
            break
    if hit is None:
        return (None, [], None)
    _n, cause, methods = hit
    dfc = code_of_cause(cause)
    try:
        import t2_stack as _S
        layer = _S.layer_of(flag)
    except Exception:
        layer = None
    if layer:
        key = next((k for k, v in LAYER_CODES.items() if v[0] == layer), None)
        mt = LAYER_CODES.get(key, (None, None))[1] if key else None
        if mt:
            return (dfc, [mt], "fixed")
    return (dfc, [code_of_method(m) for m in methods if code_of_method(m)], "inherited")


def audit_codes():
    """코드 규칙 5의 집행 — 위반 목록. 비어야 정상이다."""
    bad = []
    for cause in CAUSES:
        if code_of_cause(cause) is None:
            bad.append(("결손에 코드 없음", cause))
    for m in METHODS:
        if code_of_method(m) is None:
            bad.append(("방법에 코드 없음", m))
    for name, (cause, methods, _p, flags) in CELLS.items():
        if code_of_cause(cause) is None:
            bad.append(("셀의 결손이 미등록", "%s ← %s" % (name, cause)))
        for f in flags:
            df, mts, _how = lever_codes(f)
            if df is None or not mts:
                bad.append(("레버에 코드 없음", f))
    for d in sorted(set(CAUSE_CODES) & set(RETIRED_CODES)):
        bad.append(("폐기 코드 재사용", d))
    # ★교차 검사 — 셀이 말하는 결손과 세부 기전이 말하는 결손이 다르면 하나는 틀렸다.
    #   이것이 코드 체계의 값이다: 이름만 있을 땐 두 곳이 어긋나도 아무도 모른다.
    try:
        import t2_stack as _S
        mech_cause = {}
        for cause, _m, _e, f, _l in _S.MECHANISMS:
            if f:
                mech_cause.setdefault(f, cause)
        for name, (cause, _methods, _p, flags) in CELLS.items():
            for f in flags:
                mc = mech_cause.get(f)
                if not mc or mc == cause:
                    continue
                decl = MULTI_CAUSE.get(f)
                pair = {code_of_cause(cause), code_of_cause(mc)}
                if decl and pair <= ({decl[0]} | set(decl[1])):
                    continue          # ★선언된 다중 결손 — 통과
                bad.append(("셀↔세부기전 결손 불일치(미선언)",
                            "%s: 셀 %s(%s) ↔ 기전 %s(%s)"
                            % (f, cause, code_of_cause(cause), mc, code_of_cause(mc))))
    except Exception as e:
        bad.append(("교차 검사 불가", repr(e)))
    return bad


# 등급 → 허용 방법. 등급이 방법을 고른다(§3.2).
ALLOWED = {
    "미집행": ["차단", "출처 근거 확보", "표면화"],
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
        "완료 무검증", ["차단", "출처 근거 확보"],
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
        "전사 발산", ["출처 근거 확보"],
        "손으로 옮긴 값은 원장과 대조한다. 엔진은 고치지 않고 어긋난 사실만 말한다.",
        ["T2_TRANSCRIBE", "T2_BRANCH_REGROUND", "T2_STALE_STRIP", "T2_READ_DEDUP"]),
    "종료 판정": (
        "발화-행동 등가 오인", ["차단"],
        "종료는 남은 절차 단계의 유무로 판정한다. 피로·반복 횟수로 판정하지 않는다.",
        ["T2_TERM_GRANT", "T2_TERM_GRANT_USERDEMAND", "T2_TRANSFER_TIER", "T2_REQUIRE_DOC",
         "T2_NOTICE_REPEAT"]),
    "출처 근거 확보": (
        "미검증 단정·정박 치환", ["출처 근거 확보"],
        "행동을 좌우하는 사실 주장은 출처를 대야 한다. 엔진은 출처만 검증하고 값을 만들지 않는다.",
        ["T2_SOURCE", "T2_PROV_REGEN", "T2_GROUND", "T2_SG_GROUND", "T2_WRITE_PROV",
         "T2_WRITE_EVIDENCE", "T2_WRITE_ARG_GROUND", "T2_CLAIM_PROV", "T2_GIVE_QUOTE",
         "T2_QUOTE_HINT", "T2_QUOTE_PIN", "T2_UNLOCK_PROV", "T2_UNKNOWN_NAME_BL",
         "T2_UNLOCK_NAME", "T2_FAB_STRIP", "T2_SG_TRUTH", "T2_PROD_BIND",
         "T2_UNKNOWN_REPEAT_GUARD", "T2_SELF_DECLARATION"]),
    "역할 확인·실행 강제": (
        "발화-행동 등가 오인", ["치환·고정", "출처 근거 확보"],
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
    "T2_CLAIMPROV_CAP": "출처 근거 확보", "T2_WEV_ROUNDS": "출처 근거 확보",
    "T2_TRANSCRIBE_CAP": "정본 상태 대조",
    "T2_SEARCH_EXHAUST_TH": "부재 종결", "T2_KB_NOHIT_K": "부재 종결",
    "T2_PROC_ABSENT_K": "부재 종결", "T2_PROC_ABSENT_CAP": "부재 종결",
}

# 레버가 아니라 레버에 대한 규칙(주체=우리 층)
META = ["T2_ARBITRATE", "T2_WINDOW",
        "T2_VERIFY_DENY_CAP", "T2_PARAM_CAP", "T2_PRECLOSE_CAP"]

# ══════════════════════════════════════════════════════════════════════════════
#  ★하네스란 무엇인가 (2026-08-07 사용자 질문: *"env도 아니고 오케스트레이터도 아니고 우리 도구도 아닌가?"*)
# ══════════════════════════════════════════════════════════════════════════════
#
# **하네스는 우리 코드다.** env도 오케스트레이터도 아니다. 층위는 넷이다:
#
#   ① env (tau2 도메인)        도구·DB·문서. ⚠**수정 금지** — 고치면 벤치 변경 = 실험 무효([[03b]]).
#                              그리고 [[25]]: env 출력은 **외부 주장**이다(user-sim과 동급).
#   ② 오케스트레이터 (tau2)     실행 루프. 우리는 **감싼다**(훅), 고치지 않는다.
#   ③ 우리 층 — **레버**        도메인에 대해 **판정한다**. 모델의 결정을 좌우한다.
#   ④ 우리 층 — **하네스**      도메인에 대해 **아무것도 주장하지 않는다.** 실행 가능성만 보장한다.
#
# ③과 ④는 **같은 우리 코드**이고, 가르는 술어는 하나다(닫힘):
#
#   > 이 플래그를 끄면 **sim이 죽거나 무효가 되는가**(자원 초과·크래시·측정 오염)? → **하네스**
#   > 아니면 **모델의 결정이 달라지는가**?                                        → **레버**
#
# ★★**발화 여부는 기준이 아니다**(2026-08-07 사용자 지적으로 확정). 하네스도 **말한다** —
#   `truncguard`는 `_ap_regen`으로 모델에게 문구를 보낸다(`t2_gate_patch:7105`).
#   가르는 것은 *말하느냐*가 아니라 **무엇을 판정하느냐**다:
#     · `finish_reason == "length"`  → **채널 사실**. 도메인에 대해 아무 주장도 하지 않는다 → 하네스
#     · *"이 카드가 자격이 되는가"*   → **도메인 판정** → 레버
#   ⇒ 그래서 층 표에 **LY0 하네스**가 실재하고, 하네스가 층을 갖는 것이 모순이 아니다.
#
# ⚠**검토 후보 1건**: `T2_ENVELOPE_GUARD`(현재 조건 게이트 셀)는 *봉투 파싱 실패*를 본다 —
#   그건 **우리가 요구한 출력 형식**이므로 채널이지 도메인이 아니다. `TRUNC_GUARD`와 같은 자리일
#   가능성이 높다. C207이 셋(ENVELOPE·TRUNC·UNAVAIL_PROMISE)을 *"폭주-디코드 방어"* 로 묶었는데,
#   이 술어로 보면 **앞 둘은 채널(하네스)·`UNAVAIL_PROMISE`만 도메인(레버)** 이다.
#   지금 옮기지 않는다 — 분류 변경은 측정에 영향이 없더라도 **한 번에 하나씩** 한다.
#
# 그래서 하네스는 arm 정의에서 **상수**다 — 끄면 능력이 조용히 사라지는 게 아니라
# **측정 자체가 무효가 된다**(2026-08-07 실측 3회: 호스트·채널·죽은 훅).
#
# ★이 술어를 실제로 적용하니 **오분류 1건**이 나왔다 — `T2_A2_VARIANT`를 뺀다.
#   그것은 `verify_identity`의 **record 슬롯을 A2에서 지운다** = 모델이 낼 수 있는 호출이 달라진다
#   = 위 술어의 두 번째 가지다. 004 실측이 그 증거다(슬롯이 있으면 모델이 record를 날조하고
#   우리 도구가 그것을 자기 자신과 대조해 VERIFIED를 발급했다). ⇒ **선언면 레버**로 옮긴다.
HARNESS = ["T2_GATE_REGEN", "T2_GATE_REGEN_K", "T2_PROV_REGEN_K", "T2_PROV_MODE",
           "T2_OVERFLOW_GUARD", "T2_TRUNC_GUARD", "T2_DYN_MT", "T2_MT_FLOOR",
           "T2_DYN_MT_MARGIN", "T2_MAXPROMPT", "T2_VIEW_COMPACT", "T2_VIEW_ANNOTATE",
           "T2_VIEW_COMPACT_MINTOTAL", "T2_VIEW_MSG_CAP",
           "T2_PAIRCHECK", "T2_PAIRFIX", "T2_FAILED_PERSIST",
           "T2_KB_DOCS_DIR", "T2_LLM_TIMEOUT", "T2_LLM_RETRIES", "T2_AGENT_MAX_TOKENS",
           "T2_REGEN_BUDGET"]

# ★선언면 레버 — **말하지 않고 구조로 막는다.** 문구를 붙이지 않으므로 `speak()`의 대상이 아니고,
#   그래서 층이 비어 있는 것이 결함이 아니다(작용면이 다르다).
DECLARATIVE = ["T2_A2_VARIANT"]

# 귀속 arm 전용 — 운영 스택에서는 켜지 않는다(기본 OFF가 정상)
ARM_ONLY = ["T2_EPLAN_WALK_HOLD"]

# ★코드에서 **기본값이 ON**이라 `go_stack.sh`에 없어도 라이브인 것 (2026-08-07 배선이 드러냄)
#   라이브 판정을 go_stack 파싱으로만 하면 이들이 **보이지 않는다** — 감사의 사각이었다.
#   `T2_NOTICE_REPEAT`가 그 실물이다: `os.environ.get("T2_NOTICE_REPEAT", "1") == "1"` (`:7655`).
DEFAULT_ON = ["T2_NOTICE_REPEAT"]

# 구현은 있으나 `go_stack.sh`에 없어 **라이브가 아닌** 레버(死배선과 구분: 이쪽은 의도적 미등재)
NOT_LAUNCHED = ["T2_DISCOVERY_REQUIRED", "T2_SELF_DECLARATION"]

# [[57]] 위반(횟수로 억제) — 정체-과금·지문 억제가 대체함
#
# ★2026-08-07 정정 — `T2_UNKNOWN_REPEAT_GUARD`를 여기서 뺀다(내 오분류였다).
#   핸드오프·메모리가 *"[[57]] 위반(횟수로 억제)"* 로 적어 놓아 그대로 옮겼는데,
#   **호출부를 읽으니 술어가 이미 인자-변화 기준이다**(`:7167`·`:7190`):
#     · unkrepeat = *env가 `Unknown discoverable tool`로 반려한 **그 이름**이 응답에 다시 등장하는가*
#     · argrepeat = *env가 `Unexpected parameter`로 반려한 **그 인자**가 호출에 다시 실렸는가*
#   둘 다 **무엇이 바뀌었는가**를 보지 *몇 번 했는가*를 보지 않는다 = [[57]]이 요구하는 형태 그 자체다.
#   `cap 2`는 이 레버의 성질이 아니라 **보편 규약**이다(PROCEDURE_CAP·TRANSCRIBE_CAP과 동렬).
#   ⇒ 폐기가 아니라 **재배치**한다. 근거는 010/014/015/016 **[S]** 이고, 지우면 [[60]]이 경고한
#     *"끄면 능력이 조용히 사라진다"* 를 그대로 재현했을 것이다([[55]] 진단 순서: 코드부터 읽는다).
#
# ⚠**남은 진짜 위험은 다른 것이다([[25]])**: 이 술어의 출처가 **env의 주장**이다.
#   [[25]]는 *"env가 '없다'고 해도 레지스트리에 있으면 있는 것"* 이라고 못박았다.
#   env가 실재 도구를 `Unknown`으로 반려하면 이 레버는 **정답을 막는다**.
#   ⇒ 고칠 지점은 삭제가 아니라 **레지스트리 재검증을 술어에 추가**하는 것이다(미구현·부채로 기록).
RETIRED = ["T2_REPEAT_CAP", "T2_DD_FB"]


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
    if flag in DECLARATIVE:
        return "(선언면)"
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
    live = set(live_flags) | set(DEFAULT_ON)   # ★기본값 ON은 go_stack에 없어도 라이브다
    out = []
    for name, (_c, _m, _p, flags) in CELLS.items():
        for f in flags:
            if f not in live:
                tag = "미착수" if f in NOT_LAUNCHED else "死배선"
                out.append((name, f, tag))
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
    for cell, f, tag in dead:
        print("    %-8s %-18s %s" % (tag, cell, f))

    # ── 코드 체계 ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("[코드 — 이름은 바뀌어도 코드는 안 바뀐다]")
    print("\n  관측 족 (OB)")
    for c in sorted(FAMILIES, key=lambda k: (len(k), k)):
        dfs = [k for k, v in CAUSE_CODES.items() if v[1] == c]
        print("    %-5s %-18s ← %s" % (c, FAMILIES[c], " ".join(sorted(dfs, key=lambda x: int(x[2:])))))
    print("\n  결손 (DF) = 원인 코드")
    for c in sorted(CAUSE_CODES, key=lambda k: int(k[2:])):
        nm, fam = CAUSE_CODES[c]
        print("    %-5s %-22s [%s] %s" % (c, nm, CAUSES.get(nm, ("?",))[0], fam))
    print("\n  방법 (MT) = 해결책 코드")
    for c in sorted(METHOD_CODES, key=lambda k: int(k[2:])):
        print("    %-5s %s" % (c, METHOD_CODES[c]))
    print("\n  층 (LY) = 실시 순서  ⚠방법과 1:1 아님 — LY2·LY3 둘 다 deny를 쓴다")
    for c in sorted(LAYER_CODES, key=lambda k: int(k[2:])):
        nm, mt = LAYER_CODES[c]
        print("    %-5s %-16s %s" % (c, nm, mt or "(판정 안 함)"))

    print("\n[레버 → 코드]  fixed = 세부 기전 확정 · inherited = 셀 상속(미확정·speak 불가)")
    fixed_n = inh_n = 0
    rows = []
    for cname, (_c, _m, _p, flags) in CELLS.items():
        for f in sorted(flags):
            df, mts, how = lever_codes(f)
            rows.append((df or "??", "+".join(mts) or "??", how or "none", f, cname))
            if how == "fixed":
                fixed_n += 1
            elif how == "inherited":
                inh_n += 1
    for df, mt, how, f, cname in sorted(rows, key=lambda r: (int(r[0][2:]) if r[0][2:].isdigit() else 99, r[3])):
        print("    %-5s %-9s %-10s %-28s %s" % (df, mt, how, f.replace("T2_", ""), cname))
    print("\n    확정 %d · 상속 %d  (상속 = 다음 작업 목록)" % (fixed_n, inh_n))

    bad = audit_codes()
    print("\n코드 규칙 위반 %d건%s" % (len(bad), ":" if bad else " ✓"))
    for what, detail in bad:
        print("    %-18s %s" % (what, detail))
    print("\n[코드 관리 규칙]")
    for r in CODE_RULES:
        print("  " + r)
