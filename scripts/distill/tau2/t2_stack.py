# -*- coding: utf-8 -*-
"""**단일 분기 도구** — 층을 이루어 분기하고, 한 턴에 한 명령만 낸다.

사용자 지시 2026-08-07: *"단순 표를 만드는 게 목적이 아니라, 지금까지 실험한 모든 레버들을
통합해서 **기전과 해결책으로 재분류**하고 **충돌나지 않게 하나의 도구로 층을 이루어 분기**하게
해야 한다."* · *"이때까지 각 태스크의 레버들을 만들면서 **기전 분석한 것을 세부 분류**하여야 한다."*

## 왜 이 파일인가

레버 97개는 **각자 자기 자리에 국소적으로 붙었다**. 그래서 같은 턴에 둘이 말하고(W1 claim_prov ×
EPLAN drive), 한쪽이 다른 쪽이 금지한 도구를 권하고(022), auth 미충족 구간에서 행동-유도가 발화한다(050).
충돌 census T1~T6 12건의 뿌리는 개수가 아니라 **누가 먼저 말하는지를 아무도 안 정했다는 것**이다.

이 모듈은 그 순서를 **자료구조로** 만든다. 세 가지만 한다:

  1. `MECHANISMS` — 각 레버를 만들 때 기록한 **태스크별 기전 분석**을 세부 기전 단위로 세운다.
     (출처 = `go_stack.sh` 레버 주석 · 각 설계서 · 원장 C번호. 새로 지어낸 항목은 없다.)
  2. `LAYERS` — 층과 순서. 위층이 말하면 아래층은 **침묵이 아니라 치환**된다.
  3. `route()` — 후보 발화들을 받아 **한 턴 한 명령**으로 접는다.

## ★단일 진입점 = 이미 있는 넷을 한 줄로 꿴다 (새 출구를 만들지 않는다)

사용자 지시 2026-08-07: *"지금까지 만들어진 모든 레버를 통합하라. 충돌나지 않게 **진입점을 통일**하고
**순서대로 실시**하게 하라."*

경쟁 출구를 새로 만드는 것이 레버를 97개로 만든 바로 그 실수다. 필요한 조각은 **이미 넷 다 있고,
각자 문제의 한 조각씩만 풀고 있었다**:

| 모듈 | 이미 푸는 것 | 못 푸는 것 |
|---|---|---|
| `t2_surface_bus` | 부착의 **단일 출구** + 불변식 4종(replay·정직·예산·채널) | 순서가 **채널 4종**이라 층·결손을 모른다 |
| `t2_arbitrate` | 등급 E1..E5 · **합병**(명령 하나·사실 합집합) · 억제 자격 | 언제 부르는지 아무도 안 정함 |
| `t2_window` | **언제 말하는가**(resign ∪ acting ∪ instructing) | 누가 말하는지는 모름 |
| `t2_stack`(여기) | 층 순서 · 세부 기전 귀속 | 부착·합병·창은 위 셋이 한다 |

⇒ `speak()`가 **넷을 순서대로 실시**한다. 그리고 셋 다 **라이브가 아니다** — `go_stack.sh`에
`T2_SURFACE_BUS`·`T2_ARBITRATE`·`T2_WINDOW` 어느 것도 없다. 즉 **단일-출구 기구는 만들어져 있고
한 번도 라이브에서 돈 적이 없다**([[24]] 死코드 패턴). 통합은 그 배선까지다.

## 무엇이 아닌가 (정직)

**이 파일은 검사를 구현하지 않는다.** 각 레버의 술어는 여전히 자기 모듈에 있다
(`t2_procedure` `t2_transcribe` `t2_source` …). 이 모듈이 정하는 것은 *무엇을 검사하느냐*가 아니라
**누가 언제 말하느냐**다. 그리고 **97개 호출 지점을 `register()`로 바꾸는 편집은 아직 안 했다** —
`speak()`는 오프라인 자기검사로만 증명돼 있다. 이름을 붙였다고 구현이 된 게 아니다([[03b]]).
"""

import os

__all__ = ["LAYERS", "MECHANISMS", "layer_of", "route", "conflicts",
           "Stack", "get_stack", "register", "speak"]


# ── 층 (순서 = 권위. 위가 말하면 아래는 치환된다) ────────────────────────────
#
# 순서의 근거는 취향이 아니라 **관측된 충돌**이다:
#   · `WITHDRAWN_ROW`는 *"F5(전사 대조) 위에서만 건전"* — 오염된 입력으로 확정된 행을 지키면
#     거짓을 지킨다. ⇒ 출처 근거가 **모든 판정보다 먼저**다.
#   · `PHASE_OWNER`(050) = auth 게이트 미충족 구간에선 행동-유도 레버가 침묵.
#   · `SPEAK_PROHIBIT`(022) = push 레버가 돌고 있는 절차가 금지한 도구를 권하지 않는다.
#     ⇒ **차단이 표면화·유도보다 위**다. 이 둘은 이미 그 규칙을 국소 패치로 구현한 것이고,
#     여기서는 층으로 일반화한다(패치 2개 → 규칙 1개).
#   · C3 중재 등급 E1 실행 원장 > E2 정책 축자 > E3 env 출력 > E4 회수 산문 > E5 모델·손님.
#
LAYERS = [
    ("하네스", None,
     "판정하지 않는다. 자원 상한·설치 확인·쌍 무결성. 끄면 능력이 조용히 사라지므로 arm 상수."),
    ("출처 근거 확보", "verify",
     "이 턴의 사실 주장이 DB(실행 원장) 또는 정책(회수 문서)에 있는가. "
     "**주체를 가리지 않는다** — 모델·env 출력·손님·우리 층 넷 다 같은 검정을 받는다. "
     "여기서 걸리면 아래 층은 **오염된 입력 위에서 판정하게 되므로 돌지 않는다**."),
    ("차단", "deny",
     "정책이 선언한 조건이 미충족이면 그 호출을 열지 않는다. safety 성분만 — "
     "liveness('결국 호출해야 한다')는 원리적으로 집행 불가라 사전조건으로 환원한다."),
    ("선행", "deny/replace/pin",
     "표적의 미충족 조상이 있으면 그 요건이 먼저 말한다. 명령은 **지금 실행 가능한 걸음**. "
     "거절만으로 집행 불가인 자리에 pin(=insert)을 쓴다."),
    ("계산 이관", "compute",
     "구조화된 값 위의 산수는 엔진. 전사·해석은 모델. 엔진은 도메인 텍스트를 읽지 않는다([[59]])."),
    ("표면화", "surface",
     "출처집합 안에 있는데 아직 안 쓴 것을 그 자리에 보인다. 막지 않는다. "
     "⚠비강제 신호는 무시된다(0/40) — 필요조건이지 충분조건이 아니다."),
    ("되묻기", "ask",
     "열린 술어 잔여는 권위자에게 넘긴다. 추측으로 좁히지 않는다."),
]
LAYER_ORDER = {name: i for i, (name, _m, _d) in enumerate(LAYERS)}


# ── 세부 기전 — 레버를 만들 때 기록한 태스크별 분석 ──────────────────────────
#
# 형식: (결손, 세부 기전, 표적 태스크·실측, 레버 플래그, 층)
# 규칙: **실측 없는 항목은 넣지 않는다.** 아래 '실측' 칸은 전부 go_stack.sh 주석·설계서 축자다.
#
MECHANISMS = [
    # ── ① 날조 = 미검증 단정·정박 치환 ──────────────────────────────────────
    ("미검증 단정·정박 치환", "조회 실패 후 record를 발명 → **우리 도구가 그것을 모델 자신의 provided와 대조해 VERIFIED 발급**(가짜 검증)",
     "004: DOB 01/15/1985·'123 Main St' 날조 · record 날조 46%·grounded 0/24",
     # ★층 없음 = **말하지 않는 레버**. 이건 문구를 붙이는 게 아니라 `verify_identity`의 record
     #   슬롯을 A2에서 지워 그 경로를 **구조적으로 불가능**하게 만든다. `speak()`의 대상이 아니고,
     #   그래서 층이 비어 있는 것이 결함이 아니다 — 작용면이 다르다(선언면).
     "T2_A2_VARIANT", None),
    ("미검증 단정·정박 치환", "존재하지 않는 도구 이름을 손님에게 건넴",
     "012 `navigate_to_travel_notification` · x88: give 342 중 집합 밖 252(12 sim·통과 0) · **gold 요구 give 41 중 집합 밖 0**",
     "T2_DISPATCH_ROLE_ENVSET", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "미검증 operand로 계산 도구가 **참된 수를 참되게** 반환 → 가짜 정밀도",
     "check_rebate/apy/interest/closure 4종 · 미검증은 드롭→abstain",
     "T2_SG_GROUND", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "인용 span이 원문에 실재하지 않음",
     "010 재현 2/2 · 생성-레벨·재질의 1회 fail-open",
     "T2_GIVE_QUOTE", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "주장한 **행동**이 실행 원장에 없음",
     "035 기전 · 사임/transfer 창",
     "T2_CLAIM_PROV", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "검색이 전-0점인데 절차·주장을 지어냄",
     "012 절차 날조 · 014/015 주장 미뒷받침 · K=2",
     "T2_KB_NOHIT_SURFACE", "표면화"),
    ("미검증 단정·정박 치환", "미보유 기능을 약속(집합 대조로 차단)",
     "day4b ctxover 20건 방어 3종 중 하나",
     "T2_UNAVAIL_PROMISE", "차단"),
    ("미검증 단정·정박 치환", "열린-술어 가드를 종류별 닫힌-검사로 교체(핀 종류 라우팅)",
     "C282 라이브: 022가 5런 전패 코어에서 PASS · discrepant 10/10·`77 of 77`·드롭 0 · OFF였던 대가=022 t0/t1 2 sim 소각",
     "T2_QUOTE_PIN", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "인용 반려 시 **값이 원장에 실재할 때만** 표기를 지목",
     "046 t0/t1 자연실험",
     "T2_QUOTE_HINT", "표면화"),

    # ── ② 오선택 = 의미 소속 판정 불가 ──────────────────────────────────────
    ("의미 소속 판정 불가", "열린-문자열 선택(계좌 클래스)이 접지 없이 확정",
     "x84: 이득 7 · **gold 미접지 3이라 deny 금지**(넛지만)",
     "T2_CHOICE_GROUND", "표면화"),
    ("의미 소속 판정 불가", "참조로 넘길 것을 값으로 복사(equijoin 손실)",
     "day6 W-f 실측=GO 충족 · +F7b A2 equijoin",
     "T2_SG_BYREF", "계산 이관"),
    ("의미 소속 판정 불가", "회수가 절단됐다는 사실을 모른 채 후보를 좁힘",
     "x75 재계량: 인증 126/202=62% · P3가 만든 0점 검색 410→1024(+150%)의 짝 처방",
     "T2_MATCH_COUNT", "표면화"),
    ("의미 소속 판정 불가", "★**범주 소속 판정**(트럭=operations인가) — 변이 불변이 아닌 열린 술어",
     "격리 천장 ~.44(scale·budget·CoT·RL 전부) · 024/t0 · frontier도 동일",
     None, "되묻기"),  # ← 레버 없음 = 경계

    # ── ③ⓐ over-action(행동) = 조건 재소환 실패 ─────────────────────────────
    ("조건 재소환 실패", "정책이 **순서를 명령**하거나 **도구를 금지**한 흐름에 진입한 뒤 위반",
     "x80 전수 194 sim 과차단 **0건** · 진입 개념 없이 노드명으로만 판정한 1차 설계는 28건 오차단→폐기",
     "T2_PROCEDURE", "차단"),
    # ★2026-08-07 기전 변경(사용자 지시 *"DAG로 정의된 선행행동 순서에 따라 엔진이 동작해야 한다"*):
    #   침묵 → **치환**. 같은 auth 선언을 읽고 정반대 행동을 내던 두 기제를 DAG 하나로 모은다.
    ("사슬 역행 실패",
     "auth 조상이 미충족인 구간에서 행동-유도를 **지우기만 하고 아무도 대신 말하지 않았다** — "
     "이제 그 자리에 DAG가 낸 미충족 조상 요건을 놓는다(`requirements_for` → `merged_text`)",
     "20260807b 실측: 침묵 6회 · 우리 층 발화는 claimprov 4건뿐 · 그 게이트 `applies_to`의 read 호출 0 "
     "⇒ 자기-강화 교착(조상 미충족→침묵→명령 없음→계속 미충족). "
     "옛 계량(실패 런 9회 침묵·통과 런 0회)은 **무해**를 보였지 유익을 보이지 않았다",
     "T2_PHASE_OWNER", "선행"),
    ("조건 재소환 실패", "선행 미충족으로 **거절한 표적을 아무도 다시 꺼내지 않는다** — 모델이 그 "
     "행동을 하려던 순간은 한 번뿐이고 거절과 함께 소멸한다. 선행이 풀리면 그 표적을 "
     "**모델 자신이 시도했던 호출 형태 그대로** 한 번 다시 내민다(층 3이 `deny`만이고 "
     "`pin` 절반이 비어 있던 자리)",
     "led_j task_100 t0·t1: gold 상태변경 3개 중 **디스패처 계좌조회 하나만 누락**(더함 0). "
     "턴6에 모델이 그 호출을 시도했고 검증-선행으로 거절된 뒤 돌아오지 않았다. "
     "런-간 대조: `T2_DISCOVERY_NAMES` i=5회→호출 ✓ / j=0회→호출 ✗ (원장 C300)",
     "T2_DEFERRED", "선행"),
    ("조건 재소환 실패", "push 레버가 **돌고 있는 절차가 금지한 도구**를 권함",
     "E3-② · 022 · x104 §C: 표적 3발 침묵·over-block 0을 오프라인 전수로 사전 확정",
     "T2_SPEAK_PROHIBIT", "차단"),
    ("조건 재소환 실패", "봉투 파싱 실패로 정지에 실패(폭주-디코드)",
     "C207 day4b ctxover 20건",
     "T2_ENVELOPE_GUARD", "차단"),
    ("조건 재소환 실패", "gold 밖 write를 여는 일반 도구 게이트",
     "—", "T2_TOOLGATE", "차단"),

    # ── ③ⓑ over-action(판정) = 권한 월권 ── ★레버 없음 = 최우선 구현 후보 ──
    ("권한 월권", "마지막 안내가 적격 집합보다 **적게** 이름을 대어, 손님이 결정할 때 정답이 눈앞에 없음",
     "카드 실패 14 중 **12** · 반증 003/t0: 모른 채 열거만 하고 손님이 골라 PASS",
     None, "표면화"),
    ("권한 월권", "우리 필터가 **미검증 전제**(모델이 준 `invited=false`)로 gold를 배제",
     "023/t0 · 전 fit 호출: invited null 7 / false 18 / true 0 = 모름을 말할 수단이 있었다",
     None, "표면화"),
    ("권한 월권", "★**우리 필터가 결측 필드를 수치 제약에 통과**시켜 거짓을 사실상 단언",
     "C184: `min_cashback=5`에 cashback 필드 없는 카드가 eligible → 에이전트 'Silver 5%'(정본 4.0/1.0) → 손님 복창",
     None, "출처 근거 확보"),

    # ── ④ under-action = 완료 무검증 ────────────────────────────────────────
    ("완료 무검증", "[coverage] 미판정 행 재호출 지시를 무시하고 사임",
     "019/022/027 [S] · 1회 regen",
     "T2_COVERAGE_FOLLOWUP", "차단"),
    ("완료 무검증", "제출이 완결됐는지 표면화되지 않은 채 종료",
     "020/027 · 터미널 훅 · deny 아님 · 1회/sim",
     "T2_DISPATCH_LEDGER", "표면화"),
    ("완료 무검증", "엔진이 확정한 행을 **손님 산문에 설득당해 철회**",
     "019 t1 · x94 · ⚠**전사 대조 위에서만 건전**(1차 gold 반례 2건이 전부 오염 입력)",
     "T2_WITHDRAWN_ROW", "표면화"),
    ("완료 무검증", "abstain하면서 **결핍 필드를 지목하지 않음**",
     "P4 · coverage에 필드명+공급 지시",
     "T2_ABSTAIN_FIELDS", "표면화"),
    ("완료 무검증", "후속 필수 단계 미이행",
     "FOLLOWUP 3종 · cap 3 · 진행 있으면 환급",
     "T2_FOLLOWUP_REQUIRED", "차단"),

    # ── ④ under-action = 발화-행동 등가 오인 ───────────────────────────────
    ("발화-행동 등가 오인", "notice를 공표하고 동의까지 받고도 **호출하지 않고 종료**",
     "P3 · 1턴 유예·required · A4 유저 ###TRANSFER### 직접-방출 시 면제(008 [S])",
     "T2_TERM_GRANT", "차단"),
    ("발화-행동 등가 오인", "프로토콜 문서를 **안 읽고** 이관",
     "x93 전수: 미열람 사용 27건 · **gold이 요구한 이관인데 미열람 6건** ⇒ deny하면 정답을 막는다 ⇒ 표면화만",
     "T2_REQUIRE_DOC", "표면화"),
    ("발화-행동 등가 오인", "이관 사유가 정책 티어와 불일치",
     "004 실측 · 정책 doc_042 티어표 · A2 구동",
     "T2_TRANSFER_TIER", "표면화"),
    ("발화-행동 등가 오인", "give 서명 불일치 — ⚠**구 strip은 엔진이 대신 고쳐 로그 위반을 0으로 만들었다**(은폐)",
     "V7: strip 2 / V7 0 · 이제 모델이 고친다",
     "T2_TOOL_SIGNATURE", "차단"),
    ("발화-행동 등가 오인", "채널 오분류(출력-부착 금지·예방형 생성-레벨)",
     "041 사고",
     "T2_TOOL_CHANNEL", "차단"),
    ("발화-행동 등가 오인", "실행 주체를 레지스트리에서 도출하지 않고 말로 대체",
     "DISPATCH_ROLE 계열",
     "T2_DISPATCH_ROLE", "차단"),

    # ── ④ under-action = 부재 미종결 ───────────────────────────────────────
    ("부재 미종결", "중복 검색 소진 후에도 전략 전환 없이 반복",
     "012/033/032 [S] · TH=2 · 날조 금지 병기",
     "T2_SEARCH_EXHAUST_NUDGE", "표면화"),
    ("부재 미종결", "절차에 **들어와 놓고** K턴 동안 그쪽으로 아무 호출도 없음",
     "x86 전수 194 sim·K=3: 발화 54회/29 sim · ▶유일 98.1% · **gold-밖 지목의 write 0** · 지목 도구의 **100%가 미-unlock**(048 livelock서 모델에게 없던 유일한 정보)",
     "T2_PROC_ABSENT", "표면화"),
    ("부재 미종결", "미측정 윈도를 **부정으로 오판정**",
     "023 · A2 선언 도구만",
     "T2_SG_WINDOW_ABSTAIN", "차단"),
    # ★2026-08-07 정정: 폐기 아님. 술어가 이미 **인자-변화 기준**이다([[57]] 준수·`t2_levers` RETIRED 주석).
    ("미검증 단정·정박 치환",
     "env가 **반려한 그 이름/인자**를 다시 쓴다 — `Unknown discoverable tool` 이름을 손님에게 재지시하거나, "
     "`Unexpected parameter` 인자를 give 호출에 다시 싣는다",
     "C212/B3: 010/014/015/016 **[S]**(010/014는 에러 후 2~3회 반복) · C212/A3: 018 [S] · "
     "⚠[[25]] 위험: 술어의 출처가 **env의 주장**이라 실재 도구를 `Unknown`으로 반려하면 정답을 막는다(레지스트리 재검증 미구현=부채)",
     "T2_UNKNOWN_REPEAT_GUARD", "차단"),

    # ── ④ under-action = 유도 실패 (dual-control) ──────────────────────────
    ("유도 실패", "손님에게 도구 실행을 안내했는데 **전달 이력 0**(실행 불가 지시)",
     "012 · x82 전수: 발화 43 sim(1회/sim) · 그중 17은 나중에 실제 전달(넛지가 이르지만 문구가 '먼저 전달하라'라 무해)",
     "T2_UNINSTRUCTABLE", "차단"),
    ("유도 실패", "user-tool 안내가 비표준이라 손님이 실행 못 함",
     "018/040 · 생성-레벨 · sim당 1회",
     "T2_USER_TOOL_NOTE", "표면화"),
    ("유도 실패", "give는 성사됐는데 손님이 실행하지 않음",
     "019 [S]",
     "T2_GIVE_EXEC_NUDGE", "표면화"),
    ("유도 실패", "원장에 등장하지 않는 give(DB 오염 유발)",
     "021 [S] · **강제 금지**·cap1",
     "T2_GIVE_RELEVANCE_NUDGE", "표면화"),
    ("유도 실패", "필수 인자의 **생산자**에 도달하지 못한 채 오도구로 전환",
     "040/041",
     "T2_ARG_PRODUCERS", "선행"),

    # ── ⑤ 선행 미충족 = 사슬 역행 실패 ──────────────────────────────────────
    ("사슬 역행 실패", "선행 read를 named tool_choice + **단일값 enum**으로 고정(=Ligatti insert)",
     "x72 3/3 · 사전계측 18/18 gold · **048 5/24 → 18/24** · write는 제외",
     "T2_PIN_READ_STEPS", "선행"),
    ("사슬 역행 실패", "행동-키 검색 게이트(행동 전 관련 문서 확보)",
     "C165",
     "T2_PREKB", "선행"),
    ("사슬 역행 실패", "통지에 **접미사 포함 호출형**을 동봉하지 않아 호출로 이어지지 않음",
     "[READ-FIRST] 44발화/18 sim · **051이 이것만 없어 이관**",
     "T2_CALLABLE_HINT", "선행"),
    ("사슬 역행 실패", "계획과 실행이 분리되지 않아 요건이 늘 '나중' 칸",
     "PLAN_PROBE t99: 격리 계획선 2주문 다 정답·실제 런선 1주문 누락+날조 · 101 원장조회 2/20",
     "T2_EPLAN", "선행"),
    ("사슬 역행 실패", "분기 후 재접지 실패 → close 차단 인과",
     "C146 make-or-break GO · C149 [S]",
     "T2_BRANCH_REGROUND", "선행"),
    ("사슬 역행 실패", "조건이 unverified인 채 확정으로 진행",
     "003 [S] · 비강제",
     "T2_UNVERIFIED_FOLLOWUP", "표면화"),

    # ── ⑤ 미사용 표면화 (같은 결손·다른 방법 ⇒ 자기 진입점) ─────────────────
    ("사슬 역행 실패", "**이미 회수한 문서가 이름을 말한** 미호출 도구를 발견 문구에 병기",
     "C11b(032) · 아무도 안 부른 gold 도구 **23건 중 12건**이 그 집합",
     "T2_DISCOVERY_NAMES", "표면화"),
    ("사슬 역행 실패", "해제해 놓고 부르지 않은 도구를 사임 턴에 1회 표면화",
     "C12(053)",
     "T2_UNCALLED_UNLOCK", "표면화"),
    ("사슬 역행 실패", "판정은 실재하는데 결정 도구 미호출 → 판정을 인용하고 **선택은 남긴다**",
     "(2)",
     "T2_VERDICT_SURFACE", "표면화"),
    ("사슬 역행 실패", "이관 시도 순간에 **미완 절차 단계를 이름으로** 표면화",
     "C16(048)",
     "T2_TRANSFER_LEAVES_STEPS", "표면화"),

    # ── ⑥ 상태 오염 = 집계 미발화 ───────────────────────────────────────────
    ("집계 미발화", "동일 인자 계산도구를 반복 호출(문맥 소각) — evidence_from·fetch_formalize는 자동 제외",
     "C204/D7: 022 ctx초과 10회 · 003 5회 · 005형 정당 재호출 보호",
     "T2_SG_DEDUP", "계산 이관"),
    ("집계 미발화", "☠원장을 받고도 창_잔여·관계기간을 산출하지 않음 — **전사=모델·산수=엔진**",
     "101/102 전수: **19/22 trial 미언급** · x124 오프라인 결정론 풀이는 100/101 gold 도달",
     "T2_LEDGER", "계산 이관"),
    ("집계 미발화", "격리 서브콜로 계산 문맥을 분리(부하 제거)",
     "SG_ISOLATE/ISOFB/TRACE · ⚠서브콜·토큰 증가(20~60분/태스크 관측)",
     "T2_SG_ISOLATE", "계산 이관"),
    ("집계 미발화", "DUP-COMPUTE 스텁에 이전 결과 재제시(상한 2·shrink 시 생략)",
     "P8",
     "T2_DUP_REPRESENT", "표면화"),

    # ── ⑥ 상태 오염 = 전사 발산 ─────────────────────────────────────────────
    ("전사 발산", "행 배열 인자의 **손-전사 값이 그 대화가 읽은 원장과 어긋남**",
     "018 t0: rewards_earned 1113(원장 487) → 없는 불일치 → 여분 분쟁 1건 → db_match=False · x90 전수: 발화 3건/2 sim · **gold 자신이 걸린 횟수 0**",
     "T2_TRANSCRIBE", "출처 근거 확보"),
    ("전사 발산", "stale 상태가 잔류해 하류를 오염",
     "—", "T2_STALE_STRIP", "출처 근거 확보"),
    ("전사 발산", "중복 read가 문맥을 소각 — ⚠216줄을 감싼 것은 **코드 배치 사고**이지 이 레버의 성질이 아니다",
     "미이설 = 부채",
     "T2_READ_DEDUP", "출처 근거 확보"),

    # ══════════════════════════════════════════════════════════════════════
    # ★2026-08-07 2차 수확 — 미배정 34개 (커버리지 53/86 → 86/86)
    #   근거는 전부 `go_stack.sh` 주석 · `t2_gate_patch.py` 인라인 ★주석 · 원장 C번호에서 왔다.
    #   실측 주석이 없는 것은 **없다고 적는다**(지어내지 않는다).
    # ══════════════════════════════════════════════════════════════════════

    # ── DF1 미검증 단정·정박 치환 (12) ────────────────────────────────────
    ("미검증 단정·정박 치환", "출처 선언 4지선다 + provenance 검증 — 인자마다 *어디서 왔는가*를 유한 선택으로 강제",
     "E11 GO · C45: 32B 날조 **67% → 0%** · over-block 0 · Δspurious 0 (present 없이)",
     "T2_PROV_REGEN", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "사임 턴의 **완료-주장에 출처를 묻는다**(A2 `completion_guard.claim_question`)",
     "`_resign ∧ 미발화` 1회 · `t2_gate_patch:7596`",
     "T2_WRITE_PROV", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "write **인자값**이 출처에 있는가 — P9로 give 내포 인자까지 확장",
     "028/040 · WEV 블록과 동일 라운드·cap·배관 공유",
     "T2_WRITE_ARG_GROUND", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "증거 없는 update 차단 — ⚠**구 apply()에만 있어 unified 런에서 死코드였다**",
     "028 포렌식: deny **0회** · 증거 없는 update **6건 통과** ⇒ 생성-레벨로 이설(死배선 사고의 원형)",
     "T2_WRITE_EVIDENCE", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "P-A GROUND — 인자 접지 검사(T5-C rev3)",
     "go_stack 등재 근거만 · 태스크-단위 실측 주석 **없음**",
     "T2_GROUND", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "격리 서브콜 반환의 진위 검사 — 실패 시 **원 도구 실행으로 폴백**(거동 변화 0)",
     "`t2_gate_patch:1907`",
     "T2_SG_TRUTH", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "문맥에 없는 **id-operand를 쓴 write만** strip — read/procedural은 무해라 건드리지 않는다",
     "over-block 방지 설계가 술어에 박혀 있다 · 디스패처 nested unwrap 포함",
     "T2_FAB_STRIP", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "env가 *\"Unknown ... tool\"* 로 반려한 이름을 블랙리스트",
     "§2bt · rall11 050 실측",
     "T2_UNKNOWN_NAME_BL", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "A2 `discoverable_name_check` — **선언된 이름만** 통과",
     "§2bh · rall5 실측 · C186이 이것을 ①라우팅 표적으로 지목",
     "T2_UNLOCK_NAME", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "regen 경로의 **접미사-환각** 차단(해금 이름을 다시 지어내는 것)",
     "§2bt · rall11 050 실측",
     "T2_UNLOCK_PROV", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "producer-binding — A2 `grounded_params`로 **날조를 결핍으로 강등**(값을 만들지 않는다)",
     "P4b",
     "T2_PROD_BIND", "출처 근거 확보"),
    ("미검증 단정·정박 치환", "☠C1 출처 계약 — 유형 9종(값·행·이름·행동·사유·문서·선택·인용·생산자)을 한 술어로",
     "**미구현**(`t2_source.py`는 있으나 `go_stack.sh`에 없다) · 7키가 같은 문장을 7번 다시 쓴 것이 표적",
     "T2_SOURCE", "출처 근거 확보"),

    # ── DF2 의미 소속 판정 불가 (2) ───────────────────────────────────────
    ("의미 소속 판정 불가", "선언된 write가 **가리키는 레코드**를 결정론으로 검증(도구/필드/문구=A2)",
     "C128/C129 · C186이 ⑨값 표적으로 지목",
     "T2_REF_VERIFY", "출처 근거 확보"),
    ("의미 소속 판정 불가", "스키마 **밖 최상위 키** 위생 — 근거는 자기 도구 스키마(`properties`)뿐·값 판단 0",
     "P11 · unified 이설 완료 · **표적 0 = 위생**(효과 주장 없음)",
     "T2_ARG_SCHEMA", "출처 근거 확보"),

    # ── DF5 완료 무검증 (2) ───────────────────────────────────────────────
    ("완료 무검증", "재발행 시 **채널을 강제**(`tool_choice=required`) — 말로 하면 56%로 악화",
     "forced 프로브: 강제 하 **24/24** 정답 선택 vs 말로 지시 56%(단일변수·`forced_probe_20260718`)",
     "T2_FOLLOWUP_FORCE", "차단"),
    ("완료 무검증", "제출 후 **KB-검색류로 새는 것**을 막고 남은 단계로 되돌린다",
     "§2bk · rall7 050 실측",
     "T2_FOLLOWUP_READLOOP", "차단"),

    # ── DF6 발화-행동 등가 오인 (6) ───────────────────────────────────────
    ("발화-행동 등가 오인", "손님이 `###TRANSFER###`를 **직접 방출**하면 notice 요건을 면제 — 과차단 해제 쪽 레버",
     "A4 · 008 [S]",
     "T2_TERM_GRANT_USERDEMAND", "차단"),
    ("발화-행동 등가 오인", "**도구목록 밖 이름** 호출 차단(발명명 포함)",
     "§2bb · r095g g-t0 실측",
     "T2_TOOLLIST", "차단"),
    ("발화-행동 등가 오인", "값을 **어디서 얻는지** 경로를 표면화(쫓을 곳을 알려준다)",
     "C119 · 8-task per-step 포렌식",
     "T2_VALUE_ACQUIRE", "표면화"),
    ("발화-행동 등가 오인", "`have-value → act` 일반레버 — 값을 이미 쥐었으면 행동으로 넘긴다(도구/인자/신호/문구=A2)",
     "C115",
     "T2_HAVE_VALUE", "선행"),
    ("발화-행동 등가 오인", "그 강제판 — ⚠병리적 runaway(039 퇴행루프)만 `_gen` 폴백으로 강등",
     "C115 · `t2_gate_patch:6332`",
     "T2_HAVE_VALUE_FORCE", "선행"),
    ("발화-행동 등가 오인", "pre-gate **순서** 체인 수정 — 합성 런이 드러낸 관통의 조정물",
     "C162 실증 · C166 체인수정 ([[19]] 합성-우선의 사례)",
     "T2_GUIDED", "선행"),

    # ── DF9 사슬 역행 실패 (7) ────────────────────────────────────────────
    ("사슬 역행 실패", "say-don't-do 감지 → **다음 재생성서 `tool_choice=required`**",
     "`t2_gate_patch:4578`",
     "T2_FORCE_ACTION", "선행"),
    ("사슬 역행 실패", "E-PLAN ledger + walk — 커밋 히스토리에서 결정론 ledger 재구성(관측만·[[10]])",
     "[[14]] · discovery L1/L2 = read-강제 deny(§1.5 허용축)",
     "T2_EPLAN_WALK", "선행"),
    ("사슬 역행 실패", "A2 `scaffold_get_tools` — **검증기 GET을 주입**한다(모델이 없는 도구를 찾아 헤매지 않게)",
     "go_stack 등재 근거 · `t2_run_gated.py:248`에서 체이닝",
     "T2_SCAFFOLD_GET", "선행"),
    ("사슬 역행 실패", "read-**선행** 게이트 — 계산 전에 필요한 레코드를 먼저 읽게 한다",
     "§2aw · r095 gather-순서 실측(계산 前 저축 레코드)",
     "T2_SG_REQREADS", "선행"),
    ("사슬 역행 실패", "선행 read를 named `tool_choice` + **단일값 enum**으로 1회 고정",
     "P1 · x72 **3/3** · replay 무관 · ⚠P1 단독 기대 pass 증가 = **0으로 사전등록**",
     "T2_PIN_READ", "선행"),
    ("사슬 역행 실패", "고정의 **재무장** — 첫 라이브라 1회만(기회비용 미측정이라 보수적으로)",
     "C15 보조",
     "T2_PROC_PIN_REARM", "선행"),
    ("사슬 역행 실패", "*\"종료 시 1회\"* → **갭이 열려 있는 동안 매 드리프트마다 견인**",
     "C118 · `EPLAN_MIDDRIVE_DESIGN §2.1`",
     "T2_COV_MIDDRIVE", "선행"),

    # ── DF10 집계 미발화 (5) ──────────────────────────────────────────────
    ("집계 미발화", "liability 계산 이관 — **에이전트 제공값만** 쓰고 미확정이면 개입하지 않는다",
     "§8-3: liability만 **순 +348** · provisional 드롭 **net −4**",
     "T2_COMPUTE", "계산 이관"),
    ("집계 미발화", "per-operand **해소 디스패처**(통일 인터프리터)",
     "`UNIFIED_OPERAND_A2 §7-3`",
     "T2_RESOLVE", "계산 이관"),
    ("집계 미발화", "처방을 A2 구동으로 **결정론 산출**하고 오선택을 deny — ⚠선택 축(DF2)도 건드린다",
     "§2bu · rall11 038 실측 · 격리 L2 **8/8 = 활성화 실패**",
     "T2_PRESCRIPTION", "계산 이관"),
    ("집계 미발화", "격리 서브 **피드백** — 반환은 거동 보존이고 메인 관문1이 재검증한다(심층방어)",
     "엔진=검증+반사만 · 값 생성=LLM([[03b]]/[[10]])",
     "T2_SG_ISOFB", "계산 이관"),
    ("집계 미발화", "격리 서브콜 **궤적 기록** — 서브에서 무슨 일이 있었는지 사후 감사 가능하게",
     "`t2_gate_patch:1659` · 관측 레버(판정 안 함)",
     "T2_SG_TRACE", "계산 이관"),

    # ── ★배선이 드러낸 미등록 레버 3종 (2026-08-07) ───────────────────────
    #   `_ap_regen` tag를 게이팅 플래그까지 되짚자 **레지스트리에 없는 레버**가 나왔다.
    #   레지스트리의 라이브 목록이 `go_stack.sh` 파싱이라서, **코드에서 기본 ON인 플래그가 안 보였다.**
    ("발화-행동 등가 오인", "같은 transfer notice를 **호출 없이** 다시 보낸다 — 반복이 요청을 정체시킨다",
     "★`T2_NOTICE_REPEAT` 기본값 `\"1\"` = **라이브인데 go_stack에 없어 감사에 안 잡혔다** (`:7655`)",
     "T2_NOTICE_REPEAT", "차단"),
    ("사슬 역행 실패", "레코드를 읽고도 **판정 도구를 안 부르고** 스스로 판정 — A2 `analysis_producers`",
     "`:7826` · go_stack에 없음 = **비-라이브**",
     "T2_DISCOVERY_REQUIRED", "차단"),
    ("미검증 단정·정박 치환", "답변의 근거를 **스스로 선언**하게 하고, `INFER`인데 그걸 내주는 도구가 있으면 되돌린다",
     "A2 `assertion_operands` · `:7844` · go_stack에 없음 = **비-라이브**",
     "T2_SELF_DECLARATION", "출처 근거 확보"),
]


def layer_of(flag):
    """이 플래그가 어느 층에서 말하는가. 모르면 None."""
    for _cause, _mech, _ev, f, layer in MECHANISMS:
        if f == flag:
            return layer
    return None


def route(candidates):
    """후보 발화들을 **한 턴 한 명령**으로 접는다.

    `candidates` = [{"flag":…, "target":…, "fact":…, "order":…}, …]
      · `target` = 이 발화가 겨냥한 호출/단계. **같은 표적은 합병한다**(명령 하나·사실 합집합).
      · `order`  = 그 레버가 내리려는 명령(없으면 사실만 = 표면화).

    반환 = {"speak": [...], "suppressed": [...]}
      · 최상위 층의 명령 하나만 `order`를 갖는다.
      · 진 쪽은 **침묵이 아니라 치환** — 사실은 살아서 같은 문장에 합쳐진다([[56]] C3).
    """
    ranked = []
    for c in candidates:
        layer = c.get("layer") or layer_of(c.get("flag"))
        if layer is None:          # 폐기·미분류는 말하지 않는다
            continue
        ranked.append((LAYER_ORDER.get(layer, 99), layer, c))
    if not ranked:
        return {"speak": [], "suppressed": []}
    ranked.sort(key=lambda t: t[0])

    top_layer = ranked[0][1]
    by_target = {}
    for _i, layer, c in ranked:
        by_target.setdefault(c.get("target"), []).append((layer, c))

    speak, suppressed = [], []
    for target, group in by_target.items():
        winner_layer, winner = group[0]
        facts = [c.get("fact") for _l, c in group if c.get("fact")]
        # 명령은 **최상위 층이 그 표적을 잡았을 때만** 나간다. 아니면 사실만 (=표면화).
        order = winner.get("order") if winner_layer == top_layer else None
        speak.append({"target": target, "layer": winner_layer,
                      "order": order, "facts": facts,
                      "flags": [c.get("flag") for _l, c in group]})
        suppressed.extend(c.get("flag") for _l, c in group[1:])
    return {"speak": speak, "suppressed": suppressed}


# ══════════════════════════════════════════════════════════════════════════════
#  ★단일 진입점 — 레버는 `register()`만, 출구는 `speak()` 하나
# ══════════════════════════════════════════════════════════════════════════════
#
#  순서대로 실시한다:
#    1) 창    `t2_window.opened`      — 지금 말할 자리인가 (아니면 전부 보류)
#    2) 층    `route`                 — 누가 명령하는가 (7층·최상위만 명령)
#    3) 자격  `t2_arbitrate.may_suppress` — 남을 지우려면 자격을 대라 (못 대면 표면화로 강등)
#    4) 합병  `t2_arbitrate.merge`    — 같은 표적은 한 문장 (명령 하나·사실 합집합)
#    5) 버스  `t2_surface_bus.flush`  — replay·정직·예산·채널 불변식
#
#  각 단계는 **떨어뜨리는 이유를 기록**한다(`trace`). 조용한 침묵이 死배선을 만들었기 때문이다.

_CHANNEL_OF_LAYER = {          # 층 → 버스 채널(버스의 ④순서와 정합)
    "출처 근거 확보": "correction",
    "차단": "deny_reason",
    "선행": "deny_reason",
    "계산 이관": "correction",
    "표면화": "guidance",
    "되묻기": "guidance",
    "하네스": "mark",
}


class Stack(object):
    """orchestrator(시뮬)당 1개. 레버가 등록하고, 발화 지점에서 `speak()` 한 번."""

    def __init__(self):
        self._pending = []
        self.trace = []

    def register(self, flag, target=None, fact=None, order=None,
                 requires=None, suppresses=False, layer=None):
        """레버 쪽 API — **문구를 부착하지 않는다. 등록만 한다.**

        flag       : `T2_*` (층 귀속·`enabled` 판정에 쓴다)
        target     : 이 발화가 겨냥한 호출/단계. **같은 표적끼리 합병된다**
        fact       : 사실 진술(항상 살아남는다 — 져도 치환될 뿐 지워지지 않는다)
        order      : 이 레버가 내리려는 명령(없으면 사실만 = 표면화)
        requires   : 버스의 ②정직 불변용 — 이 문구가 전제하는 상태 키
        suppresses : 다른 레버의 발화 조건을 지우려는가(자격 검사 대상)
        """
        self._pending.append({"flag": flag, "target": target, "fact": fact,
                              "order": order, "requires": dict(requires or {}),
                              "suppresses": bool(suppresses),
                              "layer": layer or layer_of(flag)})

    # ── 출구 ────────────────────────────────────────────────────────────────
    def speak(self, orch=None, am=None, a2=None, attach_ok=True, state=None,
              targets=(), name_of=None, window_required=True):
        """**모든 레버의 단일 출구.** 위 5단계를 순서대로 실시하고 부착할 문구를 돌려준다.

        부착 자체는 호출자가 한다(메시지 소유권은 호출자에게 있다 — 버스와 같은 규약).
        반환 = [text, ...] · 떨어진 이유는 `self.trace`.
        """
        pend, self._pending = self._pending, []
        self.trace = []

        # ── 1) 창 — 지금 말할 자리인가 ──────────────────────────────────────
        if window_required and am is not None:
            try:
                import t2_window as _w
                opened = _w.opened(am, targets=targets, name_of=name_of)
            except Exception as e:
                opened, self.trace = None, self.trace + [("창", "판정 불가(통과)", repr(e))]
            if opened is not None and not opened:
                self.trace.append(("창", "닫힘 — 전부 보류", "%d건" % len(pend)))
                return []
            if opened:
                self.trace.append(("창", "열림", ",".join(sorted(opened))))

        # ── 켜짐 판정 ([[60]] 기본 항상 켬) ────────────────────────────────
        live = []
        for c in pend:
            if c["layer"] is None:
                self.trace.append(("귀속", "층 없음 — 말하지 않음", c["flag"]))
                continue
            if not _cell_enabled(c["flag"]):
                self.trace.append(("셀", "꺼짐(귀속 arm)", c["flag"]))
                continue
            live.append(c)
        if not live:
            return []

        # ── 2) 층 — 누가 명령하는가 ────────────────────────────────────────
        routed = route(live)
        for f in routed["suppressed"]:
            self.trace.append(("층", "치환(사실은 살아남음)", f))

        # ── 3) 자격 — 남을 지우려면 자격을 대라 ────────────────────────────
        others = [c["target"] for c in live if c.get("target")]
        for c in live:
            if not c["suppresses"]:
                continue
            try:
                import t2_arbitrate as _arb
                ok, why = _arb.may_suppress(c["flag"], a2, others)
            except Exception as e:
                ok, why = False, "자격 판정 불가: %r" % (e,)
            if not ok:
                for s in routed["speak"]:
                    if s["target"] == c["target"] and c["flag"] in s["flags"]:
                        s["order"] = None          # ★차단 → 표면화로 강등
                self.trace.append(("자격", "억제 자격 없음 → 표면화 강등", "%s · %s" % (c["flag"], why)))

        # ── 4) 합병 — 같은 표적은 한 문장 ──────────────────────────────────
        texts = []
        for s in routed["speak"]:
            merged = None
            if a2 is not None and s.get("target"):
                try:
                    import t2_arbitrate as _arb
                    merged = _arb.merge(s.get("reqs") or [], a2, s["target"]) or None
                except Exception as e:
                    self.trace.append(("합병", "실패(사실 나열로 대체)", repr(e)))
            body = merged or " · ".join([f for f in s["facts"] if f])
            line = ("%s %s" % (s["order"], body)).strip() if s["order"] else body
            if line:
                texts.append((s["layer"], line, s))

        # ── 5) 버스 — replay·정직·예산·채널 ────────────────────────────────
        try:
            import t2_surface_bus as _sb
            bus = _sb.get_bus(orch) if orch is not None else _sb.SurfaceBus()
            for layer, line, s in texts:
                req = {}
                for c in live:
                    if c["flag"] in s["flags"]:
                        req.update(c["requires"])
                bus.register(_CHANNEL_OF_LAYER.get(layer, "mark"), line, requires=req)
            out = bus.flush(attach_ok, state=state)
        except Exception as e:                      # 버스 결함 = fail-open(단 deny는 버스가 fail-closed)
            self.trace.append(("버스", "실패(무부착 통과)", repr(e)))
            out = []
        self.trace.append(("출구", "부착", "%d건 / 후보 %d건" % (len(out), len(pend))))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  ★배선 — `_ap_regen`의 tag가 귀속 키다 (전수 등록)
# ══════════════════════════════════════════════════════════════════════════════
#
# 조사 결과 생성면의 발화는 **전부 `_ap_regen(fbtxt, tag, ...)` 하나를 지난다**(`t2_gate_patch:6886`·
# 호출 26곳). 즉 55개를 하나씩 고칠 필요가 없다 — **그 한 자리가 이미 공유 원시연산**이다.
# tag를 플래그로 되돌리면 `register()`가 공짜로 채워진다.
#
# ⚠**모르는 tag를 추측해서 매핑하지 않는다.** 미매핑은 `flag=None`으로 등록하고 감사에 드러낸다 —
# 잘못 귀속시키는 것이 미매핑보다 나쁘다(C294: 귀속이 판정의 1차 지표다).
TAG_TO_FLAG = {
    "envguard": "T2_ENVELOPE_GUARD",
    "truncguard": "T2_TRUNC_GUARD",            # 하네스
    "giverel": "T2_GIVE_RELEVANCE_NUDGE",
    "unkrepeat": "T2_UNKNOWN_REPEAT_GUARD",    # 폐기 — 층 없음이라 말하지 않는다
    "covfollowup": "T2_COVERAGE_FOLLOWUP",
    "unverifiedfu": "T2_UNVERIFIED_FOLLOWUP",
    "givexec": "T2_GIVE_EXEC_NUDGE",
    "verdict_surface": "T2_VERDICT_SURFACE",
    "searchexhaust": "T2_SEARCH_EXHAUST_NUDGE",
    "claimprov": "T2_CLAIM_PROV",
    "channel": "T2_TOOL_CHANNEL",
    "choiceground": "T2_CHOICE_GROUND",
    "uninstructable": "T2_UNINSTRUCTABLE",
    "usertoolnote": "T2_USER_TOOL_NOTE",
    "kbnohit": "T2_KB_NOHIT_SURFACE",
    "givequote": "T2_GIVE_QUOTE",
    "argschema": "T2_ARG_SCHEMA",
    "signature": "T2_TOOL_SIGNATURE",
    "followup": "T2_FOLLOWUP_REQUIRED",
    "writeprov": "T2_WRITE_PROV",
    "transfertier": "T2_TRANSFER_TIER",
    # ★2026-08-07 호출부 정독으로 해소(게이팅 플래그를 읽어 확정 — 추측 아님):
    "argrepeat": "T2_UNKNOWN_REPEAT_GUARD",    # :7190 게이팅 (폐기 아님 — 아래 ★정정 참조)
    "unkrepeat": "T2_UNKNOWN_REPEAT_GUARD",    # :7167 게이팅
    "noticerep": "T2_NOTICE_REPEAT",           # :7655 — ★기본값 "1" = 라이브인데 go_stack에 없다
    "discreq": "T2_DISCOVERY_REQUIRED",        # :7826 — go_stack에 없음 = 비-라이브
    "selfdecl": "T2_SELF_DECLARATION",         # :7844 — go_stack에 없음 = 비-라이브
    # ★변수 tag 2곳 해소(2026-08-07·정적 확정):
    "phase_precede": "T2_PHASE_OWNER",         # ★DAG-우선 치환(2026-08-07)·게이팅 :5786
    "uncalled_unlock": "T2_UNCALLED_UNLOCK",   # :7308 리터럴(긴 주석 뒤라 1차 추출이 놓쳤다)·게이팅 :7296
    "followup_chain": "T2_FOLLOWUP_REQUIRED",  # :7581 `_tag1` ∈ {chain, decision} — 발원 :2490/:2493
    "followup_decision": "T2_FOLLOWUP_REQUIRED",  # 게이팅 :7446
}

_OBS = {}          # sim 밖에서도 세는 전역 관측 카운터(진단용)


def observe(orch, tag, text=None, target=None, order=None):
    """★거동 중립 관찰자 — `_ap_regen`이 발화할 때마다 스택에 **등록만** 한다.

    지금은 아무것도 막지 않고 아무것도 합치지 않는다. 하는 일은 둘:
      · 이 발화가 **어느 레버·어느 층**이었는지 귀속을 남긴다
      · `route()`가 **어떻게 판정했을지**를 함께 남긴다 ⇒ 순서를 뒤집기 전에 순서를 검사할 수 있다

    왜 관찰자를 먼저 두나: 거동을 바꾸는 배선을 검사 없이 켜는 것이 이 코드베이스의 반복 사고였다
    (`T2_MATCH_COUNT` 의존물 누락 · `T2_WRITE_EVIDENCE` 死코드 · `T2_LEDGER` 무음 6회).
    사이드카가 같은 이유로 기본 ON이다 — **비커밋 관측은 거동 변화 0이고, 없으면 포렌식의 절반이 불가능하다**.
    """
    if os.environ.get("T2_STACK_OBSERVE", "1") != "1":
        return None
    flag = TAG_TO_FLAG.get(tag)
    layer = layer_of(flag) if flag else None
    key = (tag, flag or "?", layer or "-")
    _OBS[key] = _OBS.get(key, 0) + 1
    try:
        st = get_stack(orch) if orch is not None else None
        if st is not None:
            st.register(flag=flag or tag, target=target, fact=(text or "")[:200],
                        order=order, layer=layer)
    except Exception:
        pass
    try:
        from t2_lever_beat import beat as _beat
        _beat("T2_STACK", "%s|%s|%s" % (tag, flag or "UNMAPPED", layer or "NOLAYER"))
    except Exception:
        pass
    return layer


def admit(orch, tag, text):
    """★출구 게이트 — **같은 입력에 같은 말을 두 번 하지 않는다**([[57]] 발화 창).

    반환 `(ok, why)`. `ok=False`면 호출자는 **보내지 않는다**.

    2026-08-07 라이브가 이 규칙의 필요를 세 번 증명했다:
      · 내 DAG 치환이 캡 없이 9회 (국소 패치로 막았다 — 그게 문제다, 레버마다 다시 짜야 한다)
      · `[ORDER] '<도구>' cannot be carried out yet` **12회** (기존 레버·패치 없음)
      · 빈 문구 11회 (다른 경로로 1건은 아직 샌다)
    셋 다 *같은 병*이고, 레버 수만큼 국소 패치하는 대신 **모든 발화가 지나는 한 자리**에 건다.

    지문 = `(tag, 정규화한 문구 전체)`. **문구가 조금이라도 달라지면 통과한다** —
    억제 기준이 *횟수*가 아니라 *인자 변화*여야 하기 때문이다([[57]]). 도구·인자가 바뀌면
    문구가 바뀌므로 자동으로 다시 말한다.

    ⚠**끄기가 아니다.** 레버는 그대로 켜져 있고, 같은 말의 재발화만 접는다([[60]]).
    """
    if os.environ.get("T2_STACK_WINDOW", "1") != "1":
        return True, "window off"
    body = " ".join(str(text or "").split())
    if not body:
        return False, "empty"
    fp = (str(tag or ""), body)
    _o = _owner(orch)          # 창도 스택과 같은 객체에 산다 — 출구가 둘인데 창이 갈리면 안 접힌다
    seen = getattr(_o, "_t2_stack_said", None)
    if seen is None:
        seen = _o._t2_stack_said = set()
    # ★기억 크기를 이유에 실어 보낸다 (2026-08-07·led_j 라이브). 접힘이 4회 발화했는데 축자 동일한
    #   `[ORDER]`가 같은 sim 안에서 계속 나갔다. 후보는 둘이고 로그가 둘을 못 갈랐다:
    #   ⓐ 창을 안 거친 경로가 따로 있다 · ⓑ 창을 거쳤지만 **기억이 비어 있었다**(주인 객체가
    #   턴마다 새로 생기면 그렇게 된다). `seen=N`이 그 자리에서 둘을 가른다 — 매번 N=0이면 ⓑ다.
    n = len(seen)
    if fp in seen:
        return False, "same fingerprint (seen=%d)" % n
    seen.add(fp)
    return True, "new (seen=%d)" % n


def observed():
    """지금까지 관측된 (tag, flag, layer) → 횟수. 미매핑·무층이 그대로 보인다."""
    return dict(_OBS)


def cell_of_note(flag):
    """층이 없는 이유를 한 마디로 — 폐기인가, 하네스인가, 아직 안 쓴 것인가."""
    try:
        import t2_levers as _L
        c = _L.cell_of(flag)
        return c or "미분류"
    except Exception:
        return "?"


def _cell_enabled(flag):
    try:
        import t2_levers as _L
        cell = _L.cell_of(flag)
        return True if cell is None or cell.startswith("(") else _L.enabled(cell)
    except Exception:
        return True


def _owner(obj):
    """스택을 **한 객체**에 모은다 — 등록과 발화가 서로 다른 훅에서 일어나기 때문이다.

    생성면(`patched`)의 `self`는 에이전트이고 결과면(`exec_augment`)의 `self`는 오케스트레이터다.
    각자에게 스택을 달면 같은 턴의 후보가 두 곳으로 갈라져 `route()`가 절반만 보고 판정한다 —
    합병하려고 만든 기구가 정확히 합병에 실패한다. 오케스트레이터면 그 에이전트로 내려간다.
    """
    ag = getattr(obj, "agent", None)
    return ag if ag is not None and hasattr(ag, "llm") else obj


def get_stack(orch):
    """스택 1개(시뮬 수명 — 예산이 sim 단위)."""
    o = _owner(orch)
    s = getattr(o, "_t2_stack", None)
    if s is None:
        s = o._t2_stack = Stack()
    return s


def register(orch, **kw):
    """레버 호출 지점이 쓰는 한 줄. `stack.register(...)`의 축약."""
    return get_stack(orch).register(**kw)


def speak(orch, **kw):
    """발화 지점이 쓰는 한 줄."""
    return get_stack(orch).speak(orch=orch, **kw)


def audit(orch, chose=None):
    """★순서를 뒤집기 **전에** 순서를 검사한다 — 등록분을 비우고 `route()`의 판정만 남긴다.

    `speak()`를 실제 출구로 쓰는 것은 거동 변경이라 측정이 먼저다. 이 함수는 등록된 후보를
    드레인해(안 비우면 sim 내내 쌓인다) `route()`가 골랐을 층·표적을 돌려주고, `chose`를 주면
    **현행 체인이 실제로 고른 것**과 다른지까지 판정한다. 두 판정이 갈리는 자리가 곧 순서
    뒤집기가 값을 만드는 자리다 — 그게 없으면 뒤집을 이유도 없다.

    반환 = None(등록 0) 또는 {"pick": [...], "chose": …, "differs": bool}
    """
    st = get_stack(orch)
    pend, st._pending = st._pending, []
    if not pend:
        return None
    r = route(pend)
    picks = [(s.get("layer"), s.get("target"), (s.get("flags") or [None])[0])
             for s in r.get("speak", [])]
    differs = bool(chose) and all(chose != f for _l, _t, f in picks)
    return {"pick": picks, "chose": chose, "differs": differs,
            "suppressed": r.get("suppressed", [])}


def conflicts():
    """같은 표적을 **다른 층에서** 잡을 수 있는 레버 쌍 — 배선 전에 봐야 할 목록.

    같은 층이면 합병으로 끝나고, 층이 다르면 순서가 결과를 바꾼다.
    """
    by_cause = {}
    for cause, mech, _ev, flag, layer in MECHANISMS:
        if flag is None or layer is None:
            continue
        by_cause.setdefault(cause, []).append((layer, flag, mech))
    out = []
    for cause, items in by_cause.items():
        layers = {l for l, _f, _m in items}
        if len(layers) > 1:
            out.append((cause, sorted(items)))
    return out


if __name__ == "__main__":
    import io
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("층 %d개 · 세부 기전 %d개\n" % (len(LAYERS), len(MECHANISMS)))
    for i, (name, method, _d) in enumerate(LAYERS):
        n = sum(1 for _c, _m, _e, _f, l in MECHANISMS if l == name)
        print("  %d. %-14s %-16s %2d개" % (i, name, method or "(판정 안 함)", n))

    print("\n[결손 → 세부 기전 수 / 레버 있음 / 레버 없음]")
    seen = {}
    for cause, _m, _e, flag, _l in MECHANISMS:
        d = seen.setdefault(cause, [0, 0])
        d[0] += 1
        if flag is None:
            d[1] += 1
    for cause, (n, gap) in seen.items():
        mark = ("  ← 레버 없음 %d" % gap) if gap else ""
        print("  %-18s %2d%s" % (cause, n, mark))

    print("\n[한 결손이 여러 층에 걸친 자리 — 순서가 결과를 바꾼다]")
    for cause, items in conflicts():
        print("  %s" % cause)
        for layer, flag, mech in items:
            print("      %-14s %-28s %s" % (layer, flag, mech[:52]))

    print("\n[route() 자기검사 — 같은 표적을 세 층이 잡은 경우]")
    demo = route([
        {"flag": "T2_DISCOVERY_NAMES", "target": "transfer_to_human", "fact": "미호출 도구 2개", "order": "이 도구들을 먼저 보라"},
        {"flag": "T2_PROCEDURE", "target": "transfer_to_human", "fact": "절차가 금지", "order": "이 호출을 열지 않는다"},
        {"flag": "T2_TRANSCRIBE", "target": "transfer_to_human", "fact": "전사값 불일치", "order": "재발행하라"},
    ])
    for s in demo["speak"]:
        print("    표적 %s · 층=%s\n      명령: %s\n      사실: %s\n      합병된 레버: %s"
              % (s["target"], s["layer"], s["order"], s["facts"], s["flags"]))
    print("    치환된 레버: %s" % demo["suppressed"])
    print("\n  ⇒ 기대: 최상위 층(출처 근거 확보)만 명령하고, 차단·표면화의 **사실은 살아서 합쳐진다**.")
    print("     이것이 W1(claim_prov × EPLAN 이중 넛지)·050·022가 각각 국소 패치로 풀던 것의 일반화다.")

    # ── 단일 진입점 5단계 전수 자기검사 ───────────────────────────────────
    print("\n" + "=" * 72)
    print("[speak() 자기검사 — 창→층→자격→합병→버스 5단계를 순서대로]")

    class _Call(object):
        def __init__(self, name):
            self.name = name

    class _Msg(object):
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _Orch(object):
        pass

    def _case(title, am, targets, a2, attach_ok, cands, expect):
        orch = _Orch()
        st = get_stack(orch)
        for c in cands:
            st.register(**c)
        out = st.speak(orch=orch, am=am, a2=a2, attach_ok=attach_ok,
                       state={"ledger_read": True}, targets=targets)
        print("\n  ── %s" % title)
        for stage, verdict, detail in st.trace:
            print("     %-6s %-26s %s" % (stage, verdict, str(detail)[:58]))
        for t in out:
            print("     → 부착: %s" % t[:100])
        ok = (len(out) == expect)
        print("     %s 기대 %d건 / 실제 %d건" % ("PASS" if ok else "**FAIL**", expect, len(out)))
        return ok

    three = [
        {"flag": "T2_DISCOVERY_NAMES", "target": "transfer_to_human_agents",
         "fact": "이미 회수한 문서가 이름을 댄 미호출 도구 2개", "order": "먼저 이것을 보라"},
        {"flag": "T2_PROCEDURE", "target": "transfer_to_human_agents",
         "fact": "돌고 있는 절차가 이 도구를 금지한다", "order": "이 호출을 열지 않는다"},
        {"flag": "T2_TRANSCRIBE", "target": "transfer_to_human_agents",
         "fact": "행 값 1113 ≠ 원장 487", "order": "재발행하라"},
    ]
    results = []
    # ① 창이 닫힘 — 호출도 없고 텍스트도 없음 ⇒ 전부 보류
    results.append(_case("창 닫힘 → 전부 보류(발화 0)",
                         _Msg(content="", tool_calls=None), ("transfer_to_human_agents",),
                         None, True, three, 0))
    # ② 창 열림(ACTING) — 표적 호출 시도 ⇒ 최상위 층 하나만 명령·사실 합집합
    results.append(_case("창 열림(acting) → 한 문장·최상위만 명령",
                         _Msg(content="", tool_calls=[_Call("transfer_to_human_agents")]),
                         ("transfer_to_human_agents",), None, True, three, 1))
    # ③ 억제 자격 없음 → 차단이 표면화로 강등 (a2에 warrant 미선언)
    results.append(_case("억제 자격 미선언 → 차단이 표면화로 강등",
                         _Msg(content="", tool_calls=[_Call("submit_referral")]),
                         ("submit_referral",), {},
                         True,
                         [{"flag": "T2_PROCEDURE", "target": "submit_referral",
                           "fact": "절차가 금지", "order": "열지 않는다", "suppresses": True}], 1))
    # ④ replay 불변 — 부착 불가 대상이면 deny는 fail-closed 고정문구, 나머지는 무부착
    results.append(_case("replay 불변(attach_ok=False) → deny만 fail-closed",
                         _Msg(content="", tool_calls=[_Call("submit_referral")]),
                         ("submit_referral",), None, False,
                         [{"flag": "T2_PROCEDURE", "target": "submit_referral",
                           "fact": "절차가 금지", "order": "열지 않는다"},
                          {"flag": "T2_DISCOVERY_NAMES", "target": "other_call",
                           "fact": "미호출 도구", "order": None}], 1))
    # ⑤ 폐기 레버는 말하지 않는다
    results.append(_case("폐기 레버(층 없음) → 말하지 않음",
                         _Msg(content="", tool_calls=[_Call("x")]), ("x",), None, True,
                         # ⚠`T2_UNKNOWN_REPEAT_GUARD`를 쓰던 케이스였으나 2026-08-07에 **폐기 판정을
                         #   철회**해 층이 생겼다(말하는 게 정상). 자기검사가 그 변경을 잡아 FAIL을 냈다 —
                         #   기대를 고치는 게 맞다. 여전히 폐기인 `T2_REPEAT_CAP`으로 바꾼다.
                         [{"flag": "T2_REPEAT_CAP", "target": "x", "fact": "반복", "order": "멈춰라"}], 0))

    print("\n  %s  (%d/%d)" % ("전부 PASS" if all(results) else "**실패 있음**",
                               sum(1 for r in results if r), len(results)))

    # ── 배선 자기검사 — `_ap_regen`의 실제 tag 전수 ──────────────────────
    print("\n" + "=" * 72)
    print("[observe() 배선 검사 — `_ap_regen` 실측 tag 26종 전수]")
    LIVE_TAGS = [  # `t2_gate_patch.py` 호출 26곳에서 추출(2026-08-07)
        "envguard", "truncguard", "giverel", "unkrepeat", "argrepeat", "covfollowup",
        "unverifiedfu", "givexec", "verdict_surface", "searchexhaust", "followup",
        "writeprov", "noticerep", "claimprov", "discreq", "selfdecl", "channel",
        "choiceground", "uninstructable", "usertoolnote", "kbnohit", "givequote",
        "transfertier", "argschema", "signature", "uncalled_unlock",
        "followup_chain", "followup_decision", "unkrepeat",
    ]
    # ★귀속은 **두 단계**다 — 섞어서 한 수로 말하면 안 된다(2026-08-07 사용자 지적):
    #     ① 플래그 귀속: tag → flag.  "이 발화가 누구 것인가"
    #     ② 층 귀속:     flag → layer. "그래서 언제 말하는가"
    #   하네스도 ①은 된다. ②가 없을 뿐이고, **그건 결함이 아니라 하네스의 정의**다.
    orch = _Orch()
    f_ok = f_no = l_ok = l_no = 0
    for t in LIVE_TAGS:
        ly = observe(orch, t, text="(검사)")
        f = TAG_TO_FLAG.get(t)
        if not f:
            f_no += 1
            print("    %-18s ✗ 플래그 미매핑 (추측 금지 — 실물 tag 보고 채운다)" % t)
            continue
        f_ok += 1
        if ly is None:
            l_no += 1
            print("    %-18s → %-26s 층 없음 %s ← 발화하지만 **도메인을 판정하지 않는다**"
                  % (t, f, cell_of_note(f)))
        else:
            l_ok += 1
    n = len(LIVE_TAGS)
    print("\n    ① 플래그 귀속 %d/%d = %d%%   (누구의 발화인가)" % (f_ok, n, 100 * f_ok // n))
    print("    ② 층   귀속 %d/%d = %d%%   (언제 말하는가 — 하네스는 층이 없는 게 정상)"
          % (l_ok, n, 100 * l_ok // n))
    print("\n    ★하네스도 **말한다**. `truncguard`는 `_ap_regen`으로 모델에게 문구를 보낸다.")
    print("      하네스를 가르는 것은 *발화 여부*가 아니라 **무엇을 판정하는가**다 —")
    print("      도메인이면 레버, **채널·자원**이면 하네스(`finish_reason=length`는 채널 사실이다).")
    print("    ⚠거동 변경 0 — 지금은 등록·로그만. 순서를 뒤집는 것은 다음 단계다.")
    print("\n  ⚠전부 **오프라인 자기검사**다. `admit()`은 이제 출구 **두 곳** 다 지킨다 —")
    print("     `_ap_regen`(텍스트 발화)과 `fb` 배치(deny·치환·지침). 후자를 빠뜨린 탓에 창을 켠")
    print("     런에서도 `[ORDER]`가 14회·바이트 동일로 나갔다(win_20260807i 사이드카 실측).")
    print("     아직 안 한 것: ①`speak()`를 실제 출구로 쓰기(순서 뒤집기) ②변수 tag 2곳 해소")
    print("     ③접힘의 라이브 실측(=`[T2_STACK] window folded fb` 발화 확인) — 아직 런이 없다.")
