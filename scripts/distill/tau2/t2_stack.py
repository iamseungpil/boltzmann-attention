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

## 무엇이 아닌가 (정직)

**이 파일은 검사를 구현하지 않는다.** 각 레버의 술어는 여전히 자기 모듈에 있다
(`t2_procedure` `t2_transcribe` `t2_source` …). 이 모듈이 정하는 것은 *무엇을 검사하느냐*가 아니라
**누가 말하느냐**다. 97개 호출 지점을 이 `route()`로 모으는 배선은 **아직 안 했다** —
이름을 붙였다고 구현이 된 게 아니라는 것이 2026-08-07의 교훈이다([[03b]]).
"""

import os

__all__ = ["LAYERS", "MECHANISMS", "layer_of", "route", "conflicts"]


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
     "T2_A2_VARIANT=ledger", "출처 근거 확보"),
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
    ("조건 재소환 실패", "auth 게이트 미충족 구간인데 **행동-유도 레버가 발화**(단계 소유권 부재)",
     "C17 · 050",
     "T2_PHASE_OWNER", "차단"),
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
    ("부재 미종결", "☠반려된 Unknown-tool 이름을 재지시 — **횟수로 억제한 폐기 레버**",
     "010/014/015/016 [S] · [[57]] 위반 ⇒ 정체-과금·지문 억제가 대체",
     "T2_UNKNOWN_REPEAT_GUARD", None),

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
