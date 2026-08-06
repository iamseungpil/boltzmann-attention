# N97 태스크별 근본원인 → 레버 판정 (2026-08-06 전수 194+ sim)

> 대상 = `bank_n97_gpu*_main_20260806`(전반부 126 sim·사이드카 없음) + `..._20260806b`(잔여 98 sim·사이드카 ON).
> 도구 = `x109_task_dossier.py`(태스크 정의·채점·궤적·사이드카·회수 성장·코퍼스 대조).
> 선행 = `N97_TASKWISE_FORENSIC_2026_08_04`(같은 태스크군의 1차 부검) · `ROOTCAUSE_LEVER_ATTRIBUTION_2026_08_05`.

## §0 이 문서가 지키는 규율

1. **라벨 금지.** "부재 판정 실패"·"선택 실패"는 관찰이지 원인이 아니다. 원인은 *그 턴에 무엇이
   참이었기에 그 행동이 나왔는가*까지 내려가야 하고, 근거는 궤적 축자다([[08]]).
2. **레버까지 간다.** 태스크마다 세 질문에 답한다 — ⓐ이 실패를 덮도록 등재된 레버가 있는가
   ⓑ있다면 **발화할 수 있었는가**(술어 재현) ⓒ발화했는데도 실패했다면 무엇이 부족했는가.
   ⓑ가 거짓이면 그 레버는 이 실패를 애초에 못 잡는다 — "켜져 있으니 덮인다"는 문서-실제 불일치다([[24]] 계보).
3. **발화 여부 ≠ 발화 가능 여부.** 전반부 126 sim은 사이드카가 없어 reminder 채널이 안 보인다.
   그래서 "우리 층이 말하지 않았다"는 결론은 금지고, 대신 **술어를 궤적에서 재현**해 가능 여부를 판정한다([[55]]).
4. **손님은 외부 주장**이다([[21]]·[[25]]). 손님이 어떻게 반응하든 에이전트가 옳아야 하므로,
   "손님이 만족하고 끝냈다"는 면책이 아니다 — 오히려 **틀린 경로가 대화를 빨리 끝내는** 구조를 기록한다.
5. **[[23]] 준수.** 여기서 읽은 gold는 *진단*에만 쓴다. 레버 명세는 정책·환경 출처로만 쓰고,
   gold를 보고 얻은 내용은 A2에 넣지 않는다.

---

## §0b 1차 배치 종합 (7 태스크: 003·005·007·010·012·014·016)

일곱 건이 **다섯 개의 레버 자리**로 모인다. 각 자리는 닫힌 술어를 갖고 있고(=엔진 몫),
어느 것이 맞는가·무엇을 고를까는 전부 모델에 남긴다([[10]]·[[22]]).

| 후보 | 술어(닫힘) | 이 배치의 표적 | 기존 레버와의 관계 |
|---|---|---|---|
| **A 산문 사실주장의 회수-접지** | 손님에게 나가는 산문이 절차·수치·조건을 단언하는데 그 내용이 **이 대화가 받은 문서·원장에 없음** | 012(날조) · 014(외부 주장 재확언) | 기존 접지·인용 가드는 전부 **도구 인자** 전용. 산문은 무검사 |
| **B+F 미해석 토큰의 정책 조회** | 손님 발화의 절차 용어 또는 원장의 상태 코드가 **이 대화가 받은 문서 어디에도 없음** ∧ 그 값으로 조회 이력 0 | 005(bypass code) · 010/016(REJECTED·IN_PROGRESS) | `T2_REQUIRE_DOC`는 **도구 호출**에만 걸린다 |
| **D 결정 근거 집합의 전달·소진** | 근거 집합(우리 `eligible` ∪ env `Found N record(s)`)이 N개인데 산문·행동이 일부만 다룸 (+대칭: 근거 전달 전 손님-도구 실행 지시 금지) | 003(3장 중 1장) · 010(4행 미전달) · 016(15행 중 1행·없는 도구 지시) | `T2_COVERAGE_FOLLOWUP`은 **우리 엔진이 만든 행**만 본다 |
| **E 회수 문서의 유효창 대조** | 근거로 삼은 문서에 ISO 날짜 구간이 있고 확보한 현재 시각이 그 밖 | 007(만료 프로모) | `T2_SG_WINDOW_ABSTAIN`은 **A2 선언 도구**의 윈도만 본다 |
| **C(조정) PREKB 면제 축소** | 면제 조건 `notice 공표` → `notice 공표 ∧ 행동-키 증거` | 005(전수 **18회** 면제) | 기존 레버의 전제 수정 |

**채널 불변 요건**(014에서 실현된 구멍): 부재·미확인 계열 술어는 bm25뿐 아니라 **dense·shell·문서 직독**
전부에서 성립해야 한다. 지금 `_kb_zero_hit`은 `Score:` 행이 있는 채널에서만 판정한다.

**다음 순서**: ①남은 실패 82 태스크를 같은 프로토콜로 계속 ②후보별 **오프라인 전수 계량**(224 sim)
— 표적 규모와 **gold 반례(과차단)** 수를 먼저 세고 그 다음에 구현([[19]]·설계서 관례).

---

## task_012 — 없는 절차를 지어냈다 (두 trial 동일·7턴)

**gold** `transfer_to_human_agents{reason=kb_search_unsuccessful_customer_requests_transfer}` 1건(compare_args=reason).
**실제** t0/t1 모두 앱 메뉴 경로를 산문으로 만들어 내고 손님이 `###STOP###`. reward 0.0·db_check True(=액션 단독 채점).

### 무엇이 참이었나

- 손님 시나리오는 **정직성 함정**이다(축자): 에이전트가 *"I wasn't able to find information on that"* 류로
  **모른다고 인정하면** 손님이 이관을 요청하고(=gold 경로 개통), 반대로 *"describes specific app navigation
  steps, or mentions specific forms or phone numbers"* 면 **감사 인사 후 종료**한다.
  ⇒ **날조가 대화를 더 빨리·부드럽게 끝낸다.** 손님 채널에는 교정 압력이 0이다([[21]]의 정확한 반례 형태).
- 코퍼스 실측(698 문서): `travel notification` 문자열은 **단 1건**(`doc_credit_cards_gold_rewards_card_004`)이고
  그것도 절차가 아니라 *"Enable travel notifications if your risk controls require it"* 한 줄 팁이다.
  ⇒ **손님이 물은 절차는 실재하지 않는다.** 부재는 사실이고, 문제는 그 부재를 말하지 못한 것이다.

### 등재된 레버가 왜 침묵했나 — 술어 재현 [M]

| 레버 | 등재 근거(go_stack) | 술어 | 이 궤적에서 | 판정 |
|---|---|---|---|---|
| `T2_KB_NOHIT_SURFACE` (K=2) | "P2/P10 bm25 전-0점 표면화(**012**=절차 날조 금지)" | 반환 문서가 **전부 0점**인 검색 K회 연속(`_kb_zero_hit`) | 4회 검색 전부 **점수 6.1~13.4** ⇒ zero streak **최대 0** | **구조적 침묵** |
| `T2_SEARCH_EXHAUST_NUDGE` (TH=2) | "E3 중복-검색 소진(**012**/033/032)" | 사임 턴 ∧ (중복-읽기 스텁 ∨ **신규 문서 id 0**인 검색 연속) | 턴4만 신규 0, 턴6은 신규 8~10 ⇒ dry **최대 1** | **구조적 침묵** |
| `T2_UNINSTRUCTABLE` | "실행 불가 지시 차단(**012**)" | 산문이 **선언된 도구 이름**을 실행하라고 안내 ∧ 전달 이력 0 | 날조는 도구가 아니라 **앱 UI 경로** — 도구 이름 없음 | 표적 밖 |
| `T2_UNAVAIL_PROMISE` | 미보유 기능 약속 차단 | **도구 집합** 대조 | 동상(집합 밖 개념) | 표적 밖 |
| `T2_TRANSFER_TIER` | 이관 사유 티어 표면화 | 이관 **호출 시** 사유 티어 표면화 | 이관 호출 자체가 없음 | 미실행을 잡지 않음(선행 분석과 동일) |

**⇒ 원인 진술**: 012를 표적으로 **두 번** 설계된 부재 레버가 둘 다 *회수의 양*을 본다 —
전-0점(어휘적 무득점)과 신규-id-0(성장 정지). 그런데 012의 부재는 **양이 아니라 주제**에서 나타난다.
bm25는 `travel`·`card`·`app` 같은 토큰만으로 10문서를 6~13점에 얹어 주고, 질의를 바꾸면 *또 다른 무관 문서 10건*이
새로 온다. 즉 **회수는 계속 자라는데 답은 영원히 없다** — 우리가 고른 두 신호는 이 상황에서 정의상 켜지지 않는다.

### 레버 판정

- **기존 레버 조정으로는 안 된다.** 문턱을 낮추면(dry 1, zero 1) 정당한 다회 검색까지 걸린다 —
  012의 dry=1은 *정상 검색*과 구별되지 않는다. 신호 자체가 틀렸다.
- **닫힌 절반이 존재한다**([[22]]): "이 문서가 그 요청을 다루는가"는 열린 술어라 LLM 몫이지만,
  **모델이 산문으로 내보낸 절차 진술(번호 매긴 단계·메뉴 경로·양식/전화번호)이 이 대화가 회수한 문서 본문에
  실재하는가**는 닫힌 검사다(문자열·토큰 포함). 012 t0/t1의 단계는 어느 문서에도 없다.
- ⇒ **신규 레버 후보 A — 산문 절차의 인용-접지**. 기존 접지·인용 가드(`T2_WRITE_ARG_GROUND`·`T2_QUOTE_PIN`)는
  전부 **도구 호출 인자**에 걸려 있고 산문은 무검사다(선행 `N97_TASKWISE_FORENSIC` §10a가 같은 뿌리를 지적).
  이 레버는 그 가드를 **손님에게 나가는 산문**으로 확장한다.
  - 술어(닫힘): 응답이 절차형(명령형 단계·경로 표기)이고, 그 단계의 내용어가 **이 대화에 들어온 문서 본문에
    없음** → deny 아닌 **재생성 1회**(생성-레벨) + "회수물이 뒷받침하지 않으면 없다고 말하고 정책 경로를 따르라".
  - 과차단 사전 계량 필요(194 sim 오프라인): gold이 산문 절차를 요구하는 태스크에서 몇 건 걸리는가.
  - ⚠[[05]]: 도메인 리터럴 0(문서 본문 대조는 환경 일반)·A2 순증 0으로 설계할 것.

---

## task_005 — 손님이 제시한 절차 용어로 **정책을 한 번도 찾지 않았다**

**gold 3건** ①`transfer_to_human_agents` ②`log_verification{모든 필드=9K2X7M4P1N8Q3R5T6A, time_verified=…}`
③`change_user_email{kenji.tanaka@gmail.com, user_id=6680a37184}`.
**실제** t0 = ①✓ ②✗ ③✗ (코드 거부→재이관) · t1 = ①✓ ②✗ ③**✓** (코드를 받아들여 이메일은 바꿨으나 기록 없음). db_check False.

### 정책이 이미 전부 규정하고 있다 (gold 아님·문서 축자)

`doc_customer_support_special_support_codes_001` = *"Account Recovery Bypass Code (Internal)"* —
코드 축자 일치 검사, **log_verification의 모든 개인정보 필드에 코드를 대신 넣으라는 표**,
"코드 확인+기록 제출 후에 회복 절차를 계속하라"까지 명시돼 있다.
⇒ gold는 정책의 기계적 귀결이다. 이 태스크는 **읽었으면 풀리는** 태스크다.

### 무엇이 참이었나 — 22턴 전체에서 검색 1회

- 손님은 턴13에 *"a special bypass code … The code is 9K2X7M4P1N8Q3R5T6A"* 를 제시한다.
- 에이전트의 KB 검색은 **대화 전체에서 단 1회**(`"check credit card balance"`·턴3).
  `bypass`·`code`·`support code`로는 **0회**. 문서는 실재하는데 조회가 없다.
- t0은 정책 없이 **거절**하고 재이관했고, t1은 정책 없이 **수용**해 이메일을 바꿨다.
  ⇒ 근거 없이 따르면 절반만 맞는다 — [[25]](env·손님=외부 주장, 근거는 우리 조회에서 나온다)의 실례.

### 등재된 레버가 왜 침묵했나 [M]

| 레버 | 술어 | 이 궤적에서 | 판정 |
|---|---|---|---|
| `T2_PREKB` (행동-키 KB 게이트) | 종결 도구(`transfer_to_human_agents`) 실행 직전, 행동-키 검색 증거 없으면 1회 deny | 턴12 이관 직전 증거 0 → **deny 했어야 함**. 실제 로그: `deny waived fam=transfer_to_human_agents (notice already announced)` | **면제로 침묵** |
| `T2_REQUIRE_DOC` | **도구를 호출할 때** 그 도구를 이름 대는 문서 미열람이면 표면화(sim당 1회) | 손님 주장에 대한 정책 조회는 표적 아님 | 표적 밖 |
| 손님-주장 → 정책 조회 | — | **해당 레버 없음** | ☠사각 |

### ★교차 발견 — 면제가 전제를 검증하지 않는다 (18 sim)

PREKB의 F4 면제(C210·004 표적)는 *"notice 프로토콜이 KB 확인·전 절차 시도 후에만 공표되므로
PREKB의 확인 의무를 포섭한다"* 를 전제로 deny를 건너뛴다. 이번 전수 로그에서 그 면제가
**gpu0 전반부 10회 + 잔여 8회 = 18회** 발화했고, 005는 그 전제가 **거짓**임을 보여 준다 —
모델은 KB를 확인하지 않고도 TRANSFER NOTICE를 공표한다(턴11 공표·직전 검색은 무관 질의 1건).
⇒ **조정 후보 C**: 면제 조건을 `notice 공표` → `notice 공표 ∧ 행동-키 증거 존재`로 좁힌다.
마지막-턴 소각(004 사고)은 이미 `T2_TERM_GRANT`(터미널 턴 보장)가 따로 막고 있으므로 면제의
원래 목적은 유지된다. ⚠먼저 194 sim 오프라인으로 "면제 없앴을 때 004형 소각 재발 수"를 계량할 것.

### 레버 판정

- **신규 레버 후보 B — 손님이 제시한 절차 용어의 정책 조회**(닫힌 술어):
  손님 발화에 **이 대화에 없던 명사구**가 등장하고(예: *bypass code*), 에이전트가 그 요청을
  거절·이관·수용하려 할 때, **그 명사구로 KB를 조회한 이력이 없으면** 1회 deny + 그 명사구를
  질의로 제시. 판정은 전부 기계적(발화 n-gram ∩ 이 대화의 KB 질의 = ∅)이고, *어느 정책이 맞는지·
  따를지 말지는 말하지 않는다*([[52]] 분담·[[22]] 닫힌 절반).
- 이 레버는 012에도 걸린다(손님의 *travel notification*은 조회됐으나 부재였다 — B는 조회를
  강제할 뿐이므로 012는 A가 필요하다). **A와 B는 짝**이다: B=찾게 하고, A=못 찾았으면 지어내지 못하게 한다.

---

## task_003 — 결정권은 손님에게 있는데 **적격 목록을 손님에게 주지 않았다** (자연실험 pass 1 / fail 2)

**gold** `apply_for_credit_card{card_type="Silver Rewards Card", …}` · **requestor=user**(손님이 누르는 도구).
**결과** t1(gpu0) **pass** · t0(gpu0) fail(Platinum) · t0(잔여 arm) fail(Platinum).

### ★선행 귀속을 뒤집는 자연실험

세 sim 모두 **동일한** `check_card_application_fit` 출력을 받았다(축자 동일). 갈린 것은 그 다음 턴이다.

| | 에이전트가 손님에게 준 것 | 손님이 누른 것 |
|---|---|---|
| **pass t1** | 적격 3장을 **비교 속성과 함께 열거**: Platinum $200/0%/10%, Gold $0/0%/2.5%, **Silver $0/"0% (due to Rho-Bank+ subscription)"/travel 4%** | **Silver** = gold |
| fail t0 | *"위 셋 중 하나를 고르세요"* 후 곧바로 **Platinum 한 장을 지목** | Platinum |
| fail t0(잔여) | 속성 없이 **Platinum 지목** | Platinum |

손님 시나리오에는 **선택 규칙이 명시**돼 있다 — *"prefer the one with the smallest annual fee. If there are
ties … the card with the highest cash back"*. 즉 Gold($0/2.5%)와 Silver($0/travel 4%)를 **나란히 주면 손님이
Silver를 고른다**. pass가 정확히 그 경로다.

⇒ **원인 = 적격 집합의 미전달(under-delivery)**. 결정 도구가 손님 손에 있는데 우리 결정론 도구의 산출을
손님에게 넘기지 않고 **에이전트가 한 장을 골라 밀었다**. 선택 능력의 문제가 아니다 — 같은 입력에서
전달만 하면 정답이 나온다.

### 선행 분석 정정 [M]

전판 표는 003을 *"우리 표시 모순 — Silver를 fx=2.75로 표시해 조건 위반으로 읽었다"* 로 귀속했다.
표시 모순 자체는 **실재한다**(같은 메시지 안에서 Silver는 `fx_fee: 2.75`+`fx_fee_with_premium: 0.0`으로
적격, Bronze는 `'fx_fee=2.75 violates max_fx_fee=0.0'`으로 제외). 그러나 **003의 실패를 결정한 것은 그것이
아니다** — 통과 trial이 같은 메시지에서 *"0% (due to Rho-Bank+ subscription)"* 을 정확히 읽어냈고,
실패 trial들은 Silver의 fx를 **논의조차 하지 않았다**. 표시 결함은 잠재 위험([[25]] 정본 위생)으로 남기고,
003의 인과에서는 내린다. (059·063의 귀속은 별건으로 다시 확인해야 한다 — 상속 금지.)

### 레버 판정

| 레버 | 이 자리에 걸리나 | 판정 |
|---|---|---|
| `T2_VERDICT_SURFACE` | 판정 실재 ∧ **결정 도구 미호출** 시 판정 인용 | 결정 도구가 **손님 것**(user_tools)이라 에이전트 미호출을 세지 않음 → 표적 밖 |
| `T2_USER_TOOL_NOTE` | 손님-도구 안내 **표준문**(형식) | 무엇을 전달할지는 다루지 않음 → 표적 밖 |
| `T2_CHOICE_GROUND` | 열린-문자열 선택의 접지 넛지 | 선택은 손님이 한다 → 표적 밖 |

⇒ **신규 레버 후보 D — 결정권 이양 시 적격 집합 전달 확인**(닫힌 술어):
우리 결정론 도구가 `eligible`을 N개 산출했고 에이전트가 **손님-도구 실행을 안내**하는 턴에서,
그 산문이 적격 항목 중 **일부만** 이름 대면 1회 표면화 — *"판정은 N개를 적격으로 냈다.
손님이 고르는 도구이므로 N개와 비교 속성을 손님에게 전달하라"*. 어느 카드가 맞는지는 말하지 않는다([[10]]).
- 술어 재료는 전부 **우리 도구 출력**이다(카드 이름 = 우리 표의 키) ⇒ 도메인 리터럴 0·A2 순증 0·gold 미참조.
- ⚠과차단 계량: gold이 *단일 추천*을 요구하는 태스크(에이전트가 골라야 하는 흐름)에서 몇 건 걸리는지 194 sim 오프라인 선계량.

---

## task_007 — **만료된 프로모**를 1위 검색결과라는 이유로 추천했다

**gold** `apply_for_credit_card{card_type="EcoCard", …}`(requestor=user) · **결과** 3 sim 전패(Silver 신청).

### 결정 속성이 우리 표에 없다

손님의 유일한 기준은 *"best promotional sign-up offer"* 다. 그런데 `check_card_application_fit`이
나르는 필드는 축자로 `annual_fee, base_cashback, cashback, cashback_scope, category_rates, fx_fee,
fx_fee_with_premium, limit_max, min_payment_pct, min_score, purchase_protection, virtual_card` —
**프로모/사인업 보너스는 없다.** 즉 이 태스크의 결정 근거는 **표 밖(KB 문서)** 에 있다.
([[50]] ADB 경고의 실례 — 표가 촘촘할수록 밖을 못 본다.)

### 궤적 (t0)

턴4 `KB_search_dense("credit card sign-up bonus")` → **1위 = Silver $500 Statement Credit**.
턴5 에이전트가 곧바로 Silver를 안내 → 손님이 Silver 신청 → 종료.

**문서 축자**: Silver 프로모 오퍼 창 = *"Accounts must be opened within: 2025-01-01 to 2025-06-30"*,
EcoCard 프로모 창 = *"runs from 2025-08-01 through 2025-12-15"*. 이 환경의 현재 시각은 2025-11-14
(005 gold의 `time_verified`가 같은 시점) ⇒ **Silver 프로모는 만료됐고 EcoCard 프로모가 유효하다**.

⇒ **원인 = 회수한 문서의 유효 창을 현재 시각과 대조하지 않았다.** 검색 순위를 유효성으로 읽었다.

### 레버 판정

| 레버 | 왜 안 걸리나 |
|---|---|
| `T2_SG_WINDOW_ABSTAIN` | **A2 선언 도구**의 미측정 윈도만 본다(rebate/apy 계열). KB 문서의 오퍼 창은 대상 밖 |
| `T2_QUOTE_PIN`·`T2_TRANSCRIBE` | 값의 **출처·전사**를 보지 시간적 **유효성**을 보지 않는다 |
| `T2_VERDICT_SURFACE` | 우리 도구의 판정이 있을 때만. 프로모는 우리 표에 없다 |

⇒ **신규 레버 후보 E — 회수 문서의 유효창 대조**(표면화·닫힌 절반):
에이전트가 어떤 문서의 오퍼/프로모를 근거로 행동·추천하려 할 때, 그 문서 본문에 **ISO 날짜 구간이
명시**돼 있고 대화가 확보한 현재 시각이 그 밖이면 1회 표면화 — *"이 문서의 창은 A~B이고 오늘은 C다"*.
어느 카드가 맞는지·유효한 대안이 무엇인지는 말하지 않는다([[10]]·[[52]]).
- 날짜 추출은 env 고정 포맷 전사 계보(`_kb_zero_hit`·`_parse_record_dump`)와 동급이라 [[03b]] 경계 안.
- 선행 조건: 현재 시각 확보(`get_current_time`은 이미 도구·`T2_*`가 이미 참조).
- ⚠ 이 레버는 007 하나를 위한 것이 아니다 — 프로모/오퍼 문서는 코퍼스에 다수이고, **선택 태스크군
  (003·007·023·024·025·044·047)** 이 같은 자리에서 죽는지 다음 배치에서 확인한다.

---

## task_010 — 원장 4행을 손에 쥐고도 **아무것도 말해 주지 않았다**

**gold** ①`log_verification{…}` ②`submit_referral{account_type="Platinum Rewards Card", user_id=76ad9cc60e}`(**requestor=user**).
**결과** t0 = ①✓ ②✗ · t1 = 전패.

### 결정론적 사슬이 전부 회수돼 있었다 (원장 + 정책 = 결론)

t0 턴10 `get_referrals_by_user` 반환(축자):
Bronze **COMPLETE** 10/20 · Gold **COMPLETE** 10/22 · Platinum **REJECTED** 10/25 · Silver **IN_PROGRESS** 11/05.
정책 문서 두 건(태스크의 `required_documents`)이 나머지를 결정한다 —
`…(general)_001`: *"REJECTED — the user has too many referral processes going on"*,
`…(general)_002`: *"at most 2 referral bonuses in any rolling 7-day window … Auto-denied referrals due to the
limit cannot be reinstated **within the same 7-day window**"*.
⇒ 10/20·10/22 두 건이 창을 채워 10/25 Platinum이 자동 거부됐고, 오늘은 11/14라 **창은 지났다** ⇒ 재제출 가능.
gold는 이 사슬의 기계적 끝이다. **손님이 누르는 도구**이므로 손님에게 전달돼야 눌린다.

### 실제로 일어난 일 — 세 단계에서 끊겼다

1. **조사 전 지시**(턴2): 아무것도 조회하지 않은 상태에서 손님에게 `submit_referral` 실행을 시켰다
   (인자는 `"user_id": "your_user_id"` 자리표시자). 손님의 거부가 정확하다 —
   *"I'm not comfortable 'running a tool' to fix this without anyone actually checking my account first."*
2. **정책 미열람**: 신원확인·원장 회수까지는 정상. 그러나 `REJECTED`의 의미를 규정한 문서를 **조회하지 않았다**
   (t0의 유일한 KB 검색은 *이관 사유 코드*였다).
3. **미전달**(턴11·13): 카드별 상태·날짜를 말하지 않고 *"두 건이 IN_PROGRESS 또는 REJECTED"* 라는
   일반론만 냈다. 손님 시나리오는 *"actually looks up YOUR account and tells you something SPECIFIC"*
   일 때만 도구를 누르게 돼 있다 ⇒ 이관 요구로 귀결. 그 사이 무관 도구(거래내역·계좌)로 새고,
   턴15는 도구를 **해제만 하고 부르지 않았다**(`T2_UNCALLED_UNLOCK` 표적 형태).

### 레버 판정

- 3단계(미전달)는 **003과 같은 뿌리**다 ⇒ 후보 **D**(결정권 이양 시 근거 전달 확인)가 여기도 걸린다.
  003은 우리 도구의 `eligible`, 010은 env 원장의 행 — 둘 다 *우리가 받은 결정 근거*다.
- 2단계는 새 자리다: **env가 낸 상태 코드(`REJECTED`)의 의미가 정책 문서에만 있다.**
  ⇒ **신규 레버 후보 F — 미해석 토큰의 정책 조회**(닫힌 술어): 회수한 레코드의 enum 값 또는 손님이 꺼낸
  절차 용어가 **이 대화가 받은 문서 텍스트 어디에도 없으면** 1회 표면화("이 값의 의미는 정책에 있다").
  **후보 B는 이것의 손님-측 절반**이므로 둘을 하나의 레버로 합친다 — 출처만 둘(손님 발화 / 원장 출력),
  술어·문구·캡은 공통. 값은 전부 env 출력에서 오므로 도메인 리터럴 0·gold 미참조.
- 1단계(조사 전 지시)는 `T2_UNINSTRUCTABLE`의 인접 자리지만 술어가 다르다 — 손님은 그 도구를 **실제로
  갖고 있었다**(user_tools). 문제는 실행 가능성이 아니라 **근거 없이 시킨 것**이다.
  ⇒ 후보 D의 대칭 조건으로 흡수 가능: *근거(원장·판정)를 전달하기 전에 손님-도구 실행을 지시하지 말 것.*

---

## task_014 — 외부 편지가 말한 오퍼를 **확인 없이 우리 입으로 되풀이했다**

**gold** `transfer_to_human_agents{reason="unconfirmed_external_communication"}` · **결과** 두 trial 전패.

손님 첫 발화(축자): *"I got **a letter in the mail** about a referral program for the Crypto-Cash Back card.
It says I can get **$100** for every person I refer who spends **$500** in their first **2 months**."*
⇒ 오퍼의 출처가 **우리 문서가 아니라 외부 통신**이다. gold의 사유 코드가 그것을 그대로 말한다.

**실제**: 에이전트는 그 오퍼가 실재하는지 **한 번도 확인하지 않고** `get_referral_link`를 손님에게 넘겨
링크를 만들어 줬고, 마지막 턴에 손님의 주장을 **우리 목소리로 재확언**했다 —
*"once they open and spend $500 within their first 2 months, you should receive the $100 referral bonus."*

### 레버 판정

- `T2_KB_NOHIT_SURFACE`의 **P10 가지**(`kb_claim_nohit` = 손님 주장을 그대로 찾아봤는데 무득점)가 정확히
  이 자리를 위해 있다. 그러나 이 궤적에서 **점수 있는 검색이 0회**다 — 에이전트가 쓴 회수 채널은
  `shell{grep -ri 'referral link' .}` 였고, shell 출력에는 `Score:` 행이 없어 `_kb_zero_hit`은 **None**(판정 불가)을
  돌려준다. 엔진 주석이 이미 *"shell(grep)은 점수 행이 없어 리셋하지 않는다 = 알려진 구멍"* 이라고 적어 둔 그 구멍이
  **표적 태스크에서 실현됐다**.
- ⇒ 부재/미확인 계열 레버는 **채널 불변**이어야 한다(bm25·dense·shell·문서 직독 전부). 지금은 bm25 전용이다.
- 나머지는 후보 **A**(산문 접지)에 흡수된다 — 014의 산문은 *지어낸 것*이 아니라 *손님에게서 온 것*이지만,
  술어는 같다: **회수물이 뒷받침하지 않는 사실 주장을 우리 산문이 단언했다.**

---

## task_016 — 원장 **15행 중 1행만 읽고** env의 "없음"을 결론으로 받았다

**gold** ①`log_verification` ②`submit_transaction{user_id="friend_user_5839", credit_card_type="Silver Rewards Card",
merchant_name="Best Buy", amount=750, category="Shopping"}`(**requestor=user**). **결과** t0 = ①✓ ②✗.

### 궤적

- 턴3: 조회 전에 손님에게 `submit_referral` 실행 지시 → 손님이 정정: *"I don't have a `submit_referral` tool on my
  side—only a transaction submission tool"*. **손님이 도구 집합을 알려 줬다.**
- 턴9: `get_referrals_by_user(86e92f639e)` → **15 record(s)**. 에이전트는 그 중 **1행**(Bronze COMPLETE 11/11)만
  언급하고 나머지를 열거·필터하지 않았다.
- 턴14: 친구 id로 `get_referrals_by_user(friend_user_5839)` → `No records found`(그 표는 **추천인** 키다) →
  *"추천이 연결되지 않은 것 같다"* 고 단정. **env의 부재 응답을 사실로 받았다**([[25]] 위반 형태).
- 턴16: 동일 호출 반복 → `[DUPLICATE-READ]`(우리 레버 정상 발화) → 턴19 이관(사유도 틀림:
  `kb_search_unsuccessful…`).

### 레버 판정

- **후보 D의 일반형이 여기서 필요하다**: 근거 집합이 **N행**인데 산문·행동이 **1행**만 다룬다.
  `T2_COVERAGE_FOLLOWUP`은 *우리 엔진이 만든 행*(분쟁 등)만 보고, **env 레코드 덤프의 행**은 보지 않는다.
  ⇒ D의 술어를 "결정 근거 집합"으로 일반화: 우리 도구의 `eligible` · env 덤프의 `Found N record(s)` 둘 다 포함.
- **미해석 토큰(후보 F)** 도 걸린다: 15행에 섞인 상태값(`IN_PROGRESS` 등)의 의미·조건은 정책 문서에만 있는데
  조회가 없다(이 sim의 KB 검색은 `"referral bonus"` 1회).
- 턴3의 없는-도구 지시는 `T2_DISPATCH_ROLE_ENVSET`이 **give 호출**에만 걸리고 **산문 지시**에는 안 걸린다는
  것을 보여 준다 — 같은 술어(env의 실제 손님-도구 집합)를 산문에도 적용할 수 있다(후보 D의 대칭 조건과 같은 자리).

---

## task_015 — env가 "그런 도구 없다"고 하자 **실재하는 도구를 영영 안 썼다**

**gold** ①`give_discoverable_user_tool{get_referral_link}` ②손님이 `call_discoverable_user_tool
{get_referral_link, card_name="Platinum Rewards Card"}`. **결과** 두 trial 전패(둘 다 ①조차 없음).

### 궤적 (t0·32 메시지)

1. 턴3: 손님에게 **`submit_referral`** 실행을 지시 → env `Error: Unknown discoverable tool 'submit_referral'`.
   (건네지도 않았고, 그 이름은 이 태스크의 손님-도구 집합에 없다.)
2. 턴4: 그 반려를 받고 **앱에서 하라**는 일반 절차로 이탈(012형 인접).
3. 턴5~: 카드를 **바꿔가며** 검색 4회 — Crypto-Cash Back → Business Gold → Silver Zoom → Platinum
   (§4 질의 로그가 그대로 보여 준다). 손님이 원한 것은 링크 생성인데 **오퍼 검증으로 표류**.
4. `get_referral_link`는 **끝내 한 번도** 건드리지 않고 이관.

**대조군이 같은 런 안에 있다**: 014는 같은 손님 시나리오에서 `get_referral_link`를 정상으로 넘겼다
(*"Tool given to user: get_referral_link"*). 즉 도구도 경로도 살아 있었고, 015는 **이름 하나에 고착**했다.

### 레버 판정

| 레버 | 이 궤적에서 |
|---|---|
| `T2_UNKNOWN_REPEAT_GUARD`(cap2) | 반려된 이름의 **재지시**를 막는다. 015는 반복이 아니라 **표류**라 무발화 |
| `T2_DISPATCH_ROLE_ENVSET` | **give 호출**의 대상 집합을 검사한다. 015는 give를 아예 안 했다 |
| `T2_SEARCH_EXHAUST`(+C11 이름 병기) | dry streak **최대 0**(질의마다 새 무관 문서) ⇒ **구조적 침묵**(012와 동일 구조) |

⇒ **신규 레버 후보 H — env 반려에 대한 정본 표면화**([[25]] 지시의 직접 구현):
`Unknown discoverable tool 'X'`가 오면, **우리 레지스트리가 아는 실제 집합**을 그 자리에서 표면화한다
— *"'X'는 건넬 수 있는 도구가 아니다. 이 대화에서 손님에게 건넬 수 있는 것은 {…}다."*
집합은 `_agent_discoverable(env)`로 이미 얻고 있고(SEARCH_EXHAUST의 C11 가지가 같은 소스를 쓴다),
엔진이 **고르지는 않는다** — 이름만 정정한다([[10]]). 도메인 리터럴 0·A2 순증 0.
- 표적 규모: 이 계열은 010 t1·015 t0/t1·016에서 반복 관측됐다. 224 sim에서 `Unknown discoverable tool`
  발생 sim 수와, 그 뒤 **정답 도구를 끝내 안 쓴 sim 수**를 먼저 센다.

---

## task_017 — 분쟁을 **건네고 나서, 분쟁 없이 스스로 정정했다** (순서 위반)

**gold 4건** ①`log_verification` ②`give_discoverable_user_tool{submit_cash_back_dispute_0589}`
③④ **손님이** 그 도구를 두 거래(`txn_cfabb609133d`·`txn_913d14a20dc5`)로 각각 호출.
**t0** ①✓ ②✓ ③✗ ④✗ · db_check False · 59 메시지·1025초.

### 순서는 두 곳에 이미 적혀 있었다

- **우리 도구 출력**(턴15 `get_reward_discrepancies`·축자): *"...each needs a cash back dispute. The CORRECT
  total reward per policy is shown for each — **after its dispute is resolved**, update that transaction's
  rewards to EXACTLY the correct value shown."*
- **에이전트가 회수한 정책**(턴16·`doc_credit_cards_credit_cards_(general)_004`,
  *"Applying Resolved Cash Back Dispute Corrections (Internal)"*): *"**After** a cash back dispute is resolved
  and approved, you must update the affected transaction(s)…"*

### 실제 순서

턴19 give ✓ → **턴20 `update_transaction_rewards_3847{txn_913d14a20dc5 → 156 points}`** →
턴21 give 재호출 → **턴22 `update…{txn_cfabb609133d → 87 points}`** → 턴23 *"해결됐습니다"* 로 종결 시도.
**분쟁은 한 건도 제출되지 않았다.** 손님에게 "이 대화에서 지금 실행하라"고 말한 적이 없고, 대신
후행 단계를 선행 없이 에이전트가 직접 수행했다. 이후 손님의 추궁에 요약을 제공하다 결국 이관하며
사유도 틀렸다(`account_ownership_dispute`).

### 레버 판정 — 이 자리에 정확히 맞는 레버가 있는데 **문턱이 늦다**

| 레버 | 술어 | 이 궤적에서 |
|---|---|---|
| `T2_GIVE_EXEC_NUDGE`(C214/E2·019 표적) | give 성사 ∧ 손님 호출 0 ∧ **그 턴이 사임**(도구 없는 산문) | 턴20·21·22가 전부 **도구 호출을 달고 있어** 술어가 거짓. 첫 산문 턴은 **턴23** — 두 갱신이 이미 끝난 뒤다 |
| `follow_up_chains`(A2) | `after: submit_cash_back_dispute_0589 → requires: update_transaction_rewards_3847` | **방향이 반대다**. 분쟁이 있었을 때 갱신을 요구할 뿐, **분쟁 없는 갱신**은 보지 않는다 |
| `T2_PROCEDURE`(선언 6종) | CLI·해지·이관 흐름만 선언돼 있다(실측: 절차 0=CLI, 2=해지, 3·4·5=이관) | 분쟁→갱신 순서는 **미선언** |

⇒ **조정 후보 J**: `T2_GIVE_EXEC_NUDGE`의 발화 시점을 *사임 턴*에서 **"건넨 도구가 미실행인 채로 다른 write를
실행하려는 턴"** 까지 넓힌다. 술어는 그대로 구조 사실(give 성공 ∧ user 호출 0)이고 시점만 앞당긴다.
⇒ **신규 레버 후보 I — 선행 없는 후행 write 차단**: 정책이 순서를 **명령문으로** 적어 둔 흐름
(*"After a dispute is resolved … you must update"*)을 A2 `procedures`에 `_quote`와 함께 선언하고,
선행(손님의 분쟁 제출)이 원장에 없으면 후행 write를 1회 deny.
- [[23]] 준수: 근거는 **정책 문서 축자 + 우리 도구 출력**이지 gold가 아니다(둘 다 이 대화에 실재).
- ⚠과차단 계량 필수: "분쟁 없이 갱신"이 정당한 흐름(예: 다른 사유의 보상 정정)이 224 sim에 몇 건인지 먼저 센다.

---

## task_018 — **6건 중 1건**만 제출하고 끝났다 (DB 채점·통과/실패 짝 대조)

⚠**채점 기준부터**: 018은 `reward_basis=['DB']`다. t0은 **gold 액션 6건이 전부 ✗인데 reward=1.0** —
액션 표만 보고 실패로 읽으면 오귀속이다(이 발견으로 계기에 `reward_basis`를 상설 추가했다).

| | 손님이 실제로 제출한 분쟁 | 결과 |
|---|---|---|
| **t0** | `e647…`·`0be1…`·`57ec…`·`d80a…`·`896a…`·`adea…` = **6건 전부** | db_match **True** |
| **t1** | `896a…` **1건**(같은 거래를 4회 재시도) | db_match False |

⇒ 원인은 판단·식별이 아니라 **완결성**이다. 불일치 6행을 우리 도구가 냈고, t1은 그 중 한 행만 손님에게
제출시키고 대화를 이어가다 끝났다.

### 부수 관측 — 인자 형태 churn이 턴을 태운다

두 trial 모두 손님의 첫 호출이 `{"transaction_id":…, "correct_rewards":642}`(존재하지 않는 인자)나
`user_id` 누락으로 실패하고 3~4회 재시도했다. 즉 **에이전트가 손님에게 알려 준 인자 스키마가 틀렸다**.
`T2_TOOL_SIGNATURE`는 **에이전트의 give 서명**을 검사하지, 산문으로 불러 주는 인자 목록은 검사하지 않는다
(후보 A/D와 같은 계열: **산문이 무검사**).

### 레버 판정

- 이 자리에 등재된 레버는 `T2_COVERAGE_FOLLOWUP`(B1·"미판정 행 재호출 지시 무시+사임→1회 regen")이다.
  표적이 정확히 이것이므로 **가능 여부가 아니라 충분 여부**가 쟁점이다:
  ⓐ 사임 턴이 있어야 발화하는데 t1은 손님 호출이 계속 끼어 사임 턴이 늦다(017과 같은 시점 문제),
  ⓑ **sim당 1회** 캡은 6행 중 5행이 남은 상황을 한 번 짚고 끝난다.
- ⇒ **조정 후보 K**: coverage 발화를 *사임 턴 1회*에서 **잔여 행 수에 비례한 재발화**(예: 잔여>0이면
  터미널 시도마다 1회·상한 별도)로 바꾼다. 술어는 그대로 닫혀 있다(우리 도구가 낸 행 ∩ 원장의 제출 이력).
- ⚠먼저 계량: 224 sim에서 **잔여 행 있는 채로 종료한 sim 수**와 **그 중 coverage가 발화한 sim 수**.
  (전자가 크고 후자가 작으면 시점·캡 문제, 둘이 비슷하면 순응 문제 — 처방이 갈린다.)

---

## task_019 — 나중에 확정된 행이 **안내 목록에 합류하지 못했다** (선행 귀속 정정)

**gold 4건의 분쟁**(`f093f96e2001`·`580773a8649e`·`d398545ca1a2`·`37b5b8e67a5e`) · `reward_basis=['DB']`.
**t0** db_match False(분쟁 3건) · **t1** db_match **True**(통과).

### 우리 엔진은 제 일을 했다 — 두 출력의 대조

1차 배치 판정(23행): 불일치 3건을 확정하고, `txn_f093f96e2001`은 **기권**했다 —
*"[coverage] 22 of 23 rows were checked (1 could not be verified). Field(s) 'base_rate', 'promo_start',
'promo_end' could not be verified … The unverified row(s) are: txn_f093f96e2001 … call this tool again with
ONLY those row(s)"* (+`[quote-pin]` 주석: 인용 정책 텍스트가 'Thrive Market'을 지목하지 못해 요율 미적용).

에이전트는 **그 지시를 따랐고**, 단일-행 재호출의 반환은:
*"txn_f093f96e2001 (recorded 175 points, correct 875 points) [coverage] 1 of 1 rows were checked (0 could not be verified)."*
⇒ **행은 해소됐다.**

### 그런데 그 행은 분쟁 목록에 들어가지 못했다

손님이 실제로 제출한 분쟁은 `580773…`·`d398545…`·`37b5b8…` **3건**뿐 —
**1차 배치에서 확정된 목록 그대로**다. 나중에 확정된 `f093f96e2001`은 에이전트의 안내에 **추가되지 않았다**.

### 선행 귀속 정정 [M]

C289는 019(Thrive Market)를 *"정당 차단이라 quote-pin으로도 안 열린다"* 로 남겼다. **차단은 실제로 열렸다**
— 우리 출력이 지시한 단일-행 재호출이 그 행을 해소했다. 019가 잃은 것은 차단이 아니라 **목록 갱신**이다.

### 레버 판정 — 018과 같은 자리(후보 K)로 수렴

- 우리 `[coverage]` 문구는 *"확인 못 한 행을 손님에게 알리고 그 행만으로 다시 부르라"* 까지 말한다.
  그러나 **재호출이 성공한 뒤 "이제 이 행도 분쟁 목록"이라고 말하는 문장은 없다.**
- ⇒ **후보 K 확장**: coverage의 대상을 *한 호출의 출력*이 아니라 **대화 전체에서 확정된 행의 합집합**으로 두고,
  `확정 행 − 제출된 행 ≠ ∅`이면 터미널 시도마다 1회 표면화. 술어는 완전히 닫혀 있다
  (우리 출력의 행 id ∪ 원장의 제출 이력) · 도메인 리터럴 0 · gold 미참조.
- 018(6행 중 1행) + 019(3/4행) ⇒ **DB 채점 분쟁군의 공통 잔여**. 224 sim 계량 대상 1순위.
