# x291 — 075 checking-픽 격리 + 조건부 checking-fit 레버 설계 (2026-08-13)

> [[62]] 절차: **레버 저작 전 격리로 결손을 잰다**. 이 문서가 사전등록 정본이며, 문턱·판정
> 매트릭스는 실행 전 고정한다. 선례 = x288→(B) `get_atm_fee_discrepancies`(C462).

## §0 실측 배경 (bank_t7275_b_20260813v · task_075 t0 · reward 0.0 · 전수 정독)

- **gold**: verify → unlock/call `open_bank_account_4821`(checking · **Green Fee-Free Account**).
  근거 문서(doc_checking_accounts_green_fee-free_account_005 축자): OON $0.00 · foreign $0.00.
- 궤적 34msg 판정: ⓐ **Green Fee-Free 문서 회수 0**(KB 검색 2회 전부 business+타클래스 —
  궤적 전체에 그 클래스명 등장 0회) ⓑ msg04 = 첫 검색 직후 **계산 0으로 즉시 Purple 픽**
  (여행-마케팅 함정 그대로)+자기서비스 포털 안내 ⓒ msg10 = "추정 도구 없음" 발화 후 수동
  계산 시도(오답) ⓓ msg26 = `check_card_application_fit`(카드용) 오선택 ⓔ open_bank_account
  언급 0 — write 시도 자체 부재(자기서비스 이관 = L1 학습행·이 프로브 범위 밖).
- 즉 075 실패는 **픽 결손(후보 커버리지+계산+정박)** 과 **write 이관(L1)** 의 복합. 비학습
  레버 후보는 픽 축뿐이며, 그 안에서 무엇이 실패 단계인지 이 프로브가 가른다.

## §1 사전등록 — 셀 5 (n=8 · TAG=bank_t7275_b_20260813v · task_075 · 8141)

컷 = 첫 "$350" 포함 user msg(=msg09·사용 패턴 확정 발화) 직후. 계기 = 응답 말미
`FINAL: <account class>` 줄(지시로 강제)의 정규화 문자열에 `feefree` 포함 여부.
보조 계기 = purple 비율·FINAL 줄 부재 수. 도구 스키마는 라이브 동일 제공하되 ASK 가
"지금 있는 재료로 텍스트로 답하라"를 명시(픽 능력 측정이지 회수 측정이 아님).

| 팔 | 문맥 | 가르는 것 |
|---|---|---|
| A_LIVE | msg00~09 그대로(문서 없음) | 라이브 재현 대조 — Purple 유지 기대 |
| B_DOCS | + 전 클래스 ATM 문서 기계선별(클래스당 ≤2편·후술) | 재료 전달만으로 픽이 열리나 |
| C_CALC | B_DOCS + 클래스별 수수료 총액 표(계기용·(B)-동형 출력 모사) | 산술 이관이 픽을 여나 |
| D_NEG | + business 클래스 문서 동수 | 길이/문서-존재 통제([[57]]) |
| E_FRESH | msg00~01+msg09+B_DOCS 문서(중간 Purple 발화 제거) | 자기-정박([[18]] C124) 분리 |

문서 기계선별([[59]] 내용판단 0): `doc_checking_accounts_<class>_*` 전 클래스 열거 →
본문 'ATM' 포함 → (Foreign-ATM 포함 우선, ATM 빈도 내림차순) 상위 2편 → 1200자 절단.
x287b 교훈(목록 위생=효과 변수: 31개 5/8 ↔ 8개 8/8)에 따라 클래스당 캡을 둔다.

C_CALC 표(계기 전용·출하물 아님): 18회×$350 foreign OON 기준 클래스별 Rho-fee 총액 —
green_fee-free $0 · bluest $36 · purple $45 · gold_years $63 · light_blue $78 · light_green $99 ·
evergreen $171 · dark_green $171 · blue $243 · green(checking) $243. 전부 정책 스케줄
(`ATM_FEE_SCHEDULE_VERBATIM` + green_ff_005·gold_years_002 축자)에서 산출·argmax/추천 문구 없음.

## §2 판정 매트릭스 (사전 고정)

- **A_LIVE ≥6/8** → 라이브 실패가 컷-의존/확률성 → 프로브 무효·재설계.
- **D_NEG ≥3/8** → 형식(FINAL 지시) 효과 오염 → 프로브 무효.
- A_LIVE ≤2/8 ∧ **B_DOCS ≥6/8** → 결손 = 회수/전달 → 레버 = **(A) 후보군 문서 표면화만**
  (기존 전달 기전 보강·결정론 op 신설 금지).
- B_DOCS ≤2/8 ∧ **C_CALC ≥6/8** → 결손 = F2b 산술(x288 동형) → **(B) checking-fit op 출시**(§3).
- B_DOCS ≤2/8 ∧ C_CALC ≤2/8 → 표를 줘도 픽 실패: **E_FRESH ≥6/8** 이면 결손=자기-정박 →
  레버 = 결정-재개방(DECIDE 계열 배치) 별도 설계 / **E_FRESH ≤2/8** 이면 L2-계열 학습행
  (`LEARNING_BACKLOG` 추가) — 결정론 저작 금지.
- 중간(3~5/8) → 해당 팔만 n=16 재측정 1회.

## §3 조건부 레버 스케치 — (B) 경로일 때만

`scaffold_get_tools` 항목 `get_checking_atm_fee_totals`(가칭·check_card_application_fit 동형):
- **입력(모델 formalize·[[22]] 근거-우선)**: 후보 클래스별 {OON 공식, foreign 공식, 무료횟수}
  — 모델이 문서에서 **복사**(출처 doc-id 병기) + 사용 패턴 {회수 n, 건당 금액, 개월}.
- **엔진(도메인-일반 산술만)**: 단가×횟수·%·min/max 캡·tier·무료횟수 차감 — (B) rewards 판
  op 재사용 축. **총액 표 반환만** — argmax·추천·정렬 없음([[62]] 최소 결정론·픽은 모델 몫).
- **A2 상수 없음이 원칙**: 스케줄은 모델이 회수 문서에서 복사(전달 레버가 문서를 올려줌).
  x288 과 달리 "재료가 손에 있으면 formalize 는 되는데 산술이 틀리는" 경계에만 이관.
- 오표적 부수효과: checking 동형이 생기면 msg26 형(`check_card_application_fit` 오선택)의
  표적 부재도 해소.

### [[05]] 3질문 ([[17]] 상설)

1. **무엇이 고정인가**: TBox weights·scaffold 엔진(산술 op = 도메인-일반: 곱·캡·tier·차감).
2. **무엇이 변경인가**: 없음이 원칙(스케줄 상수를 A2 에 넣지 않는 설계). 넣게 되면 A2 층
   (정책 축자·`_note_` 출처 병기·[[23]])만.
3. **도메인-특화 scaffold 인가**: 아니오 — op 는 select_discrepant/fee-산술 재사용, banking
   지식은 모델-복사 입력으로만 들어온다.

### [[66]] 자기점검

의도 분류 아님(모델이 자발 호출하는 도구·발화 케이스 열거 없음)·공유 상류 노드 아님
(폭발 반경 = 호출자뿐)·무측정 출시 아님(x291 이 이 문서의 문턱으로 게이트).

### [[62]] 4문

① 결손을 격리로 쟀나 — 본 프로브. ② 격리에서 열리면 — 전달만((A)). ③ 결정론은 격리에서도
실패한 단계에만 — C_CALC 매트릭스로 한정. ④ 엔진이 정답을 내나 — 아니오(총액 표만·픽은 모델).

## §4 실행

리모트 8141(유료 런 종료 후) · `T2_PROBE_URL=http://localhost:8141/v1/chat/completions
python x291_checking_pick_iso.py [N]` · 결과는 본 문서에 추기.

## §5 결과 (추기 2026-08-13 · 완주 n=8×5셀 · 8141)

| 팔 | feefree | purple | FINAL줄부재 | 문맥 |
|---|---|---|---|---|
| A_LIVE | **0/8** | 7 | 0 | 8,601자 |
| B_DOCS | **0/8** | 3 | 2 | 32,966자 |
| C_CALC | **8/8** | 0 | 0 | 33,786자 |
| D_NEG | 0/8 | 0 | 3 | 28,153자 |
| E_FRESH | **1/8** | 2 | 0 | 25,610자 |

**판정(§2 매트릭스 그대로)**: A_LIVE ≤2 ∧ D_NEG ≤2 = 프로브 유효 · B_DOCS ≤2 ∧ C_CALC ≥6 →
**(B) checking-fit op 출시**. E_FRESH 1/8 = 정박 제거+전 문서로도 실패 — 결손은 자기-정박이
아니라 **스케줄 산술(F2b)** 로 확정(x288 동형·fee-가족 2호).

**D_NEG 이상치 포렌식(리뷰 지적 ③·[[08]]·x291c 로 추기)**: A_LIVE purple 7/8 ↔ D_NEG purple
0/8 — 무관(business) 문서 추가만으로 Purple 픽이 전멸했다면 A_LIVE 의 Purple 은 함정 소화가
아니라 **문맥-민감 불안정 픽**일 수 있다. D_NEG 8건이 실제 무엇을 골랐는지 x291c(전건 FINAL
줄 로그)로 정독해 이 단락에 추기한다. 판정(프로브 유효·(B) 경로)은 D_NEG 의 feefree 0/8 로
이미 성립하며 이 포렌식은 결론 문장의 온전성 보강이다.
→ **결과(x291c·n=8 전건)**: World Blue(business 클래스) 픽 1 · Purple 1 · FINAL 부재 6
(business 문서의 rebate 조항으로 계산-표류·토큰캡). 해석: A_LIVE 의 Purple 7/8 은 함정
소화가 아니라 **문맥-민감 불안정 픽** — 무관 문서만 얹어도 픽이 흩어진다. 결론 문장 보강:
결손은 "정박"도 "일관된 오답"도 아니고, **산술 재료 부재 하에서 문맥 소음에 끌려다니는
픽**이며, 컴팩트 총액 표(C_CALC)가 그것을 8/8 로 안정화한다.

## §6 출시 기록 (2026-08-13 · **잠정 — x291b ≥6/8 통과 전까지 확정 아님**)

**§3 이탈 명기와 근거(리뷰 지적 ①·[[62]] ③)**: §3 은 "A2 상수 없음이 원칙 — 스케줄은 모델이
회수 문서에서 복사"였으나 출시본은 10클래스 요율을 A2 상수로 넣었다. 이탈 근거:
ⓐ **B_DOCS 0/8 + E_FRESH 1/8** = 문서가 손에 있어도(정박 제거 포함) 요율이 수치 경로로
넘어가지 않는다 — "모델-복사 입력 op" 팔은 별도로 돌리지 않았으므로 이는 **간접 추론**이며,
직접 실측은 x288 A_DOCS 0/8(같은 fee-가족·문서 동봉 formalize+산술 0/8)이 보강한다. 또한
x291b 1차의 미스 실물([0]/[4]: 문서 재계산 시도→오답·장황)이 복사-경로의 라이브 실패 양식을
보여준다. ⓑ 요율의 A2 이관은 **정책 산문 → 도메인 저작**([[23]] 판정 둘째 범주)이다:
`_ins_checking_fee_totals.py` 의 TABLE 은 **기계 추출이 아니라 손 전사**다 — 출처는
`ATM_FEE_SCHEDULE_VERBATIM_2026_08_13.md`(추출 에이전트가 documents/ 만 열람·문서ID 병기)
+ green_fee-free_005·gold_years_002(본 세션 직독) 축자이며, 행별 `source` 필드와 `_note_` 에
문서ID를 병기해 [[23]] 출처 의무를 이행했다. gold(태스크 정의)에서 온 수치·판단은 0
(스태킹 배제가 그 증거 — gold 를 봤다면 넣었을 축이다).

- **엔진**: `catalog_compute`(+`ref_op`·min/max 중첩-스펙 정합화) — select_discrepant steps DAG
  재사용·행별 값 열만 반환(정렬·순위·추천 0). `t2_compute.py`.
- **A2**: `get_checking_atm_fee_totals` 3사본 프로그램 삽입(`_ins_checking_fee_totals.py`) —
  10클래스 축자 요율(각 행 source 병기)·params={months, withdrawals_per_month, withdrawal_amount}.
- **[[23]] 재확인**: 스태킹(OON+foreign 동시 부과)은 personal 문서 미규정·business navy_blue_008
  은 either/or 시사 — 태스크 설명의 "BOTH apply"는 gold 유래라 **넣지 않았다**. 축별 2열 분리
  반환·결합 판단은 모델 몫. gold 픽은 green_fee-free 가 두 축 모두 $0 로 지배라 스태킹 단정
  불요(§1 C_CALC 표와 유일한 의미 차이).
- **검정**: `test_checking_fee_totals.py` 18/18(075 패턴 10클래스 수계산 대조·무료차감·tier
  경계=하위·결측=전원 보류·렌더·3사본 등가) + 기존 배터리 회귀 0.
- **x291b 사전등록**(문면 리터럴=출시본 축자·x287b 교훈): C_SHIP = B_DOCS 문서 + **A2 정본 op
  실행+return_template 렌더 축자** 주입. ≥6/8 출시 확정 · ≤2/8 보류(2열 분리/repr 렌더가 효과를
  죽였는지 포렌식) · 3~5 → n=16 1회. `x291b_shipped_render_transfer.py`.
  **1차(n=8) = 4/8 중간 대역** → n=16 재측정. 미스 정독: 토큰캡 아티팩트 2(문서 재계산
  장황→FINAL 미도달)·진짜 픽 실패 2(Purple 마케팅 정박 1·Bluest rebate 조항 견인 1).
  **재측정 계기 강화(리뷰 지적 ④a·사용자 지시)**: FINAL 줄이 **유일 클래스명**일 때만
  strict-hit(양다리 답 배제)·전건 FINAL 줄 로그([[08]]). 판정은 strict 기준.
  **재측정(n=16·strict) = 11/16 — 문턱(≥12) 1건 미달 → 사전등록대로 보류·형식 포렌식.**
  미스 5건 전건 정독: 토큰캡 아티팩트 3(도구 표를 두고 문서 재계산 장황→FINAL 미도달)·
  Bluest 픽 2(rebate 조항 견인). **형식 포렌식 결론**: C_CALC 8/8 과의 차이 = 표 형식
  (컴팩트 마크다운 라인 ↔ python-repr) — x287b "목록 위생=효과 변수" 동형. **처방**:
  `catalog_compute` 에 A2-선언 `row_template` 컴팩트 렌더 추가(값·행 순서 불변·엔진=치환만)
  + return_template 에 사실 진술 2문(totals are computed, not estimates / operator
  surcharge·rebate 는 범위 밖 — 미스 실물 2건의 견인 축을 사실로 명시). 검정 21/21.
  **최종 게이트(사전 고정): 형식 개정판 재측정 strict ≥12/16 → (B2) 확정 · <12 → (B2)
  A2 항목 회수·학습행 검토.**

## §7 동세션 병행 — fee-가족 나머지 (t7274w 판정 후속·4태스크 통합 런 준비)

- **FIX-5(출시·유닛 12/12)**: t7274w 073 실측 — (B) 도구가 id 를 내도 모델이 차액 아닌 값을
  크레딧($24.50/$15/$5 vs gold $9.50/$9.00/$1.50)+chk_3 중복 크레딧. x288 A_DOCS 0/8 이 잰 산술
  결손 범위 내에서 `_sg_details.delta`·`{delta_total}` 표면화 + 템플릿에 정책 축자
  (general_017 §2 "apply a credit for the net correction across all identified fee
  discrepancies") ONE-credit 문구.
- **FIX-4(x292 문면 격리 대기)**: 072/073/074 전부 msg02 에서 fee 도구를
  `@last:get_credit_card_transactions_by_user`(틀린 출처)로 호출 — 현행 BYREF deny "call that
  tool first" 가 틀린 도구 호출을 지시([[64]] 위반형)·074 는 그대로 credit-card 경로 표류.
  신 문면(참조 가능 출력 열거+오지시 제거) = `x292_byref_deny_probe.py` 사전등록: A_CUR ≥6/8
  ∧ B_NEW ≤2/8 → 출시.
  **1차(n=8): A_CUR BAD 4/8 · B_NEW BAD 0/8** — A_CUR 중간 대역 → 사전등록대로 n=16 재측정
  1회(게이트: A_CUR ≥12/16 ∧ B_NEW ≤4/16 → 출시 · A_CUR ≤4/16 → 보류·deny 는 무죄로 판정).
- **통합 재판정 순서(리뷰 지적 ②·명문화)**: 통합 런(`run_t7276_20260813x.sh`)은
  **ⓐ x291b strict ≥6/8 ∧ ⓑ x292 판정 완료(통과 시 FIX-4 탑재·미달 시 FIX-4 제외 명기)**
  이후에만 발사한다 — 검증 안 된 레버를 실은 합성 런은 [[09]]/[[19]] 위반(오염된 소모).
  x291b ≤2/8 이면 (B2)는 **회수(A2 항목 제거)** 하고 표 형식 포렌식으로 돌아간다.
  통과 후: FIX-4/5 + (B2) 탑재·**4태스크(072/073/074/075) nt=1 한 런** → 전수 포렌식 →
  이기는 태스크만 nt=4 확정런.
