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

## §5 결과 (추기)

(대기)
